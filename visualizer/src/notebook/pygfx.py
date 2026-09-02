"""Pygfx-based notebook visualizer for ORCHAV.

Provides interactive 3D visualization in Jupyter notebooks via ``pygfx`` +
``rendercanvas``, and headless offscreen rendering without Open3D.

Two rendering modes:

* **render()** — headless offscreen rendering to a numpy array via
  ``rendercanvas.offscreen``.  No display server needed.
* **show()** — interactive 3D widget.  In Jupyter, uses ``jupyter_rfb``
  WebSocket streaming for in-cell orbit/pan/zoom.  Outside Jupyter,
  opens a native window via ``rendercanvas.auto``.
* **widget()** — returns a Jupyter widget for embedding in ``ipywidgets``
  layouts.
* **animate()** — animated frame-by-frame playback in a Jupyter widget
  or native window.  Static scene built once, dynamic elements update
  per frame at target FPS.

Usage::

    from visualizer.src.notebook.pygfx import PygfxNotebookViz

    viz = PygfxNotebookViz("scenarios/getting_started/hello_world")

    # Headless render (remote SSH / CI)
    img = viz.render(frame=0, azimuth=45)

    # Interactive 3D in Jupyter
    viz.show()
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import TracebackType
from typing import Any, Self, Sequence

import numpy as np

from shared.logging import get_logger

from ..scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR_RGBA
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache
from ..state import MpcVisibility
from ..utils.geometry import mesh_vertices
from .frame_data import load_notebook_visual_frame
from .generator import NotebookGenerator

logger = get_logger("orchav.notebook_pygfx")

# Optional dependency guards

try:
    import pygfx as gfx

    PYGFX_AVAILABLE = True
except ImportError:
    gfx = None  # type: ignore[assignment]
    PYGFX_AVAILABLE = False

try:
    import jupyter_rfb  # noqa: F401

    JUPYTER_RFB_AVAILABLE = True
except ImportError:
    JUPYTER_RFB_AVAILABLE = False


def _running_in_jupyter() -> bool:
    """Return whether execution is inside a Jupyter kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


# Default node colors
_TX_COLOR = (1.0, 0.2, 0.2, 1.0)
_RX_COLOR = (0.2, 0.2, 1.0, 1.0)
_TARGET_COLOR = (0.8, 0.6, 0.5, 1.0)
_BG_COLOR = (0.8, 0.8, 0.8, 1.0)
_NODE_RADIUS = 1.0
_OPEN3D_IBL_INTENSITY_SCALE = 30000.0


def _normalize_pygfx_ibl_intensity(ibl_intensity: float) -> float:
    """Return pygfx's environment-map multiplier from either accepted scale."""
    try:
        value = max(0.0, float(ibl_intensity))
    except (TypeError, ValueError):
        return 1.0
    if value > 10.0:
        return value / _OPEN3D_IBL_INTENSITY_SCALE
    return value


class PygfxNotebookViz:
    """Pygfx-based notebook visualizer with interactive and headless modes.

    Loads a scenario (scene geometry + MPC frame data) and provides headless
    rendering via ``render()`` or interactive 3D via ``show()`` / ``widget()``.

    Args:
        scenario_path: Path to a scenario directory (containing
            ``scenario.yaml``) or directly to a YAML file.
        project_root: Explicit project root.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        scenario_path: str | Path,
        project_root: str | Path | None = None,
    ) -> None:
        """Load scenario metadata, scene meshes, frame source, and pygfx state."""
        if not PYGFX_AVAILABLE:
            raise ImportError(
                "pygfx is required for PygfxNotebookViz. "
                "Install the default runtime from a cloned repository with: "
                "python -m pip install -e ."
            )

        from ..io.frame_sources import FileSource
        from ..io.scenario_config import AppConfig, load_scenario

        # Resolve project root
        if project_root is not None:
            self._project_root = Path(project_root).resolve()
        else:
            from shared.scenarios.paths import find_project_root

            self._project_root = find_project_root(Path.cwd())

        scenario_abs = Path(scenario_path)
        if not scenario_abs.is_absolute():
            scenario_abs = self._project_root / scenario_path

        app_config = AppConfig.get_defaults(self._project_root)
        self._scenario = load_scenario(scenario_abs, app_config)
        logger.info("Loaded scenario: %s", self._scenario.root)

        # Resolve and load scene XML
        self._mesh_entries: list[dict[str, Any]] = []
        self._xml_root: ET.Element | None = None
        self._scene_xml_path: str | None = None
        self._load_scene(app_config)

        self._frame_source: FileSource | None = None
        if self._scenario.data_mode == "files":
            files_spec = self._scenario.data_spec.get("files", {})
            directory = files_spec.get("directory", "frames")
            fmt = files_spec.get("format", "h5")
            try:
                self._frame_source = FileSource(self._scenario.root, directory, fmt)
                self._frame_source.open()
                frames = self._frame_source.list_frames()
                logger.info("Frame source ready: %d frames available", len(frames))
            except FileNotFoundError:
                self._frame_source = None
                logger.info("No frames yet — call regenerate() or generate_frames() first")

        # MPC core for building view models
        from ..pipeline.core import MPCCore

        self._mpc_core = MPCCore(logger=logger, visualizer=None)

        # Pre-build texture cache
        from ..scene.assembly import build_texture_cache

        self._texture_cache = build_texture_cache(self._project_root)

        # Lazy-initialized generator (for regenerate())
        self._generator: NotebookGenerator | None = None

    # Scene loading

    def _load_scene(self, app_config: Any) -> None:
        """Resolve scene XML path and load meshes."""
        from ..scene.io import build_scene_from_root

        scene_spec = self._scenario.scene_spec
        scene_source = scene_spec.get("source", "library")
        scene_id = scene_spec.get("id", "default")

        xml_path: Path | None = None

        if scene_source == "sionna":
            from shared.scenarios.parsers import resolve_sionna_scene_xml

            xml_path = resolve_sionna_scene_xml(scene_id)
        elif scene_source == "library":
            candidate = app_config.scenes / scene_id
            if candidate.is_file():
                xml_path = candidate
            elif candidate.is_dir():
                xml_files = sorted(candidate.glob("*.xml"))
                if xml_files:
                    xml_path = xml_files[0]
            if xml_path is None or not xml_path.exists():
                from shared.scenarios.parsers import resolve_sionna_scene_xml

                base_id = scene_id.split("/")[0] if "/" in scene_id else scene_id
                xml_path = resolve_sionna_scene_xml(base_id)
                if xml_path is not None:
                    logger.info("Scene '%s' not in library, resolved via Sionna RT", scene_id)
        elif scene_source == "local":
            xml_path = self._scenario.root / scene_id
        elif scene_source == "osm":
            xml_path = self._scenario.root / "scene.xml"
            if not xml_path.exists():
                try:
                    from shared.scenarios import load_scenario_configuration

                    scenario_cfg = load_scenario_configuration(
                        self._scenario.root,
                        project_root=self._project_root,
                    )
                    xml_path = scenario_cfg.scene_xml
                except (ImportError, OSError, ValueError, KeyError) as exc:
                    logger.warning("Could not resolve generated city scene: %s", exc)

        if xml_path is None or not xml_path.exists():
            logger.warning("Scene XML not found for source=%s id=%s", scene_source, scene_id)
            return

        self._scene_xml_path = str(xml_path)
        self._xml_root = ET.parse(str(xml_path)).getroot()
        self._mesh_entries = build_scene_from_root(self._xml_root, self._scene_xml_path)
        logger.info(
            "Loaded %d scene meshes from %s",
            len(self._mesh_entries),
            xml_path.name,
        )

    # Public API

    def __enter__(self) -> Self:
        """Return this visualizer for an explicitly scoped notebook session."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the frame provider when leaving a scoped session."""
        self.close()

    def close(self) -> None:
        """Release frame files and invalidate cached frame data.

        This method is idempotent. Call it before an external producer replaces
        the scenario's HDF5 frame set.
        """
        source = self._frame_source
        self._frame_source = None
        try:
            if source is not None:
                source.close()
        finally:
            invalidate_visualizer_cache(
                self,
                CacheInvalidationScope.FRAME_DATA,
                reason="notebook_close",
            )

    @property
    def frames(self) -> list[int]:
        """Available frame indices."""
        if self._frame_source is None:
            return []
        return self._frame_source.list_frames()

    @property
    def num_frames(self) -> int:
        """Number of currently available HDF5 frames."""
        return len(self.frames)

    def reload(self) -> None:
        """Reload frame data from disk after external regeneration.

        Re-opens the HDF5 file source and clears the canonical data cache
        so subsequent ``render()`` calls pick up the new frames.
        """
        from ..io.frame_sources import FileSource

        self.close()

        files_spec = self._scenario.data_spec.get("files", {})
        directory = files_spec.get("directory", "frames")
        fmt = files_spec.get("format", "h5")
        replacement = FileSource(self._scenario.root, directory, fmt)
        try:
            replacement.open()
            frames = replacement.list_frames()
        except BaseException as exc:
            try:
                replacement.close()
            except BaseException as cleanup_error:
                exc.add_note(f"Replacement frame-source cleanup failed: {cleanup_error}")
            raise
        self._frame_source = replacement
        logger.info("Reloaded frame source: %d frames available", len(frames))

    def regenerate(
        self,
        tx_positions: Sequence[Sequence[float]],
        rx_positions: Sequence[Sequence[float]],
        *,
        quality: str = "low",
        steps: int = 1,
        duration: float | None = None,
        target_configs: list | None = None,
        carrier_frequency_hz: float | None = None,
        bandwidth_hz: float | None = None,
    ) -> list[int]:
        """Re-run ray tracing with new positions and reload frames.

        Requires GPU and the ``generator`` package.  Writes new HDF5 frames
        into the scenario directory, then calls :meth:`reload` so the next
        ``render()`` picks up the new data.

        Args:
            tx_positions: TX positions as ``[[x, y, z], ...]``.
            rx_positions: RX positions as ``[[x, y, z], ...]``.
            quality: Ray tracing quality preset.
            steps: Number of simulation steps.
            duration: Simulation duration in seconds.
            target_configs: Optional list of ``TargetConfig`` objects.
            carrier_frequency_hz: Override carrier frequency (Hz).
            bandwidth_hz: Override channel bandwidth (Hz).

        Returns:
            List of generated frame indices.
        """
        if self._scene_xml_path is None:
            raise RuntimeError("Cannot regenerate: no scene XML path resolved for this scenario.")

        if self._generator is None:
            from shared.scenarios import load_scenario_configuration

            scenario_abs = self._scenario.root
            scenario_cfg = load_scenario_configuration(
                scenario_abs, project_root=self._project_root
            )
            self._generator = NotebookGenerator(
                scene_xml_path=self._scene_xml_path,
                project_root=self._project_root,
                scenario_root=self._scenario.root,
                scenario_configuration=scenario_cfg,
            )

        self.close()
        try:
            frame_indices = self._generator.generate(
                tx_positions,
                rx_positions,
                quality=quality,
                steps=steps,
                duration=duration,
                target_configs=target_configs,
                carrier_frequency_hz=carrier_frequency_hz,
                bandwidth_hz=bandwidth_hz,
            )
        except BaseException as exc:
            try:
                self.reload()
            except BaseException as recovery_error:
                logger.warning(
                    "Could not reopen the rolled-back notebook frame set: %s",
                    recovery_error,
                )
                exc.add_note(
                    "Could not reopen the rolled-back notebook frame set: " f"{recovery_error}"
                )
            raise

        self.reload()
        return frame_indices

    def render(
        self,
        frame: int = 0,
        *,
        color_mode: str = "reflection_order",
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool = True,
        tx_positions: Sequence[Sequence[float]] | None = None,
        rx_positions: Sequence[Sequence[float]] | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 1280,
        height: int = 900,
        line_width: float = 2.0,
        point_size: float = 5.0,
        node_radius: float = _NODE_RADIUS,
        display: bool = True,
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
    ) -> np.ndarray:
        """Render the scene to a numpy image via offscreen pygfx rendering.

        Works over SSH and remote Jupyter — no display server needed.

        Args:
            frame: Frame index to display.
            color_mode: MPC coloring mode.
            selected_tx: TX filter (``"all"`` or 0-based index).
            selected_rx: RX filter.
            mpc_layer_enabled: Enable the MPC layer master switch.
            show_mpc_paths: Show MPC path lines while the MPC layer is enabled.
            show_mpc_bounce_points: Show physical MPC interaction points while the layer is enabled.
            tx_positions: Override TX marker positions (visual only).
            rx_positions: Override RX marker positions (visual only).
            azimuth: Camera azimuth in degrees.
            elevation: Camera elevation in degrees.
            distance: Camera distance from scene center.
            center: Camera look-at point ``[x, y, z]``.
            width: Image width in pixels.
            height: Image height in pixels.
            line_width: MPC path line width.
            point_size: Bounce point size.
            node_radius: TX/RX marker radius.
            display: If ``True`` and running in IPython/Jupyter, display
                the image inline.
            ibl_name: IBL environment map name.
            ibl_intensity: IBL intensity multiplier. Desktop-scale values above
                10, such as 30000, are normalized automatically.

        Returns:
            Rendered image as a ``(H, W, 3)`` uint8 numpy array.
        """
        from rendercanvas.offscreen import RenderCanvas

        scene, camera = self._build_pygfx_scene(
            frame=frame,
            color_mode=color_mode,
            selected_tx=selected_tx,
            selected_rx=selected_rx,
            mpc_layer_enabled=mpc_layer_enabled,
            show_mpc_paths=show_mpc_paths,
            show_mpc_bounce_points=show_mpc_bounce_points,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            center=center,
            line_width=line_width,
            point_size=point_size,
            node_radius=node_radius,
            ibl_name=ibl_name,
            ibl_intensity=ibl_intensity,
        )

        from ..renderers.pygfx.canvas import create_wgpu_renderer

        canvas = RenderCanvas(size=(width, height))
        try:
            renderer = create_wgpu_renderer(
                gfx,
                canvas,
                clear_color=tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA),
                offscreen=True,
            )

            canvas.request_draw(
                lambda: renderer.render(
                    scene,
                    camera,
                )
            )
            rgba = canvas.draw()
            image = np.asarray(rgba)[:, :, :3].copy()
        finally:
            # RenderCanvas retains the draw callback (and therefore the wgpu
            # renderer) until explicitly closed.  Breaking that ownership
            # cycle here also releases the native canvas context before Python
            # and wgpu begin their independent interpreter-shutdown cleanup.
            canvas.close()

        logger.info("Offscreen render: %dx%d, frame=%d", width, height, frame)

        if display:
            try:
                from IPython.display import display as ipy_display
                from PIL import Image as PILImage

                ipy_display(PILImage.fromarray(image))
            except ImportError:
                logger.info("IPython/Pillow not available — returning numpy array only")

        return image

    def show(
        self,
        frame: int = 0,
        *,
        color_mode: str = "reflection_order",
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool = True,
        tx_positions: Sequence[Sequence[float]] | None = None,
        rx_positions: Sequence[Sequence[float]] | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 1280,
        height: int = 900,
        line_width: float = 2.0,
        point_size: float = 5.0,
        node_radius: float = _NODE_RADIUS,
        title: str = "ORCHAV",
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
    ) -> Any:
        """Interactive 3D visualization.

        In Jupyter with ``jupyter_rfb`` installed: returns an interactive
        widget that renders in-cell with orbit/pan/zoom mouse controls.

        Outside Jupyter: opens a native window (blocks until closed).

        Args:
            frame: Frame index to display.
            color_mode: MPC coloring mode.
            selected_tx: TX filter (``"all"`` or 0-based index).
            selected_rx: RX filter.
            mpc_layer_enabled: Enable the MPC layer master switch.
            show_mpc_paths: Show MPC path lines while the MPC layer is enabled.
            show_mpc_bounce_points: Show physical MPC interaction points while the layer is enabled.
            tx_positions: Override TX marker positions (visual only).
            rx_positions: Override RX marker positions (visual only).
            azimuth: Camera azimuth in degrees.
            elevation: Camera elevation in degrees.
            distance: Camera distance from scene center.
            center: Camera look-at point ``[x, y, z]``.
            width: Window/widget width in pixels.
            height: Window/widget height in pixels.
            line_width: MPC path line width.
            point_size: Bounce point size.
            node_radius: TX/RX marker radius.
            title: Window title.
            ibl_name: IBL environment map name.
            ibl_intensity: IBL intensity multiplier. Desktop-scale values above
                10, such as 30000, are normalized automatically.

        Returns:
            In Jupyter: the widget object (displayed automatically).
            Outside Jupyter: ``None`` (blocks until window closed).
        """
        scene, camera = self._build_pygfx_scene(
            frame=frame,
            color_mode=color_mode,
            selected_tx=selected_tx,
            selected_rx=selected_rx,
            mpc_layer_enabled=mpc_layer_enabled,
            show_mpc_paths=show_mpc_paths,
            show_mpc_bounce_points=show_mpc_bounce_points,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            center=center,
            line_width=line_width,
            point_size=point_size,
            node_radius=node_radius,
            ibl_name=ibl_name,
            ibl_intensity=ibl_intensity,
        )

        controller = gfx.OrbitController(camera, register_events=None)

        if _running_in_jupyter() and JUPYTER_RFB_AVAILABLE:
            return self._show_jupyter(scene, camera, controller, width, height, title)
        else:
            return self._show_native(scene, camera, controller, width, height, title)

    def widget(
        self,
        frame: int = 0,
        *,
        color_mode: str = "reflection_order",
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool = True,
        tx_positions: Sequence[Sequence[float]] | None = None,
        rx_positions: Sequence[Sequence[float]] | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 800,
        height: int = 600,
        line_width: float = 2.0,
        point_size: float = 5.0,
        node_radius: float = _NODE_RADIUS,
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
    ) -> Any:
        """Return a Jupyter widget for embedding in ipywidgets layouts.

        Requires ``jupyter_rfb`` to be installed.

        Returns:
            A rendercanvas Jupyter widget object.

        Raises:
            ImportError: If ``jupyter_rfb`` is not installed.
        """
        if not JUPYTER_RFB_AVAILABLE:
            raise ImportError(
                "jupyter_rfb is required for widget(). " "Install with: pip install jupyter_rfb"
            )

        scene, camera = self._build_pygfx_scene(
            frame=frame,
            color_mode=color_mode,
            selected_tx=selected_tx,
            selected_rx=selected_rx,
            mpc_layer_enabled=mpc_layer_enabled,
            show_mpc_paths=show_mpc_paths,
            show_mpc_bounce_points=show_mpc_bounce_points,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            center=center,
            line_width=line_width,
            point_size=point_size,
            node_radius=node_radius,
            ibl_name=ibl_name,
            ibl_intensity=ibl_intensity,
        )

        controller = gfx.OrbitController(camera, register_events=None)

        from rendercanvas.jupyter import RenderCanvas

        canvas = RenderCanvas(size=(width, height))
        from ..renderers.pygfx.canvas import create_wgpu_renderer

        renderer = create_wgpu_renderer(
            gfx,
            canvas,
            clear_color=tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA),
        )

        controller.register_events(renderer)

        @canvas.request_draw
        def animate():
            """Render one widget frame and request the next notebook draw."""
            renderer.render(
                scene,
                camera,
            )
            canvas.request_draw()

        return canvas

    def animate(
        self,
        frames: Sequence[int] | None = None,
        *,
        fps: float = 10,
        loop: bool = True,
        color_mode: str = "reflection_order",
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool = True,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 1280,
        height: int = 900,
        line_width: float = 2.0,
        point_size: float = 5.0,
        node_radius: float = _NODE_RADIUS,
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
    ) -> Any:
        """Animated frame-by-frame playback in a Jupyter widget.

        Builds the static scene geometry (meshes, lights) once and updates
        only the dynamic elements (MPC paths, bounces, targets, TX/RX nodes)
        on each frame tick.

        Requires ``jupyter_rfb`` for in-cell display, otherwise falls back
        to a native window.

        Args:
            frames: Frame indices to animate.  Defaults to all available.
            fps: Target frames per second.
            loop: If ``True``, loop back to the first frame after the last.
            color_mode: MPC coloring mode.
            selected_tx: TX filter (``"all"`` or 0-based index).
            selected_rx: RX filter.
            mpc_layer_enabled: Enable the MPC layer master switch.
            show_mpc_paths: Show MPC path lines while the MPC layer is enabled.
            show_mpc_bounce_points: Show physical MPC interaction points while the layer is enabled.
            azimuth: Camera azimuth in degrees.
            elevation: Camera elevation in degrees.
            distance: Camera distance from scene center.
            center: Camera look-at point ``[x, y, z]``.
            width: Widget width in pixels.
            height: Widget height in pixels.
            line_width: MPC path line width.
            point_size: Bounce point size.
            node_radius: TX/RX marker radius.
            ibl_name: IBL environment map name.
            ibl_intensity: IBL intensity multiplier. Desktop-scale values above
                10, such as 30000, are normalized automatically.

        Returns:
            In Jupyter: the canvas widget (displayed automatically).
            Outside Jupyter: ``None`` (blocks until window closed).
        """
        if frames is None:
            frames = self.frames
        if not frames:
            logger.warning("No frames available for animation")
            return None

        from ..backends.pygfx_scene_helpers import (
            payload_to_pygfx_lines,
            payload_to_pygfx_mesh,
            payload_to_pygfx_points,
        )
        from ..scene.assembly import (
            compute_camera_orbit,
            make_sphere_payload,
            mesh_entry_to_payload,
            target_metadata_to_payload,
            view_model_to_mpc_payloads,
        )
        from ..types.render_payloads import MaterialPayload

        # ---- Static scene (built once) ----
        scene = gfx.Scene()
        scene.add(gfx.Background.from_color(tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA)))
        scene.add(gfx.AmbientLight(intensity=0.4))
        dir_light = gfx.DirectionalLight(intensity=0.8)
        dir_light.local.position = (50, 50, 100)
        scene.add(dir_light)

        ibl_manager = self._setup_ibl(scene, ibl_name, ibl_intensity)

        for entry in self._mesh_entries:
            result = mesh_entry_to_payload(entry, self._texture_cache)
            if result is None:
                continue
            _name, mesh_payload, mat_payload = result
            pygfx_mesh = payload_to_pygfx_mesh(
                gfx, mesh_payload, mat_payload, ibl_manager=ibl_manager
            )
            scene.add(pygfx_mesh)

        # ---- Camera (compute from first frame) ----
        first_raw, _ = self._build_frame_data(
            frames[0],
            color_mode,
            selected_tx,
            selected_rx,
            mpc_layer_enabled,
            show_mpc_paths,
            show_mpc_bounce_points,
            None,
            None,
        )
        first_tx, first_rx = self._extract_node_positions(first_raw)
        mesh_bboxes = self._extract_mesh_bboxes()

        camera = gfx.PerspectiveCamera(60, aspect=16 / 9)
        orbit = compute_camera_orbit(
            mesh_bboxes, first_tx, first_rx, azimuth, elevation, distance, center
        )
        self._apply_camera_orbit(camera, scene, orbit)

        # ---- Dynamic elements group ----
        dynamic_group = gfx.Group()
        scene.add(dynamic_group)

        import time

        state = {"frame_idx": 0, "last_time": time.perf_counter()}

        def _update_dynamic(frame_step: int) -> None:
            """Remove old dynamic objects and add new ones for the given frame."""
            dynamic_group.clear()

            raw_frame, view_model = self._build_frame_data(
                frame_step,
                color_mode,
                selected_tx,
                selected_rx,
                mpc_layer_enabled,
                show_mpc_paths,
                show_mpc_bounce_points,
                None,
                None,
            )

            if view_model is not None:
                lines_payload, points_payload = view_model_to_mpc_payloads(view_model)
                if lines_payload is not None:
                    dynamic_group.add(
                        payload_to_pygfx_lines(gfx, lines_payload, line_width=line_width)
                    )
                if points_payload is not None:
                    dynamic_group.add(
                        payload_to_pygfx_points(gfx, points_payload, point_size=point_size)
                    )

            if raw_frame is not None:
                for meta in raw_frame.get("targets_metadata", []):
                    tgt_result = target_metadata_to_payload(
                        meta, self._project_root, self._scenario.root
                    )
                    if tgt_result is None:
                        continue
                    _tgt_name, tgt_payload, _ = tgt_result
                    tgt_mat = MaterialPayload(
                        base_color=_TARGET_COLOR,
                        roughness=0.6,
                        metallic=0.0,
                        reflectance=0.4,
                    )
                    dynamic_group.add(
                        payload_to_pygfx_mesh(gfx, tgt_payload, tgt_mat, ibl_manager=ibl_manager)
                    )

            frame_tx, frame_rx = self._extract_node_positions(raw_frame)
            for pos in frame_tx:
                sp = make_sphere_payload(pos, node_radius, _TX_COLOR[:3])
                sm = MaterialPayload(base_color=_TX_COLOR, roughness=0.3, metallic=0.0)
                dynamic_group.add(payload_to_pygfx_mesh(gfx, sp, sm, ibl_manager=ibl_manager))
            for pos in frame_rx:
                sp = make_sphere_payload(pos, node_radius, _RX_COLOR[:3])
                sm = MaterialPayload(base_color=_RX_COLOR, roughness=0.3, metallic=0.0)
                dynamic_group.add(payload_to_pygfx_mesh(gfx, sp, sm, ibl_manager=ibl_manager))

        # Initial frame
        _update_dynamic(frames[0])

        frame_interval = 1.0 / fps

        # ---- Canvas + render loop ----
        if _running_in_jupyter() and JUPYTER_RFB_AVAILABLE:
            from rendercanvas.jupyter import RenderCanvas
        else:
            from rendercanvas.auto import RenderCanvas  # type: ignore[assignment]

        canvas = RenderCanvas(size=(width, height), title="ORCHAV Animation")
        from ..renderers.pygfx.canvas import create_wgpu_renderer

        renderer = create_wgpu_renderer(
            gfx,
            canvas,
            clear_color=tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA),
        )

        controller = gfx.OrbitController(camera, register_events=None)
        controller.register_events(renderer)

        def _animate_loop() -> None:
            """Advance the animation clock and render one native-canvas frame."""
            now = time.perf_counter()
            elapsed = now - state["last_time"]
            if elapsed >= frame_interval:
                state["last_time"] = now
                state["frame_idx"] += 1
                if state["frame_idx"] >= len(frames):
                    if loop:
                        state["frame_idx"] = 0
                    else:
                        state["frame_idx"] = len(frames) - 1
                _update_dynamic(frames[state["frame_idx"]])
            renderer.render(
                scene,
                camera,
            )
            canvas.request_draw()

        canvas.request_draw(_animate_loop)

        if _running_in_jupyter() and JUPYTER_RFB_AVAILABLE:
            return canvas
        else:
            from rendercanvas.auto import loop as rc_loop  # type: ignore[assignment]

            rc_loop.run()
            return None

    # Interactive display backends

    def _show_jupyter(
        self,
        scene: Any,
        camera: Any,
        controller: Any,
        width: int,
        height: int,
        title: str,
    ) -> Any:
        """Display interactive 3D widget in a Jupyter cell."""
        from rendercanvas.jupyter import RenderCanvas

        canvas = RenderCanvas(size=(width, height), title=title)
        from ..renderers.pygfx.canvas import create_wgpu_renderer

        renderer = create_wgpu_renderer(
            gfx,
            canvas,
            clear_color=tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA),
        )

        controller.register_events(renderer)

        @canvas.request_draw
        def animate():
            """Render one interactive Jupyter frame and request redraw."""
            renderer.render(
                scene,
                camera,
            )
            canvas.request_draw()

        return canvas

    def _show_native(
        self,
        scene: Any,
        camera: Any,
        controller: Any,
        width: int,
        height: int,
        title: str,
    ) -> None:
        """Display interactive 3D in a native window (blocks until closed)."""
        from rendercanvas.auto import RenderCanvas, loop

        canvas = RenderCanvas(size=(width, height), title=title)
        from ..renderers.pygfx.canvas import create_wgpu_renderer

        renderer = create_wgpu_renderer(
            gfx,
            canvas,
            clear_color=tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA),
        )

        controller.register_events(renderer)

        @canvas.request_draw
        def animate():
            """Render one interactive native-window frame and request redraw."""
            renderer.render(
                scene,
                camera,
            )
            canvas.request_draw()

        loop.run()

    # Scene building

    def _build_pygfx_scene(
        self,
        frame: int,
        color_mode: str,
        selected_tx: int | str,
        selected_rx: int | str,
        mpc_layer_enabled: bool,
        show_mpc_paths: bool,
        show_mpc_bounce_points: bool,
        tx_positions: Sequence[Sequence[float]] | None,
        rx_positions: Sequence[Sequence[float]] | None,
        azimuth: float | None,
        elevation: float | None,
        distance: float | None,
        center: Sequence[float] | None,
        line_width: float,
        point_size: float,
        node_radius: float,
        ibl_name: str,
        ibl_intensity: float,
    ) -> tuple[Any, Any]:
        """Build a complete pygfx scene with all geometry and camera.

        Returns:
            (scene, camera) tuple.
        """
        from ..backends.pygfx_scene_helpers import (
            payload_to_pygfx_lines,
            payload_to_pygfx_mesh,
            payload_to_pygfx_points,
        )
        from ..scene.assembly import (
            compute_camera_orbit,
            make_sphere_payload,
            mesh_entry_to_payload,
            target_metadata_to_payload,
            view_model_to_mpc_payloads,
        )
        from ..types.render_payloads import MaterialPayload

        scene = gfx.Scene()
        scene.add(gfx.Background.from_color(tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA)))

        # ---- Lights ----
        scene.add(gfx.AmbientLight(intensity=0.4))
        light = gfx.DirectionalLight(intensity=0.8)
        light.local.position = (50, 50, 100)
        scene.add(light)

        # ---- IBL (optional) ----
        ibl_manager = self._setup_ibl(scene, ibl_name, ibl_intensity)

        # ---- Scene meshes ----
        for entry in self._mesh_entries:
            result = mesh_entry_to_payload(entry, self._texture_cache)
            if result is None:
                continue
            name, mesh_payload, mat_payload = result
            pygfx_mesh = payload_to_pygfx_mesh(
                gfx, mesh_payload, mat_payload, ibl_manager=ibl_manager
            )
            scene.add(pygfx_mesh)

        # ---- Frame data + view model ----
        raw_frame, view_model = self._build_frame_data(
            frame,
            color_mode,
            selected_tx,
            selected_rx,
            mpc_layer_enabled,
            show_mpc_paths,
            show_mpc_bounce_points,
            tx_positions,
            rx_positions,
        )

        # ---- MPC lines and bounces ----
        if view_model is not None:
            lines_payload, points_payload = view_model_to_mpc_payloads(view_model)
            if lines_payload is not None:
                pygfx_lines = payload_to_pygfx_lines(gfx, lines_payload, line_width=line_width)
                scene.add(pygfx_lines)
            if points_payload is not None:
                pygfx_points = payload_to_pygfx_points(gfx, points_payload, point_size=point_size)
                scene.add(pygfx_points)

        # ---- Target meshes ----
        if raw_frame is not None:
            targets_metadata = raw_frame.get("targets_metadata", [])
            for i, meta in enumerate(targets_metadata):
                result = target_metadata_to_payload(meta, self._project_root, self._scenario.root)
                if result is None:
                    continue
                tgt_name, tgt_payload, _ = result
                tgt_mat = MaterialPayload(
                    base_color=_TARGET_COLOR,
                    roughness=0.6,
                    metallic=0.0,
                    reflectance=0.4,
                )
                pygfx_mesh = payload_to_pygfx_mesh(
                    gfx, tgt_payload, tgt_mat, ibl_manager=ibl_manager
                )
                scene.add(pygfx_mesh)

        # ---- TX/RX markers ----
        frame_tx, frame_rx = self._extract_node_positions(raw_frame)

        for pos in frame_tx:
            marker_payload = make_sphere_payload(pos, node_radius, _TX_COLOR[:3])
            marker_mat = MaterialPayload(base_color=_TX_COLOR, roughness=0.3, metallic=0.0)
            pygfx_mesh = payload_to_pygfx_mesh(
                gfx, marker_payload, marker_mat, ibl_manager=ibl_manager
            )
            scene.add(pygfx_mesh)

        for pos in frame_rx:
            marker_payload = make_sphere_payload(pos, node_radius, _RX_COLOR[:3])
            marker_mat = MaterialPayload(base_color=_RX_COLOR, roughness=0.3, metallic=0.0)
            pygfx_mesh = payload_to_pygfx_mesh(
                gfx, marker_payload, marker_mat, ibl_manager=ibl_manager
            )
            scene.add(pygfx_mesh)

        # ---- TX/RX labels (native pygfx Text) ----
        self._add_node_labels(scene, frame_tx, frame_rx)

        # ---- Camera ----
        camera = gfx.PerspectiveCamera(60, aspect=16 / 9)

        mesh_bboxes = self._extract_mesh_bboxes()
        orbit = compute_camera_orbit(
            mesh_bboxes, frame_tx, frame_rx, azimuth, elevation, distance, center
        )
        self._apply_camera_orbit(camera, scene, orbit)

        return scene, camera

    def _add_node_labels(
        self,
        scene: Any,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
    ) -> None:
        """Add native pygfx text labels for TX/RX markers.

        Screen-space labels so paper-figure exports stay legible at any zoom.
        """
        for idx, pos in enumerate(tx_positions):
            text = f"TX{idx + 1}"
            label = gfx.Text(
                text=text,
                font_size=18,
                screen_space=True,
                anchor="middle-center",
                material=gfx.TextMaterial(
                    color=(0.0, 0.8, 0.0),
                    outline_color=(0.0, 0.0, 0.0),
                    outline_thickness=0.15,
                    aa=True,
                ),
            )
            label.local.position = (float(pos[0]), float(pos[1]), float(pos[2]) + 2.0)
            scene.add(label)

        for idx, pos in enumerate(rx_positions):
            text = f"RX{idx + 1}"
            label = gfx.Text(
                text=text,
                font_size=18,
                screen_space=True,
                anchor="middle-center",
                material=gfx.TextMaterial(
                    color=(0.3, 0.3, 1.0),
                    outline_color=(0.0, 0.0, 0.0),
                    outline_thickness=0.15,
                    aa=True,
                ),
            )
            label.local.position = (float(pos[0]), float(pos[1]), float(pos[2]) + 2.0)
            scene.add(label)

    def _setup_ibl(self, scene: Any, ibl_name: str, ibl_intensity: float) -> Any | None:
        """Set up IBL environment lighting if available."""
        from ..renderers.pygfx.lighting import PygfxIBLManager

        ibl_dir = self._project_root / "libraries" / "ibl"
        if not ibl_dir.is_dir():
            return None

        try:
            ibl_manager = PygfxIBLManager(gfx, ibl_dir)
            texture = ibl_manager.load_ibl(ibl_name)
            if texture is not None:
                ibl_manager.apply_to_scene(scene)
                ibl_manager.set_skybox_visible(False, scene)
                ibl_manager.set_intensity(_normalize_pygfx_ibl_intensity(ibl_intensity))
                return ibl_manager
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug("Could not set up IBL: %s", exc)
        return None

    @staticmethod
    def _apply_camera_orbit(camera: Any, scene: Any, orbit: Any) -> None:
        """Position camera using orbit parameters with a safe fallback.

        ``camera.show_object(scene)`` requires the scene to have a bounding
        sphere. When the scene has no visible geometry, a synthetic bounding
        sphere from the orbit center/distance is passed
        that as a ``(x, y, z, r)`` tuple instead.
        """
        eye = orbit.eye_position()
        view_dir = tuple(orbit.center - eye)
        camera.local.position = tuple(eye)
        try:
            camera.show_object(scene, up=(0, 0, 1), view_dir=view_dir)
        except ValueError:
            r = max(float(orbit.distance) * 0.5, 1.0)
            bsphere = (*orbit.center.tolist(), r)
            camera.show_object(bsphere, up=(0, 0, 1), view_dir=view_dir)

    def _extract_mesh_bboxes(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract (center, min_bound, max_bound) for each mesh entry."""

        result = []
        for entry in self._mesh_entries:
            mesh = entry.get("mesh")
            if mesh is None:
                continue
            vertices = mesh_vertices(mesh)
            if vertices is None or vertices.size == 0:
                continue
            vertices = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
            min_b = vertices.min(axis=0)
            max_b = vertices.max(axis=0)
            center = (min_b + max_b) / 2
            result.append((center, min_b, max_b))
        return result

    # Frame data helpers

    def _build_frame_data(
        self,
        frame: int,
        color_mode: str,
        selected_tx: int | str,
        selected_rx: int | str,
        mpc_layer_enabled: bool,
        show_mpc_paths: bool,
        show_mpc_bounce_points: bool,
        tx_positions: Sequence[Sequence[float]] | None,
        rx_positions: Sequence[Sequence[float]] | None,
    ) -> tuple[dict | None, Any]:
        """Load a visual frame payload and build its view model.

        Returns:
            ``(frame_payload, view_model)``; either may be ``None``.
        """
        frame_payload = None
        if self._frame_source is not None:
            available = self._frame_source.list_frames()
            if frame in available:
                source_step = frame
            elif available:
                source_step = available[0]
                logger.warning("Frame %d not available, using frame %d", frame, source_step)
            else:
                source_step = None
            if source_step is not None:
                frame_payload = load_notebook_visual_frame(
                    self._frame_source,
                    source_step,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                )

        view_model = None
        if frame_payload is not None:
            view_model = self._mpc_core.create_view_model(
                step=frame,
                raw_frame=frame_payload,
                color_mode=color_mode,
                selected_tx=selected_tx,
                selected_rx=selected_rx,
                mpc_visibility=MpcVisibility(
                    enabled=mpc_layer_enabled,
                    paths=show_mpc_paths,
                    bounce_points=show_mpc_bounce_points,
                ),
            )

        return frame_payload, view_model

    @staticmethod
    def _extract_node_positions(
        raw_frame: dict | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract TX/RX position arrays from a raw frame."""
        frame_tx = (
            np.array(raw_frame["tx_positions"], dtype=np.float64)
            if raw_frame is not None and "tx_positions" in raw_frame
            else np.empty((0, 3))
        )
        frame_rx = (
            np.array(raw_frame["rx_positions"], dtype=np.float64)
            if raw_frame is not None and "rx_positions" in raw_frame
            else np.empty((0, 3))
        )
        return frame_tx, frame_rx

    def __repr__(self) -> str:
        """Return a compact notebook-friendly scenario summary."""
        n_meshes = len(self._mesh_entries)
        n_frames = self.num_frames
        return (
            f"PygfxNotebookViz(meshes={n_meshes}, frames={n_frames}, "
            f"root={self._scenario.root})"
        )
