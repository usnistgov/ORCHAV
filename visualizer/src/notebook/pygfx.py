"""Pygfx-based notebook visualizer for ORCHAV.

Provides interactive 3D visualization in Jupyter notebooks via ``pygfx`` +
``rendercanvas``, and headless offscreen rendering without Open3D.

Rendering modes:

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

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import TracebackType
from typing import Any, Self, Sequence
from weakref import WeakSet

import numpy as np

from shared.logging import get_logger

from ..scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR_RGBA
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache
from ..state import MpcVisibility
from ..types.label_style import LabelCalloutStyle
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


def _require_jupyter_rfb() -> None:
    """Explain how to enable in-cell rendering before allocating scene resources."""
    if not JUPYTER_RFB_AVAILABLE:
        raise ImportError(
            "jupyter_rfb is required for interactive notebook views. "
            "Install with: python -m pip install jupyter_rfb. "
            "Use render() for a static image."
        )


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
        appearance: Default 3D preset: Detailed, Paper, Dark, or Glow.
            None retains the ordinary notebook lighting and materials.
        label_style: Plain labels, callouts, cards, or a LabelCalloutStyle settings dictionary.
        tx_color: TX marker/label color as RGB or RGBA values between zero and one.
        rx_color: RX marker/label color as RGB or RGBA values between zero and one.
    """

    def __init__(
        self,
        scenario_path: str | Path,
        project_root: str | Path | None = None,
        *,
        appearance: str | None = None,
        label_style: str | dict | LabelCalloutStyle = "plain",
        tx_color: Sequence[float] = _TX_COLOR,
        rx_color: Sequence[float] = _RX_COLOR,
    ) -> None:
        """Load scenario metadata, scene meshes, frame source, and pygfx state."""
        if not PYGFX_AVAILABLE:
            raise ImportError(
                "pygfx is required for PygfxNotebookViz. "
                "Install the default runtime from a cloned repository with: "
                "python -m pip install -e ."
            )

        self._presentation_defaults = dict(
            appearance=appearance, label_style=label_style, tx_color=tx_color, rx_color=rx_color
        )
        self._new_presentation()

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

        self._mpc_core = MPCCore(logger=logger, visualizer=None, subscribe_to_theme=False)

        # Pre-build texture cache
        from ..scene.assembly import build_texture_cache

        self._texture_cache = build_texture_cache(self._project_root)

        # Lazy-initialized generator (for regenerate())
        self._generator: NotebookGenerator | None = None
        self._canvases: WeakSet[Any] = WeakSet()

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
        """Close interactive views, release frame files, and invalidate frame data.

        This method is idempotent. Call it before an external producer replaces
        the scenario's HDF5 frame set.
        """
        canvases = tuple(getattr(self, "_canvases", ()))
        # Invalidate callbacks before closing native resources or the provider.
        self._canvases = WeakSet()
        for canvas in canvases:
            try:
                canvas.close()
            except (OSError, RuntimeError) as exc:
                logger.warning("Could not close notebook canvas: %s", exc)
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

        Closes existing interactive views, re-opens the HDF5 file source, and
        clears the canonical data cache. Create a new view after reloading.
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
        color_mode: str | None = None,
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool | None = None,
        tx_positions: Sequence[Sequence[float]] | None = None,
        rx_positions: Sequence[Sequence[float]] | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 1280,
        height: int = 900,
        line_width: float | None = None,
        point_size: float | None = None,
        node_radius: float = _NODE_RADIUS,
        display: bool = True,
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
        appearance: str | None = None,
        label_style: str | dict | LabelCalloutStyle | None = None,
        tx_color: Sequence[float] | None = None,
        rx_color: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Render the scene to a numpy image via offscreen pygfx rendering.

        Works over SSH and remote Jupyter — no display server needed.

        Args:
            frame: Frame index to display.
            color_mode: MPC coloring mode; None uses the appearance preset.
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
            node_radius: TX/RX marker radius for plain labels.
            display: If ``True`` and running in IPython/Jupyter, display
                the image inline.
            ibl_name: IBL environment map name.
            ibl_intensity: IBL intensity multiplier. Desktop-scale values above
                10, such as 30000, are normalized automatically.
            appearance: Override the constructor's preset for this view.
            label_style: Override plain, callout, or card labels for this view.
            tx_color: Override the TX RGB/RGBA color for this view.
            rx_color: Override the RX RGB/RGBA color for this view.

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
            appearance=appearance,
            label_style=label_style,
            tx_color=tx_color,
            rx_color=rx_color,
        )

        from ..renderers.pygfx.canvas import create_wgpu_renderer

        canvas = RenderCanvas(size=(width, height))
        try:
            renderer = create_wgpu_renderer(
                gfx,
                canvas,
                clear_color=scene._orchav_presentation.background,
                offscreen=True,
            )

            presentation = scene._orchav_presentation
            presentation.configure_renderer(renderer)

            def draw():
                presentation.prepare_draw(canvas, camera)
                renderer.render(scene, camera)

            canvas.request_draw(draw)
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
        color_mode: str | None = None,
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool | None = None,
        tx_positions: Sequence[Sequence[float]] | None = None,
        rx_positions: Sequence[Sequence[float]] | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 1280,
        height: int = 900,
        line_width: float | None = None,
        point_size: float | None = None,
        node_radius: float = _NODE_RADIUS,
        title: str = "ORCHAV",
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
        appearance: str | None = None,
        label_style: str | dict | LabelCalloutStyle | None = None,
        tx_color: Sequence[float] | None = None,
        rx_color: Sequence[float] | None = None,
    ) -> Any:
        """Interactive 3D visualization.

        In Jupyter with ``jupyter_rfb`` installed: returns an interactive
        widget that renders in-cell with orbit/pan/zoom mouse controls.

        Outside Jupyter: opens a native window (blocks until closed).

        Args:
            frame: Frame index to display.
            color_mode: MPC coloring mode; None uses the appearance preset.
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
            node_radius: TX/RX marker radius for plain labels.
            title: Window title.
            ibl_name: IBL environment map name.
            ibl_intensity: IBL intensity multiplier. Desktop-scale values above
                10, such as 30000, are normalized automatically.
            appearance: Override the constructor's preset for this view.
            label_style: Override plain, callout, or card labels for this view.
            tx_color: Override the TX RGB/RGBA color for this view.
            rx_color: Override the RX RGB/RGBA color for this view.

        Returns:
            In Jupyter: the widget object (displayed automatically).
            Outside Jupyter: ``None`` (blocks until window closed).
        """
        in_jupyter = _running_in_jupyter()
        if in_jupyter:
            _require_jupyter_rfb()
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
            appearance=appearance,
            label_style=label_style,
            tx_color=tx_color,
            rx_color=rx_color,
        )

        controller = gfx.OrbitController(camera, register_events=None)

        if in_jupyter:
            return self._show_jupyter(scene, camera, controller, width, height, title)
        else:
            return self._show_native(scene, camera, controller, width, height, title)

    def widget(
        self,
        frame: int = 0,
        *,
        color_mode: str | None = None,
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool | None = None,
        tx_positions: Sequence[Sequence[float]] | None = None,
        rx_positions: Sequence[Sequence[float]] | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 800,
        height: int = 600,
        line_width: float | None = None,
        point_size: float | None = None,
        node_radius: float = _NODE_RADIUS,
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
        appearance: str | None = None,
        label_style: str | dict | LabelCalloutStyle | None = None,
        tx_color: Sequence[float] | None = None,
        rx_color: Sequence[float] | None = None,
    ) -> Any:
        """Return a Jupyter widget for embedding in ipywidgets layouts.

        Requires ``jupyter_rfb`` to be installed. Appearance and label options
        follow :meth:`render`.

        Returns:
            A rendercanvas Jupyter widget object.

        Raises:
            ImportError: If ``jupyter_rfb`` is not installed.
        """
        _require_jupyter_rfb()

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
            appearance=appearance,
            label_style=label_style,
            tx_color=tx_color,
            rx_color=rx_color,
        )

        controller = gfx.OrbitController(camera, register_events=None)

        return self._show_jupyter(scene, camera, controller, width, height, "ORCHAV")

    def animate(
        self,
        frames: Sequence[int] | None = None,
        *,
        fps: float = 10,
        loop: bool = True,
        color_mode: str | None = None,
        selected_tx: int | str = "all",
        selected_rx: int | str = "all",
        mpc_layer_enabled: bool = True,
        show_mpc_paths: bool = True,
        show_mpc_bounce_points: bool | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        center: Sequence[float] | None = None,
        width: int = 1280,
        height: int = 900,
        line_width: float | None = None,
        point_size: float | None = None,
        node_radius: float = _NODE_RADIUS,
        ibl_name: str = "neutral_outdoor",
        ibl_intensity: float = 1.0,
        appearance: str | None = None,
        label_style: str | dict | LabelCalloutStyle | None = None,
        tx_color: Sequence[float] | None = None,
        rx_color: Sequence[float] | None = None,
    ) -> Any:
        """Animated frame-by-frame playback in a Jupyter widget.

        Builds the static scene geometry (meshes, lights) once and updates
        only the dynamic elements (MPC paths, bounces, targets, TX/RX nodes)
        on each frame tick.

        Requires ``jupyter_rfb`` for in-cell display. Outside Jupyter, opens
        a native window. Non-looping playback leaves the last frame interactive.

        Args:
            frames: Frame indices to animate.  Defaults to all available.
            fps: Target frames per second.
            loop: If ``True``, loop back to the first frame after the last.
            color_mode: MPC coloring mode; None uses the appearance preset.
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
            node_radius: TX/RX marker radius for plain labels.
            ibl_name: IBL environment map name.
            ibl_intensity: IBL intensity multiplier. Desktop-scale values above
                10, such as 30000, are normalized automatically.
            appearance: Override the constructor's preset for this view.
            label_style: Override plain, callout, or card labels for this view.
            tx_color: Override the TX RGB/RGBA color for this view.
            rx_color: Override the RX RGB/RGBA color for this view.

        Returns:
            In Jupyter: the canvas widget (displayed automatically).
            Outside Jupyter: ``None`` (blocks until window closed).
        """
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and greater than zero")
        available = self.frames
        frames = list(available if frames is None else frames)
        if not frames:
            logger.warning("No frames available for animation")
            return None
        available_set = set(available)
        missing = [frame for frame in frames if frame not in available_set]
        if missing:
            raise ValueError(f"Animation frames are not available: {missing}")
        in_jupyter = _running_in_jupyter()
        if in_jupyter:
            _require_jupyter_rfb()

        from ..scene.assembly import compute_camera_orbit

        presentation = self._new_presentation(appearance, label_style, tx_color, rx_color)
        color_mode = color_mode if color_mode is not None else presentation.mpc["color_mode"]
        show_mpc_bounce_points = (
            presentation.mpc["bounce_points"]
            if show_mpc_bounce_points is None
            else show_mpc_bounce_points
        )
        line_width = presentation.mpc["line_width"] if line_width is None else line_width
        point_size = presentation.mpc["point_size"] if point_size is None else point_size
        scene = self._build_static_scene(presentation, ibl_name, ibl_intensity)

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
        self._apply_camera_orbit(
            camera, scene, orbit, explicit_framing=center is not None or distance is not None
        )

        # ---- Dynamic elements group ----
        dynamic_group = gfx.Group()
        scene.add(dynamic_group)

        import time

        # Start the clock on the first draw, which may follow a delayed notebook
        # display. The first stored frame must still be shown before advancing.
        state = {"frame_idx": 0, "last_time": None, "playing": len(frames) > 1}

        def _update_dynamic(frame_step: int) -> None:
            """Replace the dynamic group using the shared still-frame assembly."""
            dynamic_group.clear()
            presentation.clear_dynamic()
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
                color_palette=presentation.palette,
            )
            self._add_dynamic(
                dynamic_group,
                raw_frame,
                view_model,
                presentation,
                line_width,
                point_size,
                node_radius,
            )

        # Initial frame
        _update_dynamic(frames[0])

        frame_interval = 1.0 / fps

        # ---- Canvas + render loop ----
        if in_jupyter:
            from rendercanvas.jupyter import RenderCanvas
        else:
            from rendercanvas.auto import RenderCanvas  # type: ignore[assignment]

        canvas = RenderCanvas(size=(width, height), title="ORCHAV Animation", max_fps=fps)
        renderer = self._create_interactive_renderer(canvas, scene)

        controller = gfx.OrbitController(camera, register_events=None)
        controller.register_events(renderer)
        final_presentation_pending = in_jupyter

        def _animate_loop() -> None:
            """Advance playback while active; retain camera controls at completion."""
            nonlocal final_presentation_pending
            if canvas not in self._canvases or canvas.get_closed():
                return
            now = time.perf_counter()
            if state["last_time"] is None:
                state["last_time"] = now
            elif state["playing"] and now - state["last_time"] >= frame_interval:
                state["last_time"] += frame_interval
                state["frame_idx"] = (state["frame_idx"] + 1) % len(frames)
                _update_dynamic(frames[state["frame_idx"]])
                state["playing"] = loop or state["frame_idx"] < len(frames) - 1
            scene._orchav_presentation.prepare_draw(canvas, camera)
            renderer.render(
                scene,
                camera,
            )
            if state["playing"]:
                canvas.request_draw()
            elif final_presentation_pending:
                # Deliver the terminal frame after its asynchronous GPU readback.
                final_presentation_pending = False
                canvas.request_draw()

        canvas.request_draw(_animate_loop)

        if in_jupyter:
            return canvas
        else:
            from rendercanvas.auto import loop as rc_loop  # type: ignore[assignment]

            rc_loop.run()
            return None

    # Interactive display backends

    def _create_interactive_renderer(self, canvas: Any, scene: Any = None) -> Any:
        """Create a renderer and retain weak ownership of its interactive canvas."""
        from ..renderers.pygfx.canvas import create_wgpu_renderer

        try:
            presentation = None if scene is None else scene._orchav_presentation
            renderer = create_wgpu_renderer(
                gfx,
                canvas,
                clear_color=(
                    tuple(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA)
                    if presentation is None
                    else presentation.background
                ),
            )
            if presentation is not None:
                presentation.configure_renderer(renderer)
        except BaseException as exc:
            try:
                canvas.close()
            except BaseException as cleanup_error:
                exc.add_note(f"Notebook canvas cleanup failed: {cleanup_error}")
            raise
        self._canvases.add(canvas)
        return renderer

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
        renderer = self._create_interactive_renderer(canvas, scene)

        controller.register_events(renderer)
        first_presentation_pending = True

        @canvas.request_draw
        def animate():
            """Render on demand; the orbit controller requests interaction redraws."""
            nonlocal first_presentation_pending
            if canvas not in self._canvases or canvas.get_closed():
                return
            scene._orchav_presentation.prepare_draw(canvas, camera)
            renderer.render(
                scene,
                camera,
            )
            if first_presentation_pending:
                # The notebook transport needs another pull after the initial
                # asynchronous GPU readback has completed.
                first_presentation_pending = False
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
        renderer = self._create_interactive_renderer(canvas, scene)

        controller.register_events(renderer)

        @canvas.request_draw
        def animate():
            """Render on demand; the orbit controller requests interaction redraws."""
            if canvas not in self._canvases or canvas.get_closed():
                return
            scene._orchav_presentation.prepare_draw(canvas, camera)
            renderer.render(
                scene,
                camera,
            )

        loop.run()

    # Scene building

    def _new_presentation(self, appearance=None, label_style=None, tx_color=None, rx_color=None):
        """Resolve independent view settings from constructor defaults and call overrides."""
        from .presentation import NotebookPresentation

        options = dict(getattr(self, "_presentation_defaults", {}))
        for key, value in dict(
            appearance=appearance, label_style=label_style, tx_color=tx_color, rx_color=rx_color
        ).items():
            if value is not None:
                options[key] = value
        options.setdefault("appearance", None)
        options.setdefault("label_style", "plain")
        return NotebookPresentation(
            gfx,
            **options,
            varied_buildings=any(
                entry.get("_osm_building_parts") for entry in getattr(self, "_mesh_entries", ())
            ),
        )

    def _build_static_scene(self, presentation, ibl_name, ibl_intensity):
        """Assemble the environment once for a still view or an animation."""
        from ..materials.texture_policy import view_texture_maps
        from ..renderers.pygfx.lighting_profiles import environment_lighting_profile
        from ..scene.assembly import mesh_entry_to_payload
        from ..services.material_pbr_service import MaterialPBRService

        scene = gfx.Scene()
        scene._orchav_presentation = presentation
        settings = presentation.settings
        if settings is not None:
            ibl_name = settings.get("sky", ibl_name)
            ibl_intensity = settings["ibl"]
            if settings["skybox"]:
                profile = environment_lighting_profile(ibl_name)
                if profile is not None:
                    ibl_intensity = profile.ibl_intensity
        presentation.ibl_manager = (
            self._setup_ibl(scene, ibl_name, ibl_intensity)
            if ibl_intensity > 0 or (settings is not None and settings["skybox"])
            else None
        )
        if not (
            settings is not None and settings["skybox"] and presentation.ibl_manager is not None
        ):
            scene.add(gfx.Background.from_color(presentation.background))
        presentation.setup_lights(scene)
        if presentation.notice:
            logger.info(presentation.notice)
        material_service = MaterialPBRService(presentation.material_context())
        with view_texture_maps(presentation.textures):
            for index, original in enumerate(self._mesh_entries):
                # Procedural material resolution may replace an entry's geometry.
                # Each view owns that display choice, leaving loaded assets intact.
                entry = dict(original)
                result = mesh_entry_to_payload(
                    entry, self._texture_cache, index, material_service=material_service
                )
                if result is not None:
                    name, payload, material = result
                    presentation.add_mesh(scene, name, payload, material, entry)
        return scene

    def _add_dynamic(
        self, group, raw_frame, view_model, presentation, line_width, point_size, node_radius
    ) -> None:
        """Build paths, targets, and nodes consistently in all notebook modes."""
        from ..backends.pygfx_scene_helpers import (
            payload_to_pygfx_lines,
            payload_to_pygfx_mesh,
            payload_to_pygfx_points,
        )
        from ..scene.assembly import (
            make_sphere_payload,
            target_metadata_to_payload,
            view_model_to_mpc_payloads,
        )
        from ..types.render_payloads import MaterialPayload

        if view_model is not None:
            lines, points = view_model_to_mpc_payloads(view_model)
            if lines is not None:
                group.add(payload_to_pygfx_lines(gfx, lines, line_width=line_width))
            if points is not None:
                group.add(payload_to_pygfx_points(gfx, points, point_size=point_size))
        if raw_frame is not None:
            for index, meta in enumerate(raw_frame.get("targets_metadata", [])):
                result = target_metadata_to_payload(meta, self._project_root, self._scenario.root)
                if result is None:
                    continue
                _, payload, material = result
                if meta.get("representation") not in {"constant_rcs", "tr38901"}:
                    material = MaterialPayload(
                        base_color=_TARGET_COLOR, roughness=0.6, metallic=0.0, reflectance=0.4
                    )
                presentation.add_mesh(
                    group, f"target_{index}", payload, material, meta, target=True
                )
        frame_tx, frame_rx = self._extract_node_positions(raw_frame)
        if presentation.label_style.mode == "plain":
            for kind, positions in (("tx", frame_tx), ("rx", frame_rx)):
                color = presentation.node_colors[kind]
                for position in positions:
                    payload = make_sphere_payload(position, node_radius, color[:3])
                    material = MaterialPayload(base_color=color, roughness=0.3, metallic=0.0)
                    group.add(
                        payload_to_pygfx_mesh(
                            gfx, payload, material, ibl_manager=presentation.ibl_manager
                        )
                    )
        self._add_node_labels(group, frame_tx, frame_rx, presentation)

    def _build_pygfx_scene(
        self,
        frame: int,
        color_mode: str | None,
        selected_tx: int | str,
        selected_rx: int | str,
        mpc_layer_enabled: bool,
        show_mpc_paths: bool,
        show_mpc_bounce_points: bool | None,
        tx_positions: Sequence[Sequence[float]] | None,
        rx_positions: Sequence[Sequence[float]] | None,
        azimuth: float | None,
        elevation: float | None,
        distance: float | None,
        center: Sequence[float] | None,
        line_width: float | None,
        point_size: float | None,
        node_radius: float,
        ibl_name: str,
        ibl_intensity: float,
        appearance: str | None = None,
        label_style: str | dict | LabelCalloutStyle | None = None,
        tx_color: Sequence[float] | None = None,
        rx_color: Sequence[float] | None = None,
    ) -> tuple[Any, Any]:
        """Build a complete pygfx scene with all geometry and camera.

        Returns:
            (scene, camera) tuple.
        """
        from ..scene.assembly import compute_camera_orbit

        presentation = self._new_presentation(appearance, label_style, tx_color, rx_color)
        color_mode = color_mode if color_mode is not None else presentation.mpc["color_mode"]
        show_mpc_bounce_points = (
            presentation.mpc["bounce_points"]
            if show_mpc_bounce_points is None
            else show_mpc_bounce_points
        )
        line_width = presentation.mpc["line_width"] if line_width is None else line_width
        point_size = presentation.mpc["point_size"] if point_size is None else point_size
        scene = self._build_static_scene(presentation, ibl_name, ibl_intensity)
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
            color_palette=presentation.palette,
        )
        self._add_dynamic(
            scene, raw_frame, view_model, presentation, line_width, point_size, node_radius
        )
        frame_tx, frame_rx = self._extract_node_positions(raw_frame)

        # ---- Camera ----
        camera = gfx.PerspectiveCamera(60, aspect=16 / 9)

        mesh_bboxes = self._extract_mesh_bboxes()
        orbit = compute_camera_orbit(
            mesh_bboxes, frame_tx, frame_rx, azimuth, elevation, distance, center
        )
        self._apply_camera_orbit(
            camera, scene, orbit, explicit_framing=center is not None or distance is not None
        )

        return scene, camera

    def _add_node_labels(self, scene, tx_positions, rx_positions, presentation) -> None:
        """Use the same native text, callout, and card layout as the desktop."""
        presentation.add_node_labels(scene, tx_positions, rx_positions)

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
    def _apply_camera_orbit(
        camera: Any, scene: Any, orbit: Any, *, explicit_framing: bool = False
    ) -> None:
        """Honor requested framing, or fit the whole scene for the default view.

        ``show_object`` replaces the camera position as it fits scene bounds.
        ``show_pos`` preserves the requested eye and establishes the chosen
        orbit center. Empty scenes retain the automatic bounding-sphere fallback.
        """
        eye = orbit.eye_position()
        view_dir = tuple(orbit.center - eye)
        camera.local.position = tuple(eye)
        if explicit_framing:
            camera.show_pos(tuple(orbit.center), up=(0, 0, 1))
            return
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
        *,
        color_palette=None,
    ) -> tuple[dict | None, Any]:
        """Load a visual frame payload and build its view model.

        Returns:
            ``(frame_payload, view_model)``; either may be ``None``.
        """
        frame_payload = None
        available = self.frames
        if frame not in available:
            # Preserve the default scene-only preview before frames exist.
            if available or frame != 0:
                raise ValueError(f"Frame {frame} is not available. Use viz.frames to list frames.")
        elif self._frame_source is not None:
            frame_payload = load_notebook_visual_frame(
                self._frame_source,
                frame,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
            )

        view_model = None
        if frame_payload is not None:
            view_model = self._mpc_core.create_view_model(
                step=frame,
                raw_frame=frame_payload,
                color_mode=color_mode,
                color_palette=color_palette,
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
