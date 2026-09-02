"""Open3D/Filament renderer implementation for ORCHAV.

The backend wraps Open3D's ``O3DVisualizer`` in the shared renderer protocol.
It runs as a separate native window from the Qt controls, so runtime mixins own
the Open3D GUI event pump while app startup owns the Qt shell.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from shared.logging import get_logger

from ...materials.catalog import ITU_TO_PBR  # noqa: F401 (re-exported for external consumers)
from ...scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR_RGBA
from ...types.camera_state import CameraState
from ..base import RendererBase
from ..protocol import RendererCapabilities
from .camera import Open3DCameraMixin
from .capture import Open3DCaptureMixin
from .geometry import Open3DGeometryMixin
from .lighting import (
    PBR_USE_BASE_PREFIX,  # noqa: F401 (re-exported)
    MaterialLightingMixin,
)
from .lighting_controls import Open3DLightingControlsMixin
from .materials import Open3DMaterialMixin
from .mpc import EMPTY_COLORS_3D, EMPTY_LINES_2D, EMPTY_POINTS_3D, Open3DMpcMixin
from .runtime import Open3DRuntimeMixin
from .scene_controls import Open3DSceneControlsMixin
from .surface_overlays import Open3DSurfaceOverlayMixin, _AppliedBeamformingSurface
from .trajectories import Open3DTrajectoryMixin

try:
    from PySide6.QtCore import QTimer

    HAS_QTIMER = True
except ImportError:
    QTimer = None  # type: ignore[assignment]
    HAS_QTIMER = False

if TYPE_CHECKING:
    from ....visualizer import OrchavVisualizer
    from ...pipeline.core import FrameRenderPacket

logger = get_logger("orchav.renderer_open3d")


def _create_gui_pump_timer(callback: Any, interval_ms: int) -> Any:
    """Create the ordinary Qt timer that pumps Open3D's native GUI loop.

    Open3D owns a separate animation deadline with the same nominal interval.
    A precise external timer can repeatedly arrive just before that deadline
    and starve Open3D's animation callback.  Qt's default timer avoids coupling
    the two schedulers while still pumping native input near 60 Hz.
    """
    if QTimer is None:
        return None
    timer = QTimer()
    timer.timeout.connect(callback)
    timer.start(int(interval_ms))
    return timer


class Open3DRenderer(
    Open3DMpcMixin,
    Open3DSceneControlsMixin,
    Open3DRuntimeMixin,
    Open3DCaptureMixin,
    Open3DCameraMixin,
    Open3DGeometryMixin,
    Open3DMaterialMixin,
    Open3DSurfaceOverlayMixin,
    Open3DTrajectoryMixin,
    Open3DLightingControlsMixin,
    MaterialLightingMixin,
    RendererBase,
):
    """Renderer protocol implementation backed by ``O3DVisualizer``.

    The class composes mixins by responsibility. Shared services should treat
    it through ``RendererProtocol`` and ``renderer_capabilities``; Open3D-only
    compatibility branches belong inside this package.
    """

    capabilities = RendererCapabilities(
        pbr=True,
        material_clearcoat=True,
        material_emissive=True,
        material_anisotropy=True,
        material_transmission=True,
        material_volume_thickness=True,
        material_normal_map=True,
        scene_shader=True,
        frustum_culling=True,
        shadow_toggle=True,
        open3d_settings_panel=True,
        camera_lookat=True,
        transparency=True,
        line_width=True,
        ibl=True,
        fly_mode=True,
        trajectories=True,
        screenshot_export=True,
        skybox=True,
        axes=True,
        static_mesh_batching=True,
        physical_window_size=True,
    )

    # Stable names used by frame-packet application and reset/visibility cleanup.
    MPC_LINES_NAME = "mpc_lines"
    MPC_POINTS_NAME = "mpc_points"
    COVERAGE_MESH_NAME = "coverage_mesh"
    COVERAGE_ISOLINES_NAME = "coverage_isolines"
    TRAJECTORY_TX_LINES_NAME = "trajectory_tx_lines"
    TRAJECTORY_TX_POINTS_NAME = "trajectory_tx_points"
    TRAJECTORY_RX_LINES_NAME = "trajectory_rx_lines"
    TRAJECTORY_RX_POINTS_NAME = "trajectory_rx_points"
    TRAJECTORY_TARGET_LINES_PREFIX = "trajectory_target_lines_"
    TRAJECTORY_TARGET_POINTS_PREFIX = "trajectory_target_points_"

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Initialize Open3D renderer state before the native window exists."""
        super().__init__(visualizer)

        # Open3D-native MPC and beamforming geometry state is kept separate
        # from declarative RenderObject state because these paths update
        # mutable Open3D objects directly for frame-to-frame performance.
        self.mpc_pcd = o3d.geometry.PointCloud()
        self.mpc_lineset = o3d.geometry.LineSet()
        self._applied_beamforming_surfaces: dict[str, _AppliedBeamformingSurface] = {}
        self._beamforming_owned_names: set[str] = set()

        self._o3d_vis: Optional[o3d.visualization.O3DVisualizer] = None
        self._gui_timer: Optional[QTimer] = None
        self._gui_initialized = False
        self._closing_programmatically = False
        self._deferred_render_turn_callbacks: list[Any] = []
        self._render_turn_lifecycle_generation: int = 0
        # Open3D exposes event-loop liveness, but not a Python callback for
        # Filament frame acceptance or physical presentation completion.
        self._draw_pump_attempts: int = 0
        self._draw_pump_alive: int = 0
        self._native_redraw_pending: bool = False
        self._benchmark_telemetry_active: bool = False
        self._benchmark_telemetry_baseline: dict[str, int] = {}
        self._benchmark_frame_submissions: int = 0
        self._benchmark_redraw_pump_attempts: int = 0
        self._benchmark_redraw_pump_alive: int = 0
        self._geometry_names: set[str] = set()
        self._geometry_id_to_name: dict[int, str] = {}
        self._geometry_types: dict[str, str] = {}
        self._pbr_materials: dict[str, rendering.MaterialRecord] = {}
        self._texture_image_cache: dict[str, o3d.geometry.Image] = {}
        self._texture_source_identities: dict[str, str] = {}
        self._applied_render_objects: dict[str, Any] = {}
        # Low-level scene visibility is applied immediately, while the
        # high-level O3DVisualizer bookkeeping is flushed at batch/frame
        # boundaries to prevent per-object native redraws from exposing a
        # partially updated label or orientation-frame group.
        self._pending_o3d_visualizer_visibility: dict[str, bool] = {}
        self._last_coverage_signature: Optional[str] = None
        self._applied_coverage_state = None
        self._line_width = 2.0
        self._edge_line_width = 1.0
        self._point_size = 5.0
        self.trajectory_line_width = 3.0
        self.trajectory_point_size = 6.0

        # Shadow state is stored so UI state can be reapplied through the
        # Open3D view-level API after scene changes.
        self._shadows_enabled: bool = True
        self._shadow_type: str = "PCF"

        # Disable culling by default to avoid geometry pop-in in large outdoor
        # scenarios until the user explicitly opts in.
        self._culling_enabled: bool = False

        # Re-applying IBL intensity is the backend's reliable scene-dirty hook.
        self._ibl_intensity = 30000.0
        self._ibl_name = "default"
        self._ibl_rotation_deg = 0.0
        self._visibility_settle_redraw_pending: bool = False
        self._skybox_visible: bool = False
        self._render_debug_enabled: bool = logger.isEnabledFor(logging.DEBUG) or os.getenv(
            "ORCHAV_RENDER_DEBUG", ""
        ).lower() in ("1", "true", "yes")
        self._camera_debug_enabled: bool = os.getenv("ORCHAV_CAMERA_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._follow_debug_seq: int = 0
        self._follow_debug_baseline_radius: Optional[float] = None
        self._follow_debug_last_radius: Optional[float] = None
        self._follow_debug_last_target: Optional[np.ndarray] = None
        self._follow_debug_last_requested_eye: Optional[np.ndarray] = None
        self._follow_locked_radius: Optional[float] = None
        self._follow_last_forward: Optional[np.ndarray] = None
        self._camera_command_seq: int = 0
        self._camera_observe_last_fp: Optional[tuple[float, ...]] = None
        self._camera_observe_last_eye: Optional[np.ndarray] = None
        self._camera_observe_last_forward: Optional[np.ndarray] = None
        self._camera_observe_last_fov: Optional[float] = None
        self._camera_observe_last_command_seq: int = 0

        self._line_material = rendering.MaterialRecord()
        self._line_material.shader = "unlitLine"
        self._line_material.line_width = self._line_width

        self._edge_material = rendering.MaterialRecord()
        self._edge_material.shader = "unlitLine"
        self._edge_material.line_width = self._edge_line_width

        # Scene-edge line width is tracked separately from MPC line width.
        self._edge_geometry_names: set[str] = set()

        self._point_material = rendering.MaterialRecord()
        self._point_material.shader = "defaultUnlit"
        self._point_material.point_size = self._point_size

        self._mesh_material = rendering.MaterialRecord()
        self._mesh_material.shader = "defaultLit"

        # Text meshes have no useful normals, so labels stay unlit.
        self._label_material = rendering.MaterialRecord()
        self._label_material.shader = "defaultUnlit"

        # Follow mode retains look-at so target motion preserves camera offset.
        self._stored_lookat: Optional[np.ndarray] = None
        # Best known eye->lookat distance; used to bootstrap Follow from POV-like states.
        self._last_camera_look_distance: float = 10.0
        self._last_explicit_camera_state: Optional[CameraState] = None

        # Batch mode defers redraws until a frame or logical update is complete.
        self._batch_mode: bool = False
        self._batch_redraw_pending: bool = False

        # Redraw requests are coalesced while a frame transaction is open.
        # The Qt-driven Open3D pump cannot interleave with synchronous frame
        # mutation on the same thread, so its timer remains armed.
        self._frame_update_in_progress: bool = False
        self._frame_redraw_pending: bool = False
        self._frame_update_start_time: float = 0.0

        # Preserve visibility choices when geometry has to be re-uploaded.
        self._hidden_geometry_names: set[str] = set()
        self._pending_hidden_geometry_names: set[str] = set()

        # Open3D's separately pumped native event loop runs near 60 Hz.
        self._gui_timer_interval_ms: int = 16

        # Node position cache avoids redundant GPU transform updates.
        self._geometry_position_cache: dict[str, tuple[float, float, float]] = {}

        # Cached custom ground grid geometry for show/hide toggles.
        self._ground_grid_geometry: Optional[o3d.geometry.LineSet] = None

        # Upload center cache: vertex center snapshot at the moment the geometry
        # was last uploaded to the GPU. Object-based fast paths must compare
        # against this (not ``geometry.get_center()``) because CPU geometry may
        # be mutated after upload.
        self._geometry_upload_center: dict[str, np.ndarray] = {}

        logger.info("Open3DRenderer initialized (Open3D O3DVisualizer backend)")

    @property
    def renderer_type(self) -> str:
        """Return the stable backend identifier for Open3D."""
        return "open3d"

    def set_shadow_enabled(self, enabled: bool) -> bool:
        """Apply the default Open3D shadow type through the lighting mixin."""
        return self.set_shadowing(bool(enabled), "PCF")

    def initialize_visualizer(
        self,
        window_name: str = "ORCHAV",
        width: int = 1024,
        height: int = 768,
        left: int = -1,
        top: int = -1,
        suppress_default_camera: bool = False,
        *,
        host_parent: Any = None,
    ) -> Any:
        """Initialize the Open3D native window and event-pump integration."""
        if host_parent is not None:
            raise ValueError("Open3D does not support an embedded Qt viewport")
        if not self._gui_initialized:
            gui.Application.instance.initialize()
            self._gui_initialized = True
            logger.debug("Open3DRenderer: GUI Application initialized")

        if self._o3d_vis is None:
            self._o3d_vis = o3d.visualization.O3DVisualizer(window_name, width, height)
            self._o3d_vis.line_width = int(self._line_width)
            self._o3d_vis.point_size = int(self._point_size)
            self._install_window_close_callback()

            self._o3d_vis.set_background(DEFAULT_SCENE_BACKGROUND_COLOR_RGBA, None)

            if not suppress_default_camera:
                self._o3d_vis.reset_camera_to_default()
            gui.Application.instance.add_window(self._o3d_vis)

            self._o3d_vis.show(True)

            # Position window after it is realized so the window manager has
            # assigned decorations and os_frame coordinates are reliable.
            if left >= 0 and top >= 0:
                self._o3d_vis.os_frame = gui.Rect(left, top, width, height)
            self._render_debug(
                "init_visualizer",
                open3d_version=o3d.__version__,
                has_set_ibl=hasattr(self._o3d_vis, "set_ibl"),
                has_set_ibl_intensity=hasattr(self._o3d_vis, "set_ibl_intensity"),
                has_set_ibl_rotation=hasattr(self._o3d_vis, "set_ibl_rotation"),
                has_show_skybox=hasattr(self._o3d_vis, "show_skybox"),
                has_animation_tick=hasattr(self._o3d_vis, "set_on_animation_tick"),
            )
            scene_widget = self._o3d_vis.scene
            scene = scene_widget.scene if scene_widget is not None else None
            if scene is not None:
                self._render_debug(
                    "scene_caps",
                    enable_sun_light=hasattr(scene, "enable_sun_light"),
                    enable_indirect_light=hasattr(scene, "enable_indirect_light"),
                    set_indirect_light_rotation=hasattr(scene, "set_indirect_light_rotation"),
                    set_skybox_rotation=hasattr(scene, "set_skybox_rotation"),
                )

            try:
                self.set_ibl("default")
                self.set_ibl_intensity(30000)
                self.show_skybox(self._skybox_visible)
                logger.info("Open3DRenderer: Default IBL lighting initialized")
            except (RuntimeError, OSError) as exc:
                logger.debug(f"Open3DRenderer: Could not set up default IBL: {exc}")

            # Open3D's default far plane is too small for outdoor RF scenarios.
            self._set_far_clipping_plane(100000.0)

            # Open3D/Filament can be slow over X11 forwarding; prefer local or
            # VNC rendering for this backend in remote workflows.
            if HAS_QTIMER:
                self._gui_timer_interval_ms = 16
                self._gui_timer = _create_gui_pump_timer(
                    self._tick_o3d_gui,
                    self._gui_timer_interval_ms,
                )
                logger.info("Open3DRenderer: Started QTimer for O3D GUI updates (16ms interval)")

            self.vis = self._o3d_vis
            self.vis_initialized = True
            logger.info(f"Open3DRenderer: Created O3DVisualizer window '{window_name}'")

        return self._o3d_vis

    def _schedule_app_close(self, callback: Any) -> None:
        """Schedule *callback* through Qt when a Qt event loop is available."""
        if HAS_QTIMER:
            QTimer.singleShot(0, callback)
        else:
            callback()

    def _request_app_close_from_render_window(self) -> None:
        """Treat a user Open3D window close as a request to close the app."""

        def _close_app() -> None:
            close = getattr(self.visualizer, "close", None)
            if callable(close):
                close()
                return
            app = getattr(gui.Application, "instance", None)
            if app is not None:
                try:
                    gui.Application.instance.quit()
                except RuntimeError:
                    logger.debug("Open3DRenderer: failed to quit GUI app", exc_info=True)

        self._schedule_app_close(_close_app)

    def _install_window_close_callback(self) -> None:
        """Close the Qt shell when the native Open3D renderer window is closed."""
        if self._o3d_vis is None or not hasattr(self._o3d_vis, "set_on_close"):
            return

        def _on_close() -> bool:
            if self._closing_programmatically:
                return True
            self._request_app_close_from_render_window()
            return False

        self._o3d_vis.set_on_close(_on_close)

    def apply_frame(self, packet: "FrameRenderPacket") -> bool:
        """Apply frame-heavy renderer-native data from a new packet.

        Persistent TX/RX objects and labels are synchronized through
        ``ensure_object`` by the shared entity services, which remain their
        single owner. Updates here are batched to defer redraw until the end.
        The desired packet becomes the applied baseline only when every
        failure-reporting overlay domain reaches the backend.
        """
        import time as _time

        try:
            with self.batch_updates():
                if self.last_frame_packet is None:
                    # First render uploads every renderer-native packet domain.
                    t0 = _time.perf_counter()
                    mpc_succeeded = self._apply_mpc_data(packet)
                    t1 = _time.perf_counter()
                    coverage_succeeded = self._apply_coverage_data(packet)
                    t2 = _time.perf_counter()
                    self._apply_colorbar(packet.colorbar)
                    t3 = _time.perf_counter()
                    self._apply_stats(packet.stats_text)
                    t4 = _time.perf_counter()
                    logger.debug(
                        "⏱ [PERF] apply (first): mpc=%.1fms cov=%.1fms "
                        "colorbar=%.1fms stats=%.1fms",
                        (t1 - t0) * 1000,
                        (t2 - t1) * 1000,
                        (t3 - t2) * 1000,
                        (t4 - t3) * 1000,
                    )
                else:
                    # Later frames update only changed renderer-native domains.
                    t0 = _time.perf_counter()
                    mpc_succeeded = self._apply_mpc_data_diff(
                        self.last_frame_packet,
                        packet,
                    )
                    t1 = _time.perf_counter()
                    coverage_succeeded = self._apply_coverage_data_diff(
                        self.last_frame_packet,
                        packet,
                    )
                    t2 = _time.perf_counter()
                    self._apply_colorbar_diff(
                        self.last_frame_packet.colorbar,
                        packet.colorbar,
                    )
                    t3 = _time.perf_counter()
                    self._apply_stats_diff(
                        self.last_frame_packet.stats_text,
                        packet.stats_text,
                    )
                    t4 = _time.perf_counter()
                    logger.debug(
                        "⏱ [PERF] apply (diff): mpc=%.1fms cov=%.1fms "
                        "colorbar=%.1fms stats=%.1fms",
                        (t1 - t0) * 1000,
                        (t2 - t1) * 1000,
                        (t3 - t2) * 1000,
                        (t4 - t3) * 1000,
                    )

                t_bf_start = _time.perf_counter()
                beamforming_succeeded = self._apply_beamforming(packet)
                t_bf_end = _time.perf_counter()
                if (t_bf_end - t_bf_start) * 1000 > 1:
                    logger.debug(
                        "⏱ [PERF] beamforming: %.1fms",
                        (t_bf_end - t_bf_start) * 1000,
                    )

            succeeded = (
                bool(mpc_succeeded) and bool(coverage_succeeded) and bool(beamforming_succeeded)
            )
            if succeeded:
                self.last_frame_packet = packet
            return succeeded

        except (RuntimeError, ValueError, AttributeError, KeyError):
            logger.exception("Open3DRenderer: Error in apply_frame")
            return False

    def _apply_colorbar(self, colorbar: Optional[tuple]) -> None:
        """Leave colorbar rendering to the Qt UI layer."""
        if colorbar and hasattr(self.visualizer, "colorbar_widget"):
            title, (min_val, max_val) = colorbar
            pass

    def _apply_colorbar_diff(
        self,
        old_colorbar: Optional[tuple],
        new_colorbar: Optional[tuple],
    ) -> None:
        """Apply colorbar changes while tolerating NumPy values."""
        try:
            changed = old_colorbar != new_colorbar
            if hasattr(changed, "__len__") and not isinstance(changed, bool):
                changed = True
        except ValueError:
            changed = True
        if changed:
            self._apply_colorbar(new_colorbar)

    def _apply_stats(self, stats_text: str) -> None:
        """Mirror frame statistics text into the Qt info label."""
        if hasattr(self.visualizer, "mpc_info_label"):
            self.visualizer.mpc_info_label.setText(stats_text)

    def _apply_stats_diff(self, old_stats: str, new_stats: str) -> None:
        """Apply only changed statistics."""
        if old_stats != new_stats:
            self._apply_stats(new_stats)

    def get_native_asset_cache_info(self) -> dict[str, Any]:
        """Return Filament texture-cache inventory for lifecycle diagnostics."""
        return {
            "entries": len(self._texture_image_cache),
            # Open3D does not expose truthful native allocation sizes.
            "bytes": None,
            "max_bytes": None,
        }

    def clear_native_asset_cache(self) -> dict[str, int]:
        """Release reusable native texture handles without removing geometry."""
        entries = len(self._texture_image_cache)
        source_entries = len(self._texture_source_identities)
        self._texture_image_cache.clear()
        self._texture_source_identities.clear()
        return {"entries": entries, "source_entries": source_entries}

    def reset_state(self) -> None:
        """Clear scenario presentation state while retaining reusable native assets."""
        logger.info("Open3DRenderer: Resetting state")

        self.last_frame_packet = None
        self._visibility_settle_redraw_pending = False
        for name in list(self._geometry_names):
            self._remove_geometry(name)

        # Successful removals discard their own tracking. Preserve only
        # names whose native removal failed so reset can be retried without
        # losing ownership of live Filament geometry.
        remaining_names = set(self._geometry_names)
        for geometry_id, name in tuple(self._geometry_id_to_name.items()):
            if name not in remaining_names:
                self._geometry_id_to_name.pop(geometry_id, None)
        for mapping in (
            self._geometry_types,
            self._pbr_materials,
            self._geometry_position_cache,
            self._applied_render_objects,
        ):
            for name in tuple(mapping):
                if name not in remaining_names:
                    mapping.pop(name, None)
        self._hidden_geometry_names.intersection_update(remaining_names)
        self._pending_hidden_geometry_names.intersection_update(remaining_names)
        for name in tuple(self._pending_o3d_visualizer_visibility):
            if name not in remaining_names:
                self._pending_o3d_visualizer_visibility.pop(name, None)
        self._edge_geometry_names.intersection_update(remaining_names)
        self._last_coverage_signature = None
        self._applied_coverage_state = None
        self._beamforming_owned_names.intersection_update(remaining_names)
        for name in tuple(self._applied_beamforming_surfaces):
            if name not in self._beamforming_owned_names:
                self._applied_beamforming_surfaces.pop(name, None)
        self._ground_grid_geometry = None

        self.mpc_lineset.points = o3d.utility.Vector3dVector(EMPTY_POINTS_3D)
        self.mpc_lineset.lines = o3d.utility.Vector2iVector(EMPTY_LINES_2D)
        self.mpc_lineset.colors = o3d.utility.Vector3dVector(EMPTY_COLORS_3D)
        self.mpc_pcd.points = o3d.utility.Vector3dVector(EMPTY_POINTS_3D)
        self.mpc_pcd.colors = o3d.utility.Vector3dVector(EMPTY_COLORS_3D)

        if hasattr(self.visualizer, "_cached_bounce_points"):
            self.visualizer._cached_bounce_points = np.empty((0, 3), dtype=np.float64)
        if hasattr(self.visualizer, "_cached_bounce_colors"):
            self.visualizer._cached_bounce_colors = np.empty((0, 3), dtype=np.float64)

        if remaining_names:
            logger.warning(
                "Open3DRenderer: reset retained %d geometry name(s) after native "
                "removal failure; the next reset/frame sync will retry",
                len(remaining_names),
            )

        logger.info("Open3DRenderer: State reset complete")

    def run_event_loop(self) -> None:
        """Run the O3D GUI event loop (call from main thread)."""
        if self._gui_initialized:
            gui.Application.instance.run()

    def close(self) -> None:
        """Close the native Open3D window and release GUI resources."""
        self._render_turn_lifecycle_generation += 1
        self._deferred_render_turn_callbacks.clear()
        self._native_redraw_pending = False
        self._visibility_settle_redraw_pending = False
        self._pending_o3d_visualizer_visibility.clear()
        self._benchmark_telemetry_active = False
        # Stop the Qt timer before closing the native window so no tick races
        # with Open3D teardown.
        if self._gui_timer is not None:
            self._gui_timer.stop()
            self._gui_timer = None
            logger.info("Open3DRenderer: Stopped GUI timer")

        native_loop_running = bool(self._gui_initialized)
        if self._o3d_vis is not None:
            window = self._o3d_vis
            # Open3D retains a native shared_ptr until the next GUI turn.
            # Release every Python-owned reference before that turn so
            # CleanupAfterRunning() can destroy the final Filament window.
            self._o3d_vis = None
            self.vis = None
            self.vis_initialized = False
            try:
                self._closing_programmatically = True
                # ORCHAV initializes and pumps Open3D from Qt's main thread,
                # so close directly. Posting ``window.close`` would retain a
                # Python bound-method reference through the cleanup turn.
                window.close()
                del window

                # One native turn executes a queued close and releases the
                # Filament window. Closing Open3D's final window makes
                # run_one_tick() return False. Calling it again after that
                # terminal result can block indefinitely on Windows.
                if self._gui_initialized:
                    try:
                        native_loop_running = gui.Application.instance.run_one_tick() is not False
                    except RuntimeError as exc:
                        native_loop_running = False
                        logger.debug("Open3DRenderer: close tick stopped: %s", exc)
            except RuntimeError as exc:
                native_loop_running = False
                logger.debug("Open3DRenderer: Error closing window: %s", exc)
            finally:
                self._closing_programmatically = False
            logger.info("Open3DRenderer: Closed visualizer window")

        if self._gui_initialized:
            try:
                if native_loop_running:
                    gui.Application.instance.quit()
                    logger.info("Open3DRenderer: Quit GUI application")
                    try:
                        gui.Application.instance.run_one_tick()
                    except RuntimeError as exc:
                        logger.debug("Open3DRenderer: quit tick stopped: %s", exc)
                else:
                    logger.info("Open3DRenderer: Native GUI loop stopped with final window")
            except RuntimeError as exc:
                logger.debug("Open3DRenderer: Error quitting GUI application: %s", exc)
            self._gui_initialized = False

        self._geometry_names.clear()
        self._geometry_id_to_name.clear()
        self._geometry_types.clear()
        self._pbr_materials.clear()
        self.clear_native_asset_cache()
        self._applied_render_objects.clear()
        self._pending_hidden_geometry_names.clear()
        self._last_coverage_signature = None
        self._applied_coverage_state = None
        self._applied_beamforming_surfaces.clear()
        self._beamforming_owned_names.clear()
