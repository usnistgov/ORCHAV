"""pygfx/wgpu renderer backend for ORCHAV.

``PygfxRenderer`` implements the shared renderer protocol by composing focused
mixins for runtime, geometry, materials, MPCs, overlays, picking, and gizmos.
The class itself owns dependency loading, Qt canvas setup, and the high-level
frame-packet application sequence.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
import time
import weakref
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from ...backends.pygfx_scene_helpers import _is_pygfx_unlit_mode_enabled
from ...scene.defaults import DEFAULT_SCENE_BACKGROUND_COLOR_RGBA
from ...types.camera_state import CameraState
from ...types.render_payloads import (
    GeometryPayload,
    MaterialPayload,
)
from ..protocol import MpcPathSelectionCallback, RendererCapabilities
from .camera import PygfxCameraMixin
from .canvas import (
    _AA_MODES,
    _env_flag,
    _env_int,
    _parse_env_clipping_planes,
    _refresh_renderer_effect_passes,
    create_wgpu_renderer,
)
from .capture import PygfxCaptureMixin
from .geometry import PygfxGeometryMixin
from .labels import PygfxLabelMixin
from .lighting import PygfxIBLManager
from .lighting_profiles import INSPECTION_PROFILE, PYGFX_LIGHTING_PROFILES
from .materials import PygfxMaterialMixin
from .mpc import MpcExpandedLineCacheEntry, PygfxMpcMixin
from .mpc_selection import PygfxMpcSelectionMixin
from .overlays import PygfxOverlayMixin
from .picking import PygfxPickingMixin
from .rf_xray import PygfxRFXRayMixin
from .runtime import PygfxRuntimeMixin
from .scene_controls import PygfxSceneControlsMixin
from .surface_overlays import PygfxSurfaceOverlayMixin, _AppliedBeamformingSurface
from .trajectories import PygfxTrajectoryMixin
from .transform_gizmo import PygfxTransformGizmoMixin

if TYPE_CHECKING:
    from ....visualizer import OrchavVisualizer
    from ...pipeline.core import FrameRenderPacket

logger = logging.getLogger(__name__)


def _default_ibl_dir() -> Path:
    """Return the repo-level IBL asset directory shared with Open3D."""
    return Path(__file__).resolve().parents[4] / "libraries" / "ibl"


class PygfxRenderer(
    PygfxRuntimeMixin,
    PygfxCameraMixin,
    PygfxPickingMixin,
    PygfxMpcMixin,
    PygfxMpcSelectionMixin,
    PygfxSceneControlsMixin,
    PygfxCaptureMixin,
    PygfxGeometryMixin,
    PygfxMaterialMixin,
    PygfxOverlayMixin,
    PygfxLabelMixin,
    PygfxSurfaceOverlayMixin,
    PygfxRFXRayMixin,
    PygfxTransformGizmoMixin,
    PygfxTrajectoryMixin,
):
    """Renderer backend using pygfx + wgpu-py."""

    renderer_id = "pygfx"
    AXES_NAME = "orchav_axes"
    capabilities = RendererCapabilities(
        pbr=True,
        material_clearcoat=True,
        material_emissive=True,
        material_normal_map=True,
        clipping_planes=True,
        shadow_toggle=True,
        camera_lookat=True,
        transparency=True,
        line_width=True,
        ibl=True,
        fly_mode=True,
        camera_minimap=True,
        trajectories=True,
        screenshot_export=True,
        screen_space_labels=True,
        skybox=True,
        axes=True,
        aperture_preview=True,
        angular_preview=True,
        mpc_type_markers=True,
        mpc_path_inspection=True,
        rf_xray_overlay=True,
        viewport_hud=True,
        picking=True,
        transform_gizmo=True,
        hover_info=True,
        scenario_authoring=True,
        antialiasing=True,
        ground_grid=True,
        direct_lighting=True,
        lighting_profiles=True,
        wireframe=True,
        mesh_buffer_cache=True,
        mesh_vertex_stream_updates=True,
        static_mesh_batching=True,
        static_mesh_batch_object_threshold=400,
        static_mesh_batch_triangle_limit=500_000,
        static_mesh_batch_member_limit=1_024,
        prefer_float32_frame_data=True,
        embedded_viewport=True,
    )
    TRAJECTORY_TX_LINES_NAME = "trajectory_tx_lines"
    TRAJECTORY_TX_POINTS_NAME = "trajectory_tx_points"
    TRAJECTORY_RX_LINES_NAME = "trajectory_rx_lines"
    TRAJECTORY_RX_POINTS_NAME = "trajectory_rx_points"
    COVERAGE_MESH_NAME = "coverage_mesh"
    COVERAGE_ISOLINES_NAME = "coverage_isolines"
    BEAMFORMING_PREFIX = "beamforming:"
    TRAJECTORY_TARGET_LINES_PREFIX = "trajectory_target_lines_"
    TRAJECTORY_TARGET_POINTS_PREFIX = "trajectory_target_points_"

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Initialize pygfx state without creating the Qt canvas yet."""
        self.visualizer = visualizer
        self.last_frame_packet: Optional[FrameRenderPacket] = None

        try:
            import pygfx as gfx
            from PySide6 import QtWidgets as _QtWidgets  # noqa: F401
            from rendercanvas.qt import QRenderWidget as WgpuWidget
        except ImportError:
            # Older environments expose the Qt canvas through wgpu.gui.qt.
            try:
                import pygfx as gfx
                from wgpu.gui.qt import WgpuCanvas as WgpuWidget  # type: ignore
            except ImportError as inner_exc:
                raise ImportError(
                    "pygfx backend dependencies missing. Install the default runtime from "
                    "a cloned repository with: python -m pip install -e ."
                ) from inner_exc

        self._gfx = gfx
        self._WgpuWidget = WgpuWidget

        self._initialized: bool = False
        self._session_generation: int = 0
        self._qt_window_closed: bool = False
        self._qt_app_close_requested: bool = False
        self._qt_closing_programmatically: bool = False
        self._qt_destroyed_callbacks: list[Any] = []
        self._qt_lifecycle_connections: list[tuple[Any, Any]] = []
        self._container: Any = None
        self._owns_container: bool = True
        self._canvas: Any = None
        self._canvas_widget: Any = None
        self._canvas_draw_callback: Any = None
        self._renderer: Any = None
        self._scene: Any = None
        self._camera: Any = None
        self._controller: Any = None
        self._clear_color: tuple[float, float, float, float] = tuple(
            DEFAULT_SCENE_BACKGROUND_COLOR_RGBA
        )
        self._skybox_visible: bool = False
        self._solid_background: Optional[Any] = None
        self._unlit_mode_enabled: bool = _is_pygfx_unlit_mode_enabled()

        self._width: int = 0
        self._height: int = 0

        self._name_to_handle: dict[str, int] = {}
        self._handle_to_name: dict[int, str] = {}
        self._objects: dict[str, Any] = {}
        self._kinds: dict[str, str] = {}
        self._topology: dict[str, tuple[Any, ...]] = {}
        self._external_geometry_names: dict[int, str] = {}
        self._hidden: set[str] = set()
        self._edge_geometry_names: set[str] = set()
        self._geometry_color_sources: dict[str, Any] = {}
        self._texture_cache: dict[str, Any] = {}
        self._texture_source_identities: dict[str, str] = {}
        self._materials: dict[str, MaterialPayload] = {}
        self._material_apply_signatures: dict[str, tuple[Any, ...]] = {}
        self._transforms: dict[str, np.ndarray] = {}
        self._positions: dict[str, tuple[float, float, float]] = {}
        self._geometry_upload_center: dict[str, np.ndarray] = {}
        self._geometry_texcoords_available: dict[str, bool] = {}
        self._render_object_snapshots: dict[str, tuple[Any, bool, Any]] = {}
        self._dirty_render_object_geometry: set[str] = set()
        self._uncertain_mesh_index_buffers: set[str] = set()
        self._vertex_stream_array_tokens: OrderedDict[int, tuple[Any, int]] = OrderedDict()
        self._vertex_stream_next_array_token: int = 0
        self._vertex_stream_incompatible_transitions: OrderedDict[
            str, OrderedDict[tuple[Any, ...], None]
        ] = OrderedDict()
        self._vertex_stream_rebuild_names: set[str] = set()
        self._label_anchor_groups: dict[tuple[int, int, int], set[str]] = {}
        self._label_anchor_key_by_name: dict[str, tuple[int, int, int]] = {}
        self._label_anchor_by_name: dict[str, np.ndarray] = {}
        self._label_offset_by_name: dict[str, np.ndarray] = {}
        self._next_handle: int = 1

        self._camera_state: Optional[CameraState] = None
        self._active_controller_type: str = "orbit"  # "orbit" | "fly"
        self._follow_target_lookat: Optional[np.ndarray] = None
        self._payload_cache: dict[int, tuple[tuple[Any, ...], GeometryPayload]] = {}
        self._geometry_payload_cache_keys: dict[int, str] = {}
        self._mpc_lines_source_sig: Optional[tuple[Any, ...]] = None
        self._mpc_points_source_sig: Optional[tuple[Any, ...]] = None
        self._mpc_marker_cache_key: Optional[tuple[Any, ...]] = None
        self._mpc_marker_codes_buf: Optional[np.ndarray] = None
        self._initialize_mpc_path_inspection_state()
        self._last_coverage_signature: Optional[str] = None
        self._applied_coverage_state = None
        self._applied_beamforming_surfaces: dict[str, _AppliedBeamformingSurface] = {}
        self._beamforming_owned_names: set[str] = set()
        self._initialize_rf_xray_state()
        self._last_camera_look_distance: float = 10.0
        self._line_width: float = 2.0
        self._edge_line_width: float = 1.0
        self._point_size: float = 5.0
        self.trajectory_line_width: float = 3.0
        self.trajectory_point_size: float = 6.0

        self._created_at: float = time.perf_counter()
        self._first_present_at: Optional[float] = None
        self._initial_present_attempted: bool = False
        self._initial_present_succeeded: Optional[bool] = None
        self._initial_present_duration_ms: Optional[float] = None
        self._initial_present_error: Optional[str] = None
        self._render_attempts: int = 0
        self._render_successes: int = 0
        self._render_failures: int = 0
        self._event_pump_calls: int = 0
        self._redraw_requests: int = 0
        self._last_present_call_at: float = 0.0
        self._last_present_success_at: float = 0.0
        self._present_interval_sum_s: float = 0.0
        self._present_interval_samples: int = 0
        self._present_interval_sq_sum_s: float = 0.0
        self._present_interval_max_s: float = 0.0
        self._recent_present_intervals_s: list[float] = []
        self._last_present_was_animating: Optional[bool] = None
        self._frame_drop_count: int = 0
        self._last_update_call_at: float = 0.0
        self._update_interval_sum_s: float = 0.0
        self._update_interval_samples: int = 0
        self._update_calls_while_animating: int = 0
        self._draw_callbacks_received: int = 0
        self._forced_draw_fallbacks: int = 0
        self._draw_durations: list[float] = []
        self._draw_callback_total_durations: list[float] = []
        self._last_renderer_submit_ms: float = 0.0
        self._last_draw_callback_total_ms: float = 0.0
        self._benchmark_telemetry_baseline: dict[str, int] = {}
        self._blocking_frame_count: int = 0
        self._blocking_force_draw_callbacks: int = 0
        self._blocking_force_draw_contaminated: int = 0
        self._pending_update_start: Optional[float] = None
        self._update_to_present_times: list[float] = []
        self._max_fps: float = self._read_max_fps()
        self._min_frame_dt_s: float = 0.0 if self._max_fps <= 0.0 else (1.0 / self._max_fps)
        self._canvas_update_mode: str = "native"
        self._canvas_max_fps: Optional[float] = None
        self._canvas_vsync: Optional[bool] = None
        self._canvas_present_method_requested: str = self._implicit_canvas_present_method()
        self._canvas_present_method: str = "unresolved"
        self._canvas_present_fallback_reason: Optional[str] = None
        self._canvas_refresh_rate_hz: float = 60.0
        self._canvas_uses_display_refresh: bool = False
        self._canvas_schedule_kwargs: dict[str, Any] = {}
        self._canvas_schedule_applied: bool = False
        self._qt_screen_changed_signal: Any = None
        self._qt_screen_changed_callback: Any = None
        self._batch_mode: bool = False
        self._batch_redraw_pending: bool = False

        # Tick-based FPS tracking (decoupled from draw path)
        self._tick_count: int = 0
        self._tick_interval_sum_s: float = 0.0
        self._tick_interval_samples: int = 0
        self._last_tick_at: float = 0.0

        # Canvas-driven render loop pause flag (for atomic frame updates)
        self._frame_update_paused: bool = False

        # Clipping planes (cutaway mode). Each entry is an (nx, ny, nz, d)
        # tuple in world coordinates. pygfx discards fragments where
        # dot(world_pos, normal) < d, so (nx, ny, nz, d) keeps the half-space
        # n·p >= d. Set via set_clipping_planes(); applied to every material.
        # The env-var hook lets benchmark / headless runs exercise the
        # clipping path without going through the UI panel.
        self._clipping_planes: tuple[tuple[float, float, float, float], ...] = (
            _parse_env_clipping_planes()
        )

        # Capture/debug passes. AA is GUI-controlled; depth remains an
        # env-var-only debug path until it has normalized visual output.
        env_aa_mode = os.environ.get("ORCHAV_PYGFX_AA_MODE", "").strip().lower()
        self._aa_mode: str = env_aa_mode if env_aa_mode in _AA_MODES else "off"
        self._depth_pass_enabled: bool = _env_flag("ORCHAV_PYGFX_ENABLE_DEPTH_PASS", False)
        if _env_flag("ORCHAV_PYGFX_MPC_MARKERS", False):
            app_state = getattr(self.visualizer, "app_state", None)
            if app_state is not None and hasattr(app_state, "show_mpc_type_markers"):
                try:
                    self.visualizer.app_state = replace(app_state, show_mpc_type_markers=True)
                except (TypeError, AttributeError):
                    try:
                        setattr(app_state, "show_mpc_type_markers", True)
                    except (TypeError, AttributeError):
                        pass
        # Normal-lines debug overlay (env-flag only, no UI). Sibling meshes
        # parented next to each scene mesh in the scene graph.
        self._normal_line_overlays: dict[str, Any] = {}
        # Ground grid overlay (paper / demo helper). Lazy-built on first
        # toggle so cold startup stays unchanged.
        self._ground_grid_obj: Any = None
        self._ground_grid_visible: bool = False
        self._ground_grid_needs_rebuild: bool = False

        # Wireframe overlay mode (1.7)
        self._wireframe_enabled: bool = False
        self._frame_update_start: float = 0.0
        self._last_end_frame_update_breakdown: dict[str, float] = {}
        self._frame_update_metrics: dict[str, float] = {}
        self._last_end_frame_update_breakdown_bytes: dict[str, float] = {}
        self._frame_update_bytes: dict[str, float] = {}

        self._ambient_light: Any = None
        self._key_light: Any = None
        self._fill_light: Any = None
        self._head_light: Any = None
        inspection_lighting = PYGFX_LIGHTING_PROFILES[INSPECTION_PROFILE]
        self._shadows_enabled: bool = inspection_lighting.shadows_enabled
        self._scene_extent: float = 200.0
        self._headlight_enabled: bool = inspection_lighting.headlight_enabled
        self._headlight_follow_camera: bool = True
        self._headlight_intensity: float = inspection_lighting.headlight_intensity
        self._last_headlight_cam_key: Optional[tuple[float, ...]] = None
        self._lighting_profile_name: str = INSPECTION_PROFILE
        self._suppress_lighting_profile_custom: bool = False
        self._ibl_name: str = "default"
        # Keep pygfx's default IBL subtle: neutral_outdoor has a sky tint, and
        # high IBL values can make inspection colors read as the wrong material.
        # Users can raise this from the Render panel when they want stronger
        # environment lighting.
        self._ibl_intensity: float = inspection_lighting.ibl_intensity
        defer_default_ibl_env = os.environ.get(
            "ORCHAV_PYGFX_DEFER_DEFAULT_IBL",
            "1",
        )
        self._defer_default_ibl: bool = defer_default_ibl_env != "0"
        self._deferred_default_ibl_name: Optional[str] = None
        self._deferred_ibl_load_scheduled: bool = False
        self._base_ambient_intensity: float = inspection_lighting.ambient_intensity
        self._base_key_intensity: float = inspection_lighting.key_intensity
        self._base_fill_intensity: float = inspection_lighting.fill_intensity
        self._key_light_azimuth_deg: float = inspection_lighting.key_azimuth_deg
        self._key_light_elevation_deg: float = inspection_lighting.key_elevation_deg
        self._fill_light_azimuth_deg: float = inspection_lighting.fill_azimuth_deg
        self._fill_light_elevation_deg: float = inspection_lighting.fill_elevation_deg

        self._ibl_manager = PygfxIBLManager(gfx, _default_ibl_dir())
        self._ibl_loaded: bool = False

        self._static_group: Any = None

        # Object picking & hover tooltip state
        self._pick_metadata: dict[str, dict] = {}
        self._reverse_objects: dict[int, str] = {}
        self._hover_info_mode: str = "essential"
        self._tooltip_label: Any = None
        self._hud_overlay_labels: dict[str, Any] = {}
        self._hud_overlay_specs: dict[str, dict[str, Any]] = {}
        self._semantic_legend_cache_key: Optional[tuple[Any, ...]] = None
        self._semantic_legend_cache_html: str = ""
        self._hud_suppressed: bool = False
        self._mpc_marker_legend_requested: bool = False
        self._visible_trajectory_kinds: set[str] = set()
        self._trajectory_hud_color_mode: str = "node_color"
        self._trajectory_hud_scalar_range: Optional[tuple[float, float]] = None
        self._minimap_enabled: bool = False
        self._minimap_camera: Any = None
        self._minimap_viewport: Any = None
        self._minimap_overlay_scene: Any = None
        self._minimap_overlay_camera: Any = None
        self._minimap_overlay_size: Optional[tuple[int, int]] = None
        self._minimap_overlay_objects: dict[str, Any] = {}
        self._minimap_world_rect: Optional[tuple[float, float, float, float]] = None
        self._last_hover_identity: Optional[tuple[str, int | None]] = None
        self._mpc_path_selection_callback: Optional[MpcPathSelectionCallback] = None
        self._mpc_path_click_candidate: Optional[dict[str, Any]] = None
        self._mpc_pick_segment_map_packet: Any = None
        self._mpc_pick_canonical_segment_indices = np.empty((0,), dtype=np.int32)

        self._transform_session_callback: Any = None
        self._transform_gizmo: Any = None
        self._transform_gizmo_target_name: Optional[str] = None
        self._transform_gizmo_target_kind: Optional[str] = None
        self._transform_gizmo_target_index: Optional[int] = None
        self._transform_gizmo_last_position: Optional[np.ndarray] = None
        self._transform_gizmo_last_transform: Optional[np.ndarray] = None
        self._transform_gizmo_control_object: Any = None
        self._transform_gizmo_target_proxy: Any = None
        self._scenario_authoring_runtime: Any = None
        self._pygfx_interaction_router: Any = None

        self._mpc_lines_points_buf: Optional[np.ndarray] = None
        self._mpc_lines_indices_buf: Optional[np.ndarray] = None
        self._mpc_lines_colors_buf: Optional[np.ndarray] = None
        self._mpc_segment_points_buf: Optional[np.ndarray] = None
        self._mpc_segment_colors_buf: Optional[np.ndarray] = None
        self._mpc_segment_indices_buf: Optional[np.ndarray] = None
        self._mpc_segment_capacity: int = 0
        self._mpc_segment_color_cols: int = 0
        self._mpc_segment_capacity_floor: int = max(
            0,
            _env_int("ORCHAV_PYGFX_MPC_LINE_PREALLOC_SEGMENTS", 0),
        )
        self._mpc_segment_capacity_hint: int = self._mpc_segment_capacity_floor
        cache_mb = max(0, _env_int("ORCHAV_PYGFX_MPC_LINE_CACHE_MB", 0))
        self._mpc_expanded_line_cache_max_bytes: int = cache_mb * 1024 * 1024
        self._mpc_expanded_line_cache: OrderedDict[tuple[Any, ...], MpcExpandedLineCacheEntry] = (
            OrderedDict()
        )
        self._mpc_expanded_line_cache_bytes: int = 0
        self._mpc_expanded_line_cache_hits: int = 0
        self._mpc_expanded_line_cache_misses: int = 0
        self._mpc_expanded_line_cache_stores: int = 0
        self._mpc_expanded_line_cache_evictions: int = 0
        self._mpc_expanded_line_cache_rejected_oversize: int = 0
        self._mpc_expanded_line_cache_largest_entry_bytes: int = 0
        self._mpc_expanded_line_cache_last_entry_bytes: int = 0
        self._mpc_expanded_line_cache_prewarm_enabled: bool = _env_flag(
            "ORCHAV_PYGFX_MPC_LINE_CACHE_PREWARM",
            True,
        )
        self._mpc_expanded_line_cache_prewarm_attempts: int = 0
        self._mpc_expanded_line_cache_prewarm_stores: int = 0
        self._mpc_expanded_line_cache_prewarm_existing: int = 0
        self._mpc_expanded_line_cache_prewarm_skips: int = 0
        self._mpc_expanded_line_cache_prewarm_total_ms: float = 0.0
        self._mpc_points_points_buf: Optional[np.ndarray] = None
        self._mpc_points_colors_buf: Optional[np.ndarray] = None
        self._mpc_point_capacity: int = 0
        self._mpc_point_color_cols: int = 0
        self._mpc_point_capacity_hint: int = 0

    @staticmethod
    def _read_canvas_max_fps(default: float = 60.0) -> float:
        """Read the rendercanvas scheduler FPS cap."""
        raw = os.environ.get("ORCHAV_PYGFX_CANVAS_MAX_FPS")
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid ORCHAV_PYGFX_CANVAS_MAX_FPS=%r; using %.1f FPS canvas cap",
                raw,
                default,
            )
            return float(default)

    @staticmethod
    def _implicit_canvas_present_method() -> str:
        """Return ORCHAV's presentation method when no override is supplied."""
        if sys.platform.startswith("win"):
            # Direct screen presentation is the accelerated desktop path on
            # Windows, but it requires a native surface.  Qt's offscreen
            # platform has no such surface, so use bitmap presentation for
            # explicit headless sessions instead of letting wgpu abort.
            qt_platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
            return "bitmap" if qt_platform == "offscreen" else "screen"
        return "auto"

    @staticmethod
    def _read_canvas_present_method() -> str:
        """Return the requested rendercanvas presentation policy.

        Direct screen presentation avoids a costly bitmap readback on Windows.
        Other platforms retain rendercanvas's compatibility-oriented automatic
        policy, while explicit ``screen``, ``bitmap``, and ``auto`` values take
        precedence everywhere.
        """
        implicit_method = PygfxRenderer._implicit_canvas_present_method()
        raw = os.environ.get("ORCHAV_PYGFX_PRESENT_METHOD")
        if raw is None:
            return implicit_method
        value = raw.strip().lower()
        if value in {"screen", "bitmap", "auto"}:
            return value
        logger.warning(
            "Invalid ORCHAV_PYGFX_PRESENT_METHOD=%r; using platform default %r",
            raw,
            implicit_method,
        )
        return implicit_method

    @classmethod
    def _canvas_schedule_config(
        cls,
        *,
        display_refresh_hz: float = 60.0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return constructor kwargs and normalized canvas scheduler metadata."""
        refresh_hz = float(display_refresh_hz) if display_refresh_hz > 0.0 else 60.0
        raw_mode = os.environ.get("ORCHAV_PYGFX_CANVAS_SCHEDULER")
        mode = "" if raw_mode is None else raw_mode.strip().lower()
        if not mode:
            if "ORCHAV_PYGFX_CANVAS_MAX_FPS" in os.environ:
                mode = "fastest" if cls._read_canvas_max_fps(refresh_hz) <= 0.0 else "ondemand"
            else:
                mode = "ondemand"

        if mode in {"native", "auto"}:
            config: dict[str, Any] = {
                "update_mode": "native",
                "max_fps": None,
                "vsync": None,
                "uses_display_refresh": False,
            }
            return {}, config

        if mode not in {"manual", "ondemand", "fastest"}:
            logger.warning(
                "Invalid ORCHAV_PYGFX_CANVAS_SCHEDULER=%r; using on-demand scheduling",
                raw_mode,
            )
            mode = "ondemand"

        if mode == "manual":
            config = {
                "update_mode": "manual",
                "max_fps": None,
                "vsync": False,
                "uses_display_refresh": False,
            }
            return {
                "update_mode": config["update_mode"],
                "vsync": config["vsync"],
            }, config

        uses_display_refresh = "ORCHAV_PYGFX_CANVAS_MAX_FPS" not in os.environ
        max_fps = cls._read_canvas_max_fps(refresh_hz)
        if mode == "fastest" or max_fps <= 0.0:
            config = {
                "update_mode": "fastest",
                "max_fps": None,
                "vsync": False,
                "uses_display_refresh": False,
            }
            schedule_kwargs = {
                "update_mode": config["update_mode"],
                "vsync": config["vsync"],
            }
        else:
            config = {
                "update_mode": "ondemand",
                "max_fps": max_fps,
                "vsync": True,
                "uses_display_refresh": uses_display_refresh,
            }
            schedule_kwargs = {
                "update_mode": config["update_mode"],
                "max_fps": config["max_fps"],
                "vsync": config["vsync"],
            }
        return schedule_kwargs, config

    def _current_screen_refresh_rate(self, screen: Any = None) -> float:
        """Return a usable Qt screen refresh rate, falling back to 60 Hz."""
        candidate = screen
        if candidate is None and self._container is not None:
            try:
                candidate = self._container.screen()
            except (AttributeError, RuntimeError):
                candidate = None
        if candidate is None:
            try:
                from PySide6.QtWidgets import QApplication

                app = QApplication.instance()
                candidate = app.primaryScreen() if app is not None else None
            except (ImportError, AttributeError, RuntimeError):
                candidate = None
        try:
            refresh_hz = float(candidate.refreshRate())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            refresh_hz = 60.0
        if not np.isfinite(refresh_hz) or refresh_hz < 24.0 or refresh_hz > 480.0:
            return 60.0
        return refresh_hz

    @staticmethod
    def _filter_supported_canvas_kwargs(widget_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop scheduler kwargs unsupported by the installed Qt canvas class."""
        try:
            signature = inspect.signature(widget_cls)
        except (TypeError, ValueError):
            return dict(kwargs)

        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return dict(kwargs)
        return {key: value for key, value in kwargs.items() if key in params}

    def _create_canvas_widget(self, *, present_method: Optional[str] = None) -> Any:
        """Create the Qt canvas widget with version-tolerant scheduler kwargs."""
        refresh_hz = self._current_screen_refresh_rate()
        schedule_kwargs, config = self._canvas_schedule_config(
            display_refresh_hz=refresh_hz,
        )
        requested_present_method = present_method or self._read_canvas_present_method()
        self._canvas_present_method_requested = requested_present_method
        canvas_kwargs = dict(schedule_kwargs)
        if requested_present_method != "auto":
            canvas_kwargs["present_method"] = requested_present_method
        supported_canvas_kwargs = self._filter_supported_canvas_kwargs(
            self._WgpuWidget,
            canvas_kwargs,
        )
        canvas_cls = self._guarded_canvas_widget_class()
        kwargs = {"parent": self._container}
        kwargs.update(supported_canvas_kwargs)

        try:
            canvas = canvas_cls(**kwargs)
        except TypeError as initial_error:
            if not supported_canvas_kwargs:
                raise
            logger.warning(
                "Pygfx canvas rejected constructor kwargs %s; retrying compatible subsets",
                supported_canvas_kwargs,
                exc_info=True,
            )
            presentation_kwargs = {
                key: value
                for key, value in supported_canvas_kwargs.items()
                if key == "present_method"
            }
            scheduler_kwargs = {
                key: value
                for key, value in supported_canvas_kwargs.items()
                if key in {"update_mode", "max_fps", "vsync"}
            }
            retry_options = [presentation_kwargs, scheduler_kwargs, {}]
            canvas = None
            for candidate in retry_options:
                if candidate == supported_canvas_kwargs:
                    continue
                try:
                    canvas = canvas_cls(parent=self._container, **candidate)
                except TypeError:
                    continue
                supported_canvas_kwargs = candidate
                break
            if canvas is None:
                raise initial_error

        if requested_present_method != "auto" and "present_method" not in supported_canvas_kwargs:
            self._canvas_present_fallback_reason = (
                "Canvas constructor rejected explicit "
                f"present_method={requested_present_method!r}; rendercanvas selected the mode"
            )

        applied_schedule_kwargs = {
            key: value
            for key, value in supported_canvas_kwargs.items()
            if key in {"update_mode", "max_fps", "vsync"}
        }

        if applied_schedule_kwargs != schedule_kwargs:
            config = {
                "update_mode": applied_schedule_kwargs.get("update_mode", "native"),
                "max_fps": applied_schedule_kwargs.get("max_fps"),
                "vsync": applied_schedule_kwargs.get("vsync"),
                "uses_display_refresh": bool(
                    config["uses_display_refresh"] and "max_fps" in applied_schedule_kwargs
                ),
            }

        self._canvas_update_mode = str(config["update_mode"])
        max_fps_config = config["max_fps"]
        self._canvas_max_fps = None if max_fps_config is None else float(max_fps_config)
        vsync_config = config["vsync"]
        self._canvas_vsync = None if vsync_config is None else bool(vsync_config)
        self._canvas_refresh_rate_hz = refresh_hz
        self._canvas_uses_display_refresh = bool(config["uses_display_refresh"])
        self._canvas_schedule_kwargs = dict(applied_schedule_kwargs)
        self._canvas_schedule_applied = bool(applied_schedule_kwargs)
        logger.info(
            "Pygfx canvas scheduling: update_mode=%s max_fps=%s vsync=%s "
            "present_method=%s applied=%s",
            self._canvas_update_mode,
            self._canvas_max_fps,
            self._canvas_vsync,
            self._canvas_present_method_requested,
            sorted(supported_canvas_kwargs),
        )
        return canvas

    def _record_canvas_present_method(self) -> None:
        """Record the presentation method selected after wgpu context creation."""
        present_to_screen = getattr(self._canvas, "_present_to_screen", None)
        if present_to_screen is True:
            self._canvas_present_method = "screen"
        elif present_to_screen is False:
            self._canvas_present_method = "bitmap"
        else:
            self._canvas_present_method = "unresolved"

    def _replace_canvas_with_bitmap(self, layout: Any) -> None:
        """Recreate a failed direct-screen canvas using bitmap presentation."""
        originally_requested = self._canvas_present_method_requested
        old_canvas = self._canvas_widget
        if old_canvas is not None:
            try:
                layout.removeWidget(old_canvas)
                old_canvas.hide()
                setattr(old_canvas, "_orchav_closed", True)
                old_canvas.close()
                old_canvas.deleteLater()
            except (AttributeError, RuntimeError, TypeError):
                logger.debug("Could not fully dispose failed pygfx screen canvas", exc_info=True)

        self._qt_window_closed = False
        self._canvas = self._create_canvas_widget(present_method="bitmap")
        self._canvas_present_method_requested = originally_requested
        self._canvas_widget = self._canvas
        self._canvas_widget.setFocusPolicy(self._container.focusPolicy())
        self._container.setFocusProxy(self._canvas_widget)
        layout.addWidget(self._canvas_widget)
        # The container is already visible when renderer creation falls back.
        # Show the replacement explicitly and activate its layout so Qt assigns
        # native-window geometry before wgpu configures the bitmap surface.
        self._canvas_widget.show()
        layout.activate()

    def _create_interactive_wgpu_renderer(
        self,
        gfx: Any,
        layout: Any,
        *,
        app: Any = None,
        defer_process_events: bool = False,
    ) -> Any:
        """Create the display renderer, falling back from screen to bitmap once."""
        try:
            return create_wgpu_renderer(
                gfx,
                self._canvas,
                clear_color=self._clear_color,
                configure_effects=False,
            )
        except (TypeError, RuntimeError, OSError, ValueError) as exc:
            if self._canvas_present_method_requested != "screen":
                raise
            self._canvas_present_fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Direct pygfx screen presentation failed; retrying with bitmap mode: %s",
                exc,
            )
            self._replace_canvas_with_bitmap(layout)
            if app is not None and not defer_process_events:
                app.processEvents()
            return create_wgpu_renderer(
                gfx,
                self._canvas,
                clear_color=self._clear_color,
                configure_effects=False,
            )

    def _apply_screen_refresh_rate(self, screen: Any) -> None:
        """Update an on-demand canvas cap after the window changes screens."""
        if not self._canvas_uses_display_refresh or self._canvas is None:
            return
        refresh_hz = self._current_screen_refresh_rate(screen)
        if abs(refresh_hz - self._canvas_refresh_rate_hz) < 0.01:
            return
        set_update_mode = getattr(self._canvas, "set_update_mode", None)
        if not callable(set_update_mode):
            return
        try:
            set_update_mode("ondemand", min_fps=0.0, max_fps=refresh_hz)
        except (TypeError, ValueError, RuntimeError):
            logger.warning("Could not update pygfx canvas refresh cap", exc_info=True)
            return
        self._canvas_refresh_rate_hz = refresh_hz
        self._canvas_max_fps = refresh_hz
        logger.info("Pygfx canvas refresh cap updated to %.2f FPS", refresh_hz)

    def _install_screen_refresh_hook(self) -> None:
        """Track the refresh rate of the screen hosting the render window."""
        self._disconnect_screen_refresh_hook()
        if self._container is None:
            return
        try:
            window_handle = self._container.windowHandle()
            signal = window_handle.screenChanged if window_handle is not None else None
            if signal is None:
                return
            session_generation = self._session_generation

            def _screen_changed(screen: Any) -> None:
                if self._session_generation != session_generation:
                    return
                self._apply_screen_refresh_rate(screen)

            self._qt_screen_changed_signal = signal
            self._qt_screen_changed_callback = _screen_changed
            signal.connect(self._qt_screen_changed_callback)
        except (AttributeError, RuntimeError, TypeError):
            self._qt_screen_changed_signal = None
            self._qt_screen_changed_callback = None
            logger.debug("Could not install pygfx screen-refresh hook", exc_info=True)

    def _disconnect_screen_refresh_hook(self) -> None:
        """Disconnect the current host-screen signal, if it is still alive."""
        signal = self._qt_screen_changed_signal
        callback = self._qt_screen_changed_callback
        self._qt_screen_changed_signal = None
        self._qt_screen_changed_callback = None
        if signal is None or callback is None:
            return
        try:
            signal.disconnect(callback)
        except (AttributeError, RuntimeError, TypeError):
            logger.debug("Could not disconnect pygfx screen-refresh hook", exc_info=True)

    def _guarded_canvas_widget_class(self) -> Any:
        """Return a rendercanvas widget subclass that no-ops after Qt close."""
        base_cls = self._WgpuWidget
        renderer_ref = weakref.ref(self)
        session_generation = getattr(self, "_session_generation", 0)

        class _OrchavPygfxCanvas(base_cls):  # type: ignore[misc, valid-type]
            def _orchav_renderer_closed(self) -> bool:
                renderer = renderer_ref()
                return bool(
                    getattr(self, "_orchav_closed", False)
                    or (renderer is not None and getattr(renderer, "_qt_window_closed", False))
                )

            def _orchav_mark_closed(self) -> None:
                renderer = renderer_ref()
                if renderer is not None:
                    renderer._mark_qt_window_closed(
                        expected_generation=session_generation,
                    )
                try:
                    setattr(self, "_orchav_closed", True)
                except (RuntimeError, TypeError, AttributeError):
                    pass

            def _rc_request_paint(self) -> None:
                if self._orchav_renderer_closed():
                    return
                try:
                    super()._rc_request_paint()
                except RuntimeError:
                    self._orchav_mark_closed()

            def _rc_close(self) -> None:
                if self._orchav_renderer_closed():
                    return
                try:
                    super()._rc_close()
                except RuntimeError:
                    self._orchav_mark_closed()

        return _OrchavPygfxCanvas

    # Lifecycle

    def _start_canvas_presentation(self, *, force_initial_present: bool) -> bool:
        """Register the draw callback and optionally attempt one empty frame.

        A direct-screen Qt canvas may not yet be paintable when it is first
        inserted into the visible layout. In that case rendercanvas' synchronous
        ``force_draw()`` returns without invoking the draw callback. This is a
        deferred presentation, not a render failure; the normal queued draw
        remains responsible for the first frame. CLI benchmark and batch paths
        intentionally retain their deferred first-draw behavior.
        """
        self._redraw_requests += 1
        session_generation = self._session_generation

        def _animate_current_session() -> None:
            if self._session_generation != session_generation:
                return
            self._animate()

        self._canvas_draw_callback = _animate_current_session
        self._canvas.request_draw(_animate_current_session)
        if not force_initial_present:
            return False

        self._initial_present_attempted = True
        started_at = time.perf_counter()
        callbacks_before = self._draw_callbacks_received
        successes_before = self._render_successes
        failures_before = self._render_failures
        try:
            self._canvas.force_draw()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._initial_present_succeeded = False
            self._initial_present_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "PygfxRenderer: initial interactive canvas present failed; "
                "queued a normal draw retry: %s",
                exc,
            )
        else:
            if self._render_successes > successes_before:
                self._initial_present_succeeded = True
            elif self._draw_callbacks_received == callbacks_before:
                # Qt's direct-screen repaint is allowed to defer until the
                # widget can receive a native paint event. No callback ran, so
                # recording a failed render (or warning the user) is inaccurate.
                self._initial_present_succeeded = None
                logger.debug(
                    "PygfxRenderer: initial interactive canvas present deferred "
                    "until the Qt canvas is paintable"
                )
            else:
                self._initial_present_succeeded = False
                if self._render_failures > failures_before:
                    self._initial_present_error = "draw callback reported a render failure"
                else:
                    self._initial_present_error = (
                        "draw callback completed without a successful renderer submission"
                    )
                logger.warning(
                    "PygfxRenderer: initial interactive canvas draw failed; "
                    "queued a normal draw retry: %s",
                    self._initial_present_error,
                )
        finally:
            self._initial_present_duration_ms = (time.perf_counter() - started_at) * 1000.0

        if self._initial_present_succeeded is not True:
            self._redraw_requests += 1
            try:
                self._canvas.request_draw()
            except RuntimeError:
                logger.debug("PygfxRenderer: initial canvas retry ignored after Qt close")
        return bool(self._initial_present_succeeded)

    def _controller_event_targets(self) -> tuple[Any, ...]:
        """Return possible objects that can register pygfx controller events."""
        targets = (self._renderer, self._canvas, self._canvas_widget)
        return tuple(target for target in targets if target is not None)

    def _create_camera_controller(
        self,
        mode: str,
        camera_state: Optional[CameraState] = None,
        *,
        register_events: bool = True,
    ) -> Optional[Any]:
        """Create an orbit/fly controller, optionally without native handlers.

        Scenario authoring routes every input through one renderer-lifetime
        interaction router.  Its controller is therefore constructed without
        pygfx's automatic event registration and is driven by the renderer-side
        authoring runtime facade instead.
        """
        if self._camera is None:
            return None

        gfx = self._gfx
        if not register_events:
            try:
                if mode == "fly":
                    return gfx.FlyController(self._camera)
                ctrl = gfx.OrbitController(self._camera)
                if camera_state is not None and hasattr(ctrl, "target"):
                    ctrl.target = tuple(float(x) for x in camera_state.lookat)
                    distance = float(
                        np.linalg.norm(
                            np.asarray(camera_state.eye, dtype=np.float64)
                            - np.asarray(camera_state.lookat, dtype=np.float64)
                        )
                    )
                    if distance > 1e-6 and hasattr(ctrl, "distance"):
                        ctrl.distance = distance
                return ctrl
            except Exception as exc:
                logger.warning(
                    "PygfxRenderer: failed to create unregistered %s controller: %s",
                    mode,
                    exc,
                )
                return None

        failures: list[str] = []
        for target in self._controller_event_targets():
            try:
                if mode == "fly":
                    return gfx.FlyController(self._camera, register_events=target)

                ctrl = gfx.OrbitController(self._camera, register_events=target)
                if camera_state is not None and hasattr(ctrl, "target"):
                    ctrl.target = tuple(float(x) for x in camera_state.lookat)
                    distance = float(
                        np.linalg.norm(
                            np.asarray(camera_state.eye, dtype=np.float64)
                            - np.asarray(camera_state.lookat, dtype=np.float64)
                        )
                    )
                    if distance > 1e-6 and hasattr(ctrl, "distance"):
                        ctrl.distance = distance
                return ctrl
            except Exception as exc:
                failures.append(f"{type(target).__name__}: {exc}")

        if not failures:
            logger.warning("PygfxRenderer: no event targets available for %s controller", mode)
        else:
            logger.warning(
                "PygfxRenderer: failed to create %s controller across targets (%s)",
                mode,
                " | ".join(failures),
            )
        return None

    def _has_renderer_session_resources(self) -> bool:
        """Return whether a complete or partial renderer session is still owned."""
        return bool(
            self._initialized
            or self._container is not None
            or self._canvas is not None
            or self._canvas_widget is not None
            or self._renderer is not None
            or self._scene is not None
            or self._camera is not None
            or self._controller is not None
            or self._qt_lifecycle_connections
            or self._qt_screen_changed_signal is not None
        )

    def _reset_session_telemetry(self) -> None:
        """Reset counters that describe exactly one native canvas lifetime."""
        self._created_at = time.perf_counter()
        self._first_present_at = None
        self._initial_present_attempted = False
        self._initial_present_succeeded = None
        self._initial_present_duration_ms = None
        self._initial_present_error = None
        self._render_attempts = 0
        self._render_successes = 0
        self._render_failures = 0
        self._event_pump_calls = 0
        self._redraw_requests = 0
        self._last_present_call_at = 0.0
        self._last_present_success_at = 0.0
        self._present_interval_sum_s = 0.0
        self._present_interval_samples = 0
        self._present_interval_sq_sum_s = 0.0
        self._present_interval_max_s = 0.0
        self._recent_present_intervals_s.clear()
        self._last_present_was_animating = None
        self._frame_drop_count = 0
        self._last_update_call_at = 0.0
        self._update_interval_sum_s = 0.0
        self._update_interval_samples = 0
        self._update_calls_while_animating = 0
        self._draw_callbacks_received = 0
        self._forced_draw_fallbacks = 0
        self._draw_durations.clear()
        self._draw_callback_total_durations.clear()
        self._last_renderer_submit_ms = 0.0
        self._last_draw_callback_total_ms = 0.0
        self._benchmark_telemetry_baseline.clear()
        self._blocking_frame_count = 0
        self._blocking_force_draw_callbacks = 0
        self._blocking_force_draw_contaminated = 0
        self._pending_update_start = None
        self._update_to_present_times.clear()
        self._batch_mode = False
        self._batch_redraw_pending = False
        self._tick_count = 0
        self._tick_interval_sum_s = 0.0
        self._tick_interval_samples = 0
        self._last_tick_at = 0.0
        self._frame_update_start = 0.0
        self._last_end_frame_update_breakdown.clear()
        self._frame_update_metrics.clear()
        self._last_end_frame_update_breakdown_bytes.clear()
        self._frame_update_bytes.clear()

    def _reset_ibl_session_bindings(self) -> None:
        """Detach scene-specific IBL objects while retaining decoded texture caches."""
        manager = getattr(self, "_ibl_manager", None)
        scene = getattr(manager, "_scene", None)
        if scene is None:
            scene = self._scene
        background = getattr(manager, "_background", None)
        if background is not None and scene is not None:
            try:
                scene.remove(background)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        if scene is not None and bool(getattr(manager, "_use_scene_environment", False)):
            try:
                scene.environment = None
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        tracked_materials = getattr(manager, "_tracked_materials", None)
        if tracked_materials is not None:
            tracked_materials.clear()
        if manager is not None:
            manager._background = None
            manager._scene = None
        self._ibl_loaded = False
        self._skybox_visible = False
        self._solid_background = None
        self._deferred_default_ibl_name = None
        self._deferred_ibl_load_scheduled = False

    def _begin_renderer_session(self) -> int:
        """Start a fresh renderer generation after prior resources are gone."""
        self._session_generation += 1
        self._initialized = False
        self._qt_window_closed = False
        self._qt_app_close_requested = False
        self._qt_closing_programmatically = False
        self._qt_destroyed_callbacks.clear()
        self._frame_update_paused = False
        self._active_controller_type = "orbit"
        self._width = 0
        self._height = 0
        self._reset_ibl_session_bindings()
        self._reset_session_telemetry()
        self._canvas_update_mode = "native"
        self._canvas_max_fps = None
        self._canvas_vsync = None
        self._canvas_present_method_requested = self._implicit_canvas_present_method()
        self._canvas_present_method = "unresolved"
        self._canvas_present_fallback_reason = None
        self._canvas_refresh_rate_hz = 60.0
        self._canvas_uses_display_refresh = False
        self._canvas_schedule_kwargs.clear()
        self._canvas_schedule_applied = False
        return self._session_generation

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
        install_default_interactions: bool = True,
    ) -> Any:
        """Create one transactional Qt/pygfx renderer session."""
        if self._has_renderer_session_resources():
            self.close()
        session_generation = self._begin_renderer_session()
        try:
            return self._initialize_visualizer_session(
                window_name,
                width,
                height,
                left,
                top,
                suppress_default_camera,
                host_parent=host_parent,
                install_default_interactions=install_default_interactions,
                session_generation=session_generation,
            )
        except BaseException:
            # Initialization publishes Qt/native resources incrementally. The
            # normal close path deliberately handles incomplete sessions too.
            self.close()
            raise

    def _initialize_visualizer_session(
        self,
        window_name: str,
        width: int,
        height: int,
        left: int,
        top: int,
        suppress_default_camera: bool,
        *,
        host_parent: Any,
        install_default_interactions: bool,
        session_generation: int,
    ) -> Any:
        """Create the Qt canvas, pygfx scene, camera, and first draw callback.

        ``host_parent`` is the final Qt parent for an embedded visualization or
        Scenario Builder canvas. Detached sessions pass no parent and own their
        renderer window. The QRenderWidget is never created as a top-level
        widget and later reparented, which avoids direct-presentation failures
        on Windows.
        """
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
        except ImportError as exc:
            raise ImportError("PySide6 is required for pygfx renderer") from exc

        renderer_ref = weakref.ref(self)

        class _PygfxContainer(QWidget):
            """Qt container that lets the renderer close the whole app."""

            def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
                renderer = renderer_ref()
                if renderer is not None and renderer._session_generation == session_generation:
                    if not getattr(renderer, "_qt_closing_programmatically", False):
                        renderer._request_app_close_from_render_window(
                            expected_generation=session_generation,
                        )
                        event.ignore()
                        return
                    renderer._mark_qt_window_closed(
                        expected_generation=session_generation,
                    )
                super().closeEvent(event)

        # Use QRenderWidget directly (not QRenderCanvas wrapper).
        # QRenderCanvas is a toplevel wrapper that calls self.show() in __init__,
        # creating a separate window. The render widget is embedded in the
        # container instead, avoiding window-management conflicts on Windows
        # where the wrapper's self.show() + reparenting can leave the canvas
        # at its default 640x480 size.
        self._owns_container = host_parent is None
        if self._owns_container:
            self._container = _PygfxContainer()
            self._container.setWindowTitle(window_name)
            self._container.resize(width, height)
            self._container.setMinimumSize(320, 240)
            self._container.setFocusPolicy(Qt.StrongFocus)
            self._container.setAttribute(Qt.WA_NativeWindow, True)
            if left >= 0 and top >= 0:
                self._container.move(left, top)
            layout = QVBoxLayout(self._container)
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            self._container = host_parent
            self._container.setFocusPolicy(Qt.StrongFocus)
            layout = self._container.layout()
            if layout is None:
                layout = QVBoxLayout(self._container)
                layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = self._create_canvas_widget()
        self._canvas_widget = self._canvas
        self._canvas_widget.setFocusPolicy(Qt.StrongFocus)
        self._container.setFocusProxy(self._canvas_widget)
        layout.addWidget(self._canvas_widget)

        app = QApplication.instance()
        defer_interactive_boot = bool(getattr(self.visualizer, "_cli_driven_frame_run", False))

        if self._owns_container:
            self._container.show()
        self._install_screen_refresh_hook()
        if app is not None and not defer_interactive_boot:
            app.processEvents()

        gfx = self._gfx
        self._renderer = self._create_interactive_wgpu_renderer(
            gfx,
            layout,
            app=app,
            defer_process_events=defer_interactive_boot,
        )
        self._record_canvas_present_method()
        self._install_qt_lifecycle_hooks()
        _refresh_renderer_effect_passes(
            self._renderer,
            self._gfx,
            clear_color=self._clear_color,
            aa_mode=self._aa_mode,
            depth_pass_enabled=self._depth_pass_enabled,
        )

        self._scene = gfx.Scene()
        self._sync_solid_background()
        self._install_default_lights()
        self._camera = gfx.PerspectiveCamera(60.0, width / max(height, 1))
        self._minimap_camera = gfx.OrthographicCamera(1.0, 1.0, depth=100.0)
        self._minimap_viewport = gfx.Viewport(self._renderer)
        try:
            # Sionna RT uses Z-up; pygfx defaults to Y-up.
            self._camera.world.reference_up = (0.0, 0.0, 1.0)
            self._minimap_camera.world.reference_up = (0.0, 1.0, 0.0)
            if hasattr(self._minimap_camera, "maintain_aspect"):
                self._minimap_camera.maintain_aspect = False
            if not suppress_default_camera:
                self._camera.local.position = (0.0, -10.0, 5.0)
                self._camera.look_at((0.0, 0.0, 0.0))
        except (AttributeError, RuntimeError):
            pass

        self._controller = self._create_camera_controller(
            "orbit",
            register_events=False,
        )

        self._width = width
        self._height = height
        self._initialized = True

        # Prefer the neutral outdoor environment by default. On startup,
        # defer the HDR->cubemap conversion until after the first present so
        # the initial window can paint sooner.
        self._ibl_name = "neutral_outdoor"
        if self._defer_default_ibl:
            self._deferred_default_ibl_name = self._ibl_name
        elif self._load_ibl_into_scene(self._ibl_name):
            self._deferred_default_ibl_name = None

        # Tooltip overlay for object picking
        if not defer_interactive_boot and install_default_interactions:
            self._setup_tooltip()
            self._setup_hud_overlays()
            from ...authoring.interaction import InteractionSession

            self.pygfx_interaction_router().activate(
                InteractionSession.VISUALIZATION,
                self._ignore_interaction_input,
            )
            self._focus_canvas()

        # Register the draw callback once. Interactive startup presents the
        # empty background synchronously so the native surface cannot remain
        # black while SceneService prepares and uploads the initial scene.
        self._start_canvas_presentation(force_initial_present=not defer_interactive_boot)

        return self._container

    def initialize_authoring_viewport(
        self,
        host_parent: Any,
        *,
        width: int = 960,
        height: int = 640,
    ) -> Any:
        """Initialize an embedded Scenario Builder canvas in its final host."""

        if host_parent is None:
            raise ValueError("authoring viewport requires a final Qt host parent")
        return self.initialize_visualizer(
            "ORCHAV Scenario Builder",
            width,
            height,
            suppress_default_camera=False,
            host_parent=host_parent,
            install_default_interactions=False,
        )

    def scenario_authoring_runtime(self) -> Any:
        """Return the public facade for one initialized authoring renderer.

        The facade lives inside the pygfx backend so native objects, camera
        matrices, pick identity, and gizmo state never leak through the
        renderer-neutral authoring package.
        """

        if not self._initialized:
            raise RuntimeError("pygfx renderer must be initialized before authoring")
        from .authoring import PygfxAuthoringRuntime

        runtime = getattr(self, "_scenario_authoring_runtime", None)
        if runtime is None:
            runtime = PygfxAuthoringRuntime(self)
            self._scenario_authoring_runtime = runtime
        return runtime

    def pygfx_interaction_router(self) -> Any:
        """Return the singleton interaction router for this renderer lifetime."""

        if not self._initialized:
            raise RuntimeError("pygfx renderer must be initialized before interaction")
        from ...authoring.interaction import PygfxInteractionRouter

        router = self._pygfx_interaction_router
        if router is None:
            router = PygfxInteractionRouter(self)
            self._pygfx_interaction_router = router
        return router

    @staticmethod
    def _ignore_interaction_input(_event: Any) -> None:
        """Consume no typed input while the router owns normal visualization."""

    def begin_live_preview_transform_session(
        self,
        sink: Callable[[dict[str, Any]], None],
    ) -> bool:
        """Acquire the shared router for normal-view live-preview editing."""

        if not self._initialized or not callable(sink):
            return False
        from ...authoring.interaction import InteractionSession

        self.pygfx_interaction_router().activate(InteractionSession.LIVE_PREVIEW, sink)
        return True

    def end_live_preview_transform_session(self) -> None:
        """Release live-preview ownership without disturbing authoring."""

        router = self._pygfx_interaction_router
        if router is None:
            return
        from ...authoring.interaction import InteractionSession

        if router.session is InteractionSession.LIVE_PREVIEW:
            router.activate(
                InteractionSession.VISUALIZATION,
                self._ignore_interaction_input,
            )

    def _install_qt_lifecycle_hooks(self) -> None:
        """Observe Qt object destruction without keeping the renderer alive."""
        self._disconnect_qt_lifecycle_hooks()
        renderer_ref = weakref.ref(self)
        session_generation = self._session_generation

        def _mark_closed(*_args: Any) -> None:
            renderer = renderer_ref()
            if renderer is not None:
                renderer._mark_qt_window_closed(
                    expected_generation=session_generation,
                )

        for widget in (self._container, self._canvas_widget):
            destroyed = getattr(widget, "destroyed", None)
            connect = getattr(destroyed, "connect", None)
            if not callable(connect):
                continue
            try:
                connect(_mark_closed)
                self._qt_destroyed_callbacks.append(_mark_closed)
                self._qt_lifecycle_connections.append((destroyed, _mark_closed))
            except (RuntimeError, TypeError, AttributeError):
                logger.debug("PygfxRenderer: could not connect Qt destroyed hook", exc_info=True)

    def _disconnect_qt_lifecycle_hooks(self) -> None:
        """Disconnect destroyed hooks from the current canvas and host."""
        connections = tuple(reversed(self._qt_lifecycle_connections))
        self._qt_lifecycle_connections.clear()
        self._qt_destroyed_callbacks.clear()
        for signal, callback in connections:
            try:
                signal.disconnect(callback)
            except (AttributeError, RuntimeError, TypeError):
                logger.debug(
                    "PygfxRenderer: could not disconnect Qt destroyed hook",
                    exc_info=True,
                )

    def _schedule_qt_callback(self, callback: Any) -> None:
        """Schedule *callback* on the next Qt event-loop turn."""
        try:
            from PySide6.QtCore import QTimer
        except ImportError:
            callback()
            return
        QTimer.singleShot(0, callback)

    def _request_app_close_from_render_window(
        self,
        *,
        expected_generation: Optional[int] = None,
    ) -> None:
        """Treat a user render-window close as a request to close the app."""
        if expected_generation is not None and expected_generation != self._session_generation:
            return
        if self._qt_app_close_requested:
            return
        session_generation = self._session_generation
        self._qt_app_close_requested = True
        viz = getattr(self, "visualizer", None)

        def _close_app() -> None:
            if self._session_generation != session_generation:
                return
            close = getattr(viz, "close", None)
            if callable(close):
                accepted = close()
                if accepted is False and self._session_generation == session_generation:
                    # QWidget.close() reports False when a dirty authoring
                    # document rejects the close. The detached render window
                    # was ignored, so it remains a usable live session.
                    self._qt_app_close_requested = False
                return
            try:
                from PySide6.QtWidgets import QApplication
            except ImportError:
                if self._session_generation == session_generation:
                    self._qt_app_close_requested = False
                return
            app = QApplication.instance()
            if app is not None:
                app.quit()
            elif self._session_generation == session_generation:
                self._qt_app_close_requested = False

        self._schedule_qt_callback(_close_app)

    def _mark_qt_window_closed(
        self,
        *,
        expected_generation: Optional[int] = None,
    ) -> None:
        """Mark Qt/rendercanvas handles closed before their C++ objects disappear."""
        if expected_generation is not None and expected_generation != self._session_generation:
            return
        if getattr(self, "_qt_window_closed", False):
            return
        self._qt_window_closed = True
        self._frame_update_paused = True
        self._deferred_default_ibl_name = None
        self._deferred_ibl_load_scheduled = False

        # rendercanvas may ask closed canvases to close again during Qt app
        # shutdown. Marking its own flag keeps that cleanup path from touching
        # a QRenderWidget whose C++ object Qt has already deleted.
        for canvas in (getattr(self, "_canvas", None), getattr(self, "_canvas_widget", None)):
            if canvas is None:
                continue
            try:
                setattr(canvas, "_orchav_closed", True)
                setattr(canvas, "_is_closed", True)
                setattr(canvas, "_draw_frame", None)
                setattr(canvas, "_rc_request_paint", lambda *_, **__: None)
                setattr(canvas, "_rc_close", lambda *_, **__: None)
            except (RuntimeError, TypeError, AttributeError):
                pass

    @staticmethod
    def _qt_widget_alive(widget: Any) -> bool:
        """Return whether a Qt wrapper still owns a live C++ object."""
        if widget is None:
            return False
        try:
            widget.isVisible()
        except RuntimeError:
            return False
        return True

    def close(self) -> None:
        """Release complete or partially initialized renderer resources."""
        # Invalidate Qt callbacks before touching native objects. A destroyed
        # signal queued by an older canvas must never close a newer session.
        self._session_generation += 1
        was_initialized = self._initialized
        self._initialized = False
        self.set_mpc_path_selection_callback(None)
        self._disconnect_screen_refresh_hook()
        self._disconnect_qt_lifecycle_hooks()

        interaction_router = self._pygfx_interaction_router
        if interaction_router is not None:
            try:
                interaction_router.close()
            except Exception:
                logger.debug("PygfxRenderer: interaction-router close failed", exc_info=True)
            self._pygfx_interaction_router = None

        authoring_runtime = self._scenario_authoring_runtime
        if authoring_runtime is not None:
            try:
                authoring_runtime.close()
            except Exception:
                logger.debug("PygfxRenderer: authoring-runtime close failed", exc_info=True)
            self._scenario_authoring_runtime = None

        window_closed = bool(getattr(self, "_qt_window_closed", False))
        self._frame_update_paused = True
        self._deferred_default_ibl_name = None
        self._deferred_ibl_load_scheduled = False

        if was_initialized and not window_closed:
            try:
                self.clear()
            except Exception:
                logger.debug("PygfxRenderer: native scene clear failed", exc_info=True)
        self._name_to_handle.clear()
        self._handle_to_name.clear()
        self._objects.clear()
        self._initialize_mpc_path_inspection_state()
        self._kinds.clear()
        self._topology.clear()
        self._external_geometry_names.clear()
        self._hidden.clear()
        self._edge_geometry_names.clear()
        self._geometry_color_sources.clear()
        self._materials.clear()
        self._material_apply_signatures.clear()
        self._transforms.clear()
        self._positions.clear()
        self._geometry_upload_center.clear()
        self._geometry_texcoords_available.clear()
        self._geometry_payload_cache_keys.clear()
        self._render_object_snapshots.clear()
        self._dirty_render_object_geometry.clear()
        self._uncertain_mesh_index_buffers.clear()
        self._vertex_stream_array_tokens.clear()
        self._vertex_stream_next_array_token = 0
        self._vertex_stream_incompatible_transitions.clear()
        self._vertex_stream_rebuild_names.clear()
        self._label_anchor_groups.clear()
        self._label_anchor_key_by_name.clear()
        self._label_anchor_by_name.clear()
        self._normal_line_overlays.clear()
        try:
            self._dispose_transform_gizmo()
        except Exception:
            logger.debug("PygfxRenderer: transform-gizmo close failed", exc_info=True)
        self.last_frame_packet = None
        self._camera_state = None
        self._reset_ibl_session_bindings()

        if self._owns_container and self._qt_widget_alive(self._container):
            self._qt_closing_programmatically = True
            try:
                self._container.close()
            except Exception:
                pass
            finally:
                self._qt_closing_programmatically = False
            delete_later = getattr(self._container, "deleteLater", None)
            if callable(delete_later):
                try:
                    delete_later()
                except (RuntimeError, TypeError, AttributeError):
                    pass
        elif self._qt_widget_alive(self._canvas_widget):
            layout = self._container.layout() if self._qt_widget_alive(self._container) else None
            if layout is not None:
                try:
                    layout.removeWidget(self._canvas_widget)
                except (RuntimeError, TypeError, AttributeError):
                    pass
            if self._qt_widget_alive(self._container):
                try:
                    self._container.setFocusProxy(None)
                except (AttributeError, RuntimeError, TypeError):
                    pass
            try:
                self._canvas_widget.close()
                self._canvas_widget.deleteLater()
            except (RuntimeError, TypeError, AttributeError):
                pass

        self._qt_window_closed = False
        self._qt_app_close_requested = False
        self._qt_closing_programmatically = False
        self._container = None
        self._owns_container = True
        self._canvas = None
        self._canvas_widget = None
        self._canvas_draw_callback = None
        self._renderer = None
        self._scene = None
        self._camera = None
        self._controller = None
        self._active_controller_type = "orbit"
        self._width = 0
        self._height = 0
        self._static_group = None
        self._minimap_camera = None
        self._minimap_viewport = None
        self._minimap_overlay_scene = None
        self._minimap_overlay_camera = None
        self._minimap_overlay_size = None
        self._minimap_overlay_objects.clear()
        self._payload_cache.clear()
        self._texture_cache.clear()
        self._texture_source_identities.clear()
        self._mpc_lines_source_sig = None
        self._mpc_points_source_sig = None
        self._mpc_marker_cache_key = None
        self._mpc_marker_codes_buf = None
        self._last_coverage_signature = None
        self._applied_coverage_state = None
        self._applied_beamforming_surfaces.clear()
        self._beamforming_owned_names.clear()
        self._ambient_light = None
        self._key_light = None
        self._fill_light = None
        self._head_light = None
        self._pick_metadata.clear()
        self._reverse_objects.clear()
        self._tooltip_label = None
        self._hud_overlay_labels.clear()
        self._hud_overlay_specs.clear()
        self._hud_suppressed = False
        self._mpc_marker_legend_requested = False
        self._visible_trajectory_kinds.clear()
        self._trajectory_hud_color_mode = "node_color"
        self._trajectory_hud_scalar_range = None
        self._minimap_world_rect = None
        self._follow_target_lookat = None
        self._last_hover_identity = None
        try:
            self._clear_mpc_buffers()
        except Exception:
            logger.debug("PygfxRenderer: MPC buffer close failed", exc_info=True)
        self._ground_grid_needs_rebuild = False

    def resize(self, width: int, height: int) -> None:
        """Resize the pygfx camera/canvas bookkeeping and request a redraw."""
        if not self._initialized or width <= 0 or height <= 0:
            return

        self._width = width
        self._height = height

        try:
            if hasattr(self._camera, "aspect"):
                self._camera.aspect = width / max(height, 1)
        except Exception:
            pass
        self._minimap_overlay_size = None
        self._reposition_hud_overlays()
        self._request_canvas_draw()

    # Named geometry CRUD

    # Frame-packet application (MPC rendering)

    def apply_frame(self, packet: FrameRenderPacket) -> bool:
        """Apply a frame-heavy packet and report complete backend acceptance."""
        if not self._initialized:
            return False

        # Short-circuit when the exact same packet is applied twice
        # (e.g. repeated frame during animation), skip all frame work unless a
        # renderer-local overlay changed.
        if packet is self.last_frame_packet:
            try:
                # Coverage synchronization is keyed to backend-applied state,
                # so an identical desired packet must still retry a prior
                # native upload/material failure.
                coverage_succeeded = self._apply_coverage_data(packet)
                beamforming_succeeded = self._apply_beamforming(packet)
                if self._apply_rf_xray_overlay(packet):
                    self.request_redraw()
                return bool(coverage_succeeded) and bool(beamforming_succeeded)
            except (RuntimeError, ValueError, IndexError, TypeError) as exc:
                logger.exception("PygfxRenderer: error retrying frame overlays: %s", exc)
                return False

        # Selection is renderer-local and intentionally absent from frame
        # packets. Suspend it before mutating bulk geometry so the old path can
        # never be submitted once over a newly applied frame. Runtime commits
        # the removal only after end_frame_update() accepts presentation.
        PygfxMpcSelectionMixin._begin_mpc_path_inspection_frame_transition(self)
        try:
            if self.last_frame_packet is None:
                t0 = time.perf_counter()
                lines_succeeded = self._apply_mpc_lines(packet)
                t1 = time.perf_counter()
                points_succeeded = self._apply_mpc_points(packet)
                t2 = time.perf_counter()
                coverage_succeeded = self._apply_coverage_data(packet)
                t3 = time.perf_counter()
                self._apply_rf_xray_overlay(packet)
                beamforming_succeeded = self._apply_unsupported_features(packet)
                self._apply_stats(packet.stats_text)
                t4 = time.perf_counter()
                logger.debug(
                    "[pygfx-telemetry] apply(initial): total=%.1fms | "
                    "lines=%.1fms points=%.1fms coverage=%.1fms other=%.1fms",
                    (t4 - t0) * 1000,
                    (t1 - t0) * 1000,
                    (t2 - t1) * 1000,
                    (t3 - t2) * 1000,
                    (t4 - t3) * 1000,
                )
                self._record_profile_metric("pygfx_apply_total_ms", (t4 - t0) * 1000.0)
                self._record_profile_metric("pygfx_apply_lines_ms", (t1 - t0) * 1000.0)
                self._record_profile_metric("pygfx_apply_points_ms", (t2 - t1) * 1000.0)
                self._record_profile_metric("pygfx_apply_coverage_ms", (t3 - t2) * 1000.0)
                self._record_profile_metric("pygfx_apply_other_ms", (t4 - t3) * 1000.0)
            else:
                t0 = time.perf_counter()
                lines_succeeded = self._apply_mpc_lines(packet)
                t1 = time.perf_counter()
                points_succeeded = self._apply_mpc_points(packet)
                t2 = time.perf_counter()
                coverage_succeeded = self._apply_coverage_data_diff(
                    self.last_frame_packet,
                    packet,
                )
                t3 = time.perf_counter()
                self._apply_rf_xray_overlay(packet)
                beamforming_succeeded = self._apply_unsupported_features(packet)
                self._apply_stats_diff(self.last_frame_packet.stats_text, packet.stats_text)
                t4 = time.perf_counter()
                logger.debug(
                    "[pygfx-telemetry] apply(diff): total=%.1fms | "
                    "lines=%.1fms points=%.1fms coverage=%.1fms other=%.1fms",
                    (t4 - t0) * 1000,
                    (t1 - t0) * 1000,
                    (t2 - t1) * 1000,
                    (t3 - t2) * 1000,
                    (t4 - t3) * 1000,
                )
                self._record_profile_metric("pygfx_apply_total_ms", (t4 - t0) * 1000.0)
                self._record_profile_metric("pygfx_apply_lines_ms", (t1 - t0) * 1000.0)
                self._record_profile_metric("pygfx_apply_points_ms", (t2 - t1) * 1000.0)
                self._record_profile_metric("pygfx_apply_coverage_ms", (t3 - t2) * 1000.0)
                self._record_profile_metric("pygfx_apply_other_ms", (t4 - t3) * 1000.0)
            self._ensure_ground_grid_current()
            t_hud = time.perf_counter()
            self._update_mpc_hud_overlays(packet)
            self._record_profile_metric(
                "pygfx_apply_hud_overlays_ms",
                (time.perf_counter() - t_hud) * 1000.0,
            )
            succeeded = (
                bool(lines_succeeded)
                and bool(points_succeeded)
                and bool(coverage_succeeded)
                and bool(beamforming_succeeded)
            )
            if succeeded:
                self.last_frame_packet = packet
                PygfxPickingMixin._invalidate_mpc_pick_cache(self)
                PygfxPickingMixin._cancel_mpc_path_click_gesture(self)
            else:
                PygfxMpcSelectionMixin._finish_mpc_path_inspection_frame_transition(
                    self,
                    presented=False,
                )
            return succeeded
        except (RuntimeError, ValueError, IndexError, TypeError) as exc:
            PygfxMpcSelectionMixin._finish_mpc_path_inspection_frame_transition(
                self,
                presented=False,
            )
            logger.exception("PygfxRenderer: error in apply_frame(): %s", exc)
            return False

    def _apply_stats(self, stats_text: str) -> None:
        """Update the MPC info label with statistics text."""
        if hasattr(self.visualizer, "mpc_info_label"):
            self.visualizer.mpc_info_label.setText(stats_text)

    def _apply_stats_diff(self, old_stats: str, new_stats: str) -> None:
        """Update stats only when they changed."""
        if old_stats != new_stats:
            self._apply_stats(new_stats)

    # Sensing scene annotations (pygfx only)

    def _apply_unsupported_features(self, packet: FrameRenderPacket) -> bool:
        """Apply extra frame features and report complete backend acceptance."""
        return self._apply_beamforming(packet)

    @property
    def mpc_lineset(self) -> None:
        """Return no Open3D LineSet handle for the pygfx backend."""
        return None

    @property
    def mpc_pcd(self) -> None:
        """Return no Open3D PointCloud handle for the pygfx backend."""
        return None

    @property
    def renderer_type(self) -> str:
        """Return the backend token used by UI capability checks."""
        return "pygfx"

    def get_native_asset_cache_info(self) -> dict[str, Any]:
        """Return wgpu texture-cache inventory for lifecycle diagnostics."""
        return {
            "entries": len(self._texture_cache),
            # wgpu does not expose truthful device-allocation sizes here.
            "bytes": None,
            "max_bytes": None,
        }

    def clear_native_asset_cache(self) -> dict[str, int]:
        """Release reusable native texture handles without removing geometry."""
        entries = len(self._texture_cache)
        source_entries = len(self._texture_source_identities)
        self._texture_cache.clear()
        self._texture_source_identities.clear()
        return {"entries": entries, "source_entries": source_entries}

    @property
    def vis_initialized(self) -> bool:
        """Expose initialization state through the visualizer compatibility surface."""
        return self._initialized

    @staticmethod
    def _numpy_signature(array_like: Any) -> Optional[tuple[Any, ...]]:
        """Cheap identity signature: id + shape + dtype.

        Frame-packet arrays are created once per pipeline step and reused across
        calls, so ``id()`` is stable within a frame.  This avoids the cost of
        reshaping and sampling array data on every frame.
        """
        if array_like is None:
            return None
        arr = np.asarray(array_like)
        return (id(arr), arr.shape, arr.dtype.str)

    # Wireframe overlay (1.7)
