"""Backend-neutral renderer protocol for visualizer services and controllers.

This module defines the shared contract above concrete rendering backends.
Callers should branch on ``RendererCapabilities`` instead of backend names and
send persistent scene changes through declarative ``RenderObject`` methods.
Backend-specific geometry registries and compatibility helpers belong inside
each renderer package and are deliberately absent from this shared contract.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

import numpy as np

from ..model import RenderObject, Transform
from ..types.camera_state import CameraState
from ..types.render_payloads import MaterialPayload

if TYPE_CHECKING:
    from ..pipeline.core import FrameRenderPacket
    from .mpc_path_inspection import MpcPathInspectionSnapshot

MpcPathSelectionCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class RendererCapabilities:
    """Typed feature map exposed by renderer backends.

    A false capability means callers must skip, hide, or fall back from that
    optional feature. The persistent-object and frame-transaction contracts
    are mandatory protocol behavior, not capabilities.
    """

    # Material capabilities.
    pbr: bool = False
    material_clearcoat: bool = False
    material_emissive: bool = False
    material_anisotropy: bool = False
    material_transmission: bool = False
    material_volume_thickness: bool = False
    material_normal_map: bool = False

    # Scene appearance and renderer-settings controls.
    scene_shader: bool = False
    frustum_culling: bool = False
    shadow_toggle: bool = False
    # Retained as an explicit backend-specific settings affordance until the
    # corresponding UI is renamed or moved behind a renderer-provided panel.
    open3d_settings_panel: bool = False
    camera_lookat: bool = False
    transparency: bool = False
    line_width: bool = False
    ibl: bool = False
    clipping_planes: bool = False

    # Camera, interaction, preview, and export affordances.
    fly_mode: bool = False
    camera_minimap: bool = False
    trajectories: bool = False
    screenshot_export: bool = False
    screen_space_labels: bool = False
    skybox: bool = False
    axes: bool = False
    aperture_preview: bool = False
    angular_preview: bool = False
    mpc_type_markers: bool = False
    mpc_path_inspection: bool = False
    rf_xray_overlay: bool = False
    viewport_hud: bool = False
    picking: bool = False
    transform_gizmo: bool = False
    hover_info: bool = False
    # Dedicated, embedded Scenario Builder workspace support. Generic picking
    # is not sufficient: authoring also requires final-parent canvas hosting
    # and the feature-specific interaction port.
    scenario_authoring: bool = False
    # The backend can construct its native canvas directly inside a persistent
    # Qt parent supplied by the application workspace.
    embedded_viewport: bool = False
    antialiasing: bool = False
    ground_grid: bool = False
    direct_lighting: bool = False
    lighting_profiles: bool = False
    wireframe: bool = False
    mesh_buffer_cache: bool = False
    # The backend can update a verified fixed-topology mesh by streaming only
    # its dynamic vertex attributes. Callers must retain the complete-object
    # synchronization path as the fallback when the backend rejects an update.
    mesh_vertex_stream_updates: bool = False
    static_mesh_batching: bool = False
    static_mesh_batch_object_threshold: int = 500
    static_mesh_batch_triangle_limit: int = 750_000
    static_mesh_batch_member_limit: int = 1_024

    # Event-loop ownership flags used by startup and idle scheduling.
    continuous_redraw: bool = False
    external_event_pump: bool = False
    idle_present_loop: bool = False
    physical_window_size: bool = False
    prefer_float32_frame_data: bool = False


def renderer_capabilities(renderer: Any) -> RendererCapabilities:
    """Return the typed capability map for ``renderer``.

    ``None`` produces an all-conservative map. Every renderer and renderer test
    double must otherwise publish one explicit ``RendererCapabilities`` object;
    optional features are never inferred from method or attribute presence.
    """
    if renderer is None:
        return RendererCapabilities()
    explicit = getattr(renderer, "capabilities", None)
    if isinstance(explicit, RendererCapabilities):
        return explicit
    raise TypeError(f"{type(renderer).__name__} must expose a RendererCapabilities instance")


class RendererProtocol(Protocol):
    """Structural renderer contract used by services and controllers.

    Implementations do not need to subclass this protocol; they only need to
    provide the same public surface. Methods returning ``bool`` report whether
    the renderer accepted the requested state change.
    """

    renderer_id: str
    capabilities: RendererCapabilities

    # Lifecycle, presentation, and runtime diagnostics.
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
        """Initialize the backend window/canvas and return backend state."""
        ...

    def close(self) -> None:
        """Release backend resources owned by the renderer."""
        ...

    def update_renderer(self) -> None:
        """Present pending scene changes as soon as the backend allows."""
        ...

    def request_redraw(self) -> None:
        """Request a future draw without requiring immediate presentation."""
        ...

    def refresh_viewport_hud(self) -> None:
        """Reconcile viewport HUD presentation with current application state."""
        ...

    def defer_until_next_render_turn(self, callback: Callable[[], None]) -> bool:
        """Defer a one-shot callback until a backend-native render-loop turn."""
        ...

    def begin_frame_update(self) -> None:
        """Begin one frame mutation transaction."""
        ...

    def end_frame_update(self) -> bool:
        """Submit the completed frame and report whether the backend accepted it."""
        ...

    def get_runtime_stats(self) -> dict[str, Any]:
        """Return normalized renderer timing/event-loop telemetry."""
        ...

    def get_native_asset_cache_info(self) -> dict[str, Any]:
        """Return backend-native reusable-asset inventory."""
        ...

    def clear_native_asset_cache(self) -> dict[str, int]:
        """Release backend-native reusable assets without changing scene state."""
        ...

    def batch_updates(self) -> AbstractContextManager[None]:
        """Group scene mutations so backends may coalesce redraws."""
        ...

    # Declarative object API used by services and payload factories.
    # ``ensure_object`` creates or updates the object identified by ``obj.id``;
    # removals are idempotent so synchronizers never need backend presence queries.
    def ensure_object(self, obj: RenderObject) -> bool:
        """Create or update the declarative render object by ``obj.id``."""
        ...

    def update_mesh_vertex_stream(self, obj: RenderObject) -> bool:
        """Update a fixed-topology mesh, rejecting unsupported snapshots."""
        ...

    def remove_object(self, object_id: str) -> bool:
        """Ensure the object is absent, succeeding when it is already absent."""
        ...

    def set_visible(self, object_id: str, visible: bool) -> bool:
        """Apply visibility by declarative render-object ID."""
        ...

    def set_material(self, object_id: str, material: MaterialPayload | dict[str, Any]) -> bool:
        """Apply material by declarative render-object ID."""
        ...

    def set_transform(self, object_id: str, transform: Transform | np.ndarray) -> bool:
        """Apply transform by declarative render-object ID."""
        ...

    def begin_live_preview_transform_session(
        self,
        sink: Callable[[dict[str, Any]], None],
    ) -> bool:
        """Acquire the renderer's mutually exclusive live-preview edit session."""
        ...

    def end_live_preview_transform_session(self) -> None:
        """Release the live-preview edit session without affecting another owner."""
        ...

    # Transient MPC path inspection. Selection identity and animation timing
    # stay in the application service; renderers own only presentation objects.
    def set_mpc_path_inspection(self, snapshot: MpcPathInspectionSnapshot) -> bool:
        """Replace the transient selected-path overlay."""
        ...

    def update_mpc_path_flow(self, phase: float) -> bool:
        """Move the selected-path flow pulse to a normalized path phase."""
        ...

    def clear_mpc_path_inspection(self) -> bool:
        """Remove transient selected-path presentation objects."""
        ...

    def set_mpc_path_selection_callback(
        self,
        callback: Optional[MpcPathSelectionCallback],
    ) -> None:
        """Publish canonical path ID and source frame-packet identity on clicks."""
        ...

    def set_mpc_pick_segment_mapping(
        self,
        packet_identity: int,
        canonical_segment_indices: Any,
    ) -> bool:
        """Install a worker-prepared contiguous int32 filtered-segment mapping."""
        ...

    # Frame application and trajectory appearance.
    def apply_frame(self, packet: FrameRenderPacket) -> bool:
        """Apply a frame-heavy packet and report complete backend acceptance."""
        ...

    def set_trajectory_line_width(self, width: float) -> bool:
        """Apply trajectory line width in renderer-supported units."""
        ...

    def set_trajectory_point_size(self, size: float) -> bool:
        """Apply trajectory point marker size in renderer-supported units."""
        ...

    def set_shadow_enabled(self, enabled: bool) -> bool:
        """Enable or disable backend shadow rendering when supported."""
        ...

    # Camera state and camera-mode helpers.
    def get_camera_state(self) -> Optional[CameraState]:
        """Return a portable camera state when the backend can expose it."""
        ...

    def set_camera_state(self, state: CameraState) -> bool:
        """Apply a portable camera state to the backend."""
        ...

    def set_overview_camera(
        self,
        view: str,
        bounds: Any,
        fov: float = 60.0,
        distance: Optional[float] = None,
    ) -> bool:
        """Apply a named overview camera intent over scene bounds."""
        ...

    def focus_camera(self, target_position: Any) -> bool:
        """Retarget the current camera orbit/follow point."""
        ...

    def set_pov_camera(
        self,
        position: Any,
        orientation: Any,
        axis: str,
        *,
        defer_redraw: bool = False,
    ) -> bool:
        """Apply a first-person camera pose from entity orientation."""
        ...

    def update_follow_camera(self, target_position: Any) -> bool:
        """Move the follow target while preserving renderer camera offset."""
        ...

    def reset_follow_state(self) -> None:
        """Clear backend follow-camera tracking state."""
        ...
