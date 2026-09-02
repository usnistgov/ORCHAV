"""Abstract renderer support used by the Open3D adapter.

The backend-neutral structural contract lives in :mod:`.protocol`. This base
class supplies common lifecycle defaults for adapters that choose ABC-based
composition; pygfx satisfies the same public contract through mixins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Tuple

import numpy as np

from ..model import RenderObject, Transform
from ..types.camera_state import CameraState
from ..types.render_payloads import MaterialPayload
from .camera_ops import camera_state_for_overview, camera_state_for_pov
from .protocol import MpcPathSelectionCallback, RendererCapabilities

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..pipeline.core import FrameRenderPacket
    from .mpc_path_inspection import MpcPathInspectionSnapshot


class RendererBase(ABC):
    """Common lifecycle defaults and required object methods for ABC adapters."""

    capabilities: RendererCapabilities

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Initialize shared renderer state without creating backend resources."""
        self.visualizer = visualizer
        self.last_frame_packet: Optional[FrameRenderPacket] = None
        self.vis = None
        self.vis_initialized = False
        self._event_pump_calls: int = 0
        self._redraw_requests: int = 0
        self._present_attempts: int = 0
        self._present_successes: int = 0
        self._idle_loop_active: bool = False

    @property
    @abstractmethod
    def renderer_type(self) -> str:
        """Return the renderer type name."""
        pass

    @property
    def renderer_id(self) -> str:
        """Backend identifier used by protocol consumers."""
        return self.renderer_type

    @abstractmethod
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
        """Initialize the Open3D visualizer window.

        Args:
            window_name: Title for the visualizer window.
            width: Window width in pixels.
            height: Window height in pixels.
            left: X position of the window on screen (-1 for system default).
            top: Y position of the window on screen (-1 for system default).
            suppress_default_camera: If True, skip setting the default camera
                position so a pre-read session camera can be applied without
                a visible jump.
            host_parent: Final Qt parent for an embedded canvas. Backends that
                do not advertise ``embedded_viewport`` must reject it.
        """
        pass

    @abstractmethod
    def apply_frame(self, packet: "FrameRenderPacket") -> bool:
        """Apply a frame-heavy packet and report complete backend acceptance."""
        pass

    @abstractmethod
    def update_renderer(self) -> None:
        """Update/refresh the renderer display."""
        pass

    def request_redraw(self) -> None:
        """Request a redraw without forcing an immediate render.

        Default implementation delegates to update_renderer().
        Open3D renderers may override to defer renders during batch updates.
        """
        self.update_renderer()

    def defer_until_next_render_turn(self, callback: Callable[[], None]) -> bool:
        """Optionally invoke ``callback`` after the backend's next render-loop turn.

        Most backends either submit synchronously or own a canvas scheduler, so
        they do not need an additional pacing boundary and return ``False``.
        A backend with a separately pumped native event loop may override this
        method, retain the one-shot callback, and return ``True``. This contract
        deliberately does not claim GPU completion or physical presentation.
        """
        del callback
        return False

    @abstractmethod
    def poll_events(self) -> None:
        """Poll for window/UI events. No-op in Open3D renderer."""
        pass

    def get_runtime_stats(self) -> dict[str, Any]:
        """Return a normalized runtime telemetry snapshot."""
        return {
            "event_pump_calls": int(self._event_pump_calls),
            "redraw_requests": int(self._redraw_requests),
            "present_attempts": int(self._present_attempts),
            "present_successes": int(self._present_successes),
            "avg_present_interval_ms": None,
            "avg_update_to_present_ms": None,
            "avg_draw_ms": None,
            "present_jitter_ms": None,
            "idle_loop_active": bool(self._idle_loop_active),
        }

    def get_native_asset_cache_info(self) -> dict[str, Any]:
        """Return backend-native asset inventory when the adapter owns one."""
        return {}

    def clear_native_asset_cache(self) -> dict[str, int]:
        """Release backend-native reusable assets without changing scene state."""
        return {}

    def _record_event_pump(self) -> None:
        """Increment the event-loop pump counter for runtime telemetry."""
        self._event_pump_calls += 1

    def _record_redraw_request(self) -> None:
        """Increment the deferred redraw counter for runtime telemetry."""
        self._redraw_requests += 1

    def _record_present_attempt(self, success: bool) -> None:
        """Record a presentation attempt and whether it reached the backend."""
        self._present_attempts += 1
        if success:
            self._present_successes += 1

    @abstractmethod
    def set_line_width(self, width: float) -> bool:
        """Set line width for MPC paths. Returns True if successful."""
        pass

    def set_edge_line_width(self, width: float) -> bool:
        """Set line width for scene edge (wireframe) lines.

        Only the Open3D renderer supports per-geometry line widths. The default
        implementation is a no-op that returns False.

        Args:
            width: Line width in pixels for scene edge lines.

        Returns:
            True if the width was applied successfully.
        """
        return False

    def set_trajectory_line_width(self, width: float) -> bool:
        """Set line width for trajectory lines.

        Only the Open3D renderer supports per-geometry line widths. The default
        implementation is a no-op that returns False.

        Args:
            width: Line width in pixels for trajectory lines.

        Returns:
            True if the width was applied successfully.
        """
        return False

    def set_trajectory_point_size(self, size: float) -> bool:
        """Set point size for trajectory point markers."""
        return False

    def set_shadow_enabled(self, enabled: bool) -> bool:
        """Enable/disable shadowing when supported by backend."""
        return False

    @abstractmethod
    def set_point_size(self, size: float) -> bool:
        """Set point size for bounce points. Returns True if successful."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset renderer state - clear all caches and geometries."""
        pass

    @abstractmethod
    def set_background_color(self, color: list[float]) -> None:
        """Set the background color. Color is [r, g, b] or [r, g, b, a]."""
        pass

    @abstractmethod
    def reset_camera_bounds(self) -> None:
        """Reset camera to fit all visible geometry. Call after loading scene."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the visualizer window and clean up resources."""
        pass

    @abstractmethod
    def ensure_object(self, obj: RenderObject) -> bool:
        """Create or update one declarative object in the backend scene."""
        pass

    def update_mesh_vertex_stream(self, obj: RenderObject) -> bool:
        """Decline optional fixed-topology vertex streaming by default."""
        del obj
        return False

    @abstractmethod
    def remove_object(self, object_id: str) -> bool:
        """Ensure the declarative object is absent, including when already absent."""
        pass

    @abstractmethod
    def set_visible(self, object_id: str, visible: bool) -> bool:
        """Set visibility by stable render-object ID."""
        pass

    @abstractmethod
    def set_material(self, object_id: str, material: MaterialPayload | dict[str, Any]) -> bool:
        """Set material by stable render-object ID."""
        pass

    @abstractmethod
    def set_transform(self, object_id: str, transform: Transform | np.ndarray) -> bool:
        """Set transform by stable render-object ID."""
        pass

    def begin_live_preview_transform_session(
        self,
        sink: Callable[[dict[str, Any]], None],
    ) -> bool:
        """Decline interactive transforms for backends without a gizmo session."""
        del sink
        return False

    def end_live_preview_transform_session(self) -> None:
        """Release an optional live-preview transform session."""

    def set_mpc_path_inspection(self, snapshot: MpcPathInspectionSnapshot) -> bool:
        """Decline selected-path presentation on unsupported backends."""
        del snapshot
        return False

    def set_mpc_pick_segment_mapping(
        self,
        packet_identity: int,
        canonical_segment_indices: Any,
    ) -> bool:
        """Decline worker-prepared MPC picking maps on unsupported backends."""
        del packet_identity, canonical_segment_indices
        return False

    def update_mpc_path_flow(self, phase: float) -> bool:
        """Decline selected-path flow animation on unsupported backends."""
        del phase
        return False

    def clear_mpc_path_inspection(self) -> bool:
        """Accept idempotent cleanup when selected-path presentation is unsupported."""
        return True

    def set_mpc_path_selection_callback(
        self,
        callback: Optional[MpcPathSelectionCallback],
    ) -> None:
        """Ignore viewport path-selection callbacks on unsupported backends."""
        del callback

    # Backend-private named geometry adapter surface. Shared services enter
    # only through the declarative object methods above.

    def has_named_geometry(self, name: str) -> bool:
        """Return True if a named geometry exists in the scene."""
        return False

    def get_named_geometry_names(self) -> tuple[str, ...]:
        """Return a stable snapshot of renderer-owned geometry names."""
        return ()

    def set_named_visibility(self, name: str, visible: bool) -> bool:
        """Show/hide a named geometry."""
        return False

    def is_named_visible(self, name: str) -> Optional[bool]:
        """Return current named visibility if known."""
        return None

    def add_or_update_named_geometry(
        self,
        name: str,
        geometry: Any,
        material: Any = None,
        *,
        is_edge: bool = False,
    ) -> bool:
        """Add or update geometry under a stable name.

        Concrete renderers own backend-native geometry conversion. The base
        implementation deliberately does not fall back to unnamed geometry
        shims because shared code should use neutral render-object IDs.
        """
        return False

    def remove_named_geometry(self, name: str) -> bool:
        """Remove geometry by stable name."""
        return False

    def set_named_material(self, name: str, pbr_state: MaterialPayload | dict[str, Any]) -> bool:
        """Set material parameters for a named geometry."""
        return False

    def ensure_named_geometry(
        self,
        name: str,
        geometry: Any,
        material: Optional[MaterialPayload | dict[str, Any]] = None,
        transform: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        is_edge: bool = False,
    ) -> bool:
        """Unified named geometry ensure/update entrypoint.

        Accepts backend-neutral payload dataclasses or native geometry objects.
        """
        self.add_or_update_named_geometry(
            name=name,
            geometry=geometry,
            material=material,
            is_edge=is_edge,
        )
        if material is not None:
            self.set_named_material(name, material)
        if transform is not None:
            self.set_named_transform(name, np.asarray(transform, dtype=float))
        if visible is not None:
            self.set_named_visibility(name, bool(visible))
        return True

    def set_named_transform(self, name: str, transform: np.ndarray) -> bool:
        """Set a 4x4 transform matrix for a named geometry."""
        return False

    def get_named_position(self, name: str) -> Optional[np.ndarray]:
        """Return the cached world position for a named geometry if available."""
        return None

    def remap_external_geometry_name(
        self,
        *,
        old_geometry: Any,
        new_geometry: Any,
        name: str,
    ) -> bool:
        """Move a backend-local geometry mapping to a new object when supported."""
        return False

    def clear_scene_geometry(self) -> None:
        """Remove all object-tracked geometry from the scene.

        Renderers with a named scene graph override this method. The default
        implementation is a no-op so neutral services do not depend on
        renderer-native geometry containers.
        """
        return None

    def is_geometry_in_scene(self, geometry: Any) -> bool:
        """Return True if a geometry object is currently added to the scene.

        Renderer packages may override this for backend-local compatibility
        and diagnostics. Application services own semantic visibility and must
        not infer it from backend-native scene membership.
        """
        return False

    def is_geometry_visible(self, geometry: Any) -> Optional[bool]:
        """Return visibility state if known, else None."""
        if self.is_geometry_in_scene(geometry):
            return True
        return None

    def compute_scene_bounds(self, scope: str = "visible") -> Any:
        """Return renderer-specific scene bounds, or None if unavailable."""
        return None

    def get_camera_state(self) -> Optional[CameraState]:
        """Get current camera state if supported."""
        return None

    def set_camera_state(self, state: CameraState) -> bool:
        """Apply a renderer-neutral camera state if supported."""
        return False

    def set_overview_camera(
        self,
        view: str,
        bounds: Any,
        fov: float = 60.0,
        distance: Optional[float] = None,
    ) -> bool:
        """Apply a named overview camera intent using backend camera state."""
        state = camera_state_for_overview(
            view,
            bounds,
            fov=fov,
            distance=distance,
            aspect=self._viewport_aspect(),
        )
        return False if state is None else self.set_camera_state(state)

    def focus_camera(self, target_position: Any) -> bool:
        """Move the camera focus/orbit target to ``target_position``."""
        return self.update_follow_camera(target_position)

    def set_pov_camera(
        self,
        position: Any,
        orientation: Any,
        axis: str,
        *,
        defer_redraw: bool = False,
    ) -> bool:
        """Apply a first-person camera intent."""
        state = camera_state_for_pov(position, orientation, axis=axis)
        return False if state is None else self.set_camera_state(state)

    def update_follow_camera(self, target_position: Any) -> bool:
        """Update a follow-mode camera target."""
        state = self.get_camera_state()
        if state is None:
            return False
        target = np.asarray(target_position, dtype=np.float64).reshape(-1)[:3]
        if target.size < 3 or not np.all(np.isfinite(target)):
            return False
        eye = np.asarray(state.eye, dtype=np.float64)
        lookat = np.asarray(state.lookat, dtype=np.float64)
        offset = eye - lookat
        new_eye = target + offset
        return self.set_camera_state(
            CameraState(
                eye=(float(new_eye[0]), float(new_eye[1]), float(new_eye[2])),
                lookat=(float(target[0]), float(target[1]), float(target[2])),
                up=state.up,
                fov_deg=state.fov_deg,
            )
        )

    def reset_follow_state(self) -> None:
        """Clear renderer-owned follow-camera state."""
        return None

    def _viewport_aspect(self) -> float:
        """Return the active viewport aspect ratio with a desktop fallback."""
        width = float(getattr(self, "_width", 0.0))
        height = float(getattr(self, "_height", 0.0))
        if width > 1.0 and height > 1.0:
            return width / height
        return 16.0 / 9.0

    def set_fly_mode(self, enabled: bool) -> bool:
        """Enable or disable fly camera mode.

        Only the Open3D renderer supports this. Other renderers return False.

        Args:
            enabled: True to enable fly mode, False to restore default orbit mode.

        Returns:
            True if the mode was changed successfully.
        """
        return False

    def get_ibl_intensity(self) -> Optional[float]:
        """Return current IBL intensity if supported."""
        return None

    def get_ibl_name(self) -> Optional[str]:
        """Return current IBL environment name if supported."""
        return None

    # Batch update support

    @contextmanager
    def batch_updates(self) -> Generator[None, None, None]:
        """Context manager for batching multiple geometry updates.

        In the Open3D renderer, this defers all redraws until the context exits,
        providing a significant performance boost for bulk operations.

        Renderers that do not override this perform immediate updates.

        Usage:
            with renderer.batch_updates():
                for render_object in render_objects:
                    renderer.ensure_object(render_object)
            # A renderer override may issue a single redraw when the batch exits.
        """
        # Default implementation: no batching.
        yield

    # 3D Trajectory rendering (overridden by both renderers)

    def apply_trajectory(
        self,
        kind: str,
        trajectory_data: dict,
        color_mode: str = "node_color",
        scalar_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Build and display 3D trajectory geometry for TX, RX, or target nodes.

        Args:
            kind: ``"tx"``, ``"rx"``, or ``"target"``.
            trajectory_data: Dict with ``tx_positions`` / ``rx_positions`` /
                ``target_positions`` keys, each mapping node index (or name for
                targets) to list of ``(frame, x, y, z)`` tuples.
            color_mode: ``"node_color"`` (default per-node color),
                ``"speed"``, ``"altitude"``, ``"time"``, or ``"angular_speed"``.
            scalar_range: Optional global (vmin, vmax) so all trajectories
                share a consistent colour scale.
        """

    def remove_trajectory(self, kind: str) -> None:
        """Remove 3D trajectory geometry for TX, RX, or target nodes.

        Args:
            kind: ``"tx"``, ``"rx"``, or ``"target"``.
        """
