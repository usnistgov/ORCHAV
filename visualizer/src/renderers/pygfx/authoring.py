"""Backend-local runtime facade for pygfx interaction sessions.

The public authoring package exchanges typed, renderer-neutral values.  This
module is the deliberately narrow exception that owns native pygfx event,
camera, pick-identity, and persistent-gizmo access on the renderer side of the
boundary.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .transform_gizmo import reset_transform_gizmo_interaction


class PygfxAuthoringRuntime:
    """Expose native operations to the renderer-lifetime interaction router."""

    def __init__(self, renderer: Any) -> None:
        self._owner = renderer
        self._viewport = renderer._gfx.Viewport(renderer._renderer)
        self._registrations: list[tuple[Callable[[Any], None], tuple[str, ...]]] = []
        self._authoring_rotation_enabled = True
        self._closed = False

    @property
    def logical_size(self) -> tuple[float, float]:
        """Return the native renderer's current logical pixel size."""

        size = getattr(self._owner._renderer, "logical_size", (0.0, 0.0))
        try:
            return float(size[0]), float(size[1])
        except (IndexError, TypeError, ValueError):
            return 0.0, 0.0

    def camera_matrices(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return copies of inverse projection and view matrices."""

        camera = self._owner._camera
        if camera is None:
            return None
        try:
            projection_inverse = np.asarray(
                camera.projection_matrix_inverse,
                dtype=np.float64,
            ).copy()
            view = np.asarray(camera.view_matrix, dtype=np.float64).copy()
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            projection_inverse.shape != (4, 4)
            or view.shape != (4, 4)
            or not np.all(np.isfinite(projection_inverse))
            or not np.all(np.isfinite(view))
        ):
            return None
        return projection_inverse, view

    def add_event_handler(self, callback: Callable[[Any], None], *event_types: str) -> None:
        """Register the authoring router for native renderer events."""

        if self._closed:
            raise RuntimeError("authoring runtime is closed")
        self._owner._renderer.add_event_handler(callback, *event_types)
        self._registrations.append((callback, tuple(event_types)))

    def remove_event_handler(
        self,
        callback: Callable[[Any], None],
        *event_types: str,
    ) -> None:
        """Remove the exact authoring router registration."""

        backend = self._owner._renderer
        if backend is None:
            self._registrations = [
                registration
                for registration in self._registrations
                if registration != (callback, tuple(event_types))
            ]
            return
        backend.remove_event_handler(callback, *event_types)
        self._registrations = [
            registration
            for registration in self._registrations
            if registration != (callback, tuple(event_types))
        ]

    def object_id_for_native(self, native: Any) -> str | None:
        """Resolve one picked native object to its stable renderer ID."""

        value = self._owner._reverse_objects.get(id(native))
        return value if isinstance(value, str) else None

    def route_camera_event(self, event: Any) -> None:
        """Drive the unregistered active camera controller from the router."""

        controller = self._owner._controller
        if controller is None or self.is_gizmo_event(event):
            return
        controller.handle_event(event, self._viewport)

    def route_hover_event(self, event: Any) -> None:
        """Drive normal-view hover behavior from the shared router."""

        event_type = str(getattr(event, "type", ""))
        if event_type == "pointer_move":
            self._owner._on_pointer_move(event)
        elif event_type == "pointer_leave":
            self._owner._on_pointer_leave(event)

    def route_mpc_path_selection_event(self, event: Any) -> None:
        """Forward one visualization-session event to MPC click tracking."""
        self._owner.route_mpc_path_selection_event(event)

    def set_camera_mode(self, mode: str) -> bool:
        """Switch Orbit/Fly controllers without registering another handler."""

        normalized = str(mode).strip().lower()
        if normalized not in {"orbit", "fly"}:
            raise ValueError("authoring camera mode must be 'orbit' or 'fly'")
        if self._owner._active_controller_type == normalized:
            return True
        camera_state = self._owner.get_camera_state()
        controller = self._owner._create_camera_controller(
            normalized,
            camera_state=camera_state,
            register_events=False,
        )
        if controller is None:
            return False
        self._owner._controller = controller
        self._owner._active_controller_type = normalized
        self._owner.request_redraw()
        return True

    def ensure_gizmo(self, *, authoring: bool = True) -> Any | None:
        """Create the persistent gizmo without renderer-level handlers."""

        gizmo = self._owner._ensure_transform_gizmo(register_before_render=False)
        if gizmo is not None:
            try:
                gizmo.toggle_mode("world")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            self._set_scale_handles_visible(gizmo, False)
            self._set_rotation_handles_visible(
                gizmo,
                (
                    self._authoring_rotation_enabled
                    if authoring
                    else getattr(self._owner, "_transform_gizmo_target_kind", None) == "target"
                ),
            )
        return gizmo

    @staticmethod
    def _set_scale_handles_visible(gizmo: Any, visible: bool) -> None:
        """Set native scale-handle visibility for the active edit contract."""

        for child in getattr(gizmo, "_scale_children", ()):
            try:
                child.visible = bool(visible)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        center = getattr(gizmo, "_center_sphere", None)
        if center is not None:
            try:
                center.visible = bool(visible)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    @staticmethod
    def _set_rotation_handles_visible(gizmo: Any, visible: bool) -> None:
        """Hide meaningless rotation controls for derived orientations."""

        for collection_name in ("_rotate_children", "_arc_children"):
            for child in getattr(gizmo, collection_name, ()):
                try:
                    child.visible = bool(visible)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

    def attach_gizmo(
        self,
        object_id: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> bool:
        """Attach the persistent gizmo to an authoring-owned object."""

        metadata = self._owner._pick_metadata.get(object_id, {})
        semantic_transform = metadata.get("authoring_actor_pose")
        if semantic_transform is None:
            return False
        self._authoring_rotation_enabled = bool(metadata.get("authoring_rotation_enabled", True))
        if self.ensure_gizmo(authoring=True) is None:
            return False
        self._owner._transform_session_callback = callback
        attached = bool(
            self._owner._select_transform_gizmo_target(
                object_id,
                "authoring",
                0,
                semantic_transform=semantic_transform,
            )
        )
        if not attached:
            self._owner._transform_session_callback = None
        return attached

    def attach_live_preview_gizmo(
        self,
        object_id: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> bool:
        """Attach the persistent gizmo to one normal-view editable object."""

        parsed = self._owner._parse_transform_target_name(object_id)
        if parsed is None or self.ensure_gizmo(authoring=False) is None:
            return False
        kind, index = parsed
        self._owner._transform_session_callback = callback
        attached = bool(
            self._owner._select_transform_gizmo_target(
                object_id,
                kind,
                index,
            )
        )
        if not attached:
            self._owner._transform_session_callback = None
        else:
            gizmo = self._owner._transform_gizmo
            self._set_scale_handles_visible(gizmo, False)
            self._set_rotation_handles_visible(gizmo, kind == "target")
        return attached

    def hide_gizmo(self) -> None:
        """Detach and hide the gizmo while retaining its native instance."""

        gizmo = self._owner._transform_gizmo
        if gizmo is not None:
            reset_transform_gizmo_interaction(gizmo)
        self._owner._transform_session_callback = None
        self._owner._transform_gizmo_target_name = None
        self._owner._transform_gizmo_target_kind = None
        self._owner._transform_gizmo_target_index = None
        self._owner._transform_gizmo_last_position = None
        self._owner._transform_gizmo_last_transform = None
        self._owner._transform_gizmo_control_object = None
        self._owner._remove_transform_gizmo_proxy()
        self._authoring_rotation_enabled = True
        if gizmo is None:
            return
        try:
            gizmo.set_object(None)
            gizmo.visible = False
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def sync_gizmo_pose(self, object_id: str, transform: Any) -> bool:
        """Keep an attached semantic proxy aligned with a compiled pose."""

        synced = bool(self._owner.sync_active_transform_target_pose(object_id, transform))
        if not synced:
            return False
        metadata = self._owner._pick_metadata.get(object_id, {})
        self._authoring_rotation_enabled = bool(metadata.get("authoring_rotation_enabled", True))
        gizmo = self._owner._transform_gizmo
        if gizmo is not None:
            self._set_rotation_handles_visible(
                gizmo,
                self._authoring_rotation_enabled,
            )
        return True

    def update_before_render(
        self,
        event: Any,
        *,
        route_camera: bool = True,
        authoring: bool = True,
    ) -> None:
        """Update controller/gizmo state at the router's render boundary."""

        if route_camera:
            self.route_camera_event(event)
        gizmo = self._owner._transform_gizmo
        if gizmo is not None:
            gizmo.update_gizmo(event)
            self._set_scale_handles_visible(gizmo, False)
            self._set_rotation_handles_visible(
                gizmo,
                (
                    self._authoring_rotation_enabled
                    if authoring
                    else getattr(self._owner, "_transform_gizmo_target_kind", None) == "target"
                ),
            )

    def is_gizmo_event(self, event: Any) -> bool:
        """Return whether an input targets or actively drags the gizmo."""

        gizmo = self._owner._transform_gizmo
        if gizmo is None:
            return False
        target = getattr(event, "target", None)
        return bool(target in getattr(gizmo, "children", ()) or getattr(gizmo, "_ref", None))

    def close(self) -> None:
        """Release callbacks retained by the facade before renderer teardown."""

        if self._closed:
            return
        backend = self._owner._renderer
        if backend is not None:
            for callback, event_types in tuple(self._registrations):
                try:
                    backend.remove_event_handler(callback, *event_types)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
        self._registrations.clear()
        self.hide_gizmo()
        self._closed = True


__all__ = ["PygfxAuthoringRuntime"]
