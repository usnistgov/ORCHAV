"""Renderer-lifetime event routing and geometric picking for authoring.

The router installs one handler set for an embedded pygfx renderer. Surface
placement reconstructs the picked triangle from pygfx ``pick_info``;
empty-space placement intersects a camera ray with the visible horizontal work
plane. No depth texture is read back.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable
from uuid import UUID

import numpy as np

from .viewport_port import (
    AuthoringTool,
    HitResult,
    KeyboardInput,
    PointerInput,
    PointerPhase,
    RenderTurnInput,
    TransformInput,
    TransformPhase,
    ViewportEventSink,
    parse_renderer_id,
)


class InteractionSession(str, Enum):
    """Mutually exclusive renderer interaction sessions."""

    VISUALIZATION = "visualization"
    AUTHORING = "authoring"
    LIVE_PREVIEW = "live_preview"


_EVENT_TYPES = (
    "pointer_down",
    "pointer_up",
    "pointer_move",
    "double_click",
    "pointer_leave",
    "wheel",
    "key_down",
    "key_up",
    "before_render",
)


def _buffer_data(value: Any) -> Any:
    """Return array data from a pygfx Buffer or an array-like value."""

    return getattr(value, "data", value)


def _world_position(obj: Any, local_position: Any) -> tuple[float, float, float] | None:
    """Transform one finite local XYZ position through an object's world matrix."""

    try:
        local = np.asarray(local_position, dtype=float).reshape(-1)
        matrix = np.asarray(
            getattr(getattr(obj, "world", None), "matrix", np.eye(4)),
            dtype=float,
        )
        if local.size < 3 or matrix.shape != (4, 4):
            return None
        if not np.all(np.isfinite(local[:3])) or not np.all(np.isfinite(matrix)):
            return None
        world = matrix @ np.append(local[:3], 1.0)
        if not np.all(np.isfinite(world)) or abs(float(world[3])) < 1e-12:
            return None
        result = world[:3] / world[3]
        return tuple(float(value) for value in result)
    except (TypeError, ValueError, AttributeError):
        return None


def surface_position_from_pick_info(pick_info: Any) -> tuple[float, float, float] | None:
    """Reconstruct a picked mesh position from barycentrics and world transform."""

    if not isinstance(pick_info, dict):
        return None
    explicit = pick_info.get("world_position")
    if explicit is not None:
        point = np.asarray(explicit, dtype=float).reshape(-1)
        if point.size >= 3 and np.all(np.isfinite(point[:3])):
            return tuple(float(value) for value in point[:3])

    obj = pick_info.get("world_object")
    face_index = pick_info.get("face_index")
    face_coord = pick_info.get("face_coord")
    geometry = getattr(obj, "geometry", None)
    positions_buffer = getattr(geometry, "positions", None)
    if obj is None or face_index is None or face_coord is None or positions_buffer is None:
        return None

    try:
        positions = np.asarray(_buffer_data(positions_buffer), dtype=float).reshape((-1, 3))
        coords = np.asarray(face_coord, dtype=float).reshape(-1)
        face = int(face_index)
        if face < 0:
            return None
        indices_buffer = getattr(geometry, "indices", None)
        if indices_buffer is None:
            start = face * len(coords)
            vertex_indices = np.arange(start, start + len(coords), dtype=int)
        else:
            indices = np.asarray(_buffer_data(indices_buffer))
            vertex_indices = np.asarray(indices[face]).reshape(-1).astype(int)
        count = min(len(coords), len(vertex_indices))
        if count < 3:
            return None
        coords = coords[:count]
        vertex_indices = vertex_indices[:count]
        if (
            not np.all(np.isfinite(coords))
            or not np.all((0 <= vertex_indices) & (vertex_indices < len(positions)))
            or abs(float(coords.sum())) <= 1e-12
        ):
            return None
        coords = coords / float(coords.sum())
        local = np.sum(positions[vertex_indices] * coords[:, None], axis=0)
        return _world_position(obj, local)
    except (IndexError, OverflowError, TypeError, ValueError, AttributeError):
        return None


def vertex_position_from_pick_info(pick_info: Any) -> tuple[float, float, float] | None:
    """Return the exact picked point/line vertex position in world space."""

    if not isinstance(pick_info, dict):
        return None
    obj = pick_info.get("world_object")
    vertex_index = pick_info.get("vertex_index")
    positions_buffer = getattr(getattr(obj, "geometry", None), "positions", None)
    if obj is None or vertex_index is None or positions_buffer is None:
        return None
    try:
        index = int(vertex_index)
        positions = np.asarray(_buffer_data(positions_buffer), dtype=float).reshape((-1, 3))
        if index < 0 or index >= len(positions):
            return None
        return _world_position(obj, positions[index])
    except (OverflowError, TypeError, ValueError, AttributeError):
        return None


def camera_ray(
    screen_position: tuple[float, float],
    logical_size: tuple[float, float],
    camera: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a world-space ray for one logical viewport position."""

    try:
        projection_inverse = np.asarray(camera.projection_matrix_inverse, dtype=float)
        view = np.asarray(camera.view_matrix, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    return camera_ray_from_matrices(
        screen_position,
        logical_size,
        projection_inverse,
        view,
    )


def camera_ray_from_matrices(
    screen_position: tuple[float, float],
    logical_size: tuple[float, float],
    projection_inverse: Any,
    view_matrix: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a world-space ray from renderer-supplied camera matrices."""

    width, height = (float(logical_size[0]), float(logical_size[1]))
    if width <= 0.0 or height <= 0.0:
        return None
    x, y = float(screen_position[0]), float(screen_position[1])
    ndc_x = 2.0 * x / width - 1.0
    ndc_y = 1.0 - 2.0 * y / height
    try:
        projection_inverse_array = np.asarray(projection_inverse, dtype=float)
        view_inverse = np.linalg.inv(np.asarray(view_matrix, dtype=float))
        if projection_inverse_array.shape != (4, 4) or view_inverse.shape != (4, 4):
            return None

        def unproject(depth: float) -> np.ndarray:
            point = projection_inverse_array @ np.array(
                [ndc_x, ndc_y, depth, 1.0],
                dtype=float,
            )
            point = point / point[3]
            point = view_inverse @ point
            return point[:3] / point[3]

        near = unproject(0.0)
        far = unproject(1.0)
        direction = far - near
        norm = float(np.linalg.norm(direction))
        if not np.all(np.isfinite(near)) or not np.isfinite(norm) or norm <= 1e-12:
            return None
        return near, direction / norm
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        return None


def intersect_horizontal_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_z: float,
) -> tuple[float, float, float] | None:
    """Intersect a ray with ``z=plane_z`` in the forward ray direction."""

    origin_array = np.asarray(origin, dtype=float).reshape(-1)
    direction_array = np.asarray(direction, dtype=float).reshape(-1)
    plane = float(plane_z)
    if (
        origin_array.size < 3
        or direction_array.size < 3
        or not np.all(np.isfinite(origin_array[:3]))
        or not np.all(np.isfinite(direction_array[:3]))
        or not np.isfinite(plane)
    ):
        return None
    dz = float(direction_array[2])
    if abs(dz) <= 1e-12:
        return None
    distance = (plane - float(origin_array[2])) / dz
    if distance < 0.0:
        return None
    point = origin_array[:3] + distance * direction_array[:3]
    if not np.all(np.isfinite(point)):
        return None
    return tuple(float(value) for value in point[:3])


def intersect_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """Intersect a forward ray with an arbitrary finite world-space plane."""

    origin_array = np.asarray(origin, dtype=float).reshape(-1)
    direction_array = np.asarray(direction, dtype=float).reshape(-1)
    point_array = np.asarray(plane_point, dtype=float).reshape(-1)
    normal_array = np.asarray(plane_normal, dtype=float).reshape(-1)
    if any(value.size < 3 for value in (origin_array, direction_array, point_array, normal_array)):
        return None
    if not all(
        np.all(np.isfinite(value[:3]))
        for value in (origin_array, direction_array, point_array, normal_array)
    ):
        return None
    normal_length = float(np.linalg.norm(normal_array[:3]))
    if not np.isfinite(normal_length) or normal_length <= 1e-12:
        return None
    normal = normal_array[:3] / normal_length
    denominator = float(np.dot(direction_array[:3], normal))
    if abs(denominator) <= 1e-12:
        return None
    distance = float(np.dot(point_array[:3] - origin_array[:3], normal)) / denominator
    if distance < 0.0:
        return None
    result = origin_array[:3] + distance * direction_array[:3]
    if not np.all(np.isfinite(result)):
        return None
    return tuple(float(value) for value in result)


def snap_position(
    position: tuple[float, float, float],
    spacing: float | None,
    *,
    horizontal: bool = False,
) -> tuple[float, float, float]:
    """Snap XYZ to a positive grid spacing, or return it unchanged."""

    if spacing is None:
        return position
    step = float(spacing)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("grid spacing must be positive and finite")
    point = np.asarray(position, dtype=float).reshape(-1)
    if point.size != 3 or not np.all(np.isfinite(point)):
        raise ValueError("position must contain three finite values")
    snapped = np.round(point / step) * step
    if horizontal:
        snapped[2] = point[2]
    return tuple(float(value) for value in snapped)


def _freeze_transform_matrix(
    value: Any,
) -> tuple[tuple[float, float, float, float], ...] | None:
    """Return one finite immutable 4x4 transform, or ``None``."""

    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return tuple(tuple(float(component) for component in row) for row in matrix)


class PygfxInteractionRouter:
    """Own one complete native handler set for a pygfx renderer lifetime."""

    def __init__(self, renderer: Any) -> None:
        runtime_factory = getattr(renderer, "scenario_authoring_runtime", None)
        if not callable(runtime_factory):
            raise TypeError("pygfx authoring renderer must expose scenario_authoring_runtime()")
        self.runtime = runtime_factory()
        self._sink: Callable[[Any], None] | None = None
        self._session: InteractionSession | None = None
        self._tool = AuthoringTool.SELECT
        self._registered = False
        self._work_plane_z = 0.0
        self._work_plane_enabled = True
        self._grid_snap_m: float | None = None
        self._render_sequence = 0
        self._attached_transform_actor: UUID | None = None
        self._attached_transform_object_id: str | None = None
        self._attached_transform_matrix: tuple[tuple[float, float, float, float], ...] | None = None
        self._active_transform_actor: UUID | None = None
        self._active_transform_matrix: tuple[tuple[float, float, float, float], ...] | None = None
        self._drag_plane_point: tuple[float, float, float] | None = None
        self._drag_plane_normal: tuple[float, float, float] | None = None
        self._drag_plane_horizontal = True
        self._drag_source: HitResult | None = None

    @property
    def active(self) -> bool:
        """Return whether a handler set is currently registered."""

        return self._registered

    @property
    def handler_count(self) -> int:
        """Return the number of native event-type registrations for diagnostics."""

        return len(_EVENT_TYPES) if self._registered else 0

    @property
    def session(self) -> InteractionSession | None:
        """Return the active mutually exclusive editing session."""

        return self._session

    def activate(
        self,
        session: InteractionSession,
        sink: ViewportEventSink | Callable[[dict[str, Any]], None],
    ) -> None:
        """Register the router once, replacing any other editing session."""

        if self._registered and self._session == session:
            self._sink = sink
            return
        if self._registered:
            self._cancel_active_transform()
        else:
            self.runtime.add_event_handler(self._route_event, *_EVENT_TYPES)
            self._registered = True
        self._session = InteractionSession(session)
        self._sink = sink
        self.runtime.ensure_gizmo(authoring=self._session is InteractionSession.AUTHORING)

    def deactivate(self, session: InteractionSession | None = None) -> None:
        """Remove this session's handler set and hide transient gizmo state.

        Supplying ``session`` makes teardown ownership-safe: a stale live-preview
        service cannot deactivate an authoring session that replaced it.
        """

        if session is not None and self._session is not InteractionSession(session):
            return

        # Preserve the active sink until cancellation is delivered.  The
        # helper clears router state first, so a workspace that responds by
        # clearing its viewport cannot recursively emit a second CANCEL.
        self._cancel_active_transform()
        if self._registered:
            try:
                self.runtime.remove_event_handler(self._route_event, *_EVENT_TYPES)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._registered = False
        self._session = None
        self._sink = None

    def close(self) -> None:
        """Release the one renderer-lifetime registration, if active."""

        self.deactivate()

    def set_tool(self, tool: AuthoringTool) -> None:
        """Set the active authoring tool without adding event handlers."""

        next_tool = AuthoringTool(tool)
        if self._tool is AuthoringTool.MOVE and next_tool is not AuthoringTool.MOVE:
            self._cancel_active_transform()
        self._tool = next_tool

    def set_camera_mode(self, mode: str) -> bool:
        """Switch Orbit/Fly through the runtime without registering handlers."""

        return bool(self.runtime.set_camera_mode(mode))

    def sync_transform_pose(self, object_id: str, transform: Any) -> bool:
        """Align the gizmo and its next-gesture baseline with a compiled pose."""

        if (
            self._active_transform_actor is not None
            and object_id == self._attached_transform_object_id
        ):
            # Native pointer input owns the proxy for the duration of a
            # gesture. A synchronous document reconcile may publish the
            # corresponding semantic pose before the pointer-up callback,
            # but writing it back here would reset the proxy underneath the
            # active gizmo and corrupt its translation/rotation baseline.
            return True
        sync = getattr(self.runtime, "sync_gizmo_pose", None)
        synced = bool(callable(sync) and sync(object_id, transform))
        if not synced:
            return False
        frozen = _freeze_transform_matrix(transform)
        if (
            frozen is not None
            and self._active_transform_actor is None
            and object_id == self._attached_transform_object_id
        ):
            self._attached_transform_matrix = frozen
        return True

    def begin_drag_plane(self, z: float, source: HitResult) -> None:
        """Lock subsequent pointer motion to one horizontal actor-drag plane."""

        plane = float(z)
        if not np.isfinite(plane):
            raise ValueError("drag plane height must be finite")
        if source.actor_id is None or source.component is None:
            raise ValueError("drag source requires an actor and component")
        self._drag_plane_point = (
            source.world_position[0],
            source.world_position[1],
            plane,
        )
        self._drag_plane_normal = (0.0, 0.0, 1.0)
        self._drag_plane_horizontal = True
        self._drag_source = source

    def begin_control_drag(self, constraint: str, source: HitResult) -> None:
        """Start a semantic drag using its declared world-space constraint."""

        if source.actor_id is None or source.component is None:
            raise ValueError("drag source requires an actor and component")
        normalized = str(constraint).strip().lower()
        if normalized == "free":
            matrices = self.runtime.camera_matrices()
            if matrices is None:
                raise RuntimeError("camera matrices are unavailable for a free drag")
            try:
                view_inverse = np.linalg.inv(np.asarray(matrices[1], dtype=float))
                normal = np.asarray(view_inverse[:3, 2], dtype=float)
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                raise RuntimeError("camera view matrix is invalid for a free drag") from exc
            length = float(np.linalg.norm(normal))
            if not np.isfinite(length) or length <= 1e-12:
                raise RuntimeError("camera view direction is invalid for a free drag")
            self._drag_plane_normal = tuple(float(value) for value in normal / length)
            self._drag_plane_horizontal = False
        elif normalized in {"plane", "radial", "angular"}:
            self._drag_plane_normal = (0.0, 0.0, 1.0)
            self._drag_plane_horizontal = True
        else:
            raise ValueError(f"unsupported mobility control constraint: {constraint}")
        self._drag_plane_point = source.world_position
        self._drag_source = source

    def end_drag(self) -> None:
        """Clear the active fixed-plane handle drag."""

        self._drag_plane_point = None
        self._drag_plane_normal = None
        self._drag_plane_horizontal = True
        self._drag_source = None

    def clear_transform_gizmo(self) -> None:
        """Cancel any gesture and hide the persistent semantic gizmo."""

        self._cancel_active_transform()

    def attach_transform_gizmo(self, object_id: str) -> bool:
        """Attach the persistent gizmo to one semantic authoring pose.

        Attachment is separate from a transform gesture. Selecting an actor
        does not open a transient document command; the first native
        ``changed`` callback starts that command.
        """

        if (
            self._session is not InteractionSession.AUTHORING
            or self._tool is not AuthoringTool.MOVE
        ):
            return False
        parsed = parse_renderer_id(object_id)
        if parsed is None:
            return False
        _document_id, actor_id, _component = parsed
        if (
            self._attached_transform_actor == actor_id
            and self._attached_transform_object_id == object_id
        ):
            return True
        if self._active_transform_actor is not None:
            self._cancel_active_transform()
        elif self._attached_transform_object_id is not None:
            self.runtime.hide_gizmo()
        self._attached_transform_actor = actor_id
        self._attached_transform_object_id = object_id
        self._attached_transform_matrix = None
        if not self.runtime.attach_gizmo(object_id, self._on_gizmo_transform):
            self._attached_transform_actor = None
            self._attached_transform_object_id = None
            return False
        return True

    def set_work_plane(
        self,
        z: float,
        grid_snap_m: float | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        """Set visible work-plane and optional grid-snap state."""

        z_value = float(z)
        if not np.isfinite(z_value):
            raise ValueError("work plane height must be finite")
        if grid_snap_m is not None and (not np.isfinite(grid_snap_m) or grid_snap_m <= 0.0):
            raise ValueError("grid snap must be positive and finite")
        self._work_plane_z = z_value
        self._work_plane_enabled = bool(enabled)
        self._grid_snap_m = None if grid_snap_m is None else float(grid_snap_m)

    def resolve_hit(self, event: Any) -> HitResult | None:
        """Resolve one native event to a surface or work-plane hit."""

        pick_info = dict(getattr(event, "pick_info", None) or {})
        target = getattr(event, "target", None)
        if pick_info.get("world_object") is None and target is not None:
            pick_info["world_object"] = target
        obj = pick_info.get("world_object")
        renderer_id = self.runtime.object_id_for_native(obj)
        parsed = parse_renderer_id(renderer_id) if renderer_id else None
        actor_id: UUID | None = None
        component: str | None = None
        if parsed is not None:
            _document_id, actor_id, component = parsed
        elif renderer_id is not None:
            overlay_parts = renderer_id.split(":")
            if (
                len(overlay_parts) == 3
                and overlay_parts[0] == "authoring"
                and overlay_parts[2] == "work_plane"
            ):
                try:
                    UUID(overlay_parts[1])
                except ValueError:
                    pass
                else:
                    component = "work_plane"

        # The visible work-plane grid writes pick IDs like every other line,
        # but its endpoint vertices are only drawing geometry. Placement must
        # retain the cursor's canonical camera-ray/plane intersection.
        work_plane_pick = component == "work_plane"
        position = None if work_plane_pick else surface_position_from_pick_info(pick_info)
        surface = position is not None
        if position is None and not work_plane_pick:
            vertex_position = vertex_position_from_pick_info(pick_info)
            position = vertex_position
            if component == "trajectory_hit" and vertex_position is not None:
                # Line picking can report only a representative vertex.  A
                # body drag must start under the cursor or the trajectory
                # jumps on its first move. The edit is planar, so intersect the
                # cursor ray with that vertex's
                # horizontal plane when pygfx does not provide a precise
                # world position.
                matrices = self.runtime.camera_matrices()
                if matrices is not None:
                    ray = camera_ray_from_matrices(
                        (
                            float(getattr(event, "x", 0.0)),
                            float(getattr(event, "y", 0.0)),
                        ),
                        self.runtime.logical_size,
                        matrices[0],
                        matrices[1],
                    )
                    if ray is not None:
                        cursor_position = intersect_horizontal_plane(
                            ray[0],
                            ray[1],
                            vertex_position[2],
                        )
                        if cursor_position is not None:
                            position = cursor_position
        if position is None:
            if not self._work_plane_enabled:
                return None
            matrices = self.runtime.camera_matrices()
            if matrices is None:
                return None
            ray = camera_ray_from_matrices(
                (float(getattr(event, "x", 0.0)), float(getattr(event, "y", 0.0))),
                self.runtime.logical_size,
                matrices[0],
                matrices[1],
            )
            if ray is None:
                return None
            position = intersect_horizontal_plane(ray[0], ray[1], self._work_plane_z)
            if position is None:
                return None
            position = snap_position(position, self._grid_snap_m, horizontal=True)
        else:
            position = snap_position(position, self._grid_snap_m)

        vertex_index = pick_info.get("vertex_index")
        try:
            vertex_index = None if vertex_index is None else int(vertex_index)
            if vertex_index is not None and vertex_index < 0:
                vertex_index = None
        except (TypeError, ValueError):
            vertex_index = None
        if work_plane_pick:
            vertex_index = None
        return HitResult(
            world_position=position,
            renderer_object_id=renderer_id,
            actor_id=actor_id,
            component=component,
            vertex_index=vertex_index,
            surface=surface,
        )

    def _resolve_drag_hit(self, event: Any) -> HitResult | None:
        """Resolve pointer motion against the active semantic drag plane."""

        source = self._drag_source
        plane_point = self._drag_plane_point
        plane_normal = self._drag_plane_normal
        if source is None or plane_point is None or plane_normal is None:
            return None
        matrices = self.runtime.camera_matrices()
        if matrices is None:
            return None
        ray = camera_ray_from_matrices(
            (float(getattr(event, "x", 0.0)), float(getattr(event, "y", 0.0))),
            self.runtime.logical_size,
            matrices[0],
            matrices[1],
        )
        if ray is None:
            return None
        position = intersect_plane(ray[0], ray[1], plane_point, plane_normal)
        if position is None:
            return None
        position = snap_position(
            position,
            self._grid_snap_m,
            horizontal=self._drag_plane_horizontal,
        )
        return HitResult(
            world_position=position,
            renderer_object_id=source.renderer_object_id,
            actor_id=source.actor_id,
            component=source.component,
            vertex_index=source.vertex_index,
            surface=False,
        )

    def _on_gizmo_transform(self, event: dict[str, Any]) -> None:
        """Translate backend gizmo callbacks into typed transform phases."""

        sink = self._sink
        object_id = event.get("object_id")
        parsed = parse_renderer_id(object_id) if isinstance(object_id, str) else None
        if sink is None or parsed is None:
            return
        _document_id, actor_id, _component = parsed
        matrix_value = event.get("transform")
        if matrix_value is None:
            position = np.asarray(event.get("position", ()), dtype=float).reshape(-1)
            if position.size < 3 or not np.all(np.isfinite(position[:3])):
                return
            matrix = np.eye(4, dtype=float)
            matrix[:3, 3] = position[:3]
        else:
            try:
                matrix = np.asarray(matrix_value, dtype=float)
            except (TypeError, ValueError):
                return
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                return
        frozen_matrix = _freeze_transform_matrix(matrix)
        if frozen_matrix is None:
            return
        phase_name = str(event.get("phase", ""))
        if phase_name not in {"selected", "changed", "committed"}:
            return

        if phase_name == "selected":
            self._attached_transform_actor = actor_id
            self._attached_transform_object_id = object_id
            self._attached_transform_matrix = frozen_matrix
            return

        if (
            actor_id != self._attached_transform_actor
            or object_id != self._attached_transform_object_id
        ):
            return
        if phase_name == "changed":
            if self._active_transform_actor is None:
                initial = self._attached_transform_matrix or frozen_matrix
                self._active_transform_actor = actor_id
                self._active_transform_matrix = initial
                sink(
                    TransformInput(
                        actor_id=actor_id,
                        phase=TransformPhase.BEGIN,
                        matrix=initial,
                    )
                )
                if (
                    self._active_transform_actor != actor_id
                    or self._attached_transform_object_id != object_id
                ):
                    return
            self._active_transform_matrix = frozen_matrix
            sink(
                TransformInput(
                    actor_id=actor_id,
                    phase=TransformPhase.UPDATE,
                    matrix=frozen_matrix,
                )
            )
            return

        if self._active_transform_actor == actor_id:
            # End the router gesture before invoking application code.  A
            # commit handler may synchronously reconcile or clear the gizmo;
            # it must not resurrect partially cleared router state on return.
            self._attached_transform_matrix = frozen_matrix
            self._active_transform_actor = None
            self._active_transform_matrix = None
            sink(
                TransformInput(
                    actor_id=actor_id,
                    phase=TransformPhase.COMMIT,
                    matrix=frozen_matrix,
                )
            )

    def _on_live_preview_transform(self, event: dict[str, Any]) -> None:
        """Forward a gizmo event only while live preview owns the session."""

        sink = self._sink
        if self._session is InteractionSession.LIVE_PREVIEW and sink is not None:
            sink(dict(event))

    def _route_live_preview_event(self, event: Any) -> None:
        """Route normal-view input and optional actor selection."""

        event_type = str(getattr(event, "type", ""))
        if event_type == "before_render":
            self.runtime.update_before_render(
                event,
                route_camera=True,
                authoring=False,
            )
            return
        if self.runtime.is_gizmo_event(event):
            return
        if (
            self._session is InteractionSession.LIVE_PREVIEW
            and event_type == "pointer_down"
            and int(getattr(event, "button", 0) or 0) == 1
            and not getattr(event, "modifiers", ())
        ):
            object_id = self.runtime.object_id_for_native(getattr(event, "target", None))
            if object_id is not None and self.runtime.attach_live_preview_gizmo(
                object_id,
                self._on_live_preview_transform,
            ):
                stop = getattr(event, "stop_propagation", None)
                if callable(stop):
                    stop()
                return
        if self._session is InteractionSession.VISUALIZATION:
            route_selection = getattr(
                self.runtime,
                "route_mpc_path_selection_event",
                None,
            )
            if callable(route_selection):
                route_selection(event)
        self.runtime.route_hover_event(event)
        self.runtime.route_camera_event(event)

    def _cancel_active_transform(self) -> None:
        """Emit one typed cancel and hide the persistent gizmo."""

        sink = self._sink
        actor_id = self._active_transform_actor
        matrix = self._active_transform_matrix
        # Clear before invoking the sink.  Workspace cancellation calls back
        # into ``clear_transform_gizmo`` and must remain exactly-once.
        self.runtime.hide_gizmo()
        self._attached_transform_actor = None
        self._attached_transform_object_id = None
        self._attached_transform_matrix = None
        self._active_transform_actor = None
        self._active_transform_matrix = None
        self.end_drag()
        if sink is not None and actor_id is not None and matrix is not None:
            sink(
                TransformInput(
                    actor_id=actor_id,
                    phase=TransformPhase.CANCEL,
                    matrix=matrix,
                )
            )

    def _route_event(self, event: Any) -> None:
        if self._session in {
            InteractionSession.VISUALIZATION,
            InteractionSession.LIVE_PREVIEW,
        }:
            self._route_live_preview_event(event)
            return
        sink = self._sink
        if sink is None:
            return
        event_type = str(getattr(event, "type", ""))
        if event_type == "before_render":
            self.runtime.update_before_render(event)
            self._render_sequence += 1
            sink(RenderTurnInput(self._render_sequence))
            return
        if self.runtime.is_gizmo_event(event):
            return
        if self._should_route_camera_event(event):
            self.runtime.route_camera_event(event)
        if event_type in {"key_down", "key_up"}:
            if event_type == "key_down" and str(getattr(event, "key", "")).strip().lower() in {
                "escape",
                "esc",
            }:
                self._cancel_active_transform()
            sink(
                KeyboardInput(
                    key=str(getattr(event, "key", "")),
                    pressed=event_type == "key_down",
                    modifiers=frozenset(str(value) for value in getattr(event, "modifiers", ())),
                    repeat=bool(getattr(event, "repeat", False)),
                )
            )
            return

        phase_by_type = {
            "pointer_down": PointerPhase.DOWN,
            "pointer_up": PointerPhase.UP,
            "pointer_move": PointerPhase.MOVE,
            "double_click": PointerPhase.DOUBLE_CLICK,
            "pointer_leave": PointerPhase.LEAVE,
            "wheel": PointerPhase.WHEEL,
        }
        phase = phase_by_type.get(event_type)
        if phase is None:
            return
        wheel_delta = (
            float(getattr(event, "dx", 0.0)),
            float(getattr(event, "dy", 0.0)),
        )
        buttons_value = getattr(event, "buttons", ()) or ()
        if isinstance(buttons_value, int):
            buttons = (buttons_value,)
        else:
            buttons = tuple(int(value) for value in buttons_value)
        sink(
            PointerInput(
                phase=phase,
                position=(float(getattr(event, "x", 0.0)), float(getattr(event, "y", 0.0))),
                button=int(getattr(event, "button", 0) or 0),
                buttons=buttons,
                modifiers=frozenset(str(value) for value in getattr(event, "modifiers", ())),
                wheel_delta=wheel_delta,
            )
        )
        should_resolve = phase in {
            PointerPhase.DOWN,
            PointerPhase.MOVE,
            PointerPhase.DOUBLE_CLICK,
        } or (phase is PointerPhase.UP and self._drag_source is not None)
        if should_resolve:
            hit = (
                self._resolve_drag_hit(event)
                if self._drag_source is not None and phase in {PointerPhase.MOVE, PointerPhase.UP}
                else self.resolve_hit(event)
            )
            if hit is None and phase is PointerPhase.UP and self._drag_source is not None:
                hit = self._drag_source
            if hit is not None:
                sink(hit)
        if phase is PointerPhase.UP and self._drag_source is not None:
            self.end_drag()

    def _should_route_camera_event(self, event: Any) -> bool:
        """Keep authoring left drags and fixed-plane handle drags off the camera."""

        event_type = str(getattr(event, "type", ""))
        if self._drag_source is not None and event_type.startswith("pointer_"):
            return False
        if self._tool not in {
            AuthoringTool.PLACE,
            AuthoringTool.MOVE,
            AuthoringTool.WAYPOINT,
        }:
            return True
        if not event_type.startswith("pointer_"):
            return True
        button = int(getattr(event, "button", 0) or 0)
        buttons_value = getattr(event, "buttons", ()) or ()
        buttons = (
            (int(buttons_value),)
            if isinstance(buttons_value, int)
            else tuple(int(value) for value in buttons_value)
        )
        return button != 1 and 1 not in buttons
