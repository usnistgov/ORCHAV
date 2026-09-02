"""Embedded pygfx viewport and overlay-port implementation for authoring."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import UUID

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..model import RenderObject, Transform
from ..renderers.protocol import renderer_capabilities
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    OrientationFramePayload,
    PointCloudPayload,
    TextLabelPayload,
)
from .interaction import InteractionSession
from .viewport_port import (
    ActorOverlaySnapshot,
    ActorVisualState,
    AuthoringTool,
    HitResult,
    OverlaySnapshot,
    OverlayVisibility,
    PreviewProvenance,
    ScenarioAuthoringViewportPort,
    SceneOverlayAsset,
    TrajectoryDisplayMode,
    ViewportEventSink,
    stable_renderer_id,
    stable_scene_renderer_id,
)

_ROLE_COLORS = {
    "tx": (0.95, 0.35, 0.20),
    "rx": (0.15, 0.65, 1.00),
    "target": (0.65, 0.45, 0.95),
    "group": (0.20, 0.80, 0.45),
}
_INVALID_COLOR = (0.95, 0.18, 0.18)
_INCOMPLETE_COLOR = (1.00, 0.68, 0.12)
_PENDING_COLOR = (0.62, 0.68, 0.72)
_SELECTED_COLOR = (1.00, 1.00, 1.00)
_WORK_PLANE_COLOR = (0.30, 0.45, 0.55)
_GHOST_COLOR = (0.20, 0.95, 0.95)
_LOOK_AT_COLOR = (1.00, 0.82, 0.20)
_AUTHORED_PATH_COLOR = (0.20, 0.95, 0.75)
_MOBILITY_CONTROL_COLOR = (1.00, 0.84, 0.18)
_FRAME_SAMPLE_COLOR = (0.92, 0.92, 0.92)
_DRAFT_PATH_COLOR = (0.72, 0.72, 0.72)
_SELECTION_HALO_COLOR = (1.00, 1.00, 1.00)
_GROUP_CONTEXT_COLOR = (0.25, 0.95, 0.55)
_VISUAL_OVERLAY_RENDER_ORDER = 10


def _color(actor: ActorOverlaySnapshot) -> tuple[float, float, float]:
    if actor.status is ActorVisualState.INVALID:
        return _INVALID_COLOR
    if actor.status is ActorVisualState.INCOMPLETE:
        return _INCOMPLETE_COLOR
    if actor.status is ActorVisualState.PENDING:
        return _PENDING_COLOR
    return _ROLE_COLORS.get(actor.role.lower(), (0.75, 0.75, 0.75))


def _overlay_visible(visibility: OverlayVisibility, actor: ActorOverlaySnapshot) -> bool:
    """Return whether an actor-scoped optional overlay should be rendered."""

    return visibility is OverlayVisibility.ALL or (
        visibility is OverlayVisibility.SELECTED and actor.selected
    )


def _path_payload(points: np.ndarray, *, arrow_count: int = 1) -> LineSetPayload:
    """Create one path payload with sparse in-line direction chevrons."""

    if len(points) == 1:
        points = np.vstack((points, points))
    path_lines = np.column_stack(
        (
            np.arange(len(points) - 1, dtype=np.int32),
            np.arange(1, len(points), dtype=np.int32),
        )
    )
    lines = path_lines
    moving_lines = []
    for start_index, end_index in path_lines:
        direction = points[end_index] - points[start_index]
        length = float(np.linalg.norm(direction))
        if length > 1e-9:
            moving_lines.append((int(start_index), int(end_index), direction, length))
    count = min(max(int(arrow_count), 0), len(moving_lines))
    if count:
        selected_indices = tuple(
            dict.fromkeys(
                int(value)
                for value in np.linspace(
                    0,
                    len(moving_lines) - 1,
                    count,
                )
            )
        )
    else:
        selected_indices = ()
    for moving_index in selected_indices:
        start_index, end_index, direction, length = moving_lines[moving_index]
        direction = direction.copy()
        direction /= length
        perpendicular = np.cross(direction, np.asarray((0.0, 0.0, 1.0)))
        perpendicular_length = float(np.linalg.norm(perpendicular))
        if perpendicular_length <= 1e-9:
            perpendicular = np.cross(direction, np.asarray((1.0, 0.0, 0.0)))
            perpendicular_length = float(np.linalg.norm(perpendicular))
        perpendicular /= perpendicular_length
        arrow_length = max(0.15, min(1.0, length * 0.25))
        tip = points[end_index]
        base = tip - direction * arrow_length
        wing_offset = perpendicular * arrow_length * 0.45
        arrow_start = len(points)
        points = np.vstack((points, tip, base + wing_offset, base - wing_offset))
        lines = np.vstack(
            (
                lines,
                (arrow_start, arrow_start + 1),
                (arrow_start, arrow_start + 2),
            )
        )
    return LineSetPayload(points=points, lines=lines, line_strip=False)


def _authored_path_payload(points: np.ndarray) -> LineSetPayload:
    """Create a control polygon without generator-style direction geometry."""

    lines = np.column_stack(
        (
            np.arange(len(points) - 1, dtype=np.int32),
            np.arange(1, len(points), dtype=np.int32),
        )
    )
    return LineSetPayload(points=points, lines=lines, line_strip=False)


def _path_hit_payload(points: np.ndarray) -> LineSetPayload:
    """Create broad interaction geometry without visible direction decoration."""

    lines = np.column_stack(
        (
            np.arange(len(points) - 1, dtype=np.int32),
            np.arange(1, len(points), dtype=np.int32),
        )
    )
    return LineSetPayload(points=points, lines=lines, line_strip=False)


class PygfxScenarioAuthoringViewportPort(ScenarioAuthoringViewportPort):
    """Reconcile typed authoring snapshots into one embedded pygfx renderer."""

    def __init__(
        self,
        host_widget: QWidget,
        visualizer: Any,
        *,
        renderer_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if renderer_factory is None:
            from ..renderers.pygfx.renderer import PygfxRenderer

            renderer_factory = PygfxRenderer
        self._widget = host_widget
        self.renderer = renderer_factory(visualizer)
        if not renderer_capabilities(self.renderer).scenario_authoring:
            raise ValueError("authoring viewport requires a scenario-authoring pygfx renderer")
        initialize = getattr(self.renderer, "initialize_authoring_viewport", None)
        if not callable(initialize):
            raise TypeError("scenario-authoring renderer must provide final-parent initialization")
        initialize(host_widget, width=960, height=640)
        router_factory = getattr(self.renderer, "pygfx_interaction_router", None)
        if not callable(router_factory):
            raise TypeError("scenario-authoring renderer must own one interaction router")
        self.router = router_factory()
        self._owned_objects: dict[str, RenderObject] = {}
        self._actor_positions: dict[UUID, np.ndarray] = {}
        self._snapshot: OverlaySnapshot | None = None
        self._pending_snapshot: OverlaySnapshot | None = None
        self._desired_gizmo_actor_id: UUID | None = None
        self._closed = False
        self._flush_timer = QTimer(host_widget)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_pending_snapshot)

    @property
    def widget(self) -> QWidget:
        return self._widget

    @property
    def active(self) -> bool:
        return self.router.active

    def activate(self, sink: ViewportEventSink) -> None:
        self.router.activate(InteractionSession.AUTHORING, sink)

    def deactivate(self) -> None:
        self.router.deactivate()

    def set_tool(self, tool: AuthoringTool) -> None:
        self.router.set_tool(tool)

    def set_camera_mode(self, mode: str) -> bool:
        """Switch Orbit/Fly through the renderer-lifetime interaction route."""

        return self.router.set_camera_mode(mode)

    def begin_drag_plane(self, z: float, source: HitResult) -> None:
        self.router.begin_drag_plane(z, source)

    def begin_control_drag(self, constraint: str, source: HitResult) -> None:
        self.router.begin_control_drag(constraint, source)

    def end_drag(self) -> None:
        self.router.end_drag()

    def clear_transform_gizmo(self) -> None:
        """Clear both desired semantic ownership and the rendered gizmo."""

        self._desired_gizmo_actor_id = None
        self.router.clear_transform_gizmo()

    def show_transform_gizmo(self, actor_id: UUID) -> bool:
        """Attach the persistent gizmo to an actor's stable semantic anchor."""

        wanted = UUID(str(actor_id))
        snapshot = self._pending_snapshot or self._snapshot
        if snapshot is None or not any(actor.actor_id == wanted for actor in snapshot.actors):
            self._desired_gizmo_actor_id = None
            self.router.clear_transform_gizmo()
            return False
        self._desired_gizmo_actor_id = wanted
        if self._pending_snapshot is not None:
            return True
        return self._attach_desired_transform_gizmo(snapshot)

    def _attach_desired_transform_gizmo(self, snapshot: OverlaySnapshot) -> bool:
        """Attach the requested gizmo after its semantic anchor is reconciled."""

        wanted = self._desired_gizmo_actor_id
        if wanted is None:
            return False
        if not any(actor.actor_id == wanted for actor in snapshot.actors):
            self._desired_gizmo_actor_id = None
            self.router.clear_transform_gizmo()
            return False
        object_id = stable_renderer_id(
            snapshot.document_id,
            wanted,
            "mobility_handles",
        )
        if object_id not in self._owned_objects:
            return False
        return bool(self.router.attach_transform_gizmo(object_id))

    def reconcile(self, snapshot: OverlaySnapshot) -> None:
        """Queue the newest snapshot for one renderer mutation batch per Qt turn."""

        if self._closed:
            raise RuntimeError("authoring viewport is closed")
        current = self._pending_snapshot or self._snapshot
        if (
            current is not None
            and current.document_id == snapshot.document_id
            and snapshot.revision < current.revision
        ):
            return
        self._pending_snapshot = snapshot
        if not self._flush_timer.isActive():
            self._flush_timer.start(0)

    def _flush_pending_snapshot(self) -> None:
        """Apply the latest queued snapshot and discard superseded drag updates."""

        if self._flush_timer.isActive():
            self._flush_timer.stop()
        snapshot = self._pending_snapshot
        if snapshot is None or self._closed:
            return
        self._pending_snapshot = None
        self.router.set_work_plane(
            snapshot.work_plane_z,
            snapshot.grid_snap_m,
            enabled=snapshot.work_plane_visible,
        )
        desired: dict[str, RenderObject] = {}
        for asset in snapshot.scene_assets:
            scene_object = self._scene_object(snapshot.document_id, asset)
            desired[scene_object.id] = scene_object
        if snapshot.work_plane_visible:
            work_plane = self._work_plane_object(snapshot)
            desired[work_plane.id] = work_plane
        if snapshot.placement_ghost is not None:
            placement_ghost = self._placement_ghost_object(snapshot)
            desired[placement_ghost.id] = placement_ghost
        if snapshot.placement_ghost is not None and snapshot.placement_guide_start is not None:
            placement_guide = self._placement_guide_object(snapshot)
            desired[placement_guide.id] = placement_guide
        actor_positions: dict[UUID, np.ndarray] = {}
        for actor in snapshot.actors:
            objects = self._actor_objects(
                snapshot.document_id,
                actor,
                orientation_axes_visibility=snapshot.orientation_axes_visibility,
                look_at_visibility=snapshot.look_at_visibility,
            )
            desired.update((obj.id, obj) for obj in objects)
            actor_positions[actor.actor_id] = np.asarray(actor.positions, dtype=float)

        # Geometry payload identity is the renderer's upload boundary.  Reuse
        # the prior immutable payload whenever the workspace says the
        # geometry is unchanged, even if selection, hover, or material state
        # changed in this snapshot.
        for object_id, obj in tuple(desired.items()):
            previous = self._owned_objects.get(object_id)
            signature = obj.metadata.get("geometry_signature")
            if (
                previous is not None
                and signature is not None
                and signature == previous.metadata.get("geometry_signature")
                and type(obj.payload) is type(previous.payload)
            ):
                desired[object_id] = replace(obj, payload=previous.payload)

        batch_updates = getattr(self.renderer, "batch_updates", None)
        context = batch_updates() if callable(batch_updates) else nullcontext()
        with context:
            for object_id in sorted(set(self._owned_objects) - set(desired)):
                self.renderer.remove_object(object_id)
            for obj in desired.values():
                self.renderer.ensure_object(obj)
                semantic_pose = obj.metadata.get("authoring_actor_pose")
                if semantic_pose is not None:
                    self.router.sync_transform_pose(obj.id, semantic_pose)
            # Keep the redraw request inside the renderer batch so a complete
            # overlay reconciliation schedules exactly one canvas update.
            self.renderer.request_redraw()
        self._owned_objects = desired
        self._actor_positions = actor_positions
        self._snapshot = snapshot
        self._attach_desired_transform_gizmo(snapshot)

    @staticmethod
    def _scene_object(document_id: UUID, asset: SceneOverlayAsset) -> RenderObject:
        """Build one pickable base-scene mesh with stable asset identity."""

        return RenderObject(
            id=stable_scene_renderer_id(document_id, asset.cache_key),
            payload=asset.payload,
            material=asset.material,
            visibility=True,
            metadata={
                "type": "scenario_authoring",
                "component": "scene_surface",
                "scene_asset_name": asset.name,
                "asset_cache_key": asset.cache_key,
                "geometry_signature": ("scene", asset.cache_key),
                "surface_pickable": True,
            },
        )

    @staticmethod
    def _placement_ghost_object(snapshot: OverlaySnapshot) -> RenderObject:
        """Build the transient placement point supplied by the workspace."""

        assert snapshot.placement_ghost is not None
        point = np.asarray((snapshot.placement_ghost,), dtype=np.float32)
        return RenderObject(
            id=f"authoring:{snapshot.document_id}:placement_ghost",
            payload=PointCloudPayload(
                points=point,
                colors=np.tile(np.asarray(_GHOST_COLOR, dtype=np.float32), (1, 1)),
            ),
            material=MaterialPayload(shader="unlit", point_size=16.0),
            visibility=True,
            metadata={
                "type": "scenario_authoring",
                "component": "placement_ghost",
                "pickable": False,
                "depth_write": False,
                "depth_compare": "<=",
                "render_order": _VISUAL_OVERLAY_RENDER_ORDER,
                "interaction_role": "decorative",
                "interaction_priority": -1,
                "geometry_signature": ("placement_ghost", snapshot.placement_ghost),
            },
        )

    @staticmethod
    def _placement_guide_object(snapshot: OverlaySnapshot) -> RenderObject:
        """Build a non-pickable rubber-band guide for waypoint drawing."""

        assert snapshot.placement_ghost is not None
        assert snapshot.placement_guide_start is not None
        points = np.asarray(
            (snapshot.placement_guide_start, snapshot.placement_ghost),
            dtype=np.float32,
        )
        return RenderObject(
            id=f"authoring:{snapshot.document_id}:placement_guide",
            payload=LineSetPayload(
                points=points,
                lines=np.asarray(((0, 1),), dtype=np.int32),
                colors=np.tile(np.asarray(_GHOST_COLOR, dtype=np.float32), (2, 1)),
            ),
            material=MaterialPayload(shader="unlit", line_width=2.0),
            visibility=True,
            metadata={
                "type": "scenario_authoring",
                "component": "placement_guide",
                "pickable": False,
                "depth_write": False,
                "depth_compare": "<=",
                "render_order": _VISUAL_OVERLAY_RENDER_ORDER,
                "interaction_role": "decorative",
                "interaction_priority": -1,
                "geometry_signature": (
                    "placement_guide",
                    snapshot.placement_guide_start,
                    snapshot.placement_ghost,
                ),
            },
        )

    @staticmethod
    def _work_plane_object(snapshot: OverlaySnapshot) -> RenderObject:
        """Build a lightweight visible grid for the empty-space placement plane."""

        spacing = float(snapshot.grid_snap_m or 5.0)
        half_extent = max(25.0, min(spacing * 10.0, 500.0))
        coordinates = np.linspace(-half_extent, half_extent, 21, dtype=np.float32)
        points: list[tuple[float, float, float]] = []
        lines: list[tuple[int, int]] = []
        for coordinate in coordinates:
            base = len(points)
            points.extend(
                (
                    (-half_extent, float(coordinate), snapshot.work_plane_z),
                    (half_extent, float(coordinate), snapshot.work_plane_z),
                    (float(coordinate), -half_extent, snapshot.work_plane_z),
                    (float(coordinate), half_extent, snapshot.work_plane_z),
                )
            )
            lines.extend(((base, base + 1), (base + 2, base + 3)))
        point_array = np.asarray(points, dtype=np.float32)
        return RenderObject(
            id=f"authoring:{snapshot.document_id}:work_plane",
            payload=LineSetPayload(
                points=point_array,
                lines=np.asarray(lines, dtype=np.int32),
                colors=np.tile(
                    np.asarray(_WORK_PLANE_COLOR, dtype=np.float32),
                    (len(point_array), 1),
                ),
            ),
            material=MaterialPayload(shader="unlit", line_width=1.0),
            visibility=True,
            metadata={
                "type": "scenario_authoring",
                "component": "work_plane",
                "geometry_signature": (
                    "work_plane",
                    float(snapshot.work_plane_z),
                    float(snapshot.grid_snap_m or 5.0),
                ),
            },
        )

    def _actor_objects(
        self,
        document_id: UUID,
        actor: ActorOverlaySnapshot,
        *,
        orientation_axes_visibility: OverlayVisibility,
        look_at_visibility: OverlayVisibility,
    ) -> tuple[RenderObject, ...]:
        points = np.asarray(actor.positions, dtype=np.float32).reshape((-1, 3))
        current_position = np.asarray(
            (actor.current_position if actor.current_position is not None else points[0]),
            dtype=np.float32,
        )
        actor_pose = (
            np.asarray(actor.orientation_matrix, dtype=float).copy()
            if actor.orientation_matrix is not None
            else np.eye(4, dtype=float)
        )
        actor_pose[:3, 3] = current_position
        frozen_actor_pose = tuple(tuple(float(value) for value in row) for row in actor_pose)
        controls = actor.mobility_controls
        control_points = np.asarray(
            tuple(control.position for control in controls),
            dtype=np.float32,
        ).reshape((-1, 3))
        # The aggregate marker anchors the persistent transform gizmo.
        # Declarative rig handles render as higher-priority semantic objects.
        marker_points = (
            np.asarray((current_position,), dtype=np.float32)
            if actor.mobility_control_rig is not None
            else np.vstack((control_points, current_position))
        )
        color = _color(actor)
        trajectory_color = (
            _SELECTED_COLOR
            if actor.trajectory_hovered
            else (
                color
                if actor.preview_provenance is PreviewProvenance.GENERATOR_PREPARED
                else _DRAFT_PATH_COLOR
            )
        )
        trajectory_key = actor.trajectory_geometry_key or repr(actor.positions)
        frame_samples_key = actor.frame_samples_geometry_key or repr(actor.frame_samples)

        def prefix(component: str) -> str:
            return stable_renderer_id(document_id, actor.actor_id, component)

        points_component = "mobility_handles"
        metadata = {
            "type": "scenario_authoring",
            "actor_id": str(actor.actor_id),
            "role": actor.role,
            # HUD-like authoring overlays must not mask the scene or one
            # another in the depth buffer. Pick priority is resolved by
            # render order and the equal-depth policy on semantic controls.
            "depth_write": False,
            "depth_compare": "<=",
            "render_order": _VISUAL_OVERLAY_RENDER_ORDER,
        }
        objects: list[RenderObject] = [
            RenderObject(
                id=prefix(points_component),
                payload=PointCloudPayload(points=marker_points),
                material=MaterialPayload(
                    base_color=(*color, 1.0),
                    shader="unlit",
                    point_size=9.0,
                ),
                visibility=actor.visible,
                metadata={
                    **metadata,
                    "component": points_component,
                    "control_vertex_map": tuple(
                        (control.kind.value, control.ordinal) for control in controls
                    ),
                    "control_vertex_count": len(control_points),
                    "current_vertex_index": len(marker_points) - 1,
                    "authoring_actor_pose": frozen_actor_pose,
                    "authoring_rotation_enabled": actor.transform_rotation_enabled,
                    "pickable": True,
                    "depth_compare": "<=",
                    "render_order": 50,
                    "interaction_role": "actor_pose",
                    "interaction_priority": 50,
                    "geometry_signature": (
                        "mobility_handles",
                        tuple(tuple(float(value) for value in point) for point in marker_points),
                    ),
                },
            ),
        ]
        moving = bool(
            len(points) > 1 and np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9)
        )
        if actor.trajectory_display is TrajectoryDisplayMode.PATH:
            segment_count = max(len(points) - 1, 0)
            arrow_count = (
                min(4, max(1, segment_count // 8))
                if moving and (actor.selected or actor.closed_trajectory)
                else 1
            )
            objects.append(
                RenderObject(
                    id=prefix("path"),
                    payload=_path_payload(points, arrow_count=arrow_count),
                    material=MaterialPayload(
                        base_color=(*trajectory_color, 1.0),
                        shader="unlit",
                        line_width=5.0 if actor.trajectory_hovered else 3.0,
                    ),
                    visibility=actor.visible and actor.trajectory_visible,
                    metadata={
                        **metadata,
                        "component": "path",
                        "path_direction": moving,
                        "pickable": False,
                        "interaction_role": "trajectory",
                        "interaction_priority": -1,
                        "preview_provenance": actor.preview_provenance.value,
                        "geometry_signature": ("path", trajectory_key, arrow_count),
                    },
                )
            )
            if actor.selected and moving:
                objects.append(
                    RenderObject(
                        id=prefix("selection_path_halo"),
                        payload=_path_hit_payload(points),
                        material=MaterialPayload(
                            base_color=(*_SELECTION_HALO_COLOR, 0.35),
                            shader="transparent",
                            line_width=9.0,
                        ),
                        visibility=actor.visible and actor.trajectory_visible,
                        metadata={
                            **metadata,
                            "component": "selection_path_halo",
                            "pickable": False,
                            "render_order": 9,
                            "interaction_role": "decorative",
                            "interaction_priority": -1,
                            "geometry_signature": ("path_halo", trajectory_key),
                        },
                    )
                )
            if len(points) >= 2:
                objects.append(
                    RenderObject(
                        id=prefix("trajectory_hit"),
                        payload=_path_hit_payload(points),
                        material=MaterialPayload(
                            base_color=(1.0, 1.0, 1.0, 0.001),
                            shader="transparent",
                            line_width=16.0,
                        ),
                        visibility=actor.visible and actor.trajectory_visible,
                        metadata={
                            **metadata,
                            "component": "trajectory_hit",
                            "pickable": True,
                            "depth_compare": "<=",
                            "render_order": 20,
                            "interaction_role": "mobility_path",
                            "interaction_priority": 20,
                            "preview_provenance": actor.preview_provenance.value,
                            "geometry_signature": ("path_hit", trajectory_key),
                        },
                    )
                )
        else:
            objects.append(
                RenderObject(
                    id=prefix("observations"),
                    payload=PointCloudPayload(points=points),
                    material=MaterialPayload(
                        base_color=(*trajectory_color, 0.8),
                        shader="unlit",
                        point_size=6.0,
                    ),
                    visibility=actor.visible and actor.trajectory_visible,
                    metadata={
                        **metadata,
                        "component": "observations",
                        "pickable": False,
                        "interaction_role": "trajectory",
                        "interaction_priority": -1,
                        "preview_provenance": actor.preview_provenance.value,
                        "geometry_signature": ("observations", trajectory_key),
                    },
                )
            )
            if actor.selected:
                objects.append(
                    RenderObject(
                        id=prefix("selection_path_halo"),
                        payload=PointCloudPayload(points=points),
                        material=MaterialPayload(
                            base_color=(*_SELECTION_HALO_COLOR, 0.35),
                            shader="transparent",
                            point_size=11.0,
                        ),
                        visibility=actor.visible and actor.trajectory_visible,
                        metadata={
                            **metadata,
                            "component": "selection_path_halo",
                            "pickable": False,
                            "interaction_role": "decorative",
                            "interaction_priority": -1,
                            "geometry_signature": ("observation_halo", trajectory_key),
                        },
                    )
                )
            objects.append(
                RenderObject(
                    id=prefix("trajectory_hit"),
                    payload=PointCloudPayload(points=points),
                    material=MaterialPayload(
                        base_color=(1.0, 1.0, 1.0, 0.001),
                        shader="transparent",
                        point_size=18.0,
                    ),
                    visibility=actor.visible and actor.trajectory_visible,
                    metadata={
                        **metadata,
                        "component": "trajectory_hit",
                        "pickable": True,
                        "render_order": 20,
                        "interaction_role": "mobility_path",
                        "interaction_priority": 20,
                        "preview_provenance": actor.preview_provenance.value,
                        "geometry_signature": ("observation_hit", trajectory_key),
                    },
                )
            )

        frame_samples = np.asarray(actor.frame_samples, dtype=np.float32).reshape((-1, 3))
        if len(frame_samples):
            objects.append(
                RenderObject(
                    id=prefix("frame_samples"),
                    payload=PointCloudPayload(points=frame_samples),
                    material=MaterialPayload(
                        base_color=(*_FRAME_SAMPLE_COLOR, 1.0),
                        shader="unlit",
                        point_size=5.0,
                    ),
                    visibility=actor.visible and actor.frame_samples_visible,
                    metadata={
                        **metadata,
                        "component": "frame_samples",
                        "sample_count": len(frame_samples),
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "preview_provenance": actor.preview_provenance.value,
                        "geometry_signature": ("frame_samples", frame_samples_key),
                    },
                )
            )

        rig = actor.mobility_control_rig
        if rig is not None:
            for guide in rig.guides:
                guide_component = f"mobility_guide_{guide.key}"
                guide_points = np.asarray(guide.points, dtype=np.float32).reshape((-1, 3))
                objects.append(
                    RenderObject(
                        id=prefix(guide_component),
                        payload=_authored_path_payload(guide_points),
                        material=MaterialPayload(
                            base_color=(*_AUTHORED_PATH_COLOR, 1.0),
                            shader="unlit",
                            line_width=guide.line_width,
                        ),
                        visibility=actor.visible,
                        metadata={
                            **metadata,
                            "component": guide_component,
                            "mobility_kind": rig.mobility_kind,
                            "guide_key": guide.key,
                            "guide_style": guide.style.value,
                            "pickable": False,
                            "interaction_role": "decorative",
                            "interaction_priority": -1,
                            "geometry_signature": (
                                "mobility_guide",
                                guide.key,
                                repr(guide.points),
                            ),
                        },
                    )
                )
            for control in rig.controls:
                control_component = f"mobility_control_{control.key}"
                hovered = control.key == actor.hovered_control_key
                objects.append(
                    RenderObject(
                        id=prefix(control_component),
                        payload=PointCloudPayload(
                            points=np.asarray((control.position,), dtype=np.float32),
                        ),
                        material=MaterialPayload(
                            base_color=(
                                *(_SELECTED_COLOR if hovered else _MOBILITY_CONTROL_COLOR),
                                1.0,
                            ),
                            shader="unlit",
                            point_size=control.point_size + (4.0 if hovered else 0.0),
                        ),
                        visibility=actor.visible,
                        metadata={
                            **metadata,
                            "component": control_component,
                            "mobility_kind": rig.mobility_kind,
                            "control_key": control.key,
                            "control_label": control.label,
                            "control_tooltip": control.tooltip,
                            "control_operation": control.operation.value,
                            "control_constraint": control.constraint.value,
                            "control_glyph": control.glyph.value,
                            "control_ordinal": control.ordinal,
                            "pickable": True,
                            "depth_compare": "<=",
                            "render_order": 100,
                            "interaction_role": "mobility_control",
                            "interaction_priority": 100,
                            "geometry_signature": (
                                "mobility_control",
                                control.key,
                                control.position,
                            ),
                        },
                    )
                )
                # Fixed semantic controls benefit from always-visible names.
                # Waypoint ordinals can be unbounded, so keep those as hover
                # tooltips rather than creating one text object per vertex.
                if control.ordinal is None:
                    label_component = f"mobility_control_label_{control.key}"
                    objects.append(
                        RenderObject(
                            id=prefix(label_component),
                            payload=TextLabelPayload(
                                control.label,
                                font_size=0.24,
                                screen_space=True,
                            ),
                            material=MaterialPayload(
                                base_color=(*_MOBILITY_CONTROL_COLOR, 1.0),
                                shader="unlit",
                            ),
                            transform=Transform.from_translation(control.position),
                            visibility=actor.visible,
                            metadata={
                                **metadata,
                                "component": label_component,
                                "control_key": control.key,
                                "pickable": False,
                                "interaction_role": "decorative",
                                "interaction_priority": -1,
                                "geometry_signature": (
                                    "mobility_control_label",
                                    control.key,
                                    control.label,
                                ),
                            },
                        )
                    )
        authored_path = np.asarray(actor.authored_path, dtype=np.float32).reshape((-1, 3))
        if rig is None and len(authored_path) >= 2:
            objects.append(
                RenderObject(
                    id=prefix("authored_path"),
                    payload=_authored_path_payload(authored_path),
                    material=MaterialPayload(
                        base_color=(*_AUTHORED_PATH_COLOR, 1.0),
                        shader="unlit",
                        line_width=1.5,
                    ),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "authored_path",
                        "control_point_count": len(authored_path),
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": ("authored_path", repr(actor.authored_path)),
                    },
                )
            )

        if actor.trajectory_display is TrajectoryDisplayMode.PATH:
            start_end_points = (
                np.asarray((points[0],), dtype=np.float32)
                if actor.closed_trajectory
                else np.asarray((points[0], points[-1]), dtype=np.float32)
            )
            start_end_colors = (
                np.asarray(((0.15, 0.9, 0.35),), dtype=np.float32)
                if actor.closed_trajectory
                else np.asarray(
                    ((0.15, 0.9, 0.35), (0.95, 0.2, 0.2)),
                    dtype=np.float32,
                )
            )
            objects.append(
                RenderObject(
                    id=prefix("start_end"),
                    payload=PointCloudPayload(
                        points=start_end_points,
                        colors=start_end_colors,
                    ),
                    material=MaterialPayload(shader="unlit", point_size=12.0),
                    visibility=actor.visible and actor.trajectory_visible,
                    metadata={
                        **metadata,
                        "component": "start_end",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "start_end",
                            trajectory_key,
                            actor.closed_trajectory,
                        ),
                    },
                )
            )
        objects.append(
            RenderObject(
                id=prefix("label"),
                payload=TextLabelPayload(actor.name, font_size=0.32, screen_space=True),
                material=MaterialPayload(base_color=(*color, 1.0), shader="unlit"),
                transform=Transform.from_translation(current_position),
                visibility=actor.visible,
                metadata={
                    **metadata,
                    "component": "label",
                    "pickable": False,
                    "interaction_role": "decorative",
                    "interaction_priority": -1,
                    "geometry_signature": ("label", actor.name),
                },
            )
        )

        # Selection and validity use additional shapes as well as color so the
        # states remain distinguishable under color-vision deficiencies.
        if actor.selected or actor.status is not ActorVisualState.COMPLETE:
            status_color = _SELECTED_COLOR if actor.selected else color
            status_points = np.asarray((current_position,), dtype=np.float32)
            if actor.status is ActorVisualState.INVALID:
                epsilon = np.float32(0.12)
                status_points = np.asarray(
                    (
                        current_position + (-epsilon, 0.0, 0.0),
                        current_position + (epsilon, 0.0, 0.0),
                        current_position + (0.0, -epsilon, 0.0),
                        current_position + (0.0, epsilon, 0.0),
                    ),
                    dtype=np.float32,
                )
            objects.append(
                RenderObject(
                    id=prefix("status"),
                    payload=PointCloudPayload(
                        points=status_points,
                        colors=np.tile(status_color, (len(status_points), 1)),
                    ),
                    material=MaterialPayload(shader="unlit", point_size=16.0),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "status",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "status",
                            actor.status.value,
                            actor.selected,
                            tuple(float(value) for value in current_position),
                        ),
                    },
                )
            )

        pending_points = np.asarray(actor.pending_positions, dtype=np.float32).reshape((-1, 3))
        if len(pending_points):
            pending_key = actor.pending_geometry_key or repr(actor.pending_positions)
            pending_observations = (
                actor.pending_trajectory_display is TrajectoryDisplayMode.OBSERVATIONS
            )
            objects.append(
                RenderObject(
                    id=prefix("pending_observations" if pending_observations else "pending_path"),
                    payload=(
                        PointCloudPayload(points=pending_points)
                        if pending_observations
                        else _path_payload(pending_points, arrow_count=0)
                    ),
                    material=MaterialPayload(
                        base_color=(*_GHOST_COLOR, 0.7),
                        shader="transparent",
                        point_size=7.0 if pending_observations else None,
                        line_width=None if pending_observations else 3.0,
                    ),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": (
                            "pending_observations" if pending_observations else "pending_path"
                        ),
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "pending_observations" if pending_observations else "pending_path",
                            pending_key,
                        ),
                    },
                )
            )
            pending_current = np.asarray(
                (
                    actor.pending_current_position
                    if actor.pending_current_position is not None
                    else pending_points[min(len(pending_points) - 1, 0)]
                ),
                dtype=np.float32,
            )
            objects.append(
                RenderObject(
                    id=prefix("pending_pose"),
                    payload=PointCloudPayload(points=np.asarray((pending_current,))),
                    material=MaterialPayload(
                        base_color=(*_GHOST_COLOR, 0.75),
                        shader="transparent",
                        point_size=14.0,
                    ),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "pending_pose",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "pending_pose",
                            tuple(float(value) for value in pending_current),
                        ),
                    },
                )
            )
        if actor.pending_mobility_control_rig is not None:
            pending_rig = actor.pending_mobility_control_rig
            for guide in pending_rig.guides:
                objects.append(
                    RenderObject(
                        id=prefix(f"pending_guide_{guide.key}"),
                        payload=_authored_path_payload(
                            np.asarray(guide.points, dtype=np.float32).reshape((-1, 3))
                        ),
                        material=MaterialPayload(
                            base_color=(*_GHOST_COLOR, 0.55),
                            shader="transparent",
                            line_width=guide.line_width,
                        ),
                        visibility=actor.visible,
                        metadata={
                            **metadata,
                            "component": f"pending_guide_{guide.key}",
                            "pickable": False,
                            "interaction_role": "decorative",
                            "interaction_priority": -1,
                            "geometry_signature": (
                                "pending_guide",
                                guide.key,
                                repr(guide.points),
                            ),
                        },
                    )
                )
            for control in pending_rig.controls:
                objects.append(
                    RenderObject(
                        id=prefix(f"pending_control_{control.key}"),
                        payload=PointCloudPayload(
                            points=np.asarray((control.position,), dtype=np.float32)
                        ),
                        material=MaterialPayload(
                            base_color=(*_GHOST_COLOR, 0.65),
                            shader="transparent",
                            point_size=control.point_size,
                        ),
                        visibility=actor.visible,
                        metadata={
                            **metadata,
                            "component": f"pending_control_{control.key}",
                            "pickable": False,
                            "interaction_role": "decorative",
                            "interaction_priority": -1,
                            "geometry_signature": (
                                "pending_control",
                                control.key,
                                control.position,
                            ),
                        },
                    )
                )

        if actor.orientation_matrix is not None and _overlay_visible(
            orientation_axes_visibility, actor
        ):
            objects.append(
                RenderObject(
                    id=prefix("orientation"),
                    payload=OrientationFramePayload(size=1.0, thickness=3.0),
                    transform=Transform(actor_pose),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "orientation",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": ("orientation_frame",),
                    },
                )
            )
        if actor.pending_orientation_matrix is not None:
            pending_pose = np.asarray(actor.pending_orientation_matrix, dtype=float).copy()
            if actor.pending_current_position is not None:
                pending_pose[:3, 3] = np.asarray(actor.pending_current_position, dtype=float)
            objects.append(
                RenderObject(
                    id=prefix("pending_orientation"),
                    payload=OrientationFramePayload(size=1.1, thickness=2.0),
                    transform=Transform(pending_pose),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "pending_orientation",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": ("pending_orientation_frame",),
                    },
                )
            )
        if actor.target_asset is not None:
            target_transform = actor_pose @ np.asarray(
                actor.target_asset.local_to_actor,
                dtype=float,
            )
            objects.append(
                RenderObject(
                    id=prefix("target"),
                    payload=actor.target_asset.payload,
                    material=actor.target_asset.material,
                    transform=Transform(target_transform),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "target",
                        "asset_cache_key": actor.target_asset.cache_key,
                        "depth_write": True,
                        "depth_compare": "<",
                        "render_order": 0,
                        "authoring_actor_pose": frozen_actor_pose,
                        "authoring_rotation_enabled": actor.transform_rotation_enabled,
                        "geometry_signature": ("target", actor.target_asset.cache_key),
                    },
                )
            )
        if actor.pending_target_asset is not None:
            pending_actor_pose = (
                np.asarray(actor.pending_orientation_matrix, dtype=float).copy()
                if actor.pending_orientation_matrix is not None
                else actor_pose.copy()
            )
            if actor.pending_current_position is not None:
                pending_actor_pose[:3, 3] = np.asarray(
                    actor.pending_current_position,
                    dtype=float,
                )
            pending_target_transform = pending_actor_pose @ np.asarray(
                actor.pending_target_asset.local_to_actor,
                dtype=float,
            )
            objects.append(
                RenderObject(
                    id=prefix("pending_target"),
                    payload=actor.pending_target_asset.payload,
                    material=MaterialPayload(
                        base_color=(*_GHOST_COLOR, 0.45),
                        shader="transparent",
                    ),
                    transform=Transform(pending_target_transform),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "pending_target",
                        "asset_cache_key": actor.pending_target_asset.cache_key,
                        "pickable": False,
                        "depth_write": False,
                        "render_order": _VISUAL_OVERLAY_RENDER_ORDER,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "pending_target",
                            actor.pending_target_asset.cache_key,
                        ),
                    },
                )
            )
        if actor.look_at_position is not None and _overlay_visible(look_at_visibility, actor):
            look_at = np.asarray(actor.look_at_position, dtype=np.float32)
            ray_points = np.asarray((current_position, look_at), dtype=np.float32)
            objects.append(
                RenderObject(
                    id=prefix("look_at"),
                    payload=LineSetPayload(
                        points=ray_points,
                        lines=np.asarray(((0, 1),), dtype=np.int32),
                        colors=np.tile(
                            np.asarray(_LOOK_AT_COLOR, dtype=np.float32),
                            (2, 1),
                        ),
                    ),
                    material=MaterialPayload(shader="unlit", line_width=2.0),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "look_at",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "look_at",
                            tuple(float(value) for value in current_position),
                            actor.look_at_position,
                        ),
                    },
                )
            )
        if actor.pending_look_at_position is not None:
            pending_origin = np.asarray(
                (
                    actor.pending_current_position
                    if actor.pending_current_position is not None
                    else current_position
                ),
                dtype=np.float32,
            )
            pending_target = np.asarray(
                actor.pending_look_at_position,
                dtype=np.float32,
            )
            objects.append(
                RenderObject(
                    id=prefix("pending_look_at"),
                    payload=LineSetPayload(
                        points=np.asarray(
                            (pending_origin, pending_target),
                            dtype=np.float32,
                        ),
                        lines=np.asarray(((0, 1),), dtype=np.int32),
                    ),
                    material=MaterialPayload(
                        base_color=(*_GHOST_COLOR, 0.7),
                        shader="transparent",
                        line_width=2.0,
                    ),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "pending_look_at",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "pending_look_at",
                            tuple(float(value) for value in pending_origin),
                            actor.pending_look_at_position,
                        ),
                    },
                )
            )
        if actor.group_origin_position is not None:
            group_origin = np.asarray(actor.group_origin_position, dtype=np.float32)
            tether_points = np.asarray((current_position, group_origin), dtype=np.float32)
            objects.append(
                RenderObject(
                    id=prefix("group_tether"),
                    payload=LineSetPayload(
                        points=tether_points,
                        lines=np.asarray(((0, 1),), dtype=np.int32),
                    ),
                    material=MaterialPayload(
                        base_color=(*_GROUP_CONTEXT_COLOR, 0.75),
                        shader="transparent",
                        line_width=2.0,
                    ),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "group_tether",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": (
                            "group_tether",
                            tuple(float(value) for value in current_position),
                            actor.group_origin_position,
                        ),
                    },
                )
            )
        if actor.group_frame_matrix is not None:
            objects.append(
                RenderObject(
                    id=prefix("group_frame"),
                    payload=OrientationFramePayload(size=1.25, thickness=2.0),
                    transform=Transform(np.asarray(actor.group_frame_matrix, dtype=float)),
                    visibility=actor.visible,
                    metadata={
                        **metadata,
                        "component": "group_frame",
                        "pickable": False,
                        "interaction_role": "decorative",
                        "interaction_priority": -1,
                        "geometry_signature": ("group_frame",),
                    },
                )
            )
        return tuple(objects)

    def focus_actor(self, actor_id: UUID) -> bool:
        """Focus the camera on an actor's reconciled sample centroid."""

        self._flush_pending_snapshot()
        points = self._actor_positions.get(UUID(str(actor_id)))
        if points is None or len(points) == 0:
            return False
        return bool(self.renderer.focus_camera(points.mean(axis=0)))

    def fit_all(self) -> bool:
        """Reset camera bounds when the reconciled snapshot has actors."""

        self._flush_pending_snapshot()
        if not self._actor_positions:
            return False
        reset = getattr(self.renderer, "reset_camera_bounds", None)
        if not callable(reset):
            return False
        reset()
        return True

    def renderer_objects(self) -> Mapping[str, Any]:
        """Return a read-only snapshot after queued reconciliation completes."""

        self._flush_pending_snapshot()
        return MappingProxyType(dict(self._owned_objects))

    def current_snapshot(self) -> OverlaySnapshot | None:
        """Return the newest complete snapshot after flushing queued reconciliation."""

        self._flush_pending_snapshot()
        return self._snapshot

    def close(self) -> None:
        """Release authoring handlers before native renderer resources."""

        if self._closed:
            return
        self._flush_timer.stop()
        self._pending_snapshot = None
        self.deactivate()
        batch_updates = getattr(self.renderer, "batch_updates", None)
        context = batch_updates() if callable(batch_updates) else nullcontext()
        try:
            with context:
                for object_id in sorted(self._owned_objects):
                    self.renderer.remove_object(object_id)
                if self._owned_objects:
                    self.renderer.request_redraw()
        finally:
            self._owned_objects.clear()
            self._actor_positions.clear()
            self._snapshot = None
            self._desired_gizmo_actor_id = None
            self._closed = True
            self.renderer.close()


class ScenarioAuthoringViewport(QWidget):
    """Qt host that constructs QRenderWidget directly with its final parent."""

    input_received = Signal(object)

    def __init__(
        self,
        visualizer: Any,
        parent: QWidget | None = None,
        *,
        renderer_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("scenarioAuthoringViewport")
        self.setMinimumSize(420, 300)
        self.port: PygfxScenarioAuthoringViewportPort | None = None
        self.initialization_error: Exception | None = None
        try:
            self.port = PygfxScenarioAuthoringViewportPort(
                self,
                visualizer,
                renderer_factory=renderer_factory,
            )
            self.port.activate(self.input_received.emit)
        except Exception as exc:
            self.initialization_error = exc
            if self.layout() is None:
                layout = QVBoxLayout(self)
                layout.setContentsMargins(12, 12, 12, 12)
            else:
                layout = self.layout()
            label = QLabel(f"Pygfx authoring viewport unavailable:\n{exc}", self)
            label.setWordWrap(True)
            layout.addWidget(label)

    @property
    def available(self) -> bool:
        return self.port is not None

    def close_viewport(self) -> None:
        """Release the port once and leave the host in an unavailable state."""

        if self.port is not None:
            self.port.close()
            self.port = None
