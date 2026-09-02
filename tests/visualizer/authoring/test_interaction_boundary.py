"""Focused leak, picking, and reconciliation tests for the authoring viewport."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtWidgets import QWidget

from visualizer.src.authoring.domain import MobilityControl, MobilityControlKind
from visualizer.src.authoring.interaction import (
    InteractionSession,
    PygfxInteractionRouter,
    intersect_horizontal_plane,
    surface_position_from_pick_info,
    vertex_position_from_pick_info,
)
from visualizer.src.authoring.viewport import PygfxScenarioAuthoringViewportPort
from visualizer.src.authoring.viewport_port import (
    ActorOverlaySnapshot,
    AuthoringTool,
    HitResult,
    OverlaySnapshot,
    PointerInput,
    PointerPhase,
    TargetOverlayAsset,
    TransformInput,
    TransformPhase,
    parse_renderer_id,
    stable_renderer_id,
)
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
)


def _native_points(points: tuple[tuple[float, float, float], ...]) -> SimpleNamespace:
    return SimpleNamespace(
        geometry=SimpleNamespace(
            positions=SimpleNamespace(data=np.asarray(points, dtype=float)),
            indices=None,
        ),
        world=SimpleNamespace(matrix=np.eye(4)),
    )


def test_transformed_mesh_pick_uses_barycentrics_and_world_matrix() -> None:
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        dtype=float,
    )
    # Rotate 90 degrees around Z after nonuniform XY scale, then translate.
    transform = np.asarray(
        (
            (0.0, -3.0, 0.0, 10.0),
            (2.0, 0.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 4.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    mesh = SimpleNamespace(
        geometry=SimpleNamespace(
            positions=SimpleNamespace(data=positions),
            indices=SimpleNamespace(data=np.asarray(((0, 1, 2),))),
        ),
        world=SimpleNamespace(matrix=transform),
    )

    result = surface_position_from_pick_info(
        {
            "world_object": mesh,
            "face_index": 0,
            "face_coord": (0.25, 0.25, 0.5),
        }
    )

    assert result == pytest.approx((7.0, 21.0, 4.0))
    assert surface_position_from_pick_info(
        {
            "world_object": mesh,
            "face_index": 0,
            "face_coord": (1.0, 0.0, 0.0),
        }
    ) == pytest.approx((10.0, 20.0, 4.0))


@pytest.mark.parametrize("indices", [((-1, 1, 2),), ((0, 1, 9),)])
def test_mesh_pick_rejects_out_of_bounds_vertex_indices(indices) -> None:
    mesh = SimpleNamespace(
        geometry=SimpleNamespace(
            positions=SimpleNamespace(data=np.asarray(((0.0, 0.0, 0.0),) * 3, dtype=float)),
            indices=SimpleNamespace(data=np.asarray(indices)),
        ),
        world=SimpleNamespace(matrix=np.eye(4)),
    )
    assert (
        surface_position_from_pick_info(
            {
                "world_object": mesh,
                "face_index": 0,
                "face_coord": (0.2, 0.3, 0.5),
            }
        )
        is None
    )


class _EventBackend:
    def __init__(self) -> None:
        self.logical_size = (100.0, 100.0)
        self.add_calls = 0
        self.remove_calls = 0
        self._registrations: set[tuple[object, object, tuple[str, ...]]] = set()

    @staticmethod
    def _key(handler, events):
        return (
            getattr(handler, "__self__", None),
            getattr(handler, "__func__", handler),
            tuple(events),
        )

    def add_event_handler(self, handler, *events) -> None:
        self.add_calls += 1
        self._registrations.add(self._key(handler, events))

    def remove_event_handler(self, handler, *events) -> None:
        self.remove_calls += 1
        self._registrations.discard(self._key(handler, events))

    @property
    def active_registration_count(self) -> int:
        return len(self._registrations)


class _Gizmo:
    def __init__(self) -> None:
        self.visible = False
        self.objects: list[object | None] = []

    def set_object(self, value) -> None:
        self.objects.append(value)


class _AuthoringRuntime:
    def __init__(self, owner) -> None:
        self.owner = owner
        self.camera_events = []
        self.hover_events = []
        self.mpc_selection_events = []
        self.authoring_attachments = []
        self.authoring_callback = None
        self.live_attachments = []
        self.live_callback = None
        self.synced_poses = []
        self.hide_calls = 0

    @property
    def logical_size(self):
        return self.owner._renderer.logical_size

    def camera_matrices(self):
        camera = self.owner._camera
        return camera.projection_matrix_inverse, camera.view_matrix

    def add_event_handler(self, callback, *events) -> None:
        self.owner._renderer.add_event_handler(callback, *events)

    def remove_event_handler(self, callback, *events) -> None:
        self.owner._renderer.remove_event_handler(callback, *events)

    def object_id_for_native(self, native):
        return self.owner._reverse_objects.get(id(native))

    def set_camera_mode(self, mode):
        self.owner.camera_mode = mode
        return True

    def ensure_gizmo(self, *, authoring=True):
        if self.owner._transform_gizmo is None:
            return self.owner._ensure_transform_gizmo()
        return self.owner._transform_gizmo

    def hide_gizmo(self) -> None:
        self.hide_calls += 1
        gizmo = self.owner._transform_gizmo
        if gizmo is not None:
            gizmo.set_object(None)
            gizmo.visible = False

    def sync_gizmo_pose(self, object_id, transform) -> bool:
        self.synced_poses.append((object_id, transform))
        return True

    def route_camera_event(self, event) -> None:
        self.camera_events.append(event)

    def route_hover_event(self, event) -> None:
        self.hover_events.append(event)

    def route_mpc_path_selection_event(self, event) -> None:
        self.mpc_selection_events.append(event)

    def update_before_render(
        self,
        _event,
        *,
        route_camera=True,
        authoring=True,
    ) -> None:
        return None

    def is_gizmo_event(self, _event) -> bool:
        return False

    def attach_gizmo(self, object_id, callback) -> bool:
        self.authoring_attachments.append(object_id)
        self.authoring_callback = callback
        return True

    def attach_live_preview_gizmo(self, object_id, callback) -> bool:
        self.live_attachments.append(object_id)
        self.live_callback = callback
        return True


class _RouterRenderer:
    def __init__(self) -> None:
        self._renderer = _EventBackend()
        self._camera = SimpleNamespace(
            projection_matrix_inverse=np.eye(4),
            view_matrix=np.eye(4),
        )
        self._transform_gizmo = None
        self._reverse_objects: dict[int, str] = {}
        self.gizmo_creations = 0
        self.camera_mode = "orbit"
        self._authoring_runtime = _AuthoringRuntime(self)

    def scenario_authoring_runtime(self):
        return self._authoring_runtime

    def pygfx_interaction_router(self):
        router = getattr(self, "_interaction_router", None)
        if router is None:
            router = PygfxInteractionRouter(self)
            self._interaction_router = router
        return router

    def _ensure_transform_gizmo(self) -> _Gizmo:
        self.gizmo_creations += 1
        self._transform_gizmo = _Gizmo()
        return self._transform_gizmo


def test_work_plane_miss_and_horizontal_grid_snap_preserve_plane_height() -> None:
    renderer = _RouterRenderer()
    router = PygfxInteractionRouter(renderer)
    router.set_work_plane(0.75, grid_snap_m=0.5)
    event = SimpleNamespace(x=68.0, y=62.0, pick_info={}, target=None)

    hit = router.resolve_hit(event)

    assert hit is not None
    assert hit.world_position == pytest.approx((0.5, 0.0, 0.75))
    router.set_work_plane(-1.0, grid_snap_m=0.5)
    assert router.resolve_hit(event) is None
    assert (
        intersect_horizontal_plane(
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            0.0,
        )
        is None
    )


def test_visible_work_plane_pick_uses_cursor_ray_instead_of_grid_vertex() -> None:
    document_id = uuid4()
    native_grid = _native_points(((50.0, 50.0, 0.75), (-50.0, -50.0, 0.75)))
    renderer = _RouterRenderer()
    renderer._reverse_objects[id(native_grid)] = f"authoring:{document_id}:work_plane"
    router = PygfxInteractionRouter(renderer)
    router.set_work_plane(0.75, grid_snap_m=0.5)

    hit = router.resolve_hit(
        SimpleNamespace(
            x=68.0,
            y=62.0,
            target=native_grid,
            pick_info={"vertex_index": 0},
        )
    )

    assert hit is not None
    assert hit.component == "work_plane"
    assert hit.surface is False
    assert hit.vertex_index is None
    assert hit.world_position == pytest.approx((0.5, 0.0, 0.75))
    assert hit.world_position != pytest.approx((50.0, 50.0, 0.75))


def test_point_pick_preserves_mobility_control_vertex_index_and_world_position() -> None:
    document_id = uuid4()
    actor_id = uuid4()
    points = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    native = _native_points(points)
    native.world.matrix[:3, 3] = (10.0, 0.0, -2.0)
    renderer = _RouterRenderer()
    object_id = stable_renderer_id(document_id, actor_id, "mobility_handles")
    renderer._reverse_objects[id(native)] = object_id
    router = PygfxInteractionRouter(renderer)

    hit = router.resolve_hit(
        SimpleNamespace(
            x=0.0,
            y=0.0,
            target=native,
            pick_info={"vertex_index": 2},
        )
    )

    assert vertex_position_from_pick_info(
        {"world_object": native, "vertex_index": 2}
    ) == pytest.approx((17.0, 8.0, 7.0))
    assert hit is not None
    assert hit.actor_id == actor_id
    assert hit.component == "mobility_handles"
    assert hit.vertex_index == 2
    assert hit.world_position == pytest.approx((17.0, 8.0, 7.0))


def test_trajectory_proxy_pick_uses_cursor_plane_position_not_line_vertex() -> None:
    document_id = uuid4()
    actor_id = uuid4()
    native = _native_points(((50.0, 50.0, 0.75), (-50.0, -50.0, 0.75)))
    renderer = _RouterRenderer()
    renderer._reverse_objects[id(native)] = stable_renderer_id(
        document_id,
        actor_id,
        "trajectory_hit",
    )
    router = PygfxInteractionRouter(renderer)

    hit = router.resolve_hit(
        SimpleNamespace(
            x=68.0,
            y=62.0,
            target=native,
            pick_info={"vertex_index": 0},
        )
    )

    assert hit is not None
    assert hit.component == "trajectory_hit"
    assert hit.world_position == pytest.approx((0.36, -0.24, 0.75))
    assert hit.world_position != pytest.approx((50.0, 50.0, 0.75))


def test_router_has_one_handler_set_and_one_gizmo_across_100_cycles() -> None:
    renderer = _RouterRenderer()
    router = PygfxInteractionRouter(renderer)

    def sink(_value) -> None:
        return None

    for index in range(100):
        assert router.set_camera_mode("fly" if index % 2 else "orbit")
        router.activate(InteractionSession.AUTHORING, sink)
        router.activate(InteractionSession.AUTHORING, sink)
        assert renderer._renderer.active_registration_count == 1
        router.deactivate()
        assert renderer._renderer.active_registration_count == 0

    assert renderer.gizmo_creations == 1
    assert renderer._renderer.add_calls == 100
    assert renderer._renderer.remove_calls == 100


def test_gizmo_attachment_is_not_a_transform_until_the_first_change() -> None:
    renderer = _RouterRenderer()
    router = PygfxInteractionRouter(renderer)
    received = []
    document_id = uuid4()
    actor_id = uuid4()
    object_id = stable_renderer_id(document_id, actor_id, "mobility_handles")
    initial = np.eye(4)
    changed = np.eye(4)
    changed[:3, 3] = (2.0, 3.0, 4.0)

    router.activate(InteractionSession.AUTHORING, received.append)
    router.set_tool(AuthoringTool.MOVE)
    assert router.attach_transform_gizmo(object_id)
    assert router.attach_transform_gizmo(object_id)
    assert renderer._authoring_runtime.authoring_attachments == [object_id]

    callback = renderer._authoring_runtime.authoring_callback
    assert callback is not None
    callback({"phase": "selected", "object_id": object_id, "transform": initial})
    assert received == []

    callback({"phase": "changed", "object_id": object_id, "transform": changed})
    # A normal viewport reconcile reasserts the same semantic anchor while the
    # gesture is active. It must not switch IDs or cancel the drag.
    assert router.attach_transform_gizmo(object_id)
    callback({"phase": "committed", "object_id": object_id, "transform": changed})

    transforms = [value for value in received if isinstance(value, TransformInput)]
    assert [value.phase for value in transforms] == [
        TransformPhase.BEGIN,
        TransformPhase.UPDATE,
        TransformPhase.COMMIT,
    ]
    np.testing.assert_allclose(transforms[0].matrix, initial)
    np.testing.assert_allclose(transforms[-1].matrix, changed)
    router.deactivate()


def test_idle_pose_sync_refreshes_next_gesture_baseline() -> None:
    renderer = _RouterRenderer()
    router = PygfxInteractionRouter(renderer)
    received = []
    object_id = stable_renderer_id(uuid4(), uuid4(), "mobility_handles")
    initial = np.eye(4)
    synced = np.eye(4)
    synced[0, 3] = 10.0
    changed = np.eye(4)
    changed[0, 3] = 11.0

    router.activate(InteractionSession.AUTHORING, received.append)
    router.set_tool(AuthoringTool.MOVE)
    assert router.attach_transform_gizmo(object_id)
    callback = renderer._authoring_runtime.authoring_callback
    assert callback is not None
    callback({"phase": "selected", "object_id": object_id, "transform": initial})

    assert router.sync_transform_pose(object_id, synced)
    callback({"phase": "changed", "object_id": object_id, "transform": changed})

    transforms = [value for value in received if isinstance(value, TransformInput)]
    assert [value.phase for value in transforms] == [
        TransformPhase.BEGIN,
        TransformPhase.UPDATE,
    ]
    np.testing.assert_allclose(transforms[0].matrix, synced)
    router.clear_transform_gizmo()


@pytest.mark.parametrize("cancel_action", ["clear", "deactivate"])
def test_transform_cancellation_emits_one_reentrant_safe_cancel(
    cancel_action: str,
) -> None:
    renderer = _RouterRenderer()
    router = PygfxInteractionRouter(renderer)
    received = []
    object_id = stable_renderer_id(uuid4(), uuid4(), "mobility_handles")
    initial = np.eye(4)
    changed = np.eye(4)
    changed[1, 3] = 2.0

    def sink(value) -> None:
        received.append(value)
        if isinstance(value, TransformInput) and value.phase is TransformPhase.CANCEL:
            router.clear_transform_gizmo()

    router.activate(InteractionSession.AUTHORING, sink)
    router.set_tool(AuthoringTool.MOVE)
    assert router.attach_transform_gizmo(object_id)
    callback = renderer._authoring_runtime.authoring_callback
    assert callback is not None
    callback({"phase": "selected", "object_id": object_id, "transform": initial})
    callback({"phase": "changed", "object_id": object_id, "transform": changed})

    if cancel_action == "clear":
        router.clear_transform_gizmo()
    else:
        router.deactivate()

    transforms = [value for value in received if isinstance(value, TransformInput)]
    assert [value.phase for value in transforms] == [
        TransformPhase.BEGIN,
        TransformPhase.UPDATE,
        TransformPhase.CANCEL,
    ]
    if cancel_action == "deactivate":
        assert router.session is None
        assert renderer._renderer.active_registration_count == 0
    else:
        assert router.session is InteractionSession.AUTHORING
        router.deactivate()


def test_authoring_target_pick_does_not_bypass_workspace_gizmo_policy() -> None:
    renderer = _RouterRenderer()
    router = PygfxInteractionRouter(renderer)
    received = []
    native = object()
    document_id = uuid4()
    actor_id = uuid4()
    target_id = stable_renderer_id(document_id, actor_id, "target")
    renderer._reverse_objects[id(native)] = target_id
    router.activate(InteractionSession.AUTHORING, received.append)
    router.set_tool(AuthoringTool.MOVE)

    router._route_event(
        SimpleNamespace(
            type="pointer_down",
            x=25.0,
            y=25.0,
            button=1,
            buttons=(1,),
            modifiers=(),
            pick_info={
                "world_object": native,
                "world_position": (1.0, 2.0, 3.0),
            },
            target=native,
        )
    )

    assert renderer._authoring_runtime.authoring_attachments == []
    hits = [value for value in received if isinstance(value, HitResult)]
    assert len(hits) == 1
    assert hits[0].actor_id == actor_id
    assert hits[0].component == "target"
    router.deactivate()


def test_visualization_live_preview_and_authoring_share_one_registration() -> None:
    renderer = _RouterRenderer()
    native = object()
    renderer._reverse_objects[id(native)] = "node:tx_0::marker"
    router = renderer.pygfx_interaction_router()
    live_events = []
    authoring_events = []

    router.activate(InteractionSession.VISUALIZATION, lambda _event: None)
    pointer_move = SimpleNamespace(type="pointer_move", target=None)
    router._route_event(pointer_move)
    assert renderer._authoring_runtime.camera_events == [pointer_move]
    assert renderer._authoring_runtime.hover_events == [pointer_move]
    assert renderer._authoring_runtime.mpc_selection_events == [pointer_move]

    router.activate(InteractionSession.LIVE_PREVIEW, live_events.append)
    stops = []
    router._route_event(
        SimpleNamespace(
            type="pointer_down",
            button=1,
            modifiers=(),
            target=native,
            stop_propagation=lambda: stops.append(True),
        )
    )

    assert renderer._authoring_runtime.live_attachments == ["node:tx_0::marker"]
    assert stops == [True]
    assert renderer._authoring_runtime.mpc_selection_events == [pointer_move]
    live_pointer_move = SimpleNamespace(type="pointer_move", target=None)
    router._route_event(live_pointer_move)
    assert renderer._authoring_runtime.mpc_selection_events == [pointer_move]
    renderer._authoring_runtime.live_callback({"phase": "selected"})
    assert live_events == [{"phase": "selected"}]

    router.activate(InteractionSession.AUTHORING, authoring_events.append)
    renderer._authoring_runtime.live_callback({"phase": "changed"})
    assert live_events == [{"phase": "selected"}]
    assert authoring_events == []
    assert renderer._renderer.active_registration_count == 1
    assert renderer._renderer.add_calls == 1
    assert renderer._renderer.remove_calls == 0
    assert renderer.gizmo_creations == 1

    router.deactivate()
    assert renderer._renderer.active_registration_count == 0
    assert renderer._renderer.remove_calls == 1


@pytest.mark.parametrize(
    ("component", "vertex_index"),
    (("mobility_handles", 1), ("path", None)),
)
def test_actor_drag_uses_fixed_plane_and_preserves_source_identity(
    component: str,
    vertex_index: int | None,
) -> None:
    renderer = _RouterRenderer()
    router = PygfxInteractionRouter(renderer)
    router.set_tool(AuthoringTool.MOVE)
    received = []
    router.activate(InteractionSession.AUTHORING, received.append)
    source = HitResult(
        world_position=(0.0, 0.0, 0.5),
        renderer_object_id=stable_renderer_id(uuid4(), uuid4(), component),
        actor_id=uuid4(),
        component=component,
        vertex_index=vertex_index,
    )
    router.begin_drag_plane(0.5, source)

    router._route_event(
        SimpleNamespace(
            type="pointer_move",
            x=75.0,
            y=50.0,
            button=0,
            buttons=(1,),
            modifiers=(),
            pick_info={},
            target=None,
        )
    )
    router._route_event(
        SimpleNamespace(
            type="pointer_up",
            x=100.0,
            y=50.0,
            button=1,
            buttons=(),
            modifiers=(),
            pick_info={},
            target=None,
        )
    )

    pointer_phases = [value.phase for value in received if isinstance(value, PointerInput)]
    drag_hits = [value for value in received if isinstance(value, HitResult)]
    assert pointer_phases == [PointerPhase.MOVE, PointerPhase.UP]
    np.testing.assert_allclose(
        [hit.world_position for hit in drag_hits],
        ((0.5, 0.0, 0.5), (1.0, 0.0, 0.5)),
    )
    assert all(hit.actor_id == source.actor_id for hit in drag_hits)
    assert all(hit.component == component for hit in drag_hits)
    assert all(hit.vertex_index == vertex_index for hit in drag_hits)
    assert not renderer._authoring_runtime.camera_events
    router.deactivate()


def test_free_control_drag_uses_camera_facing_plane_and_can_change_height() -> None:
    renderer = _RouterRenderer()
    angle = np.deg2rad(-45.0)
    renderer._camera.view_matrix = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle), 0.0),
            (0.0, np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    router = PygfxInteractionRouter(renderer)
    router.set_tool(AuthoringTool.MOVE)
    received = []
    router.activate(InteractionSession.AUTHORING, received.append)
    source = HitResult(
        world_position=(0.0, 0.0, 0.5),
        renderer_object_id=stable_renderer_id(uuid4(), uuid4(), "mobility_control_end"),
        actor_id=uuid4(),
        component="mobility_control_end",
    )
    router.begin_control_drag("free", source)

    router._route_event(
        SimpleNamespace(
            type="pointer_move",
            x=50.0,
            y=25.0,
            button=0,
            buttons=(1,),
            modifiers=(),
            pick_info={},
            target=None,
        )
    )

    drag_hits = [value for value in received if isinstance(value, HitResult)]
    assert len(drag_hits) == 1
    assert drag_hits[0].actor_id == source.actor_id
    assert drag_hits[0].component == source.component
    assert drag_hits[0].world_position[2] != pytest.approx(source.world_position[2])
    router.end_drag()
    router.deactivate()


class _FakeRenderer:
    capabilities = RendererCapabilities(scenario_authoring=True)

    def __init__(self, _visualizer) -> None:
        self.objects = {}
        self.removed: list[str] = []
        self.ensure_calls = 0
        self.redraw_calls = 0
        self.batch_calls = 0
        self.closed = False
        self._renderer = _EventBackend()
        self._camera = SimpleNamespace()
        self._transform_gizmo = _Gizmo()
        self._reverse_objects: dict[int, str] = {}
        self.camera_mode = "orbit"
        self._authoring_runtime = _AuthoringRuntime(self)

    def scenario_authoring_runtime(self):
        return self._authoring_runtime

    def pygfx_interaction_router(self):
        router = getattr(self, "_interaction_router", None)
        if router is None:
            router = PygfxInteractionRouter(self)
            self._interaction_router = router
        return router

    def _ensure_transform_gizmo(self):
        return self._transform_gizmo

    def initialize_authoring_viewport(self, host_parent, **_kwargs) -> None:
        self.host_parent = host_parent

    @contextmanager
    def batch_updates(self):
        self.batch_calls += 1
        yield

    def ensure_object(self, obj) -> bool:
        self.ensure_calls += 1
        self.objects[obj.id] = obj
        return True

    def remove_object(self, object_id: str) -> bool:
        self.removed.append(object_id)
        self.objects.pop(object_id, None)
        return True

    def request_redraw(self) -> None:
        self.redraw_calls += 1

    def focus_camera(self, _position) -> bool:
        return True

    def reset_camera_bounds(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _snapshot(document_id, actor_id, revision: int, x: float) -> OverlaySnapshot:
    return OverlaySnapshot(
        document_id=document_id,
        revision=revision,
        work_plane_visible=False,
        actors=(
            ActorOverlaySnapshot(
                actor_id=actor_id,
                role="rx",
                name="RX1",
                positions=((x, 0.0, 0.0), (x + 1.0, 0.0, 0.0)),
                mobility_controls=(
                    MobilityControl(MobilityControlKind.WAYPOINT, (x, 0.0, 0.0), 0),
                    MobilityControl(
                        MobilityControlKind.WAYPOINT,
                        (x + 0.25, 1.0, 0.0),
                        1,
                    ),
                    MobilityControl(
                        MobilityControlKind.WAYPOINT,
                        (x + 1.0, 0.0, 0.0),
                        2,
                    ),
                ),
            ),
        ),
    )


def test_reconcile_coalesces_drag_updates_and_maps_control_vertices(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()

    for revision in range(1, 101):
        port.reconcile(_snapshot(document_id, actor_id, revision, float(revision)))

    objects = port.renderer_objects()
    point_id = stable_renderer_id(document_id, actor_id, "mobility_handles")
    point_payload = objects[point_id].payload
    assert isinstance(point_payload, PointCloudPayload)
    np.testing.assert_allclose(
        point_payload.points,
        (
            (100.0, 0.0, 0.0),
            (100.25, 1.0, 0.0),
            (101.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
        ),
    )
    assert objects[point_id].metadata["control_vertex_count"] == 3
    assert objects[point_id].metadata["control_vertex_map"] == (
        ("waypoint", 0),
        ("waypoint", 1),
        ("waypoint", 2),
    )
    assert objects[point_id].metadata["current_vertex_index"] == 3
    assert port.renderer.ensure_calls == len(objects)
    assert port.renderer.batch_calls == 1
    assert port.renderer.redraw_calls == 1

    # An older update for the same document is ignored after the flush.
    port.reconcile(_snapshot(document_id, actor_id, 99, -1.0))
    assert port.renderer_objects()[point_id] is objects[point_id]
    assert port.renderer.ensure_calls == len(objects)
    port.close()
    host.close()


def test_lower_revision_new_document_replaces_objects_and_close_cleans_up(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    old_document = uuid4()
    new_document = uuid4()
    old_actor = uuid4()
    new_actor = uuid4()
    port.activate(lambda _value: None)
    port.reconcile(_snapshot(old_document, old_actor, 500, 1.0))
    old_ids = set(port.renderer_objects())

    port.reconcile(_snapshot(new_document, new_actor, 0, 2.0))
    new_ids = set(port.renderer_objects())

    assert old_ids.isdisjoint(new_ids)
    assert old_ids.issubset(set(port.renderer.removed))
    assert all(parse_renderer_id(value)[0] == new_document for value in new_ids)
    assert set(port.renderer.objects) == new_ids
    port.close()
    assert not port.active
    assert port.renderer.closed
    assert not port.renderer.objects
    assert new_ids.issubset(set(port.renderer.removed))
    host.close()


def test_typed_target_asset_reuses_payload_and_composes_actor_pose(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    payload = MeshPayload(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        ),
        triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    material = MaterialPayload(base_color=(0.2, 0.4, 0.8, 1.0))
    asset = TargetOverlayAsset(
        cache_key="car.ply:123:456:scale=1",
        payload=payload,
        material=material,
        local_to_actor=(
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    orientation = (
        (0.0, -1.0, 0.0, 999.0),
        (1.0, 0.0, 0.0, 999.0),
        (0.0, 0.0, 1.0, 999.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="target",
                    name="Car",
                    positions=((10.0, 20.0, 3.0),),
                    orientation_matrix=orientation,
                    target_asset=asset,
                ),
            ),
        )
    )

    target = port.renderer_objects()[stable_renderer_id(document_id, actor_id, "target")]
    assert target.payload is payload
    assert target.material_payload is material
    assert target.metadata["asset_cache_key"] == asset.cache_key
    np.testing.assert_allclose(target.transform.translation, (10.0, 22.0, 3.0))
    port.close()
    host.close()


def test_overlay_snapshot_rejects_duplicate_actor_ids() -> None:
    actor_id = uuid4()
    actor = ActorOverlaySnapshot(
        actor_id=actor_id,
        role="tx",
        name="TX1",
        positions=((0.0, 0.0, 0.0),),
    )
    with pytest.raises(ValueError, match="unique"):
        OverlaySnapshot(
            document_id=uuid4(),
            revision=0,
            actors=(actor, actor),
        )
