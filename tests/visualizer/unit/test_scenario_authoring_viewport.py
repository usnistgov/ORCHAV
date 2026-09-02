"""Pure interaction and renderer-port tests for the authoring viewport."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtWidgets import QWidget

from shared.scenarios.actors import CircularMobilitySpec, LinearMobilitySpec
from visualizer.src.authoring.domain import MobilityControl, MobilityControlKind
from visualizer.src.authoring.interaction import (
    InteractionSession,
    PygfxInteractionRouter,
    camera_ray,
    intersect_horizontal_plane,
    snap_position,
    surface_position_from_pick_info,
)
from visualizer.src.authoring.mobility_control_rig import mobility_control_rig
from visualizer.src.authoring.viewport import PygfxScenarioAuthoringViewportPort
from visualizer.src.authoring.viewport_port import (
    ActorOverlaySnapshot,
    AuthoringTool,
    OverlaySnapshot,
    OverlayVisibility,
    PreviewProvenance,
    SceneOverlayAsset,
    TargetOverlayAsset,
    TrajectoryDisplayMode,
    TransformPhase,
    parse_renderer_id,
    stable_renderer_id,
    stable_scene_renderer_id,
)
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
)


def test_stable_renderer_ids_round_trip() -> None:
    document_id = uuid4()
    actor_id = uuid4()
    value = stable_renderer_id(document_id, actor_id, "start_end")
    assert value == f"authoring:{document_id}:{actor_id}:start_end"
    assert parse_renderer_id(value) == (document_id, actor_id, "start_end")
    assert parse_renderer_id("scene:wall") is None


def test_surface_pick_reconstructs_barycentric_world_position() -> None:
    positions = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    transform = np.eye(4)
    transform[:3, 3] = (10.0, 20.0, 3.0)
    obj = SimpleNamespace(
        geometry=SimpleNamespace(
            positions=SimpleNamespace(data=positions),
            indices=SimpleNamespace(data=np.asarray(((0, 1, 2),))),
        ),
        world=SimpleNamespace(matrix=transform),
    )
    result = surface_position_from_pick_info(
        {
            "world_object": obj,
            "face_index": 0,
            "face_coord": (0.25, 0.25, 0.5),
        }
    )
    np.testing.assert_allclose(result, (10.5, 21.0, 3.0))


def test_empty_space_camera_ray_hits_and_snaps_work_plane() -> None:
    camera = SimpleNamespace(
        projection_matrix_inverse=np.eye(4),
        view_matrix=np.eye(4),
    )
    ray = camera_ray((50.0, 50.0), (100.0, 100.0), camera)
    assert ray is not None
    hit = intersect_horizontal_plane(ray[0], ray[1], 0.75)
    assert hit is not None
    assert snap_position((0.26, -0.24, 0.74), 0.5) == (0.5, 0.0, 0.5)


class _EventBackend:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_event_handler(self, handler, *events):
        self.added.append((handler, events))

    def remove_event_handler(self, handler, *events):
        self.removed.append((handler, events))


class _FakeRuntime:
    def __init__(self, backend):
        self.backend = backend
        self.logical_size = (100.0, 100.0)

    def add_event_handler(self, handler, *events):
        self.backend.add_event_handler(handler, *events)

    def remove_event_handler(self, handler, *events):
        self.backend.remove_event_handler(handler, *events)

    def ensure_gizmo(self, *, authoring=True):
        return None

    def hide_gizmo(self):
        return None

    def set_camera_mode(self, mode):
        self.camera_mode = mode
        return True


class _TransformRuntime(_FakeRuntime):
    def __init__(self, backend, pose):
        super().__init__(backend)
        self.pose = np.asarray(pose, dtype=float)
        self.callback = None
        self.sync_calls = []

    def attach_gizmo(self, object_id, callback):
        self.callback = callback
        callback(
            {
                "phase": "selected",
                "object_id": object_id,
                "transform": self.pose,
            }
        )
        return True

    def sync_gizmo_pose(self, object_id, transform):
        self.sync_calls.append((object_id, np.asarray(transform, dtype=float).copy()))
        return True


def test_interaction_router_registration_is_constant_and_sessions_exclusive() -> None:
    backend = _EventBackend()
    runtime = _FakeRuntime(backend)
    renderer = SimpleNamespace(
        scenario_authoring_runtime=lambda: runtime,
    )
    router = PygfxInteractionRouter(renderer)

    def sink(_event):
        return None

    router.activate(InteractionSession.AUTHORING, sink)
    router.activate(InteractionSession.AUTHORING, sink)
    assert len(backend.added) == 1
    assert router.handler_count == len(backend.added[0][1])

    router.activate(InteractionSession.LIVE_PREVIEW, sink)
    assert len(backend.removed) == 0
    assert len(backend.added) == 1
    assert router.session is InteractionSession.LIVE_PREVIEW
    router.deactivate()
    assert router.handler_count == 0


def test_interaction_router_does_not_resync_proxy_during_active_rotation() -> None:
    backend = _EventBackend()
    initial = np.eye(4)
    initial[:3, 3] = (10.0, 20.0, 3.0)
    runtime = _TransformRuntime(backend, initial)
    renderer = SimpleNamespace(scenario_authoring_runtime=lambda: runtime)
    router = PygfxInteractionRouter(renderer)
    events = []
    router.activate(InteractionSession.AUTHORING, events.append)
    router.set_tool(AuthoringTool.MOVE)
    object_id = stable_renderer_id(uuid4(), uuid4(), "mobility_handles")
    assert router.attach_transform_gizmo(object_id)
    callback = runtime.callback
    assert callback is not None
    angle = np.radians(45.0)
    changed = initial.copy()
    changed[:2, :2] = (
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle), np.cos(angle)),
    )

    callback({"phase": "changed", "object_id": object_id, "transform": changed})
    assert [event.phase for event in events] == [
        TransformPhase.BEGIN,
        TransformPhase.UPDATE,
    ]

    reconciled = np.eye(4)
    assert router.sync_transform_pose(object_id, reconciled)
    assert runtime.sync_calls == []

    callback({"phase": "committed", "object_id": object_id, "transform": changed})
    assert events[-1].phase is TransformPhase.COMMIT
    np.testing.assert_allclose(events[-1].matrix, changed)
    assert router.sync_transform_pose(object_id, changed)
    assert len(runtime.sync_calls) == 1


class _FakeRenderer:
    capabilities = RendererCapabilities(scenario_authoring=True)

    def __init__(self, _visualizer):
        self.objects = {}
        self.removed = []
        self._renderer = _EventBackend()
        self._camera = SimpleNamespace()
        self._transform_gizmo = object()
        self._authoring_runtime = _FakeRuntime(self._renderer)

    def scenario_authoring_runtime(self):
        return self._authoring_runtime

    def pygfx_interaction_router(self):
        router = getattr(self, "_interaction_router", None)
        if router is None:
            router = PygfxInteractionRouter(self)
            self._interaction_router = router
        return router

    def initialize_authoring_viewport(self, host_parent, **_kwargs):
        self.host_parent = host_parent

    def batch_updates(self):
        from contextlib import nullcontext

        return nullcontext()

    def ensure_object(self, obj):
        self.objects[obj.id] = obj
        return True

    def remove_object(self, object_id):
        self.removed.append(object_id)
        self.objects.pop(object_id, None)
        return True

    def request_redraw(self):
        return None

    def focus_camera(self, _position):
        return True

    def reset_camera_bounds(self):
        return None

    def close(self):
        return None


def test_overlay_reconciliation_keeps_one_handles_and_path_object_per_actor(
    qapp,
) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    snapshot = OverlaySnapshot(
        document_id=document_id,
        revision=1,
        actors=(
            ActorOverlaySnapshot(
                actor_id=actor_id,
                role="rx",
                name="RX1",
                positions=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                selected=True,
            ),
        ),
    )
    port.reconcile(snapshot)
    assert port.current_snapshot() is snapshot
    object_ids = set(port.renderer_objects())
    assert stable_renderer_id(document_id, actor_id, "mobility_handles") in object_ids
    assert stable_renderer_id(document_id, actor_id, "path") in object_ids
    assert sum(value.endswith(":mobility_handles") for value in object_ids) == 1
    assert sum(value.endswith(":path") for value in object_ids) == 1

    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=2,
            work_plane_visible=False,
        )
    )
    assert not port.renderer_objects()
    assert object_ids.issubset(set(port.renderer.removed))
    port.close()
    host.close()


def test_authored_control_path_is_distinct_from_exact_sampled_trajectory(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    sampled = (
        (0.0, 0.0, 1.0),
        (0.5, 1.0, 1.0),
        (2.0, 1.5, 1.0),
        (4.0, 0.0, 1.0),
    )
    authored = ((0.0, 0.0, 1.0), (4.0, 0.0, 1.0))
    controls = (
        MobilityControl(MobilityControlKind.START, authored[0]),
        MobilityControl(MobilityControlKind.END, authored[1]),
    )
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="rx",
                    name="RX1",
                    positions=sampled,
                    mobility_controls=controls,
                    authored_path=authored,
                    selected=True,
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    sampled_id = stable_renderer_id(document_id, actor_id, "path")
    authored_id = stable_renderer_id(document_id, actor_id, "authored_path")
    handles_id = stable_renderer_id(document_id, actor_id, "mobility_handles")
    assert sampled_id != authored_id
    assert parse_renderer_id(authored_id) == (document_id, actor_id, "authored_path")
    assert {sampled_id, authored_id, handles_id}.issubset(objects)

    sampled_object = objects[sampled_id]
    authored_object = objects[authored_id]
    np.testing.assert_allclose(sampled_object.payload.points[: len(sampled)], sampled)
    np.testing.assert_array_equal(
        sampled_object.payload.lines[: len(sampled) - 1],
        ((0, 1), (1, 2), (2, 3)),
    )
    np.testing.assert_allclose(authored_object.payload.points, authored)
    np.testing.assert_array_equal(authored_object.payload.lines, ((0, 1),))
    assert authored_object.metadata["component"] == "authored_path"
    assert authored_object.metadata["control_point_count"] == 2
    assert authored_object.metadata["pickable"] is False
    assert authored_object.metadata["interaction_role"] == "decorative"
    assert authored_object.material_payload.line_width == 1.5
    assert sampled_object.material_payload.line_width == 3.0
    assert authored_object.material_payload.base_color != sampled_object.material_payload.base_color
    port.close()
    host.close()


@pytest.mark.parametrize("authored_path", ((), ((2.0, 3.0, 4.0),)))
def test_empty_or_stationary_authored_geometry_does_not_create_a_line(
    qapp,
    authored_path,
) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="tx",
                    name="TX1",
                    positions=((2.0, 3.0, 4.0),),
                    mobility_controls=(
                        MobilityControl(
                            MobilityControlKind.POSITION,
                            (2.0, 3.0, 4.0),
                        ),
                    ),
                    authored_path=authored_path,
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    assert stable_renderer_id(document_id, actor_id, "authored_path") not in objects
    assert stable_renderer_id(document_id, actor_id, "mobility_handles") in objects
    assert stable_renderer_id(document_id, actor_id, "path") in objects
    port.close()
    host.close()


def test_circular_authored_radius_visibility_and_removal_do_not_change_samples(
    qapp,
) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    center = (1.0, 2.0, 3.0)
    rim = (1.0, 6.0, 3.0)
    sampled = (
        rim,
        (5.0, 2.0, 3.0),
        (1.0, -2.0, 3.0),
        (-3.0, 2.0, 3.0),
    )
    controls = (
        MobilityControl(MobilityControlKind.CENTER, center),
        MobilityControl(MobilityControlKind.RIM, rim),
    )

    def snapshot(revision, *, visible=True, authored_path=(center, rim)):
        return OverlaySnapshot(
            document_id=document_id,
            revision=revision,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="target",
                    name="Target1",
                    positions=sampled,
                    mobility_controls=controls,
                    authored_path=authored_path,
                    visible=visible,
                ),
            ),
        )

    authored_id = stable_renderer_id(document_id, actor_id, "authored_path")
    sampled_id = stable_renderer_id(document_id, actor_id, "path")
    port.reconcile(snapshot(1))
    objects = port.renderer_objects()
    np.testing.assert_allclose(objects[authored_id].payload.points, (center, rim))
    np.testing.assert_array_equal(objects[authored_id].payload.lines, ((0, 1),))
    np.testing.assert_allclose(objects[sampled_id].payload.points[: len(sampled)], sampled)
    assert objects[authored_id].visible is True

    port.reconcile(snapshot(2, visible=False))
    assert port.renderer_objects()[authored_id].visible is False

    port.reconcile(snapshot(3, authored_path=()))
    assert authored_id not in port.renderer_objects()
    assert authored_id in port.renderer.removed
    np.testing.assert_allclose(
        port.renderer_objects()[sampled_id].payload.points[: len(sampled)],
        sampled,
    )
    port.close()
    host.close()


def _triangle_payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        ),
        triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
    )


def test_scene_assets_and_placement_ghost_reconcile_with_stable_ids(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    payload = _triangle_payload()
    material = MaterialPayload(base_color=(0.2, 0.3, 0.4, 1.0))
    asset = SceneOverlayAsset(
        cache_key="builtin/munich:wall-17",
        name="wall-17",
        payload=payload,
        material=material,
    )
    snapshot = OverlaySnapshot(
        document_id=document_id,
        revision=1,
        scene_assets=(asset,),
        work_plane_visible=False,
        placement_ghost=(2.0, 3.0, 4.0),
        placement_guide_start=(1.0, 1.0, 1.0),
    )

    port.reconcile(snapshot)

    objects = port.renderer_objects()
    scene_id = stable_scene_renderer_id(document_id, asset.cache_key)
    scene = objects[scene_id]
    assert scene.payload is payload
    assert scene.material_payload is material
    assert scene.metadata["component"] == "scene_surface"
    assert scene.metadata["surface_pickable"] is True
    assert parse_renderer_id(scene_id) is None
    ghost_id = f"authoring:{document_id}:placement_ghost"
    ghost = objects[ghost_id]
    assert isinstance(ghost.payload, PointCloudPayload)
    np.testing.assert_allclose(ghost.payload.points, ((2.0, 3.0, 4.0),))
    assert ghost.metadata["pickable"] is False
    assert ghost.metadata["depth_write"] is False
    assert ghost.metadata["depth_compare"] == "<="
    assert ghost.metadata["render_order"] == 10
    guide_id = f"authoring:{document_id}:placement_guide"
    guide = objects[guide_id]
    np.testing.assert_allclose(
        guide.payload.points,
        ((1.0, 1.0, 1.0), (2.0, 3.0, 4.0)),
    )
    assert guide.metadata["pickable"] is False
    assert guide.metadata["depth_write"] is False
    assert guide.metadata["depth_compare"] == "<="
    assert guide.metadata["render_order"] == 10

    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=2,
            work_plane_visible=False,
        )
    )
    assert not port.renderer_objects()
    assert {scene_id, ghost_id, guide_id}.issubset(set(port.renderer.removed))
    port.close()
    host.close()


def test_current_pose_drives_marker_label_orientation_target_and_look_at(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    orientation = (
        (0.0, -1.0, 0.0, 999.0),
        (1.0, 0.0, 0.0, 999.0),
        (0.0, 0.0, 1.0, 999.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    target_asset = TargetOverlayAsset(
        cache_key="catalog/car/mesh-0",
        payload=_triangle_payload(),
        material=MaterialPayload(base_color=(0.6, 0.2, 0.1, 1.0)),
        local_to_actor=(
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    current = (10.0, 20.0, 3.0)
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="target",
                    name="Car",
                    positions=((0.0, 0.0, 0.0), (5.0, 5.0, 0.0), (20.0, 0.0, 0.0)),
                    current_position=current,
                    look_at_position=(11.0, 24.0, 5.0),
                    selected=True,
                    orientation_matrix=orientation,
                    target_asset=target_asset,
                ),
            ),
        )
    )

    objects = port.renderer_objects()

    def get(component):
        return objects[stable_renderer_id(document_id, actor_id, component)]

    marker = get("mobility_handles")
    assert isinstance(marker.payload, PointCloudPayload)
    np.testing.assert_allclose(marker.payload.points, (current,))
    assert marker.metadata["depth_write"] is False
    assert marker.metadata["depth_compare"] == "<="
    assert marker.metadata["render_order"] == 50
    np.testing.assert_allclose(get("label").transform.translation, current)
    np.testing.assert_allclose(get("status").payload.points, (current,))
    assert get("status").metadata["pickable"] is False
    np.testing.assert_allclose(get("orientation").transform.translation, current)
    target = get("target")
    assert target.metadata["depth_write"] is True
    assert target.metadata["depth_compare"] == "<"
    assert target.metadata["render_order"] == 0
    np.testing.assert_allclose(target.transform.translation, (10.0, 22.0, 3.0))
    np.testing.assert_allclose(
        np.asarray(target.metadata["authoring_actor_pose"]),
        (
            (0.0, -1.0, 0.0, 10.0),
            (1.0, 0.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )

    start_end = get("start_end")
    assert start_end.metadata["pickable"] is False
    assert start_end.metadata["interaction_role"] == "decorative"
    np.testing.assert_allclose(
        start_end.payload.points,
        ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
    )
    path = get("path")
    assert isinstance(path.payload, LineSetPayload)
    np.testing.assert_allclose(
        path.payload.points[:3],
        ((0.0, 0.0, 0.0), (5.0, 5.0, 0.0), (20.0, 0.0, 0.0)),
    )
    assert len(path.payload.lines) == 4
    assert path.metadata["path_direction"] is True
    look_at = get("look_at")
    np.testing.assert_allclose(look_at.payload.points, (current, (11.0, 24.0, 5.0)))
    port.close()
    host.close()


@pytest.mark.parametrize(
    ("visibility", "expected_names"),
    (
        (OverlayVisibility.OFF, frozenset()),
        (OverlayVisibility.SELECTED, frozenset({"Selected"})),
        (OverlayVisibility.ALL, frozenset({"Selected", "Unselected"})),
    ),
)
def test_optional_actor_overlays_obey_off_selected_all(
    qapp,
    visibility: OverlayVisibility,
    expected_names: frozenset[str],
) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    orientation = tuple(tuple(float(value) for value in row) for row in np.eye(4))
    actors = tuple(
        ActorOverlaySnapshot(
            actor_id=uuid4(),
            role="tx",
            name=name,
            positions=((index * 2.0, 0.0, 0.0),),
            look_at_position=(index * 2.0 + 1.0, 0.0, 0.0),
            selected=selected,
            orientation_matrix=orientation,
        )
        for index, (name, selected) in enumerate((("Selected", True), ("Unselected", False)))
    )
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=actors,
            orientation_axes_visibility=visibility,
            look_at_visibility=visibility,
        )
    )

    objects = port.renderer_objects()
    orientation_names = {
        actor.name
        for actor in actors
        if stable_renderer_id(document_id, actor.actor_id, "orientation") in objects
    }
    look_at_names = {
        actor.name
        for actor in actors
        if stable_renderer_id(document_id, actor.actor_id, "look_at") in objects
    }
    assert orientation_names == expected_names
    assert look_at_names == expected_names
    port.close()
    host.close()


def test_axes_off_keeps_semantic_target_pose(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    orientation = (
        (0.0, -1.0, 0.0, 999.0),
        (1.0, 0.0, 0.0, 999.0),
        (0.0, 0.0, 1.0, 999.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    target_asset = TargetOverlayAsset(
        cache_key="catalog/car/mesh-0",
        payload=_triangle_payload(),
        material=MaterialPayload(),
        local_to_actor=(
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            orientation_axes_visibility=OverlayVisibility.OFF,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="target",
                    name="Car",
                    positions=((10.0, 20.0, 3.0),),
                    orientation_matrix=orientation,
                    target_asset=target_asset,
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    assert stable_renderer_id(document_id, actor_id, "orientation") not in objects
    target = objects[stable_renderer_id(document_id, actor_id, "target")]
    np.testing.assert_allclose(target.transform.translation, (10.0, 22.0, 3.0))
    np.testing.assert_allclose(
        np.asarray(target.metadata["authoring_actor_pose"]),
        (
            (0.0, -1.0, 0.0, 10.0),
            (1.0, 0.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    port.close()
    host.close()


def test_generic_mobility_handle_vertices_have_stable_semantic_mapping(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    controls = (
        MobilityControl(MobilityControlKind.START, (1.0, 2.0, 3.0)),
        MobilityControl(MobilityControlKind.END, (4.0, 5.0, 6.0)),
    )
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="rx",
                    name="RX1",
                    positions=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
                    current_position=(2.0, 3.0, 4.0),
                    mobility_controls=controls,
                ),
            ),
        )
    )

    handle = port.renderer_objects()[stable_renderer_id(document_id, actor_id, "mobility_handles")]
    np.testing.assert_allclose(
        handle.payload.points,
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (2.0, 3.0, 4.0)),
    )
    assert handle.metadata["control_vertex_map"] == (("start", None), ("end", None))
    assert handle.metadata["control_vertex_count"] == 2
    assert handle.metadata["current_vertex_index"] == 2
    port.close()
    host.close()


def test_declarative_linear_rig_uses_separate_top_priority_semantic_objects(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    mobility = LinearMobilitySpec(start_m=(1, 2, 3), end_m=(4, 5, 6))
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="rx",
                    name="RX1",
                    positions=(mobility.start_m, mobility.end_m),
                    current_position=(2, 3, 4),
                    mobility_control_rig=mobility_control_rig(mobility),
                    # A declarative rig replaces the older duplicate control polygon.
                    authored_path=(mobility.start_m, mobility.end_m),
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    anchor = objects[stable_renderer_id(document_id, actor_id, "mobility_handles")]
    np.testing.assert_allclose(anchor.payload.points, ((2, 3, 4),))
    assert anchor.metadata["interaction_role"] == "actor_pose"
    assert anchor.metadata["interaction_priority"] == 50
    assert stable_renderer_id(document_id, actor_id, "authored_path") not in objects

    for key, label, operation in (
        ("start", "Start point", "set_start"),
        ("end", "Arrival point", "set_end"),
    ):
        control = objects[stable_renderer_id(document_id, actor_id, f"mobility_control_{key}")]
        assert control.metadata["control_key"] == key
        assert control.metadata["control_label"] == label
        assert control.metadata["control_operation"] == operation
        assert control.metadata["control_constraint"] == "free"
        assert control.metadata["pickable"] is True
        assert control.metadata["depth_write"] is False
        assert control.metadata["depth_compare"] == "<="
        assert control.metadata["render_order"] == 100
        assert control.metadata["interaction_role"] == "mobility_control"
        assert control.metadata["interaction_priority"] == 100
        control_label = objects[
            stable_renderer_id(document_id, actor_id, f"mobility_control_label_{key}")
        ]
        assert control_label.payload.text == label
        assert control_label.metadata["pickable"] is False

    guide = objects[stable_renderer_id(document_id, actor_id, "mobility_guide_segment")]
    assert guide.metadata["guide_key"] == "segment"
    assert guide.metadata["pickable"] is False
    assert guide.metadata["interaction_role"] == "decorative"
    np.testing.assert_allclose(guide.payload.points, (mobility.start_m, mobility.end_m))
    port.close()
    host.close()


def test_trajectory_uses_a_broad_pick_proxy_below_semantic_handles(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    mobility = LinearMobilitySpec(start_m=(0, 0, 1), end_m=(10, 0, 1))
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="rx",
                    name="RX1",
                    positions=(mobility.start_m, mobility.end_m),
                    mobility_control_rig=mobility_control_rig(mobility),
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    path = objects[stable_renderer_id(document_id, actor_id, "path")]
    hit_proxy = objects[stable_renderer_id(document_id, actor_id, "trajectory_hit")]
    end = objects[stable_renderer_id(document_id, actor_id, "mobility_control_end")]
    assert path.metadata["pickable"] is False
    assert path.metadata["depth_write"] is False
    assert path.metadata["depth_compare"] == "<="
    assert path.metadata["render_order"] == 10
    assert path.material_payload.line_width == pytest.approx(3.0)
    assert hit_proxy.metadata["pickable"] is True
    assert hit_proxy.metadata["depth_write"] is False
    assert hit_proxy.metadata["depth_compare"] == "<="
    assert hit_proxy.metadata["render_order"] == 20
    assert hit_proxy.metadata["interaction_priority"] < end.metadata["interaction_priority"]
    assert hit_proxy.material_payload.line_width == pytest.approx(16.0)
    assert hit_proxy.material_payload.base_color[3] < 0.01
    port.close()
    host.close()


def test_trajectory_frame_control_layers_and_closed_marker_are_independent(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    positions = ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0))
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="tx",
                    name="TX1",
                    positions=positions,
                    frame_samples=positions[::2],
                    trajectory_visible=False,
                    frame_samples_visible=True,
                    closed_trajectory=True,
                    preview_provenance=PreviewProvenance.AUTHORED_DRAFT,
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    path = objects[stable_renderer_id(document_id, actor_id, "path")]
    proxy = objects[stable_renderer_id(document_id, actor_id, "trajectory_hit")]
    samples = objects[stable_renderer_id(document_id, actor_id, "frame_samples")]
    start = objects[stable_renderer_id(document_id, actor_id, "start_end")]
    assert path.visible is False
    assert proxy.visible is False
    assert samples.visible is True
    assert samples.metadata["pickable"] is False
    assert path.metadata["preview_provenance"] == "authored_draft"
    assert len(start.payload.points) == 1
    port.close()
    host.close()


def test_circular_rig_exposes_distinct_radius_and_start_angle_objects(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    mobility = CircularMobilitySpec(
        center_m=(1, 2, 3),
        radius_m=4.0,
        start_angle_deg=0.0,
        clockwise=False,
    )
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="target",
                    name="Target1",
                    positions=((5, 2, 3), (1, 6, 3), (-3, 2, 3)),
                    mobility_control_rig=mobility_control_rig(mobility),
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    radius = objects[stable_renderer_id(document_id, actor_id, "mobility_control_radius")]
    start_angle = objects[stable_renderer_id(document_id, actor_id, "mobility_control_start_angle")]
    assert radius.metadata["control_operation"] == "set_radius"
    assert radius.metadata["control_constraint"] == "radial"
    assert start_angle.metadata["control_operation"] == "set_start_angle"
    assert start_angle.metadata["control_constraint"] == "angular"
    np.testing.assert_allclose(radius.payload.points, ((1, 6, 3),))
    np.testing.assert_allclose(start_angle.payload.points, ((5, 2, 3),))
    assert not np.allclose(radius.payload.points, start_angle.payload.points)
    port.close()
    host.close()


def test_overlay_snapshot_rejects_duplicate_scene_keys_and_nonfinite_pose() -> None:
    asset = SceneOverlayAsset(
        cache_key="same",
        name="mesh",
        payload=_triangle_payload(),
        material=MaterialPayload(),
    )
    with np.testing.assert_raises_regex(ValueError, "scene cache keys must be unique"):
        OverlaySnapshot(
            document_id=uuid4(),
            revision=0,
            scene_assets=(asset, asset),
        )
    with np.testing.assert_raises_regex(ValueError, "current_position"):
        ActorOverlaySnapshot(
            actor_id=uuid4(),
            role="tx",
            name="TX1",
            positions=((0.0, 0.0, 0.0),),
            current_position=(float("nan"), 0.0, 0.0),
        )
    with np.testing.assert_raises_regex(ValueError, "authored control path"):
        ActorOverlaySnapshot(
            actor_id=uuid4(),
            role="rx",
            name="RX1",
            positions=((0.0, 0.0, 0.0),),
            authored_path=((float("nan"), 0.0, 0.0),),
        )
    with np.testing.assert_raises_regex(TypeError, "must be a MobilityControlRig"):
        ActorOverlaySnapshot(
            actor_id=uuid4(),
            role="rx",
            name="RX1",
            positions=((0.0, 0.0, 0.0),),
            mobility_control_rig=object(),  # type: ignore[arg-type]
        )


def test_random_sampling_renders_observations_for_applied_and_pending_candidates(
    qapp,
) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    points = ((0.0, 0.0, 1.0), (4.0, 2.0, 1.0), (-2.0, 5.0, 1.0))
    pending = ((1.0, 1.0, 1.0), (8.0, -1.0, 1.0), (0.0, 6.0, 1.0))
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="rx",
                    name="Random observations",
                    positions=points,
                    trajectory_display=TrajectoryDisplayMode.OBSERVATIONS,
                    trajectory_geometry_key="applied-random",
                    pending_positions=pending,
                    pending_trajectory_display=TrajectoryDisplayMode.OBSERVATIONS,
                    pending_geometry_key="candidate-random",
                    mobility_draft_pending=True,
                    selected=True,
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    components = {obj.metadata["component"] for obj in objects.values()}
    assert "observations" in components
    assert "pending_observations" in components
    assert "trajectory_hit" in components
    assert "path" not in components
    assert "pending_path" not in components
    assert "start_end" not in components
    assert isinstance(
        objects[stable_renderer_id(document_id, actor_id, "observations")].payload,
        PointCloudPayload,
    )
    assert isinstance(
        objects[stable_renderer_id(document_id, actor_id, "pending_observations")].payload,
        PointCloudPayload,
    )


def test_hover_and_timeline_pose_reuse_unchanged_geometry_payload_identity(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    positions = tuple((float(index), 0.0, 1.0) for index in range(20))

    def snapshot(*, current_index: int, hovered: bool) -> OverlaySnapshot:
        return OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="tx",
                    name="Cached path",
                    positions=positions,
                    frame_samples=positions,
                    current_position=positions[current_index],
                    trajectory_geometry_key="stable-path",
                    frame_samples_geometry_key="stable-frames",
                    trajectory_hovered=hovered,
                ),
            ),
        )

    port.reconcile(snapshot(current_index=0, hovered=False))
    first = port.renderer_objects()
    path_id = stable_renderer_id(document_id, actor_id, "path")
    frames_id = stable_renderer_id(document_id, actor_id, "frame_samples")
    first_path_payload = first[path_id].payload
    first_frames_payload = first[frames_id].payload
    first_line_width = first[path_id].material_payload.line_width

    port.reconcile(snapshot(current_index=9, hovered=True))
    second = port.renderer_objects()
    assert second[path_id].payload is first_path_payload
    assert second[frames_id].payload is first_frames_payload
    assert second[path_id].material_payload.line_width > first_line_width


def test_gizmo_attachment_waits_for_coalesced_snapshot_flush(qapp, monkeypatch) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    snapshot = OverlaySnapshot(
        document_id=document_id,
        revision=1,
        work_plane_visible=False,
        actors=(
            ActorOverlaySnapshot(
                actor_id=actor_id,
                role="tx",
                name="TX",
                positions=((0.0, 0.0, 1.0),),
            ),
        ),
    )
    attachments = []
    monkeypatch.setattr(
        port.router,
        "attach_transform_gizmo",
        lambda object_id: attachments.append(object_id) or True,
    )

    port.reconcile(snapshot)
    assert port.show_transform_gizmo(actor_id)
    assert port._pending_snapshot is snapshot
    assert attachments == []

    port.renderer_objects()
    assert attachments == [stable_renderer_id(document_id, actor_id, "mobility_handles")]
    assert port._pending_snapshot is None
    port.close()
    host.close()


def test_selected_path_and_group_context_have_non_pickable_shape_overlays(qapp) -> None:
    host = QWidget()
    port = PygfxScenarioAuthoringViewportPort(
        host,
        SimpleNamespace(),
        renderer_factory=_FakeRenderer,
    )
    document_id = uuid4()
    actor_id = uuid4()
    frame = (
        (1.0, 0.0, 0.0, 3.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    port.reconcile(
        OverlaySnapshot(
            document_id=document_id,
            revision=1,
            work_plane_visible=False,
            actors=(
                ActorOverlaySnapshot(
                    actor_id=actor_id,
                    role="rx",
                    name="G01 member",
                    positions=tuple((float(index), 0.0, 1.0) for index in range(24)),
                    trajectory_geometry_key="selected-long-path",
                    current_position=(7.0, 0.0, 1.0),
                    selected=True,
                    group_origin_position=(3.0, 2.0, 1.0),
                    group_frame_matrix=frame,
                ),
            ),
        )
    )

    objects = port.renderer_objects()
    halo = objects[stable_renderer_id(document_id, actor_id, "selection_path_halo")]
    tether = objects[stable_renderer_id(document_id, actor_id, "group_tether")]
    group_frame = objects[stable_renderer_id(document_id, actor_id, "group_frame")]
    path = objects[stable_renderer_id(document_id, actor_id, "path")]
    assert halo.metadata["pickable"] is False
    assert tether.metadata["pickable"] is False
    assert group_frame.metadata["pickable"] is False
    # Twenty-three route segments plus two sparse chevrons (four lines).
    assert len(path.payload.lines) == 27
