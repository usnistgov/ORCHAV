from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visualizer.src.renderers.pygfx.transform_gizmo import PygfxTransformGizmoMixin
from visualizer.src.services.object_identity import (
    ensure_target_entry_identity,
    make_node_geometry_name,
    make_target_entry_geometry_name,
)


class _FakeTransform:
    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self._matrix = np.asarray(matrix if matrix is not None else np.eye(4), dtype=float)
        self.position = self._matrix[:3, 3].copy()

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = np.asarray(value, dtype=float)
        self.position = self._matrix[:3, 3].copy()


class _FakeWorldObject:
    def __init__(self, matrix: np.ndarray | None = None) -> None:
        transform = _FakeTransform(matrix)
        self.local = transform
        self.world = transform
        self.visible = True


class _FakeScene:
    def __init__(self) -> None:
        self.objects = []

    def add(self, obj) -> None:
        self.objects.append(obj)

    def remove(self, obj) -> None:
        self.objects.remove(obj)


class _DummyTransformGizmoOwner(PygfxTransformGizmoMixin):
    def __init__(self) -> None:
        target_entry = {"target_name": "Walker", "_target_position": [10.0, 20.0, 30.0]}
        ensure_target_entry_identity(target_entry, 0)
        self.visualizer = SimpleNamespace(
            target_entries=[target_entry],
            current_target_positions=np.asarray([[40.0, 50.0, 60.0]], dtype=float),
            current_view_model=SimpleNamespace(
                target_positions=np.asarray([[70.0, 80.0, 90.0]], dtype=float)
            ),
        )
        self._transform_gizmo_target_name = None
        self._transform_gizmo_target_kind = None
        self._transform_gizmo_target_index = None
        self._transform_gizmo_control_object = None
        self._transform_gizmo_target_proxy = None
        self._transform_session_callback = None
        self._objects = {}
        self._positions = {}
        self._transforms = {}
        self._scene = _FakeScene()
        self._gfx = SimpleNamespace(WorldObject=_FakeWorldObject)
        self.redraw_count = 0

    def request_redraw(self) -> None:
        self.redraw_count += 1


def test_transform_gizmo_parses_node_marker_names() -> None:
    owner = _DummyTransformGizmoOwner()

    assert owner._parse_transform_target_name(make_node_geometry_name("tx", 2, "marker")) == (
        "tx",
        2,
    )


def test_transform_gizmo_parses_target_mesh_names() -> None:
    owner = _DummyTransformGizmoOwner()
    target_name = make_target_entry_geometry_name(owner.visualizer.target_entries[0], "mesh")

    assert owner._parse_transform_target_name(target_name) == ("target", 0)


def test_transform_gizmo_ignores_non_editable_target_components() -> None:
    owner = _DummyTransformGizmoOwner()
    label_name = make_target_entry_geometry_name(owner.visualizer.target_entries[0], "label")

    assert owner._parse_transform_target_name(label_name) is None


def test_transform_gizmo_uses_semantic_target_position_before_runtime_arrays() -> None:
    owner = _DummyTransformGizmoOwner()

    np.testing.assert_allclose(owner._target_semantic_position(0), [10.0, 20.0, 30.0])


def test_transform_gizmo_falls_back_to_current_target_positions() -> None:
    owner = _DummyTransformGizmoOwner()
    owner.visualizer.target_entries[0].pop("_target_position")
    owner.visualizer.target_entries[0].pop("position", None)

    np.testing.assert_allclose(owner._target_semantic_position(0), [40.0, 50.0, 60.0])


def test_transform_gizmo_reports_active_target() -> None:
    owner = _DummyTransformGizmoOwner()
    owner._transform_gizmo_target_name = "target:walker::mesh"
    owner._transform_gizmo_target_kind = "target"
    owner._transform_gizmo_target_index = 0

    assert owner.get_active_transform_target() == {
        "object_id": "target:walker::mesh",
        "kind": "target",
        "index": 0,
    }


def test_target_transform_proxy_uses_semantic_position_and_mesh_rotation() -> None:
    owner = _DummyTransformGizmoOwner()
    target_name = make_target_entry_geometry_name(owner.visualizer.target_entries[0], "mesh")
    mesh_transform = np.eye(4, dtype=float)
    mesh_transform[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    mesh_transform[:3, 3] = [1.0, 2.0, 3.0]

    proxy = owner._target_transform_proxy(target_name, 0, _FakeWorldObject(mesh_transform))

    assert proxy in owner._scene.objects
    np.testing.assert_allclose(proxy.local.matrix[:3, :3], mesh_transform[:3, :3])
    np.testing.assert_allclose(proxy.local.matrix[:3, 3], [10.0, 20.0, 30.0])


def test_transform_gizmo_emits_target_events_from_semantic_proxy() -> None:
    owner = _DummyTransformGizmoOwner()
    target_name = make_target_entry_geometry_name(owner.visualizer.target_entries[0], "mesh")
    events = []
    proxy_transform = np.eye(4, dtype=float)
    proxy_transform[:3, 3] = [12.0, 22.0, 32.0]
    owner._objects[target_name] = _FakeWorldObject()
    owner._transform_gizmo_control_object = _FakeWorldObject(proxy_transform)
    owner._transform_gizmo_target_name = target_name
    owner._transform_gizmo_target_kind = "target"
    owner._transform_gizmo_target_index = 0
    owner._transform_session_callback = events.append

    owner._emit_transform_gizmo_event("changed", force=True)

    assert events[0]["object_id"] == target_name
    assert events[0]["kind"] == "target"
    np.testing.assert_allclose(events[0]["position"], [12.0, 22.0, 32.0])
    assert target_name not in owner._transforms
    assert target_name not in owner._positions


def test_authoring_transform_proxy_preserves_compiled_entity_pose() -> None:
    owner = _DummyTransformGizmoOwner()
    name = "authoring:document:entity:target"
    transform = np.asarray(
        (
            (0.0, -1.0, 0.0, 10.0),
            (1.0, 0.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=float,
    )

    proxy = owner._authoring_transform_proxy(name, transform)

    assert proxy in owner._scene.objects
    assert proxy.visible is False
    np.testing.assert_allclose(proxy.local.matrix, transform)


def test_authoring_transform_attachment_failure_removes_proxy_and_selection() -> None:
    owner = _DummyTransformGizmoOwner()
    name = "authoring:document:entity:target"
    owner._objects[name] = _FakeWorldObject()

    def _fail_set_object(_obj) -> None:
        raise RuntimeError("attach failed")

    owner._transform_gizmo = SimpleNamespace(set_object=_fail_set_object, visible=False)

    assert (
        owner._select_transform_gizmo_target(
            name,
            "authoring",
            0,
            semantic_transform=np.eye(4),
        )
        is False
    )
    assert owner._transform_gizmo_target_name is None
    assert owner._transform_gizmo_control_object is None
    assert owner._transform_gizmo_target_proxy is None
    assert owner._scene.objects == []


def test_transform_gizmo_emits_authoring_events_without_mutating_mesh_cache() -> None:
    owner = _DummyTransformGizmoOwner()
    name = "authoring:document:entity:target"
    events = []
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (4.0, 5.0, 6.0)
    owner._objects[name] = _FakeWorldObject()
    owner._transform_gizmo_control_object = _FakeWorldObject(transform)
    owner._transform_gizmo_target_name = name
    owner._transform_gizmo_target_kind = "authoring"
    owner._transform_gizmo_target_index = 0
    owner._transform_session_callback = events.append

    owner._emit_transform_gizmo_event("changed", force=True)

    assert events[0]["kind"] == "authoring"
    np.testing.assert_allclose(events[0]["transform"], transform)
    assert name not in owner._transforms
    assert name not in owner._positions


def test_sync_active_transform_target_pose_updates_proxy_only() -> None:
    owner = _DummyTransformGizmoOwner()
    target_name = make_target_entry_geometry_name(owner.visualizer.target_entries[0], "mesh")
    proxy = _FakeWorldObject()
    owner._transform_gizmo_target_proxy = proxy
    owner._transform_gizmo_target_name = target_name
    owner._transform_gizmo_target_kind = "target"
    owner._transform_gizmo_target_index = 0
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = [4.0, 5.0, 6.0]

    assert owner.sync_active_transform_target_pose(target_name, transform) is True

    np.testing.assert_allclose(proxy.local.matrix[:3, 3], [4.0, 5.0, 6.0])
    assert owner.redraw_count == 1
