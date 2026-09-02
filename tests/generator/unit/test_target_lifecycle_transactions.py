from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from generator.core.exceptions import ComputationError
from generator.core.propagation.actor_state_application import apply_target_state_to_scene
from generator.core.services.scene_service import SceneService
from generator.core.target import manager as manager_module
from generator.core.target.manager import TargetManager
from shared.scenarios.actors import AlignMotionOrientationSpec


class _FakeMaterial:
    name = "fake-material"


class _FakeTargetObject:
    def __init__(
        self,
        *,
        fname: str,
        name: str,
        radio_material: Any,
        fail_position: str | None = None,
        fail_orientation: str | None = None,
        on_property_write: Callable[[], None] | None = None,
    ) -> None:
        self.fname = fname
        self.name = name
        self.radio_material = radio_material
        self._position: Any = (0.0, 0.0, 0.0)
        self._scaling: Any = (1.0, 1.0, 1.0)
        self._orientation: Any = (0.0, 0.0, 0.0)
        self.velocity: Any = (0.0, 0.0, 0.0)
        self.fail_position = fail_position
        self.fail_orientation = fail_orientation
        self.on_property_write = on_property_write

    @property
    def position(self) -> Any:
        return self._position

    @position.setter
    def position(self, value: Any) -> None:
        if self.on_property_write is not None:
            self.on_property_write()
        if self.fail_position is not None:
            raise RuntimeError(self.fail_position)
        self._position = value

    @property
    def scaling(self) -> Any:
        return self._scaling

    @scaling.setter
    def scaling(self, value: Any) -> None:
        if self.on_property_write is not None:
            self.on_property_write()
        self._scaling = value

    @property
    def orientation(self) -> Any:
        return self._orientation

    @orientation.setter
    def orientation(self, value: Any) -> None:
        if self.on_property_write is not None:
            self.on_property_write()
        if self.fail_orientation is not None:
            raise RuntimeError(self.fail_orientation)
        self._orientation = value


class _StatefulScene:
    def __init__(
        self,
        active: _FakeTargetObject | None = None,
        *,
        fail_on_edit: int | None = None,
        fail_after_remove: int | None = None,
        fail_after_edit: int | None = None,
    ) -> None:
        self.objects: dict[str, _FakeTargetObject] = {}
        if active is not None:
            self.objects[active.name] = active
        self.edits: list[dict[str, tuple[_FakeTargetObject, ...]]] = []
        self.fail_on_edit = fail_on_edit
        self.fail_after_remove = fail_after_remove
        self.fail_after_edit = fail_after_edit

    def edit(self, *, add=None, remove=None) -> None:
        additions = tuple(add or ())
        removals = tuple(remove or ())
        self.edits.append({"add": additions, "remove": removals})
        if self.fail_on_edit == len(self.edits):
            raise RuntimeError("rollback edit rejected")
        for obj in removals:
            if self.objects.get(obj.name) is obj:
                del self.objects[obj.name]
        if self.fail_after_remove == len(self.edits):
            raise RuntimeError("scene edit failed after removal")
        for obj in additions:
            self.objects[obj.name] = obj
        if self.fail_after_edit == len(self.edits):
            raise RuntimeError("scene edit failed after mutation")

    def add(self, obj: _FakeTargetObject) -> None:
        self.objects[obj.name] = obj

    def remove(self, name: str) -> None:
        self.objects.pop(name, None)

    def get(self, name: str) -> _FakeTargetObject | None:
        return self.objects.get(name)


def _write_meshes(tmp_path: Path, count: int = 2) -> list[str]:
    meshes = []
    for index in range(count):
        mesh = tmp_path / f"target_{index:02d}.obj"
        mesh.write_text("# fake mesh\n", encoding="utf-8")
        meshes.append(str(mesh))
    return meshes


def _target_config() -> SimpleNamespace:
    return SimpleNamespace(
        name="Target",
        initial_position=(1.0, 2.0, 3.0),
        mesh_start_index=0,
        mesh_frame_stride=1,
        mobility=object(),
        orientation=(0.0, 0.0, 0.0),
        scale=1.0,
        switch_meshes=True,
        use_ply_position=False,
    )


def _manager(
    scene: _StatefulScene,
    meshes: list[str],
    *,
    active: _FakeTargetObject | None,
    mesh_call_count: int = 1,
) -> TargetManager:
    manager = TargetManager.__new__(TargetManager)
    manager.config = _target_config()
    manager.scene = scene
    manager.meshes = meshes
    manager.current_mesh_idx = 0
    manager._mesh_call_count = mesh_call_count
    manager.target_object = active
    manager.target_material = _FakeMaterial()
    manager.material_type = "metal"
    manager.material_overrides = {}
    return manager


def _install_scene_object_factory(
    monkeypatch,
    *,
    fail_position: str | None = None,
    fail_orientation: str | None = None,
    on_property_write: Callable[[], None] | None = None,
) -> list[_FakeTargetObject]:
    created: list[_FakeTargetObject] = []

    def _factory(*, fname: str, name: str, radio_material: Any) -> _FakeTargetObject:
        target = _FakeTargetObject(
            fname=fname,
            name=name,
            radio_material=radio_material,
            fail_position=fail_position,
            fail_orientation=fail_orientation,
            on_property_write=on_property_write,
        )
        created.append(target)
        return target

    monkeypatch.setattr(manager_module, "SceneObject", _factory)
    monkeypatch.setattr(manager_module, "_prepare_mesh_path_for_mitsuba", lambda path: path)
    monkeypatch.setattr(manager_module, "_ply_header_has_faces", lambda _path: True)
    monkeypatch.setattr(manager_module, "point3f", lambda value: tuple(value))
    monkeypatch.setattr(
        TargetManager,
        "_apply_initial_orientation_preview",
        lambda _self, _target, _position: None,
    )
    return created


def test_initial_target_failure_removes_candidate_and_preserves_manager_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _StatefulScene()
    manager = _manager(scene, _write_meshes(tmp_path, count=1), active=None)
    created = _install_scene_object_factory(
        monkeypatch,
        fail_position="initial position rejected",
    )

    with pytest.raises(ComputationError) as caught:
        manager.create_target()

    message = str(caught.value)
    assert "Target" in message
    assert "creat" in message.lower() or "construct" in message.lower()
    assert "initial position rejected" in message
    assert manager.target_object is None
    assert scene.objects == {}
    assert created and created[0] in scene.edits[-1]["remove"]


def test_initial_add_failure_after_scene_mutation_removes_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _StatefulScene(fail_after_edit=1)
    manager = _manager(scene, _write_meshes(tmp_path, count=1), active=None)
    created = _install_scene_object_factory(monkeypatch)

    with pytest.raises(ComputationError, match="scene edit failed after mutation"):
        manager.create_target()

    assert scene.objects == {}
    assert manager.target_object is None
    assert scene.edits[-1]["remove"] == (created[0],)


def test_initial_orientation_failure_removes_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _StatefulScene()
    manager = _manager(scene, _write_meshes(tmp_path, count=1), active=None)
    created = _install_scene_object_factory(monkeypatch)

    def _reject_orientation(_self, _target, _position) -> None:
        raise RuntimeError("initial orientation rejected")

    monkeypatch.setattr(TargetManager, "_apply_initial_orientation_preview", _reject_orientation)

    with pytest.raises(ComputationError, match="initial orientation rejected"):
        manager.create_target()

    assert scene.objects == {}
    assert manager.target_object is None
    assert scene.edits[-1]["remove"] == (created[0],)


def test_initial_preview_reads_prepared_timeline_without_resampling(monkeypatch) -> None:
    class _PreparedTimelineOrientation:
        def prepare(self, _steps: int, _duration: float, context=None) -> None:
            raise AssertionError("the complete timeline must not be resampled for a preview")

        def orientations(self):
            return [(10.0, 20.0, 30.0)]

    target = _FakeTargetObject(
        fname="target.obj",
        name="Target",
        radio_material=_FakeMaterial(),
    )
    manager = TargetManager.__new__(TargetManager)
    manager.config = SimpleNamespace(
        name="Target",
        orientation=_PreparedTimelineOrientation(),
        asset_front_yaw_offset_deg=0.0,
    )
    monkeypatch.setattr(
        manager_module,
        "orientation_to_point3f_with_engine_radians",
        lambda value: (tuple(value), tuple(value)),
    )

    manager._apply_initial_orientation_preview(target, (0.0, 0.0, 0.0))

    assert target.orientation == pytest.approx((10.0, 20.0, 30.0))


def test_initial_preview_defers_unprepared_and_timeline_dependent_orientation() -> None:
    class _UnpreparedOrientation:
        def prepare(self, _steps: int, _duration: float, context=None) -> None:
            return None

        def orientations(self):
            return []

    target = _FakeTargetObject(
        fname="target.obj",
        name="Target",
        radio_material=_FakeMaterial(),
    )
    manager = TargetManager.__new__(TargetManager)

    for orientation in (_UnpreparedOrientation(), AlignMotionOrientationSpec()):
        manager.config = SimpleNamespace(name="Target", orientation=orientation)
        manager._apply_initial_orientation_preview(target, (0.0, 0.0, 0.0))

    assert target.orientation == (0.0, 0.0, 0.0)


def test_scene_service_propagates_required_target_construction_failure(monkeypatch) -> None:
    scene = _StatefulScene()
    service = SceneService(SimpleNamespace(material_overrides=None))
    service.scene = scene

    class _FailingManager:
        def __init__(self, config, manager_scene, **_kwargs) -> None:
            self.config = config
            self.scene = manager_scene

        def create_target(self):
            raise ComputationError("Target create failed: candidate rejected")

    monkeypatch.setattr(
        "generator.core.services.scene_service.TargetManager",
        _FailingManager,
    )

    with pytest.raises(ComputationError, match="Target create failed"):
        service._create_targets([SimpleNamespace(name="Target")])

    assert service.target_managers == []
    assert service.target_objects == []


def test_mesh_replacement_uses_one_combined_edit_and_commits_state_last(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = _FakeTargetObject(fname="old.obj", name="Target", radio_material=_FakeMaterial())
    old.position = (4.0, 5.0, 6.0)
    old.scaling = (2.0, 2.0, 2.0)
    old.velocity = (0.5, 0.0, 0.0)
    scene = _StatefulScene(old)
    meshes = _write_meshes(tmp_path)
    manager = _manager(scene, meshes, active=old)
    observed_states: list[tuple[Any, int, int]] = []

    def _observe_manager_state() -> None:
        observed_states.append(
            (manager.target_object, manager.current_mesh_idx, manager._mesh_call_count)
        )

    created = _install_scene_object_factory(
        monkeypatch,
        on_property_write=_observe_manager_state,
    )

    result = manager.update_mesh_for_frame(17)

    candidate = created[0]
    assert result is candidate
    assert scene.edits == [{"add": (candidate,), "remove": (old,)}]
    assert scene.objects == {"Target": candidate}
    assert observed_states
    assert all(state == (old, 0, 1) for state in observed_states)
    assert manager.target_object is candidate
    assert manager.current_mesh_idx == 1
    assert manager._mesh_call_count == 2


def test_candidate_property_failure_rolls_back_scene_and_manager_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = _FakeTargetObject(fname="old.obj", name="Target", radio_material=_FakeMaterial())
    old.position = (4.0, 5.0, 6.0)
    scene = _StatefulScene(old)
    manager = _manager(scene, _write_meshes(tmp_path), active=old)
    created = _install_scene_object_factory(
        monkeypatch,
        fail_position="candidate position rejected",
    )

    with pytest.raises(ComputationError) as caught:
        manager.update_mesh_for_frame(17)

    candidate = created[0]
    message = str(caught.value)
    assert "Target" in message
    assert "mesh" in message.lower()
    assert "17" in message
    assert "candidate position rejected" in message
    assert scene.edits == [
        {"add": (candidate,), "remove": (old,)},
        {"add": (old,), "remove": (candidate,)},
    ]
    assert scene.objects == {"Target": old}
    assert manager.target_object is old
    assert manager.current_mesh_idx == 0
    assert manager._mesh_call_count == 1


def test_combined_edit_failure_after_mutation_restores_previous_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = _FakeTargetObject(fname="old.obj", name="Target", radio_material=_FakeMaterial())
    scene = _StatefulScene(old, fail_after_edit=1)
    manager = _manager(scene, _write_meshes(tmp_path), active=old)
    candidate = _install_scene_object_factory(monkeypatch)

    with pytest.raises(ComputationError, match="scene edit failed after mutation"):
        manager.update_mesh_for_frame(19)

    assert scene.objects == {"Target": old}
    assert scene.edits[-1] == {"add": (old,), "remove": (candidate[0],)}
    assert manager.target_object is old
    assert manager.current_mesh_idx == 0
    assert manager._mesh_call_count == 1


def test_combined_edit_failure_after_removal_restores_previous_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = _FakeTargetObject(fname="old.obj", name="Target", radio_material=_FakeMaterial())
    scene = _StatefulScene(old, fail_after_remove=1)
    manager = _manager(scene, _write_meshes(tmp_path), active=old)
    _install_scene_object_factory(monkeypatch)

    with pytest.raises(ComputationError, match="scene edit failed after removal"):
        manager.update_mesh_for_frame(21)

    assert scene.objects == {"Target": old}
    assert scene.edits[-1] == {"add": (old,), "remove": ()}
    assert manager.target_object is old
    assert manager.current_mesh_idx == 0
    assert manager._mesh_call_count == 1


def test_rollback_failure_reports_primary_and_rollback_causes_without_committing_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = _FakeTargetObject(fname="old.obj", name="Target", radio_material=_FakeMaterial())
    scene = _StatefulScene(old, fail_on_edit=2)
    manager = _manager(scene, _write_meshes(tmp_path), active=old)
    created = _install_scene_object_factory(
        monkeypatch,
        fail_position="candidate position rejected",
    )

    with pytest.raises(ComputationError) as caught:
        manager.update_mesh_for_frame(23)

    message = str(caught.value)
    assert "Target" in message
    assert "mesh" in message.lower()
    assert "23" in message
    assert "candidate position rejected" in message
    assert "rollback edit rejected" in message
    assert scene.objects == {"Target": created[0]}
    assert manager.target_object is old
    assert manager.current_mesh_idx == 0
    assert manager._mesh_call_count == 1


def test_mesh_noop_returns_the_active_target(tmp_path: Path) -> None:
    old = _FakeTargetObject(fname="old.obj", name="Target", radio_material=_FakeMaterial())
    scene = _StatefulScene(old)
    manager = _manager(scene, _write_meshes(tmp_path, count=1), active=old)

    result = manager.update_mesh_for_frame(9)

    assert result is old
    assert scene.edits == []
    assert manager.target_object is old


def test_position_snapshot_failure_is_contextual(tmp_path: Path, monkeypatch) -> None:
    target = _FakeTargetObject(
        fname="old.obj",
        name="Target",
        radio_material=_FakeMaterial(),
        fail_position="position rejected",
    )
    manager = _manager(_StatefulScene(target), _write_meshes(tmp_path), active=target)
    monkeypatch.setattr(manager_module, "point3f", lambda value: tuple(value))

    with pytest.raises(ComputationError) as caught:
        manager.apply_position_snapshot((4.0, 5.0, 6.0))

    assert "Target" in str(caught.value)
    assert "position rejected" in str(caught.value)


def test_orientation_snapshot_failure_is_contextual(tmp_path: Path, monkeypatch) -> None:
    target = _FakeTargetObject(
        fname="old.obj",
        name="Target",
        radio_material=_FakeMaterial(),
        fail_orientation="orientation rejected",
    )
    manager = _manager(_StatefulScene(target), _write_meshes(tmp_path), active=target)
    monkeypatch.setattr(
        manager_module,
        "orientation_to_point3f_with_engine_radians",
        lambda value: (tuple(value), tuple(value)),
    )

    with pytest.raises(ComputationError) as caught:
        manager.apply_orientation_snapshot((10.0, 20.0, 30.0))

    assert "Target" in str(caught.value)
    assert "orientation rejected" in str(caught.value)


@pytest.mark.parametrize("field", ["position", "orientation"])
def test_actor_state_pose_failure_identifies_target_field_and_frame(
    field: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = _FakeTargetObject(
        fname="old.obj",
        name="Target",
        radio_material=_FakeMaterial(),
        fail_position="position rejected" if field == "position" else None,
        fail_orientation="orientation rejected" if field == "orientation" else None,
    )
    manager = _manager(_StatefulScene(target), _write_meshes(tmp_path), active=target)
    manager.config.switch_meshes = False
    monkeypatch.setattr(manager_module, "point3f", lambda value: tuple(value))
    monkeypatch.setattr(
        manager_module,
        "orientation_to_point3f_with_engine_radians",
        lambda value: (tuple(value), tuple(value)),
    )

    positions = [(4.0, 5.0, 6.0)] if field == "position" else [None]
    orientations = [(10.0, 20.0, 30.0)] if field == "orientation" else [None]
    with pytest.raises(ComputationError) as caught:
        apply_target_state_to_scene(
            [manager],
            step_positions=positions,
            step_orientations=orientations,
            frame_idx=29,
        )

    message = str(caught.value)
    assert "Target" in message
    assert field in message
    assert "29" in message
    assert f"{field} rejected" in message


def test_missing_prepared_orientation_leaves_current_value_unchanged() -> None:
    target = _FakeTargetObject(
        fname="old.obj",
        name="Target",
        radio_material=_FakeMaterial(),
    )
    manager = SimpleNamespace(
        config=SimpleNamespace(
            name="Target",
            mobility=None,
            switch_meshes=False,
            use_ply_position=False,
        ),
        target_object=target,
        apply_orientation_snapshot=lambda _value: pytest.fail(
            "a missing prepared orientation must not reinterpret config data"
        ),
    )

    apply_target_state_to_scene(
        [manager],
        step_positions=[None],
        step_orientations=[None],
        frame_idx=3,
    )

    assert target.orientation == (0.0, 0.0, 0.0)


def test_mesh_failure_propagates_from_actor_state_application() -> None:
    class _FailingManager:
        config = SimpleNamespace(
            name="Target",
            mobility=None,
            switch_meshes=True,
            use_ply_position=False,
        )
        target_object = _FakeTargetObject(
            fname="old.obj",
            name="Target",
            radio_material=_FakeMaterial(),
        )

        def update_mesh_for_frame(self, frame_idx: int, *, expected_call_count=None):
            raise ComputationError(f"Target mesh update failed at frame {frame_idx}")

        def apply_orientation_snapshot(self, _orientation) -> None:
            raise AssertionError("orientation must not run after a mesh failure")

    with pytest.raises(ComputationError, match="Target mesh update failed at frame 7"):
        apply_target_state_to_scene(
            [_FailingManager()],
            step_positions=[None],
            step_orientations=[None],
            frame_idx=7,
        )
