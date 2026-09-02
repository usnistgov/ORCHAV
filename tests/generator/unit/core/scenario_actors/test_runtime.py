from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.scenario_actors.quaternion import Quaternion
from generator.core.scenario_actors.runtime import (
    _target_asset_directory,
    _target_mesh_location,
    _target_runtime,
    prepare_actor_runtime,
)
from generator.core.scenario_actors.state import ActorStateManager
from generator.core.scenario_actors.types import (
    PreparedActorPose,
    PreparedMobility,
    PreparedOrientation,
)
from shared.scenarios import load_scenario_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def test_runtime_rejects_catalog_paths_outside_the_catalog_root(tmp_path: Path) -> None:
    configuration = SimpleNamespace(
        root=tmp_path,
        targets_dir=tmp_path / "libraries" / "targets",
    )

    with pytest.raises(ValueError, match="outside the catalog root"):
        _target_asset_directory(
            {"source": "catalog", "id": "../outside"},
            scenario_configuration=configuration,
        )


def test_target_mesh_location_uses_scenario_root_from_any_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_root = tmp_path / "scenario"
    mesh_directory = scenario_root / "assets" / "vehicle"
    mesh_directory.mkdir(parents=True)
    (mesh_directory / "vehicle.ply").touch()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=tmp_path,
        targets_dir=tmp_path / "libraries" / "targets",
    )

    authored, resolved, pattern = _target_mesh_location(
        {"source": "directory", "path": "assets/vehicle", "pattern": "*.ply"},
        scenario_configuration=configuration,
        target_name="Vehicle",
    )

    assert authored == "assets/vehicle"
    assert resolved == mesh_directory.resolve()
    assert pattern == "*.ply"


def test_target_mesh_location_reports_actor_field_and_resolved_path(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=tmp_path,
        targets_dir=tmp_path / "libraries" / "targets",
    )

    with pytest.raises(ValueError) as raised:
        _target_mesh_location(
            {"source": "file", "path": "assets/missing.ply"},
            scenario_configuration=configuration,
            target_name="MissingTarget",
        )

    message = str(raised.value)
    assert "MissingTarget" in message
    assert "actors.targets[].asset.path" in message
    assert "assets/missing.ply" in message
    assert str((scenario_root / "assets" / "missing.ply").resolve()) in message


def test_target_runtime_propagates_mesh_end_behavior(tmp_path: Path) -> None:
    mesh_directory = tmp_path / "assets" / "human"
    mesh_directory.mkdir(parents=True)
    (mesh_directory / "human_000.ply").touch()
    actor = PreparedActorPose(
        name="Human",
        role="target",
        mobility=PreparedMobility(
            positions_m=((0.0, 0.0, 0.0),),
            velocities_mps=((0.0, 0.0, 0.0),),
            forward_vectors=((1.0, 0.0, 0.0),),
        ),
        orientation=PreparedOrientation((Quaternion.identity(),)),
    )
    spec = SimpleNamespace(
        asset=SimpleNamespace(
            source="directory",
            path="assets/human",
            pattern="*.ply",
            mesh_end_behavior="hold_last",
        )
    )
    configuration = SimpleNamespace(
        root=tmp_path,
        project_root=tmp_path,
        targets_dir=tmp_path / "libraries" / "targets",
        timeline=SimpleNamespace(steps=1, duration_s=0.0),
    )

    target = _target_runtime(actor, spec, scenario_configuration=configuration)

    assert target.mesh_end_behavior == "hold_last"


def test_runtime_adapters_preserve_canonical_pose_and_velocity_samples() -> None:
    scenario = load_scenario_configuration(
        PROJECT_ROOT / "scenarios" / "generator" / "mobility_and_orientation" / "actor_mobility",
        project_root=PROJECT_ROOT,
    )
    runtime = prepare_actor_runtime(scenario)
    manager = ActorStateManager(
        list(runtime.transmitters),
        list(runtime.receivers),
        [],
        scenario.timeline.steps,
        scenario.timeline.duration_s,
    )

    cache = manager.prepare_cached()

    for index, actor in enumerate(runtime.prepared.actors_for_role("tx")):
        np.testing.assert_allclose(cache.tx_positions[index], actor.positions_m)
        np.testing.assert_allclose(
            cache.tx_orientations[index],
            actor.orientation.euler_deg,
        )
        np.testing.assert_allclose(manager.compute_velocities(3).tx[index], actor.velocities_mps[3])
    for index, actor in enumerate(runtime.prepared.actors_for_role("rx")):
        np.testing.assert_allclose(cache.rx_positions[index], actor.positions_m)
        np.testing.assert_allclose(
            cache.rx_orientations[index],
            actor.orientation.euler_deg,
        )
        np.testing.assert_allclose(manager.compute_velocities(3).rx[index], actor.velocities_mps[3])


def test_target_runtime_uses_prepared_orientation_after_asset_alignment() -> None:
    scenario = load_scenario_configuration(
        PROJECT_ROOT / "scenarios" / "generator" / "targets" / "mesh_targets",
        project_root=PROJECT_ROOT,
    )
    runtime = prepare_actor_runtime(scenario)
    pedestrian = next(target for target in runtime.targets if target.name == "Pedestrian")
    assert pedestrian.mesh_directory == "libraries/targets/nist_human_walking"
    assert pedestrian.mesh_end_behavior == "loop"
    assert (
        pedestrian.resolved_mesh_directory
        == (PROJECT_ROOT / "libraries" / "targets" / "nist_human_walking").resolve()
    )
    managers = [SimpleNamespace(config=target, meshes=[]) for target in runtime.targets]
    manager = ActorStateManager(
        list(runtime.transmitters),
        list(runtime.receivers),
        managers,
        scenario.timeline.steps,
        scenario.timeline.duration_s,
    )

    cache = manager.prepare_cached()

    for index, actor in enumerate(runtime.prepared.actors_for_role("target")):
        np.testing.assert_allclose(cache.target_positions[index], actor.positions_m)
        np.testing.assert_allclose(
            cache.target_orientations[index],
            actor.orientation.euler_deg,
            atol=1e-12,
        )
