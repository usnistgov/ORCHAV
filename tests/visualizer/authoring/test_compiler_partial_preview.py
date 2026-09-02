"""Generator-backed preview sampling for incomplete authoring documents."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from generator.core.scenario_actors import prepare_scenario
from shared.scenarios.actors import CircularMobilitySpec
from shared.scenarios.yaml import validate_scenario_data
from visualizer.src.authoring import (
    ActorRole,
    AuthoringActor,
    AuthoringScenario,
    ScenarioCompiler,
    SceneReference,
    TargetAsset,
    canonical_scenario_mapping,
)
from visualizer.src.authoring.orientation_models import actor_look_at_orientation

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _circular_tx(name: str = "TX Circular") -> AuthoringActor:
    return AuthoringActor.create(ActorRole.TX, name).with_changes(
        mobility=CircularMobilitySpec(
            center_m=(1.0, 2.0, 3.0),
            radius_m=4.0,
            start_angle_deg=15.0,
            clockwise=False,
        )
    )


def _compile(*actors: AuthoringActor, with_scene: bool = True):
    return ScenarioCompiler(PROJECT_ROOT).compile(
        AuthoringScenario(
            scene=(SceneReference("library", "empty/empty.xml") if with_scene else None),
            actors=actors,
        ),
        scenario_directory=PROJECT_ROOT,
    )


def test_circular_tx_without_rx_is_invalid_but_has_generator_samples() -> None:
    tx = _circular_tx()
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(tx,),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )
    prepared = prepare_scenario(
        validate_scenario_data(canonical_scenario_mapping(scenario)),
        base_dir=PROJECT_ROOT,
    )

    assert result.valid is False
    assert result.runtime is None
    assert any(issue.code == "actors.rx.required" for issue in result.issues)
    np.testing.assert_allclose(
        result.samples[tx.id].positions,
        prepared.actor(tx.name).positions_m,
    )


def test_missing_scene_still_prepares_valid_actor_samples() -> None:
    tx = _circular_tx()
    rx = AuthoringActor.create(ActorRole.RX, "RX", position=(10.0, 0.0, 1.0))

    result = _compile(tx, rx, with_scene=False)

    assert result.valid is False
    assert result.runtime is None
    assert any(issue.code == "scene.required" for issue in result.issues)
    assert set(result.samples) == {tx.id, rx.id}
    assert len(result.samples[tx.id].positions) == 30
    assert len(result.samples[rx.id].positions) == 30


def test_partial_target_preview_uses_catalog_front_alignment() -> None:
    tx = _circular_tx()
    rx = AuthoringActor.create(ActorRole.RX, "RX", position=(10.0, 0.0, 1.0))
    target = AuthoringActor.create(
        ActorRole.TARGET,
        "Car",
        position=(5.0, 0.0, 1.0),
        target=TargetAsset.from_catalog_id("car", mesh_pattern="car.ply"),
    )

    partial = _compile(tx, rx, target, with_scene=False)
    complete = _compile(tx, rx, target, with_scene=True)

    assert partial.valid is False
    assert complete.valid is True, complete.issues
    np.testing.assert_allclose(
        partial.samples[target.id].orientations,
        complete.samples[target.id].orientations,
        atol=1e-9,
    )
    assert abs(partial.samples[target.id].orientations[0][0]) == 180.0


def test_invalid_peer_does_not_suppress_valid_actor_samples() -> None:
    tx = _circular_tx()
    invalid_rx = AuthoringActor.create(ActorRole.RX, "Invalid RX").with_changes(
        mobility=CircularMobilitySpec(
            center_m=(0.0, 0.0, 0.0),
            radius_m=1.0,
        ).model_copy(update={"radius_m": 0.0})
    )

    result = _compile(tx, invalid_rx)

    assert result.valid is False
    assert result.runtime is None
    assert tx.id in result.samples
    assert invalid_rx.id not in result.samples
    assert any(
        issue.code == "schema.invalid" and issue.path == "actors.rx.0.mobility.circular.radius_m"
        for issue in result.issues
    )


def test_invalid_reference_dependency_is_not_fabricated() -> None:
    invalid_rx = AuthoringActor.create(ActorRole.RX, "Invalid RX").with_changes(
        mobility=CircularMobilitySpec(
            center_m=(0.0, 0.0, 0.0),
            radius_m=1.0,
        ).model_copy(update={"radius_m": 0.0})
    )
    tx = _circular_tx().with_changes(orientation=actor_look_at_orientation(invalid_rx.id))
    valid_rx = AuthoringActor.create(ActorRole.RX, "Valid RX", position=(8.0, 0.0, 1.0))

    result = _compile(tx, invalid_rx, valid_rx)

    assert result.valid is False
    assert result.runtime is None
    assert set(result.samples) == {valid_rx.id}
