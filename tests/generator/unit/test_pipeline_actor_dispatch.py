from pathlib import Path
from types import SimpleNamespace

import pytest

from generator.core.configuration import build_simulation_config
from generator.core.pipeline import dispatch
from shared.scenarios import load_scenario_configuration
from shared.scenarios.actors import (
    ActorsSpec,
    RxActorSpec,
    StationaryMobilitySpec,
    TxActorSpec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_yaml_pipeline_dispatches_prepared_actor_adapters(monkeypatch) -> None:
    scenario = load_scenario_configuration(
        PROJECT_ROOT / "scenarios" / "generator" / "mobility_and_orientation" / "actor_mobility",
        project_root=PROJECT_ROOT,
    )
    simulation = build_simulation_config(scenario)
    captured = {}

    def run_offline(tx, rx, targets, simulation_config, **kwargs):
        captured.update(tx=tx, rx=rx, targets=targets, scenario=kwargs["scenario_configuration"])
        assert simulation_config is simulation
        return "prepared"

    monkeypatch.setattr(dispatch, "perform_offline_pipeline", run_offline)

    result = dispatch.perform_pipeline(
        simulation_config=simulation,
        scenario_configuration=scenario,
        configure_logging_enabled=False,
        show_progress=False,
    )

    assert result == "prepared"
    assert [actor.name for actor in captured["tx"]] == ["MainTransmitter"]
    assert len(captured["rx"]) == 5
    assert captured["targets"] == []
    assert captured["scenario"] is scenario


def test_scripted_pipeline_accepts_immutable_actor_specs(monkeypatch) -> None:
    scenario = load_scenario_configuration(
        PROJECT_ROOT / "scenarios" / "getting_started" / "hello_world_scripted",
        project_root=PROJECT_ROOT,
    )
    simulation = build_simulation_config(scenario)
    actors = ActorsSpec(
        tx=(
            TxActorSpec(
                name="TX1",
                mobility=StationaryMobilitySpec(position_m=(0.0, 0.0, 2.0)),
            ),
        ),
        rx=(
            RxActorSpec(
                name="RX1",
                mobility=StationaryMobilitySpec(position_m=(3.0, 0.0, 1.0)),
            ),
        ),
    )
    captured = {}

    def run_offline(tx, rx, targets, simulation_config, **kwargs):
        captured.update(tx=tx, rx=rx, targets=targets, scenario=kwargs["scenario_configuration"])
        assert simulation_config is simulation
        return "scripted"

    monkeypatch.setattr(dispatch, "perform_offline_pipeline", run_offline)

    result = dispatch.perform_pipeline(
        simulation_config=simulation,
        scenario_configuration=scenario,
        actors=actors,
        configure_logging_enabled=False,
        show_progress=False,
    )

    assert result == "scripted"
    assert [actor.name for actor in captured["tx"]] == ["TX1"]
    assert [actor.name for actor in captured["rx"]] == ["RX1"]
    assert captured["scenario"].actors is actors


def test_streaming_dispatch_rejects_partial_frame_set_start_step(monkeypatch) -> None:
    simulation = SimpleNamespace(
        debug_level="WARNING",
        output_mode="streaming",
        start_step=2,
    )

    def unexpected_streaming_call(*_args, **_kwargs):
        pytest.fail("streaming setup must not start for an unsupported partial frame set")

    monkeypatch.setattr(dispatch, "_load_streaming_pipeline", unexpected_streaming_call)

    with pytest.raises(
        ValueError,
        match="start_step is only supported for file-output generation",
    ):
        dispatch.perform_pipeline(
            tx_configs=[],
            rx_configs=[],
            target_configs=[],
            simulation_config=simulation,
            configure_logging_enabled=False,
        )
