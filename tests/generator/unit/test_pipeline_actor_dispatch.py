import io
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


@pytest.mark.parametrize("show_progress", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_preparation_progress_respects_visibility_and_closes_on_failure(
    monkeypatch, capsys, show_progress, fail
):
    from generator.core.pipeline import progress
    from generator.core.scenario_actors import runtime

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    scenario = SimpleNamespace()
    simulation = SimpleNamespace(output_mode="file")

    def prepare(configuration, *, on_progress):
        assert configuration is scenario
        assert (on_progress is not None) == show_progress
        if on_progress is not None:
            on_progress("Walker: Verifying pose resources", 0, 1)
        if fail:
            raise ValueError("pose resource mismatch")
        if on_progress is not None:
            on_progress("Walker: Verifying pose resources", 1, 1)
        return SimpleNamespace(transmitters=(), receivers=(), targets=())

    def run_offline(*args, **kwargs):
        assert not fail
        assert kwargs["show_progress"] is show_progress
        return "prepared"

    monkeypatch.setattr(runtime, "prepare_actor_runtime", prepare)
    monkeypatch.setattr(dispatch, "perform_offline_pipeline", run_offline)
    kwargs = dict(
        simulation_config=simulation,
        scenario_configuration=scenario,
        configure_logging_enabled=False,
        show_progress=show_progress,
    )
    if fail:
        with pytest.raises(ValueError, match="pose resource mismatch"):
            dispatch.perform_pipeline(**kwargs)
    else:
        assert dispatch.perform_pipeline(**kwargs) == "prepared"

    output = stream.getvalue()
    if show_progress:
        assert "Walker: Verifying pose resources" in output
        assert output.endswith("\n")
        assert ("1/1" in output) == (not fail)
    else:
        assert output == ""
    assert capsys.readouterr().out == ""


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


@pytest.mark.parametrize("rt_enabled", [False, True])
def test_coverage_sees_initial_pose_before_topology_or_frame_zero(monkeypatch, rt_enabled):
    """Exercise the pipeline/service boundary without launching a ray solver."""
    from unittest.mock import MagicMock

    import numpy as np
    from sionna.rt import Receiver, Transmitter

    import generator.core.services.coverage_service as coverage_module
    from generator.core.pipeline import offline_pipeline as pipeline
    from generator.core.scenario_actors.state import ActorStateSnapshot, ActorStateVelocities
    from generator.core.services.actor_state_service import ActorStateService

    config = SimpleNamespace(
        num_steps=2,
        duration=1.0,
        start_step=0,
        sensing=None,
        coverage=None,
        cir_time_steps=1,
        output_mode="file",
        mesh_update_interval_s=None,
    )
    state = ActorStateSnapshot(
        [(2.0, 3.0, 4.0)], [(5.0, 6.0, 7.0)], [], [(-25.0, 0.0, 0.0)], [(75.0, 0.0, 0.0)], []
    )
    manager = MagicMock()
    manager.state_at_step.return_value = state
    manager.compute_velocities.return_value = ActorStateVelocities(
        [(1.0, 0.0, 0.0)], [(0.0, 1.0, 0.0)], []
    )
    actor = ActorStateService(config)
    actor.actor_state_manager = manager
    actor.normalize_scene_steps = MagicMock()
    actor.prepare_actor_state = MagicMock(return_value=(manager, object()))
    scene = MagicMock()
    scene.tx_list = [Transmitter("tx", position=[2.0, 3.0, 4.0])]
    scene.rx_list = [Receiver("rx", position=[5.0, 6.0, 7.0])]
    scene.target_managers = []
    scene.target_objects = []
    rt = MagicMock()
    events = []

    def assert_pose():
        np.testing.assert_allclose(np.asarray(scene.tx_list[0].position).ravel(), [2, 3, 4])
        np.testing.assert_allclose(
            np.asarray(scene.tx_list[0].orientation).ravel(), np.deg2rad([-25, 0, 0]), atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(scene.rx_list[0].orientation).ravel(), np.deg2rad([75, 0, 0]), atol=1e-7
        )

    def coverage_solver(**kwargs):
        assert kwargs["tx_list"] is scene.tx_list
        assert_pose()
        events.append("coverage")
        return {"observed_pose": True}

    def compute_step(step):
        assert events[0] == "coverage"
        assert_pose()
        events.append(step)
        return {"tx_list": scene.tx_list, "rx_list": scene.rx_list}

    rt.compute_step.side_effect = compute_step
    services = {
        pipeline.SceneService: scene,
        pipeline.ActorStateService: actor,
        pipeline.CoverageService: coverage_module.CoverageService(config),
        pipeline.RayTracingService: rt,
    }
    context = MagicMock()
    context.__enter__.return_value.get_service.side_effect = services.__getitem__
    monkeypatch.setattr(pipeline, "PipelineContext", lambda _: context)
    monkeypatch.setattr(coverage_module, "compute_coverage_map", coverage_solver)
    output_cls = MagicMock()
    output = output_cls.return_value
    output.save_frame_data.side_effect = lambda *_: assert_pose()
    monkeypatch.setattr(pipeline, "HDF5FrameOutputStrategy", output_cls)
    publication_cls = MagicMock()
    publication_cls.return_value.stage.return_value = "coverage.h5"
    monkeypatch.setattr(pipeline, "CoveragePublication", publication_cls)
    summary_cls = MagicMock()
    summary_cls.return_value.active = False
    monkeypatch.setattr(pipeline, "SummaryPublication", summary_cls)
    monkeypatch.setattr(pipeline, "SensingPipelineState", MagicMock())
    scenario = SimpleNamespace(
        root=Path("."),
        raytracing={"enabled": rt_enabled},
        coverage_cfg={"enabled": True},
        material_tuning_cfg={},
    )
    pipeline.perform_offline_pipeline([], [], [], config, scenario, show_progress=False)
    assert events == (["coverage", 0, 1] if rt_enabled else ["coverage"])
    manager.state_at_step.assert_called_once_with(0)
    assert config.num_steps == 2 and config.duration == 1.0
