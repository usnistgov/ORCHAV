import io
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from shared.source_identity import SourceIdentity, loaded_source_identity
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.services import live_preview_worker
from visualizer.src.services.live_preview_worker import _load_simulation, _solve_preview_frame


def _install_preview_runtime_modules(
    monkeypatch,
    *,
    scenario_configuration,
    simulation_config,
    actor_runtime,
    build_result=None,
):
    calls = {}

    configuration_module = ModuleType("generator.core.configuration")
    configuration_module.build_simulation_config = lambda scenario: simulation_config

    monkeypatch.setattr(
        "shared.scenarios.load_scenario_configuration",
        lambda root, *, project_root=None: scenario_configuration,
    )

    actor_runtime_module = ModuleType("generator.core.scenario_actors.runtime")
    actor_runtime_module.prepare_actor_runtime = lambda scenario: actor_runtime

    runtime_module = ModuleType("generator.core.runtime")

    def _build_on_demand_objects(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return build_result

    runtime_module.build_on_demand_objects = _build_on_demand_objects

    monkeypatch.setitem(sys.modules, configuration_module.__name__, configuration_module)
    monkeypatch.setitem(sys.modules, actor_runtime_module.__name__, actor_runtime_module)
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    return calls


def test_load_simulation_builds_preview_from_prepared_actor_runtime(tmp_path, monkeypatch):
    simulation_config = SimpleNamespace(num_steps=2)
    tx = object()
    rx = object()
    target = object()
    scenario_configuration = SimpleNamespace(
        root=tmp_path, actors=SimpleNamespace(tx=(tx,), rx=(rx,))
    )
    actor_runtime = SimpleNamespace(
        transmitters=(tx,),
        receivers=(rx,),
        targets=(target,),
    )
    simulation = object()
    calls = _install_preview_runtime_modules(
        monkeypatch,
        scenario_configuration=scenario_configuration,
        simulation_config=simulation_config,
        actor_runtime=actor_runtime,
        build_result=simulation,
    )
    settings = {"max_depth": 2}

    result = _load_simulation(
        {
            "scenario_root": str(tmp_path),
            "project_root": str(tmp_path.parent),
            "step": 4,
            "solver_settings": settings,
        }
    )

    assert result is simulation
    assert simulation_config.num_steps == 5
    assert calls["args"] == ([tx], [rx], [target], simulation_config)
    assert calls["kwargs"] == {
        "scenario_configuration": scenario_configuration,
        "motion_mode": "step",
        "raytracing_settings": settings,
    }


def test_scripted_hello_world_preview_reuses_generation_actors(monkeypatch):
    from generator.core.pipeline import dispatch

    root = Path(__file__).resolve().parents[3]
    scenario_root = root / "scenarios/getting_started/hello_world_scripted"
    driver = runpy.run_path(str(scenario_root / "generate.py"))
    generation = Mock()
    monkeypatch.setattr(dispatch, "perform_offline_pipeline", generation)
    driver["main"]()
    generated_actors = generation.call_args.kwargs["scenario_configuration"].actors
    generated_tx, generated_rx, *_ = generation.call_args.args

    runtime = ModuleType("generator.core.runtime")
    builder = Mock(return_value=object())
    runtime.build_on_demand_objects = builder
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    assert (
        _load_simulation({"scenario_root": str(scenario_root), "step": 8}) is builder.return_value
    )
    tx, rx, targets, config = builder.call_args.args
    effective_scenario = builder.call_args.kwargs["scenario_configuration"]
    assert effective_scenario.actors == generated_actors
    assert [actor.name for actor in tx + rx] == ["TX1", "RX1"]
    assert not targets
    assert config.num_steps == 16
    positions = np.asarray(rx[0].mobility.prepared_positions())
    np.testing.assert_allclose(positions, generated_rx[0].mobility.prepared_positions())
    np.testing.assert_allclose(
        tx[0].mobility.prepared_positions(), generated_tx[0].mobility.prepared_positions()
    )
    np.testing.assert_allclose(
        rx[0].orientation.orientations(), generated_rx[0].orientation.orientations()
    )
    np.testing.assert_allclose(positions[0], [-27.18, -12.61, 1.5])
    assert np.linalg.norm(positions[-1] - positions[0]) > 100
    # Every local +X axis must point at TX1 along the curved trajectory.
    yaw, pitch, _roll = np.radians(rx[0].orientation.orientations()).T
    forward = np.column_stack(
        (np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch))
    )
    direction = np.asarray(tx[0].mobility.prepared_positions()) - positions
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    np.testing.assert_allclose(forward, direction, atol=1e-12)


@pytest.mark.parametrize("yaml_actors", [False, True])
def test_preview_never_runs_generation_and_keeps_yaml_actors(tmp_path, monkeypatch, yaml_actors):
    document = (
        "schema_version: 2\nscene: {source: sionna, id: etoile}\n"
        "timeline: {steps: 2, duration_s: 1.0}\n"
    )
    if yaml_actors:
        document += """actors:
  tx:
    - name: YAML_TX
      mobility: {type: stationary, position_m: [0, 0, 2]}
  rx:
    - name: YAML_RX
      mobility: {type: stationary, position_m: [3, 0, 1]}
"""
        (tmp_path / "actors.py").write_text("raise AssertionError('YAML actors take precedence')\n")
    (tmp_path / "scenario.yaml").write_text(document)
    (tmp_path / "generate.py").write_text("raise AssertionError('Generation must stay explicit')\n")
    runtime = ModuleType("generator.core.runtime")
    runtime.build_on_demand_objects = Mock()
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    if yaml_actors:
        _load_simulation({"scenario_root": str(tmp_path)})
        tx, rx, *_ = runtime.build_on_demand_objects.call_args.args
        assert [actor.name for actor in tx + rx] == ["YAML_TX", "YAML_RX"]
    else:
        with pytest.raises(RuntimeError, match=r"actors.py factory defining build_actors"):
            _load_simulation({"scenario_root": str(tmp_path)})
        runtime.build_on_demand_objects.assert_not_called()


def test_init_reports_the_loaded_worker_source_identity(monkeypatch):
    state = SimpleNamespace(ensure_simulation=Mock())
    emitted = []
    monkeypatch.setattr(live_preview_worker, "_emit", emitted.append)

    live_preview_worker._handle_init(state, {"scenario_root": "scenario"})

    state.ensure_simulation.assert_called_once_with({"scenario_root": "scenario"})
    actual = SourceIdentity.from_mapping(emitted[0]["source_identity"])
    assert actual.matches(loaded_source_identity("visualizer"))


def test_init_reports_a_checkout_revision_changed_during_simulation_loading(monkeypatch):
    initial_identity = loaded_source_identity("visualizer")
    updated_identity = SourceIdentity(
        source_root=initial_identity.source_root,
        version=initial_identity.version,
        git_sha="revision-written-during-initialization",
    )
    disk_identity = initial_identity

    def initialize_simulation(_request):
        nonlocal disk_identity
        disk_identity = updated_identity

    def resolve_identity(root):
        assert root == initial_identity.source_root
        return disk_identity

    state = SimpleNamespace(ensure_simulation=Mock(side_effect=initialize_simulation))
    emitted = []
    monkeypatch.setattr(live_preview_worker, "_emit", emitted.append)
    monkeypatch.setattr(live_preview_worker, "resolve_source_identity", resolve_identity)

    live_preview_worker._handle_init(state, {"scenario_root": "scenario"})

    assert emitted == [{"type": "ready", "source_identity": updated_identity.to_dict()}]
    assert loaded_source_identity("visualizer") is initial_identity
    assert not initial_identity.matches(SourceIdentity.from_mapping(emitted[0]["source_identity"]))


def test_stdio_announces_identity_before_waiting_for_commands(monkeypatch):
    output = io.StringIO()
    flush = Mock(wraps=output.flush)
    load_simulation = Mock()
    monkeypatch.setattr(output, "flush", flush)

    def incoming_commands():
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        assert messages == [
            {
                "type": "identity",
                "source_identity": loaded_source_identity("visualizer").to_dict(),
            }
        ]
        flush.assert_called_once_with()
        load_simulation.assert_not_called()
        yield json.dumps({"command": "shutdown"}) + "\n"

    monkeypatch.setattr(live_preview_worker.sys, "stdin", incoming_commands())
    monkeypatch.setattr(live_preview_worker.sys, "stdout", output)
    monkeypatch.setattr(live_preview_worker, "_load_simulation", load_simulation)

    assert live_preview_worker.serve_stdio() == 0
    assert json.loads(output.getvalue().splitlines()[-1]) == {"type": "bye"}
    load_simulation.assert_not_called()


@pytest.mark.parametrize("input_text", ["", '{"command": "shutdown"}\n'])
def test_stdio_can_exit_before_init_without_loading_simulation(monkeypatch, input_text):
    output = io.StringIO()
    load_simulation = Mock()
    monkeypatch.setattr(live_preview_worker.sys, "stdin", io.StringIO(input_text))
    monkeypatch.setattr(live_preview_worker.sys, "stdout", output)
    monkeypatch.setattr(live_preview_worker, "_load_simulation", load_simulation)

    assert live_preview_worker.serve_stdio() == 0

    message_types = [json.loads(line)["type"] for line in output.getvalue().splitlines()]
    assert message_types == (["identity", "bye"] if input_text else ["identity"])
    load_simulation.assert_not_called()


def test_stdio_announces_identity_then_initializes_before_ready(monkeypatch):
    output = io.StringIO()
    request = {"scenario_root": "scenario", "step": 0}
    simulation = SimpleNamespace(simulation_config=SimpleNamespace(num_steps=1))

    def load_simulation(received_request):
        assert received_request == request
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        assert [message["type"] for message in messages] == ["identity"]
        return simulation

    loader = Mock(side_effect=load_simulation)
    input_text = "\n".join(
        json.dumps(message)
        for message in (
            {"command": "init", "request": request},
            {"command": "shutdown"},
        )
    )
    monkeypatch.setattr(live_preview_worker.sys, "stdin", io.StringIO(input_text))
    monkeypatch.setattr(live_preview_worker.sys, "stdout", output)
    monkeypatch.setattr(live_preview_worker, "_load_simulation", loader)

    assert live_preview_worker.serve_stdio() == 0

    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [message["type"] for message in messages] == ["identity", "ready", "bye"]
    assert messages[0]["source_identity"] == messages[1]["source_identity"]
    loader.assert_called_once_with(request)


@pytest.mark.parametrize("bad_command", ["{", '{"command": "unknown"}'])
def test_stdio_preserves_error_handling_before_init(monkeypatch, bad_command):
    output = io.StringIO()
    load_simulation = Mock()
    input_text = bad_command + '\n{"command": "shutdown"}\n'
    monkeypatch.setattr(live_preview_worker.sys, "stdin", io.StringIO(input_text))
    monkeypatch.setattr(live_preview_worker.sys, "stdout", output)
    monkeypatch.setattr(live_preview_worker.sys, "stderr", io.StringIO())
    monkeypatch.setattr(live_preview_worker, "_load_simulation", load_simulation)

    assert live_preview_worker.serve_stdio() == 0

    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [message["type"] for message in messages] == ["identity", "error", "bye"]
    assert messages[1]["error"]
    load_simulation.assert_not_called()


def test_solve_preview_frame_returns_canonical_frame_with_provenance(monkeypatch):
    expected = build_standard_mpc_frame("baseline", frame_idx=4)
    calls = {}

    raytracing_module = ModuleType("generator.core.propagation.raytracing")
    raytracing_module.compute_ray_tracing_step = lambda simulation, step, **kwargs: {
        "tx_list": [object()],
        "rx_list": [object()],
        "paths": object(),
        "target_objects": [],
        "target_managers": [],
        "simulation_config": SimpleNamespace(bandwidth_hz=4.0e8),
        "material_mapping": None,
    }
    builder_module = ModuleType("generator.io.frames.builder")

    def process_frame_data(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return expected

    builder_module.process_frame_data = process_frame_data
    monkeypatch.setitem(sys.modules, raytracing_module.__name__, raytracing_module)
    monkeypatch.setitem(sys.modules, builder_module.__name__, builder_module)
    monkeypatch.setattr(live_preview_worker, "build_live_overrides", lambda *_args: {})
    monkeypatch.setattr(live_preview_worker.time, "time", lambda: 123.5)

    result = _solve_preview_frame(
        {
            "step": 4,
            "scene_time_s": 2.75,
            "sequence": 9,
            "quality": "drag",
            "scenario_root": "scenario",
            "solver_settings": {"seed": 17},
            "tx_positions": [[1.0, 2.0, 3.0]],
            "rx_positions": [[4.0, 5.0, 6.0]],
            "target_positions": [[7.0, 8.0, 9.0]],
            "target_orientations": [[0.1, 0.2, 0.3]],
        },
        object(),
    )

    assert result is expected
    assert calls["args"][0] == 4
    assert calls["kwargs"]["timestamp_s"] == 2.75
    assert calls["kwargs"]["provenance"]["timestamp"] == 2.75
    assert calls["kwargs"]["provenance"]["computed_at_unix_s"] == 123.5
    assert calls["kwargs"]["provenance"]["provider"] == "live_preview"
    assert calls["kwargs"]["provenance"]["sequence_id"] == 9
    assert calls["kwargs"]["provenance"]["seed"] == 17


@pytest.mark.parametrize(
    "payload,config,expected",
    [
        ({"step": 128, "scene_time_s": 43.125}, SimpleNamespace(num_steps=200, duration=1), 43.125),
        ({"step": 128, "scene_time_s": None}, SimpleNamespace(num_steps=200, duration=1), None),
        ({"step": 3}, SimpleNamespace(num_steps=6, duration=10), 6.0),
        ({"step": 0}, SimpleNamespace(num_steps=1, duration=0), 0.0),
        ({"step": 7}, SimpleNamespace(), None),
    ],
)
def test_preview_clock_prefers_recorded_time_and_preserves_unknown(payload, config, expected):
    simulation = SimpleNamespace(simulation_config=config)
    assert live_preview_worker._preview_scene_time_s(payload, simulation) == expected


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_preview_clock_rejects_nonfinite_recorded_time(timestamp):
    with pytest.raises(ValueError, match="must be finite"):
        live_preview_worker._preview_scene_time_s({"scene_time_s": timestamp}, object())


@pytest.mark.parametrize(
    "actor_runtime",
    [
        SimpleNamespace(transmitters=(), receivers=(object(),), targets=()),
        SimpleNamespace(transmitters=(object(),), receivers=(), targets=()),
    ],
)
def test_load_simulation_requires_radio_actor_roles(tmp_path, monkeypatch, actor_runtime):
    calls = _install_preview_runtime_modules(
        monkeypatch,
        scenario_configuration=SimpleNamespace(
            root=tmp_path,
            actors=SimpleNamespace(tx=actor_runtime.transmitters, rx=actor_runtime.receivers),
        ),
        simulation_config=SimpleNamespace(num_steps=1),
        actor_runtime=actor_runtime,
    )

    with pytest.raises(RuntimeError, match="at least one TX and one RX"):
        _load_simulation({"scenario_root": str(tmp_path), "step": 0})

    assert calls == {}
