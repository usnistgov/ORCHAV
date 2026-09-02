import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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

    scenario_module = ModuleType("shared.scenarios")
    scenario_module.load_scenario_configuration = (
        lambda root, *, project_root=None: scenario_configuration
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
    monkeypatch.setitem(sys.modules, scenario_module.__name__, scenario_module)
    monkeypatch.setitem(sys.modules, actor_runtime_module.__name__, actor_runtime_module)
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    return calls


def test_load_simulation_builds_preview_from_prepared_actor_runtime(tmp_path, monkeypatch):
    scenario_configuration = object()
    simulation_config = SimpleNamespace(num_steps=2)
    tx = object()
    rx = object()
    target = object()
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


def test_init_reports_the_loaded_worker_source_identity(monkeypatch):
    state = SimpleNamespace(ensure_simulation=Mock())
    emitted = []
    monkeypatch.setattr(live_preview_worker, "_emit", emitted.append)

    live_preview_worker._handle_init(state, {"scenario_root": "scenario"})

    state.ensure_simulation.assert_called_once_with({"scenario_root": "scenario"})
    actual = SourceIdentity.from_mapping(emitted[0]["source_identity"])
    assert actual.matches(loaded_source_identity("visualizer"))


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
    assert calls["kwargs"]["timestamp_s"] == 123.5
    assert calls["kwargs"]["provenance"]["provider"] == "live_preview"
    assert calls["kwargs"]["provenance"]["sequence_id"] == 9
    assert calls["kwargs"]["provenance"]["seed"] == 17


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
        scenario_configuration=object(),
        simulation_config=SimpleNamespace(num_steps=1),
        actor_runtime=actor_runtime,
    )

    with pytest.raises(RuntimeError, match="at least one TX and one RX"):
        _load_simulation({"scenario_root": str(tmp_path), "step": 0})

    assert calls == {}
