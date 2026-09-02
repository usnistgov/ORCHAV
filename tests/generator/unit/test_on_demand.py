from types import SimpleNamespace
from unittest.mock import MagicMock

import generator.core.runtime.on_demand as on_demand
from generator.core.scenario_actors.state import ActorStateCache


def test_build_on_demand_objects_uses_scene_service(monkeypatch):
    tx_config = SimpleNamespace(name="tx")
    rx_config = SimpleNamespace(name="rx")
    target_config = SimpleNamespace(name="target")
    simulation_config = MagicMock()
    simulation_config.num_steps = 2
    simulation_config.duration = 1.0
    simulation_config.get_quality_profile.return_value = {"max_depth": 1}

    scene = object()
    tx_obj = object()
    rx_obj = object()
    target_manager = object()
    target_obj = object()
    scene_service = MagicMock()
    scene_service.build_scene.return_value = (
        scene,
        [tx_obj],
        [rx_obj],
        [target_manager],
        [target_obj],
    )
    scene_service_cls = MagicMock(return_value=scene_service)
    monkeypatch.setattr(on_demand, "SceneService", scene_service_cls)

    actor_state_manager = MagicMock()
    actor_state_manager.prepare_cached.return_value = ActorStateCache(
        tx_positions="tx_pos",
        rx_positions="rx_pos",
        target_positions="target_pos",
        tx_orientations="tx_ori",
        rx_orientations="rx_ori",
        target_orientations="target_ori",
    )
    actor_state_manager_cls = MagicMock(return_value=actor_state_manager)
    monkeypatch.setattr(on_demand, "ActorStateManager", actor_state_manager_cls)

    path_solver = object()
    monkeypatch.setattr(on_demand, "PathSolver", MagicMock(return_value=path_solver))

    result = on_demand.build_on_demand_objects(
        [tx_config],
        [rx_config],
        [target_config],
        simulation_config,
        raytracing_settings={"samples_per_src": 12},
    )

    scene_service_cls.assert_called_once_with(simulation_config)
    scene_service.build_scene.assert_called_once_with([tx_config], [rx_config], [target_config])
    actor_state_manager_cls.assert_called_once_with(
        [tx_config],
        [rx_config],
        [target_manager],
        2,
        1.0,
        "step",
    )
    assert result.scene is scene
    assert result.tx_list == [tx_obj]
    assert result.rx_list == [rx_obj]
    assert result.target_managers == [target_manager]
    assert result.settings == {"max_depth": 1, "samples_per_src": 12}
    assert result.path_solver is path_solver
