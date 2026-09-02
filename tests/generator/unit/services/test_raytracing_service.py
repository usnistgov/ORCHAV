from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from generator.core.configuration import SimulationConfig
from generator.core.services.raytracing_service import RayTracingService


class TestRayTracingService:
    @pytest.fixture
    def mock_simulation_config(self):
        return MagicMock(spec=SimulationConfig)

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.compute_ray_tracing_step")
    def test_compute_step(self, mock_compute, mock_path_solver, mock_simulation_config):
        service = RayTracingService(mock_simulation_config)
        service.simulation_objects = MagicMock()  # Manually inject prepared object

        result = service.compute_step(0, live_overrides=[])

        mock_compute.assert_called_once_with(
            service.simulation_objects,
            0,
            live_overrides=[],
        )
        assert result == mock_compute.return_value

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.freeze_frame_paths")
    @patch("generator.core.services.raytracing_service.compute_ray_tracing_step")
    def test_compute_step_freezes_paths_after_sensing(
        self,
        mock_compute,
        mock_freeze,
        mock_path_solver,
        mock_simulation_config,
    ):
        events = []
        frame_data = {"paths": object()}
        mock_compute.return_value = frame_data
        mock_freeze.side_effect = lambda _frame_data: events.append("freeze")

        service = RayTracingService(mock_simulation_config)
        service.simulation_objects = MagicMock()
        service.sensing_processor = MagicMock()
        service.sensing_processor.process.side_effect = lambda *_args, **_kwargs: events.append(
            "sensing"
        )
        service._compute_cir_stack = MagicMock(return_value=None)

        result = service.compute_step(0)

        assert result is frame_data
        mock_freeze.assert_called_once_with(frame_data)
        assert events == ["sensing", "freeze"]

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.freeze_frame_paths")
    @patch("generator.core.services.raytracing_service.compute_ray_tracing_step")
    def test_compute_step_sensing_failure_returns_none_without_freezing(
        self,
        mock_compute,
        mock_freeze,
        mock_path_solver,
        mock_simulation_config,
    ):
        frame_data = {"paths": object()}
        mock_compute.return_value = frame_data

        service = RayTracingService(mock_simulation_config)
        service.simulation_objects = MagicMock()
        service.sensing_processor = MagicMock()
        service.sensing_processor.process.side_effect = RuntimeError("sensing failed")
        service._compute_cir_stack = MagicMock(return_value=None)

        result = service.compute_step(0)

        assert result is None
        mock_freeze.assert_not_called()

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.apply_target_state_to_scene")
    def test_cached_step_refreshes_only_output_time_target_state(
        self,
        mock_apply_target_state,
        mock_path_solver,
    ):
        simulation_config = SimpleNamespace()
        service = RayTracingService(simulation_config)
        current_target = object()
        target_manager = SimpleNamespace(target_object=current_target)
        actor_state = SimpleNamespace(
            target_positions=[(70.0, 71.0, 72.0)],
            target_orientations=[(90.0, 0.0, -45.0)],
        )
        velocities = SimpleNamespace(target=[(1.0, 2.0, 3.0)])
        actor_state_manager = SimpleNamespace(
            state_at_step=MagicMock(return_value=actor_state),
            compute_velocities=MagicMock(return_value=velocities),
        )
        service.simulation_objects = SimpleNamespace(
            actor_state_manager=actor_state_manager,
            target_managers=[target_manager],
        )

        paths = object()
        tx_snapshot = np.asarray([[1.0, 2.0, 3.0]])
        rx_snapshot = np.asarray([[4.0, 5.0, 6.0]])
        acquisition_target_snapshot = np.asarray([[7.0, 8.0, 9.0]])
        sensing = {"acquisition": True}
        acquisition = {
            "frame_idx": 4,
            "paths": paths,
            "tx_positions_snapshot": tx_snapshot,
            "rx_positions_snapshot": rx_snapshot,
            "target_positions_snapshot": acquisition_target_snapshot,
            "target_objects": [object()],
            "sensing": sensing,
        }

        cached = service.compute_step_cached(7, acquisition)

        actor_state_manager.state_at_step.assert_called_once_with(7)
        actor_state_manager.compute_velocities.assert_called_once_with(7)
        mock_apply_target_state.assert_called_once_with(
            [target_manager],
            [(70.0, 71.0, 72.0)],
            [(90.0, 0.0, -45.0)],
            7,
            step_velocities=[(1.0, 2.0, 3.0)],
            simulation_config=simulation_config,
        )
        assert cached is not acquisition
        assert cached["frame_idx"] == 7
        assert cached["_cached_rt_source_frame_idx"] == 4
        assert cached["paths"] is paths
        assert cached["tx_positions_snapshot"] is tx_snapshot
        assert cached["rx_positions_snapshot"] is rx_snapshot
        assert cached["target_objects"] == [current_target]
        np.testing.assert_allclose(cached["target_positions_snapshot"], [[70.0, 71.0, 72.0]])
        np.testing.assert_allclose(
            cached["target_orientations_snapshot"],
            [[np.pi / 2.0, 0.0, -np.pi / 4.0]],
        )
        np.testing.assert_allclose(cached["target_velocities_snapshot"], [[1.0, 2.0, 3.0]])
        assert cached["sensing"] is None

        assert acquisition["frame_idx"] == 4
        assert "_cached_rt_source_frame_idx" not in acquisition
        assert acquisition["sensing"] is sensing
        assert acquisition["target_positions_snapshot"] is acquisition_target_snapshot

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.apply_target_state_to_scene")
    def test_cached_step_propagates_target_state_failure(
        self,
        mock_apply_target_state,
        mock_path_solver,
    ):
        service = RayTracingService(SimpleNamespace())
        actor_state_manager = SimpleNamespace(
            state_at_step=MagicMock(
                return_value=SimpleNamespace(
                    target_positions=[(1.0, 2.0, 3.0)],
                    target_orientations=[(0.0, 0.0, 0.0)],
                )
            ),
            compute_velocities=MagicMock(return_value=SimpleNamespace(target=[])),
        )
        service.simulation_objects = SimpleNamespace(
            actor_state_manager=actor_state_manager,
            target_managers=[SimpleNamespace(target_object=object())],
        )
        mock_apply_target_state.side_effect = RuntimeError("target update failed")

        with pytest.raises(RuntimeError, match="target update failed"):
            service.compute_step_cached(2, {"frame_idx": 0, "paths": object()})

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    def test_prepare_simulation(self, mock_sim_objs_cls, mock_path_solver, mock_simulation_config):
        service = RayTracingService(mock_simulation_config)

        mock_scene_service = MagicMock()
        mock_actor_state_service = MagicMock()
        mock_simulation_config.get_quality_profile.return_value = {}
        mock_simulation_config.cir_time_steps = 1
        mock_simulation_config.cir_sampling_frequency_hz = None
        mock_simulation_config.mesh_update_interval_s = None

        service.prepare_simulation(
            mock_scene_service, mock_actor_state_service, [], [], [], scenario_configuration=None
        )

        mock_sim_objs_cls.assert_called_once()
        assert service.simulation_objects is not None

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    def test_prepare_simulation_overrides_do_not_mutate_global_presets(
        self,
        mock_sim_objs_cls,
        mock_path_solver,
    ):
        sim_cfg = SimulationConfig(quality="medium")
        baseline = dict(sim_cfg.QUALITY_PRESETS["medium"])
        service = RayTracingService(sim_cfg)

        mock_scene_service = MagicMock()
        mock_scene_service.scene = MagicMock()
        mock_scene_service.tx_list = []
        mock_scene_service.rx_list = []
        mock_scene_service.target_managers = []
        mock_actor_state_service = MagicMock()
        mock_actor_state_service.actor_state_manager = MagicMock()

        scenario_a = SimpleNamespace(raytracing={"quality": {"custom": {"max_depth": 1}}})
        scenario_b = SimpleNamespace(raytracing={"quality": {"custom": {"max_depth": 5}}})

        service.prepare_simulation(
            mock_scene_service,
            mock_actor_state_service,
            [],
            [],
            [],
            scenario_configuration=scenario_a,
        )
        service.prepare_simulation(
            mock_scene_service,
            mock_actor_state_service,
            [],
            [],
            [],
            scenario_configuration=scenario_b,
        )

        first_settings = mock_sim_objs_cls.call_args_list[0].kwargs["settings"]
        second_settings = mock_sim_objs_cls.call_args_list[1].kwargs["settings"]
        assert first_settings["max_depth"] == 1
        assert second_settings["max_depth"] == 5
        assert sim_cfg.QUALITY_PRESETS["medium"] == baseline

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    def test_prepare_simulation_accepts_advanced_diffraction_overrides(
        self,
        mock_sim_objs_cls,
        mock_path_solver,
    ):
        sim_cfg = SimulationConfig(quality="low")
        service = RayTracingService(sim_cfg)

        mock_scene_service = MagicMock()
        mock_scene_service.scene = MagicMock()
        mock_scene_service.tx_list = []
        mock_scene_service.rx_list = []
        mock_scene_service.target_managers = []
        mock_actor_state_service = MagicMock()
        mock_actor_state_service.actor_state_manager = MagicMock()

        scenario = SimpleNamespace(
            raytracing={
                "quality": {
                    "custom": {
                        "diffraction": True,
                        "edge_diffraction": True,
                        "diffraction_lit_region": False,
                    }
                }
            }
        )

        service.prepare_simulation(
            mock_scene_service,
            mock_actor_state_service,
            [],
            [],
            [],
            scenario_configuration=scenario,
        )

        settings = mock_sim_objs_cls.call_args.kwargs["settings"]
        assert settings["diffraction"] is True
        assert settings["edge_diffraction"] is True
        assert settings["diffraction_lit_region"] is False

    @patch("generator.core.services.raytracing_service.PathSolver")
    def test_cleanup_releases_prepared_state(self, mock_path_solver, mock_simulation_config):
        service = RayTracingService(mock_simulation_config)
        service.simulation_objects = MagicMock()
        service.sensing_processor = MagicMock()
        path_solver = service.path_solver

        service.cleanup()

        assert service.simulation_objects is None
        assert service.sensing_processor is None
        assert service.path_solver is path_solver
