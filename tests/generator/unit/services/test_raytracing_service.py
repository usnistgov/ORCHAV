from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from generator.core.configuration import SensingConfig, SimulationConfig
from generator.core.exceptions import ConfigurationError
from generator.core.services.raytracing_service import RayTracingService


class TestRayTracingService:
    @pytest.fixture
    def mock_simulation_config(self):
        config = MagicMock(spec=SimulationConfig)
        config.scene_sample_rate_hz = None
        return config

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.compute_ray_tracing_step")
    def test_compute_step(self, mock_compute, mock_path_solver, mock_simulation_config):
        mock_simulation_config.num_steps = 1
        mock_simulation_config.duration = 0.0
        mock_compute.return_value = {}
        service = RayTracingService(mock_simulation_config)
        service.simulation_objects = MagicMock()  # Manually inject prepared object

        result = service.compute_step(0, live_overrides=[])

        mock_compute.assert_called_once_with(
            service.simulation_objects,
            0,
            live_overrides=[],
        )
        assert result == mock_compute.return_value
        assert "scene_time_s" not in result

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
        mock_simulation_config.num_steps = 5
        mock_simulation_config.duration = 2.0

        service = RayTracingService(mock_simulation_config)
        service.simulation_objects = MagicMock()
        service.sensing_processor = MagicMock()
        service.sensing_processor.process.side_effect = lambda *_args, **_kwargs: events.append(
            "sensing"
        )
        service._compute_cir_stack = MagicMock(return_value=None)

        result = service.compute_step(2)

        assert result is frame_data
        mock_freeze.assert_called_once_with(frame_data)
        service.sensing_processor.process.assert_called_once_with(
            service.simulation_objects,
            2,
            paths=frame_data["paths"],
            cir_stack=None,
            source_time_s=1.0,
        )
        assert events == ["sensing", "freeze"]
        assert frame_data["scene_time_s"] == 1.0

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
        mock_simulation_config.num_steps = 2
        mock_simulation_config.duration = 1.0

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
        simulation_config = SimpleNamespace(num_steps=9, duration=4.0)
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
        service.sensing_processor = MagicMock()

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
        assert cached["scene_time_s"] == 3.5
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
    def test_cached_non_sensing_step_does_not_infer_scene_timestamp(
        self,
        mock_apply_target_state,
        mock_path_solver,
    ):
        service = RayTracingService(SimpleNamespace(num_steps=9, duration=4.0))
        actor_state_manager = SimpleNamespace(
            state_at_step=MagicMock(
                return_value=SimpleNamespace(target_positions=[], target_orientations=[])
            ),
            compute_velocities=MagicMock(return_value=SimpleNamespace(target=[])),
        )
        service.simulation_objects = SimpleNamespace(
            actor_state_manager=actor_state_manager,
            target_managers=[],
        )

        cached = service.compute_step_cached(
            7,
            {"frame_idx": 4, "paths": object(), "scene_time_s": 99.0},
        )

        assert "scene_time_s" not in cached
        mock_apply_target_state.assert_called_once()

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.compute_ray_tracing_step")
    def test_sensing_rejects_implicit_one_second_timeline(
        self,
        mock_compute,
        mock_path_solver,
    ):
        simulation_config = SimpleNamespace(num_steps=4, duration=0.0)
        service = RayTracingService(simulation_config)
        service.simulation_objects = MagicMock()
        service.sensing_processor = MagicMock()
        mock_compute.return_value = {"paths": object()}

        with pytest.raises(ValueError, match="Sensing requires finite positive scenario duration"):
            service.compute_step(2)

        service.sensing_processor.process.assert_not_called()

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
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    def test_prepare_simulation_rejects_non_synthetic_array_for_sensing_before_rt_setup(
        self,
        mock_sim_objs_cls,
        mock_path_solver,
    ):
        sim_cfg = SimulationConfig(sensing=SensingConfig(enabled=True))
        service = RayTracingService(sim_cfg)
        scenario = SimpleNamespace(raytracing={"quality": {"custom": {"synthetic_array": False}}})

        with pytest.raises(ConfigurationError, match="requires synthetic_array=true"):
            service.prepare_simulation(
                MagicMock(),
                MagicMock(),
                [],
                [],
                [],
                scenario_configuration=scenario,
                sensing_execution_config=MagicMock(),
            )

        mock_sim_objs_cls.assert_not_called()
        assert service.simulation_objects is None
        assert service.sensing_processor is None

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    def test_prepare_simulation_keeps_non_sensing_non_synthetic_array_supported(
        self,
        mock_sim_objs_cls,
        mock_path_solver,
    ):
        sim_cfg = SimulationConfig(sensing=SensingConfig(enabled=False))
        service = RayTracingService(sim_cfg)
        scene_service = MagicMock()
        actor_state_service = MagicMock()
        scenario = SimpleNamespace(raytracing={"quality": {"custom": {"synthetic_array": False}}})

        service.prepare_simulation(
            scene_service,
            actor_state_service,
            [],
            [],
            [],
            scenario_configuration=scenario,
        )

        assert mock_sim_objs_cls.call_args.kwargs["settings"]["synthetic_array"] is False
        assert service.sensing_processor is None

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    @patch("generator.core.services.raytracing_service._load_sensing_processor_factory")
    def test_prepare_simulation_preserves_explicit_sensing_rng_seed(
        self,
        mock_load_sensing_processor_factory,
        mock_sim_objs_cls,
        mock_path_solver,
    ):
        sim_cfg = SimulationConfig(sensing=SensingConfig(enabled=True, rng_seed=7))
        service = RayTracingService(sim_cfg)
        sensing_execution_config = SimpleNamespace(
            processing=SimpleNamespace(mode="sequential_stft"),
            acquisition=SimpleNamespace(rt_solve_stride_scene_steps=1),
        )
        sensing_processor_factory = mock_load_sensing_processor_factory.return_value

        service.prepare_simulation(
            MagicMock(),
            MagicMock(),
            [],
            [],
            [],
            sensing_execution_config=sensing_execution_config,
        )

        assert sim_cfg.get_quality_profile()["seed"] == 42
        assert sim_cfg.sensing.rng_seed == 7
        assert sensing_processor_factory.call_args.args[0].rng_seed == 7

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    @patch("generator.core.services.raytracing_service._load_sensing_processor_factory")
    def test_prepare_simulation_inherits_rt_seed_when_sensing_seed_is_absent(
        self,
        mock_load_sensing_processor_factory,
        mock_sim_objs_cls,
        mock_path_solver,
    ):
        sim_cfg = SimulationConfig(sensing=SensingConfig(enabled=True))
        service = RayTracingService(sim_cfg)
        sensing_execution_config = SimpleNamespace(
            processing=SimpleNamespace(mode="sequential_stft"),
            acquisition=SimpleNamespace(rt_solve_stride_scene_steps=1),
        )
        sensing_processor_factory = mock_load_sensing_processor_factory.return_value

        service.prepare_simulation(
            MagicMock(),
            MagicMock(),
            [],
            [],
            [],
            scenario_configuration=SimpleNamespace(
                raytracing={"quality": {"custom": {"seed": 91}}}
            ),
            sensing_execution_config=sensing_execution_config,
        )

        assert sim_cfg.sensing.rng_seed is None
        assert sensing_processor_factory.call_args.args[0].rng_seed == 91

    @patch("generator.core.services.raytracing_service.PathSolver")
    @patch("generator.core.services.raytracing_service.SimulationObjects")
    @patch("generator.core.services.raytracing_service._load_sensing_processor_factory")
    def test_prepare_simulation_re_resolves_inherited_sensing_seed_for_each_run(
        self,
        mock_load_sensing_processor_factory,
        mock_sim_objs_cls,
        mock_path_solver,
    ):
        sim_cfg = SimulationConfig(sensing=SensingConfig(enabled=True))
        service = RayTracingService(sim_cfg)
        sensing_execution_config = SimpleNamespace(
            processing=SimpleNamespace(mode="sequential_stft"),
            acquisition=SimpleNamespace(rt_solve_stride_scene_steps=1),
        )
        sensing_processor_factory = mock_load_sensing_processor_factory.return_value

        for seed in (91, 37):
            service.prepare_simulation(
                MagicMock(),
                MagicMock(),
                [],
                [],
                [],
                scenario_configuration=SimpleNamespace(
                    raytracing={"quality": {"custom": {"seed": seed}}}
                ),
                sensing_execution_config=sensing_execution_config,
            )

        resolved_configs = [call.args[0] for call in sensing_processor_factory.call_args_list]
        assert [config.rng_seed for config in resolved_configs] == [91, 37]
        assert sim_cfg.sensing.rng_seed is None

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


@patch("generator.core.services.raytracing_service.PathSolver")
@patch("generator.core.services.raytracing_service.freeze_frame_paths")
@patch("generator.core.services.raytracing_service.compute_ray_tracing_step")
def test_live_and_cached_frames_keep_authored_clock(mock_compute, mock_freeze, mock_solver):
    config = SimpleNamespace(num_steps=150, duration=149 / 30, scene_sample_rate_hz=30)
    service = RayTracingService(config)
    state_manager = SimpleNamespace(
        state_at_step=MagicMock(
            return_value=SimpleNamespace(target_positions=[], target_orientations=[])
        ),
        compute_velocities=MagicMock(return_value=SimpleNamespace(target=[])),
    )
    service.simulation_objects = SimpleNamespace(
        actor_state_manager=state_manager, target_managers=[]
    )
    service.sensing_processor = MagicMock()
    service._compute_cir_stack = MagicMock(return_value=None)
    mock_compute.return_value = {"paths": object()}
    live = service.compute_step(23)
    cached = service.compute_step_cached(31, live)
    assert live["scene_time_s"] == 23 / 30
    assert cached["scene_time_s"] == 31 / 30
    assert service.sensing_processor.process.call_args.kwargs["source_time_s"] == 23 / 30
