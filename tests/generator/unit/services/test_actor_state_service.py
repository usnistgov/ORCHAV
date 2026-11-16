from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from generator.core.configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from generator.core.scenario_actors.state import ActorStateCache
from generator.core.target import TargetConfig

try:
    from generator.core.services.actor_state_service import ActorStateService

    _TIMELINE_AVAILABLE = True
except (ImportError, AttributeError):
    _TIMELINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TIMELINE_AVAILABLE,
    reason="ActorStateService requires Mitsuba GPU initialization",
)


class TestActorStateService:
    @pytest.fixture
    def mock_simulation_config(self):
        config = MagicMock(spec=SimulationConfig)
        config.num_steps = 10
        config.duration = 1.0
        config.scene_sample_rate_hz = None
        return config

    @patch("generator.core.services.actor_state_service.ActorStateManager")
    def test_normalize_scene_steps_expansion(self, mock_manager_cls, mock_simulation_config):
        del mock_manager_cls
        service = ActorStateService(mock_simulation_config)

        mock_mobility = MagicMock()
        mock_mobility.auto_expand_scene_steps = True
        mock_mobility.total_points = 50

        tx_config = MagicMock(spec=TransmitterConfig)
        tx_config.mobility = mock_mobility

        rx_config = MagicMock(spec=ReceiverConfig)
        rx_config.mobility = None

        service.normalize_scene_steps([tx_config], [rx_config])

        assert mock_simulation_config.num_steps == 50
        assert service.max_steps_expanded is True

    @patch("generator.core.services.actor_state_service.ActorStateManager")
    def test_normalize_no_expansion_needed(self, mock_manager_cls, mock_simulation_config):
        del mock_manager_cls
        service = ActorStateService(mock_simulation_config)
        mock_simulation_config.num_steps = 100

        mock_mobility = MagicMock()
        mock_mobility.auto_expand_scene_steps = True
        mock_mobility.total_points = 50

        tx_config = MagicMock(spec=TransmitterConfig)
        tx_config.mobility = mock_mobility

        service.normalize_scene_steps([tx_config], [])

        assert mock_simulation_config.num_steps == 100
        assert service.max_steps_expanded is False

    @patch("generator.core.services.actor_state_service.ActorStateManager")
    def test_normalize_scene_steps_expands_for_target_mobility(
        self, mock_manager_cls, mock_simulation_config
    ):
        del mock_manager_cls
        service = ActorStateService(mock_simulation_config)

        mock_mobility = MagicMock()
        mock_mobility.auto_expand_scene_steps = True
        mock_mobility.total_points = 75

        target_config = MagicMock(spec=TargetConfig)
        target_config.mobility = mock_mobility

        service.normalize_scene_steps([], [], [target_config])

        assert mock_simulation_config.num_steps == 75
        assert service.max_steps_expanded is True

    @patch("generator.core.services.actor_state_service.ActorStateManager")
    def test_prepare_actor_state(self, mock_manager_cls, mock_simulation_config):
        service = ActorStateService(mock_simulation_config)

        mock_manager = MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_cache = ActorStateCache(
            tx_positions=[],
            rx_positions=[],
            target_positions=[],
            tx_orientations=[],
            rx_orientations=[],
            target_orientations=[],
        )
        mock_manager.prepare_cached.return_value = mock_cache

        manager, cache = service.prepare_actor_state([], [], [], motion_mode="cached")

        call_args = mock_manager_cls.call_args
        assert call_args.args == (
            [],
            [],
            [],
            mock_simulation_config.num_steps,
            mock_simulation_config.duration,
            "cached",
        )
        mock_manager.prepare_cached.assert_called_once()
        assert manager == mock_manager
        assert cache == mock_cache

    def test_cleanup_releases_actor_state_manager(self, mock_simulation_config):
        service = ActorStateService(mock_simulation_config)
        service.actor_state_manager = MagicMock()

        service.cleanup()

        assert service.actor_state_manager is None


def test_initial_application_preserves_prepared_snapshot_and_target_frame_zero(monkeypatch):
    import generator.core.propagation.actor_state_application as application
    from generator.core.scenario_actors.state import ActorStateSnapshot, ActorStateVelocities

    monkeypatch.setattr(application, "point3f", tuple)
    monkeypatch.setattr(
        application, "orientation_to_point3f", lambda value: tuple(np.deg2rad(value))
    )
    config = SimpleNamespace(
        num_steps=3, duration=2.0, cir_time_steps=1, start_step=0, mesh_update_interval_s=None
    )
    service = ActorStateService(config)
    snapshot = ActorStateSnapshot(
        [(2.0, 3.0, 4.0)],
        [(5.0, 6.0, 7.0)],
        [(8.0, 9.0, 10.0)],
        [(-25.0, 0.0, 0.0)],
        [(75.0, 0.0, 0.0)],
        [(40.0, 0.0, 0.0)],
    )
    manager = MagicMock()
    manager.state_at_step.return_value = snapshot
    manager.compute_velocities.return_value = ActorStateVelocities(
        [(1.0, 0.0, 0.0)], [(0.0, 1.0, 0.0)], [(0.0, 0.0, 1.0)]
    )
    service.actor_state_manager = manager
    target = MagicMock()
    target.config = SimpleNamespace(
        name="moving",
        animated_asset=None,
        switch_meshes=True,
        mobility=object(),
        use_ply_position=False,
    )
    scene = SimpleNamespace(
        tx_list=[SimpleNamespace()],
        rx_list=[SimpleNamespace()],
        target_managers=[target],
        target_objects=[object()],
    )
    for _ in range(2):
        service.apply_initial_state(scene)
        assert scene.tx_list[0].position == (2.0, 3.0, 4.0)
        np.testing.assert_allclose(scene.tx_list[0].orientation, np.deg2rad([-25, 0, 0]))
        np.testing.assert_allclose(scene.rx_list[0].orientation, np.deg2rad([75, 0, 0]))
        target.update_mesh_for_frame.assert_called_with(0, expected_call_count=0)
        target.apply_orientation_snapshot.assert_called_with((40.0, 0.0, 0.0))
        target.apply_position_snapshot.assert_called_with((8.0, 9.0, 10.0))
        assert scene.target_objects == [target.target_object]
    assert manager.state_at_step.call_args_list == [((0,),), ((0,),)]
    manager.prepare_cached.assert_not_called()
    assert snapshot.tx_positions == [(2.0, 3.0, 4.0)]
    assert config.num_steps == 3 and config.duration == 2.0


def test_initial_application_requires_preparation():
    service = ActorStateService(SimpleNamespace())
    with pytest.raises(ValueError, match="must be prepared"):
        service.apply_initial_state(SimpleNamespace())


@pytest.mark.parametrize("mode", ["cached", "step"])
def test_service_passes_authored_clock_to_real_state_manager(mode):
    config = SimpleNamespace(num_steps=150, duration=149 / 30, scene_sample_rate_hz=30)
    manager, _ = ActorStateService(config).prepare_actor_state([], [], [], motion_mode=mode)
    assert manager._timeline.timestamps_s == tuple(i / 30 for i in range(150))
