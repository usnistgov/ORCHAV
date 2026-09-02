from unittest.mock import MagicMock, patch

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
