import logging
from unittest.mock import ANY, MagicMock, patch

import pytest

from generator.core.configuration import SimulationConfig
from generator.core.pipeline import StreamingHandle, perform_pipeline_streaming
from generator.core.pipeline.streaming import _resolve_streaming_endpoint

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_simulation_config():
    cfg = MagicMock(spec=SimulationConfig)
    cfg.grpc_config = {
        "endpoint": "grpc://scenario-host:60000",
        "advertised_host": "scenario-host",
        "port": 60000,
    }
    cfg.get_quality_profile.return_value = {}
    cfg.output_mode = "streaming"
    return cfg


@patch("generator.core.pipeline.streaming.SceneService")
@patch("generator.core.pipeline.streaming.ActorStateService")
@patch("generator.core.pipeline.streaming.RayTracingService")
@patch("generator.core.pipeline.streaming.run_generator_server")
@patch("generator.core.pipeline.streaming.GeneratorFrameCache")
def test_perform_pipeline_streaming_orchestration(
    mock_frame_cache_cls,
    mock_run_server,
    mock_rt_cls,
    mock_actor_state_cls,
    mock_scene_cls,
    mock_simulation_config,
):
    # Setup mocks
    mock_scene_service = mock_scene_cls.return_value
    mock_actor_state_service = mock_actor_state_cls.return_value
    mock_rt_service = mock_rt_cls.return_value

    # Set __name__ for PipelineContext.get_service lookup
    mock_scene_cls.__name__ = "SceneService"
    mock_actor_state_cls.__name__ = "ActorStateService"
    mock_rt_cls.__name__ = "RayTracingService"

    # Setup returns
    mock_scene_service.build_scene.return_value = (MagicMock(), [], [], [], [])
    mock_actor_state_service.prepare_actor_state.return_value = (
        MagicMock(),
        MagicMock(),
    )
    mock_server = MagicMock()
    mock_generator_service = MagicMock()
    mock_returned_cache = MagicMock()
    mock_run_server.return_value = (mock_server, mock_generator_service, mock_returned_cache)

    # Setup input
    tx_configs = [MagicMock()]
    rx_configs = [MagicMock()]
    target_configs = []

    # Scenario config
    scenario_cfg = MagicMock()
    scenario_cfg.raytracing = {"enabled": True}

    # Execute
    handle = perform_pipeline_streaming(
        tx_configs,
        rx_configs,
        target_configs,
        mock_simulation_config,
        scenario_configuration=scenario_cfg,
    )

    # Verification

    # 1. Services Instantiated
    mock_scene_cls.assert_called_with(mock_simulation_config)
    mock_actor_state_cls.assert_called_with(mock_simulation_config)
    mock_rt_cls.assert_called_with(mock_simulation_config)

    # 2. Scene Built
    mock_scene_service.build_scene.assert_called_with(tx_configs, rx_configs, target_configs)

    # 3. Actor state prepared in cached mode
    mock_actor_state_service.prepare_actor_state.assert_called_with(
        tx_configs, rx_configs, ANY, motion_mode="cached"
    )

    # 4. RayTracing Prepared
    mock_rt_service.prepare_simulation.assert_called()
    call_args = mock_rt_service.prepare_simulation.call_args
    assert call_args.kwargs["motion_mode"] == "cached"

    # 5. Server started with correct config and returns lifecycle handle
    assert isinstance(handle, StreamingHandle)
    assert handle.server is mock_server
    assert handle.generator_service is mock_generator_service
    assert handle.frame_cache is mock_returned_cache
    assert mock_run_server.called
    call_args = mock_run_server.call_args
    # call args: (grpc_port, generator_config, frame_cache)
    assert call_args[0][0] == 60000  # scenario endpoint port
    generator_config = call_args[0][1]  # second arg is generator_config

    assert "services" in generator_config
    assert generator_config["services"]["scene_service"] == mock_scene_service
    assert generator_config["services"]["actor_state_service"] == mock_actor_state_service
    assert generator_config["services"]["raytracing_service"] == mock_rt_service

    assert "configs" in generator_config
    assert generator_config["configs"]["tx_configs"] == tx_configs
    assert generator_config["configs"]["rx_configs"] == rx_configs
    assert call_args.kwargs["start_in_background"] is True
    assert call_args.kwargs["bind_host"] == "127.0.0.1"
    handle.close()

    logger.info("perform_pipeline_streaming orchestration verified.")


def test_streaming_endpoint_explicit_overrides_win_without_changing_advertised_host(
    mock_simulation_config,
) -> None:
    port, bind_host, advertised_endpoint = _resolve_streaming_endpoint(
        mock_simulation_config,
        grpc_port=61000,
        grpc_bind_host="0.0.0.0",
    )

    assert port == 61000
    assert bind_host == "0.0.0.0"
    assert advertised_endpoint == "grpc://scenario-host:61000"


def test_streaming_endpoint_rejects_unadvertisable_ephemeral_port(
    mock_simulation_config,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 65535"):
        _resolve_streaming_endpoint(
            mock_simulation_config,
            grpc_port=0,
            grpc_bind_host=None,
        )
