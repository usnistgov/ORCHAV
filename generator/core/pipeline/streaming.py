"""Streaming pipeline setup for on-demand gRPC frame generation.

Streaming uses the same service boundaries as file output: scene construction,
actor-state preparation, and ray-tracing preparation happen up front. The
difference is ownership of the frame loop. Instead of writing HDF5 frames here,
this module starts the live gRPC server and hands it the prepared services so
requested frames can be computed on demand.
"""

from __future__ import annotations

import sys
from typing import Any

from shared.grpc_transport import DEFAULT_GRPC_BIND_HOST, format_grpc_endpoint
from shared.logging import get_logger

from ...io.grpc.live_server import GeneratorFrameCache, run_generator_server
from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from ..services.actor_state_service import ActorStateService
from ..services.raytracing_service import RayTracingService
from ..services.scene_service import SceneService
from .context import PipelineContext
from .handles import StreamingHandle

logger = get_logger(__name__)

_DEFAULT_LIVE_GRPC_PORT = 50051


def _resolve_streaming_endpoint(
    simulation_config: SimulationConfig,
    *,
    grpc_port: int | None,
    grpc_bind_host: str | None,
) -> tuple[int, str, str]:
    """Resolve listener settings and the endpoint advertised to visualizers."""
    grpc_config = simulation_config.grpc_config or {}
    resolved_port = int(
        grpc_port if grpc_port is not None else grpc_config.get("port", _DEFAULT_LIVE_GRPC_PORT)
    )
    if not 1 <= resolved_port <= 65535:
        raise ValueError("live gRPC port must be between 1 and 65535")
    resolved_bind_host = str(grpc_bind_host or DEFAULT_GRPC_BIND_HOST).strip()
    if not resolved_bind_host:
        raise ValueError("gRPC bind host must not be empty")

    advertised_host = str(grpc_config.get("advertised_host", "localhost")).strip()
    advertised_address = format_grpc_endpoint(advertised_host, resolved_port)
    return resolved_port, resolved_bind_host, f"grpc://{advertised_address}"


def perform_pipeline_streaming(
    tx_configs: list[TransmitterConfig],
    rx_configs: list[ReceiverConfig],
    target_configs: list,
    simulation_config: SimulationConfig,
    scenario_configuration=None,
    grpc_port: int | None = None,
    grpc_bind_host: str | None = None,
) -> StreamingHandle:
    """Prepare a scenario and start the gRPC-backed streaming generator."""
    logger.info("Setting up streaming pipeline for frame generation with buffering")
    resolved_port, resolved_bind_host, advertised_endpoint = _resolve_streaming_endpoint(
        simulation_config,
        grpc_port=grpc_port,
        grpc_bind_host=grpc_bind_host,
    )

    # Streaming returns before the server stops, so the context cannot be a
    # normal ``with`` block. StreamingHandle closes it when the server is done.
    ctx = PipelineContext(simulation_config)
    ctx.__enter__()
    try:
        actor_state_service = ctx.get_service(ActorStateService)
        actor_state_service.normalize_scene_steps(tx_configs, rx_configs, target_configs)

        frame_cache = GeneratorFrameCache()

        generator_config: dict[str, Any] = {
            "data_mode": "live_grpc",
            "motion_mode": "step",
            "num_steps": simulation_config.num_steps,
            "duration": simulation_config.duration,
            "output_mode": "grpc",
            "enabled_patterns": ["mobility", "orientation"],
            "tx_configs": tx_configs,
            "rx_configs": rx_configs,
            "target_configs": target_configs,
            "simulation_config": simulation_config,
            "scenario_configuration": scenario_configuration,
        }

        scene_service = ctx.get_service(SceneService)
        raytracing_service = ctx.get_service(RayTracingService)

        scene_service.build_scene(tx_configs, rx_configs, target_configs)

        # gRPC frame requests are step-driven, but actor state is prepared
        # upfront so on-demand lookup is deterministic and cheap.
        _actor_state_manager, actor_state_cache = actor_state_service.prepare_actor_state(
            tx_configs, rx_configs, scene_service.target_managers, motion_mode="cached"
        )

        raytracing_service.prepare_simulation(
            scene_service,
            actor_state_service,
            tx_configs,
            rx_configs,
            target_configs,
            scenario_configuration=scenario_configuration,
            motion_mode="cached",
            actor_state_cache=actor_state_cache,
        )

        logger.info("Streaming context prepared using services")

        # The live server is the frame-loop owner in streaming mode. It receives
        # service instances instead of rebuilding scene/timeline state itself.
        services: dict[str, Any] = {
            "scene_service": scene_service,
            "actor_state_service": actor_state_service,
            "raytracing_service": raytracing_service,
        }
        generator_config["services"] = services
        generator_config["configs"] = {
            "tx_configs": tx_configs,
            "rx_configs": rx_configs,
            "target_configs": target_configs,
            "simulation_config": simulation_config,
        }

        logger.info(
            "Starting gRPC server on %s...",
            format_grpc_endpoint(resolved_bind_host, resolved_port),
        )
        logger.info(
            "   Scenario: %s steps, %ss duration",
            simulation_config.num_steps,
            simulation_config.duration,
        )
        logger.info(
            "   TX: %s, RX: %s, Targets: %s",
            len(tx_configs),
            len(rx_configs),
            len(target_configs),
        )
        logger.info("   Ready for on-demand frame generation")

        server, generator_service, returned_cache = run_generator_server(
            resolved_port,
            generator_config,
            frame_cache,
            bind_host=resolved_bind_host,
            start_in_background=True,
        )
        logger.info(
            "gRPC server started on %s",
            format_grpc_endpoint(resolved_bind_host, resolved_port),
        )
        logger.info("   Visualizers can connect to: %s", advertised_endpoint)
        logger.info("   Frames will be computed on-demand when requested")

        return StreamingHandle(
            server=server,
            generator_service=generator_service,
            frame_cache=returned_cache,
            services=services,
            pipeline_context=ctx,
        )
    except Exception:  # noqa: BLE001 - ensure context cleanup before re-raising.
        ctx.__exit__(*sys.exc_info())
        raise
