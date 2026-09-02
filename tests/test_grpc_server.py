"""Generator gRPC server startup and ownership regression."""

import logging

import pytest

logger = logging.getLogger(__name__)


def test_grpc_server() -> None:
    """Start and synchronously retire a background generator server."""

    logger.info("Testing gRPC server startup...")

    from generator.io.grpc.live_server import (
        GeneratorFrameCache,
        GeneratorService,
        run_generator_server,
    )

    frame_cache = GeneratorFrameCache()
    generator_config = {"test": True}
    service = GeneratorService(frame_cache, generator_config)
    service.close()
    logger.info("gRPC service created successfully")

    server = None
    generator_service = None
    try:
        server, generator_service, returned_cache = run_generator_server(
            0,
            generator_config,
            frame_cache,
            start_in_background=True,
        )
        assert returned_cache is frame_cache
        logger.info("gRPC server started successfully")
    except PermissionError as exc:
        pytest.skip(f"Socket creation is unavailable in this environment: {exc}")
    except OSError as exc:
        if exc.errno in (1, 13):
            pytest.skip(f"Socket creation is unavailable in this environment: {exc}")
        raise
    finally:
        if server is not None:
            server.stop(0).wait(timeout=5.0)
        if generator_service is not None:
            generator_service.close()
