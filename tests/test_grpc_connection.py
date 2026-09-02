#!/usr/bin/env python3
"""
Test gRPC connection between visualizer and generator
"""

import logging
import sys
import time
from pathlib import Path

import grpc
import pytest

from shared.protos import visualizer_pb2, visualizer_pb2_grpc
from shared.scenarios.paths import create_script_path_policy
from visualizer.src.io.grpc_provider import GrpcProvider, LiveGrpcControllerBusyError

# Create path policy for this script
HERE = Path(__file__)
POLICY = create_script_path_policy(HERE)
PROJECT_ROOT = POLICY.project_root

# Add project root to Python path for module imports
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.optional_socket


def _wait_for_server_streaming(stub, expected: bool, *, timeout_s: float = 3.0) -> None:
    """Wait until the real test server reports the expected controller state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = stub.GetGeneratorStatus(
            visualizer_pb2.GetGeneratorStatusRequest(),
            timeout=1.0,
        )
        if bool(response.is_streaming) is expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Server streaming state did not become {expected}")


def test_grpc_connection(grpc_test_server):
    """Test gRPC connection and frame listing (fast test, no frame loading)"""

    try:
        logger.info("Testing gRPC connection...")

        # Test LiveGrpcSource
        from visualizer.src.io.frame_sources import LiveGrpcSource

        logger.info(f"Creating LiveGrpcSource to {grpc_test_server}...")
        source = LiveGrpcSource(grpc_test_server)

        logger.info("Opening connection...")
        source.open()

        # Test listing frames (fast operation)
        frames = source.list_frames()
        logger.info(f"Available frames: {len(frames)} frames")
        assert len(frames) > 0, "Expected at least one frame in test server"

        # Verify frame range
        if frames:
            logger.info(f"Frame range: {min(frames)} - {max(frames)}")

        # Cleanup
        source.close()
        logger.info("✅ Connection test completed successfully")
        # Test passed

    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Connection test failed: {e}")


def test_busy_live_controller_rejects_then_accepts_after_release(grpc_test_server):
    """Reject a second real client, then admit it after the controller closes."""
    address = grpc_test_server.removeprefix("grpc://")
    status_channel = grpc.insecure_channel(address)
    status_stub = visualizer_pb2_grpc.GeneratorServiceStub(status_channel)
    first = GrpcProvider(grpc_test_server, buffer_size=2)
    second = GrpcProvider(grpc_test_server, buffer_size=2)

    try:
        first.open()
        _wait_for_server_streaming(status_stub, True)

        with pytest.raises(LiveGrpcControllerBusyError, match="active controlling visualizer"):
            second.open()
        assert second._stream_thread is None

        first.close()
        _wait_for_server_streaming(status_stub, False)

        second.open()
        _wait_for_server_streaming(status_stub, True)
    finally:
        first.close()
        second.close()
        status_channel.close()


if __name__ == "__main__":
    logger.info("Starting gRPC connection test...")
    logger.info("Make sure the gRPC server is running on localhost:50051")

    success = test_grpc_connection()
    sys.exit(0 if success else 1)
