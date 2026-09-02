#!/usr/bin/env python3
"""
Test gRPC server with test frame data
"""

import logging
import sys
from pathlib import Path

import pytest

# Get the actual project root (parent of tests directory)
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent  # tests -> project_root
TESTS_DIR = str(HERE.parent)

# Ensure project root is FIRST in sys.path to prevent tests/generator from shadowing
if TESTS_DIR in sys.path:
    sys.path.remove(TESTS_DIR)
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.optional_socket


def test_grpc_with_data(grpc_test_server):
    """Test gRPC connection with frame data verification (fast test, no frame loading)"""

    try:
        logger.info("Testing gRPC connection with frame data...")

        # Test LiveGrpcSource
        from visualizer.src.io.frame_sources import LiveGrpcSource

        logger.info("Creating LiveGrpcSource...")
        source = LiveGrpcSource(grpc_test_server)

        logger.info("Opening connection...")
        source.open()

        # Test listing frames (fast operation)
        frames = source.list_frames()
        logger.info(f"Available frames: {len(frames)} frames")
        assert len(frames) > 0, "Expected at least one frame in test server"

        # Verify has_frame for first frame
        if frames:
            frame_idx = frames[0]
            has_frame = source.has_frame(frame_idx)
            logger.info(f"Frame {frame_idx} available: {has_frame}")
            assert has_frame, f"Expected frame {frame_idx} to be available"

        # Cleanup
        source.close()
        logger.info("✅ Connection test with data completed successfully")
        # Test passed

    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Connection test failed: {e}")


if __name__ == "__main__":
    logger.info("Starting gRPC connection test with data...")
    logger.info("Make sure the gRPC server is running on localhost:50051")

    success = test_grpc_with_data()
    sys.exit(0 if success else 1)
