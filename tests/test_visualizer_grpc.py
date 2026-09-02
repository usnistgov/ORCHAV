#!/usr/bin/env python3
"""
Test visualizer with gRPC scenario
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


def test_visualizer_grpc(grpc_test_server, tmp_path):
    """Test visualizer with gRPC scenario"""

    try:
        logger.info("Testing visualizer with gRPC scenario...")

        # Test scenario loading
        from visualizer.src.io.frame_sources import make_frame_source
        from visualizer.src.io.scenario_config import load_app_config, load_scenario

        # Load app config
        app_config = load_app_config()
        logger.info("✅ App config loaded")

        # Build a minimal gRPC scenario pointing to the test server
        scenario_root = tmp_path / "grpc_scenario"
        scenario_root.mkdir(parents=True, exist_ok=True)
        scenario_yaml = scenario_root / "scenario.yaml"
        scenario_yaml.write_text(f"""
data:
  mode: live_grpc
  live_grpc:
    endpoint: "{grpc_test_server}"
    buffer_size: 10
scene:
  id: box/box.xml
  source: library
schema_version: 2
raytracing:
  enabled: true
  export_path_metrics: true
  quality:
    preset: low
timeline:
  steps: 5
  duration_s: 5.0
        """)

        scenario = load_scenario(scenario_root, app_config)
        logger.info(f"✅ Scenario loaded: {scenario.data_mode} -> {grpc_test_server}")

        # Create frame source
        frame_source = make_frame_source(scenario)
        logger.info(f"✅ Frame source created: {type(frame_source)}...")

        # Test frame source operations (fast - no frame loading)
        logger.info("Testing frame source operations...")

        # List frames (fast operation)
        frames = frame_source.list_frames()
        logger.info(f"Available frames: {len(frames)} frames")
        assert len(frames) > 0, "Expected at least one frame in test server"

        if frames:
            # Verify has_frame (fast operation)
            frame_idx = frames[0]
            has_frame = frame_source.has_frame(frame_idx)
            logger.info(f"Frame {frame_idx} available: {has_frame}")
            assert has_frame, f"Expected frame {frame_idx} to be available"

        # Cleanup
        frame_source.close()
        logger.info("✅ Visualizer gRPC test completed successfully")
        # Test passed

    except Exception as e:
        logger.error(f"❌ Visualizer gRPC test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Visualizer gRPC test failed: {e}")


if __name__ == "__main__":
    logger.info("Starting visualizer gRPC test...")
    logger.info("Make sure the gRPC server is running on localhost:50051")

    success = test_visualizer_grpc()
    sys.exit(0 if success else 1)
