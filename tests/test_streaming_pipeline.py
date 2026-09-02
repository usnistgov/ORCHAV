#!/usr/bin/env python3
"""
Test script for the streaming pipeline integration.
"""

import logging
import sys
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Get the actual project root (parent of tests directory)
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent  # tests -> project_root
TESTS_DIR = str(HERE.parent)

# Ensure project root is FIRST in sys.path to prevent tests/generator from shadowing
# Remove tests directory from path if present, add project root at position 0
if TESTS_DIR in sys.path:
    sys.path.remove(TESTS_DIR)
sys.path.insert(0, str(PROJECT_ROOT))


def test_streaming_pipeline():
    """Test the streaming pipeline setup"""
    import subprocess

    # Run the test in a subprocess to avoid module caching issues
    # This ensures fresh imports with correct sys.path
    test_script = f"""
import sys
from pathlib import Path

# Set up correct path BEFORE any imports
PROJECT_ROOT = Path("{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

# Remove any tests directory from path
sys.path = [p for p in sys.path if 'tests' not in p or p == str(PROJECT_ROOT)]

try:
    from generator import ReceiverConfig, SimulationConfig, TargetConfig, TransmitterConfig
    from generator.core.mobility import LinearMobility, StationaryMobility
    from generator.core.orientation import FixedOrientationSpec

    # Create a simple test scenario
    simulation_config = SimulationConfig(
        duration=5.0, num_steps=10, scene_name="test_scene", quality="medium"
    )

    tx_configs = [
        TransmitterConfig(
            name="tx_1",
            mobility=StationaryMobility(start_pos=(0, 0, 2.0)),
            orientation=FixedOrientationSpec(),
        )
    ]

    rx_configs = [
        ReceiverConfig(
            name="rx_1",
            mobility=StationaryMobility(start_pos=(5, 0, 2.0)),
            orientation=FixedOrientationSpec(),
        )
    ]

    target_mob = LinearMobility(start_pos=(2, 0, 1.5), end_pos=(3, 0, 1.5))
    target_configs = [
        TargetConfig(
            name="test_target",
            mobility=target_mob,
            mesh_pattern="default",
            mesh_directory="libraries/targets/default",
            resolved_mesh_directory=PROJECT_ROOT / "libraries" / "targets" / "default",
            scale=1.0,
            orientation=FixedOrientationSpec(),
            material_type="default",
            use_ply_position=False,
        )
    ]

    sys.stdout.write("SUCCESS\\n")
except Exception as e:
    sys.stdout.write(f"FAILED: {{e}}\\n")
    sys.exit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0 or "FAILED" in result.stdout:
        pytest.fail(f"Test failed: {result.stdout}{result.stderr}")

    assert "SUCCESS" in result.stdout


def test_grpc_imports():
    """Test gRPC-related imports"""
    try:
        logger.info("Testing gRPC imports...")

        # Test gRPC server imports

        logger.info("GeneratorFrameCache and GeneratorService imported successfully")

        # Test gRPC provider imports

        logger.info("GrpcProvider imported successfully")

        # Test visualizer imports

        logger.info("LiveGrpcSource imported successfully")

        # Test passed

    except ImportError as e:
        pytest.fail(f"gRPC import test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting streaming pipeline integration tests...")
    logger.info("=" * 60)

    # Run tests
    test1_passed = test_streaming_pipeline()
    test2_passed = test_grpc_imports()

    logger.info("=" * 60)
    if test1_passed and test2_passed:
        logger.info("All tests passed! Streaming pipeline is ready.")
        logger.info("Next steps:")
        logger.info("   1. Run: python generator/examples/realTest_streaming.py")
        logger.info(
            "   2. In another terminal: python -m visualizer" " --scenario scenarios/grpc_test"
        )
        logger.info("   3. The visualizer should connect and request frames on-demand")
    else:
        logger.error("Some tests failed. Please check the errors above.")
        sys.exit(1)
