"""
Pytest configuration for visualizer tests.

This module provides fixtures and configuration for testing the visualizer
in headless mode without requiring a display.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Set headless mode for Qt before importing Qt modules
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
sys.path = [path for path in sys.path if path != PROJECT_ROOT_STR]
sys.path.insert(0, PROJECT_ROOT_STR)


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "headless: mark test as runnable in headless mode")
    config.addinivalue_line("markers", "display: mark test as requiring a display")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# ============================================================================
# Mock Fixtures for Open3D
# ============================================================================


@pytest.fixture
def mock_open3d_geometry(monkeypatch):
    """Mock Open3D geometry classes for unit testing."""

    class MockPointCloud:
        def __init__(self):
            self.points = None
            self.colors = None

        def __len__(self):
            return len(self.points) if self.points is not None else 0

    class MockLineSet:
        def __init__(self):
            self.points = None
            self.lines = None
            self.colors = None

        def __len__(self):
            return len(self.lines) if self.lines is not None else 0

    class MockTriangleMesh:
        def __init__(self):
            self.vertices = None
            self.triangles = None
            self.vertex_colors = None

        @staticmethod
        def create_sphere(radius=1.0):
            mesh = MockTriangleMesh()
            mesh.vertices = np.array([[0, 0, 0]])
            mesh.triangles = np.array([[0, 0, 0]])
            return mesh

    # Mock the geometry module
    try:
        import open3d as o3d

        monkeypatch.setattr(o3d.geometry, "PointCloud", MockPointCloud)
        monkeypatch.setattr(o3d.geometry, "LineSet", MockLineSet)
        monkeypatch.setattr(o3d.geometry, "TriangleMesh", MockTriangleMesh)
    except ImportError:
        pass  # Open3D not installed, tests will skip

    return SimpleNamespace(
        PointCloud=MockPointCloud,
        LineSet=MockLineSet,
        TriangleMesh=MockTriangleMesh,
    )


# ============================================================================
# Common Test Fixtures
# ============================================================================


@pytest.fixture
def sample_app_state():
    """Provide a sample AppState for testing."""
    from visualizer.src.state import create_initial_state

    return create_initial_state(
        step=0,
        selected_tx=0,
        selected_rx=0,
        mpc_layer_enabled=True,
        show_mpc_paths=True,
        show_mpc_bounce_points=True,
        mpc_allowed_orders=frozenset([0, 1, 2]),
        mpc_allowed_types=frozenset([1, 2, 4, 8]),
        color_mode="reflection_order",
        show_labels=False,
        sync_target_position=False,
    )


@pytest.fixture
def sample_mpc_data():
    """Provide sample MPC data for testing."""
    n_mpcs = 100

    return {
        "tau": np.random.rand(n_mpcs) * 1e-6,  # Delays
        "theta_t": np.random.rand(n_mpcs) * np.pi,  # TX elevation
        "phi_t": np.random.rand(n_mpcs) * 2 * np.pi,  # TX azimuth
        "theta_r": np.random.rand(n_mpcs) * np.pi,  # RX elevation
        "phi_r": np.random.rand(n_mpcs) * 2 * np.pi,  # RX azimuth
        "a": np.random.randn(n_mpcs, 2, 2)
        + 1j * np.random.randn(n_mpcs, 2, 2),  # Channel coefficients
        "types": np.random.choice([1, 2, 4, 8], size=n_mpcs),  # Interaction types
        "num_paths": np.random.randint(1, 5, size=n_mpcs),  # Path lengths
    }


@pytest.fixture
def sample_frame_data(sample_mpc_data):
    """Provide complete sample frame data."""
    return {
        "mpc_data": sample_mpc_data,
        "tx_positions": np.array([[0.0, 0.0, 1.5]]),
        "rx_positions": np.array([[10.0, 0.0, 1.5]]),
        "tx_orientations": np.array([[0.0, 0.0, 0.0]]),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]]),
        "timestamp": 0.0,
    }


@pytest.fixture
def temp_test_dir(tmp_path):
    """Provide a temporary directory for test files."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


# ============================================================================
# Headless Testing Utilities
# ============================================================================


def is_headless():
    """Check if running in headless mode."""
    return os.environ.get("QT_QPA_PLATFORM") == "offscreen"


@pytest.fixture
def skip_if_no_display():
    """Skip test if no display is available."""
    if is_headless():
        pytest.skip("Test requires display, running in headless mode")


# ============================================================================
# Performance Testing Utilities
# ============================================================================


@pytest.fixture
def benchmark_timer():
    """Provide a simple benchmark timer."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start_time

    return Timer


# ============================================================================
# NodeService Label Test Double
# ============================================================================


@pytest.fixture
def mock_node_service_label(monkeypatch):
    """Use a lightweight label for NodeService legend construction."""
    from unittest.mock import Mock

    from visualizer.src.services import node_service as node_service_module

    class MockQLabel(Mock):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._text = args[0] if args else ""
            self._style = ""

        def setText(self, text):
            self._text = text

        def text(self):
            return self._text

        def setStyleSheet(self, style):
            self._style = style

        def styleSheet(self):
            return self._style

        def show(self):
            pass

        def hide(self):
            pass

    monkeypatch.setattr(node_service_module, "QLabel", MockQLabel)
