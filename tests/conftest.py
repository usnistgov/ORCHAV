"""Root conftest for all tests - sets up PYTHONPATH and common fixtures."""

import importlib
import os
import pathlib
import socket
import sys
import time
from contextlib import closing
from types import SimpleNamespace

import pytest

# Ensure project root on sys.path FIRST
ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

PROJECT_PACKAGES = {"generator", "shared", "visualizer"}
OPTIONAL_DEFAULT_MARKERS = {
    "pygfx_runtime",
    "optional_runtime",
    "optional_socket",
    "private_fixture",
    "soak",
}
OPTIONAL_MODULE_LEVEL_SKIP_FILES = {
    ROOT / "tests/generator/unit/test_mobility_models.py",
    ROOT / "tests/generator/unit/test_orientation_models.py",
    ROOT / "tests/generator/unit/test_orientation_patterns.py",
    ROOT / "tests/generator/unit/test_target_mesh_sequence.py",
}
OPTIONAL_TEST_ENV = "ORCHAV_RUN_OPTIONAL_TESTS"


def _optional_filter_enabled() -> bool:
    return os.environ.get(OPTIONAL_TEST_ENV, "0") != "1"


def _normalize_requested_path(raw_arg: str) -> pathlib.Path | None:
    candidate = raw_arg.split("::", 1)[0]
    if not candidate or candidate.startswith("-"):
        return None
    path = pathlib.Path(candidate)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve()
    except OSError:
        return None


def _is_explicit_file_request(config: pytest.Config, path: pathlib.Path) -> bool:
    requested = getattr(config, "args", ()) or ()
    resolved_path = path.resolve()
    for raw_arg in requested:
        requested_path = _normalize_requested_path(str(raw_arg))
        if (
            requested_path is not None
            and requested_path.is_file()
            and requested_path == resolved_path
        ):
            return True
    return False


def _path_inside_root(path: str) -> bool:
    try:
        pathlib.Path(path).resolve().relative_to(ROOT)
    except (OSError, ValueError):
        return False
    return True


def _editable_finder_points_outside_root(finder: object) -> bool:
    module_name = getattr(finder, "__module__", "")
    if not module_name.startswith("__editable___"):
        return False

    finder_module = sys.modules.get(module_name)
    mapping = getattr(finder_module, "MAPPING", {})
    project_paths = [path for package, path in mapping.items() if package in PROJECT_PACKAGES]
    return bool(project_paths) and any(not _path_inside_root(path) for path in project_paths)


sys.meta_path = [
    finder for finder in sys.meta_path if not _editable_finder_points_outside_root(finder)
]


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool:
    """Keep broad health runs free of module-level optional-runtime skips."""
    path = pathlib.Path(str(collection_path)).resolve()
    optional_files = {p.resolve() for p in OPTIONAL_MODULE_LEVEL_SKIP_FILES}
    if (
        _optional_filter_enabled()
        and path in optional_files
        and not _is_explicit_file_request(config, path)
    ):
        return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect optional/environment tests during broad default health runs.

    Explicit file requests still run the requested tests so focused optional
    diagnostics preserve their existing skip/failure behavior.
    """
    if not _optional_filter_enabled():
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        item_path = pathlib.Path(str(getattr(item, "path", item.fspath))).resolve()
        if _is_explicit_file_request(config, item_path):
            selected.append(item)
            continue
        if any(item.get_closest_marker(marker) is not None for marker in OPTIONAL_DEFAULT_MARKERS):
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


# Headless Qt for visualizer tests
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Re-install the editable finder if it's missing (pytest's assertion rewriting removes it)
try:
    import __editable___orchav_0_1_0_finder

    mapping = getattr(__editable___orchav_0_1_0_finder, "MAPPING", {})
    project_paths = [path for package, path in mapping.items() if package in PROJECT_PACKAGES]
    if project_paths and all(_path_inside_root(path) for path in project_paths):
        __editable___orchav_0_1_0_finder.install()
except ImportError:
    pass  # Not an editable install, that's fine

# Pre-import shared.statistics modules - use try/except for robustness
try:
    for module_name in (
        "shared.statistics",
        "shared.statistics.core",
        "shared.statistics.core.distributions",
        "shared.statistics.core.metrics",
        "shared.statistics.themes",
    ):
        importlib.import_module(module_name)
except ImportError as e:
    # Only warn, don't fail - these are optional for some test subsets
    import warnings

    warnings.warn(f"Could not pre-import shared.statistics: {e}")


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def grpc_test_server():
    """Start a lightweight gRPC generator server with seeded frames for tests."""
    import grpc

    from generator.io.grpc.live_server import (
        GeneratorFrameCache,
        add_test_frame_data,
        run_generator_server,
    )

    try:
        port = _get_free_port()
    except PermissionError as exc:
        pytest.skip(f"Socket creation not permitted in this environment: {exc}")
    except OSError as exc:
        if exc.errno in (1, 13):
            pytest.skip(f"Socket creation not permitted in this environment: {exc}")
        raise
    # Seed cache with synthetic frames to satisfy list_frames/load_frame
    frame_cache = GeneratorFrameCache(max_frames=50, ttl_seconds=60.0, max_size_bytes=0)
    add_test_frame_data(frame_cache, num_frames=20)
    simulation_config = SimpleNamespace(
        num_steps=frame_cache.total_frames,
        duration=float(frame_cache.total_frames),
    )
    simulation_objects = SimpleNamespace(
        simulation_config=simulation_config,
        settings={},
    )
    raytracing_service = SimpleNamespace(simulation_objects=simulation_objects)

    generator_config = {
        "data_mode": "live_grpc",
        "motion_mode": "step",
        "num_steps": frame_cache.total_frames,
        "duration": float(frame_cache.total_frames),
        "output_mode": "grpc",
        "enabled_patterns": ["mobility", "orientation"],
        "services": {"raytracing_service": raytracing_service},
        "configs": {"simulation_config": simulation_config},
    }

    # Start server with start_in_background=True - returns (server, service, cache) directly
    try:
        server, generator_service, _ = run_generator_server(
            port, generator_config, frame_cache, start_in_background=True
        )
    except PermissionError as exc:
        pytest.skip(f"gRPC server cannot start in this environment: {exc}")
    except OSError as exc:
        if exc.errno in (1, 13):
            pytest.skip(f"gRPC server cannot start in this environment: {exc}")
        raise

    # Wait for server to be actually ready by trying to connect
    endpoint = f"grpc://localhost:{port}"
    max_retries = 10
    for i in range(max_retries):
        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            grpc.channel_ready_future(channel).result(timeout=1)
            channel.close()
            break
        except Exception:
            time.sleep(0.2)
    else:
        server.stop(0)
        pytest.skip("gRPC server failed to start (environment may block networking)")

    yield endpoint

    # Cleanup
    server.stop(0)
