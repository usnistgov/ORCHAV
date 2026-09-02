"""Regression tests for the optional gRPC/protobuf import boundary."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_with_blocked_imports(code: str, *blocked_modules: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter that rejects selected imports."""
    bootstrap = f"""
import builtins

blocked_modules = {json.dumps(blocked_modules)}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == module or name.startswith(module + ".") for module in blocked_modules):
        exc = ModuleNotFoundError(f"No module named '{{name}}'")
        exc.name = name
        raise exc
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
{textwrap.dedent(code)}
"""
    return subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_missing_google_namespace_is_recognized_as_missing_protobuf() -> None:
    from generator.core.pipeline.dispatch import _is_optional_grpc_import_error as generator_check
    from visualizer.src.io.frame_sources import _is_optional_grpc_import_error as visualizer_check

    missing_google = ModuleNotFoundError("No module named 'google'")
    missing_google.name = "google"

    assert generator_check(missing_google)
    assert visualizer_check(missing_google)


def test_file_mode_modules_import_without_grpc_or_protobuf(tmp_path: Path) -> None:
    result = _run_with_blocked_imports(
        f"""
import os
import sys

import shared.protos
from generator.core.pipeline.dispatch import perform_pipeline
from shared.cli.inspect import main as inspect_main
from shared.cli.validate import main as validate_main
from visualizer.src.io.frame_sources import FileSource, make_frame_source

os.environ["MPLCONFIGDIR"] = {str(tmp_path / "mplconfig")!r}
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import visualizer.visualizer

assert perform_pipeline is not None
assert validate_main is not None
assert inspect_main is not None
assert FileSource is not None
assert make_frame_source is not None
assert "generator.core.pipeline.streaming" not in sys.modules
assert "shared.frames.remote_hdf5" not in sys.modules
assert "shared.protos.visualizer_pb2" not in sys.modules
assert "shared.protos.visualizer_pb2_grpc" not in sys.modules
assert "visualizer.src.io.grpc_provider" not in sys.modules
""",
        "grpc",
        "google.protobuf",
    )

    _assert_success(result)


def test_message_module_import_does_not_require_grpc() -> None:
    result = _run_with_blocked_imports(
        """
from shared.protos import visualizer_pb2

assert visualizer_pb2.DESCRIPTOR is not None
""",
        "grpc",
    )

    _assert_success(result)


def test_generator_streaming_reports_missing_transport_extra() -> None:
    result = _run_with_blocked_imports(
        """
from types import SimpleNamespace

from generator.core.pipeline.dispatch import perform_pipeline

simulation = SimpleNamespace(debug_level="WARNING", output_mode="streaming", start_step=0)
try:
    perform_pipeline(
        tx_configs=[],
        rx_configs=[],
        target_configs=[],
        simulation_config=simulation,
        configure_logging_enabled=False,
        blocking=False,
    )
except RuntimeError as exc:
    assert str(exc) == (
        "Generator live streaming requires the optional gRPC transport; "
        'run python -m pip install -e ".[grpc]"'
    )
else:
    raise AssertionError("streaming unexpectedly opened without the transport runtime")
""",
        "grpc",
        "google.protobuf",
    )

    _assert_success(result)


def test_visualizer_transport_sources_report_missing_extra() -> None:
    result = _run_with_blocked_imports(
        """
from visualizer.src.io.frame_sources import LiveGrpcSource, RemoteHdf5Source

for source, feature in (
    (LiveGrpcSource("localhost:50051"), "Live visualizer playback"),
    (RemoteHdf5Source("localhost:50052"), "Remote HDF5 playback"),
):
    try:
        source.open()
    except RuntimeError as exc:
        assert str(exc) == (
            f"{feature} requires the optional gRPC transport; "
            'run python -m pip install -e ".[grpc]"'
        )
    else:
        raise AssertionError(f"{feature} unexpectedly opened without the transport runtime")
""",
        "grpc",
        "google.protobuf",
    )

    _assert_success(result)


def test_unrelated_missing_imports_are_not_translated() -> None:
    result = _run_with_blocked_imports("""
import sys

from generator.core.pipeline import dispatch
from visualizer.src.io.frame_sources import RemoteHdf5Source

sys.modules["generator.core.pipeline.streaming"] = None
try:
    dispatch._load_streaming_pipeline()
except ModuleNotFoundError as exc:
    assert exc.name == "generator.core.pipeline.streaming"
else:
    raise AssertionError("unrelated generator import failure was hidden")

sys.modules["shared.frames.remote_hdf5"] = None
try:
    RemoteHdf5Source("localhost:50052").open()
except ModuleNotFoundError as exc:
    assert exc.name == "shared.frames.remote_hdf5"
else:
    raise AssertionError("unrelated visualizer import failure was hidden")
""")

    _assert_success(result)
