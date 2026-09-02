"""Regression tests for the current-renderer benchmark harness."""

from __future__ import annotations

import sys

import pytest

from scripts.benchmarks import run_current_visualizer_benchmark_pack as benchmark_pack


def test_default_chunk_size_is_accepted_by_synthetic_generator(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_current_visualizer_benchmark_pack.py"])

    assert benchmark_pack.parse_args().chunk_size == 50


def test_default_renderer_set_is_pygfx_only_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_current_visualizer_benchmark_pack.py"])
    monkeypatch.setattr(benchmark_pack.platform, "system", lambda: "Darwin")

    assert benchmark_pack.parse_args().renderers == ["pygfx"]


def test_explicit_open3d_renderer_is_rejected_on_macos() -> None:
    with pytest.raises(SystemExit, match="not supported on macOS.*--renderers pygfx"):
        benchmark_pack._validate_platform_renderers(["pygfx", "open3d"], "Darwin")


def test_open3d_renderer_remains_available_on_windows_and_linux() -> None:
    for platform_name in ("Windows", "Linux"):
        benchmark_pack._validate_platform_renderers(["pygfx", "open3d"], platform_name)


def test_nonpositive_chunk_size_fails_before_benchmarking(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_current_visualizer_benchmark_pack.py", "--chunk-size", "0"],
    )

    with pytest.raises(SystemExit, match="--chunk-size must be >= 1"):
        benchmark_pack.main()


def test_benchmark_command_requests_a_clean_launch(tmp_path) -> None:
    command = benchmark_pack._benchmark_command(
        "pygfx",
        tmp_path / "scenario",
        tmp_path / "benchmark.json",
        regime="cold",
        present_mode="request",
        timed_frames=1,
        warmup_frames=0,
        max_performance=False,
    )

    assert "--no-resume" in command
    assert "--no-session" not in command
    assert "--benchmark-previsit-all-frames" not in command
    present_mode_index = command.index("--benchmark-present-mode")
    assert command[present_mode_index + 1] == "request"


def test_warm_benchmark_command_previsits_and_forwards_performance_mode(tmp_path) -> None:
    command = benchmark_pack._benchmark_command(
        "open3d",
        tmp_path / "scenario",
        tmp_path / "benchmark.json",
        regime="warm",
        present_mode="blocking",
        timed_frames=12,
        warmup_frames=3,
        max_performance=True,
    )

    frames_index = command.index("--benchmark")
    warmup_index = command.index("--benchmark-warmup")
    present_mode_index = command.index("--benchmark-present-mode")
    assert command[frames_index + 1] == "15"
    assert command[warmup_index + 1] == "3"
    assert command[present_mode_index + 1] == "blocking"
    assert "--benchmark-previsit-all-frames" in command
    assert "--max-performance" in command


def test_posix_process_inventory_treats_no_matches_as_captured(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_pack,
        "_capture",
        lambda _command: {"returncode": 1, "stdout": "", "stderr": ""},
    )

    assert benchmark_pack._capture_visualizer_processes("Linux") == {
        "status": "captured",
        "method": "pgrep",
        "processes": [],
    }


@pytest.mark.parametrize(
    ("platform_name", "expected_flags"),
    [("Darwin", "-fl"), ("Linux", "-af")],
)
def test_posix_process_inventory_builds_platform_specific_pgrep_command(
    platform_name, expected_flags
) -> None:
    assert benchmark_pack._pgrep_visualizer_process_command(platform_name) == [
        "pgrep",
        expected_flags,
        "python -m visualizer|run_current_visualizer_benchmark_pack.py",
    ]


def test_windows_process_inventory_uses_cim_json(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_pack.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(
        benchmark_pack,
        "_capture",
        lambda _command: {
            "returncode": 0,
            "stdout": '[{"pid":42,"name":"python.exe","command_line":"python -m visualizer"}]',
            "stderr": "",
        },
    )

    inventory = benchmark_pack._capture_visualizer_processes("Windows")

    assert inventory["status"] == "captured"
    assert inventory["method"] == "powershell_cim"
    assert inventory["processes"][0]["pid"] == 42


def test_windows_process_inventory_reports_partial_fallback(monkeypatch) -> None:
    results = iter(
        [
            {"returncode": 1, "stdout": "", "stderr": "CIM unavailable"},
            {
                "returncode": 0,
                "stdout": '[{"pid":7,"name":"python"}]',
                "stderr": "",
            },
        ]
    )
    monkeypatch.setattr(benchmark_pack.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(benchmark_pack, "_capture", lambda _command: next(results))

    inventory = benchmark_pack._capture_visualizer_processes("Windows")

    assert inventory["status"] == "partial"
    assert inventory["method"] == "powershell_get_process"
    assert inventory["processes"] == [{"pid": 7, "name": "python"}]


def test_windows_process_inventory_reports_unavailable_without_powershell(
    monkeypatch,
) -> None:
    monkeypatch.setattr(benchmark_pack.shutil, "which", lambda _name: None)

    inventory = benchmark_pack._capture_visualizer_processes("Windows")

    assert inventory["status"] == "unavailable"
    assert inventory["processes"] == []
