#!/usr/bin/env python3
"""Run current ORCHAV visualizer synthetic-MPC benchmark packs.

This maintainer utility generates synthetic HDF5 frame datasets, runs selected
visualizer renderers against them, and writes benchmark artifacts under a
scratch output root.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualizer.src.benchmarking.regimes import summarize_benchmark_json  # noqa: E402
from visualizer.src.benchmarking.scenario_profile import write_scenario_profile  # noqa: E402

DEFAULT_MPC_COUNTS = (100, 1_000, 10_000, 100_000)
RENDERERS = ("pygfx", "open3d")
PRESENT_MODES = ("request", "blocking")
SYNTHETIC_GENERATOR = PROJECT_ROOT / "scenarios/visualizer/synthetic_mpc_benchmark/generate.py"


class CommandError(RuntimeError):
    """Raised when a benchmark subprocess fails."""

    def __init__(self, cmd: list[str], returncode: int, log_path: Path | None) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.log_path = log_path
        detail = " ".join(shlex.quote(part) for part in cmd)
        if log_path is not None:
            detail = f"{detail} (see {log_path})"
        super().__init__(f"command failed with exit {returncode}: {detail}")


def _default_renderers(platform_name: str | None = None) -> list[str]:
    """Return platform-supported defaults for the benchmark pack."""
    effective_platform = platform.system() if platform_name is None else platform_name
    return ["pygfx"] if effective_platform == "Darwin" else list(RENDERERS)


def _validate_platform_renderers(renderers: list[str], platform_name: str | None = None) -> None:
    """Reject renderer selections that the current platform does not support."""
    effective_platform = platform.system() if platform_name is None else platform_name
    if effective_platform == "Darwin" and "open3d" in renderers:
        raise SystemExit(
            "The Open3D/Filament renderer is not supported on macOS in ORCHAV v0.1. "
            "Use --renderers pygfx."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run current-version visualizer benchmarks on synthetic MPC datasets. "
            "Outputs are written under /tmp by default."
        )
    )
    parser.add_argument(
        "--output-root",
        default="/tmp/orchav_current_visualizer_bench",
        help="Root output directory for benchmark packs.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Benchmark pack label. Default: current timestamp.",
    )
    parser.add_argument(
        "--location",
        default="local",
        help="Free-form location tag, for example remote_vpn or onsite_lan.",
    )
    parser.add_argument(
        "--mpc-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_MPC_COUNTS),
        help="Synthetic MPC counts per frame to generate and benchmark.",
    )
    parser.add_argument(
        "--renderers",
        nargs="+",
        choices=RENDERERS,
        default=_default_renderers(),
        help="Renderer backends to benchmark (macOS default: pygfx; otherwise: both).",
    )
    parser.add_argument(
        "--dataset-frames",
        type=int,
        default=60,
        help="Number of frames to generate in each synthetic dataset.",
    )
    parser.add_argument(
        "--timed-frames",
        type=int,
        default=60,
        help="Timed frame count for each cold/warm benchmark run.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Warmup frames to discard for warm-cache benchmark runs.",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=2,
        help="Maximum synthetic interaction points per MPC.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Synthetic generator chunk size (positive integer, default: 50).",
    )
    parser.add_argument(
        "--display",
        default=None,
        help="X display to use for full visualizer runs, for example :1.",
    )
    parser.add_argument(
        "--qt-platform",
        default=None,
        help="Optional QT_QPA_PLATFORM override, for example xcb or offscreen.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override for benchmark subprocesses.",
    )
    parser.add_argument(
        "--max-performance",
        action="store_true",
        help=(
            "Pass --max-performance to visualizer benchmark subprocesses. "
            "Use this for high-memory pygfx cache-fit runs."
        ),
    )
    parser.add_argument(
        "--present-modes",
        nargs="+",
        choices=PRESENT_MODES,
        default=["request"],
        help=(
            "End-of-frame present modes to benchmark. request measures the "
            "non-blocking update/request path; blocking includes synchronous "
            "present where the renderer supports it."
        ),
    )
    parser.add_argument(
        "--reuse-scenarios",
        action="store_true",
        help="Reuse existing generated scenario directories when present.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate datasets and write environment/profile metadata without benchmarking.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip generation and benchmark existing datasets in the pack directory.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed generator or benchmark command.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even when one or more benchmark subprocesses fail.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Independent process repeats per renderer/MPC/regime. Use >=3 for "
            "remote-vs-onsite comparisons that need confidence intervals."
        ),
    )
    return parser.parse_args()


def _timestamp_label() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path | None,
    check: bool,
) -> tuple[subprocess.CompletedProcess[str], float]:
    printable = _format_cmd(cmd)
    print(f"[current-bench] {printable}")
    start = time.perf_counter()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
    else:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
    elapsed_s = time.perf_counter() - start
    if check and result.returncode != 0:
        raise CommandError(cmd, result.returncode, log_path)
    return result, elapsed_s


def _capture(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"cmd": cmd, "error": str(exc)}
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _powershell_processes(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode a PowerShell JSON process inventory."""
    if result.get("returncode") != 0:
        raise ValueError(result.get("stderr") or result.get("error") or "command failed")
    payload = json.loads(str(result.get("stdout") or "[]"))
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("PowerShell returned an unexpected process inventory")
    return payload


def _capture_windows_visualizer_processes() -> dict[str, Any]:
    """Capture Windows process evidence with a lower-fidelity fallback."""
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        return {
            "status": "unavailable",
            "method": None,
            "processes": [],
            "reason": "PowerShell was not found.",
        }

    cim_script = (
        "$pattern = 'python -m visualizer|run_current_visualizer_benchmark_pack\\.py'; "
        "$items = @(Get-CimInstance Win32_Process -ErrorAction Stop | "
        "Where-Object { $_.CommandLine -match $pattern } | "
        "Select-Object @{Name='pid';Expression={$_.ProcessId}},"
        "@{Name='name';Expression={$_.Name}},"
        "@{Name='command_line';Expression={$_.CommandLine}}); "
        "ConvertTo-Json -InputObject $items -Compress"
    )
    cim_result = _capture([powershell, "-NoProfile", "-NonInteractive", "-Command", cim_script])
    try:
        return {
            "status": "captured",
            "method": "powershell_cim",
            "processes": _powershell_processes(cim_result),
        }
    except (TypeError, ValueError, json.JSONDecodeError) as cim_error:
        fallback_script = (
            "$items = @(Get-Process -ErrorAction Stop | "
            "Where-Object { $_.ProcessName -like 'python*' } | "
            "Select-Object @{Name='pid';Expression={$_.Id}},"
            "@{Name='name';Expression={$_.ProcessName}}); "
            "ConvertTo-Json -InputObject $items -Compress"
        )
        fallback_result = _capture(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", fallback_script]
        )
        try:
            return {
                "status": "partial",
                "method": "powershell_get_process",
                "processes": _powershell_processes(fallback_result),
                "reason": f"CIM inventory failed: {cim_error}",
            }
        except (TypeError, ValueError, json.JSONDecodeError) as fallback_error:
            return {
                "status": "unavailable",
                "method": "powershell",
                "processes": [],
                "reason": (
                    f"CIM inventory failed: {cim_error}; "
                    f"Get-Process fallback failed: {fallback_error}"
                ),
            }


def _pgrep_visualizer_process_command(platform_name: str) -> list[str]:
    """Build the host-specific command used for POSIX process evidence."""
    flags = "-fl" if platform_name == "Darwin" else "-af"
    return [
        "pgrep",
        flags,
        "python -m visualizer|run_current_visualizer_benchmark_pack.py",
    ]


def _capture_visualizer_processes(platform_name: str | None = None) -> dict[str, Any]:
    """Return a structured visualizer-process inventory for the current host."""
    effective_platform = platform.system() if platform_name is None else platform_name
    if effective_platform == "Windows":
        return _capture_windows_visualizer_processes()

    result = _capture(_pgrep_visualizer_process_command(effective_platform))
    returncode = result.get("returncode")
    if returncode == 0:
        return {
            "status": "captured",
            "method": "pgrep",
            "processes": [
                {"command": line}
                for line in str(result.get("stdout") or "").splitlines()
                if line.strip()
            ],
        }
    if returncode == 1:
        return {"status": "captured", "method": "pgrep", "processes": []}
    return {
        "status": "unavailable",
        "method": "pgrep",
        "processes": [],
        "reason": result.get("error") or result.get("stderr") or "pgrep failed",
    }


def _package_versions() -> dict[str, str | None]:
    packages = ["open3d", "pygfx", "wgpu", "PySide6", "numpy", "h5py", "PyYAML"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _benchmark_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.display:
        env["DISPLAY"] = args.display
        env["QT_QPA_PLATFORM"] = args.qt_platform or "xcb"
        env.pop("OPEN3D_CPU_RENDERING", None)
        env.pop("OPEN3D_RENDERING_HEADLESS", None)
    elif args.qt_platform:
        env["QT_QPA_PLATFORM"] = args.qt_platform
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def _write_environment(
    pack_dir: Path, args: argparse.Namespace, env: dict[str, str]
) -> dict[str, Any]:
    git_commit = _capture(["git", "rev-parse", "HEAD"])
    git_branch = _capture(["git", "branch", "--show-current"])
    payload: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label,
        "location": args.location,
        "project_root": str(PROJECT_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "git": {
            "commit": git_commit.get("stdout"),
            "branch": git_branch.get("stdout"),
            "status_short": {
                "status": "not_captured",
                "reason": (
                    "Avoids Git LFS clean filters in restricted benchmark "
                    "environments; capture status manually before release runs."
                ),
            },
        },
        "benchmark_options": {
            "mpc_counts": list(args.mpc_counts),
            "renderers": list(args.renderers),
            "dataset_frames": int(args.dataset_frames),
            "timed_frames": int(args.timed_frames),
            "warmup_frames": int(args.warmup_frames),
            "max_interactions": int(args.max_interactions),
            "chunk_size": int(args.chunk_size),
            "repeats": int(args.repeats),
            "max_performance": bool(args.max_performance),
            "present_modes": list(args.present_modes),
        },
        "env": {
            key: env.get(key)
            for key in (
                "DISPLAY",
                "QT_QPA_PLATFORM",
                "CUDA_VISIBLE_DEVICES",
                "MITSUBA_VARIANT",
                "WGPU_BACKEND",
                "OPEN3D_CPU_RENDERING",
                "OPEN3D_RENDERING_HEADLESS",
            )
        },
        "packages": _package_versions(),
        "nvidia_smi": _capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "visualizer_processes": _capture_visualizer_processes(),
    }
    (pack_dir / "environment.json").write_text(json.dumps(payload, indent=2))
    return payload


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _scenario_dir(pack_dir: Path, mpcs: int) -> Path:
    return pack_dir / "scenarios" / f"mpcs_{mpcs}"


def _generate_dataset(
    pack_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    mpcs: int,
) -> dict[str, Any]:
    scenario_dir = _scenario_dir(pack_dir, mpcs)
    log_path = pack_dir / "logs" / f"generate_mpcs_{mpcs}.log"
    if args.benchmark_only:
        return {
            "mpcs": mpcs,
            "scenario": str(scenario_dir),
            "status": "skipped_benchmark_only",
        }
    if scenario_dir.exists() and not args.reuse_scenarios:
        if not str(scenario_dir.resolve()).startswith(str(pack_dir.resolve())):
            raise RuntimeError(f"refusing to remove scenario outside pack: {scenario_dir}")
        shutil.rmtree(scenario_dir)
    if scenario_dir.exists() and args.reuse_scenarios:
        status = "reused"
        elapsed_s = 0.0
    else:
        cmd = [
            sys.executable,
            str(SYNTHETIC_GENERATOR),
            "--frames",
            str(args.dataset_frames),
            "--mpcs",
            str(mpcs),
            "--max-interactions",
            str(args.max_interactions),
            "--chunk-size",
            str(args.chunk_size),
            "--output",
            str(scenario_dir),
        ]
        try:
            _, elapsed_s = _run(cmd, env=env, log_path=log_path, check=True)
            status = "generated"
        except CommandError as exc:
            if args.stop_on_error:
                raise
            return {
                "mpcs": mpcs,
                "scenario": str(scenario_dir),
                "status": "failed",
                "error": str(exc),
                "log": str(log_path),
            }

    profile_path = pack_dir / "profiles" / f"mpcs_{mpcs}_profile.json"
    try:
        profile = write_scenario_profile(scenario_dir, profile_path)
    except Exception as exc:  # pragma: no cover - diagnostic path
        profile = {"status": "fail", "issues": [str(exc)]}
    return {
        "mpcs": mpcs,
        "scenario": str(scenario_dir),
        "status": status,
        "elapsed_s": elapsed_s,
        "bytes": _dir_size_bytes(scenario_dir / "frames"),
        "profile": profile,
        "profile_path": str(profile_path),
        "log": str(log_path),
    }


def _benchmark_command(
    renderer: str,
    scenario_dir: Path,
    output_path: Path,
    *,
    regime: str,
    present_mode: str,
    timed_frames: int,
    warmup_frames: int,
    max_performance: bool,
) -> list[str]:
    if regime == "cold":
        benchmark_frames = timed_frames
        benchmark_warmup = 0
        previsit = False
    elif regime == "warm":
        benchmark_frames = timed_frames + warmup_frames
        benchmark_warmup = warmup_frames
        previsit = True
    else:
        raise ValueError(f"unknown regime: {regime}")

    cmd = [
        sys.executable,
        "-m",
        "visualizer",
        "--renderer",
        renderer,
        "--scenario",
        str(scenario_dir),
        "--benchmark",
        str(benchmark_frames),
        "--benchmark-warmup",
        str(benchmark_warmup),
        "--benchmark-present-mode",
        present_mode,
        "--benchmark-output",
        str(output_path),
        "--no-resume",
    ]
    if previsit:
        cmd.append("--benchmark-previsit-all-frames")
    if max_performance:
        cmd.append("--max-performance")
    return cmd


def _result_summary(path: Path) -> dict[str, Any]:
    summary = summarize_benchmark_json(path)
    avg_ms = summary.get("avg_total_ms")
    summary["avg_fps_equiv"] = 1000.0 / avg_ms if avg_ms else None
    avg_before_end_ms = summary.get("avg_total_before_end_ms")
    summary["avg_before_end_fps_equiv"] = 1000.0 / avg_before_end_ms if avg_before_end_ms else None
    return summary


def _run_benchmarks(
    pack_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    mpc_counts: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for renderer in args.renderers:
        for mpcs in mpc_counts:
            scenario_dir = _scenario_dir(pack_dir, mpcs)
            for regime in ("cold", "warm"):
                for present_mode in args.present_modes:
                    for repeat in range(1, args.repeats + 1):
                        repeat_suffix = f"_rep{repeat:02d}" if args.repeats > 1 else ""
                        stem = f"{renderer}_mpcs_{mpcs}_{regime}_{present_mode}{repeat_suffix}"
                        output_path = pack_dir / "results" / f"{stem}.json"
                        log_path = pack_dir / "logs" / f"{stem}.log"
                        cmd = _benchmark_command(
                            renderer,
                            scenario_dir,
                            output_path,
                            regime=regime,
                            present_mode=present_mode,
                            timed_frames=args.timed_frames,
                            warmup_frames=args.warmup_frames,
                            max_performance=bool(args.max_performance),
                        )
                        try:
                            _, elapsed_s = _run(cmd, env=env, log_path=log_path, check=True)
                            compact = _result_summary(output_path)
                            results.append(
                                {
                                    "renderer": renderer,
                                    "mpcs": mpcs,
                                    "regime": regime,
                                    "present_mode": present_mode,
                                    "repeat": repeat,
                                    "elapsed_s": elapsed_s,
                                    "output": str(output_path),
                                    "log": str(log_path),
                                    "summary": compact,
                                }
                            )
                        except Exception as exc:
                            failure = {
                                "renderer": renderer,
                                "mpcs": mpcs,
                                "regime": regime,
                                "present_mode": present_mode,
                                "repeat": repeat,
                                "error": str(exc),
                                "log": str(log_path),
                                "command": cmd,
                            }
                            failures.append(failure)
                            print(f"[current-bench] FAILED: {failure['error']}", file=sys.stderr)
                            if args.stop_on_error:
                                raise
    return results, failures


def _t_critical_95(df: int) -> float:
    tcrit = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df <= 0:
        return 0.0
    return tcrit.get(df, 1.96)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "stdev": None,
            "stderr": None,
            "ci95": None,
        }
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if n > 1 else 0.0
    stderr = stdev / (n**0.5) if n > 1 else 0.0
    ci95 = _t_critical_95(n - 1) * stderr if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "stdev": stdev,
        "stderr": stderr,
        "ci95": ci95,
    }


def _aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "wall_update_rate_hz",
        "avg_total_before_end_ms",
        "avg_total_ms",
        "avg_before_end_fps_equiv",
        "avg_fps_equiv",
        "p95_total_before_end_ms",
        "p95_total_ms",
        "avg_viewmodel_ms",
        "avg_render_ms",
        "avg_end_frame_update_ms",
        "avg_force_draw_ms",
        "avg_request_draw_ms",
        "startup_to_first_frame_ms",
    ]
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            str(result["renderer"]),
            int(result["mpcs"]),
            str(result["regime"]),
            str(
                result.get("present_mode")
                or result.get("summary", {}).get("benchmark_present_mode")
            ),
        )
        groups.setdefault(key, []).append(result)

    aggregate: list[dict[str, Any]] = []
    for (renderer, mpcs, regime, present_mode), group in sorted(groups.items()):
        metric_stats: dict[str, Any] = {}
        for metric in metrics:
            values = [
                float(value)
                for result in group
                for value in [result.get("summary", {}).get(metric)]
                if isinstance(value, (int, float))
            ]
            metric_stats[metric] = _stats(values)
        aggregate.append(
            {
                "renderer": renderer,
                "mpcs": mpcs,
                "regime": regime,
                "present_mode": present_mode,
                "n_repeats": len(group),
                "metrics": metric_stats,
            }
        )
    return aggregate


def _write_markdown_summary(pack_dir: Path, summary: dict[str, Any]) -> None:
    analysis = summary.get("analysis") or {}
    aggregate = summary.get("aggregate_results") or []
    lines = [
        f"# Current Visualizer Benchmark Pack: {summary['label']}",
        "",
        "This is a current-version benchmark pack for local renderer trend measurements.",
        "",
        f"- Location: `{summary['location']}`",
        f"- Output directory: `{pack_dir}`",
        f"- Git commit: `{summary['environment'].get('git', {}).get('commit')}`",
        f"- Max performance profile: `{summary['environment'].get('benchmark_options', {}).get('max_performance', False)}`",
        "",
    ]
    if aggregate:
        lines.extend(
            [
                "## Aggregate Results",
                "",
                (
                    "| Renderer | MPCs/frame | Regime | Present mode | Repeats | "
                    "Pipeline avg (ms) | Total avg (ms) | Total 95% CI (ms) | "
                    "Wall updates/s | CPU equiv/s | End-frame avg (ms) | Total p95 (ms) |"
                ),
                (
                    "|----------|-----------:|--------|--------------|--------:|"
                    "------------------:|---------------:|------------------:|"
                    "---------------:|------------:|-----------------:|---------------:|"
                ),
            ]
        )
        for row in aggregate:
            avg_before_end = row["metrics"]["avg_total_before_end_ms"]
            avg_total = row["metrics"]["avg_total_ms"]
            wall_rate = row["metrics"]["wall_update_rate_hz"]
            fps = row["metrics"]["avg_fps_equiv"]
            p95 = row["metrics"]["p95_total_ms"]
            present = row["metrics"]["avg_end_frame_update_ms"]
            before_end_mean = avg_before_end["mean"]
            avg_mean = avg_total["mean"]
            avg_ci = avg_total["ci95"]
            wall_rate_mean = wall_rate["mean"]
            fps_mean = fps["mean"]
            p95_mean = p95["mean"]
            present_mean = present["mean"]
            lines.append(
                (
                    "| {renderer} | {mpcs} | {regime} | {present_mode} | {n} | "
                    "{before_end} | {avg} | {ci} | {wall_rate} | {fps} | {present} | {p95} |"
                ).format(
                    renderer=row["renderer"],
                    mpcs=row["mpcs"],
                    regime=row["regime"],
                    present_mode=row.get("present_mode", "n/a"),
                    n=row["n_repeats"],
                    before_end=(
                        f"{before_end_mean:.2f}"
                        if isinstance(before_end_mean, (int, float))
                        else "n/a"
                    ),
                    avg=f"{avg_mean:.2f}" if isinstance(avg_mean, (int, float)) else "n/a",
                    ci=f"{avg_ci:.2f}" if isinstance(avg_ci, (int, float)) else "n/a",
                    wall_rate=(
                        f"{wall_rate_mean:.1f}"
                        if isinstance(wall_rate_mean, (int, float))
                        else "n/a"
                    ),
                    fps=f"{fps_mean:.1f}" if isinstance(fps_mean, (int, float)) else "n/a",
                    present=(
                        f"{present_mean:.2f}" if isinstance(present_mean, (int, float)) else "n/a"
                    ),
                    p95=f"{p95_mean:.2f}" if isinstance(p95_mean, (int, float)) else "n/a",
                )
            )
        lines.extend(["", "## Per-Repeat Results", ""])
    lines.extend(
        [
            (
                "| Renderer | MPCs/frame | Regime | Present mode | Pipeline avg (ms) | "
                "Total avg (ms) | Wall updates/s | CPU equiv/s | Total p95 (ms) | JSON |"
            ),
            (
                "|----------|-----------:|--------|--------------|------------------:|"
                "---------------:|---------------:|------------:|---------------:|------|"
            ),
        ]
    )
    for result in summary.get("results", []):
        compact = result.get("summary", {})
        avg_before_end = compact.get("avg_total_before_end_ms")
        avg_ms = compact.get("avg_total_ms")
        wall_rate = compact.get("wall_update_rate_hz")
        fps = compact.get("avg_fps_equiv")
        p95 = compact.get("p95_total_ms")
        regime = result["regime"]
        if result.get("repeat") is not None:
            regime = f"{regime} r{int(result['repeat']):02d}"
        lines.append(
            (
                "| {renderer} | {mpcs} | {regime} | {present_mode} | "
                "{before_end} | {avg} | {wall_rate} | {fps} | {p95} | `{path}` |"
            ).format(
                renderer=result["renderer"],
                mpcs=result["mpcs"],
                regime=regime,
                present_mode=result.get(
                    "present_mode", compact.get("benchmark_present_mode", "n/a")
                ),
                before_end=(
                    f"{avg_before_end:.2f}" if isinstance(avg_before_end, (int, float)) else "n/a"
                ),
                avg=f"{avg_ms:.2f}" if isinstance(avg_ms, (int, float)) else "n/a",
                wall_rate=(f"{wall_rate:.1f}" if isinstance(wall_rate, (int, float)) else "n/a"),
                fps=f"{fps:.1f}" if isinstance(fps, (int, float)) else "n/a",
                p95=f"{p95:.2f}" if isinstance(p95, (int, float)) else "n/a",
                path=result["output"],
            )
        )
    failures = summary.get("failures", [])
    if failures:
        lines.extend(
            [
                "",
                "## Failures",
                "",
                "| Renderer | MPCs/frame | Regime | Present mode | Error | Log |",
                "|----------|-----------:|--------|--------------|-------|-----|",
            ]
        )
        for failure in failures:
            error = str(failure.get("error", "")).replace("|", "\\|")
            lines.append(
                f"| {failure['renderer']} | {failure['mpcs']} | {failure['regime']} | "
                f"{failure.get('present_mode', 'n/a')} | "
                f"{error} | `{failure['log']}` |"
            )
    artifacts = analysis.get("artifacts") or []
    if artifacts:
        lines.extend(["", "## Analysis Artifacts", ""])
        for artifact in artifacts:
            lines.append(f"- `{artifact}`")
    if analysis.get("error"):
        lines.extend(["", f"Analysis artifact generation failed: `{analysis['error']}`"])
    lines.append("")
    (pack_dir / "summary.md").write_text("\n".join(lines))


def _write_summary_csv(pack_dir: Path, summary: dict[str, Any]) -> Path:
    csv_path = pack_dir / "summary.csv"
    fields = [
        "renderer",
        "mpcs",
        "regime",
        "present_mode",
        "repeat",
        "wall_update_rate_hz",
        "avg_total_before_end_ms",
        "avg_total_ms",
        "avg_before_end_fps_equiv",
        "avg_fps_equiv",
        "p95_total_before_end_ms",
        "p95_total_ms",
        "avg_viewmodel_ms",
        "avg_render_ms",
        "avg_end_frame_update_ms",
        "avg_force_draw_ms",
        "avg_request_draw_ms",
        "startup_to_first_frame_ms",
        "json_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in summary.get("results", []):
            compact = result.get("summary", {})
            writer.writerow(
                {
                    "renderer": result.get("renderer"),
                    "mpcs": result.get("mpcs"),
                    "regime": result.get("regime"),
                    "present_mode": result.get(
                        "present_mode", compact.get("benchmark_present_mode")
                    ),
                    "repeat": result.get("repeat"),
                    "wall_update_rate_hz": compact.get("wall_update_rate_hz"),
                    "avg_total_before_end_ms": compact.get("avg_total_before_end_ms"),
                    "avg_total_ms": compact.get("avg_total_ms"),
                    "avg_before_end_fps_equiv": compact.get("avg_before_end_fps_equiv"),
                    "avg_fps_equiv": compact.get("avg_fps_equiv"),
                    "p95_total_before_end_ms": compact.get("p95_total_before_end_ms"),
                    "p95_total_ms": compact.get("p95_total_ms"),
                    "avg_viewmodel_ms": compact.get("avg_viewmodel_ms"),
                    "avg_render_ms": compact.get("avg_render_ms"),
                    "avg_end_frame_update_ms": compact.get("avg_end_frame_update_ms"),
                    "avg_force_draw_ms": compact.get("avg_force_draw_ms"),
                    "avg_request_draw_ms": compact.get("avg_request_draw_ms"),
                    "startup_to_first_frame_ms": compact.get("startup_to_first_frame_ms"),
                    "json_path": result.get("output"),
                }
            )
    return csv_path


def _write_aggregate_csv(pack_dir: Path, summary: dict[str, Any]) -> Path:
    csv_path = pack_dir / "aggregate_summary.csv"
    fields = [
        "renderer",
        "mpcs",
        "regime",
        "present_mode",
        "n_repeats",
        "metric",
        "mean",
        "stdev",
        "stderr",
        "ci95",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary.get("aggregate_results", []):
            for metric, stats in row.get("metrics", {}).items():
                writer.writerow(
                    {
                        "renderer": row["renderer"],
                        "mpcs": row["mpcs"],
                        "regime": row["regime"],
                        "present_mode": row.get("present_mode"),
                        "n_repeats": row["n_repeats"],
                        "metric": metric,
                        "mean": stats.get("mean"),
                        "stdev": stats.get("stdev"),
                        "stderr": stats.get("stderr"),
                        "ci95": stats.get("ci95"),
                    }
                )
    return csv_path


def _plot_metric(
    results: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    renderers = sorted({str(result["renderer"]) for result in results})
    regimes = ("cold", "warm")
    present_modes = sorted({str(result.get("present_mode", "request")) for result in results})
    fig, axes = plt.subplots(
        len(present_modes),
        len(regimes),
        figsize=(10, 4 * len(present_modes)),
        sharey=True,
        squeeze=False,
    )
    if len(regimes) == 1:
        axes = [axes]
    for row_idx, present_mode in enumerate(present_modes):
        for col_idx, regime in enumerate(regimes):
            ax = axes[row_idx][col_idx]
            for renderer in renderers:
                points = [
                    (
                        int(result["mpcs"]),
                        result.get("metrics", {}).get(metric, {}).get("mean"),
                    )
                    for result in results
                    if result.get("renderer") == renderer
                    and result.get("regime") == regime
                    and result.get("present_mode") == present_mode
                ]
                points = [
                    (mpcs, float(value))
                    for mpcs, value in sorted(points)
                    if isinstance(value, (int, float))
                ]
                if points:
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, marker="o", label=renderer)
            ax.set_xscale("log")
            ax.set_xlabel("MPCs per frame")
            ax.set_title(f"{regime.capitalize()} / {present_mode}")
            ax.grid(True, which="both", alpha=0.3)
        axes[row_idx][0].set_ylabel(ylabel)
        axes[row_idx][-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_analysis_artifacts(pack_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    artifacts: list[str] = []
    csv_path = _write_summary_csv(pack_dir, summary)
    artifacts.append(str(csv_path))
    aggregate_csv_path = _write_aggregate_csv(pack_dir, summary)
    artifacts.append(str(aggregate_csv_path))

    results = list(summary.get("aggregate_results", []))
    if not results:
        return {"artifacts": artifacts}
    try:
        figures_dir = pack_dir / "figures"
        avg_path = figures_dir / "avg_total_ms_by_mpcs.png"
        fps_path = figures_dir / "fps_equiv_by_mpcs.png"
        before_end_path = figures_dir / "avg_total_before_end_ms_by_mpcs.png"
        _plot_metric(
            results,
            metric="avg_total_ms",
            ylabel="Average frame time (ms)",
            title="Average Frame Time by MPC Density",
            output_path=avg_path,
        )
        _plot_metric(
            results,
            metric="avg_fps_equiv",
            ylabel="FPS equivalent",
            title="FPS Equivalent by MPC Density",
            output_path=fps_path,
        )
        _plot_metric(
            results,
            metric="avg_total_before_end_ms",
            ylabel="Average pre-present pipeline time (ms)",
            title="Average Pipeline Time by MPC Density",
            output_path=before_end_path,
        )
        artifacts.extend([str(avg_path), str(fps_path), str(before_end_path)])
    except Exception as exc:  # pragma: no cover - optional plotting diagnostics
        return {"artifacts": artifacts, "error": str(exc)}
    return {"artifacts": artifacts}


def main() -> int:
    args = parse_args()
    _validate_platform_renderers(args.renderers)
    if args.generate_only and args.benchmark_only:
        raise SystemExit("--generate-only and --benchmark-only are mutually exclusive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")
    label = args.label or _timestamp_label()
    args.label = label
    pack_dir = Path(args.output_root).expanduser().resolve() / label
    pack_dir.mkdir(parents=True, exist_ok=True)

    env = _benchmark_env(args)
    environment = _write_environment(pack_dir, args, env)

    datasets: list[dict[str, Any]] = []
    for mpcs in args.mpc_counts:
        datasets.append(_generate_dataset(pack_dir, args, env, mpcs))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = [
        dataset for dataset in datasets if dataset.get("status") == "failed"
    ]
    if not args.generate_only:
        benchmarkable_mpcs = [
            int(dataset["mpcs"]) for dataset in datasets if dataset.get("status") != "failed"
        ]
        bench_results, bench_failures = _run_benchmarks(
            pack_dir,
            args,
            env,
            benchmarkable_mpcs,
        )
        results.extend(bench_results)
        failures.extend(bench_failures)

    summary = {
        "label": label,
        "location": args.location,
        "pack_dir": str(pack_dir),
        "environment": environment,
        "datasets": datasets,
        "results": results,
        "aggregate_results": _aggregate_results(results),
        "failures": failures,
    }
    summary["analysis"] = _write_analysis_artifacts(pack_dir, summary)
    (pack_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    _write_markdown_summary(pack_dir, summary)
    print(f"[current-bench] wrote {pack_dir / 'summary.md'}")
    if failures and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
