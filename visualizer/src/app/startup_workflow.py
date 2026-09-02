"""Command-line startup orchestration for the visualizer app shell.

This module is the stable path behind ``python -m visualizer`` after GPU
preflight. It parses CLI flags, applies launch-time environment policy, creates
the Qt application, performs deferred visualizer initialization, opens the
optional scenario, and then starts GUI, benchmark, or batch-render mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from shared.logging import get_logger, set_log_level

from ..authoring.feature import scenario_builder_enabled
from ..benchmarking.harness import (
    previsit_benchmark_steps,
    resolve_benchmark_steps,
)
from ..benchmarking.recorder import BenchmarkRecorder
from ..materials.texture_policy import apply_texture_launch_policy
from ..renderers.protocol import renderer_capabilities
from ..renderers.registry import renderer_choices
from ..services.cache_service import CacheInvalidationScope
from ..services.session_service import (
    WorkspaceSnapshot,
    normalize_scenario_root,
    read_workspace_snapshot,
)
from ..state import AppState, create_initial_state
from .app_identity import apply_application_identity
from .lifecycle import shutdown_visualizer
from .window_manager import LAYOUT_PROFILE_CHOICES

logger = get_logger(__name__)

MAX_PERFORMANCE_CANON_CACHE_MB = 4096
MAX_PERFORMANCE_PYGFX_LINE_CACHE_MB = 1024
CLI_DATA_MODE_CHOICES = ("files", "live_grpc", "remote_hdf5")


class ViewportMode(str, Enum):
    """Supported renderer-host modes exposed by the CLI."""

    AUTO = "auto"
    EMBEDDED = "embedded"
    DETACHED = "detached"


class LaunchKind(str, Enum):
    """Mutually exclusive top-level visualizer launch behaviors."""

    INTERACTIVE = "interactive"
    AUTHORING = "authoring"
    BENCHMARK = "benchmark"
    RENDER_FRAMES = "render-frames"


@dataclass(frozen=True, slots=True)
class ResolvedLaunchMode:
    """Validated launch policy resolved before any Qt application is created."""

    kind: LaunchKind
    viewport_mode: ViewportMode
    layout_profile: str


class _CliScenarioOpenError(RuntimeError):
    """A handled explicit-scenario failure that must end CLI startup."""


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the visualizer CLI parser."""
    parser = argparse.ArgumentParser(
        description="ORCHAV - Radio Propagation Analysis & Visualization Tool"
    )
    parser.add_argument("--scenario", type=str, help="Path to scenario folder or YAML file")
    parser.add_argument(
        "--data-mode",
        choices=CLI_DATA_MODE_CHOICES,
        help="Override the scenario data source for this visualizer process.",
    )
    parser.add_argument(
        "--grpc-port",
        type=_parse_grpc_port,
        help=(
            "Override the client port for live_grpc or remote_hdf5 while retaining "
            "the configured host."
        ),
    )
    parser.add_argument(
        "--author",
        action="store_true",
        help="Open the feature-gated pygfx Scenario Builder workspace.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=renderer_choices(),
        default="pygfx",
        help="Renderer backend: 'pygfx' (default) or 'open3d' (Open3D/Filament).",
    )
    parser.add_argument(
        "--viewport-mode",
        choices=[mode.value for mode in ViewportMode],
        default=ViewportMode.AUTO.value,
        help=(
            "Renderer hosting: 'auto' embeds normal pygfx visualization and keeps "
            "Open3D, capture, benchmark, and frame-render runs detached."
        ),
    )
    parser.add_argument(
        "--pygfx-present-method",
        choices=["screen", "bitmap", "auto"],
        default=None,
        help=(
            "pygfx canvas presentation: the platform default uses 'screen' on Windows "
            "and rendercanvas automatic selection elsewhere; explicit 'auto' delegates "
            "to rendercanvas, 'screen' uses the direct GPU surface, and 'bitmap' uses "
            "the compatible GPU-readback/Qt-composition path."
        ),
    )
    parser.add_argument(
        "--pygfx-adapter-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Select the pygfx/wgpu adapter by name via "
            "PYGFX_WGPU_ADAPTER_NAME before renderer startup."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run in benchmark mode: auto-step through N frames, record per-stage "
            "pipeline timings, write JSON results, and exit."
        ),
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=5,
        metavar="W",
        help="Number of warmup frames to discard in benchmark mode (default: 5).",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for benchmark JSON results (default: benchmark_render_results.json).",
    )
    parser.add_argument(
        "--benchmark-previsit-all-frames",
        action="store_true",
        help=(
            "In benchmark mode, previsit every available frame step in-process before "
            "timed playback so the measured run reflects a true warm-cache regime."
        ),
    )
    parser.add_argument(
        "--benchmark-present-mode",
        type=str,
        choices=["blocking", "request"],
        default="blocking",
        help=(
            "In benchmark mode, choose whether end-of-frame work synchronously services "
            "one backend draw (`blocking`) or only queues a draw request (`request`). "
            "Neither mode measures physical display presentation."
        ),
    )
    parser.add_argument(
        "--benchmark-state-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Benchmark-only AppState override JSON used for scripted ablations.",
    )
    parser.add_argument(
        "--render-frames",
        type=str,
        default=None,
        metavar="DIR",
        help="Batch render all frames as PNGs to DIR and exit (requires --scenario).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Skip automatic resume on startup.",
    )
    parser.add_argument(
        "--camera-debug",
        action="store_true",
        help="Enable structured Follow/POV camera diagnostics in logs.",
    )
    parser.add_argument(
        "--layout-profile",
        type=str,
        choices=LAYOUT_PROFILE_CHOICES,
        default="auto",
        help=(
            "Initial detached-window layout. Use 'capture-renderer' for a 1920x1080 "
            "renderer target, or 'capture-workspace' for a 1920x1080 "
            "controls-plus-renderer target, when the display can fit it."
        ),
    )
    parser.add_argument(
        "--max-performance",
        action="store_true",
        help=(
            "Enable a high-memory performance profile for heavy playback. This raises "
            f"VIZ_CANON_CACHE_MB to at least {MAX_PERFORMANCE_CANON_CACHE_MB} and, "
            "for pygfx, ORCHAV_PYGFX_MPC_LINE_CACHE_MB to at least "
            f"{MAX_PERFORMANCE_PYGFX_LINE_CACHE_MB}."
        ),
    )
    parser.add_argument(
        "--pygfx-mpc-line-cache-mb",
        type=int,
        default=None,
        metavar="MB",
        help=(
            "Set the pygfx expanded MPC-line cache budget in MB. Default is 0 "
            "(disabled). Useful only when the cache can hold the full replay working set."
        ),
    )
    texture_group = parser.add_mutually_exclusive_group()
    texture_group.add_argument(
        "--enable-textures",
        action="store_true",
        help="Enable albedo/detail texture maps for PBR-capable renderers.",
    )
    texture_group.add_argument(
        "--disable-textures",
        action="store_true",
        help="Disable all albedo/detail texture maps at launch (default).",
    )
    return parser


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate visualizer CLI arguments."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.benchmark < 0:
        parser.error("--benchmark must be >= 0")
    if args.benchmark_warmup < 0:
        parser.error("--benchmark-warmup must be >= 0")
    if args.render_frames and args.benchmark > 0:
        parser.error("--render-frames and --benchmark are mutually exclusive")
    if args.render_frames and not args.scenario:
        parser.error("--render-frames requires --scenario")
    if args.benchmark > 0 and not args.scenario:
        parser.error("--benchmark requires --scenario")
    if args.benchmark_previsit_all_frames and args.benchmark <= 0:
        parser.error("--benchmark-previsit-all-frames requires --benchmark")
    if args.benchmark_present_mode != "blocking" and args.benchmark <= 0:
        parser.error("--benchmark-present-mode requires --benchmark")
    if args.benchmark_state_json and args.benchmark <= 0:
        parser.error("--benchmark-state-json requires --benchmark")
    if args.benchmark_output and args.benchmark <= 0:
        parser.error("--benchmark-output requires --benchmark")
    if args.pygfx_mpc_line_cache_mb is not None and args.pygfx_mpc_line_cache_mb < 0:
        parser.error("--pygfx-mpc-line-cache-mb must be >= 0")
    if args.pygfx_adapter_name is not None and args.renderer != "pygfx":
        parser.error("--pygfx-adapter-name requires --renderer pygfx")
    if args.author and args.renderer != "pygfx":
        parser.error("--author requires --renderer pygfx")
    if args.author and (args.benchmark > 0 or args.render_frames):
        parser.error("--author cannot be combined with benchmark or batch-render mode")
    if args.author and (args.data_mode is not None or args.grpc_port is not None):
        parser.error("--author cannot be combined with data-source overrides")
    if (args.data_mode is not None or args.grpc_port is not None) and not args.scenario:
        parser.error("--data-mode and --grpc-port require --scenario")
    if args.data_mode == "files" and args.grpc_port is not None:
        parser.error("--grpc-port cannot be combined with --data-mode files")
    if args.author and not scenario_builder_enabled():
        parser.error("--author requires ORCHAV_ENABLE_SCENARIO_BUILDER=1")
    try:
        resolved = resolve_launch_mode(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.resolved_launch_mode = resolved
    args.resolved_viewport_mode = resolved.viewport_mode.value
    return args


def _parse_grpc_port(value: str) -> int:
    """Parse a nonzero TCP port for a visualizer gRPC client override."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gRPC port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("gRPC port must be between 1 and 65535")
    return port


def resolve_launch_mode(args: argparse.Namespace) -> ResolvedLaunchMode:
    """Validate cross-option host policy and resolve ``auto`` deterministically."""

    requested = ViewportMode(args.viewport_mode)
    capture_layout = args.layout_profile != "auto"

    if args.author:
        if capture_layout:
            raise ValueError("--author cannot be combined with a capture layout profile")
        if requested is ViewportMode.DETACHED:
            raise ValueError("--author does not support --viewport-mode detached")
        return ResolvedLaunchMode(LaunchKind.AUTHORING, ViewportMode.EMBEDDED, "auto")

    if args.render_frames:
        if requested is ViewportMode.EMBEDDED:
            raise ValueError("--render-frames requires a detached viewport")
        return ResolvedLaunchMode(
            LaunchKind.RENDER_FRAMES,
            ViewportMode.DETACHED,
            args.layout_profile,
        )

    if args.benchmark > 0:
        if requested is ViewportMode.EMBEDDED and (args.renderer != "pygfx" or capture_layout):
            raise ValueError(
                "embedded benchmarks require --renderer pygfx and --layout-profile auto"
            )
        resolved = (
            ViewportMode.EMBEDDED if requested is ViewportMode.EMBEDDED else ViewportMode.DETACHED
        )
        return ResolvedLaunchMode(LaunchKind.BENCHMARK, resolved, args.layout_profile)

    if requested is ViewportMode.EMBEDDED and args.renderer != "pygfx":
        raise ValueError("--viewport-mode embedded requires --renderer pygfx")
    if requested is ViewportMode.EMBEDDED and capture_layout:
        raise ValueError("capture layout profiles require a detached viewport")

    if requested is ViewportMode.AUTO:
        resolved = (
            ViewportMode.DETACHED
            if args.renderer == "open3d" or capture_layout
            else ViewportMode.EMBEDDED
        )
    else:
        resolved = requested
    return ResolvedLaunchMode(LaunchKind.INTERACTIVE, resolved, args.layout_profile)


def select_startup_workspace(
    scenario_path: Optional[str],
    session_dir: Optional[Path] = None,
) -> Optional[WorkspaceSnapshot]:
    """Select the newest usable autosave in one directory scan.

    With a scenario hint, only that logical scenario matches. Without a hint,
    stale autosaves are skipped until the newest snapshot whose scenario still
    exists is found.
    """
    if session_dir is None:
        session_dir = Path.home() / ".orchav" / "sessions"
    if not session_dir.is_dir():
        return None

    resolved_scenario = normalize_scenario_root(scenario_path)
    if scenario_path is not None and resolved_scenario is None:
        return None

    def _modified_at(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    try:
        candidates = sorted(session_dir.glob("*.json"), key=_modified_at, reverse=True)
    except OSError:
        return None

    for path in candidates:
        snapshot = read_workspace_snapshot(path)
        if snapshot is None or not snapshot.summary.is_autosave:
            continue
        summary = snapshot.summary
        if resolved_scenario is not None and os.path.normcase(
            str(summary.scenario_root)
        ) != os.path.normcase(str(resolved_scenario)):
            continue
        if not (summary.scenario_root / "scenario.yaml").is_file():
            continue
        return snapshot
    return None


def load_benchmark_state_overrides(path: str) -> dict[str, Any]:
    """Load a JSON object containing AppState fields for benchmark ablations."""
    override_path = Path(path).expanduser()
    with override_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark state JSON must contain an object: {override_path}")

    allowed = {field.name for field in fields(AppState)}
    ignored = sorted(key for key in payload if key not in allowed)
    for key in ignored:
        logger.warning("Ignoring unknown benchmark AppState override key: %s", key)

    overrides = {key: value for key, value in payload.items() if key in allowed}
    if not overrides:
        return {}

    base = create_initial_state().to_dict()
    base.update(overrides)
    normalized = AppState.from_dict(base)
    return {key: getattr(normalized, key) for key in overrides}


def apply_benchmark_state_overrides(viz: Any, overrides: dict[str, Any]) -> None:
    """Apply benchmark-only AppState overrides before previsit/warmup."""
    if not overrides:
        return
    viz.set_state(**overrides)
    viz.cache_service.invalidate(CacheInvalidationScope.FILTERS, reason="benchmark_state_overrides")
    viz.force_update_next_frame = True
    logger.info("Applied benchmark AppState overrides: %s", sorted(overrides))


def set_env_min_int(name: str, minimum: int) -> tuple[int, bool]:
    """Set an integer environment variable to at least ``minimum``."""
    raw = os.environ.get(name)
    try:
        current = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        current = None
    if current is None or current < minimum:
        os.environ[name] = str(int(minimum))
        return int(minimum), True
    return int(current), False


def read_env_int(name: str, default: int = 0) -> int:
    """Read an integer environment variable with a robust fallback."""
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def apply_launch_environment(args: argparse.Namespace) -> list[str]:
    """Apply CLI-driven environment settings and return user-facing notes.

    Environment updates happen before renderer/service construction so cache
    sizing, texture policy, and camera diagnostics are visible to downstream
    modules during startup.
    """
    if args.camera_debug:
        os.environ["ORCHAV_CAMERA_DEBUG"] = "1"
        set_log_level("INFO")
        logger.info("Camera diagnostics enabled (ORCHAV_CAMERA_DEBUG=1)")
    apply_texture_launch_policy(
        enable_textures=args.enable_textures,
        disable_textures=args.disable_textures,
    )

    performance_notes: list[str] = []
    if args.pygfx_present_method is not None:
        os.environ["ORCHAV_PYGFX_PRESENT_METHOD"] = args.pygfx_present_method
        performance_notes.append(f"ORCHAV_PYGFX_PRESENT_METHOD={args.pygfx_present_method}")
    if args.pygfx_adapter_name is not None:
        os.environ["PYGFX_WGPU_ADAPTER_NAME"] = args.pygfx_adapter_name
        performance_notes.append(f"PYGFX_WGPU_ADAPTER_NAME={args.pygfx_adapter_name}")
    if (
        args.renderer == "pygfx"
        and args.benchmark > 0
        and args.benchmark_present_mode == "blocking"
        and "ORCHAV_PYGFX_CANVAS_SCHEDULER" not in os.environ
        and "ORCHAV_PYGFX_CANVAS_MAX_FPS" not in os.environ
    ):
        os.environ["ORCHAV_PYGFX_CANVAS_SCHEDULER"] = "manual"
        performance_notes.append("blocking benchmark: ORCHAV_PYGFX_CANVAS_SCHEDULER=manual")
    if args.max_performance:
        canon_mb, canon_changed = set_env_min_int(
            "VIZ_CANON_CACHE_MB",
            MAX_PERFORMANCE_CANON_CACHE_MB,
        )
        performance_notes.append(
            f"max-performance: VIZ_CANON_CACHE_MB={canon_mb}"
            + ("" if canon_changed else " (already set)")
        )
        if args.renderer == "pygfx" and args.pygfx_mpc_line_cache_mb is None:
            line_cache_mb, line_cache_changed = set_env_min_int(
                "ORCHAV_PYGFX_MPC_LINE_CACHE_MB",
                MAX_PERFORMANCE_PYGFX_LINE_CACHE_MB,
            )
            performance_notes.append(
                f"max-performance: ORCHAV_PYGFX_MPC_LINE_CACHE_MB={line_cache_mb}"
                + ("" if line_cache_changed else " (already set)")
            )
    if args.pygfx_mpc_line_cache_mb is not None:
        os.environ["ORCHAV_PYGFX_MPC_LINE_CACHE_MB"] = str(args.pygfx_mpc_line_cache_mb)
        performance_notes.append(f"ORCHAV_PYGFX_MPC_LINE_CACHE_MB={args.pygfx_mpc_line_cache_mb}")
    return performance_notes


def workspace_resume_enabled(args: argparse.Namespace) -> bool:
    """Return whether automatic workspace resume should run for these arguments."""
    return args.benchmark == 0 and not args.render_frames and not args.no_resume and not args.author


def configure_cli_driven_frame_run(viz: Any, reporter: Any, enabled: bool) -> None:
    """Disable background work for benchmark or batch-render stepping."""
    viz._cli_driven_frame_run = enabled
    if not enabled:
        return
    viz.set_background_update_enabled(False)
    viz.use_preload_mode = False
    viz.cancel_startup_preload()
    viz.cancel_scheduled_update()
    reporter.note("Background updates and startup preload disabled for CLI-driven frame stepping")


def restore_startup_workspace(
    viz: Any,
    selection: Optional[WorkspaceSnapshot],
    reporter: Any,
) -> Optional[Path]:
    """Restore the workspace chosen before scenario startup without rescanning."""
    if selection is None or not hasattr(viz, "session_service"):
        return None
    try:
        restored = viz.session_service.load_session(
            selection,
            skip_camera=(selection.camera is not None),
        )
    except FileNotFoundError as exc:
        logger.warning("Workspace resume skipped: %s", exc)
        reporter.note(f"Workspace resume skipped: {exc}")
        return None
    if not restored:
        return None
    display_name = selection.scenario_name.replace("_", " ").replace("-", " ").title()
    reporter.note(f"Resumed workspace: {display_name}, frame {selection.frame}")
    return selection.path


def _benchmark_overrides_metadata(overrides: dict[str, Any]) -> dict[str, Any]:
    """Normalize benchmark state override values for JSON metadata output."""
    return {
        key: (
            sorted(value)
            if isinstance(value, frozenset)
            else list(value) if isinstance(value, tuple) else value
        )
        for key, value in sorted(overrides.items())
    }


def _viewport_metadata(viz: Any) -> dict[str, Any]:
    """Return resolved host and current logical/physical viewport dimensions."""

    mode = str(getattr(viz, "_viewport_mode", "detached"))
    width = int(getattr(getattr(viz, "renderer", None), "_width", 0) or 0)
    height = int(getattr(getattr(viz, "renderer", None), "_height", 0) or 0)
    ratio = 1.0
    host = getattr(viz, "_viewport_host", None) if mode == "embedded" else None
    physical_width: int | None = None
    physical_height: int | None = None
    if host is not None:
        size = host.canvas_parent.size()
        width = max(0, int(size.width()))
        height = max(0, int(size.height()))
        try:
            ratio = float(host.devicePixelRatioF())
        except (AttributeError, TypeError, ValueError):
            ratio = 1.0
    elif mode == "detached":
        layout = getattr(viz, "_window_layout", None)
        if layout is not None:
            width = max(0, int(layout.renderer_logical_width))
            height = max(0, int(layout.renderer_logical_height))
            ratio = float(layout.device_pixel_ratio)
            if ratio <= 0.0:
                ratio = 1.0
            physical_width = max(0, int(layout.renderer_physical_width))
            physical_height = max(0, int(layout.renderer_physical_height))
        else:
            container = getattr(getattr(viz, "renderer", None), "_container", None)
            try:
                ratio = float(container.devicePixelRatioF())
                if ratio <= 0.0:
                    ratio = 1.0
            except (AttributeError, TypeError, ValueError):
                ratio = 1.0
    if physical_width is None:
        physical_width = int(round(width * ratio))
    if physical_height is None:
        physical_height = int(round(height * ratio))
    return {
        "viewport_mode": mode,
        "viewport_logical_width": width,
        "viewport_logical_height": height,
        "viewport_device_pixel_ratio": ratio,
        "viewport_physical_width": physical_width,
        "viewport_physical_height": physical_height,
    }


def _close_renderer_and_quit(viz: Any, app: QApplication) -> None:
    """Run application teardown before ending a CLI-driven Qt event loop."""
    try:
        shutdown_visualizer(viz)
    finally:
        app.quit()


def _fail_cli_run(viz: Any, app: QApplication, message: str) -> None:
    """Terminate a timer-driven CLI run with a preserved nonzero failure."""

    failure = RuntimeError(message)
    viz._cli_run_failure = failure
    logger.error(message)
    try:
        shutdown_visualizer(viz)
    finally:
        exit_app = getattr(app, "exit", None)
        if callable(exit_app):
            exit_app(1)
        else:
            app.quit()


def _run_cli_timer_transaction(
    viz: Any,
    app: QApplication,
    timer: QTimer,
    label: str,
    operation: Any,
) -> None:
    """Convert exceptions from a Qt timer slot into one nonzero CLI exit."""

    try:
        operation()
    except Exception as exc:
        timer.stop()
        logger.exception("%s failed", label)
        _fail_cli_run(viz, app, f"{label} failed: {exc}")


def start_benchmark_mode(
    *,
    args: argparse.Namespace,
    app: QApplication,
    reporter: Any,
    viz: Any,
    benchmark_state_overrides: dict[str, Any],
) -> None:
    """Attach benchmark recorder and start auto-stepping."""
    output_path = Path(args.benchmark_output) if args.benchmark_output else None
    recorder = BenchmarkRecorder(
        n_frames=args.benchmark,
        n_warmup=args.benchmark_warmup,
        output_path=output_path,
    )
    recorder.set_metadata("renderer", args.renderer)
    recorder.set_metadata("scenario", args.scenario or "")
    for key, value in _viewport_metadata(viz).items():
        recorder.set_metadata(key, value)
    recorder.set_metadata("benchmark_present_mode", args.benchmark_present_mode)
    recorder.set_metadata("max_performance", bool(args.max_performance))
    recorder.set_metadata(
        "pygfx_mpc_line_cache_mb",
        read_env_int("ORCHAV_PYGFX_MPC_LINE_CACHE_MB", 0),
    )
    recorder.set_metadata(
        "canon_cache_mb",
        read_env_int("VIZ_CANON_CACHE_MB", 0),
    )
    if args.benchmark_state_json:
        recorder.set_metadata("benchmark_state_json", args.benchmark_state_json)
        recorder.set_metadata(
            "benchmark_state_overrides",
            _benchmark_overrides_metadata(benchmark_state_overrides),
        )

    viz.pipeline.benchmark_recorder = None
    setattr(viz.renderer, "_benchmark_present_mode", args.benchmark_present_mode)

    benchmark_plan = resolve_benchmark_steps(
        viz.total_animation_steps,
        getattr(viz, "frame_source", None),
    )
    benchmark_steps = benchmark_plan.steps
    recorder.set_metadata("benchmark_step_source", benchmark_plan.source)
    recorder.set_metadata("benchmark_step_count", len(benchmark_steps))
    if not benchmark_steps:
        raise RuntimeError("Benchmark mode requires at least one available frame")
    recorder.set_metadata(
        "benchmark_previsit_all_frames",
        bool(args.benchmark_previsit_all_frames),
    )

    if args.benchmark_previsit_all_frames and benchmark_steps:
        reporter.note(f"Benchmark previsit: warming {len(benchmark_steps)} steps")

        def _reserve_benchmark_cache(step_count: int) -> None:
            """Reserve cache space before the previsit warms every benchmark step."""
            cache_service = getattr(viz, "cache_service", None)
            if cache_service is not None and hasattr(cache_service, "ensure_frame_cache_capacity"):
                cache_service.ensure_frame_cache_capacity(step_count)
                recorder.set_metadata("benchmark_reserved_cache_steps", step_count)

        def _previsit_step(step: int) -> None:
            """Force a synchronous frame update during benchmark warmup."""
            viz.force_update_next_frame = True
            viz.update_frame(step)

        previsit_stats = previsit_benchmark_steps(
            benchmark_steps,
            _previsit_step,
            process_events=app.processEvents,
            reserve_cache=_reserve_benchmark_cache,
        )
        for key, value in previsit_stats.items():
            recorder.set_metadata(key, value)

        viz.force_update_next_frame = True
        viz.update_frame(benchmark_steps[0])
        app.processEvents()

    viz.pipeline.benchmark_recorder = recorder
    begin_telemetry = getattr(viz.renderer, "begin_benchmark_telemetry", None)
    if callable(begin_telemetry):
        begin_telemetry()
    bench_timer = QTimer()
    # Mirror interactive playback's serial event-loop boundary: one complete
    # frame transaction, then one newly queued callback. Blocking regimes also
    # honor a backend-native render-turn permit when the renderer provides one.
    bench_timer.setSingleShot(True)
    viz._benchmark_timer = bench_timer
    bench_step = [0]

    def _start_benchmark_timer() -> None:
        """Queue work only while application shutdown has not begun."""
        if not bool(getattr(viz, "_shutdown_started", False)):
            bench_timer.start(0)

    def _bench_tick_impl() -> None:
        """Advance benchmark frames and finalize recorder/process shutdown."""
        if bool(getattr(viz, "_shutdown_started", False)):
            return
        if recorder.is_done:
            bench_timer.stop()
            if getattr(viz, "scene_boot_duration_ms", None) is not None:
                recorder.set_metadata(
                    "scene_boot_duration_ms",
                    round(float(viz.scene_boot_duration_ms), 3),
                )
            startup_profile = viz.get_startup_timing_profile()
            if startup_profile:
                recorder.set_metadata("startup_timing_profile", startup_profile)
            stats_fn = getattr(viz.renderer, "get_runtime_stats", None)
            if callable(stats_fn):
                try:
                    stats = stats_fn() or {}
                    recorder.set_runtime_stats(stats)
                except Exception as exc:
                    logger.debug("Benchmark: failed to collect renderer runtime stats: %s", exc)
            result_path = recorder.finalize()
            logger.info("Benchmark complete. Results: %s", result_path)
            _close_renderer_and_quit(viz, app)
            return
        step = benchmark_steps[bench_step[0] % len(benchmark_steps)]
        viz.force_update_next_frame = True
        if not bool(viz.update_frame(step)):
            bench_timer.stop()
            _fail_cli_run(viz, app, f"Benchmark frame {step} was not accepted")
            return
        bench_step[0] += 1
        if args.benchmark_present_mode == "blocking":
            defer = getattr(viz.renderer, "defer_until_next_render_turn", None)
            if callable(defer):
                try:
                    if bool(defer(_start_benchmark_timer)):
                        return
                except (RuntimeError, AttributeError, TypeError) as exc:
                    logger.debug("Benchmark renderer-turn permit failed: %s", exc)
        _start_benchmark_timer()

    def _bench_tick() -> None:
        _run_cli_timer_transaction(
            viz,
            app,
            bench_timer,
            "Benchmark",
            _bench_tick_impl,
        )

    bench_timer.timeout.connect(_bench_tick)
    QTimer.singleShot(500, _start_benchmark_timer)
    reporter.note(f"Benchmark mode: {args.benchmark} frames")


def start_batch_render_mode(
    *, args: argparse.Namespace, app: QApplication, reporter: Any, viz: Any
):
    """Render all available frames as PNGs and quit."""
    render_plan = resolve_benchmark_steps(
        viz.total_animation_steps,
        getattr(viz, "frame_source", None),
    )
    render_steps = render_plan.steps
    if not render_steps:
        raise RuntimeError("Frame rendering requires at least one available frame")
    if not renderer_capabilities(viz.renderer).screenshot_export:
        raise RuntimeError("Selected renderer cannot export screenshots")

    output_dir = Path(args.render_frames)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_step = [0]
    render_total = len(render_steps)
    render_timer = QTimer()
    render_timer.setSingleShot(True)
    viz._batch_render_timer = render_timer

    def _start_render_timer() -> None:
        """Queue work only while application shutdown has not begun."""
        if not bool(getattr(viz, "_shutdown_started", False)):
            render_timer.start(0)

    def _render_tick_impl() -> None:
        """Render one frame per Qt timer tick so screenshots see fresh state."""
        if bool(getattr(viz, "_shutdown_started", False)):
            return
        render_index = render_step[0]
        if render_index >= render_total:
            render_timer.stop()
            logger.info("Batch render complete: %d frames in %s", render_total, output_dir)
            _close_renderer_and_quit(viz, app)
            return
        step = render_steps[render_index]
        viz.force_update_next_frame = True
        if not bool(viz.update_frame(step)):
            render_timer.stop()
            _fail_cli_run(viz, app, f"Frame {step} was not accepted for rendering")
            return
        app.processEvents()
        if bool(getattr(viz, "_shutdown_started", False)):
            return
        filepath = output_dir / f"frame_{step:04d}.png"
        if not bool(viz.renderer.export_screenshot(str(filepath))):
            render_timer.stop()
            _fail_cli_run(viz, app, f"Screenshot export failed for frame {step}")
            return
        logger.info("Rendered frame %d/%d: %s", render_index + 1, render_total, filepath)
        render_step[0] += 1
        _start_render_timer()

    def _render_tick() -> None:
        _run_cli_timer_transaction(
            viz,
            app,
            render_timer,
            "Batch render",
            _render_tick_impl,
        )

    render_timer.timeout.connect(_render_tick)
    QTimer.singleShot(500, _start_render_timer)
    reporter.note(f"Batch render mode: {render_total} frames -> {output_dir}")


def _complete_cli_startup(
    *,
    args: argparse.Namespace,
    app: QApplication,
    reporter: Any,
    viz: Any,
) -> None:
    """Initialize services and open CLI resources from inside the Qt loop."""
    workspace_selection = (
        select_startup_workspace(args.scenario) if workspace_resume_enabled(args) else None
    )
    pending_camera = workspace_selection.camera if workspace_selection is not None else None
    if pending_camera:
        reporter.note("Pre-read workspace camera for jump-free startup")

    viz._deferred_init(pending_camera=pending_camera)
    app.processEvents()
    if bool(getattr(viz, "_shutdown_started", False)):
        raise RuntimeError("Visualizer startup was cancelled during shutdown")

    if args.author:
        if args.scenario:
            viz.open_for_authoring(args.scenario)
        else:
            viz.new_authoring_scenario()
        return

    cli_driven_frame_run = args.benchmark > 0 or bool(args.render_frames)
    configure_cli_driven_frame_run(viz, reporter, cli_driven_frame_run)

    opened_scenario = False
    scenario_to_open = (
        args.scenario
        if args.scenario
        else (str(workspace_selection.scenario_root) if workspace_selection is not None else None)
    )
    if scenario_to_open:
        previous_cli_scenario_startup = bool(getattr(viz, "_explicit_cli_scenario_startup", False))
        viz._explicit_cli_scenario_startup = args.scenario is not None
        try:
            with reporter.task(f"Opening scenario {scenario_to_open}"):
                source_overrides: dict[str, Any] = {}
                if args.data_mode is not None:
                    source_overrides["data_mode_override"] = args.data_mode
                if args.grpc_port is not None:
                    source_overrides["grpc_port_override"] = args.grpc_port
                open_outcome = viz.open_scenario(
                    scenario_to_open,
                    pending_camera=pending_camera,
                    autorun_initial_frame=(
                        not cli_driven_frame_run and workspace_selection is None
                    ),
                    **source_overrides,
                )
        finally:
            viz._explicit_cli_scenario_startup = previous_cli_scenario_startup
        opened_scenario = bool(getattr(open_outcome, "succeeded", False))
        if not opened_scenario:
            message = getattr(open_outcome, "message", None) or (
                f"Scenario did not open: {scenario_to_open}"
            )
            reporter.note(message)
            if cli_driven_frame_run or args.scenario is not None:
                raise _CliScenarioOpenError(message)
        if workspace_selection is not None:
            outcome_root = getattr(open_outcome, "scenario_root", None)
            current_root = normalize_scenario_root(outcome_root)
            selected_root = normalize_scenario_root(workspace_selection.scenario_root)
            opened_scenario = opened_scenario and (
                current_root is not None
                and selected_root is not None
                and os.path.normcase(str(current_root)) == os.path.normcase(str(selected_root))
            )
            if not opened_scenario:
                logger.error(
                    "Startup did not activate the selected workspace scenario: "
                    "expected=%s actual=%s",
                    workspace_selection.scenario_root,
                    current_root,
                )
                reporter.note("Workspace resume skipped because its scenario did not open")
        if cli_driven_frame_run:
            viz.cancel_scheduled_update()
    else:
        reporter.note("No scenario provided on CLI")

    benchmark_state_overrides: dict[str, Any] = {}
    if args.benchmark_state_json:
        if not opened_scenario:
            raise RuntimeError("Benchmark state cannot be applied because the scenario failed")
        with reporter.task(f"Applying benchmark state {args.benchmark_state_json}"):
            benchmark_state_overrides = load_benchmark_state_overrides(args.benchmark_state_json)
            apply_benchmark_state_overrides(viz, benchmark_state_overrides)

    restored_workspace = (
        restore_startup_workspace(viz, workspace_selection, reporter) if opened_scenario else None
    )
    if (
        workspace_selection is not None
        and restored_workspace is None
        and opened_scenario
        and not cli_driven_frame_run
    ):
        # The selected restore suppressed scenario autorun. If the snapshot
        # becomes unusable after selection, still present the scenario once.
        viz.force_update_next_frame = True
        viz.schedule_update()

    if args.benchmark > 0:
        start_benchmark_mode(
            args=args,
            app=app,
            reporter=reporter,
            viz=viz,
            benchmark_state_overrides=benchmark_state_overrides,
        )

    if args.render_frames:
        start_batch_render_mode(args=args, app=app, reporter=reporter, viz=viz)


def _exec_with_deferred_cli_startup(
    *,
    args: argparse.Namespace,
    app: QApplication,
    reporter: Any,
    viz: Any,
) -> int:
    """Enter Qt first, then run startup while preserving callback exceptions."""
    startup_failure: list[tuple[type[BaseException], BaseException, Any]] = []

    def _start() -> None:
        """Run startup on the first normal Qt event-loop turn."""
        try:
            _complete_cli_startup(args=args, app=app, reporter=reporter, viz=viz)
        except BaseException:
            exc_type, exc, traceback = sys.exc_info()
            if exc_type is not None and exc is not None:
                startup_failure.append((exc_type, exc, traceback))
            try:
                _close_renderer_and_quit(viz, app)
            except Exception:
                logger.exception("CLI startup cleanup failed")

    QTimer.singleShot(0, _start)
    exit_code = int(app.exec())
    if startup_failure:
        _exc_type, exc, traceback = startup_failure[0]
        raise exc.with_traceback(traceback)
    cli_failure = getattr(viz, "_cli_run_failure", None)
    if isinstance(cli_failure, BaseException):
        raise cli_failure
    return exit_code


def run_visualizer_cli(
    visualizer_cls: Any,
    progress_reporter_cls: Any,
    argv: Optional[list[str]] = None,
) -> None:
    """Run the visualizer command-line startup path."""
    args = parse_cli_args(argv)
    performance_notes = apply_launch_environment(args)

    reporter = progress_reporter_cls(enabled=True)
    reporter.note("Bootstrapping Qt application")
    reporter.note(f"Using {args.renderer} renderer")
    for note in performance_notes:
        reporter.note(note)
    if args.enable_textures:
        reporter.note("Scene textures enabled for this launch")
    elif args.disable_textures:
        reporter.note("Scene textures disabled for this launch")
    if args.layout_profile != "auto":
        reporter.note(f"Using {args.layout_profile} window layout profile")
    reporter.note(f"Using {args.resolved_viewport_mode} renderer viewport")

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    apply_application_identity(app)

    viz = visualizer_cls(
        progress=reporter,
        renderer_type=args.renderer,
        layout_profile=args.layout_profile,
        viewport_mode=args.resolved_viewport_mode,
    )
    viz.show()
    app.processEvents()

    reporter.note("Starting GUI")
    try:
        exit_code = _exec_with_deferred_cli_startup(
            args=args,
            app=app,
            reporter=reporter,
            viz=viz,
        )
    except _CliScenarioOpenError as exc:
        logger.error("Visualizer startup aborted: %s", exc)
        exit_code = 1
    sys.exit(exit_code)
