#!/usr/bin/env python3
"""Measure rendercanvas FPS caps without importing ORCHAV application code.

The probe uses a small, empty pygfx scene so the measured cadence primarily
reflects the Qt/rendercanvas scheduler and presentation path. It intentionally
does not use ORCHAV's renderer, animation controller, or benchmark harness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pygfx as gfx
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from rendercanvas.qt import QRenderWidget


def _default_present_method() -> str:
    """Return the platform-aware implicit ORCHAV presentation method."""
    return "screen" if sys.platform.startswith("win") else "auto"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the standalone scheduler probe."""
    parser = argparse.ArgumentParser(
        description="Measure actual pygfx/rendercanvas callback FPS at several caps.",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        default=["30", "60", "120", "monitor"],
        help="FPS caps to test; values may be numbers or 'monitor'.",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--warmup", type=float, default=0.75)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--vsync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable canvas vsync (disabled by default to isolate the FPS cap).",
    )
    parser.add_argument(
        "--present-method",
        choices=["screen", "bitmap", "auto"],
        default=_default_present_method(),
        help=(
            "Canvas presentation method. The default matches ORCHAV: 'screen' on "
            "Windows and rendercanvas 'auto' elsewhere; explicit values force or "
            "delegate to the requested path."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _finite_refresh_rate(app: QApplication) -> float:
    """Return the primary monitor refresh rate with a conservative fallback."""
    screen = app.primaryScreen()
    try:
        refresh_hz = float(screen.refreshRate()) if screen is not None else 60.0
    except (AttributeError, TypeError, ValueError, RuntimeError):
        refresh_hz = 60.0
    if not math.isfinite(refresh_hz) or refresh_hz < 24.0 or refresh_hz > 480.0:
        return 60.0
    return refresh_hz


def _size(value: Any) -> list[float] | None:
    """Normalize an optional two-element size for JSON output."""
    try:
        return [float(value[0]), float(value[1])]
    except (IndexError, TypeError, ValueError):
        return None


def run_case(
    app: QApplication,
    *,
    label: str,
    cap_hz: float,
    duration_s: float,
    warmup_s: float,
    width: int,
    height: int,
    vsync: bool,
    present_method: str,
) -> dict[str, Any]:
    """Run one continuous-scheduler case and return measured cadence."""
    canvas_kwargs: dict[str, Any] = {
        "update_mode": "continuous",
        "min_fps": 0.0,
        "max_fps": float(cap_hz),
        "vsync": bool(vsync),
    }
    if present_method != "auto":
        canvas_kwargs["present_method"] = present_method
    canvas = QRenderWidget(**canvas_kwargs)
    canvas.setWindowTitle(f"pygfx FPS probe: {label}")
    canvas.resize(int(width), int(height))
    canvas.show()
    app.processEvents()

    renderer = gfx.WgpuRenderer(
        canvas,
        pixel_scale=1.0,
        ppaa="none",
        sort_objects=False,
    )
    actual_present_method = "screen" if canvas._present_to_screen else "bitmap"
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(60.0, width / max(height, 1))

    started_at = time.perf_counter()
    record_after = started_at + max(0.0, float(warmup_s))
    callback_times: list[float] = []

    def draw_frame() -> None:
        """Render one empty frame and record callback start time after warmup."""
        now = time.perf_counter()
        if now >= record_after:
            callback_times.append(now)
        renderer.render(scene, camera, flush=True)

    canvas.request_draw(draw_frame)
    loop = QEventLoop()
    total_runtime_ms = max(1, int(round((warmup_s + duration_s) * 1000.0)))
    QTimer.singleShot(total_runtime_ms, loop.quit)
    loop.exec()

    logical_size = _size(canvas.get_logical_size())
    physical_size = _size(canvas.get_physical_size())
    internal_size = _size(renderer.physical_size)
    pixel_ratio = float(canvas.get_pixel_ratio())
    renderer_pixel_scale = float(renderer.pixel_scale)

    canvas.close()
    canvas.deleteLater()
    app.processEvents()

    intervals_ms = [
        (current - previous) * 1000.0
        for previous, current in zip(callback_times, callback_times[1:])
    ]
    elapsed_s = callback_times[-1] - callback_times[0] if len(callback_times) > 1 else 0.0
    actual_fps = (len(callback_times) - 1) / elapsed_s if elapsed_s > 0.0 else 0.0

    return {
        "label": label,
        "requested_cap_hz": float(cap_hz),
        "vsync": bool(vsync),
        "present_method_requested": present_method,
        "present_method": actual_present_method,
        "callback_count": len(callback_times),
        "actual_callback_fps": round(actual_fps, 3),
        "mean_interval_ms": (round(statistics.fmean(intervals_ms), 3) if intervals_ms else None),
        "median_interval_ms": (round(statistics.median(intervals_ms), 3) if intervals_ms else None),
        "min_interval_ms": round(min(intervals_ms), 3) if intervals_ms else None,
        "max_interval_ms": round(max(intervals_ms), 3) if intervals_ms else None,
        "logical_size": logical_size,
        "physical_size": physical_size,
        "internal_size": internal_size,
        "canvas_pixel_ratio": pixel_ratio,
        "renderer_pixel_scale": renderer_pixel_scale,
    }


def main() -> None:
    """Run requested cap cases and print/write a JSON report."""
    args = parse_args()
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    if args.warmup < 0.0:
        raise SystemExit("--warmup must be non-negative")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")

    app = QApplication.instance() or QApplication([])
    monitor_hz = _finite_refresh_rate(app)

    cases: list[tuple[str, float]] = []
    for raw in args.caps:
        label = str(raw).strip().lower()
        if label == "monitor":
            cases.append(("monitor", monitor_hz))
            continue
        try:
            cap_hz = float(label)
        except ValueError as exc:
            raise SystemExit(f"Invalid cap {raw!r}; use a positive number or 'monitor'") from exc
        if not math.isfinite(cap_hz) or cap_hz <= 0.0:
            raise SystemExit(f"Invalid cap {raw!r}; cap must be positive")
        cases.append((label, cap_hz))

    report = {
        "monitor_refresh_hz": monitor_hz,
        "duration_s": float(args.duration),
        "warmup_s": float(args.warmup),
        "present_method": args.present_method,
        "cases": [
            run_case(
                app,
                label=label,
                cap_hz=cap_hz,
                duration_s=float(args.duration),
                warmup_s=float(args.warmup),
                width=int(args.width),
                height=int(args.height),
                vsync=bool(args.vsync),
                present_method=str(args.present_method),
            )
            for label, cap_hz in cases
        ],
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
