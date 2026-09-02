"""Helpers for cold/warm/play benchmark regime orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkRegimeSpec:
    """One visualizer benchmark regime."""

    name: str
    frames: int
    warmup: int
    previsit_all_frames: bool
    present_mode: str
    description: str


def build_standard_regimes(
    *,
    frames: int,
    warmup: int,
    play_frames: int,
) -> list[BenchmarkRegimeSpec]:
    """Return the standard cold/warm/play benchmark regime set."""
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if play_frames < 1:
        raise ValueError("play_frames must be >= 1")
    return [
        BenchmarkRegimeSpec(
            name="cold",
            frames=frames,
            warmup=warmup,
            previsit_all_frames=False,
            present_mode="request",
            description=(
                "Cold state-update throughput; non-blocking draw requests may coalesce "
                "and are not a visible-smoothness measurement"
            ),
        ),
        BenchmarkRegimeSpec(
            name="warm",
            frames=frames,
            warmup=warmup,
            previsit_all_frames=True,
            present_mode="request",
            description=(
                "Warm state-update throughput; non-blocking draw requests may coalesce "
                "and are not a visible-smoothness measurement"
            ),
        ),
        BenchmarkRegimeSpec(
            name="play",
            frames=play_frames,
            warmup=0,
            previsit_all_frames=False,
            present_mode="blocking",
            description=(
                "Synchronous renderer microtrace; each state services one backend draw "
                "callback/event pump before advancing, without claiming physical present"
            ),
        ),
    ]


def summarize_benchmark_json(path: str | Path) -> dict[str, Any]:
    """Return a compact summary for one benchmark JSON file."""
    benchmark_path = Path(path)
    with benchmark_path.open() as handle:
        payload = json.load(handle)

    metadata = payload.get("metadata", {})
    runtime = payload.get("runtime_stats", {})
    summary = payload.get("summary", {})
    timed = payload.get("timed") or []
    first_timed = timed[0] if timed else {}
    avg_breakdown = summary.get("avg_breakdown_ms", {})
    p95_breakdown = summary.get("p95_breakdown_ms", {})
    benchmark_frames = int(metadata.get("n_frames", 0) or 0)
    benchmark_draw_callbacks = runtime.get("benchmark_draw_callbacks")
    blocking_frames = int(runtime.get("blocking_frame_count", 0) or 0)
    blocking_draw_callbacks = runtime.get("blocking_force_draw_callbacks")
    frame_submissions = int(runtime.get("benchmark_frame_submissions", 0) or 0)
    redraw_pump_attempts = int(runtime.get("benchmark_redraw_pump_attempts", 0) or 0)
    redraw_pump_alive = int(runtime.get("benchmark_redraw_pump_alive", 0) or 0)
    redraw_pumps_per_submission = (
        float(redraw_pump_alive) / float(frame_submissions) if frame_submissions > 0 else None
    )
    draw_callbacks_per_frame = None
    if (
        metadata.get("benchmark_present_mode") == "blocking"
        and blocking_frames > 0
        and blocking_draw_callbacks is not None
    ):
        draw_callbacks_per_frame = float(blocking_draw_callbacks) / float(blocking_frames)
    elif benchmark_frames > 0 and benchmark_draw_callbacks is not None:
        draw_callbacks_per_frame = float(benchmark_draw_callbacks) / float(benchmark_frames)

    return {
        "path": str(benchmark_path),
        "n_timed": metadata.get("n_timed", 0),
        "wall_update_rate_hz": metadata.get("wall_update_rate_hz"),
        "benchmark_present_mode": metadata.get("benchmark_present_mode", "blocking"),
        "avg_prepare_step_ms": summary.get("avg_prepare_step_ms"),
        "avg_total_ms": summary.get("avg_total_ms"),
        "avg_total_before_end_ms": summary.get("avg_total_before_end_ms")
        or avg_breakdown.get("total_before_end_ms"),
        "avg_end_frame_update_ms": avg_breakdown.get("end_frame_update_ms"),
        "avg_force_draw_ms": avg_breakdown.get("force_draw_ms"),
        "avg_draw_callback_total_ms": avg_breakdown.get("draw_callback_total_ms"),
        "avg_renderer_submit_ms": avg_breakdown.get("renderer_submit_ms"),
        "avg_canvas_present_residual_ms": avg_breakdown.get("canvas_present_residual_ms"),
        "renderer_content_size": runtime.get("renderer_content_size"),
        "renderer_window_size": runtime.get("renderer_window_size"),
        "renderer_internal_size": runtime.get("renderer_internal_size"),
        "renderer_pixel_scale": runtime.get("renderer_pixel_scale"),
        "renderer_pixel_ratio": runtime.get("renderer_pixel_ratio"),
        "benchmark_draw_callbacks": benchmark_draw_callbacks,
        "blocking_frame_count": blocking_frames,
        "blocking_force_draw_callbacks": blocking_draw_callbacks,
        "draw_callbacks_per_frame": draw_callbacks_per_frame,
        "benchmark_frame_submissions": frame_submissions,
        "benchmark_redraw_pump_attempts": redraw_pump_attempts,
        "benchmark_redraw_pump_alive": redraw_pump_alive,
        "redraw_pumps_per_submission": redraw_pumps_per_submission,
        "avg_request_draw_ms": avg_breakdown.get("request_draw_ms"),
        "avg_geometry_update_ms": avg_breakdown.get("geometry_update_ms"),
        "avg_canonical_lookup_ms": avg_breakdown.get("canonical_lookup_ms"),
        "avg_targets_ms": avg_breakdown.get("targets_ms"),
        "p95_total_ms": summary.get("p95_total_ms"),
        "p95_total_before_end_ms": summary.get("p95_total_before_end_ms")
        or p95_breakdown.get("total_before_end_ms"),
        "avg_viewmodel_ms": summary.get("avg_viewmodel_ms"),
        "avg_render_ms": summary.get("avg_render_ms"),
        "startup_to_first_frame_ms": runtime.get("startup_to_first_frame_ms"),
        "avg_update_to_present_ms": runtime.get("avg_update_to_present_ms"),
        "benchmark_previsit_all_frames": metadata.get("benchmark_previsit_all_frames", False),
        "previsit_wall_ms": metadata.get("previsit_wall_ms"),
        "first_timed_total_ms": first_timed.get("total_ms"),
        "first_timed_step": first_timed.get("step"),
    }
