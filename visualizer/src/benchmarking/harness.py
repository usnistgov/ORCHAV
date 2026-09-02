"""Helpers for benchmark stepping and warm-cache previsit runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class BenchmarkStepPlan:
    """Resolved step sequence for benchmark or render-frame runs."""

    steps: list[int]
    source: str


def resolve_benchmark_steps(total_animation_steps: int, frame_source: Any) -> BenchmarkStepPlan:
    """Resolve the ordered step list for benchmark-style frame stepping."""
    steps = list(range(total_animation_steps))
    source = "animation_range"
    if frame_source is not None and hasattr(frame_source, "list_frames"):
        try:
            available_frames = frame_source.list_frames()
        except (OSError, ValueError):
            available_frames = None
        if available_frames:
            steps = list(available_frames)
            source = "frame_source"
    return BenchmarkStepPlan(steps=steps, source=source)


def previsit_benchmark_steps(
    steps: Sequence[int],
    update_step: Callable[[int], None],
    *,
    process_events: Callable[[], None] | None = None,
    reserve_cache: Callable[[int], None] | None = None,
) -> dict[str, float]:
    """Walk all steps once before a benchmarked pass to warm in-process caches."""
    if not steps:
        return {
            "previsit_step_count": 0.0,
            "previsit_wall_ms": 0.0,
            "previsit_avg_ms": 0.0,
        }

    if reserve_cache is not None:
        reserve_cache(len(steps))

    wall_start = time.perf_counter()
    for step in steps:
        update_step(int(step))
        if process_events is not None:
            process_events()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return {
        "previsit_step_count": float(len(steps)),
        "previsit_wall_ms": round(wall_ms, 3),
        "previsit_avg_ms": round(wall_ms / max(len(steps), 1), 3),
    }
