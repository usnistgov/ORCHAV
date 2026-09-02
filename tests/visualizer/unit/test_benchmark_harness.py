from __future__ import annotations

from visualizer.src.benchmarking.harness import (
    BenchmarkStepPlan,
    previsit_benchmark_steps,
    resolve_benchmark_steps,
)


class _FrameSource:
    def __init__(self, frames):
        self._frames = list(frames)

    def list_frames(self):
        return list(self._frames)


def test_resolve_benchmark_steps_prefers_frame_source():
    plan = resolve_benchmark_steps(10, _FrameSource([63, 127, 191]))
    assert isinstance(plan, BenchmarkStepPlan)
    assert plan.steps == [63, 127, 191]
    assert plan.source == "frame_source"


def test_resolve_benchmark_steps_falls_back_to_animation_range():
    plan = resolve_benchmark_steps(4, None)
    assert plan.steps == [0, 1, 2, 3]
    assert plan.source == "animation_range"


def test_previsit_benchmark_steps_updates_all_steps():
    visited = []
    processed = []
    reserved = []

    stats = previsit_benchmark_steps(
        [3, 5, 7],
        visited.append,
        process_events=lambda: processed.append(True),
        reserve_cache=reserved.append,
    )

    assert visited == [3, 5, 7]
    assert len(processed) == 3
    assert reserved == [3]
    assert stats["previsit_step_count"] == 3.0
    assert stats["previsit_wall_ms"] >= 0.0
    assert stats["previsit_avg_ms"] >= 0.0
