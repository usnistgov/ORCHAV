"""Benchmark recorder for measuring per-frame, per-stage pipeline latency.

Activated only when the visualizer is launched with ``--benchmark N``.
When not active, the recorder is ``None`` and zero overhead is incurred.

Usage from the CLI::

    python -m visualizer --benchmark 50 --scenario path/to/scenario.yaml

After *N* frames have been processed the recorder writes a JSON file to the
working directory and the application exits.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _FrameRecord:
    """Timing for a single pipeline frame."""

    step: int = 0
    prepare_step_ms: float = 0.0
    load_ms: float = 0.0
    viewmodel_ms: float = 0.0
    render_ms: float = 0.0
    total_before_end_ms: float = 0.0
    total_ms: float = 0.0
    n_mpc_points: int = 0
    n_mpc_lines: int = 0
    breakdown_ms: Dict[str, float] = field(default_factory=dict)
    breakdown_bytes: Dict[str, float] = field(default_factory=dict)


class BenchmarkRecorder:
    """Collects per-frame per-stage timings during a benchmark run.

    Parameters
    ----------
    n_frames:
        Total number of frames to benchmark.
    n_warmup:
        Number of initial frames whose timings are discarded.
    output_path:
        Where to write the JSON results file.  Defaults to
        ``benchmark_results.json`` in the current directory.
    """

    def __init__(
        self,
        n_frames: int,
        n_warmup: int = 5,
        output_path: Optional[Path] = None,
    ) -> None:
        """Initialize benchmark counters, output path, and metadata stores."""
        self.n_frames = n_frames
        self.n_warmup = n_warmup
        self.output_path = output_path or Path("benchmark_render_results.json")

        self._records: List[_FrameRecord] = []
        self._current: Optional[_FrameRecord] = None
        self._frame_count: int = 0
        self._wall_start: Optional[float] = None
        self._done: bool = False
        self._pending_prepare_step_ms: float = 0.0

        # Metadata collected once at start
        self._meta: Dict[str, Any] = {}
        self._runtime_stats: Dict[str, Any] = {}

    # Public API called from FramePipeline

    def begin_frame(self, step: int) -> None:
        """Call at the very start of ``FramePipeline.update``."""
        if self._done:
            return
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        self._current = _FrameRecord(step=step)
        if self._pending_prepare_step_ms > 0.0:
            self._current.prepare_step_ms = self._pending_prepare_step_ms
            self._pending_prepare_step_ms = 0.0

    def record_prepare_step(self, elapsed_ms: float) -> None:
        """Record frame-step preparation time before or during a frame update."""
        if self._current is not None:
            self._current.prepare_step_ms = elapsed_ms
        else:
            self._pending_prepare_step_ms = float(elapsed_ms)

    def record_load(self, elapsed_ms: float) -> None:
        """Record frame-source load time for the current benchmark frame."""
        if self._current is not None:
            self._current.load_ms = elapsed_ms

    def record_viewmodel(self, elapsed_ms: float) -> None:
        """Record ViewModel derivation time for the current benchmark frame."""
        if self._current is not None:
            self._current.viewmodel_ms = elapsed_ms

    def record_render(self, elapsed_ms: float) -> None:
        """Record renderer apply/update time before end-frame submission."""
        if self._current is not None:
            self._current.render_ms = elapsed_ms

    def record_total_before_end(self, elapsed_ms: float) -> None:
        """Record total frame time before renderer end-frame work is included."""
        if self._current is not None:
            self._current.total_before_end_ms = elapsed_ms

    def record_geometry(self, n_points: int, n_lines: int) -> None:
        """Record MPC geometry size for the current benchmark frame."""
        if self._current is not None:
            self._current.n_mpc_points = n_points
            self._current.n_mpc_lines = n_lines

    def record_breakdown(self, name: str, elapsed_ms: float) -> None:
        """Record one named millisecond subphase for the current frame."""
        if self._current is not None:
            self._current.breakdown_ms[name] = float(elapsed_ms)

    def record_breakdowns(self, breakdowns: Optional[Dict[str, Any]]) -> None:
        """Record many named millisecond subphases, ignoring non-numeric values."""
        if self._current is None or not breakdowns:
            return
        for name, value in breakdowns.items():
            try:
                self._current.breakdown_ms[str(name)] = float(value)
            except (TypeError, ValueError):
                continue

    def record_breakdown_bytes(self, name: str, value: float) -> None:
        """Record one named byte-count subphase for the current frame."""
        if self._current is not None:
            self._current.breakdown_bytes[name] = float(value)

    def record_breakdown_bytes_many(self, breakdowns: Optional[Dict[str, Any]]) -> None:
        """Record many named byte-count subphases, ignoring non-numeric values."""
        if self._current is None or not breakdowns:
            return
        for name, value in breakdowns.items():
            try:
                self._current.breakdown_bytes[str(name)] = float(value)
            except (TypeError, ValueError):
                continue

    def end_frame(self, total_ms: float) -> None:
        """Call at the end of ``FramePipeline.update`` (completed frames only)."""
        if self._current is None or self._done:
            return
        self._current.total_ms = total_ms
        self._records.append(self._current)
        self._current = None
        self._frame_count += 1

        warmup_tag = " (warmup)" if self._frame_count <= self.n_warmup else ""
        logger.info(
            "Benchmark frame %d/%d: total=%.1f ms%s",
            self._frame_count,
            self.n_frames,
            total_ms,
            warmup_tag,
        )

    @property
    def is_done(self) -> bool:
        """Return True once the configured benchmark frame count is reached."""
        return self._frame_count >= self.n_frames

    def set_metadata(self, key: str, value: Any) -> None:
        """Attach one benchmark metadata field to the output JSON."""
        self._meta[key] = value

    def set_runtime_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """Store normalized renderer runtime stats in their dedicated block."""
        self._runtime_stats = dict(stats or {})

    # Finalization

    def finalize(self) -> Path:
        """Write results to JSON and return the path."""
        self._done = True
        wall_elapsed = (time.perf_counter() - self._wall_start) if self._wall_start else 0.0
        wall_update_rate_hz = float(self._frame_count) / wall_elapsed if wall_elapsed > 0.0 else 0.0

        warmup = self._records[: self.n_warmup]
        timed = self._records[self.n_warmup :]
        summary = self._build_summary(timed)

        result: Dict[str, Any] = {
            "metadata": {
                "n_frames": self.n_frames,
                "n_warmup": self.n_warmup,
                "n_timed": len(timed),
                "wall_time_s": round(wall_elapsed, 3),
                # End-to-end scheduler throughput, including renderer permits
                # and cadence waits. This is not physical display FPS.
                "wall_update_rate_hz": round(wall_update_rate_hz, 3),
                "platform": platform.platform(),
                "python": platform.python_version(),
                **self._meta,
            },
            "runtime_stats": dict(self._runtime_stats),
            "summary": summary,
            "warmup": [self._record_to_dict(r) for r in warmup],
            "timed": [self._record_to_dict(r) for r in timed],
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("Benchmark results written to %s", self.output_path)
        return self.output_path

    @staticmethod
    def _record_to_dict(rec: _FrameRecord) -> Dict[str, Any]:
        """Convert a frame timing record to stable JSON-friendly fields."""
        return {
            "step": rec.step,
            "prepare_step_ms": round(rec.prepare_step_ms, 3),
            "load_ms": round(rec.load_ms, 3),
            "viewmodel_ms": round(rec.viewmodel_ms, 3),
            "render_ms": round(rec.render_ms, 3),
            "total_before_end_ms": round(rec.total_before_end_ms, 3),
            "total_ms": round(rec.total_ms, 3),
            "n_mpc_points": rec.n_mpc_points,
            "n_mpc_lines": rec.n_mpc_lines,
            "breakdown_ms": {
                key: round(float(value), 3) for key, value in sorted(rec.breakdown_ms.items())
            },
            "breakdown_bytes": {
                key: int(value) for key, value in sorted(rec.breakdown_bytes.items())
            },
        }

    @staticmethod
    def _pctl(values: list[float], percentile: float) -> float:
        """Return an interpolated percentile for a list of timings."""
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        sorted_vals = sorted(float(v) for v in values)
        rank = (len(sorted_vals) - 1) * (percentile / 100.0)
        lo = int(rank)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = rank - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    @classmethod
    def _build_summary(cls, timed: list[_FrameRecord]) -> Dict[str, Any]:
        """Build aggregate timing and geometry statistics for timed frames."""
        if not timed:
            return {}
        prepare = [r.prepare_step_ms for r in timed]
        load = [r.load_ms for r in timed]
        viewmodel = [r.viewmodel_ms for r in timed]
        render = [r.render_ms for r in timed]
        total_before_end = [r.total_before_end_ms for r in timed]
        total = [r.total_ms for r in timed]
        avg_prepare = statistics.fmean(prepare)
        avg_load = statistics.fmean(load)
        avg_viewmodel = statistics.fmean(viewmodel)
        avg_render = statistics.fmean(render)
        avg_total_before_end = statistics.fmean(total_before_end)
        avg_total = statistics.fmean(total)
        p95_prepare = cls._pctl(prepare, 95.0)
        p95_load = cls._pctl(load, 95.0)
        p95_viewmodel = cls._pctl(viewmodel, 95.0)
        p95_render = cls._pctl(render, 95.0)
        p95_total_before_end = cls._pctl(total_before_end, 95.0)
        p95_total = cls._pctl(total, 95.0)

        def fps_equivalent(ms: float) -> float:
            """Convert milliseconds-per-frame to FPS-like reciprocal units."""
            return round(1000.0 / ms, 3) if ms > 0.0 else 0.0

        summary = {
            "avg_prepare_step_ms": round(avg_prepare, 3),
            "avg_load_ms": round(avg_load, 3),
            "avg_viewmodel_ms": round(avg_viewmodel, 3),
            "avg_render_ms": round(avg_render, 3),
            "avg_total_before_end_ms": round(avg_total_before_end, 3),
            "avg_total_ms": round(avg_total, 3),
            "p95_prepare_step_ms": round(p95_prepare, 3),
            "p95_load_ms": round(p95_load, 3),
            "p95_viewmodel_ms": round(p95_viewmodel, 3),
            "p95_render_ms": round(p95_render, 3),
            "p95_total_before_end_ms": round(p95_total_before_end, 3),
            "p95_total_ms": round(p95_total, 3),
            "avg_render_fps_equiv": fps_equivalent(avg_render),
            "avg_total_before_end_fps_equiv": fps_equivalent(avg_total_before_end),
            "avg_total_fps_equiv": fps_equivalent(avg_total),
            "p95_render_fps_equiv": fps_equivalent(p95_render),
            "p95_total_before_end_fps_equiv": fps_equivalent(p95_total_before_end),
            "p95_total_fps_equiv": fps_equivalent(p95_total),
            "avg_mpc_points": round(statistics.fmean([r.n_mpc_points for r in timed]), 3),
            "avg_mpc_lines": round(statistics.fmean([r.n_mpc_lines for r in timed]), 3),
        }
        breakdown_keys = sorted({key for rec in timed for key in rec.breakdown_ms})
        if breakdown_keys:
            avg_breakdown_ms: Dict[str, float] = {}
            p95_breakdown_ms: Dict[str, float] = {}
            for key in breakdown_keys:
                values = [float(rec.breakdown_ms.get(key, 0.0)) for rec in timed]
                avg_breakdown_ms[key] = round(statistics.fmean(values), 3)
                p95_breakdown_ms[key] = round(cls._pctl(values, 95.0), 3)
            summary["avg_breakdown_ms"] = avg_breakdown_ms
            summary["p95_breakdown_ms"] = p95_breakdown_ms
        byte_keys = sorted({key for rec in timed for key in rec.breakdown_bytes})
        if byte_keys:
            avg_breakdown_bytes: Dict[str, float] = {}
            p95_breakdown_bytes: Dict[str, float] = {}
            for key in byte_keys:
                values = [float(rec.breakdown_bytes.get(key, 0.0)) for rec in timed]
                avg_breakdown_bytes[key] = round(statistics.fmean(values), 3)
                p95_breakdown_bytes[key] = round(cls._pctl(values, 95.0), 3)
            summary["avg_breakdown_bytes"] = avg_breakdown_bytes
            summary["p95_breakdown_bytes"] = p95_breakdown_bytes
        return summary
