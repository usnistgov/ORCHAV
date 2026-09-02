"""Pipeline progress callback objects."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass(frozen=True)
class ProgressInfo:
    """Progress information passed to the ``on_step_complete`` callback.

    Attributes:
        step: Zero-based index of the step that just completed.
        total_steps: Total number of steps in the pipeline run.
        elapsed_s: Wall-clock seconds since the pipeline loop started.
        step_duration_s: Wall-clock seconds for this individual step.
    """

    step: int
    total_steps: int
    elapsed_s: float
    step_duration_s: float


def format_duration(seconds: float) -> str:
    """Return a compact wall-clock duration for CLI progress output."""
    seconds_i = max(0, int(seconds))
    hours = seconds_i // 3600
    minutes = (seconds_i % 3600) // 60
    secs = seconds_i % 60
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"


@dataclass
class StderrProgress:
    """Render simple in-place progress for long-running file generation."""

    first_step: int
    total_steps: int
    bar_width: int = 30
    label: str = ""
    start_time: float = field(default_factory=time.time)

    def update(self, step_idx: int) -> None:
        done = step_idx - self.first_step + 1
        elapsed = time.time() - self.start_time
        avg = (elapsed / done) if done > 0 else 0.0
        remaining = avg * max(0, self.total_steps - done)
        pct = 100.0 * done / max(1, self.total_steps)
        fill = int(self.bar_width * done / max(1, self.total_steps))
        bar = "#" * fill + "." * (self.bar_width - fill)
        prefix = f"{self.label} " if self.label else ""
        msg = (
            f"\r{prefix}[{bar}] {done}/{self.total_steps} {pct:5.1f}%"
            f" | elapsed {format_duration(elapsed)}"
            f" | eta {format_duration(remaining)}"
        )
        try:
            sys.stderr.write(msg)
            sys.stderr.flush()
        except OSError:
            pass

    @staticmethod
    def newline() -> None:
        try:
            sys.stderr.write("\n")
        except OSError:
            pass


@dataclass
class JsonlProgressReporter:
    """Write the stable machine-readable generator progress protocol.

    JSONL is written exclusively to ``stream`` (stdout by default). Normal
    application logging continues to use stderr, which lets subprocess clients
    parse progress without discarding diagnostic output.
    """

    first_step: int
    total_steps: int
    source_identity: Mapping[str, Any]
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    completed_steps: int = 0

    SCHEMA_VERSION = 1

    def run_started(self) -> None:
        """Emit the run metadata known before pipeline execution.

        A nonzero ``first_step`` identifies a fresh partial frame set. It never
        means that the generator will merge this run with previously published
        frames.
        """
        self._emit(
            "run_started",
            first_step=int(self.first_step),
            total_steps=int(self.total_steps),
            frame_set_mode=("fresh_partial" if int(self.first_step) > 0 else "fresh_full"),
            source_identity=dict(self.source_identity),
        )

    def step_completed(self, info: ProgressInfo) -> None:
        """Emit one completed output step."""
        self.completed_steps += 1
        # Actor-state normalization may adjust the step count after the CLI
        # constructs this reporter. The callback owns the authoritative value.
        self.total_steps = int(info.total_steps)
        self._emit(
            "step_completed",
            step=int(info.step),
            completed_steps=int(self.completed_steps),
            total_steps=int(info.total_steps),
            elapsed_s=round(float(info.elapsed_s), 6),
            step_duration_s=round(float(info.step_duration_s), 6),
        )

    def run_completed(self, output: str | None) -> None:
        """Emit successful process-level completion."""
        self._emit(
            "run_completed",
            completed_steps=int(self.completed_steps),
            total_steps=int(self.total_steps),
            output=output,
        )

    def run_failed(self, message: str, *, error_type: str | None = None) -> None:
        """Emit a terminal failure event before the CLI exits nonzero."""
        payload: dict[str, Any] = {
            "completed_steps": int(self.completed_steps),
            "total_steps": int(self.total_steps),
            "message": str(message),
        }
        if error_type:
            payload["error_type"] = str(error_type)
        self._emit("run_failed", **payload)

    def _emit(self, event: str, **payload: Any) -> None:
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "event": event,
            **payload,
        }
        try:
            self.stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            self.stream.write("\n")
            self.stream.flush()
        except OSError:
            # Match the human progress renderer: losing a display stream must
            # not corrupt or abort frame generation.
            pass
