"""Playback-mode values and pure cadence calculations.

Playback cadence controls when the application advances scenario frames. It
does not configure a renderer's camera-interaction or presentation scheduler.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

DEFAULT_FIXED_PLAYBACK_FPS = 30
MIN_PLAYBACK_FPS = 1
MAX_PLAYBACK_FPS = 240
DEFAULT_PLAYBACK_INTERVAL_MS = round(1000 / DEFAULT_FIXED_PLAYBACK_FPS)
MAX_QT_TIMER_INTERVAL_MS = 2_147_483_647
DEFAULT_PLAYBACK_CADENCE_WINDOW = 30
MAX_PLAYBACK_CADENCE_SAMPLES = 120


class PlaybackMode(str, Enum):
    """User-facing scenario-frame playback policies."""

    REAL_TIME = "real_time"
    FIXED_FPS = "fixed_fps"
    MAXIMUM = "maximum"


@dataclass
class PlaybackCadenceTracker:
    """Measure completed scenario-frame cadence independently of the renderer.

    Each sample is the wall-clock interval between two completed frame
    pipelines. Because both renderers finish their frame transaction before
    the pipeline reports completion, this gives the user-facing playback rate
    without interpreting backend-specific draw counters as playback FPS.
    """

    max_samples: int = MAX_PLAYBACK_CADENCE_SAMPLES
    _last_completion_s: float | None = None
    _intervals_s: list[float] = field(default_factory=list)

    def reset(self) -> None:
        """Start a fresh playback measurement window."""
        self._last_completion_s = None
        self._intervals_s.clear()

    def record_completion(self, completed_at_s: float | None = None) -> None:
        """Record one completed frame using a supplied or monotonic timestamp."""
        try:
            now = time.perf_counter() if completed_at_s is None else float(completed_at_s)
        except (TypeError, ValueError):
            return
        if not math.isfinite(now):
            return

        previous = self._last_completion_s
        if previous is None:
            self._last_completion_s = now
            return

        interval_s = now - previous
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            return
        self._last_completion_s = now
        self._intervals_s.append(interval_s)
        limit = max(1, int(self.max_samples))
        if len(self._intervals_s) > limit:
            self._intervals_s = self._intervals_s[-limit:]

    def frames_per_second(self, window: int = DEFAULT_PLAYBACK_CADENCE_WINDOW) -> float | None:
        """Return recent completed-frame cadence, or ``None`` before two frames."""
        if not self._intervals_s:
            return None
        size = max(1, int(window))
        intervals = self._intervals_s[-size:]
        elapsed_s = sum(intervals)
        if elapsed_s <= 0.0:
            return None
        return len(intervals) / elapsed_s

    def mean_interval_ms(self, window: int = DEFAULT_PLAYBACK_CADENCE_WINDOW) -> float | None:
        """Return the recent mean completed-frame interval in milliseconds."""
        fps = self.frames_per_second(window)
        return None if fps is None or fps <= 0.0 else 1000.0 / fps

    def percentile_interval_ms(
        self,
        percentile: float,
        window: int = DEFAULT_PLAYBACK_CADENCE_WINDOW,
    ) -> float | None:
        """Return a recent completed-update interval percentile in milliseconds."""
        if not self._intervals_s:
            return None
        try:
            requested = min(100.0, max(0.0, float(percentile)))
        except (TypeError, ValueError):
            return None
        size = max(1, int(window))
        values = sorted(self._intervals_s[-size:])
        if len(values) == 1:
            return values[0] * 1000.0
        rank = (len(values) - 1) * requested / 100.0
        lower = int(math.floor(rank))
        upper = min(lower + 1, len(values) - 1)
        fraction = rank - lower
        value_s = values[lower] * (1.0 - fraction) + values[upper] * fraction
        return value_s * 1000.0


def normalize_playback_mode(value: object) -> PlaybackMode:
    """Return a supported playback mode, defaulting to maximum throughput."""
    if isinstance(value, PlaybackMode):
        return value
    try:
        return PlaybackMode(str(value))
    except ValueError:
        return PlaybackMode.MAXIMUM


def fixed_fps_interval_ms(fps: object) -> int:
    """Convert a fixed playback FPS value to a positive Qt interval."""
    try:
        value = float(fps)
    except (TypeError, ValueError):
        value = float(DEFAULT_FIXED_PLAYBACK_FPS)
    if not math.isfinite(value) or value <= 0.0:
        value = float(DEFAULT_FIXED_PLAYBACK_FPS)
    return max(1, min(MAX_QT_TIMER_INTERVAL_MS, round(1000.0 / value)))


def real_time_interval_ms(
    duration_s: object,
    frame_count: object,
    *,
    frame_stride: object = 1,
) -> int:
    """Return the wall-clock interval that preserves uniform scenario time.

    The configured duration is the total playback time, so a scenario with 12
    frames over 12 seconds advances once per second. A larger playback stride
    uses a proportionally larger timer interval instead of accelerating
    simulation time.
    """
    try:
        duration = float(duration_s)
        count = int(frame_count)
        stride = max(1, int(frame_stride))
    except (TypeError, ValueError):
        return DEFAULT_PLAYBACK_INTERVAL_MS
    if not math.isfinite(duration) or duration <= 0.0 or count < 2:
        return DEFAULT_PLAYBACK_INTERVAL_MS
    interval_ms = (duration * 1000.0 / float(count)) * stride
    return max(1, min(MAX_QT_TIMER_INTERVAL_MS, round(interval_ms)))


def timestamp_interval_ms(current_s: object, next_s: object) -> int | None:
    """Return a positive interval between two frame timestamps, if valid."""
    try:
        delta_s = abs(float(next_s) - float(current_s))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(delta_s) or delta_s <= 0.0:
        return None
    return max(1, min(MAX_QT_TIMER_INTERVAL_MS, round(delta_s * 1000.0)))


def format_playback_fps(value: object) -> str | None:
    """Format a positive playback cadence without rounding it down to zero."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fps) or fps <= 0.0:
        return None
    if fps >= 10.0:
        return f"{fps:.0f}"
    if fps >= 0.1:
        return f"{fps:.1f}"
    if fps >= 0.01:
        return f"{fps:.2f}"
    return "<0.01"
