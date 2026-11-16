"""Optional wall-clock measurements for sensing processing stages.

``SensingStageProfiler`` accumulates milliseconds by stage name and attaches
them to the emitted payload's timing metadata when profiling is enabled.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def sensing_profiling_enabled() -> bool:
    """Return True when per-stage sensing profiling is enabled."""
    raw = os.getenv("ORCHAV_SENSING_PROFILE", "")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


class SensingStageProfiler:
    """Collect per-stage sensing timings with near-zero disabled overhead."""

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = sensing_profiling_enabled() if enabled is None else bool(enabled)
        self._stage_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Accumulate elapsed time under ``name``, including failed stage calls."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._stage_ms[name] = self._stage_ms.get(name, 0.0) + elapsed_ms

    def attach(self, sensing_data: dict[str, Any], *, total_ms: float | None = None) -> None:
        """Add millisecond stage totals to the payload in place when enabled."""
        if not self.enabled:
            return

        stage_profile = {
            name: round(float(elapsed_ms), 3) for name, elapsed_ms in sorted(self._stage_ms.items())
        }
        if total_ms is not None:
            stage_profile["total_strategy"] = round(float(total_ms), 3)

        sensing_data.setdefault("timing", {})["stage_profile_ms"] = stage_profile
