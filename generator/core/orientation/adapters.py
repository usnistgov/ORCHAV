"""Thin conversions into canonical orientation-model parameters."""

from __future__ import annotations

import math


def smoothing_time_from_step_fraction(
    fraction: float,
    *,
    steps: int,
    duration_s: float,
) -> float:
    """Convert a per-step exponential fraction to canonical time seconds."""

    alpha = float(fraction)
    step_count = int(steps)
    duration = float(duration_s)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("smoothing fraction must be in (0, 1]")
    if alpha == 1.0 or step_count <= 1 or duration <= 0.0:
        return 0.0
    interval = duration / (step_count - 1)
    return -interval / math.log1p(-alpha)


__all__ = [
    "smoothing_time_from_step_fraction",
]
