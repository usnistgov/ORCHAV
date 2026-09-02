"""Renderer-neutral per-frame MPC statistics and order histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameStats:
    """Per-frame channel statistics computed from canonical MPC data.

    Aggregates path counts, interaction-order distribution, delay/path-loss
    ranges, and optional advanced channel metrics. Optional fields are ``None``
    when they were not requested or their required path metrics are absent.

    Attributes:
        total_paths: Total number of multipath components in the frame.
        orders_hist: Histogram mapping interaction order (bounce count) to
            path count.
        delay_range_ns: (min, max) propagation delay in nanoseconds, or
            ``None`` when delay data is unavailable.
        path_loss_range: (min, max) path loss in dB, or ``None`` when
            path-loss data is unavailable.
        delay_spread_ns: RMS delay spread in nanoseconds.
        angular_spread_deg: Power-weighted wrapped RMS AoD azimuth spread in
            degrees.
        signal_strength_distribution: ``(bin_centers_dbm, path_counts)``.
        binned_power_delay_profile: Nearby paths combined as ``(delays,
            summed_powers_db)`` arrays, in nanoseconds and dB, without phase
            assumptions.
        snr_distribution: ``(bin_centers_db, path_counts)``.
    """

    total_paths: int
    orders_hist: Dict[int, int]
    delay_range_ns: Optional[Tuple[float, float]] = None
    path_loss_range: Optional[Tuple[float, float]] = None

    delay_spread_ns: Optional[float] = None
    angular_spread_deg: Optional[float] = None
    signal_strength_distribution: Optional[Tuple[np.ndarray, np.ndarray]] = None
    binned_power_delay_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None
    snr_distribution: Optional[Tuple[np.ndarray, np.ndarray]] = None


def compute_orders_hist(reflection_order_counts: Any) -> Dict[int, int]:
    """Build an interaction-order histogram from raw bounce counts.

    Args:
        reflection_order_counts: Either a ``dict`` mapping order to count, or
            an iterable whose index is the order and value is the count.
            Values are coerced to ``int``; zero-count orders are omitted.

    Returns:
        Histogram mapping each non-zero interaction order to its path count.
    """
    hist: Dict[int, int] = {}

    def coerce_positive_count(value: Any) -> int | None:
        try:
            count = int(value) if isinstance(value, (int, np.integer)) else int(value.item())
        except (AttributeError, TypeError, ValueError):
            return None
        return count if count > 0 else None

    if isinstance(reflection_order_counts, dict):
        for order, count in reflection_order_counts.items():
            value = coerce_positive_count(count)
            if value is None:
                continue
            try:
                hist[int(order)] = value
            except (TypeError, ValueError):
                continue
    elif hasattr(reflection_order_counts, "__iter__"):
        for idx, count in enumerate(reflection_order_counts):
            value = coerce_positive_count(count)
            if value is not None:
                hist[idx] = value
    return hist
