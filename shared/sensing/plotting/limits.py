"""Shared view limits for interactive and exported sensing figures."""

from collections.abc import Sequence

import numpy as np

_ZOOM_MARGIN_FRACTION = 0.2  # Context on each side of the visible signal span.


def range_doppler_zoom_limits(
    power_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    dynamic_range_db: float = 50.0,
    extra_points: Sequence[tuple[float, float]] = (),
    range_padding_bins: float = 1.5,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Bound visible signal and overlays, with the GUI's 20-percent margin.

    This selects view limits only. It never crops, smooths or edits the arrays.
    ``power_db`` is indexed by Doppler row and range column. Neighboring range
    cells remain visible so the linked sampled range profile has context.
    """
    power = np.asarray(power_db)
    ranges = np.asarray(range_axis)
    velocities = np.asarray(velocity_axis)
    if power.shape != (velocities.size, ranges.size):
        raise ValueError("RD view limits require Doppler-by-range power")
    finite = np.isfinite(power)
    if not finite.any():
        return None
    rows, cols = np.nonzero(finite & (power >= np.max(power[finite]) - dynamic_range_db))
    if not cols.size:
        return None
    r_lo, r_hi = float(ranges[cols.min()]), float(ranges[cols.max()])
    v_lo, v_hi = float(velocities[rows.min()]), float(velocities[rows.max()])
    for r, v in extra_points:
        if np.isfinite(r) and np.isfinite(v):
            r_lo, r_hi = min(r_lo, r), max(r_hi, r)
            v_lo, v_hi = min(v_lo, v), max(v_hi, v)
    dr = abs(float(ranges[1] - ranges[0])) if ranges.size > 1 else 1.0
    dv = abs(float(velocities[1] - velocities[0])) if velocities.size > 1 else 1.0
    pad_r = max(r_hi - r_lo, dr) * _ZOOM_MARGIN_FRACTION
    if ranges.size > 1:
        pad_r = max(pad_r, range_padding_bins * dr)
    pad_v = max(v_hi - v_lo, dv) * _ZOOM_MARGIN_FRACTION
    return (r_lo - pad_r, r_hi + pad_r), (v_lo - pad_v, v_hi + pad_v)
