"""Selection of retained cell centres; no interpolation or solver operations."""

import numpy as np


def coverage_roi_mask(axes, bounds=None):
    """Return a ZYX mask for inclusive XYZ centre-coordinate bounds.

    Bounds select recorded samples, not intersections of a continuous field
    with an arbitrary box. A reversed, nonfinite or empty selection is rejected.
    """
    if bounds is None:
        return np.ones(tuple(len(a) for a in axes[::-1]), dtype=bool)
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (3, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("Region requires finite minimum ≤ maximum for X, Y and Z")
    selected = [(a >= low) & (a <= high) for a, (low, high) in zip(axes, bounds, strict=True)]
    if any(not mask.any() for mask in selected):
        raise ValueError("Region contains no recorded cell centres")
    return selected[2][:, None, None] & selected[1][None, :, None] & selected[0][None, None, :]
