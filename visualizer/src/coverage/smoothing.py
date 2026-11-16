"""Display-only normalized Gaussian smoothing, shared by all coverage views."""

import numpy as np

from shared.logging import get_logger

# Historical Smooth/Smooth+ strengths, in cells along each filtered axis.
COVERAGE_SMOOTH_SIGMA_LINEAR = 1.5
COVERAGE_SMOOTH_SIGMA_CUBIC = 3.0


def smooth_coverage_values(values, method="none"):
    """Smooth finite samples without filling holes or changing the input array.

    The operation acts in the supplied metric's units. Callers pass an XY, XZ
    or YZ plane; no samples from a neighbouring plane participate.
    """
    sigma = {"linear": COVERAGE_SMOOTH_SIGMA_LINEAR, "cubic": COVERAGE_SMOOTH_SIGMA_CUBIC}.get(
        method
    )
    if sigma is None:
        return values
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        get_logger("orchav").warning("scipy not available for smoothing, returning raw values")
        return values
    invalid = ~np.isfinite(values)
    numerator = gaussian_filter(np.where(invalid, 0.0, values), sigma=sigma)
    weights = gaussian_filter((~invalid).astype(np.float64), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weights > 1e-10, numerator / weights, np.nan)
    result[invalid] = np.nan
    return result
