"""Distribution helpers for MPC statistics and comparison plots.

Functions here are deliberately small wrappers around NumPy so callers use the
same histogram binning, CDF convention, percentile behavior, and empty-input
handling across generator summaries and visualizer diagnostics. NaN and
infinite input values are excluded consistently; inputs with no finite values
use the same result semantics as empty inputs.
"""

from typing import Optional, Tuple

import numpy as np

# Shared default bin count for ORCHAV statistics histograms.
DEFAULT_BINS = 50


def _finite_values(values: np.ndarray) -> np.ndarray:
    """Return the finite values used by all distribution helpers."""
    array = np.asarray(values)
    return array[np.isfinite(array)]


def compute_histogram(
    values: np.ndarray,
    bins: int = DEFAULT_BINS,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram with consistent defaults.

    Args:
        values: Input values, shape (N,). NaN and infinite values are ignored.
        bins: Number of histogram bins. Default: 50
        range: Optional (min, max) range. If None, uses data range.
        density: If True, normalize to probability density. Default: False

    Returns:
        Tuple of (bin_centers, counts)
        bin_centers has shape (bins,), counts has shape (bins,)

    Example:
        >>> values = np.random.randn(1000)
        >>> centers, counts = compute_histogram(values)
        >>> len(centers)
        50
    """
    finite_values = _finite_values(values)
    if len(finite_values) == 0:
        return np.array([]), np.array([])

    counts, bin_edges = np.histogram(finite_values, bins=bins, range=range, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, counts


def compute_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical Cumulative Distribution Function (CDF).

    Args:
        values: Input values, shape (N,). NaN and infinite values are ignored.

    Returns:
        Tuple of (sorted_values, cdf)
        sorted_values: The sorted input values
        cdf: Cumulative probability at each value (0 to 1)

    Example:
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> x, cdf = compute_cdf(values)
        >>> cdf[-1]
        1.0
    """
    finite_values = _finite_values(values)
    if len(finite_values) == 0:
        return np.array([]), np.array([])

    sorted_values = np.sort(finite_values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    return sorted_values, cdf


def compute_percentile(values: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """
    Compute percentiles of a distribution.

    Args:
        values: Input values, shape (N,). NaN and infinite values are ignored.
        percentiles: Percentiles to compute (0-100), shape (M,)

    Returns:
        Array of percentile values, shape (M,)

    Example:
        >>> values = np.arange(100)
        >>> compute_percentile(values, np.array([25, 50, 75]))
        array([24.75, 49.5 , 74.25])
    """
    finite_values = _finite_values(values)
    if len(finite_values) == 0:
        return np.full(len(percentiles), np.nan)

    return np.percentile(finite_values, percentiles)


def compute_statistics_summary(values: np.ndarray) -> dict:
    """
    Compute common statistical summary of a distribution.

    Args:
        values: Input values, shape (N,). NaN and infinite values are ignored.

    Returns:
        Dictionary with keys: count, mean, std, min, max, median, q25, q75

    Example:
        >>> values = np.arange(100)
        >>> stats = compute_statistics_summary(values)
        >>> stats['mean']
        49.5
    """
    finite_values = _finite_values(values)
    if len(finite_values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    return {
        "count": len(finite_values),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "median": float(np.median(finite_values)),
        "q25": float(np.percentile(finite_values, 25)),
        "q75": float(np.percentile(finite_values, 75)),
    }
