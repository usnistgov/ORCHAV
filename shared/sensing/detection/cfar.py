"""CFAR, non-maximum suppression, and 2-D detection clustering helpers.

These routines operate on range-Doppler power maps stored in optional sensing
payloads. Inputs are linear power arrays; thresholds are returned in the same
linear units. Conversion to range, Doppler, and velocity happens only at the
clustering boundary where physical axes are available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Literal

import numpy as np

from shared.sensing.signal_processing import (
    doppler_to_monostatic_equivalent_velocity_m_s,
    doppler_to_path_closing_rate_m_s,
)

CFARMethod = Literal["ca", "os"]
CFAREdgePolicy = Literal["full_window", "truncate"]
_FLOAT_EPS = np.finfo(np.float64).eps
_MIN_SUBNORMAL = np.nextafter(0.0, 1.0)
_MIN_NORMAL = np.finfo(np.float64).tiny


@dataclass
class ClusteredDetection:
    """Single detection cluster expressed in physical range/velocity units."""

    range_m: float
    velocity_m_s: float
    doppler_hz: float
    path_closing_rate_m_s: float
    monostatic_equivalent_velocity_m_s: float
    power_linear: float
    snr_db: float
    range_bin: int
    doppler_bin: int
    num_detections: int  # Number of detections in this cluster
    cluster_id: int
    # Sub-bin centroid estimates (interpolated, more precise than integer-bin)
    centroid_range_m: float = 0.0
    centroid_velocity_m_s: float = 0.0
    centroid_doppler_hz: float = 0.0
    centroid_path_closing_rate_m_s: float = 0.0
    centroid_monostatic_equivalent_velocity_m_s: float = 0.0
    # Per-cluster AoA (power-weighted mean, NaN if unavailable)
    aoa_az_deg: float = float("nan")
    aoa_el_deg: float = float("nan")


@dataclass
class CFAR1DResult:
    """Result arrays for 1-D CFAR in linear power units."""

    detections: np.ndarray  # bool
    threshold: np.ndarray
    noise: np.ndarray
    eligible: np.ndarray  # bool; threshold/noise are meaningful only here
    training_cell_count: np.ndarray  # number of training cells actually used


@dataclass
class CFAR2DResult:
    """Result arrays for 2-D CFAR over ``(doppler, range)`` grids."""

    detections: np.ndarray  # bool matrix
    threshold: np.ndarray
    noise: np.ndarray
    eligible: np.ndarray  # bool; threshold/noise are meaningful only here
    training_cell_count: np.ndarray  # number of training cells actually used


def _validated_power_array(power: np.ndarray, *, dimensions: int, name: str) -> np.ndarray:
    """Return one finite, nonnegative linear-power array."""

    raw = np.asarray(power)
    if raw.ndim != dimensions:
        raise ValueError(f"{name} expects a {dimensions}D array")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} power must be a real numeric array")
    arr = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} power must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} power must be nonnegative")
    return arr


def _validated_boolean_mask(mask: np.ndarray, *, dimensions: int, name: str) -> np.ndarray:
    """Return one exact boolean mask without numeric truth-value coercion."""

    arr = np.asarray(mask)
    if arr.ndim != dimensions:
        raise ValueError(f"{name} expects a {dimensions}D array")
    if not np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f"{name} must be a boolean array")
    return arr


def _validated_real_scalar(
    value: float,
    *,
    name: str,
    positive: bool,
) -> float:
    """Validate one finite positive or nonnegative real scalar."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} real value")
    scalar = float(value)
    if not np.isfinite(scalar) or (scalar <= 0.0 if positive else scalar < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} real value")
    return scalar


def _validated_axis(
    values: np.ndarray,
    *,
    expected_length: int,
    name: str,
    nonnegative: bool,
) -> np.ndarray:
    """Validate one finite, strictly increasing physical coordinate axis."""

    raw = np.asarray(values)
    if raw.shape != (expected_length,):
        raise ValueError(f"{name} must have shape {(expected_length,)}, got {raw.shape}")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a real numeric array")
    axis = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(axis < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _validated_optional_map(
    values: np.ndarray | None,
    *,
    expected_shape: tuple[int, int],
    name: str,
    allow_nan: bool,
) -> np.ndarray | None:
    """Validate an optional real map aligned with a range-Doppler grid."""

    if values is None:
        return None
    raw = np.asarray(values)
    if raw.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {raw.shape}")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a real numeric array")
    arr = raw.astype(np.float64, copy=False)
    valid_values = np.isfinite(arr) | (np.isnan(arr) if allow_nan else False)
    if not np.all(valid_values):
        qualifier = "finite values or NaN" if allow_nan else "only finite values"
        raise ValueError(f"{name} must contain {qualifier}")
    return arr


def _validated_cell_count(value: int, *, name: str, positive: bool) -> int:
    """Validate one exact integral guard/reference-cell count."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be an exact {qualifier} integer")
    count = int(value)
    if (positive and count <= 0) or (not positive and count < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return count


def _validated_cell_pair(
    value: tuple[int, int],
    *,
    name: str,
    positive: bool,
) -> tuple[int, int]:
    """Validate one exact ``(doppler, range)`` cell-count pair."""

    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item (doppler, range) tuple")
    return (
        _validated_cell_count(value[0], name=f"{name}[0]", positive=positive),
        _validated_cell_count(value[1], name=f"{name}[1]", positive=positive),
    )


def _validated_cfar_parameters(
    *,
    threshold_scale: float,
    method: CFARMethod,
    os_rank: float,
    edge_policy: CFAREdgePolicy,
) -> tuple[float, CFARMethod, float, CFAREdgePolicy]:
    """Validate scalar CFAR semantics shared by the 1-D and 2-D paths."""

    if isinstance(threshold_scale, (bool, np.bool_)) or not isinstance(threshold_scale, Real):
        raise ValueError("threshold_scale must be a finite nonnegative linear multiplier")
    scale = float(threshold_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("threshold_scale must be a finite nonnegative linear multiplier")
    if not isinstance(method, str) or method not in ("ca", "os"):
        raise ValueError("method must be exactly 'ca' or 'os'")
    if isinstance(os_rank, (bool, np.bool_)) or not isinstance(os_rank, Real):
        raise ValueError("os_rank must be finite and lie between zero and one")
    rank = float(os_rank)
    if not np.isfinite(rank) or not 0.0 <= rank <= 1.0:
        raise ValueError("os_rank must be finite and lie between zero and one")
    if not isinstance(edge_policy, str) or edge_policy not in (
        "full_window",
        "truncate",
    ):
        raise ValueError("edge_policy must be exactly 'full_window' or 'truncate'")
    return scale, method, rank, edge_policy


def _os_rank_index(training_cell_count: int, os_rank: float) -> int:
    """Return the OS index using nearest-integer, ties-to-even rounding."""

    return int(round(os_rank * (training_cell_count - 1)))


def _nonnegative_difference_with_roundoff_guard(
    minuend: np.ndarray | float,
    subtrahend: np.ndarray | float,
    *,
    accumulation_length: int,
    name: str,
    error_scale: float | None = None,
) -> np.ndarray:
    """Subtract nonnegative accumulated powers and clamp only roundoff.

    A CFAR training sum is mathematically nonnegative because its guard sum is
    a subset of its outer-window sum. Floating-point accumulation/subtraction
    can nevertheless produce values a few ulps below zero. Values beyond a
    conservative accumulation-length-scaled bound indicate numerical failure
    and are rejected rather than silently floored.
    """

    lhs = np.asarray(minuend, dtype=np.float64)
    rhs = np.asarray(subtrahend, dtype=np.float64)
    difference = lhs - rhs
    scale = np.maximum(np.abs(lhs), np.abs(rhs))
    if error_scale is not None:
        scale = np.maximum(scale, abs(float(error_scale)))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(int(accumulation_length), 1) * scale
    if np.any(difference < -tolerance):
        minimum = float(np.min(difference))
        raise ValueError(f"{name} became materially negative ({minimum:.17g})")
    return np.maximum(difference, 0.0)


def _accurate_ca_decision(
    samples: list[float], cut: float, scale: float
) -> tuple[float, float, bool]:
    """Resolve a numerically ambiguous CA comparison without a power floor.

    The threshold is the exact training mean times the supplied multiplier,
    rounded to float64 for reporting. Detection compares the exact quantities,
    so equality never detects and a representable increase is not discarded by
    an arbitrary tolerance. Rational arithmetic is restricted to comparisons
    that cannot be certified after local compensated summation.
    """
    count = len(samples)
    mean = math.fsum(samples) / count
    threshold = mean * scale
    bound = 4.0 * _FLOAT_EPS * max(abs(cut), abs(threshold))
    bound += 4.0 * _MIN_SUBNORMAL * max(1.0, scale)
    amplified_subnormal = mean < _MIN_NORMAL and scale > 1.0 and any(samples)
    if abs(cut - threshold) > bound and not amplified_subnormal:
        return mean, threshold, cut > threshold
    # Flat reference rings are common, and have an exactly known mean.
    first = samples[0]
    if all(value == first for value in samples):
        exact_mean = Fraction(float(first))
    else:
        exact_mean = sum((Fraction(float(value)) for value in samples), Fraction()) / count
    exact_threshold = exact_mean * Fraction(float(scale))
    return float(exact_mean), float(exact_threshold), Fraction(float(cut)) > exact_threshold


def _training_values_2d(
    arr: np.ndarray,
    i: int,
    j: int,
    guard_cells: tuple[int, int],
    reference_cells: tuple[int, int],
) -> list[float]:
    """Gather a clipped training annulus without subtracting excluded power."""
    gd, gr = guard_cells
    rd, rr = reference_cells
    top, bottom = max(0, i - gd - rd), min(arr.shape[0], i + gd + rd + 1)
    left, right = max(0, j - gr - rr), min(arr.shape[1], j + gr + rr + 1)
    gt, gb = max(0, i - gd), min(arr.shape[0], i + gd + 1)
    gl, gright = max(0, j - gr), min(arr.shape[1], j + gr + 1)
    return [
        float(value)
        for strip in (
            arr[top:gt, left:right],
            arr[gb:bottom, left:right],
            arr[gt:gb, left:gl],
            arr[gt:gb, gright:right],
        )
        for value in strip.flat
    ]


def cfar_1d(
    power: np.ndarray,
    *,
    guard_cells: int,
    reference_cells: int,
    threshold_scale: float,
    method: CFARMethod = "ca",
    os_rank: float = 0.75,
    edge_policy: CFAREdgePolicy = "full_window",
) -> CFAR1DResult:
    """Run 1-D CFAR over a power array.

    Guard and reference cell counts are applied on both sides of each test
    cell. CA-CFAR averages training cells; OS-CFAR uses the configured
    ascending rank-order training sample. ``threshold_scale`` is a raw linear
    multiplier on that estimate, not a dB offset or requested false-alarm
    probability. CA decisions use a strict comparison against the exact
    training mean times that multiplier at floating-point ambiguities; returned
    noise and threshold arrays are rounded float64 representations.

    ``full_window`` makes a CUT eligible only when its complete two-sided
    training window exists. ``truncate`` uses every available training cell at
    an edge and exposes the exact count in the returned result.
    """
    arr = _validated_power_array(power, dimensions=1, name="cfar_1d")
    guard = _validated_cell_count(guard_cells, name="guard_cells", positive=False)
    reference = _validated_cell_count(
        reference_cells,
        name="reference_cells",
        positive=True,
    )
    scale, method, rank_fraction, edge_policy = _validated_cfar_parameters(
        threshold_scale=threshold_scale,
        method=method,
        os_rank=os_rank,
        edge_policy=edge_policy,
    )

    n = arr.shape[0]
    detections = np.zeros(n, dtype=bool)
    threshold = np.zeros(n, dtype=np.float64)
    noise = np.zeros(n, dtype=np.float64)
    eligible = np.zeros(n, dtype=bool)
    training_cell_count = np.zeros(n, dtype=np.int32)
    # Prefix sums make the CA-CFAR training-window average O(1) per cell.
    prefix: np.ndarray | None = None
    if method == "ca":
        with np.errstate(over="ignore", invalid="ignore"):
            prefix = np.concatenate(([0.0], np.cumsum(arr)))
        if not np.all(np.isfinite(prefix)):
            raise ValueError("cfar_1d CA training-window sums exceed finite float64 range")

    sum_error = 64.0 * _FLOAT_EPS * n * prefix[-1] if prefix is not None else 0.0
    underflow_error = 4.0 * _MIN_SUBNORMAL * max(1.0, scale)
    for idx in range(n):
        requested_left_start = idx - guard - reference
        requested_left_end = idx - guard
        requested_right_start = idx + guard + 1
        requested_right_end = idx + guard + 1 + reference
        if edge_policy == "full_window" and (requested_left_start < 0 or requested_right_end > n):
            continue

        left_start = max(0, min(requested_left_start, n))
        left_end = max(0, min(requested_left_end, n))
        right_start = max(0, min(requested_right_start, n))
        right_end = max(0, min(requested_right_end, n))
        count = (left_end - left_start) + (right_end - right_start)
        if count <= 0:
            continue

        eligible[idx] = True
        training_cell_count[idx] = count

        if method == "os":
            samples = np.concatenate((arr[left_start:left_end], arr[right_start:right_end]))
            rank = _os_rank_index(count, rank_fraction)
            noise_val = np.partition(samples, rank)[rank]
        else:
            assert prefix is not None
            left_sum = float(
                _nonnegative_difference_with_roundoff_guard(
                    prefix[left_end],
                    prefix[left_start],
                    accumulation_length=n,
                    name="cfar_1d left training sum",
                )
            )
            right_sum = float(
                _nonnegative_difference_with_roundoff_guard(
                    prefix[right_end],
                    prefix[right_start],
                    accumulation_length=n,
                    name="cfar_1d right training sum",
                )
            )
            # Subtracting large prefixes can erase small positive local
            # training power. Re-sum only ill-conditioned intervals directly.
            for start, end, value, side in (
                (left_start, left_end, left_sum, "left"),
                (right_start, right_end, right_sum, "right"),
            ):
                error_bound = (
                    64.0 * np.finfo(np.float64).eps * n * (abs(prefix[end]) + abs(prefix[start]))
                )
                if value <= error_bound * 1e6:
                    value = math.fsum(arr[start:end])
                    if side == "left":
                        left_sum = value
                    else:
                        right_sum = value
            noise_val = (left_sum + right_sum) / count

        noise[idx] = noise_val
        threshold_val = noise_val * scale
        if not np.isfinite(threshold_val):
            raise ValueError("cfar_1d threshold exceeds finite float64 range")
        threshold[idx] = threshold_val
        detections[idx] = arr[idx] > threshold[idx]
        if method == "ca":
            assert prefix is not None
            comparison_error = sum_error / count * scale
            comparison_error += 4.0 * _FLOAT_EPS * max(arr[idx], threshold_val)
            comparison_error += underflow_error
            # Preserve threshold diagnostics even for a zero CUT when a
            # subnormal mean loses precision before a large multiplier.
            amplified_subnormal = (
                noise_val < _MIN_NORMAL and scale > 1.0 and left_sum + right_sum > 0.0
            )
            if amplified_subnormal or (
                arr[idx] > 0.0 and abs(arr[idx] - threshold_val) <= comparison_error
            ):
                samples = [*arr[left_start:left_end], *arr[right_start:right_end]]
                noise[idx], threshold[idx], detections[idx] = _accurate_ca_decision(
                    samples, arr[idx], scale
                )

    return CFAR1DResult(
        detections=detections,
        threshold=threshold,
        noise=noise,
        eligible=eligible,
        training_cell_count=training_cell_count,
    )


def _integral_rect_sum(
    integral: np.ndarray,
    top: np.ndarray,
    left: np.ndarray,
    bottom: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Vectorized rectangle sum from an integral image (2-D indices)."""
    bottom_right = integral[bottom, right]
    top_right = integral[top, right]
    bottom_left = integral[bottom, left]
    top_left = integral[top, left]
    rectangle_sum = bottom_right - top_right - bottom_left + top_left
    # Each exact rectangle sum is nonnegative. Its four large integral-image
    # terms can cancel near an explicitly zeroed region, so base the ulp bound
    # on those terms rather than on the already-cancelled result.
    corner_scale = np.abs(bottom_right) + np.abs(top_right) + np.abs(bottom_left) + np.abs(top_left)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(integral.shape) * corner_scale
    if np.any(rectangle_sum < -tolerance):
        minimum = float(np.min(rectangle_sum))
        raise ValueError(f"CFAR integral-image rectangle sum became negative ({minimum:.17g})")
    return np.maximum(rectangle_sum, 0.0)


def _accurate_positive_row_sums(values: np.ndarray) -> np.ndarray:
    """Sum nonnegative float64 rows, retaining fsum at rounding ambiguities.

    Pairwise TwoSum records each addition's exact residual (including gradual
    underflow). Summing those residuals corrects the ordinary pairwise result.
    A conservative bound on the residual accumulation certifies rounding only
    when its entire interval lies inside one float64 rounding cell. Otherwise
    use math.fsum on the original row. This also handles exact midpoint ties
    and subnormal results without assuming a wider platform-specific dtype.

    TwoSum: Ogita, Rump and Oishi, SIAM SISC 26(6), 2005, Algorithm 3.1.
    The bound uses n rather than log2(n), covers both residual reductions, and
    includes an absolute underflow allowance. It is not a signal power floor.
    """
    count = values.shape[1]
    work = values
    correction = np.zeros(len(values))
    absolute_residual = np.zeros(len(values))
    while work.shape[1] > 1:
        pairs = work.shape[1] // 2
        left, right = work[:, : 2 * pairs : 2], work[:, 1 : 2 * pairs : 2]
        total = left + right
        virtual_right = total - left
        residual = (left - (total - virtual_right)) + (right - virtual_right)
        correction += residual.sum(axis=1)
        absolute_residual += np.abs(residual).sum(axis=1)
        work = np.concatenate((total, work[:, -1:]), axis=1) if work.shape[1] % 2 else total

    high = work[:, 0]
    rounded = high + correction
    virtual_correction = rounded - high
    remainder = (high - (rounded - virtual_correction)) + (correction - virtual_correction)
    bound = 8 * count * np.finfo(np.float64).eps * absolute_residual + 8 * count * np.nextafter(
        0.0, 1.0
    )
    with np.errstate(over="ignore"):
        upper_half_gap = (np.nextafter(rounded, np.inf) - rounded) * 0.5
    lower_half_gap = (rounded - np.nextafter(rounded, -np.inf)) * 0.5
    uncertain = (
        (remainder + bound >= upper_half_gap)
        | (remainder - bound <= -lower_half_gap)
        | (rounded >= np.finfo(np.float64).max)
    )
    for index in np.flatnonzero(uncertain):
        rounded[index] = math.fsum(values[index])
    return rounded


def _local_training_sums(
    arr: np.ndarray,
    locations: np.ndarray,
    *,
    guard_cells: tuple[int, int],
    reference_cells: tuple[int, int],
) -> np.ndarray:
    """Stable annulus sums for only the cancellation-risk cells.

    Gather positive training values directly, never subtract guard power.
    Blocks bound the temporary matrices to about 1 MiB each. Clipping offsets
    to the map dimensions and zero-padding implements truncated edges while
    the caller retains the exact original training-cell counts.
    """
    gd, gr = guard_cells
    rd, rr = reference_cells
    if len(locations) < 16:
        # Avoid allocating neighborhood grids for a few exceptional cells.
        result = []
        for i, j in locations:
            top, bottom = max(0, i - gd - rd), min(arr.shape[0], i + gd + rd + 1)
            left, right = max(0, j - gr - rr), min(arr.shape[1], j + gr + rr + 1)
            gt, gb = max(0, i - gd), min(arr.shape[0], i + gd + 1)
            gl, gright = max(0, j - gr), min(arr.shape[1], j + gr + 1)
            strips = (
                arr[top:gt, left:right],
                arr[gb:bottom, left:right],
                arr[gt:gb, left:gl],
                arr[gt:gb, gright:right],
            )
            result.append(math.fsum(value for strip in strips for value in strip.flat))
        return np.asarray(result, dtype=np.float64)

    wd, wr = min(gd + rd, arr.shape[0] - 1), min(gr + rr, arr.shape[1] - 1)
    di, dj = np.meshgrid(np.arange(-wd, wd + 1), np.arange(-wr, wr + 1), indexing="ij")
    training = (np.abs(di) > gd) | (np.abs(dj) > gr)
    di, dj = di[training], dj[training]
    padded = np.pad(arr, ((wd, wd), (wr, wr)))
    # Linear offsets gather the same row-major annulus with one index matrix,
    # instead of allocating separate row/column matrices for every block.
    # Padding keeps even clipped edge neighborhoods inside the flattened map.
    offsets = di * padded.shape[1] + dj
    origins = (locations[:, 0] + wd) * padded.shape[1] + locations[:, 1] + wr
    flat = padded.ravel()
    block_size = max(1, 131072 // len(di))
    result = np.empty(len(locations))
    for start in range(0, len(locations), block_size):
        values = flat[origins[start : start + block_size, None] + offsets]
        result[start : start + block_size] = _accurate_positive_row_sums(values)
    return result


def _cfar_2d_ca_vectorized(
    arr: np.ndarray,
    *,
    guard_cells: tuple[int, int],
    reference_cells: tuple[int, int],
    threshold_scale: float,
    edge_policy: CFAREdgePolicy,
) -> CFAR2DResult:
    """Fully vectorized CA-CFAR using integral image lookups."""
    rows, cols = arr.shape
    gd, gr = guard_cells
    rd, rr = reference_cells
    window_d = gd + rd
    window_r = gr + rr

    # Build integral image (1-indexed: pad top-left with a row/col of zeros)
    padded = np.pad(arr, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        integral = padded.cumsum(axis=0).cumsum(axis=1)
    if not np.all(np.isfinite(integral)):
        raise ValueError("cfar_2d CA training-window sums exceed finite float64 range")

    # Broadcast row/column coordinates without allocating dense index grids.
    ri = np.arange(rows)
    ci = np.arange(cols)
    R, C = ri[:, None], ci[None, :]

    # Outer window clipped to array bounds (integral-image coords are 0-based)
    out_top = np.maximum(R - window_d, 0)
    out_bot = np.minimum(R + window_d + 1, rows)
    out_lft = np.maximum(C - window_r, 0)
    out_rgt = np.minimum(C + window_r + 1, cols)

    # Guard window clipped to array bounds
    grd_top = np.maximum(R - gd, 0)
    grd_bot = np.minimum(R + gd + 1, rows)
    grd_lft = np.maximum(C - gr, 0)
    grd_rgt = np.minimum(C + gr + 1, cols)

    # Rectangle sums via integral image
    outer_sum = _integral_rect_sum(integral, out_top, out_lft, out_bot, out_rgt)
    guard_sum = _integral_rect_sum(integral, grd_top, grd_lft, grd_bot, grd_rgt)

    outer_count = (out_bot - out_top) * (out_rgt - out_lft)
    guard_count = (grd_bot - grd_top) * (grd_rgt - grd_lft)
    training_count = outer_count - guard_count
    training_sum = _nonnegative_difference_with_roundoff_guard(
        outer_sum,
        guard_sum,
        accumulation_length=max(rows, cols),
        name="cfar_2d training sum",
        error_scale=float(integral[-1, -1]),
    )

    # A full-window CUT requires the complete rectangular annulus. Truncated
    # mode instead uses the clipped count computed above.
    if edge_policy == "full_window":
        valid = (
            (R - window_d >= 0)
            & (R + window_d < rows)
            & (C - window_r >= 0)
            & (C + window_r < cols)
            & (training_count > 0)
        )
    else:
        valid = training_count > 0
    # Every nonempty training ring in a constant field has exactly this mean,
    # including truncated edges. Resolve the common strict comparison once,
    # rather than sending every cell through the local ambiguity fallback.
    if arr.size and np.any(valid) and np.all(arr == arr.flat[0]):
        value = float(arr.flat[0])
        exact_threshold = Fraction(value) * Fraction(float(threshold_scale))
        try:
            threshold_value = float(exact_threshold)
        except OverflowError as exc:
            raise ValueError("cfar_2d threshold exceeds finite float64 range") from exc
        return CFAR2DResult(
            detections=valid & (Fraction(value) > exact_threshold),
            threshold=np.where(valid, threshold_value, 0.0),
            noise=np.where(valid, value, 0.0),
            eligible=valid,
            training_cell_count=np.where(valid, training_count, 0).astype(np.int32),
        )
    # A distant large sample, or a large CUT excluded by the guard, can
    # dominate the integral image. Detect cancellation before trusting zero
    # or small positive sums. Local annulus sums never subtract guard power.
    error_bound = 128.0 * np.finfo(np.float64).eps * max(rows, cols) * integral[-1, -1]
    cancellation_risk = valid & (training_sum <= error_bound * 1e6)
    if np.any(cancellation_risk):
        # Exact integer counts distinguish genuinely empty training rings
        # from positive power erased by cancellation. Sparse/zero maps must
        # not turn the fast integral path into a dense local-summation loop.
        positive_integral = (
            np.pad(arr > 0.0, ((1, 0), (1, 0)))
            .cumsum(axis=0, dtype=np.int64)
            .cumsum(axis=1, dtype=np.int64)
        )
        positive_outer = (
            positive_integral[out_bot, out_rgt]
            - positive_integral[out_top, out_rgt]
            - positive_integral[out_bot, out_lft]
            + positive_integral[out_top, out_lft]
        )
        positive_guard = (
            positive_integral[grd_bot, grd_rgt]
            - positive_integral[grd_top, grd_rgt]
            - positive_integral[grd_bot, grd_lft]
            + positive_integral[grd_top, grd_lft]
        )
        has_training_power = positive_outer > positive_guard
        training_sum[cancellation_risk & ~has_training_power] = 0.0
        cancellation_risk &= has_training_power
    if np.any(cancellation_risk):
        training_sum[cancellation_risk] = _local_training_sums(
            arr,
            np.argwhere(cancellation_risk),
            guard_cells=guard_cells,
            reference_cells=reference_cells,
        )
    used_training_count = np.where(valid, training_count, 0).astype(np.int32, copy=False)
    noise = np.zeros_like(arr)
    noise[valid] = training_sum[valid] / training_count[valid]

    with np.errstate(over="ignore", invalid="ignore"):
        threshold = noise * threshold_scale
    if not np.all(np.isfinite(threshold[valid])):
        raise ValueError("cfar_2d threshold exceeds finite float64 range")
    detections = (arr > threshold) & (arr > 0.0) & valid
    # A well-conditioned sum can still round to the wrong side of a strict
    # threshold. Revisit only decisions inside the accumulation error interval.
    comparison_error = np.zeros_like(arr)
    with np.errstate(over="ignore", invalid="ignore"):
        comparison_error[valid] = error_bound / training_count[valid] * threshold_scale
        # Locally repaired sums no longer carry integral-image cancellation.
        comparison_error[cancellation_risk] = 0.0
        comparison_error += 4.0 * _FLOAT_EPS * np.maximum(arr, threshold)
        comparison_error += 4.0 * _MIN_SUBNORMAL * max(1.0, threshold_scale)
    amplified_subnormal = (noise < _MIN_NORMAL) & (threshold_scale > 1.0) & (training_sum > 0.0)
    ambiguous = valid & (
        amplified_subnormal | ((arr > 0.0) & (np.abs(arr - threshold) <= comparison_error))
    )
    for i, j in np.argwhere(ambiguous):
        samples = _training_values_2d(arr, i, j, guard_cells, reference_cells)
        noise[i, j], threshold[i, j], detections[i, j] = _accurate_ca_decision(
            samples, arr[i, j], threshold_scale
        )

    return CFAR2DResult(
        detections=detections,
        threshold=threshold,
        noise=noise,
        eligible=valid,
        training_cell_count=used_training_count,
    )


def _cfar_2d_os_loop(
    arr: np.ndarray,
    *,
    guard_cells: tuple[int, int],
    reference_cells: tuple[int, int],
    threshold_scale: float,
    os_rank: float,
    edge_policy: CFAREdgePolicy,
) -> CFAR2DResult:
    """OS-CFAR via per-cell loop (rank-order statistics not vectorizable)."""
    rows, cols = arr.shape
    detections = np.zeros_like(arr, dtype=bool)
    threshold = np.zeros_like(arr, dtype=np.float64)
    noise = np.zeros_like(arr, dtype=np.float64)
    eligible = np.zeros_like(arr, dtype=bool)
    training_cell_count = np.zeros_like(arr, dtype=np.int32)

    gd, gr = guard_cells
    rd, rr = reference_cells
    window_d = gd + rd
    window_r = gr + rr

    for r in range(rows):
        requested_top = r - window_d
        requested_bottom = r + window_d + 1
        if edge_policy == "full_window" and (requested_top < 0 or requested_bottom > rows):
            continue
        top = max(0, requested_top)
        bottom = min(rows, requested_bottom)
        guard_top = max(0, r - gd)
        guard_bottom = min(rows, r + gd + 1)
        for c in range(cols):
            requested_left = c - window_r
            requested_right = c + window_r + 1
            if edge_policy == "full_window" and (requested_left < 0 or requested_right > cols):
                continue
            left = max(0, requested_left)
            right = min(cols, requested_right)
            guard_left = max(0, c - gr)
            guard_right = min(cols, c + gr + 1)

            patch = arr[top:bottom, left:right]
            guard_mask = np.zeros_like(patch, dtype=bool)
            gt = guard_top - top
            gb = guard_bottom - top
            gl = guard_left - left
            gr_ = guard_right - left
            guard_mask[gt:gb, gl:gr_] = True
            guard_mask[r - top, c - left] = True
            training = patch[~guard_mask]
            if training.size == 0:
                continue
            count = int(training.size)
            eligible[r, c] = True
            training_cell_count[r, c] = count
            rank = _os_rank_index(count, os_rank)
            noise_val = np.partition(training, rank)[rank]

            noise[r, c] = noise_val
            threshold_val = noise_val * threshold_scale
            if not np.isfinite(threshold_val):
                raise ValueError("cfar_2d threshold exceeds finite float64 range")
            threshold[r, c] = threshold_val
            if arr[r, c] <= 0.0:
                continue
            detections[r, c] = arr[r, c] > threshold[r, c]

    return CFAR2DResult(
        detections=detections,
        threshold=threshold,
        noise=noise,
        eligible=eligible,
        training_cell_count=training_cell_count,
    )


def cfar_2d(
    power: np.ndarray,
    *,
    guard_cells: tuple[int, int],
    reference_cells: tuple[int, int],
    threshold_scale: float,
    method: CFARMethod = "ca",
    os_rank: float = 0.75,
    edge_policy: CFAREdgePolicy = "full_window",
) -> CFAR2DResult:
    """Run 2D CFAR over a Doppler-Range map.

    Guard/reference cells are tuples ``(doppler_axis, range_axis)``.
    CA-CFAR uses integral images with direct local sums for ill-conditioned
    differences. This OS-CFAR implementation evaluates rank statistics per cell.
    ``threshold_scale`` is a raw linear multiplier on the CA mean or selected
    OS training sample; it is not a dB offset or requested false-alarm
    probability. At floating-point ambiguities, CA decisions strictly compare
    the exact training mean times the multiplier; reported arrays are rounded
    float64 representations.

    ``full_window`` makes only CUTs with a complete rectangular training
    annulus eligible. ``truncate`` uses every available edge training cell and
    exposes the exact count in the result.
    """
    arr = _validated_power_array(power, dimensions=2, name="cfar_2d")
    validated_guard_cells = _validated_cell_pair(
        guard_cells,
        name="guard_cells",
        positive=False,
    )
    validated_reference_cells = _validated_cell_pair(
        reference_cells,
        name="reference_cells",
        positive=True,
    )
    scale, method, rank_fraction, edge_policy = _validated_cfar_parameters(
        threshold_scale=threshold_scale,
        method=method,
        os_rank=os_rank,
        edge_policy=edge_policy,
    )

    if method == "os":
        return _cfar_2d_os_loop(
            arr,
            guard_cells=validated_guard_cells,
            reference_cells=validated_reference_cells,
            threshold_scale=scale,
            os_rank=rank_fraction,
            edge_policy=edge_policy,
        )

    return _cfar_2d_ca_vectorized(
        arr,
        guard_cells=validated_guard_cells,
        reference_cells=validated_reference_cells,
        threshold_scale=scale,
        edge_policy=edge_policy,
    )


def nms_peaks_2d(
    detections_mask: np.ndarray,
    power_map: np.ndarray,
    *,
    suppression_radius: tuple[int, int] = (2, 4),
) -> np.ndarray:
    """Extract local-maximum peaks from a CFAR detection mask using NMS.

    Implements the NMS stage used in 5GNRad's detection pipeline: after
    CFAR produces a binary mask, find local maxima within the mask, sort
    by power descending, and greedily suppress neighbors within the
    suppression radius. This reduces nearby sidelobe peaks; separated lobes
    and downstream clustering still depend on their physical gates. Neither
    range nor Doppler wraps at the array boundary.

    Parameters
    ----------
    detections_mask : np.ndarray
        Boolean [doppler, range] mask from CFAR.
    power_map : np.ndarray
        Power values [doppler, range].
    suppression_radius : tuple[int, int]
        (doppler_radius, range_radius) for NMS suppression window.
        Typically matches the CFAR guard cell size.

    Returns
    -------
    np.ndarray
        Boolean mask of the same shape with only NMS-surviving peaks.
    """
    mask = _validated_boolean_mask(
        detections_mask,
        dimensions=2,
        name="detections_mask",
    )
    power = _validated_power_array(power_map, dimensions=2, name="nms_peaks_2d")
    if mask.shape != power.shape:
        raise ValueError(
            "detections_mask and power_map must have identical shapes, "
            f"got {mask.shape} and {power.shape}"
        )
    rd, rr = _validated_cell_pair(
        suppression_radius,
        name="suppression_radius",
        positive=False,
    )
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    rows, cols = power.shape

    # Step 1: local-maximum filter (3x3 neighborhood)
    masked_power = np.where(mask, power, 0.0)
    local_max = np.zeros_like(mask, dtype=bool)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(masked_power, -dr, axis=0), -dc, axis=1)
            # Boundary: zero out wrapped edges
            if dr == -1:
                shifted[0, :] = 0.0
            elif dr == 1:
                shifted[-1, :] = 0.0
            if dc == -1:
                shifted[:, 0] = 0.0
            elif dc == 1:
                shifted[:, -1] = 0.0
            local_max |= masked_power < shifted

    # Invert: a cell is a local max if it is NOT less than any neighbor
    candidates = mask & ~local_max & (power > 0.0)

    # Step 2: sort candidates by power descending
    cand_coords = np.argwhere(candidates)
    if cand_coords.size == 0:
        return np.zeros_like(mask, dtype=bool)

    cand_powers = power[cand_coords[:, 0], cand_coords[:, 1]]
    # Stable ordering makes equal-power ties deterministic in row-major order.
    order = np.argsort(-cand_powers, kind="stable")
    cand_coords = cand_coords[order]

    # Step 3: greedy NMS — accept peaks, suppress neighbors
    suppressed = np.zeros((rows, cols), dtype=bool)
    result = np.zeros((rows, cols), dtype=bool)

    for dopp_idx, range_idx in cand_coords:
        if suppressed[dopp_idx, range_idx]:
            continue
        result[dopp_idx, range_idx] = True
        # Suppress neighborhood
        d_lo = max(0, dopp_idx - rd)
        d_hi = min(rows, dopp_idx + rd + 1)
        r_lo = max(0, range_idx - rr)
        r_hi = min(cols, range_idx + rr + 1)
        suppressed[d_lo:d_hi, r_lo:r_hi] = True

    return result


def _cluster_unit_radius_components(physical_coordinates: np.ndarray) -> np.ndarray:
    """Return DBSCAN-equivalent labels for ``eps=1, min_samples=1``.

    Scikit-learn is used when available. The dependency-free path computes the
    same connected components of the unit-radius neighbourhood graph instead
    of silently changing the scientific result to one cluster per detection.
    """

    try:
        from sklearn.cluster import DBSCAN

        return np.asarray(
            DBSCAN(eps=1.0, min_samples=1).fit(physical_coordinates).labels_,
            dtype=np.int32,
        )
    except ImportError:
        count = physical_coordinates.shape[0]
        labels = np.full(count, -1, dtype=np.int32)
        next_label = 0
        for seed in range(count):
            if labels[seed] >= 0:
                continue
            labels[seed] = next_label
            pending = [seed]
            while pending:
                current = pending.pop()
                deltas = physical_coordinates - physical_coordinates[current]
                neighbours = np.flatnonzero(np.einsum("ij,ij->i", deltas, deltas) <= 1.0)
                for neighbour in neighbours:
                    neighbour_index = int(neighbour)
                    if labels[neighbour_index] >= 0:
                        continue
                    labels[neighbour_index] = next_label
                    pending.append(neighbour_index)
            next_label += 1
        return labels


def cluster_detections_2d(
    detections_mask: np.ndarray,
    power_map: np.ndarray,
    range_axis_m: np.ndarray,
    doppler_axis_hz: np.ndarray,
    *,
    wavelength: float,
    snr_map: np.ndarray | None = None,
    snr_valid_mask: np.ndarray | None = None,
    aoa_az_map: np.ndarray | None = None,
    aoa_el_map: np.ndarray | None = None,
    cluster_range_gate_m: float = 1.0,
    cluster_velocity_gate_m_s: float = 5.0,
    return_cluster_labels: bool = False,
    range_mode: str = "monostatic",
    tx_pos: np.ndarray | None = None,
    rx_pos: np.ndarray | None = None,
) -> list[ClusteredDetection] | tuple[list[ClusteredDetection], np.ndarray]:
    """Cluster CFAR detections in range-Doppler space to identify distinct targets.

    Uses DBSCAN clustering with **anisotropic distance in physical space**
    (metres for range, m/s for velocity).  This correctly groups extended
    targets like walking pedestrians whose micro-Doppler signature spans many
    Doppler bins but only a few range bins.

    Parameters
    ----------
    detections_mask : np.ndarray
        Boolean [doppler, range] mask from CFAR.
    power_map : np.ndarray
        Power values [doppler, range] for finding peak within cluster.
    range_axis_m : np.ndarray
        Range axis in meters.
    doppler_axis_hz : np.ndarray
        Doppler axis in Hz.
    wavelength : float
        Radar wavelength in meters (for velocity conversion).
    snr_map : np.ndarray, optional
        SNR map [doppler, range] in dB.
    snr_valid_mask : np.ndarray, optional
        Boolean map declaring where ``snr_map`` is scientifically defined.
        Every selected detection cell must be valid when this mask is supplied.
    cluster_range_gate_m : float
        Range gate for clustering in metres (default: 1.0).  Detections
        within this range difference are candidates for the same cluster.
    cluster_velocity_gate_m_s : float
        Velocity gate for clustering in m/s (default: 5.0).  Detections
        within this velocity difference are candidates for the same cluster.
        Set wide for extended targets with micro-Doppler (pedestrians).
    range_mode : str
        ``"monostatic"`` or ``"bistatic"``. Retained for sensing call-site
        compatibility; detections always use the target-independent
        ``wavelength * doppler / 2`` coordinate.
    tx_pos : np.ndarray, optional
        Observation-system transmitter position, reserved for future
        geometry-estimating processors. It does not affect clustering.
    rx_pos : np.ndarray, optional
        Observation-system receiver position, reserved for future
        geometry-estimating processors. It does not affect clustering.
    Returns
    -------
    list[ClusteredDetection] or (list[ClusteredDetection], np.ndarray)
        Clustered detections, optionally with cluster label map.
    """
    mask = _validated_boolean_mask(
        detections_mask,
        dimensions=2,
        name="detections_mask",
    )
    power = _validated_power_array(power_map, dimensions=2, name="cluster_detections_2d")
    if mask.shape != power.shape:
        raise ValueError(
            "detections_mask and power_map must have identical shapes, "
            f"got {mask.shape} and {power.shape}"
        )
    range_axis = _validated_axis(
        range_axis_m,
        expected_length=power.shape[1],
        name="range_axis_m",
        nonnegative=True,
    )
    doppler_axis = _validated_axis(
        doppler_axis_hz,
        expected_length=power.shape[0],
        name="doppler_axis_hz",
        nonnegative=False,
    )
    wavelength_m = _validated_real_scalar(
        wavelength,
        name="wavelength",
        positive=True,
    )
    range_gate_m = _validated_real_scalar(
        cluster_range_gate_m,
        name="cluster_range_gate_m",
        positive=True,
    )
    velocity_gate_m_s = _validated_real_scalar(
        cluster_velocity_gate_m_s,
        name="cluster_velocity_gate_m_s",
        positive=True,
    )
    if not isinstance(range_mode, str) or range_mode not in ("monostatic", "bistatic"):
        raise ValueError("range_mode must be exactly 'monostatic' or 'bistatic'")

    snr = _validated_optional_map(
        snr_map,
        expected_shape=power.shape,
        name="snr_map",
        allow_nan=False,
    )
    aoa_az = _validated_optional_map(
        aoa_az_map,
        expected_shape=power.shape,
        name="aoa_az_map",
        allow_nan=True,
    )
    aoa_el = _validated_optional_map(
        aoa_el_map,
        expected_shape=power.shape,
        name="aoa_el_map",
        allow_nan=True,
    )
    if snr_valid_mask is None:
        snr_valid = None
    else:
        snr_valid = _validated_boolean_mask(
            snr_valid_mask,
            dimensions=2,
            name="snr_valid_mask",
        )
        if snr_valid.shape != power.shape:
            raise ValueError(f"snr_valid_mask must have shape {power.shape}, got {snr_valid.shape}")
        if snr is None:
            raise ValueError("snr_valid_mask requires snr_map")

    if np.any(mask & (power <= 0.0)):
        raise ValueError("detections_mask may select only strictly positive power cells")
    if snr_valid is not None and np.any(mask & ~snr_valid):
        raise ValueError("detections_mask selects a cell where SNR is undefined")

    if not np.any(mask):
        empty_labels = np.full(mask.shape, -1, dtype=np.int32)
        return ([], empty_labels) if return_cluster_labels else []

    # Get detection coordinates
    det_coords = np.argwhere(mask)  # [[dopp_idx, range_idx], ...]

    # Convert detection bin coordinates to physical space (metres, m/s).
    # Velocity axis derived from Doppler: v = λ·f_d / 2 (monostatic convention).
    velocity_axis = doppler_axis * wavelength_m / 2.0
    det_range_m = range_axis[det_coords[:, 1]]
    det_velocity = velocity_axis[det_coords[:, 0]]

    # Scale to gate-normalised units so eps=1.0 gives the desired gates.
    phys_coords = np.column_stack([det_range_m / range_gate_m, det_velocity / velocity_gate_m_s])
    labels = _cluster_unit_radius_components(phys_coords)

    cluster_labels = np.full(mask.shape, -1, dtype=np.int32)
    cluster_labels[det_coords[:, 0], det_coords[:, 1]] = labels.astype(np.int32, copy=False)

    # Group detections by cluster
    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
    clustered = []

    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_indices = det_coords[cluster_mask]

        cluster_power = power[cluster_indices[:, 0], cluster_indices[:, 1]]
        peak_offset = int(np.argmax(cluster_power))
        peak_doppler_idx, peak_range_idx = cluster_indices[peak_offset]
        max_power = float(cluster_power[peak_offset])

        # Normalize before summing so finite, very large power values cannot
        # overflow the centroid calculation.
        normalized_weights = cluster_power / max_power
        weight_sum = float(np.sum(normalized_weights))

        # Integer-bin fields identify an actual detected peak. Continuous
        # centroid fields below preserve the sub-bin cluster estimate.
        range_m = float(range_axis[peak_range_idx])
        doppler_hz = float(doppler_axis[peak_doppler_idx])

        # Weight physical coordinates directly: interpolating a weighted bin
        # index is equivalent only when the accepted axes are uniformly spaced.
        weights = normalized_weights / weight_sum
        centroid_range_m = float(np.dot(weights, range_axis[cluster_indices[:, 1]]))
        centroid_doppler_hz = float(np.dot(weights, doppler_axis[cluster_indices[:, 0]]))

        # A detection owns an observed Doppler, not a target identity. Keep
        # the legacy velocity projection target-independent; geometry-aware
        # bistatic speed belongs exclusively to truth evaluation.
        path_closing_rate_m_s = doppler_to_path_closing_rate_m_s(doppler_hz, wavelength_m)
        velocity_m_s = doppler_to_monostatic_equivalent_velocity_m_s(
            doppler_hz,
            wavelength_m,
        )
        centroid_path_closing_rate_m_s = doppler_to_path_closing_rate_m_s(
            centroid_doppler_hz,
            wavelength_m,
        )
        centroid_velocity_m_s = doppler_to_monostatic_equivalent_velocity_m_s(
            centroid_doppler_hz,
            wavelength_m,
        )

        if snr is not None:
            snr_db = float(snr[peak_doppler_idx, peak_range_idx])
        else:
            snr_db = 0.0

        # Power-weighted mean AoA per cluster (circular mean for azimuth)
        cluster_aoa_az = float("nan")
        cluster_aoa_el = float("nan")
        if aoa_az is not None:
            # Circular mean via atan2(sum(w*sin), sum(w*cos)) for ±180 wrap
            azimuth_values = aoa_az[cluster_indices[:, 0], cluster_indices[:, 1]]
            valid_azimuth = np.isfinite(azimuth_values)
            if np.any(valid_azimuth):
                azimuth_radians = np.radians(azimuth_values[valid_azimuth])
                azimuth_weights = normalized_weights[valid_azimuth]
                sin_sum = float(np.dot(azimuth_weights, np.sin(azimuth_radians)))
                cos_sum = float(np.dot(azimuth_weights, np.cos(azimuth_radians)))
                cluster_aoa_az = float(np.degrees(np.arctan2(sin_sum, cos_sum)))
        if aoa_el is not None:
            # Elevation does not wrap; ignore explicitly unavailable NaN cells.
            elevation_values = aoa_el[cluster_indices[:, 0], cluster_indices[:, 1]]
            valid_elevation = np.isfinite(elevation_values)
            if np.any(valid_elevation):
                elevation_weights = normalized_weights[valid_elevation]
                cluster_aoa_el = float(
                    np.dot(elevation_weights, elevation_values[valid_elevation])
                    / np.sum(elevation_weights)
                )

        clustered.append(
            ClusteredDetection(
                range_m=range_m,
                velocity_m_s=velocity_m_s,
                doppler_hz=doppler_hz,
                path_closing_rate_m_s=path_closing_rate_m_s,
                monostatic_equivalent_velocity_m_s=velocity_m_s,
                power_linear=max_power,
                snr_db=snr_db,
                range_bin=int(peak_range_idx),
                doppler_bin=int(peak_doppler_idx),
                num_detections=int(np.sum(cluster_mask)),
                cluster_id=int(cluster_id),
                centroid_range_m=centroid_range_m,
                centroid_velocity_m_s=centroid_velocity_m_s,
                centroid_doppler_hz=centroid_doppler_hz,
                centroid_path_closing_rate_m_s=centroid_path_closing_rate_m_s,
                centroid_monostatic_equivalent_velocity_m_s=centroid_velocity_m_s,
                aoa_az_deg=cluster_aoa_az,
                aoa_el_deg=cluster_aoa_el,
            )
        )

    # Sort by power (strongest first)
    clustered.sort(key=lambda d: d.power_linear, reverse=True)

    if return_cluster_labels:
        return clustered, cluster_labels
    return clustered
