"""Independent numeric references for sensing detection and clutter removal.

The production detector remains in :mod:`shared.sensing.detection.cfar` and
:mod:`shared.sensing.detection.pipeline`.  This module intentionally contains small,
direct equations used to validate configuration semantics, Monte Carlo
evidence, and clutter-removal claims without calling the production paths.

All CFAR equations below assume independent exponential power samples, as
obtained from circular complex Gaussian noise.  ``threshold_scale`` is a raw
linear multiplier applied to the estimated noise power; it is never a dB
offset.  A requested cell false-alarm probability is converted to a multiplier
only after the exact number of eligible training cells is known.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def cfar_training_cell_count_1d(reference_cells: int) -> int:
    """Return the full-window 1-D training-cell count.

    ``reference_cells`` is the count on each side of the cell under test.
    """

    reference = int(reference_cells)
    if reference <= 0:
        raise ValueError("reference_cells must be positive")
    return 2 * reference


def cfar_training_cell_count_2d(
    guard_cells: tuple[int, int],
    reference_cells: tuple[int, int],
) -> int:
    """Return the full rectangular-ring 2-D training-cell count."""

    guard_doppler, guard_range = (int(value) for value in guard_cells)
    reference_doppler, reference_range = (int(value) for value in reference_cells)
    if min(guard_doppler, guard_range) < 0:
        raise ValueError("guard cell counts must be nonnegative")
    if min(reference_doppler, reference_range) <= 0:
        raise ValueError("reference cell counts must be positive")

    outer = (2 * (guard_doppler + reference_doppler) + 1) * (
        2 * (guard_range + reference_range) + 1
    )
    excluded_guard_and_cut = (2 * guard_doppler + 1) * (2 * guard_range + 1)
    return outer - excluded_guard_and_cut


def ca_cfar_false_alarm_probability(threshold_scale: float, training_cells: int) -> float:
    """Return ideal CA-CFAR cell ``Pfa`` for exponential noise.

    The noise estimate is the arithmetic mean of ``training_cells`` samples
    and the independent cell under test is compared with
    ``threshold_scale * mean(training)``.
    """

    alpha = float(threshold_scale)
    count = int(training_cells)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("threshold_scale must be finite and nonnegative")
    if count <= 0:
        raise ValueError("training_cells must be positive")
    return float((1.0 + alpha / count) ** (-count))


def ca_cfar_threshold_scale(requested_pfa: float, training_cells: int) -> float:
    """Convert requested CA-CFAR cell ``Pfa`` to a linear multiplier."""

    probability = float(requested_pfa)
    count = int(training_cells)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("requested_pfa must lie strictly between zero and one")
    if count <= 0:
        raise ValueError("training_cells must be positive")
    return float(count * math.expm1(-math.log(probability) / count))


def ca_cfar_detection_probability_nonfluctuating_target(
    threshold_scale: float,
    training_cells: int,
    target_snr_linear: float,
) -> float:
    """Return ideal CA-CFAR ``Pd`` for a deterministic complex target.

    Training powers are independent unit-mean exponentials.  The cell under
    test is ``|sqrt(target_snr_linear) + w|**2`` for circular complex Gaussian
    ``w`` with unit total power.  The Gamma-distributed training sum is
    evaluated through its exact noncentral-F ratio, independently of the production detector.
    """

    from scipy.stats import f, ncf

    alpha = float(threshold_scale)
    count = int(training_cells)
    snr = float(target_snr_linear)
    ca_cfar_false_alarm_probability(alpha, count)
    if not math.isfinite(snr) or snr < 0.0:
        raise ValueError("target_snr_linear must be finite and nonnegative")
    if alpha == 0.0:
        return 1.0

    # X = 2*|sqrt(snr)+w|**2 ~ ncx2(2, 2*snr), Y = 2*sum(training)
    # ~ chi2(2*N), independently. Thus (X/2)/(Y/(2*N)) is noncentral F.
    # Direct quadrature on [0, infinity] can miss the concentrated Gamma mass
    # for large N and silently return near zero. Use the distribution SF.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html
    probability = float(
        f.sf(alpha, 2, 2 * count) if snr == 0.0 else ncf.sf(alpha, 2, 2 * count, 2.0 * snr)
    )
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("CA-CFAR detection probability is outside the supported numeric range")
    return probability


def os_cfar_rank_index(training_cells: int, rank_fraction: float) -> int:
    """Return the production-compatible zero-based ascending OS rank.

    The nearest integer is selected with ties to even, matching Python's
    :func:`round` and the production CFAR implementation exactly.
    """

    count = int(training_cells)
    fraction = float(rank_fraction)
    if count <= 0:
        raise ValueError("training_cells must be positive")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("rank_fraction must lie between zero and one")
    return int(np.clip(round(fraction * (count - 1)), 0, count - 1))


def os_cfar_false_alarm_probability(
    threshold_scale: float,
    training_cells: int,
    rank_index: int,
) -> float:
    """Return ideal OS-CFAR cell ``Pfa`` for exponential noise.

    ``rank_index`` is zero-based in the ascending ordered training samples.
    The result is the Laplace transform of that exponential order statistic.
    A log-product form avoids overflow for practical window sizes.
    """

    alpha = float(threshold_scale)
    count = int(training_cells)
    rank = int(rank_index)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("threshold_scale must be finite and nonnegative")
    if count <= 0:
        raise ValueError("training_cells must be positive")
    if not 0 <= rank < count:
        raise ValueError("rank_index must identify one ordered training cell")

    # For ascending order statistic j=rank+1 of N iid Exp(1), spacings have
    # rates N, N-1, ..., N-j+1.  Multiplying their Laplace transforms gives
    # E[exp(-alpha X_(j))].
    log_probability = 0.0
    for offset in range(rank + 1):
        rate = count - offset
        log_probability += math.log(rate) - math.log(rate + alpha)
    return float(math.exp(log_probability))


def os_cfar_threshold_scale(
    requested_pfa: float,
    training_cells: int,
    rank_index: int,
    *,
    relative_tolerance: float = 1e-12,
) -> float:
    """Invert the OS-CFAR false-alarm equation by monotone bisection."""

    probability = float(requested_pfa)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("requested_pfa must lie strictly between zero and one")
    # Validate the remaining arguments through the forward equation.
    os_cfar_false_alarm_probability(0.0, training_cells, rank_index)

    lower = 0.0
    upper = 1.0
    while os_cfar_false_alarm_probability(upper, training_cells, rank_index) > probability:
        upper *= 2.0
        if not math.isfinite(upper):
            raise RuntimeError("Could not bracket the OS-CFAR threshold multiplier")

    for _ in range(160):
        midpoint = (lower + upper) / 2.0
        observed = os_cfar_false_alarm_probability(midpoint, training_cells, rank_index)
        if observed > probability:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= relative_tolerance * max(1.0, midpoint):
            break
    return float((lower + upper) / 2.0)


def relative_complex_awgn_power(signal_power: float, snr_db: float) -> float:
    """Return total complex-noise power for a target-relative SNR.

    This is a statistical relative-power model.  It is not thermal noise and
    carries no watts, noise temperature, bandwidth, receiver-noise figure, or
    absolute link-budget scale.
    """

    power = float(signal_power)
    snr = float(snr_db)
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("signal_power must be finite and nonnegative")
    if not math.isfinite(snr):
        raise ValueError("snr_db must be finite")
    if power == 0.0:
        return 0.0
    try:
        noise_power = power * (10.0 ** (-snr / 10.0))
    except OverflowError as exc:
        raise ValueError("snr_db is outside the finite floating-point range") from exc
    if not math.isfinite(noise_power) or noise_power == 0.0:
        raise ValueError("snr_db is outside the finite floating-point range")
    return float(noise_power)


def finite_cpi_mean_removal_power_gain(
    normalized_frequency_cycles_per_sample: float,
    sample_count: int,
) -> float:
    """Return tone power retained by finite-CPI mean subtraction.

    For ``x[n] = exp(j 2 pi f n)``, subtracting the mean of the complete
    ``sample_count``-sample vector retains
    ``1 - |mean(x)|**2`` of its input power.  This finite-record projection is
    not an LTI high-pass filter and has no streaming group-delay claim.
    """

    frequency = float(normalized_frequency_cycles_per_sample)
    count = int(sample_count)
    if not math.isfinite(frequency):
        raise ValueError("normalized frequency must be finite")
    if count <= 0:
        raise ValueError("sample_count must be positive")
    samples = np.exp(2j * np.pi * frequency * np.arange(count, dtype=np.float64))
    retained = 1.0 - float(abs(np.mean(samples)) ** 2)
    return float(np.clip(retained, 0.0, 1.0))


def sliding_baseline_frequency_response(
    normalized_frequency_cycles_per_sample: float,
    window: int,
) -> complex:
    """Return the steady-state response of the previous-sample baseline remover.

    Once its history is full, the production streaming operation is
    ``y[n] = x[n] - mean(x[n-1], ..., x[n-window])``.
    """

    frequency = float(normalized_frequency_cycles_per_sample)
    count = int(window)
    if not math.isfinite(frequency):
        raise ValueError("normalized frequency must be finite")
    if count <= 0:
        raise ValueError("window must be positive")
    delays = np.arange(1, count + 1, dtype=np.float64)
    baseline_response = np.mean(np.exp(-2j * np.pi * frequency * delays))
    return complex(1.0 - baseline_response)


def wilson_binomial_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    hits = int(successes)
    count = int(trials)
    level = float(confidence)
    if count <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= hits <= count:
        raise ValueError("successes must lie between zero and trials")
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
    estimate = hits / count
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / count
    center = (estimate + z_squared / (2.0 * count)) / denominator
    half_width = (
        z_value
        * math.sqrt(estimate * (1.0 - estimate) / count + z_squared / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


__all__ = [
    "ca_cfar_detection_probability_nonfluctuating_target",
    "ca_cfar_false_alarm_probability",
    "ca_cfar_threshold_scale",
    "cfar_training_cell_count_1d",
    "cfar_training_cell_count_2d",
    "finite_cpi_mean_removal_power_gain",
    "os_cfar_false_alarm_probability",
    "os_cfar_rank_index",
    "os_cfar_threshold_scale",
    "relative_complex_awgn_power",
    "sliding_baseline_frequency_response",
    "wilson_binomial_interval",
]
