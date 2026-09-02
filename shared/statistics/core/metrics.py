"""Core MPC metric computations.

These helpers operate on already-loaded arrays from canonical frame data. Units
are explicit in function names and docstrings: delays are nanoseconds, angular
spreads are degrees, powers are linear unless a function explicitly returns dB
or dBm.
"""

from typing import Tuple

import numpy as np


def compute_delay_spread(delays_ns: np.ndarray, powers_linear: np.ndarray) -> float:
    """
    Compute RMS delay spread in nanoseconds.

    Formula: σ_τ = sqrt(Σ(P_i × (τ_i - μ)²) / Σ P_i)
    where μ = Σ(P_i × τ_i) / Σ P_i (power-weighted mean delay)

    This is the IEEE 802.11 / 3GPP standard definition.

    Args:
        delays_ns: Path delays in nanoseconds, shape (N,)
        powers_linear: Path powers in linear scale (NOT dB), shape (N,)

    Returns:
        RMS delay spread in nanoseconds. Returns 0.0 for empty input or
        zero total power.

    Example:
        >>> delays = np.array([10, 20, 30])
        >>> powers = np.array([1, 2, 3])
        >>> spread = compute_delay_spread(delays, powers)
        >>> round(spread, 2)
        7.45
    """
    if len(delays_ns) == 0:
        return 0.0
    total_power = np.sum(powers_linear)
    if total_power <= 0:
        return 0.0

    mean_delay = np.average(delays_ns, weights=powers_linear)
    variance = np.average((delays_ns - mean_delay) ** 2, weights=powers_linear)
    return float(np.sqrt(variance))


def compute_angular_spread(
    angles_deg: np.ndarray,
    powers_linear: np.ndarray,
) -> float:
    """
    Compute a power-weighted RMS of wrapped azimuth deviations in degrees.

    Uses complex arithmetic to handle angle wrapping correctly.
    The result is the weighted RMS distance from the circular mean, not the
    conventional circular-variance statistic ``1 - R``.

    Args:
        angles_deg: Angles in degrees, shape (N,)
        powers_linear: Path powers in linear scale for weighting, shape (N,)

    Returns:
        RMS angular spread in degrees. Returns 0.0 when fewer than two
        finite angles have finite positive power.

    Example:
        >>> angles = np.array([350, 0, 10])  # Around 0 degrees
        >>> powers = np.array([1, 1, 1])
        >>> spread = compute_angular_spread(angles, powers)
        >>> round(spread, 1)
        8.2
    """
    if len(angles_deg) == 0:
        return 0.0

    angles_deg = np.asarray(angles_deg)
    powers_linear = np.asarray(powers_linear)
    usable = np.isfinite(angles_deg) & np.isfinite(powers_linear) & (powers_linear > 0)
    if np.count_nonzero(usable) < 2:
        return 0.0

    angles_deg = angles_deg[usable]
    powers_linear = powers_linear[usable]
    total_power = np.sum(powers_linear)

    # Convert to radians for circular statistics
    angles_rad = np.radians(angles_deg)

    # Compute mean direction using complex arithmetic
    complex_vectors = powers_linear * np.exp(1j * angles_rad)
    mean_complex = np.sum(complex_vectors) / total_power
    mean_angle_rad = np.angle(mean_complex)

    # Compute angular variance
    angular_diff = np.angle(np.exp(1j * (angles_rad - mean_angle_rad)))
    variance = np.average(angular_diff**2, weights=powers_linear)

    # RMS angular spread (convert back to degrees)
    return float(np.sqrt(variance) * 180 / np.pi)


def compute_linear_angular_spread(
    angles_deg: np.ndarray,
    powers_linear: np.ndarray,
) -> float:
    """Compute a power-weighted RMS spread for non-periodic angles.

    Elevation has physical endpoints rather than a 360-degree wrap, so it must
    use ordinary weighted differences. Azimuth should use
    :func:`compute_angular_spread` instead.

    Args:
        angles_deg: Elevation-like angles in degrees, shape ``(N,)``.
        powers_linear: Linear path-power weights, shape ``(N,)``.

    Returns:
        RMS angular spread in degrees, or ``0.0`` for empty or zero-weight
        input.
    """
    if len(angles_deg) == 0:
        return 0.0
    total_power = np.sum(powers_linear)
    if total_power <= 0:
        return 0.0

    mean_angle = np.average(angles_deg, weights=powers_linear)
    variance = np.average((angles_deg - mean_angle) ** 2, weights=powers_linear)
    return float(np.sqrt(variance))


def compute_power_delay_profile(
    delays_ns: np.ndarray,
    powers_linear: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Delay Profile (PDP) - power vs delay.

    Returns individual finite, positive MPC powers at their respective delays,
    sorted by delay. Invalid records and zero-power records are omitted.

    Args:
        delays_ns: Path delays in nanoseconds, shape (N,)
        powers_linear: Path powers in linear scale, shape (N,)

    Returns:
        Tuple of (sorted_delays_ns, powers_db)
        Both arrays have shape ``(M,)`` for the retained records.

    Raises:
        ValueError: If the inputs are not aligned one-dimensional arrays.

    Example:
        >>> delays = np.array([30, 10, 20])
        >>> powers = np.array([0.1, 1.0, 0.5])
        >>> d, p = compute_power_delay_profile(delays, powers)
        >>> d.tolist()
        [10, 20, 30]
    """
    delays = np.asarray(delays_ns, dtype=np.float64)
    powers = np.asarray(powers_linear, dtype=np.float64)
    if delays.ndim != 1 or powers.ndim != 1 or delays.shape != powers.shape:
        raise ValueError("delays_ns and powers_linear must be aligned one-dimensional arrays")

    usable = np.isfinite(delays) & np.isfinite(powers) & (powers > 0.0)
    delays = delays[usable]
    powers = powers[usable]
    if delays.size == 0:
        return np.array([]), np.array([])

    # Sort by delay for proper visualization
    sorted_indices = np.argsort(delays)
    sorted_delays = delays[sorted_indices]
    sorted_powers = powers[sorted_indices]

    powers_db = 10.0 * np.log10(sorted_powers)

    return sorted_delays, powers_db


def compute_binned_power_delay_profile(
    delays_ns: np.ndarray,
    powers_linear: np.ndarray,
    coalescence_threshold_ns: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine nearby paths into an incoherent power-delay profile.

    A bin starts at the earliest unassigned delay and includes later samples
    no more than ``coalescence_threshold_ns`` from that first delay. The bin
    delay is power weighted and its value is
    ``10 * log10(sum(powers_linear))``. This operation does not construct a
    complex channel impulse response because no path phase is available.

    Args:
        delays_ns: Path delays in nanoseconds, shape ``(N,)``.
        powers_linear: Linear path-gain ratios, shape ``(N,)``. Only finite,
            positive values are retained.
        coalescence_threshold_ns: Maximum separation from the first path delay
            in one bin.

    Returns:
        Tuple of sorted bin delays in nanoseconds and summed bin power in dB.

    Raises:
        ValueError: If inputs are not aligned one-dimensional arrays or the
            threshold is negative or non-finite.
    """
    delays = np.asarray(delays_ns, dtype=np.float64)
    powers = np.asarray(powers_linear, dtype=np.float64)
    if delays.ndim != 1 or powers.ndim != 1 or delays.shape != powers.shape:
        raise ValueError("delays_ns and powers_linear must be aligned one-dimensional arrays")
    if not np.isfinite(coalescence_threshold_ns) or coalescence_threshold_ns < 0.0:
        raise ValueError("coalescence_threshold_ns must be finite and non-negative")

    valid = np.isfinite(delays) & np.isfinite(powers) & (powers > 0.0)
    delays = delays[valid]
    powers = powers[valid]
    if delays.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    order = np.argsort(delays)
    delays = delays[order]
    powers = powers[order]
    group_starts = [0]
    current_start = 0
    for index in range(1, delays.size):
        if delays[index] - delays[current_start] > coalescence_threshold_ns:
            group_starts.append(index)
            current_start = index

    starts = np.asarray(group_starts, dtype=np.intp)
    bin_powers = np.add.reduceat(powers, starts)
    weighted_delays = np.add.reduceat(delays * powers, starts)
    bin_delays = weighted_delays / bin_powers
    powers_db = 10.0 * np.log10(bin_powers)
    return bin_delays, powers_db


def compute_signal_strength_distribution(
    powers_linear: np.ndarray,
    num_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of signal strengths in dBm.

    Args:
        powers_linear: Path powers in linear scale (Watts), shape (N,)
        num_bins: Number of histogram bins. Default: 20

    Returns:
        Tuple of (bin_centers_dBm, counts)
        bin_centers_dBm is in dBm, counts is the histogram count.

    Example:
        >>> powers = np.array([1e-6, 1e-5, 1e-4])
        >>> centers, counts = compute_signal_strength_distribution(powers)
    """
    if len(powers_linear) == 0:
        return np.array([]), np.array([])

    # Convert to dBm for better visualization
    # P_dBm = 10 * log10(P_W * 1000) = 10 * log10(P_W) + 30
    powers_dbm = 10 * np.log10(np.maximum(powers_linear, 1e-18) * 1e3)

    # Create histogram
    counts, bin_edges = np.histogram(powers_dbm, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, counts


def compute_snr_distribution(
    powers_linear: np.ndarray,
    noise_power_w: float = 1e-12,
    num_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SNR distribution histogram.

    Args:
        powers_linear: Path powers in linear scale (Watts), shape (N,)
        noise_power_w: Noise power in Watts. Default: 1e-12 W (-90 dBm)
        num_bins: Number of histogram bins. Default: 20

    Returns:
        Tuple of (bin_centers_dB, counts)
        bin_centers_dB is SNR in dB, counts is the histogram count.

    Example:
        >>> powers = np.array([1e-6, 1e-7, 1e-8])
        >>> centers, counts = compute_snr_distribution(powers)
    """
    if len(powers_linear) == 0:
        return np.array([]), np.array([])

    # Calculate SNR for each MPC
    snr_linear = powers_linear / noise_power_w
    snr_db = 10 * np.log10(np.maximum(snr_linear, 1e-6))

    # Create histogram
    counts, bin_edges = np.histogram(snr_db, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, counts


def pathloss_db_to_power_linear(pathloss_db: np.ndarray) -> np.ndarray:
    """
    Convert path loss in dB to linear power.

    Formula: P_linear = 10^(-PL_dB / 10)

    Args:
        pathloss_db: Path loss in dB (positive values)

    Returns:
        Linear power (dimensionless, relative to TX power)
    """
    return 10 ** (-pathloss_db / 10)


def power_linear_to_db(power_linear: np.ndarray) -> np.ndarray:
    """
    Convert linear power to dB.

    Formula: P_dB = 10 * log10(P_linear)

    Args:
        power_linear: Linear power values

    Returns:
        Power in dB
    """
    return 10 * np.log10(np.maximum(power_linear, 1e-18))
