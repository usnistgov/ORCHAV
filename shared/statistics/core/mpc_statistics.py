"""Phase-free MPC statistics for frame summaries and comparisons.

``compute_mpc_statistics`` consumes per-path delay, path-loss, and optional
angle arrays and returns an immutable ``MPCStatistics`` record. The statistics
avoid absolute phase assumptions, so they are suitable for both simulation-only
summaries and measurement/reference comparisons.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .metrics import (
    compute_angular_spread,
    compute_delay_spread,
    compute_linear_angular_spread,
)


@dataclass(frozen=True)
class MPCStatistics:
    """Immutable container for phase-free MPC statistics.

    This is the canonical representation for MPC statistics used across
    comparison and visualization modules.

    All statistics are computed from magnitude and delay information only
    (no phase required), making them suitable for reference comparisons.

    Attributes:
        P_incoh: Sum of retained dimensionless linear path-gain ratios
        P_incoh_db: The retained aggregate path-gain ratio in dB
        sigma_tau_ns: RMS delay spread in nanoseconds
        mean_delay_ns: Power-weighted mean delay in nanoseconds
        N_paths: Number of paths above power threshold
        max_delay_ns: Maximum delay
        min_delay_ns: Minimum delay
        earliest_path_delay_ns: Delay of the earliest retained path, or None
        earliest_path_loss_db: Path loss of the earliest retained path, or None
        rms_aoa_az_deg: RMS azimuth spread at receiver (AoA)
        rms_aoa_el_deg: RMS elevation spread at receiver (AoA)
        rms_aod_az_deg: RMS azimuth spread at transmitter (AoD)
        rms_aod_el_deg: RMS elevation spread at transmitter (AoD)
        mean_aoa_az_deg: Power-weighted mean AoA azimuth
        mean_aoa_el_deg: Power-weighted mean AoA elevation
        mean_aod_az_deg: Power-weighted mean AoD azimuth
        mean_aod_el_deg: Power-weighted mean AoD elevation
    """

    # Power metrics
    P_incoh: float
    P_incoh_db: float

    # Delay metrics
    sigma_tau_ns: float
    mean_delay_ns: float
    max_delay_ns: float
    min_delay_ns: float

    # Path count
    N_paths: int

    # Earliest path. Reflection order, not delay alone, determines LoS status.
    earliest_path_delay_ns: Optional[float]
    earliest_path_loss_db: Optional[float]

    # Angular spreads (AoA - receiver side)
    rms_aoa_az_deg: float
    rms_aoa_el_deg: float
    mean_aoa_az_deg: float
    mean_aoa_el_deg: float

    # Angular spreads (AoD - transmitter side)
    rms_aod_az_deg: float
    rms_aod_el_deg: float
    mean_aod_az_deg: float
    mean_aod_el_deg: float

    def __str__(self) -> str:
        """Return a human-readable summary of the statistics."""
        lines = [
            "MPC Statistics",
            f"  Paths:       {self.N_paths}",
            f"  Retained gain: {self.P_incoh_db:.1f} dB",
            f"  Delay:       mean={self.mean_delay_ns:.1f} ns, "
            f"spread={self.sigma_tau_ns:.2f} ns, "
            f"range=[{self.min_delay_ns:.1f}, {self.max_delay_ns:.1f}] ns",
        ]
        if self.earliest_path_delay_ns is not None:
            lines.append(
                f"  Earliest:    delay={self.earliest_path_delay_ns:.1f} ns, "
                f"loss={self.earliest_path_loss_db:.1f} dB"
            )
        if self.rms_aoa_az_deg > 0 or self.rms_aod_az_deg > 0:
            lines.append(
                f"  AoA spread:  az={self.rms_aoa_az_deg:.1f} deg, "
                f"el={self.rms_aoa_el_deg:.1f} deg"
            )
            lines.append(
                f"  AoD spread:  az={self.rms_aod_az_deg:.1f} deg, "
                f"el={self.rms_aod_el_deg:.1f} deg"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "P_incoh": self.P_incoh,
            "P_incoh_db": self.P_incoh_db,
            "sigma_tau_ns": self.sigma_tau_ns,
            "mean_delay_ns": self.mean_delay_ns,
            "max_delay_ns": self.max_delay_ns,
            "min_delay_ns": self.min_delay_ns,
            "N_paths": self.N_paths,
            "earliest_path_delay_ns": self.earliest_path_delay_ns,
            "earliest_path_loss_db": self.earliest_path_loss_db,
            "rms_aoa_az_deg": self.rms_aoa_az_deg,
            "rms_aoa_el_deg": self.rms_aoa_el_deg,
            "mean_aoa_az_deg": self.mean_aoa_az_deg,
            "mean_aoa_el_deg": self.mean_aoa_el_deg,
            "rms_aod_az_deg": self.rms_aod_az_deg,
            "rms_aod_el_deg": self.rms_aod_el_deg,
            "mean_aod_az_deg": self.mean_aod_az_deg,
            "mean_aod_el_deg": self.mean_aod_el_deg,
        }


def _empty_statistics() -> MPCStatistics:
    """Return the neutral statistics record for empty or invalid path sets."""
    return MPCStatistics(
        P_incoh=0.0,
        P_incoh_db=float("-inf"),
        sigma_tau_ns=0.0,
        mean_delay_ns=0.0,
        max_delay_ns=0.0,
        min_delay_ns=0.0,
        N_paths=0,
        earliest_path_delay_ns=None,
        earliest_path_loss_db=None,
        rms_aoa_az_deg=0.0,
        rms_aoa_el_deg=0.0,
        mean_aoa_az_deg=0.0,
        mean_aoa_el_deg=0.0,
        rms_aod_az_deg=0.0,
        rms_aod_el_deg=0.0,
        mean_aod_az_deg=0.0,
        mean_aod_el_deg=0.0,
    )


def _circular_mean(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Compute power-weighted circular mean for angles."""
    if len(angles_deg) == 0:
        return 0.0
    angles_rad = np.radians(angles_deg)
    complex_vectors = weights * np.exp(1j * angles_rad)
    mean_complex = np.sum(complex_vectors) / np.sum(weights)
    return float(np.degrees(np.angle(mean_complex)))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute power-weighted mean."""
    if len(values) == 0:
        return 0.0
    return float(np.average(values, weights=weights))


def _path_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return one float64 value per path or reject an ambiguous shape."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {result.shape}")
    return result


def compute_mpc_statistics(
    delays_ns: np.ndarray,
    pathloss_db: np.ndarray,
    aoa_az_deg: Optional[np.ndarray] = None,
    aoa_el_deg: Optional[np.ndarray] = None,
    aod_az_deg: Optional[np.ndarray] = None,
    aod_el_deg: Optional[np.ndarray] = None,
    threshold_db: float = -30.0,
) -> MPCStatistics:
    """
    Compute all MPC statistics from raw path data.

    This shared implementation is used by comparison tools, visualization
    summaries, and analysis diagnostics.

    Args:
        delays_ns: Path delays in nanoseconds, shape (N,)
        pathloss_db: Path loss in dB (positive values), shape (N,)
        aoa_az_deg: Angle of arrival azimuth in degrees, shape (N,) or None
        aoa_el_deg: Angle of arrival elevation in degrees, shape (N,) or None
        aod_az_deg: Angle of departure azimuth in degrees, shape (N,) or None
        aod_el_deg: Angle of departure elevation in degrees, shape (N,) or None
        threshold_db: Path-gain threshold in dB relative to the pre-threshold
            sum of all finite path-gain ratios.
            Paths weaker than this are excluded from statistics.
            Default: -30 dB (0.1% of that aggregate)

    Returns:
        MPCStatistics: Frozen dataclass with all computed statistics

    Example:
        >>> delays = np.array([10, 20, 30])
        >>> pathloss = np.array([60, 70, 80])
        >>> stats = compute_mpc_statistics(delays, pathloss)
        >>> stats.N_paths
        3
    """
    if not np.isfinite(threshold_db) or threshold_db > 0.0:
        raise ValueError("threshold_db must be finite and less than or equal to 0 dB")

    delays_ns = _path_vector(delays_ns, "delays_ns")
    pathloss_db = _path_vector(pathloss_db, "pathloss_db")
    if delays_ns.shape != pathloss_db.shape:
        raise ValueError(
            "delays_ns and pathloss_db must describe the same paths: "
            f"{delays_ns.shape} != {pathloss_db.shape}"
        )
    if delays_ns.size == 0:
        return _empty_statistics()

    source_indices = np.arange(delays_ns.size, dtype=np.intp)
    finite = np.isfinite(delays_ns) & np.isfinite(pathloss_db)
    delays_ns = delays_ns[finite]
    pathloss_db = pathloss_db[finite]
    source_indices = source_indices[finite]
    if delays_ns.size == 0:
        return _empty_statistics()

    # Path loss weights statistics by its dimensionless received-power ratio.
    with np.errstate(over="ignore", under="ignore"):
        powers = 10.0 ** (-pathloss_db / 10.0)
    usable_power = np.isfinite(powers) & (powers > 0.0)
    delays_ns = delays_ns[usable_power]
    pathloss_db = pathloss_db[usable_power]
    source_indices = source_indices[usable_power]
    powers = powers[usable_power]
    if powers.size == 0:
        return _empty_statistics()

    # The cutoff is relative to the pre-threshold path-gain aggregate so weak
    # numerical tails do not dominate delay and angular spread statistics.
    P_total = float(np.sum(powers))
    if not np.isfinite(P_total) or P_total <= 0.0:
        return _empty_statistics()

    threshold_linear = P_total * 10 ** (threshold_db / 10)
    mask = powers >= threshold_linear

    # If no paths above threshold, return empty
    if not np.any(mask):
        return _empty_statistics()

    valid_powers = powers[mask]
    valid_delays = delays_ns[mask]
    valid_pathloss = pathloss_db[mask]
    valid_source_indices = source_indices[mask]

    # Retained aggregate path-gain ratio after the relative cutoff.
    P_incoh = float(np.sum(valid_powers))
    P_incoh_db = float(10.0 * np.log10(P_incoh))

    # Delay statistics (power-weighted)
    sigma_tau = compute_delay_spread(valid_delays, valid_powers)
    mean_delay = _weighted_mean(valid_delays, valid_powers)

    earliest_idx = int(np.argmin(valid_delays))
    earliest_path_delay = float(valid_delays[earliest_idx])
    earliest_path_loss = float(valid_pathloss[earliest_idx])

    # Angular statistics
    def compute_angular_stats(
        angles_deg: Optional[np.ndarray], *, circular: bool
    ) -> tuple[float, float]:
        """Compute one angle's statistics on finite retained paths."""
        if angles_deg is None:
            return 0.0, 0.0
        angle_vector = _path_vector(angles_deg, "angle array")
        if angle_vector.shape[0] != finite.shape[0]:
            raise ValueError(
                "angle arrays must describe the same paths as delays_ns and pathloss_db: "
                f"{angle_vector.shape} != {finite.shape}"
            )
        selected_angles = angle_vector[valid_source_indices]
        angle_valid = np.isfinite(selected_angles)
        if not np.any(angle_valid):
            return 0.0, 0.0
        selected_angles = selected_angles[angle_valid]
        angle_powers = valid_powers[angle_valid]
        if circular:
            rms = compute_angular_spread(selected_angles, angle_powers)
            mean = _circular_mean(selected_angles, angle_powers)
        else:
            rms = compute_linear_angular_spread(selected_angles, angle_powers)
            mean = _weighted_mean(selected_angles, angle_powers)
        return rms, mean

    rms_aoa_az, mean_aoa_az = compute_angular_stats(aoa_az_deg, circular=True)
    rms_aoa_el, mean_aoa_el = compute_angular_stats(aoa_el_deg, circular=False)
    rms_aod_az, mean_aod_az = compute_angular_stats(aod_az_deg, circular=True)
    rms_aod_el, mean_aod_el = compute_angular_stats(aod_el_deg, circular=False)

    return MPCStatistics(
        P_incoh=P_incoh,
        P_incoh_db=P_incoh_db,
        sigma_tau_ns=sigma_tau,
        mean_delay_ns=mean_delay,
        max_delay_ns=float(np.max(valid_delays)),
        min_delay_ns=float(np.min(valid_delays)),
        N_paths=int(np.sum(mask)),
        earliest_path_delay_ns=earliest_path_delay,
        earliest_path_loss_db=earliest_path_loss,
        rms_aoa_az_deg=rms_aoa_az,
        rms_aoa_el_deg=rms_aoa_el,
        mean_aoa_az_deg=mean_aoa_az,
        mean_aoa_el_deg=mean_aoa_el,
        rms_aod_az_deg=rms_aod_az,
        rms_aod_el_deg=rms_aod_el,
        mean_aod_az_deg=mean_aod_az,
        mean_aod_el_deg=mean_aod_el,
    )


def compute_comparison_stats(meas_stats: MPCStatistics, sim_stats: MPCStatistics) -> Dict[str, Any]:
    """
    Compute comparison statistics between measurement and simulation.

    Args:
        meas_stats: Statistics from measurements
        sim_stats: Statistics from simulation

    Returns:
        Dictionary with comparison metrics (differences, ratios, etc.)
    """
    return {
        # Power differences
        "P_incoh_diff_db": sim_stats.P_incoh_db - meas_stats.P_incoh_db,
        # Delay differences
        "sigma_tau_diff_ns": sim_stats.sigma_tau_ns - meas_stats.sigma_tau_ns,
        "sigma_tau_ratio": (
            sim_stats.sigma_tau_ns / meas_stats.sigma_tau_ns
            if meas_stats.sigma_tau_ns > 0
            else float("inf")
        ),
        "mean_delay_diff_ns": sim_stats.mean_delay_ns - meas_stats.mean_delay_ns,
        # Path count
        "N_paths_diff": sim_stats.N_paths - meas_stats.N_paths,
        "N_paths_ratio": (
            sim_stats.N_paths / meas_stats.N_paths if meas_stats.N_paths > 0 else float("inf")
        ),
        # Earliest-path comparison. Delay alone does not establish LoS status.
        "earliest_path_delay_diff_ns": (
            sim_stats.earliest_path_delay_ns - meas_stats.earliest_path_delay_ns
            if sim_stats.earliest_path_delay_ns is not None
            and meas_stats.earliest_path_delay_ns is not None
            else None
        ),
        "earliest_path_loss_diff_db": (
            sim_stats.earliest_path_loss_db - meas_stats.earliest_path_loss_db
            if sim_stats.earliest_path_loss_db is not None
            and meas_stats.earliest_path_loss_db is not None
            else None
        ),
        # Angular spreads
        "rms_aoa_az_diff_deg": sim_stats.rms_aoa_az_deg - meas_stats.rms_aoa_az_deg,
        "rms_aod_az_diff_deg": sim_stats.rms_aod_az_deg - meas_stats.rms_aod_az_deg,
    }
