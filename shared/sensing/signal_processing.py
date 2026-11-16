"""Shared signal-processing helpers for optional frame sensing payloads.

The generator may attach sensing-derived arrays to ``StandardMPCFrame`` under
the optional ``sensing`` extension key. This module contains reusable numerical
helpers and presentation metadata for those payloads so generation,
post-processing, and visualization interpret range, Doppler, and detection
velocity consistently.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from shared.sensing.targets import (
    INSTANTANEOUS_VELOCITY_REFERENCE_POLICY,
    target_rows_by_id,
)

# Use shared logger if available, else standard logging
try:
    from shared.logging import get_logger

    logger = get_logger("shared.sensing")
except ImportError:
    logger = logging.getLogger("shared.sensing")

C0 = 299_792_458.0  # speed of light [m/s]

# Machine-readable names for the current CIR-domain sensing approximation.
# These values are persisted with generated products so consumers do not infer
# waveform-level range processing that did not occur.
DELAY_PROJECTION_POINT_BIN_FLOOR = "point_bin_floor"
RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS = "delay_domain_cir_bins"
DOPPLER_FFT_NORMALIZATION_UNITARY = "unitary_1_over_sqrt_n"
WINDOW_NORMALIZATION_ENERGY = "energy_preserving_rms"
# Historical sampled RD policy. Coherent processing can explicitly select
# none or windowed_mean; these are distinct from sliding-baseline removal.
SLOW_TIME_MEAN_REMOVAL_POLICY_NONZERO_VARIATION = "when_row_variation_nonzero"
SLOW_TIME_MEAN_REMOVAL_POLICY_WINDOWED = "window_weighted_mean"


def coherent_clutter_metadata(mode: str) -> dict[str, Any]:
    """Describe the executed coherent projection for generation and replay.

    Omitted configuration is resolved to legacy by callers for compatibility.
    This metadata describes the operator, not whether a particular legacy CPI
    had sufficient row variation to execute it.
    """
    if mode not in {"none", "legacy", "windowed_mean"}:
        raise ValueError("coherent_clutter_mode must be none, legacy or windowed_mean")
    disabled = mode == "none"
    weighted = mode == "windowed_mean"
    return {
        "slow_time_mean_removal_policy": {
            "none": "disabled",
            "legacy": SLOW_TIME_MEAN_REMOVAL_POLICY_NONZERO_VARIATION,
            "windowed_mean": SLOW_TIME_MEAN_REMOVAL_POLICY_WINDOWED,
        }[mode],
        "slow_time_mean_removal_operator": "identity" if disabled else "finite_cpi_mean_projection",
        "slow_time_mean_removal_transfer_model": (
            "identity" if disabled else "finite_record_projection_not_lti"
        ),
        "slow_time_mean_removal_tone_power_gain": (
            "1"
            if disabled
            else (
                "window_dependent_finite_record_projection"
                if weighted
                else "1-abs(mean(exp(j*2*pi*f*n)))**2"
            )
        ),
        "slow_time_mean_removal_group_delay": (
            "0" if disabled else "undefined_for_non_lti_projection"
        ),
        "slow_time_mean_removal_result_availability": "after_complete_cpi",
        "slow_time_mean_removal_decision_statistic": (
            "disabled_by_configuration"
            if disabled
            else "configured_every_complete_cpi" if weighted else "max_abs_first_difference"
        ),
        "slow_time_mean_removal_decision_threshold": 0.0,
    }


def remove_window_weighted_mean(
    samples: np.ndarray, window: np.ndarray, *, axis: int
) -> np.ndarray:
    """Remove a constant component whose windowed Doppler DC is zero.

    Center on the first sample before summation so an exactly constant input
    becomes exactly zero, even when its magnitude is far above a moving echo.
    Subtract the taper-weighted mean *before* applying that taper. This is a
    finite-record projection: it also attenuates genuine near-zero Doppler,
    and must not be interpreted as a stationary-target detector.
    """
    values = np.asarray(samples)
    weights = np.asarray(window, dtype=float)
    if (
        values.ndim == 0
        or not -values.ndim <= axis < values.ndim
        or weights.shape != (values.shape[axis],)
        or not weights.size
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(weights < -np.finfo(float).eps * max(1.0, float(np.max(abs(weights)))))
    ):
        raise ValueError("Mean projection requires finite samples and a matching nonnegative taper")
    weights = np.maximum(weights, 0.0)
    shape = [1] * values.ndim
    shape[axis] = len(weights)
    # Degenerate tapers (e.g. two-point symmetric Hann) produce zero output
    # later; use uniform weights to keep this intermediate defined.
    total = float(weights.sum())
    weights = weights / total if total > 0 else np.full(weights.shape, 1 / len(weights))
    centered = values - np.take(values, [0], axis=axis)
    return centered - np.sum(centered * weights.reshape(shape), axis=axis, keepdims=True)


# The analytic path-Doppler oracle can reject low-|Doppler| paths directly
# before nearest-bin deposition.  Persist these policy names on its typed RD
# products so consumers do not confuse that operation with slow-time clutter
# removal used by sampled processors.
DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_DISABLED = "disabled"
DIRECT_DOPPLER_CLUTTER_FILTER_POLICY_EXCLUDE_BELOW_CUTOFF = "exclude_abs_doppler_below_cutoff"


def normalize_range_mode(range_mode: str | None) -> str:
    """Normalize a range-mode string to ``monostatic`` or ``bistatic``."""
    if isinstance(range_mode, str) and range_mode.lower().startswith("mono"):
        return "monostatic"
    return "bistatic"


@dataclass(frozen=True)
class DetectionVelocityDisplay:
    """Presentation metadata for detection-reported velocity values."""

    semantics: str
    axis_label: str
    inline_suffix: str
    is_approximate: bool


def detection_velocity_display(
    range_mode: str = "bistatic",
) -> DetectionVelocityDisplay:
    """Describe how detection velocity should be presented to users.

    A range-Doppler cell directly measures Doppler, not a target velocity
    vector. ``velocity_m_s`` is therefore always the target-independent
    conversion ``wavelength * doppler_hz / 2``. It is radial closing velocity
    for a monostatic link and a monostatic-equivalent observable for a
    bistatic link. Geometry-conditioned bistatic speed belongs to evaluation,
    not detection.
    """
    mode = normalize_range_mode(range_mode)
    if mode == "monostatic":
        return DetectionVelocityDisplay(
            semantics="monostatic_radial_closing_velocity",
            axis_label="Radial Closing Velocity (m/s)",
            inline_suffix="",
            is_approximate=False,
        )

    return DetectionVelocityDisplay(
        semantics="monostatic_equivalent_velocity",
        axis_label="Monostatic-equivalent Velocity (m/s)",
        inline_suffix=" mono. equiv.",
        is_approximate=False,
    )


def format_detection_velocity_m_s(
    value_m_s: float,
    *,
    range_mode: str = "bistatic",
    precision: int = 1,
) -> str:
    """Format a detection velocity for UI display."""
    display = detection_velocity_display(range_mode=range_mode)
    return f"{float(value_m_s):.{precision}f} m/s{display.inline_suffix}"


def apply_detection_velocity_metadata(
    mapping: dict[str, object],
    *,
    range_mode: str = "bistatic",
) -> None:
    """Attach detection-velocity presentation metadata to a sensing mapping."""
    display = detection_velocity_display(range_mode=range_mode)
    mapping["detection_velocity_semantics"] = display.semantics
    mapping["detection_velocity_axis_label"] = display.axis_label
    mapping["detection_velocity_is_approximate"] = display.is_approximate
    mapping["doppler_sign_convention"] = "positive_approaching"
    mapping["path_closing_rate_formula"] = "wavelength_m * doppler_hz"
    mapping["detection_velocity_formula"] = "0.5 * wavelength_m * doppler_hz"
    mapping["detection_velocity_geometry_source"] = "none"


def _safe_normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None
    return vector / norm


def radial_velocity_axis(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    range_mode: str = "bistatic",
) -> np.ndarray:
    """Return the monostatic LOS or observable bistatic unit bisector.

    For monostatic mode this is the TX-target line of sight. For bistatic mode
    this is the unit bisector of the TX-target and RX-target directions.
    Coincident endpoints and the forward-scatter singularity are rejected;
    substituting a TX line of sight would fabricate observability.
    """
    tx = np.asarray(tx_pos, dtype=np.float64)
    rx = np.asarray(rx_pos, dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)

    los_tx = _safe_normalize(tx - target)
    if los_tx is None:
        raise ValueError("radial velocity axis is undefined at a coincident TX endpoint")

    if str(range_mode).lower().startswith("mono"):
        return los_tx

    los_rx = _safe_normalize(rx - target)
    if los_rx is None:
        raise ValueError("bistatic velocity axis is undefined at a coincident RX endpoint")

    bisector = _safe_normalize(los_tx + los_rx)
    if bisector is None:
        raise ValueError("bistatic velocity axis is unobservable near forward scatter")
    return bisector


def _finite_vector3(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return *value* as a finite three-vector or raise ``ValueError``."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return vector


def _positive_finite_wavelength(wavelength_m: float) -> float:
    """Validate and return a physical wavelength in metres."""
    wavelength = float(wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength_m must be finite and greater than zero")
    return wavelength


def doppler_to_path_closing_rate_m_s(doppler_hz: float, wavelength_m: float) -> float:
    """Convert Doppler to path-closing rate, positive for decreasing path length.

    ORCHAV follows the Sionna RT phase convention
    ``a(t) = a(0) * exp(j * 2*pi*doppler_hz*t)``. For a path of length
    ``L``, ``doppler_hz = -(1 / wavelength_m) * dL/dt``. The returned
    observable is therefore ``-dL/dt = wavelength_m * doppler_hz``.
    """
    doppler = float(doppler_hz)
    if not np.isfinite(doppler):
        raise ValueError("doppler_hz must be finite")
    return doppler * _positive_finite_wavelength(wavelength_m)


def doppler_to_monostatic_equivalent_velocity_m_s(
    doppler_hz: float,
    wavelength_m: float,
) -> float:
    """Return ``wavelength_m * doppler_hz / 2`` without using target truth.

    The result is radial closing velocity for a monostatic path. For a
    bistatic path it is an exactly defined monostatic-equivalent observable,
    not an estimate of the target's physical speed.
    """
    return 0.5 * doppler_to_path_closing_rate_m_s(doppler_hz, wavelength_m)


def doppler_to_velocity_m_s(doppler_hz: float, wavelength_m: float) -> float:
    """Return the target-independent monostatic-equivalent velocity.

    This narrow compatibility name never accepts target geometry. New code
    should prefer :func:`doppler_to_monostatic_equivalent_velocity_m_s`.
    """
    return doppler_to_monostatic_equivalent_velocity_m_s(doppler_hz, wavelength_m)


def bistatic_path_closing_rate_m_s(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    target_pos: np.ndarray,
    target_velocity_m_s: np.ndarray,
    *,
    tx_velocity_m_s: np.ndarray | None = None,
    rx_velocity_m_s: np.ndarray | None = None,
) -> float:
    """Return ``-dL/dt`` for a bistatic path, positive while it shortens.

    ``L = ||target - tx|| + ||target - rx||``. Sensor velocities default to
    zero. This helper is suitable for truth/evaluation and analytic reference
    cases; an estimator must not fill ``target_pos`` from ground truth.
    """
    tx = _finite_vector3(tx_pos, name="tx_pos")
    rx = _finite_vector3(rx_pos, name="rx_pos")
    target = _finite_vector3(target_pos, name="target_pos")
    target_velocity = _finite_vector3(target_velocity_m_s, name="target_velocity_m_s")
    tx_velocity = (
        np.zeros(3, dtype=np.float64)
        if tx_velocity_m_s is None
        else _finite_vector3(tx_velocity_m_s, name="tx_velocity_m_s")
    )
    rx_velocity = (
        np.zeros(3, dtype=np.float64)
        if rx_velocity_m_s is None
        else _finite_vector3(rx_velocity_m_s, name="rx_velocity_m_s")
    )

    target_to_tx = _safe_normalize(tx - target)
    target_to_rx = _safe_normalize(rx - target)
    if target_to_tx is None or target_to_rx is None:
        raise ValueError("bistatic path direction is undefined at a coincident endpoint")

    return float(
        np.dot(target_to_tx, target_velocity - tx_velocity)
        + np.dot(target_to_rx, target_velocity - rx_velocity)
    )


def bistatic_bisector_velocity_from_doppler_m_s(
    doppler_hz: float,
    wavelength_m: float,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    tx_velocity_m_s: np.ndarray | None = None,
    rx_velocity_m_s: np.ndarray | None = None,
    minimum_geometry_scale: float = 1e-9,
) -> float:
    """Return target velocity projected onto the bistatic unit bisector.

    This geometry-conditioned quantity is defined only when the TX/RX look
    direction sum is observable. It corrects for known sensor velocities and
    raises ``ValueError`` at coincident endpoints or near forward scatter.
    It belongs to evaluation or a geometry-estimating processor, never a
    truth-assisted detector.
    """
    tx = _finite_vector3(tx_pos, name="tx_pos")
    rx = _finite_vector3(rx_pos, name="rx_pos")
    target = _finite_vector3(target_pos, name="target_pos")
    tx_velocity = (
        np.zeros(3, dtype=np.float64)
        if tx_velocity_m_s is None
        else _finite_vector3(tx_velocity_m_s, name="tx_velocity_m_s")
    )
    rx_velocity = (
        np.zeros(3, dtype=np.float64)
        if rx_velocity_m_s is None
        else _finite_vector3(rx_velocity_m_s, name="rx_velocity_m_s")
    )

    target_to_tx = _safe_normalize(tx - target)
    target_to_rx = _safe_normalize(rx - target)
    if target_to_tx is None or target_to_rx is None:
        raise ValueError("bistatic path direction is undefined at a coincident endpoint")
    direction_sum = target_to_tx + target_to_rx
    geometry_scale = float(np.linalg.norm(direction_sum))
    threshold = float(minimum_geometry_scale)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_geometry_scale must be finite and greater than zero")
    if geometry_scale <= threshold:
        raise ValueError("bistatic bisector velocity is unobservable near forward scatter")

    path_closing_rate = doppler_to_path_closing_rate_m_s(doppler_hz, wavelength_m)
    sensor_term = float(np.dot(target_to_tx, tx_velocity) + np.dot(target_to_rx, rx_velocity))
    return (path_closing_rate + sensor_term) / geometry_scale


def doppler_axis_to_velocity_m_s(
    doppler_hz: np.ndarray,
    carrier_frequency_hz: float,
) -> np.ndarray:
    """Convert a Doppler axis in Hz to the reported velocity axis in m/s."""
    arr = np.asarray(doppler_hz, dtype=np.float64)
    carrier_frequency_hz = float(carrier_frequency_hz)
    if carrier_frequency_hz <= 0.0:
        return np.zeros_like(arr)
    lambda_c = C0 / carrier_frequency_hz
    return arr * lambda_c / 2.0


def sensing_window(name: str | None, length: int, **kwargs: Any) -> np.ndarray:
    """Return a 1-D window of *length* samples.

    Supported names (case-insensitive):
        hann / hanning, hamming, blackman, blackman-harris / blackmanharris,
        kaiser (requires ``beta`` kwarg, default 14.0),
        rect / rectangular / boxcar / none.

    Unrecognised names fall back to Hann.
    """
    _positive_integer(length, "window length")
    if length <= 1:
        return np.ones(max(length, 0), dtype=np.float64)
    if not name:
        return np.ones(length, dtype=np.float64)
    name = name.lower().replace("-", "").replace("_", "")
    if name in ("hann", "hanning"):
        return np.hanning(length)
    if name == "hamming":
        return np.hamming(length)
    if name == "blackman":
        return np.blackman(length)
    if name in ("blackmanharris",):
        # 4-term Blackman-Harris (minimum sidelobe, -92 dB)
        n = np.arange(length, dtype=np.float64)
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        return (
            a0
            - a1 * np.cos(2 * np.pi * n / (length - 1))
            + a2 * np.cos(4 * np.pi * n / (length - 1))
            - a3 * np.cos(6 * np.pi * n / (length - 1))
        )
    if name == "kaiser":
        beta = float(kwargs.get("beta", 14.0))
        if not np.isfinite(beta):
            raise ValueError("Kaiser beta must be finite")
        return np.kaiser(length, beta)
    if name in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(length, dtype=np.float64)
    # Preserve the shared helper's historical Hann fallback
    return np.hanning(length)


def window_energy_correction(window: np.ndarray) -> float:
    """
    Compute energy correction factor for a window to preserve power.
    Returns the scaling factor to multiply windowed signal by.
    """
    window = np.asarray(window, dtype=float)
    if window.ndim != 1 or not window.size or not np.isfinite(window).all():
        raise ValueError("Window must be a finite nonempty vector")
    energy = np.sum(window**2)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Energy normalization requires a nonzero finite window")
    return float(np.sqrt(window.size / energy))


# Internal callers and recorded reproduction scripts use the same implementation.
_window = sensing_window
_window_energy_correction = window_energy_correction


def _positive_integer(value: int, name: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _positive_rate(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _finite_sample_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Samples must be a finite [slow time, range] matrix")
    return values


def project_cir_to_delay_bins(
    cir: np.ndarray,
    delays_s: np.ndarray,
    *,
    sampling_frequency: float,
    fft_size: int,
) -> np.ndarray:
    """Project sparse CIR paths onto delay bins with a point-bin model.

    Each path is deposited into ``floor(delay_s * sampling_frequency)``.
    This is an explicit histogram-like approximation: it does not perform
    fractional-delay interpolation, pulse compression, matched filtering, or
    a range FFT.
    """
    _positive_integer(fft_size, "fft_size")
    _positive_rate(sampling_frequency, "sampling_frequency")
    cir = np.asarray(cir)
    delays_s = np.asarray(delays_s, dtype=float)
    if (
        cir.ndim != 1
        or delays_s.shape != (cir.shape[0],)
        or not np.isfinite(cir).all()
        or not np.isfinite(delays_s).all()
    ):
        raise ValueError("CIR values and delays must be finite and path-aligned")
    samples = np.zeros(fft_size, dtype=np.complex128)
    if cir.size == 0 or delays_s.size == 0:
        return samples
    valid = (delays_s >= 0) & (delays_s < fft_size / sampling_frequency)
    bin_indices = np.zeros(delays_s.shape, dtype=np.int64)
    bin_indices[valid] = np.floor(delays_s[valid] * sampling_frequency).astype(np.int64)
    valid &= bin_indices < fft_size
    if not np.any(valid):
        return samples
    np.add.at(samples, bin_indices[valid], cir[valid])
    return samples


def cir_to_samples(
    cir: np.ndarray,
    delays_s: np.ndarray,
    *,
    sampling_frequency: float,
    fft_size: int,
) -> np.ndarray:
    """Compatibility name for :func:`project_cir_to_delay_bins`.

    The returned axis is a delay-domain CIR-bin axis, not waveform fast-time
    samples awaiting a range FFT.
    """
    return project_cir_to_delay_bins(
        cir,
        delays_s,
        sampling_frequency=sampling_frequency,
        fft_size=fft_size,
    )


def cir_to_samples_batch(
    cir_matrix: np.ndarray,
    delays_s: np.ndarray,
    *,
    sampling_frequency: float,
    fft_size: int,
) -> np.ndarray:
    """Map a (N_paths, N_cpi) CIR matrix onto uniform sample grids.

    Computes bin indices and valid mask once, then scatters all time steps.
    Equivalent to calling :func:`cir_to_samples` per column, but ~2x faster
    because ``np.floor``, ``np.clip``, and validity masking are done once.

    Args:
        cir_matrix: Complex CIR values, shape ``(N_paths, N_cpi)``.
        delays_s: Delay per path in seconds, shape ``(N_paths,)``.
        sampling_frequency: Sampling rate in Hz.
        fft_size: Number of range bins.

    Returns:
        Slow-time matrix, shape ``(N_cpi, fft_size)``, complex128.
    """
    _positive_integer(fft_size, "fft_size")
    _positive_rate(sampling_frequency, "sampling_frequency")
    cir_matrix = np.asarray(cir_matrix)
    delays_s = np.asarray(delays_s, dtype=float)
    if (
        cir_matrix.ndim != 2
        or delays_s.shape != (cir_matrix.shape[0],)
        or not np.isfinite(cir_matrix).all()
        or not np.isfinite(delays_s).all()
    ):
        raise ValueError("CIR values and delays must be finite and path-aligned")
    n_cpi = cir_matrix.shape[1]
    result = np.zeros((n_cpi, fft_size), dtype=np.complex128)
    if cir_matrix.size == 0 or delays_s.size == 0:
        return result
    valid = (delays_s >= 0) & (delays_s < fft_size / sampling_frequency)
    bin_indices = np.zeros(delays_s.shape, dtype=np.int64)
    bin_indices[valid] = np.floor(delays_s[valid] * sampling_frequency).astype(np.int64)
    valid &= bin_indices < fft_size
    if not np.any(valid):
        return result
    valid_bins = bin_indices[valid]
    for t in range(n_cpi):
        np.add.at(result[t], valid_bins, cir_matrix[valid, t])
    return result


def cir_to_samples_per_antenna(
    cir_tensor: np.ndarray,
    delays_s: np.ndarray,
    *,
    sampling_frequency: float,
    fft_size: int,
) -> np.ndarray:
    """Project per-antenna CIR tensors onto uniform sample grids.

    Args:
        cir_tensor: Complex CIR values with shape ``(N_ant, N_paths, N_cpi)``.
        delays_s: Delay per path in seconds, shape ``(N_paths,)``.
        sampling_frequency: Sampling rate in Hz.
        fft_size: Number of range bins.

    Returns:
        Per-antenna slow-time matrices with shape
        ``(N_ant, N_cpi, fft_size)``.
    """
    cir_tensor = np.asarray(cir_tensor)
    if cir_tensor.ndim != 3:
        raise ValueError(f"cir_tensor must be 3-D, got shape {cir_tensor.shape}")

    num_ant, _, num_cpi = cir_tensor.shape
    result = np.zeros((num_ant, num_cpi, fft_size), dtype=np.complex128)
    if cir_tensor.size == 0 or delays_s.size == 0:
        return result

    for ant_idx in range(num_ant):
        result[ant_idx] = cir_to_samples_batch(
            cir_tensor[ant_idx],
            delays_s,
            sampling_frequency=sampling_frequency,
            fft_size=fft_size,
        )

    return result


@dataclass(frozen=True)
class RelativeComplexAwgnResult:
    """Samples and scale evidence for target-relative complex AWGN.

    The numeric domain is deliberately dimensionless.  This result does not
    claim a receiver temperature, bandwidth, noise figure, impedance, watts,
    or any other absolute link-budget scale.
    """

    samples: np.ndarray
    requested_snr_db: float
    reference_signal_power: float
    expected_noise_power: float
    realized_noise_power: float
    realized_snr_db: float | None
    reference_sample_count: int
    total_sample_count: int
    applied: bool
    disposition: str

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-safe scientific description of the injection."""

        return {
            "model": "target_relative_complex_awgn",
            "sample_domain": "relative_complex_voltage",
            "power_unit": "1",
            "scale_semantics": "relative_unitless",
            "reference_policy": "mean_power_over_nonzero_input_samples",
            "application_support": "all_input_samples",
            "requested_snr_db": self.requested_snr_db,
            "reference_signal_power": self.reference_signal_power,
            "expected_noise_power": self.expected_noise_power,
            "realized_noise_power": self.realized_noise_power,
            "realized_snr_db": self.realized_snr_db,
            "reference_sample_count": self.reference_sample_count,
            "total_sample_count": self.total_sample_count,
            "applied": self.applied,
            "disposition": self.disposition,
            "physical_thermal_noise_model": False,
        }


def _finite_mean_complex_power(values: np.ndarray, *, label: str) -> tuple[float, float]:
    """Return finite mean complex power and its natural logarithm.

    Scaling by the largest magnitude avoids overflowing the intermediate
    square for finite complex64/complex128 values.  A power that cannot be
    represented by the public float metadata contract is rejected instead of
    being returned as zero or infinity.
    """

    values = np.asarray(values)
    if values.size == 0:
        raise ValueError(f"{label} has no samples")
    magnitudes = np.abs(values)
    scale = float(np.max(magnitudes))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} is outside numeric range")
    normalized = np.asarray(magnitudes / scale, dtype=np.float64)
    mean_normalized_power = float(np.mean(normalized * normalized))
    if not np.isfinite(mean_normalized_power) or mean_normalized_power <= 0.0:
        raise ValueError(f"{label} is outside numeric range")
    log_power = 2.0 * float(np.log(scale)) + float(np.log(mean_normalized_power))
    max_log = float(np.log(np.finfo(np.float64).max))
    min_log = float(np.log(np.nextafter(0.0, 1.0, dtype=np.float64)))
    if not np.isfinite(log_power) or log_power > max_log or log_power < min_log:
        raise ValueError(f"{label} is outside numeric range")
    power = float(np.exp(log_power))
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError(f"{label} is outside numeric range")
    return power, log_power


def inject_target_relative_complex_awgn(
    samples: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator | None = None,
) -> RelativeComplexAwgnResult:
    """Add target-relative circular complex AWGN in a relative unitless domain.

    The reference signal power is the mean squared magnitude over nonzero
    input samples.  Noise with that relative power is then added to every
    input sample. A zero-only input has no target-relative power reference, so
    it is returned unchanged with ``applied=False``.

    This is intentionally *not* a physical thermal-noise or link-budget
    model. In particular, the input has no asserted watt scale and the
    function does not use temperature, bandwidth, receiver noise figure, or
    impedance.

    Args:
        samples: Complex floating-point samples in relative-voltage units.
        snr_db: Desired reference-signal to total-complex-noise power ratio.
        rng: NumPy random generator for reproducibility.

    Returns:
        Noisy samples plus expected and realized relative-scale evidence.
    """
    samples = np.asarray(samples)
    if samples.dtype.kind != "c":
        raise TypeError("target-relative complex AWGN requires a complex floating array")
    if not np.all(np.isfinite(samples.real)) or not np.all(np.isfinite(samples.imag)):
        raise ValueError("target-relative complex AWGN input samples must be finite")
    if isinstance(snr_db, (bool, np.bool_)):
        raise ValueError("target-relative complex AWGN snr_db must be finite")
    requested_snr_db = float(snr_db)
    if not np.isfinite(requested_snr_db):
        raise ValueError("target-relative complex AWGN snr_db must be finite")
    if rng is None:
        rng = np.random.default_rng()

    nonzero_mask = samples != 0
    if not np.any(nonzero_mask):
        return RelativeComplexAwgnResult(
            samples=samples.copy(),
            requested_snr_db=requested_snr_db,
            reference_signal_power=0.0,
            expected_noise_power=0.0,
            realized_noise_power=0.0,
            realized_snr_db=None,
            reference_sample_count=0,
            total_sample_count=int(samples.size),
            applied=False,
            disposition="no_nonzero_signal_reference",
        )

    signal_power, log_signal_power = _finite_mean_complex_power(
        samples[nonzero_mask],
        label="target-relative complex AWGN reference signal power",
    )
    # Work in the log domain so an extreme but finite dB request cannot raise
    # Python's OverflowError before it reaches the public numeric-range check.
    # ``np.errstate`` also turns exponential overflow/underflow into values the
    # explicit finite/positive validation below can reject deterministically.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_noise_power = log_signal_power - (np.float64(requested_snr_db) * (np.log(10.0) / 10.0))
        expected_noise_power = float(np.exp(log_noise_power))
    if not np.isfinite(expected_noise_power) or expected_noise_power <= 0.0:
        raise ValueError("target-relative complex AWGN requested SNR is outside numeric range")

    sigma = float(np.sqrt(expected_noise_power / 2.0))
    generated_noise = sigma * (
        rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        noise = generated_noise.astype(samples.dtype)
    if not np.all(np.isfinite(noise.real)) or not np.all(np.isfinite(noise.imag)):
        raise ValueError(
            "target-relative complex AWGN requested SNR is outside input dtype numeric range"
        )
    realized_noise_power, log_realized_noise_power = _finite_mean_complex_power(
        noise,
        label="target-relative complex AWGN realized noise power",
    )
    realized_snr_db = float((10.0 / np.log(10.0)) * (log_signal_power - log_realized_noise_power))
    with np.errstate(over="ignore", invalid="ignore"):
        noisy_samples = samples + noise
    if not np.all(np.isfinite(noisy_samples.real)) or not np.all(np.isfinite(noisy_samples.imag)):
        raise ValueError(
            "target-relative complex AWGN requested SNR is outside input dtype numeric range"
        )
    return RelativeComplexAwgnResult(
        samples=noisy_samples,
        requested_snr_db=requested_snr_db,
        reference_signal_power=signal_power,
        expected_noise_power=expected_noise_power,
        realized_noise_power=realized_noise_power,
        realized_snr_db=realized_snr_db,
        reference_sample_count=int(np.count_nonzero(nonzero_mask)),
        total_sample_count=int(samples.size),
        applied=True,
        disposition="applied",
    )


@dataclass
class RangeProfileResult:
    """Range-domain power profile and axis metadata for one CIR snapshot."""

    power_linear: np.ndarray
    range_axis_m: np.ndarray
    fft_spectrum: np.ndarray | None
    range_axis_label: str = "Path Length (m)"
    metadata: dict | None = None


def _range_axis(
    fft_size: int, sampling_frequency: float, range_mode: str = "bistatic"
) -> tuple[np.ndarray, str]:
    """
    Build a range/path-length axis and a human-readable label.
    """
    mode = normalize_range_mode(range_mode)
    _positive_integer(fft_size, "fft_size")
    _positive_rate(sampling_frequency, "sampling_frequency")
    sf = float(sampling_frequency)
    base = (np.arange(fft_size) / sf) * C0
    if mode == "monostatic":
        return base / 2.0, "Range (m)"
    return base, "Path Length (m)"


def compute_range_profile(
    cir: np.ndarray,
    delays_s: np.ndarray,
    *,
    sampling_frequency: float,
    fft_size: int = 256,
    window: str | None = "hann",
    range_mode: str = "bistatic",
) -> RangeProfileResult:
    """Compute a range/path-length profile from point-binned CIR delays.

    The default assumes bistatic/link-style path length (no /2). Set
    range_mode=\"monostatic\" to apply the round-trip (/2) convention.
    Only a delay-domain taper and magnitude square are applied; there is no
    waveform synthesis, matched filter, pulse compression, or range FFT.
    """
    samples = project_cir_to_delay_bins(
        cir, delays_s, sampling_frequency=sampling_frequency, fft_size=fft_size
    )

    # Debug/diagnostic: log bin occupancy and delay spread to explain single-peak issues.
    if logger.isEnabledFor(logging.DEBUG):
        bin_indices = np.floor(delays_s * sampling_frequency).astype(np.int64)
        valid = (bin_indices >= 0) & (bin_indices < fft_size)
        unique_bins = np.unique(bin_indices[valid]) if np.any(valid) else np.array([])
        nonzero_bins = int(np.count_nonzero(np.abs(samples) > 0))
        delay_min_ns = float(np.min(delays_s) * 1e9) if delays_s.size else float("nan")
        delay_max_ns = float(np.max(delays_s) * 1e9) if delays_s.size else float("nan")
        spread_bins = int(unique_bins.max() - unique_bins.min()) if unique_bins.size else 0
        logger.debug(
            "RangeProfile bins: nonzero=%d unique=%d spread_bins=%d "
            "delay_ns[min=%.3f max=%.3f] first_bins=%s",
            nonzero_bins,
            unique_bins.size,
            spread_bins,
            delay_min_ns,
            delay_max_ns,
            unique_bins[:8],
        )
    win = _window(window, len(samples))
    win_correction = _window_energy_correction(win)

    weighted_samples = samples * win * win_correction

    # The point-binned CIR is already on the delay grid used by this range
    # product. No waveform synthesis, matched filter, or range FFT is applied.
    power_linear = np.abs(weighted_samples) ** 2

    range_axis, axis_label = _range_axis(
        fft_size=fft_size, sampling_frequency=sampling_frequency, range_mode=range_mode
    )

    # Compute spectrum for optional analysis (not used for power)
    spectrum = np.fft.fft(weighted_samples, n=fft_size)

    return RangeProfileResult(
        power_linear=power_linear,
        range_axis_m=range_axis,
        fft_spectrum=spectrum,
        range_axis_label=axis_label,
        metadata={
            "delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR,
            "range_processing": RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
            "range_fft_applied": False,
            "delay_taper": str(window) if window is not None else "rect",
            "window_normalization": WINDOW_NORMALIZATION_ENERGY,
        },
    )


@dataclass
class RangeDopplerResult:
    """Two-dimensional range-Doppler product derived from slow-time samples."""

    power_linear: np.ndarray  # [slow, fast]
    range_axis_m: np.ndarray
    doppler_axis_hz: np.ndarray
    cube: np.ndarray | None  # complex RD cube before magnitude
    range_axis_label: str = "Path Length (m)"
    doppler_axis_label: str = "Doppler (Hz)"
    metadata: dict | None = None


@dataclass
class DopplerProfileResult:
    """1D Doppler power profile collapsed across a range gate."""

    doppler_axis_hz: np.ndarray  # (fft_d,)
    power_db: np.ndarray  # (fft_d,)
    range_bin_start: int
    range_bin_end: int
    range_m: float


def compute_collapsed_doppler_profile(
    rd_map: np.ndarray,
    range_axis_m: np.ndarray,
    doppler_axis_hz: np.ndarray,
    target_range_m: float | None = None,
    gate_half_width: int = 3,
) -> DopplerProfileResult:
    """Extract range-collapsed Doppler profile from an RD map.

    Sums RD map power across a range gate centered on *target_range_m*
    (or the peak-power range bin when ``None``), returning a 1D Doppler
    spectrum in dB.

    Args:
        rd_map: Range-Doppler power map, shape ``(fft_d, fft_r)``, linear.
        range_axis_m: Range axis in metres, length ``fft_r``.
        doppler_axis_hz: Doppler axis in Hz, length ``fft_d``.
        target_range_m: Centre of the range gate.  ``None`` → peak range bin.
        gate_half_width: Half-width of range gate in bins (default 3).

    Returns:
        DopplerProfileResult with the collapsed 1D Doppler spectrum.
    """
    rd_map = np.asarray(rd_map, dtype=np.float64)
    range_axis_m = np.asarray(range_axis_m, dtype=np.float64)
    doppler_axis_hz = np.asarray(doppler_axis_hz, dtype=np.float64)

    if (
        rd_map.ndim != 2
        or not all(rd_map.shape)
        or not np.isfinite(rd_map).all()
        or np.any(rd_map < 0)
        or range_axis_m.shape != (rd_map.shape[1],)
        or doppler_axis_hz.shape != (rd_map.shape[0],)
        or not np.isfinite(range_axis_m).all()
        or not np.isfinite(doppler_axis_hz).all()
    ):
        raise ValueError(
            "Power and matching axes must be nonempty, finite and nonnegative in power"
        )
    if not isinstance(gate_half_width, (int, np.integer)) or gate_half_width < 0:
        raise ValueError("gate_half_width must be a nonnegative integer")
    if target_range_m is not None and not np.isfinite(target_range_m):
        raise ValueError("target_range_m must be finite")

    fft_d, fft_r = rd_map.shape

    if target_range_m is not None:
        center_bin = int(np.argmin(np.abs(range_axis_m - target_range_m)))
    else:
        # Use range bin with highest total Doppler energy
        center_bin = int(np.argmax(np.sum(rd_map, axis=0)))

    lo = max(0, center_bin - gate_half_width)
    hi = min(fft_r, center_bin + gate_half_width + 1)

    # Sum across range gate → 1D Doppler spectrum
    collapsed_power = np.sum(rd_map[:, lo:hi], axis=1)
    power_db = 10.0 * np.log10(np.maximum(collapsed_power, 1e-30))

    range_m = float(range_axis_m[center_bin]) if center_bin < len(range_axis_m) else 0.0

    return DopplerProfileResult(
        doppler_axis_hz=doppler_axis_hz,
        power_db=power_db,
        range_bin_start=lo,
        range_bin_end=hi - 1,
        range_m=range_m,
    )


def apply_pre_cfar_doppler_guards(
    rd_power: np.ndarray,
    *,
    edge_guard_bins: int = 0,
    dc_guard_bins: int = 0,
) -> np.ndarray:
    """Return a guarded copy of a range-Doppler power map for CFAR input.

    The Doppler axis is assumed to be axis 0 after ``fftshift``:

    - ``edge_guard_bins`` zeros bins at both Doppler extremes
    - ``dc_guard_bins`` zeros a band centered on the zero-Doppler bin

    The input array is returned unchanged when no guards are requested.
    """
    rd_power = np.asarray(rd_power)
    if (
        rd_power.ndim != 2
        or np.iscomplexobj(rd_power)
        or not np.isfinite(rd_power).all()
        or np.any(rd_power < 0)
    ):
        raise ValueError("CFAR power must be a finite nonnegative 2-D array")
    for name, count in (("edge_guard_bins", edge_guard_bins), ("dc_guard_bins", dc_guard_bins)):
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer")
    if edge_guard_bins == 0 and dc_guard_bins == 0:
        return rd_power

    fft_size_doppler = int(rd_power.shape[0])
    guarded = rd_power.copy()

    if edge_guard_bins > 0:
        if 2 * edge_guard_bins >= fft_size_doppler:
            guarded[...] = 0.0
            return guarded
        guarded[:edge_guard_bins, :] = 0.0
        guarded[-edge_guard_bins:, :] = 0.0

    if dc_guard_bins > 0:
        dc_guard_bins = min(dc_guard_bins, fft_size_doppler)
        dc_bin = fft_size_doppler // 2
        dc_lo = max(0, dc_bin - dc_guard_bins // 2)
        dc_hi = min(fft_size_doppler, dc_lo + dc_guard_bins)
        if dc_hi - dc_lo < dc_guard_bins:
            dc_lo = max(0, dc_hi - dc_guard_bins)
        guarded[dc_lo:dc_hi, :] = 0.0

    return guarded


@dataclass
class MicroDopplerResult:
    """Micro-Doppler spectrogram from STFT on selected range bin."""

    spectrogram: np.ndarray  # [time_frames, freq_bins] power
    time_axis_s: np.ndarray
    doppler_axis_hz: np.ndarray
    range_bin: int
    range_m: float
    collapsed: bool = False
    range_bin_start: int | None = None
    range_bin_end: int | None = None
    range_tracking: bool = False
    tracked_range_bins: np.ndarray | None = None
    tracked_range_m: np.ndarray | None = None
    truth_assisted: bool = False
    selection_source: str = "fixed_or_observation"
    target_id: str | None = None


def _stft_time_axis_s(
    *,
    n_segments: int,
    stft_window_frames: int,
    stft_hop_frames: int,
    frame_duration_s: float,
) -> np.ndarray:
    return (
        np.arange(n_segments, dtype=np.float64) * stft_hop_frames + stft_window_frames / 2
    ) * frame_duration_s


def _stft_doppler_axis_hz(
    *,
    stft_window_frames: int,
    prf_hz: float,
) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(stft_window_frames, d=1.0 / prf_hz))


def _compute_stft_power_spectrogram(
    slow_time_signal: np.ndarray,
    *,
    stft_window_frames: int,
    stft_hop_frames: int,
    window: str | None,
    remove_dc: bool = True,
) -> np.ndarray:
    """Compute a windowed STFT power spectrogram from 1-D or 2-D slow-time data."""
    _positive_integer(stft_window_frames, "stft_window_frames")
    _positive_integer(stft_hop_frames, "stft_hop_frames")
    signal = np.asarray(slow_time_signal, dtype=np.complex128)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if signal.ndim != 2:
        raise ValueError("slow_time_signal must be 1-D or 2-D")

    if not np.isfinite(signal).all():
        raise ValueError("STFT samples must be finite; missing samples cannot be zero-filled")
    n_frames = int(signal.shape[0])
    if n_frames < stft_window_frames:
        return np.zeros((0, stft_window_frames), dtype=np.float64)

    win = _window(window, stft_window_frames)
    win_correction = _window_energy_correction(win)

    segments = sliding_window_view(signal, stft_window_frames, axis=0)[::stft_hop_frames]
    if segments.size == 0:
        return np.zeros((0, stft_window_frames), dtype=np.float64)

    # sliding_window_view(..., axis=0) returns [segments, channels, window]
    # This operator projects out the DC bin after applying the taper.
    # It differs from rejecting a constant component before tapering (or a
    # taper-weighted constant fit); those are distinct clutter filters.
    windowed = segments * (win[np.newaxis, np.newaxis, :] * win_correction)
    if remove_dc:
        windowed = windowed - np.mean(windowed, axis=-1, keepdims=True)
    fft_result = np.fft.fftshift(
        np.fft.fft(windowed, n=stft_window_frames, axis=-1),
        axes=-1,
    ) / np.sqrt(stft_window_frames)
    power = np.abs(fft_result) ** 2
    spectrogram = np.sum(power, axis=1)
    # Preserve physical power, including weak returns and exact zeros. A log
    # display protects against log(0) at rendering time; flooring here would
    # invent background energy and erase sub-floor motion signatures.
    return spectrogram


def has_fresh_sensing_data(sensing_block: Mapping[str, Any] | None) -> bool:
    """Return whether a schema-v4 payload declares a processed result.

    Freshness is temporal result state, not the presence, shape, or dtype of a
    convenience array.  A valid result with an empty detection list is still
    fresh, while accumulating and incomplete acquisitions have no result.
    Pre-v4 sensing payloads fail with regeneration guidance; the generic frame
    path remains unaffected when the optional sensing mapping is absent.
    """

    if sensing_block is None or not sensing_block:
        return False
    from shared.extensions.sensing import assert_supported_sensing_schema

    assert_supported_sensing_schema(sensing_block)
    temporal = sensing_block.get("temporal")
    if not isinstance(temporal, Mapping):
        raise ValueError("Schema-v4 sensing freshness requires temporal metadata")
    result = temporal.get("result")
    if result is None:
        return False
    if not isinstance(result, Mapping):
        raise ValueError("Schema-v4 sensing temporal.result must be a mapping when present")
    return True


def compute_range_doppler(
    cir_sequence: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    *,
    fast_time_sequence: Iterable[np.ndarray] | None = None,
    sampling_frequency: float,
    prf_hz: float,
    fft_size_range: int = 256,
    fft_size_doppler: int = 64,
    window_range: str | None = "hann",
    window_doppler: str | None = "hann",
    range_mode: str = "bistatic",
) -> RangeDopplerResult | None:
    """Compute delay-bin–Doppler power from CIR-domain inputs.

    Sparse CIR inputs use the explicit point-bin-floor delay projection.  The
    only FFT is the unitary FFT over slow time; no range FFT is applied.
    """
    _positive_rate(sampling_frequency, "sampling_frequency")
    _positive_rate(prf_hz, "prf_hz")
    _positive_integer(fft_size_range, "fft_size_range")
    _positive_integer(fft_size_doppler, "fft_size_doppler")
    fast_time_arr: np.ndarray | None = None
    if fast_time_sequence is not None:
        fast_list = [np.asarray(samples) for samples in fast_time_sequence]
        if not fast_list:
            return None
        fast_time_arr = np.stack(fast_list, axis=0)
    elif cir_sequence is not None:
        cir_list = list(cir_sequence)
        if len(cir_list) == 0:
            return None
        fast_time = []
        for cir, delays_s in cir_list:
            fast_samples = cir_to_samples(
                cir, delays_s, sampling_frequency=sampling_frequency, fft_size=fft_size_range
            )
            fast_time.append(fast_samples)
        fast_time_arr = np.stack(fast_time, axis=0)  # [slow, fast]
    else:
        return None

    fast_time_arr = _finite_sample_matrix(fast_time_arr)
    if fast_time_arr.shape[1] != fft_size_range:
        raise ValueError("Sample range dimension must equal fft_size_range")
    if fast_time_arr.shape[0] > fft_size_doppler:
        raise ValueError("Doppler FFT must not silently truncate observed samples")

    wr = _window(window_range, fft_size_range)
    wd = _window(window_doppler, fast_time_arr.shape[0])

    # Compute window energy corrections to preserve power
    wr_correction = _window_energy_correction(wr)
    wd_correction = _window_energy_correction(wd)

    windowed = fast_time_arr * wr[np.newaxis, :] * wr_correction
    windowed = windowed * wd[:, np.newaxis] * wd_correction

    # FFT on slow-time (Doppler) axis with normalization
    # Normalize by sqrt(N) for power to scale correctly
    rd_cube = np.fft.fftshift(np.fft.fft(windowed, n=fft_size_doppler, axis=0), axes=0) / np.sqrt(
        fft_size_doppler
    )

    rd_power = np.abs(rd_cube) ** 2

    range_axis, range_label = _range_axis(
        fft_size=fft_size_range,
        sampling_frequency=sampling_frequency,
        range_mode=range_mode,
    )
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_size_doppler, d=1.0 / prf_hz))

    return RangeDopplerResult(
        power_linear=rd_power,
        range_axis_m=range_axis,
        doppler_axis_hz=doppler_axis,
        cube=rd_cube,
        range_axis_label=range_label,
        doppler_axis_label="Doppler (Hz)",
        metadata={
            "delay_projection": DELAY_PROJECTION_POINT_BIN_FLOOR,
            "range_processing": RANGE_PROCESSING_DELAY_DOMAIN_CIR_BINS,
            "range_fft_applied": False,
            "delay_taper": str(window_range) if window_range is not None else "rect",
            "doppler_taper": str(window_doppler) if window_doppler is not None else "rect",
            "fft_axis": "slow_time",
            "fft_normalization": DOPPLER_FFT_NORMALIZATION_UNITARY,
            "window_normalization": WINDOW_NORMALIZATION_ENERGY,
        },
    )


def _required_tracking_sample_times(
    sample_times_s: Iterable[float],
    *,
    sample_count: int,
) -> tuple[np.ndarray, float]:
    """Return exact, uniformly spaced physical timestamps for an FFT product."""

    try:
        times = np.asarray(list(sample_times_s), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Truth-assisted range tracking requires finite physical sample timestamps"
        ) from exc
    if times.ndim != 1 or times.size != sample_count or not np.all(np.isfinite(times)):
        raise ValueError(
            "Truth-assisted range tracking requires exactly one finite physical timestamp "
            f"per signal sample ({times.size} != {sample_count})"
        )
    if sample_count < 2:
        raise ValueError("Truth-assisted range tracking requires at least two physical samples")
    intervals = np.diff(times)
    if not np.all(intervals > 0.0):
        raise ValueError("Truth-assisted range-tracking timestamps must be strictly increasing")
    interval_s = float(intervals[0])
    tolerance_s = max(1e-12, abs(interval_s) * 1e-9)
    if not np.allclose(intervals, interval_s, rtol=1e-6, atol=tolerance_s):
        raise ValueError(
            "Truth-assisted FFT tracking requires uniform physical timestamps; "
            "resample explicitly before tracking"
        )
    return times, interval_s


def _required_exact_range_axis(
    range_axis_m: Iterable[float],
    *,
    range_bin_count: int,
    label: str,
) -> np.ndarray:
    """Return one exact, producer-authored physical range coordinate vector."""

    try:
        axis = np.asarray(list(range_axis_m), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} requires an exact finite range_axis_m") from exc
    if axis.ndim != 1 or axis.size != range_bin_count or not np.all(np.isfinite(axis)):
        raise ValueError(
            f"{label} requires range_axis_m to contain exactly one "
            f"finite coordinate per signal bin ({axis.size} != {range_bin_count})"
        )
    if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{label} range_axis_m must be strictly increasing")
    return axis


def _target_range_for_exact_sample(
    entry: Mapping[str, Any],
    *,
    target_id: str,
    sample_time_s: float,
    sample_index: int,
) -> float:
    """Join one truth snapshot to one physical sample by time and target ID."""

    raw_time = entry.get("reference_time_s")
    try:
        truth_time_s = float(raw_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Truth sample {sample_index} lacks a finite reference_time_s") from exc
    if not np.isfinite(truth_time_s):
        raise ValueError(f"Truth sample {sample_index} lacks a finite reference_time_s")
    if truth_time_s != float(sample_time_s):
        raise ValueError(
            f"Truth sample {sample_index} is not exactly time-aligned: "
            f"{truth_time_s!r} != {float(sample_time_s)!r}"
        )

    target_ids = entry.get("target_ids")
    target_ranges = entry.get("target_ranges")
    if (
        not isinstance(target_ids, (list, tuple))
        or not isinstance(target_ranges, (list, tuple))
        or len(target_ids) != len(target_ranges)
    ):
        raise ValueError(
            f"Truth sample {sample_index} requires target_ids aligned one-to-one with "
            "target_ranges"
        )
    if any(not isinstance(value, str) or not value for value in target_ids):
        raise ValueError(f"Truth sample {sample_index} contains an invalid target_id")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(f"Truth sample {sample_index} contains duplicate target_ids")
    matches = [index for index, value in enumerate(target_ids) if value == target_id]
    if len(matches) != 1:
        raise ValueError(
            f"Truth sample {sample_index} must contain target_id {target_id!r} exactly once"
        )
    try:
        target_range_m = float(target_ranges[matches[0]])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Truth sample {sample_index} has no finite range for target_id {target_id!r}"
        ) from exc
    if not np.isfinite(target_range_m):
        raise ValueError(
            f"Truth sample {sample_index} has no finite range for target_id {target_id!r}"
        )
    return target_range_m


def _resolve_tracked_range_bins(
    gt_entries: Iterable[Mapping[str, Any]],
    *,
    target_id: str,
    sample_times_s: Iterable[float],
    range_axis_m: Iterable[float],
    n_time_steps: int,
    n_range_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Resolve bins only through an exact identity-and-time truth join."""

    if not isinstance(target_id, str) or not target_id:
        raise ValueError("Truth-assisted range tracking requires an explicit target_id")
    if n_time_steps <= 0 or n_range_bins <= 0:
        raise ValueError("Truth-assisted range tracking requires a non-empty signal matrix")
    times, sample_interval_s = _required_tracking_sample_times(
        sample_times_s,
        sample_count=n_time_steps,
    )
    range_axis = _required_exact_range_axis(
        range_axis_m,
        range_bin_count=n_range_bins,
        label="Truth-assisted range tracking",
    )
    gt_list = list(gt_entries)
    if len(gt_list) != n_time_steps:
        raise ValueError(
            "Truth-assisted range tracking requires exactly one truth snapshot per "
            f"physical sample ({len(gt_list)} != {n_time_steps}); truth expansion and "
            "interpolation are forbidden"
        )

    timeline_bins = np.empty(n_time_steps, dtype=np.int64)
    for sample_index, (entry, sample_time_s) in enumerate(zip(gt_list, times, strict=True)):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Truth sample {sample_index} must be a mapping")
        target_range_m = _target_range_for_exact_sample(
            entry,
            target_id=target_id,
            sample_time_s=float(sample_time_s),
            sample_index=sample_index,
        )
        if target_range_m < float(range_axis[0]) or target_range_m > float(range_axis[-1]):
            raise ValueError(
                f"Truth sample {sample_index} range {target_range_m!r} m for target_id "
                f"{target_id!r} lies outside the supplied range_axis_m"
            )
        timeline_bins[sample_index] = int(np.argmin(np.abs(range_axis - target_range_m)))

    return timeline_bins, range_axis, times, sample_interval_s


def compute_range_tracked_range_doppler(
    gt_entries: Iterable[Mapping[str, Any]],
    *,
    target_id: str,
    fast_time_sequence: Iterable[np.ndarray],
    sample_times_s: Iterable[float],
    range_axis_m: Iterable[float],
    fft_size_doppler: int = 64,
    window_range: str | None = "hann",
    window_doppler: str | None = "hann",
    gate_half_width_bins: int = 4,
    mti_enabled: bool = True,
) -> RangeDopplerResult | None:
    """Compute a truth-assisted oracle range-Doppler diagnostic.

    This function uses ground-truth target range to move the extraction gate.
    Its output is evaluation evidence, not a blind estimator product.

    Args:
        gt_entries: One identity-keyed truth snapshot per physical sample.
        target_id: Stable target identity selected for the oracle diagnostic.
        fast_time_sequence: Per-pulse fast-time IQ samples (one array per pulse).
        sample_times_s: Exact physical timestamp aligned with every signal row.
        range_axis_m: Exact producer-authored physical range coordinate vector.
        fft_size_doppler: Doppler FFT size (last N pulses are used).
        window_range: Window function for the gated range axis.
        window_doppler: Window function for the Doppler (slow-time) axis.
        gate_half_width_bins: Half-width of the range gate centered on the
            tracked target bin.
        mti_enabled: If True, remove the taper-weighted slow-time constant
            component unconditionally before Doppler FFT. A constant CPI is
            exactly zero; genuine near-zero Doppler can also be attenuated.

    Returns:
        RangeDopplerResult with gated range-Doppler power map, or None if input
        is empty or malformed.
    """
    _positive_integer(fft_size_doppler, "fft_size_doppler")
    fast_list = [np.asarray(samples) for samples in fast_time_sequence]
    if not fast_list:
        return None

    fast_time_arr = np.stack(fast_list, axis=0)
    if fast_time_arr.ndim != 2:
        return None

    fast_time_arr = _finite_sample_matrix(fast_time_arr)

    n_time_steps, n_range_bins = fast_time_arr.shape
    if n_time_steps == 0 or n_range_bins == 0:
        return None

    gate_half_width_bins = max(0, int(gate_half_width_bins))
    gate_width_bins = 2 * gate_half_width_bins + 1
    fft_size_doppler = max(1, min(int(fft_size_doppler), n_time_steps))

    tracked_bins, physical_range_axis, times, sample_interval_s = _resolve_tracked_range_bins(
        gt_entries,
        target_id=target_id,
        sample_times_s=sample_times_s,
        range_axis_m=range_axis_m,
        n_time_steps=n_time_steps,
        n_range_bins=n_range_bins,
    )

    gated = np.zeros((n_time_steps, gate_width_bins), dtype=np.complex128)
    for time_idx, center_bin in enumerate(tracked_bins):
        src_lo = max(0, int(center_bin) - gate_half_width_bins)
        src_hi = min(n_range_bins, int(center_bin) + gate_half_width_bins + 1)
        dst_lo = gate_half_width_bins - (int(center_bin) - src_lo)
        dst_hi = dst_lo + (src_hi - src_lo)
        gated[time_idx, dst_lo:dst_hi] = fast_time_arr[time_idx, src_lo:src_hi]

    tracked_tail = gated[-fft_size_doppler:]
    range_win = _window(window_range, gate_width_bins)
    doppler_win = _window(window_doppler, fft_size_doppler)
    range_win_correction = _window_energy_correction(range_win)
    doppler_win_correction = _window_energy_correction(doppler_win)

    rd_input = tracked_tail * range_win[np.newaxis, :] * range_win_correction
    if mti_enabled:
        rd_input = remove_window_weighted_mean(rd_input, doppler_win, axis=0)
    rd_input = rd_input * doppler_win[:, np.newaxis] * doppler_win_correction

    rd_cube = np.fft.fftshift(
        np.fft.fft(rd_input, n=fft_size_doppler, axis=0),
        axes=0,
    ) / np.sqrt(fft_size_doppler)
    rd_power = np.abs(rd_cube) ** 2

    if physical_range_axis.size > 1:
        range_steps_m = np.diff(physical_range_axis)
        range_resolution_m = float(range_steps_m[0])
        if not np.allclose(
            range_steps_m,
            range_resolution_m,
            rtol=1e-9,
            atol=max(1e-12, abs(range_resolution_m) * 1e-12),
        ):
            raise ValueError(
                "Truth-tracked range-Doppler requires a uniform supplied range_axis_m "
                "for one exact relative gate axis"
            )
    else:
        range_resolution_m = 0.0
    range_axis = (
        np.arange(gate_width_bins, dtype=np.float64) - float(gate_half_width_bins)
    ) * range_resolution_m
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(fft_size_doppler, d=sample_interval_s))

    return RangeDopplerResult(
        power_linear=rd_power,
        range_axis_m=range_axis,
        doppler_axis_hz=doppler_axis,
        cube=rd_cube,
        range_axis_label="Range Offset From Tracked Target (m)",
        doppler_axis_label="Doppler (Hz)",
        metadata={
            "truth_assisted": True,
            "slow_time_mean_removal_policy": "window_weighted_mean" if mti_enabled else "disabled",
            "selection_source": "ground_truth_target_range",
            "product_role": "evaluation_oracle",
            "target_id": target_id,
            "sample_time_start_s": float(times[-fft_size_doppler]),
            "sample_time_end_s": float(times[-1]),
            "sample_interval_s": sample_interval_s,
        },
    )


def compute_micro_doppler(
    range_bin: int,
    *,
    fast_time_sequence: Iterable[np.ndarray],
    prf_hz: float,
    frame_duration_s: float,
    stft_window_frames: int = 32,
    stft_hop_frames: int = 8,
    window: str | None = "hann",
    sampling_frequency: float | None = None,
    range_mode: str = "bistatic",
    mti_enabled: bool = True,
    range_axis_m: Iterable[float] | None = None,
) -> MicroDopplerResult | None:
    """
    Compute micro-Doppler spectrogram for a selected range bin via STFT.

    Parameters:
    -----------
    range_bin : int
        Range bin index to extract slow-time signal from
    fast_time_sequence : Iterable[np.ndarray]
        Sequence of fast-time CIR samples (one per frame)
    prf_hz : float
        Pulse repetition frequency
    frame_duration_s : float
        Duration of each frame in seconds
    stft_window_frames : int
        STFT window size in frames (default: 32)
    stft_hop_frames : int
        STFT hop size in frames (default: 8, 75% overlap)
    window : str
        Window function name (default: "hann")
    sampling_frequency : float, optional
        Sampling frequency in Hz. When provided, ``range_m`` is computed
        from the range axis instead of using the bin index as a placeholder.
    range_mode : str
        ``"bistatic"`` (path length) or ``"monostatic"`` (range = path/2).

    Returns:
    --------
    MicroDopplerResult or None if insufficient data
    """
    # Convert to array [frames, range_bins]
    _positive_rate(prf_hz, "prf_hz")
    _positive_rate(frame_duration_s, "frame_duration_s")
    if not np.isclose(prf_hz * frame_duration_s, 1.0, rtol=1e-7, atol=1e-12):
        raise ValueError("prf_hz and frame_duration_s must describe the same sample clock")
    _positive_integer(stft_window_frames, "stft_window_frames")
    _positive_integer(stft_hop_frames, "stft_hop_frames")
    fast_list = [np.asarray(samples) for samples in fast_time_sequence]
    if not fast_list:
        return None

    fast_time_arr = np.stack(fast_list, axis=0)
    fast_time_arr = _finite_sample_matrix(fast_time_arr)
    n_frames, n_range_bins = fast_time_arr.shape

    if range_bin < 0 or range_bin >= n_range_bins:
        return None
    if n_frames < stft_window_frames:
        return None

    # Extract slow-time signal at selected range bin
    slow_time_signal = fast_time_arr[:, range_bin]

    # MTI: subtract temporal mean to remove static clutter (zero-Doppler).
    # The RD map pipeline applies this before its Doppler FFT; the micro-Doppler
    # STFT must do the same to avoid a strong DC artifact in each segment.
    if mti_enabled:
        centered = slow_time_signal - slow_time_signal[0]
        slow_time_signal = centered - np.mean(centered)
    spectrogram = _compute_stft_power_spectrogram(
        slow_time_signal,
        stft_window_frames=stft_window_frames,
        stft_hop_frames=stft_hop_frames,
        window=window,
        remove_dc=mti_enabled,
    )
    time_axis_s = _stft_time_axis_s(
        n_segments=spectrogram.shape[0],
        stft_window_frames=stft_window_frames,
        stft_hop_frames=stft_hop_frames,
        frame_duration_s=frame_duration_s,
    )
    doppler_axis_hz = _stft_doppler_axis_hz(
        stft_window_frames=stft_window_frames,
        prf_hz=prf_hz,
    )

    # Native persisted routes supply the producer-authored range coordinates.
    # The config-derived branch remains an explicit low-level in-memory API.
    if range_axis_m is not None:
        physical_range_axis = _required_exact_range_axis(
            range_axis_m,
            range_bin_count=n_range_bins,
            label="Micro-Doppler",
        )
        range_m = float(physical_range_axis[range_bin])
    elif sampling_frequency is not None:
        derived_range_axis, _ = _range_axis(n_range_bins, sampling_frequency, range_mode)
        range_m = float(derived_range_axis[range_bin])
    else:
        range_m = float(range_bin)

    return MicroDopplerResult(
        spectrogram=spectrogram,
        time_axis_s=time_axis_s,
        doppler_axis_hz=doppler_axis_hz,
        range_bin=range_bin,
        range_m=range_m,
    )


def compute_range_tracked_micro_doppler(
    gt_entries: Iterable[Mapping[str, Any]],
    *,
    target_id: str,
    fast_time_sequence: Iterable[np.ndarray],
    sample_times_s: Iterable[float],
    range_axis_m: Iterable[float],
    stft_window_frames: int = 32,
    stft_hop_frames: int = 8,
    window: str | None = "hann",
    mti_enabled: bool = True,
) -> MicroDopplerResult | None:
    """Compute a truth-assisted oracle micro-Doppler diagnostic.

    Unlike :func:`compute_micro_doppler` (single fixed range bin), this variant
    follows the target range over time using ``gt_entries`` and extracts one
    slow-time sample per frame from the tracked bin. This mitigates range
    migration smearing for moving targets. The output is evaluation evidence,
    not a blind range-tracking estimate.
    """
    _positive_integer(stft_window_frames, "stft_window_frames")
    _positive_integer(stft_hop_frames, "stft_hop_frames")
    fast_list = [np.asarray(samples) for samples in fast_time_sequence]
    if not fast_list:
        return None

    fast_time_arr = np.stack(fast_list, axis=0)
    fast_time_arr = _finite_sample_matrix(fast_time_arr)
    n_frames, n_range_bins = fast_time_arr.shape
    if n_frames < stft_window_frames:
        return None

    tracked_bins, range_axis, times, sample_interval_s = _resolve_tracked_range_bins(
        gt_entries,
        target_id=target_id,
        sample_times_s=sample_times_s,
        range_axis_m=range_axis_m,
        n_time_steps=n_frames,
        n_range_bins=n_range_bins,
    )

    slow_time_signal = fast_time_arr[np.arange(n_frames), tracked_bins]
    if mti_enabled:
        centered = slow_time_signal - slow_time_signal[0]
        slow_time_signal = centered - np.mean(centered)
    spectrogram = _compute_stft_power_spectrogram(
        slow_time_signal,
        stft_window_frames=stft_window_frames,
        stft_hop_frames=stft_hop_frames,
        window=window,
        remove_dc=mti_enabled,
    )
    time_axis_s = float(times[0]) + _stft_time_axis_s(
        n_segments=spectrogram.shape[0],
        stft_window_frames=stft_window_frames,
        stft_hop_frames=stft_hop_frames,
        frame_duration_s=sample_interval_s,
    )
    doppler_axis_hz = _stft_doppler_axis_hz(
        stft_window_frames=stft_window_frames,
        prf_hz=1.0 / sample_interval_s,
    )

    tracked_ranges = range_axis[tracked_bins]
    return MicroDopplerResult(
        spectrogram=spectrogram,
        time_axis_s=time_axis_s,
        doppler_axis_hz=doppler_axis_hz,
        range_bin=int(np.rint(np.mean(tracked_bins))),
        range_m=float(np.mean(tracked_ranges)),
        range_tracking=True,
        tracked_range_bins=tracked_bins,
        tracked_range_m=tracked_ranges,
        truth_assisted=True,
        selection_source="ground_truth_target_range",
        target_id=target_id,
    )


def compute_range_collapsed_micro_doppler(
    *,
    fast_time_sequence: Iterable[np.ndarray],
    prf_hz: float,
    frame_duration_s: float,
    stft_window_frames: int = 32,
    stft_hop_frames: int = 8,
    window: str | None = "hann",
    range_bins: Iterable[int] | None = None,
    sampling_frequency: float | None = None,
    range_mode: str = "bistatic",
    mti_enabled: bool = True,
    range_axis_m: Iterable[float] | None = None,
) -> MicroDopplerResult | None:
    """
    Compute a range-collapsed micro-Doppler spectrogram.

    The collapse is performed non-coherently: STFT power is computed for each
    selected range bin and then summed across range. This avoids phase
    cancellation between bins with different delays.
    """
    _positive_rate(prf_hz, "prf_hz")
    _positive_rate(frame_duration_s, "frame_duration_s")
    if not np.isclose(prf_hz * frame_duration_s, 1.0, rtol=1e-7, atol=1e-12):
        raise ValueError("prf_hz and frame_duration_s must describe the same sample clock")
    _positive_integer(stft_window_frames, "stft_window_frames")
    _positive_integer(stft_hop_frames, "stft_hop_frames")
    fast_list = [np.asarray(samples) for samples in fast_time_sequence]
    if not fast_list:
        return None

    fast_time_arr = np.stack(fast_list, axis=0)
    fast_time_arr = _finite_sample_matrix(fast_time_arr)
    n_frames, n_range_bins = fast_time_arr.shape
    if n_frames < stft_window_frames:
        return None

    if range_bins is None:
        selected_bins = np.arange(n_range_bins, dtype=np.int64)
    else:
        supplied_bins = np.asarray(list(range_bins))
        if supplied_bins.size and (supplied_bins.dtype.kind not in "iu" or supplied_bins.ndim != 1):
            raise ValueError("range_bins must contain integer indices")
        selected_bins = np.unique(supplied_bins.astype(np.int64))
        if selected_bins.size == 0:
            return None
        selected_bins = selected_bins[(selected_bins >= 0) & (selected_bins < n_range_bins)]
        if selected_bins.size == 0:
            return None

    selected = fast_time_arr[:, selected_bins]

    if mti_enabled:
        centered = selected - selected[:1]
        selected = centered - np.mean(centered, axis=0, keepdims=True)
    spectrogram = _compute_stft_power_spectrogram(
        selected,
        stft_window_frames=stft_window_frames,
        stft_hop_frames=stft_hop_frames,
        window=window,
        remove_dc=mti_enabled,
    )
    time_axis_s = _stft_time_axis_s(
        n_segments=spectrogram.shape[0],
        stft_window_frames=stft_window_frames,
        stft_hop_frames=stft_hop_frames,
        frame_duration_s=frame_duration_s,
    )
    doppler_axis_hz = _stft_doppler_axis_hz(
        stft_window_frames=stft_window_frames,
        prf_hz=prf_hz,
    )

    if range_axis_m is not None:
        physical_range_axis = _required_exact_range_axis(
            range_axis_m,
            range_bin_count=n_range_bins,
            label="Range-collapsed micro-Doppler",
        )
        mean_range = float(np.mean(physical_range_axis[selected_bins]))
    elif sampling_frequency is not None:
        derived_range_axis, _ = _range_axis(n_range_bins, sampling_frequency, range_mode)
        mean_range = float(np.mean(derived_range_axis[selected_bins]))
    else:
        mean_range = float(np.mean(selected_bins))

    return MicroDopplerResult(
        spectrogram=spectrogram,
        time_axis_s=time_axis_s,
        doppler_axis_hz=doppler_axis_hz,
        range_bin=-1,
        range_m=mean_range,
        collapsed=True,
        range_bin_start=int(selected_bins.min()),
        range_bin_end=int(selected_bins.max()),
    )


def compute_analytical_micro_doppler(
    analytical_paths_sequence: list[dict],
    *,
    prf_hz: float = 1000.0,
    frame_duration_s: float = 1e-3,
    n_doppler_bins: int = 128,
    sampling_frequency: float | None = None,
    fft_size_range: int = 128,
    range_mode: str = "bistatic",
) -> MicroDopplerResult | None:
    """Build an oracle micro-Doppler spectrogram from per-path analytical Doppler.

    Unlike FFT-based micro-Doppler (which needs coherent slow-time phase),
    this uses Sionna's per-path Doppler values directly.  Each frame
    contributes a row to the spectrogram by summing path powers in circular
    nearest Doppler bins (half-bin ties round upward). This is noncoherent
    power deposition, not a sampled FFT. Empty frames retain exact zero power;
    when all frames are empty, range_m is NaN because no range is available.

    Args:
        analytical_paths_sequence: List of dicts (one per frame) with keys
            ``amplitudes`` (complex), ``delays_s`` (float), ``doppler_hz``
            (float).
        prf_hz: Pulse repetition frequency in Hz (defines Doppler axis extent).
        frame_duration_s: Time between frames in seconds.
        n_doppler_bins: Number of Doppler bins in the output spectrogram.
        sampling_frequency: Bandwidth in Hz for range axis computation.
        fft_size_range: FFT size for range binning.
        range_mode: ``"monostatic"`` or ``"bistatic"``.

    Returns:
        MicroDopplerResult with spectrogram shape ``[n_frames, n_doppler_bins]``,
        or ``None`` if input is empty.
    """
    if not analytical_paths_sequence:
        return None

    _positive_rate(prf_hz, "prf_hz")
    _positive_rate(frame_duration_s, "frame_duration_s")
    _positive_integer(n_doppler_bins, "n_doppler_bins")
    _positive_integer(fft_size_range, "fft_size_range")
    mode = normalize_range_mode(range_mode)
    n_frames = len(analytical_paths_sequence)
    doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(n_doppler_bins, d=1.0 / prf_hz))
    spectrogram = np.zeros((n_frames, n_doppler_bins), dtype=np.float64)
    delay_blocks = []
    for t, paths in enumerate(analytical_paths_sequence):
        amps = np.asarray(paths["amplitudes"], dtype=np.complex128)
        dopplers = np.asarray(paths["doppler_hz"], dtype=np.float64)
        delays = np.asarray(paths["delays_s"], dtype=np.float64)
        if (
            amps.ndim != 1
            or amps.shape != dopplers.shape
            or amps.shape != delays.shape
            or not np.isfinite(amps).all()
            or not np.isfinite(dopplers).all()
            or not np.isfinite(delays).all()
            or np.any(delays < 0)
        ):
            raise ValueError(
                "Analytical paths require aligned finite amplitudes, Dopplers and nonnegative delays"
            )
        if delays.size:
            delay_blocks.append(delays)
        # Circular nearest-bin quantization, including odd N and Nyquist wrap.
        # Exact half-bin ties choose the higher unshifted frequency index.
        bins = np.floor((dopplers % prf_hz) * n_doppler_bins / prf_hz + 0.5).astype(np.int64)
        shifted_bins = (bins + n_doppler_bins // 2) % n_doppler_bins
        np.add.at(spectrogram[t], shifted_bins, np.abs(amps) ** 2)
    time_axis_s = np.arange(n_frames) * frame_duration_s
    range_factor = 2.0 if mode == "monostatic" else 1.0
    mean_range = (
        float(np.median(np.concatenate(delay_blocks)) * C0 / range_factor)
        if delay_blocks
        else float("nan")
    )

    return MicroDopplerResult(
        spectrogram=spectrogram,
        time_axis_s=time_axis_s,
        doppler_axis_hz=doppler_axis_hz,
        range_bin=-1,
        range_m=mean_range,
        collapsed=True,
        truth_assisted=True,
        selection_source="analytical_path_doppler",
        range_bin_start=0,
        range_bin_end=fft_size_range - 1,
    )


# ---------------------------------------------------------------------------
# Ground-truth computation utilities
# ---------------------------------------------------------------------------


@dataclass
class GTEntry:
    """Per-frame ground-truth observables and detection values.

    ``target_speeds`` is the transitional alias for
    ``target_monostatic_equivalent_velocities_m_s`` so it can be compared
    like-for-like with the detection ``velocity_m_s`` field. An unavailable
    velocity is represented by ``None`` and accompanied by aligned validity,
    source, elapsed-time, and reason fields; it is never replaced by zero.
    """

    target_ranges: list[float]
    target_speeds: list[float | None]
    det_ranges: list[float]
    det_velocities: list[float]
    target_ids: list[str] = field(default_factory=list)
    reference_time_s: float | None = None
    target_path_closing_rates_m_s: list[float | None] = field(default_factory=list)
    target_monostatic_equivalent_velocities_m_s: list[float | None] = field(default_factory=list)
    target_bistatic_bisector_velocities_m_s: list[float | None] = field(default_factory=list)
    target_velocity_valid: list[bool] = field(default_factory=list)
    target_velocity_sources: list[str] = field(default_factory=list)
    target_velocity_reference_dt_s: list[float | None] = field(default_factory=list)
    target_velocity_unavailable_reasons: list[str | None] = field(default_factory=list)
    target_speed_semantics: str = "monostatic_equivalent_velocity"
    target_velocity_reference_evidence: list[dict[str, Any]] = field(
        default_factory=list, kw_only=True
    )


def extract_target_positions(frame: dict) -> list | None:
    """Return target positions from a frame dict, checking common key names.

    Args:
        frame: Frame dictionary (pipeline frame_data or HDF5-loaded frame).

    Returns:
        List of target position arrays, or ``None`` if no targets found.
    """
    for key in ("target_positions_snapshot", "target_pos"):
        val = frame.get(key)
        if val is not None:
            return val
    return None


def compute_gt_entry(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    target_positions: list,
    *,
    range_mode: str = "bistatic",
    dt: float | None = None,
    prev_target_ranges: list[float | None] | None = None,
    detections: list | None = None,
    target_velocities: np.ndarray | None = None,
    target_ids: Iterable[str] | None = None,
    reference_time_s: float | None = None,
    velocity_reference_policy: str = "legacy",
    tx_velocity_m_s: np.ndarray | None = None,
    rx_velocity_m_s: np.ndarray | None = None,
    pair_id: str | None = None,
) -> GTEntry:
    """Compute ground-truth ranges and radial speeds for sensing targets.

    This is the single source of truth for GT computation.  All consumers
    (pipeline, diagnostic tool, visualizer panel, sensing window) should
    call this instead of duplicating the math.

    Args:
        tx_pos: Selected TX position, shape ``(3,)``.
        rx_pos: Selected RX position, shape ``(3,)``.
        target_positions: List of target position arrays, each shape ``(3,)``.
        range_mode: ``"monostatic"`` (round-trip / 2) or ``"bistatic"`` (path length).
        dt: Exact elapsed time between the current and previous GT reference
            epochs. Required only when a previous range is used.
        prev_target_ranges: Previous target ranges aligned by exact target ID.
            ``None`` entries explicitly mean that target has no prior sample.
        detections: List of CFAR detection dicts with ``"range_m"`` and
            ``"velocity_m_s"`` keys.
        target_velocities: Optional (N_targets, 3) velocity array.  When
            provided and ``prev_target_ranges`` is not available, radial speed
            is computed analytically as the projection of the velocity vector
            onto the TX-target line of sight.
        target_ids: Optional exact authored IDs aligned with target positions.
        reference_time_s: Exact physical epoch of positions and velocities.
        velocity_reference_policy: ``legacy`` preserves interval precedence and
            the fixed-sensor analytic fallback. Explicit
            ``instantaneous_relative_kinematics_v1`` uses complete relative
            kinematics at the declared epoch, retaining separately labeled
            interval evidence when instantaneous state is unavailable.
        tx_velocity_m_s: Known solve-epoch TX velocity for the new policy.
            ``None`` means unknown; stationary sensors require explicit zeros.
        rx_velocity_m_s: Known solve-epoch RX velocity for the new policy.
        pair_id: Canonical acquisition pair ID, required by the new policy.

    Returns:
        GTEntry with computed ranges, speeds, and extracted detections.
    """
    tx = _finite_vector3(tx_pos, name="tx_pos")
    rx = _finite_vector3(rx_pos, name="rx_pos")
    normalized_range_mode = str(range_mode).strip().lower()
    if normalized_range_mode not in {"monostatic", "bistatic"}:
        raise ValueError("range_mode must be exactly 'monostatic' or 'bistatic'")
    mono = normalized_range_mode == "monostatic"
    if velocity_reference_policy not in {"legacy", INSTANTANEOUS_VELOCITY_REFERENCE_POLICY}:
        raise ValueError("Unknown velocity_reference_policy")
    instantaneous_policy = velocity_reference_policy == INSTANTANEOUS_VELOCITY_REFERENCE_POLICY
    if not instantaneous_policy and (tx_velocity_m_s is not None or rx_velocity_m_s is not None):
        raise ValueError("Explicit sensor velocities require the instantaneous reference policy")

    positions = [
        _finite_vector3(position, name=f"target_positions[{index}]")
        for index, position in enumerate(target_positions)
    ]
    target_count = len(positions)

    ids: list[str] = []
    if target_ids is not None:
        ids = list(target_ids)
        if len(ids) != target_count:
            raise ValueError("target_ids must align one-to-one with target_positions")
        if any(
            not isinstance(target_id, str) or not target_id or target_id != target_id.strip()
            for target_id in ids
        ):
            raise ValueError("target_ids must contain exact non-empty authored strings")
        if len(set(ids)) != len(ids):
            raise ValueError("target_ids must be unique")

    epoch_s: float | None = None
    if reference_time_s is not None:
        if isinstance(reference_time_s, (bool, np.bool_)):
            raise ValueError("reference_time_s must be finite")
        epoch_s = float(reference_time_s)
        if not np.isfinite(epoch_s):
            raise ValueError("reference_time_s must be finite")
    if instantaneous_policy:
        if epoch_s is None:
            raise ValueError("Instantaneous velocity references require reference_time_s")
        if target_ids is None:
            raise ValueError("Instantaneous velocity references require exact target_ids")
        target_rows_by_id([{"target_id": target_id} for target_id in ids])
        if not isinstance(pair_id, str) or re.fullmatch(r"[0-9a-f]{64}", pair_id) is None:
            raise ValueError("Instantaneous velocity references require canonical pair_id")

    tx_velocity = (
        None
        if tx_velocity_m_s is None
        else _finite_vector3(tx_velocity_m_s, name="tx_velocity_m_s")
    )
    rx_velocity = (
        None
        if rx_velocity_m_s is None
        else _finite_vector3(rx_velocity_m_s, name="rx_velocity_m_s")
    )

    elapsed_s: float | None = None
    if dt is not None:
        if isinstance(dt, (bool, np.bool_)):
            raise ValueError("dt must be finite and strictly positive")
        elapsed_s = float(dt)
        if not np.isfinite(elapsed_s) or elapsed_s <= 0.0:
            raise ValueError("dt must be finite and strictly positive")

    prior_ranges: list[float | None]
    if prev_target_ranges is None:
        prior_ranges = [None] * target_count
    else:
        prior_ranges = list(prev_target_ranges)
        if len(prior_ranges) != target_count:
            raise ValueError("prev_target_ranges must align one-to-one with target_positions")
        for index, prior_range in enumerate(prior_ranges):
            if prior_range is None:
                continue
            value = float(prior_range)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"prev_target_ranges[{index}] must be finite and non-negative")
            prior_ranges[index] = value

    velocity_rows: np.ndarray | None = None
    if target_velocities is not None:
        velocity_rows = np.asarray(target_velocities, dtype=np.float64)
        if target_count == 0 and velocity_rows.size == 0:
            velocity_rows = np.empty((0, 3), dtype=np.float64)
        if velocity_rows.shape != (target_count, 3) or not np.all(np.isfinite(velocity_rows)):
            raise ValueError("target_velocities must be a finite array with shape (N_targets, 3)")

    if any(prior_range is not None for prior_range in prior_ranges) and elapsed_s is None:
        raise ValueError("dt is required when prev_target_ranges contains prior samples")

    target_ranges: list[float] = []
    target_speeds: list[float | None] = []
    target_path_closing_rates: list[float | None] = []
    target_monostatic_equivalent_velocities: list[float | None] = []
    target_bistatic_bisector_velocities: list[float | None] = []
    target_velocity_valid: list[bool] = []
    target_velocity_sources: list[str] = []
    target_velocity_reference_dt_s: list[float | None] = []
    target_velocity_unavailable_reasons: list[str | None] = []
    target_velocity_reference_evidence: list[dict[str, Any]] = []

    for tidx, p in enumerate(positions):
        dist_tx = float(np.linalg.norm(tx - p))
        dist_rx = float(np.linalg.norm(rx - p))
        r = 0.5 * (dist_tx + dist_rx) if mono else dist_tx + dist_rx
        if not np.isfinite(r):
            raise ValueError(f"target_positions[{tidx}] produced a non-finite range")
        target_ranges.append(r)

        # Path-closing rate is positive while the full propagation path
        # shortens. The detection velocity coordinate is exactly half of that
        # value: physical radial velocity for monostatic sensing and a
        # monostatic-equivalent observable for bistatic sensing.
        prior_range = prior_ranges[tidx]
        velocity_valid = False
        velocity_source = "unavailable"
        velocity_reference_dt_s = None
        velocity_unavailable_reason: str | None = None
        path_closing_rate: float | None
        fallback_reason: str | None = None
        if instantaneous_policy:
            missing = [
                name
                for name, value in (
                    ("tx", tx_velocity),
                    ("rx", rx_velocity),
                    ("target", velocity_rows),
                )
                if value is None
            ]
            path_closing_rate = None
            if missing:
                fallback_reason = "missing_instantaneous_velocity:" + ",".join(missing)
            else:
                assert velocity_rows is not None
                try:
                    path_closing_rate = bistatic_path_closing_rate_m_s(
                        tx,
                        rx,
                        p,
                        velocity_rows[tidx],
                        tx_velocity_m_s=tx_velocity,
                        rx_velocity_m_s=rx_velocity,
                    )
                except ValueError:
                    fallback_reason = "instantaneous_projection_undefined_geometry"
                else:
                    velocity_valid = True
                    velocity_source = "analytic_velocity_projection"
            if not velocity_valid and prior_range is not None:
                assert elapsed_s is not None
                range_closing_rate = -(r - prior_range) / elapsed_s
                path_closing_rate = 2.0 * range_closing_rate if mono else range_closing_rate
                velocity_valid = True
                velocity_source = "exact_epoch_finite_difference"
                velocity_reference_dt_s = elapsed_s
            elif not velocity_valid:
                velocity_unavailable_reason = fallback_reason
        elif prior_range is not None:
            assert elapsed_s is not None
            range_closing_rate = -(r - prior_range) / elapsed_s
            path_closing_rate = 2.0 * range_closing_rate if mono else range_closing_rate
            velocity_valid = True
            velocity_source = "exact_epoch_finite_difference"
            velocity_reference_dt_s = elapsed_s
        elif velocity_rows is not None:
            try:
                path_closing_rate = bistatic_path_closing_rate_m_s(
                    tx,
                    rx,
                    p,
                    velocity_rows[tidx],
                )
            except ValueError:
                path_closing_rate = None
                velocity_unavailable_reason = "analytic_velocity_projection_undefined_geometry"
            else:
                velocity_valid = True
                velocity_source = "analytic_velocity_projection"
        else:
            path_closing_rate = None
            velocity_unavailable_reason = "no_prior_reference_epoch_or_target_velocity"

        monostatic_equivalent_velocity = (
            None if path_closing_rate is None else 0.5 * path_closing_rate
        )
        bistatic_bisector_velocity: float | None = None
        if velocity_valid and mono:
            bistatic_bisector_velocity = monostatic_equivalent_velocity
        elif velocity_valid:
            assert path_closing_rate is not None
            try:
                bistatic_bisector_velocity = bistatic_bisector_velocity_from_doppler_m_s(
                    path_closing_rate,
                    1.0,
                    tx,
                    rx,
                    p,
                )
            except ValueError:
                bistatic_bisector_velocity = None

        target_path_closing_rates.append(path_closing_rate)
        target_monostatic_equivalent_velocities.append(monostatic_equivalent_velocity)
        target_bistatic_bisector_velocities.append(bistatic_bisector_velocity)
        target_speeds.append(monostatic_equivalent_velocity)
        target_velocity_valid.append(velocity_valid)
        target_velocity_sources.append(velocity_source)
        target_velocity_reference_dt_s.append(velocity_reference_dt_s)
        target_velocity_unavailable_reasons.append(velocity_unavailable_reason)
        if instantaneous_policy:
            assert epoch_s is not None
            if velocity_source == "analytic_velocity_projection":
                support = {
                    "kind": "instantaneous",
                    "reference_time_s": epoch_s,
                    "start_time_s": epoch_s,
                    "end_time_s": epoch_s,
                }
            elif velocity_source == "exact_epoch_finite_difference":
                assert elapsed_s is not None
                start_time_s = epoch_s - elapsed_s
                support = {
                    "kind": "interval_mean",
                    "reference_time_s": (start_time_s + epoch_s) / 2.0,
                    "start_time_s": start_time_s,
                    "end_time_s": epoch_s,
                }
            else:
                support = {
                    "kind": "unavailable",
                    "reference_time_s": None,
                    "start_time_s": None,
                    "end_time_s": None,
                }
            target_velocity_reference_evidence.append(
                {
                    "schema_version": 1,
                    "policy": INSTANTANEOUS_VELOCITY_REFERENCE_POLICY,
                    "pair_id": pair_id,
                    "target_id": ids[tidx],
                    "evaluation_epoch_s": epoch_s,
                    "support": support,
                    "instantaneous_eligible": velocity_source == "analytic_velocity_projection",
                    "source": velocity_source,
                    "tx_position_m": tx.tolist(),
                    "rx_position_m": rx.tolist(),
                    "target_position_m": p.tolist(),
                    "tx_velocity_m_s": None if tx_velocity is None else tx_velocity.tolist(),
                    "rx_velocity_m_s": None if rx_velocity is None else rx_velocity.tolist(),
                    "target_velocity_m_s": (
                        None if velocity_rows is None else velocity_rows[tidx].tolist()
                    ),
                    "fallback_reason": fallback_reason,
                }
            )

    det_list = detections or []
    det_ranges: list[float] = []
    det_velocities: list[float] = []
    for detection_index, detection in enumerate(det_list):
        if "range_m" not in detection:
            continue
        if "velocity_m_s" not in detection:
            raise ValueError(
                f"detections[{detection_index}] with range_m requires exact velocity_m_s"
            )
        detection_range = float(detection["range_m"])
        detection_velocity = float(detection["velocity_m_s"])
        if (
            not np.isfinite(detection_range)
            or detection_range < 0.0
            or not np.isfinite(detection_velocity)
        ):
            raise ValueError(
                f"detections[{detection_index}] range_m/velocity_m_s must be finite and physical"
            )
        det_ranges.append(detection_range)
        det_velocities.append(detection_velocity)

    return GTEntry(
        target_ranges=target_ranges,
        target_speeds=target_speeds,
        det_ranges=det_ranges,
        det_velocities=det_velocities,
        target_ids=ids,
        reference_time_s=epoch_s,
        target_path_closing_rates_m_s=target_path_closing_rates,
        target_monostatic_equivalent_velocities_m_s=target_monostatic_equivalent_velocities,
        target_bistatic_bisector_velocities_m_s=target_bistatic_bisector_velocities,
        target_velocity_valid=target_velocity_valid,
        target_velocity_sources=target_velocity_sources,
        target_velocity_reference_dt_s=target_velocity_reference_dt_s,
        target_velocity_unavailable_reasons=target_velocity_unavailable_reasons,
        target_velocity_reference_evidence=target_velocity_reference_evidence,
    )


def auto_select_range_bin(
    gt_entries: list,
    sensing_data: dict,
    fft_size_range: int,
    sampling_frequency: float | None = None,
    range_mode: str = "bistatic",
    *,
    truth_assisted: bool = False,
    range_axis_m: Iterable[float] | None = None,
) -> int:
    """Pick a range bin from an explicit observation-derived detection.

    This selector is intentionally blind. Truth-assisted tracking belongs in
    the identity/time-keyed helpers above; it must never share a fallback path
    with detection-driven selection. Native persisted callers should supply
    their producer-authored ``range_axis_m``.

    Args:
        gt_entries: Optional evaluation truth entries.
        sensing_data: Last sensing data dict (for fallback detections).
        fft_size_range: Number of range bins.
        sampling_frequency: Sampling frequency in Hz.
        range_mode: ``"monostatic"`` or ``"bistatic"``.
        truth_assisted: Removed compatibility switch. ``True`` is rejected.
        range_axis_m: Optional exact range coordinate vector. Low-level callers
            may omit it and explicitly derive coordinates from sampling config.

    Returns:
        Selected range bin index.
    """
    if truth_assisted:
        raise ValueError(
            "auto_select_range_bin is blind-only; use an explicit target_id with the "
            "truth-assisted tracking helpers"
        )
    if range_axis_m is None:
        if sampling_frequency is None:
            raise ValueError(
                "Blind range-bin selection requires exact range_axis_m or an explicit "
                "low-level sampling_frequency"
            )
        range_axis, _ = _range_axis(fft_size_range, sampling_frequency, range_mode)
    else:
        range_axis = _required_exact_range_axis(
            range_axis_m,
            range_bin_count=fft_size_range,
            label="Blind range-bin selection",
        )

    # Prefer the strongest observation-derived detection.
    detections = sensing_data.get("detections") or []
    candidates: list[tuple[float, float]] = []
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        try:
            detection_range_m = float(detection["range_m"])
            detection_power = float(detection.get("power_linear", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(detection_range_m) and np.isfinite(detection_power):
            candidates.append((detection_power, detection_range_m))
    if candidates:
        _power, r = max(candidates, key=lambda item: item[0])
        bin_idx = int(np.argmin(np.abs(range_axis - r)))
        logger.info("Auto range bin from detection: bin=%d (%.2f m)", bin_idx, r)
        return bin_idx

    raise ValueError(
        "Blind range-bin selection requires a detection with explicit finite range_m; "
        "center-bin and truth fallbacks are disabled"
    )


# ---------------------------------------------------------------------------
# CIR extraction helpers (shared across snapshot and sequential strategies)
# ---------------------------------------------------------------------------


def extract_cir_link(
    paths: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened CIR amplitudes and delays from a Sionna paths object.

    Handles tensor-to-numpy conversion and the float32 real/imag split used
    by some Sionna versions.

    Returns:
        (a_link, tau_link) — 1-D complex amplitudes and 1-D float delays.
    """
    a, tau = paths.cir(normalize_delays=False)
    if hasattr(a, "numpy"):
        a = a.numpy()
    if hasattr(tau, "numpy"):
        tau = tau.numpy()
    a = np.asarray(a)
    tau = np.asarray(tau)

    if a.dtype == np.float32 and a.shape[0] == 2 and a.size == 2 * tau.size:
        a_link = (a[0] + 1j * a[1]).flatten()
        tau_link = tau.flatten()
    elif a.ndim >= 2:
        a_link = a.flatten()
        tau_link = tau.flatten()
    else:
        a_link = a
        tau_link = tau

    return a_link, tau_link


def apply_aoa_filter(
    a_2d: np.ndarray,
    tau_1d: np.ndarray,
    paths: Any,
    config: Any,
    frame_idx: int,
    *,
    path_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter CIR paths by AoA bounds before summation.

    Extracts ``phi_r`` (azimuth) and ``theta_r`` (zenith) from the Sionna
    paths object, converts to degrees (azimuth wrapped to ``[-180, 180)``,
    elevation ``= 90 - zenith``), and masks out paths outside the configured
    angular bounds.

    Args:
        a_2d: Complex amplitude array, shape ``(N_paths, ...)`` or 1-D.
        tau_1d: Delay array, shape ``(N_paths,)``.
        paths: Sionna paths object with ``phi_r`` / ``theta_r``.
        config: Sensing config with ``aoa_filter_enabled`` and bounds.
        frame_idx: Current frame index (for logging).
        path_mask: Optional selection on the original solver path axis, already
            applied to gains and delays. Apply it to angles before filtering.

    Returns:
        Filtered ``(a_2d, tau_1d)`` with rejected paths removed.
    """
    if not getattr(config, "aoa_filter_enabled", False):
        return a_2d, tau_1d
    if a_2d.shape[0] == 0 or tau_1d.size == 0:
        return a_2d, tau_1d

    phi_r = getattr(paths, "phi_r", None)
    theta_r = getattr(paths, "theta_r", None)
    if phi_r is None or theta_r is None:
        logger.warning("AoA filter enabled but paths lack phi_r/theta_r; skipping")
        return a_2d, tau_1d

    phi_r_np = np.asarray(phi_r.numpy() if hasattr(phi_r, "numpy") else phi_r)
    theta_r_np = np.asarray(theta_r.numpy() if hasattr(theta_r, "numpy") else theta_r)

    # Convert: azimuth in degrees wrapped to [-180, 180), elevation = 90 - zenith
    az_deg = np.degrees(phi_r_np).flatten()
    az_deg = (az_deg + 180.0) % 360.0 - 180.0
    el_deg = (90.0 - np.degrees(theta_r_np)).flatten()

    n = tau_1d.size
    if path_mask is not None:
        selection = np.asarray(path_mask)
        if (
            selection.ndim != 1
            or selection.dtype != np.dtype(bool)
            or np.count_nonzero(selection) != n
            or az_deg.size != selection.size
            or el_deg.size != selection.size
        ):
            raise ValueError("AoA filter angle/path axes must agree with the original path mask")
        az_deg, el_deg = az_deg[selection], el_deg[selection]
    if az_deg.size != n or el_deg.size != n or a_2d.shape[0] != n:
        raise ValueError("AoA filter angle/path axes must agree; implicit truncation is forbidden")

    mask = np.ones(n, dtype=bool)
    az_min = getattr(config, "aoa_az_min_deg", None)
    az_max = getattr(config, "aoa_az_max_deg", None)
    if az_min is not None and az_max is not None:
        if az_min > az_max:  # wrap-around
            mask &= (az_deg >= az_min) | (az_deg <= az_max)
        else:
            mask &= (az_deg >= az_min) & (az_deg <= az_max)
    elif az_min is not None:
        mask &= az_deg >= az_min
    elif az_max is not None:
        mask &= az_deg <= az_max

    el_min = getattr(config, "aoa_el_min_deg", None)
    el_max = getattr(config, "aoa_el_max_deg", None)
    if el_min is not None:
        mask &= el_deg >= el_min
    if el_max is not None:
        mask &= el_deg <= el_max

    kept = int(np.sum(mask))
    total = n
    if kept < total:
        logger.debug(
            "AoA filter frame %d: %d/%d paths kept (az=[%s,%s] el=[%s,%s])",
            frame_idx,
            kept,
            total,
            az_min,
            az_max,
            el_min,
            el_max,
        )
    return a_2d[mask], tau_1d[mask]
