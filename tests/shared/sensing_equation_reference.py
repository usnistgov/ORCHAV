"""Independent numerical references for the sensing equation contract.

This module deliberately does not import :mod:`shared.sensing.signal_processing` or any
generator implementation.  Expected values in the Phase 1B tests therefore
come from closed-form equations or a direct DFT, not from the kernels under
test.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0


def point_bin_floor_projection(
    amplitudes: Iterable[complex],
    delays_s: Iterable[float],
    *,
    sampling_frequency_hz: float,
    bin_count: int,
) -> np.ndarray:
    """Project sparse paths with the explicit ``floor(tau * Fs)`` model."""
    result = np.zeros(bin_count, dtype=np.complex128)
    for amplitude, delay_s in zip(amplitudes, delays_s, strict=True):
        bin_index = int(np.floor(float(delay_s) * sampling_frequency_hz))
        if 0 <= bin_index < bin_count:
            result[bin_index] += complex(amplitude)
    return result


def bandlimited_delay_projection(
    amplitudes: Iterable[complex],
    delays_s: Iterable[float],
    *,
    sampling_frequency_hz: float,
    bin_count: int,
) -> np.ndarray:
    """Return the ideal infinite-support sinc reference on the delay grid."""
    delay_bins = np.arange(bin_count, dtype=np.float64)
    result = np.zeros(bin_count, dtype=np.complex128)
    for amplitude, delay_s in zip(amplitudes, delays_s, strict=True):
        fractional_bin = float(delay_s) * sampling_frequency_hz
        result += complex(amplitude) * np.sinc(delay_bins - fractional_bin)
    return result


def range_axis_m(
    bin_count: int,
    *,
    sampling_frequency_hz: float,
    monostatic: bool,
) -> np.ndarray:
    """Map delay-bin indices to monostatic range or bistatic path length."""
    propagation_factor = 2.0 if monostatic else 1.0
    return (
        np.arange(bin_count, dtype=np.float64)
        * SPEED_OF_LIGHT_M_S
        / (sampling_frequency_hz * propagation_factor)
    )


def centered_doppler_axis_hz(sample_count: int, *, prf_hz: float) -> np.ndarray:
    """Construct a centered DFT-frequency axis without ``np.fft.fftfreq``."""
    first_index = -(sample_count // 2)
    last_index = first_index + sample_count
    return np.arange(first_index, last_index, dtype=np.float64) * prf_hz / sample_count


def wrap_doppler_hz(doppler_hz: float, *, prf_hz: float) -> float:
    """Wrap a continuous Doppler frequency into ``[-PRF/2, PRF/2)``."""
    return (float(doppler_hz) + prf_hz / 2.0) % prf_hz - prf_hz / 2.0


def complex_tone(
    doppler_hz: float,
    *,
    prf_hz: float,
    sample_count: int,
    amplitude: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Sample ``A exp(+j 2 pi f n / PRF)`` on the slow-time grid."""
    sample_indices = np.arange(sample_count, dtype=np.float64)
    return complex(amplitude) * np.exp(2j * np.pi * float(doppler_hz) * sample_indices / prf_hz)


def centered_unitary_dft(signal: np.ndarray, *, prf_hz: float) -> np.ndarray:
    """Evaluate the centered unitary DFT directly from its matrix equation."""
    values = np.asarray(signal, dtype=np.complex128)
    sample_count = values.shape[0]
    sample_indices = np.arange(sample_count, dtype=np.float64)
    frequencies_hz = centered_doppler_axis_hz(sample_count, prf_hz=prf_hz)
    kernel = np.exp(
        -2j * np.pi * frequencies_hz[:, np.newaxis] * sample_indices[np.newaxis, :] / prf_hz
    )
    return kernel @ values / np.sqrt(sample_count)


def reference_window(name: str | None, sample_count: int, *, beta: float = 14.0) -> np.ndarray:
    """Evaluate every supported symmetric sensing window from its equation."""
    if sample_count <= 1:
        return np.ones(max(sample_count, 0), dtype=np.float64)

    normalized = "rect" if name is None else name.lower().replace("-", "").replace("_", "")
    sample_indices = np.arange(sample_count, dtype=np.float64)
    phase = 2.0 * np.pi * sample_indices / (sample_count - 1)

    if normalized in {"rect", "rectangular", "boxcar", "none"}:
        return np.ones(sample_count, dtype=np.float64)
    if normalized in {"hann", "hanning"}:
        return 0.5 - 0.5 * np.cos(phase)
    if normalized == "hamming":
        return 0.54 - 0.46 * np.cos(phase)
    if normalized == "blackman":
        return 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    if normalized == "blackmanharris":
        return (
            0.35875
            - 0.48829 * np.cos(phase)
            + 0.14128 * np.cos(2.0 * phase)
            - 0.01168 * np.cos(3.0 * phase)
        )
    if normalized == "kaiser":
        coordinate = 2.0 * sample_indices / (sample_count - 1) - 1.0
        return np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - coordinate**2))) / np.i0(beta)
    raise ValueError(f"Unsupported reference window {name!r}")


def coherent_gain(window: np.ndarray) -> float:
    """Return coherent amplitude gain ``sum(w) / N``."""
    values = np.asarray(window, dtype=np.float64)
    return float(np.sum(values) / values.size) if values.size else 1.0


def equivalent_noise_bandwidth_bins(window: np.ndarray) -> float:
    """Return ENBW in DFT bins, ``N sum(w^2) / |sum(w)|^2``."""
    values = np.asarray(window, dtype=np.float64)
    denominator = float(np.abs(np.sum(values)) ** 2)
    if values.size == 0 or denominator == 0.0:
        return 0.0
    return float(values.size * np.sum(values**2) / denominator)


def energy_correction(window: np.ndarray) -> float:
    """Return RMS window correction ``sqrt(N / sum(w^2))``."""
    values = np.asarray(window, dtype=np.float64)
    energy = float(np.sum(values**2))
    return float(np.sqrt(values.size / energy)) if values.size and energy > 0.0 else 1.0


def path_closing_rate_from_doppler_m_s(doppler_hz: float, *, wavelength_m: float) -> float:
    """Return the observable ``-dL/dt = wavelength * Doppler``."""
    return float(wavelength_m) * float(doppler_hz)


def monostatic_radial_closing_velocity_m_s(
    doppler_hz: float,
    *,
    wavelength_m: float,
) -> float:
    """Return monostatic radial closing velocity ``lambda * f_d / 2``."""
    return 0.5 * path_closing_rate_from_doppler_m_s(
        doppler_hz,
        wavelength_m=wavelength_m,
    )


def _unit_direction(origin: np.ndarray, destination: np.ndarray) -> np.ndarray:
    vector = np.asarray(destination, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction is undefined for coincident points")
    return vector / norm


def bistatic_path_closing_rate_m_s(
    tx_position_m: np.ndarray,
    rx_position_m: np.ndarray,
    target_position_m: np.ndarray,
    target_velocity_m_s: np.ndarray,
    *,
    tx_velocity_m_s: np.ndarray | None = None,
    rx_velocity_m_s: np.ndarray | None = None,
) -> float:
    """Evaluate ``-d(||p-tx|| + ||p-rx||)/dt`` from geometry."""
    target_to_tx = _unit_direction(target_position_m, tx_position_m)
    target_to_rx = _unit_direction(target_position_m, rx_position_m)
    target_velocity = np.asarray(target_velocity_m_s, dtype=np.float64)
    tx_velocity = (
        np.zeros(3, dtype=np.float64)
        if tx_velocity_m_s is None
        else np.asarray(tx_velocity_m_s, dtype=np.float64)
    )
    rx_velocity = (
        np.zeros(3, dtype=np.float64)
        if rx_velocity_m_s is None
        else np.asarray(rx_velocity_m_s, dtype=np.float64)
    )
    return float(
        np.dot(target_to_tx, target_velocity - tx_velocity)
        + np.dot(target_to_rx, target_velocity - rx_velocity)
    )


def bistatic_bisector_velocity_m_s(
    doppler_hz: float,
    *,
    wavelength_m: float,
    tx_position_m: np.ndarray,
    rx_position_m: np.ndarray,
    target_position_m: np.ndarray,
    tx_velocity_m_s: np.ndarray | None = None,
    rx_velocity_m_s: np.ndarray | None = None,
    minimum_geometry_scale: float = 1.0e-9,
) -> float:
    """Convert Doppler to velocity projected on the bistatic unit bisector.

    The conversion is undefined when the two target-to-sensor unit vectors
    cancel at forward scatter.
    """
    target_to_tx = _unit_direction(target_position_m, tx_position_m)
    target_to_rx = _unit_direction(target_position_m, rx_position_m)
    direction_sum = target_to_tx + target_to_rx
    geometry_scale = float(np.linalg.norm(direction_sum))
    if geometry_scale <= float(minimum_geometry_scale):
        raise ValueError("bistatic bisector is unobservable near forward scatter")

    tx_velocity = (
        np.zeros(3, dtype=np.float64)
        if tx_velocity_m_s is None
        else np.asarray(tx_velocity_m_s, dtype=np.float64)
    )
    rx_velocity = (
        np.zeros(3, dtype=np.float64)
        if rx_velocity_m_s is None
        else np.asarray(rx_velocity_m_s, dtype=np.float64)
    )
    sensor_term = float(np.dot(target_to_tx, tx_velocity) + np.dot(target_to_rx, rx_velocity))
    path_closing_rate = path_closing_rate_from_doppler_m_s(
        doppler_hz,
        wavelength_m=wavelength_m,
    )
    return (path_closing_rate + sensor_term) / geometry_scale
