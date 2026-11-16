"""Finite-bandwidth response on an explicit baseband frequency grid.

Coefficients already contain carrier phase and slow-time evolution. This CPU
boundary never synthesizes either again. It models an ideal channel observation,
not transmitted symbols or a receiver. All delays are absolute seconds.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Strict

C0 = 299_792_458.0
CFR_MODEL = "cfr_ifft"


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return int(value)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class FrequencyNoise:
    """Circular complex observation noise; variance is E[abs(noise)**2]."""

    variance: Annotated[float, Strict()] = 0.0
    seed: Annotated[int, Strict()] | None = None

    def __post_init__(self) -> None:
        _number(self.variance, "frequency_noise.variance")
        if self.seed is not None:
            _integer(self.seed, "frequency_noise.seed", 0)
        if self.variance > 0 and self.seed is None:
            raise ValueError("Nonzero frequency_noise.variance requires an explicit seed")


@dataclass(frozen=True)
class RangeResponseConfig:
    """One shared generation/replay configuration; omission preserves point bins."""

    model: Literal["point_bin_floor", "cfr_ifft"] = "point_bin_floor"
    num_frequency_bins: Annotated[int, Strict()] | None = None
    frequency_window: Literal["boxcar", "hann"] = "boxcar"
    frequency_noise: FrequencyNoise = field(default_factory=FrequencyNoise)

    def __post_init__(self) -> None:
        if isinstance(self.frequency_noise, Mapping):
            object.__setattr__(self, "frequency_noise", FrequencyNoise(**self.frequency_noise))
        if not isinstance(self.frequency_noise, FrequencyNoise):
            raise ValueError("frequency_noise must be a FrequencyNoise configuration")
        if self.model not in {"point_bin_floor", CFR_MODEL}:
            raise ValueError("range_response.model must be point_bin_floor or cfr_ifft")
        if self.frequency_window not in {"boxcar", "hann"}:
            raise ValueError("frequency_window must be boxcar or periodic hann")
        if self.model == CFR_MODEL:
            _integer(self.num_frequency_bins, "num_frequency_bins", 2)
        elif (
            self.num_frequency_bins is not None
            or self.frequency_window != "boxcar"
            or self.frequency_noise != FrequencyNoise()
        ):
            raise ValueError("Frequency grid/window/noise settings require model: cfr_ifft")

    @classmethod
    def parse(cls, value: Any) -> RangeResponseConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("range_response must be a mapping")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"Invalid range_response fields: {exc}") from exc

    def validate_processing(
        self,
        *,
        bandwidth: float,
        fft_size: int,
        window_range: str,
        noise_enabled: bool = False,
        noise_snr_db: float | None = None,
        processing_mode: str = "coherent_cpi",
    ) -> None:
        if self.model != CFR_MODEL:
            return
        _number(bandwidth, "bandwidth", positive=True)
        _integer(fft_size, "fft_size_range", int(self.num_frequency_bins))
        if window_range != "boxcar":
            raise ValueError(
                "cfr_ifft requires explicit window_range: boxcar; use frequency_window for spectral weighting"
            )
        if noise_enabled or noise_snr_db is not None:
            raise ValueError(
                "cfr_ifft conflicts with legacy noise_enabled/noise_snr_db; use range_response.frequency_noise"
            )
        if processing_mode not in {"coherent_cpi", "sequential_stft", "auto"}:
            raise ValueError("cfr_ifft supports coherent_cpi or SISO sequential_stft processing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frequency_grid(bandwidth: float, num_bins: int) -> np.ndarray:
    """Increasing signed offsets q * B/N, q = k - floor(N/2)."""
    bandwidth = _number(bandwidth, "bandwidth", positive=True)
    num_bins = _integer(num_bins, "num_frequency_bins", 2)
    return (np.arange(num_bins, dtype=np.float64) - num_bins // 2) * (bandwidth / num_bins)


def frequency_weights(size: int, window: str) -> np.ndarray:
    size = _integer(size, "num_frequency_bins", 2)
    if window == "boxcar":
        return np.ones(size)
    if window == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    raise ValueError("frequency_window must be boxcar or periodic hann")


def validate_sample_times(sample_times: Any, size: int, *, minimum_samples: int = 2) -> np.ndarray:
    """Validate physical sampling; projection alone also permits one pulse."""
    times = np.asarray(sample_times, dtype=np.float64)
    if times.shape != (size,) or size < minimum_samples or not np.all(np.isfinite(times)):
        raise ValueError(
            "sample_times must contain one finite physical timestamp per slow-time sample "
            f"(at least {minimum_samples})"
        )
    if size == 1:
        return times
    steps = np.diff(times)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-12):
        raise ValueError("cfr_ifft requires strictly increasing uniform sample_times")
    return times


def _project_valid_paths(
    a: np.ndarray, delays: np.ndarray, valid: np.ndarray, frequencies: np.ndarray
) -> np.ndarray:
    """Reuse only exactly shared kernels; keep each matrix product unchanged.

    Validity is already sanitized by ``paths_to_frequency``. Equal delay values
    alone are insufficient: the selected path order must also match, so different
    element masks retain the original per-element calculation. No tolerance is
    used when deciding whether delays are shared.
    """
    e, _, t = a.shape
    h = np.zeros((e, t, frequencies.size), dtype=np.complex128)
    selected = np.flatnonzero(valid[0])
    # Check one row at a time so eligibility scratch grows with paths, not
    # elements x paths. A zero element stride guarantees identical delay rows
    # (the common one-dimensional-delay broadcast) without any delay gather.
    shared = e > 1 and all(np.array_equal(valid[element], valid[0]) for element in range(1, e))
    if shared and delays.strides[0] != 0:
        reference_delays = delays[0, selected]
        shared = all(
            np.array_equal(delays[element, selected], reference_delays) for element in range(1, e)
        )
    if shared:
        for first in range(0, selected.size, 128):
            paths = selected[first : first + 128]
            for k in range(0, frequencies.size, 256):
                kernel = np.exp(
                    -2j * np.pi * delays[0, paths, None] * frequencies[None, k : k + 256]
                )
                for element in range(e):
                    for m in range(0, t, 128):
                        h[element, m : m + 128, k : k + 256] += (
                            a[element, paths, m : m + 128].T @ kernel
                        )
        return h
    for element in range(e):
        selected = np.flatnonzero(valid[element])
        for first in range(0, selected.size, 128):
            paths = selected[first : first + 128]
            for k in range(0, frequencies.size, 256):
                kernel = np.exp(
                    -2j * np.pi * delays[element, paths, None] * frequencies[None, k : k + 256]
                )
                for m in range(0, t, 128):
                    h[element, m : m + 128, k : k + 256] += (
                        a[element, paths, m : m + 128].T @ kernel
                    )
    return h


def paths_to_frequency(
    coefficients: np.ndarray,
    delays_s: np.ndarray,
    validity: np.ndarray,
    sample_times_s: np.ndarray,
    *,
    bandwidth: float,
    config: RangeResponseConfig,
    realization_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project [element,path,time] into unwindowed [element,time,frequency].

    Delays and validity have explicit [path] or [element,path] axes. Invalid
    padding is excluded; a nonfinite valid path is dropped across all elements,
    matching native coherent path selection. Empty channels are valid. Missing
    acquisition is an error. Temporary kernels are bounded by 256 x 128 cells.
    """
    if coefficients is None or delays_s is None or validity is None:
        raise ValueError("Unavailable acquisition: coefficients, delays and validity are required")
    if config.model != CFR_MODEL:
        raise ValueError("paths_to_frequency requires model: cfr_ifft")
    a = np.asarray(coefficients)
    if a.ndim != 3 or not np.iscomplexobj(a) or a.shape[0] < 1:
        raise ValueError("coefficients must be complex [receive_element,path,slow_time]")
    e, p, t = a.shape
    validate_sample_times(sample_times_s, t, minimum_samples=1)
    delays = np.asarray(delays_s, dtype=np.float64)
    valid = np.asarray(validity)
    if (
        delays.shape not in {(p,), (e, p)}
        or valid.shape not in {(p,), (e, p)}
        or valid.dtype != bool
    ):
        raise ValueError(
            "delays and boolean validity must have explicit [path] or [element,path] axes"
        )
    delays = np.broadcast_to(delays, (e, p))
    valid = np.broadcast_to(valid, (e, p)).copy()
    finite = np.isfinite(delays) & np.all(np.isfinite(a), axis=2)
    nonfinite_paths = np.any(valid & ~finite, axis=0)
    valid[:, nonfinite_paths] = False
    frequencies = frequency_grid(bandwidth, config.num_frequency_bins)
    period = len(frequencies) / bandwidth
    outside = valid & ((delays < 0) | (delays >= period))
    if np.any(outside):
        bad = delays[outside]
        raise ValueError(
            f"{bad.size} valid delays outside [0, {period:.12g}) s: {bad.min():.12g}..{bad.max():.12g}; increase num_frequency_bins/B for a longer ambiguity period"
        )
    h = _project_valid_paths(a, delays, valid, frequencies)
    noise = config.frequency_noise
    realization_index = _integer(realization_index, "realization_index", 0)
    if noise.variance:
        h = add_frequency_noise(h, noise, realization_index=realization_index)
    return h, {
        "invalid_entry_count": int((~np.broadcast_to(np.asarray(validity), (e, p))).sum()),
        "nonfinite_path_count": int(nonfinite_paths.sum()),
        "contributing_element_path_count": int(valid.sum()),
        "noise": {
            **asdict(noise),
            "realization_index": realization_index,
            "generator": "PCG64/SeedSequence(seed,realization_index)",
            "domain": "unwindowed_frequency_observation",
            "power_unit": "1",
            "observation_sha256": hashlib.sha256(h.astype("<c16").tobytes()).hexdigest(),
        },
    }


def add_frequency_noise(
    observation: np.ndarray,
    noise: FrequencyNoise,
    *,
    realization_index: int = 0,
) -> np.ndarray:
    """Copy and perturb an unwindowed CFR using the native acquisition injector.

    Caller owns provenance and must ensure noise is applied only once. Variance
    is dimensionless E[abs(noise)**2], not a receiver thermal/noise-figure model.
    This preserves the original native PCG64 stream and arithmetic bit for bit.
    Explicit variance also supports a zero-input, target-absent control.
    """
    realization_index = _integer(realization_index, "realization_index", 0)
    if not isinstance(noise, FrequencyNoise):
        raise ValueError("noise must be FrequencyNoise")
    h = np.array(observation, dtype=np.complex128, order="C", copy=True)
    if h.ndim != 3 or not np.iscomplexobj(observation) or not np.all(np.isfinite(h)):
        raise ValueError("observation must be finite complex [element,time,frequency]")
    if noise.variance:
        rng = np.random.default_rng(np.random.SeedSequence([noise.seed, realization_index]))
        # Draw real/imaginary pairs together; bounded blocks preserve stream order.
        flat = h.reshape(-1)
        for first in range(0, flat.size, 65536):
            count = min(65536, flat.size - first)
            draws = rng.standard_normal((count, 2))
            flat[first : first + count] += np.sqrt(noise.variance / 2) * (
                draws[:, 0] + 1j * draws[:, 1]
            )
    return h


def frequency_to_delay(
    observation: np.ndarray,
    *,
    bandwidth: float,
    config: RangeResponseConfig,
    fft_size: int,
    range_mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Weighted IFFT with signed-frequency placement and unit coherent peak."""
    config.validate_processing(bandwidth=bandwidth, fft_size=fft_size, window_range="boxcar")
    if config.model != CFR_MODEL or range_mode not in {"monostatic", "bistatic"}:
        raise ValueError(
            "frequency_to_delay requires cfr_ifft and monostatic or bistatic coordinates"
        )
    h = np.asarray(observation)
    n = int(config.num_frequency_bins)
    if h.ndim != 3 or h.shape[-1] != n or not np.iscomplexobj(h) or not np.all(np.isfinite(h)):
        raise ValueError(
            "observation must be finite complex [element,slow_time,frequency] on the declared grid"
        )
    frequencies = frequency_grid(bandwidth, n)
    w = frequency_weights(n, config.frequency_window)
    bins = (np.arange(n) - n // 2) % fft_size
    spectrum = np.zeros((*h.shape[:-1], fft_size), dtype=np.complex128)
    spectrum[..., bins] = h * w
    response = np.fft.ifft(spectrum, axis=-1) * (fft_size / w.sum())
    delta_f = bandwidth / n
    delays = np.arange(fft_size, dtype=np.float64) / (fft_size * delta_f)
    factor = 2 if range_mode == "monostatic" else 1
    axes = {
        "frequency_axis_hz": frequencies,
        "delay_axis_s": delays,
        "range_axis_m": C0 * delays / factor,
    }
    metadata = {
        "model": CFR_MODEL,
        "model_version": 1,
        "input_phase_convention": "baseband_coefficients_carrier_and_doppler_already_applied",
        "delay_reference": "absolute_zero",
        "frequency_reference": "offset_from_carrier",
        "bandwidth_hz": bandwidth,
        "bandwidth_convention": "N_times_frequency_spacing",
        "num_frequency_bins": n,
        "frequency_spacing_hz": delta_f,
        "fft_size_range": fft_size,
        "delay_ambiguity_period_s": 1 / delta_f,
        "delay_sample_spacing_s": 1 / (fft_size * delta_f),
        "range_sample_spacing_m": C0 / (factor * fft_size * delta_f),
        "range_resolution_scale_m": C0 / (factor * bandwidth),
        "measurement_uncertainty_policy": "configured_tracker_covariance_independent_of_sample_spacing",
        "frequency_window": config.frequency_window,
        "hann_convention": "periodic",
        "frequency_window_definition": (
            "1" if config.frequency_window == "boxcar" else "0.5-0.5*cos(2*pi*k/N)"
        ),
        "coherent_gain": float(w.sum()),
        "noise_gain": float(np.sum(w * w) / w.sum() ** 2),
        "range_transform": "ifft",
        "range_transform_axis": "frequency",
        "range_transform_normalization": "unit_coherent_peak",
        "delay_projection": CFR_MODEL,
        "range_processing": "finite_bandwidth_delay_response",
        "range_fft_applied": True,
    }
    return response, axes, metadata


def validate_execution_range_response(config: Mapping[str, Any]) -> RangeResponseConfig:
    """Version 1 is the unchanged legacy reader; version 2 requires CFR semantics."""
    response = RangeResponseConfig.parse(config.get("range_response"))
    expected = 2 if response.model == CFR_MODEL else 1
    if config.get("contract_version") != expected:
        raise ValueError(f"{response.model} requires execution contract_version {expected}")
    if expected == 2:
        recipe = config.get("processing_recipe", {})
        window_range = config.get("window_range", recipe.get("window_range"))
        mode = config.get("processing_mode")
        response.validate_processing(
            bandwidth=config.get("bandwidth", config.get("sampling_frequency")),
            fft_size=config.get("fft_size_range"),
            window_range=window_range,
            processing_mode="coherent_cpi" if mode == "offline_reprocess" else mode,
        )
        processing = config.get(
            "source_processing" if mode == "offline_reprocess" else "processing", {}
        )
        if processing.get("product_semantics") != "cfr_ifft_slow_time_fft":
            raise ValueError("CFR execution contract requires cfr_ifft_slow_time_fft semantics")
    return response
