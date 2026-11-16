"""Pre-optimization complex128 CFR reference pinned to main b14bd1e9.

Only the deterministic input factory and original per-element projector are
copied from the study; no private benchmark or alternate kernels are imported.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np

from shared.sensing.range_response import (
    CFR_MODEL,
    FrequencyNoise,
    RangeResponseConfig,
    _integer,
    frequency_grid,
    validate_sample_times,
)


def reference_paths_to_frequency(
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
    validate_sample_times(sample_times_s, t)
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
    h = np.zeros((e, t, frequencies.size), dtype=np.complex128)
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
    noise = config.frequency_noise
    realization_index = _integer(realization_index, "realization_index", 0)
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


def make_case(elements, paths, times, frequencies, delay_kind="shared", seed=20261014):
    rng = np.random.default_rng(seed)
    a = (
        rng.normal(size=(elements, paths, times)) + 1j * rng.normal(size=(elements, paths, times))
    ).astype(np.complex128)
    # All three candidate grids use the same native coefficients/delays. Most
    # paths lie below 80 m and the last valid path is just below the 320 period.
    delays = rng.uniform(0, 550e-9, paths)
    if paths:
        delays[0] = 0
        delays[-1] = np.nextafter(320 / 500e6, 0)
    valid = np.ones(paths, dtype=bool)
    if paths > 2:
        delays[1] = 13 / 500e6  # exact 512-grid delay sample
        valid[2] = False
    if delay_kind == "element_specific":
        delays = np.broadcast_to(delays, (elements, paths)).copy()
        if paths:
            delays[:, 0] = np.arange(elements) * 1e-12
        valid = np.broadcast_to(valid, (elements, paths)).copy()
    cfg = RangeResponseConfig("cfr_ifft", frequencies, "boxcar", FrequencyNoise(1e-8, 52))
    return (a, delays, valid, 7 + np.arange(times) / 1000), dict(
        bandwidth=500e6, config=cfg, realization_index=3
    )
