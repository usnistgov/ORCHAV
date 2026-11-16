"""Read-only signal inspection on one recorded pair and acquisition.

Saved variants are compared only with matching axes, provenance and receiver
combination. An explicitly requested missing spectrum can be reconstructed from
retained complex samples with the declared recipe, provided the same operation
also reproduces the saved primary spectrum. No detector or tracker is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from shared.sensing.products import select_product_by_contract
from shared.sensing.signal_processing import compute_range_doppler

from .inspection_display import range_label

MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_WORK_BYTES = 1024 * 1024 * 1024
_RD_AXES = ("pair", "doppler_frequency", "range")
_INPUT_AXES = ("pair", "rx_element", "slow_time", "range")
_RECIPE = {
    "range_processing": "delay_domain_cir_bins",
    "range_fft_applied": False,
    "fft_axis": "slow_time",
    "fft_normalization": "unitary_1_over_sqrt_n",
    "window_normalization": "energy_preserving_rms",
    "slow_time_mean_removal_operator": "finite_cpi_mean_projection",
}
_WINDOWS = {
    "hann",
    "hanning",
    "hamming",
    "blackman",
    "blackmanharris",
    "kaiser",
    "rect",
    "rectangular",
    "boxcar",
    "none",
}


@dataclass(frozen=True)
class ProcessingInspection:
    """Bounded plot data and its recorded/reconstructed evidence status."""

    mode: str
    ranges: np.ndarray | None = None
    doppler: np.ndarray | None = None
    before: np.ndarray | None = None
    after: np.ndarray | None = None
    samples: np.ndarray | None = None
    sample_power: np.ndarray | None = None
    times: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    elements: tuple[str, ...] = ()
    notice: str = ""
    details: str = ""
    can_reconstruct: bool = False
    reconstructed: bool = False
    range_label: str = "Range"
    identity: tuple = ()
    scale_scope: tuple = ()
    # CIR range bin or CFR slow-time sample; no detector output is consulted.
    default_coordinate: int | None = None


def preflight_input(descriptor: dict) -> int:
    """Refuse unsafe dense input reads from shape/dtype metadata, before allocation.

    Current provider projections select whole products, not element hyperslabs.
    Budget the input, validation/immutable copies and conservative complex128
    combination, FFT and power temporaries. This is a worker estimate, not RSS.
    """
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 4
        or any(type(s) is not int or s <= 0 for s in shape)
    ):
        raise ValueError(
            "Input shape is missing or invalid; dense read declined before allocation."
        )
    dtype = np.dtype(descriptor.get("dtype"))
    if dtype.kind != "c" or shape[0] != 1:
        raise ValueError("Input inspection requires one pair of complex samples.")
    size = prod(shape) * dtype.itemsize
    work = 4 * size + 10 * prod(shape[-2:]) * 16
    if size > MAX_INPUT_BYTES or work > MAX_WORK_BYTES:
        raise ValueError(
            f"Input read declined before allocation: exceeds {MAX_INPUT_BYTES // 2**20} MiB "
            f"selected input or {MAX_WORK_BYTES // 2**20} MiB estimated worker buffers. "
            "No implicit regeneration."
        )
    return work


def _preflight_frequency_reconstruction(
    sensing: dict, descriptor: dict, *, include_doppler: bool = False
) -> int:
    """Budget padded delay arrays before an explicit frequency reconstruction."""
    source_work = preflight_input(descriptor)
    range_bins = sensing.get("config_resolved", {}).get("fft_size_range")
    if type(range_bins) is not int or range_bins <= 0:
        raise ValueError("Delay reconstruction requires a positive integer range FFT size")
    shape = descriptor["shape"]
    source_bytes = prod(shape) * np.dtype(descriptor["dtype"]).itemsize
    output_bytes = prod(shape[:-1]) * range_bins * np.dtype(np.complex128).itemsize
    # Frequency padding, FFT work and the owned delay copy coexist with the
    # retained source. Spectrum replay also retains both per-element RD cubes.
    output_buffers = 6 if include_doppler else 4
    combined_work = 10 * shape[-2] * range_bins * np.dtype(np.complex128).itemsize
    work = max(source_work, 2 * source_bytes + output_buffers * output_bytes + combined_work)
    if work > MAX_WORK_BYTES:
        raise ValueError(
            f"Projected reconstruction exceeds {MAX_WORK_BYTES // 2**20} MiB estimated worker "
            f"buffers with {range_bins} range bins; no transform was allocated."
        )
    return work


def _context(sensing: dict, descriptor: dict) -> dict:
    lineage = descriptor.get("lineage", {})
    mode = sensing.get("config_resolved", {}).get("range_mode", "unknown")
    return dict(
        range_label=range_label(mode),
        identity=(
            descriptor.get("product_id"),
            descriptor.get("data_sha256"),
            str(descriptor.get("axes")),
            str(descriptor.get("combine")),
        ),
        scale_scope=(
            lineage.get("processing_run_id"),
            lineage.get("processing_recipe_id"),
            str(descriptor.get("pair_ids")),
            mode,
            str(descriptor.get("combine")),
            descriptor.get("scale_semantics"),
            descriptor.get("unit"),
        ),
    )


def _frozen(values: Any) -> np.ndarray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _select(sensing: dict, kind: str, *, variant: str = "primary") -> tuple:
    return select_product_by_contract(
        sensing,
        kind=kind,
        variant=variant,
        axis_names=_INPUT_AXES if kind == "delay_domain_slow_time" else _RD_AXES,
        representations=(
            ("complex_voltage",)
            if kind == "delay_domain_slow_time"
            else ("complex_voltage", "power_linear")
        ),
    )


def _power(values: np.ndarray, descriptor: dict) -> np.ndarray:
    values = values[0]
    return _frozen(
        np.abs(values) ** 2 if descriptor["representation"] == "complex_voltage" else values
    )


def _same_event(left: dict, right: dict) -> None:
    for key in ("pair_ids", "time_span", "combine", "scale_semantics", "unit"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Products have different {key}; comparison is unavailable.")
    for key in ("acquisition_id", "processing_run_id", "processing_recipe_id"):
        if left["lineage"].get(key) != right["lineage"].get(key):
            raise ValueError(f"Products have different {key}; comparison is unavailable.")


def _input(sensing: dict) -> tuple:
    _, values, axes, descriptor = _select(sensing, "delay_domain_slow_time")
    if values.shape[0] != 1 or values.nbytes > MAX_INPUT_BYTES:
        raise ValueError(
            f"Input inspection requires one pair and at most {MAX_INPUT_BYTES // 2**20} MiB "
            "of samples."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Retained complex samples contain nonfinite values.")
    return values[0], axes, descriptor


def _power_input(sensing: dict, mode: str) -> ProcessingInspection:
    """Inspect an authored combined power product without inventing its phase."""
    frequency = mode == "frequency"
    _, values, axes, descriptor = select_product_by_contract(
        sensing,
        kind="baseband_frequency_power" if frequency else "delay_domain_power",
        variant="primary",
        axis_names=("pair", "slow_time", "frequency" if frequency else "range"),
        representations=("power_linear",),
    )
    if values.shape[0] != 1 or values.nbytes > MAX_INPUT_BYTES:
        raise ValueError(
            f"Power inspection requires one pair and at most {MAX_INPUT_BYTES // 2**20} MiB"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Saved input power must be finite and nonnegative")
    power = _frozen(values[0][None])
    return ProcessingInspection(
        mode,
        sample_power=power,
        times=_frozen(axes[1]),
        frequencies=_frozen(axes[2]) if frequency else None,
        ranges=None if frequency else _frozen(axes[2]),
        elements=("Recorded receiver combination",),
        default_coordinate=int(np.argmax(power.sum(axis=(0, 2) if frequency else (0, 1)))),
        **_context(sensing, descriptor),
        notice="Saved receiver-combined power · phase was not retained",
        details=f"Product {descriptor['product_id']}. Receiver combination: {descriptor.get('combine')}. "
        "The numeric display product preserves the recorded power; antenna and phase processing require complex observations.",
    )


def _frequency_delay_input(sensing: dict) -> ProcessingInspection:
    """Reconstruct delay samples on the reader worker from the retained CFR."""
    from shared.sensing.cfr_replay import reconstruct_delay_observation

    _, _, _, source = select_product_by_contract(
        sensing,
        kind="baseband_frequency_observation",
        variant="primary",
        axis_names=("pair", "rx_element", "slow_time", "frequency"),
        representations=("complex_voltage",),
    )
    _preflight_frequency_reconstruction(sensing, source)
    rebuilt = reconstruct_delay_observation(sensing)
    values, axes, descriptor = _input(rebuilt)
    samples = _frozen(values)
    return ProcessingInspection(
        "inputs",
        samples=samples,
        times=_frozen(axes[2]),
        ranges=_frozen(axes[3]),
        elements=tuple(f"Element {i}" for i in range(samples.shape[0])),
        default_coordinate=int(np.argmax(np.sum(np.abs(samples) ** 2, axis=(0, 1)))),
        **_context(sensing, descriptor),
        reconstructed=True,
        notice="CIR reconstructed from saved frequency observations",
        details=f"Source product {source['product_id']}. Uses the recorded frequency window and range transform; no detector or tracker is rerun.",
    )


def _reconstruct(sensing: dict, primary: tuple) -> tuple[np.ndarray, str]:
    _, saved, axes, descriptor = primary
    kinds = {d.get("kind") for d in sensing.get("products", {}).values()}
    if "delay_domain_slow_time" not in kinds and "baseband_frequency_observation" in kinds:
        from shared.sensing.cfr_replay import (
            rebuild_frequency_observation,
            select_frequency_observation,
        )

        _, _, _, source = select_frequency_observation(sensing)
        _preflight_frequency_reconstruction(sensing, source, include_doppler=True)
        if source["pair_ids"] != descriptor["pair_ids"] or any(
            source["lineage"].get(key) != descriptor["lineage"].get(key)
            for key in ("acquisition_id", "processing_run_id")
        ):
            raise ValueError("Retained frequency input belongs to a different result")
        rebuilt = rebuild_frequency_observation(sensing)
        if not np.array_equal(rebuilt["range_axis_m"], axes[2]) or not np.array_equal(
            rebuilt["doppler_axis_hz"], axes[1]
        ):
            raise ValueError("Retained frequency recipe does not reproduce the saved grid")
        actual = rebuilt["rd_cube"]
        if descriptor["representation"] == "power_linear":
            actual = np.abs(actual) ** 2
        if not np.allclose(actual, saved[0], rtol=1e-9, atol=0):
            raise ValueError("Retained frequency recipe does not reproduce the saved spectrum")
        return (
            _frozen(np.abs(rebuilt["rd_raw"]) ** 2),
            f"Verified against the saved primary; source frequency product {source['product_id']}.",
        )
    samples, input_axes, input_descriptor = _input(sensing)
    acquisition_samples = sensing.get("temporal", {}).get("acquisition", {}).get("samples", [])
    if any(sample.get("valid") is False for sample in acquisition_samples):
        raise ValueError("Reconstruction requires valid acquisition samples throughout the window.")
    _same_event(descriptor, input_descriptor)
    if input_descriptor["product_id"] not in descriptor["lineage"]["source_product_ids"]:
        raise ValueError("The retained input is not a declared parent of this spectrum.")
    recipe = descriptor.get("processing", {})
    if any(recipe.get(k) != v for k, v in _RECIPE.items()):
        raise ValueError("The recorded spectral recipe is not supported for reconstruction.")
    if descriptor["representation"] != "complex_voltage":
        raise ValueError("Reconstruction requires a complex primary spectrum for verification.")
    applied = recipe.get("slow_time_mean_removal_applied")
    if type(applied) is not bool:
        raise ValueError("The recorded mean-removal decision is missing.")
    windows = (recipe.get("delay_taper"), recipe.get("doppler_taper"))
    if any(
        not isinstance(w, str) or w.lower().replace("-", "").replace("_", "") not in _WINDOWS
        for w in windows
    ):
        raise ValueError("The recorded taper is missing or unsupported.")
    combine = descriptor.get("combine", {})
    if combine.get("policy") == "siso_identity" and samples.shape[0] == 1:
        combined = samples[0]
    elif combine == {
        "policy": "rx_broadside_unitary_voltage_sum",
        "input_axis": "rx_element",
        "normalization": "one_over_sqrt_element_count",
    }:
        combined = samples.sum(axis=0) / np.sqrt(samples.shape[0])
    else:
        raise ValueError("The recorded receiver combination is unsupported.")
    if combined.shape != saved.shape[1:] or not np.array_equal(input_axes[-1], axes[-1]):
        raise ValueError("Retained samples and the spectrum have different bin grids.")
    times = np.asarray(input_axes[2], dtype=float)
    cfg = sensing["config_resolved"]
    if len(times) < 2 or not np.allclose(np.diff(times), 1 / cfg["prf_hz"], rtol=1e-7, atol=1e-12):
        raise ValueError("Reconstruction requires contiguous samples at the recorded PRF.")
    settings = dict(
        sampling_frequency=cfg["sampling_frequency"],
        prf_hz=cfg["prf_hz"],
        fft_size_doppler=combined.shape[0],
        fft_size_range=combined.shape[1],
        window_range=windows[0],
        window_doppler=windows[1],
        range_mode=cfg["range_mode"],
    )
    raw = compute_range_doppler(fast_time_sequence=combined, **settings)
    processed_input = combined
    if applied:
        if recipe.get("slow_time_mean_removal_policy") == "window_weighted_mean":
            from shared.sensing.signal_processing import remove_window_weighted_mean, sensing_window

            processed_input = remove_window_weighted_mean(
                combined, sensing_window(windows[1], combined.shape[0]), axis=0
            )
        elif recipe.get("slow_time_mean_removal_policy") in (None, "when_row_variation_nonzero"):
            processed_input = combined - combined.mean(axis=0, keepdims=True)
        else:
            raise ValueError("The recorded mean-removal policy is unsupported.")
    processed = compute_range_doppler(fast_time_sequence=processed_input, **settings)
    if not np.allclose(raw.doppler_axis_hz, axes[1], rtol=1e-9, atol=1e-9) or not np.allclose(
        raw.range_axis_m, axes[2], rtol=1e-9, atol=1e-9
    ):
        raise ValueError("The declared recipe does not reproduce the saved coordinate axes.")
    error = np.linalg.norm(processed.cube - saved[0]) / max(np.linalg.norm(saved[0]), 1e-30)
    if error > 1e-6:
        raise ValueError(
            f"Reconstruction does not reproduce the saved primary spectrum (relative error {error:.3g})."
        )
    return (
        _frozen(np.abs(raw.cube) ** 2),
        f"Verified against saved complex primary; relative error {error:.3g}. Input product {input_descriptor['product_id']}.",
    )


def inspect_processing(
    sensing: dict, mode: str, *, reconstruct: bool = False
) -> ProcessingInspection:
    """Build selected-view data on the background reader; never alter saved arrays."""
    kinds = {d.get("kind") for d in sensing.get("products", {}).values()}
    power_kind = "baseband_frequency_power" if mode == "frequency" else "delay_domain_power"
    complex_kind = (
        "baseband_frequency_observation" if mode == "frequency" else "delay_domain_slow_time"
    )
    if mode in {"inputs", "frequency"} and complex_kind not in kinds:
        try:
            if mode == "inputs" and "baseband_frequency_observation" in kinds:
                return _frequency_delay_input(sensing)
            if power_kind in kinds:
                return _power_input(sensing, mode)
        except (KeyError, ValueError) as exc:
            return ProcessingInspection(mode, notice=f"Input inspection unavailable: {exc}")
    if mode == "frequency":
        try:
            from shared.sensing.cfr_replay import select_frequency_observation

            _, values, axes, descriptor = select_frequency_observation(sensing)
            preflight_input(descriptor)
            if not np.all(np.isfinite(values)):
                raise ValueError("Retained complex samples contain nonfinite values.")
            return ProcessingInspection(
                mode,
                samples=_frozen(values[0]),
                default_coordinate=int(np.argmax(np.sum(np.abs(values[0]) ** 2, axis=(0, 2)))),
                times=_frozen(axes[2]),
                frequencies=_frozen(axes[3]),
                elements=tuple(f"Element {i}" for i in range(values.shape[1])),
                **_context(sensing, descriptor),
                notice="Saved unwindowed CFR",
                details=f"Product {descriptor['product_id']}. Magnitude is in relative channel units; phase below 1e-10 of the selected peak is masked. Change frequency window or IFFT size through an explicit orchav-sensing processing run.",
            )
        except (KeyError, ValueError) as exc:
            return ProcessingInspection(mode, notice=f"CFR inspection unavailable: {exc}")
    if mode == "inputs":
        try:
            samples, axes, descriptor = _input(sensing)
            elements = tuple(
                f"Element {v.get('element_index', i)}" if isinstance(v, dict) else f"Element {i}"
                for i, v in enumerate(axes[1])
            )
            return ProcessingInspection(
                mode,
                ranges=_frozen(axes[-1]),
                samples=_frozen(samples),
                default_coordinate=int(np.argmax(np.sum(np.abs(samples) ** 2, axis=(0, 1)))),
                times=_frozen(axes[2]),
                elements=elements,
                **_context(sensing, descriptor),
                notice="Saved CIR samples",
                details=f"Input product {descriptor['product_id']}. Scale: {descriptor['scale_semantics']}. These are simulated CIR-domain samples, not ADC samples. Phase is wrapped; it is undefined at zero magnitude. "
                + f"Phase contract: {descriptor.get('processing', {}).get('phase_contract', sensing.get('config_resolved', {}).get('acquisition', {}).get('phase_contract', 'not recorded'))}.",
            )
        except (KeyError, ValueError) as exc:
            return ProcessingInspection(mode, notice=f"Input samples unavailable: {exc}")
    try:
        primary = _select(sensing, "range_doppler")
        _, values, axes, descriptor = primary
        if values.shape[0] != 1:
            raise ValueError("Comparison requires one selected pair.")
        after = _power(values, descriptor)
        recipe = descriptor.get("processing", {})
        applied = recipe.get("slow_time_mean_removal_applied")
        if type(applied) is not bool:
            raise ValueError("This recording does not declare a mean-removal comparison.")
        if not applied:
            return ProcessingInspection(
                mode,
                notice="Clutter suppression off",
            )
        base = dict(
            mode=mode,
            ranges=_frozen(axes[2]),
            doppler=_frozen(axes[1]),
            after=after,
            **_context(sensing, descriptor),
        )
        details = f"Primary product {descriptor['product_id']}. Receiver combination: {descriptor.get('combine')}. Delay taper: {recipe.get('delay_taper')}; Doppler taper: {recipe.get('doppler_taper')}. Mean removal applied: {applied}. No detector or tracker is rerun."
        try:
            _, raw, raw_axes, raw_descriptor = _select(sensing, "range_doppler", variant="raw")
        except KeyError:
            retained = any(
                d.get("kind") in {"delay_domain_slow_time", "baseband_frequency_observation"}
                for d in sensing["products"].values()
            )
            if reconstruct:
                before, verification = _reconstruct(sensing, primary)
                return ProcessingInspection(
                    **base,
                    before=before,
                    reconstructed=True,
                    notice="Before: reconstructed · After: saved",
                    details=details + " " + verification,
                )
            return ProcessingInspection(
                **base,
                can_reconstruct=retained,
                notice="Before-filtering spectrum not recorded."
                + (" Reconstruction available." if retained else " Input samples unavailable."),
                details=details,
            )
        _same_event(descriptor, raw_descriptor)
        if (
            descriptor["lineage"]["source_product_ids"]
            != raw_descriptor["lineage"]["source_product_ids"]
        ):
            raise ValueError("Saved spectra have different source products.")
        if any(not np.array_equal(a, b) for a, b in zip(axes[1:], raw_axes[1:], strict=True)):
            raise ValueError("Saved spectra have different coordinate grids.")
        raw_recipe = raw_descriptor.get("processing", {})

        def matching_recipe(value: dict) -> dict:
            return {
                key: item for key, item in value.items() if key != "slow_time_mean_removal_applied"
            }

        if raw_recipe.get("slow_time_mean_removal_applied") is not False or (
            matching_recipe(recipe) != matching_recipe(raw_recipe)
        ):
            raise ValueError("Saved variants do not declare matching before/after recipes.")
        return ProcessingInspection(
            **base,
            before=_power(raw, raw_descriptor),
            notice="Saved before / after spectra",
            details=details + f" Before product {raw_descriptor['product_id']}.",
        )
    except (KeyError, ValueError) as exc:
        return ProcessingInspection(mode, notice=f"Comparison unavailable: {exc}")
