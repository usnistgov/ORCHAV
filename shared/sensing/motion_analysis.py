"""Reproducible motion spectra from retained, untapered complex observations.

This optional sensing analysis never invokes a detector or ray tracer. Native
schema-v4 CIR samples supply their own physical epochs; saved FFT products are
not relabeled as raw observations. UI, CLI, and article exports share this path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SCHEMA = "orchav.motion_comparison.v1"
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_RECORDINGS = 16


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class MotionRecipe:
    """Inclusive physical gate/interval and explicit finite-window STFT recipe."""

    range_min_m: float = 0.0
    range_max_m: float = 100.0
    time_start_s: float | None = None
    time_end_s: float | None = None
    window_samples: int = 128
    hop_samples: int = 32
    window: str = "hann"
    mean: str = "both"

    def __post_init__(self):
        if (
            not np.isfinite([self.range_min_m, self.range_max_m]).all()
            or self.range_min_m > self.range_max_m
        ):
            raise ValueError("Range limits must be finite and ordered")
        for value in (self.time_start_s, self.time_end_s):
            if value is not None and not np.isfinite(value):
                raise ValueError("Time limits must be finite or omitted")
        if (
            self.time_start_s is not None
            and self.time_end_s is not None
            and self.time_start_s > self.time_end_s
        ):
            raise ValueError("Time limits must be ordered")
        if type(self.window_samples) is not int or not 3 <= self.window_samples <= 65536:
            raise ValueError("Window length must be an integer from 3 to 65536")
        if type(self.hop_samples) is not int or not 1 <= self.hop_samples <= self.window_samples:
            raise ValueError("Hop must be a positive integer no larger than the window")
        if self.window not in {"hann", "boxcar"} or self.mean not in {
            "retained",
            "removed",
            "both",
        }:
            raise ValueError("Unsupported window or mean policy")


def compute_motion_spectra(samples, times_s, ranges_m, recipe: MotionRecipe) -> dict:
    """Sum delay-bin PSDs after a temporal FFT, without a delay taper.

    Arithmetic complex mean removal is per delay bin and per STFT window,
    before a symmetric Hann (NumPy hanning) or boxcar temporal window. Missing
    epochs/invalid samples are refused, while a valid all-zero channel remains
    a valid zero-power observation. No FFT bridges missing acquisition epochs.
    """
    x, times, ranges = (
        np.asarray(samples),
        np.asarray(times_s, dtype=float),
        np.asarray(ranges_m, dtype=float),
    )
    if x.ndim != 2 or x.dtype.kind != "c" or x.shape != (times.size, ranges.size):
        raise ValueError("Complex samples must have shape [physical time, range]")
    if x.nbytes > MAX_INPUT_BYTES:
        raise ValueError("Selected samples exceed the 256 MiB analysis limit")
    if (
        times.ndim != 1
        or ranges.ndim != 1
        or not np.isfinite(times).all()
        or not np.isfinite(ranges).all()
        or not np.isfinite(x).all()
    ):
        raise ValueError("Samples and coordinates must be finite")
    if len(times) < 2 or np.any(np.diff(times) <= 0) or np.any(np.diff(ranges) <= 0):
        raise ValueError("Physical sample epochs and range coordinates must increase")
    selected = np.ones(times.size, dtype=bool)
    if recipe.time_start_s is not None:
        selected &= times >= recipe.time_start_s
    if recipe.time_end_s is not None:
        selected &= times <= recipe.time_end_s
    gate = (ranges >= recipe.range_min_m) & (ranges <= recipe.range_max_m)
    times = times[selected]
    if not gate.any() or len(times) < recipe.window_samples:
        raise ValueError("Selection needs at least one range bin and one complete temporal window")
    delta = np.diff(times)
    dt = float(np.median(delta))
    if not np.allclose(delta, dt, rtol=1e-7, atol=1e-12):
        raise ValueError("STFT requires uniform physical epochs without gaps")
    x = x[selected][:, gate]
    window = (
        np.hanning(recipe.window_samples)
        if recipe.window == "hann"
        else np.ones(recipe.window_samples)
    )
    starts = np.arange(0, len(times) - recipe.window_samples + 1, recipe.hop_samples)
    modes = ("retained", "removed") if recipe.mean == "both" else (recipe.mean,)
    if recipe.window_samples * len(starts) * len(modes) * 8 > MAX_INPUT_BYTES:
        raise ValueError(
            "Requested spectral output exceeds 256 MiB; shorten the interval or increase hop"
        )
    spectra = {mode: np.empty((recipe.window_samples, len(starts))) for mode in modes}
    for column, start in enumerate(starts):
        raw = x[start : start + recipe.window_samples]
        for mode in modes:
            centered = raw - raw[:1]
            values = centered - centered.mean(axis=0, keepdims=True) if mode == "removed" else raw
            fft = np.fft.fftshift(np.fft.fft(values * window[:, None], axis=0), axes=0)
            spectra[mode][:, column] = np.sum(abs(fft) ** 2, axis=1) * dt / np.sum(window**2)
    loss = None
    if len(modes) == 2:
        numerator, denominator = spectra["retained"].sum(), spectra["removed"].sum()
        if numerator > 0 and denominator > 0:
            loss = float(10 * np.log10(numerator / denominator))
    return {
        "time_s": (times[starts] + times[starts + recipe.window_samples - 1]) / 2,
        "doppler_hz": np.fft.fftshift(np.fft.fftfreq(recipe.window_samples, dt)),
        "range_m": ranges[gate],
        **{f"{mode}_psd": value for mode, value in spectra.items()},
        "metadata": {
            "recipe": asdict(recipe),
            "sample_interval_s": dt,
            "sample_rate_hz": 1 / dt,
            "sample_count": len(times),
            "source_interval_s": [float(times[0]), float(times[-1])],
            "window_count": len(starts),
            "delay_taper": "none",
            "hann_convention": "symmetric",
            "mean_operation": "Per-window per-delay arithmetic complex mean, before temporal window",
            "psd_definition": "sum_range(abs(FFT(x*window))**2)/(fs*sum(window**2))",
            "power_unit": "unit-input channel power per Hz",
            "power_reference": "relative_unitless",
            "mean_removal_loss_db": loss,
            "metric_frequency_support": "all DFT bins",
        },
    }


def load_motion_recording(source: Path, progress: Callable[[str], None] | None = None) -> tuple:
    """Select native CIR products through the existing packed-frame provider.

    The supported route is a continuous sequential acquisition of raw CIRs.
    Independent coherent CPIs cannot be concatenated under this contract. A
    product's recorded combine policy remains authoritative for receive arrays.
    """
    from shared.frames.contracts import FrameComponent, FrameReadRequest
    from shared.frames.manifest import load_frame_manifest
    from shared.frames.providers import Hdf5Provider
    from shared.sensing.products import select_product_by_contract

    source = Path(source).resolve()
    if not (source / "frames_manifest.json").is_file():
        source = source / "frames"
    manifest = load_frame_manifest(source)
    if not manifest.frame_ids:
        raise ValueError("Recording contains no observation frames")
    provider = Hdf5Provider(source.parent, frames_subdir=source.name)
    rows, times, product_ids, contexts, paths = [], [], [], set(), {}
    reference = None
    size = 0
    try:
        probe = provider.load_frame_projection(
            manifest.frame_ids[0], FrameReadRequest(sensing_products=frozenset({"products"}))
        )
        descriptors = (probe.frame.sensing or {}).get("products", {})
        cir = [
            d
            for d in descriptors.values()
            if d.get("kind") == "delay_domain_cir" and d.get("variant") == "primary"
        ]
        if len(cir) != 1:
            if any(
                d.get("kind") in {"delay_domain_slow_time", "baseband_frequency_observation"}
                for d in descriptors.values()
            ):
                return _load_canonical_motion_recording(source, manifest, provider, progress)
            raise ValueError(
                "Untapered typed CIR is unavailable; a saved raw spectrum cannot substitute for it"
            )
        requested = {cir[0]["data_key"], "products", "config_resolved", "temporal"}
        requested.update(axis["coordinate_key"] for axis in cir[0]["axes"])
        request = FrameReadRequest(
            components=frozenset({FrameComponent.SENSING}), sensing_products=frozenset(requested)
        )
        for index, projection in enumerate(
            provider.iter_frame_projections(manifest.frame_ids, request)
        ):
            sensing = dict(projection.frame.sensing or {})
            config = sensing.get("config_resolved", {})
            if config.get("processing_mode") != "sequential_stft":
                raise ValueError(
                    "Cross-record motion spectra require sequential_stft CIR acquisition"
                )
            _, values, axes, descriptor = select_product_by_contract(
                sensing,
                kind="delay_domain_cir",
                variant="primary",
                axis_names=("pair", "rx_element", "range"),
                representations=("complex_voltage",),
            )
            if values.shape[0] != 1 or not np.isfinite(values).all():
                raise ValueError("Motion comparison requires one valid sensor pair")
            if set(descriptor.get("processing", {})) != {"delay_projection"}:
                raise ValueError("CIR processing is not the supported untapered delay projection")
            combine = descriptor["combine"]
            if (
                combine.get("policy") == "siso_identity"
                and values.shape[1] == 1
                and combine.get("normalization") == "identity"
            ):
                row = values[0, 0]
            elif combine == {
                "policy": "rx_broadside_unitary_voltage_sum",
                "input_axis": "rx_element",
                "normalization": "one_over_sqrt_element_count",
            }:
                row = values[0].sum(axis=0) / np.sqrt(values.shape[1])
            else:
                raise ValueError("Unsupported declared receiver combination")
            span = descriptor["time_span"]
            timestamp = span.get("reference_time_s")
            acquisition = sensing.get("temporal", {}).get("acquisition", {})
            matches = [
                s
                for s in acquisition.get("samples", [])
                if s.get("sample_time_s") == timestamp and s.get("valid") is True
            ]
            if (
                len(matches) != 1
                or span.get("start_time_s") != timestamp
                or span.get("end_time_s") != timestamp
            ):
                raise ValueError("CIR sample must bind exactly one valid acquisition epoch")
            context = (
                span.get("clock_id"),
                tuple(descriptor["pair_ids"]),
                json.dumps(combine, sort_keys=True),
                json.dumps(axes[1], sort_keys=True),
                config.get("carrier_freq_hz"),
                config.get("range_mode"),
                descriptor["scale_semantics"],
                json.dumps(descriptor["processing"], sort_keys=True),
            )
            contexts.add(context)
            if len(contexts) != 1:
                raise ValueError(
                    "Recording changes its clock, sensor pair, array, or physical signal contract"
                )
            if reference is None:
                reference = np.asarray(axes[-1])
            elif not np.array_equal(reference, axes[-1]):
                raise ValueError("Recording changes its exact range coordinates")
            size += row.nbytes
            if size > MAX_INPUT_BYTES:
                raise ValueError("CIR recording exceeds the 256 MiB analysis limit")
            rows.append(np.array(row, copy=True))
            times.append(timestamp)
            product_ids.append(descriptor["product_id"])
            if progress and index % 128 == 0:
                progress(f"{source.parent.name}: {index + 1}/{len(manifest.frame_ids)} CIR samples")
        if not rows:
            raise ValueError("No raw CIR observations are retained")
        for chunk in manifest.chunks:
            path = source / chunk.file
            paths[str(path)] = file_identity(path)
        return (
            np.asarray(rows),
            np.asarray(times),
            reference,
            {
                "frames_directory": str(source),
                "generation_id": manifest.generation_id,
                "frame_set_id": manifest.frame_set_id,
                "manifest": file_identity(source / "frames_manifest.json"),
                "native_files": list(paths.values()),
                "product_ids_sha256": hashlib.sha256("\n".join(product_ids).encode()).hexdigest(),
                "product_count": len(product_ids),
                "pair_ids": descriptor["pair_ids"],
                "combine": combine,
                "carrier_hz": config["carrier_freq_hz"],
                "range_mode": config["range_mode"],
                "scale_semantics": descriptor["scale_semantics"],
                "source_kind": "delay_domain_cir",
                "recorded_delay_taper_for_fft_products": config.get("delay_taper"),
                "acquisition_phase_contract": config.get("acquisition", {}).get("phase_contract"),
            },
        )
    finally:
        provider.close()


def _partial_canonical_delay_observation(sensing):
    """Read one validated SISO warm-up pulse for explicit motion analysis only."""
    from shared.sensing.delay import DelayDomainSamples
    from shared.sensing.products import select_product_by_contract

    key, values, axes, descriptor = select_product_by_contract(
        sensing,
        kind="delay_domain_slow_time",
        variant="primary",
        axis_names=("pair", "rx_element", "slow_time", "range"),
        representations=("complex_voltage",),
    )
    acquisition = sensing["temporal"]["acquisition"]
    valid = [s for s in acquisition["samples"] if s.get("valid") is True]
    if not valid:
        raise ValueError("Partial observation has no acquired sample")
    sample = valid[-1]
    support = {
        "type": "acquisition_sample",
        "selection_policy": "last_valid_acquisition_sample",
        "sample_index": sample["sample_index"],
    }
    timestamp = sample["sample_time_s"]
    span = descriptor["time_span"]
    if (
        values.shape[:3] != (1, 1, 1)
        or not np.iscomplexobj(values)
        or not np.isfinite(values).all()
        or tuple(a.get("unit") for a in descriptor["axes"]) != ("1", "1", "s", "m")
        or descriptor.get("temporal_support") != support
        or not np.array_equal(axes[2], [timestamp])
        or any(span.get(k) != timestamp for k in ("start_time_s", "end_time_s", "reference_time_s"))
        or descriptor.get("combine") != {"policy": "siso_identity", "normalization": "identity"}
        or axes[1] != [{"element_index": 0, "row": 0, "column": 0}]
        or descriptor.get("pair_ids") != [acquisition["pair_id"]]
    ):
        raise ValueError("Partial delay observation must bind the exact last acquired SISO pulse")
    return DelayDomainSamples(
        data_key=key,
        kind="delay_domain_slow_time",
        samples=values[0, 0].copy(),
        sample_times_s=np.asarray(axes[2]).copy(),
        range_axis_m=np.asarray(axes[-1]).copy(),
        pair_id=acquisition["pair_id"],
        rx_element_identities=((0, 0, 0),),
        combine_policy="siso_identity",
    )


def _load_canonical_motion_recording(source, manifest, provider, progress):
    """Read one retained complex representation through explicit analysis only."""
    # Imports stay local: the generator uses the numeric MotionRecipe/kernel
    # without importing providers or creating a dependency cycle.
    from shared.frames.contracts import FrameComponent, FrameReadRequest
    from shared.sensing.cfr_replay import reconstruct_delay_observation
    from shared.sensing.delay import (
        append_delay_domain_samples,
        extract_delay_domain_samples,
        require_compatible_delay_sequence,
    )

    probe_request = FrameReadRequest(
        components=frozenset({FrameComponent.SENSING}), sensing_products=frozenset({"products"})
    )
    rows, times, ids = [], [], []
    reference, context = None, None
    size = 0
    for index, frame_id in enumerate(manifest.frame_ids):
        metadata = provider.load_frame_projection(frame_id, probe_request).frame.sensing or {}
        config = metadata.get("config_resolved", {})
        if config.get("processing_mode") != "sequential_stft":
            raise ValueError("Cross-record motion spectra require sequential_stft CIR acquisition")
        # Canonical warm-up products preserve real observations even when
        # no configured Doppler window completed. Overlapping epochs are checked
        # and deduplicated with the same sequence logic as complete windows.
        descriptors = metadata.get("products", {})
        candidates = [
            d
            for d in descriptors.values()
            if d.get("kind") == "delay_domain_slow_time" and d.get("variant") == "primary"
        ]
        if not candidates:
            candidates = [
                d
                for d in descriptors.values()
                if d.get("kind") == "baseband_frequency_observation"
                and d.get("variant") == "primary"
            ]
        if len(candidates) != 1:
            raise ValueError("Recording does not retain one canonical complex observation")
        original = candidates[0]
        requested = {original["data_key"], "products", "config_resolved", "temporal"}
        requested.update(axis["coordinate_key"] for axis in original["axes"])
        sensing = dict(
            provider.load_frame_projection(
                frame_id,
                FrameReadRequest(
                    components=frozenset({FrameComponent.SENSING}),
                    sensing_products=frozenset(requested),
                ),
            ).frame.sensing
            or {}
        )
        kind = original["kind"]
        if kind == "baseband_frequency_observation":
            sensing = reconstruct_delay_observation(sensing)
        product = (
            extract_delay_domain_samples(sensing)
            if isinstance(sensing.get("temporal", {}).get("result"), dict)
            else _partial_canonical_delay_observation(sensing)
        )
        if product is None:
            raise ValueError("Canonical complex observation cannot supply typed delay samples")
        current_context = (
            sensing["temporal"]["clock"]["clock_id"],
            config.get("carrier_freq_hz"),
            config.get("range_mode"),
            original["scale_semantics"],
            kind,
            json.dumps(config.get("range_response"), sort_keys=True),
        )
        if context is None:
            context, reference = current_context, product
        elif current_context != context:
            raise ValueError("Recording changes its clock, carrier, or physical signal contract")
        require_compatible_delay_sequence(reference, product)
        added = append_delay_domain_samples(product, rows, times)
        size += added * product.samples.shape[1] * product.samples.dtype.itemsize
        if size > MAX_INPUT_BYTES:
            raise ValueError("CIR recording exceeds the 256 MiB analysis limit")
        ids.append(original["product_id"])
        if progress and index % 128 == 0:
            progress(f"{source.parent.name}: {len(rows)} canonical complex samples")
    if reference is None or not rows:
        raise ValueError("No canonical complex observations are retained")
    return (
        np.asarray(rows),
        np.asarray(times),
        reference.range_axis_m,
        {
            "frames_directory": str(source),
            "generation_id": manifest.generation_id,
            "frame_set_id": manifest.frame_set_id,
            "manifest": file_identity(source / "frames_manifest.json"),
            "native_files": [file_identity(source / chunk.file) for chunk in manifest.chunks],
            "product_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "product_count": len(ids),
            "pair_ids": original["pair_ids"],
            "combine": original["combine"],
            "carrier_hz": config["carrier_freq_hz"],
            "range_mode": config["range_mode"],
            "scale_semantics": original["scale_semantics"],
            "source_kind": kind,
            "delay_reconstructed": kind == "baseband_frequency_observation",
            "recorded_delay_taper_for_fft_products": config.get("delay_taper"),
            "acquisition_phase_contract": config.get("acquisition", {}).get("phase_contract"),
        },
    )


def load_definition(path: Path) -> dict:
    """Resolve explicit file references relative to a saved comparison definition."""
    path = Path(path).resolve()
    definition = json.loads(path.read_text())
    if definition.get("schema") != SCHEMA:
        raise ValueError("Unsupported motion comparison schema")
    MotionRecipe(**definition.get("recipe", {}))
    if definition.get("display", {}).get("normalization", "absolute") not in {
        "absolute",
        "per_record_peak",
    }:
        raise ValueError("Unsupported power normalization")
    for source in definition.get("sources", []):
        for key in ("frames", "root_motion", "motion_reference"):
            if source.get(key):
                source[key] = str((path.parent / source[key]).resolve())
    return definition


def analyze_comparison(definition: dict, progress: Callable[[str], None] | None = None) -> dict:
    sources = definition.get("sources", [])
    if (
        not 1 <= len(sources) <= MAX_RECORDINGS
        or len({s.get("label") for s in sources}) != len(sources)
        or any(not s.get("label") for s in sources)
    ):
        raise ValueError("Choose 1–16 distinctly labeled recordings")
    recipe = MotionRecipe(**definition.get("recipe", {}))
    outputs, compatibility, reference_ranges = [], None, None
    for source in sources:
        samples, times, ranges, provenance = load_motion_recording(Path(source["frames"]), progress)
        result = compute_motion_spectra(samples, times, ranges, recipe)
        if reference_ranges is not None and not np.array_equal(reference_ranges, result["range_m"]):
            raise ValueError("Comparison requires the same selected range coordinates")
        reference_ranges = result["range_m"]
        contract = (
            provenance["carrier_hz"],
            provenance["range_mode"],
            provenance["scale_semantics"],
            provenance["combine"],
            result["metadata"]["sample_rate_hz"],
        )
        if compatibility is not None and (
            contract[:4] != compatibility[:4]
            or not np.isclose(contract[4], compatibility[4], rtol=1e-9)
        ):
            raise ValueError(
                "Comparison requires matching carrier, range convention, receiver combination, power reference and cadence"
            )
        compatibility = contract
        result["label"], result["provenance"] = source["label"], provenance
        if source.get("root_motion"):
            path = Path(source["root_motion"])
            root = json.loads(path.read_text())
            rt, xyz = np.asarray(root["time_s"]), np.asarray(root["root_world_m"])
            if (
                rt.ndim != 1
                or xyz.shape != (rt.size, 3)
                or not np.isfinite(xyz).all()
                or not np.isfinite(rt).all()
                or np.any(np.diff(rt) <= 0)
            ):
                raise ValueError(
                    "Root reference requires finite increasing time and XYZ coordinates"
                )
            result.update(
                root_time_s=rt,
                root_world_m=xyz,
                root_displacement_m=np.linalg.norm(xyz - xyz[0], axis=1),
            )
            result["provenance"]["root_motion"] = file_identity(path)
            result["provenance"][
                "root_reference_role"
            ] = "Declared external source-motion reference at authored times; not used for spectral selection"
        if source.get("motion_reference"):
            result["provenance"]["motion_reference"] = file_identity(
                Path(source["motion_reference"])
            )
        outputs.append(result)
    return {"schema": SCHEMA, "definition": definition, "results": outputs}


def export_comparison(comparison: dict, destination: Path, *, plots: bool = True) -> Path:
    """Write compact immutable analysis evidence, leaving native sources untouched."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    arrays, records = {}, []
    for index, result in enumerate(comparison["results"]):
        row = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        row["arrays"] = {}
        for name, value in result.items():
            if isinstance(value, np.ndarray):
                key = f"source_{index}_{name}"
                arrays[key] = value
                row["arrays"][name] = key
        records.append(row)
    np.savez_compressed(destination / "spectra.npz", **arrays)
    metadata = {
        "schema": SCHEMA,
        "definition": comparison["definition"],
        "results": records,
        "arrays_sha256": file_identity(destination / "spectra.npz")["sha256"],
    }
    (destination / "comparison.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    (destination / "definition.json").write_text(
        json.dumps(comparison["definition"], indent=2, allow_nan=False) + "\n"
    )
    if plots:
        from shared.sensing.plotting.motion import render_motion_comparison

        render_motion_comparison(comparison, destination)
    return destination


def load_export(path: Path) -> dict:
    """Load authenticated spectra for portable inspection, without raw-CIR claims."""
    path = Path(path)
    meta = json.loads((path / "comparison.json").read_text())
    if (
        meta.get("schema") != SCHEMA
        or file_identity(path / "spectra.npz")["sha256"] != meta["arrays_sha256"]
    ):
        raise ValueError("Changed or unsupported motion export")
    with np.load(path / "spectra.npz", allow_pickle=False) as arrays:
        results = [
            {
                **{k: v for k, v in row.items() if k != "arrays"},
                **{name: np.array(arrays[key]) for name, key in row["arrays"].items()},
            }
            for row in meta["results"]
        ]
    _validate_export_results(meta["definition"], results)
    return {"schema": SCHEMA, "definition": meta["definition"], "results": results}


def _validate_export_results(definition: dict, results: list[dict]) -> None:
    """Bind portable numeric arrays to their declared axes and processing recipe."""
    if definition.get("schema") != SCHEMA:
        raise ValueError("Unsupported exported comparison definition")
    recipe = MotionRecipe(**definition.get("recipe", {}))
    labels = [source["label"] for source in definition.get("sources", [])]
    if (
        not 1 <= len(results) <= MAX_RECORDINGS
        or labels != [r["label"] for r in results]
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Exported recordings do not match the labeled source definition")
    modes = ("retained", "removed") if recipe.mean == "both" else (recipe.mean,)
    for record in results:
        metadata = record["metadata"]
        if (
            metadata.get("power_reference") != "relative_unitless"
            or metadata.get("power_unit") != "unit-input channel power per Hz"
        ):
            raise ValueError("Exported spectral power reference is unsupported")
        if metadata["recipe"] != asdict(recipe) or metadata.get("delay_taper") != "none":
            raise ValueError("Exported processing metadata differs from the declared recipe")
        dt = metadata["sample_interval_s"]
        count = metadata["sample_count"]
        if (
            not isinstance(dt, (int, float))
            or not np.isfinite(dt)
            or dt <= 0
            or type(count) is not int
            or count < recipe.window_samples
        ):
            raise ValueError("Exported physical cadence or sample count is invalid")
        windows = 1 + (count - recipe.window_samples) // recipe.hop_samples
        interval = np.asarray(metadata["source_interval_s"], dtype=float)
        if (
            interval.shape != (2,)
            or not np.isfinite(interval).all()
            or not np.isclose(interval[1] - interval[0], (count - 1) * dt, rtol=1e-9, atol=1e-12)
            or metadata["window_count"] != windows
            or not np.isclose(metadata["sample_rate_hz"] * dt, 1, rtol=1e-9)
        ):
            raise ValueError("Exported temporal support does not match its sample count")
        expected_time = (
            interval[0]
            + (np.arange(windows) * recipe.hop_samples + (recipe.window_samples - 1) / 2) * dt
        )
        expected_frequency = np.fft.fftshift(np.fft.fftfreq(recipe.window_samples, dt))
        for name, expected in (("time_s", expected_time), ("doppler_hz", expected_frequency)):
            actual = record[name]
            if (
                actual.shape != expected.shape
                or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
            ):
                raise ValueError(f"Exported {name} axis differs from the recipe")
        ranges = record["range_m"]
        if (
            ranges.ndim != 1
            or not len(ranges)
            or not np.isfinite(ranges).all()
            or np.any(np.diff(ranges) <= 0)
            or ranges.min() < recipe.range_min_m
            or ranges.max() > recipe.range_max_m
        ):
            raise ValueError("Exported range coordinates differ from the declared selection")
        if {name for name in record if name.endswith("_psd")} != {mode + "_psd" for mode in modes}:
            raise ValueError("Exported mean variants differ from the recipe")
        for mode in modes:
            power = record[mode + "_psd"]
            if (
                power.shape != (recipe.window_samples, windows)
                or power.dtype.kind not in "fi"
                or not np.isfinite(power).all()
                or np.any(power < 0)
            ):
                raise ValueError(
                    "Exported spectral power must be finite, nonnegative and match its axes"
                )
        if len(modes) == 2:
            numerator, denominator = record["retained_psd"].sum(), record["removed_psd"].sum()
            expected_loss = (
                float(10 * np.log10(numerator / denominator))
                if numerator > 0 and denominator > 0
                else None
            )
            loss = metadata["mean_removal_loss_db"]
            if (expected_loss is None and loss is not None) or (
                expected_loss is not None
                and (loss is None or not np.isclose(loss, expected_loss, rtol=1e-10, atol=1e-12))
            ):
                raise ValueError("Exported mean-removal metric disagrees with the spectral arrays")
        root_names = {"root_time_s", "root_world_m", "root_displacement_m"}
        if root_names.intersection(record):
            if not root_names.issubset(record):
                raise ValueError("Exported root-motion reference is incomplete")
            time, xyz, displacement = (
                record[name] for name in ("root_time_s", "root_world_m", "root_displacement_m")
            )
            if (
                time.ndim != 1
                or len(time) < 1
                or xyz.shape != (len(time), 3)
                or displacement.shape != time.shape
                or not np.isfinite(time).all()
                or not np.isfinite(xyz).all()
                or np.any(np.diff(time) <= 0)
                or not np.allclose(
                    displacement, np.linalg.norm(xyz - xyz[0], axis=1), rtol=1e-10, atol=1e-12
                )
            ):
                raise ValueError(
                    "Exported source-root coordinates or displacement are inconsistent"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--render-export",
        action="store_true",
        help="Render saved spectra; no reprocessing or raw observations implied",
    )
    args = parser.parse_args()
    if args.render_export:
        from shared.sensing.plotting.motion import render_motion_comparison

        args.output_dir.mkdir(parents=True, exist_ok=False)
        render_motion_comparison(load_export(args.definition), args.output_dir)
    else:
        export_comparison(analyze_comparison(load_definition(args.definition)), args.output_dir)


if __name__ == "__main__":
    main()
