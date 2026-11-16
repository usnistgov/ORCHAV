"""Compact views of saved primary Doppler products, without signal reprocessing."""

from collections.abc import Mapping, Sequence
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from shared.sensing.plotting.style import BLUE, sensing_figure_style
from shared.sensing.products import select_product_by_contract


def _replay_display_identity(
    sensing: Mapping[str, Any], descriptor: Mapping[str, Any]
) -> str | None:
    """Identify equal replay settings while retaining source-bound recipe IDs.

    Older CFR replay recipes included path counts and observation fingerprints
    from each source CPI. Only those explicitly named evidence fields are
    removed from this display identity; all declared settings remain compared.
    """
    from shared.extensions.sensing import canonical_processing_recipe_id, canonical_sensing_hash

    processing = descriptor.get("processing", {})
    config = sensing.get("config_resolved", {})
    recipe = config.get("processing_recipe", {})
    if (
        processing.get("tool") != "shared.sensing_postprocessor"
        or processing.get("input_kind") != "cfr_by_pair_rx_element"
        or recipe.get("tool") != processing["tool"]
        or recipe.get("input_kind") != processing["input_kind"]
    ):
        return None
    recipe_id = descriptor.get("lineage", {}).get("processing_recipe_id")
    if (
        processing.get("recipe_id") != recipe_id
        or canonical_processing_recipe_id(recipe) != recipe_id
    ):
        raise ValueError("Saved replay descriptor and declared processing recipe do not agree")
    declared = deepcopy(config)
    inherited = declared["processing_recipe"].get("inherited_numeric_model", {})
    for response in (
        declared.get("range_response_processing"),
        inherited.get("range_response_processing"),
    ):
        if not isinstance(response, dict):
            raise ValueError("Saved CFR replay requires its declared range-response processing")
        for key in (
            "source_product_id",
            "contributing_element_path_count",
            "discarded_nonfinite_paths_before_aoa_selection",
            "invalid_entry_count",
            "nonfinite_path_count",
        ):
            response.pop(key, None)
        noise = response.get("noise", {})
        for key in ("observation_sha256", "realization_index"):
            noise.pop(key, None)
    return canonical_sensing_hash(
        {
            "config_resolved": declared,
            "descriptor": {
                key: descriptor.get(key) for key in ("unit", "scale_semantics", "combine")
            },
            "producer": {key: value for key, value in processing.items() if key != "recipe_id"},
        }
    )


def doppler_processing_matches(first: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    """Accept the same recipe or independently verified equal replay settings."""
    return first.get("processing_recipe_id") == other.get("processing_recipe_id") or (
        first.get("processing_display_id") is not None
        and first["processing_display_id"] == other.get("processing_display_id")
    )


def doppler_summary_event(sensing: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reduce one saved primary RD map to power versus Doppler.

    No taper, mean subtraction, transform or normalization is added. Keeping
    only one real vector per result avoids retaining full complex RD histories.
    Missing products (including acquisition warm-up) produce no event; malformed
    declared products fail visibly. The existing summary supports one sensor pair.
    """
    products = sensing.get("products", {})
    if not any(
        isinstance(p, Mapping)
        and p.get("kind") == "range_doppler"
        and p.get("variant") == "primary"
        for p in products.values()
    ):
        return None
    _, values, coordinates, descriptor = select_product_by_contract(
        sensing,
        kind="range_doppler",
        variant="primary",
        axis_names=("pair", "doppler_frequency", "range"),
        representations=("complex_voltage", "power_linear"),
    )
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError("Saved Doppler summary requires exactly one sensor pair")
    if tuple(a["unit"] for a in descriptor["axes"]) != ("1", "Hz", "m"):
        raise ValueError("Saved Doppler summary requires axes in Hz and m")
    if descriptor["representation"] == "complex_voltage":
        if not np.iscomplexobj(values):
            raise ValueError("complex_voltage requires complex RD samples")
        power = np.abs(values[0]) ** 2
    else:
        if np.iscomplexobj(values):
            raise ValueError("power_linear requires real RD samples")
        power = np.asarray(values[0], dtype=np.float64)
    if not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("Saved RD power must be finite and nonnegative")
    time_s = float(descriptor["time_span"]["reference_time_s"])
    doppler = np.asarray(coordinates[1], dtype=np.float64)
    if not np.isfinite(time_s) or not np.all(np.isfinite(doppler)):
        raise ValueError("Saved Doppler coordinates must be finite")
    if doppler.shape != (power.shape[0],) or np.any(np.diff(doppler) <= 0):
        raise ValueError("Saved Doppler coordinates must match the ordered spectrum")
    return {
        "time_s": time_s,
        "pair_id": descriptor["pair_ids"][0],
        "doppler_axis_hz": doppler.copy(),
        "power_linear": np.sum(power, axis=-1, dtype=np.float64),
        "processing": dict(descriptor.get("processing", {})),
        "processing_recipe_id": descriptor.get("lineage", {}).get("processing_recipe_id"),
        "processing_display_id": _replay_display_identity(sensing, descriptor),
    }


@sensing_figure_style
def plot_doppler_history(
    history: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    axis: str = "doppler",
    carrier_frequency_hz: float | None = None,
    range_mode: str = "monostatic",
    time_reference: str = "result",
    title: str | None = None,
    power_reference: float | None = None,
    dynamic_range_db: float = 80.0,
    time_limits_s: tuple[float, float] | None = None,
    force_map: bool = False,
) -> Path:
    """Render the standard saved spectrum or time–Doppler summary.

    Defaults preserve the ordinary summary. Comparisons can supply a common
    power reference, time limits and calibrated velocity coordinates to render
    separate standard figures without changing any saved signal samples.
    """
    from shared.sensing.plotting.viz import SUMMARY_FIGURE_DPI, plt
    from shared.sensing.signal_processing import (
        detection_velocity_display,
        doppler_axis_to_velocity_m_s,
    )

    if not history:
        raise ValueError("Saved Doppler history is empty")
    first = history[0]
    frequency = np.asarray(first["doppler_axis_hz"])
    if any(
        e["pair_id"] != first["pair_id"]
        or not doppler_processing_matches(first, e)
        or not np.array_equal(e["doppler_axis_hz"], frequency)
        for e in history
    ):
        raise ValueError("Doppler history must share a sensor pair, recipe and frequency axis")
    if axis not in {"doppler", "velocity"} or time_reference not in {"result", "midpoint"}:
        raise ValueError("Choose doppler/velocity and result/midpoint display coordinates")
    coordinate = frequency
    coordinate_label = "Doppler frequency (Hz)"
    if axis == "velocity":
        if (
            carrier_frequency_hz is None
            or not np.isfinite(carrier_frequency_hz)
            or carrier_frequency_hz <= 0
        ):
            raise ValueError("Velocity display requires a finite positive carrier frequency")
        coordinate = doppler_axis_to_velocity_m_s(frequency, carrier_frequency_hz)
        coordinate_label = detection_velocity_display(range_mode).axis_label
    time_key = "midpoint_s" if time_reference == "midpoint" else "time_s"
    times = np.asarray([e[time_key] for e in history], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Saved result times must be strictly increasing")
    power = np.asarray([e["power_linear"] for e in history], dtype=float)
    if (
        power.shape != (len(history), len(frequency))
        or np.any(power < 0)
        or not np.all(np.isfinite(power))
    ):
        raise ValueError("Saved Doppler power must match the history axes")
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0:
        raise ValueError("Display dynamic range must be positive and finite")
    if power_reference is None:
        power_db = np.full(power.shape, np.nan)
        np.log10(power, out=power_db, where=power > 0)
        power_db *= 10.0
        peak_db = float(np.nanmax(power_db)) if np.any(power > 0) else 0.0
        power_label = "Range-summed power (dB)"
    else:
        if not np.isfinite(power_reference) or power_reference < 0:
            raise ValueError("Shared power reference must be finite and nonnegative")
        power_db = np.full(power.shape, -dynamic_range_db, dtype=np.float64)
        if power_reference > 0:
            np.log10(
                np.maximum(power / power_reference, 10 ** (-dynamic_range_db / 10)), out=power_db
            )
            power_db *= 10
        peak_db = 0.0
        power_label = "Range-summed power (dB relative to the common peak)"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    single_spectrum = len(history) == 1 and not force_map
    ax.set_title(title or ("Doppler spectrum" if single_spectrum else "Time–Doppler evolution"))
    if single_spectrum:
        ax.plot(coordinate, power_db[0], linewidth=1.5, color=BLUE)
        ax.set_xlabel(coordinate_label)
        ax.set_ylabel(power_label)
        ax.text(
            0.98,
            0.97,
            f"Result time: {times[0]:g} s",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        ax.grid(alpha=0.25)
        if np.any(power > 0):
            ax.set_ylim(peak_db - dynamic_range_db, peak_db + 3)
    else:
        im = _draw_doppler_mesh(
            ax,
            times,
            coordinate,
            power_db,
            vmin=peak_db - dynamic_range_db,
            vmax=peak_db,
            single_time_width=(
                len(first["sample_times_s"]) * first["sample_interval_s"] if force_map else None
            ),
        )
        ax.set_xlabel(
            "Observation-window midpoint (s)" if time_reference == "midpoint" else "Time (s)"
        )
        ax.set_ylabel(coordinate_label)
        if time_limits_s is not None:
            ax.set_xlim(time_limits_s)
        fig.colorbar(im, ax=ax, label=power_label)
    if not np.any(power > 0):
        ax.text(
            0.5,
            0.5,
            "No nonzero power in the saved product",
            ha="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# Comparisons retain the saved sample epochs. This allowance covers alternate
# floating-point evaluation orders of the same clock, not nearest-time matching.
_TIMESTAMP_ULPS = 8


def _time_tolerance(times: np.ndarray) -> float:
    return _TIMESTAMP_ULPS * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(times))))


def _complete_sample_times(sensing: Mapping[str, Any]) -> np.ndarray:
    acquisition = sensing["temporal"]["acquisition"]
    samples = acquisition["samples"]
    expected = acquisition["expected_sample_count"]
    if (
        acquisition.get("completeness") != "complete"
        or expected < 2
        or len(samples) != expected
        or acquisition.get("observed_sample_count") != expected
        or acquisition.get("valid_sample_count") != expected
        or not all(sample.get("valid") for sample in samples)
    ):
        raise ValueError("Doppler comparison requires complete valid observation windows")
    times = np.asarray([sample["sample_time_s"] for sample in samples], dtype=np.float64)
    intervals = np.diff(times)
    if (
        times.shape != (expected,)
        or not np.all(np.isfinite(times))
        or np.any(intervals <= 0)
        or not np.allclose(intervals, intervals[0], rtol=0, atol=_time_tolerance(times))
    ):
        raise ValueError("Doppler comparison requires uniformly sampled complete windows")
    return times


def _comparison_calibration(sensing: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict:
    config = sensing["config_resolved"]
    carrier = float(config["carrier_freq_hz"])
    mode = config["range_mode"]
    if not np.isfinite(carrier) or carrier <= 0 or mode not in {"monostatic", "bistatic"}:
        raise ValueError(
            "Doppler comparison requires finite carrier and explicit range interpretation"
        )
    if descriptor.get("unit") != "1" or descriptor.get("scale_semantics") != "relative_unitless":
        raise ValueError("Doppler comparison requires the declared relative-unitless power scale")
    processing = descriptor.get("processing", {})
    if _replay_display_identity(sensing, descriptor) is not None:
        processing = config
    combine = dict(descriptor.get("combine", {}))
    if not all(
        isinstance(processing.get(key), str) and processing[key]
        for key in ("range_processing", "fft_normalization", "window_normalization")
    ) or not all(
        isinstance(combine.get(key), str) and combine[key] for key in ("policy", "normalization")
    ):
        raise ValueError(
            "Doppler comparison requires declared signal normalization and combination"
        )
    elements = config["topology"]["rx_elements"]
    # Unitary broadside voltage summation and identity are the same operation
    # for one receive element, as used by coherent and sequential producers.
    if elements == 1 and combine.get("policy") in {
        "siso_identity",
        "rx_broadside_unitary_voltage_sum",
    }:
        combine = {"policy": "single_element_identity", "normalization": "identity"}
    clock = sensing["temporal"]["clock"]
    return {
        "carrier_frequency_hz": carrier,
        "range_mode": mode,
        "unit": descriptor["unit"],
        "scale_semantics": descriptor["scale_semantics"],
        "rx_elements": elements,
        "combine": combine,
        "clock": {key: clock[key] for key in ("domain", "origin", "origin_time_s", "unit")},
        "processing": {
            key: processing.get(key)
            for key in (
                "bandwidth_hz",
                "model",
                "range_processing",
                "delay_reference",
                "range_transform_normalization",
                "fft_normalization",
                "window_normalization",
                "frequency_reference",
                "num_frequency_bins",
            )
        },
    }


def load_doppler_recording(source: Path, label: str) -> dict[str, Any]:
    """Stream a recording into compact saved spectra and complete sample epochs."""
    from shared.sensing.loader import iter_frames_sensing

    events = []
    skipped = 0
    calibration = None
    range_axis = None
    frequency = None
    with closing(iter_frames_sensing(source)) as frames:
        for frame in frames:
            sensing = frame["_typed_sensing"]
            event = doppler_summary_event(sensing)
            if event is None:
                acquisition = sensing.get("temporal", {}).get("acquisition", {})
                if acquisition.get("completeness") == "incomplete_unprocessed":
                    skipped += 1
                    continue
                raise ValueError("Complete acquisition has no saved primary range-Doppler product")
            samples = _complete_sample_times(sensing)
            _, values, coordinates, descriptor = select_product_by_contract(
                sensing,
                kind="range_doppler",
                variant="primary",
                axis_names=("pair", "doppler_frequency", "range"),
                representations=("complex_voltage", "power_linear"),
            )
            current_range = np.asarray(coordinates[2], dtype=np.float64)
            if (
                current_range.shape != (values.shape[-1],)
                or not np.all(np.isfinite(current_range))
                or np.any(np.diff(current_range) <= 0)
            ):
                raise ValueError("Saved range coordinates must match the ordered range samples")
            current_calibration = _comparison_calibration(sensing, descriptor)
            if events and (
                current_calibration != calibration
                or event["pair_id"] != events[0]["pair_id"]
                or not doppler_processing_matches(events[0], event)
                or not np.array_equal(current_range, range_axis)
                or not np.array_equal(event["doppler_axis_hz"], frequency)
                or len(samples) != len(events[0]["sample_times_s"])
                or not np.isclose(
                    samples[1] - samples[0],
                    events[0]["sample_interval_s"],
                    rtol=0,
                    atol=_time_tolerance(samples),
                )
            ):
                raise ValueError(
                    "Recording changed its pair, recipe, calibration, window or coordinates"
                )
            calibration, range_axis, frequency = (
                current_calibration,
                current_range,
                event["doppler_axis_hz"],
            )
            event.update(
                sample_times_s=samples,
                sample_phase_coherent=[
                    sample["phase_coherent"]
                    for sample in sensing["temporal"]["acquisition"]["samples"]
                ],
                sample_interval_s=float(samples[1] - samples[0]),
                midpoint_s=float((samples[0] + samples[-1]) / 2),
                reference_time_policy=descriptor["time_span"]["reference_time_policy"],
                product_id=descriptor["product_id"],
                frame_index=frame["_frame_index"],
            )
            if events and (
                event["time_s"] <= events[-1]["time_s"]
                or samples[0] <= events[-1]["sample_times_s"][0]
            ):
                raise ValueError("Saved result and observation-window times must increase")
            events.append(event)
    if not events:
        raise ValueError(f"No complete saved Doppler observations in {source}")
    return {
        "label": label,
        "source": Path(source),
        "events": events,
        "calibration": calibration,
        "range_axis_m": range_axis,
        "doppler_axis_hz": frequency,
        "skipped_incomplete": skipped,
    }


def validate_doppler_comparison(recordings: Sequence[Mapping[str, Any]]) -> None:
    """Require shared axes and calibrated power/velocity interpretations."""
    if not recordings:
        raise ValueError("At least one recording is required")
    first = recordings[0]
    for recording in recordings[1:]:
        if (
            recording["calibration"] != first["calibration"]
            or not np.array_equal(recording["range_axis_m"], first["range_axis_m"])
            or not np.array_equal(recording["doppler_axis_hz"], first["doppler_axis_hz"])
        ):
            raise ValueError(
                "Recordings have incompatible coordinates or power/velocity calibration"
            )
        a, b = first["events"][0], recording["events"][0]
        if len(a["sample_times_s"]) != len(b["sample_times_s"]) or not np.isclose(
            a["sample_interval_s"],
            b["sample_interval_s"],
            rtol=0,
            atol=max(_time_tolerance(a["sample_times_s"]), _time_tolerance(b["sample_times_s"])),
        ):
            raise ValueError("Recordings have incompatible analysis windows or sample cadence")


def matching_window_indices(
    reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> list[int]:
    """Find the unique window with every sample epoch equal within float roundoff."""
    matches = []
    for event in reference:
        times = np.asarray(event["sample_times_s"])
        found = [
            i
            for i, other in enumerate(candidate)
            if np.shape(other["sample_times_s"]) == times.shape
            and np.allclose(other["sample_times_s"], times, rtol=0, atol=_time_tolerance(times))
        ]
        if len(found) != 1:
            raise ValueError("Each reference window requires one exact complete sample-time match")
        matches.append(found[0])
    return matches


def _doppler_time_edges(times, single_time_width=None):
    from shared.sensing.plotting.viz import _coordinate_edges

    if len(times) == 1 and single_time_width is not None:
        return np.asarray([times[0] - single_time_width / 2, times[0] + single_time_width / 2])
    return _coordinate_edges(times, name="time")


def _draw_doppler_mesh(ax, times, coordinates, power_db, *, vmin, vmax, single_time_width=None):
    """Shared map renderer for ordinary summaries and recording comparisons."""
    from shared.sensing.plotting.viz import _coordinate_edges

    return ax.pcolormesh(
        _doppler_time_edges(times, single_time_width),
        _coordinate_edges(coordinates, name="doppler"),
        power_db.T,
        cmap="viridis",
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )


@sensing_figure_style
def plot_doppler_comparison(
    recordings: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    axis: str = "doppler",
    time_reference: str = "result",
    match_windows: bool = False,
    dynamic_range_db: float = 40.0,
    layout: str = "combined",
) -> dict:
    """Compare saved spectra as combined panels or separate standard summaries.

    ``output_path`` is the image path for the combined layout or an existing
    output directory for the separate layout. No signal is transformed or filtered.
    """
    from shared.sensing.plotting.viz import SUMMARY_FIGURE_DPI, plt
    from shared.sensing.signal_processing import (
        detection_velocity_display,
        doppler_axis_to_velocity_m_s,
    )

    validate_doppler_comparison(recordings)
    if axis not in {"doppler", "velocity"} or time_reference not in {"result", "midpoint"}:
        raise ValueError("Choose doppler/velocity and result/midpoint display coordinates")
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0:
        raise ValueError("Display dynamic range must be positive and finite")
    if match_windows and len(recordings) != 2:
        raise ValueError("Matched-window display requires exactly two recordings")
    if layout not in {"combined", "separate"}:
        raise ValueError("Choose combined or separate summary layout")
    first = recordings[0]
    coordinate = np.asarray(first["doppler_axis_hz"])
    y_label = "Doppler frequency (Hz)"
    if axis == "velocity":
        coordinate = doppler_axis_to_velocity_m_s(
            coordinate, first["calibration"]["carrier_frequency_hz"]
        )
        y_label = detection_velocity_display(first["calibration"]["range_mode"]).axis_label
    panels = [(r["label"], r["events"]) for r in recordings]
    matching = None
    if match_windows:
        matching = matching_window_indices(first["events"], recordings[1]["events"])
        panels.append(
            (
                f"{recordings[1]['label']} · windows matching {first['label']}",
                [recordings[1]["events"][i] for i in matching],
            )
        )
    reference_power = max(
        float(np.max(event["power_linear"])) for r in recordings for event in r["events"]
    )
    receipt = {
        "axis": axis,
        "time_reference": time_reference,
        "common_power_reference": reference_power,
        "dynamic_range_db": dynamic_range_db,
        "matching_window_indices": matching,
        "timestamp_match": f"Every saved sample epoch; at most {_TIMESTAMP_ULPS} float64 epsilon units at the time scale, no nearest-time selection",
        "layout": layout,
    }
    if layout == "separate":
        time_key = "midpoint_s" if time_reference == "midpoint" else "time_s"
        edges = [
            _doppler_time_edges(
                np.asarray([event[time_key] for event in events]),
                len(events[0]["sample_times_s"]) * events[0]["sample_interval_s"],
            )
            for _, events in panels
        ]
        limits = (min(e[0] for e in edges), max(e[-1] for e in edges))
        receipt["time_limits_s"] = [float(value) for value in limits]
        receipt["outputs"] = []
        for index, (label, events) in enumerate(panels):
            name = (
                "matched_doppler.png"
                if index == len(recordings)
                else f"source_{index + 1}_doppler.png"
            )
            plot_doppler_history(
                events,
                Path(output_path) / name,
                axis=axis,
                carrier_frequency_hz=first["calibration"]["carrier_frequency_hz"],
                range_mode=first["calibration"]["range_mode"],
                time_reference=time_reference,
                title=f"{label} · {len(events):,} saved spectra",
                power_reference=reference_power,
                dynamic_range_db=dynamic_range_db,
                time_limits_s=limits,
                force_map=True,
            )
            receipt["outputs"].append(
                {"image": name, "label": label, "spectrum_count": len(events)}
            )
        return receipt
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(11, 3.3 * len(panels)),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, (label, events) in zip(axes[:, 0], panels):
        times = np.asarray(
            [e["midpoint_s"] if time_reference == "midpoint" else e["time_s"] for e in events]
        )
        power = np.asarray([e["power_linear"] for e in events])
        db = np.full(power.shape, -dynamic_range_db, dtype=np.float64)
        if reference_power > 0:
            np.log10(np.maximum(power / reference_power, 10 ** (-dynamic_range_db / 10)), out=db)
            db *= 10
        width = len(events[0]["sample_times_s"]) * events[0]["sample_interval_s"]
        mesh = _draw_doppler_mesh(
            ax, times, coordinate, db, vmin=-dynamic_range_db, vmax=0, single_time_width=width
        )
        ax.set(title=f"{label} · {len(events):,} saved spectra", ylabel=y_label)
        if reference_power == 0:
            ax.text(
                0.5,
                0.5,
                "No nonzero power in the saved products",
                transform=ax.transAxes,
                ha="center",
            )
    axes[-1, 0].set_xlabel(
        "Observation-window midpoint (s)"
        if time_reference == "midpoint"
        else "Saved result reference time (s)"
    )
    fig.colorbar(
        mesh, ax=list(axes[:, 0]), label="Range-summed power (dB relative to the common peak)"
    )
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI)
    plt.close(fig)
    return receipt


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Export a comparison and exact observation provenance to a new directory."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=("doppler", "velocity"), default="doppler")
    parser.add_argument("--time-reference", choices=("result", "midpoint"), default="result")
    parser.add_argument("--match-windows", action="store_true")
    parser.add_argument("--dynamic-range-db", type=float, default=40.0)
    parser.add_argument("--layout", choices=("combined", "separate"), default="combined")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory")
    if len({label for label, _ in args.source}) != len(args.source):
        parser.error("Source labels must be distinct")
    if not np.isfinite(args.dynamic_range_db) or args.dynamic_range_db <= 0:
        parser.error("Display dynamic range must be positive and finite")
    recordings = [load_doppler_recording(Path(path), label) for label, path in args.source]
    validate_doppler_comparison(recordings)
    if args.match_windows:
        if len(recordings) != 2:
            parser.error("--match-windows requires exactly two sources")
        matching_window_indices(recordings[0]["events"], recordings[1]["events"])
    args.output.mkdir(parents=True)
    image_path = args.output / "doppler_comparison.png"
    receipt = plot_doppler_comparison(
        recordings,
        args.output if args.layout == "separate" else image_path,
        axis=args.axis,
        time_reference=args.time_reference,
        match_windows=args.match_windows,
        dynamic_range_db=args.dynamic_range_db,
        layout=args.layout,
    )
    receipt.update(
        plotter_sha256=_sha256(Path(__file__)),
        provenance="ORCHAV saved primary RD products; range-power sum only, no signal reprocessing",
        recordings=[],
    )
    if args.layout == "combined":
        receipt.update(image=image_path.name, sha256=_sha256(image_path))
    else:
        for output in receipt["outputs"]:
            output["sha256"] = _sha256(args.output / output["image"])
    for recording in recordings:
        source = recording["source"]
        frames = source if (source / "frames_manifest.json").is_file() else source / "frames"
        manifest_path = frames / "frames_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source_files = [manifest_path, *(frames / chunk["file"] for chunk in manifest["chunks"])]
        receipt["recordings"].append(
            {
                "label": recording["label"],
                "source": str(source),
                "calibration": recording["calibration"],
                "skipped_incomplete": recording["skipped_incomplete"],
                "doppler_axis_hz": recording["doppler_axis_hz"].tolist(),
                "range_axis_m": recording["range_axis_m"].tolist(),
                "source_hashes": {
                    p.relative_to(frames).as_posix(): _sha256(p) for p in source_files
                },
                "windows": [
                    {
                        "product_id": e["product_id"],
                        "frame_index": e["frame_index"],
                        "sample_times_s": e["sample_times_s"].tolist(),
                        "sample_phase_coherent": e["sample_phase_coherent"],
                        "saved_result_time_s": e["time_s"],
                        "saved_reference_policy": e["reference_time_policy"],
                        "midpoint_s": e["midpoint_s"],
                        "processing_recipe_id": e["processing_recipe_id"],
                        "processing_display_id": e.get("processing_display_id"),
                    }
                    for e in recording["events"]
                ],
            }
        )
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.layout == "separate":
        for output in receipt["outputs"]:
            image_receipt = {
                **output,
                "plotter_sha256": receipt["plotter_sha256"],
                "provenance": receipt["provenance"],
                "comparison_receipt": receipt_path.name,
                "comparison_receipt_sha256": _sha256(receipt_path),
                **{
                    key: receipt[key]
                    for key in (
                        "axis",
                        "time_reference",
                        "common_power_reference",
                        "dynamic_range_db",
                        "time_limits_s",
                    )
                },
            }
            (args.output / Path(output["image"]).with_suffix(".json")).write_text(
                json.dumps(image_receipt, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
