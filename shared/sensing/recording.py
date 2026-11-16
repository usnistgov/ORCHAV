"""Lossless sensing retention policies and fixed numeric display products.

Profiles affect publication, never the measurements supplied to the detector or
tracker. The canonical complex observation is selected at the latest stochastic
input boundary so replay never silently replaces an acquired noise realization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np

from shared.sensing.motion_analysis import MotionRecipe, compute_motion_spectra
from shared.sensing.products import (
    TEMPORAL_SUPPORT_FULL_ACQUISITION,
    TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
    add_product_spec,
    catalog_axis,
    dense_axis,
)

RECORDING_PROFILES = ("results", "maps", "reprocessing", "full")
DISPLAY_PRODUCT_KEYS = frozenset(
    {
        "delay_power_by_pair",
        "frequency_power_by_pair",
        "micro_doppler_by_pair",
        "micro_doppler_raw_by_pair",
    }
)


def _sequential_display_hop(window: int, requested: int | None, result_hop: int) -> int:
    """Align saved current-window columns with the native result-publication grid."""
    if window < result_hop:
        raise ValueError(
            "recording.display.window_samples must be at least sensing.cpi_hop_samples"
        )
    if requested is not None:
        if type(requested) is not int or requested < 1 or requested % result_hop:
            raise ValueError(
                "recording.display.hop_samples must be a positive integer multiple of "
                "sensing.cpi_hop_samples"
            )
        return requested
    nominal = max(1, window // 4)
    return ((nominal + result_hop - 1) // result_hop) * result_hop


def validate_recording_options(
    profile: str,
    display: Mapping | None,
    *,
    doppler_window_samples: int = 128,
    result_hop_samples: int = 1,
) -> None:
    """Reject invalid profile/recipe choices before an acquisition starts."""
    if profile not in RECORDING_PROFILES:
        raise ValueError(f"recording.profile must be one of {RECORDING_PROFILES}")
    if display is None:
        return
    if not isinstance(display, Mapping):
        raise ValueError("recording.display must be a mapping of MotionRecipe settings")
    unknown = set(display) - {item.name for item in fields(MotionRecipe)}
    if unknown:
        raise ValueError(f"Unknown recording.display settings: {sorted(unknown)}")
    if profile == "results" and display:
        raise ValueError("recording.display requires maps, reprocessing or full")
    if not display:
        return
    window = display.get("window_samples", doppler_window_samples)
    if type(window) is not int:
        raise ValueError("recording.display.window_samples must be an integer")
    defaults = {"window_samples": window, "hop_samples": min(32, window)}
    if result_hop_samples > 1:
        defaults["hop_samples"] = _sequential_display_hop(
            window, display.get("hop_samples"), result_hop_samples
        )
    # Only an explicitly supplied pair of bounds can be ordered here. The
    # omitted endpoint is resolved from the acquired range span at generation.
    for supplied, omitted in (("range_min_m", "range_max_m"), ("range_max_m", "range_min_m")):
        if supplied in display and omitted not in display:
            defaults[omitted] = display[supplied]
    MotionRecipe(**(defaults | dict(display)))


def canonical_observation_key(config: Any) -> str:
    """Choose frequency samples only when no later noise changes the signal."""
    post_projection_noise = bool(config.noise_enabled and config.noise_snr_db is not None)
    if config.range_response.model == "cfr_ifft" and not post_projection_noise:
        return "cfr_by_pair_rx_element"
    return "slow_time_by_pair_rx_element"


def retain_raw_product(config: Any, key: str) -> bool:
    """Select only the canonical complex input and useful before-filter maps."""
    profile = config.recording_profile
    canonical = canonical_observation_key(config)
    if key == "cfr_by_pair_rx_element":
        return profile in {"reprocessing", "full"} and canonical == key
    if key == "slow_time_matrix":
        return profile in {"reprocessing", "full"} and canonical == "slow_time_by_pair_rx_element"
    return profile != "results" and key == "range_doppler_map_raw"


def add_display_products(
    sensing: dict[str, Any],
    config: Any,
    *,
    samples: np.ndarray,
    times_s: np.ndarray,
    range_axis_m: np.ndarray,
    frequency_samples: np.ndarray | None = None,
    sequential: bool = False,
    last_spectrum_time_s: float | None = None,
) -> float | None:
    """Queue combined power and authored STFT products before discarding phase.

    Sequential histories overlap; publish the current power sample only and
    emit each STFT column once at the authored hop. Coherent calls are confined
    to one solve's CPI. The caller owns the small per-pair cadence state.
    """
    if config.recording_profile in {None, "results"}:
        return last_spectrum_time_s
    x = np.asarray(samples)
    times = np.asarray(times_s, dtype=np.float64)
    ranges = np.asarray(range_axis_m, dtype=np.float64)
    if x.ndim != 2 or x.shape != (len(times), len(ranges)):
        raise ValueError("Display samples must match physical time and range coordinates")
    pair = catalog_axis("pair", size=1, catalog_key="pair_axis")
    support = (
        TEMPORAL_SUPPORT_LAST_VALID_SAMPLE if sequential else TEMPORAL_SUPPORT_FULL_ACQUISITION
    )
    shown = x[-1:] if sequential else x
    shown_times = times[-1:] if sequential else times
    sensing["display_sample_time_s"] = shown_times.copy()
    sensing["range_axis_m"] = ranges
    time_axis = dense_axis(
        "slow_time",
        size=len(shown),
        unit="s",
        coordinate_key="display_sample_time_s",
        reference_frame="scenario_clock",
    )
    range_axis = dense_axis(
        "range",
        size=len(ranges),
        unit="m",
        coordinate_key="range_axis_m",
        reference_frame="tx_target_rx_path",
    )
    combine = {
        "policy": "rx_broadside_unitary_voltage_sum",
        "normalization": "one_over_sqrt_element_count",
    }
    sensing["delay_power_by_pair"] = (np.abs(shown) ** 2)[None]
    add_product_spec(
        sensing,
        data_key="delay_power_by_pair",
        kind="delay_domain_power",
        axes=(pair, time_axis, range_axis),
        representation="power_linear",
        unit="1",
        temporal_support=support,
        combine=combine,
        processing={"phase_retained": False, "projection": "squared_magnitude"},
    )
    if frequency_samples is not None:
        frequency = np.asarray(frequency_samples)
        if frequency.ndim != 2 or frequency.shape[0] != len(times):
            raise ValueError("Combined frequency display samples must share physical times")
        frequency = frequency[-1:] if sequential else frequency
        sensing["frequency_power_by_pair"] = (np.abs(frequency) ** 2)[None]
        add_product_spec(
            sensing,
            data_key="frequency_power_by_pair",
            kind="baseband_frequency_power",
            axes=(
                pair,
                time_axis,
                dense_axis(
                    "frequency",
                    size=frequency.shape[-1],
                    unit="Hz",
                    coordinate_key="frequency_axis_hz",
                    reference_frame="carrier_frequency_hz",
                ),
            ),
            representation="power_linear",
            unit="1",
            temporal_support=support,
            combine=combine,
            processing={"phase_retained": False, "window_applied": False},
        )

    requested = dict(config.recording_display or {})
    window = int(requested.get("window_samples", config.fft_size_doppler))
    if window < 3:
        sensing["display_status"] = {"micro_doppler": "STFT requires at least three samples"}
        return last_spectrum_time_s
    defaults = dict(
        range_min_m=float(ranges[0]),
        range_max_m=float(ranges[-1]),
        window_samples=window,
        hop_samples=(
            _sequential_display_hop(
                window,
                requested.get("hop_samples"),
                getattr(config, "cpi_hop_samples", None) or 1,
            )
            if sequential
            else window
        ),
        window="hann",
        mean="both",
    )
    recipe = MotionRecipe(**(defaults | requested))
    if len(times) < window:
        sensing["display_status"] = {"micro_doppler": "insufficient temporal support"}
        return last_spectrum_time_s
    dt = 1.0 / config.prf_hz
    if len(times) > 1 and not np.allclose(np.diff(times), dt, rtol=1e-7, atol=1e-12):
        sensing["display_status"] = {"micro_doppler": "acquisition gap"}
        return None
    if sequential and last_spectrum_time_s is not None:
        if times[-1] - last_spectrum_time_s < recipe.hop_samples * dt - 1e-12:
            return last_spectrum_time_s
    # A sliding acquisition's existing bounded buffer has exact sample support.
    # Evaluate its newest complete window; earlier overlapping columns were saved.
    analysis_x = x[-window:] if sequential else x
    analysis_times = times[-window:] if sequential else times
    if recipe.time_start_s is not None or recipe.time_end_s is not None:
        eligible = np.ones(len(analysis_times), dtype=bool)
        if recipe.time_start_s is not None:
            eligible &= analysis_times >= recipe.time_start_s
        if recipe.time_end_s is not None:
            eligible &= analysis_times <= recipe.time_end_s
        if eligible.sum() < window:
            sensing["display_status"] = {"micro_doppler": "outside saved display interval"}
            return last_spectrum_time_s
    result = compute_motion_spectra(analysis_x, analysis_times, ranges, recipe)
    # Subtracting large physical epochs introduces roundoff in a locally
    # estimated dt. The acquisition declares an exact sampling cadence; use it
    # for the saved frequency grid and PSD scale so adjacent CPIs share a grid.
    estimated_dt = result["metadata"]["sample_interval_s"]
    result["doppler_hz"] = np.fft.fftshift(np.fft.fftfreq(window, d=dt))
    for name in ("removed_psd", "retained_psd"):
        if name in result:
            result[name] = result[name] * (dt / estimated_dt)
    result["metadata"].update(
        sample_interval_s=dt,
        sample_rate_hz=float(config.prf_hz),
        sample_interval_origin="declared_acquisition_prf",
        kernel_estimated_sample_interval_s=estimated_dt,
    )
    sensing["micro_doppler_time_s"] = result["time_s"]
    sensing["micro_doppler_frequency_hz"] = result["doppler_hz"]
    for mode, key, variant in (
        ("removed", "micro_doppler_by_pair", "primary"),
        ("retained", "micro_doppler_raw_by_pair", "raw"),
    ):
        if f"{mode}_psd" not in result:
            continue
        sensing[key] = result[f"{mode}_psd"].T[None]
        add_product_spec(
            sensing,
            data_key=key,
            kind="micro_doppler",
            variant=variant,
            axes=(
                pair,
                dense_axis(
                    "slow_time",
                    size=len(result["time_s"]),
                    unit="s",
                    coordinate_key="micro_doppler_time_s",
                    reference_frame="scenario_clock",
                ),
                dense_axis(
                    "doppler_frequency",
                    size=len(result["doppler_hz"]),
                    unit="Hz",
                    coordinate_key="micro_doppler_frequency_hz",
                ),
            ),
            representation="power_linear",
            unit="1",
            combine=combine,
            processing={
                **result["metadata"],
                "mean": mode,
                "hop_samples": recipe.hop_samples,
                "fixed_display_product": True,
            },
        )
    return float(times[-1])


def apply_recording_profile(sensing: dict[str, Any], config: Any) -> dict[str, Any]:
    """Select numeric products before descriptor hashing and HDF5 publication."""
    profile = config.recording_profile
    if profile is None:
        return sensing
    specs = sensing.get("_product_specs", [])
    if profile == "results":
        keep = set()
        if config.persist_detection_stage_artifacts:
            keep = {
                spec["data_key"]
                for spec in specs
                if "cfar" in spec["data_key"]
                or any(
                    part in spec["data_key"]
                    for part in ("detection_input", "nms", "snr", "cluster_labels")
                )
            }
            # Diagnostic products require the actual source power products.
            changed = True
            while changed:
                previous = set(keep)
                for spec in specs:
                    if spec["data_key"] in keep:
                        keep.update(spec.get("source_data_keys", ()))
                changed = previous != keep
        for spec in specs:
            if spec["data_key"] not in keep:
                sensing.pop(spec["data_key"], None)
        specs = [spec for spec in specs if spec["data_key"] in keep]

    renamed = {}
    for source, target in (
        ("range_doppler_by_pair", "range_doppler_power_by_pair"),
        ("range_doppler_raw_by_pair", "range_doppler_raw_power_by_pair"),
    ):
        if source in sensing:
            sensing[target] = np.abs(sensing.pop(source)) ** 2
            renamed[source] = target
    for spec in specs:
        key = spec["data_key"]
        if key in renamed:
            spec["data_key"] = renamed[key]
            spec["representation"] = "power_linear"
            spec["processing"] = {
                **(spec.get("processing") or {}),
                "retained_projection": "squared_magnitude",
            }
        spec["source_data_keys"] = [
            renamed.get(key, key) for key in spec.get("source_data_keys", ())
        ]
    primary, raw = "range_doppler_power_by_pair", "range_doppler_raw_power_by_pair"
    if primary in sensing and raw in sensing and np.array_equal(sensing[primary], sensing[raw]):
        sensing.pop(raw)
        specs = [spec for spec in specs if spec["data_key"] != raw]
        for spec in specs:
            spec["source_data_keys"] = [
                primary if key == raw else key for key in spec.get("source_data_keys", ())
            ]
    sensing["_product_specs"] = specs
    canonical = canonical_observation_key(config)
    sensing["recording"] = {
        "profile": profile,
        "canonical_observation": canonical if canonical in sensing else None,
        "display_recipe": dict(config.recording_display or {}),
    }
    for key in ("config", "config_resolved"):
        sensing[key]["recording_profile"] = profile
        sensing[key]["recording_display"] = dict(config.recording_display or {})
    return sensing
