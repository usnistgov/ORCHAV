"""Grouped sensing authoring, normalized to the established flat runtime contract.

Only explicitly grouped input is translated. Old YAML, saved recipes and runtime
field defaults retain their meaning. This module does no signal processing.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

# Each leaf maps one authored name to one existing runtime/configuration field.
# Keep this table exhaustive: tests compare it with SensingModel's field set.
GROUPS: dict[str, Any] = {
    "acquisition": {
        "mode": "processing_mode",
        "bandwidth_hz": "bandwidth",
        "frequency_samples": "range_response.num_frequency_bins",
        "sample_rate_hz": "prf_hz",
        "samples_per_cpi": "fft_size_doppler",
    },
    "processing": {
        "range": {
            "interpretation": "range_mode",
            "fft_size": "fft_size_range",
            "delay_window": "window_range",
            "model": "range_response.model",
            "frequency_window": "range_response.frequency_window",
        },
        "doppler": {
            "fft_size": "fft_size_doppler",
            "window": "window_doppler",
            "hop_samples": "cpi_hop_samples",
        },
        "clutter": {
            "coherent_mode": "coherent_clutter_mode",
            "history_enabled": "clutter_removal_enabled",
            "history_window": "clutter_removal_window",
        },
        "path_selection": {
            "enabled": "aoa_filter_enabled",
            "azimuth_min_deg": "aoa_az_min_deg",
            "azimuth_max_deg": "aoa_az_max_deg",
            "elevation_min_deg": "aoa_el_min_deg",
            "elevation_max_deg": "aoa_el_max_deg",
        },
    },
    "noise": {
        "target_relative": {
            "enabled": "noise_enabled",
            "snr_db": "noise_snr_db",
            "seed": "rng_seed",
        },
        "frequency": {
            "variance": "range_response.frequency_noise.variance",
            "seed": "range_response.frequency_noise.seed",
        },
    },
    "detection": {
        "enabled": "output_detections",
        "mode": "array_detection_mode",
        "angle_oversampling": "beam_angle_oversampling",
        "cfar": {
            "type": "cfar_type",
            "edge_policy": "cfar_edge_policy",
            "threshold_scale": "cfar_threshold_scale",
            "os_rank": "cfar_os_rank",
            "range_profile": {
                "guard_cells": "cfar_guard_cells",
                "reference_cells": "cfar_ref_cells",
            },
            "range_doppler": {
                "guard_range_cells": "cfar_rd_guard_cells_range",
                "guard_doppler_cells": "cfar_rd_guard_cells_doppler",
                "reference_range_cells": "cfar_rd_ref_cells_range",
                "reference_doppler_cells": "cfar_rd_ref_cells_doppler",
            },
        },
        "exclusions": {
            "doppler_edge_bins": "pre_cfar_doppler_edge_guard_bins",
            "doppler_dc_bins": "pre_cfar_doppler_dc_guard_bins",
        },
        "filters": {
            "minimum_snr_db": "cfar_min_snr_db",
            "maximum_dynamic_range_db": "cfar_max_dynamic_range_db",
            "minimum_range_m": "min_detection_range_m",
            "maximum_range_m": "max_detection_range_m",
        },
        "clustering": {
            "range_gate_m": "cluster_range_gate_m",
            "velocity_gate_m_s": "cluster_velocity_gate_m_s",
        },
        "shortest_range_filter": {
            "velocity_gate_m_s": "ghost_filter_velocity_gate_m_s",
            "azimuth_gate_deg": "ghost_filter_azimuth_gate_deg",
        },
    },
    "tracking": {
        "enabled": "track_confirmation_enabled",
        "type": "tracker_type",
        "confirmation": {
            "hits": "track_confirmation_m",
            "window": "track_confirmation_n",
            "mode": "track_confirmation_mode",
        },
        "max_missed_updates": "track_max_missed_frames",
        "output_tentative": "track_output_tentative",
        "acceleration_psd_m2_s3": "track_acceleration_psd_m2_s3",
        "jpda": {
            "max_joint_nodes": "jpda_max_joint_nodes",
            "birth_confirmation_frames": "jpda_birth_confirmation_frames",
            "birth_separation_chi2": "jpda_birth_separation_chi2",
        },
        "association": {
            "range_gate_m": "track_association_range_gate_m",
            "velocity_gate_m_s": "track_association_velocity_gate_m_s",
            "azimuth_gate_deg": "track_association_azimuth_gate_deg",
            "position_gate_m": "track_association_position_gate_m",
        },
    },
    "evaluation": {
        "range_gate_m": "eval_association_range_gate_m",
        "velocity_gate_m_s": "eval_association_velocity_gate_m_s",
    },
    "recording": {
        "profile": "recording_profile",
        "display": "recording_display",
        "range_doppler": "output_range_doppler",
        "range_profile": "output_range_profile",
        "complex_observations": "raw_payload_policy",
        "detection_stages": "persist_detection_stage_artifacts",
        "tracking_diagnostics": "persist_tracking_diagnostics",
    },
}

# Processing choices only. Bandwidth, observation clocks/counts and range
# interpretation describe the experiment and remain independently configured.
COHERENT_HANN = {
    "processing_mode": "coherent_cpi",
    "fft_size_range": 512,
    "window_range": "boxcar",
    "window_doppler": "hann",
    "coherent_clutter_mode": "windowed_mean",
    "range_response": {"model": "cfr_ifft", "frequency_window": "hann"},
}


def _put(result: dict, path: str, value: Any, source: str) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        if part in result and not isinstance(result[part], dict):
            raise ValueError(f"{source} conflicts with {path}")
        result = result.setdefault(part, {})
    if parts[-1] in result:
        raise ValueError(f"{source} duplicates {path}; declare the setting only once")
    result[parts[-1]] = value


def _expand(values: Mapping, spec: dict, result: dict, prefix: str) -> None:
    for key, value in values.items():
        source = f"{prefix}.{key}"
        if key not in spec:
            raise ValueError(f"Unknown sensing setting: {source}")
        destination = spec[key]
        if isinstance(destination, dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"{source} must be a mapping")
            _expand(value, destination, result, source)
        else:
            _put(result, destination, deepcopy(value), source)


def _with_defaults(values: dict, defaults: dict) -> dict:
    result = deepcopy(defaults)
    for key, value in values.items():
        result[key] = (
            _with_defaults(value, result[key])
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def normalize_sensing_groups(value: Any) -> Any:
    """Validate grouped names and return flat fields for existing validators."""
    if not isinstance(value, Mapping):
        return value
    result = deepcopy(dict(value))
    preset = result.pop("preset", None)
    grouped = {key: result.pop(key) for key in GROUPS if key in result}
    _expand(grouped, GROUPS, result, "sensing")
    if result.get("recording_profile") is not None and result.get("raw_payload_policy") is not None:
        raise ValueError(
            "recording.profile cannot be combined with complex_observations/raw_payload_policy"
        )
    if result.get("recording_display") is not None and result.get("recording_profile") is None:
        raise ValueError("recording.display requires an explicit recording.profile")
    if result.get("recording_profile") == "full" and any(
        result.get(key) is False
        for key in ("persist_detection_stage_artifacts", "persist_tracking_diagnostics")
    ):
        raise ValueError("recording.profile full requires detector and tracker diagnostics")
    if preset is not None:
        if preset != "coherent_hann":
            raise ValueError(f"Unknown sensing preset: {preset}")
        result = _with_defaults(result, COHERENT_HANN)
    if grouped and result.get("processing_mode") == "coherent_cpi":
        result.setdefault("coherent_clutter_mode", "windowed_mean")
    # Descriptive authoring name; preserve the established serialized recipe
    # identity and its conditional unweighted operator for existing consumers.
    if result.get("coherent_clutter_mode") == "unweighted_mean":
        result["coherent_clutter_mode"] = "legacy"
    # CFR synthesis already applies its frequency taper. This is derived only
    # for new grouped CFR authoring, never for old flat recipes.
    range_group = grouped.get("processing", {}).get("range", {})
    if range_group.get("model") == "cfr_ifft":
        result.setdefault("window_range", "boxcar")
    return result


def normalize_scenario_sensing(value: Any) -> Any:
    """Resolve coherent acquisition clocks and view hints at the YAML boundary."""
    if not isinstance(value, Mapping) or not isinstance(value.get("sensing"), Mapping):
        return value
    result = deepcopy(dict(value))
    authored = result["sensing"]
    sensing = normalize_sensing_groups(authored)
    acquisition = authored.get("acquisition", {})
    coherent = sensing.get("processing_mode") == "coherent_cpi"
    if "samples_per_cpi" in acquisition and not coherent:
        raise ValueError("acquisition.samples_per_cpi requires mode: coherent_cpi")
    if coherent and acquisition:
        raytracing = result.setdefault("raytracing", {})
        for source, target, requested in (
            ("prf_hz", "cir_sampling_frequency_hz", "sample_rate_hz"),
            ("fft_size_doppler", "cir_time_steps", "samples_per_cpi"),
        ):
            if requested not in acquisition:
                continue
            if target in raytracing and raytracing[target] != sensing[source]:
                raise ValueError(
                    f"sensing.acquisition.{requested} conflicts with raytracing.{target}"
                )
            raytracing[target] = sensing[source]
    views = (result.get("view_defaults") or {}).get("sensing") or {}
    for name, legacy in (
        ("range_xlim", "display_range_xlim"),
        ("velocity_ylim", "display_velocity_ylim"),
    ):
        if name in views:
            if legacy in sensing and sensing[legacy] != views[name]:
                raise ValueError(f"view_defaults.sensing.{name} conflicts with sensing.{legacy}")
            sensing[legacy] = deepcopy(views[name])
    result["sensing"] = sensing
    return result
