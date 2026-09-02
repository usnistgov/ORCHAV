"""Build ``SimulationConfig`` from scenario models and explicit overrides.

This module is the YAML-to-runtime-config adapter.  It understands the
``ScenarioConfiguration`` shape produced by the shared scenario loader and maps
those fields into ``SimulationConfig`` dataclasses. It stops before scene
construction: services decide how to turn the normalized config
into Sionna objects, actor-state caches, coverage maps, or streaming contexts.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any, Dict, Optional

from shared.extensions.sensing import resolve_sensing_config
from shared.grpc_transport import format_grpc_endpoint, parse_grpc_endpoint
from shared.logging import get_logger

from .defaults import (
    DEFAULT_FILE_FORMAT,
    DEFAULT_GRPC_SERVER_ADDRESS,
    SCENARIO_HDF5_FILE_FORMAT_ALIAS,
)
from .models import AntennaConfig, SensingConfig, SimulationConfig

logger = get_logger(__name__)


def load_simulation_config(
    scenario_configuration: Any = None,
    overrides: Optional[Dict[str, Any]] = None,
    base: Optional[SimulationConfig] = None,
) -> SimulationConfig:
    """Build a ``SimulationConfig`` from defaults, scenario YAML, and overrides.

    Precedence (lowest -> highest):
      - SimulationConfig defaults (or provided base)
      - ScenarioConfiguration explicit values (if provided)
      - Overrides dict (script/CLI/env)
    """
    cfg = _copy_base_config(base)

    if scenario_configuration is not None:
        rt_cfg = _as_mapping(getattr(scenario_configuration, "raytracing", {}) or {})
        sensing_cfg = _as_mapping(getattr(scenario_configuration, "sensing", {}) or {})

        _apply_file_output(cfg, scenario_configuration)
        _apply_scene(cfg, scenario_configuration)
        _apply_debug(cfg, scenario_configuration)
        _apply_timeline(cfg, scenario_configuration)
        _apply_raytracing(cfg, rt_cfg)
        _apply_rf(cfg, rt_cfg)
        _apply_materials(cfg, rt_cfg)
        _apply_sensing(cfg, sensing_cfg, rt_cfg)
        _apply_antenna(cfg, rt_cfg)
        _apply_streaming(cfg, scenario_configuration)

    _apply_overrides(cfg, overrides)
    return cfg


def _copy_base_config(base: Optional[SimulationConfig]) -> SimulationConfig:
    """Return a run-local config copy.

    ``SimulationConfig`` is config-only state and should remain copyable.  A
    full deep copy keeps nested mutable fields such as sensing, coverage,
    material overrides, transmitters, and receivers isolated from the caller's
    base object.
    """
    if base is None:
        return SimulationConfig()
    return copy.deepcopy(base)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _assign_if_present(
    target: Any,
    field_name: str,
    source: Mapping[str, Any],
    source_key: str | None = None,
    transform: Callable[[Any], Any] | None = None,
) -> None:
    """Assign one simple field when the source explicitly provides a value."""
    key = field_name if source_key is None else source_key
    if key not in source:
        return
    value = source[key]
    if value is None:
        return
    setattr(target, field_name, transform(value) if transform is not None else value)


def _apply_file_output(cfg: SimulationConfig, scenario_configuration: Any) -> None:
    frames_format = getattr(scenario_configuration, "frames_format", None)
    if frames_format == SCENARIO_HDF5_FILE_FORMAT_ALIAS:
        cfg.file_format = DEFAULT_FILE_FORMAT
    elif frames_format is not None:
        cfg.file_format = str(frames_format)


def _apply_scene(cfg: SimulationConfig, scenario_configuration: Any) -> None:
    src = getattr(scenario_configuration, "scene_source", None)
    if src == "library":
        scene_xml = getattr(scenario_configuration, "scene_xml", None)
        if scene_xml:
            cfg.scene_name = str(scene_xml)
        else:
            cfg.scene_name = str(getattr(scenario_configuration, "scene_id", cfg.scene_name))
    elif src == "sionna":
        cfg.scene_name = str(getattr(scenario_configuration, "scene_id", cfg.scene_name))
    else:
        scene_xml = getattr(scenario_configuration, "scene_xml", None)
        if scene_xml:
            cfg.scene_name = str(scene_xml)


def _apply_debug(cfg: SimulationConfig, scenario_configuration: Any) -> None:
    debug_level = getattr(scenario_configuration, "debug_level", None)
    if debug_level is not None:
        cfg.debug_level = str(debug_level).upper()


def _apply_timeline(cfg: SimulationConfig, scenario_configuration: Any) -> None:
    """Apply the scenario timeline to generator step and duration settings."""

    timeline = getattr(scenario_configuration, "timeline", None)
    if timeline is None:
        return
    steps = getattr(timeline, "steps", None)
    duration_s = getattr(timeline, "duration_s", None)
    if steps is not None:
        cfg.num_steps = int(steps)
    if duration_s is not None:
        cfg.duration = float(duration_s)


def _apply_raytracing(cfg: SimulationConfig, rt_cfg: Mapping[str, Any]) -> None:
    _assign_if_present(cfg, "export_path_metrics", rt_cfg, transform=bool)
    _assign_if_present(cfg, "start_step", rt_cfg, transform=int)

    quality_cfg = _as_mapping(rt_cfg.get("quality") or {})
    preset = quality_cfg.get("preset")
    if preset:
        cfg.quality = str(preset)

    view = rt_cfg.get("view")
    if view:
        cfg.view = str(view)


def _apply_rf(cfg: SimulationConfig, rt_cfg: Mapping[str, Any]) -> None:
    _assign_if_present(cfg, "carrier_frequency_hz", rt_cfg, transform=float)
    _assign_if_present(cfg, "bandwidth_hz", rt_cfg, transform=float)
    _assign_if_present(cfg, "temperature_k", rt_cfg, transform=float)
    _assign_if_present(cfg, "mesh_update_interval_s", rt_cfg, transform=float)
    _assign_if_present(cfg, "cir_time_steps", rt_cfg, transform=int)
    _assign_if_present(cfg, "cir_sampling_frequency_hz", rt_cfg, transform=float)


def _apply_materials(cfg: SimulationConfig, rt_cfg: Mapping[str, Any]) -> None:
    scene_materials_cfg = _as_mapping(rt_cfg.get("scene_materials") or {})
    preset = scene_materials_cfg.get("scattering_coefficient_preset")
    if preset is not None:
        cfg.scene_material_scattering_coefficient_preset = str(preset)

    if "materials" not in rt_cfg:
        return
    materials_cfg = rt_cfg.get("materials")
    if isinstance(materials_cfg, Mapping):
        cfg.material_overrides = {
            str(name): dict(values) if isinstance(values, Mapping) else {}
            for name, values in materials_cfg.items()
        }


def _apply_sensing(
    cfg: SimulationConfig,
    sensing_cfg: Mapping[str, Any],
    rt_cfg: Mapping[str, Any],
) -> None:
    if cfg.sensing is None:
        cfg.sensing = SensingConfig()

    _assign_if_present(cfg.sensing, "enabled", sensing_cfg, transform=bool)
    if not cfg.sensing.enabled:
        return

    resolved_sensing = resolve_sensing_config(
        scenario_sensing=sensing_cfg,
        scenario_raytracing=rt_cfg,
        defaults={
            "sampling_frequency": cfg.sensing.bandwidth,
            "prf_hz": cfg.sensing.prf_hz,
            "fft_size_range": cfg.sensing.fft_size_range,
            "fft_size_doppler": cfg.sensing.fft_size_doppler,
            "carrier_freq_hz": cfg.carrier_frequency_hz,
            "range_mode": cfg.sensing.range_mode,
            "cir_time_steps": cfg.cir_time_steps,
        },
    )
    cfg.sensing.bandwidth = float(resolved_sensing["sampling_frequency"])
    cfg.sensing.prf_hz = float(resolved_sensing["prf_hz"])
    cfg.sensing.fft_size_range = int(resolved_sensing["fft_size_range"])
    cfg.sensing.fft_size_doppler = int(resolved_sensing["fft_size_doppler"])
    cfg.sensing.range_mode = str(resolved_sensing["range_mode"])

    _assign_if_present(cfg.sensing, "bandwidth", sensing_cfg, transform=float)
    _assign_if_present(cfg.sensing, "chirps_per_frame", sensing_cfg, transform=int)
    _assign_if_present(cfg.sensing, "prf_hz", sensing_cfg, transform=float)
    _assign_if_present(cfg.sensing, "range_mode", sensing_cfg, transform=str)
    _assign_if_present(cfg.sensing, "fft_size_range", sensing_cfg, transform=int)
    _assign_if_present(cfg.sensing, "fft_size_doppler", sensing_cfg, transform=int)

    for field_name in (
        "window_range",
        "window_doppler",
        "cfar_type",
        "raw_payload_policy",
        "tracker_type",
        "processing_mode",
    ):
        _assign_if_present(cfg.sensing, field_name, sensing_cfg, transform=str)

    for field_name in (
        "cfar_guard_cells",
        "cfar_ref_cells",
        "cfar_rd_guard_cells_range",
        "cfar_rd_guard_cells_doppler",
        "cfar_rd_ref_cells_range",
        "cfar_rd_ref_cells_doppler",
        "pre_cfar_doppler_edge_guard_bins",
        "pre_cfar_doppler_dc_guard_bins",
        "track_confirmation_m",
        "track_confirmation_n",
        "track_max_missed_frames",
        "rng_seed",
    ):
        _assign_if_present(cfg.sensing, field_name, sensing_cfg, transform=int)

    for field_name in (
        "cfar_threshold_scale",
        "cfar_os_rank",
        "cfar_min_snr_db",
        "min_detection_range_m",
        "max_detection_range_m",
        "track_association_range_gate_m",
        "track_association_velocity_gate_m_s",
        "eval_association_range_gate_m",
        "eval_association_velocity_gate_m_s",
        "noise_snr_db",
        "aoa_az_min_deg",
        "aoa_az_max_deg",
        "aoa_el_min_deg",
        "aoa_el_max_deg",
    ):
        _assign_if_present(cfg.sensing, field_name, sensing_cfg, transform=float)

    for field_name in (
        "track_confirmation_enabled",
        "track_output_tentative",
        "output_range_doppler",
        "output_range_profile",
        "output_detections",
        "persist_detection_stage_artifacts",
        "clutter_removal_enabled",
        "noise_enabled",
        "aoa_filter_enabled",
    ):
        _assign_if_present(cfg.sensing, field_name, sensing_cfg, transform=bool)

    if sensing_cfg.get("clutter_removal_window") is not None:
        cfg.sensing.clutter_removal_window = max(1, int(sensing_cfg["clutter_removal_window"]))

    _assign_if_present(cfg.sensing, "display_range_xlim", sensing_cfg, transform=list)
    _assign_if_present(cfg.sensing, "display_velocity_ylim", sensing_cfg, transform=list)

    cfg.sensing.__post_init__()


def _apply_antenna(cfg: SimulationConfig, rt_cfg: Mapping[str, Any]) -> None:
    antenna_value = rt_cfg.get("antenna")
    if not isinstance(antenna_value, Mapping):
        return
    antenna_cfg = antenna_value

    tx_ant = antenna_cfg.get("tx")
    if isinstance(tx_ant, Mapping):
        cfg.tx_antenna = _build_antenna_config(tx_ant)

    rx_ant = antenna_cfg.get("rx")
    if isinstance(rx_ant, Mapping):
        cfg.rx_antenna = _build_antenna_config(rx_ant)


def _build_antenna_config(values: Mapping[str, Any]) -> AntennaConfig:
    """Build one runtime array while preserving schema normalization."""

    normalized = dict(values)
    pattern = normalized.get("pattern")
    if isinstance(pattern, str):
        normalized["pattern"] = pattern.strip()
    return AntennaConfig(**normalized)


def _apply_streaming(cfg: SimulationConfig, scenario_configuration: Any) -> None:
    data_mode = getattr(scenario_configuration, "data_mode", None)
    endpoints = getattr(scenario_configuration, "live_grpc_endpoints", {}) or {}
    if data_mode != "live_grpc":
        return

    cfg.output_mode = "grpc"
    endpoint = str(endpoints.get("sionna", f"grpc://{DEFAULT_GRPC_SERVER_ADDRESS}"))
    advertised_host, port = parse_grpc_endpoint(endpoint)
    cfg.grpc_config = {
        "endpoint": f"grpc://{format_grpc_endpoint(advertised_host, port)}",
        "advertised_host": advertised_host,
        "port": port,
    }


def _apply_overrides(
    cfg: SimulationConfig,
    overrides: Optional[Dict[str, Any]],
) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)
