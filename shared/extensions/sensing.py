"""Contract helpers for optional ``StandardMPCFrame.sensing`` payloads.

The core frame schema carries path geometry and per-path metrics. The optional
``sensing`` extension adds derived arrays, detection metadata, processing
configuration, and timing fields. This module keeps the frame-facing extension
contract in one place for HDF5 storage, reading, and structural validation. The
protobuf frame codec intentionally does not carry this extension.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, TypedDict

from shared.frames.payloads import (
    FRAME_EXTENSION_DENSE_DATASET_KEYS,
    FRAME_EXTENSION_DENSE_HDF5_MIN_SCHEMA_VERSION,
    FRAME_EXTENSION_STAGE_ARTIFACT_DATASET_ALIASES,
)

SENSING_SCHEMA_VERSION = 2
SENSING_DENSE_HDF5_MIN_SCHEMA_VERSION = FRAME_EXTENSION_DENSE_HDF5_MIN_SCHEMA_VERSION
SENSING_STAGE_ARTIFACT_DATASET_ALIASES = FRAME_EXTENSION_STAGE_ARTIFACT_DATASET_ALIASES
SENSING_DENSE_DATASET_KEYS = FRAME_EXTENSION_DENSE_DATASET_KEYS


class SensingResolvedConfig(TypedDict):
    """Normalized sensing settings read from payload, scenario, or defaults.

    Frequency and bandwidth values are in hertz; FFT sizes and CIR time steps
    are sample counts. ``bandwidth`` mirrors ``sampling_frequency``, and
    ``carrier_frequency`` mirrors ``carrier_freq_hz`` for callers using either
    accepted field name.
    """

    range_mode: str
    sampling_frequency: float
    bandwidth: float
    prf_hz: float
    fft_size_range: int
    fft_size_doppler: int
    carrier_freq_hz: float
    carrier_frequency: float
    cir_time_steps: int
    processing_mode: str


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    fvalue = _to_float(value)
    if fvalue is None:
        return None
    try:
        return int(round(fvalue))
    except (TypeError, ValueError):
        return None


def normalize_range_mode(value: Any) -> str:
    """Normalize range-mode aliases to the two stored contract values."""
    if isinstance(value, str) and value.lower().startswith("mono"):
        return "monostatic"
    return "bistatic"


def _is_range_mode_alias(value: str) -> bool:
    """Return whether a string names one of the supported range modes."""
    normalized = value.lower()
    return normalized.startswith("mono") or normalized.startswith("bi")


def _first_float(default: float, *values: Any) -> float:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return float(default)


def _first_int(default: int, *values: Any) -> int:
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return int(default)


def resolve_sensing_config(
    *,
    sensing_block: Optional[Mapping[str, Any]] = None,
    scenario_sensing: Optional[Mapping[str, Any]] = None,
    scenario_raytracing: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
) -> SensingResolvedConfig:
    """Resolve sensing config fields with one precedence chain.

    Precedence is highest-to-lowest:
    1. direct sensing payload fields
    2. sensing payload ``config`` dict
    3. scenario YAML sensing/raytracing config
    4. provided defaults
    """

    block: Mapping[str, Any] = sensing_block or {}
    payload_cfg_raw = block.get("config")
    payload_cfg = payload_cfg_raw if isinstance(payload_cfg_raw, Mapping) else {}
    payload_resolved_raw = block.get("config_resolved")
    payload_resolved = payload_resolved_raw if isinstance(payload_resolved_raw, Mapping) else {}
    scenario_s = scenario_sensing or {}
    scenario_rt = scenario_raytracing or {}
    resolved_defaults = defaults or {}

    default_fs = _first_float(1.0, resolved_defaults.get("sampling_frequency"), 1.0)
    default_prf = _first_float(200.0, resolved_defaults.get("prf_hz"), 200.0)
    default_fft_r = _first_int(128, resolved_defaults.get("fft_size_range"), 128)
    default_fft_d = _first_int(64, resolved_defaults.get("fft_size_doppler"), 64)
    default_carrier = _first_float(3.5e9, resolved_defaults.get("carrier_freq_hz"), 3.5e9)
    default_cir_steps = _first_int(1, resolved_defaults.get("cir_time_steps"), 1)

    range_mode = normalize_range_mode(
        block.get("range_mode")
        or payload_cfg.get("range_mode")
        or payload_resolved.get("range_mode")
        or scenario_s.get("range_mode")
        or resolved_defaults.get("range_mode")
        or "bistatic"
    )
    sampling_frequency = _first_float(
        default_fs,
        block.get("sampling_frequency"),
        block.get("bandwidth_hz"),
        payload_cfg.get("sampling_frequency"),
        payload_cfg.get("bandwidth_hz"),
        payload_cfg.get("bandwidth"),
        payload_resolved.get("sampling_frequency"),
        payload_resolved.get("bandwidth_hz"),
        payload_resolved.get("bandwidth"),
        scenario_s.get("sampling_frequency"),
        scenario_s.get("bandwidth_hz"),
        scenario_s.get("bandwidth"),
    )
    prf_hz = _first_float(
        default_prf,
        block.get("prf_hz"),
        payload_cfg.get("prf_hz"),
        payload_resolved.get("prf_hz"),
        scenario_s.get("prf_hz"),
    )
    fft_size_range = _first_int(
        default_fft_r,
        block.get("fft_size_range"),
        payload_cfg.get("fft_size_range"),
        payload_resolved.get("fft_size_range"),
        scenario_s.get("fft_size_range"),
    )
    fft_size_doppler = _first_int(
        default_fft_d,
        block.get("fft_size_doppler"),
        payload_cfg.get("fft_size_doppler"),
        payload_resolved.get("fft_size_doppler"),
        scenario_s.get("fft_size_doppler"),
    )
    carrier_freq = _first_float(
        default_carrier,
        block.get("carrier_freq_hz"),
        block.get("carrier_frequency"),
        payload_cfg.get("carrier_freq_hz"),
        payload_cfg.get("carrier_frequency"),
        payload_resolved.get("carrier_freq_hz"),
        payload_resolved.get("carrier_frequency"),
        scenario_rt.get("carrier_frequency_hz"),
    )
    cir_time_steps = _first_int(
        default_cir_steps,
        block.get("cir_time_steps"),
        payload_cfg.get("cir_time_steps"),
        payload_resolved.get("cir_time_steps"),
        scenario_rt.get("cir_time_steps"),
    )
    if cir_time_steps < 1:
        cir_time_steps = 1

    processing_mode = (
        block.get("processing_mode")
        or payload_cfg.get("processing_mode")
        or payload_resolved.get("processing_mode")
        or scenario_s.get("processing_mode")
        or resolved_defaults.get("processing_mode")
        or "auto"
    )

    return {
        "range_mode": range_mode,
        "sampling_frequency": sampling_frequency,
        "bandwidth": sampling_frequency,
        "prf_hz": prf_hz,
        "fft_size_range": fft_size_range,
        "fft_size_doppler": fft_size_doppler,
        "carrier_freq_hz": carrier_freq,
        "carrier_frequency": carrier_freq,
        "cir_time_steps": cir_time_steps,
        "processing_mode": str(processing_mode),
    }


def validate_sensing_contract(
    frame: Mapping[str, Any], *, require_schema_version: bool = False
) -> list[str]:
    """Return contract violations from a frame-like mapping's ``sensing`` key.

    Validation is intentionally structural: it checks field types, accepted
    policy strings, and timing/config scalar types while leaving numerical
    correctness to the producing pipeline. To validate a canonical frame, pass
    ``{"sensing": frame.sensing}``; this helper accepts the wrapper mapping,
    not the ``StandardMPCFrame`` object itself.
    """

    errors: list[str] = []
    sensing = frame.get("sensing")
    if sensing is None:
        return errors
    if not isinstance(sensing, Mapping):
        return ["sensing must be a mapping when present"]

    schema_version = sensing.get("schema_version")
    if require_schema_version and schema_version is None:
        errors.append("sensing.schema_version is required")
    if schema_version is not None and _to_int(schema_version) is None:
        errors.append("sensing.schema_version must be an integer")

    if "range_mode" in sensing:
        mode = sensing.get("range_mode")
        if not isinstance(mode, str):
            errors.append("sensing.range_mode must be a string")
        elif not _is_range_mode_alias(mode):
            errors.append("sensing.range_mode must be monostatic or bistatic")

    if "pair_index" in sensing and _to_int(sensing.get("pair_index")) is None:
        errors.append("sensing.pair_index must be an integer")

    processing_strategy = sensing.get("processing_strategy")
    if processing_strategy is not None and not isinstance(processing_strategy, str):
        errors.append("sensing.processing_strategy must be a string")

    if "raw_payload_policy" in sensing:
        policy = sensing.get("raw_payload_policy")
        if not isinstance(policy, str) or policy not in {"none", "diagnostic", "always"}:
            errors.append("sensing.raw_payload_policy must be one of: none, diagnostic, always")

    config_resolved = sensing.get("config_resolved")
    if config_resolved is not None:
        if not isinstance(config_resolved, Mapping):
            errors.append("sensing.config_resolved must be a mapping")
        else:
            seed_value = config_resolved.get("rng_seed")
            if seed_value is not None and _to_int(seed_value) is None:
                errors.append("sensing.config_resolved.rng_seed must be an integer or null")

    timing = sensing.get("timing")
    if timing is not None:
        if not isinstance(timing, Mapping):
            errors.append("sensing.timing must be a mapping")
        else:
            for key in (
                "frame_dt_s",
                "pri_s",
                "cpi_duration_s",
                "revisit_interval_s",
            ):
                if key in timing and _to_float(timing.get(key)) is None:
                    errors.append(f"sensing.timing.{key} must be numeric")
            for key in ("coherent_samples_per_cpi", "cpi_index", "pri_index_in_cpi"):
                if key in timing and _to_int(timing.get(key)) is None:
                    errors.append(f"sensing.timing.{key} must be an integer")
            if "timing_role" in timing:
                role = timing.get("timing_role")
                if not isinstance(role, str):
                    errors.append("sensing.timing.timing_role must be a string")

    if "rd_fresh" in sensing and not isinstance(sensing.get("rd_fresh"), bool):
        errors.append("sensing.rd_fresh must be a bool")
    if "rp_fresh" in sensing and not isinstance(sensing.get("rp_fresh"), bool):
        errors.append("sensing.rp_fresh must be a bool")

    return errors


def assert_sensing_contract(
    frame: Mapping[str, Any], *, require_schema_version: bool = False
) -> None:
    """Raise ``ValueError`` if the optional sensing extension is malformed."""
    errors = validate_sensing_contract(frame, require_schema_version=require_schema_version)
    if errors:
        raise ValueError("Invalid sensing contract: " + "; ".join(errors))


__all__ = [
    "SENSING_DENSE_DATASET_KEYS",
    "SENSING_DENSE_HDF5_MIN_SCHEMA_VERSION",
    "SENSING_SCHEMA_VERSION",
    "SENSING_STAGE_ARTIFACT_DATASET_ALIASES",
    "SensingResolvedConfig",
    "assert_sensing_contract",
    "normalize_range_mode",
    "resolve_sensing_config",
    "validate_sensing_contract",
]
