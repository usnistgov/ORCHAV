"""Contract helpers for optional ``StandardMPCFrame.sensing`` payloads.

The core frame schema carries path geometry and per-path metrics. The optional
``sensing`` extension adds derived arrays, detection metadata, processing
configuration, and timing fields. This module keeps the frame-facing extension
contract in one place for HDF5 storage, reading, and structural validation. The
protobuf frame codec intentionally does not carry this extension.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, NotRequired, Optional, TypedDict
from uuid import UUID

import numpy as np

from shared.frames.json_codec import dumps_frame_json
from shared.frames.payloads import (
    FRAME_EXTENSION_DENSE_DATASET_KEYS,
    FRAME_EXTENSION_DENSE_HDF5_MIN_SCHEMA_VERSION,
    FRAME_EXTENSION_STAGE_ARTIFACT_DATASET_ALIASES,
)

SENSING_SCHEMA_VERSION = 4
SENSING_DENSE_HDF5_MIN_SCHEMA_VERSION = FRAME_EXTENSION_DENSE_HDF5_MIN_SCHEMA_VERSION
SENSING_STAGE_ARTIFACT_DATASET_ALIASES = FRAME_EXTENSION_STAGE_ARTIFACT_DATASET_ALIASES
SENSING_DENSE_DATASET_KEYS = FRAME_EXTENSION_DENSE_DATASET_KEYS

SENSING_CLOCK_DOMAIN = "scenario"
SENSING_CLOCK_ORIGIN = "scenario_start"
SENSING_CLOCK_UNIT = "s"
SENSING_DEFAULT_REFERENCE_POLICY = "cpi_center"
SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS = frozenset(
    {
        "compatibility_views",
        "cir",
        "slow_time_matrix",
        "range_profile",
        "range_profile_raw",
        "range_doppler_map",
        "range_doppler_map_raw",
        "range_profile_kind",
        "azimuth_axis_deg",
        "elevation_axis_deg",
    }
)
"""Transitional sensing aliases forbidden at the native v4 publication boundary."""


def _contains_ndarray(value: Any, *, _seen: set[int] | None = None) -> bool:
    """Return whether a metadata value contains an ndarray at any depth."""

    if isinstance(value, np.ndarray):
        return True
    if not isinstance(value, (Mapping, list, tuple)):
        return False
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    children = value.values() if isinstance(value, Mapping) else value
    return any(_contains_ndarray(child, _seen=seen) for child in children)


def sensing_v4_undeclared_numeric_payload_keys(sensing: Mapping[str, Any]) -> frozenset[str]:
    """Return ndarray-bearing roots not bound by a schema-v4 descriptor.

    The generic packed-HDF5 v2 writer persists top-level ndarrays as sensing
    datasets and serializes nested ndarrays into JSON metadata. Published v4
    payloads permit only descriptor ``data_key`` arrays and descriptor-bound
    coordinate arrays. Recursing through metadata also catches internal
    scientific payloads such as ``analytical_paths`` before they can bypass
    the typed-product catalog.
    """

    referenced_keys: set[str] = set()
    products = sensing.get("products")
    if isinstance(products, Mapping):
        for raw_descriptor in products.values():
            if not isinstance(raw_descriptor, Mapping):
                continue
            data_key = raw_descriptor.get("data_key")
            if isinstance(data_key, str) and data_key:
                referenced_keys.add(data_key)
            axes = raw_descriptor.get("axes")
            if not isinstance(axes, list):
                continue
            for raw_axis in axes:
                if not isinstance(raw_axis, Mapping):
                    continue
                coordinate_key = raw_axis.get("coordinate_key")
                if isinstance(coordinate_key, str) and coordinate_key:
                    referenced_keys.add(coordinate_key)
                alternate_keys = raw_axis.get("alternate_coordinate_keys")
                if isinstance(alternate_keys, list):
                    referenced_keys.update(
                        key for key in alternate_keys if isinstance(key, str) and key
                    )

    return frozenset(
        key
        for key, value in sensing.items()
        if key not in referenced_keys and _contains_ndarray(value)
    )


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPLETENESS_VALUES = {
    "complete",
    "incomplete_unprocessed",
    "incomplete_processed_by_explicit_policy",
}
_NATIVE_V4_REQUIRED_CONFIG_FIELDS = (
    "contract_version",
    "fft_size_range",
    "fft_size_doppler",
    "sampling_frequency",
    "prf_hz",
    "carrier_freq_hz",
    "range_mode",
    "processing_mode",
    "processing_recipe",
    "acquisition",
    "processing",
    "topology",
)
_RESULT_BEARING_DENSE_KEYS = frozenset(
    key
    for key in SENSING_DENSE_DATASET_KEYS
    if key.startswith("range_doppler_") or key == "range_profile_cfar_mask"
)
_RESULT_BEARING_METADATA_KEYS = frozenset(
    {
        "detections",
        "detections_before_tracking",
        "detections_before_tracking_metadata",
        "detection_filter_metadata",
        "cfar_metadata",
    }
)


class UnsupportedSensingSchemaError(ValueError):
    """Raised when a sensing consumer encounters a pre-v4 payload.

    The packed-HDF5 container is still supported. Only the optional sensing
    extension is incompatible and must be regenerated with current ORCHAV.
    """


class SensingSampleSource(TypedDict):
    """Stable composite reference for one source sample.

    ``source_generation_id`` is available before a frame set is finalized.
    ``source_frame_set_id`` may be added by readers or reprocessors once the
    published manifest identity is actually known; producers must not guess it.
    """

    source_generation_id: str
    scene_frame_id: int | str
    sample_offset: int
    source_frame_set_id: NotRequired[str]
    solve_id: NotRequired[str]


class SensingAcquisitionSample(TypedDict):
    """One ordered expected slow-time slot in a sensing acquisition."""

    sample_index: int
    sample_time_s: float
    valid: bool
    phase_coherent: bool
    source: NotRequired[SensingSampleSource]
    invalid_reason: NotRequired[str]


class SensingClock(TypedDict):
    """Named monotonic scenario-clock declaration."""

    clock_id: str
    domain: str
    origin: str
    origin_time_s: float
    unit: str


class SensingAcquisition(TypedDict):
    """Ordered acquisition membership and its physical-time span."""

    acquisition_id: str
    acquisition_config_id: str
    pair_id: str
    tx_sensor_id: str
    rx_sensor_id: str
    start_time_s: float
    end_time_s: float
    default_reference_time_s: float
    default_reference_policy: str
    expected_sample_count: int
    observed_sample_count: int
    valid_sample_count: int
    completeness: str
    samples: list[SensingAcquisitionSample]


class SensingResultReference(TypedDict):
    """Processing-run identity and physical reference epoch for one result."""

    processing_run_id: str
    processing_recipe_id: str
    reference_time_s: float
    reference_time_policy: str


class SensingDisplayBinding(TypedDict):
    """Explicit mapping from a sensing result to a scene frame for display."""

    scene_frame_id: int | str | None
    policy: str
    reason: str


class SensingTruthReference(TypedDict):
    """Optional evaluation-only truth epoch and association policy."""

    reference_time_s: float
    policy: str
    scene_frame_id: NotRequired[int | str | None]
    source: NotRequired[str]


class SensingTemporalContract(TypedDict):
    """Complete temporal/lineage metadata required by sensing schema v4."""

    clock: SensingClock
    acquisition: SensingAcquisition
    result: NotRequired[SensingResultReference]
    display_binding: NotRequired[SensingDisplayBinding]
    truth: NotRequired[SensingTruthReference]


class SensingAcquisitionPlan(TypedDict):
    """Persisted immutable acquisition semantics resolved before RT."""

    sample_source: str
    rt_solve_stride_scene_steps: int
    sample_period_s: float | None
    sample_rate_hz: float | None
    cpi_length_samples: int
    cpi_hop_samples: int
    overlap_samples: int
    partial_window_policy: str
    phase_contract: str


class SensingProcessingPlan(TypedDict):
    """Persisted explicit processing mode and product claim."""

    mode: str
    strategy: str
    fft_size_doppler: int
    result_stride_samples: int
    product_semantics: str
    evidence_status: str


class SensingTopologyPlan(TypedDict):
    """Persisted device/element topology admitted by a strategy."""

    pair_count: int
    pair_index: int
    tx_device_count: int
    rx_device_count: int
    tx_elements: int
    rx_elements: int
    rx_array_processing: str


class SensingExecutionContract(TypedDict):
    """JSON form of the generator's frozen resolved execution contract."""

    contract_version: int
    acquisition: SensingAcquisitionPlan
    processing: SensingProcessingPlan
    topology: SensingTopologyPlan


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


def canonical_sensing_hash(value: Any) -> str:
    """Return the lowercase SHA-256 identity of canonical frame JSON.

    Acquisition configurations and processing recipes use content identities,
    not occurrence UUIDs. The shared JSON codec normalizes NumPy values and
    sorts mapping keys, so semantically identical mappings hash identically.
    """

    encoded = dumps_frame_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_processing_recipe_id(processing_recipe: Mapping[str, Any]) -> str:
    """Return the cross-tool identity of one immutable processing recipe.

    Acquisition, pair, strategy occurrence, and storage context have their own
    schema-v4 identities.  Recipe identity therefore hashes exactly the
    producer-authored ``config_resolved.processing_recipe`` mapping and no
    surrounding projection envelope.
    """
    if not isinstance(processing_recipe, Mapping) or not processing_recipe:
        raise ValueError("processing_recipe must be a non-empty mapping")
    return canonical_sensing_hash(dict(processing_recipe))


def assert_processing_recipe_identity(sensing: Mapping[str, Any]) -> None:
    """Check a result's recipe identity using only its persisted metadata."""
    temporal = sensing.get("temporal")
    result = temporal.get("result") if isinstance(temporal, Mapping) else None
    if not isinstance(result, Mapping):
        return
    config = sensing.get("config_resolved")
    recipe = config.get("processing_recipe") if isinstance(config, Mapping) else None
    if not isinstance(recipe, Mapping) or not recipe:
        raise ValueError("sensing.config_resolved.processing_recipe must be a non-empty mapping")
    if result.get("processing_recipe_id") != canonical_processing_recipe_id(recipe):
        raise ValueError(
            "sensing.temporal.result.processing_recipe_id must match "
            "sensing.config_resolved.processing_recipe"
        )


def _is_exact_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def canonical_sensing_pair_id(tx_sensor_id: str, rx_sensor_id: str) -> str:
    """Return the stable identity of one ordered sensing TX/RX pair.

    Sensor identifiers name devices, while ``pair_id`` names their ordered
    sensing relationship. No occurrence UUID is needed for the pair.
    """

    if not _is_nonempty_string(tx_sensor_id):
        raise ValueError("tx_sensor_id must be a non-empty string")
    if not _is_nonempty_string(rx_sensor_id):
        raise ValueError("rx_sensor_id must be a non-empty string")
    return canonical_sensing_hash(
        {
            "tx_sensor_id": tx_sensor_id,
            "rx_sensor_id": rx_sensor_id,
        }
    )


def _is_canonical_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _is_opaque_frame_id(value: Any) -> bool:
    return _is_exact_int(value) or _is_nonempty_string(value)


def assert_supported_sensing_schema(sensing: Mapping[str, Any] | None) -> None:
    """Require sensing schema v4 when the optional extension is present.

    This check deliberately says nothing about the generic packed-HDF5 v2
    container. A frame without sensing remains valid, while a pre-v4 sensing
    payload must be regenerated rather than interpreted with guessed semantics.
    """

    if sensing is None:
        return
    if not isinstance(sensing, Mapping):
        raise ValueError("sensing must be a mapping when present")
    # Preserve the generic frame contract's intentional distinction between
    # an absent optional mapping (None) and a present-but-empty mapping ({}).
    # An empty mapping contains no sensing payload to interpret.
    if not sensing:
        return
    version = sensing.get("schema_version")
    if not _is_exact_int(version) or int(version) != SENSING_SCHEMA_VERSION:
        rendered = "missing" if version is None else repr(version)
        raise UnsupportedSensingSchemaError(
            "Unsupported sensing schema version "
            f"{rendered}; sensing schema v{SENSING_SCHEMA_VERSION} is required. "
            "Regenerate this sensing dataset with the current ORCHAV version. "
            "The generic packed-HDF5 container is still supported."
        )


def build_sensing_temporal_contract(
    *,
    clock_id: str,
    acquisition_id: str,
    acquisition_config_id: str,
    pair_id: str,
    tx_sensor_id: str,
    rx_sensor_id: str,
    samples: Sequence[Mapping[str, Any]],
    processing_run_id: str | None = None,
    processing_recipe_id: str | None = None,
    reference_time_s: float | None = None,
    reference_time_policy: str | None = None,
    display_scene_frame_id: int | str | None = None,
    display_policy: str | None = None,
    display_reason: str | None = None,
    completeness: str | None = None,
    truth: Mapping[str, Any] | None = None,
) -> SensingTemporalContract:
    """Build and validate the lean schema-v4 temporal record.

    Counts, acquisition span, and the default CPI-center epoch are derived from
    the ordered samples so producers cannot persist contradictory summaries.
    ``processing_run_id`` is also the result occurrence identity; together
    with ``acquisition_id`` it uniquely names the result without another UUID.
    """

    normalized_samples = [dict(sample) for sample in samples]
    expected_count = len(normalized_samples)
    observed_count = sum(isinstance(sample.get("source"), Mapping) for sample in samples)
    valid_count = sum(sample.get("valid") is True for sample in samples)
    if completeness is None:
        completeness = (
            "complete"
            if expected_count > 0 and observed_count == expected_count == valid_count
            else "incomplete_unprocessed"
        )

    if normalized_samples:
        start_time_s = normalized_samples[0].get("sample_time_s")
        end_time_s = normalized_samples[-1].get("sample_time_s")
        if _is_finite_number(start_time_s) and _is_finite_number(end_time_s):
            default_reference_time_s = (float(start_time_s) + float(end_time_s)) / 2.0
        else:
            # Preserve the malformed values in a validation-friendly record;
            # the assertion below provides field-specific diagnostics.
            default_reference_time_s = math.nan
    else:
        start_time_s = math.nan
        end_time_s = math.nan
        default_reference_time_s = math.nan

    temporal: SensingTemporalContract = {
        "clock": {
            "clock_id": clock_id,
            "domain": SENSING_CLOCK_DOMAIN,
            "origin": SENSING_CLOCK_ORIGIN,
            "origin_time_s": 0.0,
            "unit": SENSING_CLOCK_UNIT,
        },
        "acquisition": {
            "acquisition_id": acquisition_id,
            "acquisition_config_id": acquisition_config_id,
            "pair_id": pair_id,
            "tx_sensor_id": tx_sensor_id,
            "rx_sensor_id": rx_sensor_id,
            "start_time_s": float(start_time_s),
            "end_time_s": float(end_time_s),
            "default_reference_time_s": default_reference_time_s,
            "default_reference_policy": SENSING_DEFAULT_REFERENCE_POLICY,
            "expected_sample_count": expected_count,
            "observed_sample_count": observed_count,
            "valid_sample_count": valid_count,
            "completeness": completeness,
            "samples": normalized_samples,  # type: ignore[typeddict-item]
        },
    }
    result_values = (
        processing_run_id,
        processing_recipe_id,
        reference_time_s,
        reference_time_policy,
    )
    if any(value is not None for value in result_values):
        if not all(value is not None for value in result_values):
            raise ValueError(
                "processing_run_id, processing_recipe_id, reference_time_s, and "
                "reference_time_policy must be supplied together"
            )
        temporal["result"] = {
            "processing_run_id": processing_run_id,  # type: ignore[typeddict-item]
            "processing_recipe_id": processing_recipe_id,  # type: ignore[typeddict-item]
            "reference_time_s": float(reference_time_s),  # type: ignore[arg-type]
            "reference_time_policy": reference_time_policy,  # type: ignore[typeddict-item]
        }

    display_values = (display_policy, display_reason)
    if any(value is not None for value in display_values):
        if not all(value is not None for value in display_values):
            raise ValueError("display_policy and display_reason must be supplied together")
        temporal["display_binding"] = {
            "scene_frame_id": display_scene_frame_id,
            "policy": display_policy,  # type: ignore[typeddict-item]
            "reason": display_reason,  # type: ignore[typeddict-item]
        }
    if truth is not None:
        temporal["truth"] = dict(truth)  # type: ignore[typeddict-item]

    temporal_errors: list[str] = []
    _validate_temporal_contract(
        temporal,
        temporal_errors,
        result_products_present=False,
    )
    if temporal_errors:
        raise ValueError("Invalid sensing temporal contract: " + "; ".join(temporal_errors))
    return temporal


def _required_mapping(
    parent: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{path}.{key} must be a mapping")
        return None
    return value


def _validate_hash(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path} must be a 64-character lowercase SHA-256 hex digest")


def _validate_uuid(value: Any, path: str, errors: list[str]) -> None:
    if not _is_canonical_uuid(value):
        errors.append(f"{path} must be a canonical lowercase UUID string")


def _validate_finite(value: Any, path: str, errors: list[str]) -> float | None:
    if not _is_finite_number(value):
        errors.append(f"{path} must be a finite number of seconds")
        return None
    return float(value)


def _validate_temporal_contract(
    temporal: Any,
    errors: list[str],
    *,
    result_products_present: bool,
) -> None:
    root = "sensing.temporal"
    if not isinstance(temporal, Mapping):
        errors.append(f"{root} must be a mapping")
        return

    clock = _required_mapping(temporal, "clock", root, errors)
    clock_id: str | None = None
    if clock is not None:
        raw_clock_id = clock.get("clock_id")
        _validate_uuid(raw_clock_id, f"{root}.clock.clock_id", errors)
        if isinstance(raw_clock_id, str):
            clock_id = raw_clock_id
        for key, expected in (
            ("domain", SENSING_CLOCK_DOMAIN),
            ("origin", SENSING_CLOCK_ORIGIN),
            ("unit", SENSING_CLOCK_UNIT),
        ):
            if clock.get(key) != expected:
                errors.append(f"{root}.clock.{key} must equal {expected!r}")
        origin_time = _validate_finite(
            clock.get("origin_time_s"),
            f"{root}.clock.origin_time_s",
            errors,
        )
        if origin_time is not None and origin_time != 0.0:
            errors.append(f"{root}.clock.origin_time_s must equal 0.0")

    acquisition = _required_mapping(temporal, "acquisition", root, errors)
    acquisition_start: float | None = None
    acquisition_end: float | None = None
    completeness: Any = None
    if acquisition is not None:
        _validate_uuid(
            acquisition.get("acquisition_id"),
            f"{root}.acquisition.acquisition_id",
            errors,
        )
        _validate_hash(
            acquisition.get("acquisition_config_id"),
            f"{root}.acquisition.acquisition_config_id",
            errors,
        )
        pair_id = acquisition.get("pair_id")
        tx_sensor_id = acquisition.get("tx_sensor_id")
        rx_sensor_id = acquisition.get("rx_sensor_id")
        _validate_hash(pair_id, f"{root}.acquisition.pair_id", errors)
        for key, value in (
            ("tx_sensor_id", tx_sensor_id),
            ("rx_sensor_id", rx_sensor_id),
        ):
            if not _is_nonempty_string(value):
                errors.append(f"{root}.acquisition.{key} must be a non-empty string")
        if _is_nonempty_string(tx_sensor_id) and _is_nonempty_string(rx_sensor_id):
            expected_pair_id = canonical_sensing_pair_id(tx_sensor_id, rx_sensor_id)
            if pair_id != expected_pair_id:
                errors.append(
                    f"{root}.acquisition.pair_id must equal the canonical SHA-256 "
                    "identity of tx_sensor_id and rx_sensor_id"
                )
        acquisition_start = _validate_finite(
            acquisition.get("start_time_s"),
            f"{root}.acquisition.start_time_s",
            errors,
        )
        acquisition_end = _validate_finite(
            acquisition.get("end_time_s"),
            f"{root}.acquisition.end_time_s",
            errors,
        )
        default_reference = _validate_finite(
            acquisition.get("default_reference_time_s"),
            f"{root}.acquisition.default_reference_time_s",
            errors,
        )
        if acquisition.get("default_reference_policy") != SENSING_DEFAULT_REFERENCE_POLICY:
            errors.append(
                f"{root}.acquisition.default_reference_policy must equal "
                f"{SENSING_DEFAULT_REFERENCE_POLICY!r}"
            )

        samples = acquisition.get("samples")
        sample_times: list[float] = []
        observed_count = 0
        valid_count = 0
        source_references: set[tuple[str, int | str, int]] = set()
        if not isinstance(samples, list):
            errors.append(f"{root}.acquisition.samples must be an ordered JSON list")
            samples = []
        if not samples:
            errors.append(f"{root}.acquisition.samples must contain at least one expected slot")

        for position, sample_value in enumerate(samples):
            sample_path = f"{root}.acquisition.samples[{position}]"
            if not isinstance(sample_value, Mapping):
                errors.append(f"{sample_path} must be a mapping")
                continue
            sample_index = sample_value.get("sample_index")
            if not _is_exact_int(sample_index):
                errors.append(f"{sample_path}.sample_index must be an integer")
            elif int(sample_index) != position:
                errors.append(
                    f"{sample_path}.sample_index must equal its ordered position {position}"
                )

            sample_time = _validate_finite(
                sample_value.get("sample_time_s"),
                f"{sample_path}.sample_time_s",
                errors,
            )
            if sample_time is not None:
                if sample_times and sample_time <= sample_times[-1]:
                    errors.append(
                        f"{sample_path}.sample_time_s must be strictly greater than the prior slot"
                    )
                sample_times.append(sample_time)

            valid = sample_value.get("valid")
            if not isinstance(valid, bool):
                errors.append(f"{sample_path}.valid must be a bool")
            elif valid:
                valid_count += 1
            if not isinstance(sample_value.get("phase_coherent"), bool):
                errors.append(f"{sample_path}.phase_coherent must be a bool")

            source = sample_value.get("source")
            if source is None:
                if valid is True:
                    errors.append(f"{sample_path}.source is required for a valid sample")
            elif not isinstance(source, Mapping):
                errors.append(f"{sample_path}.source must be a mapping when present")
            else:
                observed_count += 1
                source_generation_id = source.get("source_generation_id")
                _validate_uuid(
                    source_generation_id,
                    f"{sample_path}.source.source_generation_id",
                    errors,
                )
                if (
                    clock_id is not None
                    and isinstance(source_generation_id, str)
                    and source_generation_id != clock_id
                ):
                    errors.append(
                        f"{sample_path}.source.source_generation_id must match "
                        f"{root}.clock.clock_id"
                    )
                scene_frame_id = source.get("scene_frame_id")
                if not _is_opaque_frame_id(scene_frame_id):
                    errors.append(
                        f"{sample_path}.source.scene_frame_id must be an integer or non-empty string"
                    )
                sample_offset = source.get("sample_offset")
                if not _is_exact_int(sample_offset) or int(sample_offset) < 0:
                    errors.append(
                        f"{sample_path}.source.sample_offset must be a nonnegative integer"
                    )
                for optional_key in ("source_frame_set_id", "solve_id"):
                    if optional_key in source and not _is_nonempty_string(source.get(optional_key)):
                        errors.append(f"{sample_path}.source.{optional_key} must be non-empty")
                if (
                    isinstance(source_generation_id, str)
                    and _is_opaque_frame_id(scene_frame_id)
                    and _is_exact_int(sample_offset)
                ):
                    reference = (
                        source_generation_id,
                        scene_frame_id,
                        int(sample_offset),
                    )
                    if reference in source_references:
                        errors.append(
                            f"{sample_path}.source duplicates an earlier composite source"
                        )
                    source_references.add(reference)

            invalid_reason = sample_value.get("invalid_reason")
            if valid is False and not _is_nonempty_string(invalid_reason):
                errors.append(f"{sample_path}.invalid_reason is required when valid is false")
            if valid is True and "invalid_reason" in sample_value:
                errors.append(f"{sample_path}.invalid_reason is forbidden when valid is true")

        expected_count = acquisition.get("expected_sample_count")
        stored_observed_count = acquisition.get("observed_sample_count")
        stored_valid_count = acquisition.get("valid_sample_count")
        for key, value, actual in (
            ("expected_sample_count", expected_count, len(samples)),
            ("observed_sample_count", stored_observed_count, observed_count),
            ("valid_sample_count", stored_valid_count, valid_count),
        ):
            if not _is_exact_int(value) or int(value) < 0:
                errors.append(f"{root}.acquisition.{key} must be a nonnegative integer")
            elif int(value) != actual:
                errors.append(f"{root}.acquisition.{key} must equal {actual}")

        completeness = acquisition.get("completeness")
        if completeness not in _COMPLETENESS_VALUES:
            errors.append(
                f"{root}.acquisition.completeness must be one of: "
                + ", ".join(sorted(_COMPLETENESS_VALUES))
            )
        elif completeness == "complete" and not (
            len(samples) > 0 and observed_count == len(samples) == valid_count
        ):
            errors.append(
                f"{root}.acquisition.completeness='complete' requires every slot to be observed and valid"
            )
        elif (
            completeness != "complete"
            and len(samples) > 0
            and (observed_count == len(samples) == valid_count)
        ):
            errors.append(
                f"{root}.acquisition.completeness must be 'complete' when every slot is observed and valid"
            )

        if sample_times and len(sample_times) == len(samples):
            first_time = sample_times[0]
            last_time = sample_times[-1]
            if acquisition_start is not None and not math.isclose(
                acquisition_start, first_time, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"{root}.acquisition.start_time_s must equal the first sample time")
            if acquisition_end is not None and not math.isclose(
                acquisition_end, last_time, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"{root}.acquisition.end_time_s must equal the last sample time")
            center_time = (first_time + last_time) / 2.0
            if default_reference is not None and not math.isclose(
                default_reference, center_time, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(
                    f"{root}.acquisition.default_reference_time_s must equal the CPI center"
                )

    raw_result = temporal.get("result")
    result: Mapping[str, Any] | None
    if raw_result is None:
        result = None
        if completeness == "incomplete_processed_by_explicit_policy":
            errors.append(f"{root}.result is required for a processed acquisition")
        if result_products_present:
            errors.append(
                f"{root}.result is required when result-bearing sensing products "
                "or metadata exist"
            )
    elif not isinstance(raw_result, Mapping):
        errors.append(f"{root}.result must be a mapping when present")
        result = None
    else:
        result = raw_result
    if result is not None and completeness == "incomplete_unprocessed":
        errors.append(f"{root}.result is forbidden for an incomplete_unprocessed acquisition")
    result_reference: float | None = None
    if result is not None:
        _validate_uuid(
            result.get("processing_run_id"),
            f"{root}.result.processing_run_id",
            errors,
        )
        _validate_hash(
            result.get("processing_recipe_id"),
            f"{root}.result.processing_recipe_id",
            errors,
        )
        result_reference = _validate_finite(
            result.get("reference_time_s"),
            f"{root}.result.reference_time_s",
            errors,
        )
        if not _is_nonempty_string(result.get("reference_time_policy")):
            errors.append(f"{root}.result.reference_time_policy must be a non-empty string")
        if (
            result_reference is not None
            and acquisition_start is not None
            and acquisition_end is not None
            and not acquisition_start <= result_reference <= acquisition_end
        ):
            errors.append(f"{root}.result.reference_time_s must lie within the acquisition span")

    raw_display = temporal.get("display_binding")
    display: Mapping[str, Any] | None
    if raw_display is None:
        display = None
        if result is not None:
            errors.append(f"{root}.display_binding is required when a result exists")
    elif not isinstance(raw_display, Mapping):
        errors.append(f"{root}.display_binding must be a mapping when present")
        display = None
    else:
        display = raw_display
    if display is not None:
        if result is None:
            errors.append(f"{root}.display_binding is forbidden when no result exists")
        display_frame_id = display.get("scene_frame_id")
        if display_frame_id is not None and not _is_opaque_frame_id(display_frame_id):
            errors.append(
                f"{root}.display_binding.scene_frame_id must be null, an integer, or a non-empty string"
            )
        for key in ("policy", "reason"):
            if not _is_nonempty_string(display.get(key)):
                errors.append(f"{root}.display_binding.{key} must be a non-empty string")

    truth = temporal.get("truth")
    if truth is not None:
        if result is None:
            errors.append(f"{root}.truth is forbidden when no result exists")
        if not isinstance(truth, Mapping):
            errors.append(f"{root}.truth must be a mapping when present")
        else:
            truth_reference = _validate_finite(
                truth.get("reference_time_s"),
                f"{root}.truth.reference_time_s",
                errors,
            )
            if not _is_nonempty_string(truth.get("policy")):
                errors.append(f"{root}.truth.policy must be a non-empty string")
            if "scene_frame_id" in truth:
                truth_scene_id = truth.get("scene_frame_id")
                if truth_scene_id is not None and not _is_opaque_frame_id(truth_scene_id):
                    errors.append(
                        f"{root}.truth.scene_frame_id must be null, an integer, or a non-empty string"
                    )
            if "source" in truth and not _is_nonempty_string(truth.get("source")):
                errors.append(f"{root}.truth.source must be a non-empty string")
            if (
                truth_reference is not None
                and result_reference is not None
                and not math.isclose(
                    truth_reference,
                    result_reference,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                errors.append(f"{root}.truth.reference_time_s must equal the result reference time")


def validate_sensing_contract(
    frame: Mapping[str, Any],
    *,
    require_schema_version: bool = False,
    require_described_products: bool = False,
) -> list[str]:
    """Return contract violations from a frame-like mapping's ``sensing`` key.

    Schema-v4 validation checks temporal, typed-product, target-identity, and
    lineage invariants needed to interpret products scientifically, in addition
    to scalar execution fields that remain in current producer payloads. To validate a canonical
    frame, pass ``{"sensing": frame.sensing}``; this helper accepts the wrapper
    mapping, not the ``StandardMPCFrame`` object itself.

    Explicit pre-v4 payloads raise :class:`UnsupportedSensingSchemaError`
    instead of returning ordinary field errors. A schema-less fragment can
    still be checked when ``require_schema_version`` is false, which is useful
    for configuration validation outside a stored sensing payload.
    """

    errors: list[str] = []
    sensing = frame.get("sensing")
    if sensing is None:
        return errors
    if not isinstance(sensing, Mapping):
        return ["sensing must be a mapping when present"]

    schema_version = sensing.get("schema_version")
    if require_schema_version or schema_version is not None:
        assert_supported_sensing_schema(sensing)
    if schema_version == SENSING_SCHEMA_VERSION:
        from shared.sensing.products import validate_product_descriptors

        config_resolved = sensing.get("config_resolved")
        if not isinstance(config_resolved, Mapping):
            errors.append("sensing schema v4 requires a complete sensing.config_resolved mapping")
        else:
            missing_config_fields = [
                key for key in _NATIVE_V4_REQUIRED_CONFIG_FIELDS if key not in config_resolved
            ]
            if missing_config_fields:
                errors.append(
                    "sensing schema v4 config_resolved is incomplete; missing fields: "
                    + ", ".join(missing_config_fields)
                )

            contract_version = config_resolved.get("contract_version")
            if not _is_exact_int(contract_version) or int(contract_version) not in {1, 2}:
                errors.append("sensing.config_resolved.contract_version must equal integer 1 or 2")
            else:
                from shared.sensing.range_response import validate_execution_range_response

                try:
                    validate_execution_range_response(config_resolved)
                except ValueError as exc:
                    errors.append(str(exc))
            for key in ("fft_size_range", "fft_size_doppler"):
                value = config_resolved.get(key)
                if not _is_exact_int(value) or int(value) < 1:
                    errors.append(f"sensing.config_resolved.{key} must be a positive integer")
            for key in ("sampling_frequency", "prf_hz", "carrier_freq_hz"):
                value = config_resolved.get(key)
                if not _is_finite_number(value) or float(value) <= 0.0:
                    errors.append(f"sensing.config_resolved.{key} must be a finite positive number")
            if config_resolved.get("range_mode") not in {"monostatic", "bistatic"}:
                errors.append(
                    "sensing.config_resolved.range_mode must be exactly monostatic or bistatic"
                )

            processing_recipe = config_resolved.get("processing_recipe")
            if isinstance(processing_recipe, Mapping):
                for window_key in ("window_range", "window_doppler"):
                    top_window = config_resolved.get(window_key)
                    recipe_window = processing_recipe.get(window_key)
                    if (
                        top_window is not None
                        and recipe_window is not None
                        and top_window != recipe_window
                    ):
                        errors.append(
                            f"sensing.config_resolved.{window_key} must agree with "
                            f"processing_recipe.{window_key}"
                        )
                    resolved_window = top_window if top_window is not None else recipe_window
                    if not _is_nonempty_string(resolved_window):
                        errors.append(
                            f"sensing.config_resolved.{window_key} or "
                            f"processing_recipe.{window_key} must be a non-empty string"
                        )

            processing_plan = config_resolved.get("processing")
            if isinstance(processing_plan, Mapping):
                top_fft_doppler = config_resolved.get("fft_size_doppler")
                plan_fft_doppler = processing_plan.get("fft_size_doppler")
                if (
                    _is_exact_int(top_fft_doppler)
                    and _is_exact_int(plan_fft_doppler)
                    and int(top_fft_doppler) != int(plan_fft_doppler)
                ):
                    errors.append(
                        "sensing.config_resolved.fft_size_doppler must mirror "
                        "processing.fft_size_doppler"
                    )

            if "config" in sensing:
                projected_config = sensing.get("config")
                if not isinstance(projected_config, Mapping):
                    errors.append("sensing.config must be a mapping when present")
                elif projected_config != config_resolved:
                    errors.append(
                        "sensing.config and sensing.config_resolved must be exact mirrors"
                    )

        legacy_truth_keys = tuple(
            key
            for key in (
                "gt",
                "gt_state",
                "gt_sensing",
                "target_ranges",
                "target_speeds",
            )
            if key in sensing
        )
        if legacy_truth_keys:
            errors.append(
                "sensing schema v4 forbids positional/alias truth fields: "
                + ", ".join(legacy_truth_keys)
            )
        compatibility_keys = sorted(SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS.intersection(sensing))
        if compatibility_keys:
            errors.append(
                "sensing schema v4 forbids compatibility aliases: " + ", ".join(compatibility_keys)
            )
        if require_described_products:
            undeclared_numeric_keys = sorted(sensing_v4_undeclared_numeric_payload_keys(sensing))
            if undeclared_numeric_keys:
                errors.append(
                    "sensing schema v4 forbids undeclared numeric payloads: "
                    + ", ".join(undeclared_numeric_keys)
                )

        result_products_present = any(
            sensing.get(key) is not None for key in _RESULT_BEARING_DENSE_KEYS
        ) or any(key in sensing for key in _RESULT_BEARING_METADATA_KEYS)
        _validate_temporal_contract(
            sensing.get("temporal"),
            errors,
            result_products_present=result_products_present,
        )
        try:
            assert_processing_recipe_identity(sensing)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        if (
            "detections_before_tracking" in sensing
            or "detections_before_tracking_metadata" in sensing
        ):
            reports = sensing.get("detections_before_tracking")
            metadata = sensing.get("detections_before_tracking_metadata")
            measurement_reports = (
                isinstance(metadata, Mapping)
                and metadata.get("schema_version") == 2
                and metadata.get("coordinate_frame") == "receiver_measurement"
            )
            expected = {
                "schema_version": 2 if measurement_reports else 1,
                "stage": "after_angle_refinement_before_tracking",
                "coordinate_frame": (
                    "receiver_measurement" if measurement_reports else "receiver_local"
                ),
                "tracking_applied": False,
            }
            if not isinstance(metadata, Mapping) or any(
                type(metadata.get(key)) is not type(value) or metadata[key] != value
                for key, value in expected.items()
            ):
                errors.append(
                    "sensing.detections_before_tracking_metadata has an invalid source contract"
                )
            else:
                temporal = sensing.get("temporal")
                result = temporal.get("result") if isinstance(temporal, Mapping) else None
                time_s = metadata.get("reference_time_s")
                if (
                    not _is_finite_number(time_s)
                    or not isinstance(result, Mapping)
                    or time_s != result.get("reference_time_s")
                ):
                    errors.append(
                        "sensing.detections_before_tracking_metadata.reference_time_s must match the result"
                    )
            if not isinstance(reports, list):
                errors.append("sensing.detections_before_tracking must be an explicit list")
            else:
                for index, report in enumerate(reports):
                    if not isinstance(report, Mapping):
                        errors.append(
                            f"sensing.detections_before_tracking[{index}] must be a mapping"
                        )
                        continue
                    if any(str(key).startswith("track_") for key in report):
                        errors.append(
                            f"sensing.detections_before_tracking[{index}] cannot contain track state"
                        )
                    required = {"range_m", "velocity_m_s", "x_m", "y_m", "aoa_az_deg"}
                    if measurement_reports:
                        required = {"range_m", "velocity_m_s"}
                    if (
                        not measurement_reports
                        and isinstance(config_resolved, Mapping)
                        and config_resolved.get("tracker_type") == "cartesian3d"
                    ):
                        required.update({"z_m", "aoa_el_deg"})
                    optional = {"x_m", "y_m", "z_m", "aoa_az_deg", "aoa_el_deg"} & report.keys()
                    if measurement_reports:
                        optional = {key for key in optional if report[key] is not None}
                    for key in required | optional:
                        if not _is_finite_number(report.get(key)):
                            errors.append(
                                f"sensing.detections_before_tracking[{index}].{key} must be finite"
                            )
        errors.extend(
            validate_product_descriptors(
                sensing,
                require_described_data=require_described_products,
            )
        )

        target_catalog = sensing.get("target_catalog")
        target_state = sensing.get("target_state")
        target_truth = sensing.get("truth")
        temporal = sensing.get("temporal")
        temporal_truth = temporal.get("truth") if isinstance(temporal, Mapping) else None
        if any(value is not None for value in (target_catalog, target_state, target_truth)):
            from shared.sensing.targets import (
                validate_target_catalog,
                validate_target_state,
                validate_target_truth,
            )

            errors.extend(validate_target_catalog(target_catalog))
            if target_state is None:
                errors.append(
                    "sensing.target_state is required when a target catalog or truth exists"
                )
            else:
                errors.extend(validate_target_state(target_catalog, target_state))
            if target_truth is not None:
                errors.extend(validate_target_truth(target_catalog, target_truth))
        if target_catalog is not None and target_truth is None and temporal_truth is not None:
            errors.append(
                "sensing.truth snapshot is required when sensing.temporal.truth is present"
            )
        if target_truth is not None and temporal_truth is None:
            errors.append(
                "sensing.temporal.truth is required when the canonical truth snapshot exists"
            )

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
            if "contract_version" in config_resolved:
                if config_resolved.get("contract_version") not in {1, 2}:
                    errors.append("sensing.config_resolved.contract_version must equal 1 or 2")
                acquisition_plan = config_resolved.get("acquisition")
                processing_plan = config_resolved.get("processing")
                topology_plan = config_resolved.get("topology")
                if not isinstance(acquisition_plan, Mapping):
                    errors.append("sensing.config_resolved.acquisition must be a mapping")
                else:
                    sample_source = acquisition_plan.get("sample_source")
                    if sample_source not in {
                        "coherent_one_solve",
                        "independent_solves",
                        "analytic_path_snapshot",
                        "measured_pulse_stream",
                    }:
                        errors.append(
                            "sensing.config_resolved.acquisition.sample_source is invalid"
                        )
                    acquisition_ints: dict[str, int] = {}
                    for key in ("cpi_length_samples", "cpi_hop_samples"):
                        value = acquisition_plan.get(key)
                        if not _is_exact_int(value) or int(value) < 1:
                            errors.append(
                                f"sensing.config_resolved.acquisition.{key} must be a "
                                "positive integer"
                            )
                        else:
                            acquisition_ints[key] = int(value)
                    rt_stride_raw = acquisition_plan.get("rt_solve_stride_scene_steps")
                    if sample_source == "measured_pulse_stream":
                        if rt_stride_raw is not None:
                            errors.append(
                                "measured_pulse_stream acquisition must not claim an RT solve stride"
                            )
                    elif not _is_exact_int(rt_stride_raw) or int(rt_stride_raw) < 1:
                        errors.append(
                            "sensing.config_resolved.acquisition."
                            "rt_solve_stride_scene_steps must be a positive integer"
                        )
                    else:
                        acquisition_ints["rt_solve_stride_scene_steps"] = int(rt_stride_raw)
                    overlap_raw = acquisition_plan.get("overlap_samples")
                    overlap = int(overlap_raw) if _is_exact_int(overlap_raw) else None
                    if overlap is None or overlap < 0:
                        errors.append(
                            "sensing.config_resolved.acquisition.overlap_samples must be a "
                            "nonnegative integer"
                        )
                    cpi_length = acquisition_ints.get("cpi_length_samples")
                    cpi_hop = acquisition_ints.get("cpi_hop_samples")
                    if cpi_length is not None and cpi_hop is not None:
                        if cpi_hop > cpi_length:
                            errors.append(
                                "sensing.config_resolved.acquisition.cpi_hop_samples must "
                                "not exceed cpi_length_samples"
                            )
                        if overlap is not None and overlap != cpi_length - cpi_hop:
                            errors.append(
                                "sensing.config_resolved.acquisition.overlap_samples must "
                                "equal cpi_length_samples - cpi_hop_samples"
                            )
                    sample_period = acquisition_plan.get("sample_period_s")
                    sample_rate = acquisition_plan.get("sample_rate_hz")
                    if sample_source == "analytic_path_snapshot":
                        if sample_period is not None or sample_rate is not None:
                            errors.append(
                                "analytic_path_snapshot acquisition must not claim a "
                                "sample period or sample rate"
                            )
                    elif not (
                        _is_finite_number(sample_period)
                        and float(sample_period) > 0.0
                        and _is_finite_number(sample_rate)
                        and float(sample_rate) > 0.0
                    ):
                        errors.append(
                            "sampled sensing acquisition requires finite positive "
                            "sample_period_s and sample_rate_hz"
                        )
                    elif not math.isclose(
                        float(sample_period) * float(sample_rate),
                        1.0,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    ):
                        errors.append(
                            "sensing acquisition sample_period_s and sample_rate_hz "
                            "must be reciprocal"
                        )
                    rt_stride = acquisition_ints.get("rt_solve_stride_scene_steps")
                    if sample_source == "coherent_one_solve" and (
                        cpi_length is not None and rt_stride != cpi_length
                    ):
                        errors.append("coherent_one_solve RT stride must equal cpi_length_samples")
                    if sample_source != "coherent_one_solve" and rt_stride not in {None, 1}:
                        errors.append(
                            "non-coherent acquisition RT stride must equal one scene step"
                        )
                    if acquisition_plan.get("partial_window_policy") != "retain_unprocessed":
                        errors.append(
                            "sensing.config_resolved.acquisition.partial_window_policy "
                            "must equal retain_unprocessed"
                        )
                    if not _is_nonempty_string(acquisition_plan.get("phase_contract")):
                        errors.append(
                            "sensing.config_resolved.acquisition.phase_contract must be non-empty"
                        )
                if not isinstance(processing_plan, Mapping):
                    errors.append("sensing.config_resolved.processing must be a mapping")
                else:
                    resolved_mode = processing_plan.get("mode")
                    if resolved_mode == "auto":
                        errors.append(
                            "sensing.config_resolved.processing.mode must be explicit, not auto"
                        )
                    if not _is_nonempty_string(resolved_mode):
                        errors.append("sensing.config_resolved.processing.mode must be non-empty")
                    if not _is_nonempty_string(processing_plan.get("strategy")):
                        errors.append(
                            "sensing.config_resolved.processing.strategy must be non-empty"
                        )
                    allowed_mode_strategies = {
                        "coherent_cpi": "coherent_cpi_fft",
                        "sequential_stft": "sequential_stft",
                        "analytic_path_doppler_oracle": "analytic_path_doppler_oracle",
                        "offline_reprocess": "shared_sensing_postprocessor",
                        "measurement_import": "nist_mpc_csv_import",
                    }
                    expected_strategy = allowed_mode_strategies.get(str(resolved_mode))
                    if expected_strategy is None:
                        errors.append(
                            "sensing.config_resolved.processing.mode is not a supported "
                            "explicit v0.2 mode"
                        )
                    elif processing_plan.get("strategy") != expected_strategy:
                        errors.append(
                            "sensing.config_resolved.processing strategy does not match "
                            f"mode {resolved_mode!r}"
                        )
                    fft_size = processing_plan.get("fft_size_doppler")
                    if not _is_exact_int(fft_size) or int(fft_size) < 1:
                        errors.append(
                            "sensing.config_resolved.processing.fft_size_doppler must be "
                            "a positive integer"
                        )
                    result_stride = processing_plan.get("result_stride_samples")
                    if not _is_exact_int(result_stride) or int(result_stride) < 1:
                        errors.append(
                            "sensing.config_resolved.processing.result_stride_samples "
                            "must be a positive integer"
                        )
                    if not _is_nonempty_string(processing_plan.get("product_semantics")):
                        errors.append(
                            "sensing.config_resolved.processing.product_semantics must "
                            "be non-empty"
                        )
                    if not _is_nonempty_string(processing_plan.get("evidence_status")):
                        errors.append(
                            "sensing.config_resolved.processing.evidence_status must be "
                            "non-empty"
                        )
                    if isinstance(acquisition_plan, Mapping):
                        cpi_length_raw = acquisition_plan.get("cpi_length_samples")
                        cpi_hop_raw = acquisition_plan.get("cpi_hop_samples")
                        inherited_processing = config_resolved.get("source_processing")
                        analytic_processing = resolved_mode == "analytic_path_doppler_oracle" or (
                            resolved_mode == "offline_reprocess"
                            and isinstance(inherited_processing, Mapping)
                            and inherited_processing.get("mode") == "analytic_path_doppler_oracle"
                            and inherited_processing.get("strategy")
                            == "analytic_path_doppler_oracle"
                        )
                        analytic_snapshot = (
                            acquisition_plan.get("sample_source") == "analytic_path_snapshot"
                        )
                        if (
                            resolved_mode == "offline_reprocess"
                            and analytic_snapshot != analytic_processing
                        ):
                            errors.append(
                                "offline analytic acquisition requires matching source_processing oracle ancestry"
                            )
                        if (
                            _is_exact_int(fft_size)
                            and _is_exact_int(cpi_length_raw)
                            and not (analytic_snapshot and analytic_processing)
                            and int(fft_size) != int(cpi_length_raw)
                        ):
                            errors.append(
                                "processing fft_size_doppler must equal acquisition "
                                "cpi_length_samples"
                            )
                        if (
                            _is_exact_int(result_stride)
                            and _is_exact_int(cpi_hop_raw)
                            and int(result_stride) != int(cpi_hop_raw)
                        ):
                            errors.append(
                                "processing result_stride_samples must equal acquisition "
                                "cpi_hop_samples"
                            )
                        expected_sources = {
                            "coherent_cpi": "coherent_one_solve",
                            "sequential_stft": "independent_solves",
                            "analytic_path_doppler_oracle": "analytic_path_snapshot",
                            "measurement_import": "measured_pulse_stream",
                        }
                        expected_source = expected_sources.get(str(resolved_mode))
                        if (
                            expected_source is not None
                            and acquisition_plan.get("sample_source") != expected_source
                        ):
                            errors.append(
                                "sensing acquisition sample_source does not match "
                                f"processing mode {resolved_mode!r}"
                            )
                if not isinstance(topology_plan, Mapping):
                    errors.append("sensing.config_resolved.topology must be a mapping")
                else:
                    if topology_plan.get("pair_count") != 1:
                        errors.append("sensing.config_resolved.topology.pair_count must equal 1")
                    if topology_plan.get("pair_index") != 0:
                        errors.append("sensing.config_resolved.topology.pair_index must equal 0")
                    for key in (
                        "tx_device_count",
                        "rx_device_count",
                        "tx_elements",
                        "rx_elements",
                    ):
                        value = topology_plan.get(key)
                        if not _is_exact_int(value) or int(value) < 1:
                            errors.append(
                                f"sensing.config_resolved.topology.{key} must be a "
                                "positive integer"
                            )
                    if topology_plan.get("tx_device_count") != 1:
                        errors.append(
                            "sensing.config_resolved.topology.tx_device_count must equal 1"
                        )
                    if topology_plan.get("rx_device_count") != 1:
                        errors.append(
                            "sensing.config_resolved.topology.rx_device_count must equal 1"
                        )
                    if not _is_nonempty_string(topology_plan.get("rx_array_processing")):
                        errors.append(
                            "sensing.config_resolved.topology.rx_array_processing must "
                            "be non-empty"
                        )
                    if isinstance(processing_plan, Mapping) and processing_plan.get("mode") not in {
                        "coherent_cpi",
                        "offline_reprocess",
                    }:
                        if (
                            topology_plan.get("tx_elements") != 1
                            or topology_plan.get("rx_elements") != 1
                        ):
                            errors.append("non-coherent Phase-2 acquisition must use SISO topology")
                processing_recipe = config_resolved.get("processing_recipe")
                if not isinstance(processing_recipe, Mapping):
                    errors.append("sensing.config_resolved.processing_recipe must be a mapping")
                elif (
                    isinstance(processing_plan, Mapping)
                    and processing_plan.get("mode") == "measurement_import"
                ):
                    acquisition_layout = processing_recipe.get("acquisition_layout")
                    if acquisition_layout not in {"per_pulse", "pregrouped_cpi"}:
                        errors.append(
                            "measurement_import processing_recipe.acquisition_layout must be "
                            "per_pulse or pregrouped_cpi"
                        )
                    temporal = sensing.get("temporal")
                    temporal_acquisition = (
                        temporal.get("acquisition") if isinstance(temporal, Mapping) else None
                    )
                    temporal_result = (
                        temporal.get("result") if isinstance(temporal, Mapping) else None
                    )
                    expected_count = (
                        temporal_acquisition.get("expected_sample_count")
                        if isinstance(temporal_acquisition, Mapping)
                        else None
                    )
                    declared_cpi_length = (
                        acquisition_plan.get("cpi_length_samples")
                        if isinstance(acquisition_plan, Mapping)
                        else None
                    )
                    if acquisition_layout == "per_pulse" and isinstance(temporal, Mapping):
                        if expected_count != 1:
                            errors.append(
                                "per_pulse measurement_import frames must contain exactly one "
                                "raw acquisition sample"
                            )
                        if temporal_result is not None:
                            errors.append(
                                "per_pulse measurement_import frames must not fabricate a "
                                "processed result"
                            )
                    elif (
                        acquisition_layout == "pregrouped_cpi"
                        and isinstance(temporal, Mapping)
                        and _is_exact_int(declared_cpi_length)
                        and expected_count != int(declared_cpi_length)
                    ):
                        errors.append(
                            "pregrouped_cpi measurement_import membership must equal the "
                            "declared cpi_length_samples"
                        )
                config_mode = config_resolved.get("processing_mode")
                if isinstance(processing_plan, Mapping) and config_mode != processing_plan.get(
                    "mode"
                ):
                    errors.append(
                        "sensing.config_resolved.processing_mode must mirror processing.mode"
                    )
                top_strategy = sensing.get("processing_strategy")
                if (
                    top_strategy is not None
                    and isinstance(processing_plan, Mapping)
                    and top_strategy != processing_plan.get("strategy")
                ):
                    errors.append("sensing.processing_strategy must mirror the resolved strategy")

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
    frame: Mapping[str, Any],
    *,
    require_schema_version: bool = False,
    require_described_products: bool = False,
) -> None:
    """Raise ``ValueError`` if the optional sensing extension is malformed."""
    errors = validate_sensing_contract(
        frame,
        require_schema_version=require_schema_version,
        require_described_products=require_described_products,
    )
    if errors:
        raise ValueError("Invalid sensing contract: " + "; ".join(errors))


def finalize_sensing_v4_publication(sensing: dict[str, Any]) -> None:
    """Remove internal numeric payloads and validate one complete v4 result.

    Internal producers may use transitional aliases and intermediate ndarrays,
    but neither is part of the persisted schema-v4 contract. Unsupported older
    sensing schemas fail before mutation so callers retain regeneration
    guidance; generic and schema-less payloads must not call this v4 boundary.
    """

    if not isinstance(sensing, dict):
        raise TypeError("sensing must be a mutable mapping at publication")
    if sensing.get("schema_version") != SENSING_SCHEMA_VERSION:
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
        return
    for key in SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS:
        sensing.pop(key, None)
    for key in sensing_v4_undeclared_numeric_payload_keys(sensing):
        sensing.pop(key, None)
    assert_sensing_contract(
        {"sensing": sensing},
        require_schema_version=True,
        require_described_products=True,
    )


__all__ = [
    "SENSING_CLOCK_DOMAIN",
    "SENSING_CLOCK_ORIGIN",
    "SENSING_CLOCK_UNIT",
    "SENSING_DEFAULT_REFERENCE_POLICY",
    "SENSING_DENSE_DATASET_KEYS",
    "SENSING_DENSE_HDF5_MIN_SCHEMA_VERSION",
    "SENSING_SCHEMA_VERSION",
    "SENSING_STAGE_ARTIFACT_DATASET_ALIASES",
    "SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS",
    "SensingAcquisition",
    "SensingAcquisitionPlan",
    "SensingAcquisitionSample",
    "SensingClock",
    "SensingDisplayBinding",
    "SensingExecutionContract",
    "SensingProcessingPlan",
    "SensingResolvedConfig",
    "SensingResultReference",
    "SensingSampleSource",
    "SensingTemporalContract",
    "SensingTopologyPlan",
    "SensingTruthReference",
    "UnsupportedSensingSchemaError",
    "assert_sensing_contract",
    "assert_supported_sensing_schema",
    "build_sensing_temporal_contract",
    "canonical_processing_recipe_id",
    "assert_processing_recipe_identity",
    "canonical_sensing_pair_id",
    "canonical_sensing_hash",
    "finalize_sensing_v4_publication",
    "normalize_range_mode",
    "resolve_sensing_config",
    "sensing_v4_undeclared_numeric_payload_keys",
    "validate_sensing_contract",
]
