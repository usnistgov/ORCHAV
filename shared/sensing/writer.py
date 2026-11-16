"""Shared writer for postprocessed sensing results.

``ColumnarResultWriter`` is the output boundary for the sensing postprocessor.
It attaches reprocessed extension data to complete canonical frames and
publishes them through the shared frame-set writer.

This module does not own the generator's primary scenario-output lifecycle;
it reuses the shared physical frame codec for sensing-derived frame sets.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from shared.extensions.sensing import (
    SENSING_DENSE_DATASET_KEYS,
    SENSING_SCHEMA_VERSION,
    canonical_processing_recipe_id,
    finalize_sensing_v4_publication,
)
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.types import StandardMPCFrame
from shared.sensing.cfr_replay import processing_config_dict
from shared.sensing.detection.pipeline import (
    SNR_CENSOR_CEILING_DB,
    SNR_CENSOR_FLOOR_DB,
    DetectionStageArtifacts,
)
from shared.sensing.products import (
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)
from shared.sensing.signal_processing import C0, apply_detection_velocity_metadata

# Use shared logger if available, else standard logging
try:
    from shared.logging import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_postprocessed_result(result: dict[str, Any]) -> None:
    """Validate the minimal result contract before writing files."""
    rd_map = np.asarray(result.get("rd_map"))
    range_axis = np.asarray(result.get("range_axis"))
    doppler_axis = np.asarray(result.get("doppler_axis"))
    detections = result.get("detections")
    config = result.get("config")

    if rd_map.ndim != 2:
        raise ValueError("result['rd_map'] must be a 2-D array")
    if range_axis.ndim != 1 or range_axis.shape[0] != rd_map.shape[1]:
        raise ValueError("result['range_axis'] must be 1-D and match rd_map range bins")
    if doppler_axis.ndim != 1 or doppler_axis.shape[0] != rd_map.shape[0]:
        raise ValueError("result['doppler_axis'] must be 1-D and match rd_map Doppler bins")
    if detections is None or not isinstance(detections, list):
        raise ValueError("result['detections'] must be a list")
    if config is None or not isinstance(config, dict):
        raise ValueError("result['config'] must be a dict")

    rd_cube = result.get("rd_cube")
    if rd_cube is not None and np.asarray(rd_cube).shape != rd_map.shape:
        raise ValueError("result['rd_cube'] must match rd_map shape when present")
    json.dumps(detections, default=_json_default)
    json.dumps(config, default=_json_default)


# ---------------------------------------------------------------------------
# Stage artifact helpers
# ---------------------------------------------------------------------------


def artifacts_to_dict(
    stages: DetectionStageArtifacts | None,
) -> dict[str, np.ndarray] | None:
    """Copy detection-stage artifacts into a plain dataset dictionary."""

    if stages is None:
        return None
    artifacts = {
        "cfar_input_power": np.array(stages.cfar_input_power, copy=True),
        "cfar_input_power_db": np.array(stages.cfar_input_power_db, copy=True),
        "cfar_mask": np.array(stages.cfar_mask, copy=True),
        "cfar_noise_floor": np.array(stages.cfar_noise_floor, copy=True),
        "cfar_threshold": np.array(stages.cfar_threshold, copy=True),
        "cfar_training_cell_count": np.array(stages.cfar_training_cell_count, copy=True),
        "cfar_eligible_mask": np.array(stages.cfar_eligible_mask, copy=True),
        "nms_mask": np.array(stages.nms_mask, copy=True),
        "snr_map": np.array(stages.snr_map, copy=True),
        "snr_valid_mask": np.array(stages.snr_valid_mask, copy=True),
        "cluster_labels": np.array(stages.cluster_labels, copy=True),
    }
    for key, value in (stages.extra_artifacts or {}).items():
        if value is None:
            continue
        artifacts[key] = np.array(value, copy=True)
    return artifacts


_REPROCESSED_DETECTION_STAGE_PRODUCTS = (
    (
        "cfar_input_power",
        "range_doppler_detection_input_power_by_pair",
        "range_doppler_detection_input_power",
        "power_linear",
        "1",
        ("range_doppler_power_by_pair",),
    ),
    (
        "cfar_noise_floor",
        "range_doppler_cfar_noise_power_by_pair",
        "range_doppler_cfar_noise_power",
        "power_linear",
        "1",
        ("range_doppler_detection_input_power_by_pair",),
    ),
    (
        "cfar_threshold",
        "range_doppler_cfar_threshold_power_by_pair",
        "range_doppler_cfar_threshold_power",
        "power_linear",
        "1",
        (
            "range_doppler_cfar_noise_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
    ),
    (
        "cfar_training_cell_count",
        "range_doppler_cfar_training_cell_count_by_pair",
        "range_doppler_cfar_training_cell_count",
        "integer_count",
        "1",
        ("range_doppler_detection_input_power_by_pair",),
    ),
    (
        "cfar_eligible_mask",
        "range_doppler_cfar_eligible_mask_by_pair",
        "range_doppler_cfar_eligible_mask",
        "boolean_mask",
        "1",
        (
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_training_cell_count_by_pair",
        ),
    ),
    (
        "cfar_mask",
        "range_doppler_cfar_mask_by_pair",
        "range_doppler_cfar_mask",
        "boolean_mask",
        "1",
        (
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_threshold_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
    ),
    (
        "nms_mask",
        "range_doppler_nms_mask_by_pair",
        "range_doppler_nms_mask",
        "boolean_mask",
        "1",
        (
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_mask_by_pair",
            "range_doppler_snr_valid_mask_by_pair",
        ),
    ),
    (
        "snr_map",
        "range_doppler_snr_db_by_pair",
        "range_doppler_snr_db",
        "log_power_ratio_db",
        "dB",
        (
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_cfar_noise_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
    ),
    (
        "snr_valid_mask",
        "range_doppler_snr_valid_mask_by_pair",
        "range_doppler_snr_valid_mask",
        "boolean_mask",
        "1",
        (
            "range_doppler_cfar_noise_power_by_pair",
            "range_doppler_cfar_eligible_mask_by_pair",
        ),
    ),
    (
        "cluster_labels",
        "range_doppler_cluster_labels_by_pair",
        "range_doppler_cluster_labels",
        "integer_label",
        "1",
        (
            "range_doppler_nms_mask_by_pair",
            "range_doppler_snr_db_by_pair",
            "range_doppler_snr_valid_mask_by_pair",
        ),
    ),
)


def _publish_reprocessed_detection_stages(
    sensing: dict[str, Any],
    *,
    result: Mapping[str, Any],
    postprocessing_config: Any,
    axes: tuple[dict[str, Any], ...],
    recipe_id: str,
    evidence_role: str = "observation_derived",
) -> None:
    """Persist one complete typed detector DAG from an offline recipe.

    Stage sidecars remain useful for low-level debugging, but replay needs the
    exact same typed products in the derived frame set.  Partial stage sets are
    not published: a dashboard must never assemble a plausible pipeline from
    arrays that did not belong to one detector execution.
    """

    if not bool(getattr(postprocessing_config, "include_stage_artifacts", False)):
        return
    raw_artifacts = result.get("stage_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        return

    required_names = tuple(item[0] for item in _REPROCESSED_DETECTION_STAGE_PRODUCTS)
    present = tuple(name for name in required_names if name in raw_artifacts)
    if not present:
        return
    missing = tuple(name for name in required_names if name not in raw_artifacts)
    if missing:
        raise ValueError(
            "Reprocessed detector-stage publication requires one complete stage set; "
            + ", ".join(missing)
        )

    expected_shape = np.asarray(result["rd_map"]).shape
    for (
        artifact_name,
        data_key,
        _kind,
        _representation,
        _unit,
        _sources,
    ) in _REPROCESSED_DETECTION_STAGE_PRODUCTS:
        dtype = {
            "power_linear": np.float64,
            "log_power_ratio_db": np.float64,
            "integer_count": np.int32,
            "integer_label": np.int32,
            "boolean_mask": np.bool_,
        }[_representation]
        values = np.asarray(raw_artifacts[artifact_name], dtype=dtype)
        if values.shape != expected_shape:
            raise ValueError(
                f"Reprocessed detector stage {artifact_name!r} has shape {values.shape}; "
                f"expected {expected_shape}"
            )
        sensing[data_key] = np.array(values[np.newaxis, ...], copy=True)

    resolved = sensing.get("config_resolved")
    if not isinstance(resolved, Mapping):
        raise ValueError("Detector-stage publication requires config_resolved")
    source_processing = resolved.get("source_processing")
    source_mode = (
        str(source_processing.get("mode", ""))
        if isinstance(source_processing, Mapping)
        else str(resolved.get("processing_mode", ""))
    ).lower()
    explicit_guard_override = (
        postprocessing_config.pre_cfar_doppler_edge_guard_bins is not None
        or postprocessing_config.pre_cfar_doppler_dc_guard_bins is not None
    )
    if source_mode != "coherent_cpi" and not explicit_guard_override:
        edge_guard_bins = 0
        dc_guard_bins = 0
    else:
        edge_guard_bins = int(resolved.get("pre_cfar_doppler_edge_guard_bins", 0))
        dc_guard_bins = int(resolved.get("pre_cfar_doppler_dc_guard_bins", 0))

    cfar_processing = {
        "algorithm": "two_dimensional_constant_false_alarm_rate",
        "input_domain": "power_linear",
        "method": str(getattr(postprocessing_config, "cfar_type", "ca")).lower(),
        "threshold_scale": float(postprocessing_config.cfar_threshold_scale),
        "threshold_semantics": "raw_linear_power_multiplier",
        "max_dynamic_range_db": getattr(postprocessing_config, "cfar_max_dynamic_range_db", None),
        "power_floor_reference": result.get("detection_filter_metadata", {}).get(
            "cfar_power_floor_reference",
            "strongest_guarded_cell_per_detector_input_or_beam",
        ),
        "os_rank_fraction": float(postprocessing_config.cfar_os_rank),
        "os_rank_rounding": "nearest_integer_ties_to_even_of_fraction_times_k_minus_one",
        "edge_policy": str(postprocessing_config.cfar_edge_policy),
        "cut_comparison": "strict_greater_than_threshold",
        "guard_cells": {
            "doppler": int(postprocessing_config.cfar_guard_cells_doppler),
            "range": int(postprocessing_config.cfar_guard_cells_range),
        },
        "reference_cells": {
            "doppler": int(postprocessing_config.cfar_ref_cells_doppler),
            "range": int(postprocessing_config.cfar_ref_cells_range),
        },
    }
    input_processing = {
        "operation": "detector_input_power_projection",
        "input_domain": "power_linear",
        "output_domain": "power_linear",
        "pre_cfar_doppler_guard": {
            "operation": "zero_selected_doppler_rows_before_detection",
            "edge_guard_bins_each_side": edge_guard_bins,
            "dc_guard_bins_each_side": dc_guard_bins,
            "fill_value": 0.0,
        },
    }
    nms_processing = {
        "algorithm": "greedy_local_maximum_nonmaximum_suppression_2d",
        "candidate_mask": "cfar_mask and snr_valid_mask",
        "local_maximum_neighborhood_bins": {"doppler": 3, "range": 3},
        "candidate_order": "descending_input_power_via_numpy_argsort",
        "suppression_radius_bins": {
            "doppler": int(postprocessing_config.nms_radius_doppler),
            "range": int(postprocessing_config.nms_radius_range),
        },
        "boundary_policy": "clip_without_wrap",
    }
    carrier_freq_hz = float(resolved["carrier_freq_hz"])
    cluster_processing = {
        "algorithm": "dbscan_on_gate_normalized_range_velocity_coordinates",
        "input_mask": "nms_mask",
        "range_coordinate": "range_axis_m / cluster_range_gate_m",
        "velocity_coordinate": ("(doppler_axis_hz * wavelength_m / 2) / cluster_velocity_gate_m_s"),
        "cluster_range_gate_m": float(postprocessing_config.cluster_range_gate_m),
        "cluster_velocity_gate_m_s": float(postprocessing_config.cluster_velocity_gate_m_s),
        "wavelength_m": C0 / carrier_freq_hz,
        "velocity_semantics": "monostatic_equivalent_velocity",
        "dbscan_eps": 1.0,
        "dbscan_min_samples": 1,
        "background_label": -1,
        "missing_sklearn_fallback": "connected_components_of_unit_radius_graph",
    }
    processing_by_artifact: dict[str, dict[str, Any]] = {
        "cfar_input_power": input_processing,
        "cfar_noise_floor": {
            **cfar_processing,
            "operation": "training_cell_noise_power_estimate",
            "ineligible_fill_value": 0.0,
        },
        "cfar_training_cell_count": {
            **cfar_processing,
            "operation": "actual_training_cell_count",
            "count_semantics": "training_cells_used_for_each_cell_under_test",
            "ineligible_fill_value": 0,
        },
        "cfar_eligible_mask": {
            **cfar_processing,
            "operation": "training_window_eligibility",
            "true_semantics": "cell_has_training_cells_required_by_edge_policy",
        },
        "cfar_threshold": {
            **cfar_processing,
            "operation": "cfar_threshold_power",
            "formula": (
                "max(noise_power * threshold_scale, input_peak_power * 10**(-max_dynamic_range_db/10))"
                if getattr(postprocessing_config, "cfar_max_dynamic_range_db", None) is not None
                else "noise_power * threshold_scale"
            ),
            "ineligible_fill_value": 0.0,
        },
        "cfar_mask": {
            **cfar_processing,
            "operation": "cfar_cell_under_test_decision",
            "formula": "eligible and input_power > threshold_power and input_power > 0",
        },
        "snr_map": {
            "operation": "cell_power_over_cfar_noise_estimate",
            "formula": "10 * log10(input_power / noise_power)",
            "validity": "eligible_training_window_and_strictly_positive_noise_power",
            "zero_input_valid_value_db": SNR_CENSOR_FLOOR_DB,
            "finite_censor_floor_db": SNR_CENSOR_FLOOR_DB,
            "finite_censor_ceiling_db": SNR_CENSOR_CEILING_DB,
            "invalid_fill_value_db": 0.0,
        },
        "snr_valid_mask": {
            "operation": "snr_denominator_validity",
            "formula": "cfar_eligible and noise_power > 0",
            "invalid_snr_fill_value_db": 0.0,
        },
        "nms_mask": nms_processing,
        "cluster_labels": cluster_processing,
    }
    for (
        artifact_name,
        data_key,
        kind,
        representation,
        unit,
        source_keys,
    ) in _REPROCESSED_DETECTION_STAGE_PRODUCTS:
        beam_scan = getattr(postprocessing_config, "array_detection_mode", "rd") == "beam_scan"
        angular_source = result.get("angular_input_product_id")
        if beam_scan and not isinstance(angular_source, str):
            raise ValueError(
                "beam_scan stage publication requires its exact angular source product ID"
            )
        add_product_spec(
            sensing,
            data_key=data_key,
            kind=kind,
            axes=axes,
            representation=representation,
            unit=unit,
            evidence_role=evidence_role,
            source_data_keys=(
                () if beam_scan and artifact_name == "cfar_input_power" else source_keys
            ),
            source_product_ids=(angular_source,) if beam_scan else (),
            processing={
                "tool": "shared.sensing_postprocessor",
                "recipe_id": recipe_id,
                **processing_by_artifact[artifact_name],
                **(
                    {
                        "execution": "independently_per_visible_beam_before_projection",
                        "stage_projection": "strongest_visible_beam_per_rd_cell",
                    }
                    if beam_scan
                    else {}
                ),
            },
        )


# ---------------------------------------------------------------------------
# Sensing block builder
# ---------------------------------------------------------------------------


_REPROCESSING_ACQUISITION_FIELDS = (
    "sample_source",
    "rt_solve_stride_scene_steps",
    "sample_period_s",
    "sample_rate_hz",
    "cpi_length_samples",
    "cpi_hop_samples",
    "overlap_samples",
    "partial_window_policy",
    "phase_contract",
)

_REPROCESSING_PROCESSING_FIELDS = (
    "mode",
    "strategy",
    "fft_size_doppler",
    "result_stride_samples",
    "product_semantics",
    "evidence_status",
)

_REPROCESSING_TOPOLOGY_FIELDS = (
    "pair_count",
    "pair_index",
    "tx_device_count",
    "rx_device_count",
    "tx_elements",
    "rx_elements",
    "rx_array_processing",
)


def _require_complete_reprocessing_source_config(
    source_sensing: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], int, str, int, str, str]:
    """Return writer-used v4 values without config/default fallback.

    The postprocessor performs the full semantic validation before invoking
    this writer. This publication boundary repeats complete-field checks for
    direct callers so it can never reconstruct output semantics with the
    low-level sensing-config resolver's compatibility defaults.
    """

    if not isinstance(source_sensing, Mapping):
        raise ValueError("Reprocessed schema-v4 output requires source sensing metadata")
    if source_sensing.get("schema_version") != SENSING_SCHEMA_VERSION:
        raise ValueError(
            f"Reprocessed output requires source sensing schema version {SENSING_SCHEMA_VERSION}"
        )
    config = source_sensing.get("config_resolved")
    if not isinstance(config, Mapping):
        raise ValueError(
            "Reprocessed schema-v4 output requires source config_resolved; "
            "source config is not a fallback"
        )
    scalar_fields = (
        "fft_size_range",
        "fft_size_doppler",
        "sampling_frequency",
        "prf_hz",
        "range_mode",
        "carrier_freq_hz",
        "processing_mode",
        "processing_recipe",
        "contract_version",
    )
    missing_scalars = [field for field in scalar_fields if field not in config]
    if missing_scalars:
        raise ValueError(
            "Reprocessed schema-v4 source config_resolved is incomplete; missing fields: "
            + ", ".join(missing_scalars)
        )
    contract_version = config.get("contract_version")
    if (
        isinstance(contract_version, (bool, np.bool_))
        or not isinstance(contract_version, (int, np.integer))
        or int(contract_version) not in {1, 2}
    ):
        raise ValueError("Reprocessed schema-v4 source contract_version must equal 1 or 2")
    from shared.sensing.range_response import validate_execution_range_response

    validate_execution_range_response(config)

    processing_recipe = config["processing_recipe"]
    if not isinstance(processing_recipe, Mapping):
        raise ValueError(
            "Reprocessed schema-v4 source config_resolved.processing_recipe must be a mapping"
        )
    source_windows: list[str] = []
    for field in ("window_range", "window_doppler"):
        top_value = config.get(field)
        recipe_value = processing_recipe.get(field)
        if top_value is not None and recipe_value is not None and top_value != recipe_value:
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{field} disagrees with "
                f"config_resolved.processing_recipe.{field}"
            )
        value = top_value if top_value is not None else recipe_value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{field} or "
                f"processing_recipe.{field} must be a non-empty string"
            )
        source_windows.append(value.strip())

    for field in ("fft_size_range", "fft_size_doppler"):
        value = config[field]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{field} must be an exact integer"
            )
        if int(value) < 1:
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{field} must be positive"
            )
    for field in ("sampling_frequency", "prf_hz", "carrier_freq_hz"):
        value = config[field]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{field} must be a finite "
                "positive number"
            )
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{field} must be a finite "
                "positive number"
            ) from exc
        if not np.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{field} must be a finite "
                "positive number"
            )

    sections = (
        ("acquisition", _REPROCESSING_ACQUISITION_FIELDS),
        ("processing", _REPROCESSING_PROCESSING_FIELDS),
        ("topology", _REPROCESSING_TOPOLOGY_FIELDS),
    )
    normalized_sections: dict[str, Mapping[str, Any]] = {}
    for section_name, fields in sections:
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{section_name} must be a mapping"
            )
        missing = [field for field in fields if field not in section]
        if missing:
            raise ValueError(
                f"Reprocessed schema-v4 source config_resolved.{section_name} is incomplete; "
                "missing fields: " + ", ".join(missing)
            )
        normalized_sections[section_name] = section

    fft_d_raw = config["fft_size_doppler"]
    if isinstance(fft_d_raw, (bool, np.bool_)) or not isinstance(fft_d_raw, (int, np.integer)):
        raise ValueError("Reprocessed schema-v4 source fft_size_doppler must be an exact integer")
    fft_d = int(fft_d_raw)
    if fft_d < 1:
        raise ValueError("Reprocessed schema-v4 source fft_size_doppler must be positive")
    range_mode = config["range_mode"]
    if range_mode not in {"monostatic", "bistatic"}:
        raise ValueError(
            "Reprocessed schema-v4 source range_mode must be exactly monostatic or bistatic"
        )
    hop_raw = normalized_sections["acquisition"]["cpi_hop_samples"]
    if isinstance(hop_raw, (bool, np.bool_)) or not isinstance(hop_raw, (int, np.integer)):
        raise ValueError(
            "Reprocessed schema-v4 source acquisition.cpi_hop_samples must be an exact integer"
        )
    result_stride_samples = int(hop_raw)
    if result_stride_samples < 1:
        raise ValueError(
            "Reprocessed schema-v4 source acquisition.cpi_hop_samples must be positive"
        )
    return (
        config,
        fft_d,
        str(range_mode),
        result_stride_samples,
        source_windows[0],
        source_windows[1],
    )


def build_reprocessed_sensing_block(
    *,
    source_sensing: Mapping[str, Any] | None,
    result: dict[str, Any],
    postprocessing_config: Any,
    input_kind: str,
    source_display_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sensing block for a reprocessed frame.

    Args:
        source_sensing: Original sensing block from the source frame.
        result: Postprocessor result dict.
        postprocessing_config: ``PostProcessingConfig`` dataclass instance.
        input_kind: Exact schema-v4 source-product ``data_key``.
        source_display_binding: Binding from the source product before its
            physical scene is copied into the derived frame namespace.

    Returns:
        A sensing block dict suitable for HDF5 persistence.
    """
    (
        source_resolved,
        source_fft_d,
        source_range_mode,
        result_stride_samples,
        source_window_range,
        source_window_doppler,
    ) = _require_complete_reprocessing_source_config(source_sensing)
    sensing = deepcopy(dict(source_sensing))
    source_products = (
        source_sensing.get("products") if isinstance(source_sensing, Mapping) else None
    )
    if not isinstance(source_products, Mapping):
        raise ValueError(f"Reprocessed input_kind {input_kind!r} has no exact source descriptor")
    source_descriptor = source_products.get(input_kind)
    if not isinstance(source_descriptor, Mapping):
        raise ValueError(f"Reprocessed input_kind {input_kind!r} has no exact source descriptor")
    if source_descriptor.get("data_key") != input_kind:
        raise ValueError(
            f"Reprocessed input_kind {input_kind!r} source descriptor has a different data_key"
        )
    source_product_id = source_descriptor.get("product_id")
    if (
        not isinstance(source_product_id, str)
        or len(source_product_id) != 64
        or any(character not in "0123456789abcdef" for character in source_product_id)
    ):
        raise ValueError(
            f"Reprocessed input_kind {input_kind!r} has no canonical source product_id"
        )
    source_combine: dict[str, Any] | None = None
    combine = source_descriptor.get("combine")
    if isinstance(combine, Mapping):
        source_combine = deepcopy(dict(combine))

    # Old descriptors are immutable records bound to their original run. Keep
    # their exact IDs as ancestry, but do not silently rebind either their
    # arrays or descriptors to this derived result's temporal run.
    for source_data_key in source_products:
        if isinstance(source_data_key, str):
            sensing.pop(source_data_key, None)
    sensing["products"] = {}
    sensing.pop("_product_specs", None)
    sensing.pop("compatibility_views", None)

    # The immutable source frame remains the authoritative home of every input
    # product. A derived frame stores only products owned by this processing
    # run, plus exact source product IDs in lineage.
    for key in SENSING_DENSE_DATASET_KEYS:
        sensing.pop(key, None)
    for key in (
        "detections",
        "detections_before_tracking",
        "detections_before_tracking_metadata",
        "detection_filter_metadata",
        "cfar_metadata",
        "azimuth_axis_deg",
        "elevation_axis_deg",
        "beamforming_mode",
        "antenna_rows",
        "antenna_cols",
        "postprocessing",
        "gt",
        "gt_state",
        "gt_sensing",
    ):
        sensing.pop(key, None)

    temporal = result.get("temporal")
    if not isinstance(temporal, Mapping):
        raise ValueError("result['temporal'] must contain the schema-v4 temporal contract")

    sensing["schema_version"] = SENSING_SCHEMA_VERSION
    sensing["temporal"] = deepcopy(dict(temporal))
    effective_recipe = result.get("processing_recipe_resolved")
    if not isinstance(effective_recipe, Mapping):
        raise ValueError(
            "result['processing_recipe_resolved'] must describe the effective reprocessing recipe"
        )
    temporal_result = temporal.get("result")
    recipe_id = (
        temporal_result.get("processing_recipe_id")
        if isinstance(temporal_result, Mapping)
        else None
    )
    if recipe_id != canonical_processing_recipe_id(effective_recipe):
        raise ValueError("Temporal processing_recipe_id does not match processing_recipe_resolved")

    result_resolved = deepcopy(dict(source_resolved))
    source_processing = result_resolved.pop("processing", None)
    source_recipe = result_resolved.pop("processing_recipe", None)
    prior_source_processing = result_resolved.pop("source_processing", None)
    prior_source_recipe = result_resolved.pop("source_processing_recipe", None)
    if (
        isinstance(source_processing, Mapping)
        and source_processing.get("mode") == "offline_reprocess"
        and isinstance(prior_source_processing, Mapping)
    ):
        source_processing = prior_source_processing
        if isinstance(prior_source_recipe, Mapping):
            source_recipe = prior_source_recipe
    result_resolved["source_processing"] = deepcopy(source_processing)
    result_resolved["source_processing_recipe"] = deepcopy(source_recipe)
    result_resolved["processing"] = {
        "mode": "offline_reprocess",
        "strategy": "shared_sensing_postprocessor",
        "fft_size_doppler": source_fft_d,
        "result_stride_samples": result_stride_samples,
        "product_semantics": "derived_sensing_result",
        "evidence_status": "derived_recipe_recorded",
    }
    if effective_recipe.get("output_selection") is not None:
        result_resolved["processing"]["output_selection"] = deepcopy(
            effective_recipe["output_selection"]
        )
    result_resolved["processing_mode"] = "offline_reprocess"
    result_resolved["processing_recipe"] = deepcopy(dict(effective_recipe))
    overrides = effective_recipe.get("effective_overrides")
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            if value is not None:
                result_resolved[str(key)] = deepcopy(value)
    # Native tracking settings describe the source result, not this newly
    # detected collection. In particular, disabling tracking must make the
    # derived observations usable without inheriting native confirmation.
    tracker_metadata = (result.get("detection_filter_metadata") or {}).get("tracker")
    tracking_applied = bool(
        isinstance(tracker_metadata, Mapping) and tracker_metadata.get("enabled", False)
    )
    result_resolved["track_confirmation_enabled"] = tracking_applied
    if tracking_applied:
        result_resolved["track_confirmation_mode"] = tracker_metadata.get(
            "confirmation_mode", postprocessing_config.tracker_confirmation_mode
        )
    if not tracking_applied:
        result_resolved["tracker_type"] = "none"
        for key in (
            "tracking_coordinate_frame",
            "track_confirmation_m",
            "track_confirmation_n",
            "track_confirmation_mode",
            "track_max_missed_frames",
            "track_association_range_gate_m",
            "track_association_velocity_gate_m_s",
            "track_association_position_gate_m",
            "track_acceleration_psd_m2_s3",
            "track_output_tentative",
        ):
            result_resolved.pop(key, None)
    result_resolved.setdefault("window_range", source_window_range)
    result_resolved.setdefault("window_doppler", source_window_doppler)
    effective_clutter = result.get("rd_processing", {}).get("coherent_clutter_mode")
    if effective_clutter in {"none", "legacy", "windowed_mean"}:
        result_resolved["coherent_clutter_mode"] = effective_clutter
    rebuilt_cfr = result.get("cfr_rebuilt")
    if rebuilt_cfr is not None:
        response_metadata = rebuilt_cfr["metadata"]
        result_resolved.update(
            {
                "fft_size_range": response_metadata["fft_size_range"],
                "range_response": rebuilt_cfr["response_config"],
                "range_response_processing": response_metadata,
                "range_sample_spacing_m": response_metadata["range_sample_spacing_m"],
                "range_resolution_scale_m": response_metadata["range_resolution_scale_m"],
                "coherent_clutter_mode": response_metadata["coherent_clutter_mode"],
            }
        )

    source_detection_mode = source_resolved["processing"]["mode"]
    if source_detection_mode == "offline_reprocess" and isinstance(
        source_resolved.get("source_processing"), Mapping
    ):
        source_detection_mode = source_resolved["source_processing"].get("mode")
    if source_detection_mode == "coherent_cpi":
        for guard_key in (
            "pre_cfar_doppler_edge_guard_bins",
            "pre_cfar_doppler_dc_guard_bins",
        ):
            guard_value = result_resolved.get(guard_key)
            if isinstance(guard_value, (bool, np.bool_)) or not isinstance(
                guard_value, (int, np.integer)
            ):
                raise ValueError(
                    "Reprocessed schema-v4 coherent output requires exact "
                    f"{guard_key} in source config_resolved or effective overrides"
                )
            if int(guard_value) < 0:
                raise ValueError(
                    f"Reprocessed schema-v4 coherent output {guard_key} must be non-negative"
                )
    oracle_derived = source_detection_mode == "analytic_path_doppler_oracle"
    evidence_role = "oracle_derived" if oracle_derived else str(source_descriptor["evidence_role"])
    sensing["config_resolved"] = result_resolved
    # Keep the transitional projection an exact mirror so consumers cannot
    # choose different effective configurations by precedence.
    sensing["config"] = deepcopy(result_resolved)
    sensing["processing_strategy"] = "shared_sensing_postprocessor"
    sensing["rd_source"] = input_kind
    sensing["rd_model"] = "derived_sensing_result"

    # The shared detector canonicalizes its input power to float64. Preserve
    # that precision in the typed parent so an unguarded detector-input stage
    # is exactly replayable rather than descended from a lossy float32 copy.
    rd_power = np.asarray(result["rd_map"], dtype=np.float64)
    range_axis = np.asarray(result["range_axis"], dtype=np.float64)
    doppler_axis = np.asarray(result["doppler_axis"], dtype=np.float64)
    sensing["range_doppler_power_by_pair"] = rd_power[np.newaxis, ...]
    sensing["range_axis_m"] = range_axis
    sensing["doppler_axis_hz"] = doppler_axis
    sensing["detections"] = deepcopy(result["detections"])
    sensing["detection_filter_metadata"] = deepcopy(result.get("detection_filter_metadata") or {})
    if result.get("cfar_metadata") is not None:
        sensing["cfar_metadata"] = deepcopy(result["cfar_metadata"])
    sensing["postprocessing"] = {
        "tool": "shared.sensing_postprocessor",
        "config": processing_config_dict(postprocessing_config),
        "input_kind": input_kind,
        "output_representation": "power_linear",
        "source_product_id": source_product_id,
    }
    if source_display_binding is not None:
        sensing["postprocessing"]["source_display_binding"] = deepcopy(dict(source_display_binding))
    apply_detection_velocity_metadata(
        sensing,
        range_mode=source_range_mode,
    )
    if isinstance(sensing.get("config"), dict):
        apply_detection_velocity_metadata(
            sensing["config"],
            range_mode=source_range_mode,
        )
    if isinstance(sensing.get("config_resolved"), dict):
        apply_detection_velocity_metadata(
            sensing["config_resolved"],
            range_mode=source_range_mode,
        )

    rd_axes = (
        catalog_axis("pair", size=1, catalog_key="pair_axis"),
        dense_axis(
            "doppler_frequency",
            size=rd_power.shape[0],
            unit="Hz",
            coordinate_key="doppler_axis_hz",
            coordinate_semantics="fftshifted_doppler_frequency",
        ),
        dense_axis(
            "range",
            size=rd_power.shape[1],
            unit="m",
            coordinate_key="range_axis_m",
            coordinate_semantics=(
                "monostatic_range" if source_range_mode == "monostatic" else "bistatic_path_length"
            ),
            reference_frame="tx_target_rx_path",
        ),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_power_by_pair",
        kind="range_doppler",
        axes=rd_axes,
        representation="power_linear",
        unit="1",
        source_product_ids=(source_product_id,),
        evidence_role=evidence_role,
        combine=source_combine,
        # The power map and retained complex evidence represent the same
        # rebuilt spectrum. Preserve its actual transform/clutter recipe so
        # readers can compare it with the saved pre-suppression spectrum.
        processing=(
            rebuilt_cfr["metadata"]
            if rebuilt_cfr is not None
            else {
                "tool": "shared.sensing_postprocessor",
                "recipe_id": recipe_id,
                "input_kind": input_kind,
                **result.get("rd_processing", {}),
            }
        ),
    )
    if rebuilt_cfr is not None:
        from shared.sensing.cfr_replay import publish_rebuilt_frequency

        publish_rebuilt_frequency(sensing, rebuilt_cfr)
        if (
            postprocessing_config.array_detection_mode == "beam_scan"
            and rebuilt_cfr.get("spatial") is not None
        ):
            angular_key = "range_doppler_angle_by_pair"
            if postprocessing_config.include_stage_artifacts:
                # The rebuilt detector uses full-precision angular power.
                # Retain that exact parent for its stage trace instead of the
                # display-oriented float32 projection published by default.
                sensing[angular_key] = (abs(rebuilt_cfr["spatial"].spectrum) ** 2)[None]
            # Product IDs depend on the new processing occurrence and exact
            # numeric content. Bind the rebuilt parent before recording the
            # detector-stage ancestry; the upstream angular map has a different
            # transform and must not be substituted here.
            materialize_product_specs(sensing)
            angular_id = sensing["products"][angular_key]["product_id"]
            result = {**result, "angular_input_product_id": angular_id}
            sensing["detection_filter_metadata"]["angular_input_product_id"] = angular_id
    if effective_recipe.get("output_recording_profile") is not None:
        if effective_recipe["output_recording_profile"] != "maps" or rebuilt_cfr is None:
            raise ValueError("Maps retention requires a complete rebuilt frequency observation")
        _retain_replayed_maps(sensing, rebuilt_cfr, postprocessing_config)
    if effective_recipe.get("output_selection") is not None:
        from shared.sensing.window_selection import output_hop_samples

        output_hop_samples(sensing)
    _publish_reprocessed_detection_stages(
        sensing,
        result=result,
        postprocessing_config=postprocessing_config,
        axes=rd_axes,
        recipe_id=recipe_id,
        evidence_role=evidence_role,
    )
    materialize_product_specs(sensing)
    finalize_sensing_v4_publication(sensing)
    return sensing


def _retain_replayed_maps(sensing, rebuilt, postprocessing_config):
    """Publish the existing display kernel before discarding replay-only phase."""
    from shared.sensing.range_response import RangeResponseConfig
    from shared.sensing.recording import add_display_products, apply_recording_profile

    resolved = sensing["config_resolved"]
    source_mode = resolved["source_processing"]["mode"]
    sequential = source_mode == "sequential_stft"
    display_recipe = dict(resolved.get("recording_display") or {})
    selection = resolved["processing"].get("output_selection")
    if selection is not None:
        display_recipe["hop_samples"] = selection["hop_samples"]
    config = SimpleNamespace(
        recording_profile="maps",
        recording_display=display_recipe,
        fft_size_doppler=resolved["fft_size_doppler"],
        cpi_hop_samples=resolved["acquisition"]["cpi_hop_samples"],
        prf_hz=resolved["prf_hz"],
        range_response=RangeResponseConfig.parse(rebuilt["response_config"]),
        noise_enabled=False,
        noise_snr_db=None,
        persist_detection_stage_artifacts=postprocessing_config.include_stage_artifacts,
    )
    # Unit-norm broadside combining matches both native publication and replay.
    delay = rebuilt["delay"]
    frequency = rebuilt["observation"][0]
    add_display_products(
        sensing,
        config,
        samples=delay.sum(axis=0) / np.sqrt(delay.shape[0]),
        frequency_samples=frequency.sum(axis=0) / np.sqrt(frequency.shape[0]),
        times_s=rebuilt["slow_time_axis_s"],
        range_axis_m=rebuilt["range_axis_m"],
        sequential=sequential,
    )
    apply_recording_profile(sensing, config)
    retained = []
    for spec in sensing["_product_specs"]:
        if spec["representation"] == "complex_voltage":
            sensing.pop(spec["data_key"], None)
        else:
            retained.append(spec)
    sensing["_product_specs"] = retained
    sensing["recording"]["canonical_observation"] = None


def build_incomplete_sensing_block(
    *,
    source_sensing: Mapping[str, Any] | None,
    temporal: Mapping[str, Any],
    postprocessing_config: Any,
) -> dict[str, Any]:
    """Retain acquired observations without fabricating a coherent result."""

    _require_complete_reprocessing_source_config(source_sensing)
    sensing = deepcopy(dict(source_sensing))
    acquisition_id = temporal["acquisition"]["acquisition_id"]
    preserve_observations = temporal["acquisition"]["completeness"] == "complete"
    products = sensing.get("products", {})
    # A grouped partial CPI has a new acquisition identity; one representative
    # pulse's products cannot be relabeled as that larger unobserved window.
    for key, descriptor in tuple(products.items()):
        if descriptor.get("lineage", {}).get("acquisition_id") != acquisition_id or (
            not preserve_observations and key in SENSING_DENSE_DATASET_KEYS
        ):
            sensing.pop(key, None)
            products.pop(key)
    retained = set(products)
    if "_product_specs" in sensing:
        sensing["_product_specs"] = [
            spec for spec in sensing["_product_specs"] if spec["data_key"] in retained
        ]
    # Incomplete replay keeps its established product stripping policy. Newer
    # CFR/display products were not stripped, so retain their required axes too.
    for descriptor in sensing.get("products", {}).values():
        for axis in descriptor.get("axes", ()):
            retained.add(axis.get("coordinate_key"))
            retained.update(axis.get("alternate_coordinate_keys", ()))
    for key in SENSING_DENSE_DATASET_KEYS:
        if key not in retained:
            sensing.pop(key, None)
    for key in (
        "cfar_metadata",
        "detections",
        "detections_before_tracking",
        "detections_before_tracking_metadata",
        "detection_filter_metadata",
        "postprocessing",
    ):
        sensing.pop(key, None)
    sensing["schema_version"] = SENSING_SCHEMA_VERSION
    sensing["temporal"] = deepcopy(dict(temporal))
    sensing["postprocessing"] = {
        "tool": "shared.sensing_postprocessor",
        "config": processing_config_dict(postprocessing_config),
        "input_kind": (
            "incomplete_acquisition"
            if temporal["acquisition"]["completeness"] == "incomplete_unprocessed"
            else "off_cadence_acquisition"
        ),
        "status": "not_processed",
        "reason": (
            "incomplete_acquisition"
            if temporal["acquisition"]["completeness"] == "incomplete_unprocessed"
            else "outside_result_cadence"
        ),
    }
    for key in ("gt", "gt_state", "gt_sensing", "truth"):
        sensing.pop(key, None)
    materialize_product_specs(sensing)
    finalize_sensing_v4_publication(sensing)
    return sensing


# ---------------------------------------------------------------------------
# Frame data builder
# ---------------------------------------------------------------------------


def attach_sensing_to_standard_frame(
    source_frame: StandardMPCFrame,
    *,
    sensing_block: dict[str, Any],
    frame_index: int,
) -> StandardMPCFrame:
    """Attach sensing while reusing every unchanged canonical MPC array."""

    provenance = dict(source_frame.provenance or {})
    provenance["frame_idx"] = int(frame_index)
    return replace(
        source_frame,
        frame_index=int(frame_index),
        provenance=provenance,
        sensing=sensing_block,
    )


# ---------------------------------------------------------------------------
# Columnar result writer
# ---------------------------------------------------------------------------


class ColumnarResultWriter:
    """Attach sensing results and publish one new derived frame set.

    The analysis root must already exist and its ``frames`` child must be
    absent. :class:`FrameSetWriter` owns chunk rotation, manifest publication,
    and abort cleanup; this class only performs sensing-specific normalization.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        chunk_size: int = 100,
        compression: str | None = "lzf",
    ) -> None:
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be at least one frame")
        self._output_dir = Path(output_dir)
        if not self._output_dir.is_dir():
            raise FileNotFoundError(
                f"Sensing analysis root must exist before frame output begins: "
                f"{self._output_dir}"
            )
        self._frames_dir = self._output_dir / "frames"
        self._writer = FrameSetWriter.create_new(
            self._frames_dir,
            chunk_size=int(chunk_size),
            compression=compression,
        )
        self._finalized = False

    def add_frame(
        self,
        result: dict[str, Any],
        *,
        frame_index: int,
        source_frame: StandardMPCFrame,
        source_sensing: Mapping[str, Any] | None,
        postprocessing_config: Any,
        input_kind: str,
        source_display_binding: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one postprocessed sensing result to bounded packed output."""
        if self._finalized:
            raise RuntimeError("Cannot add a frame after sensing output was finalized")
        frame_index = int(frame_index)

        sensing_block = build_reprocessed_sensing_block(
            source_sensing=source_sensing,
            result=result,
            postprocessing_config=postprocessing_config,
            input_kind=input_kind,
            source_display_binding=source_display_binding,
        )
        frame_data = attach_sensing_to_standard_frame(
            source_frame,
            sensing_block=sensing_block,
            frame_index=frame_index,
        )

        self._writer.append(frame_data)

    def add_incomplete_acquisition(
        self,
        *,
        temporal: Mapping[str, Any],
        frame_index: int,
        source_frame: StandardMPCFrame,
        source_sensing: Mapping[str, Any] | None,
        postprocessing_config: Any,
    ) -> None:
        """Persist an unprocessed acquisition without fabricating a sensing result."""

        if self._finalized:
            raise RuntimeError("Cannot add a frame after sensing output was finalized")
        sensing_block = build_incomplete_sensing_block(
            source_sensing=source_sensing,
            temporal=temporal,
            postprocessing_config=postprocessing_config,
        )
        self._writer.append(
            attach_sensing_to_standard_frame(
                source_frame,
                sensing_block=sensing_block,
                frame_index=int(frame_index),
            )
        )

    def finalize(self) -> list[dict[str, Any]]:
        """Publish the complete derived frame set and return chunk records."""
        if self._finalized:
            return [chunk.to_dict() for chunk in self._writer.generated_chunks]
        manifest = self._writer.finalize(
            provenance={
                "producer": "shared.sensing_writer.ColumnarResultWriter",
                "output_kind": "reprocessed_sensing",
            }
        )
        if manifest is not None:
            logger.info(
                "Packed sensing output: %d chunk(s) -> %s",
                len(manifest.chunks),
                self._frames_dir,
            )
        self._finalized = True
        return [] if manifest is None else [chunk.to_dict() for chunk in manifest.chunks]

    def abort(self) -> None:
        """Discard the active private chunk after a failed processing run."""

        self._writer.abort()

    def __del__(self) -> None:
        try:
            self.abort()
        except (OSError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# Stage artifacts sidecar
# ---------------------------------------------------------------------------


def save_stage_artifacts(
    output_dir: Path,
    result: dict[str, Any],
    *,
    frame_index: int,
) -> Path | None:
    """Write optional detection-stage arrays beside reprocessed sensing outputs."""

    stage_artifacts = result.get("stage_artifacts")
    if not isinstance(stage_artifacts, dict) or not stage_artifacts:
        return None

    artifacts_dir = output_dir / "stage_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"frame_{frame_index:04d}"
    npz_path = artifacts_dir / f"{prefix}_stage_artifacts.npz"
    np.savez_compressed(npz_path, **stage_artifacts)

    metadata = result.get("detection_filter_metadata")
    if isinstance(metadata, dict):
        with open(artifacts_dir / f"{prefix}_detection_filter_metadata.json", "w") as handle:
            json.dump(metadata, handle, indent=2, default=_json_default)

    return npz_path
