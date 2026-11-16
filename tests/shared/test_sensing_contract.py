from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS,
    UnsupportedSensingSchemaError,
    assert_sensing_contract,
    build_sensing_temporal_contract,
    canonical_processing_recipe_id,
    canonical_sensing_hash,
    finalize_sensing_v4_publication,
    resolve_sensing_config,
    validate_sensing_contract,
)
from shared.sensing.products import (
    SENSING_PRODUCT_CONTRACT_VERSION,
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)

_CLOCK_ID = "5d6ac341-f653-4b6d-a856-f4c4efdd86fa"
_ACQUISITION_ID = "3d354378-98ee-4d11-841a-f5588ae10928"
_PROCESSING_RUN_ID = "ba246b78-fdc9-4a4b-a65e-4e9a58a73467"


def _valid_resolved_config(
    *,
    mode: str = "coherent_cpi",
    strategy: str = "coherent_cpi_fft",
    sample_source: str = "coherent_one_solve",
    cpi_length_samples: int = 4,
    cpi_hop_samples: int | None = None,
    sample_rate_hz: float = 1000.0,
) -> dict:
    hop = cpi_length_samples if cpi_hop_samples is None else cpi_hop_samples
    config = {
        "contract_version": 1,
        "fft_size_range": 64,
        "fft_size_doppler": cpi_length_samples,
        "sampling_frequency": 20.0e6,
        "prf_hz": 1000.0,
        "carrier_freq_hz": 3.5e9,
        "range_mode": "monostatic",
        "acquisition": {
            "sample_source": sample_source,
            "rt_solve_stride_scene_steps": (
                cpi_length_samples if sample_source == "coherent_one_solve" else 1
            ),
            "sample_period_s": 1.0 / sample_rate_hz,
            "sample_rate_hz": sample_rate_hz,
            "cpi_length_samples": cpi_length_samples,
            "cpi_hop_samples": hop,
            "overlap_samples": cpi_length_samples - hop,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "explicit_test_phase_contract",
        },
        "processing": {
            "mode": mode,
            "strategy": strategy,
            "fft_size_doppler": cpi_length_samples,
            "result_stride_samples": hop,
            "product_semantics": "explicit_test_product",
            "evidence_status": "explicit_test_evidence",
        },
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "tx_device_count": 1,
            "rx_device_count": 1,
            "tx_elements": 1,
            "rx_elements": 1,
            "rx_array_processing": "siso",
        },
        "processing_mode": mode,
        "processing_recipe": {
            "window_range": "hann",
            "window_doppler": "hann",
        },
    }
    return config


def _valid_sensing() -> dict:
    config_resolved = _valid_resolved_config()
    samples = [
        {
            "sample_index": index,
            "sample_time_s": index * 0.001,
            "valid": True,
            "phase_coherent": True,
            "source": {
                "source_generation_id": _CLOCK_ID,
                # Deliberately gapped opaque IDs; timestamps carry time.
                "scene_frame_id": (100, 101, 103, 104)[index],
                "sample_offset": 0,
            },
        }
        for index in range(4)
    ]
    temporal = build_sensing_temporal_contract(
        clock_id=_CLOCK_ID,
        acquisition_id=_ACQUISITION_ID,
        acquisition_config_id=canonical_sensing_hash({"samples": 4, "prf_hz": 1000.0}),
        pair_id=canonical_sensing_hash(
            {"tx_sensor_id": "contract-tx", "rx_sensor_id": "contract-rx"}
        ),
        tx_sensor_id="contract-tx",
        rx_sensor_id="contract-rx",
        samples=samples,
        processing_run_id=_PROCESSING_RUN_ID,
        processing_recipe_id=canonical_processing_recipe_id(config_resolved["processing_recipe"]),
        reference_time_s=0.0015,
        reference_time_policy="cpi_center",
        display_scene_frame_id=101,
        display_policy="nearest_at_or_before_reference",
        display_reason="Frame 101 is the latest scene state at the result epoch.",
    )
    return {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
        "products": {},
        "processing_strategy": "coherent_cpi_fft",
        "config_resolved": config_resolved,
        "config": deepcopy(config_resolved),
    }


def _incomplete_sensing() -> dict:
    samples = [
        {
            "sample_index": 0,
            "sample_time_s": 0.0,
            "valid": True,
            "phase_coherent": True,
            "source": {
                "source_generation_id": _CLOCK_ID,
                "scene_frame_id": 100,
                "sample_offset": 0,
            },
        },
        {
            "sample_index": 1,
            "sample_time_s": 0.001,
            "valid": False,
            "phase_coherent": False,
            "invalid_reason": "not_yet_observed",
        },
    ]
    temporal = build_sensing_temporal_contract(
        clock_id=_CLOCK_ID,
        acquisition_id=_ACQUISITION_ID,
        acquisition_config_id=canonical_sensing_hash({"samples": 2}),
        pair_id=canonical_sensing_hash(
            {"tx_sensor_id": "contract-tx", "rx_sensor_id": "contract-rx"}
        ),
        tx_sensor_id="contract-tx",
        rx_sensor_id="contract-rx",
        samples=samples,
    )
    config_resolved = _valid_resolved_config(cpi_length_samples=2)
    return {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
        "products": {},
        "processing_strategy": "coherent_cpi_fft",
        "config_resolved": config_resolved,
        "config": deepcopy(config_resolved),
    }


def _valid_execution_contract_sensing() -> dict:
    config_resolved = {
        "contract_version": 1,
        "fft_size_range": 64,
        "fft_size_doppler": 4,
        "sampling_frequency": 20.0e6,
        "prf_hz": 1000.0,
        "carrier_freq_hz": 3.5e9,
        "range_mode": "monostatic",
        "acquisition": {
            "sample_source": "independent_solves",
            "rt_solve_stride_scene_steps": 1,
            "sample_period_s": 0.001,
            "sample_rate_hz": 1000.0,
            "cpi_length_samples": 4,
            "cpi_hop_samples": 1,
            "overlap_samples": 3,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "independent_solve_phase_continuity_controlled_specular_only",
        },
        "processing": {
            "mode": "sequential_stft",
            "strategy": "sequential_stft",
            "fft_size_doppler": 4,
            "result_stride_samples": 1,
            "product_semantics": "cross_solve_cir_doppler_fft",
            "evidence_status": "phase2_controlled_specular_validated_experimental",
        },
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "tx_device_count": 1,
            "rx_device_count": 1,
            "tx_elements": 1,
            "rx_elements": 1,
            "rx_array_processing": "siso",
        },
        "processing_mode": "sequential_stft",
        "processing_recipe": {"window_range": "hann", "window_doppler": "hann"},
    }
    return {
        "processing_strategy": "sequential_stft",
        "config_resolved": config_resolved,
        "config": deepcopy(config_resolved),
    }


def test_resolve_sensing_config_precedence_chain() -> None:
    resolved = resolve_sensing_config(
        sensing_block={
            "sampling_frequency": 42.0,
            "config": {
                "prf_hz": 1200.0,
                "fft_size_range": 512,
                "carrier_freq_hz": 24.0e9,
            },
        },
        scenario_sensing={
            "range_mode": "monostatic",
            "fft_size_doppler": 96,
            "bandwidth": 600.0e6,
        },
        scenario_raytracing={"carrier_frequency_hz": 28.0e9, "cir_time_steps": 16},
        defaults={
            "sampling_frequency": 1.0,
            "prf_hz": 100.0,
            "fft_size_range": 128,
            "fft_size_doppler": 64,
            "carrier_freq_hz": 3.5e9,
        },
    )

    assert resolved["sampling_frequency"] == pytest.approx(42.0)
    assert resolved["prf_hz"] == pytest.approx(1200.0)
    assert resolved["fft_size_range"] == 512
    assert resolved["fft_size_doppler"] == 96
    assert resolved["carrier_freq_hz"] == pytest.approx(24.0e9)
    assert resolved["range_mode"] == "monostatic"
    assert resolved["cir_time_steps"] == 16


def test_resolve_sensing_config_defaults_when_missing() -> None:
    resolved = resolve_sensing_config(
        sensing_block={},
        scenario_sensing={},
        scenario_raytracing={},
        defaults={
            "sampling_frequency": 800.0e6,
            "prf_hz": 900.0,
            "fft_size_range": 256,
            "fft_size_doppler": 32,
            "carrier_freq_hz": 26.0e9,
            "range_mode": "bistatic",
            "cir_time_steps": 8,
        },
    )

    assert resolved["sampling_frequency"] == pytest.approx(800.0e6)
    assert resolved["prf_hz"] == pytest.approx(900.0)
    assert resolved["fft_size_range"] == 256
    assert resolved["fft_size_doppler"] == 32
    assert resolved["carrier_freq_hz"] == pytest.approx(26.0e9)
    assert resolved["range_mode"] == "bistatic"
    assert resolved["cir_time_steps"] == 8


def test_validate_sensing_contract_schema_requirement() -> None:
    with pytest.raises(UnsupportedSensingSchemaError, match="Regenerate this sensing dataset"):
        validate_sensing_contract(
            {"sensing": {"range_mode": "monostatic"}}, require_schema_version=True
        )


def test_validate_sensing_contract_rejects_unknown_range_mode() -> None:
    errors = validate_sensing_contract({"sensing": {"range_mode": "sideways"}})
    assert any("range_mode" in err for err in errors)


def test_assert_sensing_contract_raises_for_invalid_pair_index() -> None:
    sensing = _valid_sensing()
    sensing["pair_index"] = "not-an-int"
    with pytest.raises(ValueError, match="pair_index"):
        assert_sensing_contract(
            {"sensing": sensing},
            require_schema_version=True,
        )


@pytest.mark.parametrize("version", [1, 2, 3, "4", 4.0, True, None])
def test_pre_v4_or_noninteger_sensing_schema_requires_regeneration(version: object) -> None:
    with pytest.raises(
        UnsupportedSensingSchemaError,
        match=r"sensing schema v4 is required.*Regenerate",
    ):
        assert_sensing_contract(
            {"sensing": {"schema_version": version}},
            require_schema_version=True,
        )


def test_canonical_sensing_hash_is_key_order_independent() -> None:
    left = canonical_sensing_hash({"algorithm": "fft", "config": {"n": 4, "window": "hann"}})
    right = canonical_sensing_hash({"config": {"window": "hann", "n": 4}, "algorithm": "fft"})

    assert left == right
    assert len(left) == 64
    assert set(left) <= set("0123456789abcdef")


def test_processing_recipe_identity_hashes_only_the_immutable_recipe() -> None:
    recipe = {"window_doppler": "hann", "window_range": "hann", "threshold": 3.0}

    assert canonical_processing_recipe_id(recipe) == canonical_sensing_hash(recipe)
    with pytest.raises(ValueError, match="non-empty mapping"):
        canonical_processing_recipe_id({})


def test_schema_v4_rejects_a_stale_processing_recipe_identity() -> None:
    sensing = _valid_sensing()
    sensing["config_resolved"]["processing_recipe"]["window_doppler"] = "none"
    sensing["config"] = deepcopy(sensing["config_resolved"])

    errors = validate_sensing_contract({"sensing": sensing})

    assert any("processing_recipe_id must match" in error for error in errors)
    sensing["temporal"]["result"]["processing_recipe_id"] = canonical_processing_recipe_id(
        sensing["config_resolved"]["processing_recipe"]
    )
    assert validate_sensing_contract({"sensing": sensing}) == []


def test_schema_v4_accepts_explicit_times_with_gapped_opaque_source_ids() -> None:
    sensing = _valid_sensing()

    assert validate_sensing_contract({"sensing": sensing}) == []
    samples = sensing["temporal"]["acquisition"]["samples"]
    assert [sample["source"]["scene_frame_id"] for sample in samples] == [100, 101, 103, 104]
    assert [sample["sample_time_s"] for sample in samples] == [0.0, 0.001, 0.002, 0.003]


@pytest.mark.parametrize(
    "missing_field",
    [
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
    ],
)
def test_schema_v4_requires_complete_config_resolved(missing_field: str) -> None:
    sensing = _valid_sensing()
    sensing["config_resolved"].pop(missing_field)
    sensing["config"] = deepcopy(sensing["config_resolved"])

    with pytest.raises(ValueError, match=rf"config_resolved is incomplete.*{missing_field}"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


def test_schema_v4_does_not_accept_config_as_resolved_fallback() -> None:
    sensing = _valid_sensing()
    sensing.pop("config_resolved")

    with pytest.raises(ValueError, match="requires a complete sensing.config_resolved"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


def test_schema_v4_requires_exact_stored_processing_windows() -> None:
    sensing = _valid_sensing()
    sensing["config_resolved"]["processing_recipe"].pop("window_range")
    sensing["config"] = deepcopy(sensing["config_resolved"])

    with pytest.raises(ValueError, match="window_range.*must be a non-empty string"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


@pytest.mark.parametrize("legacy_key", ["gt", "gt_state", "gt_sensing", "target_ranges"])
def test_schema_v4_rejects_legacy_positional_truth_aliases(legacy_key: str) -> None:
    sensing = _valid_sensing()
    sensing[legacy_key] = {}

    with pytest.raises(ValueError, match="forbids positional/alias truth fields"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


@pytest.mark.parametrize("alias_key", sorted(SENSING_V4_FORBIDDEN_COMPATIBILITY_KEYS))
def test_schema_v4_rejects_sensing_compatibility_aliases(alias_key: str) -> None:
    sensing = _valid_sensing()
    sensing[alias_key] = {}

    with pytest.raises(ValueError, match=rf"forbids compatibility aliases:.*{alias_key}"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


@pytest.mark.parametrize(
    ("payload_key", "payload"),
    [
        ("range_doppler_cfar_mask", np.ones((2, 3), dtype=np.uint8)),
        (
            "analytical_paths",
            {
                "amplitudes": np.ones((2,), dtype=np.complex64),
                "delays_s": np.ones((2,), dtype=np.float64),
                "doppler_hz": np.ones((2,), dtype=np.float64),
            },
        ),
    ],
)
def test_schema_v4_rejects_undeclared_numeric_payloads(payload_key: str, payload: object) -> None:
    sensing = _valid_sensing()
    sensing[payload_key] = payload

    with pytest.raises(ValueError, match=rf"forbids undeclared numeric payloads:.*{payload_key}"):
        assert_sensing_contract(
            {"sensing": sensing},
            require_schema_version=True,
            require_described_products=True,
        )


def test_publication_boundary_strips_undeclared_top_level_and_nested_arrays() -> None:
    sensing = _valid_sensing()
    sensing["range_doppler_cfar_mask"] = np.ones((2, 3), dtype=np.uint8)
    sensing["analytical_paths"] = {"doppler_hz": np.ones((2,), dtype=np.float64)}

    finalize_sensing_v4_publication(sensing)

    assert "range_doppler_cfar_mask" not in sensing
    assert "analytical_paths" not in sensing
    assert (
        validate_sensing_contract(
            {"sensing": sensing},
            require_schema_version=True,
            require_described_products=True,
        )
        == []
    )


def test_publication_boundary_keeps_declared_typed_detection_stage_products() -> None:
    sensing = _valid_sensing()
    primary_key = "range_doppler_by_pair"
    input_key = "range_doppler_detection_input_power_by_pair"
    mask_key = "range_doppler_cfar_mask_by_pair"
    sensing["range_axis_m"] = np.arange(64, dtype=np.float64)
    sensing["doppler_axis_hz"] = np.arange(4, dtype=np.float64)
    sensing[primary_key] = np.ones((1, 4, 64), dtype=np.complex64)
    sensing[input_key] = np.ones((1, 4, 64), dtype=np.float64)
    sensing[mask_key] = np.zeros((1, 4, 64), dtype=np.bool_)
    axes = (
        catalog_axis("pair", size=1, catalog_key="pair_axis"),
        dense_axis(
            "doppler_frequency",
            size=4,
            unit="Hz",
            coordinate_key="doppler_axis_hz",
        ),
        dense_axis("range", size=64, unit="m", coordinate_key="range_axis_m"),
    )
    add_product_spec(
        sensing,
        data_key=primary_key,
        kind="range_doppler",
        axes=axes,
        representation="complex_voltage",
        unit="1",
    )
    add_product_spec(
        sensing,
        data_key=input_key,
        kind="range_doppler_detection_input_power",
        axes=axes,
        representation="power_linear",
        unit="1",
        source_data_keys=(primary_key,),
    )
    add_product_spec(
        sensing,
        data_key=mask_key,
        kind="range_doppler_cfar_mask",
        axes=axes,
        representation="boolean_mask",
        unit="1",
        source_data_keys=(input_key,),
    )
    materialize_product_specs(sensing)
    sensing["range_doppler_cfar_mask"] = np.ones((4, 64), dtype=np.uint8)

    finalize_sensing_v4_publication(sensing)

    assert input_key in sensing
    assert mask_key in sensing
    assert "range_doppler_cfar_mask" not in sensing
    input_product_id = sensing["products"][input_key]["product_id"]
    assert sensing["products"][mask_key]["lineage"]["source_product_ids"] == [input_product_id]
    assert (
        validate_sensing_contract(
            {"sensing": sensing},
            require_schema_version=True,
            require_described_products=True,
        )
        == []
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda sensing: sensing["temporal"]["acquisition"]["samples"][1].update(sample_index=7),
            "ordered position 1",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"]["samples"][1].update(
                sample_time_s=0.0
            ),
            "strictly greater",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"]["samples"][1]["source"].update(
                scene_frame_id=100
            ),
            "duplicates an earlier composite source",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].update(
                acquisition_config_id="SHA256:not-canonical"
            ),
            "lowercase SHA-256",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].update(pair_id="0" * 64),
            "canonical SHA-256 identity",
        ),
        (
            lambda sensing: sensing["temporal"]["acquisition"].pop("tx_sensor_id"),
            "tx_sensor_id must be a non-empty string",
        ),
    ],
)
def test_schema_v4_rejects_temporal_integrity_errors(mutation, match: str) -> None:
    sensing = deepcopy(_valid_sensing())
    mutation(sensing)

    with pytest.raises(ValueError, match=match):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


def test_incomplete_unprocessed_acquisition_may_omit_result_and_display() -> None:
    sensing = _incomplete_sensing()
    temporal = sensing["temporal"]

    assert validate_sensing_contract({"sensing": sensing}) == []
    assert "result" not in temporal
    assert "display_binding" not in temporal

    sensing["range_doppler_map"] = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="result is required"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


def test_complete_metadata_only_acquisition_may_omit_result() -> None:
    source_acquisition = _valid_sensing()["temporal"]["acquisition"]
    temporal = build_sensing_temporal_contract(
        clock_id=_CLOCK_ID,
        acquisition_id=_ACQUISITION_ID,
        acquisition_config_id=canonical_sensing_hash({"samples": 4, "metadata_only": True}),
        pair_id=canonical_sensing_hash(
            {"tx_sensor_id": "contract-tx", "rx_sensor_id": "contract-rx"}
        ),
        tx_sensor_id="contract-tx",
        rx_sensor_id="contract-rx",
        samples=source_acquisition["samples"],
    )
    sensing = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
        "products": {},
    }
    config_resolved = _valid_resolved_config()
    sensing["processing_strategy"] = "coherent_cpi_fft"
    sensing["config_resolved"] = config_resolved
    sensing["config"] = deepcopy(config_resolved)

    assert temporal["acquisition"]["completeness"] == "complete"
    assert "result" not in temporal
    assert "display_binding" not in temporal
    assert validate_sensing_contract({"sensing": sensing}) == []


def test_incomplete_unprocessed_acquisition_rejects_result_record() -> None:
    sensing = _incomplete_sensing()
    sensing["temporal"]["result"] = {
        "processing_run_id": _PROCESSING_RUN_ID,
        "processing_recipe_id": canonical_sensing_hash({"algorithm": "forbidden"}),
        "reference_time_s": 0.0,
        "reference_time_policy": "sample_time",
    }
    sensing["temporal"]["display_binding"] = {
        "scene_frame_id": 100,
        "policy": "exact",
        "reason": "forbidden incomplete result",
    }

    with pytest.raises(ValueError, match="forbidden for an incomplete_unprocessed"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("detections", []),
        ("detection_filter_metadata", {}),
        ("cfar_metadata", {}),
    ],
)
def test_result_bearing_metadata_requires_temporal_result(key: str, value: object) -> None:
    sensing = _incomplete_sensing()
    sensing[key] = value

    with pytest.raises(ValueError, match="result-bearing sensing products or metadata"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (
            "display_binding",
            {"scene_frame_id": 100, "policy": "diagnostic", "reason": "no result"},
        ),
        (
            "truth",
            {"reference_time_s": 0.0, "policy": "diagnostic", "scene_frame_id": 100},
        ),
    ],
)
def test_result_binding_and_truth_are_forbidden_without_result(
    key: str,
    value: object,
) -> None:
    sensing = _incomplete_sensing()
    sensing["temporal"][key] = value

    with pytest.raises(ValueError, match=rf"{key} is forbidden when no result exists"):
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)


def test_validate_raw_payload_policy_enum() -> None:
    errors = validate_sensing_contract({"sensing": {"raw_payload_policy": "invalid"}})
    assert any("raw_payload_policy" in err for err in errors)

    assert validate_sensing_contract({"sensing": {"raw_payload_policy": "none"}}) == []
    assert validate_sensing_contract({"sensing": {"raw_payload_policy": "diagnostic"}}) == []
    assert validate_sensing_contract({"sensing": {"raw_payload_policy": "always"}}) == []


def test_resolve_sensing_config_uses_config_resolved_when_config_missing() -> None:
    resolved = resolve_sensing_config(
        sensing_block={
            "config_resolved": {
                "range_mode": "monostatic",
                "bandwidth": 250.0e6,
                "prf_hz": 1200.0,
                "fft_size_range": 96,
                "fft_size_doppler": 48,
                "carrier_freq_hz": 24.5e9,
                "cir_time_steps": 12,
            }
        },
        defaults={
            "sampling_frequency": 1.0,
            "prf_hz": 100.0,
            "fft_size_range": 128,
            "fft_size_doppler": 64,
            "carrier_freq_hz": 3.5e9,
            "range_mode": "bistatic",
            "cir_time_steps": 1,
        },
    )

    assert resolved["range_mode"] == "monostatic"
    assert resolved["sampling_frequency"] == pytest.approx(250.0e6)
    assert resolved["prf_hz"] == pytest.approx(1200.0)
    assert resolved["fft_size_range"] == 96
    assert resolved["fft_size_doppler"] == 48
    assert resolved["carrier_freq_hz"] == pytest.approx(24.5e9)
    assert resolved["cir_time_steps"] == 12


def test_validate_config_resolved_rng_seed_type() -> None:
    errors = validate_sensing_contract({"sensing": {"config_resolved": {"rng_seed": "not-an-int"}}})
    assert any("config_resolved.rng_seed" in err for err in errors)
    assert validate_sensing_contract({"sensing": {"config_resolved": {"rng_seed": 42}}}) == []
    assert validate_sensing_contract({"sensing": {"config_resolved": {"rng_seed": None}}}) == []


def test_v02_execution_contract_accepts_one_exact_resolved_configuration() -> None:
    sensing = _valid_execution_contract_sensing()

    assert validate_sensing_contract({"sensing": sensing}) == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda cfg: cfg["acquisition"].update(cpi_hop_samples=5),
            "must not exceed cpi_length_samples",
        ),
        (
            lambda cfg: cfg["acquisition"].update(overlap_samples=2),
            "overlap_samples must equal",
        ),
        (
            lambda cfg: cfg["acquisition"].update(sample_period_s=0.002),
            "must be reciprocal",
        ),
        (
            lambda cfg: cfg["processing"].update(result_stride_samples=2),
            "result_stride_samples must equal",
        ),
        (
            lambda cfg: cfg["processing"].update(fft_size_doppler=8),
            "fft_size_doppler must equal",
        ),
        (
            lambda cfg: cfg["processing"].update(strategy="coherent_cpi_fft"),
            "strategy does not match",
        ),
        (
            lambda cfg: cfg["topology"].update(rx_elements=2),
            "must use SISO topology",
        ),
    ],
)
def test_v02_execution_contract_rejects_internal_contradictions(
    mutation,
    match: str,
) -> None:
    sensing = _valid_execution_contract_sensing()
    mutation(sensing["config_resolved"])
    sensing["config"] = deepcopy(sensing["config_resolved"])

    with pytest.raises(ValueError, match=match):
        assert_sensing_contract({"sensing": sensing})


def test_validate_timing_block_types() -> None:
    errors = validate_sensing_contract({"sensing": {"timing": {"frame_dt_s": "bad"}}})
    assert any("timing.frame_dt_s" in err for err in errors)
    errors = validate_sensing_contract({"sensing": {"timing": {"coherent_samples_per_cpi": "bad"}}})
    assert any("timing.coherent_samples_per_cpi" in err for err in errors)
    errors = validate_sensing_contract({"sensing": {"timing": {"timing_role": 123}}})
    assert any("timing.timing_role" in err for err in errors)

    assert (
        validate_sensing_contract(
            {
                "sensing": {
                    "timing": {
                        "frame_dt_s": 0.1,
                        "pri_s": 0.01,
                        "coherent_samples_per_cpi": 10,
                        "cpi_duration_s": 0.1,
                        "revisit_interval_s": 0.1,
                        "cpi_index": 2,
                        "pri_index_in_cpi": 9,
                        "timing_role": "cpi_end",
                        "stage_profile_ms": {"rd_fft": 2.5},
                    }
                }
            }
        )
        == []
    )
