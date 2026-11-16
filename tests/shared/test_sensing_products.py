"""Contract tests for typed sensing products and derivation lineage."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.products import (
    ACQUISITION_SAMPLE_REFERENCE_POLICY,
    SENSING_PRODUCT_CONTRACT_VERSION,
    TEMPORAL_SUPPORT_FULL_ACQUISITION,
    TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
    add_product_spec,
    attach_product_descriptors,
    build_product_descriptor,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
    product_with_coordinates,
    select_product_by_contract,
    validate_product_descriptors,
)

PAIR_ID = "1" * 64
RECIPE_ID = "2" * 64


def _sensing_payload() -> dict[str, object]:
    return {
        "pair_axis": [
            {
                "pair_index": 0,
                "pair_id": PAIR_ID,
                "tx_sensor_id": "tx",
                "rx_sensor_id": "rx",
            }
        ],
        "doppler_axis_hz": np.array([-10.0, 0.0, 10.0]),
        "range_axis_m": np.array([0.0, 1.5, 3.0, 4.5]),
        "range_doppler_by_pair": np.ones((1, 3, 4), dtype=np.complex64),
        "temporal": {
            "clock": {"clock_id": "scenario-clock"},
            "acquisition": {
                "acquisition_id": "41d72ca4-43e2-48fb-a85c-c196366f2ca2",
                "pair_id": PAIR_ID,
                "tx_sensor_id": "tx",
                "rx_sensor_id": "rx",
                "start_time_s": 1.0,
                "end_time_s": 1.2,
                "default_reference_time_s": 1.1,
                "default_reference_policy": "cpi_center",
                "samples": [
                    {"sample_index": 0, "sample_time_s": 1.0, "valid": True},
                    {"sample_index": 1, "sample_time_s": 1.2, "valid": True},
                ],
            },
            "result": {
                "processing_run_id": "d3c58889-3a8a-40d0-9126-42db981d6c99",
                "processing_recipe_id": RECIPE_ID,
                "reference_time_s": 1.1,
                "reference_time_policy": "cpi_center",
            },
        },
    }


def _descriptor(
    sensing: dict[str, object],
    *,
    temporal_support: str = TEMPORAL_SUPPORT_FULL_ACQUISITION,
) -> dict[str, object]:
    return build_product_descriptor(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler_complex",
        axes=(
            catalog_axis("pair", size=1, catalog_key="pair_axis"),
            dense_axis(
                "doppler_frequency",
                size=3,
                unit="Hz",
                coordinate_key="doppler_axis_hz",
            ),
            dense_axis("range", size=4, unit="m", coordinate_key="range_axis_m"),
        ),
        representation="complex_voltage",
        unit="1",
        scale_semantics="relative_unitless",
        pair_ids=(PAIR_ID,),
        temporal_support=temporal_support,
        combine={
            "input_axis": "rx_element",
            "policy": "coherent_voltage_sum",
            "normalization": "none",
        },
        processing={"fft_normalization": "unitary"},
    )


def _range_doppler_axes() -> tuple[dict[str, object], ...]:
    return (
        catalog_axis("pair", size=1, catalog_key="pair_axis"),
        dense_axis(
            "doppler_frequency",
            size=3,
            unit="Hz",
            coordinate_key="doppler_axis_hz",
        ),
        dense_axis("range", size=4, unit="m", coordinate_key="range_axis_m"),
    )


def _build_stage_descriptor(
    values: np.ndarray,
    *,
    representation: str,
    unit: str,
) -> tuple[dict[str, object], dict[str, object]]:
    sensing = _sensing_payload()
    sensing["stage_product"] = values
    descriptor = build_product_descriptor(
        sensing,
        data_key="stage_product",
        kind="test_stage",
        axes=_range_doppler_axes(),
        representation=representation,
        unit=unit,
        scale_semantics="relative_unitless",
        pair_ids=(PAIR_ID,),
    )
    return sensing, descriptor


def test_product_descriptor_is_content_addressed_and_axis_exact() -> None:
    sensing = _sensing_payload()
    descriptor = _descriptor(sensing)
    attach_product_descriptors(sensing, [descriptor])

    data, coordinates, loaded_descriptor = product_with_coordinates(
        sensing, "range_doppler_by_pair"
    )
    assert data.shape == (1, 3, 4)
    assert coordinates[0] == sensing["pair_axis"]
    np.testing.assert_array_equal(coordinates[1], sensing["doppler_axis_hz"])
    np.testing.assert_array_equal(coordinates[2], sensing["range_axis_m"])
    assert loaded_descriptor["product_id"] == descriptor["product_id"]
    assert loaded_descriptor["pair"] == {
        "pair_id": PAIR_ID,
        "tx_sensor_id": "tx",
        "rx_sensor_id": "rx",
    }
    assert loaded_descriptor["time_span"]["clock_id"] == "scenario-clock"
    assert loaded_descriptor["contract_version"] == SENSING_PRODUCT_CONTRACT_VERSION == 2
    assert loaded_descriptor["scale_semantics"] == "relative_unitless"
    assert validate_product_descriptors(sensing, require_described_data=True) == []


def test_product_builder_rejects_unsupported_scale_semantics() -> None:
    sensing = _sensing_payload()

    with pytest.raises(ValueError, match="scale_semantics must be exactly"):
        build_product_descriptor(
            sensing,
            data_key="range_doppler_by_pair",
            kind="range_doppler_complex",
            axes=_range_doppler_axes(),
            representation="complex_voltage",
            unit="1",
            scale_semantics="absolute_scale",
            pair_ids=(PAIR_ID,),
        )


def test_descriptor_validation_rejects_unknown_fields() -> None:
    sensing = _sensing_payload()
    attach_product_descriptors(sensing, [_descriptor(sensing)])
    sensing["products"]["range_doppler_by_pair"]["unexpected_extension"] = "value"

    errors = validate_product_descriptors(sensing, require_described_data=True)

    assert any("unexpected fields ['unexpected_extension']" in error for error in errors)


@pytest.mark.parametrize(
    ("values", "representation", "unit"),
    [
        (np.zeros((1, 3, 4), dtype=np.bool_), "boolean_mask", "1"),
        (np.arange(12, dtype=np.uint16).reshape(1, 3, 4), "integer_count", "1"),
        (np.arange(-1, 11, dtype=np.int16).reshape(1, 3, 4), "integer_label", "1"),
        (np.ones((1, 3, 4), dtype=np.complex64), "complex_voltage", "1"),
        (np.ones((1, 3, 4), dtype=np.float32), "power_linear", "1"),
        (np.zeros((1, 3, 4), dtype=np.float64), "log_power_ratio_db", "dB"),
    ],
    ids=(
        "boolean-mask",
        "integer-count",
        "integer-label",
        "complex-voltage",
        "linear-power",
        "log-power-ratio-db",
    ),
)
def test_product_contract_v2_accepts_exact_representation_dtype_and_unit(
    values: np.ndarray,
    representation: str,
    unit: str,
) -> None:
    sensing, descriptor = _build_stage_descriptor(
        values,
        representation=representation,
        unit=unit,
    )

    attach_product_descriptors(sensing, [descriptor])

    assert descriptor["contract_version"] == 2
    assert validate_product_descriptors(sensing, require_described_data=True) == []


@pytest.mark.parametrize(
    ("values", "representation", "unit", "match"),
    [
        (
            np.zeros((1, 3, 4), dtype=np.uint8),
            "boolean_mask",
            "1",
            "boolean dtype",
        ),
        (
            np.zeros((1, 3, 4), dtype=np.bool_),
            "boolean_mask",
            "dB",
            "requires unit '1'",
        ),
        (
            np.zeros((1, 3, 4), dtype=np.float32),
            "integer_count",
            "1",
            "integer dtype",
        ),
        (
            np.full((1, 3, 4), -1, dtype=np.int16),
            "integer_count",
            "1",
            "must be nonnegative",
        ),
        (
            np.zeros((1, 3, 4), dtype=np.float32),
            "integer_label",
            "1",
            "integer dtype",
        ),
        (
            np.zeros((1, 3, 4), dtype=np.float32),
            "complex_voltage",
            "1",
            "complex floating dtype",
        ),
        (
            np.zeros((1, 3, 4), dtype=np.int16),
            "power_linear",
            "1",
            "real floating dtype",
        ),
        (
            np.full((1, 3, 4), -1.0, dtype=np.float32),
            "power_linear",
            "1",
            "must be nonnegative",
        ),
        (
            np.full((1, 3, 4), np.nan, dtype=np.float32),
            "power_linear",
            "1",
            "must be finite",
        ),
        (
            np.zeros((1, 3, 4), dtype=np.int16),
            "log_power_ratio_db",
            "dB",
            "real floating dtype",
        ),
        (
            np.full((1, 3, 4), -np.inf, dtype=np.float64),
            "log_power_ratio_db",
            "dB",
            "must be finite",
        ),
        (
            np.zeros((1, 3, 4), dtype=np.float64),
            "log_power_ratio_db",
            "1",
            "requires unit 'dB'",
        ),
    ],
)
def test_product_builder_rejects_invalid_representation_dtype_value_or_unit(
    values: np.ndarray,
    representation: str,
    unit: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _build_stage_descriptor(
            values,
            representation=representation,
            unit=unit,
        )


def test_descriptor_validation_rejects_loaded_value_semantics_drift() -> None:
    sensing, descriptor = _build_stage_descriptor(
        np.ones((1, 3, 4), dtype=np.float32),
        representation="power_linear",
        unit="1",
    )
    attach_product_descriptors(sensing, [descriptor])
    sensing["stage_product"] = np.full((1, 3, 4), -1.0, dtype=np.float32)

    errors = validate_product_descriptors(sensing, require_described_data=True)

    assert any("power_linear values must be nonnegative" in error for error in errors)


def test_single_sample_support_uses_last_valid_acquisition_epoch_and_changes_hash() -> None:
    sensing = _sensing_payload()
    full_acquisition = _descriptor(sensing)
    last_sample = _descriptor(
        sensing,
        temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
    )

    assert last_sample["temporal_support"] == {
        "type": "acquisition_sample",
        "selection_policy": TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
        "sample_index": 1,
    }
    assert last_sample["time_span"] == {
        "clock_id": "scenario-clock",
        "start_time_s": 1.2,
        "end_time_s": 1.2,
        "reference_time_s": 1.2,
        "reference_time_policy": ACQUISITION_SAMPLE_REFERENCE_POLICY,
    }
    assert last_sample["product_id"] != full_acquisition["product_id"]

    attach_product_descriptors(sensing, [last_sample])
    assert validate_product_descriptors(sensing, require_described_data=True) == []


def test_sample_support_rejects_invalid_or_non_last_acquisition_binding() -> None:
    sensing = _sensing_payload()
    attach_product_descriptors(
        sensing,
        [
            _descriptor(
                sensing,
                temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            )
        ],
    )
    sensing["temporal"]["acquisition"]["samples"][1]["valid"] = False

    errors = validate_product_descriptors(sensing, require_described_data=True)
    assert any("must name a valid acquisition slot" in error for error in errors)
    assert any("must name the last valid acquisition slot" in error for error in errors)

    no_valid_samples = _sensing_payload()
    for sample in no_valid_samples["temporal"]["acquisition"]["samples"]:
        sample["valid"] = False
    with pytest.raises(ValueError, match="at least one valid acquisition slot"):
        _descriptor(
            no_valid_samples,
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
        )


def test_product_selection_uses_exact_scientific_contract() -> None:
    sensing = _sensing_payload()
    attach_product_descriptors(sensing, [_descriptor(sensing)])

    data_key, values, coordinates, descriptor = select_product_by_contract(
        sensing,
        kind="range_doppler_complex",
        axis_names=("pair", "doppler_frequency", "range"),
        representations=("complex_voltage",),
    )

    assert data_key == "range_doppler_by_pair"
    np.testing.assert_array_equal(values, sensing["range_doppler_by_pair"])
    assert coordinates[0] == sensing["pair_axis"]
    assert descriptor["product_id"] == sensing["products"][data_key]["product_id"]

    with pytest.raises(KeyError, match="No loaded sensing product matches"):
        select_product_by_contract(
            sensing,
            kind="range_doppler_complex",
            axis_names=("pair", "doppler_frequency", "range"),
            representations=("power_linear",),
        )


def test_product_selection_rejects_ambiguous_contract() -> None:
    sensing = _sensing_payload()
    attach_product_descriptors(sensing, [_descriptor(sensing)])
    sensing["relocated_rd"] = sensing["range_doppler_by_pair"]
    sensing["products"]["relocated_rd"] = {
        **sensing["products"]["range_doppler_by_pair"],
        "data_key": "relocated_rd",
    }

    with pytest.raises(ValueError, match="Ambiguous sensing products"):
        select_product_by_contract(
            sensing,
            kind="range_doppler_complex",
            axis_names=("pair", "doppler_frequency", "range"),
            representations=("complex_voltage",),
        )


def test_product_id_changes_with_semantics_but_not_mapping_order() -> None:
    sensing = _sensing_payload()
    first = _descriptor(sensing)
    reordered = {key: first[key] for key in reversed(first)}
    reordered_id = reordered.pop("product_id")
    rebuilt = build_product_descriptor(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler_complex",
        axes=first["axes"],
        representation="complex_voltage",
        unit="1",
        scale_semantics="relative_unitless",
        pair_ids=(PAIR_ID,),
        combine=first["combine"],
        processing=first["processing"],
    )
    assert reordered_id == rebuilt["product_id"] == first["product_id"]

    changed = build_product_descriptor(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler_complex_normalized",
        axes=first["axes"],
        representation="complex_voltage",
        unit="1",
        scale_semantics="relative_unitless",
        pair_ids=(PAIR_ID,),
        combine=first["combine"],
        processing=first["processing"],
    )
    assert changed["product_id"] != first["product_id"]


def test_validation_detects_shape_dtype_coordinate_and_identity_drift() -> None:
    sensing = _sensing_payload()
    attach_product_descriptors(sensing, [_descriptor(sensing)])
    broken = deepcopy(sensing)
    broken["range_doppler_by_pair"] = np.ones((1, 2, 4), dtype=np.complex128)
    broken["doppler_axis_hz"] = np.array([-1.0, 1.0])

    errors = validate_product_descriptors(broken, require_described_data=True)
    assert any("semantic descriptor" in error for error in errors) is False
    assert any("shape does not match" in error for error in errors)
    assert any("dtype does not match" in error for error in errors)
    assert any("has size 2, expected 3" in error for error in errors)


def test_selective_projection_can_retain_descriptor_without_data() -> None:
    sensing = _sensing_payload()
    attach_product_descriptors(sensing, [_descriptor(sensing)])
    sensing.pop("range_doppler_by_pair")

    assert validate_product_descriptors(sensing, require_described_data=False) == []
    with pytest.raises(KeyError, match="is not loaded"):
        product_with_coordinates(sensing, "range_doppler_by_pair")


def test_product_builder_rejects_rank_without_declared_axis() -> None:
    sensing = _sensing_payload()
    with pytest.raises(ValueError, match="rank 3, but 2 axes"):
        build_product_descriptor(
            sensing,
            data_key="range_doppler_by_pair",
            kind="range_doppler_complex",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                dense_axis("range", size=4, unit="m", coordinate_key="range_axis_m"),
            ),
            representation="complex_voltage",
            unit="1",
            scale_semantics="relative_unitless",
            pair_ids=(PAIR_ID,),
        )


def test_product_id_ignores_storage_bindings_and_normalizes_byte_order() -> None:
    original = _sensing_payload()
    first = _descriptor(original)

    relocated = deepcopy(original)
    relocated["relocated_pair_axis"] = relocated.pop("pair_axis")
    relocated["relocated_doppler"] = np.asarray(relocated.pop("doppler_axis_hz"), dtype=">f8")
    relocated["relocated_range"] = np.asarray(relocated.pop("range_axis_m"), dtype=">f8")
    relocated["relocated_rd"] = np.asarray(relocated.pop("range_doppler_by_pair"), dtype=">c8")
    second = build_product_descriptor(
        relocated,
        data_key="relocated_rd",
        kind="range_doppler_complex",
        axes=(
            catalog_axis("pair", size=1, catalog_key="relocated_pair_axis"),
            dense_axis(
                "doppler_frequency",
                size=3,
                unit="Hz",
                coordinate_key="relocated_doppler",
            ),
            dense_axis(
                "range",
                size=4,
                unit="m",
                coordinate_key="relocated_range",
            ),
        ),
        representation="complex_voltage",
        unit="1",
        scale_semantics="relative_unitless",
        pair_ids=(PAIR_ID,),
        combine=first["combine"],
        processing=first["processing"],
    )

    assert second["product_id"] == first["product_id"]
    assert second["data_sha256"] == first["data_sha256"]
    assert second["dtype"] == first["dtype"] == "<c8"


def test_product_builder_rejects_pair_identity_disagreement() -> None:
    sensing = _sensing_payload()
    with pytest.raises(ValueError, match="must match temporal acquisition"):
        build_product_descriptor(
            sensing,
            data_key="range_doppler_by_pair",
            kind="range_doppler_complex",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                dense_axis(
                    "doppler_frequency",
                    size=3,
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                ),
                dense_axis("range", size=4, unit="m", coordinate_key="range_axis_m"),
            ),
            representation="complex_voltage",
            unit="1",
            scale_semantics="relative_unitless",
            pair_ids=("f" * 64,),
        )


def test_materialize_product_specs_resolves_dependencies_independent_of_queue_order() -> None:
    sensing = _sensing_payload()
    sensing["derived_power"] = np.abs(sensing["range_doppler_by_pair"]) ** 2
    add_product_spec(
        sensing,
        data_key="derived_power",
        kind="range_doppler_power",
        axes=_range_doppler_axes(),
        representation="power_linear",
        unit="1",
        source_data_keys=("range_doppler_by_pair",),
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler_complex",
        axes=_range_doppler_axes(),
        representation="complex_voltage",
        unit="1",
    )

    materialize_product_specs(sensing)

    assert list(sensing["products"]) == ["range_doppler_by_pair", "derived_power"]
    assert sensing["products"]["derived_power"]["lineage"]["source_product_ids"] == [
        sensing["products"]["range_doppler_by_pair"]["product_id"]
    ]


def test_materialize_product_specs_rejects_missing_source_data_key() -> None:
    sensing = _sensing_payload()
    sensing["derived_power"] = np.ones((1, 3, 4), dtype=np.float32)
    add_product_spec(
        sensing,
        data_key="derived_power",
        kind="range_doppler_power",
        axes=_range_doppler_axes(),
        representation="power_linear",
        unit="1",
        source_data_keys=("missing_input",),
    )

    with pytest.raises(ValueError, match=r"missing source_data_keys:.*missing_input"):
        materialize_product_specs(sensing)


def test_materialize_product_specs_rejects_source_dependency_cycle() -> None:
    sensing = _sensing_payload()
    sensing["stage_a"] = np.ones((1, 3, 4), dtype=np.float32)
    sensing["stage_b"] = np.ones((1, 3, 4), dtype=np.float32)
    add_product_spec(
        sensing,
        data_key="stage_a",
        kind="test_stage",
        axes=_range_doppler_axes(),
        representation="power_linear",
        unit="1",
        source_data_keys=("stage_b",),
    )
    add_product_spec(
        sensing,
        data_key="stage_b",
        kind="test_stage",
        axes=_range_doppler_axes(),
        representation="power_linear",
        unit="1",
        source_data_keys=("stage_a",),
    )

    with pytest.raises(
        ValueError,
        match=r"dependency cycle: stage_a -> stage_b -> stage_a",
    ):
        materialize_product_specs(sensing)
