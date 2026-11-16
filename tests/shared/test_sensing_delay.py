"""Contract tests for strict typed delay-domain consumption."""

from __future__ import annotations

from copy import deepcopy
from uuid import UUID, uuid5

import numpy as np
import pytest

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
    canonical_sensing_pair_id,
)
from shared.sensing.delay import (
    append_delay_domain_samples,
    extract_delay_domain_samples,
)
from shared.sensing.products import (
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)

_GENERATION_ID = "00000000-0000-4000-8000-0000000000d1"
_PROCESSING_RUN_ID = "00000000-0000-4000-8000-0000000000d2"


def _delay_payload(
    values: np.ndarray,
    *,
    kind: str,
    sample_times_s: list[float],
    combine: dict | None,
    second_values: np.ndarray | None = None,
) -> dict:
    pair_id = canonical_sensing_pair_id("delay-tx", "delay-rx")
    samples = [
        {
            "sample_index": index,
            "sample_time_s": sample_time_s,
            "valid": True,
            "phase_coherent": kind == "delay_domain_slow_time",
            "source": {
                "source_generation_id": _GENERATION_ID,
                "scene_frame_id": f"delay-scene-{index}",
                "sample_offset": index,
            },
        }
        for index, sample_time_s in enumerate(sample_times_s)
    ]
    sensing = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "range_axis_m": np.asarray([0.0, 1.25, 4.5], dtype=np.float64),
        "rx_element_axis": [
            {"element_index": index, "row": 0, "column": index} for index in range(values.shape[1])
        ],
    }
    sensing["temporal"] = build_sensing_temporal_contract(
        clock_id=_GENERATION_ID,
        acquisition_id=str(
            uuid5(
                UUID(_GENERATION_ID),
                f"delay:{kind}:{sample_times_s}:{values.shape}:{float(np.abs(values).sum())}",
            )
        ),
        acquisition_config_id=canonical_sensing_hash({"fixture": "typed-delay"}),
        pair_id=pair_id,
        tx_sensor_id="delay-tx",
        rx_sensor_id="delay-rx",
        samples=samples,
        processing_run_id=_PROCESSING_RUN_ID,
        processing_recipe_id=canonical_sensing_hash({"recipe": "typed-delay"}),
        reference_time_s=sample_times_s[-1],
        reference_time_policy="latest_sample",
        display_scene_frame_id=samples[-1]["source"]["scene_frame_id"],
        display_policy="exact_sample",
        display_reason="focused typed delay fixture",
    )

    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    element_axis = catalog_axis(
        "rx_element",
        size=values.shape[1],
        catalog_key="rx_element_axis",
        coordinate_semantics="sionna_rx_element_index",
        reference_frame="rx_array_local",
    )
    range_axis = dense_axis(
        "range",
        size=values.shape[-1],
        unit="m",
        coordinate_key="range_axis_m",
        coordinate_semantics="fixture_range_bin_centers",
    )
    if kind == "delay_domain_slow_time":
        axes = (
            pair_axis,
            element_axis,
            dense_axis(
                "slow_time",
                size=values.shape[2],
                unit="s",
                coordinate_key="slow_time_axis_s",
                coordinate_semantics="scenario_clock_sample_time",
                reference_frame="scenario_clock",
            ),
            range_axis,
        )
        data_key = "authored_slow_time"
    else:
        axes = (pair_axis, element_axis, range_axis)
        data_key = "authored_cir"
    sensing[data_key] = np.array(values, copy=True)
    add_product_spec(
        sensing,
        data_key=data_key,
        kind=kind,
        axes=axes,
        representation="complex_voltage",
        unit="1",
        combine=combine,
    )
    if second_values is not None:
        sensing["second_authored_product"] = np.array(second_values, copy=True)
        add_product_spec(
            sensing,
            data_key="second_authored_product",
            kind=kind,
            axes=axes,
            representation="complex_voltage",
            unit="1",
            combine=combine,
        )
    materialize_product_specs(sensing)
    return sensing


def test_extract_delay_domain_cir_uses_typed_value_and_descriptor_epoch() -> None:
    typed = np.asarray([[[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]]])
    sensing = _delay_payload(
        typed,
        kind="delay_domain_cir",
        sample_times_s=[7.25],
        combine={"policy": "siso_identity", "normalization": "identity"},
    )
    sensing["cir"] = np.full(3, 999.0 + 0.0j)

    product = extract_delay_domain_samples(sensing)

    assert product is not None
    assert product.data_key == "authored_cir"
    assert product.pair_id == canonical_sensing_pair_id("delay-tx", "delay-rx")
    np.testing.assert_array_equal(product.samples, typed[0])
    np.testing.assert_array_equal(product.sample_times_s, [7.25])
    np.testing.assert_array_equal(product.range_axis_m, [0.0, 1.25, 4.5])


def test_extract_delay_domain_slow_time_preserves_nonuniform_axis_and_unitary_sum() -> None:
    values = np.asarray(
        [
            [
                [[1.0j, 2.0j, 3.0j], [4.0j, 5.0j, 6.0j], [7.0j, 8.0j, 9.0j]],
                [[2.0j, 4.0j, 6.0j], [8.0j, 10.0j, 12.0j], [14.0j, 16.0j, 18.0j]],
            ]
        ],
        dtype=np.complex64,
    )
    sensing = _delay_payload(
        values,
        kind="delay_domain_slow_time",
        sample_times_s=[0.0, 0.1, 0.35],
        combine={
            "input_axis": "rx_element",
            "policy": "rx_broadside_unitary_voltage_sum",
            "normalization": "one_over_sqrt_element_count",
        },
    )
    sensing["slow_time_matrix"] = np.full((3, 3), 999.0 + 0.0j)

    product = extract_delay_domain_samples(sensing)

    assert product is not None
    np.testing.assert_allclose(product.samples, values[0].sum(axis=0) / np.sqrt(2.0))
    np.testing.assert_array_equal(product.sample_times_s, [0.0, 0.1, 0.35])
    assert product.rx_element_identities == ((0, 0, 0), (1, 0, 1))


def test_extract_delay_domain_rejects_ambiguous_primary_products() -> None:
    values = np.ones((1, 1, 2, 3), dtype=np.complex64)
    sensing = _delay_payload(
        values,
        kind="delay_domain_slow_time",
        sample_times_s=[1.0, 1.5],
        combine={"policy": "siso_identity", "normalization": "identity"},
        second_values=2.0 * values,
    )

    with pytest.raises(ValueError, match="Ambiguous sensing products"):
        extract_delay_domain_samples(sensing)


def test_extract_delay_domain_cir_requires_explicit_combine() -> None:
    sensing = _delay_payload(
        np.ones((1, 1, 3), dtype=np.complex64),
        kind="delay_domain_cir",
        sample_times_s=[3.0],
        combine=None,
    )

    with pytest.raises(ValueError, match="explicit named rx_element combine"):
        extract_delay_domain_samples(sensing)


def test_extract_delay_domain_rejects_zero_element_topology() -> None:
    sensing = _delay_payload(
        np.empty((1, 0, 3), dtype=np.complex64),
        kind="delay_domain_cir",
        sample_times_s=[3.0],
        combine={
            "input_axis": "rx_element",
            "policy": "rx_broadside_unitary_voltage_sum",
            "normalization": "one_over_sqrt_element_count",
        },
    )

    with pytest.raises(ValueError, match="at least one rx_element"):
        extract_delay_domain_samples(sensing)


def test_extract_delay_domain_never_falls_back_to_legacy_arrays() -> None:
    sensing = {
        "temporal": {"result": {"reference_time_s": 1.0}},
        "cir": np.ones(3, dtype=np.complex64),
        "slow_time_matrix": np.ones((2, 3), dtype=np.complex64),
    }

    assert extract_delay_domain_samples(sensing) is None


def test_append_delay_domain_samples_deduplicates_only_identical_epochs() -> None:
    first = extract_delay_domain_samples(
        _delay_payload(
            np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3).astype(np.complex64),
            kind="delay_domain_slow_time",
            sample_times_s=[0.0, 0.1],
            combine={"policy": "siso_identity", "normalization": "identity"},
        )
    )
    assert first is not None
    samples: list[np.ndarray] = []
    sample_times_s: list[float] = []
    assert append_delay_domain_samples(first, samples, sample_times_s) == 2
    assert append_delay_domain_samples(first, samples, sample_times_s) == 0

    conflicting = deepcopy(first)
    object.__setattr__(conflicting, "samples", first.samples + 1.0)
    with pytest.raises(ValueError, match="conflicting persisted values"):
        append_delay_domain_samples(conflicting, samples, sample_times_s)
