"""Tests for the typed Phase-4 detector-stage inspection model."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.products import add_product_spec, materialize_product_specs
from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from visualizer.src.sensing.stage_inspector import (
    DETECTION_STAGE_KINDS,
    CellSelection,
    DetectionStageContractError,
    InspectionDisposition,
    build_detection_stage_bundle,
    load_detection_stage_bundle,
)


def _with_detection(sensing: dict[str, object]) -> dict[str, object]:
    temporal = sensing["temporal"]
    assert isinstance(temporal, dict)
    sensing["detections"] = [
        {
            "cluster_id": 0,
            "doppler_bin": 1,
            "range_bin": 2,
            "range_m": 2.0,
            "doppler_hz": 0.0,
            "snr_db": 3.0102999566,
        }
    ]
    return sensing


def _rematerialize_products(sensing: dict[str, object]) -> None:
    """Re-hash fixture products after deliberately changing test values."""

    raw_products = sensing.pop("products")
    assert isinstance(raw_products, dict)
    sensing.pop("product_contract_version")
    product_key_by_id = {
        descriptor["product_id"]: data_key for data_key, descriptor in raw_products.items()
    }
    for data_key, raw_descriptor in raw_products.items():
        descriptor = deepcopy(raw_descriptor)
        lineage = descriptor["lineage"]
        source_ids = lineage["source_product_ids"]
        source_data_keys = [
            product_key_by_id[source_id]
            for source_id in source_ids
            if source_id in product_key_by_id
        ]
        external_ids = [source_id for source_id in source_ids if source_id not in product_key_by_id]
        add_product_spec(
            sensing,
            data_key=data_key,
            kind=descriptor["kind"],
            variant=descriptor["variant"],
            axes=descriptor["axes"],
            representation=descriptor["representation"],
            unit=descriptor["unit"],
            scale_semantics=descriptor["scale_semantics"],
            evidence_role=descriptor["evidence_role"],
            source_data_keys=source_data_keys,
            source_product_ids=external_ids,
            temporal_support=descriptor["temporal_support"]["type"],
            combine=descriptor.get("combine"),
            processing=descriptor.get("processing"),
        )
    materialize_product_specs(sensing)


def test_complete_bundle_inspects_exact_cell_and_links_stable_detection() -> None:
    sensing = _with_detection(_typed_sensing())

    bundle = build_detection_stage_bundle(sensing)

    assert bundle.disposition is InspectionDisposition.AVAILABLE
    assert tuple(product.kind for product in bundle.products) == DETECTION_STAGE_KINDS
    assert all(
        product.disposition is InspectionDisposition.AVAILABLE for product in bundle.products
    )
    assert bundle.event is not None
    selection = CellSelection(
        pair_id=bundle.event.pair_id,
        doppler_bin=1,
        range_bin=2,
        cluster_id=0,
    )
    inspection = bundle.inspect(selection)

    assert inspection.disposition is InspectionDisposition.AVAILABLE
    assert inspection.doppler_hz == pytest.approx(0.0)
    assert inspection.range_m == pytest.approx(2.0)
    assert inspection.detection_input_power == pytest.approx(1.0)
    assert inspection.cfar_noise_power == pytest.approx(0.5)
    assert inspection.cfar_threshold_power == pytest.approx(1.5)
    assert inspection.cfar_training_cell_count == 8
    assert inspection.eligibility_disposition == "eligible"
    assert inspection.cfar_disposition == "passed"
    assert inspection.nms_disposition == "retained"
    assert inspection.snr_db == pytest.approx(3.0102999566)
    assert inspection.cluster_id == 0
    assert inspection.cluster_disposition == "clustered"
    assert inspection.detection_disposition == "linked"
    assert inspection.detection_link is not None
    assert inspection.detection_link.stable_key == (
        bundle.event.acquisition_id,
        bundle.event.processing_run_id,
        bundle.event.pair_id,
        0,
        1,
        2,
    )


def test_explicit_cluster_selection_disambiguates_angle_clusters_on_one_rd_bin() -> None:
    sensing = _with_detection(_typed_sensing())
    detections = sensing["detections"]
    assert isinstance(detections, list)
    detections.append(
        {
            "cluster_id": 1,
            "doppler_bin": 1,
            "range_bin": 2,
            "range_m": 2.0,
            "doppler_hz": 0.0,
            "snr_db": 3.0102999566,
        }
    )
    bundle = build_detection_stage_bundle(sensing)
    assert bundle.event is not None

    inspection = bundle.inspect(
        CellSelection(
            pair_id=bundle.event.pair_id,
            doppler_bin=1,
            range_bin=2,
            cluster_id=1,
        )
    )

    assert inspection.cluster_id == 0  # Exact collapsed cell label remains visible.
    assert inspection.detection_link is not None
    assert inspection.detection_link.cluster_id == 1
    assert inspection.detection_disposition == "linked"


def test_bundle_arrays_are_backed_by_immutable_storage() -> None:
    bundle = build_detection_stage_bundle(_typed_sensing())
    stage = bundle.product("range_doppler_detection_input_power")
    assert stage.values is not None
    assert not stage.values.flags.writeable

    with pytest.raises(ValueError):
        stage.values.setflags(write=True)


def test_lineage_trace_reaches_the_primary_range_doppler_source() -> None:
    bundle = build_detection_stage_bundle(_typed_sensing())

    trace = bundle.trace_lineage("range_doppler_cluster_labels")

    assert trace is not None
    assert trace.nodes[0].kind == "range_doppler_cluster_labels"
    assert any(node.kind == "range_doppler" for node in trace.nodes)
    assert trace.unresolved_source_product_ids == ()


def test_snr_fill_value_is_suppressed_when_validity_mask_is_false() -> None:
    sensing = _typed_sensing()
    snr_valid = sensing["range_doppler_snr_valid_mask_by_pair"]
    snr = sensing["range_doppler_snr_db_by_pair"]
    assert isinstance(snr_valid, np.ndarray) and isinstance(snr, np.ndarray)
    snr_valid[0, 0, 0] = False
    snr[0, 0, 0] = 0.0
    _rematerialize_products(sensing)
    bundle = build_detection_stage_bundle(sensing)
    assert bundle.event is not None

    inspection = bundle.inspect(
        CellSelection(pair_id=bundle.event.pair_id, doppler_bin=0, range_bin=0)
    )

    assert inspection.snr_valid is False
    assert inspection.snr_db is None
    assert inspection.eligibility_disposition == "eligible"
    assert inspection.cfar_disposition == "rejected"
    assert inspection.nms_disposition == "not_candidate"


@pytest.mark.parametrize(
    ("remove_descriptor", "expected_reason"),
    [(True, "descriptor_absent"), (False, "data_not_loaded")],
)
def test_missing_optional_stage_is_explicitly_unavailable(
    remove_descriptor: bool,
    expected_reason: str,
) -> None:
    sensing = _typed_sensing()
    data_key = "range_doppler_snr_db_by_pair"
    sensing.pop(data_key)
    if remove_descriptor:
        products = sensing["products"]
        assert isinstance(products, dict)
        products.pop(data_key)

    bundle = build_detection_stage_bundle(sensing)
    stage = bundle.product("range_doppler_snr_db")

    assert bundle.disposition is InspectionDisposition.AVAILABLE
    assert stage.disposition is InspectionDisposition.UNAVAILABLE
    assert stage.reason == expected_reason
    assert stage.values is None
    assert bundle.event is not None
    inspection = bundle.inspect(
        CellSelection(pair_id=bundle.event.pair_id, doppler_bin=1, range_bin=2)
    )
    assert inspection.snr_valid is True
    assert inspection.snr_db is None


def test_selection_uses_descriptor_contract_instead_of_producer_storage_key() -> None:
    sensing = _typed_sensing()
    old_key = "range_doppler_cfar_noise_power_by_pair"
    new_key = "renamed_noise_stage"
    sensing[new_key] = sensing.pop(old_key)
    products = sensing["products"]
    assert isinstance(products, dict)
    descriptor = dict(products.pop(old_key))
    descriptor["data_key"] = new_key
    products[new_key] = descriptor

    bundle = build_detection_stage_bundle(sensing)

    stage = bundle.product("range_doppler_cfar_noise_power")
    assert stage.disposition is InspectionDisposition.AVAILABLE
    assert stage.data_key == new_key


@pytest.mark.parametrize("contradiction", ["lineage", "coordinates", "time_span", "variant"])
def test_contradictory_stage_metadata_fails_closed(contradiction: str) -> None:
    sensing = _typed_sensing()
    products = sensing["products"]
    assert isinstance(products, dict)
    data_key = "range_doppler_cfar_noise_power_by_pair"
    descriptor = deepcopy(products[data_key])
    if contradiction == "lineage":
        descriptor["lineage"]["processing_run_id"] = "wrong-run"
    elif contradiction == "coordinates":
        descriptor["axes"][1]["coordinate_sha256"] = "0" * 64
    elif contradiction == "time_span":
        descriptor["time_span"]["reference_time_s"] += 0.01
    else:
        descriptor["variant"] = "alternate"
    products[data_key] = descriptor

    with pytest.raises(DetectionStageContractError):
        build_detection_stage_bundle(sensing)

    bundle = load_detection_stage_bundle(sensing)
    assert bundle.disposition is InspectionDisposition.INVALID
    assert bundle.event is None
    assert all(product.values is None for product in bundle.products)
    selection = CellSelection(pair_id="untrusted", doppler_bin=0, range_bin=0)
    inspection = bundle.inspect(selection)
    assert inspection.disposition is InspectionDisposition.INVALID
    assert inspection.detection_input_power is None


def test_contradictory_stage_decisions_fail_closed() -> None:
    sensing = _typed_sensing()
    eligible = sensing["range_doppler_cfar_eligible_mask_by_pair"]
    cfar = sensing["range_doppler_cfar_mask_by_pair"]
    assert isinstance(eligible, np.ndarray) and isinstance(cfar, np.ndarray)
    eligible[0, 1, 2] = False
    cfar[0, 1, 2] = True
    _rematerialize_products(sensing)

    invalid = load_detection_stage_bundle(sensing)

    assert invalid.disposition is InspectionDisposition.INVALID
    assert invalid.reason is not None
    assert "CFAR-positive cells must be eligible" in invalid.reason


def test_cell_selection_rejects_wrong_pair_and_out_of_bounds_bins() -> None:
    bundle = build_detection_stage_bundle(_typed_sensing())
    assert bundle.event is not None

    with pytest.raises(ValueError, match="pair_id"):
        bundle.inspect(CellSelection(pair_id="other", doppler_bin=0, range_bin=0))
    with pytest.raises(IndexError, match="doppler_bin"):
        bundle.inspect(CellSelection(pair_id=bundle.event.pair_id, doppler_bin=3, range_bin=0))
    with pytest.raises(IndexError, match="range_bin"):
        bundle.inspect(CellSelection(pair_id=bundle.event.pair_id, doppler_bin=0, range_bin=4))
