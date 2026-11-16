"""Determinism and traceability tests for detector-stage exports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from visualizer.src.sensing import stage_export
from visualizer.src.sensing.stage_export import (
    build_stage_export_manifest,
    detections_csv_bytes,
    diagnostic_png_bytes,
    export_detection_stage_bundle,
    selected_cell_csv_bytes,
    stage_manifest_bytes,
)
from visualizer.src.sensing.stage_inspector import (
    DETECTION_STAGE_KINDS,
    CellSelection,
    InspectionDisposition,
    build_detection_stage_bundle,
)


def _bundle_and_selection():
    sensing = _typed_sensing()
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
    bundle = build_detection_stage_bundle(sensing)
    assert bundle.event is not None
    selection = CellSelection(
        pair_id=bundle.event.pair_id,
        doppler_bin=1,
        range_bin=2,
        cluster_id=0,
    )
    return bundle, selection


def test_manifest_is_sorted_json_safe_and_traceable_to_exact_products() -> None:
    bundle, selection = _bundle_and_selection()
    selected_stages = (
        "range_doppler_snr_db",
        "range_doppler_cfar_mask",
    )

    manifest = build_stage_export_manifest(
        bundle,
        selection,
        stage_kinds=selected_stages,
    )
    encoded = stage_manifest_bytes(bundle, selection, stage_kinds=selected_stages)

    assert list(manifest) == sorted(manifest)
    assert manifest["stage_export_version"] == 1
    assert manifest["disposition"] == "available"
    assert manifest["event"] == {
        "acquisition_id": bundle.event.acquisition_id,
        "clock_id": bundle.event.clock_id,
        "pair_id": bundle.event.pair_id,
        "processing_recipe_id": bundle.event.processing_recipe_id,
        "processing_run_id": bundle.event.processing_run_id,
        "reference_time_s": bundle.event.reference_time_s,
    }
    assert manifest["axes"]["doppler_frequency"]["values"] == [-5.0, 0.0, 5.0]
    assert manifest["axes"]["range"]["values"] == [0.0, 1.0, 2.0, 3.0]
    assert manifest["diagnostic_stage_kinds"] == [
        "range_doppler_cfar_mask",
        "range_doppler_snr_db",
    ]
    products = manifest["products"]
    assert [product["kind"] for product in products] == sorted(DETECTION_STAGE_KINDS)
    assert all(product["available"] is True for product in products)
    assert all(len(product["product_id"]) == 64 for product in products)
    cluster_product = next(
        product for product in products if product["kind"] == "range_doppler_cluster_labels"
    )
    assert cluster_product["lineage"]["root_product_id"] == cluster_product["product_id"]
    assert cluster_product["lineage"]["nodes"][0]["kind"] == ("range_doppler_cluster_labels")
    assert manifest["selected_cell"]["cfar_passed"] is True
    assert manifest["selected_cell"]["cfar_disposition"] == "passed"
    assert manifest["selected_cell"]["nms_retained"] is True
    assert manifest["selected_cell"]["detection_link"]["cluster_id"] == 0
    assert b"NaN" not in encoded and b"Infinity" not in encoded
    assert json.loads(encoded) == manifest


def test_manifest_and_csv_normalize_nonfinite_values_to_null_and_blank() -> None:
    bundle, selection = _bundle_and_selection()
    assert bundle.event is not None and bundle.doppler_axis_hz is not None
    nonfinite_bundle = replace(
        bundle,
        event=replace(bundle.event, reference_time_s=float("nan")),
        doppler_axis_hz=np.asarray([-5.0, np.nan, 5.0]),
    )

    manifest = json.loads(stage_manifest_bytes(nonfinite_bundle, selection))
    detection_rows = list(
        csv.DictReader(io.StringIO(detections_csv_bytes(nonfinite_bundle).decode("utf-8")))
    )
    cell_rows = list(
        csv.DictReader(
            io.StringIO(selected_cell_csv_bytes(nonfinite_bundle, selection).decode("utf-8"))
        )
    )

    assert manifest["event"]["reference_time_s"] is None
    assert manifest["axes"]["doppler_frequency"]["values"][1] is None
    assert manifest["selected_cell"]["doppler_hz"] is None
    assert detection_rows[0]["doppler_hz"] == ""
    assert cell_rows[0]["reference_time_s"] == ""
    assert cell_rows[0]["doppler_hz"] == ""


def test_detection_and_cell_csv_have_fixed_columns_and_stable_rows() -> None:
    bundle, selection = _bundle_and_selection()

    detections = detections_csv_bytes(bundle)
    cell = selected_cell_csv_bytes(bundle, selection)

    assert detections == detections_csv_bytes(bundle)
    assert cell == selected_cell_csv_bytes(bundle, selection)
    detection_reader = csv.DictReader(io.StringIO(detections.decode("utf-8")))
    detection_rows = list(detection_reader)
    assert detection_reader.fieldnames == [
        "acquisition_id",
        "processing_run_id",
        "pair_id",
        "cluster_id",
        "doppler_bin",
        "range_bin",
        "doppler_hz",
        "range_m",
    ]
    assert detection_rows[0]["cluster_id"] == "0"
    assert detection_rows[0]["doppler_bin"] == "1"
    assert detection_rows[0]["range_m"] == "2"

    cell_reader = csv.DictReader(io.StringIO(cell.decode("utf-8")))
    cell_rows = list(cell_reader)
    assert cell_reader.fieldnames is not None
    assert cell_reader.fieldnames[:6] == [
        "clock_id",
        "acquisition_id",
        "processing_run_id",
        "processing_recipe_id",
        "pair_id",
        "reference_time_s",
    ]
    assert cell_rows[0]["cfar_eligible"] == "true"
    assert cell_rows[0]["cfar_disposition"] == "passed"
    assert cell_rows[0]["snr_db"] == "3.0102999565999999"
    assert cell_rows[0]["linked_cluster_id"] == "0"
    assert selected_cell_csv_bytes(bundle).count(b"\n") == 1


def test_diagnostic_png_is_byte_identical_and_marker_changes_render() -> None:
    bundle, selection = _bundle_and_selection()
    stages = ("range_doppler_cfar_mask", "range_doppler_snr_db")

    first = diagnostic_png_bytes(bundle, selection, stage_kinds=stages)
    repeated = diagnostic_png_bytes(bundle, selection, stage_kinds=tuple(reversed(stages)))
    without_marker = diagnostic_png_bytes(bundle, stage_kinds=stages)

    assert first == repeated
    assert first != without_marker
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(first)) as image:
        assert image.format == "PNG"
        assert image.size == (1000, 340)
        assert image.info["Software"] == "ORCHAV deterministic sensing stage export"
        assert "Creation Time" not in image.info


def test_requested_unavailable_stage_is_rendered_without_recomputation() -> None:
    sensing = _typed_sensing()
    data_key = "range_doppler_snr_db_by_pair"
    sensing.pop(data_key)
    products = sensing["products"]
    assert isinstance(products, dict)
    products.pop(data_key)
    bundle = build_detection_stage_bundle(sensing)

    manifest = build_stage_export_manifest(
        bundle,
        stage_kinds=("range_doppler_snr_db",),
    )
    png = diagnostic_png_bytes(bundle, stage_kinds=("range_doppler_snr_db",))

    snr_product = next(
        product for product in manifest["products"] if product["kind"] == "range_doppler_snr_db"
    )
    assert bundle.disposition is InspectionDisposition.AVAILABLE
    assert snr_product["disposition"] == "unavailable"
    assert snr_product["reason"] == "descriptor_absent"
    assert png.startswith(b"\x89PNG")


def test_atomic_export_returns_exact_paths_sizes_and_hashes(tmp_path) -> None:
    bundle, selection = _bundle_and_selection()
    output = tmp_path / "exports"
    stages = ("range_doppler_cfar_mask", "range_doppler_snr_db")

    first = export_detection_stage_bundle(
        bundle,
        output,
        selection,
        stage_kinds=stages,
    )
    first_payloads = {
        artifact.path.name: artifact.path.read_bytes() for artifact in first.artifacts
    }
    repeated = export_detection_stage_bundle(
        bundle,
        output,
        selection,
        stage_kinds=tuple(reversed(stages)),
    )

    assert [artifact.path.name for artifact in first.artifacts] == [
        "detection_stages_manifest.json",
        "detection_stages_detections.csv",
        "detection_stages_cell.csv",
        "detection_stages_diagnostic.png",
    ]
    for artifact in repeated.artifacts:
        payload = artifact.path.read_bytes()
        assert payload == first_payloads[artifact.path.name]
        assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
        assert artifact.size_bytes == len(payload)
        assert artifact.path.is_absolute()
    assert not list(output.glob(".*.tmp"))
    manifest_text = repeated.manifest.path.read_text(encoding="utf-8")
    assert "created_at" not in manifest_text
    assert "export_id" not in manifest_text


def test_atomic_write_cleans_temporary_file_on_replace_failure(tmp_path, monkeypatch) -> None:
    bundle, selection = _bundle_and_selection()
    output = tmp_path / "failed-export"

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(stage_export.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        export_detection_stage_bundle(
            bundle,
            output,
            selection,
            stage_kinds=("range_doppler_cfar_mask",),
        )

    assert list(output.iterdir()) == []


@pytest.mark.parametrize("stem", ["../escape", ".hidden", "bad/name", ""])
def test_export_rejects_unsafe_stem(tmp_path, stem: str) -> None:
    bundle, _selection = _bundle_and_selection()

    with pytest.raises(ValueError, match="stem"):
        export_detection_stage_bundle(bundle, tmp_path, stem=stem)
