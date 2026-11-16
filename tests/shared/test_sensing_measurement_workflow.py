"""Runnable workflow tests for already-calibrated sensing comparison."""

from __future__ import annotations

import builtins
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import shared.sensing.measurement_workflow as workflow
from shared.sensing.measurement import (
    MeasurementContractError,
    canonical_measurement_sha256,
    load_measurement_alignment_recipe,
    load_measurement_manifest,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inventory(root: Path, inventory_path: Path) -> dict:
    inventory = _read_json(inventory_path)
    for entry in inventory["files"]:
        path = root / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert "hash_inventory.json" not in {item["path"] for item in inventory["files"]}
    return inventory


def test_synthetic_workflow_publishes_complete_traceable_evidence(tmp_path: Path) -> None:
    publication = workflow.publish_synthetic_measurement_workflow(tmp_path / "result")

    assert publication.root == (tmp_path / "result").resolve()
    assert publication.comparison_json.is_file()
    assert publication.residual_csv.is_file()
    assert publication.synchronized_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    measurement = load_measurement_manifest(
        publication.root / "inputs" / "measurement" / "manifest.json"
    )
    simulation = load_measurement_manifest(
        publication.root / "inputs" / "simulation" / "manifest.json"
    )
    assert [item.valid for item in measurement.acquisitions] == [True, False, True]
    assert measurement.acquisitions[1].missing is not None
    assert measurement.acquisitions[1].missing.reason == "not_measured"
    assert [item.sample_id for item in measurement.acquisitions] == [
        "synthetic:alpha",
        "synthetic:missing-middle",
        "synthetic:beta",
    ]
    assert all(len(product.product_id) == 64 for product in measurement.products)
    assert all(
        product.pair_id == measurement.sensor_topology[0].pair_id
        for product in measurement.products
    )
    assert measurement.calibration_record.scope is not None
    assert set(measurement.calibration_record.scope.acquisition_ids) == {
        measurement.acquisitions[0].acquisition_id,
        measurement.acquisitions[2].acquisition_id,
    }
    assert simulation.calibration_record.missing is not None

    comparison = _read_json(publication.comparison_json)
    expected = (
        np.array(
            [
                [0.01, -0.02, 0.03, -0.01],
                [0.02, 0.05, -0.04, 0.01],
                [-0.01, 0.02, 0.01, -0.02],
            ]
        ),
        np.array(
            [
                [-0.02, 0.01, 0.02, -0.01],
                [0.03, -0.04, 0.06, -0.02],
                [0.01, -0.01, 0.02, 0.01],
            ]
        ),
    )
    np.testing.assert_allclose(comparison["residuals"][0]["residual"]["values"], expected[0])
    np.testing.assert_allclose(comparison["residuals"][1]["residual"]["values"], expected[1])
    flattened = np.concatenate([item.reshape(-1) for item in expected])
    assert comparison["aggregate"]["rmse"] == pytest.approx(
        float(np.sqrt(np.mean(np.square(flattened))))
    )
    assert comparison["alignment_id"] == ("sha256:" + comparison["alignment_sha256"])

    with publication.residual_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert {row["sample_id"] for row in rows} == {"synthetic:alpha", "synthetic:beta"}
    assert float(rows[0]["aggregate_rmse"]) == pytest.approx(comparison["aggregate"]["rmse"])

    evidence = _read_json(publication.root / "comparison" / "workflow_evidence.json")
    assert evidence["calibration_record"]["content_included"] is False
    assert len(evidence["calibration_record"]["sha256"]) == 64
    assert evidence["synthetic_inputs"]["numeric_resources_retained"] is True
    assert evidence["runtime"]["python"]
    assert evidence["runtime"]["numpy"]
    assert evidence["runtime"]["matplotlib"]
    assert evidence["runtime"]["matplotlib_backend"].casefold() == "agg"
    assert evidence["runtime"]["png_portability"] == "runtime_qualified"
    boundary = evidence["storage_boundary"]
    assert boundary == {
        "ingress_format": "json_manifest_plus_numpy",
        "ingress_only": True,
        "native_frame_schema_version": 4,
        "native_product_contract_version": 2,
        "native_container": "packed_hdf5_v2",
        "packed_hdf5_adapter_status": "not_implemented_fail_closed",
        "statement": (
            "JSON/NumPy is an already-calibrated ingress envelope only; packed-HDF5 v2 "
            "with schema-v4 typed sensing products remains the native storage boundary."
        ),
    }
    assert evidence["native_crosswalk"]["scope"] == {
        "observable": "range_doppler_power",
        "variant": "primary",
        "pair_topology": "siso",
        "context": "standalone_ingress_evidence",
        "device_indices": "local_single_pair_zero_indices",
        "temporal_support": "instantaneous_acquisition_timestamp",
        "native_hdf5_published": False,
    }
    for source in ("measurement", "simulation"):
        records = evidence["native_crosswalk"][source]
        assert len(records) == 2
        assert all(record["native_kind"] == "range_doppler" for record in records)
        assert all(
            record["native_axes"] == ["pair", "doppler_frequency", "range"] for record in records
        )
        assert all(len(record["native_schema_v4_product_id"]) == 64 for record in records)
        assert all(
            record["native_schema_v4_product_id"] != record["ingress_product_id"]
            for record in records
        )
        for record in records:
            binding = record["ingress_binding"]
            descriptor = record["native_descriptor"]
            assert binding["pair"]["topology"] == "siso"
            assert binding["clock"]["clock_id"]
            assert binding["sign_convention"] in {
                "doppler_frequency_positive_approaching",
                "doppler_frequency_positive_receding",
            }
            assert [axis["name"] for axis in binding["axes"]] == [
                "doppler_frequency",
                "range",
            ]
            assert descriptor["product_id"] == record["native_schema_v4_product_id"]
            assert descriptor["pair"] == {
                "pair_id": binding["pair"]["pair_id"],
                "tx_sensor_id": binding["pair"]["tx_sensor_id"],
                "rx_sensor_id": binding["pair"]["rx_sensor_id"],
            }
            assert descriptor["time_span"]["reference_time_s"] == binding["timestamp_s"]
            assert descriptor["processing"]["sign_convention"] == binding["sign_convention"]
            assert all(len(axis["coordinate_sha256"]) == 64 for axis in descriptor["axes"])
            assert "authenticated_external_ingress_resource" in record["numeric_reverification"]
    assert not list(publication.root.rglob("*.h5"))
    inventory = _validate_inventory(publication.root, publication.hash_inventory)
    assert inventory["inventory_sha256"] == publication.inventory_sha256
    assert inventory["runtime"]["png_portability"] == "runtime_qualified"
    with publication.residual_csv.open(newline="", encoding="utf-8") as stream:
        first_residual = next(csv.DictReader(stream))
    assert json.loads(first_residual["axis_indices_json"]) == [0, 0]
    assert json.loads(first_residual["axis_coordinates_json"]) == {
        "doppler_frequency": -200.0,
        "range": 1.0,
    }


def test_synthetic_json_and_csv_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first = workflow.publish_synthetic_measurement_workflow(tmp_path / "first")
    second = workflow.publish_synthetic_measurement_workflow(tmp_path / "second")

    relative_paths = (
        "inputs/measurement/manifest.json",
        "inputs/simulation/manifest.json",
        "comparison/alignment_recipe.json",
        "comparison/comparison.json",
        "comparison/residual_metrics.csv",
        "comparison/workflow_evidence.json",
    )
    for relative in relative_paths:
        assert (first.root / relative).read_bytes() == (second.root / relative).read_bytes()


def test_native_crosswalk_binds_ingress_identity_sign_and_supported_scope(
    tmp_path: Path,
) -> None:
    measurement_path = workflow._write_synthetic_manifest(
        tmp_path / "measurement",
        source_kind="measurement",
    )
    simulation_path = workflow._write_synthetic_manifest(
        tmp_path / "simulation",
        source_kind="simulation",
    )
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    recipe = load_measurement_alignment_recipe(
        workflow._build_alignment_recipe(measurement, simulation)
    )
    original_ids = [
        record["native_schema_v4_product_id"]
        for record in workflow._native_crosswalk(measurement, recipe)
    ]

    raw = _read_json(measurement_path)
    raw["products"][0]["sign_convention"] = "doppler_frequency_positive_receding"
    raw["products"][0]["product_sha256"] = canonical_measurement_sha256(
        {key: value for key, value in raw["products"][0].items() if key != "product_sha256"},
        domain="product",
    )
    raw["manifest_sha256"] = canonical_measurement_sha256(
        {key: value for key, value in raw.items() if key != "manifest_sha256"},
        domain="manifest",
    )
    measurement_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    changed = load_measurement_manifest(measurement_path)
    changed_ids = [
        record["native_schema_v4_product_id"]
        for record in workflow._native_crosswalk(changed, recipe)
    ]
    assert changed_ids[0] != original_ids[0]

    raw = _read_json(measurement_path)
    raw["clock"]["origin"] = "synthetic_epoch_revised"
    raw["manifest_sha256"] = canonical_measurement_sha256(
        {key: value for key, value in raw.items() if key != "manifest_sha256"},
        domain="manifest",
    )
    measurement_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lineage_changed = load_measurement_manifest(measurement_path)
    lineage_changed_ids = [
        record["native_schema_v4_product_id"]
        for record in workflow._native_crosswalk(lineage_changed, recipe)
    ]
    assert lineage_changed_ids[0] != changed_ids[0]

    with pytest.raises(MeasurementContractError, match="range_doppler_power/primary"):
        workflow._native_crosswalk(
            changed,
            replace(recipe, observable="range_profile_power"),
        )
    non_siso = replace(
        changed,
        sensor_topology=(replace(changed.sensor_topology[0], topology="mimo"),),
    )
    with pytest.raises(MeasurementContractError, match="SISO"):
        workflow._native_crosswalk(non_siso, recipe)

    wrong_frame_product = replace(
        changed.products[0],
        axes=(
            replace(changed.products[0].axes[0], reference_frame="world"),
            changed.products[0].axes[1],
        ),
    )
    wrong_frame = replace(
        changed,
        products=(wrong_frame_product, *changed.products[1:]),
    )
    with pytest.raises(MeasurementContractError, match="reference frames"):
        workflow._native_crosswalk(wrong_frame, recipe)

    wrong_sign = replace(
        changed,
        products=(
            replace(changed.products[0], sign_convention="vendor_positive_left"),
            *changed.products[1:],
        ),
    )
    with pytest.raises(MeasurementContractError, match="Doppler sign conventions"):
        workflow._native_crosswalk(wrong_sign, recipe)


def test_existing_manifest_workflow_reproduces_synthetic_comparison(tmp_path: Path) -> None:
    synthetic = workflow.publish_synthetic_measurement_workflow(tmp_path / "synthetic")
    replay = workflow.publish_measurement_comparison(
        synthetic.root / "inputs" / "measurement" / "manifest.json",
        synthetic.root / "inputs" / "simulation" / "manifest.json",
        synthetic.root / "comparison" / "alignment_recipe.json",
        tmp_path / "replay",
    )

    assert replay.comparison_json.read_bytes() == synthetic.comparison_json.read_bytes()
    assert replay.residual_csv.read_bytes() == synthetic.residual_csv.read_bytes()
    evidence = _read_json(replay.root / "workflow_evidence.json")
    assert (
        evidence["alignment"]["alignment_id"]
        == _read_json(replay.root / "alignment_recipe.json")["alignment_id"]
    )
    _validate_inventory(replay.root, replay.hash_inventory)


def test_publication_never_overwrites_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    first = workflow.publish_synthetic_measurement_workflow(destination)
    before = first.hash_inventory.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        workflow.publish_synthetic_measurement_workflow(destination)

    assert first.hash_inventory.read_bytes() == before


def test_atomic_failure_leaves_no_partial_output_or_lock(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "failed"

    def fail_plot(*args, **kwargs):
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(workflow, "_write_synchronized_plot", fail_plot)
    with pytest.raises(RuntimeError, match="synthetic plot failure"):
        workflow.publish_synthetic_measurement_workflow(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
    assert not (tmp_path / ".failed.orchav-publish.lock").exists()


def test_staging_creation_failure_releases_publication_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "failed-staging"

    def fail_staging(*args, **kwargs):
        raise OSError("synthetic staging failure")

    monkeypatch.setattr(workflow.tempfile, "mkdtemp", fail_staging)
    with pytest.raises(OSError, match="synthetic staging failure"):
        workflow._atomic_publish(destination, lambda staging: None)

    assert not destination.exists()
    assert not (tmp_path / ".failed-staging.orchav-publish.lock").exists()


def test_atomic_promotion_rejects_destination_created_during_build(tmp_path: Path) -> None:
    destination = tmp_path / "raced"

    def racing_builder(staging: Path) -> None:
        (staging / "complete.txt").write_text("complete", encoding="utf-8")
        destination.mkdir()
        (destination / "external.txt").write_text("external", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        workflow._atomic_publish(destination, racing_builder)

    assert (destination / "external.txt").read_text(encoding="utf-8") == "external"
    assert not (destination / "complete.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="renameat2 fallback is POSIX-specific")
def test_atomic_promotion_fails_closed_without_renameat2(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import ctypes

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "complete.txt").write_text("complete", encoding="utf-8")

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        workflow.os,
        "rename",
        lambda *args, **kwargs: pytest.fail("unsafe POSIX rename fallback was called"),
    )

    with pytest.raises(OSError, match="Atomic no-replace"):
        workflow._promote_directory_noreplace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_tampered_alignment_recipe_fails_before_output_claim(tmp_path: Path) -> None:
    synthetic = workflow.publish_synthetic_measurement_workflow(tmp_path / "synthetic")
    recipe = _read_json(synthetic.root / "comparison" / "alignment_recipe.json")
    recipe["measurement_to_simulation_time_offset_s"] = 0.0
    destination = tmp_path / "tampered"

    with pytest.raises(MeasurementContractError, match="alignment_sha256"):
        workflow.publish_measurement_comparison(
            synthetic.root / "inputs" / "measurement" / "manifest.json",
            synthetic.root / "inputs" / "simulation" / "manifest.json",
            recipe,
            destination,
        )

    assert not destination.exists()


def test_workflow_import_is_matplotlib_and_calibration_free() -> None:
    code = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise AssertionError(name)
    if name.startswith('generator'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import shared.sensing.measurement_workflow
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
assert not any(name.startswith('generator') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_missing_matplotlib_has_standard_install_guidance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "matplotlib":
            error = ModuleNotFoundError("No module named 'matplotlib'")
            error.name = "matplotlib"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    destination = tmp_path / "missing-matplotlib"
    with pytest.raises(
        RuntimeError,
        match=r"python -m pip install orchav",
    ):
        workflow.publish_synthetic_measurement_workflow(destination)

    assert not destination.exists()


def test_workflow_source_has_no_generator_or_calibration_import() -> None:
    source = Path(workflow.__file__).read_text(encoding="utf-8")
    assert "from generator" not in source
    assert "import generator" not in source
    assert "generator.core.calibration" not in source
