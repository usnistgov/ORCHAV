"""Contract tests for already-calibrated sensing measurement comparison."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shared.sensing.measurement import (
    AGGREGATE_WEIGHTING_POLICY,
    INDEPENDENT_VALUE_UNCERTAINTY_POLICY,
    RESIDUAL_DEFINITION,
    AcquisitionPair,
    AxisAlignment,
    MeasurementAlignment,
    MeasurementContractError,
    alignment_recipe_mapping,
    canonical_measurement_sha256,
    compare_measurement_to_simulation,
    comparison_export_mapping,
    load_measurement_alignment_recipe,
    load_measurement_manifest,
)

MEASUREMENT_DATASET_ID = "202ce869-0670-53da-8249-cfb1bea15a92"
SIMULATION_DATASET_ID = "a2f45acc-f55c-541e-98ee-0aeef31bdc07"
MEASUREMENT_ACQUISITION_IDS = (
    "ff022150-ee47-5080-962b-c4e68a9f6d2f",
    "3f353300-843e-5a79-af75-1300de9ab997",
)
SIMULATION_ACQUISITION_IDS = (
    "c6007adc-8fe4-507e-9d1d-a702577ad4f4",
    "648b3c2c-1888-527d-bd4b-48d78111b283",
)
SAMPLE_IDS = ("synthetic:alpha", "synthetic:beta")
PAIR_ID = "f" * 64
PRODUCT_IDS = {
    "measurement": (
        "391e95d6-113f-5d50-8021-ac20cfcd5ea6",
        "19da2d12-9b89-5b3f-9149-a84723d14583",
    ),
    "simulation": (
        "df88a371-e6c3-5358-89d5-59b3ce416ebd",
        "344362ee-fe69-536c-872c-540287371a91",
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rehash_transform(transform: dict[str, Any]) -> None:
    transform["sha256"] = canonical_measurement_sha256(
        {key: value for key, value in transform.items() if key != "sha256"},
        domain="reference_transform",
    )


def _rehash_chain(chain: dict[str, Any]) -> None:
    for transform in chain["transforms"]:
        _rehash_transform(transform)
    chain["sha256"] = canonical_measurement_sha256(
        {key: value for key, value in chain.items() if key != "sha256"},
        domain="reference_chain",
    )


def _rehash_product(product: dict[str, Any]) -> None:
    product["product_sha256"] = canonical_measurement_sha256(
        {key: value for key, value in product.items() if key != "product_sha256"},
        domain="product",
    )


def _rehash_split(manifest: dict[str, Any]) -> None:
    manifest["dataset"]["split_sha256"] = canonical_measurement_sha256(
        {
            "split_id": manifest["dataset"]["split_id"],
            "sample_ids": [sample["sample_id"] for sample in manifest["acquisitions"]],
        },
        domain="dataset_split",
    )


def _rehash_manifest(manifest: dict[str, Any]) -> None:
    manifest["manifest_sha256"] = canonical_measurement_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"},
        domain="manifest",
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    _rehash_manifest(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_product_array(manifest_path: Path, product: dict[str, Any]) -> np.ndarray:
    path = manifest_path.parent / product["data"]["path"]
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    with np.load(path, allow_pickle=False) as archive:
        return np.array(archive[product["data"]["array_key"]], copy=True)


def _replace_product_array(
    manifest_path: Path,
    product: dict[str, Any],
    values: np.ndarray,
) -> None:
    path = manifest_path.parent / product["data"]["path"]
    if path.suffix == ".npy":
        np.save(path, values, allow_pickle=False)
        product["data"].pop("array_key", None)
    else:
        np.savez(path, values=values)
        product["data"]["array_key"] = "values"
    product["data"].update(
        {
            "sha256": _sha256_bytes(path.read_bytes()),
            "dtype": values.dtype.str,
            "shape": list(values.shape),
        }
    )
    _rehash_product(product)


def _synthetic_manifest(
    root: Path,
    *,
    source_kind: str,
    uncertainty_status: str = "known",
) -> tuple[Path, dict[str, Any]]:
    """Write an ORCHAV-authored deterministic two-acquisition fixture."""

    products_dir = root / "products"
    products_dir.mkdir(parents=True, exist_ok=True)
    if source_kind == "measurement":
        arrays = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64),
        )
        dataset_id = MEASUREMENT_DATASET_ID
        acquisition_ids = MEASUREMENT_ACQUISITION_IDS
        timestamps = (0.0, 1.0)
        range_axis = [1.1, 2.1]
        calibration: dict[str, Any] = {
            "status": "known",
            "record_id": "urn:orchav:synthetic-calibration-record:v1",
            "sha256": _sha256_bytes(b"ORCHAV synthetic opaque calibration record v1"),
            "scope": {
                "pair_ids": [PAIR_ID],
                "acquisition_ids": list(acquisition_ids),
                "observable_variants": [
                    {"observable": "range_doppler_power", "variant": "primary"}
                ],
            },
        }
        uncertainty_value = 0.05
    else:
        arrays = (
            np.array([[0.9, 2.1], [2.8, 4.2]], dtype=np.float64),
            np.array([[2.1, 3.9], [5.8, 8.2]], dtype=np.float64),
        )
        dataset_id = SIMULATION_DATASET_ID
        acquisition_ids = SIMULATION_ACQUISITION_IDS
        timestamps = (0.025, 1.025)
        range_axis = [1.0, 2.0]
        calibration = {"status": "missing", "reason": "not_applicable"}
        uncertainty_value = 0.04

    products: list[dict[str, Any]] = []
    for index, array in enumerate(arrays):
        if index == 0:
            filename = f"{source_kind}-{index}.npy"
            product_path = products_dir / filename
            np.save(product_path, array, allow_pickle=False)
            data: dict[str, Any] = {
                "path": f"products/{filename}",
                "sha256": _sha256_bytes(product_path.read_bytes()),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        else:
            filename = f"{source_kind}-{index}.npz"
            product_path = products_dir / filename
            np.savez(product_path, values=array)
            data = {
                "path": f"products/{filename}",
                "array_key": "values",
                "sha256": _sha256_bytes(product_path.read_bytes()),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        uncertainty = (
            {
                "status": "known",
                "unit": "1",
                "standard_deviation": uncertainty_value,
            }
            if uncertainty_status == "known"
            else {"status": "missing", "unit": "1", "reason": "not_provided"}
        )
        product = {
            "product_id": PRODUCT_IDS[source_kind][index],
            "acquisition_id": acquisition_ids[index],
            "pair_id": PAIR_ID,
            "observable": "range_doppler_power",
            "variant": "primary",
            "representation": "power_linear",
            "unit": "1",
            "sign_convention": "radial_velocity_positive_receding",
            "axes": [
                {
                    "name": "range",
                    "unit": "m",
                    "reference_frame": "orchav_world",
                    "coordinates": range_axis,
                    "uncertainty": {
                        "status": "known",
                        "unit": "m",
                        "standard_deviation": 0.01,
                    },
                },
                {
                    "name": "radial_velocity",
                    "unit": "m/s",
                    "reference_frame": "sensor_line_of_sight",
                    "coordinates": [-1.0, 1.0],
                    "uncertainty": {
                        "status": "known",
                        "unit": "m/s",
                        "standard_deviation": 0.02,
                    },
                },
            ],
            "data": data,
            "uncertainty": uncertainty,
            "product_sha256": "",
        }
        _rehash_product(product)
        products.append(product)

    transform = {
        "transform_id": "f5cf0f8c-80a1-536e-9b90-8c7a91d81202",
        "from_frame": "fixture_sensor",
        "to_frame": "orchav_world",
        "matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "sha256": "",
    }
    _rehash_transform(transform)
    chain = {
        "source_frame": "fixture_sensor",
        "target_frame": "orchav_world",
        "transforms": [transform],
        "sha256": "",
    }
    _rehash_chain(chain)
    manifest: dict[str, Any] = {
        "contract_version": 1,
        "dataset": {
            "dataset_id": dataset_id,
            "source_kind": source_kind,
            "role": "evaluation",
            "split_id": "synthetic-evaluation-v1",
            "split_sha256": "",
        },
        "governance": {
            "provenance_id": "orchav:synthetic-measurement-fixture-v1",
            "license_spdx": "GPL-2.0-only",
            "redistribution": "permitted",
            "privacy": "none",
            "authority_id": "urn:orchav:authored-synthetic-fixture",
        },
        "acquisitions": [
            {
                "acquisition_id": acquisition_id,
                "sample_id": sample_id,
                "timestamp_s": timestamp,
                "valid": True,
            }
            for acquisition_id, sample_id, timestamp in zip(
                acquisition_ids,
                SAMPLE_IDS,
                timestamps,
            )
        ],
        "sensor_topology": [
            {
                "pair_id": PAIR_ID,
                "tx_sensor_id": "sensor:synthetic-tx",
                "rx_sensor_id": "sensor:synthetic-rx",
                "topology": "siso",
            }
        ],
        "clock": {
            "clock_id": f"{source_kind}-fixture-clock",
            "origin": "synthetic_fixture_start",
            "unit": "s",
        },
        "calibration_record": calibration,
        "reference_chain": chain,
        "processing_lineage": {
            "processing_run_id": (
                "4c741e5b-f9d4-54ec-a538-7385123f30ae"
                if source_kind == "measurement"
                else "8ecb2188-006f-55e7-96b1-92d4c070dac0"
            ),
            "processor": f"orchav.synthetic.{source_kind}",
            "processor_version": "1",
            "recipe_sha256": _sha256_bytes(f"{source_kind}-recipe".encode()),
            "source_sha256": [_sha256_bytes(f"{source_kind}-source".encode())],
        },
        "products": products,
        "manifest_sha256": "",
    }
    _rehash_split(manifest)
    path = root / f"{source_kind}.json"
    return _write_manifest(path, manifest), manifest


def _fixture_pair(tmp_path: Path, **measurement_options: Any):
    measurement_path, measurement_raw = _synthetic_manifest(
        tmp_path / "measurement",
        source_kind="measurement",
        **measurement_options,
    )
    simulation_path, simulation_raw = _synthetic_manifest(
        tmp_path / "simulation",
        source_kind="simulation",
    )
    return measurement_path, measurement_raw, simulation_path, simulation_raw


def _alignment(measurement, simulation):
    body = {
        "contract": "orchav.sensing.measurement_alignment",
        "contract_version": 1,
        "measurement_dataset_id": measurement.dataset.dataset_id,
        "simulation_dataset_id": simulation.dataset.dataset_id,
        "measurement_manifest_sha256": measurement.manifest_sha256,
        "simulation_manifest_sha256": simulation.manifest_sha256,
        "split_sha256": measurement.dataset.split_sha256,
        "observable": "range_doppler_power",
        "variant": "primary",
        "uncertainty_combination_policy": INDEPENDENT_VALUE_UNCERTAINTY_POLICY,
        "acquisition_pairs": [
            {
                "sample_id": sample_id,
                "measurement_acquisition_id": measurement_id,
                "simulation_acquisition_id": simulation_id,
            }
            for sample_id, measurement_id, simulation_id in zip(
                SAMPLE_IDS,
                MEASUREMENT_ACQUISITION_IDS,
                SIMULATION_ACQUISITION_IDS,
            )
        ],
        "axis_alignments": [
            {
                "name": "range",
                "unit": "m",
                "measurement_to_simulation_scale": 1,
                "measurement_to_simulation_offset": -0.1,
                "absolute_tolerance": 1e-12,
            },
            {
                "name": "radial_velocity",
                "unit": "m/s",
                "measurement_to_simulation_scale": 1,
                "measurement_to_simulation_offset": 0.0,
                "absolute_tolerance": 1e-12,
            },
        ],
        "measurement_to_simulation_time_offset_s": 0.025,
        "timestamp_tolerance_s": 1e-12,
    }
    digest = canonical_measurement_sha256(body, domain="alignment_recipe")
    body["alignment_id"] = f"sha256:{digest}"
    body["alignment_sha256"] = digest
    return load_measurement_alignment_recipe(body)


def _compare_paths(measurement_path: Path, simulation_path: Path):
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    return compare_measurement_to_simulation(
        measurement,
        simulation,
        _alignment(measurement, simulation),
    )


def _modified_recipe(recipe, **updates):
    mapping = alignment_recipe_mapping(recipe)
    mapping.update(updates)
    mapping.pop("alignment_id")
    mapping.pop("alignment_sha256")
    digest = canonical_measurement_sha256(mapping, domain="alignment_recipe")
    mapping["alignment_id"] = f"sha256:{digest}"
    mapping["alignment_sha256"] = digest
    return load_measurement_alignment_recipe(mapping)


def test_loader_accepts_typed_npy_and_npz_products(tmp_path: Path) -> None:
    measurement_path, _, simulation_path, _ = _fixture_pair(tmp_path)

    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)

    assert measurement.dataset.source_kind == "measurement"
    assert measurement.governance.provenance_id == "orchav:synthetic-measurement-fixture-v1"
    assert measurement.governance.license_spdx == "GPL-2.0-only"
    assert measurement.governance.redistribution == "permitted"
    assert measurement.calibration_record.record_id == (
        "urn:orchav:synthetic-calibration-record:v1"
    )
    assert simulation.calibration_record.missing is not None
    assert simulation.calibration_record.missing.reason == "not_applicable"
    assert [product.sample_id for product in measurement.products] == list(SAMPLE_IDS)
    assert [product.timestamp_s for product in simulation.products] == [0.025, 1.025]
    np.testing.assert_array_equal(measurement.products[0].values, [[1.0, 2.0], [3.0, 4.0]])
    assert not measurement.products[0].values.flags.writeable
    assert not measurement.products[0].axes[0].coordinates.flags.writeable
    assert not measurement.reference_chain.transforms[0].matrix.flags.writeable


def test_numeric_product_is_parsed_from_the_authenticated_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    product_path = measurement_path.parent / raw["products"][0]["data"]["path"]
    original_values = np.asarray(np.load(product_path, allow_pickle=False))
    original_bytes = product_path.read_bytes()
    np.save(product_path, np.full(original_values.shape, 9.0), allow_pickle=False)
    changed_bytes = product_path.read_bytes()
    product_path.write_bytes(original_bytes)
    original_read_bytes = Path.read_bytes
    product_reads = 0

    def swap_after_authenticated_read(path: Path) -> bytes:
        nonlocal product_reads
        if path == product_path:
            product_reads += 1
            if product_reads == 1:
                product_path.write_bytes(changed_bytes)
                return original_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_after_authenticated_read)
    loaded = load_measurement_manifest(measurement_path)

    np.testing.assert_allclose(loaded.products[0].values, original_values, atol=0.0, rtol=0.0)
    assert product_reads == 1


def test_contract_versions_require_exact_non_boolean_integers(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["contract_version"] = 1.0
    _write_manifest(measurement_path, raw)
    with pytest.raises(MeasurementContractError, match="contract_version must be 1"):
        load_measurement_manifest(measurement_path)

    measurement_path, _, simulation_path, _ = _fixture_pair(tmp_path / "recipe")
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    with pytest.raises(MeasurementContractError, match="contract_version must be 1"):
        _modified_recipe(_alignment(measurement, simulation), contract_version=1.0)


def test_shared_import_never_loads_generator_calibration() -> None:
    code = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('generator.core.calibration'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import shared.sensing.measurement
assert not any(name.startswith('generator.core.calibration') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_loader_accepts_semantic_hash_and_uuid_stable_identity_forms(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["dataset"]["dataset_id"] = "dataset:synthetic-measurement-v1"
    raw["processing_lineage"]["processing_run_id"] = "processing:synthetic-run-alpha"
    raw["reference_chain"]["transforms"][0]["transform_id"] = "transform:fixture-sensor-to-world"
    _rehash_chain(raw["reference_chain"])
    raw["products"][0]["product_id"] = "a" * 64
    raw["products"][0]["acquisition_id"] = "acquisition:synthetic-alpha"
    raw["acquisitions"][0]["acquisition_id"] = "acquisition:synthetic-alpha"
    raw["calibration_record"]["scope"]["acquisition_ids"][0] = "acquisition:synthetic-alpha"
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)

    loaded = load_measurement_manifest(measurement_path)

    assert loaded.dataset.dataset_id == "dataset:synthetic-measurement-v1"
    assert loaded.products[0].product_id == "a" * 64
    assert loaded.products[0].acquisition_id == "acquisition:synthetic-alpha"
    assert loaded.processing_lineage.processing_run_id == "processing:synthetic-run-alpha"


def test_loader_rejects_nonportable_extended_float_dtype(tmp_path: Path) -> None:
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform long double is not wider than float64")
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    product = raw["products"][0]
    values = _read_product_array(measurement_path, product).astype(np.longdouble)
    _replace_product_array(measurement_path, product, values)
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="float32, float64"):
        load_measurement_manifest(measurement_path)


def test_big_endian_float64_products_are_admitted_and_exportable(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for manifest_path, raw in (
        (measurement_path, measurement_raw),
        (simulation_path, simulation_raw),
    ):
        for product in raw["products"]:
            values = _read_product_array(manifest_path, product).astype(">f8")
            _replace_product_array(manifest_path, product, values)
        _write_manifest(manifest_path, raw)

    comparison = _compare_paths(measurement_path, simulation_path)
    exported = comparison_export_mapping(comparison)

    assert exported["residuals"][0]["residual"]["encoding"] == "real"
    json.dumps(exported, sort_keys=True, allow_nan=False)


def test_loader_rejects_ordinal_acquisition_identity(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["products"][0]["acquisition_id"] = "acquisition:1"
    raw["acquisitions"][0]["acquisition_id"] = "acquisition:1"
    raw["calibration_record"]["scope"]["acquisition_ids"][0] = "acquisition:1"
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="non-ordinal namespaced ID"):
        load_measurement_manifest(measurement_path)


def test_comparison_aligns_ids_times_axes_and_exports_deterministically(tmp_path: Path) -> None:
    measurement_path, _, simulation_path, _ = _fixture_pair(tmp_path)
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    recipe = _alignment(measurement, simulation)

    comparison = compare_measurement_to_simulation(
        measurement,
        simulation,
        recipe,
    )

    assert comparison.measurement_dataset_id == MEASUREMENT_DATASET_ID
    assert comparison.simulation_dataset_id == SIMULATION_DATASET_ID
    assert comparison.alignment_id == recipe.alignment_id
    assert comparison.alignment_sha256 == recipe.alignment_sha256
    np.testing.assert_allclose(
        comparison.residuals[0].residual,
        [[0.1, -0.1], [0.2, -0.2]],
    )
    assert comparison.aggregate_rmse == pytest.approx(np.sqrt(0.025))
    assert comparison.aggregate_max_absolute_error == pytest.approx(0.2)
    assert comparison.aggregate_mean_residual == pytest.approx(0.0 + 0.0j)
    assert comparison.residual_definition == RESIDUAL_DEFINITION
    assert comparison.aggregate_weighting_policy == AGGREGATE_WEIGHTING_POLICY
    assert comparison.aggregate_normalized_rmse == pytest.approx(
        np.sqrt(0.025) / np.hypot(0.05, 0.04)
    )
    assert comparison.uncertainty_combination_policy == INDEPENDENT_VALUE_UNCERTAINTY_POLICY

    first = comparison_export_mapping(comparison)
    second = comparison_export_mapping(comparison)
    assert first == second
    export_hash = first.pop("export_sha256")
    assert export_hash == canonical_measurement_sha256(first, domain="comparison_export")
    json.dumps(first, sort_keys=True, allow_nan=False)
    assert first["product_contract"]["residual_definition"] == RESIDUAL_DEFINITION
    assert first["aggregate_contract"] == {
        "weighting_policy": AGGREGATE_WEIGHTING_POLICY,
    }
    assert first["uncertainty_contract"] == {
        "value_combination_policy": INDEPENDENT_VALUE_UNCERTAINTY_POLICY,
        "axis_uncertainty_use": (
            "retained_for_axis_contract_validation_not_propagated_into_value_residuals"
        ),
    }


def test_aggregate_direction_and_weighting_are_explicit_for_unequal_shapes(
    tmp_path: Path,
) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for source_path, raw, second_value in (
        (measurement_path, measurement_raw, 2.0),
        (simulation_path, simulation_raw, 0.0),
    ):
        first = raw["products"][0]
        _replace_product_array(
            source_path,
            first,
            np.zeros((2, 2), dtype=np.float64),
        )
        second = raw["products"][1]
        _replace_product_array(
            source_path,
            second,
            np.full((3, 2), second_value, dtype=np.float64),
        )
        second["axes"][0]["coordinates"] = (
            [1.1, 2.1, 3.1] if raw["dataset"]["source_kind"] == "measurement" else [1.0, 2.0, 3.0]
        )
        _rehash_product(second)
        _write_manifest(source_path, raw)

    comparison = _compare_paths(measurement_path, simulation_path)

    assert comparison.residual_definition == "axis_aligned_measurement_minus_simulation"
    assert comparison.aggregate_mean_residual == pytest.approx(1.2 + 0.0j)
    assert comparison.aggregate_rmse == pytest.approx(np.sqrt(2.4))
    assert comparison.aggregate_weighting_policy == (
        "equal_weight_per_residual_element_across_selected_products"
    )


def test_alignment_requires_explicit_supported_uncertainty_policy(tmp_path: Path) -> None:
    measurement_path, _, simulation_path, _ = _fixture_pair(tmp_path)
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    recipe = _alignment(measurement, simulation)

    with pytest.raises(MeasurementContractError, match="uncertainty_combination_policy"):
        _modified_recipe(recipe, uncertainty_combination_policy="unspecified")


def test_missing_uncertainty_is_explicit_and_disables_normalized_aggregate(
    tmp_path: Path,
) -> None:
    measurement_path, _, simulation_path, _ = _fixture_pair(
        tmp_path,
        uncertainty_status="missing",
    )
    comparison = _compare_paths(measurement_path, simulation_path)

    assert comparison.aggregate_normalized_rmse is None
    assert all(item.normalized_rmse is None for item in comparison.residuals)
    exported = comparison_export_mapping(comparison)
    assert exported["residuals"][0]["combined_standard_uncertainty"] == {
        "status": "missing",
        "unit": "1",
        "reason": "not_provided",
    }


@pytest.mark.parametrize("sample_id", ["0", "frame-1", "sample:2", "capture"])
def test_loader_rejects_ordinal_or_unnamespaced_sample_identity(
    tmp_path: Path,
    sample_id: str,
) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["acquisitions"][0]["sample_id"] = sample_id
    _rehash_split(raw)
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="stable|ordinal|namespaced"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_out_of_order_and_duplicate_acquisitions(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["acquisitions"].reverse()
    _rehash_split(raw)
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="strictly increasing"):
        load_measurement_manifest(measurement_path)

    raw["acquisitions"].reverse()
    raw["products"][1]["acquisition_id"] = raw["products"][0]["acquisition_id"]
    _rehash_product(raw["products"][1])
    _write_manifest(measurement_path, raw)
    with pytest.raises(MeasurementContractError, match="duplicate acquisition ID"):
        load_measurement_manifest(measurement_path)


def test_ordered_acquisition_table_preserves_missing_slot_without_product(
    tmp_path: Path,
) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["acquisitions"].insert(
        1,
        {
            "acquisition_id": "acquisition:synthetic-missing-middle",
            "sample_id": "synthetic:missing-middle",
            "timestamp_s": 0.5,
            "valid": False,
            "missing_reason": "not_measured",
        },
    )
    raw["acquisitions"][2]["timestamp_s"] = 1.0
    _rehash_split(raw)
    _write_manifest(measurement_path, raw)

    loaded = load_measurement_manifest(measurement_path)

    assert [item.valid for item in loaded.acquisitions] == [True, False, True]
    assert loaded.acquisitions[1].missing is not None
    assert loaded.acquisitions[1].missing.reason == "not_measured"
    assert len(loaded.products) == 2


def test_product_cannot_bind_missing_acquisition_or_unknown_pair(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["acquisitions"][0].update({"valid": False, "missing_reason": "not_measured"})
    _write_manifest(measurement_path, raw)
    with pytest.raises(MeasurementContractError, match="invalid acquisition slot"):
        load_measurement_manifest(measurement_path)

    raw["acquisitions"][0] = {
        "acquisition_id": MEASUREMENT_ACQUISITION_IDS[0],
        "sample_id": SAMPLE_IDS[0],
        "timestamp_s": 0.0,
        "valid": True,
    }
    raw["products"][0]["pair_id"] = "e" * 64
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)
    with pytest.raises(MeasurementContractError, match="absent from sensor_topology"):
        load_measurement_manifest(measurement_path)


def test_calibration_scope_must_cover_every_measurement_product(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["calibration_record"]["scope"]["acquisition_ids"] = [MEASUREMENT_ACQUISITION_IDS[0]]
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="scope does not cover"):
        load_measurement_manifest(measurement_path)


@pytest.mark.parametrize(
    "unsafe_path",
    ["/tmp/product.npy", "../product.npy", "private/product.npy", "C:\\product.npy"],
)
def test_loader_rejects_absolute_traversal_and_private_product_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["products"][0]["data"]["path"] = unsafe_path
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="relative|unsafe|private|portable"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_product_symlinks(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    product = raw["products"][0]
    source = measurement_path.parent / product["data"]["path"]
    link = source.with_name("measurement-linked.npy")
    try:
        link.symlink_to(source.name)
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symlink creation privilege (WinError 1314)")
        raise
    product["data"]["path"] = f"products/{link.name}"
    _rehash_product(product)
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="must not traverse a symlink"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_changed_product_bytes(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    product_path = measurement_path.parent / raw["products"][0]["data"]["path"]
    np.save(product_path, np.zeros((2, 2), dtype=np.float64), allow_pickle=False)

    with pytest.raises(MeasurementContractError, match="sha256 does not match product bytes"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_declared_shape_mismatch(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["products"][0]["data"]["shape"] = [4]
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="shape does not match"):
        load_measurement_manifest(measurement_path)


def test_loader_never_unpickles_object_arrays(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    product = raw["products"][0]
    path = measurement_path.parent / product["data"]["path"]
    values = np.array([{"hidden": "object"}], dtype=object)
    np.save(path, values, allow_pickle=True)
    product["data"].update(
        {
            "sha256": _sha256_bytes(path.read_bytes()),
            "dtype": values.dtype.str,
            "shape": list(values.shape),
        }
    )
    _rehash_product(product)
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="loaded safely"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_npz_with_undeclared_payload(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    product = raw["products"][1]
    product_path = measurement_path.parent / product["data"]["path"]
    np.savez(product_path, values=np.ones((2, 2)), hidden=np.ones(1))
    product["data"]["sha256"] = _sha256_bytes(product_path.read_bytes())
    _rehash_product(product)
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="exactly the declared array_key"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_missing_field_instead_of_guessing(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    del raw["products"][0]["uncertainty"]
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="missing required fields.*uncertainty"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    measurement_path, _, _, _ = _fixture_pair(tmp_path)
    text = measurement_path.read_text(encoding="utf-8")
    text = text.replace(
        '"contract_version": 1,',
        '"contract_version": 1,\n  "contract_version": 1,',
        1,
    )
    measurement_path.write_text(text, encoding="utf-8")

    with pytest.raises(MeasurementContractError, match="duplicate key"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_private_calibration_path(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["calibration_record"]["record_id"] = "/private/calibration/result.json"
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="opaque ID"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_improper_reference_transform(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    transform = raw["reference_chain"]["transforms"][0]
    transform["matrix"][0][0] = -1.0
    _rehash_chain(raw["reference_chain"])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="preserve handedness"):
        load_measurement_manifest(measurement_path)


def test_loader_rejects_split_and_product_metadata_hash_mismatches(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["dataset"]["split_sha256"] = "0" * 64
    _write_manifest(measurement_path, raw)
    with pytest.raises(MeasurementContractError, match="split_sha256 does not match"):
        load_measurement_manifest(measurement_path)

    _rehash_split(raw)
    raw["products"][0]["product_sha256"] = "1" * 64
    _write_manifest(measurement_path, raw)
    with pytest.raises(MeasurementContractError, match="product_sha256 does not match"):
        load_measurement_manifest(measurement_path)


def test_comparison_rejects_calibration_or_tuning_role_leakage(tmp_path: Path) -> None:
    measurement_path, raw, simulation_path, _ = _fixture_pair(tmp_path)
    raw["dataset"]["role"] = "tuning"
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="role.*leakage"):
        _compare_paths(measurement_path, simulation_path)


def test_governance_is_required_and_controls_export(tmp_path: Path) -> None:
    measurement_path, raw, simulation_path, _ = _fixture_pair(tmp_path)
    raw_without_governance = deepcopy(raw)
    del raw_without_governance["governance"]
    _write_manifest(measurement_path, raw_without_governance)
    with pytest.raises(MeasurementContractError, match="missing required fields.*governance"):
        load_measurement_manifest(measurement_path)

    raw["governance"]["redistribution"] = "prohibited"
    _write_manifest(measurement_path, raw)
    comparison = _compare_paths(measurement_path, simulation_path)
    with pytest.raises(MeasurementContractError, match="does not authorize.*export"):
        comparison_export_mapping(comparison)


def test_governance_authority_cannot_be_a_private_path(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["governance"]["authority_id"] = "/private/release-approval.txt"
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="opaque ID"):
        load_measurement_manifest(measurement_path)


def test_nonzero_residual_with_zero_uncertainty_fails_finitely(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for manifest_path, raw in (
        (measurement_path, measurement_raw),
        (simulation_path, simulation_raw),
    ):
        for product in raw["products"]:
            product["uncertainty"]["standard_deviation"] = 0.0
            _rehash_product(product)
        _write_manifest(manifest_path, raw)

    with pytest.raises(MeasurementContractError, match="zero combined uncertainty"):
        _compare_paths(measurement_path, simulation_path)


def test_combined_uncertainty_overflow_fails_during_comparison(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for manifest_path, raw in (
        (measurement_path, measurement_raw),
        (simulation_path, simulation_raw),
    ):
        for product in raw["products"]:
            product["uncertainty"]["standard_deviation"] = 1.7e308
            _rehash_product(product)
        _write_manifest(manifest_path, raw)

    with pytest.raises(MeasurementContractError, match="combined product-value"):
        _compare_paths(measurement_path, simulation_path)


def test_arbitrarily_large_json_integer_raises_measurement_contract_error(
    tmp_path: Path,
) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["products"][0]["uncertainty"]["standard_deviation"] = 10**1000
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="must be a finite number"):
        load_measurement_manifest(measurement_path)


def test_large_json_integer_in_reference_matrix_raises_contract_error(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    transform = raw["reference_chain"]["transforms"][0]
    transform["matrix"][0][0] = 10**1000
    _rehash_transform(transform)
    _rehash_chain(raw["reference_chain"])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="finite 4x4 matrix"):
        load_measurement_manifest(measurement_path)


def test_large_json_integer_in_axis_coordinates_raises_contract_error(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    raw["products"][0]["axes"][0]["coordinates"][0] = 10**1000
    _rehash_product(raw["products"][0])
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="coordinates.*finite values"):
        load_measurement_manifest(measurement_path)


def test_extreme_finite_values_use_overflow_safe_residual_metrics(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for manifest_path, raw, value in (
        (measurement_path, measurement_raw, 1.0e200),
        (simulation_path, simulation_raw, 0.0),
    ):
        for product in raw["products"]:
            values = np.full(tuple(product["data"]["shape"]), value, dtype=np.float64)
            _replace_product_array(manifest_path, product, values)
        _write_manifest(manifest_path, raw)

    comparison = _compare_paths(measurement_path, simulation_path)

    assert comparison.aggregate_rmse == pytest.approx(1.0e200)
    assert math.isfinite(comparison.aggregate_rmse)


def test_nonfinite_derived_residual_fails_before_comparison_output(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for manifest_path, raw, value in (
        (measurement_path, measurement_raw, 1.0e308),
        (simulation_path, simulation_raw, -1.0e308),
    ):
        for product in raw["products"]:
            values = np.full(tuple(product["data"]["shape"]), value, dtype=np.float64)
            _replace_product_array(manifest_path, product, values)
            product["representation"] = "log_power_ratio_db"
            product["unit"] = "dB"
            product["uncertainty"]["unit"] = "dB"
            _rehash_product(product)
        _write_manifest(manifest_path, raw)

    with pytest.raises(MeasurementContractError, match="derived residual values"):
        _compare_paths(measurement_path, simulation_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observable", "range_profile_power", "observable"),
        ("sign_convention", "radial_velocity_positive_approaching", "sign_convention"),
    ],
)
def test_comparison_rejects_observable_and_sign_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    measurement_path, _, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    simulation_raw["products"][0][field] = value
    _rehash_product(simulation_raw["products"][0])
    _write_manifest(simulation_path, simulation_raw)

    with pytest.raises(MeasurementContractError, match=message):
        _compare_paths(measurement_path, simulation_path)


def test_comparison_rejects_representation_and_unit_mismatch(tmp_path: Path) -> None:
    measurement_path, _, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    product = simulation_raw["products"][0]
    product["representation"] = "log_power_ratio_db"
    product["unit"] = "dB"
    product["uncertainty"]["unit"] = "dB"
    _rehash_product(product)
    _write_manifest(simulation_path, simulation_raw)

    with pytest.raises(MeasurementContractError, match="representation"):
        _compare_paths(measurement_path, simulation_path)


def test_comparison_rejects_mixed_contracts_across_acquisitions(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for manifest_path, raw in (
        (measurement_path, measurement_raw),
        (simulation_path, simulation_raw),
    ):
        raw["products"][1]["sign_convention"] = "radial_velocity_positive_approaching"
        _rehash_product(raw["products"][1])
        _write_manifest(manifest_path, raw)

    with pytest.raises(MeasurementContractError, match="share one representation.*sign"):
        _compare_paths(measurement_path, simulation_path)


def test_power_linear_products_must_be_nonnegative(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    product = raw["products"][0]
    values = _read_product_array(measurement_path, product)
    values[0, 0] = -1.0
    _replace_product_array(measurement_path, product, values)
    _write_manifest(measurement_path, raw)

    with pytest.raises(MeasurementContractError, match="power_linear.*nonnegative"):
        load_measurement_manifest(measurement_path)


def test_explicit_negative_axis_transform_reverses_coordinates_and_values(
    tmp_path: Path,
) -> None:
    measurement_path, measurement_raw, simulation_path, _ = _fixture_pair(tmp_path)
    for product in measurement_raw["products"]:
        product["sign_convention"] = "radial_velocity_positive_approaching"
        _replace_product_array(
            measurement_path,
            product,
            np.flip(_read_product_array(measurement_path, product), axis=1),
        )
    _write_manifest(measurement_path, measurement_raw)
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    base = _alignment(measurement, simulation)
    axes = alignment_recipe_mapping(base)["axis_alignments"]
    axes[1]["measurement_to_simulation_scale"] = -1
    signed_recipe = _modified_recipe(
        base,
        axis_alignments=axes,
    )

    comparison = compare_measurement_to_simulation(
        measurement,
        simulation,
        signed_recipe,
    )

    np.testing.assert_allclose(
        comparison.residuals[0].residual,
        [[0.1, -0.1], [0.2, -0.2]],
    )
    assert comparison.sign_convention == "radial_velocity_positive_receding"


def test_negative_axis_rejects_unrecognized_sign_convention_pair(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, _ = _fixture_pair(tmp_path)
    for product in measurement_raw["products"]:
        product["sign_convention"] = "vendor_velocity_positive_left"
        _replace_product_array(
            measurement_path,
            product,
            np.flip(_read_product_array(measurement_path, product), axis=1),
        )
    _write_manifest(measurement_path, measurement_raw)
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    base = _alignment(measurement, simulation)
    axes = alignment_recipe_mapping(base)["axis_alignments"]
    axes[1]["measurement_to_simulation_scale"] = -1

    with pytest.raises(MeasurementContractError, match="not an admitted approaching/receding"):
        compare_measurement_to_simulation(
            measurement,
            simulation,
            _modified_recipe(base, axis_alignments=axes),
        )


def test_v1_comparison_rejects_multi_pair_topology(tmp_path: Path) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    extra_pair = {
        "pair_id": "e" * 64,
        "tx_sensor_id": "sensor:synthetic-tx-secondary",
        "rx_sensor_id": "sensor:synthetic-rx-secondary",
        "topology": "siso",
    }
    measurement_raw["sensor_topology"].append(deepcopy(extra_pair))
    simulation_raw["sensor_topology"].append(deepcopy(extra_pair))
    _write_manifest(measurement_path, measurement_raw)
    _write_manifest(simulation_path, simulation_raw)
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)

    with pytest.raises(MeasurementContractError, match="exactly one sensor pair"):
        compare_measurement_to_simulation(
            measurement,
            simulation,
            _alignment(measurement, simulation),
        )


def test_axis_alignment_scale_is_exactly_positive_or_negative_one() -> None:
    with pytest.raises(MeasurementContractError, match="scale must be exactly"):
        AxisAlignment(
            "radial_velocity",
            "m/s",
            0.0,
            measurement_to_simulation_scale=0,
        )


@pytest.mark.parametrize(
    ("axis_field", "value"),
    [("unit", "cm"), ("reference_frame", "another_world")],
)
def test_comparison_rejects_axis_unit_or_reference_mismatch(
    tmp_path: Path,
    axis_field: str,
    value: str,
) -> None:
    measurement_path, _, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    simulation_raw["products"][0]["axes"][0][axis_field] = value
    if axis_field == "unit":
        simulation_raw["products"][0]["axes"][0]["uncertainty"]["unit"] = value
    _rehash_product(simulation_raw["products"][0])
    _write_manifest(simulation_path, simulation_raw)

    with pytest.raises(MeasurementContractError, match="axis name, unit, or reference"):
        _compare_paths(measurement_path, simulation_path)


def test_comparison_rejects_reference_chain_mismatch(tmp_path: Path) -> None:
    measurement_path, _, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    simulation_raw["reference_chain"]["transforms"][0]["matrix"][0][3] = 0.25
    _rehash_chain(simulation_raw["reference_chain"])
    _write_manifest(simulation_path, simulation_raw)

    with pytest.raises(MeasurementContractError, match="reference transform chains"):
        _compare_paths(measurement_path, simulation_path)


def test_comparison_rejects_missing_identity_or_wrong_axis_alignment(tmp_path: Path) -> None:
    measurement_path, _, simulation_path, _ = _fixture_pair(tmp_path)
    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    recipe = _alignment(measurement, simulation)
    mapping = alignment_recipe_mapping(recipe)
    missing_recipe = _modified_recipe(
        recipe,
        acquisition_pairs=mapping["acquisition_pairs"][:1],
    )
    with pytest.raises(MeasurementContractError, match="cover every selected measurement"):
        compare_measurement_to_simulation(
            measurement,
            simulation,
            missing_recipe,
        )

    wrong_axes = mapping["axis_alignments"]
    wrong_axes[0]["measurement_to_simulation_offset"] = 0.0
    wrong_recipe = _modified_recipe(
        recipe,
        axis_alignments=wrong_axes,
    )
    with pytest.raises(MeasurementContractError, match="aligned coordinates"):
        compare_measurement_to_simulation(
            measurement,
            simulation,
            wrong_recipe,
        )


def test_alignment_rejects_duplicate_identity_mapping() -> None:
    pair = AcquisitionPair(
        SAMPLE_IDS[0],
        MEASUREMENT_ACQUISITION_IDS[0],
        SIMULATION_ACQUISITION_IDS[0],
    )
    with pytest.raises(MeasurementContractError, match="duplicate"):
        MeasurementAlignment(
            acquisition_pairs=(pair, pair),
            axis_alignments=(AxisAlignment("range", "m", 0.0),),
            measurement_to_simulation_time_offset_s=0.0,
        )


def test_numeric_products_and_contract_objects_are_not_mutable(tmp_path: Path) -> None:
    measurement_path, _, _, _ = _fixture_pair(tmp_path)
    product = load_measurement_manifest(measurement_path).products[0]

    with pytest.raises(ValueError, match="read-only"):
        product.values[0, 0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        product.axes[0].coordinates[0] = 99.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        product.values.setflags(write=True)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        product.axes[0].coordinates.setflags(write=True)


def test_source_has_no_calibration_or_generator_dependency() -> None:
    import shared.sensing.measurement as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "generator.core.calibration" not in source
    assert "from generator" not in source
    assert "import generator" not in source
    assert builtins is not None


def test_complex_products_require_explicit_phase_contract_before_comparison(
    tmp_path: Path,
) -> None:
    measurement_path, measurement_raw, simulation_path, simulation_raw = _fixture_pair(tmp_path)
    for manifest_path, raw, delta in (
        (measurement_path, measurement_raw, 1.0j),
        (simulation_path, simulation_raw, 0.5j),
    ):
        for product in raw["products"]:
            values = _read_product_array(manifest_path, product).astype(np.complex128)
            values += delta
            _replace_product_array(manifest_path, product, values)
            product["representation"] = "complex_voltage"
            _rehash_product(product)
        _write_manifest(manifest_path, raw)

    measurement = load_measurement_manifest(measurement_path)
    simulation = load_measurement_manifest(simulation_path)
    assert measurement.products[0].representation == "complex_voltage"
    assert simulation.products[0].representation == "complex_voltage"

    with pytest.raises(MeasurementContractError, match="phase-reference"):
        compare_measurement_to_simulation(
            measurement,
            simulation,
            _alignment(measurement, simulation),
        )


def test_manifest_is_loaded_from_json_not_mutated_caller_state(tmp_path: Path) -> None:
    measurement_path, raw, _, _ = _fixture_pair(tmp_path)
    original = deepcopy(raw)

    load_measurement_manifest(measurement_path)

    assert raw == original
