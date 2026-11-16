r"""Runnable evidence workflow for already-calibrated sensing products.

The JSON/NumPy bundle produced here is an ingress and comparison artifact. It
does not become a second native frame format: schema-v4 typed sensing products
inside packed-HDF5 v2 remain ORCHAV's native storage boundary. The workflow
recomputes native descriptor identities as a crosswalk, but intentionally does
not claim or perform packed-HDF5 ingestion.

The CLI accepts calibrated range-Doppler power for one SISO pair, described by
JSON manifests and numeric NumPy resources. Calibration must happen before
import. The comparison preserves its record identity and scope. To inspect
the input/output contract with synthetic products, use a new directory::

    orchav-sensing-compare demo --output-dir analysis/measurement-comparison-demo

For supplied manifests, declare time/axis alignment and selected acquisitions
in an explicit recipe::

    orchav-sensing-compare compare \
        --measurement-manifest path/to/measurement/manifest.json \
        --simulation-manifest path/to/simulation/manifest.json \
        --alignment-recipe path/to/alignment_recipe.json \
        --output-dir analysis/measurement-comparison

``comparison/measured_simulated_residual.png`` and ``residual_metrics.csv``
describe aligned residuals. ``comparison.json`` contains aggregate metrics,
and ``alignment_recipe.json`` records the executed choices. Outputs are
JSON/CSV/PNG artifacts, not native measurement HDF5 frames.

Residuals are aligned measurement minus simulation. Aggregate errors weight
each selected value equally, so products with more bins contribute more values.
Normalized residuals require explicit mutually uncorrelated standard
uncertainties. Axis uncertainty remains metadata and is not propagated into
residuals. Complex-voltage comparisons and correlated uncertainty are unsupported
by this workflow. See :mod:`shared.sensing.measurement` for typed contracts.

Retain input manifests, numeric resources, executed recipes and the output
inventory together. JSON/CSV carry numerical comparisons. PNG bytes depend on
the plotting environment. Real-data interpretation also requires its source,
alignment and calibration context.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shared.extensions.sensing import SENSING_SCHEMA_VERSION
from shared.sensing.measurement import (
    AGGREGATE_WEIGHTING_POLICY,
    INDEPENDENT_VALUE_UNCERTAINTY_POLICY,
    RESIDUAL_DEFINITION,
    MeasurementAlignmentRecipe,
    MeasurementComparison,
    MeasurementContractError,
    MeasurementManifest,
    alignment_recipe_mapping,
    canonical_measurement_sha256,
    compare_measurement_to_simulation,
    comparison_export_mapping,
    load_measurement_alignment_recipe,
    load_measurement_manifest,
)
from shared.sensing.products import (
    SENSING_PRODUCT_CONTRACT_VERSION,
    attach_product_descriptors,
    build_product_descriptor,
    catalog_axis,
    dense_axis,
)

_ALIGNMENT_CONTRACT = "orchav.sensing.measurement_alignment"
_WORKFLOW_CONTRACT = "orchav.sensing.measurement_workflow"
_PAIR_ID = canonical_measurement_sha256(
    {"tx_sensor_id": "sensor:synthetic-tx", "rx_sensor_id": "sensor:synthetic-rx"},
    domain="synthetic_pair",
)
_SAMPLE_IDS = ("synthetic:alpha", "synthetic:missing-middle", "synthetic:beta")
_MEASUREMENT_ACQUISITION_IDS = (
    "acquisition:synthetic-measurement-alpha",
    "acquisition:synthetic-measurement-missing-middle",
    "acquisition:synthetic-measurement-beta",
)
_SIMULATION_ACQUISITION_IDS = (
    "acquisition:synthetic-simulation-alpha",
    "acquisition:synthetic-simulation-missing-middle",
    "acquisition:synthetic-simulation-beta",
)
_VALID_SAMPLE_INDICES = (0, 2)


@dataclass(frozen=True)
class MeasurementWorkflowPublication:
    root: Path
    comparison_json: Path
    residual_csv: Path
    synchronized_png: Path
    hash_inventory: Path
    inventory_sha256: str


@dataclass(frozen=True)
class _PlottingRuntime:
    pyplot: Any
    matplotlib_version: str
    matplotlib_backend: str

    def evidence(self) -> dict[str, str]:
        return {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "matplotlib": self.matplotlib_version,
            "matplotlib_backend": self.matplotlib_backend,
            "png_portability": "runtime_qualified",
            "png_portability_note": (
                "PNG bytes are exact only for the declared runtime; JSON and CSV carry "
                "the portable semantic evidence."
            ),
        }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_publish(
    destination: Path,
    builder: Callable[[Path], None],
) -> None:
    destination = Path(os.path.abspath(os.fspath(destination)))
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    lock = parent / f".{destination.name}.orchav-publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Another publisher owns output lock: {lock}") from exc
    staging: Path | None = None
    try:
        os.close(descriptor)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
        builder(staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        _promote_directory_noreplace(staging, destination)
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _promote_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically promote a sibling directory without replacing a destination."""

    if os.name == "posix":
        try:
            import ctypes

            renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
            if error_number not in (errno.ENOSYS, errno.EINVAL):
                raise OSError(error_number, os.strerror(error_number), str(destination))
        except AttributeError:
            pass
        raise OSError(
            errno.ENOTSUP,
            "Atomic no-replace directory promotion is unavailable on this platform",
            str(destination),
        )
    if os.name == "nt":
        # Windows rename fails when the destination already exists.
        os.rename(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "Atomic no-replace directory promotion is unavailable on this platform",
        str(destination),
    )


def _load_pyplot() -> _PlottingRuntime:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise RuntimeError(
                "Sensing measurement figures require Matplotlib. "
                "Install ORCHAV with: python -m pip install orchav"
            ) from exc
        raise
    matplotlib.use("Agg", force=True)
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "matplotlib" or exc.name.startswith("matplotlib.")):
            raise RuntimeError(
                "Sensing measurement figures require Matplotlib. "
                "Install ORCHAV with: python -m pip install orchav"
            ) from exc
        raise
    return _PlottingRuntime(
        pyplot=pyplot,
        matplotlib_version=str(matplotlib.__version__),
        matplotlib_backend=str(matplotlib.get_backend()),
    )


def _inventory(root: Path, plotting: _PlottingRuntime) -> dict[str, Any]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "hash_inventory.json":
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "contract": "orchav.sensing.measurement_workflow_inventory",
        "contract_version": 1,
        "files": entries,
        "runtime": plotting.evidence(),
        "inventory_sha256": canonical_measurement_sha256(
            entries,
            domain="workflow_inventory",
        ),
    }


def _write_inventory(root: Path, plotting: _PlottingRuntime) -> dict[str, Any]:
    inventory = _inventory(root, plotting)
    _write_json(root / "hash_inventory.json", inventory)
    return inventory


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


def _write_synthetic_manifest(
    root: Path,
    *,
    source_kind: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    products_dir = root / "products"
    products_dir.mkdir()
    simulation_values = (
        np.array(
            [
                [0.1, 0.2, 0.5, 0.2],
                [0.2, 1.0, 4.0, 1.0],
                [0.1, 0.4, 0.8, 0.3],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.2, 0.3, 0.6, 0.2],
                [0.4, 1.5, 5.0, 1.1],
                [0.1, 0.5, 1.0, 0.4],
            ],
            dtype=np.float64,
        ),
    )
    residual_values = (
        np.array(
            [
                [0.01, -0.02, 0.03, -0.01],
                [0.02, 0.05, -0.04, 0.01],
                [-0.01, 0.02, 0.01, -0.02],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [-0.02, 0.01, 0.02, -0.01],
                [0.03, -0.04, 0.06, -0.02],
                [0.01, -0.01, 0.02, 0.01],
            ],
            dtype=np.float64,
        ),
    )
    is_measurement = source_kind == "measurement"
    acquisition_ids = (
        _MEASUREMENT_ACQUISITION_IDS if is_measurement else _SIMULATION_ACQUISITION_IDS
    )
    timestamps = (0.0, 0.5, 1.0) if is_measurement else (0.05, 0.55, 1.05)
    dataset_id = (
        "dataset:orchav-synthetic-measurement-v1"
        if is_measurement
        else "dataset:orchav-synthetic-simulation-v1"
    )
    products = []
    for product_index, sample_index in enumerate(_VALID_SAMPLE_INDICES):
        aligned = simulation_values[product_index]
        values = (
            np.flip(aligned + residual_values[product_index], axis=0) if is_measurement else aligned
        )
        filename = f"range_doppler_{product_index}.npy"
        product_path = products_dir / filename
        np.save(product_path, values, allow_pickle=False)
        product_id = hashlib.sha256(
            f"orchav-synthetic-{source_kind}-product-{product_index}".encode("ascii")
        ).hexdigest()
        product = {
            "product_id": product_id,
            "acquisition_id": acquisition_ids[sample_index],
            "pair_id": _PAIR_ID,
            "observable": "range_doppler_power",
            "variant": "primary",
            "representation": "power_linear",
            "unit": "1",
            "sign_convention": (
                "doppler_frequency_positive_approaching"
                if is_measurement
                else "doppler_frequency_positive_receding"
            ),
            "axes": [
                {
                    "name": "doppler_frequency",
                    "unit": "Hz",
                    "reference_frame": "sensor_line_of_sight",
                    "coordinates": [-200.0, 0.0, 200.0],
                    "uncertainty": {
                        "status": "known",
                        "unit": "Hz",
                        "standard_deviation": 1.0,
                    },
                },
                {
                    "name": "range",
                    "unit": "m",
                    "reference_frame": "sensor_range",
                    "coordinates": (
                        [1.25, 2.25, 3.25, 4.25] if is_measurement else [1.0, 2.0, 3.0, 4.0]
                    ),
                    "uncertainty": {
                        "status": "known",
                        "unit": "m",
                        "standard_deviation": 0.01,
                    },
                },
            ],
            "data": {
                "path": f"products/{filename}",
                "sha256": _sha256_file(product_path),
                "dtype": values.dtype.str,
                "shape": list(values.shape),
            },
            "uncertainty": {
                "status": "known",
                "unit": "1",
                "standard_deviation": 0.05 if is_measurement else 0.03,
            },
            "product_sha256": "",
        }
        _rehash_product(product)
        products.append(product)

    transform = {
        "transform_id": "transform:synthetic-sensor-to-orchav-world",
        "from_frame": "synthetic_sensor",
        "to_frame": "orchav_world",
        "matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "sha256": "",
    }
    chain = {
        "source_frame": "synthetic_sensor",
        "target_frame": "orchav_world",
        "transforms": [transform],
        "sha256": "",
    }
    _rehash_chain(chain)
    acquisitions = [
        {
            "acquisition_id": acquisition_id,
            "sample_id": sample_id,
            "timestamp_s": timestamp,
            "valid": index in _VALID_SAMPLE_INDICES,
            **({} if index in _VALID_SAMPLE_INDICES else {"missing_reason": "not_measured"}),
        }
        for index, (acquisition_id, sample_id, timestamp) in enumerate(
            zip(acquisition_ids, _SAMPLE_IDS, timestamps)
        )
    ]
    split_sha256 = canonical_measurement_sha256(
        {
            "split_id": "orchav-synthetic-evaluation-v1",
            "sample_ids": list(_SAMPLE_IDS),
        },
        domain="dataset_split",
    )
    calibration = (
        {
            "status": "known",
            "record_id": "urn:orchav:synthetic-calibration-record:v1",
            "sha256": hashlib.sha256(
                b"ORCHAV authored opaque synthetic calibration identity v1"
            ).hexdigest(),
            "scope": {
                "pair_ids": [_PAIR_ID],
                "acquisition_ids": [acquisition_ids[index] for index in _VALID_SAMPLE_INDICES],
                "observable_variants": [
                    {"observable": "range_doppler_power", "variant": "primary"}
                ],
            },
        }
        if is_measurement
        else {"status": "missing", "reason": "not_applicable"}
    )
    manifest = {
        "contract_version": 1,
        "dataset": {
            "dataset_id": dataset_id,
            "source_kind": source_kind,
            "role": "evaluation",
            "split_id": "orchav-synthetic-evaluation-v1",
            "split_sha256": split_sha256,
        },
        "governance": {
            "provenance_id": "orchav:authored-synthetic-measurement-workflow-v1",
            "license_spdx": "GPL-3.0-only",
            "redistribution": "permitted",
            "privacy": "none",
            "authority_id": "urn:orchav:authored-synthetic-fixture",
        },
        "acquisitions": acquisitions,
        "sensor_topology": [
            {
                "pair_id": _PAIR_ID,
                "tx_sensor_id": "sensor:synthetic-tx",
                "rx_sensor_id": "sensor:synthetic-rx",
                "topology": "siso",
            }
        ],
        "clock": {
            "clock_id": f"clock:synthetic-{source_kind}",
            "origin": "synthetic_fixture_start",
            "unit": "s",
        },
        "calibration_record": calibration,
        "reference_chain": chain,
        "processing_lineage": {
            "processing_run_id": f"processing:synthetic-{source_kind}-run-v1",
            "processor": f"orchav.synthetic.{source_kind}",
            "processor_version": "1",
            "recipe_sha256": hashlib.sha256(f"{source_kind}-recipe-v1".encode()).hexdigest(),
            "source_sha256": [hashlib.sha256(f"{source_kind}-source-v1".encode()).hexdigest()],
        },
        "products": products,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = canonical_measurement_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"},
        domain="manifest",
    )
    path = root / "manifest.json"
    _write_json(path, manifest)
    load_measurement_manifest(path)
    return path


def _build_alignment_recipe(
    measurement: MeasurementManifest,
    simulation: MeasurementManifest,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "contract": _ALIGNMENT_CONTRACT,
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
                "sample_id": _SAMPLE_IDS[index],
                "measurement_acquisition_id": _MEASUREMENT_ACQUISITION_IDS[index],
                "simulation_acquisition_id": _SIMULATION_ACQUISITION_IDS[index],
            }
            for index in _VALID_SAMPLE_INDICES
        ],
        "axis_alignments": [
            {
                "name": "doppler_frequency",
                "unit": "Hz",
                "measurement_to_simulation_scale": -1,
                "measurement_to_simulation_offset": 0.0,
                "absolute_tolerance": 1e-12,
            },
            {
                "name": "range",
                "unit": "m",
                "measurement_to_simulation_scale": 1,
                "measurement_to_simulation_offset": -0.25,
                "absolute_tolerance": 1e-12,
            },
        ],
        "measurement_to_simulation_time_offset_s": 0.05,
        "timestamp_tolerance_s": 1e-12,
    }
    digest = canonical_measurement_sha256(body, domain="alignment_recipe")
    body["alignment_id"] = f"sha256:{digest}"
    body["alignment_sha256"] = digest
    return body


def _native_crosswalk(
    manifest: MeasurementManifest,
    recipe: MeasurementAlignmentRecipe,
) -> list[dict[str, Any]]:
    if recipe.observable != "range_doppler_power" or recipe.variant != "primary":
        raise MeasurementContractError(
            "Workflow native crosswalk supports only range_doppler_power/primary"
        )
    topology = {item.pair_id: item for item in manifest.sensor_topology}
    records = []
    for product in manifest.products:
        if product.observable != recipe.observable or product.variant != recipe.variant:
            continue
        if (
            product.representation != "power_linear"
            or product.unit != "1"
            or len(product.axes) != 2
            or [(axis.name, axis.unit, axis.reference_frame) for axis in product.axes]
            != [
                ("doppler_frequency", "Hz", "sensor_line_of_sight"),
                ("range", "m", "sensor_range"),
            ]
            or product.sign_convention
            not in {
                "doppler_frequency_positive_approaching",
                "doppler_frequency_positive_receding",
            }
        ):
            raise MeasurementContractError(
                "Workflow native crosswalk supports exact range-Doppler power axes, "
                "reference frames, and Doppler sign conventions only"
            )
        pair = topology[product.pair_id]
        if pair.topology != "siso":
            raise MeasurementContractError(
                "Workflow native crosswalk supports exact SISO pair topology only"
            )
        data_key = "range_doppler_power_by_pair"
        sensing: dict[str, Any] = {
            "pair_axis": [
                {
                    "pair_index": 0,
                    "frame_pair_index": 0,
                    "pair_id": pair.pair_id,
                    "tx_sensor_id": pair.tx_sensor_id,
                    "rx_sensor_id": pair.rx_sensor_id,
                    "tx_device_index": 0,
                    "rx_device_index": 0,
                }
            ],
            "doppler_axis_hz": product.axes[0].coordinates,
            "range_axis_m": product.axes[1].coordinates,
            data_key: product.values[np.newaxis, ...],
            "temporal": {
                "clock": {"clock_id": manifest.clock.clock_id},
                "acquisition": {
                    "acquisition_id": product.acquisition_id,
                    "pair_id": pair.pair_id,
                    "tx_sensor_id": pair.tx_sensor_id,
                    "rx_sensor_id": pair.rx_sensor_id,
                    "start_time_s": product.timestamp_s,
                    "end_time_s": product.timestamp_s,
                    "default_reference_time_s": product.timestamp_s,
                    "default_reference_policy": "measured_sample_exact",
                },
                "result": {
                    "processing_run_id": manifest.processing_lineage.processing_run_id,
                    "processing_recipe_id": manifest.processing_lineage.recipe_sha256,
                    "reference_time_s": product.timestamp_s,
                    "reference_time_policy": "measured_sample_exact",
                },
            },
        }
        descriptor = build_product_descriptor(
            sensing,
            data_key=data_key,
            kind="range_doppler",
            axes=(
                catalog_axis("pair", size=1, catalog_key="pair_axis"),
                dense_axis(
                    "doppler_frequency",
                    size=product.values.shape[0],
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                    coordinate_semantics=(
                        "fftshifted_doppler_frequency__" + product.sign_convention
                    ),
                    reference_frame="sensor_line_of_sight",
                ),
                dense_axis(
                    "range",
                    size=product.values.shape[1],
                    unit="m",
                    coordinate_key="range_axis_m",
                    reference_frame="sensor_range",
                ),
            ),
            representation="power_linear",
            unit="1",
            scale_semantics="relative_unitless",
            pair_ids=(pair.pair_id,),
            evidence_role=(
                "observation_derived"
                if manifest.dataset.source_kind == "measurement"
                else "simulation_derived"
            ),
            source_product_ids=(product.product_sha256,),
            combine={"policy": "siso_identity", "normalization": "identity"},
            processing={
                "ingress_contract": "orchav.sensing.measurement.v1",
                "ingress_manifest_sha256": manifest.manifest_sha256,
                "ingress_observable": product.observable,
                "ingress_variant": product.variant,
                "sign_convention": product.sign_convention,
                "sample_id": product.sample_id,
                "clock_origin": manifest.clock.origin,
                "processor": manifest.processing_lineage.processor,
                "processor_version": manifest.processing_lineage.processor_version,
                "processing_source_sha256": list(manifest.processing_lineage.source_sha256),
                "calibration_record_sha256": manifest.calibration_record.sha256,
                "reference_chain_sha256": manifest.reference_chain.sha256,
            },
        )
        attach_product_descriptors(sensing, [descriptor])
        records.append(
            {
                "acquisition_id": product.acquisition_id,
                "ingress_product_id": product.product_id,
                "ingress_product_sha256": product.product_sha256,
                "native_schema_v4_product_id": descriptor["product_id"],
                "native_data_sha256": descriptor["data_sha256"],
                "native_data_key": data_key,
                "native_kind": descriptor["kind"],
                "native_shape": descriptor["shape"],
                "native_axes": [axis["name"] for axis in descriptor["axes"]],
                "ingress_binding": {
                    "sample_id": product.sample_id,
                    "timestamp_s": product.timestamp_s,
                    "clock": {
                        "clock_id": manifest.clock.clock_id,
                        "origin": manifest.clock.origin,
                        "unit": manifest.clock.unit,
                    },
                    "pair": {
                        "pair_id": pair.pair_id,
                        "tx_sensor_id": pair.tx_sensor_id,
                        "rx_sensor_id": pair.rx_sensor_id,
                        "topology": pair.topology,
                    },
                    "observable": product.observable,
                    "variant": product.variant,
                    "representation": product.representation,
                    "unit": product.unit,
                    "sign_convention": product.sign_convention,
                    "axes": [
                        {
                            "name": axis.name,
                            "unit": axis.unit,
                            "reference_frame": axis.reference_frame,
                            "coordinates": axis.coordinates.tolist(),
                        }
                        for axis in product.axes
                    ],
                    "data": {
                        "path": product.data_path,
                        "sha256": product.data_sha256,
                        "dtype": product.values.dtype.str,
                        "shape": list(product.values.shape),
                    },
                    "processing": {
                        "processing_run_id": (manifest.processing_lineage.processing_run_id),
                        "processor": manifest.processing_lineage.processor,
                        "processor_version": (manifest.processing_lineage.processor_version),
                        "recipe_sha256": manifest.processing_lineage.recipe_sha256,
                        "source_sha256": list(manifest.processing_lineage.source_sha256),
                    },
                },
                "native_descriptor": descriptor,
                "numeric_reverification": (
                    "requires_the_authenticated_external_ingress_resource; "
                    "the evidence retains semantic bindings and content hashes"
                ),
            }
        )
    if not records:
        raise MeasurementContractError("No products were available for native crosswalk")
    return records


def _residual_csv(
    comparison: MeasurementComparison,
    simulation: MeasurementManifest,
) -> bytes:
    stream = io.StringIO(newline="")
    fields = [
        "sample_id",
        "measurement_acquisition_id",
        "simulation_acquisition_id",
        "simulation_timestamp_s",
        "timestamp_residual_s",
        "flat_index",
        "axis_indices_json",
        "axis_coordinates_json",
        "residual_real",
        "residual_imag",
        "residual_absolute",
        "product_rmse",
        "product_max_absolute_error",
        "product_normalized_rmse",
        "aggregate_rmse",
        "aggregate_max_absolute_error",
        "aggregate_normalized_rmse",
        "residual_definition",
        "aggregate_weighting_policy",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    simulated = {
        product.acquisition_id: product
        for product in simulation.products
        if product.observable == comparison.observable and product.variant == comparison.variant
    }
    for residual in comparison.residuals:
        simulation_product = simulated[residual.simulation_acquisition_id]
        for index, value in enumerate(residual.residual.reshape(-1)):
            scalar = complex(value)
            indices = np.unravel_index(index, residual.residual.shape)
            coordinates = {
                axis.name: float(axis.coordinates[axis_index])
                for axis, axis_index in zip(simulation_product.axes, indices)
            }
            writer.writerow(
                {
                    "sample_id": residual.sample_id,
                    "measurement_acquisition_id": residual.measurement_acquisition_id,
                    "simulation_acquisition_id": residual.simulation_acquisition_id,
                    "simulation_timestamp_s": format(residual.simulation_timestamp_s, ".17g"),
                    "timestamp_residual_s": format(residual.timestamp_residual_s, ".17g"),
                    "flat_index": index,
                    "axis_indices_json": json.dumps(
                        [int(axis_index) for axis_index in indices],
                        separators=(",", ":"),
                    ),
                    "axis_coordinates_json": json.dumps(
                        coordinates,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "residual_real": format(scalar.real, ".17g"),
                    "residual_imag": format(scalar.imag, ".17g"),
                    "residual_absolute": format(abs(scalar), ".17g"),
                    "product_rmse": format(residual.rmse, ".17g"),
                    "product_max_absolute_error": format(residual.max_absolute_error, ".17g"),
                    "product_normalized_rmse": (
                        ""
                        if residual.normalized_rmse is None
                        else format(residual.normalized_rmse, ".17g")
                    ),
                    "aggregate_rmse": format(comparison.aggregate_rmse, ".17g"),
                    "aggregate_max_absolute_error": format(
                        comparison.aggregate_max_absolute_error, ".17g"
                    ),
                    "aggregate_normalized_rmse": (
                        ""
                        if comparison.aggregate_normalized_rmse is None
                        else format(comparison.aggregate_normalized_rmse, ".17g")
                    ),
                    "residual_definition": RESIDUAL_DEFINITION,
                    "aggregate_weighting_policy": AGGREGATE_WEIGHTING_POLICY,
                }
            )
    return stream.getvalue().encode("utf-8")


def _write_synchronized_plot(
    path: Path,
    comparison: MeasurementComparison,
    simulation: MeasurementManifest,
    pyplot: Any,
) -> None:
    simulated = {
        product.acquisition_id: product
        for product in simulation.products
        if product.observable == comparison.observable and product.variant == comparison.variant
    }
    rows = len(comparison.residuals)
    figure, axes = pyplot.subplots(rows, 3, figsize=(12, 3.5 * rows), squeeze=False)
    for row, residual in enumerate(comparison.residuals):
        simulation_values = simulated[residual.simulation_acquisition_id].values
        measured_values = simulation_values + residual.residual
        if np.iscomplexobj(simulation_values):
            simulation_values = np.abs(simulation_values)
            measured_values = np.abs(measured_values)
        limit_min = float(min(np.min(simulation_values), np.min(measured_values)))
        limit_max = float(max(np.max(simulation_values), np.max(measured_values)))
        residual_limit = float(np.max(np.abs(residual.residual)))
        images = (
            axes[row, 0].imshow(
                measured_values,
                origin="lower",
                aspect="auto",
                vmin=limit_min,
                vmax=limit_max,
            ),
            axes[row, 1].imshow(
                simulation_values,
                origin="lower",
                aspect="auto",
                vmin=limit_min,
                vmax=limit_max,
            ),
            axes[row, 2].imshow(
                residual.residual.real,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-residual_limit,
                vmax=residual_limit,
            ),
        )
        for column, title in enumerate(
            (
                "Measured (aligned)",
                "Simulated",
                "Residual (aligned measurement - simulation)",
            )
        ):
            axes[row, column].set_title(f"{title} — {residual.sample_id}")
            axes[row, column].set_xlabel("Range bin")
            axes[row, column].set_ylabel("Doppler bin")
            figure.colorbar(images[column], ax=axes[row, column], fraction=0.046, pad=0.04)
    figure.suptitle("Already-calibrated sensing comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=140, metadata={"Software": "ORCHAV"})
    pyplot.close(figure)


def _comparison_evidence(
    measurement: MeasurementManifest,
    simulation: MeasurementManifest,
    recipe: MeasurementAlignmentRecipe,
    plotting: _PlottingRuntime,
) -> dict[str, Any]:
    calibration = measurement.calibration_record
    assert calibration.record_id is not None and calibration.sha256 is not None
    return {
        "contract": _WORKFLOW_CONTRACT,
        "contract_version": 1,
        "calibration_record": {
            "record_id": calibration.record_id,
            "sha256": calibration.sha256,
            "content_included": False,
        },
        "governance": {
            "measurement": {
                "provenance_id": measurement.governance.provenance_id,
                "license_spdx": measurement.governance.license_spdx,
                "redistribution": measurement.governance.redistribution,
                "privacy": measurement.governance.privacy,
                "authority_id": measurement.governance.authority_id,
            },
            "simulation": {
                "provenance_id": simulation.governance.provenance_id,
                "license_spdx": simulation.governance.license_spdx,
                "redistribution": simulation.governance.redistribution,
                "privacy": simulation.governance.privacy,
                "authority_id": simulation.governance.authority_id,
            },
        },
        "alignment": {
            "alignment_id": recipe.alignment_id,
            "alignment_sha256": recipe.alignment_sha256,
        },
        "runtime": plotting.evidence(),
        "storage_boundary": {
            "ingress_format": "json_manifest_plus_numpy",
            "ingress_only": True,
            "native_frame_schema_version": SENSING_SCHEMA_VERSION,
            "native_product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
            "native_container": "packed_hdf5_v2",
            "packed_hdf5_adapter_status": "not_implemented_fail_closed",
            "statement": (
                "JSON/NumPy is an already-calibrated ingress envelope only; packed-HDF5 v2 "
                "with schema-v4 typed sensing products remains the native storage boundary."
            ),
        },
        "native_crosswalk": {
            "scope": {
                "observable": "range_doppler_power",
                "variant": "primary",
                "pair_topology": "siso",
                "context": "standalone_ingress_evidence",
                "device_indices": "local_single_pair_zero_indices",
                "temporal_support": "instantaneous_acquisition_timestamp",
                "native_hdf5_published": False,
            },
            "measurement": _native_crosswalk(measurement, recipe),
            "simulation": _native_crosswalk(simulation, recipe),
        },
    }


def publish_measurement_comparison(
    measurement_manifest_path: str | Path,
    simulation_manifest_path: str | Path,
    alignment_recipe: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
) -> MeasurementWorkflowPublication:
    """Atomically publish comparison evidence from existing ingress manifests."""

    plotting = _load_pyplot()
    measurement = load_measurement_manifest(measurement_manifest_path)
    simulation = load_measurement_manifest(simulation_manifest_path)
    recipe = load_measurement_alignment_recipe(alignment_recipe)
    comparison = compare_measurement_to_simulation(measurement, simulation, recipe)
    destination = Path(output_dir)
    inventory_result: dict[str, Any] = {}

    def build(staging: Path) -> None:
        _write_json(staging / "alignment_recipe.json", alignment_recipe_mapping(recipe))
        _write_json(staging / "comparison.json", comparison_export_mapping(comparison))
        (staging / "residual_metrics.csv").write_bytes(_residual_csv(comparison, simulation))
        _write_synchronized_plot(
            staging / "measured_simulated_residual.png",
            comparison,
            simulation,
            plotting.pyplot,
        )
        _write_json(
            staging / "workflow_evidence.json",
            _comparison_evidence(measurement, simulation, recipe, plotting),
        )
        inventory_result.update(_write_inventory(staging, plotting))

    _atomic_publish(destination, build)
    return MeasurementWorkflowPublication(
        root=destination.resolve(),
        comparison_json=destination.resolve() / "comparison.json",
        residual_csv=destination.resolve() / "residual_metrics.csv",
        synchronized_png=destination.resolve() / "measured_simulated_residual.png",
        hash_inventory=destination.resolve() / "hash_inventory.json",
        inventory_sha256=inventory_result["inventory_sha256"],
    )


def publish_synthetic_measurement_workflow(
    output_dir: str | Path,
) -> MeasurementWorkflowPublication:
    """Atomically publish the deterministic ORCHAV-authored synthetic workflow."""

    plotting = _load_pyplot()
    destination = Path(output_dir)
    inventory_result: dict[str, Any] = {}

    def build(staging: Path) -> None:
        measurement_path = _write_synthetic_manifest(
            staging / "inputs" / "measurement",
            source_kind="measurement",
        )
        simulation_path = _write_synthetic_manifest(
            staging / "inputs" / "simulation",
            source_kind="simulation",
        )
        measurement = load_measurement_manifest(measurement_path)
        simulation = load_measurement_manifest(simulation_path)
        raw_recipe = _build_alignment_recipe(measurement, simulation)
        recipe = load_measurement_alignment_recipe(raw_recipe)
        comparison = compare_measurement_to_simulation(measurement, simulation, recipe)
        comparison_dir = staging / "comparison"
        comparison_dir.mkdir()
        _write_json(
            comparison_dir / "alignment_recipe.json",
            alignment_recipe_mapping(recipe),
        )
        _write_json(
            comparison_dir / "comparison.json",
            comparison_export_mapping(comparison),
        )
        (comparison_dir / "residual_metrics.csv").write_bytes(_residual_csv(comparison, simulation))
        _write_synchronized_plot(
            comparison_dir / "measured_simulated_residual.png",
            comparison,
            simulation,
            plotting.pyplot,
        )
        evidence = _comparison_evidence(measurement, simulation, recipe, plotting)
        evidence["synthetic_inputs"] = {
            "measurement_manifest": "inputs/measurement/manifest.json",
            "simulation_manifest": "inputs/simulation/manifest.json",
            "numeric_resources_retained": True,
            "missing_sample_id": _SAMPLE_IDS[1],
        }
        _write_json(comparison_dir / "workflow_evidence.json", evidence)
        _write_inventory(comparison_dir, plotting)
        inventory_result.update(_write_inventory(staging, plotting))

    _atomic_publish(destination, build)
    root = destination.resolve()
    return MeasurementWorkflowPublication(
        root=root,
        comparison_json=root / "comparison" / "comparison.json",
        residual_csv=root / "comparison" / "residual_metrics.csv",
        synchronized_png=root / "comparison" / "measured_simulated_residual.png",
        hash_inventory=root / "hash_inventory.json",
        inventory_sha256=inventory_result["inventory_sha256"],
    )


__all__ = [
    "MeasurementWorkflowPublication",
    "publish_measurement_comparison",
    "publish_synthetic_measurement_workflow",
]
