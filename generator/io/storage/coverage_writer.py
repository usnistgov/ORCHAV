"""HDF5 writer for schema-v2 coverage maps.

Coverage output is separate from MPC frame output. It stores gridded coverage
metrics, TX/RX metadata, solver metadata, and derived values under the shared
coverage schema rather than the packed MPC-frame schema.

The writer owns the on-disk compact layout. It materializes canonical
``path_gain_linear`` and compact derived labels, then records which additional
metrics consumers can derive from recipes. It intentionally does not expand the
file into one dense dataset per requested metric.
"""

import json
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from shared.coverage.schema import (
    COVERAGE_CANONICAL_VALUE_METRICS,
    COVERAGE_COMPACT_DERIVED_METRICS,
    COVERAGE_DERIVABLE_METRICS,
    COVERAGE_FRAME_GENERATION_ID_ATTR,
    COVERAGE_FRAME_SET_ID_ATTR,
    COVERAGE_HDF5_SCHEMA_VERSION,
    COVERAGE_HDF5_STORAGE_LAYOUT,
    COVERAGE_NO_SERVING_TX,
    coverage_available_metrics,
    coverage_metric_recipe_metadata,
    normalise_path_gain_tensor,
)
from shared.logging import get_logger

logger = get_logger(__name__)

COVERAGE_HDF5_SUFFIXES = {".h5", ".hdf5"}
COVERAGE_VALUE_CHUNK_EDGE_LIMIT = 128
COVERAGE_PARTIAL_CREATE_ATTEMPTS = 8


@contextmanager
def _atomic_coverage_file(output_path: Path) -> Iterator[Any]:
    """Yield a private HDF5 file and atomically publish it on success."""
    import h5py

    partial_path: Path | None = None
    try:
        for _attempt in range(COVERAGE_PARTIAL_CREATE_ATTEMPTS):
            partial_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.partial")
            try:
                coverage_file = h5py.File(partial_path, "x")
            except FileExistsError:
                # This invocation does not own the colliding path. Leave it
                # untouched and choose another private name.
                partial_path = None
                continue
            break
        else:
            raise FileExistsError(
                f"Could not allocate a private coverage output after "
                f"{COVERAGE_PARTIAL_CREATE_ATTEMPTS} attempts"
            )

        assert partial_path is not None
        with coverage_file:
            yield coverage_file
            coverage_file.flush()
        os.replace(partial_path, output_path)
    except BaseException:
        if partial_path is not None:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.warning(
                    "Could not remove partial coverage output %s: %s",
                    partial_path,
                    cleanup_error,
                )
        raise


def save_coverage_map(
    coverage_data: dict[str, Any],
    scenario_configuration: Any,
    *,
    output_path: Path | None = None,
    frame_generation_id: str | None = None,
    frame_set_id: str | None = None,
) -> str | None:
    """Write one scenario's canonical coverage HDF5 file.

    ``None`` is reserved for an intentional ``save.data.enabled: false``
    configuration. Path, schema, and filesystem failures retain their original
    exception type so the generator pipeline cannot report a successful run
    without its enabled primary coverage output.
    """
    try:
        root_value = getattr(scenario_configuration, "root", None)
        if not isinstance(root_value, (str, os.PathLike)):
            raise ValueError("Scenario coverage output requires a concrete scenario root")
        coverage_config = getattr(scenario_configuration, "coverage_cfg", None)
        if not isinstance(coverage_config, Mapping):
            raise ValueError("Scenario coverage output requires coverage configuration")

        save_root = coverage_config.get("save", {}) or {}
        save_spec = save_root.get("data", save_root)
        if save_spec.get("enabled", True):
            if output_path is None:
                output_dir = Path(root_value) / "coverage"
                output_dir.mkdir(parents=True, exist_ok=True)
                resolved_output_path = output_dir / "coverage_maps.h5"
            else:
                resolved_output_path = Path(output_path)
        else:
            logger.info("Coverage HDF5 persistence is disabled")
            return None

        resolved_output_path = _ensure_coverage_hdf5_path(resolved_output_path)
        logger.info("Saving coverage map to: %s", resolved_output_path)
        save_coverage_hdf5(
            coverage_data,
            resolved_output_path,
            compression=save_root.get("compression", "lzf"),
            frame_generation_id=frame_generation_id,
            frame_set_id=frame_set_id,
        )
        if not resolved_output_path.is_file():
            raise OSError(
                "Coverage writer returned without creating the enabled output: "
                f"{resolved_output_path}"
            )

        logger.info("Coverage map saved successfully")
        return str(resolved_output_path)
    except (OSError, KeyError, ValueError, TypeError):
        logger.exception("Error saving coverage map")
        raise


def _ensure_coverage_hdf5_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() not in COVERAGE_HDF5_SUFFIXES:
        raise ValueError("coverage output path must end in .h5 or .hdf5")
    return output_path


def _string_dataset_values(values: list[str]) -> np.ndarray:
    return np.asarray([str(value).encode("utf-8") for value in values])


def _coverage_dataset_kwargs(arr: np.ndarray, compression: str | None) -> dict[str, Any]:
    """Return HDF5 creation options for gridded coverage tensors.

    These are h5py dataset chunks, not ORCHAV frame chunks. Coverage arrays are
    read one metric or height slice at a time, so chunking favors small spatial
    tiles while keeping the step/height/TX axes narrow.
    """
    kwargs: dict[str, Any] = {}
    if arr.ndim == 5:
        kwargs["chunks"] = (
            1,
            1,
            1,
            min(arr.shape[-2], COVERAGE_VALUE_CHUNK_EDGE_LIMIT),
            min(arr.shape[-1], COVERAGE_VALUE_CHUNK_EDGE_LIMIT),
        )
    elif arr.ndim == 4:
        kwargs["chunks"] = (
            1,
            1,
            min(arr.shape[-2], COVERAGE_VALUE_CHUNK_EDGE_LIMIT),
            min(arr.shape[-1], COVERAGE_VALUE_CHUNK_EDGE_LIMIT),
        )
    elif arr.ndim == 3:
        kwargs["chunks"] = (
            1,
            min(arr.shape[-2], COVERAGE_VALUE_CHUNK_EDGE_LIMIT),
            min(arr.shape[-1], COVERAGE_VALUE_CHUNK_EDGE_LIMIT),
        )
    elif arr.ndim >= 2 and arr.size > 0:
        kwargs["chunks"] = True
    if arr.ndim >= 2 and arr.size > 0 and compression is not None:
        kwargs["compression"] = compression
        if np.issubdtype(arr.dtype, np.number):
            kwargs["shuffle"] = True
    return kwargs


def _serving_tx_dtype(tx_count: int) -> np.dtype:
    """Use the smallest signed dtype that can represent the no-service sentinel."""
    if tx_count <= np.iinfo(np.int8).max:
        return np.dtype(np.int8)
    if tx_count <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    return np.dtype(np.int32)


def _materialized_derived_layers(
    derived: dict[str, Any],
    *,
    tx_count: int,
) -> dict[str, np.ndarray]:
    """Exclude derived layers that coverage recipes can reproduce.

    Scalar views are computed from ``path_gain_linear`` and metadata on read.
    ``serving_tx`` remains physical because it is a compact categorical result,
    not merely another scalar view of path gain.
    """
    materialized: dict[str, np.ndarray] = {}
    for key, value in derived.items():
        if key not in COVERAGE_COMPACT_DERIVED_METRICS and key in COVERAGE_DERIVABLE_METRICS:
            continue
        arr = np.asarray(value)
        if key == "serving_tx":
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError("serving_tx values must be numeric TX indices")
            invalid = ~np.isfinite(arr) | (arr < COVERAGE_NO_SERVING_TX) | (arr >= tx_count)
            if np.any(invalid):
                raise ValueError(
                    f"serving_tx values must be {COVERAGE_NO_SERVING_TX} or 0..{tx_count - 1}"
                )
            arr = arr.astype(_serving_tx_dtype(tx_count), copy=False)
        materialized[key] = arr
    return materialized


def save_coverage_hdf5(
    coverage_data: dict[str, Any],
    output_path: Path,
    compression: str | None = "lzf",
    *,
    frame_generation_id: str | None = None,
    frame_set_id: str | None = None,
) -> None:
    """Write schema-v2 gridded coverage data to HDF5.

    The v2 layout stores one canonical dense physical layer
    (``path_gain_linear``) plus compact derived classifications. Common scalar
    views such as path loss, RSS, SINR, and TX margin are advertised through
    recipes and derived by consumers on demand.
    """
    output_path = _ensure_coverage_hdf5_path(output_path)
    compression = None if compression in (None, "none", "None") else str(compression)
    for field_name, value in (
        (COVERAGE_FRAME_GENERATION_ID_ATTR, frame_generation_id),
        (COVERAGE_FRAME_SET_ID_ATTR, frame_set_id),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field_name} must be a non-empty string when supplied")
    logger.debug("Coverage data keys: %s", list(coverage_data.keys()))

    metadata = coverage_data.get("metadata", {}) or {}
    # Path gain is the canonical dense tensor. Recipes below advertise scalar
    # views that consumers derive from it instead of storing duplicate arrays.
    stored_values = coverage_data.get("stored_values") or {
        "path_gain_linear": coverage_data["path_gain_linear"]
    }
    path_gain = normalise_path_gain_tensor(
        np.asarray(stored_values.get("path_gain_linear", coverage_data["path_gain_linear"]))
    )
    tx_count = int(path_gain.shape[2])
    tx_names = list(coverage_data.get("tx_names", []))
    metrics_store = list(metadata.get("metrics_store", []))
    metrics_derived = list(metadata.get("metrics_derived", []))
    materialized_derived = _materialized_derived_layers(
        coverage_data.get("derived", {}) or {},
        tx_count=tx_count,
    )
    available_metrics = coverage_available_metrics(
        tx_names=tx_names,
        tx_count=tx_count,
        metrics_store=metrics_store or COVERAGE_CANONICAL_VALUE_METRICS,
        metrics_derived=metrics_derived,
        primary_metric=str(coverage_data.get("metric_name", "best_path_loss_db")),
        materialized_values=COVERAGE_CANONICAL_VALUE_METRICS,
        materialized_derived=materialized_derived.keys(),
    )

    with _atomic_coverage_file(output_path) as f:
        f.attrs["coverage_schema_version"] = COVERAGE_HDF5_SCHEMA_VERSION
        f.attrs["coverage_storage_layout"] = COVERAGE_HDF5_STORAGE_LAYOUT
        if frame_generation_id is not None:
            f.attrs[COVERAGE_FRAME_GENERATION_ID_ATTR] = frame_generation_id
        if frame_set_id is not None:
            f.attrs[COVERAGE_FRAME_SET_ID_ATTR] = frame_set_id
        f.attrs["metric_name"] = coverage_data.get("metric_name", "path_loss_db")
        f.attrs["value_min"] = float(coverage_data.get("value_min", 0.0))
        f.attrs["value_max"] = float(coverage_data.get("value_max", 0.0))

        grid = f.create_group("grid")
        origin = np.asarray(coverage_data["grid_origin"], dtype=np.float64)
        spacing = np.asarray(coverage_data["grid_spacing"], dtype=np.float64)
        shape = np.asarray(coverage_data["grid_shape"], dtype=np.int32)
        heights = np.asarray(coverage_data.get("heights", []), dtype=np.float64)
        grid.create_dataset("origin_xy", data=origin[:2])
        grid.create_dataset("origin", data=origin)
        grid.create_dataset("spacing_xy", data=spacing[:2])
        grid.create_dataset("shape_yx", data=np.asarray([shape[1], shape[0]], dtype=np.int32))
        grid.create_dataset("shape_xyz", data=shape)
        grid.create_dataset("heights_m", data=heights)

        bbox_xy = metadata.get("bbox_xy")
        if bbox_xy is not None:
            grid.create_dataset("bbox_xy", data=np.asarray(bbox_xy, dtype=np.float64))

        tx_group = f.create_group("tx")
        tx_group.create_dataset("positions_m", data=coverage_data.get("tx_positions", []))
        tx_group.create_dataset("powers_dbm", data=coverage_data.get("tx_power_dbm", []))
        tx_group.create_dataset(
            "names", data=_string_dataset_values(coverage_data.get("tx_names", []))
        )

        rx_group = f.create_group("rx")
        rx_positions = np.asarray(coverage_data.get("rx_positions", []), dtype=np.float32)
        rx_positions = rx_positions.reshape((0, 3)) if rx_positions.size == 0 else rx_positions
        rx_group.create_dataset("positions_m", data=rx_positions)
        rx_group.create_dataset(
            "names", data=_string_dataset_values(coverage_data.get("rx_names", []))
        )

        solver_group = f.create_group("solver")
        solver = metadata.get("solver", {}) or {}
        for key, value in solver.items():
            if value is None:
                continue
            solver_group.attrs[key] = value

        values_group = f.create_group("values")
        values_group.create_dataset(
            "path_gain_linear",
            data=path_gain.astype(np.float32, copy=False),
            **_coverage_dataset_kwargs(path_gain, compression),
        )

        derived_group = f.create_group("derived")
        for key, arr in materialized_derived.items():
            dataset = derived_group.create_dataset(
                key, data=arr, **_coverage_dataset_kwargs(arr, compression)
            )
            if key == "serving_tx":
                dataset.attrs["no_service_value"] = COVERAGE_NO_SERVING_TX
                dataset.attrs["description"] = (
                    "zero-based serving TX index; no_service_value marks cells "
                    "without finite signal"
                )

        metadata_group = f.create_group("metadata")
        metadata_group.attrs["tx_mode"] = str(metadata.get("tx_mode", "per_tx"))
        metadata_group.attrs["metrics_store"] = json.dumps(metrics_store)
        metadata_group.attrs["metrics_derived"] = json.dumps(metrics_derived)
        # ``materialized_*`` is the physical on-disk inventory; ``available`` is
        # the larger logical menu that readers can derive lazily.
        metadata_group.attrs["materialized_values"] = json.dumps(["path_gain_linear"])
        metadata_group.attrs["materialized_derived"] = json.dumps(sorted(materialized_derived))
        metadata_group.attrs["available_metrics"] = json.dumps(available_metrics)
        metadata_group.attrs["derived_metric_recipes"] = json.dumps(
            coverage_metric_recipe_metadata()
        )
        metadata_group.attrs["noise_power_w"] = float(metadata.get("noise_power_w", 0.0))
        metadata_group.attrs["bandwidth_hz"] = float(metadata.get("bandwidth_hz", 0.0))
        metadata_group.attrs["temperature_k"] = float(metadata.get("temperature_k", 0.0))
