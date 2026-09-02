"""Selective HDF5 reads for compact coverage-map files.

Coverage schema v2 stores one canonical dense tensor with axes
``(step, height, transmitter, y, x)``.  The writer chunks the first three axes
individually, so callers should select those axes before NumPy materializes the
data.  This module owns that storage-facing behavior while
``shared.coverage.schema`` remains responsible for the metric formulas.

The high-level :meth:`CoverageHDF5Reader.read_metric` method preserves the
existing logical result shape ``(height, y, x)``.  Omitting
``height_indices`` therefore remains the full-height API.  Callers that need
raw canonical inputs can use :meth:`CoverageHDF5Reader.read_path_gain` and
select both height and transmitter layers explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .schema import (
    COVERAGE_PER_TX_METRICS,
    coverage_metric_base,
    derive_coverage_metric_layer,
    normalise_coverage_tx_names,
    resolve_coverage_tx_index,
    validate_coverage_hdf5_contract,
)

CoverageIndexSelection = int | Sequence[int] | None
CoverageDatasetReadObserver = Callable[[str, tuple[Any, ...], tuple[int, ...]], None]

_FULL_AXIS = slice(None)


def _decode_string_array(values: Any) -> list[str]:
    """Decode an HDF5 string array into ordinary Python strings."""
    result: list[str] = []
    for item in np.asarray(values).reshape(-1).tolist():
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _normalise_indices(
    selection: CoverageIndexSelection,
    *,
    axis_name: str,
    axis_size: int,
) -> tuple[int, ...] | None:
    """Validate one logical axis selection while preserving request order."""
    if selection is None:
        return None
    if isinstance(selection, (int, np.integer)):
        raw_indices = (selection,)
    else:
        raw_indices = tuple(selection)
    if not raw_indices:
        raise ValueError(f"{axis_name} selection must not be empty")
    for index in raw_indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError(f"{axis_name} indices must be integers, got {index!r}")
    indices = tuple(int(index) for index in raw_indices)
    invalid = [index for index in indices if index < 0 or index >= axis_size]
    if invalid:
        raise IndexError(
            f"{axis_name} index {invalid[0]} is outside the valid range "
            f"0..{max(axis_size - 1, 0)}"
        )
    return indices


class CoverageHDF5Reader:
    """Read logical coverage layers through bounded HDF5 hyperslabs.

    The reader opens the file only for the duration of each public read.  The
    optional observer is a diagnostics/testing hook: it receives each physical
    dataset path, HDF5 selection, and materialized result shape.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        read_observer: CoverageDatasetReadObserver | None = None,
    ) -> None:
        self.path = Path(path)
        self._read_observer = read_observer

    @staticmethod
    def _validate_file(h5: h5py.File) -> None:
        validate_coverage_hdf5_contract(
            int(h5.attrs.get("coverage_schema_version", 0)),
            h5.attrs.get("coverage_storage_layout"),
        )

    def _read(
        self,
        dataset: h5py.Dataset,
        selection: tuple[Any, ...],
        *,
        dtype: Any = None,
    ) -> np.ndarray:
        """Materialize exactly one recorded HDF5 dataset selection."""
        result = np.asarray(dataset[selection], dtype=dtype)
        if self._read_observer is not None:
            self._read_observer(dataset.name, selection, result.shape)
        return result

    def _read_whole(self, dataset: h5py.Dataset, *, dtype: Any = None) -> np.ndarray:
        """Read a small metadata dataset through the same observable boundary."""
        selection = tuple(_FULL_AXIS for _ in dataset.shape)
        return self._read(dataset, selection, dtype=dtype)

    @staticmethod
    def _path_gain_dataset(h5: h5py.File) -> h5py.Dataset:
        dataset = h5["values/path_gain_linear"]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("/values/path_gain_linear must be an HDF5 dataset")
        if dataset.ndim != 5:
            raise ValueError(
                "/values/path_gain_linear must have shape "
                f"(step, height, transmitter, y, x), got {dataset.shape}"
            )
        return dataset

    @staticmethod
    def _validate_step_index(step_index: int, step_count: int) -> int:
        index = int(step_index)
        if index < 0 or index >= step_count:
            raise IndexError(
                f"coverage step index {index} is outside the valid range "
                f"0..{max(step_count - 1, 0)}"
            )
        return index

    def _read_path_gain_from_open_file(
        self,
        h5: h5py.File,
        *,
        step_index: int,
        height_indices: CoverageIndexSelection,
        transmitter_indices: CoverageIndexSelection,
    ) -> np.ndarray:
        """Read selected canonical axes from an already validated file."""
        dataset = self._path_gain_dataset(h5)
        step = self._validate_step_index(step_index, int(dataset.shape[0]))
        heights = _normalise_indices(
            height_indices,
            axis_name="coverage height",
            axis_size=int(dataset.shape[1]),
        )
        transmitters = _normalise_indices(
            transmitter_indices,
            axis_name="coverage transmitter",
            axis_size=int(dataset.shape[2]),
        )

        # One unbounded request intentionally remains the existing full read.
        if heights is None and transmitters is None:
            return self._read(
                dataset,
                (
                    slice(step, step + 1),
                    _FULL_AXIS,
                    _FULL_AXIS,
                    _FULL_AXIS,
                    _FULL_AXIS,
                ),
                dtype=np.float32,
            )

        if heights is None and transmitters is not None:
            result = np.empty(
                (
                    1,
                    int(dataset.shape[1]),
                    len(transmitters),
                    int(dataset.shape[3]),
                    int(dataset.shape[4]),
                ),
                dtype=np.float32,
            )
            for output_tx, transmitter in enumerate(transmitters):
                result[:, :, output_tx : output_tx + 1] = self._read(
                    dataset,
                    (
                        slice(step, step + 1),
                        _FULL_AXIS,
                        slice(transmitter, transmitter + 1),
                        _FULL_AXIS,
                        _FULL_AXIS,
                    ),
                    dtype=np.float32,
                )
            return result

        if heights is not None and transmitters is None:
            result = np.empty(
                (
                    1,
                    len(heights),
                    int(dataset.shape[2]),
                    int(dataset.shape[3]),
                    int(dataset.shape[4]),
                ),
                dtype=np.float32,
            )
            for output_height, height in enumerate(heights):
                result[:, output_height : output_height + 1] = self._read(
                    dataset,
                    (
                        slice(step, step + 1),
                        slice(height, height + 1),
                        _FULL_AXIS,
                        _FULL_AXIS,
                        _FULL_AXIS,
                    ),
                    dtype=np.float32,
                )
            return result

        assert heights is not None
        assert transmitters is not None
        result = np.empty(
            (
                1,
                len(heights),
                len(transmitters),
                int(dataset.shape[3]),
                int(dataset.shape[4]),
            ),
            dtype=np.float32,
        )

        # h5py supports only a restricted form of multi-axis fancy indexing.
        # Small basic hyperslabs are both predictable and aligned with the
        # writer's (1, 1, 1, tile_y, tile_x) chunk shape.
        for output_height, height in enumerate(heights):
            for output_tx, transmitter in enumerate(transmitters):
                slab = self._read(
                    dataset,
                    (
                        slice(step, step + 1),
                        slice(height, height + 1),
                        slice(transmitter, transmitter + 1),
                        _FULL_AXIS,
                        _FULL_AXIS,
                    ),
                    dtype=np.float32,
                )
                result[:, output_height : output_height + 1, output_tx : output_tx + 1] = slab
        return result

    def read_path_gain(
        self,
        *,
        step_index: int = 0,
        height_indices: CoverageIndexSelection = None,
        transmitter_indices: CoverageIndexSelection = None,
    ) -> np.ndarray:
        """Return canonical path gain as ``(1, height, transmitter, y, x)``.

        Requested height and transmitter order is preserved.  A single-height,
        single-transmitter request touches only that HDF5 hyperslab.
        """
        with h5py.File(self.path, "r") as h5:
            self._validate_file(h5)
            return self._read_path_gain_from_open_file(
                h5,
                step_index=step_index,
                height_indices=height_indices,
                transmitter_indices=transmitter_indices,
            )

    def _read_tx_metadata(
        self,
        h5: h5py.File,
        tx_count: int,
    ) -> tuple[list[str], np.ndarray]:
        tx_group = h5.get("tx")
        if not isinstance(tx_group, h5py.Group):
            return normalise_coverage_tx_names([], tx_count), np.zeros(tx_count, dtype=np.float32)
        names_dataset = tx_group.get("names")
        names = (
            _decode_string_array(self._read_whole(names_dataset))
            if isinstance(names_dataset, h5py.Dataset)
            else []
        )
        powers_dataset = tx_group.get("powers_dbm")
        powers = (
            self._read_whole(powers_dataset, dtype=np.float32).reshape(-1)
            if isinstance(powers_dataset, h5py.Dataset)
            else np.zeros(tx_count, dtype=np.float32)
        )
        return normalise_coverage_tx_names(names, tx_count), powers

    def _read_materialized_metric(
        self,
        dataset: h5py.Dataset,
        *,
        step_index: int,
        height_indices: CoverageIndexSelection,
    ) -> np.ndarray:
        """Read a materialized scalar metric as ``(height, y, x)``."""
        if dataset.ndim not in (3, 4):
            raise ValueError(
                f"{dataset.name} must have shape (height, y, x) or "
                f"(step, height, y, x), got {dataset.shape}"
            )
        height_axis = 1 if dataset.ndim == 4 else 0
        height_count = int(dataset.shape[height_axis])
        heights = _normalise_indices(
            height_indices,
            axis_name="coverage height",
            axis_size=height_count,
        )
        selected_heights = tuple(range(height_count)) if heights is None else heights

        if heights is None:
            if dataset.ndim == 4:
                step = self._validate_step_index(step_index, int(dataset.shape[0]))
                return self._read(
                    dataset,
                    (
                        slice(step, step + 1),
                        _FULL_AXIS,
                        _FULL_AXIS,
                        _FULL_AXIS,
                    ),
                )[0]
            if int(step_index) != 0:
                raise IndexError("coverage dataset without a step axis only supports step 0")
            return self._read(
                dataset,
                (_FULL_AXIS, _FULL_AXIS, _FULL_AXIS),
            )

        if dataset.ndim == 4:
            step = self._validate_step_index(step_index, int(dataset.shape[0]))
            output_shape = (
                len(selected_heights),
                int(dataset.shape[2]),
                int(dataset.shape[3]),
            )
        else:
            step = 0
            if int(step_index) != 0:
                raise IndexError("coverage dataset without a step axis only supports step 0")
            output_shape = (
                len(selected_heights),
                int(dataset.shape[1]),
                int(dataset.shape[2]),
            )

        result = np.empty(output_shape, dtype=dataset.dtype)
        for output_height, height in enumerate(selected_heights):
            if dataset.ndim == 4:
                selection = (
                    slice(step, step + 1),
                    slice(height, height + 1),
                    _FULL_AXIS,
                    _FULL_AXIS,
                )
                slab = self._read(dataset, selection)[0]
            else:
                selection = (
                    slice(height, height + 1),
                    _FULL_AXIS,
                    _FULL_AXIS,
                )
                slab = self._read(dataset, selection)
            result[output_height : output_height + 1] = slab
        return result

    def read_metric(
        self,
        metric_name: str,
        *,
        step_index: int = 0,
        height_indices: CoverageIndexSelection = None,
    ) -> np.ndarray:
        """Read or derive one logical metric as ``(height, y, x)``.

        Materialized scalar layers are read directly. Per-transmitter path-gain,
        path-loss, and RSS recipes read only the selected transmitter's chunks.
        Serving-path-gain and SINR recipes necessarily read every transmitter to
        select the serving TX or account for interference, but still honor the
        requested height selection.
        """
        metric_name = str(metric_name)
        base_name, selector = coverage_metric_base(metric_name)

        with h5py.File(self.path, "r") as h5:
            self._validate_file(h5)
            derived = h5.get("derived")
            materialized_dataset = (
                derived.get(base_name) if isinstance(derived, h5py.Group) else None
            )
            if selector is None and isinstance(materialized_dataset, h5py.Dataset):
                return self._read_materialized_metric(
                    materialized_dataset,
                    step_index=step_index,
                    height_indices=height_indices,
                )

            path_gain_dataset = self._path_gain_dataset(h5)
            tx_count = int(path_gain_dataset.shape[2])
            if tx_count <= 0:
                raise ValueError("coverage metric has no transmitters")
            tx_names, tx_power_dbm = self._read_tx_metadata(h5, tx_count)
            requested_tx: int | None = None
            derived_metric_name = metric_name
            if base_name in COVERAGE_PER_TX_METRICS:
                requested_tx = resolve_coverage_tx_index(selector, tx_names, tx_count)
                # The sliced tensor contains one transmitter.  Removing the
                # selector lets the shared formula resolve that sole layer.
                derived_metric_name = base_name

            path_gain = self._read_path_gain_from_open_file(
                h5,
                step_index=step_index,
                height_indices=height_indices,
                transmitter_indices=requested_tx,
            )
            if requested_tx is not None:
                formula_tx_names = [tx_names[requested_tx]]
                formula_tx_power = tx_power_dbm[requested_tx : requested_tx + 1]
            else:
                formula_tx_names = tx_names
                formula_tx_power = tx_power_dbm

            metadata = h5.get("metadata")
            noise_power_w = (
                float(metadata.attrs.get("noise_power_w", 0.0))
                if isinstance(metadata, h5py.Group)
                else 0.0
            )
            return derive_coverage_metric_layer(
                path_gain,
                derived_metric_name,
                tx_power_dbm=formula_tx_power,
                noise_power_w=noise_power_w,
                tx_names=formula_tx_names,
            )


__all__ = [
    "CoverageDatasetReadObserver",
    "CoverageHDF5Reader",
    "CoverageIndexSelection",
]
