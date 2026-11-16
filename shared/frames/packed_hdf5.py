"""Complete and selective reads for the packed-ragged HDF5 v2 layout.

``PackedHDF5Reader`` maps frame IDs through the authoritative manifest and
opens chunk files lazily through a bounded handle pool. Complete reads produce
``StandardMPCFrame``; selective reads produce ``FrameProjection`` with an
explicit inventory of loaded logical fields. Logical requests are mapped to
physical datasets here so unrequested path, metric, material, target, or
extension groups are not materialized.
"""

from __future__ import annotations

import hashlib
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from threading import Condition, RLock
from typing import Any, cast

import h5py
import numpy as np

from .contracts import (
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
    PATH_METRIC_VALIDITY_BITS,
    FrameComponent,
    FrameReadRequest,
    PathMetric,
)
from .json_codec import dumps_frame_json, loads_frame_json
from .manifest import FrameChunkManifest, FrameSetManifest, load_frame_manifest
from .packed import FrameProjection, ProjectedMPCFrame
from .types import StandardMPCFrame

_METRIC_DATASETS: dict[PathMetric, str] = {
    PathMetric.DELAY_NS: "delay_ns",
    PathMetric.PATH_LOSS_DB: "path_loss_db",
    PathMetric.AOA_AZ_DEG: "aoa_az_deg",
    PathMetric.AOA_EL_DEG: "aoa_el_deg",
    PathMetric.AOD_AZ_DEG: "aod_az_deg",
    PathMetric.AOD_EL_DEG: "aod_el_deg",
}

_METRIC_FRAME_FIELDS: dict[PathMetric, str] = {
    PathMetric.DELAY_NS: "delays_ns",
    PathMetric.PATH_LOSS_DB: "path_loss_db",
    PathMetric.AOA_AZ_DEG: "aoa_az_deg",
    PathMetric.AOA_EL_DEG: "aoa_el_deg",
    PathMetric.AOD_AZ_DEG: "aod_az_deg",
    PathMetric.AOD_EL_DEG: "aod_el_deg",
}

_FULL_FRAME_REQUEST = FrameReadRequest.full()
_MISSING_HDF5_OBJECT = object()
# Bound read-ahead independently of the writer's configurable chunk size.
_PROJECTION_BATCH_ROWS = 64
_PROJECTION_BATCH_CACHE_BYTES = 8 * 1024 * 1024


class PackedHDF5Error(ValueError):
    """Raised when a packed HDF5 chunk violates the v2 contract."""


class _CachedHDF5File:
    """One pooled file plus caches valid for that immutable open handle."""

    __slots__ = ("file", "identity_validated", "_objects", "_static_values")

    def __init__(self, file: h5py.File) -> None:
        self.file = file
        self.identity_validated = False
        self._objects: dict[str, h5py.Dataset | h5py.Group | object] = {}
        self._static_values: dict[str, Any] = {}

    @property
    def attrs(self) -> h5py.AttributeManager:
        return self.file.attrs

    @property
    def id(self) -> Any:
        return self.file.id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.file, name)

    def get(
        self,
        name: str,
        default: Any = None,
        *,
        getclass: bool = False,
        getlink: bool = False,
    ) -> Any:
        """Resolve and cache normal objects; delegate metadata queries."""
        if getclass or getlink:
            return self.file.get(
                name,
                default,
                getclass=getclass,
                getlink=getlink,
            )
        cached = self._objects.get(name, _MISSING_HDF5_OBJECT)
        if cached is not _MISSING_HDF5_OBJECT:
            return cached
        if name in self._objects:
            return default
        node = self.file.get(name)
        if isinstance(node, (h5py.Dataset, h5py.Group)):
            self._objects[name] = node
            return node
        self._objects[name] = _MISSING_HDF5_OBJECT
        return default

    def __getitem__(self, name: str) -> h5py.Dataset | h5py.Group:
        node = self.get(name)
        if not isinstance(node, (h5py.Dataset, h5py.Group)):
            raise KeyError(f"Unable to synchronously open object (object {name!r} doesn't exist)")
        return node

    def __contains__(self, name: str) -> bool:
        return name in self.file

    def static_value(self, key: str, load: Callable[[], Any]) -> Any:
        """Return one decoded immutable value for this open file handle."""
        if key not in self._static_values:
            self._static_values[key] = load()
        return self._static_values[key]

    def close(self) -> None:
        self._static_values.clear()
        self._objects.clear()
        self.file.close()


def _dataset(
    root: h5py.Group | h5py.File | _CachedHDF5File,
    path: str,
) -> h5py.Dataset:
    node = root.get(path)
    if not isinstance(node, h5py.Dataset):
        raise PackedHDF5Error(f"Missing HDF5 dataset /{path}")
    return node


class _BatchDataset:
    """Selection facade that reuses one bounded, contiguous dataset slice."""

    def __init__(self, batch: "_ProjectionReadBatch", path: str) -> None:
        self._batch = batch
        self._path = path

    def __getitem__(self, selection: int | slice) -> Any:
        return self._batch.read(self._path, selection)


class _ProjectionReadBatch:
    """Read requested columns together without changing per-frame validation.

    Only columns reached by the ordinary projection reader are loaded. Numeric
    slices exceeding the remaining budget fall back to individual selections.
    Variable-length JSON has no known decoded size before I/O: its bounded row
    slice is retained only if it fits, otherwise subsequent reads stay scalar.
    Large numeric sensing products always use their existing selective reader.
    """

    def __init__(self, h5: _CachedHDF5File, start: int, stop: int) -> None:
        self._h5 = h5
        self._start = start
        self._stop = stop
        self._columns: dict[str, tuple[int, np.ndarray]] = {}
        self._uncached: set[str] = set()
        self.cache_bytes = 0

    def dataset(self, path: str) -> _BatchDataset:
        return _BatchDataset(self, path)

    def _bounds(self, path: str) -> tuple[int, int]:
        if path == "index/frame_target_offsets":
            return self._start, self._stop + 1
        if path.startswith("frames/") or path == "index/frame_pair_path_offsets":
            return self._start, self._stop
        if path.startswith("targets/"):
            offsets = self.read("index/frame_target_offsets", slice(self._start, self._stop + 1))
            return int(offsets[0]), int(offsets[-1])
        if path.startswith(("paths/", "bounces/")):
            offsets = self.read("index/frame_pair_path_offsets", slice(self._start, self._stop))
            start, stop = int(offsets[0, 0]), int(offsets[-1, -1])
            if path == "paths/bounce_offsets":
                return start, stop + 1
            if path.startswith("bounces/"):
                bounce_offsets = self.read("paths/bounce_offsets", slice(start, stop + 1))
                return int(bounce_offsets[0]), int(bounce_offsets[-1])
            return start, stop
        raise ValueError(f"No projection batch bounds for {path}")

    def read(self, path: str, selection: int | slice) -> Any:
        column = self._columns.get(path)
        dataset = _dataset(self._h5, path)
        if column is None and path not in self._uncached:
            start, stop = self._bounds(path)
            estimate = (stop - start) * prod(dataset.shape[1:]) * dataset.dtype.itemsize
            remaining = _PROJECTION_BATCH_CACHE_BYTES - self.cache_bytes
            if estimate <= remaining:
                values = np.asarray(dataset[start:stop])
                size = values.nbytes
                if values.dtype.hasobject:
                    size += sum(sys.getsizeof(value) for value in values.flat)
                if size <= remaining:
                    values.setflags(write=False)
                    column = self._columns[path] = (start, values)
                    self.cache_bytes += size
                else:
                    self._uncached.add(path)
                    # Reuse this just-read selection before dropping an
                    # oversized JSON block; never retain it in the cache.
                    return self._select(values, start, selection)
            else:
                self._uncached.add(path)
        if column is None:
            return dataset[selection]
        return self._select(column[1], column[0], selection)

    @staticmethod
    def _select(values: np.ndarray, start: int, selection: int | slice) -> Any:
        if isinstance(selection, slice):
            result = values[slice(selection.start - start, selection.stop - start, selection.step)]
        else:
            result = values[selection - start]
        # Returned frame arrays must not expose the private read-ahead buffer.
        return result.copy() if isinstance(result, np.ndarray) else result


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _integer_attribute(
    h5: h5py.File | _CachedHDF5File,
    name: str,
    *,
    chunk_file: str,
) -> int:
    """Return one required scalar integer HDF5 attribute."""
    if name not in h5.attrs:
        raise PackedHDF5Error(f"{chunk_file} is missing HDF5 attribute {name!r}")
    value = h5.attrs[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise PackedHDF5Error(f"{chunk_file} HDF5 attribute {name!r} must be an integer")
    return int(value)


def _string_attribute(
    h5: h5py.File | _CachedHDF5File,
    name: str,
    *,
    chunk_file: str,
) -> str:
    """Return one required scalar string HDF5 attribute."""
    if name not in h5.attrs:
        raise PackedHDF5Error(f"{chunk_file} is missing HDF5 attribute {name!r}")
    value = h5.attrs[name]
    if not isinstance(value, (str, bytes, np.bytes_)):
        raise PackedHDF5Error(f"{chunk_file} HDF5 attribute {name!r} must be a string")
    return _decode_string(value)


def _decode_string_array(dataset: h5py.Dataset) -> tuple[str, ...]:
    return tuple(_decode_string(value) for value in dataset[:])


def _cached_static_array(
    h5: _CachedHDF5File,
    path: str,
    *,
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    """Decode a static numeric dataset once and keep its private cache read-only."""

    resolved_dtype = np.dtype(dtype)

    def load() -> np.ndarray:
        values = np.asarray(_dataset(h5, path)[:], dtype=resolved_dtype)
        values.setflags(write=False)
        return values

    return cast(
        np.ndarray,
        h5.static_value(f"array:{path}:{resolved_dtype.str}", load),
    )


def _cached_static_strings(
    h5: _CachedHDF5File,
    path: str,
) -> tuple[str, ...]:
    """Decode an immutable static string catalog once per pooled handle."""

    return cast(
        tuple[str, ...],
        h5.static_value(
            f"strings:{path}",
            lambda: _decode_string_array(_dataset(h5, path)),
        ),
    )


def _decode_json_cell(dataset: h5py.Dataset | _BatchDataset, row: Any, *, default: Any) -> Any:
    raw = dataset[row]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if raw in ("", None):
        return default
    if raw == "{}":
        return {}
    return loads_frame_json(cast(str, raw))


def _load_static_sensing_config(
    h5: h5py.File | _CachedHDF5File,
) -> Mapping[str, Any] | None:
    """Verify static sensing identity without reading numerical product arrays.

    The writer hashes exact config JSON, fixed-product shape/dtype metadata,
    and the CIR storage dtype/rank. A zero-width CIR shape table denotes an
    absent CIR layout, including its placeholder values dataset.
    """
    raw_config = _decode_string(_dataset(h5, "static/sensing_config_json")[()])
    config = loads_frame_json(raw_config)
    if config is not None:
        if not isinstance(config, Mapping):
            raise PackedHDF5Error("static/sensing_config_json must contain a JSON object or null")
        for key in {"config", "config_resolved"} & set(config):
            if not isinstance(config[key], Mapping):
                raise PackedHDF5Error(f"static/sensing_config_json.{key} must be a JSON object")
    chunk_file = Path(h5.filename).name
    fixed = _required_group(h5, "sensing/fixed", chunk_file=chunk_file)
    fixed_layout = {}
    for product in sorted(fixed.keys()):
        data = _dataset(h5, f"sensing/fixed/{product}/data")
        fixed_layout[product] = {"shape": list(data.shape[1:]), "dtype": data.dtype.str}
    shapes = _dataset(h5, "sensing/ragged/cir/shapes")
    if shapes.ndim != 2:
        raise PackedHDF5Error("sensing/ragged/cir/shapes must be a rank-two dataset")
    rank = shapes.shape[1] or None
    cir_values = _dataset(h5, "sensing/ragged/cir/values")
    if rank is None and cir_values.shape != (0,):
        raise PackedHDF5Error("An absent CIR layout must have an empty values dataset")
    payload = {
        "fixed": fixed_layout,
        "cir": {
            "storage_dtype": cir_values.dtype.str if rank is not None else None,
            "rank": rank,
        },
        "static_config": raw_config if config is not None else None,
    }
    actual = hashlib.sha256(dumps_frame_json(payload).encode("utf-8")).hexdigest()
    stored = _string_attribute(h5, "sensing_layout_id", chunk_file=chunk_file)
    if actual != stored:
        raise PackedHDF5Error(
            f"{chunk_file} sensing_layout_id does not match its static sensing payload"
        )
    return dict(config) if config is not None else None


def _merge_static_sensing_config(
    sensing: dict[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge verified static metadata without allowing per-frame overrides."""
    if config is None:
        return sensing
    if sensing is None:
        sensing = {}
    static_keys = {"config", "config_resolved", "target_catalog"} & set(config)
    bundle = (
        {key: config[key] for key in static_keys}
        if static_keys
        else {"config": config, "config_resolved": config}
    )
    for key, value in bundle.items():
        if not isinstance(value, Mapping):
            raise PackedHDF5Error(f"static/sensing_config_json.{key} must be a JSON object")
        if key in sensing and dumps_frame_json(sensing[key]) != dumps_frame_json(value):
            raise PackedHDF5Error(
                f"Per-frame sensing.{key} conflicts with verified static metadata"
            )
        sensing[key] = deepcopy(value)
    return sensing


def _decode_json_optional_mapping_cell(
    dataset: h5py.Dataset | _BatchDataset,
    row: Any,
    *,
    label: str,
) -> dict[str, Any] | None:
    """Decode an optional metadata mapping without collapsing null and empty."""

    value = _decode_json_cell(dataset, row, default=None)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PackedHDF5Error(f"{label} must contain a JSON object or null")
    return dict(value)


def _decode_target_metadata_cell(
    dataset: h5py.Dataset | _BatchDataset,
    target_row: int,
    *,
    chunk_file: str,
    frame_id: int,
) -> dict[str, Any]:
    """Decode one target record with enough location data to repair the file."""
    context = (
        f"{chunk_file} output frame {frame_id} " f"/targets/metadata_json target row {target_row}"
    )
    try:
        value = _decode_json_cell(dataset, target_row, default=None)
    except Exception as exc:
        raise PackedHDF5Error(f"{context} could not be decoded: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PackedHDF5Error(f"{context} must contain a JSON object")
    return dict(value)


@dataclass(slots=True)
class _HandleEntry:
    handle: _CachedHDF5File
    leases: int = 0
    io_lock: RLock = field(default_factory=RLock)


class HDF5HandleLRU:
    """Ref-counted, bounded LRU of read-only HDF5 file handles.

    Closing rejects new leases immediately. Already accepted reads retain their
    handles until their last lease exits, so cancellation never invalidates an
    HDF5 object in a worker or blocks the GUI waiting for disk I/O.
    """

    def __init__(self, max_open: int = 4) -> None:
        if max_open <= 0:
            raise ValueError("max_open must be positive")
        self._max_open = int(max_open)
        self._entries: OrderedDict[Path, _HandleEntry] = OrderedDict()
        self._condition = Condition(RLock())
        self._closed = False

    @property
    def open_count(self) -> int:
        with self._condition:
            return len(self._entries)

    @property
    def max_open(self) -> int:
        return self._max_open

    def _evict_one_unleased(self) -> bool:
        for path, entry in tuple(self._entries.items()):
            if entry.leases == 0:
                self._entries.pop(path)
                entry.handle.close()
                return True
        return False

    @contextmanager
    def lease(self, path: str | Path) -> Iterator[_CachedHDF5File]:
        resolved = Path(path)
        with self._condition:
            if self._closed:
                raise RuntimeError("HDF5 handle pool is closed")
            entry = self._entries.get(resolved)
            while entry is None and len(self._entries) >= self._max_open:
                if self._evict_one_unleased():
                    break
                self._condition.wait()
                if self._closed:
                    raise RuntimeError("HDF5 handle pool is closed")
                entry = self._entries.get(resolved)
            if entry is None:
                entry = _HandleEntry(_CachedHDF5File(h5py.File(resolved, "r")))
                self._entries[resolved] = entry
            entry.leases += 1
            self._entries.move_to_end(resolved)
        try:
            # h5py does not promise that callers may use one File object
            # concurrently.  Keep separate chunk files independent while
            # serializing reads that happen to share this cached handle.
            with entry.io_lock:
                yield entry.handle
        finally:
            with self._condition:
                entry.leases -= 1
                if self._closed and entry.leases == 0:
                    self._entries.pop(resolved)
                    entry.handle.close()
                else:
                    self._entries.move_to_end(resolved)
                self._condition.notify_all()

    def close(self) -> None:
        """Retire the pool, closing active handles when their reads finish."""
        with self._condition:
            self._closed = True
            for path, entry in tuple(self._entries.items()):
                if entry.leases == 0:
                    self._entries.pop(path)
                    entry.handle.close()
            self._condition.notify_all()


class PackedHDF5Reader:
    """Manifest-driven packed-frame reader with true component projections."""

    def __init__(
        self,
        frames_dir: str | Path,
        *,
        max_open_files: int = 4,
        verify_manifest_files: bool = True,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.manifest: FrameSetManifest = load_frame_manifest(
            self.frames_dir,
            verify_files=verify_manifest_files,
        )
        self._locations = self.manifest.frame_locations()
        self._handles = HDF5HandleLRU(max_open=max_open_files)

    @property
    def frame_ids(self) -> list[int]:
        return list(self.manifest.frame_ids)

    @property
    def bulk_files(self) -> list[Path]:
        return [self.frames_dir / chunk.file for chunk in self.manifest.chunks]

    @property
    def open_handle_count(self) -> int:
        return self._handles.open_count

    def has_frame(self, step: int) -> bool:
        return int(step) in self._locations

    def frame_source_path(self, step: int) -> Path | None:
        """Return the manifest-bound chunk for a frame without opening it."""
        location = self._locations.get(int(step))
        return None if location is None else self.frames_dir / location[0].file

    def close(self) -> None:
        self._handles.close()

    def load_projection(
        self,
        step: int,
        request: FrameReadRequest,
    ) -> FrameProjection:
        try:
            chunk, row = self._locations[int(step)]
        except KeyError as exc:
            raise KeyError(f"Frame {step} is not listed in the frame manifest") from exc
        path = self.frames_dir / chunk.file
        with self._handles.lease(path) as h5:
            self._ensure_file_identity(h5, chunk)
            return self._read_projection(h5, chunk, row, int(step), request)

    def iter_projections(
        self,
        steps: Sequence[int],
        request: FrameReadRequest,
    ) -> Iterator[FrameProjection]:
        """Yield projections in caller order with bounded contiguous read-ahead."""
        locations: list[tuple[int, FrameChunkManifest, int]] = []
        for raw_step in steps:
            step = int(raw_step)
            try:
                chunk, row = self._locations[step]
            except KeyError as exc:
                raise KeyError(f"Frame {step} is not listed in the frame manifest") from exc
            locations.append((step, chunk, row))

        cursor = 0
        while cursor < len(locations):
            chunk = locations[cursor][1]
            run_end = cursor + 1
            while (
                run_end < len(locations)
                and locations[run_end][1].file == chunk.file
                and (
                    self._can_coalesce_path_projection(request)
                    or run_end - cursor < _PROJECTION_BATCH_ROWS
                )
            ):
                run_end += 1

            run = locations[cursor:run_end]
            with self._handles.lease(self.frames_dir / chunk.file) as h5:
                self._ensure_file_identity(h5, chunk)
                consecutive_rows = all(right[2] == left[2] + 1 for left, right in zip(run, run[1:]))
                if (
                    len(run) > 1
                    and consecutive_rows
                    and self._can_coalesce_path_projection(request)
                ):
                    projections = tuple(
                        self._read_coalesced_path_projections(
                            h5,
                            chunk,
                            run,
                            request,
                        )
                    )
                else:
                    batch = (
                        _ProjectionReadBatch(h5, run[0][2], run[-1][2] + 1)
                        if len(run) > 1 and consecutive_rows
                        else None
                    )
                    projections = tuple(
                        self._read_projection(h5, chunk, row, step, request, batch=batch)
                        for step, _chunk, row in run
                    )
            # Do not hold the per-file h5py lock while callers process a
            # projection. A bounded chunk is fully materialized above.
            yield from projections
            cursor = run_end

    @staticmethod
    def _can_coalesce_path_projection(request: FrameReadRequest) -> bool:
        """Return whether a request uses only batch-friendly path/device columns."""
        unsupported = {
            FrameComponent.TARGETS,
            FrameComponent.SENSING,
            FrameComponent.BEAMFORMING,
            FrameComponent.PROVENANCE,
        }
        return request.components.isdisjoint(unsupported)

    def _read_coalesced_path_projections(
        self,
        h5: _CachedHDF5File,
        chunk: FrameChunkManifest,
        run: Sequence[tuple[int, FrameChunkManifest, int]],
        request: FrameReadRequest,
    ) -> Iterator[FrameProjection]:
        """Read one contiguous row run with one selection per requested dataset."""
        row_start = run[0][2]
        row_end = run[-1][2] + 1
        expected_steps = np.asarray(
            [step for step, _chunk, _row in run],
            dtype=np.int64,
        )
        stored_steps = _cached_static_array(
            h5,
            "frames/id",
            dtype=np.int64,
        )[row_start:row_end]
        if not np.array_equal(stored_steps, expected_steps):
            raise PackedHDF5Error(
                f"{chunk.file} rows {row_start}:{row_end} do not match requested frame IDs"
            )

        components = request.components
        pairs: np.ndarray | None = None
        tx_names: tuple[str, ...] | None = None
        rx_names: tuple[str, ...] | None = None
        device_columns: dict[str, np.ndarray] = {}
        if FrameComponent.PATH_TOPOLOGY in components:
            pairs = _cached_static_array(
                h5,
                "static/tx_rx_pairs",
                dtype=np.int32,
            ).copy()
        if FrameComponent.DEVICES in components:
            tx_names = _cached_static_strings(h5, "static/tx_names")
            rx_names = _cached_static_strings(h5, "static/rx_names")
            for field_name, dataset_name in (
                ("tx_positions", "frames/tx_position_m"),
                ("rx_positions", "frames/rx_position_m"),
                ("tx_orientations", "frames/tx_orientation_rad"),
                ("rx_orientations", "frames/rx_orientation_rad"),
            ):
                device_columns[field_name] = np.asarray(
                    _dataset(h5, dataset_name)[row_start:row_end],
                    dtype=np.float64,
                )

        pair_offsets: np.ndarray | None = None
        batch_bounce_offsets: np.ndarray | None = None
        path_base = path_end = bounce_base = bounce_end = 0
        if FrameComponent.PATH_TOPOLOGY in components:
            pair_offsets = np.asarray(
                _dataset(h5, "index/frame_pair_path_offsets")[row_start:row_end],
                dtype=np.int64,
            )
            path_base = int(pair_offsets[0, 0])
            path_end = int(pair_offsets[-1, -1])
        if FrameComponent.PATH_BOUNCE_TOPOLOGY in components:
            batch_bounce_offsets = np.asarray(
                _dataset(h5, "paths/bounce_offsets")[path_base : path_end + 1],
                dtype=np.int64,
            )
            if batch_bounce_offsets.size == 0:
                raise PackedHDF5Error(f"{chunk.file} has an empty bounce-offset slice")
            bounce_base = int(batch_bounce_offsets[0])
            bounce_end = int(batch_bounce_offsets[-1])

        bounce_columns: dict[str, np.ndarray] = {}
        if FrameComponent.PATH_GEOMETRY in components:
            bounce_columns["bounce_xyz_m"] = np.asarray(
                _dataset(h5, "bounces/xyz_m")[bounce_base:bounce_end],
                dtype=np.float32,
            )
        if FrameComponent.PATH_INTERACTIONS in components:
            bounce_columns["interactions"] = np.asarray(
                _dataset(h5, "bounces/interaction")[bounce_base:bounce_end],
                dtype=np.uint8,
            )

        material_names: tuple[str, ...] | None = None
        material_itu_types: tuple[str, ...] | None = None
        if FrameComponent.PATH_MATERIALS in components:
            bounce_columns["material_ids"] = np.asarray(
                _dataset(h5, "bounces/material_id")[bounce_base:bounce_end],
                dtype=np.uint16,
            )
            material_names = _cached_static_strings(h5, "static/materials/name")
            material_itu_types = _cached_static_strings(
                h5,
                "static/materials/itu_type",
            )

        metric_validity: np.ndarray | None = None
        metric_columns: dict[PathMetric, np.ndarray] = {}
        if FrameComponent.PATH_METRICS in components:
            metric_validity = np.asarray(
                _dataset(h5, "paths/metric_valid_bits")[path_base:path_end],
                dtype=np.uint8,
            )
            for metric in request.metrics:
                metric_columns[metric] = np.asarray(
                    _dataset(h5, f"paths/{_METRIC_DATASETS[metric]}")[path_base:path_end],
                    dtype=np.float32,
                )

        assert pairs is not None or FrameComponent.PATH_TOPOLOGY not in components
        for run_index, (step, _chunk, _row) in enumerate(run):
            kwargs: dict[str, Any] = {"frame_index": step}
            if FrameComponent.DEVICES in components:
                kwargs.update(
                    tx_names=tx_names,
                    rx_names=rx_names,
                    **{
                        field_name: values[run_index]
                        for field_name, values in device_columns.items()
                    },
                )

            local_path_start = local_path_end = 0
            local_bounce_start = local_bounce_end = 0
            if FrameComponent.PATH_TOPOLOGY in components:
                assert pair_offsets is not None
                absolute_pair_offsets = pair_offsets[run_index]
                frame_path_start = int(absolute_pair_offsets[0])
                frame_path_end = int(absolute_pair_offsets[-1])
                local_path_start = frame_path_start - path_base
                local_path_end = frame_path_end - path_base
                kwargs.update(
                    tx_rx_pairs=pairs,
                    pair_path_offsets=absolute_pair_offsets - frame_path_start,
                )
            if FrameComponent.PATH_BOUNCE_TOPOLOGY in components:
                assert batch_bounce_offsets is not None
                absolute_bounce_offsets = batch_bounce_offsets[
                    local_path_start : local_path_end + 1
                ]
                frame_bounce_start = int(absolute_bounce_offsets[0])
                frame_bounce_end = int(absolute_bounce_offsets[-1])
                local_bounce_start = frame_bounce_start - bounce_base
                local_bounce_end = frame_bounce_end - bounce_base
                kwargs.update(
                    bounce_offsets=absolute_bounce_offsets - frame_bounce_start,
                )

            for field_name, values in bounce_columns.items():
                kwargs[field_name] = values[local_bounce_start:local_bounce_end]
            if FrameComponent.PATH_MATERIALS in components:
                kwargs.update(
                    material_names=material_names,
                    material_itu_types=material_itu_types,
                )
            if FrameComponent.PATH_METRICS in components:
                assert metric_validity is not None
                kwargs["metric_valid_bits"] = metric_validity[local_path_start:local_path_end]
                for metric, values in metric_columns.items():
                    kwargs[_METRIC_FRAME_FIELDS[metric]] = values[local_path_start:local_path_end]

            yield FrameProjection(
                frame=ProjectedMPCFrame(**kwargs),
                loaded_components=request.components,
                loaded_path_metrics=request.metrics,
            )

    def load_standard_frame(self, step: int) -> StandardMPCFrame:
        """Load one complete canonical frame without an intermediate model.

        Stored provenance is returned without injecting provider or file-access
        identity; that operational identity belongs to ``ProviderInfo``.
        """

        try:
            chunk, row = self._locations[int(step)]
        except KeyError as exc:
            raise KeyError(f"Frame {step} is not listed in the frame manifest") from exc
        with self._handles.lease(self.frames_dir / chunk.file) as h5:
            self._ensure_file_identity(h5, chunk)
            fields, _products, _all_products = self._read_frame_fields(
                h5,
                chunk,
                row,
                int(step),
                _FULL_FRAME_REQUEST,
                validate_sensing=False,
            )

        return StandardMPCFrame(**fields)

    def _ensure_file_identity(
        self,
        h5: _CachedHDF5File,
        chunk: FrameChunkManifest,
    ) -> None:
        if h5.identity_validated:
            return
        self._check_file_identity(
            h5,
            chunk,
            generation_id=self.manifest.generation_id,
        )
        h5.identity_validated = True

    @staticmethod
    def _check_file_identity(
        h5: h5py.File | _CachedHDF5File,
        chunk: FrameChunkManifest,
        *,
        generation_id: str,
    ) -> None:
        if (
            _string_attribute(
                h5,
                "file_kind",
                chunk_file=chunk.file,
            )
            != "mpc_frames"
        ):
            raise PackedHDF5Error(f"{chunk.file} is not an MPC frame chunk")
        if (
            _integer_attribute(
                h5,
                "schema_version",
                chunk_file=chunk.file,
            )
            != MPC_HDF5_SCHEMA_VERSION
        ):
            raise PackedHDF5Error(f"{chunk.file} has an unsupported schema version")
        if (
            _string_attribute(
                h5,
                "storage_layout",
                chunk_file=chunk.file,
            )
            != MPC_HDF5_LAYOUT
        ):
            raise PackedHDF5Error(f"{chunk.file} has an unsupported storage layout")
        expected_integer_attributes = {
            "packed_frame_version": PACKED_MPC_FRAME_VERSION,
            "num_frames": chunk.count,
            "start_frame": chunk.start_frame,
            "end_frame": chunk.end_frame,
        }
        for name, expected in expected_integer_attributes.items():
            actual_integer = _integer_attribute(
                h5,
                name,
                chunk_file=chunk.file,
            )
            if actual_integer != expected:
                raise PackedHDF5Error(
                    f"{chunk.file} HDF5 attribute {name!r} does not match "
                    f"the manifest ({actual_integer} != {expected})"
                )

        expected_string_attributes = {
            "publication_state": "complete",
            "generation_id": generation_id,
            "topology_id": chunk.topology_id,
            "sensing_layout_id": chunk.sensing_layout_id,
        }
        for name, expected in expected_string_attributes.items():
            actual_string = _string_attribute(
                h5,
                name,
                chunk_file=chunk.file,
            )
            if actual_string != expected:
                raise PackedHDF5Error(
                    f"{chunk.file} HDF5 attribute {name!r} does not match "
                    f"the manifest ({actual_string!r} != {expected!r})"
                )

        frame_ids_dataset = _dataset(h5, "frames/id")
        expected_shape = (chunk.count,)
        if frame_ids_dataset.shape != expected_shape or frame_ids_dataset.dtype != np.dtype(
            np.int64
        ):
            raise PackedHDF5Error(
                f"{chunk.file} /frames/id must have shape {expected_shape} " "and dtype int64"
            )
        if isinstance(h5, _CachedHDF5File):
            stored_frame_ids = _cached_static_array(
                h5,
                "frames/id",
                dtype=np.int64,
            )
        else:
            stored_frame_ids = np.asarray(
                _dataset(h5, "frames/id")[:],
                dtype=np.int64,
            )
        expected_frame_ids: np.ndarray = np.asarray(chunk.frame_ids, dtype=np.int64)
        if not np.array_equal(stored_frame_ids, expected_frame_ids):
            raise PackedHDF5Error(f"{chunk.file} frame IDs do not match the published manifest")

    def validate_chunk_identities(self) -> None:
        """Validate finalized chunk metadata without reading frame payloads."""
        for chunk in self.manifest.chunks:
            path = self.frames_dir / chunk.file
            with self._handles.lease(path) as h5:
                self._check_file_identity(
                    h5,
                    chunk,
                    generation_id=self.manifest.generation_id,
                )
                h5.identity_validated = True

    def validate_chunk_structures(self) -> None:
        """Validate visualizable v2 structure without reading payload arrays.

        Chunk identity is checked first unless it was already cached by
        :meth:`validate_chunk_identities`. The structural pass reads compact
        topology and offset metadata, but only inspects shape and dtype metadata
        for geometry, path-metric, device-position, and target-position payloads.
        """
        for chunk in self.manifest.chunks:
            path = self.frames_dir / chunk.file
            with self._handles.lease(path) as h5:
                self._ensure_file_identity(h5, chunk)
                validate_packed_chunk_structure(h5, chunk)

    def _read_frame_fields(
        self,
        h5: _CachedHDF5File,
        chunk: FrameChunkManifest,
        row: int,
        step: int,
        request: FrameReadRequest,
        *,
        validate_sensing: bool = True,
        batch: _ProjectionReadBatch | None = None,
    ) -> tuple[dict[str, Any], frozenset[str], bool]:
        """Read exactly the compact fields selected by ``request``.

        A complete generic frame load transports the optional sensing mapping
        without interpreting its independently versioned schema. Explicit
        sensing projections validate that schema before returning it.
        """

        def dataset(path: str) -> h5py.Dataset | _BatchDataset:
            return batch.dataset(path) if batch is not None else _dataset(h5, path)

        components = request.components
        kwargs: dict[str, Any] = {"frame_index": step}
        path_start = path_end = bounce_start = bounce_end = 0
        pairs: np.ndarray | None = None
        if FrameComponent.PATH_TOPOLOGY in components:
            # Keep the cached array private so a caller that violates the
            # packed-frame immutability convention cannot poison later reads.
            pairs = _cached_static_array(
                h5,
                "static/tx_rx_pairs",
                dtype=np.int32,
            ).copy()

        if FrameComponent.DEVICES in components:
            kwargs.update(
                tx_positions=np.asarray(dataset("frames/tx_position_m")[row], dtype=np.float64),
                rx_positions=np.asarray(dataset("frames/rx_position_m")[row], dtype=np.float64),
                tx_orientations=np.asarray(
                    dataset("frames/tx_orientation_rad")[row], dtype=np.float64
                ),
                rx_orientations=np.asarray(
                    dataset("frames/rx_orientation_rad")[row], dtype=np.float64
                ),
                tx_names=_cached_static_strings(h5, "static/tx_names"),
                rx_names=_cached_static_strings(h5, "static/rx_names"),
            )

        if FrameComponent.PATH_TOPOLOGY in components:
            absolute_pair_offsets = np.asarray(
                dataset("index/frame_pair_path_offsets")[row],
                dtype=np.int64,
            )
            path_start = int(absolute_pair_offsets[0])
            path_end = int(absolute_pair_offsets[-1])
            local_pair_offsets = absolute_pair_offsets - path_start
            kwargs.update(
                tx_rx_pairs=pairs,
                pair_path_offsets=local_pair_offsets,
            )
        if FrameComponent.PATH_BOUNCE_TOPOLOGY in components:
            absolute_bounce_offsets = np.asarray(
                dataset("paths/bounce_offsets")[path_start : path_end + 1],
                dtype=np.int64,
            )
            if absolute_bounce_offsets.size == 0:
                raise PackedHDF5Error(f"{chunk.file} has an empty bounce-offset slice")
            bounce_start = int(absolute_bounce_offsets[0])
            bounce_end = int(absolute_bounce_offsets[-1])
            kwargs.update(
                bounce_offsets=absolute_bounce_offsets - bounce_start,
            )

        if FrameComponent.PATH_GEOMETRY in components:
            kwargs["bounce_xyz_m"] = np.asarray(
                dataset("bounces/xyz_m")[bounce_start:bounce_end],
                dtype=np.float32,
            )

        if FrameComponent.PATH_INTERACTIONS in components:
            kwargs["interactions"] = np.asarray(
                dataset("bounces/interaction")[bounce_start:bounce_end],
                dtype=np.uint8,
            )

        if FrameComponent.PATH_MATERIALS in components:
            kwargs.update(
                material_ids=np.asarray(
                    dataset("bounces/material_id")[bounce_start:bounce_end],
                    dtype=np.uint16,
                ),
                material_names=_cached_static_strings(h5, "static/materials/name"),
                material_itu_types=_cached_static_strings(
                    h5,
                    "static/materials/itu_type",
                ),
            )

        if FrameComponent.PATH_METRICS in components:
            kwargs["metric_valid_bits"] = np.asarray(
                dataset("paths/metric_valid_bits")[path_start:path_end],
                dtype=np.uint8,
            )
            for metric in request.metrics:
                dataset_name = _METRIC_DATASETS[metric]
                field_name = _METRIC_FRAME_FIELDS[metric]
                kwargs[field_name] = np.asarray(
                    dataset(f"paths/{dataset_name}")[path_start:path_end],
                    dtype=np.float32,
                )

        if FrameComponent.TARGETS in components:
            target_interval = np.asarray(
                dataset("index/frame_target_offsets")[row : row + 2],
                dtype=np.int64,
            )
            target_start = int(target_interval[0])
            target_end = int(target_interval[1])
            positions = np.asarray(
                dataset("targets/position_m")[target_start:target_end],
                dtype=np.float64,
            )
            metadata_ds = dataset("targets/metadata_json")
            metadata = tuple(
                _decode_target_metadata_cell(
                    metadata_ds,
                    index,
                    chunk_file=chunk.file,
                    frame_id=step,
                )
                for index in range(target_start, target_end)
            )
            kwargs.update(
                target_positions_m=positions,
                targets_metadata=metadata,
            )

        if FrameComponent.PROVENANCE in components:
            source = _decode_json_optional_mapping_cell(
                dataset("frames/source_json"),
                row,
                label="frames/source_json",
            )
            timestamp = float(dataset("frames/timestamp_s")[row])
            kwargs.update(
                provenance=source,
                timestamp_s=timestamp if np.isfinite(timestamp) else None,
                recomputed_from_stored_positions=bool(dataset("frames/recomputed")[row]),
            )

        if FrameComponent.BEAMFORMING in components:
            kwargs["beamforming"] = _decode_json_optional_mapping_cell(
                dataset("frames/beamforming_json"),
                row,
                label="frames/beamforming_json",
            )

        loaded_sensing_products: frozenset[str] = frozenset()
        all_sensing_products_loaded = False
        if FrameComponent.SENSING in components:
            sensing = _decode_json_optional_mapping_cell(
                dataset("frames/sensing_metadata_json"),
                row,
                label="frames/sensing_metadata_json",
            )
            cached_config = self._static_sensing_config(h5)
            sensing = _merge_static_sensing_config(sensing, cached_config)
            available_products = self._available_sensing_products(h5)
            if request.all_sensing_products:
                selected_products = available_products
                all_sensing_products_loaded = True
            else:
                selected_products = set(request.sensing_products)
                # A typed numeric product is unusable without its exact axis
                # coordinates.  Resolve that dependency from persisted v4
                # descriptors instead of making each consumer guess companion
                # dataset names.
                descriptors = sensing.get("products") if isinstance(sensing, Mapping) else None
                if isinstance(descriptors, Mapping):
                    for product in tuple(selected_products):
                        descriptor = descriptors.get(product)
                        axes = descriptor.get("axes") if isinstance(descriptor, Mapping) else None
                        if not isinstance(axes, list):
                            continue
                        for axis in axes:
                            if not isinstance(axis, Mapping):
                                continue
                            coordinate_key = axis.get("coordinate_key")
                            if isinstance(coordinate_key, str):
                                selected_products.add(coordinate_key)
                            alternate_keys = axis.get("alternate_coordinate_keys")
                            if isinstance(alternate_keys, list):
                                selected_products.update(
                                    key for key in alternate_keys if isinstance(key, str)
                                )
            for product in selected_products:
                value = self._read_sensing_product(h5, product, row)
                if value is not None:
                    if sensing is None:
                        sensing = {}
                    sensing[product] = value
            if sensing is not None and validate_sensing:
                from shared.extensions.sensing import assert_sensing_contract

                assert_sensing_contract(
                    {"sensing": sensing},
                    require_schema_version=True,
                )
            elif sensing is not None:
                from shared.extensions.sensing import (
                    SENSING_SCHEMA_VERSION,
                    assert_processing_recipe_identity,
                )

                if sensing.get("schema_version") == SENSING_SCHEMA_VERSION:
                    assert_processing_recipe_identity(sensing)
            kwargs["sensing"] = sensing
            loaded_sensing_products = frozenset(selected_products)

        return kwargs, loaded_sensing_products, all_sensing_products_loaded

    def _read_projection(
        self,
        h5: _CachedHDF5File,
        chunk: FrameChunkManifest,
        row: int,
        step: int,
        request: FrameReadRequest,
        *,
        batch: _ProjectionReadBatch | None = None,
    ) -> FrameProjection:
        fields, loaded_sensing_products, all_sensing_products_loaded = self._read_frame_fields(
            h5, chunk, row, step, request, batch=batch
        )
        return FrameProjection(
            frame=ProjectedMPCFrame(**fields),
            loaded_components=request.components,
            loaded_path_metrics=request.metrics,
            loaded_sensing_products=loaded_sensing_products,
            all_sensing_products_loaded=all_sensing_products_loaded,
        )

    @staticmethod
    def _static_sensing_config(
        h5: _CachedHDF5File,
    ) -> Mapping[str, Any] | None:
        return cast(
            Mapping[str, Any] | None,
            h5.static_value(
                "json:static/sensing_config_json", lambda: _load_static_sensing_config(h5)
            ),
        )

    @staticmethod
    def _available_sensing_products(h5: _CachedHDF5File) -> frozenset[str]:
        def load() -> frozenset[str]:
            products: set[str] = set()
            fixed = h5.get("sensing/fixed")
            if isinstance(fixed, h5py.Group):
                products.update(str(key) for key in fixed.keys())
            ragged = h5.get("sensing/ragged")
            if isinstance(ragged, h5py.Group):
                products.update(str(key) for key in ragged.keys())
            return frozenset(products)

        return cast(
            frozenset[str],
            h5.static_value("sensing:available_products", load),
        )

    @staticmethod
    def _read_sensing_product(
        h5: _CachedHDF5File,
        product: str,
        row: int,
    ) -> np.ndarray | None:
        fixed_path = f"sensing/fixed/{product}"
        fixed = h5.get(fixed_path)
        if isinstance(fixed, h5py.Group):
            present = _dataset(h5, f"{fixed_path}/present")
            if not bool(present[row]):
                return None
            return np.asarray(_dataset(h5, f"{fixed_path}/data")[row])

        ragged_path = f"sensing/ragged/{product}"
        ragged = h5.get(ragged_path)
        if isinstance(ragged, h5py.Group):
            present = _dataset(h5, f"{ragged_path}/present")
            if not bool(present[row]):
                return None
            offsets = _dataset(h5, f"{ragged_path}/offsets")
            start = int(offsets[row])
            end = int(offsets[row + 1])
            shape_row = np.asarray(
                _dataset(h5, f"{ragged_path}/shapes")[row],
                dtype=np.int64,
            )
            shape = tuple(int(size) for size in shape_row if int(size) >= 0)
            values = np.asarray(_dataset(h5, f"{ragged_path}/values")[start:end])
            return values.reshape(shape).copy()
        return None

    def validate_all_chunks(self) -> None:
        """Open and deeply validate every chunk advertised by the manifest."""
        for chunk in self.manifest.chunks:
            path = self.frames_dir / chunk.file
            with self._handles.lease(path) as h5:
                self._check_file_identity(
                    h5,
                    chunk,
                    generation_id=self.manifest.generation_id,
                )
                h5.identity_validated = True
                validate_packed_chunk(h5, chunk)


def _required_group(
    h5: h5py.File | _CachedHDF5File,
    path: str,
    *,
    chunk_file: str,
) -> h5py.Group:
    node = h5.get(path)
    if not isinstance(node, h5py.Group):
        raise PackedHDF5Error(f"{chunk_file} is missing HDF5 group /{path}")
    return node


def _require_exact_dataset(
    h5: h5py.File | _CachedHDF5File,
    path: str,
    *,
    chunk_file: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> h5py.Dataset:
    dataset = _dataset(h5, path)
    expected_dtype = np.dtype(dtype)
    if dataset.shape != shape or dataset.dtype != expected_dtype:
        raise PackedHDF5Error(
            f"{chunk_file} /{path} must have shape {shape} and dtype {expected_dtype}"
        )
    return dataset


def _require_string_dataset(
    h5: h5py.File | _CachedHDF5File,
    path: str,
    *,
    chunk_file: str,
    shape: tuple[int, ...] | None = None,
    rank: int | None = None,
) -> h5py.Dataset:
    dataset = _dataset(h5, path)
    string_info = h5py.check_string_dtype(dataset.dtype)
    if string_info is None or string_info.encoding.lower() != "utf-8":
        raise PackedHDF5Error(f"{chunk_file} /{path} must use a UTF-8 string dtype")
    if shape is not None and dataset.shape != shape:
        raise PackedHDF5Error(f"{chunk_file} /{path} must have shape {shape}")
    if rank is not None and dataset.ndim != rank:
        raise PackedHDF5Error(f"{chunk_file} /{path} must have rank {rank}")
    return dataset


def _require_numeric_dataset(
    h5: h5py.File | _CachedHDF5File,
    path: str,
    *,
    chunk_file: str,
) -> h5py.Dataset:
    dataset = _dataset(h5, path)
    if not (np.issubdtype(dataset.dtype, np.number) or np.issubdtype(dataset.dtype, np.bool_)):
        raise PackedHDF5Error(f"{chunk_file} /{path} must use a numeric dtype")
    return dataset


def _read_int64_offsets(
    h5: h5py.File | _CachedHDF5File,
    path: str,
    *,
    chunk_file: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    dataset = _require_exact_dataset(
        h5,
        path,
        chunk_file=chunk_file,
        shape=shape,
        dtype=np.int64,
    )
    return cast(np.ndarray, np.asarray(dataset[:], dtype=np.int64))


def validate_packed_chunk_structure(
    h5: h5py.File | _CachedHDF5File,
    chunk: FrameChunkManifest,
) -> None:
    """Validate packed-v2 dataset structure without reading large payloads.

    Geometry, per-path metric, device-position, and target-position datasets
    are checked through HDF5 metadata only. Compact topology, offset, presence,
    and ragged-shape arrays are read because they define the boundaries needed
    to interpret those payloads safely.
    """

    chunk_file = chunk.file
    frame_count = chunk.count
    for group_path in (
        "static",
        "static/materials",
        "frames",
        "index",
        "paths",
        "bounces",
        "targets",
        "sensing",
        "sensing/fixed",
        "sensing/ragged",
        "sensing/ragged/cir",
    ):
        _required_group(h5, group_path, chunk_file=chunk_file)

    _require_exact_dataset(
        h5,
        "frames/id",
        chunk_file=chunk_file,
        shape=(frame_count,),
        dtype=np.int64,
    )

    tx_names = _require_string_dataset(
        h5,
        "static/tx_names",
        chunk_file=chunk_file,
        rank=1,
    )
    rx_names = _require_string_dataset(
        h5,
        "static/rx_names",
        chunk_file=chunk_file,
        rank=1,
    )
    tx_count = int(tx_names.shape[0])
    rx_count = int(rx_names.shape[0])

    pairs_dataset = _dataset(h5, "static/tx_rx_pairs")
    if (
        pairs_dataset.ndim != 2
        or pairs_dataset.shape[1:] != (2,)
        or pairs_dataset.dtype != np.dtype(np.int32)
    ):
        raise PackedHDF5Error(
            f"{chunk_file} /static/tx_rx_pairs must have shape (pairs, 2) and dtype int32"
        )
    pairs = np.asarray(pairs_dataset[:], dtype=np.int64)
    if np.any(pairs < 0) or (
        pairs.size
        and (int(np.max(pairs[:, 0])) >= tx_count or int(np.max(pairs[:, 1])) >= rx_count)
    ):
        raise PackedHDF5Error(f"{chunk_file} TX/RX pairs reference an unknown device")
    pair_count = int(pairs.shape[0])

    pair_offsets = _read_int64_offsets(
        h5,
        "index/frame_pair_path_offsets",
        chunk_file=chunk_file,
        shape=(frame_count, pair_count + 1),
    )
    if np.any(pair_offsets[:, 1:] < pair_offsets[:, :-1]):
        raise PackedHDF5Error(f"{chunk_file} has decreasing pair path offsets")
    if pair_offsets[0, 0] != 0:
        raise PackedHDF5Error(f"{chunk_file} path offsets must start at zero")
    if frame_count > 1 and np.any(pair_offsets[1:, 0] != pair_offsets[:-1, -1]):
        raise PackedHDF5Error(f"{chunk_file} frame path intervals are not contiguous")
    path_count = int(pair_offsets[-1, -1])

    bounce_offsets = _read_int64_offsets(
        h5,
        "paths/bounce_offsets",
        chunk_file=chunk_file,
        shape=(path_count + 1,),
    )
    if bounce_offsets[0] != 0 or np.any(bounce_offsets[1:] < bounce_offsets[:-1]):
        raise PackedHDF5Error(f"{chunk_file} has invalid bounce offsets")
    bounce_count = int(bounce_offsets[-1])

    _require_exact_dataset(
        h5,
        "paths/metric_valid_bits",
        chunk_file=chunk_file,
        shape=(path_count,),
        dtype=np.uint8,
    )
    for dataset_name in _METRIC_DATASETS.values():
        _require_exact_dataset(
            h5,
            f"paths/{dataset_name}",
            chunk_file=chunk_file,
            shape=(path_count,),
            dtype=np.float32,
        )
    for path, shape, dtype in (
        ("bounces/xyz_m", (bounce_count, 3), np.float32),
        ("bounces/interaction", (bounce_count,), np.uint8),
        ("bounces/material_id", (bounce_count,), np.uint16),
    ):
        _require_exact_dataset(
            h5,
            path,
            chunk_file=chunk_file,
            shape=shape,
            dtype=dtype,
        )

    material_names = _require_string_dataset(
        h5,
        "static/materials/name",
        chunk_file=chunk_file,
        rank=1,
    )
    material_itu_types = _require_string_dataset(
        h5,
        "static/materials/itu_type",
        chunk_file=chunk_file,
        rank=1,
    )
    if material_names.shape != material_itu_types.shape or material_names.shape[0] < 1:
        raise PackedHDF5Error(f"{chunk_file} material catalog columns are misaligned")
    if _decode_string(material_names[0]) != "" or _decode_string(material_itu_types[0]) != "":
        raise PackedHDF5Error(f"{chunk_file} material catalog row zero must be empty")

    _require_exact_dataset(
        h5,
        "frames/timestamp_s",
        chunk_file=chunk_file,
        shape=(frame_count,),
        dtype=np.float64,
    )
    _require_exact_dataset(
        h5,
        "frames/recomputed",
        chunk_file=chunk_file,
        shape=(frame_count,),
        dtype=np.uint8,
    )
    for path in (
        "frames/source_json",
        "frames/beamforming_json",
        "frames/sensing_metadata_json",
    ):
        _require_string_dataset(
            h5,
            path,
            chunk_file=chunk_file,
            shape=(frame_count,),
        )
    for path, count in (
        ("frames/tx_position_m", tx_count),
        ("frames/tx_orientation_rad", tx_count),
        ("frames/rx_position_m", rx_count),
        ("frames/rx_orientation_rad", rx_count),
    ):
        _require_exact_dataset(
            h5,
            path,
            chunk_file=chunk_file,
            shape=(frame_count, count, 3),
            dtype=np.float64,
        )

    target_offsets = _read_int64_offsets(
        h5,
        "index/frame_target_offsets",
        chunk_file=chunk_file,
        shape=(frame_count + 1,),
    )
    if target_offsets[0] != 0 or np.any(target_offsets[1:] < target_offsets[:-1]):
        raise PackedHDF5Error(f"{chunk_file} target offsets are invalid")
    target_count = int(target_offsets[-1])
    _require_exact_dataset(
        h5,
        "targets/position_m",
        chunk_file=chunk_file,
        shape=(target_count, 3),
        dtype=np.float64,
    )
    _require_string_dataset(
        h5,
        "targets/metadata_json",
        chunk_file=chunk_file,
        shape=(target_count,),
    )
    _require_string_dataset(
        h5,
        "static/sensing_config_json",
        chunk_file=chunk_file,
        shape=(),
    )

    fixed_sensing = _required_group(h5, "sensing/fixed", chunk_file=chunk_file)
    for product, node in fixed_sensing.items():
        if not isinstance(node, h5py.Group):
            raise PackedHDF5Error(f"{chunk_file} fixed sensing product {product!r} is not a group")
        data_path = f"sensing/fixed/{product}/data"
        data = _require_numeric_dataset(h5, data_path, chunk_file=chunk_file)
        if data.ndim < 1 or data.shape[0] != frame_count:
            raise PackedHDF5Error(
                f"{chunk_file} fixed sensing product {product!r} is not frame-aligned"
            )
        present_path = f"sensing/fixed/{product}/present"
        present = _require_exact_dataset(
            h5,
            present_path,
            chunk_file=chunk_file,
            shape=(frame_count,),
            dtype=np.uint8,
        )
        if np.any(np.asarray(present[:], dtype=np.uint8) > 1):
            raise PackedHDF5Error(
                f"{chunk_file} fixed sensing product {product!r} has invalid presence bits"
            )

    ragged_sensing = _required_group(h5, "sensing/ragged", chunk_file=chunk_file)
    for product, node in ragged_sensing.items():
        if not isinstance(node, h5py.Group):
            raise PackedHDF5Error(f"{chunk_file} ragged sensing product {product!r} is not a group")
        base_path = f"sensing/ragged/{product}"
        offsets = _read_int64_offsets(
            h5,
            f"{base_path}/offsets",
            chunk_file=chunk_file,
            shape=(frame_count + 1,),
        )
        shapes = _dataset(h5, f"{base_path}/shapes")
        if shapes.ndim != 2 or shapes.shape[0] != frame_count or shapes.dtype != np.dtype(np.int64):
            raise PackedHDF5Error(
                f"{chunk_file} ragged sensing product {product!r} "
                "shapes are not frame-aligned int64 metadata"
            )
        present = _require_exact_dataset(
            h5,
            f"{base_path}/present",
            chunk_file=chunk_file,
            shape=(frame_count,),
            dtype=np.uint8,
        )
        values = _require_numeric_dataset(
            h5,
            f"{base_path}/values",
            chunk_file=chunk_file,
        )
        if values.ndim != 1:
            raise PackedHDF5Error(
                f"{chunk_file} ragged sensing product {product!r} values must be a vector"
            )

        presence = np.asarray(present[:], dtype=np.uint8)
        shape_values = np.asarray(shapes[:], dtype=np.int64)
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise PackedHDF5Error(
                f"{chunk_file} ragged sensing product {product!r} has invalid offsets"
            )
        if int(offsets[-1]) != values.shape[0]:
            raise PackedHDF5Error(
                f"{chunk_file} ragged sensing product {product!r} offsets disagree with values"
            )
        if np.any(presence > 1):
            raise PackedHDF5Error(
                f"{chunk_file} ragged sensing product {product!r} has invalid presence bits"
            )
        for row in range(frame_count):
            item_count = int(offsets[row + 1] - offsets[row])
            if presence[row]:
                dimensions = tuple(int(value) for value in shape_values[row])
                if not dimensions or any(value < 0 for value in dimensions):
                    raise PackedHDF5Error(
                        f"{chunk_file} ragged sensing product {product!r} row {row} "
                        "has invalid shape metadata"
                    )
                expected_count = 1
                for dimension in dimensions:
                    expected_count *= dimension
                if expected_count != item_count:
                    raise PackedHDF5Error(
                        f"{chunk_file} ragged sensing product {product!r} row {row} "
                        "shape disagrees with its value interval"
                    )
            elif item_count:
                raise PackedHDF5Error(
                    f"{chunk_file} absent ragged sensing product {product!r} row {row} "
                    "owns stored values"
                )


def validate_packed_chunk(
    h5: h5py.File | _CachedHDF5File,
    chunk: FrameChunkManifest,
) -> None:
    """Deeply validate the offset and dataset alignment invariants of one chunk."""
    frame_ids = np.asarray(_dataset(h5, "frames/id")[:], dtype=np.int64)
    expected_ids = np.asarray(chunk.frame_ids, dtype=np.int64)
    if not np.array_equal(frame_ids, expected_ids):
        raise PackedHDF5Error(f"{chunk.file} frame IDs do not match its manifest entry")
    frame_count = len(frame_ids)

    pairs = np.asarray(_dataset(h5, "static/tx_rx_pairs")[:], dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs < 0):
        raise PackedHDF5Error(f"{chunk.file} has invalid TX/RX pairs")
    tx_count = len(_dataset(h5, "static/tx_names"))
    rx_count = len(_dataset(h5, "static/rx_names"))
    if pairs.size and (
        int(np.max(pairs[:, 0])) >= tx_count or int(np.max(pairs[:, 1])) >= rx_count
    ):
        raise PackedHDF5Error(f"{chunk.file} TX/RX pairs reference an unknown device")
    pair_offsets = np.asarray(_dataset(h5, "index/frame_pair_path_offsets")[:], dtype=np.int64)
    if pair_offsets.shape != (frame_count, pairs.shape[0] + 1):
        raise PackedHDF5Error(f"{chunk.file} has invalid frame/pair path offsets")
    if np.any(pair_offsets[:, 1:] < pair_offsets[:, :-1]):
        raise PackedHDF5Error(f"{chunk.file} has decreasing pair path offsets")
    if frame_count:
        if pair_offsets[0, 0] != 0:
            raise PackedHDF5Error(f"{chunk.file} path offsets must start at zero")
        if np.any(pair_offsets[1:, 0] != pair_offsets[:-1, -1]):
            raise PackedHDF5Error(f"{chunk.file} frame path intervals are not contiguous")
        path_count = int(pair_offsets[-1, -1])
    else:
        path_count = 0

    bounce_offsets = np.asarray(_dataset(h5, "paths/bounce_offsets")[:], dtype=np.int64)
    if bounce_offsets.shape != (path_count + 1,):
        raise PackedHDF5Error(f"{chunk.file} bounce-offset length does not match paths")
    if bounce_offsets[0] != 0 or np.any(bounce_offsets[1:] < bounce_offsets[:-1]):
        raise PackedHDF5Error(f"{chunk.file} has invalid bounce offsets")
    bounce_count = int(bounce_offsets[-1])

    validity_dataset = _dataset(h5, "paths/metric_valid_bits")
    if validity_dataset.shape != (path_count,) or validity_dataset.dtype != np.dtype(np.uint8):
        raise PackedHDF5Error(
            f"{chunk.file} /paths/metric_valid_bits is not a path-aligned uint8 vector"
        )
    validity = np.asarray(validity_dataset[:], dtype=np.uint8)
    known_bits = sum(PATH_METRIC_VALIDITY_BITS.values())
    if np.any(np.bitwise_and(validity, np.uint8(0xFF ^ known_bits))):
        raise PackedHDF5Error(f"{chunk.file} metric validity contains unknown bits")
    for metric, dataset_name in _METRIC_DATASETS.items():
        dataset = _dataset(h5, f"paths/{dataset_name}")
        if dataset.shape != (path_count,) or dataset.dtype != np.dtype(np.float32):
            raise PackedHDF5Error(
                f"{chunk.file} /paths/{dataset_name} is not a path-aligned float32 vector"
            )
        values = np.asarray(dataset[:], dtype=np.float32)
        valid = (validity & PATH_METRIC_VALIDITY_BITS[metric]) != 0
        if np.any(~np.isfinite(values[valid])) or np.any(~np.isnan(values[~valid])):
            raise PackedHDF5Error(
                f"{chunk.file} /paths/{dataset_name} disagrees with metric validity bits"
            )

    for path, expected_shape, expected_dtype in (
        ("bounces/xyz_m", (bounce_count, 3), np.dtype(np.float32)),
        ("bounces/interaction", (bounce_count,), np.dtype(np.uint8)),
        ("bounces/material_id", (bounce_count,), np.dtype(np.uint16)),
    ):
        dataset = _dataset(h5, path)
        if dataset.shape != expected_shape or dataset.dtype != expected_dtype:
            raise PackedHDF5Error(
                f"{chunk.file} /{path} is not aligned to bounces with dtype {expected_dtype}"
            )
    bounce_xyz = np.asarray(_dataset(h5, "bounces/xyz_m")[:], dtype=np.float32)
    if np.any(~np.isfinite(bounce_xyz)):
        raise PackedHDF5Error(f"{chunk.file} contains non-finite bounce coordinates")
    interactions = np.asarray(_dataset(h5, "bounces/interaction")[:], dtype=np.uint8)
    if np.any(interactions == 0):
        raise PackedHDF5Error(f"{chunk.file} contains a zero physical-bounce interaction")

    material_count = len(_dataset(h5, "static/materials/name"))
    material_names = _decode_string_array(_dataset(h5, "static/materials/name"))
    material_itu_types = _decode_string_array(_dataset(h5, "static/materials/itu_type"))
    if not material_names or material_names[0] != "" or material_itu_types[0] != "":
        raise PackedHDF5Error(f"{chunk.file} material catalog row zero must be empty")
    material_ids = np.asarray(_dataset(h5, "bounces/material_id")[:], dtype=np.uint64)
    if material_ids.size and int(np.max(material_ids)) >= material_count:
        raise PackedHDF5Error(f"{chunk.file} contains an unknown material ID")
    if len(material_itu_types) != material_count:
        raise PackedHDF5Error(f"{chunk.file} material catalog columns are misaligned")

    target_offsets = np.asarray(_dataset(h5, "index/frame_target_offsets")[:], dtype=np.int64)
    if target_offsets.shape != (frame_count + 1,):
        raise PackedHDF5Error(f"{chunk.file} target offsets do not match frames")
    if target_offsets[0] != 0 or np.any(target_offsets[1:] < target_offsets[:-1]):
        raise PackedHDF5Error(f"{chunk.file} target offsets are invalid")
    target_count = int(target_offsets[-1])
    target_positions = _dataset(h5, "targets/position_m")
    if target_positions.shape != (target_count, 3) or target_positions.dtype != np.dtype(
        np.float64
    ):
        raise PackedHDF5Error(f"{chunk.file} target positions are not aligned")
    if np.any(~np.isfinite(np.asarray(target_positions[:], dtype=np.float64))):
        raise PackedHDF5Error(f"{chunk.file} contains non-finite target coordinates")
    if len(_dataset(h5, "targets/metadata_json")) != target_count:
        raise PackedHDF5Error(f"{chunk.file} target metadata is not aligned")
    target_metadata = _dataset(h5, "targets/metadata_json")
    for target_index in range(target_count):
        frame_row = int(
            np.searchsorted(
                target_offsets[1:],
                target_index,
                side="right",
            )
        )
        _decode_target_metadata_cell(
            target_metadata,
            target_index,
            chunk_file=chunk.file,
            frame_id=int(frame_ids[frame_row]),
        )

    frame_shapes = {
        "frames/timestamp_s": (frame_count,),
        "frames/recomputed": (frame_count,),
        "frames/source_json": (frame_count,),
        "frames/beamforming_json": (frame_count,),
        "frames/sensing_metadata_json": (frame_count,),
    }
    for path, shape in frame_shapes.items():
        if _dataset(h5, path).shape != shape:
            raise PackedHDF5Error(f"{chunk.file} /{path} is not aligned to frames")

    for path, count in (
        ("frames/tx_position_m", tx_count),
        ("frames/tx_orientation_rad", tx_count),
        ("frames/rx_position_m", rx_count),
        ("frames/rx_orientation_rad", rx_count),
    ):
        dataset = _dataset(h5, path)
        if dataset.shape != (frame_count, count, 3) or dataset.dtype != np.dtype(np.float64):
            raise PackedHDF5Error(f"{chunk.file} /{path} is not aligned to frames/devices")
        if np.any(~np.isfinite(np.asarray(dataset[:], dtype=np.float64))):
            raise PackedHDF5Error(f"{chunk.file} /{path} contains non-finite values")

    for path in ("frames/source_json", "frames/beamforming_json", "frames/sensing_metadata_json"):
        dataset = _dataset(h5, path)
        for row in range(frame_count):
            _decode_json_optional_mapping_cell(dataset, row, label=f"{path}[{row}]")

    config = _load_static_sensing_config(h5)
    from shared.extensions.sensing import SENSING_SCHEMA_VERSION, assert_processing_recipe_identity

    metadata = _dataset(h5, "frames/sensing_metadata_json")
    for row in range(frame_count):
        sensing = _decode_json_optional_mapping_cell(metadata, row, label=metadata.name)
        sensing = _merge_static_sensing_config(sensing, config)
        if sensing is not None and sensing.get("schema_version") == SENSING_SCHEMA_VERSION:
            assert_processing_recipe_identity(sensing)

    fixed_sensing = h5.get("sensing/fixed")
    if not isinstance(fixed_sensing, h5py.Group):
        raise PackedHDF5Error(f"{chunk.file} is missing /sensing/fixed")
    for product, node in fixed_sensing.items():
        if not isinstance(node, h5py.Group):
            raise PackedHDF5Error(f"{chunk.file} fixed sensing product {product!r} is not a group")
        data = _dataset(node, "data")
        present = _dataset(node, "present")
        if data.shape[:1] != (frame_count,) or present.shape != (frame_count,):
            raise PackedHDF5Error(
                f"{chunk.file} fixed sensing product {product!r} is not frame-aligned"
            )
        if not (np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)):
            raise PackedHDF5Error(f"{chunk.file} fixed sensing product {product!r} is not numeric")
        presence = np.asarray(present[:], dtype=np.uint8)
        if np.any(presence > 1):
            raise PackedHDF5Error(
                f"{chunk.file} fixed sensing product {product!r} has invalid presence bits"
            )

    ragged_sensing = h5.get("sensing/ragged")
    if not isinstance(ragged_sensing, h5py.Group):
        raise PackedHDF5Error(f"{chunk.file} is missing /sensing/ragged")
    for product, node in ragged_sensing.items():
        if not isinstance(node, h5py.Group):
            raise PackedHDF5Error(f"{chunk.file} ragged sensing product {product!r} is not a group")
        offsets = np.asarray(_dataset(node, "offsets")[:], dtype=np.int64)
        shapes = _dataset(node, "shapes")
        present = np.asarray(_dataset(node, "present")[:], dtype=np.uint8)
        values = _dataset(node, "values")
        if not (np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)):
            raise PackedHDF5Error(f"{chunk.file} ragged sensing product {product!r} is not numeric")
        if offsets.shape != (frame_count + 1,) or offsets[0] != 0:
            raise PackedHDF5Error(
                f"{chunk.file} ragged sensing product {product!r} has invalid offsets"
            )
        if np.any(offsets[1:] < offsets[:-1]) or int(offsets[-1]) != len(values):
            raise PackedHDF5Error(
                f"{chunk.file} ragged sensing product {product!r} offsets disagree with values"
            )
        if shapes.ndim != 2 or shapes.shape[0] != frame_count:
            raise PackedHDF5Error(
                f"{chunk.file} ragged sensing product {product!r} shapes are not frame-aligned"
            )
        if present.shape != (frame_count,) or np.any(present > 1):
            raise PackedHDF5Error(
                f"{chunk.file} ragged sensing product {product!r} has invalid presence bits"
            )
        shape_values = np.asarray(shapes[:], dtype=np.int64)
        for row in range(frame_count):
            item_count = int(offsets[row + 1] - offsets[row])
            if present[row]:
                expected_count = int(np.prod(shape_values[row], dtype=np.int64))
                if np.any(shape_values[row] < 0) or expected_count != item_count:
                    raise PackedHDF5Error(
                        f"{chunk.file} ragged sensing product {product!r} row {row} "
                        "shape disagrees with its value interval"
                    )
            elif item_count:
                raise PackedHDF5Error(
                    f"{chunk.file} absent ragged sensing product {product!r} row {row} "
                    "owns stored values"
                )


__all__ = [
    "HDF5HandleLRU",
    "PackedHDF5Error",
    "PackedHDF5Reader",
    "validate_packed_chunk",
    "validate_packed_chunk_structure",
]
