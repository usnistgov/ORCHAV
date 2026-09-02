"""Append canonical MPC frames to the packed-ragged HDF5 v2 layout.

The writer owns one private ``.partial`` file at a time. A complete
``StandardMPCFrame`` is validated by its constructor, prepared for the current
chunk's material catalog, and appended to resizable datasets. Only
:meth:`finalize_to_range_name` publishes the file under its manifest-visible
range name.

This module is the shared physical-storage boundary used by generator,
sensing, benchmark, and fixture publishers. It does not convert the canonical
frame into a second complete in-memory representation.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from shared.frames.contracts import (
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
    PATH_METRIC_ORDER,
    PathMetric,
)
from shared.frames.json_codec import dumps_frame_json
from shared.frames.manifest import FrameChunkManifest
from shared.frames.types import PATH_METRIC_ARRAY_FIELDS, StandardMPCFrame

_METRIC_UNITS: dict[PathMetric, str] = {
    PathMetric.DELAY_NS: "ns",
    PathMetric.PATH_LOSS_DB: "dB",
    PathMetric.AOA_AZ_DEG: "deg",
    PathMetric.AOA_EL_DEG: "deg",
    PathMetric.AOD_AZ_DEG: "deg",
    PathMetric.AOD_EL_DEG: "deg",
}

_PRODUCT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_EMPTY_MATERIAL = ("", "")
_NUMERIC_CHUNK_TARGET_BYTES = 64 * 1024
_FRAME_CHUNK_ROWS = 128
_STRING_CHUNK_ROWS = 256


class PackedMPCChunkBoundaryError(ValueError):
    """Signal that the frame must be retried in a new physical HDF5 chunk.

    The current chunk remains valid and unchanged by the rejected append.
    """


@dataclass(frozen=True, slots=True)
class PreparedPackedMPCFrame:
    """A canonical frame with HDF5-specific metadata prepared for append.

    Run-level output can inspect ``estimated_uncompressed_bytes`` before
    choosing a chunk without preparing a large frame twice. Boundary checks
    remain an append-time operation and therefore happen before any HDF5
    dataset is resized.
    """

    packed: StandardMPCFrame
    material_ids: np.ndarray
    material_names: tuple[str, ...]
    material_itu_types: tuple[str, ...]
    fixed_sensing: Mapping[str, np.ndarray]
    cir: np.ndarray | None
    sensing_config_json: str | None
    source_json: str
    beamforming_json: str
    sensing_metadata_json: str
    target_metadata_json: tuple[str, ...]
    estimated_uncompressed_bytes: int


def _json_size(value: Any) -> int:
    return len(dumps_frame_json(value).encode("utf-8"))


def _packed_frame_bytes(
    frame: StandardMPCFrame,
    *,
    material_names: Sequence[str] | None = None,
    material_itu_types: Sequence[str] | None = None,
) -> int:
    arrays = (
        frame.tx_rx_pairs,
        frame.pair_path_offsets,
        frame.bounce_offsets,
        frame.tx_positions,
        frame.rx_positions,
        frame.tx_orientations,
        frame.rx_orientations,
        frame.bounce_xyz_m,
        frame.interactions,
        frame.material_ids,
        frame.metric_valid_bits,
        frame.target_positions_m,
        *frame.path_metrics.values(),
    )
    total = sum(int(value.nbytes) for value in arrays if value is not None)
    total += sum(len(value.encode("utf-8")) for value in frame.tx_names or ())
    total += sum(len(value.encode("utf-8")) for value in frame.rx_names or ())
    total += sum(
        len(value.encode("utf-8"))
        for value in (frame.material_names if material_names is None else material_names)
    )
    total += sum(
        len(value.encode("utf-8"))
        for value in (
            frame.material_itu_types if material_itu_types is None else material_itu_types
        )
    )
    total += sum(_json_size(value) for value in frame.targets_metadata or ())
    total += _json_size(frame.provenance)
    total += _json_size(frame.beamforming)
    fixed_sensing, cir, sensing_metadata = _sensing_parts(frame.sensing)
    total += sum(int(values.nbytes) for values in fixed_sensing.values())
    total += 0 if cir is None else int(cir.nbytes)
    total += _json_size(sensing_metadata)
    return total


def estimate_packed_frame_bytes(frame: StandardMPCFrame) -> int:
    """Estimate the uncompressed v2 payload bytes for one complete frame."""
    return _packed_frame_bytes(frame)


def estimate_prepared_frame_bytes(frame: PreparedPackedMPCFrame) -> int:
    """Return the precomputed uncompressed byte estimate for a prepared frame."""
    return frame.estimated_uncompressed_bytes


def _compression_options(compression: str | None) -> tuple[str | None, int | None]:
    if compression is None:
        return None, None
    normalized = compression.strip().lower()
    if normalized in {"none", "fast"}:
        return None, None
    if normalized in {"lzf", "balanced"}:
        return "lzf", None
    if normalized in {"gzip", "gzip-4", "compact"}:
        return "gzip", 4
    raise ValueError("compression must be one of None, 'lzf'/'balanced', or 'gzip-4'/'compact'")


def _numeric_chunk_shape(dtype: Any, *trailing_shape: int) -> tuple[int, ...]:
    """Choose a moderate chunk near 64 KiB for an append-only numeric column.

    h5py's automatic choice is pathological for datasets initialized with one
    offset: it selects ``(1,)`` and creates one HDF5 chunk per path.  Besides
    wasting metadata, that turns a contiguous offset read into thousands of
    tiny chunk lookups.  An explicit byte-oriented policy keeps both random
    frame reads and append memory bounded.
    """

    normalized_shape = tuple(max(1, int(size)) for size in trailing_shape)
    row_values = int(np.prod(normalized_shape, dtype=np.int64)) if normalized_shape else 1
    row_bytes = max(1, np.dtype(dtype).itemsize * row_values)
    rows = max(1, _NUMERIC_CHUNK_TARGET_BYTES // row_bytes)
    return (rows, *normalized_shape)


def _topology_payload(frame: StandardMPCFrame) -> dict[str, Any]:
    return {
        "tx_rx_pairs": frame.tx_rx_pairs.tolist(),
        "tx_names": list(frame.tx_names or ()),
        "rx_names": list(frame.rx_names or ()),
    }


def _material_payload_for_catalog(
    frame: StandardMPCFrame,
    material_catalog: Sequence[tuple[str, str]],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Return IDs and catalogs rebased to the chunk-wide material table.

    Canonical frames own a self-contained material catalog. HDF5 stores one
    catalog per chunk, so independently produced frames can require a small ID
    translation when their catalogs differ. The common identical/prefix case
    returns the original material-ID array.
    """

    chunk_catalog = [(str(name), str(itu_type)) for name, itu_type in material_catalog]
    frame_catalog = list(zip(frame.material_names, frame.material_itu_types, strict=True))
    if not chunk_catalog:
        return frame.material_ids, frame.material_names, frame.material_itu_types
    if frame_catalog[: len(chunk_catalog)] == chunk_catalog:
        return frame.material_ids, frame.material_names, frame.material_itu_types

    combined_catalog = list(chunk_catalog)
    combined_index = {entry: index for index, entry in enumerate(combined_catalog)}
    local_to_chunk = np.empty((len(frame_catalog),), dtype=np.uint16)
    identity_mapping = True
    for local_id, material in enumerate(frame_catalog):
        chunk_id = combined_index.get(material)
        if chunk_id is None:
            if len(combined_catalog) > np.iinfo(np.uint16).max:
                raise ValueError("Material catalog exceeds the uint16 material ID range")
            chunk_id = len(combined_catalog)
            combined_catalog.append(material)
            combined_index[material] = chunk_id
        local_to_chunk[local_id] = chunk_id
        identity_mapping = identity_mapping and chunk_id == local_id

    material_ids = frame.material_ids if identity_mapping else local_to_chunk[frame.material_ids]
    return (
        material_ids,
        tuple(name for name, _ in combined_catalog),
        tuple(itu_type for _, itu_type in combined_catalog),
    )


def _stable_id(payload: Any) -> str:
    return hashlib.sha256(dumps_frame_json(payload).encode("utf-8")).hexdigest()


def _sensing_parts(
    sensing: Mapping[str, Any] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, dict[str, Any]]:
    """Split top-level sensing arrays from small JSON metadata."""
    fixed: dict[str, np.ndarray] = {}
    cir: np.ndarray | None = None
    metadata: dict[str, Any] = {}
    for key, value in (sensing or {}).items():
        if key == "cir" and value is not None:
            cir = np.asarray(value)
            continue
        if isinstance(value, np.ndarray):
            if not _PRODUCT_NAME_RE.fullmatch(key):
                raise ValueError(f"Invalid HDF5 sensing product name {key!r}")
            if value.dtype.hasobject:
                raise ValueError(f"Sensing product {key!r} cannot use object dtype")
            fixed[key] = np.asarray(value)
            continue
        metadata[key] = value
    return fixed, cir, metadata


class PackedMPCChunkWriter:
    """Append validated frames to one private packed HDF5 v2 chunk.

    ``prepare()`` performs material rebasing and metadata serialization without
    touching HDF5. ``append_prepared()`` checks the strictly increasing frame
    axis plus chunk-wide TX/RX topology and sensing-layout invariants before it
    resizes a dataset. ``finalize_to_range_name()`` publishes only this chunk
    and returns its inventory entry; ``FrameSetWriter`` owns whole-directory
    publication and the authoritative manifest.
    """

    def __init__(
        self,
        frames_dir: str | Path,
        *,
        generation_id: str,
        compression: str | None = "lzf",
        partial_name: str | None = None,
    ) -> None:
        self._frames_dir = Path(frames_dir)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise ValueError("generation_id must be a non-empty string")
        self._generation_id = generation_id
        if partial_name is None:
            partial_name = f".mpc_frames.{uuid.uuid4().hex}.h5.partial"
        if Path(partial_name).name != partial_name:
            raise ValueError("partial_name must be a plain filename")
        if not partial_name.startswith(".") or not partial_name.endswith(".partial"):
            raise ValueError("partial_name must be private (leading '.') and end in '.partial'")
        self._partial_path = self._frames_dir / partial_name
        if self._partial_path.exists():
            raise FileExistsError(self._partial_path)

        self._compression, self._compression_opts = _compression_options(compression)
        self._compression_label = (
            "none"
            if self._compression is None
            else (
                f"{self._compression}-{self._compression_opts}"
                if self._compression_opts is not None
                else self._compression
            )
        )
        self._h5 = h5py.File(self._partial_path, "x")
        self._frame_ids: list[int] = []
        self._uncompressed_bytes = 0
        self._topology: dict[str, Any] | None = None
        self._topology_id = ""
        self._material_catalog: list[tuple[str, str]] = [_EMPTY_MATERIAL]
        self._fixed_sensing_layout: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        self._cir_rank: int | None = None
        self._cir_dtype: np.dtype[Any] | None = None
        self._sensing_config_json: str | None = None
        self._sensing_config_initialized = False
        self._closed = False
        self._finalized = False
        self._initialize_file()

    @property
    def partial_path(self) -> Path:
        return self._partial_path

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(self._frame_ids)

    @property
    def current_frame_ids(self) -> tuple[int, ...]:
        return self.frame_ids

    @property
    def frame_count(self) -> int:
        return len(self._frame_ids)

    @property
    def current_frame_count(self) -> int:
        return self.frame_count

    @property
    def uncompressed_bytes(self) -> int:
        return self._uncompressed_bytes

    @property
    def topology_id(self) -> str:
        return self._topology_id

    @property
    def sensing_layout_id(self) -> str:
        return _stable_id(self._sensing_layout_payload())

    def _numeric_options(self) -> dict[str, Any]:
        if self._compression is None:
            return {}
        options: dict[str, Any] = {
            "compression": self._compression,
            "shuffle": True,
        }
        if self._compression_opts is not None:
            options["compression_opts"] = self._compression_opts
        return options

    def _initialize_file(self) -> None:
        h5 = self._h5
        h5.attrs["schema_version"] = MPC_HDF5_SCHEMA_VERSION
        h5.attrs["storage_layout"] = MPC_HDF5_LAYOUT
        h5.attrs["file_kind"] = "mpc_frames"
        h5.attrs["packed_frame_version"] = PACKED_MPC_FRAME_VERSION
        h5.attrs["generation_id"] = self._generation_id
        h5.attrs["compression"] = self._compression_label
        h5.attrs["publication_state"] = "partial"

        static = h5.create_group("static")
        materials = static.create_group("materials")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        materials.create_dataset(
            "name",
            data=np.asarray([""], dtype=object),
            dtype=string_dtype,
            maxshape=(None,),
            chunks=(_STRING_CHUNK_ROWS,),
        )
        materials.create_dataset(
            "itu_type",
            data=np.asarray([""], dtype=object),
            dtype=string_dtype,
            maxshape=(None,),
            chunks=(_STRING_CHUNK_ROWS,),
        )
        static.create_dataset(
            "sensing_config_json",
            data=dumps_frame_json(None),
            dtype=string_dtype,
        )

        frames = h5.create_group("frames")
        frame_ids = frames.create_dataset(
            "id",
            shape=(0,),
            maxshape=(None,),
            chunks=(_FRAME_CHUNK_ROWS,),
            dtype=np.int64,
            **self._numeric_options(),
        )
        frame_ids.make_scale("frame")
        self._create_frame_vector("timestamp_s", np.float64, units="s")
        self._create_frame_vector("recomputed", np.uint8)
        for name in ("source_json", "beamforming_json", "sensing_metadata_json"):
            dataset = frames.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(_FRAME_CHUNK_ROWS,),
                dtype=string_dtype,
            )
            self._attach_frame_scale(dataset)

        index = h5.create_group("index")
        index.create_dataset(
            "frame_target_offsets",
            data=np.asarray([0], dtype=np.int64),
            maxshape=(None,),
            chunks=_numeric_chunk_shape(np.int64),
            **self._numeric_options(),
        )

        paths = h5.create_group("paths")
        paths.create_dataset(
            "bounce_offsets",
            data=np.asarray([0], dtype=np.int64),
            maxshape=(None,),
            chunks=_numeric_chunk_shape(np.int64),
            **self._numeric_options(),
        )
        for metric in PATH_METRIC_ORDER:
            dataset = paths.create_dataset(
                metric.value,
                shape=(0,),
                maxshape=(None,),
                chunks=_numeric_chunk_shape(np.float32),
                dtype=np.float32,
                **self._numeric_options(),
            )
            dataset.attrs["units"] = _METRIC_UNITS[metric]
        paths.create_dataset(
            "metric_valid_bits",
            shape=(0,),
            maxshape=(None,),
            chunks=_numeric_chunk_shape(np.uint8),
            dtype=np.uint8,
            **self._numeric_options(),
        )

        bounces = h5.create_group("bounces")
        xyz = bounces.create_dataset(
            "xyz_m",
            shape=(0, 3),
            maxshape=(None, 3),
            chunks=_numeric_chunk_shape(np.float32, 3),
            dtype=np.float32,
            **self._numeric_options(),
        )
        xyz.attrs["units"] = "m"
        for name, dtype in (("interaction", np.uint8), ("material_id", np.uint16)):
            bounces.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=_numeric_chunk_shape(dtype),
                dtype=dtype,
                **self._numeric_options(),
            )

        targets = h5.create_group("targets")
        positions = targets.create_dataset(
            "position_m",
            shape=(0, 3),
            maxshape=(None, 3),
            chunks=_numeric_chunk_shape(np.float64, 3),
            dtype=np.float64,
            **self._numeric_options(),
        )
        positions.attrs["units"] = "m"
        targets.create_dataset(
            "metadata_json",
            shape=(0,),
            maxshape=(None,),
            chunks=(_STRING_CHUNK_ROWS,),
            dtype=string_dtype,
        )

        sensing = h5.create_group("sensing")
        sensing.create_group("fixed")
        cir = sensing.create_group("ragged").create_group("cir")
        cir.create_dataset(
            "offsets",
            data=np.asarray([0], dtype=np.int64),
            maxshape=(None,),
            chunks=_numeric_chunk_shape(np.int64),
            **self._numeric_options(),
        )
        shapes = cir.create_dataset(
            "shapes",
            shape=(0, 0),
            maxshape=(None, None),
            chunks=(1, 1),
            dtype=np.int64,
            **self._numeric_options(),
        )
        self._attach_frame_scale(shapes)
        present = cir.create_dataset(
            "present",
            shape=(0,),
            maxshape=(None,),
            chunks=(_FRAME_CHUNK_ROWS,),
            dtype=np.uint8,
            **self._numeric_options(),
        )
        self._attach_frame_scale(present)

    def _create_frame_vector(
        self,
        name: str,
        dtype: Any,
        *,
        units: str | None = None,
    ) -> h5py.Dataset:
        dataset = self._h5["frames"].create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(_FRAME_CHUNK_ROWS,),
            dtype=dtype,
            **self._numeric_options(),
        )
        self._attach_frame_scale(dataset)
        if units is not None:
            dataset.attrs["units"] = units
        return dataset

    def _attach_frame_scale(self, dataset: h5py.Dataset) -> None:
        dataset.dims[0].attach_scale(self._h5["frames/id"])
        dataset.dims[0].label = "frame"

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PackedMPCChunkWriter is closed")

    def _validate_frame_id(self, frame_id: int) -> None:
        if self._frame_ids and frame_id <= self._frame_ids[-1]:
            raise ValueError("Frame IDs must be appended in strictly increasing order")

    def _check_topology(self, packed: StandardMPCFrame) -> dict[str, Any]:
        topology = _topology_payload(packed)
        if self._topology is not None and topology != self._topology:
            raise PackedMPCChunkBoundaryError(
                "TX/RX topology or device names changed; start a new HDF5 chunk"
            )
        return topology

    def _check_sensing_layout(
        self,
        fixed: Mapping[str, np.ndarray],
        cir: np.ndarray | None,
        sensing_config_json: str | None,
    ) -> None:
        for product, values in fixed.items():
            if not (
                np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)
            ):
                raise ValueError(f"Fixed sensing product {product!r} must use a numeric dtype")
            try:
                h5py.h5t.py_create(values.dtype)
            except TypeError as exc:
                raise ValueError(
                    f"Fixed sensing product {product!r} dtype "
                    f"{values.dtype} is not supported by HDF5"
                ) from exc
            previous = self._fixed_sensing_layout.get(product)
            current = (tuple(int(size) for size in values.shape), values.dtype)
            if previous is not None and previous != current:
                raise PackedMPCChunkBoundaryError(
                    f"Fixed sensing product {product!r} changed shape or dtype "
                    f"from {previous} to {current}; start a new HDF5 chunk"
                )
        if cir is not None:
            if cir.ndim == 0:
                raise ValueError("sensing.cir must have at least one dimension")
            if not (np.issubdtype(cir.dtype, np.number) or np.issubdtype(cir.dtype, np.bool_)):
                raise ValueError("sensing.cir must use a numeric dtype")
            if self._cir_rank is not None and cir.ndim != self._cir_rank:
                raise PackedMPCChunkBoundaryError(
                    "sensing.cir rank changed; start a new HDF5 chunk"
                )
            if self._cir_dtype is not None and cir.dtype != self._cir_dtype:
                raise PackedMPCChunkBoundaryError(
                    "sensing.cir dtype changed; start a new HDF5 chunk"
                )
            try:
                h5py.h5t.py_create(cir.dtype)
            except TypeError as exc:
                raise ValueError(f"sensing.cir dtype {cir.dtype} is not supported by HDF5") from exc
        if self._sensing_config_initialized and sensing_config_json != self._sensing_config_json:
            raise PackedMPCChunkBoundaryError(
                "Static sensing configuration changed; start a new HDF5 chunk"
            )

    def _accept_sensing_config(self, sensing_config_json: str | None) -> None:
        if self._sensing_config_initialized:
            return
        self._sensing_config_json = sensing_config_json
        self._sensing_config_initialized = True
        self._h5["static/sensing_config_json"][()] = (
            dumps_frame_json(None) if sensing_config_json is None else sensing_config_json
        )

    def _initialize_topology(
        self,
        packed: StandardMPCFrame,
        topology: dict[str, Any],
    ) -> None:
        if self._topology is not None:
            return
        self._topology = topology
        self._topology_id = _stable_id(topology)
        static = self._h5["static"]
        assert packed.tx_rx_pairs is not None
        static.create_dataset("tx_rx_pairs", data=packed.tx_rx_pairs.astype(np.int32, copy=False))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        static.create_dataset(
            "tx_names",
            data=np.asarray(packed.tx_names, dtype=object),
            dtype=string_dtype,
        )
        static.create_dataset(
            "rx_names",
            data=np.asarray(packed.rx_names, dtype=object),
            dtype=string_dtype,
        )

        frames = self._h5["frames"]
        for name, count in (
            ("tx_position_m", len(packed.tx_positions)),
            ("rx_position_m", len(packed.rx_positions)),
            ("tx_orientation_rad", len(packed.tx_orientations)),
            ("rx_orientation_rad", len(packed.rx_orientations)),
        ):
            dataset = frames.create_dataset(
                name,
                shape=(0, count, 3),
                maxshape=(None, None if count == 0 else count, 3),
                chunks=(1, max(1, count), 3),
                dtype=np.float64,
                **self._numeric_options(),
            )
            self._attach_frame_scale(dataset)
            dataset.attrs["units"] = "m" if "position" in name else "rad"

        pair_count = len(packed.tx_rx_pairs)
        pair_offsets = self._h5["index"].create_dataset(
            "frame_pair_path_offsets",
            shape=(0, pair_count + 1),
            maxshape=(None, pair_count + 1),
            chunks=(1, pair_count + 1),
            dtype=np.int64,
            **self._numeric_options(),
        )
        self._attach_frame_scale(pair_offsets)
        pair_offsets.attrs["semantics"] = "global [start,end) path boundaries per TX/RX pair"

    def _extend_material_catalog(
        self,
        material_names: Sequence[str],
        material_itu_types: Sequence[str],
    ) -> None:
        complete_catalog = list(zip(material_names, material_itu_types, strict=True))
        if complete_catalog[: len(self._material_catalog)] != self._material_catalog:
            raise ValueError("Canonical frame material catalog does not extend the chunk catalog")
        new_entries = complete_catalog[len(self._material_catalog) :]
        if not new_entries:
            return
        materials = self._h5["static/materials"]
        start = len(self._material_catalog)
        end = len(complete_catalog)
        for name, column in (("name", 0), ("itu_type", 1)):
            dataset = materials[name]
            dataset.resize((end,))
            dataset[start:end] = [entry[column] for entry in new_entries]
        self._material_catalog = complete_catalog

    @staticmethod
    def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
        values = np.asarray(values)
        count = values.shape[0]
        if count == 0:
            return
        start = dataset.shape[0]
        dataset.resize((start + count, *dataset.shape[1:]))
        dataset[start:] = values

    @staticmethod
    def _append_scalar(dataset: h5py.Dataset, value: Any) -> None:
        start = dataset.shape[0]
        dataset.resize((start + 1, *dataset.shape[1:]))
        dataset[start] = value

    def _append_fixed_sensing(
        self,
        fixed: Mapping[str, np.ndarray],
        prior_frame_count: int,
    ) -> None:
        root = self._h5["sensing/fixed"]
        for product, values in fixed.items():
            if product in self._fixed_sensing_layout:
                continue
            shape = tuple(int(size) for size in values.shape)
            self._fixed_sensing_layout[product] = (shape, values.dtype)
            group = root.create_group(product)
            maxshape = (None, *(None if size == 0 else size for size in shape))
            data = group.create_dataset(
                "data",
                shape=(prior_frame_count, *shape),
                maxshape=maxshape,
                chunks=True,
                dtype=values.dtype,
                **self._numeric_options(),
            )
            self._attach_frame_scale(data)
            present = group.create_dataset(
                "present",
                shape=(prior_frame_count,),
                maxshape=(None,),
                chunks=True,
                dtype=np.uint8,
                **self._numeric_options(),
            )
            self._attach_frame_scale(present)

        for product in sorted(self._fixed_sensing_layout):
            group = root[product]
            product_values = fixed.get(product)
            data = group["data"]
            present = group["present"]
            data.resize((prior_frame_count + 1, *data.shape[1:]))
            present.resize((prior_frame_count + 1,))
            if product_values is None:
                data[prior_frame_count] = np.zeros(data.shape[1:], dtype=data.dtype)
                present[prior_frame_count] = np.uint8(0)
            else:
                data[prior_frame_count] = product_values
                present[prior_frame_count] = np.uint8(1)

    def _append_cir(self, cir: np.ndarray | None, prior_frame_count: int) -> None:
        group = self._h5["sensing/ragged/cir"]
        offsets = group["offsets"]
        shapes = group["shapes"]
        present = group["present"]

        if cir is not None and self._cir_rank is None:
            self._cir_rank = cir.ndim
            self._cir_dtype = cir.dtype
            shapes.resize((prior_frame_count, self._cir_rank))
            group.create_dataset(
                "values",
                shape=(0,),
                maxshape=(None,),
                chunks=_numeric_chunk_shape(self._cir_dtype),
                dtype=self._cir_dtype,
                **self._numeric_options(),
            )
        rank = self._cir_rank or 0
        if shapes.shape[1] != rank:
            shapes.resize((prior_frame_count, rank))
        shapes.resize((prior_frame_count + 1, rank))
        present.resize((prior_frame_count + 1,))

        start = int(offsets[-1])
        if cir is None:
            end = start
            if rank:
                shapes[prior_frame_count] = np.zeros((rank,), dtype=np.int64)
            present[prior_frame_count] = np.uint8(0)
        else:
            flattened = np.asarray(cir).reshape(-1)
            self._append(group["values"], flattened)
            end = start + flattened.size
            shapes[prior_frame_count] = np.asarray(cir.shape, dtype=np.int64)
            present[prior_frame_count] = np.uint8(1)
        offsets.resize((offsets.shape[0] + 1,))
        offsets[-1] = end

    def prepare(self, frame: StandardMPCFrame) -> PreparedPackedMPCFrame:
        """Prepare one validated canonical frame without mutating HDF5."""
        self._require_open()
        if not isinstance(frame, StandardMPCFrame):
            raise TypeError("frame must be a complete StandardMPCFrame")
        # Complete frames validate once at construction. Their NumPy buffers
        # remain immutable by contract until every consumer releases them.
        material_ids, material_names, material_itu_types = _material_payload_for_catalog(
            frame,
            self._material_catalog,
        )
        packed = frame
        fixed_sensing, cir, sensing_metadata = _sensing_parts(packed.sensing)
        sensing_metadata = dict(sensing_metadata)
        static_sensing_config = {
            key: sensing_metadata.pop(key)
            for key in ("config", "config_resolved")
            if key in sensing_metadata
        }
        sensing_config_json = (
            None if not static_sensing_config else dumps_frame_json(static_sensing_config)
        )
        return PreparedPackedMPCFrame(
            packed=packed,
            material_ids=material_ids,
            material_names=material_names,
            material_itu_types=material_itu_types,
            fixed_sensing=fixed_sensing,
            cir=cir,
            sensing_config_json=sensing_config_json,
            source_json=dumps_frame_json(packed.provenance),
            beamforming_json=dumps_frame_json(packed.beamforming),
            sensing_metadata_json=dumps_frame_json(
                None if packed.sensing is None else sensing_metadata
            ),
            target_metadata_json=tuple(
                dumps_frame_json(metadata) for metadata in packed.targets_metadata or ()
            ),
            estimated_uncompressed_bytes=_packed_frame_bytes(
                packed,
                material_names=material_names,
                material_itu_types=material_itu_types,
            ),
        )

    def append_prepared(self, prepared: PreparedPackedMPCFrame) -> None:
        """Append a prepared frame after side-effect-free boundary checks."""
        self._require_open()
        packed = prepared.packed
        frame_id = int(packed.frame_index)
        self._validate_frame_id(frame_id)

        topology = self._check_topology(packed)
        fixed_sensing = prepared.fixed_sensing
        cir = prepared.cir
        sensing_config_json = prepared.sensing_config_json
        self._check_sensing_layout(fixed_sensing, cir, sensing_config_json)

        prior_frame_count = self.frame_count
        self._initialize_topology(packed, topology)
        self._extend_material_catalog(
            prepared.material_names,
            prepared.material_itu_types,
        )
        self._accept_sensing_config(sensing_config_json)

        path_base = self._h5["paths/metric_valid_bits"].shape[0]
        bounce_base = self._h5["bounces/material_id"].shape[0]
        target_base = self._h5["targets/position_m"].shape[0]

        self._append_scalar(self._h5["frames/id"], frame_id)
        self._append_scalar(
            self._h5["frames/timestamp_s"],
            np.nan if packed.timestamp_s is None else packed.timestamp_s,
        )
        self._append_scalar(
            self._h5["frames/recomputed"],
            np.uint8(packed.recomputed_from_stored_positions),
        )
        self._append_scalar(
            self._h5["frames/source_json"],
            prepared.source_json,
        )
        self._append_scalar(
            self._h5["frames/beamforming_json"],
            prepared.beamforming_json,
        )
        self._append_scalar(
            self._h5["frames/sensing_metadata_json"],
            prepared.sensing_metadata_json,
        )
        for name, values in (
            ("tx_position_m", packed.tx_positions),
            ("rx_position_m", packed.rx_positions),
            ("tx_orientation_rad", packed.tx_orientations),
            ("rx_orientation_rad", packed.rx_orientations),
        ):
            self._append_scalar(self._h5[f"frames/{name}"], values)

        self._append_scalar(
            self._h5["index/frame_pair_path_offsets"],
            packed.pair_path_offsets + path_base,
        )
        target_offsets = self._h5["index/frame_target_offsets"]
        target_offsets.resize((target_offsets.shape[0] + 1,))
        target_offsets[-1] = target_base + len(packed.target_positions_m)

        path_bounce_offsets = self._h5["paths/bounce_offsets"]
        adjusted_bounce_offsets = packed.bounce_offsets[1:] + bounce_base
        self._append(path_bounce_offsets, adjusted_bounce_offsets)
        for metric in PATH_METRIC_ORDER:
            self._append(
                self._h5[f"paths/{metric.value}"],
                getattr(packed, PATH_METRIC_ARRAY_FIELDS[metric]),
            )
        self._append(self._h5["paths/metric_valid_bits"], packed.metric_valid_bits)

        self._append(self._h5["bounces/xyz_m"], packed.bounce_xyz_m)
        self._append(self._h5["bounces/interaction"], packed.interactions)
        self._append(self._h5["bounces/material_id"], prepared.material_ids)

        self._append(self._h5["targets/position_m"], packed.target_positions_m)
        target_metadata = self._h5["targets/metadata_json"]
        for metadata_json in prepared.target_metadata_json:
            self._append_scalar(target_metadata, metadata_json)

        self._append_fixed_sensing(fixed_sensing, prior_frame_count)
        self._append_cir(cir, prior_frame_count)

        self._frame_ids.append(frame_id)
        self._uncompressed_bytes += prepared.estimated_uncompressed_bytes
        self._h5.flush()

    def append(self, frame: StandardMPCFrame) -> None:
        """Prepare and append one complete canonical frame."""
        self.append_prepared(self.prepare(frame))

    def _sensing_layout_payload(self) -> dict[str, Any]:
        return {
            "fixed": {
                product: {
                    "shape": list(shape),
                    "dtype": dtype.str,
                }
                for product, (shape, dtype) in sorted(self._fixed_sensing_layout.items())
            },
            "cir": {
                "storage_dtype": None if self._cir_dtype is None else self._cir_dtype.str,
                "rank": self._cir_rank,
            },
            "static_config": self._sensing_config_json,
        }

    def _ensure_empty_cir_values_dataset(self) -> None:
        group = self._h5["sensing/ragged/cir"]
        if "values" in group:
            return
        group.create_dataset(
            "values",
            shape=(0,),
            maxshape=(None,),
            chunks=_numeric_chunk_shape(np.complex64),
            dtype=np.complex64,
            **self._numeric_options(),
        )

    def finalize_to_range_name(self) -> FrameChunkManifest:
        """Close and publish the private file under its inclusive frame range."""
        self._require_open()
        if not self._frame_ids:
            raise ValueError("Cannot finalize an empty HDF5 chunk")

        start_frame = self._frame_ids[0]
        end_frame = self._frame_ids[-1]
        destination = self._frames_dir / (f"mpc_frames_{start_frame:05d}-{end_frame:05d}.h5")
        if destination.exists():
            raise FileExistsError(destination)

        # Preserve the physical schema even for chunks in which CIR is absent.
        # The empty dtype carries no scientific values and readers use
        # ``present``/``offsets`` to distinguish that case.
        self._ensure_empty_cir_values_dataset()
        self._h5.attrs["publication_state"] = "complete"
        self._h5.attrs["num_frames"] = self.frame_count
        self._h5.attrs["start_frame"] = start_frame
        self._h5.attrs["end_frame"] = end_frame
        self._h5.attrs["topology_id"] = self.topology_id
        self._h5.attrs["sensing_layout_id"] = self.sensing_layout_id
        self._h5.flush()
        self._h5.close()
        self._closed = True
        os.rename(self._partial_path, destination)
        self._finalized = True
        return FrameChunkManifest(
            file=destination.name,
            frame_ids=self.frame_ids,
            size_bytes=destination.stat().st_size,
            uncompressed_bytes=self.uncompressed_bytes,
            topology_id=self.topology_id,
            sensing_layout_id=self.sensing_layout_id,
        )

    def close(self) -> None:
        """Close the private file without publishing it."""
        if self._closed:
            return
        self._h5.close()
        self._closed = True

    def discard(self) -> None:
        """Close and remove an unpublished private file."""
        if self._finalized:
            raise RuntimeError("A finalized chunk cannot be discarded")
        self.close()
        if self._partial_path.exists():
            self._partial_path.unlink()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def write_packed_mpc_frame_chunk(
    frames_dir: str | Path,
    frames: Iterable[StandardMPCFrame],
    *,
    generation_id: str,
    compression: str | None = "lzf",
) -> FrameChunkManifest:
    """Finalize one packed chunk without publishing a complete frame set.

    This low-level assembly and test helper returns the manifest entry for the
    chunk. A caller creating a consumable frame set must combine its chunk
    entries into a :class:`~shared.frames.manifest.FrameSetManifest` and publish
    ``frames_manifest.json`` separately. Complete frame-set publishers should
    use :class:`~shared.frames.frame_set_writer.FrameSetWriter` instead.
    """
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id=generation_id,
        compression=compression,
    )
    try:
        for frame in frames:
            writer.append(frame)
        return writer.finalize_to_range_name()
    except Exception:
        writer.discard()
        raise


__all__ = [
    "PackedMPCChunkBoundaryError",
    "PackedMPCChunkWriter",
    "PreparedPackedMPCFrame",
    "estimate_prepared_frame_bytes",
    "estimate_packed_frame_bytes",
]
