"""Canonical in-memory representation of one complete MPC frame.

``StandardMPCFrame`` is independent of ray tracers, storage formats, and
visualization code. Producers construct one complete compact frame, providers
return it, and consumers can inspect its pair/path/bounce relationships without
reconstructing padded arrays.

The dataclass is structurally immutable and never copies NumPy arrays supplied
to its direct constructor. Callers must therefore treat the underlying buffers
as immutable after construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .contracts import PATH_METRIC_VALIDITY_BITS, PathMetric

# Increment when the complete in-memory frame contract changes incompatibly.
FRAME_FORMAT_VERSION = 2

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Int32Array: TypeAlias = NDArray[np.int32]
Int64Array: TypeAlias = NDArray[np.int64]
UInt8Array: TypeAlias = NDArray[np.uint8]
UInt16Array: TypeAlias = NDArray[np.uint16]

PATH_METRIC_ARRAY_FIELDS = MappingProxyType(
    {
        PathMetric.DELAY_NS: "delays_ns",
        PathMetric.PATH_LOSS_DB: "path_loss_db",
        PathMetric.AOA_AZ_DEG: "aoa_az_deg",
        PathMetric.AOA_EL_DEG: "aoa_el_deg",
        PathMetric.AOD_AZ_DEG: "aod_az_deg",
        PathMetric.AOD_EL_DEG: "aod_el_deg",
    }
)
"""Canonical attribute containing each physical path-metric vector."""


def _validate_integer_vector(
    name: str,
    values: np.ndarray,
    dtype: np.dtype[Any] | type[np.generic],
) -> None:
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {values.shape}")
    expected = np.dtype(dtype)
    if values.dtype != expected:
        raise ValueError(f"{name} must use dtype {expected}, got {values.dtype}")


def _validate_offsets(name: str, values: np.ndarray) -> None:
    _validate_integer_vector(name, values, np.int64)
    if values.size == 0:
        raise ValueError(f"{name} must contain at least the initial zero")
    if int(values[0]) != 0:
        raise ValueError(f"{name} must start at zero")
    if np.any(values < 0):
        raise ValueError(f"{name} cannot contain negative offsets")
    if np.any(values[1:] < values[:-1]):
        raise ValueError(f"{name} must be monotonically non-decreasing")


def _validate_xyz(
    name: str,
    values: np.ndarray,
    dtype: np.dtype[Any] | type[np.generic],
) -> None:
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {values.shape}")
    expected = np.dtype(dtype)
    if values.dtype != expected:
        raise ValueError(f"{name} must use dtype {expected}, got {values.dtype}")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite coordinates")


def _validate_pairs(values: np.ndarray) -> None:
    if not isinstance(values, np.ndarray):
        raise ValueError("tx_rx_pairs must be a numpy array")
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"tx_rx_pairs must have shape (N, 2), got {values.shape}")
    if values.dtype != np.dtype(np.int32):
        raise ValueError(f"tx_rx_pairs must use dtype int32, got {values.dtype}")
    if np.any(values < 0):
        raise ValueError("tx_rx_pairs cannot contain negative indices")


def _validate_device_axis(
    prefix: str,
    positions: np.ndarray,
    orientations: np.ndarray,
    names: tuple[str, ...],
) -> None:
    _validate_xyz(f"{prefix}_positions", positions, np.float64)
    _validate_xyz(f"{prefix}_orientations", orientations, np.float64)
    if len(orientations) != len(positions):
        raise ValueError(f"{prefix}_orientations count must match {prefix}_positions")
    if not isinstance(names, tuple) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{prefix}_names must be a tuple of strings")
    if len(names) != len(positions):
        raise ValueError(f"{prefix}_names count must match {prefix}_positions")


class _CompactFrameProperties:
    """Shared read-only helpers for complete frames and partial projections."""

    __slots__ = ()

    _path_fields: ClassVar[Mapping[PathMetric, str]] = PATH_METRIC_ARRAY_FIELDS

    @property
    def path_metrics(self) -> Mapping[PathMetric, Float32Array]:
        """Return resident metric vectors keyed by their logical metric."""

        return MappingProxyType(
            {
                metric: values
                for metric, field_name in self._path_fields.items()
                if (values := getattr(self, field_name)) is not None
            }
        )

    @property
    def num_pairs(self) -> int | None:
        """Return the pair count when pair topology is resident."""

        pairs = getattr(self, "tx_rx_pairs")
        if pairs is not None:
            return int(pairs.shape[0])
        offsets = getattr(self, "pair_path_offsets")
        return None if offsets is None else int(offsets.size - 1)

    @property
    def num_paths(self) -> int | None:
        """Return the path count when any path-aligned array is resident."""

        pair_offsets = getattr(self, "pair_path_offsets")
        if pair_offsets is not None:
            return int(pair_offsets[-1])
        bounce_offsets = getattr(self, "bounce_offsets")
        if bounce_offsets is not None:
            return int(bounce_offsets.size - 1)
        metrics = self.path_metrics
        return None if not metrics else int(len(next(iter(metrics.values()))))

    @property
    def num_bounces(self) -> int | None:
        """Return the physical-bounce count when bounce data is resident."""

        offsets = getattr(self, "bounce_offsets")
        if offsets is not None:
            return int(offsets[-1])
        for field_name in ("bounce_xyz_m", "interactions", "material_ids"):
            values = getattr(self, field_name)
            if values is not None:
                return int(len(values))
        return None

    def metric_is_valid(self, metric: PathMetric | str) -> NDArray[np.bool_]:
        """Return the per-path validity mask for a resident metric."""

        normalized = PathMetric(metric)
        if normalized not in self.path_metrics:
            raise KeyError(f"Metric {normalized.value!r} is not resident")
        bits = getattr(self, "metric_valid_bits")
        if bits is None:
            raise ValueError("metric_valid_bits is required for resident metrics")
        return (bits & PATH_METRIC_VALIDITY_BITS[normalized]) != 0


@dataclass(frozen=True, slots=True, eq=False)
class StandardMPCFrame(_CompactFrameProperties):
    """One complete canonical MPC frame in compact ragged form.

    All numeric core fields are required, including empty arrays for an empty
    frame and NaN-filled metric vectors for unavailable metrics. The direct
    constructor accepts only canonical dtypes and performs no normalization or
    hidden array copies. Use :func:`standard_mpc_frame_from_pair_data` for
    familiar per-pair padded or ragged source arrays.

    ``tx_rx_pairs`` has shape ``(pair, 2)`` and indexes the transmitter and
    receiver axes. Pair row ``p`` owns paths
    ``pair_path_offsets[p]:pair_path_offsets[p + 1]``; path ``q`` owns physical
    bounces ``bounce_offsets[q]:bounce_offsets[q + 1]``. Direct paths therefore
    have zero bounces, while ``bounce_xyz_m``, ``interactions``, and
    ``material_ids`` align on the bounce axis.

    Device positions and orientations have shape ``(device, 3)``; target
    positions have shape ``(target, 3)``; and bounce positions have shape
    ``(bounce, 3)``. Positions are in meters and orientations are
    ``(yaw, pitch, roll)`` in radians. Every metric vector aligns with the path
    axis; an unavailable value is NaN with the corresponding
    ``metric_valid_bits`` flag clear. Material catalog row zero is the empty
    no-material sentinel. Target positions and metadata are row-aligned, and a
    metadata ``current_position`` must equal the canonical row when present.
    """

    frame_index: int

    tx_rx_pairs: Int32Array
    pair_path_offsets: Int64Array
    bounce_offsets: Int64Array

    tx_positions: Float64Array
    rx_positions: Float64Array
    tx_orientations: Float64Array
    rx_orientations: Float64Array
    tx_names: tuple[str, ...]
    rx_names: tuple[str, ...]

    bounce_xyz_m: Float32Array
    interactions: UInt8Array
    material_ids: UInt16Array
    material_names: tuple[str, ...]
    material_itu_types: tuple[str, ...]

    delays_ns: Float32Array
    path_loss_db: Float32Array
    aoa_az_deg: Float32Array
    aoa_el_deg: Float32Array
    aod_az_deg: Float32Array
    aod_el_deg: Float32Array
    metric_valid_bits: UInt8Array

    target_positions_m: Float64Array
    targets_metadata: tuple[Mapping[str, Any], ...]

    version: int = FRAME_FORMAT_VERSION
    timestamp_s: float | None = None
    sensing: Mapping[str, Any] | None = None
    beamforming: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None
    recomputed_from_stored_positions: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_tx(self) -> int:
        """Number of transmitters in this frame."""

        return int(len(self.tx_positions))

    @property
    def num_rx(self) -> int:
        """Number of receivers in this frame."""

        return int(len(self.rx_positions))

    @property
    def num_targets(self) -> int:
        """Number of targets in this frame."""

        return int(len(self.target_positions_m))

    def validate(self) -> None:
        """Validate the complete compact-frame contract."""

        if (
            isinstance(self.frame_index, (bool, np.bool_))
            or not isinstance(self.frame_index, (int, np.integer))
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if self.version != FRAME_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported StandardMPCFrame version {self.version}; "
                f"expected {FRAME_FORMAT_VERSION}"
            )
        if self.timestamp_s is not None and not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite when present")

        _validate_pairs(self.tx_rx_pairs)
        _validate_offsets("pair_path_offsets", self.pair_path_offsets)
        _validate_offsets("bounce_offsets", self.bounce_offsets)
        if self.pair_path_offsets.size != len(self.tx_rx_pairs) + 1:
            raise ValueError("pair_path_offsets length must equal the pair count plus one")
        num_paths = int(self.pair_path_offsets[-1])
        if self.bounce_offsets.size != num_paths + 1:
            raise ValueError("bounce_offsets length must equal the path count plus one")

        _validate_device_axis("tx", self.tx_positions, self.tx_orientations, self.tx_names)
        _validate_device_axis("rx", self.rx_positions, self.rx_orientations, self.rx_names)
        if self.tx_rx_pairs.size:
            if int(np.max(self.tx_rx_pairs[:, 0])) >= self.num_tx:
                raise ValueError("tx_rx_pairs references an unknown transmitter")
            if int(np.max(self.tx_rx_pairs[:, 1])) >= self.num_rx:
                raise ValueError("tx_rx_pairs references an unknown receiver")

        _validate_xyz("bounce_xyz_m", self.bounce_xyz_m, np.float32)
        _validate_integer_vector("interactions", self.interactions, np.uint8)
        _validate_integer_vector("material_ids", self.material_ids, np.uint16)
        num_bounces = int(self.bounce_offsets[-1])
        for name, values in (
            ("bounce_xyz_m", self.bounce_xyz_m),
            ("interactions", self.interactions),
            ("material_ids", self.material_ids),
        ):
            if len(values) != num_bounces:
                raise ValueError(
                    f"{name} length {len(values)} does not match " f"the bounce count {num_bounces}"
                )
        if np.any(self.interactions == 0):
            raise ValueError("interactions must contain positive physical-bounce codes")

        self._validate_material_catalog()
        self._validate_metrics(num_paths)

        _validate_xyz("target_positions_m", self.target_positions_m, np.float64)
        if not isinstance(self.targets_metadata, tuple):
            raise ValueError("targets_metadata must be a tuple of mappings")
        if len(self.targets_metadata) != self.num_targets:
            raise ValueError("targets_metadata length must match target_positions_m")
        if any(not isinstance(item, Mapping) for item in self.targets_metadata):
            raise ValueError("Each targets_metadata item must be a mapping")
        for target_index, metadata in enumerate(self.targets_metadata):
            current_position = metadata.get("current_position")
            if current_position is None:
                continue
            try:
                metadata_position = np.asarray(current_position, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"targets_metadata[{target_index}].current_position must contain "
                    "three numeric coordinates"
                ) from exc
            if metadata_position.shape != (3,) or np.any(~np.isfinite(metadata_position)):
                raise ValueError(
                    f"targets_metadata[{target_index}].current_position must have "
                    "shape (3,) with finite coordinates"
                )
            if not np.array_equal(
                metadata_position,
                self.target_positions_m[target_index],
            ):
                raise ValueError(
                    f"targets_metadata[{target_index}].current_position must match "
                    "target_positions_m"
                )

        for name, value in (
            ("sensing", self.sensing),
            ("beamforming", self.beamforming),
            ("provenance", self.provenance),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping or None")
        if not isinstance(self.recomputed_from_stored_positions, (bool, np.bool_)):
            raise ValueError("recomputed_from_stored_positions must be boolean")

    def _validate_material_catalog(self) -> None:
        if not isinstance(self.material_names, tuple) or any(
            not isinstance(value, str) for value in self.material_names
        ):
            raise ValueError("material_names must be a tuple of strings")
        if not isinstance(self.material_itu_types, tuple) or any(
            not isinstance(value, str) for value in self.material_itu_types
        ):
            raise ValueError("material_itu_types must be a tuple of strings")
        if len(self.material_names) != len(self.material_itu_types):
            raise ValueError("material_names and material_itu_types must have equal lengths")
        if not self.material_names or (self.material_names[0], self.material_itu_types[0]) != (
            "",
            "",
        ):
            raise ValueError("Material catalog row zero must be the no-material empty row")
        if len(self.material_names) > np.iinfo(np.uint16).max + 1:
            raise ValueError("Material catalog exceeds the uint16 material ID range")
        catalog = tuple(zip(self.material_names, self.material_itu_types, strict=True))
        if len(catalog) != len(set(catalog)):
            raise ValueError("Material catalog cannot contain duplicate rows")
        if self.material_ids.size and int(np.max(self.material_ids)) >= len(catalog):
            raise ValueError("material_ids references an unknown material")

    def _validate_metrics(self, num_paths: int) -> None:
        for metric, values in self.path_metrics.items():
            if not isinstance(values, np.ndarray):
                raise ValueError(f"Path metric {metric.value} must be a numpy array")
            if values.ndim != 1:
                raise ValueError(
                    f"Path metric {metric.value} must be one-dimensional, "
                    f"got shape {values.shape}"
                )
            if values.dtype != np.dtype(np.float32):
                raise ValueError(
                    f"Path metric {metric.value} must use dtype float32, got {values.dtype}"
                )
            if len(values) != num_paths:
                raise ValueError(
                    f"Path metric {metric.value} length {len(values)} does not "
                    f"match the path count {num_paths}"
                )

        _validate_integer_vector("metric_valid_bits", self.metric_valid_bits, np.uint8)
        if len(self.metric_valid_bits) != num_paths:
            raise ValueError("metric_valid_bits length must match the path count")
        known_bits = sum(PATH_METRIC_VALIDITY_BITS.values())
        unknown_bits = np.uint8(0xFF ^ known_bits)
        if np.any(np.bitwise_and(self.metric_valid_bits, unknown_bits) != 0):
            raise ValueError("metric_valid_bits contains unknown metric bits")

        for metric, values in self.path_metrics.items():
            valid = self.metric_is_valid(metric)
            if np.any(~np.isnan(values[~valid])):
                raise ValueError(f"Invalid {metric.value} entries must contain NaN")
            if np.any(~np.isfinite(values[valid])):
                raise ValueError(f"Valid {metric.value} entries must be finite")


__all__ = [
    "FRAME_FORMAT_VERSION",
    "Float32Array",
    "Float64Array",
    "Int32Array",
    "Int64Array",
    "PATH_METRIC_ARRAY_FIELDS",
    "StandardMPCFrame",
    "UInt8Array",
    "UInt16Array",
]
