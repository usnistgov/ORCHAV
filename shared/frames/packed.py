"""Selective in-memory projections of canonical MPC frames.

``StandardMPCFrame`` is always complete. ``ProjectedMPCFrame`` is deliberately
different: it may contain only the arrays named by a ``FrameProjection``
inventory. This prevents a selective HDF5 read from being mistaken for a
complete frame that can be published or transported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import (
    PATH_METRIC_VALIDITY_BITS,
    FrameComponent,
    FrameReadRequest,
    PathMetric,
    _coerce_components,
    _coerce_metrics,
    _component_closure,
)
from .types import (
    FRAME_FORMAT_VERSION,
    PATH_METRIC_ARRAY_FIELDS,
    Float32Array,
    Float64Array,
    Int32Array,
    Int64Array,
    UInt8Array,
    UInt16Array,
    _CompactFrameProperties,
    _validate_integer_vector,
    _validate_offsets,
    _validate_pairs,
    _validate_xyz,
)


@dataclass(frozen=True, slots=True, eq=False)
class ProjectedMPCFrame(_CompactFrameProperties):
    """The compact arrays resident for one selective frame read.

    Every field except ``frame_index`` is optional because the enclosing
    :class:`FrameProjection` records exactly which logical components were
    loaded. Present arrays still obey the canonical v2 dtypes and every
    relationship observable from those arrays is validated.
    """

    frame_index: int
    version: int = FRAME_FORMAT_VERSION
    timestamp_s: float | None = None

    tx_rx_pairs: Int32Array | None = None
    pair_path_offsets: Int64Array | None = None
    bounce_offsets: Int64Array | None = None

    tx_positions: Float64Array | None = None
    rx_positions: Float64Array | None = None
    tx_orientations: Float64Array | None = None
    rx_orientations: Float64Array | None = None
    tx_names: tuple[str, ...] | None = None
    rx_names: tuple[str, ...] | None = None

    bounce_xyz_m: Float32Array | None = None
    interactions: UInt8Array | None = None
    material_ids: UInt16Array | None = None
    material_names: tuple[str, ...] | None = None
    material_itu_types: tuple[str, ...] | None = None

    delays_ns: Float32Array | None = None
    path_loss_db: Float32Array | None = None
    aoa_az_deg: Float32Array | None = None
    aoa_el_deg: Float32Array | None = None
    aod_az_deg: Float32Array | None = None
    aod_el_deg: Float32Array | None = None
    metric_valid_bits: UInt8Array | None = None

    target_positions_m: Float64Array | None = None
    targets_metadata: tuple[Mapping[str, Any], ...] | None = None
    sensing: Mapping[str, Any] | None = None
    beamforming: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None
    recomputed_from_stored_positions: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_tx(self) -> int | None:
        """Return the transmitter count when device state is resident."""

        return None if self.tx_positions is None else int(len(self.tx_positions))

    @property
    def num_rx(self) -> int | None:
        """Return the receiver count when device state is resident."""

        return None if self.rx_positions is None else int(len(self.rx_positions))

    @property
    def num_targets(self) -> int | None:
        """Return the target count when target state is resident."""

        return None if self.target_positions_m is None else int(len(self.target_positions_m))

    def validate(self) -> None:
        """Validate all compact relationships observable in this projection."""

        if (
            isinstance(self.frame_index, (bool, np.bool_))
            or not isinstance(self.frame_index, (int, np.integer))
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if self.version != FRAME_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported projected frame version {self.version}; "
                f"expected {FRAME_FORMAT_VERSION}"
            )
        if self.timestamp_s is not None and not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite when present")

        if self.tx_rx_pairs is not None:
            _validate_pairs(self.tx_rx_pairs)
        if self.pair_path_offsets is not None:
            _validate_offsets("pair_path_offsets", self.pair_path_offsets)
            if (
                self.tx_rx_pairs is not None
                and self.pair_path_offsets.size != len(self.tx_rx_pairs) + 1
            ):
                raise ValueError("pair_path_offsets length must equal the pair count plus one")
        if self.bounce_offsets is not None:
            _validate_offsets("bounce_offsets", self.bounce_offsets)
            if (
                self.pair_path_offsets is not None
                and self.bounce_offsets.size != int(self.pair_path_offsets[-1]) + 1
            ):
                raise ValueError("bounce_offsets length must equal the path count plus one")

        self._validate_device_axis("tx")
        self._validate_device_axis("rx")
        if self.tx_rx_pairs is not None and self.tx_rx_pairs.size:
            if self.tx_positions is not None and int(np.max(self.tx_rx_pairs[:, 0])) >= len(
                self.tx_positions
            ):
                raise ValueError("tx_rx_pairs references an unknown transmitter")
            if self.rx_positions is not None and int(np.max(self.tx_rx_pairs[:, 1])) >= len(
                self.rx_positions
            ):
                raise ValueError("tx_rx_pairs references an unknown receiver")

        if self.bounce_xyz_m is not None:
            _validate_xyz("bounce_xyz_m", self.bounce_xyz_m, np.float32)
        if self.interactions is not None:
            _validate_integer_vector("interactions", self.interactions, np.uint8)
            if np.any(self.interactions == 0):
                raise ValueError("interactions must contain positive physical-bounce codes")
        if self.material_ids is not None:
            _validate_integer_vector("material_ids", self.material_ids, np.uint16)

        num_bounces = self.num_bounces
        for name, values in (
            ("bounce_xyz_m", self.bounce_xyz_m),
            ("interactions", self.interactions),
            ("material_ids", self.material_ids),
        ):
            if values is not None and num_bounces is not None and len(values) != num_bounces:
                raise ValueError(
                    f"{name} length {len(values)} does not match " f"the bounce count {num_bounces}"
                )

        self._validate_material_catalog()
        self._validate_metrics()
        self._validate_targets()
        for name, value in (
            ("sensing", self.sensing),
            ("beamforming", self.beamforming),
            ("provenance", self.provenance),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping or None")

    def _validate_device_axis(self, prefix: str) -> None:
        positions = getattr(self, f"{prefix}_positions")
        orientations = getattr(self, f"{prefix}_orientations")
        names = getattr(self, f"{prefix}_names")
        if positions is not None:
            _validate_xyz(f"{prefix}_positions", positions, np.float64)
        if orientations is not None:
            _validate_xyz(f"{prefix}_orientations", orientations, np.float64)
        expected = None if positions is None else len(positions)
        if orientations is not None:
            if expected is None:
                expected = len(orientations)
            elif len(orientations) != expected:
                raise ValueError(f"{prefix}_orientations count must match {prefix}_positions")
        if names is not None:
            if not isinstance(names, tuple) or any(not isinstance(name, str) for name in names):
                raise ValueError(f"{prefix}_names must be a tuple of strings")
            if expected is not None and len(names) != expected:
                raise ValueError(f"{prefix}_names count must match {prefix} device arrays")

    def _validate_material_catalog(self) -> None:
        if self.material_names is not None:
            if not isinstance(self.material_names, tuple) or any(
                not isinstance(value, str) for value in self.material_names
            ):
                raise ValueError("material_names must be a tuple of strings")
            if not self.material_names or self.material_names[0] != "":
                raise ValueError("material_names index zero must be the no-material empty string")
        if self.material_itu_types is not None:
            if not isinstance(self.material_itu_types, tuple) or any(
                not isinstance(value, str) for value in self.material_itu_types
            ):
                raise ValueError("material_itu_types must be a tuple of strings")
            if not self.material_itu_types or self.material_itu_types[0] != "":
                raise ValueError(
                    "material_itu_types index zero must be the no-material empty string"
                )
        if self.material_names is not None and self.material_itu_types is not None:
            if len(self.material_names) != len(self.material_itu_types):
                raise ValueError("material_names and material_itu_types must have equal lengths")
        if (
            self.material_ids is not None
            and self.material_names is not None
            and self.material_ids.size
            and int(np.max(self.material_ids)) >= len(self.material_names)
        ):
            raise ValueError("material_ids references an unknown material")

    def _validate_metrics(self) -> None:
        num_paths = self.num_paths
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
            if num_paths is not None and len(values) != num_paths:
                raise ValueError(
                    f"Path metric {metric.value} length {len(values)} does not "
                    f"match the path count {num_paths}"
                )
        if self.metric_valid_bits is None:
            if self.path_metrics:
                raise ValueError("metric_valid_bits is required when metrics are resident")
            return

        _validate_integer_vector("metric_valid_bits", self.metric_valid_bits, np.uint8)
        if num_paths is not None and len(self.metric_valid_bits) != num_paths:
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

    def _validate_targets(self) -> None:
        if self.target_positions_m is not None:
            _validate_xyz("target_positions_m", self.target_positions_m, np.float64)
        if self.targets_metadata is None:
            return
        if not isinstance(self.targets_metadata, tuple) or any(
            not isinstance(item, Mapping) for item in self.targets_metadata
        ):
            raise ValueError("targets_metadata must be a tuple of mappings")
        if self.target_positions_m is None:
            raise ValueError("target_positions_m is required with targets_metadata")
        if len(self.targets_metadata) != len(self.target_positions_m):
            raise ValueError("targets_metadata length must match target_positions_m")


@dataclass(frozen=True, slots=True, eq=False)
class FrameProjection:
    """A partial frame plus an exact inventory of fields resident in memory."""

    frame: ProjectedMPCFrame
    loaded_components: frozenset[FrameComponent] = frozenset()
    loaded_path_metrics: frozenset[PathMetric] = frozenset()
    loaded_sensing_products: frozenset[str] = frozenset()
    all_sensing_products_loaded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ProjectedMPCFrame):
            raise TypeError("FrameProjection.frame must be a ProjectedMPCFrame")
        components = _coerce_components(self.loaded_components)
        metrics = _coerce_metrics(self.loaded_path_metrics)
        products = frozenset(self.loaded_sensing_products)
        if any(not isinstance(product, str) or not product.strip() for product in products):
            raise ValueError("Loaded sensing product names must be non-empty strings")
        if metrics and FrameComponent.PATH_METRICS not in components:
            raise ValueError("loaded_path_metrics requires the PATH_METRICS loaded component")
        if FrameComponent.PATH_METRICS in components and not metrics:
            raise ValueError("PATH_METRICS requires at least one loaded_path_metrics entry")
        missing_dependencies = _component_closure(components) - components
        if missing_dependencies:
            names = ", ".join(sorted(component.value for component in missing_dependencies))
            raise ValueError(f"Loaded components are missing interpretation dependencies: {names}")
        if products and FrameComponent.SENSING not in components:
            raise ValueError("loaded_sensing_products requires the SENSING loaded component")
        if self.all_sensing_products_loaded and FrameComponent.SENSING not in components:
            raise ValueError("all_sensing_products_loaded requires the SENSING component")
        if (
            FrameComponent.SENSING in components
            and not products
            and not self.all_sensing_products_loaded
        ):
            raise ValueError("A sensing projection must list loaded products or mark all loaded")
        if metrics != frozenset(self.frame.path_metrics):
            raise ValueError("loaded_path_metrics must exactly match frame.path_metrics")

        object.__setattr__(self, "loaded_components", components)
        object.__setattr__(self, "loaded_path_metrics", metrics)
        object.__setattr__(self, "loaded_sensing_products", products)
        self._validate_loaded_component_fields()

    @classmethod
    def from_request(
        cls,
        frame: ProjectedMPCFrame,
        request: FrameReadRequest,
        *,
        all_sensing_products_loaded: bool | None = None,
    ) -> "FrameProjection":
        """Create a projection whose inventory matches a normalized request."""

        return cls(
            frame=frame,
            loaded_components=request.components,
            loaded_path_metrics=request.metrics,
            loaded_sensing_products=request.sensing_products,
            all_sensing_products_loaded=(
                request.all_sensing_products
                if all_sensing_products_loaded is None
                else all_sensing_products_loaded
            ),
        )

    def satisfies(self, request: FrameReadRequest) -> bool:
        """Return whether this projection can answer ``request`` without I/O."""

        if not request.components.issubset(self.loaded_components):
            return False
        if not request.metrics.issubset(self.loaded_path_metrics):
            return False
        if FrameComponent.SENSING not in request.components:
            return True
        if request.all_sensing_products:
            return self.all_sensing_products_loaded
        return self.all_sensing_products_loaded or request.sensing_products.issubset(
            self.loaded_sensing_products
        )

    def _validate_loaded_component_fields(self) -> None:
        frame = self.frame
        required_by_component: dict[FrameComponent, tuple[str, ...]] = {
            FrameComponent.DEVICES: (
                "tx_positions",
                "rx_positions",
                "tx_orientations",
                "rx_orientations",
                "tx_names",
                "rx_names",
            ),
            FrameComponent.PATH_TOPOLOGY: (
                "tx_rx_pairs",
                "pair_path_offsets",
            ),
            FrameComponent.PATH_BOUNCE_TOPOLOGY: ("bounce_offsets",),
            FrameComponent.PATH_GEOMETRY: ("bounce_xyz_m",),
            FrameComponent.PATH_INTERACTIONS: ("interactions",),
            FrameComponent.PATH_MATERIALS: (
                "material_ids",
                "material_names",
                "material_itu_types",
            ),
            FrameComponent.TARGETS: (
                "target_positions_m",
                "targets_metadata",
            ),
        }
        for component in self.loaded_components:
            for field_name in required_by_component.get(component, ()):
                if getattr(frame, field_name) is None:
                    raise ValueError(
                        f"Loaded component {component.value!r} requires frame.{field_name}"
                    )
        if FrameComponent.PATH_METRICS in self.loaded_components:
            if frame.metric_valid_bits is None:
                raise ValueError("Loaded path metrics require frame.metric_valid_bits")
            missing = self.loaded_path_metrics - frozenset(frame.path_metrics)
            if missing:
                names = ", ".join(sorted(metric.value for metric in missing))
                raise ValueError(f"Loaded path metric arrays are missing: {names}")


__all__ = [
    "FrameProjection",
    "PATH_METRIC_ARRAY_FIELDS",
    "ProjectedMPCFrame",
]
