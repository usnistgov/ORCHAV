"""Logical read contracts and independently versioned frame boundaries.

The durable HDF5 layout, frame-set manifest, physical packed payload, and
complete :class:`~shared.frames.types.StandardMPCFrame` contract can evolve at
different times. Their separate identifiers let each boundary advertise the
exact contract it implements.

``FrameReadRequest`` describes data by meaning rather than by HDF5 dataset
name. Readers are responsible for mapping these logical groups to their
physical representation. The request is immutable and normalized to include
the dependencies needed to interpret the requested data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Iterable

MPC_HDF5_SCHEMA_VERSION = 2
MPC_HDF5_LAYOUT = "packed_ragged_v2"
MPC_FRAME_MANIFEST_VERSION = 2
PACKED_MPC_FRAME_VERSION = 1


class FrameComponent(StrEnum):
    """Logical groups that a frame consumer can load independently."""

    DEVICES = "devices"
    TARGETS = "targets"
    PATH_TOPOLOGY = "path_topology"
    PATH_BOUNCE_TOPOLOGY = "path_bounce_topology"
    PATH_GEOMETRY = "path_geometry"
    PATH_INTERACTIONS = "path_interactions"
    PATH_MATERIALS = "path_materials"
    PATH_METRICS = "path_metrics"
    SENSING = "sensing"
    BEAMFORMING = "beamforming"
    PROVENANCE = "provenance"


class PathMetric(StrEnum):
    """One-value-per-path channel metrics exposed by a canonical frame."""

    DELAY_NS = "delay_ns"
    PATH_LOSS_DB = "path_loss_db"
    AOA_AZ_DEG = "aoa_az_deg"
    AOA_EL_DEG = "aoa_el_deg"
    AOD_AZ_DEG = "aod_az_deg"
    AOD_EL_DEG = "aod_el_deg"


PATH_METRIC_ORDER = tuple(PathMetric)
"""Stable bit and storage order for packed path metrics."""

PATH_METRIC_VALIDITY_BITS = MappingProxyType(
    {metric: 1 << index for index, metric in enumerate(PATH_METRIC_ORDER)}
)
"""Bit assigned to each metric in the per-path validity byte."""


_COMPONENT_DEPENDENCIES: dict[FrameComponent, frozenset[FrameComponent]] = {
    FrameComponent.PATH_BOUNCE_TOPOLOGY: frozenset({FrameComponent.PATH_TOPOLOGY}),
    FrameComponent.PATH_GEOMETRY: frozenset(
        {FrameComponent.DEVICES, FrameComponent.PATH_BOUNCE_TOPOLOGY}
    ),
    FrameComponent.PATH_INTERACTIONS: frozenset({FrameComponent.PATH_BOUNCE_TOPOLOGY}),
    FrameComponent.PATH_MATERIALS: frozenset({FrameComponent.PATH_BOUNCE_TOPOLOGY}),
    FrameComponent.PATH_METRICS: frozenset({FrameComponent.PATH_TOPOLOGY}),
}


def _coerce_components(
    values: Iterable[FrameComponent | str],
) -> frozenset[FrameComponent]:
    try:
        return frozenset(FrameComponent(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"Unknown frame component: {exc}") from exc


def _coerce_metrics(values: Iterable[PathMetric | str]) -> frozenset[PathMetric]:
    try:
        return frozenset(PathMetric(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"Unknown path metric: {exc}") from exc


def _component_closure(
    components: frozenset[FrameComponent],
) -> frozenset[FrameComponent]:
    expanded = set(components)
    pending = list(components)
    while pending:
        component = pending.pop()
        for dependency in _COMPONENT_DEPENDENCIES.get(component, ()):
            if dependency not in expanded:
                expanded.add(dependency)
                pending.append(dependency)
    return frozenset(expanded)


@dataclass(frozen=True, slots=True)
class FrameReadRequest:
    """Immutable logical selection for one or more frame reads.

    ``components`` and ``metrics`` accept enum instances or their stable string
    values. Construction canonicalizes them to enum-valued ``frozenset``
    instances and adds interpretation dependencies:

    * bounce topology requires pair-to-path topology;
    * geometry requires device endpoints and bounce topology;
    * interactions and materials require bounce topology;
    * any metric requires the path-metrics component and path topology.

    Requesting ``PATH_METRICS`` without naming metrics means all metrics.
    Requesting ``SENSING`` without naming products means every available
    sensing product. An empty request is valid and represents frame identity
    only, which is useful for manifest and metadata inspection.
    """

    components: frozenset[FrameComponent] = frozenset()
    metrics: frozenset[PathMetric] = frozenset()
    sensing_products: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        raw_components = _coerce_components(self.components)
        metrics = _coerce_metrics(self.metrics)
        products = frozenset(self.sensing_products)

        if any(not isinstance(product, str) or not product.strip() for product in products):
            raise ValueError("Sensing product names must be non-empty strings")

        components = set(raw_components)
        if metrics:
            components.add(FrameComponent.PATH_METRICS)
        elif FrameComponent.PATH_METRICS in raw_components:
            metrics = frozenset(PATH_METRIC_ORDER)

        if products:
            components.add(FrameComponent.SENSING)

        object.__setattr__(
            self,
            "components",
            _component_closure(frozenset(components)),
        )
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "sensing_products", products)

    @classmethod
    def full(cls) -> "FrameReadRequest":
        """Return a request for every logical frame component and metric."""

        return cls(components=frozenset(FrameComponent))

    @classmethod
    def for_metrics(
        cls,
        metrics: Iterable[PathMetric | str] = PATH_METRIC_ORDER,
        *,
        include_devices: bool = False,
        include_interactions: bool = False,
    ) -> "FrameReadRequest":
        """Build a metric request with common statistics dependencies."""

        components: set[FrameComponent] = set()
        if include_devices:
            components.add(FrameComponent.DEVICES)
        if include_interactions:
            components.add(FrameComponent.PATH_INTERACTIONS)
        return cls(
            components=frozenset(components),
            metrics=frozenset(metrics),
        )

    @property
    def all_sensing_products(self) -> bool:
        """Whether the request asks for sensing without a product restriction."""

        return FrameComponent.SENSING in self.components and not self.sensing_products

    def includes_component(self, component: FrameComponent | str) -> bool:
        """Return whether a logical component is part of this request."""

        return FrameComponent(component) in self.components

    def includes_metric(self, metric: PathMetric | str) -> bool:
        """Return whether a path metric is part of this request."""

        return PathMetric(metric) in self.metrics

    def union(self, other: "FrameReadRequest") -> "FrameReadRequest":
        """Return the canonical union of two requests.

        An unrestricted sensing selection dominates a product-restricted
        selection.
        """

        sensing_requested = (
            FrameComponent.SENSING in self.components or FrameComponent.SENSING in other.components
        )
        all_sensing = (FrameComponent.SENSING in self.components and self.all_sensing_products) or (
            FrameComponent.SENSING in other.components and other.all_sensing_products
        )
        products = (
            frozenset()
            if sensing_requested and all_sensing
            else self.sensing_products | other.sensing_products
        )
        return FrameReadRequest(
            components=self.components | other.components,
            metrics=self.metrics | other.metrics,
            sensing_products=products,
        )


__all__ = [
    "FrameComponent",
    "FrameReadRequest",
    "MPC_FRAME_MANIFEST_VERSION",
    "MPC_HDF5_LAYOUT",
    "MPC_HDF5_SCHEMA_VERSION",
    "PACKED_MPC_FRAME_VERSION",
    "PATH_METRIC_ORDER",
    "PATH_METRIC_VALIDITY_BITS",
    "PathMetric",
]
