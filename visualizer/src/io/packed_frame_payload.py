"""Provider-neutral projection loading for normal visualizer playback.

The visualizer consumes one compact payload regardless of where a frame came
from. Providers with selective storage can satisfy the request without reading
unrelated arrays; other providers use :class:`DataProvider`'s complete-frame
fallback and are projected in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.frames import (
    PATH_METRIC_ORDER,
    FrameComponent,
    FrameProjection,
    FrameReadRequest,
    PathMetric,
    StandardMPCFrame,
    project_standard_mpc_frame,
)
from shared.frames.provider_base import DataProvider

from ..metrics.packed_canon import canonical_from_projection

_BASE_VISUAL_COMPONENTS = frozenset(
    {
        FrameComponent.DEVICES,
        FrameComponent.TARGETS,
        FrameComponent.PATH_TOPOLOGY,
        FrameComponent.PATH_BOUNCE_TOPOLOGY,
        FrameComponent.PATH_GEOMETRY,
        FrameComponent.PATH_INTERACTIONS,
        # Material IDs/catalogs are small relative to geometry and are needed
        # to populate filter choices before a user selects a material mode.
        FrameComponent.PATH_MATERIALS,
        FrameComponent.PROVENANCE,
        # Beamforming metadata is compact and keeps the Antennas panel ready
        # while large sensing products remain unloaded.
        FrameComponent.BEAMFORMING,
    }
)

PACKED_PROJECTION_KEY = "_packed_frame_projection"
"""Private payload key retaining exact loaded-component inventory."""

PACKED_SENSING_PROJECTION_KEY = "_packed_sensing_projection"
"""Private payload key retaining independently loaded sensing-field inventory."""


@dataclass(frozen=True, slots=True)
class _ProjectionInventory:
    """Field inventory retained after projected arrays become canonical arrays."""

    loaded_components: frozenset[FrameComponent]
    loaded_path_metrics: frozenset[PathMetric]
    loaded_sensing_products: frozenset[str]
    all_sensing_products_loaded: bool

    @classmethod
    def from_projection(cls, projection: FrameProjection) -> "_ProjectionInventory":
        """Copy only the request-coverage metadata from a loaded projection."""
        return cls(
            loaded_components=projection.loaded_components,
            loaded_path_metrics=projection.loaded_path_metrics,
            loaded_sensing_products=projection.loaded_sensing_products,
            all_sensing_products_loaded=projection.all_sensing_products_loaded,
        )

    def satisfies(self, request: FrameReadRequest) -> bool:
        """Return whether the discarded projection had every requested field."""
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


def _projection_inventory(value: Any) -> _ProjectionInventory | None:
    """Return lightweight inventory from a cached payload value."""
    if isinstance(value, _ProjectionInventory):
        return value
    if isinstance(value, FrameProjection):
        return _ProjectionInventory.from_projection(value)
    return None


def visual_frame_read_request(
    *,
    include_sensing: bool = False,
    sensing_products: frozenset[str] | None = None,
) -> FrameReadRequest:
    """Return the logical columns needed by normal visual construction."""
    components = set(_BASE_VISUAL_COMPONENTS)
    if include_sensing:
        components.add(FrameComponent.SENSING)
    return FrameReadRequest(
        components=frozenset(components),
        metrics=frozenset(PATH_METRIC_ORDER),
        sensing_products=(
            frozenset() if not include_sensing or sensing_products is None else sensing_products
        ),
    )


def _sensing_dashboard_visible(viz: Any) -> bool:
    """Return whether the optional sensing analysis window is visible."""
    panel_manager = getattr(viz, "panel_manager", None)
    panels = getattr(panel_manager, "panels", None)
    panel = panels.get("sensing") if isinstance(panels, dict) else None
    if panel is None:
        panel = getattr(viz, "sensing_panel", None)
    window = getattr(panel, "_sensing_window", None)
    is_visible = getattr(window, "isVisible", None)
    try:
        return bool(callable(is_visible) and is_visible())
    except RuntimeError:
        return False


def visualizer_needs_sensing(viz: Any) -> bool:
    """Return whether the current scene actually draws sensing products."""
    if _sensing_dashboard_visible(viz):
        return True
    services = getattr(viz, "extension_services", None)
    if not isinstance(services, dict) or "sensing" not in services:
        return False
    state = getattr(viz, "app_state", None)
    namespaces = getattr(state, "extension_state", {}) or {}
    sensing_state = namespaces.get("sensing", {}) if isinstance(namespaces, dict) else {}
    return bool(isinstance(sensing_state, dict) and sensing_state.get("show_detections", False))


def visual_frame_read_request_for_visualizer(viz: Any) -> FrameReadRequest:
    """Build the current request, keeping large sensing arrays demand-driven."""
    if _sensing_dashboard_visible(viz):
        # The dashboard supports RP/RD, CFAR, angle cubes, CIR, micro-Doppler,
        # and history views, so correctness requires all stored products.
        return visual_frame_read_request(include_sensing=True)
    if visualizer_needs_sensing(viz):
        # 3D overlays consume JSON detection/track metadata. Naming this
        # product keeps numeric RP/RD/CIR datasets cold; metadata is always
        # decoded with a sensing projection.
        return visual_frame_read_request(
            include_sensing=True,
            sensing_products=frozenset({"detections"}),
        )
    return visual_frame_read_request()


def visualizer_frame_provider(viz: Any) -> DataProvider | None:
    """Return the shared provider exposed by the active loader or source."""
    loader = getattr(viz, "frame_loader", None)
    provider = getattr(loader, "provider", None)
    if isinstance(provider, DataProvider):
        return provider
    return frame_source_provider(getattr(viz, "frame_source", None))


def frame_source_provider(frame_source: Any) -> DataProvider | None:
    """Resolve the provider contract without bypassing a source's step mapping.

    Measurement-backed sources are themselves providers and translate
    contiguous visualizer steps to external measurement identifiers. Prefer
    that public source before inspecting a nested storage provider.
    """
    if isinstance(frame_source, DataProvider):
        return frame_source
    provider = getattr(frame_source, "provider", None)
    return provider if isinstance(provider, DataProvider) else None


def projection_to_visual_frame(
    projection: FrameProjection,
    *,
    points_dtype: np.dtype = np.float32,
) -> dict[str, Any]:
    """Build the compact dictionary seam consumed by the existing pipeline."""
    frame = projection.frame
    canonical = canonical_from_projection(projection, points_dtype=points_dtype)
    tx_positions = np.asarray(frame.tx_positions)
    rx_positions = np.asarray(frame.rx_positions)
    target_positions = (
        np.asarray(frame.target_positions_m)
        if frame.target_positions_m is not None
        else np.empty((0, 3), dtype=np.float64)
    )

    source = dict(frame.provenance or {})
    source.setdefault("provider", "frame_projection")
    source["frame_idx"] = int(frame.frame_index)
    if frame.timestamp_s is not None:
        source.setdefault("timestamp", float(frame.timestamp_s))

    payload: dict[str, Any] = {
        "canonical_data": canonical,
        "tx_rx_pairs": np.asarray(frame.tx_rx_pairs),
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "tx_orientations": np.asarray(frame.tx_orientations),
        "rx_orientations": np.asarray(frame.rx_orientations),
        "tx_names": list(frame.tx_names or ()),
        "rx_names": list(frame.rx_names or ()),
        "num_tx": len(tx_positions),
        "num_rx": len(rx_positions),
        "target_pos": target_positions,
        "targets_metadata": [dict(item) for item in frame.targets_metadata or ()],
        "num_targets": len(target_positions),
        "_source": source,
        # Canonicalization above copied every heavy geometry/metric array needed
        # by visual playback. Retain only request coverage so the frame cache
        # can decide whether a later overlay needs an incremental read.
        PACKED_PROJECTION_KEY: _ProjectionInventory.from_projection(projection),
    }
    if frame.beamforming:
        payload["beamforming"] = dict(frame.beamforming)
    if frame.sensing:
        payload["sensing"] = dict(frame.sensing)
    if frame.recomputed_from_stored_positions:
        payload["recomputed_from_stored_positions"] = True
    return payload


def standard_frame_to_visual_frame(
    frame: StandardMPCFrame,
    *,
    request: FrameReadRequest,
    points_dtype: np.dtype = np.float32,
) -> dict[str, Any]:
    """Project one complete canonical frame into the visual payload contract."""
    if not isinstance(frame, StandardMPCFrame):
        raise TypeError("visual frame conversion requires StandardMPCFrame")
    return projection_to_visual_frame(
        project_standard_mpc_frame(frame, request),
        points_dtype=points_dtype,
    )


def try_load_packed_visual_frame(
    provider: DataProvider,
    step: int,
    *,
    request: FrameReadRequest,
    points_dtype: np.dtype = np.float32,
) -> dict[str, Any] | None:
    """Load one provider projection and adapt it to the visual payload seam.

    ``DataProvider`` supplies a complete-frame projection fallback, so this
    path is valid for local, remote, live, measurement, and future providers.
    A provider may raise ``NotImplementedError`` when it cannot satisfy the
    projection request; callers then continue through their non-provider
    source path.
    """
    try:
        projection = provider.load_frame_projection(step, request)
    except NotImplementedError:
        return None
    return projection_to_visual_frame(projection, points_dtype=points_dtype)


def packed_payload_satisfies(
    payload: dict[str, Any],
    request: FrameReadRequest,
) -> bool:
    """Return whether a cached compact payload already contains ``request``."""
    inventory = _projection_inventory(payload.get(PACKED_PROJECTION_KEY))
    if inventory is None:
        return False
    if inventory.satisfies(request):
        return True
    if FrameComponent.SENSING not in request.components:
        return False

    non_sensing = FrameReadRequest(
        components=request.components - {FrameComponent.SENSING},
        metrics=request.metrics,
    )
    if not inventory.satisfies(non_sensing):
        return False
    sensing_inventory = _projection_inventory(payload.get(PACKED_SENSING_PROJECTION_KEY))
    sensing_request = FrameReadRequest(
        components=frozenset({FrameComponent.SENSING}),
        sensing_products=request.sensing_products,
    )
    return sensing_inventory is not None and sensing_inventory.satisfies(sensing_request)


def try_upgrade_packed_visual_frame(
    provider: DataProvider,
    step: int,
    payload: dict[str, Any],
    *,
    request: FrameReadRequest,
    points_dtype: np.dtype = np.float32,
) -> dict[str, Any] | None:
    """Upgrade a cached compact payload while avoiding redundant geometry I/O."""
    if packed_payload_satisfies(payload, request):
        return payload
    base_inventory = _projection_inventory(payload.get(PACKED_PROJECTION_KEY))
    if base_inventory is None:
        return None

    non_sensing = FrameReadRequest(
        components=request.components - {FrameComponent.SENSING},
        metrics=request.metrics,
    )
    if FrameComponent.SENSING in request.components and base_inventory.satisfies(non_sensing):
        sensing_request = FrameReadRequest(
            components=frozenset({FrameComponent.SENSING}),
            sensing_products=request.sensing_products,
        )
        try:
            sensing_projection = provider.load_frame_projection(step, sensing_request)
        except NotImplementedError:
            return None
        upgraded = dict(payload)
        upgraded[PACKED_PROJECTION_KEY] = base_inventory
        upgraded[PACKED_SENSING_PROJECTION_KEY] = _ProjectionInventory.from_projection(
            sensing_projection
        )
        sensing = sensing_projection.frame.sensing
        if sensing:
            upgraded["sensing"] = dict(sensing)
        else:
            upgraded.pop("sensing", None)
        return upgraded

    return try_load_packed_visual_frame(
        provider,
        step,
        request=request,
        points_dtype=points_dtype,
    )


__all__ = [
    "PACKED_PROJECTION_KEY",
    "PACKED_SENSING_PROJECTION_KEY",
    "frame_source_provider",
    "packed_payload_satisfies",
    "projection_to_visual_frame",
    "standard_frame_to_visual_frame",
    "try_load_packed_visual_frame",
    "try_upgrade_packed_visual_frame",
    "visual_frame_read_request",
    "visual_frame_read_request_for_visualizer",
    "visualizer_needs_sensing",
    "visualizer_frame_provider",
]
