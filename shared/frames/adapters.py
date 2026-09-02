"""Project complete canonical MPC frames into requested in-memory fields.

Storage-aware providers may construct :class:`FrameProjection` directly.
Providers without selective I/O call :func:`project_standard_mpc_frame`, which
retains references to the requested canonical arrays without padding or
repacking them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .contracts import FrameComponent, FrameReadRequest
from .packed import FrameProjection, ProjectedMPCFrame
from .types import PATH_METRIC_ARRAY_FIELDS, StandardMPCFrame


def _sensing_projection(
    sensing: Mapping[str, Any] | None,
    request: FrameReadRequest,
) -> tuple[dict[str, Any] | None, frozenset[str], bool]:
    """Select numeric sensing products while retaining small metadata."""

    if sensing is None:
        # A selective read records that each requested product was checked even
        # when the source frame has no sensing payload. This matches the HDF5
        # reader and lets callers distinguish known absence from an incomplete
        # projection.
        return None, request.sensing_products, request.all_sensing_products
    source = dict(sensing)
    available_products = frozenset(
        name for name, value in source.items() if isinstance(value, np.ndarray)
    )
    if request.all_sensing_products:
        return source, available_products, True

    selected = request.sensing_products
    projected = {
        name: value
        for name, value in source.items()
        if not isinstance(value, np.ndarray) or name in selected
    }
    return projected, selected, False


def project_standard_mpc_frame(
    frame: StandardMPCFrame,
    request: FrameReadRequest,
) -> FrameProjection:
    """Return a zero-copy logical projection of one complete frame."""

    if not isinstance(frame, StandardMPCFrame):
        raise TypeError("frame must be a complete StandardMPCFrame")

    components = request.components
    kwargs: dict[str, Any] = {"frame_index": frame.frame_index}

    if FrameComponent.DEVICES in components:
        kwargs.update(
            tx_positions=frame.tx_positions,
            rx_positions=frame.rx_positions,
            tx_orientations=frame.tx_orientations,
            rx_orientations=frame.rx_orientations,
            tx_names=frame.tx_names,
            rx_names=frame.rx_names,
        )
    if FrameComponent.PATH_TOPOLOGY in components:
        kwargs.update(
            tx_rx_pairs=frame.tx_rx_pairs,
            pair_path_offsets=frame.pair_path_offsets,
        )
    if FrameComponent.PATH_BOUNCE_TOPOLOGY in components:
        kwargs["bounce_offsets"] = frame.bounce_offsets
    if FrameComponent.PATH_GEOMETRY in components:
        kwargs["bounce_xyz_m"] = frame.bounce_xyz_m
    if FrameComponent.PATH_INTERACTIONS in components:
        kwargs["interactions"] = frame.interactions
    if FrameComponent.PATH_MATERIALS in components:
        kwargs.update(
            material_ids=frame.material_ids,
            material_names=frame.material_names,
            material_itu_types=frame.material_itu_types,
        )
    if FrameComponent.PATH_METRICS in components:
        kwargs["metric_valid_bits"] = frame.metric_valid_bits
        for metric in request.metrics:
            kwargs[PATH_METRIC_ARRAY_FIELDS[metric]] = getattr(
                frame, PATH_METRIC_ARRAY_FIELDS[metric]
            )
    if FrameComponent.TARGETS in components:
        kwargs.update(
            target_positions_m=frame.target_positions_m,
            targets_metadata=frame.targets_metadata,
        )
    if FrameComponent.BEAMFORMING in components:
        kwargs["beamforming"] = None if frame.beamforming is None else dict(frame.beamforming)
    if FrameComponent.PROVENANCE in components:
        kwargs.update(
            provenance=None if frame.provenance is None else dict(frame.provenance),
            timestamp_s=frame.timestamp_s,
            recomputed_from_stored_positions=frame.recomputed_from_stored_positions,
        )

    loaded_sensing_products: frozenset[str] = frozenset()
    all_sensing_products_loaded = False
    if FrameComponent.SENSING in components:
        sensing, loaded_sensing_products, all_sensing_products_loaded = _sensing_projection(
            frame.sensing, request
        )
        kwargs["sensing"] = sensing

    return FrameProjection(
        frame=ProjectedMPCFrame(**kwargs),
        loaded_components=request.components,
        loaded_path_metrics=request.metrics,
        loaded_sensing_products=loaded_sensing_products,
        all_sensing_products_loaded=all_sensing_products_loaded,
    )


__all__ = ["project_standard_mpc_frame"]
