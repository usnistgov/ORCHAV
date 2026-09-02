"""Build renderer canonical MPC arrays from compact frame projections.

This module is the boundary between provider-neutral ragged path arrays and
the ``CanonicalStepData`` contract used by rendering, filtering, statistics,
and picking. It reconstructs TX/bounce/RX polylines without creating padded
per-pair arrays.
"""

from __future__ import annotations

import time

import numpy as np

from shared.frames import (
    PATH_METRIC_ARRAY_FIELDS,
    FrameComponent,
    FrameProjection,
    PathMetric,
    ProjectedMPCFrame,
)

from ..materials.canonical_materials import _normalize_material_value
from .canonical_metrics import _compute_path_lengths
from .mpc_canon import CanonicalStepData, _bare_material_type, _canon_profile_enabled

_GEOMETRY_COMPONENTS = frozenset(
    {
        FrameComponent.DEVICES,
        FrameComponent.PATH_TOPOLOGY,
        FrameComponent.PATH_BOUNCE_TOPOLOGY,
        FrameComponent.PATH_GEOMETRY,
        FrameComponent.PATH_INTERACTIONS,
    }
)


def _required_array(frame: ProjectedMPCFrame, name: str) -> np.ndarray:
    """Return one required array with a focused projection error."""
    value = getattr(frame, name)
    if value is None:
        raise ValueError(f"Canonical projection requires frame.{name}")
    return np.asarray(value)


def _empty_canonical(points_dtype: np.dtype, profile_start: float) -> CanonicalStepData:
    """Return an aligned empty renderer-frame contract."""
    empty_u8 = np.empty((0,), dtype=np.uint8)
    empty_i32 = np.empty((0,), dtype=np.int32)
    empty_i16 = np.empty((0,), dtype=np.int16)
    empty_f32 = np.empty((0,), dtype=np.float32)
    profile_ms = None
    if profile_start:
        profile_ms = {"total_ms": (time.perf_counter() - profile_start) * 1000.0}
    return CanonicalStepData(
        points=np.empty((0, 3), dtype=points_dtype),
        lines=np.empty((0, 2), dtype=np.int32),
        order=empty_u8,
        itype=empty_u8,
        delay=empty_f32,
        loss=empty_f32,
        tx_id=empty_i16,
        rx_id=empty_i16,
        path_id=empty_i32,
        path_start_indices=empty_i32,
        path_orders=empty_u8,
        path_delays=empty_f32,
        path_losses=empty_f32,
        path_tx=empty_i16,
        path_rx=empty_i16,
        path_delay_is_estimated=np.empty((0,), dtype=np.bool_),
        path_loss_is_estimated=np.empty((0,), dtype=np.bool_),
        path_aoa_az=None,
        path_aoa_el=None,
        path_aod_az=None,
        path_aod_el=None,
        segment_start_indices=empty_i32,
        segment_end_indices=empty_i32,
        segment_order=empty_u8,
        segment_itype=empty_u8,
        segment_delay=empty_f32,
        segment_loss=empty_f32,
        segment_tx_id=empty_i16,
        segment_rx_id=empty_i16,
        segment_path_id=empty_i32,
        segment_material_ids=empty_i16,
        material_names=None,
        material_ids=None,
        material_itu_types=None,
        delay_min=0.0,
        delay_max=1.0,
        loss_min=0.0,
        loss_max=1.0,
        aoa_az=None,
        aoa_el=None,
        aod_az=None,
        aod_el=None,
        profile_ms=profile_ms,
    )


def _canonical_materials(
    frame: ProjectedMPCFrame,
    *,
    pair_path_offsets: np.ndarray,
    bounce_offsets: np.ndarray,
    bounce_destinations: np.ndarray,
    total_points: int,
) -> tuple[
    np.ndarray,
    dict[int, str] | None,
    dict[int, str] | None,
    dict[int, str] | None,
]:
    """Map catalog IDs to pair-major canonical material-name IDs.

    Canonical IDs follow pair traversal, sorted source IDs within each pair,
    and display-name de-duplication. This deterministic order keeps material
    filters, colors, and picking identities aligned without a padded matrix.
    """
    point_ids = np.zeros((total_points,), dtype=np.int16)
    raw_ids = frame.material_ids
    names = frame.material_names
    if raw_ids is None or names is None:
        return point_ids, None, None, None

    packed_ids = np.asarray(raw_ids)
    name_catalog = tuple(names)
    itu_catalog = tuple(frame.material_itu_types or ())
    if packed_ids.size == 0 or not name_catalog:
        return point_ids, None, None, None

    name_to_id: dict[str, int] = {}
    name_to_itu: dict[str, str] = {}
    raw_to_canonical = np.zeros((len(name_catalog),), dtype=np.int16)
    next_id = 1

    for pair_index in range(len(pair_path_offsets) - 1):
        path_start = int(pair_path_offsets[pair_index])
        path_end = int(pair_path_offsets[pair_index + 1])
        bounce_start = int(bounce_offsets[path_start])
        bounce_end = int(bounce_offsets[path_end])
        pair_raw_ids = packed_ids[bounce_start:bounce_end]
        if pair_raw_ids.size == 0:
            continue
        for raw_id_value in np.unique(pair_raw_ids):
            raw_id = int(raw_id_value)
            if raw_id < 0 or raw_id >= len(name_catalog):
                continue
            name = _normalize_material_value(name_catalog[raw_id])
            if not name:
                continue
            canonical_id = name_to_id.get(name)
            if canonical_id is None:
                if next_id > np.iinfo(np.int16).max:
                    raise ValueError("Canonical material count exceeds int16 capacity")
                canonical_id = next_id
                name_to_id[name] = canonical_id
                next_id += 1
            raw_to_canonical[raw_id] = canonical_id
            if name not in name_to_itu and raw_id < len(itu_catalog):
                itu = _normalize_material_value(itu_catalog[raw_id])
                if itu:
                    name_to_itu[name] = itu

    valid_raw = (packed_ids >= 0) & (packed_ids < len(raw_to_canonical))
    bounce_canonical = np.zeros(packed_ids.shape, dtype=np.int16)
    bounce_canonical[valid_raw] = raw_to_canonical[packed_ids[valid_raw]]
    point_ids[bounce_destinations] = bounce_canonical

    if not name_to_id:
        return point_ids, None, None, None
    id_to_name = {0: ""}
    id_to_itu = {0: ""}
    id_to_bare = {0: ""}
    for name, material_id in name_to_id.items():
        id_to_name[material_id] = name
        id_to_itu[material_id] = name_to_itu.get(name, "")
        id_to_bare[material_id] = _bare_material_type(name)
    return point_ids, id_to_name, id_to_itu, id_to_bare


def _projected_metric(
    frame: ProjectedMPCFrame,
    metric: PathMetric,
    path_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one float32 metric and its explicit per-path validity mask."""
    values = np.full((path_count,), np.nan, dtype=np.float32)
    valid = np.zeros((path_count,), dtype=np.bool_)
    field_name = PATH_METRIC_ARRAY_FIELDS[metric]
    source = getattr(frame, field_name)
    if source is None or frame.metric_valid_bits is None:
        return values, valid
    source_values = np.asarray(source, dtype=np.float32)
    valid = frame.metric_is_valid(metric)
    if np.all(valid):
        # A fully valid float32 metric does not need a writable fallback buffer,
        # so CanonicalStepData can retain the provider-owned array directly.
        return source_values, valid
    values[valid] = source_values[valid]
    return values, valid


def canonical_from_projection(
    value: FrameProjection | ProjectedMPCFrame,
    *,
    points_dtype: np.dtype = np.float32,
) -> CanonicalStepData:
    """Convert projected paths directly to the visualizer canonical contract.

    Required geometry is checked against the projection inventory when a
    ``FrameProjection`` is supplied. Delay and path-loss values use their
    explicit validity bits; unavailable values receive a geometric fallback.
    Optional angle channels preserve missing measurements as NaN and remain
    ``None`` when no real measurements exist.
    """
    dtype = np.dtype(points_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Canonical renderer points must use float32 or float64")
    profile_start = time.perf_counter() if _canon_profile_enabled() else 0.0

    if isinstance(value, FrameProjection):
        missing = _GEOMETRY_COMPONENTS - value.loaded_components
        if missing:
            names = ", ".join(sorted(component.value for component in missing))
            raise ValueError(f"Canonical projection is missing: {names}")
        frame = value.frame
    else:
        frame = value

    pairs = _required_array(frame, "tx_rx_pairs")
    pair_path_offsets = _required_array(frame, "pair_path_offsets").astype(np.int64, copy=False)
    bounce_offsets = _required_array(frame, "bounce_offsets").astype(np.int64, copy=False)
    tx_positions = _required_array(frame, "tx_positions")
    rx_positions = _required_array(frame, "rx_positions")
    bounce_xyz = _required_array(frame, "bounce_xyz_m")
    interactions = _required_array(frame, "interactions")

    path_count = len(bounce_offsets) - 1
    if path_count == 0:
        return _empty_canonical(dtype, profile_start)

    pair_path_counts = np.diff(pair_path_offsets)
    path_pair = np.repeat(np.arange(len(pairs), dtype=np.int32), pair_path_counts)
    if len(path_pair) != path_count:
        raise ValueError("Pair/path offsets disagree with bounce offsets")
    path_tx_i64 = np.asarray(pairs[path_pair, 0], dtype=np.int64)
    path_rx_i64 = np.asarray(pairs[path_pair, 1], dtype=np.int64)

    bounce_counts = np.diff(bounce_offsets)
    nodes_per_path_i64 = bounce_counts + 2
    total_points = int(np.sum(nodes_per_path_i64, dtype=np.int64))
    if total_points > np.iinfo(np.int32).max:
        raise ValueError("Canonical point count exceeds int32 line-index capacity")
    path_ordinals_i64 = np.arange(path_count, dtype=np.int64)
    path_starts_i64 = bounce_offsets[:-1] + 2 * path_ordinals_i64
    path_starts = np.ascontiguousarray(path_starts_i64, dtype=np.int32)
    nodes_per_path = np.ascontiguousarray(nodes_per_path_i64, dtype=np.int32)

    # Renderer geometry is quantized to float32 before the optional float64
    # output cast so drawing and picking use the same coordinates.
    points = np.empty((total_points, 3), dtype=dtype)
    points[path_starts] = np.asarray(tx_positions[path_tx_i64], dtype=np.float32)
    path_ends_i64 = path_starts_i64 + nodes_per_path_i64 - 1
    points[path_ends_i64] = np.asarray(rx_positions[path_rx_i64], dtype=np.float32)

    total_bounces = int(bounce_offsets[-1])
    bounce_path = np.repeat(path_ordinals_i64, bounce_counts)
    bounce_destinations_i64 = np.arange(total_bounces, dtype=np.int64) + 2 * bounce_path + 1
    bounce_destinations = np.ascontiguousarray(bounce_destinations_i64, dtype=np.int32)
    if total_bounces:
        points[bounce_destinations] = np.asarray(bounce_xyz, dtype=np.float32)

    itype = np.zeros((total_points,), dtype=np.uint8)
    if total_bounces:
        itype[bounce_destinations] = np.asarray(interactions, dtype=np.uint8)

    all_indices = np.arange(total_points, dtype=np.int32)
    non_last = np.ones((total_points,), dtype=np.bool_)
    non_last[path_ends_i64] = False
    starts = all_indices[non_last]
    lines = np.ascontiguousarray(np.column_stack((starts, starts + 1)), dtype=np.int32)

    # Interactions contain physical bounces only and exclude the zero endpoint
    # sentinel, so path order equals the bounce count.
    path_orders = bounce_counts.astype(np.uint8, copy=False)
    order = np.repeat(path_orders, nodes_per_path).astype(np.uint8, copy=False)
    path_id = np.repeat(np.arange(path_count, dtype=np.int32), nodes_per_path)
    path_tx = path_tx_i64.astype(np.int16, copy=False)
    path_rx = path_rx_i64.astype(np.int16, copy=False)
    tx_id = np.repeat(path_tx, nodes_per_path)
    rx_id = np.repeat(path_rx, nodes_per_path)

    path_delays, delay_valid = _projected_metric(frame, PathMetric.DELAY_NS, path_count)
    path_losses, loss_valid = _projected_metric(frame, PathMetric.PATH_LOSS_DB, path_count)
    delay_missing = ~delay_valid
    loss_missing = ~loss_valid
    if np.any(delay_missing) or np.any(loss_missing):
        lengths_m = _compute_path_lengths(
            points,
            path_starts,
            nodes_per_path,
            total_points,
            path_count,
        )
        if np.any(delay_missing):
            geometric_delay = lengths_m / np.float32(0.3)
            path_delays[delay_missing] = geometric_delay[delay_missing]
        if np.any(loss_missing):
            safe_lengths = np.maximum(lengths_m, np.float32(1e-6))
            geometric_loss = 20.0 * np.log10(safe_lengths) + 30.0
            path_losses[loss_missing] = geometric_loss[loss_missing]
    delay = np.repeat(path_delays, nodes_per_path)
    loss = np.repeat(path_losses, nodes_per_path)

    material_ids, id_to_name, id_to_itu, id_to_bare = _canonical_materials(
        frame,
        pair_path_offsets=pair_path_offsets,
        bounce_offsets=bounce_offsets,
        bounce_destinations=bounce_destinations,
        total_points=total_points,
    )

    angle_by_metric: dict[PathMetric, np.ndarray | None] = {}
    angle_ranges: dict[PathMetric, tuple[float, float]] = {}
    defaults = {
        PathMetric.AOA_AZ_DEG: (0.0, 360.0),
        PathMetric.AOA_EL_DEG: (-90.0, 90.0),
        PathMetric.AOD_AZ_DEG: (0.0, 360.0),
        PathMetric.AOD_EL_DEG: (-90.0, 90.0),
    }
    for metric, default_range in defaults.items():
        path_values, valid = _projected_metric(frame, metric, path_count)
        if np.any(valid):
            broadcast = np.repeat(path_values, nodes_per_path)
            angle_by_metric[metric] = np.ascontiguousarray(broadcast, dtype=np.float32)
            angle_ranges[metric] = (
                float(np.nanmin(path_values)),
                float(np.nanmax(path_values)),
            )
        else:
            angle_by_metric[metric] = None
            angle_ranges[metric] = default_range

    segment_start = lines[:, 0].astype(np.int32, copy=False)
    segment_end = lines[:, 1].astype(np.int32, copy=False)
    segment_order = order[segment_start].astype(np.uint8, copy=False)
    segment_itype_start = itype[segment_start]
    segment_itype_end = itype[segment_end]
    segment_itype = segment_itype_start.copy()
    inherit_first_bounce = (
        (segment_order > 0) & (segment_itype_start == 0) & (segment_itype_end != 0)
    )
    segment_itype[inherit_first_bounce] = segment_itype_end[inherit_first_bounce]

    profile_ms = None
    if profile_start:
        profile_ms = {"total_ms": (time.perf_counter() - profile_start) * 1000.0}

    return CanonicalStepData(
        points=np.ascontiguousarray(points),
        lines=lines,
        order=np.ascontiguousarray(order),
        itype=np.ascontiguousarray(itype),
        delay=np.ascontiguousarray(delay, dtype=np.float32),
        loss=np.ascontiguousarray(loss, dtype=np.float32),
        tx_id=np.ascontiguousarray(tx_id, dtype=np.int16),
        rx_id=np.ascontiguousarray(rx_id, dtype=np.int16),
        path_id=np.ascontiguousarray(path_id, dtype=np.int32),
        path_start_indices=path_starts,
        path_orders=np.ascontiguousarray(path_orders, dtype=np.uint8),
        path_delays=np.ascontiguousarray(path_delays, dtype=np.float32),
        path_losses=np.ascontiguousarray(path_losses, dtype=np.float32),
        path_tx=np.ascontiguousarray(path_tx, dtype=np.int16),
        path_rx=np.ascontiguousarray(path_rx, dtype=np.int16),
        path_delay_is_estimated=np.ascontiguousarray(delay_missing, dtype=np.bool_),
        path_loss_is_estimated=np.ascontiguousarray(loss_missing, dtype=np.bool_),
        path_aoa_az=None,
        path_aoa_el=None,
        path_aod_az=None,
        path_aod_el=None,
        segment_start_indices=segment_start,
        segment_end_indices=segment_end,
        segment_order=np.ascontiguousarray(segment_order, dtype=np.uint8),
        segment_itype=np.ascontiguousarray(segment_itype, dtype=np.uint8),
        segment_delay=np.ascontiguousarray(delay[segment_start], dtype=np.float32),
        segment_loss=np.ascontiguousarray(loss[segment_start], dtype=np.float32),
        segment_tx_id=np.ascontiguousarray(tx_id[segment_start], dtype=np.int16),
        segment_rx_id=np.ascontiguousarray(rx_id[segment_start], dtype=np.int16),
        segment_path_id=np.ascontiguousarray(path_id[segment_start], dtype=np.int32),
        segment_material_ids=np.ascontiguousarray(material_ids[segment_start], dtype=np.int16),
        material_names=None,
        material_ids=np.ascontiguousarray(material_ids, dtype=np.int16),
        material_itu_types=None,
        material_id_to_name=id_to_name,
        material_id_to_itu=id_to_itu,
        material_id_to_bare=id_to_bare,
        delay_min=float(np.nanmin(path_delays)),
        delay_max=float(np.nanmax(path_delays)),
        loss_min=float(np.nanmin(path_losses)),
        loss_max=float(np.nanmax(path_losses)),
        aoa_az=angle_by_metric[PathMetric.AOA_AZ_DEG],
        aoa_el=angle_by_metric[PathMetric.AOA_EL_DEG],
        aod_az=angle_by_metric[PathMetric.AOD_AZ_DEG],
        aod_el=angle_by_metric[PathMetric.AOD_EL_DEG],
        aoa_az_min=angle_ranges[PathMetric.AOA_AZ_DEG][0],
        aoa_az_max=angle_ranges[PathMetric.AOA_AZ_DEG][1],
        aoa_el_min=angle_ranges[PathMetric.AOA_EL_DEG][0],
        aoa_el_max=angle_ranges[PathMetric.AOA_EL_DEG][1],
        aod_az_min=angle_ranges[PathMetric.AOD_AZ_DEG][0],
        aod_az_max=angle_ranges[PathMetric.AOD_AZ_DEG][1],
        aod_el_min=angle_ranges[PathMetric.AOD_EL_DEG][0],
        aod_el_max=angle_ranges[PathMetric.AOD_EL_DEG][1],
        profile_ms=profile_ms,
    )


__all__ = ["canonical_from_projection"]
