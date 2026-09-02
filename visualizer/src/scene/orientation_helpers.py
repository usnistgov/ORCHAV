"""Orientation-frame synchronization for TX, RX, and target scene objects.

Frame data stores Sionna yaw/pitch/roll values in radians. This module turns
those values into renderer-neutral RGB axis handles and keeps the handle names
stable so Open3D, pygfx, sessions, and visibility controls synchronize the
same objects.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Iterable

import numpy as np

from shared.logging import get_logger

from ..model import RenderObjectState, Transform, render_state_center
from ..services.object_identity import (
    ensure_target_entry_identity,
    make_node_geometry_name,
    make_target_entry_geometry_name,
)
from ..types.render_payloads import OrientationFramePayload
from .orientation_frame_payloads import (
    DEFAULT_ORIENTATION_FRAME_THICKNESS,
    make_orientation_frame_handle,
    make_orientation_frame_payload,
    orientation_frame_size,
)
from .target_transforms import target_entry_anchor_position

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.orientation_helpers")


def _benchmark_active(viz: OrchavVisualizer) -> bool:
    """Return True when frame benchmark metrics should be collected."""
    pipeline = getattr(viz, "pipeline", None)
    return getattr(pipeline, "benchmark_recorder", None) is not None


def _reset_orientation_metrics(viz: OrchavVisualizer) -> None:
    """Reset per-frame orientation metrics when benchmark mode is active."""
    if _benchmark_active(viz):
        viz._orientation_frame_breakdown = {}


def _record_orientation_metric(viz: OrchavVisualizer, name: str, value: float = 1.0) -> None:
    """Accumulate one orientation-frame metric for benchmark output."""
    if not _benchmark_active(viz):
        return
    metrics = getattr(viz, "_orientation_frame_breakdown", None)
    if not isinstance(metrics, dict):
        metrics = {}
        viz._orientation_frame_breakdown = metrics
    metrics[name] = float(metrics.get(name, 0.0)) + float(value)


def _orientation_axis_thickness(viz: OrchavVisualizer) -> float:
    """Return a finite native-axis thickness for orientation-frame helpers."""
    try:
        value = float(getattr(viz, "orientation_axis_thickness"))
    except (TypeError, ValueError):
        value = DEFAULT_ORIENTATION_FRAME_THICKNESS
    if not np.isfinite(value):
        return DEFAULT_ORIENTATION_FRAME_THICKNESS
    return max(value, 0.0)


def _orientation_frame_visible(
    viz: OrchavVisualizer,
    node_type: str,
    index: int,
) -> bool:
    """Resolve effective frame visibility through the node policy owner."""
    node_service = getattr(viz, "node_service", None)
    resolve_visibility = getattr(node_service, "orientation_frame_visible", None)
    if not callable(resolve_visibility):
        return False
    visible = resolve_visibility(node_type, index)
    return bool(visible) if isinstance(visible, (bool, np.bool_)) else False


def _pending_orientation_syncs(
    viz: OrchavVisualizer,
) -> dict[str, tuple[RenderObjectState, bool]]:
    """Return renderer snapshots awaiting a successful orientation sync."""
    pending = vars(viz).get("_pending_orientation_frame_syncs")
    if not isinstance(pending, dict):
        pending = {}
        setattr(viz, "_pending_orientation_frame_syncs", pending)
    return pending


def _pending_orientation_removals(
    viz: OrchavVisualizer,
) -> dict[str, RenderObjectState]:
    """Return orientation handles awaiting successful native removal."""
    pending = vars(viz).get("_pending_orientation_frame_removals")
    if not isinstance(pending, dict):
        pending = {}
        setattr(viz, "_pending_orientation_frame_removals", pending)
    return pending


def _sync_orientation_frame(
    viz: OrchavVisualizer,
    frame: RenderObjectState,
    *,
    effective_visible: bool,
) -> bool:
    """Ensure one complete orientation-frame snapshot in the active renderer."""
    renderer = getattr(viz, "renderer", None)
    if renderer is None:
        _pending_orientation_syncs(viz)[frame.id] = (frame, bool(effective_visible))
        return False
    _pending_orientation_removals(viz).pop(frame.id, None)
    start = time.perf_counter()
    ensure_object = getattr(renderer, "ensure_object", None)
    synced = bool(
        callable(ensure_object)
        and ensure_object(frame.to_render_object(effective_visible=bool(effective_visible)))
    )
    _record_orientation_metric(
        viz,
        "orientation_frame_sync_ms",
        (time.perf_counter() - start) * 1000.0,
    )
    _record_orientation_metric(viz, "orientation_frame_sync_count")
    pending = _pending_orientation_syncs(viz)
    if synced:
        pending.pop(frame.id, None)
    else:
        pending[frame.id] = (frame, bool(effective_visible))
    return synced


def sync_orientation_frame_visibility(
    viz: OrchavVisualizer,
    frame: RenderObjectState,
    visible: bool,
) -> bool:
    """Publish one complete orientation snapshot after a visibility change."""
    _retry_pending_orientation_operations(viz)
    return _sync_orientation_frame(viz, frame, effective_visible=bool(visible))


def _remove_orientation_frame(
    viz: OrchavVisualizer,
    frame: RenderObjectState,
) -> bool:
    """Remove one frame without forgetting an incomplete backend operation."""
    _pending_orientation_syncs(viz).pop(frame.id, None)
    renderer = getattr(viz, "renderer", None)
    remove_object = getattr(renderer, "remove_object", None)
    removed = bool(callable(remove_object) and remove_object(frame.id))
    pending = _pending_orientation_removals(viz)
    if removed:
        pending.pop(frame.id, None)
    else:
        pending[frame.id] = frame
    return removed


def _retry_pending_orientation_operations(viz: OrchavVisualizer) -> bool:
    """Retry incomplete removals and snapshots from the previous update."""
    all_synced = True
    for frame in tuple(_pending_orientation_removals(viz).values()):
        if not _remove_orientation_frame(viz, frame):
            all_synced = False
    pending_removal_ids = set(_pending_orientation_removals(viz))
    for render_id, (frame, visible) in tuple(_pending_orientation_syncs(viz).items()):
        if render_id in pending_removal_ids:
            continue
        if not _sync_orientation_frame(viz, frame, effective_visible=visible):
            all_synced = False
    return all_synced


def _trim_orientation_frames(viz: OrchavVisualizer, frames: list, desired_count: int) -> bool:
    """Remove stale orientation-frame objects after count changes."""
    if len(frames) <= desired_count:
        return True
    all_removed = True
    for frame in frames[desired_count:]:
        if isinstance(frame, RenderObjectState):
            all_removed = _remove_orientation_frame(viz, frame) and all_removed
    del frames[desired_count:]
    return all_removed


def _orientation_frame(
    frames: list,
    index: int,
    render_id: str,
    orientation_scale: float,
    orientation_thickness: float,
    visible: bool,
) -> RenderObjectState:
    """Return a stable orientation-frame handle at *index*."""
    if index < len(frames) and isinstance(frames[index], RenderObjectState):
        frame = frames[index]
        if frame.id == render_id:
            _resize_orientation_frame(frame, orientation_scale, orientation_thickness)
            return frame
    frame = make_orientation_frame_handle(
        render_id,
        size=orientation_scale,
        thickness=orientation_thickness,
        visible=visible,
    )
    if index < len(frames):
        frames[index] = frame
    else:
        frames.append(frame)
    return frame


def _resize_orientation_frame(
    frame: RenderObjectState,
    orientation_scale: float,
    orientation_thickness: float = DEFAULT_ORIENTATION_FRAME_THICKNESS,
) -> bool:
    """Refresh a stable orientation-frame handle when the configured scale changes."""
    axis_size = orientation_frame_size(orientation_scale)
    axis_thickness = max(float(orientation_thickness), 0.0)
    try:
        current_size = float(frame.metadata.get("size"))
    except (TypeError, ValueError):
        current_size = float("nan")
    try:
        current_thickness = float(frame.metadata.get("thickness"))
    except (TypeError, ValueError):
        current_thickness = float("nan")
    if (
        isinstance(frame.payload, OrientationFramePayload)
        and np.isfinite(current_size)
        and abs(current_size - axis_size) < 1e-9
        and np.isfinite(current_thickness)
        and abs(current_thickness - axis_thickness) < 1e-9
    ):
        return False
    frame.replace_payload(make_orientation_frame_payload(axis_size, thickness=axis_thickness))
    frame.material = replace(frame.material, line_width=axis_thickness)
    frame.is_edge = False
    frame.metadata.update(
        {
            "type": "orientation_frame",
            "size": axis_size,
            "thickness": axis_thickness,
            "pickable": False,
        }
    )
    return True


def create_orientation_frames(visualizer: OrchavVisualizer, step: int) -> bool:
    """Create or update TX/RX/Target orientation frames for a specific step."""
    viz = visualizer
    _reset_orientation_metrics(viz)
    total_start = time.perf_counter()
    data = viz.cache_service.get_frame(step)
    if data is None:
        logger.debug("No frame data for step %s to create orientation frames", step)
        return True
    tx_raw = _to_orientation_list(data.get("tx_orientations"))
    rx_raw = _to_orientation_list(data.get("rx_orientations"))
    tx_orientations, rx_orientations = normalize_node_orientation_lists(
        viz,
        tx_raw,
        rx_raw,
        data,
    )
    target_orientations = _extract_target_orientations(data)

    logger.debug(
        "Orientation frames step %s: TX=%s RX=%s Target=%s (raw TX=%s RX=%s)",
        step,
        len(tx_orientations),
        len(rx_orientations),
        len(target_orientations),
        len(tx_raw),
        len(rx_raw),
    )
    if len(tx_orientations) > 0:
        logger.debug("TX orientations: %s", tx_orientations)
    if len(rx_orientations) > 0:
        logger.debug("RX orientations: %s", rx_orientations)
    if len(target_orientations) > 0:
        logger.debug("Target orientations: %s", target_orientations)

    tx_synced = update_tx_orientation_frames(viz, tx_orientations)
    rx_synced = update_rx_orientation_frames(viz, rx_orientations)
    targets_synced = update_target_orientation_frames(viz, target_orientations)
    _record_orientation_metric(
        viz,
        "orientation_frame_total_ms",
        (time.perf_counter() - total_start) * 1000.0,
    )
    return bool(tx_synced and rx_synced and targets_synced)


def update_tx_orientation_frames(visualizer: OrchavVisualizer, tx_orientations: Iterable) -> bool:
    """Update TX orientation coordinate frames."""
    viz = visualizer
    try:
        all_synced = _retry_pending_orientation_operations(viz)
        orientations = list(tx_orientations)
        orientation_scale = getattr(viz, "orientation_scale", 3.0)
        orientation_thickness = _orientation_axis_thickness(viz)
        frames = getattr(viz, "tx_orientation_frames", [])
        viz.tx_orientation_frames = frames
        all_synced = _trim_orientation_frames(viz, frames, len(orientations)) and all_synced

        tx_positions = getattr(viz, "current_tx_positions", [])

        for i, orientation in enumerate(orientations):
            if i < len(viz.tx_markers):
                frame = _orientation_frame(
                    frames,
                    i,
                    make_node_geometry_name("tx", i, "orientation_frame"),
                    orientation_scale,
                    orientation_thickness,
                    False,
                )
                marker = viz.tx_markers[i]
                if frame is not None and marker is not None:
                    # Use cached world position; state payload centers can be
                    # stale when the renderer positions
                    # nodes via Filament scene transforms.
                    if i < len(tx_positions):
                        position = np.asarray(tx_positions[i], dtype=np.float64)
                    else:
                        position = np.asarray(render_state_center(marker))
                    yaw, pitch, roll = orientation
                    transform_matrix = create_orientation_transform(position, yaw, pitch, roll)
                    frame.world_transform = Transform(transform_matrix)
                    all_synced = (
                        _sync_orientation_frame(
                            viz,
                            frame,
                            effective_visible=_orientation_frame_visible(viz, "tx", i),
                        )
                        and all_synced
                    )
                    _record_orientation_metric(viz, "tx_orientation_frame_sync_count")
                    logger.debug(
                        "Updated TX%s orientation frame at %s with %s",
                        i + 1,
                        position,
                        orientation,
                    )
        return all_synced
    except (RuntimeError, ValueError):  # pragma: no cover - depends on Open3D runtime
        logger.exception("Error updating TX orientation frames")
        return False


def update_rx_orientation_frames(visualizer: OrchavVisualizer, rx_orientations: Iterable) -> bool:
    """Update RX orientation coordinate frames."""
    viz = visualizer
    try:
        all_synced = _retry_pending_orientation_operations(viz)
        orientations = list(rx_orientations)
        orientation_scale = getattr(viz, "orientation_scale", 3.0)
        orientation_thickness = _orientation_axis_thickness(viz)
        frames = getattr(viz, "rx_orientation_frames", [])
        viz.rx_orientation_frames = frames
        all_synced = _trim_orientation_frames(viz, frames, len(orientations)) and all_synced

        rx_positions = getattr(viz, "current_rx_positions", [])

        for i, orientation in enumerate(orientations):
            if i < len(viz.rx_markers):
                frame = _orientation_frame(
                    frames,
                    i,
                    make_node_geometry_name("rx", i, "orientation_frame"),
                    orientation_scale,
                    orientation_thickness,
                    False,
                )
                marker = viz.rx_markers[i]
                if frame is not None and marker is not None:
                    # Use cached world position; state payload centers can be
                    # stale when the renderer positions
                    # nodes via Filament scene transforms.
                    if i < len(rx_positions):
                        position = np.asarray(rx_positions[i], dtype=np.float64)
                    else:
                        position = np.asarray(render_state_center(marker))
                    yaw, pitch, roll = orientation
                    transform_matrix = create_orientation_transform(position, yaw, pitch, roll)
                    frame.world_transform = Transform(transform_matrix)
                    all_synced = (
                        _sync_orientation_frame(
                            viz,
                            frame,
                            effective_visible=_orientation_frame_visible(viz, "rx", i),
                        )
                        and all_synced
                    )
                    _record_orientation_metric(viz, "rx_orientation_frame_sync_count")
                    logger.debug(
                        "Updated RX%s orientation frame at %s with %s",
                        i + 1,
                        position,
                        orientation,
                    )
        return all_synced
    except (RuntimeError, ValueError):  # pragma: no cover
        logger.exception("Error updating RX orientation frames")
        return False


def update_target_orientation_frames(
    visualizer: OrchavVisualizer,
    target_orientations: Mapping[str, object],
) -> bool:
    """Update target frames by stable target name, independent of frame order."""
    viz = visualizer
    try:
        all_synced = _retry_pending_orientation_operations(viz)
        orientations_by_name = dict(target_orientations)
        orientation_scale = getattr(viz, "orientation_scale", 3.0)
        orientation_thickness = _orientation_axis_thickness(viz)
        previous_frames = getattr(viz, "target_orientation_frames", [])
        frames_by_id = {
            frame.id: frame for frame in previous_frames if isinstance(frame, RenderObjectState)
        }
        current_frames: list[RenderObjectState | None] = []
        retained_ids: set[str] = set()

        for i, target_entry in enumerate(getattr(viz, "target_entries", [])):
            ensure_target_entry_identity(target_entry, i)
            render_id = make_target_entry_geometry_name(target_entry, "orientation_frame")
            retained_ids.add(render_id)
            frame = frames_by_id.get(render_id)
            orientation = _target_orientation_for_entry(orientations_by_name, target_entry)

            if orientation is None:
                if frame is not None:
                    all_synced = (
                        _sync_orientation_frame(viz, frame, effective_visible=False) and all_synced
                    )
                current_frames.append(frame)
                continue

            if frame is None:
                frame = make_orientation_frame_handle(
                    render_id,
                    size=orientation_scale,
                    thickness=orientation_thickness,
                    visible=False,
                )
            else:
                _resize_orientation_frame(frame, orientation_scale, orientation_thickness)
            current_frames.append(frame)

            if target_entry.get("mesh") is None:
                all_synced = (
                    _sync_orientation_frame(viz, frame, effective_visible=False) and all_synced
                )
                continue
            position = target_entry_anchor_position(target_entry)
            if position is None:
                all_synced = (
                    _sync_orientation_frame(viz, frame, effective_visible=False) and all_synced
                )
                continue
            yaw, pitch, roll = orientation
            transform_matrix = create_orientation_transform(position, yaw, pitch, roll)
            frame.world_transform = Transform(transform_matrix)
            all_synced = (
                _sync_orientation_frame(
                    viz,
                    frame,
                    effective_visible=_orientation_frame_visible(viz, "target", i),
                )
                and all_synced
            )
            _record_orientation_metric(viz, "target_orientation_frame_sync_count")
            logger.debug(
                "Updated Target%s orientation frame at %s with %s",
                i + 1,
                position,
                orientation,
            )

        for render_id, frame in frames_by_id.items():
            if render_id not in retained_ids:
                all_synced = _remove_orientation_frame(viz, frame) and all_synced
        viz.target_orientation_frames = current_frames
        return all_synced
    except (RuntimeError, ValueError):  # pragma: no cover
        logger.exception("Error updating Target orientation frames")
        return False


def create_orientation_transform(position, yaw, pitch, roll):
    """Create a 4x4 transformation matrix from yaw/pitch/roll."""
    rx, ry, rz = _sionna_ypr_to_xyz_rotation(yaw, pitch, roll)
    rotation_matrix = np.eye(3)

    if abs(rx) > 1e-6:
        roll_matrix = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )
        rotation_matrix = roll_matrix @ rotation_matrix

    if abs(ry) > 1e-6:
        pitch_matrix = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )
        rotation_matrix = pitch_matrix @ rotation_matrix

    if abs(rz) > 1e-6:
        yaw_matrix = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )
        rotation_matrix = yaw_matrix @ rotation_matrix

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    return transform_matrix


def _sionna_ypr_to_xyz_rotation(yaw, pitch, roll):
    """Map Sionna yaw/pitch/roll radians into X/Y/Z rotation components.

    The pipeline guarantees that orientation values are stored in radians, so
    this helper only remaps component order for matrix construction.
    """
    yaw_rad = float(yaw)
    pitch_rad = float(pitch)
    roll_rad = float(roll)
    return roll_rad, pitch_rad, yaw_rad


def _extract_target_orientations(data: dict) -> dict[str, object]:
    """Extract target orientations keyed by stable frame target name."""
    target_orientations: dict[str, object] = {}
    if "targets_metadata" in data and data["targets_metadata"]:
        for index, target_meta in enumerate(data["targets_metadata"]):
            if not isinstance(target_meta, dict):
                continue
            target_name = str(
                target_meta.get("stable_target_id")
                or target_meta.get("target_name")
                or target_meta.get("name")
                or target_meta.get("node_name")
                or f"target_{index}"
            )
            orientation = target_meta.get("orientation", [0, 0, 0])
            if hasattr(orientation, "tolist"):
                orientation = orientation.tolist()
            target_orientations[target_name] = orientation
    else:
        logger.debug("No targets_metadata found in frame data for orientations")
    return target_orientations


def _target_orientation_for_entry(
    orientations_by_name: Mapping[str, object],
    target_entry: dict,
) -> object | None:
    """Resolve one target orientation through stable entry identity aliases."""
    for key in (
        target_entry.get("stable_target_id"),
        target_entry.get("target_name"),
        target_entry.get("name"),
        target_entry.get("node_name"),
    ):
        if key is not None and str(key) in orientations_by_name:
            return orientations_by_name[str(key)]
    return None


def _to_orientation_list(values) -> list:
    """Normalize list-like orientation payloads from frame data."""
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if hasattr(values, "tolist"):
        try:
            return values.tolist()
        except (ValueError, TypeError):
            pass
    try:
        return list(values)
    except TypeError:
        return []


def _safe_len(values) -> int:
    """Return collection length without triggering NumPy truthiness errors."""
    if values is None:
        return 0
    try:
        return len(values)
    except TypeError:
        return 0


def normalize_node_orientation_lists(
    visualizer: OrchavVisualizer,
    tx_values,
    rx_values,
    frame_data: dict,
) -> tuple[list, list]:
    """Normalize orientation arrays to unique TX/RX node cardinalities.

    Some frame formats store RX orientation per TX/RX pair. The visualizer
    renders one marker per RX node, so those pairwise arrays collapse to the
    first orientation for each RX when ``num_tx`` is available.
    """
    viz = visualizer
    tx_list = _to_orientation_list(tx_values)
    rx_list = _to_orientation_list(rx_values)

    tx_target_count = _safe_len(getattr(viz, "current_tx_positions", None))
    rx_target_count = _safe_len(getattr(viz, "current_rx_positions", None))
    if tx_target_count <= 0:
        tx_target_count = _safe_len(getattr(viz, "tx_markers", None))
    if rx_target_count <= 0:
        rx_target_count = _safe_len(getattr(viz, "rx_markers", None))

    try:
        num_tx = int(frame_data.get("num_tx"))
    except (TypeError, ValueError, AttributeError):
        num_tx = 0

    return (
        _normalize_tx_orientations(tx_list, tx_target_count),
        _normalize_rx_orientations(rx_list, rx_target_count, num_tx),
    )


def orientations_empty(values) -> bool:
    """Return whether an orientation container has no usable entries."""
    if values is None:
        return True
    try:
        return len(values) == 0
    except TypeError:
        return False


def _normalize_tx_orientations(values: list, target_count: int) -> list:
    """Trim TX orientations to the currently rendered TX count when known."""
    if target_count <= 0:
        return values
    if len(values) <= target_count:
        return values
    return values[:target_count]


def _normalize_rx_orientations(values: list, target_count: int, num_tx: int) -> list:
    """Trim or deinterleave RX orientations to the currently rendered RX count."""
    if target_count <= 0:
        return values
    if len(values) <= target_count:
        return values

    if num_tx > 0:
        reduced: list = []
        for rx_idx in range(target_count):
            src_idx = rx_idx * num_tx
            if src_idx < len(values):
                reduced.append(values[src_idx])
        if len(reduced) == target_count:
            return reduced

    return values[:target_count]
