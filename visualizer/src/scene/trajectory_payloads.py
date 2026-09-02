"""Backend-neutral trajectory payload construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..types.render_payloads import LineSetPayload, PointCloudPayload
from ..utils.trajectory_colors import (
    TARGET_TRAJECTORY_PALETTE,
    compute_trajectory_colors,
    compute_trajectory_point_colors,
)


@dataclass(frozen=True)
class TrajectoryNaming:
    """Renderer-provided stable names for trajectory geometries."""

    tx_lines: str
    tx_points: str
    rx_lines: str
    rx_points: str
    target_lines_prefix: str
    target_points_prefix: str


@dataclass(frozen=True)
class NamedTrajectoryPayload:
    """Line and point payloads for one named trajectory."""

    lines_name: str
    points_name: str
    line_payload: Optional[LineSetPayload]
    point_payload: PointCloudPayload


@dataclass(frozen=True)
class TrajectoryPayloadBatch:
    """Payloads to apply plus stale names to remove."""

    payloads: tuple[NamedTrajectoryPayload, ...]
    stale_names: tuple[str, ...] = ()


def sanitize_trajectory_name(name: str) -> str:
    """Return a renderer-safe trajectory name suffix."""
    safe = str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in safe)


def build_trajectory_payloads(
    *,
    kind: str,
    trajectory_data: Mapping[str, Any],
    visualizer: Any,
    naming: TrajectoryNaming,
    color_mode: str = "node_color",
    scalar_range: Optional[tuple[float, float]] = None,
    existing_names: tuple[str, ...] = (),
) -> TrajectoryPayloadBatch:
    """Build named line/point payloads for TX, RX, or target trajectories."""
    if kind == "target":
        return _build_target_trajectory_payloads(
            trajectory_data=trajectory_data,
            visualizer=visualizer,
            naming=naming,
            color_mode=color_mode,
            scalar_range=scalar_range,
            existing_names=existing_names,
        )
    if kind in {"tx", "rx"}:
        return _build_node_trajectory_payloads(
            kind=kind,
            trajectory_data=trajectory_data,
            visualizer=visualizer,
            naming=naming,
            color_mode=color_mode,
            scalar_range=scalar_range,
        )
    return TrajectoryPayloadBatch(payloads=())


def _build_node_trajectory_payloads(
    *,
    kind: str,
    trajectory_data: Mapping[str, Any],
    visualizer: Any,
    naming: TrajectoryNaming,
    color_mode: str,
    scalar_range: Optional[tuple[float, float]],
) -> TrajectoryPayloadBatch:
    """Build the aggregate TX or RX trajectory payload pair.

    TX/RX trajectories share one line-set and point-cloud per kind. Individual
    node coloring is applied by fixed point/segment ranges so scalar color modes
    can still use the shared trajectory color helpers.
    """
    if kind == "tx":
        lines_name = naming.tx_lines
        points_name = naming.tx_points
        default_color = [1.0, 0.0, 0.0]
        positions_key = "tx_positions"
    else:
        lines_name = naming.rx_lines
        points_name = naming.rx_points
        default_color = [0.0, 0.0, 1.0]
        positions_key = "rx_positions"

    node_positions = trajectory_data.get(positions_key, {})
    if not node_positions:
        return TrajectoryPayloadBatch(payloads=(), stale_names=(lines_name, points_name))

    use_individual = (
        color_mode == "node_color"
        and getattr(visualizer, "node_coloring_mode", "per_type") == "individual"
    )
    individual_colors = getattr(visualizer, "individual_node_colors", None) or []
    num_tx = len(getattr(visualizer, "tx_markers", []))

    all_points: list[list[float]] = []
    all_lines: list[list[int]] = []
    all_frames: list[float] = []
    node_seg_ranges: list[tuple[int, int, list[float]]] = []
    node_pt_ranges: list[tuple[int, int, list[float]]] = []
    offset = 0
    seg_offset = 0

    for node_idx, pos_list in node_positions.items():
        sorted_pos = sorted(pos_list, key=lambda p: p[0])
        pts = [[p[1], p[2], p[3]] for p in sorted_pos]
        frames = [p[0] for p in sorted_pos]
        n_points = len(pts)
        if n_points == 0:
            continue

        if use_individual and individual_colors:
            try:
                idx = int(node_idx)
            except (TypeError, ValueError):
                idx = 0
            color_idx = (num_tx + idx) if kind == "rx" else idx
            node_color = (
                individual_colors[color_idx]
                if color_idx < len(individual_colors)
                else default_color
            )
        else:
            node_color = default_color

        all_points.extend(pts)
        all_frames.extend(frames)
        n_segments = n_points - 1
        for local_idx in range(n_segments):
            all_lines.append([offset + local_idx, offset + local_idx + 1])
        node_seg_ranges.append((seg_offset, n_segments, list(node_color)))
        node_pt_ranges.append((offset, n_points, list(node_color)))
        seg_offset += n_segments
        offset += n_points

    if not all_points:
        return TrajectoryPayloadBatch(payloads=(), stale_names=(lines_name, points_name))

    points_arr = np.asarray(all_points, dtype=np.float64)
    frames_arr = np.asarray(all_frames, dtype=np.float64)
    line_colors, point_colors = _trajectory_colors(
        points_arr=points_arr,
        frames_arr=frames_arr,
        lines=all_lines,
        default_color=default_color,
        color_mode=color_mode,
        scalar_range=scalar_range,
        use_fixed_ranges=use_individual and bool(individual_colors),
        node_seg_ranges=node_seg_ranges,
        node_pt_ranges=node_pt_ranges,
    )

    payload = _named_payload(
        lines_name, points_name, points_arr, all_lines, line_colors, point_colors
    )
    return TrajectoryPayloadBatch(payloads=(payload,))


def _build_target_trajectory_payloads(
    *,
    trajectory_data: Mapping[str, Any],
    visualizer: Any,
    naming: TrajectoryNaming,
    color_mode: str,
    scalar_range: Optional[tuple[float, float]],
    existing_names: tuple[str, ...],
) -> TrajectoryPayloadBatch:
    """Build one payload pair per target and report stale target names."""
    target_positions = trajectory_data.get("target_positions", {})
    existing_target_names = tuple(
        name
        for name in existing_names
        if name.startswith(naming.target_lines_prefix)
        or name.startswith(naming.target_points_prefix)
    )
    if not target_positions:
        return TrajectoryPayloadBatch(payloads=(), stale_names=existing_target_names)

    use_material_color = (
        color_mode == "node_color"
        and getattr(visualizer, "node_coloring_mode", "per_type") == "individual"
    )
    target_color_map: dict[str, list[float]] = {}
    if use_material_color:
        for entry in getattr(visualizer, "target_entries", []):
            target_name = entry.get("target_name") or entry.get("node_name")
            color = entry.get("color")
            if target_name and color:
                target_color_map[str(target_name)] = list(color)

    payloads: list[NamedTrajectoryPayload] = []
    seen_names: set[str] = set()
    for idx, (target_name, pos_list) in enumerate(target_positions.items()):
        default_color = (
            target_color_map.get(str(target_name))
            if use_material_color
            else TARGET_TRAJECTORY_PALETTE[idx % len(TARGET_TRAJECTORY_PALETTE)]
        )
        if default_color is None:
            default_color = TARGET_TRAJECTORY_PALETTE[idx % len(TARGET_TRAJECTORY_PALETTE)]

        sorted_pos = sorted(pos_list, key=lambda p: p[0])
        points = [[p[1], p[2], p[3]] for p in sorted_pos]
        frames = [p[0] for p in sorted_pos]
        if not points:
            continue

        lines = [[line_idx, line_idx + 1] for line_idx in range(len(points) - 1)]
        points_arr = np.asarray(points, dtype=np.float64)
        frames_arr = np.asarray(frames, dtype=np.float64)
        safe_name = sanitize_trajectory_name(str(target_name))
        lines_name = f"{naming.target_lines_prefix}{safe_name}"
        points_name = f"{naming.target_points_prefix}{safe_name}"
        seen_names.add(lines_name)
        seen_names.add(points_name)

        line_colors = compute_trajectory_colors(
            points_arr,
            frames_arr,
            lines,
            list(default_color),
            color_mode,
            scalar_range=scalar_range,
        )
        point_colors = compute_trajectory_point_colors(
            points_arr,
            frames_arr,
            list(default_color),
            color_mode,
            scalar_range=scalar_range,
        )
        payloads.append(
            _named_payload(lines_name, points_name, points_arr, lines, line_colors, point_colors)
        )

    stale_names = tuple(name for name in existing_target_names if name not in seen_names)
    return TrajectoryPayloadBatch(payloads=tuple(payloads), stale_names=stale_names)


def _trajectory_colors(
    *,
    points_arr: np.ndarray,
    frames_arr: np.ndarray,
    lines: list[list[int]],
    default_color: list[float],
    color_mode: str,
    scalar_range: Optional[tuple[float, float]],
    use_fixed_ranges: bool,
    node_seg_ranges: list[tuple[int, int, list[float]]],
    node_pt_ranges: list[tuple[int, int, list[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-segment and per-point colors for one trajectory payload."""
    if use_fixed_ranges:
        line_colors = np.empty((len(lines), 3), dtype=np.float64)
        point_colors = np.empty((len(points_arr), 3), dtype=np.float64)
        for seg_start, seg_count, color in node_seg_ranges:
            line_colors[seg_start : seg_start + seg_count] = color
        for pt_start, pt_count, color in node_pt_ranges:
            point_colors[pt_start : pt_start + pt_count] = color
        return line_colors, point_colors

    line_colors = compute_trajectory_colors(
        points_arr,
        frames_arr,
        lines,
        default_color,
        color_mode,
        scalar_range=scalar_range,
    )
    point_colors = compute_trajectory_point_colors(
        points_arr,
        frames_arr,
        default_color,
        color_mode,
        scalar_range=scalar_range,
    )
    return line_colors, point_colors


def _named_payload(
    lines_name: str,
    points_name: str,
    points_arr: np.ndarray,
    lines: list[list[int]],
    line_colors: np.ndarray,
    point_colors: np.ndarray,
) -> NamedTrajectoryPayload:
    """Wrap trajectory arrays in renderer-neutral line and point payloads."""
    line_payload = None
    if lines:
        line_payload = LineSetPayload(
            points=np.asarray(points_arr, dtype=np.float64),
            lines=np.asarray(lines, dtype=np.int32),
            colors=np.asarray(line_colors, dtype=np.float64),
        )
    point_payload = PointCloudPayload(
        points=np.asarray(points_arr, dtype=np.float64),
        colors=np.asarray(point_colors, dtype=np.float64),
    )
    return NamedTrajectoryPayload(
        lines_name=lines_name,
        points_name=points_name,
        line_payload=line_payload,
        point_payload=point_payload,
    )
