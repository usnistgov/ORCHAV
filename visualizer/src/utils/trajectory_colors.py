"""Trajectory color computation helpers.

Pure NumPy functions for computing per-line and per-point trajectory colours.
These are shared by the Open3D and pygfx renderers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .colors import ensure_viridis_lut

# Public API

#: Recognised metric-based colour modes (beyond ``"node_color"``).
METRIC_MODES = ("speed", "altitude", "time", "angular_speed")

#: Default target trajectory colour palette (cycles for multiple targets).
TARGET_TRAJECTORY_PALETTE: list[list[float]] = [
    [0.6, 0.2, 0.8],  # purple
    [0.9, 0.4, 0.1],  # orange
    [0.1, 0.7, 0.7],  # teal
    [0.8, 0.2, 0.5],  # pink
    [0.4, 0.6, 0.2],  # olive
    [0.2, 0.5, 0.8],  # sky blue
    [0.7, 0.7, 0.1],  # yellow
    [0.5, 0.3, 0.1],  # brown
]


def map_scalar_to_colors(
    values: np.ndarray,
    scalar_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Map a 1-D scalar array to viridis RGB colors (Nx3, float64).

    Args:
        values: 1-D array of scalar values.
        scalar_range: Optional (vmin, vmax) for global normalization.
            When provided, values are normalized against this range so that
            multiple trajectories share the same colour scale.  When *None*,
            the local min/max of *values* is used.

    Returns:
        (N, 3) float64 colour array.
    """
    # Geometry, the Nodes colorbar, and renderer HUDs share this exact LUT.
    lut = ensure_viridis_lut()
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if scalar_range is not None:
        vmin, vmax = scalar_range
    else:
        vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin
    scale = max(abs(vmin), abs(vmax), 1e-12)
    if span / scale < 1e-6:
        indices = np.full(values.shape, 0, dtype=np.int32)
    else:
        normalized = (values - vmin) / span
        indices = np.clip((normalized * 255).astype(np.int32), 0, 255)
    return np.asarray(lut[indices], dtype=np.float64)


def compute_trajectory_colors(
    points: np.ndarray,
    frames: np.ndarray,
    lines: list[list[int]],
    default_color: list[float],
    color_mode: str,
    scalar_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Compute per-line-segment colours for a trajectory.

    Args:
        points: (N, 3) float64 array of trajectory positions.
        frames: (N,) float64 array of frame indices per point.
        lines: List of ``[i, j]`` index pairs into *points*.
        default_color: RGB fallback for ``"node_color"`` mode.
        color_mode: One of ``"node_color"``, ``"speed"``, ``"altitude"``,
            ``"time"``, ``"angular_speed"``.
        scalar_range: Optional global (vmin, vmax) for consistent colouring.

    Returns:
        (M, 3) float64 colour array (one row per line segment).
    """
    n_lines = len(lines)
    if n_lines == 0:
        return np.empty((0, 3), dtype=np.float64)

    if color_mode == "node_color" or color_mode not in METRIC_MODES:
        return np.tile(default_color, (n_lines, 1)).astype(np.float64)

    lines_arr = np.array(lines, dtype=np.int32)
    start_idx = lines_arr[:, 0]
    end_idx = lines_arr[:, 1]

    if color_mode == "speed":
        dp = points[end_idx] - points[start_idx]
        dist = np.linalg.norm(dp, axis=1)
        dt = np.abs(frames[end_idx] - frames[start_idx])
        dt = np.where(dt < 1e-12, 1.0, dt)
        values = dist / dt
    elif color_mode == "altitude":
        values = points[start_idx, 2]
    elif color_mode == "angular_speed":
        dp = points[end_idx] - points[start_idx]
        headings = np.arctan2(dp[:, 1], dp[:, 0])
        dh = np.zeros_like(headings)
        if len(headings) > 1:
            dh[1:] = np.abs(np.diff(headings))
            dh[1:] = np.minimum(dh[1:], 2 * np.pi - dh[1:])
        dt = np.abs(frames[end_idx] - frames[start_idx])
        dt = np.where(dt < 1e-12, 1.0, dt)
        values = dh / dt
    else:  # "time"
        values = frames[start_idx]

    return map_scalar_to_colors(values, scalar_range=scalar_range)


def compute_trajectory_point_colors(
    points: np.ndarray,
    frames: np.ndarray,
    default_color: list[float],
    color_mode: str,
    scalar_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Compute per-point colours for a trajectory.

    Args:
        points: (N, 3) float64 positions.
        frames: (N,) float64 frame indices.
        default_color: RGB fallback.
        color_mode: ``"node_color"``, ``"speed"``, ``"altitude"``, ``"time"``,
            or ``"angular_speed"``.
        scalar_range: Optional global (vmin, vmax) for consistent colouring.

    Returns:
        (N, 3) float64 colour array.
    """
    n = len(points)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    if color_mode == "node_color" or color_mode not in METRIC_MODES:
        return np.tile(default_color, (n, 1)).astype(np.float64)

    if color_mode == "speed":
        speed = np.zeros(n, dtype=np.float64)
        if n > 1:
            dp = np.diff(points, axis=0)
            dt = np.abs(np.diff(frames))
            dt = np.where(dt < 1e-12, 1.0, dt)
            seg_speed = np.linalg.norm(dp, axis=1) / dt
            speed[:-1] += seg_speed
            speed[1:] += seg_speed
            speed[1:-1] /= 2.0
        values = speed
    elif color_mode == "altitude":
        values = points[:, 2]
    elif color_mode == "angular_speed":
        ang = np.zeros(n, dtype=np.float64)
        if n > 2:
            dp = np.diff(points, axis=0)
            headings = np.arctan2(dp[:, 1], dp[:, 0])
            dh = np.abs(np.diff(headings))
            dh = np.minimum(dh, 2 * np.pi - dh)
            dt = np.abs(np.diff(frames[:-1]))
            dt = np.where(dt < 1e-12, 1.0, dt)
            seg_ang = dh / dt
            ang[1:-1] = seg_ang
            if len(seg_ang) > 0:
                ang[0] = seg_ang[0]
                ang[-1] = seg_ang[-1]
        values = ang
    else:  # "time"
        values = frames

    return map_scalar_to_colors(values, scalar_range=scalar_range)


def find_position_at_step(pos_list: list, step: int) -> list[float] | None:
    """Find the position closest to the given frame step.

    Args:
        pos_list: List of ``(frame, x, y, z)`` tuples.
        step: Target frame index.

    Returns:
        ``[x, y, z]`` or ``None`` if the list is empty.
    """
    if not pos_list:
        return None
    best = min(pos_list, key=lambda p: abs(p[0] - step))
    return [best[1], best[2], best[3]]
