"""Geometric helpers for renderer-facing canonical MPC arrays."""

from __future__ import annotations

import numpy as np


def _compute_path_lengths(
    flat_points: np.ndarray,
    offsets: np.ndarray,
    nodes_per_path: np.ndarray,
    total_points: int,
    num_paths: int,
) -> np.ndarray:
    """Compute geometric path length for each path using vectorized ops.

    For each path, sums the Euclidean distances between consecutive points.
    Cross-path boundary segments are zeroed out so they do not contribute.

    Args:
        flat_points: [total_pts, 3] float32.
        offsets: [P] int32 - start index of each path.
        nodes_per_path: [P] int32 - number of points per path.
        total_points: Total number of points.
        num_paths: Number of paths (P).

    Returns:
        float32 [P] - geometric path length in meters for each path.
    """
    if total_points <= 1:
        return np.zeros(num_paths, dtype=np.float32)

    segments = flat_points[1:] - flat_points[:-1]  # [total_pts-1, 3]
    seg_norms = np.linalg.norm(segments, axis=1)  # [total_pts-1]

    # ``reduceat`` will include the segment between adjacent flattened paths,
    # so zero those boundaries before reducing per path.
    if num_paths > 1:
        last_points = offsets[:-1] + nodes_per_path[:-1] - 1  # [P-1]
        seg_norms[last_points] = 0.0

    path_lengths = np.add.reduceat(seg_norms, offsets)  # [P]
    return path_lengths.astype(np.float32)
