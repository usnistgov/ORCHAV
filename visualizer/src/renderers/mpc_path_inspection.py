"""Renderer-neutral payload for transient MPC path inspection.

The Explorer and selection service own selection identity and animation timing.
Renderers receive only the small, selected-path snapshot defined here; the
frame-wide MPC population remains in the normal bulk render packet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def _readonly_array(values: Any, dtype: np.dtype) -> np.ndarray:
    """Return a detached, contiguous, read-only numeric array."""
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _rgba(values: Any) -> tuple[float, float, float, float]:
    """Normalize a finite RGB/RGBA sequence to clipped RGBA."""
    color = np.asarray(values, dtype=np.float64).reshape(-1)
    if color.size not in {3, 4} or not np.all(np.isfinite(color)):
        raise ValueError("path_color must contain three or four finite values")
    if color.size == 3:
        color = np.concatenate((color, np.ones(1, dtype=np.float64)))
    color = np.clip(color, 0.0, 1.0)
    return tuple(float(value) for value in color)


@dataclass(frozen=True, slots=True)
class MpcPathInspectionSnapshot:
    """Small immutable snapshot for one frame-local canonical MPC path.

    ``frame_token`` is deliberately opaque to the renderer. The transient
    selection service validates it before issuing updates. Arc-length arrays
    are precomputed once so an animation tick is proportional only to the
    number of pulse particles, not to the number of path segments.
    """

    frame_token: Any
    canonical_path_id: int
    points: np.ndarray
    bounce_interaction_types: Optional[np.ndarray] = None
    bounce_colors: Optional[np.ndarray] = None
    bounce_labels: tuple[str, ...] = ()
    path_color: tuple[float, float, float, float] = (0.12, 0.92, 1.0, 1.0)
    segment_vectors: np.ndarray = field(init=False, repr=False)
    segment_lengths: np.ndarray = field(init=False, repr=False)
    cumulative_lengths: np.ndarray = field(init=False, repr=False)
    total_length: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate selected-path alignment and prepare immutable arc data."""
        path_id = int(self.canonical_path_id)
        if path_id < 0:
            raise ValueError("canonical_path_id must be non-negative")
        object.__setattr__(self, "canonical_path_id", path_id)

        points = _readonly_array(self.points, np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            raise ValueError("points must have shape (N, 3) with N >= 2")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")
        object.__setattr__(self, "points", points)

        bounce_count = max(0, len(points) - 2)
        interaction_types = self.bounce_interaction_types
        if interaction_types is not None:
            interaction_types = _readonly_array(interaction_types, np.int32).reshape(-1)
            if interaction_types.size != bounce_count:
                raise ValueError(
                    "bounce_interaction_types must align with the interior path points"
                )
        object.__setattr__(self, "bounce_interaction_types", interaction_types)

        colors = self.bounce_colors
        if colors is not None:
            colors = _readonly_array(colors, np.float32)
            if (
                colors.ndim != 2
                or colors.shape[0] != bounce_count
                or colors.shape[1]
                not in {
                    3,
                    4,
                }
            ):
                raise ValueError("bounce_colors must have shape (bounce_count, 3 or 4)")
            if not np.all(np.isfinite(colors)):
                raise ValueError("bounce_colors must contain only finite values")
            colors = np.clip(colors, 0.0, 1.0)
            colors.setflags(write=False)
        object.__setattr__(self, "bounce_colors", colors)

        labels = tuple(str(label) for label in self.bounce_labels)
        if labels and len(labels) != bounce_count:
            raise ValueError("bounce_labels must be empty or contain one label per bounce")
        if not labels:
            labels = tuple(str(index + 1) for index in range(bounce_count))
        object.__setattr__(self, "bounce_labels", labels)
        object.__setattr__(self, "path_color", _rgba(self.path_color))

        vectors = np.diff(points.astype(np.float64, copy=False), axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(lengths)))
        vectors.setflags(write=False)
        lengths.setflags(write=False)
        cumulative.setflags(write=False)
        object.__setattr__(self, "segment_vectors", vectors)
        object.__setattr__(self, "segment_lengths", lengths)
        object.__setattr__(self, "cumulative_lengths", cumulative)
        object.__setattr__(self, "total_length", float(cumulative[-1]))
