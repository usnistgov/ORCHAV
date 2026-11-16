"""Joined barrier meshes and explicit provenance for unsurveyed dimensions.

Walls remain walls unless OSM explicitly identifies a railing. These are coarse
geometric approximations, including railing spacing, not measured structures.
The functions use local metre coordinates and have no acquisition dependencies.
"""

from __future__ import annotations

import math
import re

import numpy as np

DEFAULT_HEIGHTS = {"wall": 2.0, "retaining_wall": 2.0, "city_wall": 2.0, "fence": 1.5, "hedge": 1.5}


def _metres(value) -> float | None:
    if isinstance(value, bool):
        return None
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d*)?|\.\d+)\s*(m|meters?|metres?|ft|feet)?\s*", str(value), re.I
    )
    if not match:
        return None
    number = float(match[1]) * (0.3048 if (match[2] or "").lower() in {"ft", "feet"} else 1.0)
    return number if math.isfinite(number) and number > 0 else None


def barrier_spec(tags: dict) -> dict | None:
    """Resolve dimensions while retaining which values were assumed."""
    kind = str(tags.get("barrier", "")).strip().lower()
    if kind not in DEFAULT_HEIGHTS:
        return None
    height, width = _metres(tags.get("height")), _metres(tags.get("width"))
    railing = kind == "fence" and str(tags.get("fence_type", "")).lower() in {"railing", "railings"}
    return {
        "barrier": kind,
        "height_m": height if height is not None else DEFAULT_HEIGHTS[kind],
        "height_source": "osm_height" if height is not None else "default",
        "width_m": width if width is not None else (0.08 if railing else 0.3),
        "width_source": "osm_width" if width is not None else "default",
        "construction": "open_railing" if railing else "joined_solid",
        "material": "metal" if kind == "fence" else "wood" if kind == "hedge" else "concrete",
    }


def joined_strip(points: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a connected XY polyline with per-point ground Z and capped miters.

    Adjacent sections share vertices and have no internal end walls. Closed
    rings join across their seam; duplicate consecutive coordinates are ignored.
    Miters are limited to twice the half-width to avoid spikes at sharp turns.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("barrier points must be finite XYZ coordinates")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("barrier dimensions must be positive and finite")
    points = (
        points[np.r_[True, np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > 1e-6]]
        if len(points)
        else points
    )
    if len(points) < 2:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    closed = len(points) > 2 and np.linalg.norm(points[0, :2] - points[-1, :2]) < 1e-6
    if closed:
        points = points[:-1]
    segments = np.diff(np.vstack((points, points[0])) if closed else points, axis=0)[:, :2]
    tangents = segments / np.linalg.norm(segments, axis=1)[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    offsets = []
    for i in range(len(points)):
        if not closed and i == 0:
            offset = normals[0] * width / 2
        elif not closed and i == len(points) - 1:
            offset = normals[-1] * width / 2
        else:
            previous, following = normals[(i - 1) % len(normals)], normals[i % len(normals)]
            bisector = previous + following
            length = np.linalg.norm(bisector)
            direction = bisector / length if length > 1e-6 else following
            distance = min(width, width / (2 * max(float(np.dot(direction, following)), 0.5)))
            offset = direction * distance
        offsets.append(np.r_[offset, 0.0])
    offsets = np.asarray(offsets)
    left, right = points + offsets, points - offsets
    rise = np.array([0, 0, height])
    vertices = np.stack((left, right, left + rise, right + rise), axis=1).reshape(-1, 3)
    faces = []
    for i in range(len(segments)):
        a, b = 4 * i, 4 * ((i + 1) % len(points))
        faces.extend(
            (
                (a, a + 2, b + 2),
                (a, b + 2, b),
                (a + 1, b + 1, b + 3),
                (a + 1, b + 3, a + 3),
                (a + 2, a + 3, b + 3),
                (a + 2, b + 3, b + 2),
            )
        )
    if not closed:
        b = 4 * (len(points) - 1)
        faces.extend(((0, 1, 3), (0, 3, 2), (b, b + 2, b + 3), (b, b + 3, b + 1)))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32)


def barrier_mesh(points: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build a joined solid or an explicitly tagged railing with open gaps."""
    if spec["construction"] != "open_railing":
        return joined_strip(points, spec["width_m"], spec["height_m"])
    points = np.asarray(points, dtype=float)
    rail_size = min(0.06, spec["width_m"], spec["height_m"] / 6)
    pieces = []
    for fraction in (0.35, 0.9):
        raised = points + np.array([0, 0, fraction * spec["height_m"]])
        pieces.append(joined_strip(raised, spec["width_m"], rail_size))
    # Approximate 25 cm picket spacing; bound complexity on long source ways.
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    spacing = max(0.25, float(lengths.sum()) / 4096)
    for a, b, length in zip(points[:-1], points[1:], lengths):
        if length < 1e-6:
            continue
        tangent = (b - a) / length
        count = max(1, int(np.ceil(length / spacing)))
        for fraction in np.arange(count) / count:
            center = a + fraction * (b - a)
            ends = np.array([center - rail_size * tangent / 2, center + rail_size * tangent / 2])
            pieces.append(joined_strip(ends, rail_size, spec["height_m"]))
    # End post, also closes the last gap on open ways.
    if len(points) >= 2 and lengths[-1] > 1e-6:
        tangent = (points[-1] - points[-2]) / lengths[-1]
        pieces.append(
            joined_strip(
                np.array(
                    [points[-1] - rail_size * tangent / 2, points[-1] + rail_size * tangent / 2]
                ),
                rail_size,
                spec["height_m"],
            )
        )
    vertices, faces, offset = [], [], 0
    for v, f in pieces:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)
