"""Coplanar render-only bay subdivision for the Haussmann appearance preset.

Source mesh arrays remain immutable. Clipping existing triangles into vertical
bays preserves their planes, winding and total area; no balcony, roof or other
scattering surface is added. The original payload is restored by building_style.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ..types.render_payloads import MeshPayload
from .haussmann_textures import (
    FLOORS,
    ORIGINAL_SLOT,
    ROOF_METRES,
    ROOF_REPEATS,
    STONE_SLOT,
    VERSION,
    tile_uv,
)

MAX_RENDER_VERTICES = 600_000


class SubdivisionBudgetExceeded(ValueError):
    """Deterministic render-only size rejection, safe to cache for unchanged inputs."""


def _clip(polygon, tangent, boundary, greater):
    """Sutherland–Hodgman clip against one vertical bay boundary."""
    if not len(polygon):
        return polygon
    distances = polygon @ tangent - boundary
    inside = distances >= 0 if greater else distances <= 0
    if inside.all():
        return polygon
    if not inside.any():
        return np.empty((0, 3))
    result = []
    previous = polygon[-1]
    previous_distance = distances[-1]
    previous_inside = inside[-1]
    for point, distance, point_inside in zip(polygon, distances, inside, strict=True):
        if point_inside != previous_inside:
            fraction = previous_distance / (previous_distance - distance)
            result.append(previous + fraction * (point - previous))
        if point_inside:
            result.append(point)
        previous, previous_distance, previous_inside = point, distance, point_inside
    return np.asarray(result)


def _triangle_cross(triangle):
    """Single-triangle cross product without NumPy's batched-axis dispatch."""
    a, b = triangle[1] - triangle[0], triangle[2] - triangle[0]
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )


def facade_payload(source: MeshPayload, parts, identity: str, *, exclusions=()) -> MeshPayload:
    """Fit ~3 m bays and a height-matched floor family to classified walls."""
    from .scene_variation import valid_parts

    count = sum(part[1] for part in parts)
    if not valid_parts(parts, count) or count > len(source.triangles):
        raise ValueError("Haussmann appearance requires a valid building partition")
    digest = hashlib.sha256(f"{VERSION}|{identity}|{json.dumps(parts)}|{exclusions}".encode())
    digest.update(source.vertices.tobytes())
    digest.update(source.triangles.tobytes())
    if source.normals is not None:
        digest.update(source.normals.tobytes())
    vertices, normals, uvs = [], [], []
    corners = source.vertices[source.triangles].astype(np.float64)
    corner_normals = source.normals[source.triangles] if source.normals is not None else None
    excluded = set(exclusions)

    def emit(triangle, uv, original_normals=None):
        normal = _triangle_cross(triangle)
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            return
        vertices.extend(triangle)
        normals.extend([normal / length] * 3 if original_normals is None else original_normals)
        uvs.extend(uv)
        if len(vertices) > MAX_RENDER_VERTICES:
            raise SubdivisionBudgetExceeded(
                "Haussmann appearance exceeds the render subdivision budget"
            )

    def flat(index, slot):
        emit(
            corners[index],
            np.tile(tile_uv(slot, 0.5, 0.5), (3, 1)),
            None if corner_normals is None else corner_normals[index],
        )

    def roof(triangle, slot):
        """Fit slate at physical scale, splitting UV repeats on the existing plane."""
        normal = _triangle_cross(triangle)
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            return
        normal /= length
        # Undersides are not roofs.
        if normal[2] < 0:
            emit(triangle, np.tile(tile_uv(STONE_SLOT, 0.5, 0.5), (3, 1)))
            return
        tangent = np.cross((0, 0, 1), normal)
        tangent_length = np.linalg.norm(tangent)
        tangent = tangent / tangent_length if tangent_length > 1e-8 else np.array([1.0, 0, 0])
        uphill = np.cross(normal, tangent)
        projected = np.column_stack((triangle @ tangent, triangle @ uphill))
        tile_metres = np.multiply(ROOF_METRES, ROOF_REPEATS)
        first = np.floor(projected.min(axis=0) / tile_metres).astype(int)
        last = np.ceil(projected.max(axis=0) / tile_metres).astype(int)
        if np.prod(last - first) > 4096:
            raise SubdivisionBudgetExceeded("Haussmann roof exceeds the render subdivision budget")
        for x in range(first[0], last[0]):
            strip = _clip(triangle, tangent, x * tile_metres[0], True)
            strip = _clip(strip, tangent, (x + 1) * tile_metres[0], False)
            if len(strip) < 3:
                continue
            for y in range(first[1], last[1]):
                polygon = _clip(strip, uphill, y * tile_metres[1], True)
                polygon = _clip(polygon, uphill, (y + 1) * tile_metres[1], False)
                for i in range(1, len(polygon) - 1):
                    tri = polygon[[0, i, i + 1]]
                    u = np.clip(tri @ tangent / tile_metres[0] - x, 0, 1)
                    v = np.clip(tri @ uphill / tile_metres[1] - y, 0, 1)
                    emit(tri, tile_uv(slot, u, v))

    for first, size in parts:
        faces = corners[first : first + size]
        low, high = faces.min(axis=(0, 1)), faces.max(axis=(0, 1))
        height = high[2] - low[2]
        if (first, size) in excluded or height < 5 or height > 36:
            for index in range(first, first + size):
                flat(index, STONE_SLOT)
            continue
        seed = hashlib.sha256(f"{identity}|{low.round(3)}|{high.round(3)}".encode()).digest()[0]
        style = seed % 3
        floor_index = int(np.argmin(abs(np.array(FLOORS) - height / 3.4)))
        slot = style * len(FLOORS) + floor_index
        cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        wall = (lengths > 1e-10) & (abs(cross[:, 2]) < 0.15 * lengths)
        for triangle in faces[~wall]:
            roof(triangle, 18 + style)
        ids = np.flatnonzero(wall)
        if not len(ids):
            continue
        unit = cross[ids] / lengths[ids, None]
        keys = np.column_stack((unit, np.einsum("ij,ij->i", unit, faces[ids, 0])))
        _, groups = np.unique(keys.round(4), axis=0, return_inverse=True)
        for group in range(int(groups.max()) + 1):
            selected = ids[groups == group]
            tangent = np.cross((0, 0, 1), cross[selected[0]])
            tangent /= np.linalg.norm(tangent)
            horizontal = faces[selected] @ tangent
            minimum, maximum = float(horizontal.min()), float(horizontal.max())
            width = maximum - minimum
            if width <= 1e-8:
                for index in selected:
                    flat(first + index, STONE_SLOT)
                continue
            bays = max(1, min(64, round(width / 3.0)))
            bay_width = width / bays
            for triangle in faces[selected]:
                projection = triangle @ tangent
                start = max(0, int(np.floor((projection.min() - minimum) / bay_width)))
                stop = min(bays, int(np.ceil((projection.max() - minimum) / bay_width)))
                for bay in range(start, stop):
                    left = minimum + bay * bay_width
                    polygon = _clip(triangle, tangent, left, True)
                    polygon = _clip(polygon, tangent, left + bay_width, False)
                    for i in range(1, len(polygon) - 1):
                        tri = polygon[[0, i, i + 1]]
                        x = np.clip((tri @ tangent - left) / bay_width, 0, 1)
                        y = np.clip((tri[:, 2] - low[2]) / height, 0, 1)
                        emit(tri, tile_uv(slot, x, y))
    for index in range(count, len(corners)):
        flat(index, ORIGINAL_SLOT)
    return MeshPayload(
        vertices=np.asarray(vertices, dtype=source.vertices.dtype).reshape(-1, 3),
        triangles=np.arange(len(vertices), dtype=np.int32).reshape(-1, 3),
        normals=np.asarray(normals, dtype=np.float32).reshape(-1, 3),
        triangle_uvs=np.asarray(uvs, dtype=np.float32).reshape(-1, 2),
        cache_key=f"{VERSION}/{digest.hexdigest()}",
    )
