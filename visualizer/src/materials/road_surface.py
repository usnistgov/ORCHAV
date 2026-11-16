"""Remove coplanar overlap only from the derived, marked-road render payload.

OSM graph directions can duplicate a pavement quad, and adjoining road/path
quads overlap at junctions. One visible surface owns each covered patch. Source
geometry and radio materials remain unchanged; disabling road styling restores
the original payload through the existing building-style cache.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from ..types.render_payloads import MeshPayload
from .haussmann_geometry import _clip

VERSION = "road-surfaces-v1"
_GRID_METRES = 32.0  # Candidate lookup only; it does not quantize geometry.
_AREA_EPSILON = 1e-12
_MAX_RENDER_VERTICES = 600_000
_MAX_GRID_REFERENCES = 250_000
_MAX_INTERSECTION_PAIRS = 200_000


def _area(polygon):
    """Signed horizontal area, using local coordinates for stable products."""
    if len(polygon) < 3:
        return 0.0
    p = polygon[:, :2] - polygon[0, :2]
    return float(np.sum(p[:-1, 0] * p[1:, 1] - p[:-1, 1] * p[1:, 0]) / 2)


def _subtract_quad(polygon, quad):
    """Subtract one convex horizontal quad using the existing half-plane clip."""
    edges = []
    intersection = polygon
    for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        inward = np.array([a[1] - b[1], b[0] - a[0], 0.0])
        boundary = float(a @ inward)
        edges.append((inward, boundary))
        intersection = _clip(intersection, inward, boundary, True)
        if abs(_area(intersection)) <= _AREA_EPSILON:
            return [polygon]
    pieces = []
    remainder = polygon
    for inward, boundary in edges:
        outside = _clip(remainder, inward, boundary, False)
        if abs(_area(outside)) > _AREA_EPSILON:
            pieces.append(outside)
        remainder = _clip(remainder, inward, boundary, True)
        if not len(remainder):
            break
    return pieces


def _cells(bounds, height):
    low = np.floor(bounds[0] / _GRID_METRES).astype(int)
    high = np.floor(bounds[1] / _GRID_METRES).astype(int)
    if int(np.prod(high - low + 1)) > _MAX_GRID_REFERENCES:
        raise ValueError("Road appearance exceeds the spatial lookup budget")
    return [(height, x, y) for x in range(low[0], high[0] + 1) for y in range(low[1], high[1] + 1)]


def visible_road_payload(source: MeshPayload, segments, uvs, *, cache_key: str) -> MeshPayload:
    """Keep a single textured face at each coplanar road/path overlap.

    Only validated, convex, exactly horizontal producer quads participate.
    Sloped roads and bridges at different heights retain their source geometry.
    Motor carriageways own crossings over cycleways and pedestrian paths;
    wider roads and then source order resolve ties without inventing markings.
    """
    from .scene_variation import _MOTOR_ROADS, valid_roads

    uvs = np.asarray(uvs).reshape(-1, 3, 2)
    if uvs.shape != (len(source.triangles), 3, 2) or not np.isfinite(uvs).all():
        raise ValueError("Road overlap removal requires finite per-corner UVs")
    key = f"{VERSION}/{cache_key}"
    if not valid_roads({"total_triangles": len(source.triangles), "segments": segments}, source):
        # Existing partial/custom classifications still receive their atlas;
        # overlap removal requires a verified complete producer partition.
        return replace(source, triangle_uvs=uvs.reshape(-1, 2), cache_key=key)
    corners = source.vertices[source.triangles].astype(np.float64)
    quads = np.concatenate((corners[::2], corners[1::2, 2:3]), axis=1)
    bounds = np.stack((quads[:, :, :2].min(1), quads[:, :, :2].max(1)), axis=1)
    order = sorted(
        range(len(segments)),
        key=lambda i: (
            (
                0
                if segments[i]["highway"] in _MOTOR_ROADS
                else 1 if segments[i]["highway"] == "cycleway" else 2
            ),
            -segments[i]["width_m"],
            i,
        ),
    )
    grid = defaultdict(list)
    seen = set()
    pieces_by_face = {}
    changed = False
    references = 0
    intersection_pairs = 0
    render_vertices = 0
    for index in order:
        quad = quads[index]
        direction = np.roll(quad[:, :2], -1, axis=0) - quad[:, :2]
        turns = direction[:, 0] * np.roll(direction[:, 1], -1) - direction[:, 1] * np.roll(
            direction[:, 0], -1
        )
        if not np.all(quad[:, 2] == quad[0, 2]) or not (np.all(turns > 0) or np.all(turns < 0)):
            continue
        signature = tuple(sorted(map(tuple, quad)))
        if signature in seen:
            pieces_by_face[2 * index] = pieces_by_face[2 * index + 1] = []
            changed = True
            continue
        seen.add(signature)
        cells = _cells(bounds[index], float(quad[0, 2]))
        candidates = sorted({prior for cell in cells for prior in grid[cell]})
        candidates = [
            prior
            for prior in candidates
            if np.all(bounds[prior, 0] < bounds[index, 1])
            and np.all(bounds[index, 0] < bounds[prior, 1])
        ]
        intersection_pairs += len(candidates)
        if intersection_pairs > _MAX_INTERSECTION_PAIRS:
            raise ValueError("Road appearance exceeds the intersection budget")
        for face_index in (2 * index, 2 * index + 1):
            pieces = [corners[face_index]]
            for prior in candidates:
                cover = quads[prior]
                if _area(cover) < 0:
                    cover = cover[::-1]
                retained = [part for polygon in pieces for part in _subtract_quad(polygon, cover)]
                if len(retained) != len(pieces) or any(
                    a is not b for a, b in zip(retained, pieces)
                ):
                    changed = True
                pieces = retained
                if sum(max(0, len(p) - 2) * 3 for p in pieces) > _MAX_RENDER_VERTICES:
                    raise ValueError("Road appearance exceeds the subdivision budget")
                if not pieces:
                    break
            pieces_by_face[face_index] = pieces
            render_vertices += sum(max(0, len(p) - 2) * 3 for p in pieces)
            if changed and render_vertices > _MAX_RENDER_VERTICES:
                raise ValueError("Road appearance exceeds the subdivision budget")
        references += len(cells)
        if references > _MAX_GRID_REFERENCES:
            raise ValueError("Road appearance exceeds the spatial lookup budget")
        for cell in cells:
            grid[cell].append(index)
    if not changed:
        return replace(source, triangle_uvs=uvs.reshape(-1, 2), cache_key=key)

    vertices, texture_coordinates, normals, colors = [], [], [], []
    source_normals = source.normals[source.triangles] if source.normals is not None else None
    source_colors = (
        source.vertex_colors[source.triangles] if source.vertex_colors is not None else None
    )
    for index, triangle in enumerate(corners):
        for polygon in pieces_by_face.get(index, [triangle]):
            for j in range(1, len(polygon) - 1):
                points = polygon[[0, j, j + 1]]
                if not np.array_equal(points, triangle) and abs(_area(points)) <= _AREA_EPSILON:
                    continue
                a, b = triangle[1, :2] - triangle[0, :2], triangle[2, :2] - triangle[0, :2]
                determinant = a[0] * b[1] - a[1] * b[0]
                if np.array_equal(points, triangle):
                    weights = np.eye(3)
                else:
                    xy = points[:, :2] - triangle[0, :2]
                    u = (xy[:, 0] * b[1] - xy[:, 1] * b[0]) / determinant
                    v = (xy[:, 1] * a[0] - xy[:, 0] * a[1]) / determinant
                    weights = np.column_stack((1 - u - v, u, v))
                vertices.extend(points)
                texture_coordinates.extend(weights @ uvs[index])
                if source_normals is not None:
                    normals.extend(weights @ source_normals[index])
                if source_colors is not None:
                    colors.extend(weights @ source_colors[index])
                if len(vertices) > _MAX_RENDER_VERTICES:
                    raise ValueError("Road appearance exceeds the subdivision budget")
    return MeshPayload(
        vertices=np.asarray(vertices, dtype=source.vertices.dtype).reshape(-1, 3),
        triangles=np.arange(len(vertices), dtype=source.triangles.dtype).reshape(-1, 3),
        normals=(
            np.asarray(normals, dtype=source.normals.dtype).reshape(-1, 3)
            if source_normals is not None
            else None
        ),
        vertex_colors=(
            np.asarray(colors, dtype=source.vertex_colors.dtype).reshape(-1, 3)
            if source_colors is not None
            else None
        ),
        triangle_uvs=np.asarray(texture_coordinates, dtype=uvs.dtype).reshape(-1, 2),
        color_source=source.color_source,
        cache_key=key,
    )
