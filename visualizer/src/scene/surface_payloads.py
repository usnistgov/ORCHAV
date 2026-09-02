"""Backend-neutral surface mesh payload helpers for overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..types.render_payloads import LineSetPayload, MeshPayload, SurfaceColorSource


@dataclass(frozen=True)
class CoveragePayloads:
    """Coverage mesh and optional isoline payloads."""

    mesh: Optional[MeshPayload]
    isolines: Optional[LineSetPayload]


@dataclass(frozen=True, slots=True)
class BeamformingSurface:
    """One fully prepared renderer-neutral beamforming surface."""

    id: str
    payload: MeshPayload

    def __post_init__(self) -> None:
        """Require the service-provided stable object ID and mesh payload."""
        surface_id = str(self.id).strip()
        if not surface_id:
            raise ValueError("BeamformingSurface id must be non-empty")
        if not isinstance(self.payload, MeshPayload):
            raise TypeError("BeamformingSurface payload must be a MeshPayload")
        object.__setattr__(self, "id", surface_id)


def compute_vertex_normals(vertices: Any, triangles: Any) -> np.ndarray:
    """Compute per-vertex normals from triangle indices."""
    vertices_arr = np.asarray(vertices, dtype=np.float32)
    triangles_arr = np.asarray(triangles, dtype=np.int32)
    normals = np.zeros_like(vertices_arr, dtype=np.float32)
    if vertices_arr.size == 0 or triangles_arr.size == 0:
        return normals

    v0 = vertices_arr[triangles_arr[:, 0]]
    v1 = vertices_arr[triangles_arr[:, 1]]
    v2 = vertices_arr[triangles_arr[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for index in range(3):
        np.add.at(normals, triangles_arr[:, index], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)
    return normals


def build_coverage_payloads(view_model: Any) -> CoveragePayloads:
    """Build coverage mesh and isoline payloads from a ViewModel-like object."""
    coverage_vertices = getattr(view_model, "coverage_vertices", None)
    if not getattr(view_model, "show_coverage", False) or coverage_vertices is None:
        return CoveragePayloads(mesh=None, isolines=None)

    vertices = np.asarray(coverage_vertices, dtype=np.float32)
    triangles = np.asarray(getattr(view_model, "coverage_triangles", []), dtype=np.int32)
    colors = np.asarray(getattr(view_model, "coverage_colors", []), dtype=np.float32)
    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
        return CoveragePayloads(mesh=None, isolines=None)

    mesh = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        normals=compute_vertex_normals(vertices, triangles),
        vertex_colors=colors if colors.shape[0] == vertices.shape[0] else None,
        color_source=SurfaceColorSource.VERTEX,
    )
    return CoveragePayloads(mesh=mesh, isolines=build_coverage_isolines_payload(view_model))


def build_coverage_isolines_payload(view_model: Any) -> Optional[LineSetPayload]:
    """Build optional coverage isoline payload from a ViewModel-like object."""
    points = getattr(view_model, "coverage_isoline_points", None)
    lines = getattr(view_model, "coverage_isoline_lines", None)
    if points is None or lines is None:
        return None

    points_arr = np.asarray(points, dtype=np.float32)
    lines_arr = np.asarray(lines, dtype=np.int32)
    if points_arr.shape[0] == 0 or lines_arr.shape[0] == 0:
        return None

    colors = getattr(view_model, "coverage_isoline_colors", None)
    colors_arr = None if colors is None else np.asarray(colors, dtype=np.float32)
    return LineSetPayload(points=points_arr, lines=lines_arr, colors=colors_arr)
