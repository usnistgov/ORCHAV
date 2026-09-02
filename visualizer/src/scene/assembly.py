"""Renderer-agnostic scene building helpers.

Extracts payload-based scene construction logic into reusable functions for
the pygfx notebook and standalone headless interfaces. All outputs use the
backend-neutral payload types from ``types.render_payloads``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from shared.logging import get_logger

from ..materials.catalog import material_preset, resolve_pbr_material
from ..scene.geometry_payload_factory import load_mesh_payload
from ..types.render_payloads import (
    LineSetPayload,
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
    mesh_payload_for_pbr_material,
)
from ..utils.geometry import build_rotation_matrix_from_angles

logger = get_logger("orchav.scene.assembly")

# Default node colors (RGB 0-1)
TX_COLOR = [1.0, 0.2, 0.2]
RX_COLOR = [0.2, 0.2, 1.0]
DEFAULT_NODE_RADIUS = 1.0


# CameraOrbit


@dataclass
class CameraOrbit:
    """Spherical camera parameters for orbit-style viewing."""

    center: np.ndarray  # Look-at target (3,)
    distance: float  # Distance from center
    azimuth_deg: float  # Horizontal angle (0 = +X)
    elevation_deg: float  # Vertical angle (0 = horizon, 90 = top-down)

    def eye_position(self) -> np.ndarray:
        """Compute camera eye position from orbit parameters."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        x = self.center[0] + self.distance * math.cos(el) * math.cos(az)
        y = self.center[1] + self.distance * math.cos(el) * math.sin(az)
        z = self.center[2] + self.distance * math.sin(el)
        return np.array([x, y, z], dtype=np.float64)


# Texture helpers


def build_texture_cache(project_root: Path) -> dict[str, str]:
    """Build a texture filename-to-path mapping from the texture library.

    Args:
        project_root: Project root directory containing ``libraries/textures/``.

    Returns:
        Mapping from lowercase stem (e.g. ``"asphalt"``) to absolute file path.
    """
    tex_dir = project_root / "libraries" / "textures"
    cache: dict[str, str] = {}
    if tex_dir.is_dir():
        for f in tex_dir.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                cache[f.stem.lower()] = str(f)
    return cache


def resolve_texture(
    material_type: str,
    material_id: str | None,
    texture_cache: dict[str, str],
    *,
    allow_material_type_fallback: bool = True,
) -> str | None:
    """Look up a texture path, trying material_id first, then material_type.

    Args:
        material_type: ITU material type string (e.g. ``"concrete"``).
        material_id: Optional material identifier from the scene XML.
        texture_cache: Mapping from ``build_texture_cache()``.
        allow_material_type_fallback: When false, only material-ID matches
            are considered. This prevents a plain material such as ``wood``
            from becoming textured only because ``libraries/textures/wood.png``
            exists.

    Returns:
        Absolute texture file path, or ``None`` if no match.
    """
    if material_id:
        id_key = material_id.lower()
        for prefix in ("mat-", "mat_"):
            if id_key.startswith(prefix):
                id_key = id_key[len(prefix) :]
                break
        material_key = str(material_type or "").lower()
        generic_keys = {material_key, f"itu_{material_key}", f"itu-{material_key}"}
        if id_key not in generic_keys:
            path = texture_cache.get(id_key)
            if path is not None:
                return path
    if not allow_material_type_fallback:
        return None
    return texture_cache.get(material_type.lower())


# Mesh entry -> payload


def _pbr_for_material_type(material_type: str) -> dict[str, Any]:
    """Look up PBR properties for an ITU material type."""
    return material_preset(material_type)


def mesh_entry_to_payload(
    entry: dict[str, Any],
    texture_cache: dict[str, str],
    index: int = 0,
) -> tuple[str, MeshPayload, MaterialPayload] | None:
    """Convert a scene mesh entry dict to payload objects.

    Args:
        entry: Scene mesh entry dict from ``build_scene_from_root()``.
            Expected keys: ``mesh`` (MeshPayload), ``name``, ``material_type``,
            ``material_id``, ``color``, ``visible``.
        texture_cache: Texture lookup from ``build_texture_cache()``.
        index: Entry index used for the geometry name.

    Returns:
        ``(name, mesh_payload, material_payload)`` or ``None`` if the entry
        is invisible or has no mesh.
    """
    mesh = entry.get("mesh")
    if mesh is None or not entry.get("visible", True):
        return None
    if not isinstance(mesh, MeshPayload) or len(mesh.vertices) == 0:
        return None

    name = f"scene_{index}_{entry.get('name', 'mesh')}"
    mat_type = entry.get("material_type", "default")
    mat_id = entry.get("material_id")
    pbr = dict(_pbr_for_material_type(mat_type))

    color = entry.get("color", pbr["color"])
    # Prefer preset-embedded albedo (bundled PBR pack) over explicit
    # material-ID texture matches. Do not fall back to material type here:
    # solid editable materials such as ``wood`` should not become textured
    # only because ``libraries/textures/wood.png`` exists.
    pbr["texture_path"] = pbr.get("texture_path") or resolve_texture(
        mat_type,
        mat_id,
        texture_cache,
        allow_material_type_fallback=False,
    )
    resolved_material = resolve_pbr_material(
        color,
        pbr,
        context=str(entry.get("name", mat_type)),
    )
    pbr = resolved_material.properties_copy(mark_texture_policy=True)

    # Box-project UVs when any texture map is bound and the mesh lacks
    # UVs (the extruded-polygon city-scene case). The notebook static paths
    # use this shared pure conversion rather than SceneService, so neutral UV
    # generation stays at this assembly boundary. Without it pygfx raises
    # "Texture <map> requires geometry.texcoords" in its mesh shader.
    needs_uvs = resolved_material.texture_policy.has_active_maps
    mesh_payload = mesh_payload_for_pbr_material(mesh)
    if needs_uvs and mesh_payload.triangle_uvs is None:
        from .io import _generate_box_projection_uvs

        uv_scale = 1.0 / max(0.1, float(pbr.get("uv_scale_meters", 2.0)))
        auto_uvs = _generate_box_projection_uvs(mesh_payload, scale=uv_scale)
        if auto_uvs is not None and len(auto_uvs) > 0:
            mesh_payload = replace(
                mesh_payload,
                triangle_uvs=np.asarray(auto_uvs, dtype=np.float32),
            )

    # Scene XML vertex colors are often stale material tints. Static scene
    # meshes should use the resolved PBR material color/texture instead.
    return name, mesh_payload, resolved_material.payload


# MPC view model -> payloads


def view_model_to_mpc_payloads(
    view_model: Any,
) -> tuple[LineSetPayload | None, PointCloudPayload | None]:
    """Convert a ViewModel's MPC data to line set and point cloud payloads.

    Args:
        view_model: ``ViewModel`` instance from ``pipeline/core.py``.  Expected
            attributes: ``mpc_points`` (N,3), ``mpc_lines`` (M,2),
            ``mpc_colors`` (M,3), and dedicated bounce arrays.
    Returns:
        ``(line_payload, point_payload)`` — either or both may be ``None``
        if the view model has no MPC data.
    """
    if view_model is None:
        return None, None

    line_payload: LineSetPayload | None = None
    point_payload: PointCloudPayload | None = None

    visibility = view_model.mpc_visibility
    if visibility.effective_paths and view_model.mpc_lines.size > 0:
        line_payload = LineSetPayload(
            points=view_model.mpc_points.astype(np.float64),
            lines=view_model.mpc_lines,
            colors=view_model.mpc_colors.astype(np.float64),
        )

    bounce_points = getattr(view_model, "mpc_bounce_points", None)
    bounce_colors = getattr(view_model, "mpc_bounce_colors", None)

    if (
        visibility.effective_bounce_points
        and bounce_points is not None
        and bounce_points.shape[0] > 0
    ):
        colors = None
        if bounce_colors is not None and bounce_colors.shape[0] == bounce_points.shape[0]:
            colors = bounce_colors.astype(np.float64)
        point_payload = PointCloudPayload(
            points=bounce_points.astype(np.float64),
            colors=colors,
        )

    return line_payload, point_payload


# Sphere payload (pure numpy icosphere)


def _icosphere(subdivisions: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Generate a unit icosphere (vertices, triangles) via subdivision.

    Args:
        subdivisions: Number of subdivision iterations (0 = icosahedron,
            1 = 42 verts, 2 = 162 verts).

    Returns:
        ``(vertices, triangles)`` — float64 and int32 arrays.
    """
    # Golden ratio
    t = (1.0 + math.sqrt(5.0)) / 2.0

    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int32)

    # Normalize to unit sphere
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    # Subdivide
    midpoint_cache: dict[tuple[int, int], int] = {}

    def _midpoint(i1: int, i2: int) -> int:
        """Return the cached normalized midpoint vertex for one edge."""
        key = (min(i1, i2), max(i1, i2))
        if key in midpoint_cache:
            return midpoint_cache[key]
        nonlocal V
        mid = (V[i1] + V[i2]) / 2.0
        mid /= np.linalg.norm(mid)
        idx = len(V)
        V = np.vstack([V, mid[np.newaxis, :]])
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        for tri in F:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = _midpoint(a, b)
            bc = _midpoint(b, c)
            ca = _midpoint(c, a)
            new_faces.extend(
                [
                    [a, ab, ca],
                    [b, bc, ab],
                    [c, ca, bc],
                    [ab, bc, ca],
                ]
            )
        F = np.array(new_faces, dtype=np.int32)
        midpoint_cache.clear()

    return V, F


def _compute_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from triangle mesh."""
    normals = np.zeros_like(vertices)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return normals / norms


def make_sphere_payload(
    position: np.ndarray | Sequence[float],
    radius: float,
    color: Sequence[float],
    subdivisions: int = 2,
) -> MeshPayload:
    """Create a sphere MeshPayload at a given position.

    Uses a pure-numpy icosphere (no Open3D dependency).

    Args:
        position: Center ``[x, y, z]``.
        radius: Sphere radius.
        color: RGB color ``[r, g, b]`` in ``[0, 1]``.
        subdivisions: Icosphere subdivision level (2 = 162 verts).

    Returns:
        MeshPayload for the sphere.
    """
    pos = np.asarray(position, dtype=np.float64)
    verts, tris = _icosphere(subdivisions)
    verts = verts * radius + pos
    normals = _compute_normals(verts, tris)

    n_verts = len(verts)
    colors = np.empty((n_verts, 3), dtype=np.float64)
    colors[:, 0] = color[0]
    colors[:, 1] = color[1]
    colors[:, 2] = color[2]

    return MeshPayload(
        vertices=verts,
        triangles=tris,
        normals=normals,
        vertex_colors=colors,
    )


# Target mesh -> payload


def _load_target_mesh_to_payload(
    mesh_path: str,
    position: np.ndarray,
    scale: float,
    orientation: list[float],
) -> MeshPayload | None:
    """Load and transform a target mesh as a renderer-neutral payload.

    Applies scale, ZYX Euler rotation, and translation.  The mesh is
    centered at its AABB center before transforms (matching Sionna RT).
    """
    path = Path(mesh_path)
    if not path.is_file():
        logger.debug("Target mesh not found: %s", mesh_path)
        return None

    mesh = load_mesh_payload(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        return None

    aabb_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    verts = vertices - aabb_center

    verts *= float(scale)

    rotation_matrix = np.eye(3, dtype=np.float64)
    if orientation and any(abs(a) > 1e-6 for a in orientation):
        yaw, pitch, roll = [float(value) for value in orientation[:3]]
        rotation_matrix = build_rotation_matrix_from_angles((yaw, pitch, roll))
        verts = (rotation_matrix @ verts.T).T

    normals = mesh.normals
    if normals is not None:
        normals_arr = np.asarray(normals, dtype=np.float64).reshape((-1, 3))
        normals = (rotation_matrix @ normals_arr.T).T
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0.0
        normals[valid] = normals[valid] / lengths[valid, None]

    position_arr = np.asarray(position, dtype=np.float64).reshape((3,))
    rotated_aabb_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts += position_arr - rotated_aabb_center

    return MeshPayload(
        vertices=verts,
        triangles=np.asarray(mesh.triangles, dtype=np.int32).copy(),
        normals=None if normals is None else np.asarray(normals, dtype=np.float64),
        vertex_colors=(
            None
            if mesh.vertex_colors is None
            else np.asarray(mesh.vertex_colors, dtype=np.float64).copy()
        ),
        triangle_uvs=(
            None
            if mesh.triangle_uvs is None
            else np.asarray(mesh.triangle_uvs, dtype=np.float64).copy()
        ),
        cache_key=mesh.cache_key,
    )


def target_metadata_to_payload(
    meta: dict[str, Any],
    project_root: Path,
    scenario_root: Path,
) -> tuple[str, MeshPayload, MaterialPayload] | None:
    """Convert a single target metadata dict to payload objects.

    Args:
        meta: Target metadata from ``raw_frame["targets_metadata"][i]``.
            Expected keys: ``name``, ``mesh_file``, ``mesh_directory``,
            ``current_position``, ``scale``, ``orientation``.
        project_root: Project root for resolving ``libraries/`` paths.
        scenario_root: Scenario directory for relative mesh paths.

    Returns:
        ``(name, mesh_payload, material_payload)`` or ``None`` if the
        mesh cannot be loaded.
    """
    mesh_file = meta.get("mesh_file")
    if not mesh_file:
        return None

    name = meta.get("name", "target")
    mesh_directory = meta.get("mesh_directory", "")

    if mesh_directory.startswith("libraries/"):
        mesh_dir = project_root / mesh_directory
    else:
        mesh_dir = scenario_root / mesh_directory
    mesh_path = str(mesh_dir / mesh_file)

    position = np.array(meta.get("current_position", [0, 0, 0]), dtype=np.float64)
    scale = float(meta.get("scale", 1.0))
    orientation = list(meta.get("orientation", [0, 0, 0]))

    payload = _load_target_mesh_to_payload(mesh_path, position, scale, orientation)
    if payload is None:
        logger.debug("Could not load target mesh: %s", mesh_path)
        return None

    # Default skin-like PBR for targets
    pbr = _pbr_for_material_type("default")
    color = pbr["color"]
    alpha = float(pbr.get("alpha", 1.0))

    material = MaterialPayload(
        base_color=(float(color[0]), float(color[1]), float(color[2]), alpha),
        roughness=float(pbr.get("roughness", 0.6)),
        metallic=float(pbr.get("metallic", 0.0)),
        reflectance=float(pbr.get("reflectance", 0.4)),
    )

    return name, payload, material


# Camera orbit computation


def compute_camera_orbit(
    mesh_bboxes: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_tx: np.ndarray,
    frame_rx: np.ndarray,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    center: Sequence[float] | None = None,
) -> CameraOrbit:
    """Compute camera orbit from scene extents and node positions.

    Args:
        mesh_bboxes: List of ``(center, min_bound, max_bound)`` tuples for
            each scene mesh bounding box.
        frame_tx: TX positions array ``(N, 3)``.
        frame_rx: RX positions array ``(M, 3)``.
        azimuth: Camera azimuth in degrees (default 45).
        elevation: Camera elevation in degrees (default 30).
        distance: Camera distance (auto-computed from scene extents if ``None``).
        center: Camera look-at point ``[x, y, z]`` (auto-computed if ``None``).

    Returns:
        CameraOrbit with computed parameters.
    """
    all_points: list[np.ndarray] = []
    for bbox_center, _, _ in mesh_bboxes:
        all_points.append(np.asarray(bbox_center))
    for pos in frame_tx:
        all_points.append(np.asarray(pos))
    for pos in frame_rx:
        all_points.append(np.asarray(pos))

    scene_center = np.mean(all_points, axis=0) if all_points else np.zeros(3)

    default_distance = 100.0
    if mesh_bboxes:
        all_min = np.min([bb[1] for bb in mesh_bboxes], axis=0)
        all_max = np.max([bb[2] for bb in mesh_bboxes], axis=0)
        extent = float(np.linalg.norm(all_max - all_min))
        default_distance = max(extent * 0.8, 10.0)

    return CameraOrbit(
        center=np.array(center, dtype=np.float64) if center is not None else scene_center,
        distance=distance if distance is not None else default_distance,
        azimuth_deg=azimuth if azimuth is not None else 45.0,
        elevation_deg=elevation if elevation is not None else 30.0,
    )
