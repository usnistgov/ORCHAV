"""Geometry helper utilities used by multiple modules."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from shared.logging import get_logger

from ..model import RenderObjectState, render_state_points
from ..types.render_payloads import LineSetPayload, MeshPayload, PointCloudPayload

logger = get_logger("orchav.utils.geometry")


def build_rotation_matrix_from_angles(
    rotation_deg: Optional[tuple[float, float, float]],
) -> np.ndarray:
    """Build a 3x3 rotation matrix using Mitsuba/Sionna order (Z -> Y -> X)."""
    rotation = rotation_deg or (0.0, 0.0, 0.0)
    yaw, pitch, roll = map(np.radians, rotation)

    rotation_matrix = np.eye(3)

    if abs(roll) > 1e-6:
        rot_x = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
        )
        rotation_matrix = rot_x @ rotation_matrix

    if abs(pitch) > 1e-6:
        rot_y = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
        )
        rotation_matrix = rot_y @ rotation_matrix

    if abs(yaw) > 1e-6:
        rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = rot_z @ rotation_matrix

    return rotation_matrix


def build_transform_matrix(transform_state: dict[str, Any]) -> np.ndarray:
    """Construct a 4x4 transform matrix (scale → rotate → translate)."""
    scale = float(transform_state.get("scale", 1.0))
    rotation = tuple(transform_state.get("rotation", (0.0, 0.0, 0.0)))
    translate_vec = np.asarray(transform_state.get("translate", (0.0, 0.0, 0.0)), dtype=float)

    matrix = np.eye(4)
    scale_matrix = np.diag([scale, scale, scale, 1.0])
    matrix = scale_matrix @ matrix

    rotation_matrix = build_rotation_matrix_from_angles(rotation)
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix
    matrix = rotation_matrix_4x4 @ matrix

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translate_vec
    matrix = translation_matrix @ matrix
    return matrix


def compute_position_after_scale_rotation(
    original_center: np.ndarray, transform_state: dict[str, Any]
) -> np.ndarray:
    """Compute position after applying scale and rotation (before translation)."""
    rotation_matrix = build_rotation_matrix_from_angles(
        tuple(transform_state.get("rotation", (0.0, 0.0, 0.0)))
    )
    scaled_center = float(transform_state.get("scale", 1.0)) * np.asarray(original_center)
    return rotation_matrix @ scaled_center


def mesh_vertices(mesh: Any) -> np.ndarray | None:
    """Return mesh vertices as a float array for renderer-neutral mesh objects."""
    if isinstance(mesh, RenderObjectState) and isinstance(mesh.payload, MeshPayload):
        return np.asarray(mesh.payload.vertices, dtype=float)
    if isinstance(mesh, MeshPayload):
        return np.asarray(mesh.vertices, dtype=float)
    return None


def geometry_bounds(geometry: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return axis-aligned bounds for Open3D-shaped or renderer-neutral geometry."""
    if geometry is None:
        return None

    if hasattr(geometry, "get_axis_aligned_bounding_box"):
        try:
            bbox = geometry.get_axis_aligned_bounding_box()
            min_bound = (
                bbox.get_min_bound()
                if hasattr(bbox, "get_min_bound")
                else getattr(bbox, "min_bound", None)
            )
            max_bound = (
                bbox.get_max_bound()
                if hasattr(bbox, "get_max_bound")
                else getattr(bbox, "max_bound", None)
            )
            bounds = _coerce_bounds(min_bound, max_bound)
            if bounds is not None:
                return bounds
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    if isinstance(geometry, RenderObjectState):
        points = np.asarray(render_state_points(geometry), dtype=float)
        if points.size == 0:
            return None
        points = points.reshape((-1, 3))
        transform = np.asarray(geometry.world_transform.matrix, dtype=float)
        hom = np.column_stack([points, np.ones(len(points), dtype=float)])
        points = (transform @ hom.T).T[:, :3]
        return _points_bounds(points)

    payload = getattr(geometry, "payload", geometry)
    if isinstance(payload, MeshPayload):
        return _points_bounds(payload.vertices)
    if isinstance(payload, (LineSetPayload, PointCloudPayload)):
        return _points_bounds(payload.points)
    return None


def _coerce_bounds(min_bound: Any, max_bound: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate and return a pair of 3D min/max bounds arrays."""
    if min_bound is None or max_bound is None:
        return None
    try:
        min_arr = np.asarray(min_bound, dtype=float).reshape(-1)
        max_arr = np.asarray(max_bound, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if min_arr.size < 3 or max_arr.size < 3:
        return None
    min_arr = min_arr[:3]
    max_arr = max_arr[:3]
    if not np.all(np.isfinite(min_arr)) or not np.all(np.isfinite(max_arr)):
        return None
    return min_arr, max_arr


def _points_bounds(points: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return finite 3D bounds for a point-like array."""
    try:
        arr = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    try:
        arr = arr.reshape((-1, 3))
    except ValueError:
        return None
    finite = np.all(np.isfinite(arr), axis=1)
    if not np.any(finite):
        return None
    arr = arr[finite]
    return arr.min(axis=0), arr.max(axis=0)


def mesh_center(mesh: Any) -> np.ndarray:
    """Return the vertex-mean center for renderer-neutral mesh objects."""
    vertices = mesh_vertices(mesh)
    if vertices is None or vertices.size == 0:
        return np.zeros(3, dtype=float)
    return np.asarray(vertices, dtype=float).reshape((-1, 3)).mean(axis=0)


def paint_mesh_payload(mesh: MeshPayload, color: Any) -> MeshPayload:
    """Return a mesh payload carrying a uniform vertex color."""
    rgb = np.asarray(color, dtype=float).reshape(-1)
    if rgb.size < 3:
        return mesh
    rgb = np.clip(rgb[:3], 0.0, 1.0)
    return replace(mesh, vertex_colors=np.tile(rgb, (len(mesh.vertices), 1)))


def transform_mesh_payload(
    mesh: MeshPayload,
    original_vertices: np.ndarray,
    transform_state: dict[str, Any],
) -> MeshPayload:
    """Return a mesh payload with the XML transform applied to original vertices."""
    vertices = np.asarray(original_vertices, dtype=float).reshape((-1, 3))
    transform_matrix = build_transform_matrix(transform_state)
    hom = np.column_stack([vertices, np.ones(len(vertices), dtype=float)])
    transformed_vertices = (transform_matrix @ hom.T).T[:, :3]

    normals = mesh.normals
    if normals is not None:
        rotation_matrix = build_rotation_matrix_from_angles(
            tuple(transform_state.get("rotation", (0.0, 0.0, 0.0)))
        )
        normals_arr = np.asarray(normals, dtype=float).reshape((-1, 3))
        normals = (rotation_matrix @ normals_arr.T).T
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0.0
        normals[valid] = normals[valid] / lengths[valid, None]

    return replace(mesh, vertices=transformed_vertices, normals=normals)


def apply_transform_to_payload(
    mesh: MeshPayload,
    original_vertices: np.ndarray,
    original_center: np.ndarray,
    transform_state: dict[str, Any],
) -> tuple[MeshPayload, np.ndarray, np.ndarray]:
    """Apply an XML transform to a mesh payload and return updated centers."""
    transformed = transform_mesh_payload(mesh, original_vertices, transform_state)
    position_after_scale_rotation = compute_position_after_scale_rotation(
        original_center, transform_state
    )
    final_center = mesh_center(transformed)
    return transformed, position_after_scale_rotation, final_center
