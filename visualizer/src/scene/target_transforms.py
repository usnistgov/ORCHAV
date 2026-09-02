"""Backend-neutral target transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..model import (
    RenderObjectState,
    offset_render_state_transform,
    render_state_aabb_center,
    set_render_state_points,
    tint_render_state_payload,
    transform_render_state_payload,
)
from ..types.render_payloads import MeshPayload


@dataclass
class TargetGeometryMeta:
    """Cached local-space geometry metadata for one target mesh."""

    scaled_aabb_center: np.ndarray
    rotated_aabb_center: Optional[np.ndarray] = None
    rotation_key: Optional[bytes] = None


def mesh_aabb_center(mesh: RenderObjectState) -> np.ndarray:
    """Return the axis-aligned bounding-box center used by Sionna target placement."""
    if not isinstance(mesh, RenderObjectState):
        raise TypeError(f"{type(mesh).__name__} is not a target render state")
    return render_state_aabb_center(mesh)


def target_mesh_world_center(mesh: RenderObjectState) -> np.ndarray:
    """Return a target mesh AABB center in world coordinates."""
    local_center = mesh_aabb_center(mesh)
    world_center = mesh.world_transform.matrix @ np.append(local_center, 1.0)
    return np.asarray(world_center[:3], dtype=np.float64)


def target_mesh_payload(mesh: RenderObjectState) -> Optional[MeshPayload]:
    """Return a mesh payload from an application-owned target state."""
    if isinstance(mesh, RenderObjectState) and isinstance(mesh.payload, MeshPayload):
        return mesh.payload
    return None


def target_mesh_vertices(mesh: RenderObjectState) -> np.ndarray:
    """Return target mesh vertices from application-owned state."""
    payload = target_mesh_payload(mesh)
    if payload is None:
        raise TypeError(f"{type(mesh).__name__} is not a neutral target mesh")
    return np.asarray(payload.vertices, dtype=np.float64)


def target_mesh_triangles(mesh: RenderObjectState) -> np.ndarray:
    """Return target mesh triangles from application-owned state."""
    payload = target_mesh_payload(mesh)
    if payload is None:
        raise TypeError(f"{type(mesh).__name__} is not a neutral target mesh")
    return np.asarray(payload.triangles, dtype=np.int32)


def target_mesh_vertex_colors(mesh: RenderObjectState) -> Optional[np.ndarray]:
    """Return target vertex colors from application-owned state."""
    payload = target_mesh_payload(mesh)
    if payload is None or payload.vertex_colors is None:
        return None
    return np.asarray(payload.vertex_colors, dtype=np.float64)


def set_target_mesh_vertices(mesh: RenderObjectState, vertices: Any) -> bool:
    """Replace vertices on a mutable neutral target mesh."""
    if not isinstance(mesh, RenderObjectState):
        return False
    set_render_state_points(mesh, np.asarray(vertices, dtype=np.float64))
    return True


def scale_target_mesh(mesh: RenderObjectState, scale: float, center: Any) -> bool:
    """Scale a mutable neutral target mesh around ``center``."""
    if not isinstance(mesh, RenderObjectState):
        return False
    factor = float(scale)
    center_arr = np.asarray(center, dtype=np.float64).reshape(-1)[:3]
    vertices = target_mesh_vertices(mesh)
    set_render_state_points(mesh, (vertices - center_arr) * factor + center_arr)
    return True


def transform_target_mesh_payload(mesh: RenderObjectState, transform: Any) -> bool:
    """Apply a geometry-space transform to a mutable neutral target mesh."""
    if not isinstance(mesh, RenderObjectState):
        return False
    transform_render_state_payload(mesh, transform)
    return True


def translate_target_mesh_by(mesh: RenderObjectState, delta: Any) -> bool:
    """Translate a mutable neutral target mesh by a relative delta."""
    if not isinstance(mesh, RenderObjectState):
        return False
    offset_render_state_transform(mesh, delta)
    return True


def tint_target_mesh_payload(mesh: RenderObjectState, color: Any) -> bool:
    """Apply a uniform color to a mutable neutral target mesh payload."""
    if not isinstance(mesh, RenderObjectState):
        return False
    tint_render_state_payload(mesh, color)
    return True


def _coerce_position(value: Any) -> Optional[np.ndarray]:
    """Return a finite XYZ position array or None."""
    try:
        position = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if position.size < 3 or not np.all(np.isfinite(position[:3])):
        return None
    return position[:3]


def target_entry_anchor_position(entry: dict[str, Any]) -> Optional[np.ndarray]:
    """Return the world anchor used for target labels and orientation frames."""
    for key in ("position", "_target_position"):
        position = _coerce_position(entry.get(key))
        if position is not None:
            return position
    mesh = entry.get("mesh")
    if mesh is None:
        return None
    try:
        return target_mesh_world_center(mesh)
    except TypeError:
        return None


def sionna_ypr_to_xyz_rotation(yaw: float, pitch: float, roll: float) -> tuple[float, float, float]:
    """Map Sionna yaw/pitch/roll radians into X/Y/Z rotation components."""
    return float(roll), float(pitch), float(yaw)


def build_sionna_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build the current target rotation matrix for Sionna yaw/pitch/roll radians."""
    rx, ry, rz = sionna_ypr_to_xyz_rotation(yaw, pitch, roll)
    rotation = np.eye(3, dtype=np.float64)

    if abs(rx) > 1e-6:
        c, s = np.cos(rx), np.sin(rx)
        roll_matrix = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=np.float64,
        )
        rotation = roll_matrix @ rotation

    if abs(ry) > 1e-6:
        c, s = np.cos(ry), np.sin(ry)
        pitch_matrix = np.asarray(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )
        rotation = pitch_matrix @ rotation

    if abs(rz) > 1e-6:
        c, s = np.cos(rz), np.sin(rz)
        yaw_matrix = np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rotation = yaw_matrix @ rotation

    return rotation


def build_sionna_rotation_transform(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a 4x4 rotation-only transform for Sionna yaw/pitch/roll radians."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = build_sionna_rotation_matrix(yaw, pitch, roll)
    return transform


def closest_proper_rotation(values: Any) -> Optional[np.ndarray]:
    """Return the closest proper 3D rotation, discarding scale and shear.

    Transform gizmos can return matrices that contain a uniform scale in the
    rotation block.  Projecting that block with an SVD keeps authoring and live
    preview on the same well-defined rotation conversion path.
    """
    try:
        raw = np.asarray(values, dtype=np.float64)
        if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
            return None
        u_mat, _singular_values, vh_mat = np.linalg.svd(raw)
        rotation = u_mat @ vh_mat
        if np.linalg.det(rotation) < 0.0:
            u_mat[:, -1] *= -1.0
            rotation = u_mat @ vh_mat
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(rotation)):
        return None
    return rotation


def sionna_orientation_from_rotation_matrix(values: Any) -> Optional[np.ndarray]:
    """Convert a rotation-like matrix to Sionna yaw/pitch/roll radians.

    The input is first projected to the closest proper rotation so callers may
    safely pass the upper-left block of a transform that also contains scale.
    """
    rotation = closest_proper_rotation(values)
    if rotation is None:
        return None
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-8:
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    else:
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        roll = 0.0
    return np.asarray([yaw, pitch, roll], dtype=np.float64)


def sionna_orientation_from_transform(values: Any) -> Optional[np.ndarray]:
    """Extract Sionna yaw/pitch/roll radians from a finite 4x4 transform."""
    try:
        transform = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    return sionna_orientation_from_rotation_matrix(transform[:3, :3])


def rotated_aabb_center(vertices: Any, rotation_matrix: np.ndarray) -> Optional[np.ndarray]:
    """Return the AABB center after applying ``rotation_matrix`` to local vertices."""
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.size == 0:
        return None
    rotated = (np.asarray(rotation_matrix, dtype=np.float64) @ verts.T).T
    return (rotated.min(axis=0) + rotated.max(axis=0)) / 2.0


def target_transform_matrix(
    *,
    position: Any,
    mesh_center: Any = None,
    rotation_matrix: Any = None,
    rotated_center: Any = None,
) -> Optional[np.ndarray]:
    """Build a target scene-graph transform matching Sionna AABB placement."""
    try:
        pos_arr = np.asarray(position, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if pos_arr.size != 3:
        return None

    transform = np.eye(4, dtype=np.float32)
    if rotation_matrix is not None:
        try:
            rotation_arr = np.asarray(rotation_matrix, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if rotation_arr.shape != (3, 3):
            return None
        transform[:3, :3] = rotation_arr.astype(np.float32)

        if rotated_center is not None:
            try:
                rotated_arr = np.asarray(rotated_center, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return None
            if rotated_arr.size != 3:
                return None
            transform[:3, 3] = (pos_arr - rotated_arr).astype(np.float32)
            return transform

        if mesh_center is not None:
            try:
                center_arr = np.asarray(mesh_center, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return None
            if center_arr.size != 3:
                return None
            transform[:3, 3] = (pos_arr - rotation_arr @ center_arr).astype(np.float32)
            return transform

    if mesh_center is not None:
        try:
            center_arr = np.asarray(mesh_center, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if center_arr.size != 3:
            return None
        transform[:3, 3] = (pos_arr - center_arr).astype(np.float32)
    else:
        transform[:3, 3] = pos_arr.astype(np.float32)
    return transform


def orientation_metadata(orientation: Any) -> tuple[list[float], list[float]]:
    """Return orientation values as radians plus matching degree display values."""
    if orientation is None:
        radians = [0.0, 0.0, 0.0]
    elif hasattr(orientation, "tolist"):
        radians = [float(value) for value in orientation.tolist()]
    else:
        radians = [float(value) for value in orientation]
    return radians, [float(np.degrees(value)) for value in radians]
