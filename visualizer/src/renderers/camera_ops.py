"""Renderer-neutral camera intent helpers used by renderer backends."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..types.camera_state import CameraState

_CAMERA_BOUNDS_EXCLUDED_PREFIXES = (
    "mpc_",
    "coverage_",
    "beamforming:",
    "trajectory_",
    "orientation_",
    "target_orientation_",
    "tx_label_",
    "rx_label_",
    "bldg_label_",
    "vm_label_",
    "target_label_",
    "ground_grid",
)
_CAMERA_BOUNDS_EXCLUDED_SUFFIXES = (
    ":label",
    ":work_plane",
)
_CAMERA_BOUNDS_EXCLUDED_FRAGMENTS = (":mobility_control_label_",)


def object_contributes_to_camera_bounds(object_id: str) -> bool:
    """Return whether an applied render object belongs in camera-fit bounds."""
    name = str(object_id)
    return (
        not name.startswith(_CAMERA_BOUNDS_EXCLUDED_PREFIXES)
        and not name.endswith(_CAMERA_BOUNDS_EXCLUDED_SUFFIXES)
        and not any(fragment in name for fragment in _CAMERA_BOUNDS_EXCLUDED_FRAGMENTS)
        and "::label" not in name
    )


def bounds_center_extent(bounds: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(center, extent)`` from a bbox-like object or tuple."""
    if bounds is None:
        return None
    try:
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            center = np.asarray(bounds[0], dtype=np.float64).reshape(-1)[:3]
            extent = np.asarray(bounds[1], dtype=np.float64).reshape(-1)[:3]
        elif hasattr(bounds, "get_center") and hasattr(bounds, "get_extent"):
            center = np.asarray(bounds.get_center(), dtype=np.float64).reshape(-1)[:3]
            extent = np.asarray(bounds.get_extent(), dtype=np.float64).reshape(-1)[:3]
        else:
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    if center.size < 3 or extent.size < 3:
        return None
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
        return None
    return center.astype(np.float64), np.maximum(extent.astype(np.float64), 0.0)


def camera_state_for_overview(
    view: str,
    bounds: Any,
    *,
    fov: float = 60.0,
    distance: Optional[float] = None,
    aspect: float = 16.0 / 9.0,
) -> Optional[CameraState]:
    """Build a portable camera state for a named overview camera intent."""
    parsed = bounds_center_extent(bounds)
    if parsed is None:
        return None
    center, extent = parsed
    view_key = _normalize_view(view)
    if view_key is None:
        return None
    fov_value = float(fov)
    if distance is None:
        distance_value = _compute_camera_distance(extent, fov_value, aspect=aspect, view=view_key)
    else:
        try:
            distance_value = max(0.1, float(distance))
        except (TypeError, ValueError):
            distance_value = _compute_camera_distance(
                extent, fov_value, aspect=aspect, view=view_key
            )
    eye, up = _compute_view_eye_up(view_key, center, distance_value)
    return CameraState(
        eye=(float(eye[0]), float(eye[1]), float(eye[2])),
        lookat=(float(center[0]), float(center[1]), float(center[2])),
        up=(float(up[0]), float(up[1]), float(up[2])),
        fov_deg=fov_value,
    )


def camera_state_for_pov(
    position: Any,
    orientation: Any,
    *,
    axis: str = "forward",
    fov: float = 60.0,
    look_distance: float = 10.0,
) -> Optional[CameraState]:
    """Build a portable POV camera state from entity position/orientation."""
    try:
        eye = np.asarray(position, dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None
    if eye.size < 3 or not np.all(np.isfinite(eye)):
        return None
    yaw, pitch, roll = _normalize_orientation(orientation)
    forward, up = compute_pov_forward_up(yaw, pitch, roll, axis=axis)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-9:
        return None
    forward = forward / forward_norm
    up = _orthogonal_up(forward, up)
    distance = max(1e-3, float(look_distance))
    lookat = eye + forward * distance
    return CameraState(
        eye=(float(eye[0]), float(eye[1]), float(eye[2])),
        lookat=(float(lookat[0]), float(lookat[1]), float(lookat[2])),
        up=(float(up[0]), float(up[1]), float(up[2])),
        fov_deg=float(fov),
    )


def _normalize_view(view: str) -> Optional[str]:
    """Normalize supported overview names and aliases."""
    aliases = {
        "top": "top",
        "side": "side",
        "front": "front",
        "isometric": "isometric",
        "iso": "isometric",
    }
    return aliases.get(str(view).strip().lower())


def _view_plane_extents(extent: np.ndarray, view: str) -> tuple[float, float]:
    """Return horizontal/vertical extents that must fit the requested view."""
    if view == "top":
        return float(extent[0]), float(extent[1])
    if view == "front":
        return float(extent[1]), float(extent[2])
    if view == "side":
        return float(extent[0]), float(extent[2])
    horizontal = max(float(extent[0]), float(extent[1]))
    vertical = max(float(extent[2]), horizontal * 0.6)
    return horizontal, vertical


def _compute_camera_distance(
    extent: np.ndarray,
    fov: float,
    *,
    aspect: float,
    view: str,
) -> float:
    """Compute a conservative camera distance that fits scene bounds."""
    width_axis, height_axis = _view_plane_extents(extent[:3], view)
    width_axis = max(float(width_axis), 1e-3)
    height_axis = max(float(height_axis), 1e-3)
    fov_rad = np.radians(max(10.0, min(120.0, float(fov))))
    aspect = max(float(aspect), 1e-3)
    h_fov_rad = 2.0 * np.arctan(np.tan(fov_rad / 2.0) * aspect)
    fit_height = (height_axis * 0.5) / max(np.tan(fov_rad / 2.0), 1e-3)
    fit_width = (width_axis * 0.5) / max(np.tan(h_fov_rad / 2.0), 1e-3)
    return max(fit_height, fit_width, 10.0) * 1.2


def _compute_view_eye_up(
    view: str, center: np.ndarray, distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return overview camera eye/up vectors for normalized view names."""
    if view == "top":
        return center + np.array([0.0, 0.0, distance]), np.array([0.0, 1.0, 0.0])
    if view == "side":
        return center + np.array([0.0, -distance, distance * 0.2]), np.array([0.0, 0.0, 1.0])
    if view == "front":
        return center + np.array([-distance, 0.0, distance * 0.2]), np.array([0.0, 0.0, 1.0])
    return center + np.array([distance * 0.7, -distance * 0.7, distance * 0.7]), np.array(
        [0.0, 0.0, 1.0]
    )


def _normalize_orientation(orientation: Any) -> tuple[float, float, float]:
    """Coerce yaw/pitch/roll input, defaulting invalid values to zeros."""
    if orientation is None:
        return 0.0, 0.0, 0.0
    try:
        values = np.asarray(orientation, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0
    if values.size < 3 or not np.all(np.isfinite(values[:3])):
        return 0.0, 0.0, 0.0
    return float(values[0]), float(values[1]), float(values[2])


def compute_pov_forward_up(
    yaw: float,
    pitch: float,
    roll: float,
    *,
    axis: str = "forward",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-space POV forward/up vectors for supported axis modes."""
    if axis == "forward":
        rotation_matrix = np.eye(3)
        if abs(roll) > 1e-6:
            roll_matrix = np.array(
                [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
            )
            rotation_matrix = roll_matrix @ rotation_matrix
        if abs(pitch) > 1e-6:
            pitch_matrix = np.array(
                [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
            )
            rotation_matrix = pitch_matrix @ rotation_matrix
        if abs(yaw) > 1e-6:
            yaw_matrix = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )
            rotation_matrix = yaw_matrix @ rotation_matrix
        return rotation_matrix[:, 0].astype(np.float64), rotation_matrix[:, 2].astype(np.float64)

    axis_map: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "x": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "-x": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "y": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "-y": (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "z": (np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
        "-z": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    }
    return axis_map.get(
        axis,
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    )


def _orthogonal_up(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return an up vector orthogonal enough for backend camera state."""
    up_arr = np.asarray(up, dtype=np.float64).reshape(-1)[:3]
    if up_arr.size < 3 or not np.all(np.isfinite(up_arr)):
        up_arr = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_norm = float(np.linalg.norm(up_arr))
    if up_norm < 1e-9:
        up_arr = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up_arr = up_arr / up_norm
    if abs(float(np.dot(forward, up_arr))) > 0.999:
        up_arr = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(forward, up_arr))) > 0.95:
            up_arr = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_arr)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        return up_arr
    right = right / right_norm
    true_up = np.cross(right, forward)
    return true_up / max(float(np.linalg.norm(true_up)), 1e-9)
