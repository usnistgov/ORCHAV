"""Deterministic camera-only motion; scientific frames are never interpolated."""

from __future__ import annotations

import numpy as np
import pylinalg as la

from ..types.camera_state import CameraState


def validate_pose(state: CameraState) -> CameraState:
    """Reject malformed saved poses before changing any camera state."""
    if not isinstance(state, CameraState):
        raise ValueError("Camera is not ready")
    vectors = np.asarray([state.eye, state.lookat, state.up], dtype=float)
    if vectors.shape != (3, 3) or not np.isfinite(vectors).all():
        raise ValueError("Camera coordinates must be three finite values")
    if not 1 <= state.fov_deg <= 179:
        raise ValueError("Camera FOV must be between 1 and 179 degrees")
    if np.linalg.norm(vectors[0] - vectors[1]) < 1e-6 or np.linalg.norm(vectors[2]) < 1e-6:
        raise ValueError("Camera direction and up vector must be nonzero")
    return state


def rotation(state: CameraState) -> np.ndarray:
    """Return a right-handed camera basis, including safe near-pole handling."""
    back = np.asarray(state.eye) - state.lookat
    back = back / np.linalg.norm(back)
    up = np.asarray(state.up, dtype=float)
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-6:
        up = np.eye(3)[np.argmin(abs(back))]
        right = np.cross(up, back)
    right /= np.linalg.norm(right)
    return la.quat_from_mat(np.column_stack([right, np.cross(back, right), back]))


def interpolate(start: CameraState, end: CameraState, fraction: float) -> CameraState:
    """Ease position, orientation and FOV along a shortest-rotation camera move."""
    t = float(np.clip(fraction, 0, 1))
    if t == 0:
        return start
    if t == 1:
        return end
    t = t * t * (3 - 2 * t)
    q0, q1 = rotation(start), rotation(end)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        basis = (1 - t) * q0 + t * q1
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        basis = (np.sin((1 - t) * angle) * q0 + np.sin(t * angle) * q1) / np.sin(angle)
    basis /= np.linalg.norm(basis)
    eye = (1 - t) * np.asarray(start.eye) + t * np.asarray(end.eye)
    distance = (1 - t) * np.linalg.norm(np.subtract(start.eye, start.lookat)) + t * np.linalg.norm(
        np.subtract(end.eye, end.lookat)
    )
    lookat = eye + la.vec_transform_quat([0, 0, -distance], basis)
    return CameraState(
        tuple(eye),
        tuple(lookat),
        tuple(la.vec_transform_quat([0, 1, 0], basis)),
        (1 - t) * start.fov_deg + t * end.fov_deg,
    )


def translated(state: CameraState, delta) -> CameraState:
    """Translate a camera without changing its framing."""
    return CameraState(
        tuple(np.asarray(state.eye) + delta),
        tuple(np.asarray(state.lookat) + delta),
        state.up,
        state.fov_deg,
    )


def orbit(state: CameraState, center, degrees: float) -> CameraState:
    """Orbit about the scene's Z-up axis at fixed elevation and radius."""
    transform = la.quat_from_axis_angle([0, 0, 1], np.radians(degrees))
    center = np.asarray(center)
    return CameraState(
        tuple(center + la.vec_transform_quat(np.asarray(state.eye) - center, transform)),
        tuple(center + la.vec_transform_quat(np.asarray(state.lookat) - center, transform)),
        tuple(la.vec_transform_quat(state.up, transform)),
        state.fov_deg,
    )
