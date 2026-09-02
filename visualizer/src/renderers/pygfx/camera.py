"""Camera helpers and camera-state controls for the pygfx renderer.

The shared camera helpers consume an Open3D-like bounds object. ``SceneBounds``
adapts pygfx/native numpy bounds to that small API without importing Open3D.
``PygfxCameraMixin`` owns renderer-neutral camera state, follow/look-at updates,
and pygfx quaternion conversions.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ...types.camera_state import CameraState
from ..camera_ops import bounds_center_extent, camera_state_for_overview, camera_state_for_pov

__all__ = ["PygfxCameraMixin", "SceneBounds"]

logger = logging.getLogger(__name__)


def _pygfx_overview_distance_scale(aspect: float) -> float:
    """Return the distance scale that maps vertical-FOV fit math to pygfx FOV."""
    try:
        aspect_value = max(float(aspect), 1e-3)
    except (TypeError, ValueError):
        aspect_value = 16.0 / 9.0
    return 0.5 * (1.0 + aspect_value)


def _overview_view_vector_up(view: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the camera offset vector and up vector for an overview preset."""
    view_key = str(view).strip().lower()
    if view_key == "top":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), np.array(
            [0.0, 1.0, 0.0], dtype=np.float64
        )
    if view_key == "side":
        return np.array([0.0, -1.0, 0.2], dtype=np.float64), np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
    if view_key == "front":
        return np.array([-1.0, 0.0, 0.2], dtype=np.float64), np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
    return np.array([0.7, -0.7, 0.7], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _pygfx_projection_tangents(fov: float, aspect: float) -> tuple[float, float]:
    """Return horizontal and vertical half-angle tangents for pygfx's extent FOV."""
    fov_rad = np.radians(max(10.0, min(120.0, float(fov))))
    aspect_value = max(float(aspect), 1e-3)
    extent_tan = np.tan(fov_rad / 2.0)
    tan_y = max(2.0 * extent_tan / (1.0 + aspect_value), 1e-3)
    tan_x = max(tan_y * aspect_value, 1e-3)
    return float(tan_x), float(tan_y)


def _pygfx_overview_fit_distance(
    extent: Any,
    *,
    fov: float,
    aspect: float,
    view: str,
) -> Optional[float]:
    """Return an auto-fit overview distance that accounts for full bbox depth."""
    try:
        extent_arr = np.asarray(extent, dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None
    if extent_arr.size < 3 or not np.all(np.isfinite(extent_arr)):
        return None

    view_vec, up = _overview_view_vector_up(view)
    view_norm = float(np.linalg.norm(view_vec))
    if view_norm < 1e-6:
        return None

    forward = -view_vec / view_norm
    up_norm = float(np.linalg.norm(up))
    up = up / up_norm if up_norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right_norm = 1.0
    right = right / right_norm
    true_up = np.cross(right, forward)
    true_up = true_up / max(float(np.linalg.norm(true_up)), 1e-6)

    tan_x, tan_y = _pygfx_projection_tangents(fov, aspect)
    half = np.maximum(extent_arr, 1e-3) * 0.5
    required_distance = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
                x = float(np.dot(corner, right))
                y = float(np.dot(corner, true_up))
                z = float(np.dot(corner, forward))
                required_distance = max(
                    required_distance,
                    abs(x) / tan_x - z,
                    abs(y) / tan_y - z,
                    -z + 1.0,
                )

    return max(0.1, required_distance * 1.2 / view_norm, 10.0)


class SceneBounds:
    """Minimal bounding-box adapter with the camera-helper bounds API."""

    def __init__(self, min_bound: np.ndarray, max_bound: np.ndarray) -> None:
        """Store min/max corners as float64 arrays for camera math."""
        self.min_bound = np.asarray(min_bound, dtype=np.float64)
        self.max_bound = np.asarray(max_bound, dtype=np.float64)

    def get_center(self) -> np.ndarray:
        """Return the midpoint used by overview and minimap cameras."""
        return (self.min_bound + self.max_bound) * 0.5

    def get_extent(self) -> np.ndarray:
        """Return non-negative side lengths for camera distance fitting."""
        return np.maximum(self.max_bound - self.min_bound, 0.0)

    def get_min_bound(self) -> np.ndarray:
        """Return the lower world-space corner."""
        return self.min_bound

    def get_max_bound(self) -> np.ndarray:
        """Return the upper world-space corner."""
        return self.max_bound


class PygfxCameraMixin:
    """Mixin providing pygfx camera state and follow/look-at operations."""

    def get_camera_state(self) -> Optional[CameraState]:
        """Return the current renderer-neutral camera state."""
        if not self._initialized or self._camera is None:
            return self._camera_state
        try:
            local = self._camera.local
            eye = tuple(float(x) for x in local.position)
            target = (
                getattr(self._controller, "target", None) if self._controller is not None else None
            )
            if target is not None:
                target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
                if target_arr.size >= 3 and np.all(np.isfinite(target_arr[:3])):
                    lookat = tuple(float(x) for x in target_arr[:3])
                else:
                    rot = np.array(local.rotation)
                    fwd = self._quat_forward(rot)
                    look_distance = self._fallback_camera_look_distance(eye)
                    lookat = tuple(float(eye[i] + fwd[i] * look_distance) for i in range(3))
            else:
                rot = np.array(local.rotation)
                fwd = self._quat_forward(rot)
                look_distance = self._fallback_camera_look_distance(eye)
                lookat = tuple(float(eye[i] + fwd[i] * look_distance) for i in range(3))
            fov = float(getattr(self._camera, "fov", 60.0))
            world = getattr(self._camera, "world", None)
            up_raw = getattr(world, "reference_up", None) if world is not None else None
            up_arr = np.asarray(up_raw, dtype=np.float64).reshape(-1)[:3]
            if up_arr.size < 3 or not np.all(np.isfinite(up_arr)):
                rot = np.array(local.rotation)
                up_arr = self._quat_up(rot)
            up_norm = float(np.linalg.norm(up_arr))
            if up_norm < 1e-6:
                up_arr = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                up_arr = up_arr / up_norm
            up = tuple(float(x) for x in up_arr)
            self._camera_state = CameraState(eye=eye, lookat=lookat, up=up, fov_deg=fov)
        except (AttributeError, TypeError, ValueError, IndexError):
            pass
        return self._camera_state

    def _fallback_camera_look_distance(self, eye: tuple[float, float, float]) -> float:
        """Return a finite look distance for controllers without a target."""
        distance = float(getattr(self, "_last_camera_look_distance", 0.0) or 0.0)
        if distance > 1e-6 and np.isfinite(distance):
            return distance
        state = self._camera_state
        if state is not None:
            try:
                distance = float(
                    np.linalg.norm(
                        np.asarray(eye, dtype=np.float64)
                        - np.asarray(state.lookat, dtype=np.float64)
                    )
                )
            except (TypeError, ValueError):
                distance = 0.0
        return distance if distance > 1e-6 and np.isfinite(distance) else 1.0

    def set_camera_state(self, state: CameraState) -> bool:
        """Apply a renderer-neutral camera state."""
        if not self._initialized or not isinstance(state, CameraState):
            return False

        eye_arr = np.asarray(state.eye, dtype=np.float32)
        look_arr = np.asarray(state.lookat, dtype=np.float32)
        up_arr = np.asarray(state.up, dtype=np.float32)
        if not np.all(np.isfinite(eye_arr)) or not np.all(np.isfinite(look_arr)):
            return False
        if up_arr.size < 3 or not np.all(np.isfinite(up_arr[:3])):
            up_arr = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        forward = look_arr - eye_arr
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-6:
            return False
        forward = forward / forward_norm
        up_arr = up_arr[:3]
        up_norm = float(np.linalg.norm(up_arr))
        if up_norm < 1e-6:
            up_arr = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            up_norm = 1.0
        up_arr = up_arr / up_norm
        if abs(float(np.dot(up_arr, forward))) > 0.999:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(fallback, forward))) > 0.95:
                fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            up_arr = fallback

        try:
            if hasattr(self._camera, "fov"):
                self._camera.fov = float(state.fov_deg)
            local = getattr(self._camera, "local", None)
            if local is not None and hasattr(local, "position"):
                local.position = tuple(float(x) for x in eye_arr)
            world = getattr(self._camera, "world", None)
            if world is not None and hasattr(world, "reference_up"):
                world.reference_up = tuple(float(x) for x in up_arr)
            if hasattr(self._camera, "look_at"):
                self._camera.look_at(tuple(float(x) for x in look_arr))
            if self._controller is not None and hasattr(self._controller, "target"):
                self._controller.target = tuple(float(x) for x in look_arr)
                if hasattr(self._controller, "distance"):
                    self._controller.distance = float(np.linalg.norm(eye_arr - look_arr))
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.warning("PygfxRenderer: set_camera failed: %s", exc)
            return False

        self._camera_state = CameraState(
            eye=tuple(float(x) for x in state.eye),
            lookat=tuple(float(x) for x in state.lookat),
            up=(float(up_arr[0]), float(up_arr[1]), float(up_arr[2])),
            fov_deg=float(state.fov_deg),
        )
        self._last_camera_look_distance = float(np.linalg.norm(eye_arr - look_arr))
        # Only steal focus for interactive (user-initiated) camera changes.
        # During animation, _focus_canvas() calls activateWindow() which on
        # Windows deactivates the control panel window and swallows button
        # clicks (play/pause becomes unresponsive in follow/POV mode).
        if not getattr(self.visualizer, "animation_running", False):
            self._focus_canvas()
        self.request_redraw()
        return True

    def set_overview_camera(
        self,
        view: str,
        bounds: Any,
        fov: float = 60.0,
        distance: Optional[float] = None,
    ) -> bool:
        """Apply a named overview camera using renderer-neutral camera helpers."""
        aspect = self._viewport_aspect()
        if distance is None:
            parsed = bounds_center_extent(bounds)
            if parsed is not None:
                _, extent = parsed
                distance = _pygfx_overview_fit_distance(
                    extent,
                    fov=float(fov),
                    aspect=aspect,
                    view=view,
                )
        state = camera_state_for_overview(
            view,
            bounds,
            fov=float(fov),
            distance=distance,
            aspect=aspect,
        )
        return False if state is None else self.set_camera_state(state)

    def adjust_overview_distance(
        self,
        distance: float,
        *,
        fov: float = 60.0,
        aspect: float = 16.0 / 9.0,
        view: str = "isometric",
        extent: Any = None,
        **_kwargs: Any,
    ) -> float:
        """Adjust controller-computed overview fit distance for pygfx FOV semantics."""
        fit_distance = _pygfx_overview_fit_distance(
            extent,
            fov=fov,
            aspect=aspect,
            view=view,
        )
        scaled_distance = max(0.1, float(distance)) * _pygfx_overview_distance_scale(aspect)
        if fit_distance is None:
            return scaled_distance
        return max(scaled_distance, fit_distance)

    def focus_camera(self, target_position: Any) -> bool:
        """Focus the camera on a target while clearing follow state."""
        self.reset_follow_state()
        return self.set_lookat(np.asarray(target_position, dtype=np.float64))

    def set_pov_camera(
        self,
        position: Any,
        orientation: Any,
        axis: str,
        *,
        defer_redraw: bool = False,
    ) -> bool:
        """Apply an entity point-of-view camera from position/orientation data."""
        state = camera_state_for_pov(position, orientation, axis=axis)
        return False if state is None else self.set_camera_state(state)

    def update_follow_camera(self, target_position: Any) -> bool:
        """Move follow mode's look-at point while preserving camera offset."""
        return self.set_lookat(np.asarray(target_position, dtype=np.float64))

    def _viewport_aspect(self) -> float:
        """Return current viewport aspect with a stable fallback."""
        width = float(getattr(self, "_width", 0.0))
        height = float(getattr(self, "_height", 0.0))
        if width > 1.0 and height > 1.0:
            return width / height
        return 16.0 / 9.0

    def set_lookat(self, lookat: np.ndarray) -> bool:
        """Move the look-at point while preserving the current eye offset."""
        if not self._initialized:
            return False
        current = self.get_camera_state()
        if current is None:
            return False
        eye = np.asarray(current.eye, dtype=np.float64)
        old_lookat = np.asarray(current.lookat, dtype=np.float64)
        new_lookat = np.asarray(lookat, dtype=np.float64)
        offset = eye - old_lookat
        offset_norm = float(np.linalg.norm(offset))
        if not np.isfinite(offset_norm) or offset_norm < 1e-3:
            offset_norm = max(float(self._last_camera_look_distance), 2.0)
            fallback = eye - new_lookat
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm < 1e-9:
                fallback = np.array([0.0, -1.0, 0.35], dtype=np.float64)
                fallback_norm = float(np.linalg.norm(fallback))
            offset = fallback / max(fallback_norm, 1e-9) * offset_norm
        new_eye = new_lookat + offset
        ok = self.set_camera_state(
            CameraState(
                eye=tuple(new_eye.tolist()),
                lookat=tuple(new_lookat.tolist()),
                up=current.up,
                fov_deg=current.fov_deg,
            )
        )
        if ok and self._controller is not None and hasattr(self._controller, "target"):
            self._controller.target = tuple(float(x) for x in new_lookat)
            self._follow_target_lookat = new_lookat.copy()
        return ok

    def reset_follow_state(self) -> None:
        """Reset follow bookkeeping to the current camera state."""
        self._camera_state = self.get_camera_state()
        self._follow_target_lookat = None
        if self._controller is not None and hasattr(self._controller, "target"):
            if self._camera_state is not None:
                self._controller.target = tuple(float(x) for x in self._camera_state.lookat)
            else:
                self._controller.target = (0.0, 0.0, 0.0)

    @staticmethod
    def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Rotate a vector by a pygfx camera quaternion."""
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rot = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )
        return rot @ np.asarray(vector, dtype=np.float64)

    @staticmethod
    def _quat_forward(q: np.ndarray) -> np.ndarray:
        """Extract forward (-Z) direction from a quaternion (x, y, z, w)."""
        return PygfxCameraMixin._quat_rotate(q, np.array([0.0, 0.0, -1.0], dtype=np.float64))

    @staticmethod
    def _quat_up(q: np.ndarray) -> np.ndarray:
        """Extract world-space camera up (+Y) direction from a quaternion."""
        return PygfxCameraMixin._quat_rotate(q, np.array([0.0, 1.0, 0.0], dtype=np.float64))
