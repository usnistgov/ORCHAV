"""Opt-in directional shadow fitting using pygfx's public light/camera API.

pygfx 0.17 allocates a fixed 1024-square map. These controls improve world-space
texel density, not texture resolution. Focus is a pinned inspection region;
ordinary camera movement does not move the map or make its edges shimmer.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

logger = logging.getLogger(__name__)
SHADOW_MAP_SIZE = 1024
SHADOW_QUALITY_LABELS = {
    "standard": "Standard (whole scene)",
    "scene": "Fitted scene",
    "focus": "Inspection focus",
}


def bounds_corners(bounds: np.ndarray) -> np.ndarray:
    """Return all eight corners of a finite, ordered world-space box."""
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
        raise ValueError("Shadow bounds must be a finite ordered (2, 3) box")
    return np.array(list(itertools.product(*zip(box[0], box[1]))), dtype=np.float64)


def fitted_shadow_box(
    caster_corners: np.ndarray,
    rotation: np.ndarray,
    *,
    focus_center: np.ndarray | None = None,
    focus_radius: float = 25.0,
) -> tuple[np.ndarray, float, float, float]:
    """Fit a light-space box, retaining every caster's depth in focus mode.

    XY bounds restrict receivers; Z always spans the entire physical scene.
    An upstream caster outside the focus cube can therefore still cast into it.
    Quantized XY centers and padding keep stable geometry within the frustum.
    """
    light_points = np.asarray(caster_corners, dtype=np.float64) @ rotation
    lo, hi = light_points.min(axis=0), light_points.max(axis=0)
    if focus_center is not None:
        focus = (
            bounds_corners(np.array([focus_center - focus_radius, focus_center + focus_radius]))
            @ rotation
        )
        lo[:2], hi[:2] = focus[:, :2].min(axis=0), focus[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 0.01)
    # Room for PCF samples and center snapping, also for near-planar scenes.
    size = np.maximum(span[:2] * 1.05, 0.05)
    texel = size / SHADOW_MAP_SIZE
    center = (lo + hi) * 0.5
    center[:2] = np.round(center[:2] / texel) * texel
    depth_pad = max(float(span[2]) * 0.05, 0.05)
    center[2] = hi[2] + depth_pad
    far = float(span[2] + 2.0 * depth_pad)
    return rotation @ center, float(size[0]), float(size[1]), far


class PygfxShadowMixin:
    """Public runtime shadow-quality controls and cached native fitting."""

    def get_shadow_quality(self) -> str:
        return str(getattr(self, "_shadow_quality", "standard"))

    def get_available_shadow_qualities(self) -> dict[str, str]:
        return dict(SHADOW_QUALITY_LABELS)

    def set_shadow_quality(self, quality: str) -> bool:
        """Choose standard coverage, fitted scene, or a pinned focus region."""
        if quality not in SHADOW_QUALITY_LABELS:
            return False
        if quality == "focus" and getattr(self, "_shadow_focus_center", None) is None:
            if not self.refocus_shadows():
                return False
        self._shadow_quality = quality
        self._shadow_fit_signature = None
        if quality == "standard":
            self._restore_standard_shadow_camera()
        else:
            self._update_shadow_quality(force=True)
        self.request_redraw()
        return True

    def set_shadow_focus_radius(self, radius: float) -> bool:
        """Set the half-size (world units) of the pinned inspection cube."""
        try:
            value = float(radius)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(value) or not 0.1 <= value <= 100000.0:
            return False
        self._shadow_focus_radius = value
        self._shadow_fit_signature = None
        self._update_shadow_quality(force=True)
        self.request_redraw()
        return True

    def refocus_shadows(self) -> bool:
        """Pin shadow detail around the current camera look-at point."""
        state = self.get_camera_state()
        if state is None:
            return False
        center = np.asarray(state.lookat, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            return False
        self._shadow_focus_center = center.copy()
        self._shadow_fit_signature = None
        self._update_shadow_quality(force=True)
        self.request_redraw()
        return True

    def set_shadow_bias_texels(self, value: float) -> bool:
        """Control fitted-mode depth offset in world texels, not scene metres."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(value) or not 0.0 <= value <= 8.0:
            return False
        self._shadow_bias_texels = value
        self._apply_fitted_shadow_bias()
        self.request_redraw()
        return True

    def get_shadow_quality_state(self) -> dict[str, Any]:
        """Return actual map coverage and fit controls for UI/diagnostics."""
        shadow = getattr(getattr(self, "_key_light", None), "shadow", None)
        camera = getattr(shadow, "camera", None)
        width = float(getattr(camera, "width", 0.0))
        height = float(getattr(camera, "height", 0.0))
        center = getattr(self, "_shadow_focus_center", None)
        return {
            "quality": self.get_shadow_quality(),
            "map_size": SHADOW_MAP_SIZE,
            "width": width,
            "height": height,
            "world_units_per_texel": max(width, height) / SHADOW_MAP_SIZE,
            "focus_radius": float(getattr(self, "_shadow_focus_radius", 25.0)),
            "focus_center": None if center is None else tuple(float(v) for v in center),
            "bias_texels": float(getattr(self, "_shadow_bias_texels", 1.5)),
            "fit_available": bool(getattr(self, "_shadow_fit_available", False)),
        }

    def _restore_standard_shadow_camera(self) -> None:
        light = getattr(self, "_key_light", None)
        if light is None:
            return
        light.target.world.position = (0.0, 0.0, 0.0)
        self._set_directional_light_from_angles(
            light=light,
            azimuth_deg=self._key_light_azimuth_deg,
            elevation_deg=self._key_light_elevation_deg,
            radius=80.0,
        )
        camera = light.shadow.camera
        camera.width = camera.height = float(getattr(self, "_scene_extent", 200.0)) * 1.5
        camera.depth_range = (-500.0, 500.0)
        light.shadow.bias = 0.001
        self._apply_profile_shadow_bias(self.get_lighting_profile())
        self._shadow_fit_available = False

    def _apply_fitted_shadow_bias(self) -> None:
        if self.get_shadow_quality() == "standard":
            return
        shadow = getattr(getattr(self, "_key_light", None), "shadow", None)
        if shadow is None:
            return
        camera = shadow.camera
        texel = max(float(camera.width), float(camera.height)) / SHADOW_MAP_SIZE
        depth = max(float(camera.far - camera.near), 1e-6)
        shadow.bias = float(getattr(self, "_shadow_bias_texels", 1.5)) * texel / depth

    def _mark_shadow_positions_changed(self, buffer: Any) -> None:
        """Track ORCHAV writes even before pygfx consumes an already-dirty buffer."""
        versions = getattr(self, "_shadow_position_versions", None)
        if versions is None:
            versions = self._shadow_position_versions = WeakKeyDictionary()
        versions[buffer] = versions.get(buffer, 0) + 1

    def _shadow_object_corners(self, obj: Any) -> np.ndarray | None:
        """Use cached native bounds, with explicit invalidation for streamed data."""
        positions = obj.geometry.positions
        version = getattr(self, "_shadow_position_versions", {}).get(positions, 0)
        if not version:
            box = obj.get_world_bounding_box()
            return None if box is None else bounds_corners(box)
        # Buffer.update_full() may coalesce writes before the first upload.
        # The app's own write generation prevents stale native cached bounds.
        cache = getattr(self, "_shadow_stream_bounds", None)
        if cache is None:
            cache = self._shadow_stream_bounds = WeakKeyDictionary()
        cached = cache.get(positions)
        key = (positions.rev, version)
        if cached is None or cached[0] != key:
            points = np.asarray(positions.data, dtype=np.float64)
            if points.ndim != 2 or len(points) == 0 or points.shape[1] < 3:
                return None
            local = bounds_corners(np.array([points[:, :3].min(0), points[:, :3].max(0)]))
            cache[positions] = (key, local)
        else:
            local = cached[1]
        homogeneous = np.column_stack([local, np.ones(8)])
        return (homogeneous @ obj.world.matrix.T)[:, :3]

    def _update_shadow_quality(self, *, force: bool = False) -> None:
        """Refit changed physical geometry; unchanged frames only check revisions.

        Bounding boxes are pygfx-cached and include parent/world transforms.
        This never scans mesh vertices on the steady-state frame path. Only
        physical mesh casters participate, never labels/MPC/coverage overlays.
        """
        if self.get_shadow_quality() == "standard":
            return
        light = getattr(self, "_key_light", None)
        if light is None or not getattr(self, "_shadows_enabled", True):
            return
        objects = []
        signature = []
        for name, obj in getattr(self, "_objects", {}).items():
            if not (self._is_scene_mesh_name(name) or self._is_target_mesh_name(name)):
                continue
            if not getattr(obj, "cast_shadow", False):
                continue
            geometry = getattr(obj, "geometry", None)
            positions = getattr(geometry, "positions", None)
            if positions is None:
                continue
            objects.append(obj)
            write_version = getattr(self, "_shadow_position_versions", {}).get(positions, 0)
            signature.append((id(obj), positions.rev, write_version, obj.world.last_modified))
        key = (
            tuple(signature),
            float(self._key_light_azimuth_deg),
            float(self._key_light_elevation_deg),
            self.get_shadow_quality(),
        )
        if not force and key == getattr(self, "_shadow_fit_signature", None):
            return
        corners = []
        for obj in objects:
            try:
                object_corners = self._shadow_object_corners(obj)
                if object_corners is not None:
                    corners.append(object_corners)
            except ValueError:
                continue
        if not corners:
            self._shadow_fit_available = False
            return
        # Use the native camera's own up convention to derive the light basis.
        az = np.deg2rad(float(self._key_light_azimuth_deg))
        el = np.deg2rad(float(self._key_light_elevation_deg))
        direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        camera = light.shadow.camera
        camera.world.position = (0.0, 0.0, 0.0)
        camera.look_at(direction)
        rotation = np.asarray(camera.world.matrix, dtype=np.float64)[:3, :3]
        focus = (
            getattr(self, "_shadow_focus_center", None)
            if self.get_shadow_quality() == "focus"
            else None
        )
        all_corners = np.concatenate(corners)
        if self.get_shadow_quality() == "focus" and focus is None:
            # A new scene invalidates the previous scene's pinned region.
            focus = (all_corners.min(axis=0) + all_corners.max(axis=0)) * 0.5
            self._shadow_focus_center = focus.copy()
        eye, width, height, far = fitted_shadow_box(
            all_corners,
            rotation,
            focus_center=focus,
            focus_radius=float(getattr(self, "_shadow_focus_radius", 25.0)),
        )
        light.world.position = eye
        light.target.world.position = eye + direction
        camera.width, camera.height = width, height
        camera.depth_range = (0.0, far)
        camera.world.position = eye
        camera.look_at(eye + direction)
        self._apply_fitted_shadow_bias()
        self._shadow_fit_signature = key
        self._shadow_fit_available = True
