"""Open3D camera-state and follow-mode helpers for the renderer backend.

This mixin adapts ORCHAV's renderer-neutral camera intents to Open3D's
``O3DVisualizer`` camera API. The implementation is careful about coordinate
conventions: ORCHAV scenes use world Z-up while Open3D exposes OpenGL-style
view matrices through its scene camera.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from shared.logging import get_logger

from ...types.camera_state import CameraState
from ...utils.geometry import geometry_bounds
from ..camera_ops import (
    camera_state_for_overview,
    camera_state_for_pov,
    object_contributes_to_camera_bounds,
)

logger = get_logger("orchav.renderer_open3d")


class Open3DCameraMixin:
    """Own Open3D camera state, scene bounds, and follow-mode behavior.

    Public methods implement the shared renderer camera contract. Private
    helpers provide passive camera-debug logging and the locked-radius policy
    used by follow mode.
    """

    def _setup_camera(
        self, fov: float, lookat: list[float], eye: list[float], up: list[float]
    ) -> bool:
        """Set camera parameters via O3DVisualizer in an adapter-safe way."""
        if self._o3d_vis is None:
            return False
        try:
            self._observe_camera_state("setup_camera_pre")
            self._o3d_vis.setup_camera(float(fov), lookat, eye, up)
            self._remember_explicit_camera_state(
                float(fov),
                np.asarray(lookat, dtype=np.float64),
                np.asarray(eye, dtype=np.float64),
                np.asarray(up, dtype=np.float64),
            )
            self._mark_camera_command("setup_camera")
            self._observe_camera_state("setup_camera_post")
            self._set_far_clipping_plane()
            self._post_redraw()
            return True
        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.debug("Open3DRenderer: setup_camera failed: %s", exc)
            return False

    def _remember_explicit_camera_state(
        self, fov: float, lookat: np.ndarray, eye: np.ndarray, up: np.ndarray
    ) -> None:
        """Store the last camera state issued through ORCHAV's renderer API."""
        self._last_explicit_camera_state = CameraState(
            eye=(float(eye[0]), float(eye[1]), float(eye[2])),
            lookat=(float(lookat[0]), float(lookat[1]), float(lookat[2])),
            up=(float(up[0]), float(up[1]), float(up[2])),
            fov_deg=float(fov),
        )

    def set_fly_mode(self, enabled: bool) -> bool:
        """Enable or disable fly camera mode via O3DVisualizer mouse_mode."""
        if self._o3d_vis is None:
            return False
        try:
            if enabled:
                self._o3d_vis.mouse_mode = gui.SceneWidget.Controls.FLY
            else:
                self._o3d_vis.mouse_mode = gui.SceneWidget.Controls.ROTATE_CAMERA
            logger.info("Open3DRenderer: fly mode %s", "enabled" if enabled else "disabled")
            return True
        except (AttributeError, RuntimeError) as exc:
            logger.warning("Open3DRenderer: set_fly_mode failed: %s", exc)
            return False

    def compute_scene_bounds(
        self, scope: str = "visible"
    ) -> Optional[o3d.geometry.AxisAlignedBoundingBox]:
        """Return scene bounds for camera fitting and external adapters."""
        return self._compute_scene_bounds(scope=scope)

    def _camera_debug(self, message: str, **fields: Any) -> None:
        """Emit structured camera diagnostics when ORCHAV_CAMERA_DEBUG is enabled."""
        if not getattr(self, "_camera_debug_enabled", False):
            return
        if fields:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            logger.info("CameraDebug: %s %s", message, details)
        else:
            logger.info("CameraDebug: %s", message)

    def _capture_camera_observation(self) -> Optional[dict[str, Any]]:
        """Capture a lightweight camera snapshot for passive mutation diagnostics."""
        if self._o3d_vis is None:
            return None
        try:
            scene = self._o3d_vis.scene
            if scene is None or not hasattr(scene, "camera"):
                return None
            camera = scene.camera
            view = camera.get_view_matrix()
            R = view[:3, :3]
            t = view[:3, 3]
            eye = -np.dot(R.T, t)
            forward = -view[2, :3].copy()
            forward_norm = float(np.linalg.norm(forward))
            if forward_norm > 1e-9:
                forward = forward / forward_norm
            else:
                forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            fov = float(camera.get_field_of_view())
            fp_values = np.concatenate(
                [view[:3, :4].reshape(-1), np.array([fov], dtype=np.float64)]
            )
            fp = tuple(float(v) for v in np.round(fp_values, 6))
            return {"fp": fp, "eye": eye, "forward": forward, "fov": fov}
        except (RuntimeError, AttributeError, ValueError):
            return None

    def _observe_camera_state(self, source: str, log_internal: bool = False) -> None:
        """Passively report camera changes that occur between explicit renderer camera commands."""
        if not getattr(self, "_camera_debug_enabled", False):
            return

        obs = self._capture_camera_observation()
        if obs is None:
            return

        last_fp = getattr(self, "_camera_observe_last_fp", None)
        command_seq = int(getattr(self, "_camera_command_seq", 0))
        last_command_seq = int(getattr(self, "_camera_observe_last_command_seq", 0))

        if last_fp is not None and obs["fp"] != last_fp:
            last_eye = getattr(self, "_camera_observe_last_eye", None)
            last_forward = getattr(self, "_camera_observe_last_forward", None)
            last_fov = getattr(self, "_camera_observe_last_fov", None)

            eye_delta = 0.0
            if isinstance(last_eye, np.ndarray):
                eye_delta = float(np.linalg.norm(obs["eye"] - last_eye))

            forward_delta_deg = 0.0
            if isinstance(last_forward, np.ndarray):
                dot = float(np.clip(np.dot(obs["forward"], last_forward), -1.0, 1.0))
                forward_delta_deg = float(np.degrees(np.arccos(dot)))

            fov_delta = 0.0
            if last_fov is not None:
                fov_delta = float(obs["fov"] - float(last_fov))

            external_candidate = command_seq == last_command_seq
            if external_candidate or log_internal:
                step = getattr(getattr(self.visualizer, "app_state", None), "step", None)
                self._camera_debug(
                    "camera_state_change",
                    source=source,
                    step=step,
                    external_candidate=external_candidate,
                    command_seq=command_seq,
                    last_command_seq=last_command_seq,
                    eye_delta=f"{eye_delta:.6f}",
                    forward_delta_deg=f"{forward_delta_deg:.3f}",
                    fov_delta=f"{fov_delta:.4f}",
                )

        self._camera_observe_last_fp = obs["fp"]
        self._camera_observe_last_eye = obs["eye"]
        self._camera_observe_last_forward = obs["forward"]
        self._camera_observe_last_fov = float(obs["fov"])
        self._camera_observe_last_command_seq = command_seq

    def _mark_camera_command(self, reason: str) -> None:
        """Mark an internal camera command for passive external-mutation detection."""
        self._camera_command_seq = int(getattr(self, "_camera_command_seq", 0)) + 1
        self._camera_debug(
            "camera_command",
            reason=reason,
            command_seq=self._camera_command_seq,
        )

    def reset_camera_bounds(self) -> None:
        """Reset the Open3D camera to frame all known scene geometry."""
        if self._o3d_vis is None:
            return

        try:
            combined_bbox = self._compute_scene_bounds(scope="whole")

            if combined_bbox is not None:
                center = combined_bbox.get_center()
                extent = combined_bbox.get_extent()
                max_extent = max(extent)

                if max_extent > 1e-6:
                    fov_rad = np.radians(60.0)
                    distance = (max_extent / 2.0) / np.tan(fov_rad / 2.0) * 1.5

                    # Slight X/Y/Z offsets give a stable overview perspective.
                    eye = np.array(
                        [
                            center[0] + distance * 0.3,
                            center[1] - distance * 0.3,
                            center[2] + distance * 0.7,
                        ]
                    )

                    lookat = np.array(center)
                    up = np.array([0.0, 0.0, 1.0])

                    self._o3d_vis.setup_camera(60.0, lookat.tolist(), eye.tolist(), up.tolist())
                    self._remember_explicit_camera_state(60.0, lookat, eye, up)

                    self._set_far_clipping_plane()

                    self._post_redraw()

                    logger.info(
                        f"Open3DRenderer: Camera fit to scene bounds - "
                        f"center={center}, extent={extent}, distance={distance:.1f}"
                    )
                    return

            self._o3d_vis.reset_camera_to_default()
            self._set_far_clipping_plane()
            self._post_redraw()
            logger.info("Open3DRenderer: Reset camera to default (no scene bounds)")

        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.warning(f"Open3DRenderer: Failed to fit camera to bounds: {exc}")
            self._o3d_vis.reset_camera_to_default()
            self._post_redraw()

    def _compute_scene_bounds(
        self, scope: str = "visible"
    ) -> Optional[o3d.geometry.AxisAlignedBoundingBox]:
        """Compute bounds from renderer objects that reached Open3D successfully."""
        include_hidden = scope == "whole"
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []

        for name, applied in self._open3d_applied_objects().items():
            if not object_contributes_to_camera_bounds(name):
                continue
            if not include_hidden and not applied.visible:
                continue
            bounds = geometry_bounds(applied.payload)
            if bounds is None:
                continue
            min_bound, max_bound = bounds
            corners = np.asarray(
                [
                    [x, y, z]
                    for x in (min_bound[0], max_bound[0])
                    for y in (min_bound[1], max_bound[1])
                    for z in (min_bound[2], max_bound[2])
                ],
                dtype=np.float64,
            )
            try:
                transform = np.asarray(applied.transform, dtype=np.float64)
                if transform.shape != (4, 4):
                    continue
                homogeneous = np.column_stack([corners, np.ones(len(corners))])
                world = homogeneous @ transform.T
                points = world[:, :3]
            except (TypeError, ValueError):
                continue
            finite = np.all(np.isfinite(points), axis=1)
            if not np.any(finite):
                continue
            points = points[finite]
            mins.append(np.min(points, axis=0))
            maxs.append(np.max(points, axis=0))

        if not mins:
            return None
        return o3d.geometry.AxisAlignedBoundingBox(
            np.min(np.vstack(mins), axis=0),
            np.max(np.vstack(maxs), axis=0),
        )

    def _set_far_clipping_plane(self, far: float = 100000.0, near: float = 0.01) -> bool:
        """Set near/far clipping planes after Open3D camera changes.

        By default, O3DVisualizer uses a far plane around 1000 units which causes
        distant geometry to be culled. Large outdoor scenes such as cities
        require a much larger far plane.
        """
        if self._o3d_vis is None:
            return False

        try:
            scene = self._o3d_vis.scene
            if scene is not None and hasattr(scene, "camera"):
                camera = scene.camera
                fov = camera.get_field_of_view()

                try:
                    content_rect = self._o3d_vis.content_rect
                    if content_rect.width > 0 and content_rect.height > 0:
                        aspect = content_rect.width / content_rect.height
                    else:
                        aspect = 16.0 / 9.0
                except (AttributeError, ZeroDivisionError):
                    aspect = 16.0 / 9.0

                fov_type = rendering.Camera.FovType.Vertical
                camera.set_projection(fov, aspect, near, far, fov_type)

                logger.debug(
                    f"Open3DRenderer: Set clipping planes - near={near}, far={far}, aspect={aspect:.2f}"
                )
                return True
        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.debug(f"Open3DRenderer: Could not set clipping planes: {exc}")
        return False

    def get_camera_state(self) -> Optional[CameraState]:
        """Return the current Open3D camera as a renderer-neutral ``CameraState``.

        Values are extracted directly from the Open3D scene camera so user mouse
        interaction is reflected. If the current view still matches the last
        explicit ORCHAV camera command, the original lookat/up vectors are
        reused because Open3D's view matrix does not preserve an orbit target.
        """
        if self._o3d_vis is None:
            return None

        try:
            scene = self._o3d_vis.scene
            if scene is None or not hasattr(scene, "camera"):
                return None

            camera = scene.camera
            view_matrix = camera.get_view_matrix()

            R = view_matrix[:3, :3]
            t = view_matrix[:3, 3]
            eye = -np.dot(R.T, t)

            forward = -view_matrix[2, :3]
            forward = forward / (np.linalg.norm(forward) + 1e-9)

            fov = camera.get_field_of_view()

            lookat: np.ndarray
            up: np.ndarray
            explicit = self._last_explicit_camera_state
            if explicit is not None:
                explicit_eye = np.asarray(explicit.eye, dtype=np.float64)
                explicit_lookat = np.asarray(explicit.lookat, dtype=np.float64)
                explicit_forward = explicit_lookat - explicit_eye
                explicit_forward_norm = np.linalg.norm(explicit_forward)
                if explicit_forward_norm > 1e-9:
                    explicit_forward = explicit_forward / explicit_forward_norm
                eye_matches = float(np.linalg.norm(explicit_eye - eye)) <= max(
                    1e-3, 0.02 * max(np.linalg.norm(explicit_eye), 1.0)
                )
                fwd_matches = (
                    explicit_forward_norm > 1e-9
                    and float(np.dot(explicit_forward, forward)) >= 0.999
                )
                fov_matches = abs(float(explicit.fov_deg) - float(fov)) <= 0.5
                if eye_matches and fwd_matches and fov_matches:
                    lookat = explicit_lookat
                    up = np.asarray(explicit.up, dtype=np.float64)
                else:
                    explicit = None

            if explicit is None:
                # External Open3D interaction has no stored lookat, so infer a
                # practical target distance from scene bounds when available.
                lookat_distance = 10.0
                try:
                    bounds = self._o3d_vis.scene.bounding_box
                    if bounds is not None:
                        center = np.array(bounds.get_center())
                        lookat_distance = max(1.0, np.linalg.norm(center - eye))
                except (RuntimeError, AttributeError):
                    logger.debug("Could not access scene bounding box for lookat distance")

                lookat = eye + forward * lookat_distance

                # Always use world Z-up as the fallback hint for camera states
                # produced by external Open3D interaction instead of the renderer API.
                up = np.array([0.0, 0.0, 1.0])

            return CameraState(
                eye=(float(eye[0]), float(eye[1]), float(eye[2])),
                lookat=(float(lookat[0]), float(lookat[1]), float(lookat[2])),
                up=(float(up[0]), float(up[1]), float(up[2])),
                fov_deg=float(fov),
            )

        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.debug(f"Open3DRenderer: Could not get camera state: {exc}")
            return None

    def set_camera_state(self, state: CameraState) -> bool:
        """Apply a renderer-neutral camera state."""
        if self._o3d_vis is None or not isinstance(state, CameraState):
            return False

        eye = list(state.eye)
        lookat = list(state.lookat)
        up = list(state.up)
        fov = float(state.fov_deg)

        try:
            if not self._setup_camera(fov=fov, lookat=lookat, eye=eye, up=up):
                return False
            eye_arr = np.asarray(eye, dtype=np.float64)
            lookat_arr = np.asarray(lookat, dtype=np.float64)
            look_dist = float(np.linalg.norm(lookat_arr - eye_arr))
            if np.isfinite(look_dist) and look_dist > 1e-6:
                self._last_camera_look_distance = look_dist
            self._camera_debug("set_camera_state", look_dist=f"{look_dist:.6f}")

            logger.info(f"Camera restored: eye={eye[:2]}..., lookat={lookat[:2]}...")
            return True

        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.warning(f"Failed to restore camera: {exc}")
            return False

    def set_overview_camera(
        self,
        view: str,
        bounds: Any,
        fov: float = 60.0,
        distance: Optional[float] = None,
    ) -> bool:
        """Apply a named overview camera intent."""
        state = camera_state_for_overview(
            view,
            bounds,
            fov=float(fov),
            distance=distance,
            aspect=self._viewport_aspect(),
        )
        return False if state is None else self.set_camera_state(state)

    def focus_camera(self, target_position: Any) -> bool:
        """Focus camera on a target and reset follow-orbit history."""
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
        """Apply a first-person camera intent from entity pose."""
        state = camera_state_for_pov(position, orientation, axis=axis)
        return False if state is None else self.set_camera_state(state)

    def update_follow_camera(self, target_position: Any) -> bool:
        """Update follow-mode camera target while preserving renderer orbit state."""
        return self.set_lookat(np.asarray(target_position, dtype=np.float64))

    def _viewport_aspect(self) -> float:
        """Return Open3D content-rect aspect ratio with desktop fallback."""
        try:
            rect = self._o3d_vis.content_rect if self._o3d_vis is not None else None
            width = float(getattr(rect, "width", 0.0))
            height = float(getattr(rect, "height", 0.0))
            if width > 1.0 and height > 1.0:
                return width / height
        except (AttributeError, TypeError, ValueError):
            pass
        return 16.0 / 9.0

    def apply_camera_delta(self, delta: np.ndarray, target_pos: np.ndarray = None) -> bool:
        """Apply a translation delta to the camera position for Follow mode.

        This moves the camera eye by delta while keeping the target as the
        rotation center (lookat point), so rotation orbits around the followed object.
        """
        if self._o3d_vis is None:
            return False

        try:
            scene = self._o3d_vis.scene
            if scene is None or not hasattr(scene, "camera"):
                return False

            camera = scene.camera
            view_matrix = camera.get_view_matrix()

            R = view_matrix[:3, :3]
            t = view_matrix[:3, 3]
            eye = -np.dot(R.T, t)

            up = view_matrix[1, :3].copy()
            up = up / (np.linalg.norm(up) + 1e-9)

            fov = camera.get_field_of_view()

            new_eye = eye + delta

            if target_pos is not None:
                new_lookat = np.asarray(target_pos, dtype=np.float64)
            else:
                forward = -view_matrix[2, :3]
                forward = forward / (np.linalg.norm(forward) + 1e-9)
                lookat_dist = 10.0
                old_lookat = eye + forward * lookat_dist
                new_lookat = old_lookat + delta

            self._o3d_vis.setup_camera(fov, new_lookat.tolist(), new_eye.tolist(), up.tolist())
            self._remember_explicit_camera_state(fov, new_lookat, new_eye, up)
            self._mark_camera_command("apply_camera_delta")
            self._observe_camera_state("apply_camera_delta_post")

            self._set_far_clipping_plane()

            # Frame pipeline owns redraw here to avoid presenting partial frames.

            logger.debug(f"Open3DRenderer: Follow camera - eye={new_eye}, lookat={new_lookat}")
            return True

        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.debug(f"Open3DRenderer: Could not apply camera delta: {exc}")
            return False

    def set_lookat(self, lookat: np.ndarray) -> bool:
        """Set the camera lookat point for Follow mode.

        This keeps the camera looking AT the target position while preserving
        the user's orbit offset (their distance and angle relative to the target).
        The locked-radius policy adopts small zoom gestures but blocks large
        drift-like radius jumps while the target is moving. A fixed world Z-up
        vector prevents rotational drift from round-tripping Open3D matrices.
        """
        if self._o3d_vis is None:
            return False

        try:
            scene = self._o3d_vis.scene
            if scene is None or not hasattr(scene, "camera"):
                return False

            camera = scene.camera
            new_lookat = np.asarray(lookat, dtype=np.float64)
            self._observe_camera_state("set_lookat_pre")
            follow_locked_radius = getattr(self, "_follow_locked_radius", None)
            follow_last_forward = getattr(self, "_follow_last_forward", None)
            follow_debug_last_requested_eye = getattr(
                self, "_follow_debug_last_requested_eye", None
            )
            camera_debug_enabled = bool(getattr(self, "_camera_debug_enabled", False))
            follow_debug_seq = int(getattr(self, "_follow_debug_seq", 0))
            follow_debug_baseline_radius = getattr(self, "_follow_debug_baseline_radius", None)
            follow_debug_last_radius = getattr(self, "_follow_debug_last_radius", None)
            follow_debug_last_target = getattr(self, "_follow_debug_last_target", None)

            view_matrix = camera.get_view_matrix()
            R = view_matrix[:3, :3]
            t = view_matrix[:3, 3]
            eye = -np.dot(R.T, t)
            forward = -view_matrix[2, :3].copy()
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 1e-9:
                forward = forward / forward_norm
            else:
                forward = np.array([1.0, 0.0, 0.0])
            fov = camera.get_field_of_view()

            # POV mode stores a synthetic 10 m distance; use it to bootstrap
            # Follow when eye and target are degenerate.
            look_distance_hint = float(getattr(self, "_last_camera_look_distance", 10.0))
            if not np.isfinite(look_distance_hint) or look_distance_hint <= 1e-6:
                look_distance_hint = 10.0

            min_distance = 1e-3
            first_follow_frame = self._stored_lookat is None
            bootstrap_used = False
            zoom_radius_adopted = False
            zoom_adoption_blocked = False
            stabilization_applied = False
            orientation_delta_deg = 0.0
            target_step = 0.0
            radial_delta = 0.0

            if first_follow_frame:
                offset = eye - new_lookat
                offset_norm = float(np.linalg.norm(offset))
                if offset_norm < min_distance:
                    offset = -forward * look_distance_hint
                    bootstrap_used = True
                    offset_norm = float(np.linalg.norm(offset))
            else:
                old_lookat = np.asarray(self._stored_lookat, dtype=np.float64)
                raw_offset = eye - old_lookat
                raw_radius = float(np.linalg.norm(raw_offset))
                offset_norm = raw_radius
                target_step = float(np.linalg.norm(new_lookat - old_lookat))

                prev_radius = follow_locked_radius
                if prev_radius is None:
                    prev_radius = raw_radius if raw_radius > min_distance else look_distance_hint
                prev_radius = float(prev_radius)
                if not np.isfinite(prev_radius) or prev_radius <= min_distance:
                    prev_radius = look_distance_hint

                prev_forward = follow_last_forward
                if (
                    prev_forward is not None
                    and isinstance(prev_forward, np.ndarray)
                    and prev_forward.shape == (3,)
                ):
                    prev_forward_norm = np.linalg.norm(prev_forward)
                    if prev_forward_norm > 1e-9:
                        prev_forward = prev_forward / prev_forward_norm
                        forward_dot = float(np.clip(np.dot(prev_forward, forward), -1.0, 1.0))
                        orientation_delta_deg = float(np.degrees(np.arccos(forward_dot)))

                zoom_radius_threshold = max(0.2, 0.02 * prev_radius)
                zoom_angle_threshold_deg = 1.0
                radial_delta = abs(raw_radius - prev_radius)
                zoom_radius_adopted = (
                    radial_delta > zoom_radius_threshold
                    and orientation_delta_deg < zoom_angle_threshold_deg
                )
                # Keep small wheel-zoom adjustments responsive while blocking
                # large injected radius jumps as the target moves.
                moving_target = target_step > 1e-3
                max_motion_zoom_delta = max(0.2, 0.75 * target_step)
                if zoom_radius_adopted and moving_target and radial_delta > max_motion_zoom_delta:
                    zoom_radius_adopted = False
                    zoom_adoption_blocked = True
                follow_radius = raw_radius if zoom_radius_adopted else prev_radius
                if not np.isfinite(follow_radius) or follow_radius <= min_distance:
                    follow_radius = max(raw_radius, look_distance_hint, min_distance)

                offset = -forward * follow_radius
                offset_norm = float(np.linalg.norm(offset))
                stabilization_applied = (
                    not zoom_radius_adopted and abs(raw_radius - follow_radius) > 1e-4
                )

            new_eye = new_lookat + offset

            up = np.array([0.0, 0.0, 1.0])

            eye_to_target = new_lookat - new_eye
            eye_to_target_norm = float(np.linalg.norm(eye_to_target))
            if eye_to_target_norm < min_distance:
                new_eye = new_lookat - forward * look_distance_hint
                eye_to_target = new_lookat - new_eye
                eye_to_target_norm = float(np.linalg.norm(eye_to_target))
                bootstrap_used = True

            if (
                eye_to_target_norm > 1e-9
                and abs(np.dot(eye_to_target / eye_to_target_norm, up)) > 0.999
            ):
                new_eye[0] += 0.01 * eye_to_target_norm
                eye_to_target = new_lookat - new_eye
                eye_to_target_norm = float(np.linalg.norm(eye_to_target))

            if eye_to_target_norm > min_distance:
                # First transition frame can carry stale mode-switch distance.
                if first_follow_frame and follow_locked_radius is None:
                    follow_last_forward = eye_to_target / eye_to_target_norm
                else:
                    follow_locked_radius = eye_to_target_norm
                    follow_last_forward = eye_to_target / eye_to_target_norm
            else:
                follow_locked_radius = None
                follow_last_forward = None

            prev_requested_eye = follow_debug_last_requested_eye

            self._o3d_vis.setup_camera(fov, new_lookat.tolist(), new_eye.tolist(), up.tolist())
            self._remember_explicit_camera_state(fov, new_lookat, new_eye, up)
            self._mark_camera_command("set_lookat")
            self._observe_camera_state("set_lookat_post")
            self._stored_lookat = new_lookat.copy()
            if eye_to_target_norm > 1e-6:
                self._last_camera_look_distance = eye_to_target_norm

            # Compare requested and observed eye to detect stale/frozen camera state.
            apply_error: Optional[float] = None
            apply_prev_error: Optional[float] = None
            post_radius: Optional[float] = None
            post_stale = False
            try:
                post_view = camera.get_view_matrix()
                post_R = post_view[:3, :3]
                post_t = post_view[:3, 3]
                post_eye = -np.dot(post_R.T, post_t)
                apply_error = float(np.linalg.norm(post_eye - new_eye))
                post_radius = float(np.linalg.norm(new_lookat - post_eye))
                if prev_requested_eye is not None:
                    apply_prev_error = float(np.linalg.norm(post_eye - prev_requested_eye))
                stale_tol = max(1e-3, 0.05 * max(eye_to_target_norm, 1.0))
                if apply_prev_error is not None:
                    post_stale = apply_error > stale_tol and apply_prev_error <= stale_tol
            except (RuntimeError, ValueError):
                pass
            follow_debug_last_requested_eye = new_eye.copy()

            if camera_debug_enabled:
                follow_debug_seq += 1
                if first_follow_frame or follow_debug_baseline_radius is None:
                    follow_debug_baseline_radius = eye_to_target_norm
                baseline = follow_debug_baseline_radius or eye_to_target_norm
                last_radius = follow_debug_last_radius
                last_target = follow_debug_last_target
                delta_target = 0.0
                if last_target is not None:
                    delta_target = float(np.linalg.norm(new_lookat - last_target))
                delta_radius = 0.0
                if last_radius is not None:
                    delta_radius = eye_to_target_norm - last_radius
                drift_from_baseline = eye_to_target_norm - baseline
                step = getattr(getattr(self.visualizer, "app_state", None), "step", None)
                self._camera_debug(
                    "follow_update",
                    seq=follow_debug_seq,
                    step=step,
                    first=first_follow_frame,
                    bootstrap=bootstrap_used,
                    offset_norm=f"{offset_norm:.6f}",
                    radius=f"{eye_to_target_norm:.6f}",
                    baseline=f"{baseline:.6f}",
                    drift=f"{drift_from_baseline:.6f}",
                    delta_radius=f"{delta_radius:.6f}",
                    delta_target=f"{delta_target:.6f}",
                    target_step=f"{target_step:.6f}",
                    orient_delta_deg=f"{orientation_delta_deg:.3f}",
                    zoom_adopted=zoom_radius_adopted,
                    zoom_blocked=zoom_adoption_blocked,
                    radial_delta=f"{radial_delta:.6f}",
                    stabilized=stabilization_applied,
                    apply_err="nan" if apply_error is None else f"{apply_error:.6f}",
                    apply_prev_err="nan" if apply_prev_error is None else f"{apply_prev_error:.6f}",
                    post_radius="nan" if post_radius is None else f"{post_radius:.6f}",
                    post_stale=post_stale,
                )
                follow_debug_last_radius = eye_to_target_norm
                follow_debug_last_target = new_lookat.copy()

            self._follow_locked_radius = follow_locked_radius
            self._follow_last_forward = follow_last_forward
            self._follow_debug_last_requested_eye = follow_debug_last_requested_eye
            self._follow_debug_seq = follow_debug_seq
            self._follow_debug_baseline_radius = follow_debug_baseline_radius
            self._follow_debug_last_radius = follow_debug_last_radius
            self._follow_debug_last_target = follow_debug_last_target

            if first_follow_frame:
                logger.debug(
                    "Follow transition init: bootstrap=%s offset_norm=%.6f eye_to_target_norm=%.6f",
                    bootstrap_used,
                    offset_norm,
                    eye_to_target_norm,
                )

            self._set_far_clipping_plane()
            self._post_redraw()

            return True

        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.error("set_lookat failed: %s", exc)
            return False

    def reset_follow_state(self) -> None:
        """Clear Follow-mode orbit state before leaving or re-entering Follow."""
        if self._camera_debug_enabled and self._follow_debug_last_radius is not None:
            baseline = self._follow_debug_baseline_radius or self._follow_debug_last_radius
            self._camera_debug(
                "follow_reset",
                updates=self._follow_debug_seq,
                last_radius=f"{self._follow_debug_last_radius:.6f}",
                baseline=f"{baseline:.6f}",
                drift=f"{(self._follow_debug_last_radius - baseline):.6f}",
            )
        self._stored_lookat = None
        self._follow_debug_seq = 0
        self._follow_debug_baseline_radius = None
        self._follow_debug_last_radius = None
        self._follow_debug_last_target = None
        self._follow_debug_last_requested_eye = None
        self._follow_locked_radius = None
        self._follow_last_forward = None
        logger.debug("Open3DRenderer: Reset follow state (cleared stored lookat)")
