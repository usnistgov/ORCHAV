"""Shared camera UI actions for the visualizer and Scenario Builder."""

from __future__ import annotations

import os
from typing import Any

from shared.logging import get_logger

from ..renderers.protocol import renderer_capabilities

logger = get_logger("orchav.camera_ui")


class CameraUIActions:
    """Route camera widgets through the common camera controller."""

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self._camera_debug_enabled = os.getenv("ORCHAV_CAMERA_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def _camera_debug(self, message: str, **fields: Any) -> None:
        """Emit camera UI diagnostics when ORCHAV_CAMERA_DEBUG is enabled."""
        if not getattr(self, "_camera_debug_enabled", False):
            return
        if fields:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            logger.info("CameraDebugUI: %s %s", message, details)
        else:
            logger.info("CameraDebugUI: %s", message)

    def handle_target_focus_changed(self, text: str) -> None:
        """Route focus-target changes to follow or POV camera updates."""
        logger.debug("Target focus selection changed to: %s", text)
        viz = self.visualizer
        camera = getattr(viz, "camera_controller", None)
        if camera:
            camera.remember_focus_selection()
            mode = getattr(getattr(viz, "app_state", None), "camera_mode", None)
            tools = camera.presentation() if hasattr(camera, "presentation") else None
            if tools is not None and mode in {"follow", "pov"}:
                tools.prepare()
                viz.renderer.reset_follow_state()
                camera._last_follow_target_pos = None
            self._camera_debug("target_focus_changed", mode=mode, text=text)
            if mode == "follow":
                camera.update_follow_camera_focus()
            elif mode == "pov":
                camera.set_pov_camera()

    def handle_camera_mode_changed(self, mode: str, checked: bool) -> None:
        """Switch overview/follow/POV camera modes and related UI state."""
        if not checked:
            return  # Radio groups emit for the button turning off as well.

        logger.info("Camera mode changed to: %s", mode)
        self._camera_debug("camera_mode_changed", mode=mode)

        viz = self.visualizer
        previous_mode = getattr(getattr(viz, "app_state", None), "camera_mode", None)
        previous_fly_mode = bool(getattr(getattr(viz, "app_state", None), "fly_mode", False))
        if hasattr(viz, "set_state"):
            viz.set_state(camera_mode=mode)

        camera = getattr(viz, "camera_controller", None)
        if not camera:
            return

        tools = camera.presentation() if hasattr(camera, "presentation") else None
        if tools is not None:
            tools.prepare()

        # Track controls are meaningful only when a selected entity drives view.
        track_group = getattr(viz, "track_group", None)
        advanced = getattr(viz, "advanced_camera_cb", None)
        show_advanced_track = bool(advanced is not None and advanced.isChecked())
        if track_group:
            track_group.setVisible(mode in ("follow", "pov") or show_advanced_track)

        # POV axis controls only affect first-person camera construction.
        pov_axis_container = getattr(viz, "pov_axis_container", None)
        if pov_axis_container:
            pov_axis_container.setVisible(mode == "pov")

        if mode == "overview":
            leaving_pov = previous_mode == "pov"
            camera.restore_pov_entity_visibility(update_renderer=not leaving_pov)
            if hasattr(viz, "set_state"):
                viz.set_state(pov_hidden_node=None)
            if previous_mode == "pov" and hasattr(camera, "restore_pre_pov_camera_state"):
                restored = camera.restore_pre_pov_camera_state(update_renderer=True)
                if not restored:
                    renderer = getattr(viz, "renderer", None)
                    if renderer is not None and hasattr(renderer, "update_renderer"):
                        renderer.update_renderer()
                    elif renderer is not None and hasattr(renderer, "request_redraw"):
                        renderer.request_redraw()
            logger.debug("Overview mode: free camera control enabled")

        elif mode == "follow":
            leaving_pov = previous_mode == "pov"
            camera.restore_pov_entity_visibility(update_renderer=not leaving_pov)
            if hasattr(viz, "set_state"):
                viz.set_state(pov_hidden_node=None)
            if previous_mode == "pov" and hasattr(camera, "clear_pre_pov_camera_state"):
                camera.clear_pre_pov_camera_state()
            if hasattr(viz.renderer, "reset_follow_state"):
                viz.renderer.reset_follow_state()
            camera.focus_on_target()
            logger.debug("Follow mode: camera will track selected entity")

        elif mode == "pov":
            if previous_mode != "pov" and hasattr(camera, "capture_pre_pov_camera_state"):
                camera.capture_pre_pov_camera_state()
            if not camera.set_pov_camera():
                if hasattr(viz, "set_state"):
                    viz.set_state(
                        camera_mode=previous_mode,
                        fly_mode=previous_fly_mode,
                    )
                if previous_mode != "pov" and hasattr(camera, "clear_pre_pov_camera_state"):
                    camera.clear_pre_pov_camera_state()
                if track_group:
                    track_group.setVisible(
                        previous_mode in ("follow", "pov") or show_advanced_track
                    )
                if pov_axis_container:
                    pov_axis_container.setVisible(previous_mode == "pov")
                logger.warning("POV mode change rolled back because camera setup failed")
                return
            logger.debug("POV mode: first-person view from selected entity")

    def handle_fly_mode_toggled(self, checked: bool) -> None:
        """Toggle fly mode, forcing overview camera mode when needed."""
        viz = self.visualizer
        enabled = bool(checked)

        if enabled and getattr(getattr(viz, "app_state", None), "camera_mode", "overview") != (
            "overview"
        ):
            self.handle_camera_mode_changed("overview", True)

        viz.set_state(fly_mode=enabled)
        camera = getattr(viz, "camera_controller", None)
        tools = camera.presentation() if camera and hasattr(camera, "presentation") else None
        if tools is not None:
            tools.stop()

    def handle_camera_minimap_toggled(self, checked: bool) -> None:
        """Toggle the renderer-provided minimap inset."""
        viz = self.visualizer
        viz.set_state(show_camera_minimap=bool(checked))
        renderer = getattr(viz, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).camera_minimap:
            try:
                renderer.set_camera_minimap_visible(bool(checked))
            except (AttributeError, RuntimeError, ValueError) as exc:
                logger.debug("Failed to toggle camera minimap: %s", exc)

    def handle_pov_axis_combo_changed(self, axis: str) -> None:
        """Update POV forward-axis selection and reapply active POV camera."""
        if not axis:
            return

        logger.debug("POV axis changed to: %s", axis)

        viz = self.visualizer
        if hasattr(viz, "app_state"):
            viz.set_state(pov_axis=axis)

        if hasattr(viz, "app_state") and viz.app_state.camera_mode == "pov":
            camera = getattr(viz, "camera_controller", None)
            if camera:
                camera.set_pov_camera()
                logger.debug("Reapplied POV camera with axis: %s", axis)
            if viz.renderer is not None:
                if hasattr(viz.renderer, "request_redraw"):
                    viz.renderer.request_redraw()
                else:
                    viz.renderer.update_renderer()
