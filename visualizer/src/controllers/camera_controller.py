"""Camera intent controller for overview, follow, POV, and saved views.

The controller resolves user-facing selections (focus dropdowns, view buttons,
POV mode, saved-view slots) into renderer-neutral camera calls. Renderers own
backend camera math and redraw mechanics; this layer owns selection policy and
session-friendly preset serialization.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from shared.logging import get_logger

from ..services.camera_scene_query_service import CameraSceneQueryService
from ..services.pov_visibility_service import PovVisibilityService
from ..types.camera_state import CameraState

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.camera_controller")


class CameraController:
    """Translate UI camera intent into renderer protocol calls."""

    MAX_PRESETS = 4
    VIEW_PRESETS = {
        "top": "top",
        "side": "side",
        "front": "front",
        "isometric": "isometric",
        "iso": "isometric",
    }

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        camera_scene_query_service: CameraSceneQueryService,
    ) -> None:
        """Bind camera workflows to the active visualizer instance."""
        self.visualizer = visualizer
        self.scene_query = camera_scene_query_service
        self._preferred_focus_data: Optional[dict[str, Any]] = {"type": "auto"}
        self._pov_visibility = PovVisibilityService(visualizer)
        # Presets store neutral CameraState so sessions are backend-independent.
        self._camera_presets: dict[int, dict[str, Any]] = {}
        # Follow mode suppresses tiny target deltas to avoid jitter.
        self._last_follow_target_pos: Optional[np.ndarray] = None
        # Camera state to restore when leaving POV back to Overview.
        self._pre_pov_camera_state: Optional[CameraState] = None
        # Guard against re-entrant POV data bootstrapping.
        self._pov_bootstrap_in_progress: bool = False
        self._camera_debug_enabled: bool = os.getenv("ORCHAV_CAMERA_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def _camera_debug(self, message: str, **fields: Any) -> None:
        """Emit camera controller diagnostics when ORCHAV_CAMERA_DEBUG is enabled."""
        if not self._camera_debug_enabled:
            return
        if fields:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            logger.info("CameraDebugController: %s %s", message, details)
        else:
            logger.info("CameraDebugController: %s", message)

    def set_overview_view(
        self,
        view: str,
        camera_dist: Optional[float] = None,
        fov: Optional[float] = None,
    ) -> bool:
        """Set a named overview view using whole-scene bounds."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if not hasattr(viz, "vis_initialized") or not viz.vis_initialized or renderer is None:
            logger.warning("Cannot set overview view: visualizer not initialized")
            return False

        view_key = self._normalize_view(view)
        if view_key is None:
            logger.warning("Unknown overview view '%s'", view)
            return False

        bounds = self.scene_query.compute_scene_bounds(scope="whole")
        if bounds is None:
            logger.warning("Cannot set overview view: no scene bounds available")
            return False

        center, extent = bounds
        fov_value = float(fov) if fov is not None else 60.0
        if camera_dist is None:
            distance = self._compute_camera_distance(
                extent,
                fov_value,
                aspect=self.scene_query.viewport_aspect(),
                view=view_key,
            )
            adjust = getattr(renderer, "adjust_overview_distance", None)
            if callable(adjust):
                try:
                    distance = float(
                        adjust(
                            distance,
                            fov=fov_value,
                            aspect=self.scene_query.viewport_aspect(),
                            view=view_key,
                            center=center,
                            extent=extent,
                            bounds=(center, extent),
                        )
                    )
                except TypeError:
                    try:
                        distance = float(
                            adjust(
                                distance,
                                fov=fov_value,
                                aspect=self.scene_query.viewport_aspect(),
                                view=view_key,
                            )
                        )
                    except (TypeError, ValueError, AttributeError):
                        logger.debug("Renderer overview-distance adjustment failed", exc_info=True)
                except (ValueError, AttributeError):
                    logger.debug("Renderer overview-distance adjustment failed", exc_info=True)
        else:
            try:
                distance = max(0.1, float(camera_dist))
            except (TypeError, ValueError):
                distance = self._compute_camera_distance(
                    extent,
                    fov_value,
                    aspect=self.scene_query.viewport_aspect(),
                    view=view_key,
                )

        if not hasattr(renderer, "set_overview_camera"):
            logger.warning("Cannot set overview view: renderer camera API unavailable")
            return False

        try:
            success = renderer.set_overview_camera(
                view_key,
                (center, extent),
                fov_value,
                distance=distance,
            )
            if success:
                logger.debug("Applied overview view '%s' via renderer camera intent", view_key)
            return bool(success)
        except (RuntimeError, AttributeError, ValueError, TypeError) as exc:
            logger.warning("Failed to set overview view '%s': %s", view_key, exc)
            return False

    def save_camera_preset(self, preset_num: int, name: Optional[str] = None) -> bool:
        """Save the current renderer camera state to a preset slot."""
        if not 1 <= preset_num <= self.MAX_PRESETS:
            logger.warning(f"Invalid preset number {preset_num}, must be 1-{self.MAX_PRESETS}")
            return False

        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if not hasattr(viz, "vis_initialized") or not viz.vis_initialized or renderer is None:
            logger.warning("Cannot save camera preset: visualizer not initialized")
            return False

        try:
            preset_name = name or f"Preset {preset_num}"
            cam_state = None
            if hasattr(renderer, "get_camera_state"):
                try:
                    candidate = renderer.get_camera_state()
                    cam_state = candidate if isinstance(candidate, CameraState) else None
                except (RuntimeError, AttributeError, ValueError, TypeError) as exc:
                    logger.debug("Camera state unavailable for preset save: %s", exc)

            if cam_state is None:
                logger.warning("Cannot save camera preset: camera state unavailable")
                return False

            self._camera_presets[preset_num] = {
                "state": cam_state,
                "name": preset_name,
            }

            logger.info("Saved camera view slot %s: '%s'", preset_num, preset_name)
            return True

        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            logger.error("Error saving camera view slot %s: %s", preset_num, e)
            return False

    def load_camera_preset(self, preset_num: int) -> bool:
        """Load a saved camera preset through the renderer camera API."""
        if not 1 <= preset_num <= self.MAX_PRESETS:
            logger.warning(f"Invalid preset number {preset_num}, must be 1-{self.MAX_PRESETS}")
            return False

        if preset_num not in self._camera_presets:
            logger.debug("Camera view slot %s not set", preset_num)
            return False

        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if not hasattr(viz, "vis_initialized") or not viz.vis_initialized or renderer is None:
            logger.warning("Cannot load camera preset: visualizer not initialized")
            return False

        try:
            preset = self._camera_presets[preset_num]
            cam_state = preset["state"]
            if not isinstance(cam_state, CameraState):
                logger.warning("Cannot load camera preset: invalid camera state")
                return False
            if not hasattr(renderer, "set_camera_state"):
                logger.warning("Cannot load camera preset: set_camera_state unavailable")
                return False
            if not renderer.set_camera_state(cam_state):
                logger.warning("Cannot load camera preset: set_camera_state failed")
                return False

            logger.info("Loaded camera view slot %s: '%s'", preset_num, preset["name"])
            return True

        except (RuntimeError, ValueError) as e:
            logger.error("Error loading camera view slot %s: %s", preset_num, e)
            return False

    def get_preset_info(self, preset_num: int) -> Optional[dict[str, Any]]:
        """Return display metadata for a preset slot, if populated."""
        if preset_num in self._camera_presets:
            return {"name": self._camera_presets[preset_num]["name"]}
        return None

    def get_all_presets(self) -> dict[int, str]:
        """Return saved preset names keyed by preset slot."""
        return {num: p["name"] for num, p in self._camera_presets.items()}

    def clear_preset(self, preset_num: int) -> bool:
        """Clear one preset slot."""
        if preset_num in self._camera_presets:
            del self._camera_presets[preset_num]
            logger.info("Cleared camera view slot %s", preset_num)
            return True
        return False

    def _normalize_view(self, view: str) -> Optional[str]:
        """Normalize a user/session view alias to a preset key."""
        if not view:
            return None
        return self.VIEW_PRESETS.get(str(view).strip().lower())

    def _compute_camera_distance(
        self,
        extent: np.ndarray,
        fov: float,
        *,
        aspect: float = 16.0 / 9.0,
        view: str = "isometric",
    ) -> float:
        """Compute an overview distance that fits the scene in the viewport."""
        extent_arr = np.asarray(extent, dtype=np.float64).reshape(-1)
        if extent_arr.size < 3:
            return 10.0
        width_axis, height_axis = self._view_plane_extents(extent_arr[:3], view)
        width_axis = max(float(width_axis), 1e-3)
        height_axis = max(float(height_axis), 1e-3)
        fov_rad = np.radians(max(10.0, min(120.0, float(fov))))
        aspect = max(float(aspect), 1e-3)
        h_fov_rad = 2.0 * np.arctan(np.tan(fov_rad / 2.0) * aspect)
        fit_height = (height_axis * 0.5) / max(np.tan(fov_rad / 2.0), 1e-3)
        fit_width = (width_axis * 0.5) / max(np.tan(h_fov_rad / 2.0), 1e-3)
        return max(fit_height, fit_width, 10.0) * 1.2

    def _view_plane_extents(self, extent: np.ndarray, view: str) -> tuple[float, float]:
        """Map 3D scene extent to the 2D plane visible for a named view."""
        if view == "top":
            return float(extent[0]), float(extent[1])
        if view == "front":
            return float(extent[1]), float(extent[2])
        if view == "side":
            return float(extent[0]), float(extent[2])
        horizontal = max(float(extent[0]), float(extent[1]))
        vertical = max(float(extent[2]), horizontal * 0.6)
        return horizontal, vertical

    def focus_on_target(self) -> None:
        """Focus camera on the selected target, TX, or RX."""
        viz = self.visualizer
        if not hasattr(viz, "vis_initialized") or not viz.vis_initialized or viz.vis is None:
            logger.warning("Cannot focus: visualizer not initialized")
            return

        try:
            # The focus dropdown owns user selection; the renderer owns camera math.
            target_pos = self.scene_query.get_focus_position()

            if target_pos is None:
                logger.warning("No target position available for focus")
                return

            if not hasattr(viz.renderer, "focus_camera"):
                logger.warning("Cannot focus: renderer camera API unavailable")
                return
            success = viz.renderer.focus_camera(np.asarray(target_pos, dtype=np.float64))
            if success:
                self._last_follow_target_pos = np.asarray(target_pos, dtype=np.float64)
            else:
                logger.debug("Renderer focus_camera failed")
            self._camera_debug(
                "focus_on_target",
                success=success,
                target=np.asarray(target_pos, dtype=np.float64).tolist(),
            )

            logger.debug("Camera focused on target position")

        except (RuntimeError, ValueError) as e:
            logger.error("Error in focus_on_target: %s", e)

    def capture_pre_pov_camera_state(self) -> bool:
        """Capture the current camera before entering POV, if available."""
        if self._pre_pov_camera_state is not None:
            return True

        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is None or not hasattr(renderer, "get_camera_state"):
            return False

        try:
            candidate = renderer.get_camera_state()
            camera_state = candidate if isinstance(candidate, CameraState) else None
        except (RuntimeError, AttributeError, ValueError, TypeError) as exc:
            logger.debug("Could not capture pre-POV camera state: %s", exc)
            return False

        if camera_state is None:
            logger.debug("Could not capture pre-POV camera state: unavailable")
            return False

        self._pre_pov_camera_state = camera_state
        return True

    def clear_pre_pov_camera_state(self) -> None:
        """Discard any pending pre-POV camera restore state."""
        self._pre_pov_camera_state = None

    def restore_pre_pov_camera_state(self, *, update_renderer: bool = True) -> bool:
        """Restore the camera captured before entering POV."""
        state = self._pre_pov_camera_state
        if state is None:
            return False

        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is None or not hasattr(renderer, "set_camera_state"):
            return False

        try:
            if not renderer.set_camera_state(state):
                logger.debug("Could not restore pre-POV camera state")
                return False
            self._pre_pov_camera_state = None
            return True
        except (RuntimeError, AttributeError, ValueError, TypeError) as exc:
            logger.debug("Could not restore pre-POV camera state: %s", exc)
            return False

    def set_pov_camera(self, *, defer_redraw: bool = False) -> bool:
        """Set first-person POV from the selected target, TX, or RX."""
        viz = self.visualizer
        if not hasattr(viz, "vis_initialized") or not viz.vis_initialized or viz.vis is None:
            logger.warning("Cannot set POV: visualizer not initialized")
            return False

        previous_hidden_entity: tuple[str, int] | None = None
        visibility_staged = False
        camera_applied = False
        try:
            position, orientation, entity_info = (
                self.scene_query.get_entity_position_orientation_and_info()
            )
            if position is None:
                # POV can be toggled before current frame data is fully materialized.
                # Do a one-shot frame refresh so POV applies immediately without
                # requiring manual "next frame" interaction.
                if self._bootstrap_pov_frame_data():
                    (
                        position,
                        orientation,
                        entity_info,
                    ) = self.scene_query.get_entity_position_orientation_and_info()

            if position is None:
                logger.warning("No entity position available for POV")
                self._camera_debug("set_pov_camera_failed_no_position")
                return False

            # Hide the selected entity so first-person views do not render from
            # inside the same marker/orientation frame.
            previous_hidden_entity = self._pov_visibility.current_hidden_entity()
            if entity_info is not None:
                if not self._hide_pov_entity(entity_info):
                    logger.warning("Cannot set POV: entity visibility synchronization failed")
                    self._camera_debug("set_pov_camera_failed_visibility")
                    return False
                visibility_staged = True

            # AppState owns which local entity axis acts as the camera forward.
            pov_axis = (
                getattr(viz.app_state, "pov_axis", "forward")
                if hasattr(viz, "app_state")
                else "forward"
            )
            logger.debug("Using POV axis: %s", pov_axis)

            if orientation is None:
                logger.debug("No orientation data, using default forward direction (+X)")
                orientation = [0.0, 0.0, 0.0]  # yaw=0, pitch=0, roll=0

            renderer = getattr(viz, "renderer", None)
            if renderer is None or not hasattr(renderer, "set_pov_camera"):
                logger.warning("POV camera requires renderer camera API")
                return False

            ok = renderer.set_pov_camera(
                position,
                orientation,
                pov_axis,
                defer_redraw=defer_redraw,
            )
            if not ok:
                logger.warning("POV camera set_pov_camera failed")
                return False
            camera_applied = True
            logger.debug("POV camera set, axis=%s", pov_axis)
            self._camera_debug(
                "set_pov_camera",
                axis=pov_axis,
                position=np.asarray(position, dtype=np.float64).tolist(),
                orientation=[float(x) for x in orientation[:3]],
            )
            return True

        except (RuntimeError, ValueError) as e:
            self._camera_debug("set_pov_camera_error", error=str(e))
            logger.exception("Error in set_pov_camera")
            return False
        finally:
            if visibility_staged and not camera_applied:
                self._pov_visibility.set_hidden_entity(previous_hidden_entity)

    def _bootstrap_pov_frame_data(self) -> bool:
        """Best-effort one-shot frame bootstrap so POV can resolve entity positions."""
        viz = self.visualizer

        if self._pov_bootstrap_in_progress:
            return False

        if not hasattr(viz, "_process_frame_step"):
            return False

        renderer = getattr(viz, "renderer", None)
        # If already inside frame pipeline update, never recurse.
        if bool(getattr(renderer, "_frame_update_in_progress", False)):
            return False

        step = getattr(viz, "animation_step", None)
        if step is None:
            step = getattr(getattr(viz, "app_state", None), "step", 0)

        try:
            self._pov_bootstrap_in_progress = True
            viz._process_frame_step(int(step))
            return True
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("POV bootstrap frame refresh failed: %s", exc)
            return False
        finally:
            self._pov_bootstrap_in_progress = False

    def restore_pov_entity_visibility(self, update_renderer: bool = True) -> None:
        """Restore visibility of entities hidden during POV mode."""
        self._pov_visibility.restore(update_renderer=update_renderer)

    def update_follow_camera_focus(self) -> None:
        """Update follow-mode camera focus during animation ticks."""
        viz = self.visualizer
        if not hasattr(viz, "vis_initialized") or not viz.vis_initialized or viz.vis is None:
            return

        try:
            # Follow camera updates are valid only in Follow mode.
            camera_mode = getattr(getattr(viz, "app_state", None), "camera_mode", None)
            if camera_mode != "follow":
                self._last_follow_target_pos = None
                if hasattr(viz.renderer, "reset_follow_state"):
                    viz.renderer.reset_follow_state()
                return

            logger.debug("Auto-focus enabled, checking target position")

            target_pos = self.scene_query.get_focus_position()
            if target_pos is None:
                logger.debug("No target position available for auto-focus")
                self._last_follow_target_pos = None
                return

            target_pos_array = np.array(target_pos, dtype=np.float64)

            if not hasattr(viz.renderer, "update_follow_camera"):
                logger.warning("Renderer missing update_follow_camera")
                return

            if self._last_follow_target_pos is not None:
                distance = np.linalg.norm(target_pos_array - self._last_follow_target_pos)
                if distance < 0.1:
                    self._camera_debug(
                        "follow_skip_small_target_delta",
                        target_delta=f"{distance:.6f}",
                        threshold="0.100000",
                    )
                    return

            success = viz.renderer.update_follow_camera(target_pos_array)
            if success:
                self._last_follow_target_pos = target_pos_array.copy()
                logger.debug("Renderer follow camera updated")
                self._camera_debug(
                    "follow_update",
                    step=getattr(getattr(viz, "app_state", None), "step", None),
                    target=target_pos_array.tolist(),
                )
            else:
                self._camera_debug("follow_update_failed", target=target_pos_array.tolist())

        except (RuntimeError, ValueError) as e:
            logger.error("Error in update_follow_camera_focus: %s", e)

    def _get_camera_eye_position(self) -> Optional[np.ndarray]:
        """Return the current renderer camera eye position, if available."""
        viz = self.visualizer
        try:
            if hasattr(viz.renderer, "get_camera_state"):
                candidate = viz.renderer.get_camera_state()
                cam_state = candidate if isinstance(candidate, CameraState) else None
                if cam_state is not None:
                    return np.asarray(cam_state.eye, dtype=np.float64)
        except (RuntimeError, ValueError) as exc:
            logger.debug("Could not get camera position: %s", exc)
        return None

    def update_target_focus_dropdown(self) -> None:
        """Populate focus choices from the current ViewModel and frame mirrors."""
        viz = self.visualizer
        if not hasattr(viz, "target_focus_dropdown"):
            return

        try:
            dropdown = viz.target_focus_dropdown
            previous_data = dropdown.currentData()
            previous_text = dropdown.currentText()
            desired_data = self._preferred_focus_data or previous_data

            dropdown.blockSignals(True)
            dropdown.clear()

            dropdown.addItem("Auto (First Target)", {"type": "auto"})

            # Prefer ViewModel data because it matches the frame currently shown.
            target_count = 0
            if hasattr(viz, "current_view_model") and viz.current_view_model:
                if (
                    hasattr(viz.current_view_model, "target_metadata")
                    and viz.current_view_model.target_metadata
                ):
                    for i in range(len(viz.current_view_model.target_metadata)):
                        selection = self.scene_query.target_focus_selection(i)
                        if selection is None:
                            continue
                        display = self.scene_query.format_target_label(
                            i,
                            canonical_index=selection["index"],
                        )
                        dropdown.addItem(display, selection)
                        target_count += 1

            from ..scene.geometry_helpers import resolve_node_label

            app_state = getattr(viz, "app_state", None)
            tx_custom = getattr(app_state, "tx_labels", ())
            rx_custom = getattr(app_state, "rx_labels", ())
            tx_device_names = getattr(app_state, "tx_device_names", ())
            rx_device_names = getattr(app_state, "rx_device_names", ())
            label_mode = getattr(app_state, "node_label_mode", "role")

            tx_count = 0
            if hasattr(viz, "current_view_model") and viz.current_view_model:
                if hasattr(viz.current_view_model, "tx_positions"):
                    tx_count = len(viz.current_view_model.tx_positions)
                    for i in range(tx_count):
                        text = resolve_node_label(
                            "TX",
                            i,
                            tx_custom,
                            label_mode=label_mode,
                            device_names=tx_device_names,
                        )
                        dropdown.addItem(text, {"type": "tx", "index": i})

            rx_count = 0
            if hasattr(viz, "current_view_model") and viz.current_view_model:
                if hasattr(viz.current_view_model, "rx_positions"):
                    rx_count = len(viz.current_view_model.rx_positions)
                    for i in range(rx_count):
                        text = resolve_node_label(
                            "RX",
                            i,
                            rx_custom,
                            label_mode=label_mode,
                            device_names=rx_device_names,
                        )
                        dropdown.addItem(text, {"type": "rx", "index": i})

            # Startup and partial-frame paths may only have frame mirrors.
            if tx_count == 0 and hasattr(viz, "current_tx_positions"):
                tx_count = (
                    len(viz.current_tx_positions) if viz.current_tx_positions is not None else 0
                )
                for i in range(tx_count):
                    text = resolve_node_label(
                        "TX",
                        i,
                        tx_custom,
                        label_mode=label_mode,
                        device_names=tx_device_names,
                    )
                    dropdown.addItem(text, {"type": "tx", "index": i})

            if rx_count == 0 and hasattr(viz, "current_rx_positions"):
                rx_count = (
                    len(viz.current_rx_positions) if viz.current_rx_positions is not None else 0
                )
                for i in range(rx_count):
                    text = resolve_node_label(
                        "RX",
                        i,
                        rx_custom,
                        label_mode=label_mode,
                        device_names=rx_device_names,
                    )
                    dropdown.addItem(text, {"type": "rx", "index": i})

            restored = False
            if desired_data:
                idx = self._find_matching_index(dropdown, desired_data)
                if idx is not None:
                    dropdown.setCurrentIndex(idx)
                    restored = True

            if not restored and previous_text:
                idx = dropdown.findText(previous_text)
                if idx != -1:
                    dropdown.setCurrentIndex(idx)
                    restored = True

            if not restored:
                dropdown.setCurrentIndex(0)

            dropdown.blockSignals(False)

            self._preferred_focus_data = dropdown.currentData()

            total_items = target_count + tx_count + rx_count + 1

            logger.debug(
                "Updated focus dropdown: %s items (%s targets, %s TX, %s RX)",
                total_items,
                target_count,
                tx_count,
                rx_count,
            )

        except (KeyError, AttributeError, ValueError, TypeError) as e:
            logger.error("Error updating target focus dropdown: %s", e)

    def remember_focus_selection(self) -> None:
        """Persist the current dropdown selection as the preferred focus target."""
        dropdown = getattr(self.visualizer, "target_focus_dropdown", None)
        if dropdown is None:
            return
        self._preferred_focus_data = dropdown.currentData()

    def _find_matching_index(self, dropdown, desired_data) -> Optional[int]:
        """Return the dropdown index whose item data matches *desired_data*."""
        for idx in range(dropdown.count()):
            if dropdown.itemData(idx) == desired_data:
                return idx
        return None

    def _hide_pov_entity(self, entity_info: Optional[dict]) -> bool:
        """Hide the sphere and orientation frame of the POV entity."""
        return self._pov_visibility.hide(entity_info)
