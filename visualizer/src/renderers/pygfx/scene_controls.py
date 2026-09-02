"""Scene-control helpers for the pygfx renderer.

This mixin contains runtime controls that affect the pygfx scene as a whole:
background and IBL state, light rig controls, line/point size overrides,
scene-bounds computation, renderer cleanup, antialiasing/depth passes, and
compatibility methods expected by controller code.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...types.camera_state import CameraState
from ..camera_ops import object_contributes_to_camera_bounds
from .camera import SceneBounds
from .canvas import _AA_MODES, _refresh_renderer_effect_passes
from .lighting_profiles import (
    CUSTOM_PROFILE,
    PYGFX_LIGHTING_PROFILE_LABELS,
    PYGFX_LIGHTING_PROFILES,
)

logger = logging.getLogger(__name__)


class PygfxSceneControlsMixin:
    """Scene-wide controls and compatibility adapters for ``PygfxRenderer``."""

    def _remove_solid_background(self) -> None:
        """Remove the solid-color background object if it is attached."""
        if self._solid_background is None:
            return
        if self._scene is not None:
            try:
                self._scene.remove(self._solid_background)
            except (ValueError, RuntimeError):
                pass
        self._solid_background = None

    def _sync_solid_background(self) -> bool:
        """Ensure a solid background is present when no skybox is visible."""
        if self._scene is None or self._skybox_visible:
            return False
        self._remove_solid_background()
        try:
            self._solid_background = self._gfx.Background.from_color(self._clear_color)
            self._scene.add(self._solid_background)
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("PygfxRenderer: failed to set solid background: %s", exc)
            self._solid_background = None
            return False

    def _apply_active_ibl_to_scene_materials(self) -> bool:
        """Apply the loaded IBL to the scene and all eligible materials."""
        if self._scene is None:
            return False
        if not self._ibl_manager.apply_to_scene(self._scene):
            return False
        if self._skybox_visible:
            self._remove_solid_background()
        else:
            self._ibl_manager.set_skybox_visible(False, self._scene)
            self._sync_solid_background()
        for name, obj in self._objects.items():
            if name == self.COVERAGE_MESH_NAME:
                continue
            mat = getattr(obj, "material", None)
            if mat is not None and hasattr(mat, "env_map"):
                self._ibl_manager.apply_to_material(mat)
        self._ibl_loaded = True
        # Replay cached material payloads through set_named_material so every
        # object re-runs the full material-setup path under _ibl_loaded=True
        # (metallic clamp unlocked, env_map applied via the complete
        # pipeline, not just the attribute setter).
        cached_materials = list(self._materials.items())
        for name, mat_payload in cached_materials:
            if name in self._objects:
                self.set_named_material(name, mat_payload)
        return True

    def _load_ibl_into_scene(self, name: str) -> bool:
        """Load an IBL asset by canonical name and apply it to the scene."""
        if self._ibl_manager.load_ibl(name) is None:
            return False
        self._ibl_name = name
        return self._apply_active_ibl_to_scene_materials()

    def _load_deferred_default_ibl(self) -> None:
        """Load the default IBL after initial canvas setup has returned to Qt."""
        self._deferred_ibl_load_scheduled = False
        name = self._deferred_default_ibl_name
        if name is None:
            return
        if not self._initialized or self._scene is None:
            return
        self._deferred_default_ibl_name = None
        started = time.perf_counter()
        if self._load_ibl_into_scene(name):
            logger.info(
                "PygfxRenderer: deferred default IBL '%s' loaded in %.1f ms",
                name,
                (time.perf_counter() - started) * 1000.0,
            )
            # The IBL mutation is complete; let rendercanvas present it on the
            # next Qt turn instead of blocking inside a nested paint.
            self.request_redraw()

    def _schedule_deferred_default_ibl_load(self) -> None:
        """Queue default IBL loading on the next Qt turn when configured."""
        if (
            not self._defer_default_ibl
            or self._ibl_loaded
            or self._deferred_default_ibl_name is None
            or self._deferred_ibl_load_scheduled
        ):
            return
        try:
            from PySide6.QtCore import QTimer
        except ImportError:
            self._load_deferred_default_ibl()
            return
        self._deferred_ibl_load_scheduled = True
        session_generation = getattr(self, "_session_generation", 0)

        def _load_for_current_session() -> None:
            if getattr(self, "_session_generation", 0) != session_generation:
                return
            self._load_deferred_default_ibl()

        QTimer.singleShot(0, _load_for_current_session)

    def set_background_color(self, color: list[float]) -> None:
        """Set clear color and rebuild passes that bake the background color."""
        if len(color) >= 3:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
            a = float(color[3]) if len(color) > 3 else 1.0
            self._clear_color = (r, g, b, a)
            self._rebuild_effect_passes()
            if not self._skybox_visible:
                self._sync_solid_background()

    def set_line_width(self, width: float) -> bool:
        """Update MPC line thickness for all live MPC line geometries."""
        self._line_width = float(width)
        updated = False
        for name in self._mpc_line_names():
            updated = self._apply_named_visual_overrides(name) or updated
        if updated:
            self.request_redraw()
        return True

    def set_edge_line_width(self, width: float) -> bool:
        """Update outline/edge line thickness for live edge geometries."""
        self._edge_line_width = float(width)
        updated = False
        for name in self._edge_line_names():
            updated = self._apply_named_visual_overrides(name, is_edge=True) or updated
        if updated:
            self.request_redraw()
        return True

    def set_trajectory_line_width(self, width: float) -> bool:
        """Update thickness on live trajectory line materials."""
        self.trajectory_line_width = float(width)
        updated = False
        for name in self._trajectory_line_names():
            obj = self._objects.get(name)
            if obj is None:
                continue
            mat = getattr(obj, "material", None)
            if mat is None or not hasattr(mat, "thickness"):
                continue
            try:
                mat.thickness = float(width)
                updated = True
            except Exception:
                continue
        if updated:
            self.request_redraw()
        return True

    def set_trajectory_point_size(self, size: float) -> bool:
        """Update marker size on live trajectory point materials."""
        self.trajectory_point_size = float(size)
        updated = False
        for name in self._trajectory_point_names():
            obj = self._objects.get(name)
            if obj is None:
                continue
            mat = getattr(obj, "material", None)
            if mat is None or not hasattr(mat, "size"):
                continue
            try:
                mat.size = float(size)
                updated = True
            except Exception:
                continue
        if updated:
            self.request_redraw()
        return True

    def set_point_size(self, size: float) -> bool:
        """Update the global MPC point size and refresh live point geometry."""
        self._point_size = float(size)
        updated = self._apply_named_visual_overrides("mpc_points")
        if updated:
            self.request_redraw()
        return True

    def reset_state(self) -> None:
        """Clear scene objects and forget the last applied frame packet."""
        self.clear()
        self.last_frame_packet = None

    def clear(self) -> None:
        """Remove renderer-owned scene objects and reset cached render state."""
        clear_mpc_selection = getattr(self, "clear_mpc_path_inspection", None)
        if callable(clear_mpc_selection):
            clear_mpc_selection()
        cancel_mpc_click = getattr(self, "_cancel_mpc_path_click_gesture", None)
        if callable(cancel_mpc_click):
            cancel_mpc_click()
        invalidate_mpc_pick = getattr(self, "_invalidate_mpc_pick_cache", None)
        if callable(invalidate_mpc_pick):
            invalidate_mpc_pick()
        clear_gizmo = getattr(self, "clear_transform_gizmo", None)
        if callable(clear_gizmo):
            clear_gizmo()
        self._ground_grid_needs_rebuild = bool(self._ground_grid_visible)
        self._remove_ground_grid()
        self._set_marker_legend_visible(False)
        self._update_mpc_hud_overlays(None)
        for name in list(self._name_to_handle):
            self.remove_named_geometry(name)
        remaining_names = set(self._name_to_handle)
        self._payload_cache.clear()
        for name in tuple(self._render_object_snapshots):
            if name not in remaining_names:
                self._render_object_snapshots.pop(name, None)
        dirty_geometry = getattr(self, "_dirty_render_object_geometry", None)
        if dirty_geometry is not None:
            dirty_geometry.intersection_update(remaining_names)
        uncertain_indices = getattr(self, "_uncertain_mesh_index_buffers", None)
        if uncertain_indices is not None:
            uncertain_indices.intersection_update(remaining_names)
        incompatible_transitions = getattr(
            self,
            "_vertex_stream_incompatible_transitions",
            None,
        )
        if incompatible_transitions is not None:
            for name in tuple(incompatible_transitions):
                if name not in remaining_names:
                    incompatible_transitions.pop(name, None)
        rebuild_names = getattr(self, "_vertex_stream_rebuild_names", None)
        if rebuild_names is not None:
            rebuild_names.intersection_update(remaining_names)
        self._mpc_lines_source_sig = None
        self._mpc_points_source_sig = None
        self._last_coverage_signature = None
        self._applied_coverage_state = None
        self._beamforming_owned_names.intersection_update(remaining_names)
        for name in tuple(self._applied_beamforming_surfaces):
            if name not in self._beamforming_owned_names:
                self._applied_beamforming_surfaces.pop(name, None)
        self._edge_geometry_names.intersection_update(remaining_names)
        self._clear_mpc_buffers()
        if remaining_names:
            logger.warning(
                "PygfxRenderer: clear retained %d object name(s) after native "
                "removal failure; the next clear/frame sync will retry",
                len(remaining_names),
            )

    def clear_scene_geometry(self) -> None:
        """Compatibility alias for clearing renderer-owned scene objects."""
        self.clear()

    def reset_camera_bounds(self) -> None:
        """Reframe the camera and shadow extent around the current whole scene."""
        bbox = self.compute_scene_bounds(scope="whole")
        if bbox is None:
            return
        center = np.asarray(bbox.get_center(), dtype=np.float64)
        extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        max_extent = float(np.max(extent))
        if not np.isfinite(max_extent) or max_extent < 1e-3:
            max_extent = 10.0
        fov_deg = float(getattr(self._camera, "fov", 60.0))
        set_overview = getattr(self, "set_overview_camera", None)
        if callable(set_overview) and bool(
            set_overview(
                "isometric",
                bbox,
                fov=fov_deg,
            )
        ):
            self._update_shadow_extent(max_extent)
            return

        fov_rad = np.deg2rad(max(15.0, min(120.0, fov_deg)))
        distance = max((max_extent * 0.5) / max(np.tan(fov_rad * 0.5), 1e-3) * 1.35, 10.0)
        eye = center + np.array(
            [distance * 0.35, -distance * 0.55, distance * 0.45], dtype=np.float64
        )
        self.set_camera_state(
            CameraState(
                eye=tuple(float(x) for x in eye),
                lookat=tuple(float(x) for x in center),
                up=(0.0, 0.0, 1.0),
                fov_deg=fov_deg,
            )
        )
        if self._controller is not None and hasattr(self._controller, "target"):
            try:
                self._controller.target = tuple(float(x) for x in center)
            except Exception:
                pass
        self._update_shadow_extent(max_extent)

    def set_fly_mode(self, enabled: bool) -> bool:
        """Swap between orbit and fly camera controllers while preserving view."""
        if not self._initialized or self._camera is None:
            return False
        target_type = "fly" if enabled else "orbit"
        if self._active_controller_type == target_type:
            return True
        cam_state = self.get_camera_state()

        router = getattr(self, "_pygfx_interaction_router", None)
        ctrl = self._create_camera_controller(
            target_type,
            camera_state=cam_state,
            register_events=router is None,
        )
        if ctrl is None:
            return False
        self._controller = ctrl
        self._active_controller_type = target_type
        self._focus_canvas()
        self.request_redraw()
        logger.info("PygfxRenderer: fly mode %s", "enabled" if enabled else "disabled")
        return True

    def get_ibl_intensity(self) -> Optional[float]:
        """Return the UI-scale IBL intensity currently stored by the renderer."""
        return float(self._ibl_intensity)

    def get_ibl_name(self) -> Optional[str]:
        """Return the canonical IBL name currently selected for the scene."""
        return self._ibl_name

    def set_ibl(self, ibl: str) -> bool:
        """Select an IBL by canonical name or accepted ``*_ibl.ktx`` path."""
        raw = str(ibl or "default")
        # The render panel may pass a KTX path (e.g. ".../neutral_outdoor_ibl.ktx").
        # Extract the canonical IBL name from the path so the manager can resolve
        # the corresponding HDR file.
        name = raw
        if raw.lower().endswith("_ibl.ktx"):
            stem = Path(raw).stem  # "neutral_outdoor_ibl"
            name = stem[:-4]  # strip "_ibl" suffix
        self._deferred_default_ibl_name = None
        self._deferred_ibl_load_scheduled = False
        loaded = self._load_ibl_into_scene(name)
        if loaded:
            self.request_redraw()
        return loaded

    def set_ibl_intensity(self, intensity: float) -> bool:
        """Set IBL intensity using the Open3D-scale UI value."""
        try:
            value = max(0.0, float(intensity))
        except (TypeError, ValueError):
            return False
        self._ibl_intensity = value
        # Map Open3D-scale (0-100k, default 30k) to pygfx (0-2, default 1).
        pygfx_value = max(0.0, value) / 30000.0
        self._ibl_manager.set_intensity(pygfx_value)
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def _mark_lighting_profile_custom(self) -> None:
        """Record that manual lighting edits no longer match a named profile."""
        if getattr(self, "_suppress_lighting_profile_custom", False):
            return
        self._lighting_profile_name = CUSTOM_PROFILE

    def get_lighting_profile(self) -> str:
        """Return the active pygfx lighting profile key."""
        return str(getattr(self, "_lighting_profile_name", CUSTOM_PROFILE))

    def get_available_lighting_profiles(self) -> dict[str, str]:
        """Return selectable pygfx lighting profile keys and display labels."""
        return dict(PYGFX_LIGHTING_PROFILE_LABELS)

    def set_lighting_profile(self, name: str) -> bool:
        """Apply one named pygfx lighting profile as renderer runtime state."""
        profile = PYGFX_LIGHTING_PROFILES.get(str(name or ""))
        if profile is None:
            return False

        previous_suppress = getattr(self, "_suppress_lighting_profile_custom", False)
        self._suppress_lighting_profile_custom = True
        try:
            self._base_ambient_intensity = float(profile.ambient_intensity)
            self._base_key_intensity = float(profile.key_intensity)
            self._base_fill_intensity = float(profile.fill_intensity)
            self._headlight_enabled = bool(profile.headlight_enabled)
            self._headlight_intensity = float(profile.headlight_intensity)
            self._key_light_azimuth_deg = float(profile.key_azimuth_deg)
            self._key_light_elevation_deg = float(profile.key_elevation_deg)
            self._fill_light_azimuth_deg = float(profile.fill_azimuth_deg)
            self._fill_light_elevation_deg = float(profile.fill_elevation_deg)
            self._shadows_enabled = bool(profile.shadows_enabled)

            self._apply_light_intensities()
            self._apply_world_light_transforms()
            if self._key_light is not None:
                try:
                    self._key_light.cast_shadow = self._shadows_enabled
                except (AttributeError, RuntimeError):
                    pass
            self.set_ibl_intensity(float(profile.ibl_intensity))
        finally:
            self._suppress_lighting_profile_custom = previous_suppress

        self._lighting_profile_name = profile.key
        self.request_redraw()
        return True

    def set_headlight_enabled(self, enabled: bool) -> bool:
        """Toggle the camera-follow headlight while preserving stored intensity."""
        self._headlight_enabled = bool(enabled)
        self._apply_light_intensities()
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def set_headlight_intensity(self, intensity: float) -> bool:
        """Set the headlight intensity used when the headlight is enabled."""
        try:
            value = max(0.0, float(intensity))
        except (TypeError, ValueError):
            return False
        self._headlight_intensity = value
        self._apply_light_intensities()
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def set_key_light_angles(self, azimuth_deg: float, elevation_deg: float) -> bool:
        """Set key-light direction using UI azimuth/elevation degrees."""
        try:
            self._key_light_azimuth_deg = float(azimuth_deg)
            self._key_light_elevation_deg = float(elevation_deg)
        except (TypeError, ValueError):
            return False
        self._apply_world_light_transforms()
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def set_fill_light_angles(self, azimuth_deg: float, elevation_deg: float) -> bool:
        """Set fill-light direction using UI azimuth/elevation degrees."""
        try:
            self._fill_light_azimuth_deg = float(azimuth_deg)
            self._fill_light_elevation_deg = float(elevation_deg)
        except (TypeError, ValueError):
            return False
        self._apply_world_light_transforms()
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def set_key_light_intensity(self, intensity: float) -> bool:
        """Set the stored key-light intensity and apply it to the live light."""
        try:
            self._base_key_intensity = max(0.0, float(intensity))
        except (TypeError, ValueError):
            return False
        self._apply_light_intensities()
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def set_fill_light_intensity(self, intensity: float) -> bool:
        """Set the stored fill-light intensity and apply it to the live light."""
        try:
            self._base_fill_intensity = max(0.0, float(intensity))
        except (TypeError, ValueError):
            return False
        self._apply_light_intensities()
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def get_light_rig_state(self) -> dict[str, float | bool | str]:
        """Return renderer-neutral light controls for session/UI mirroring."""
        return {
            "lighting_profile": self.get_lighting_profile(),
            "ambient_intensity": float(self._base_ambient_intensity),
            "headlight_enabled": bool(self._headlight_enabled),
            "headlight_intensity": float(self._headlight_intensity),
            "key_azimuth_deg": float(self._key_light_azimuth_deg),
            "key_elevation_deg": float(self._key_light_elevation_deg),
            "key_intensity": float(self._base_key_intensity),
            "fill_azimuth_deg": float(self._fill_light_azimuth_deg),
            "fill_elevation_deg": float(self._fill_light_elevation_deg),
            "fill_intensity": float(self._base_fill_intensity),
            "shadows_enabled": bool(self._shadows_enabled),
        }

    def poll_events(self) -> None:
        """Process pending Qt events so that canvas draws can complete."""
        self._event_pump_calls += 1
        try:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        except (ImportError, RuntimeError):
            pass

    def _include_object_in_scene_bounds(self, name: str) -> bool:
        """Return whether a named object should influence scene camera bounds."""
        return object_contributes_to_camera_bounds(name)

    def _compute_object_scene_bounds(self, *, include_hidden: bool) -> Optional[SceneBounds]:
        """Compute bounds from currently realized pygfx objects."""
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for name, obj in self._objects.items():
            if not self._include_object_in_scene_bounds(name):
                continue
            if not include_hidden and name in self._hidden:
                continue
            geom = getattr(obj, "geometry", None)
            pos_buf = getattr(geom, "positions", None)
            if pos_buf is None:
                continue
            points = getattr(pos_buf, "data", pos_buf)
            pts = np.asarray(points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
                continue
            pts3 = pts[:, :3]
            mat = getattr(getattr(obj, "local", None), "matrix", None)
            try:
                m = None if mat is None else np.asarray(mat, dtype=np.float64)
                if m is not None and m.shape == (4, 4):
                    ones = np.ones((pts3.shape[0], 1), dtype=np.float64)
                    world = np.concatenate([pts3, ones], axis=1) @ m.T
                    pts3 = world[:, :3]
            except Exception:
                pass
            mins.append(np.min(pts3, axis=0))
            maxs.append(np.max(pts3, axis=0))

        if not mins or not maxs:
            return None
        min_bound = np.min(np.vstack(mins), axis=0)
        max_bound = np.max(np.vstack(maxs), axis=0)
        return SceneBounds(min_bound=min_bound, max_bound=max_bound)

    def compute_scene_bounds(self, scope: str = "visible") -> Any:
        """Return bounds from pygfx objects that reached the active scene."""
        return self._compute_object_scene_bounds(include_hidden=scope == "whole")

    def is_geometry_in_scene(self, geometry: Any) -> bool:
        """Return whether an external geometry adapter is currently realized."""
        name = self._external_name_for_geometry(geometry)
        return name is not None and self.has_named_geometry(name)

    def is_geometry_visible(self, geometry: Any) -> Optional[bool]:
        """Return visibility for an external geometry adapter, if known."""
        name = self._external_name_for_geometry(geometry)
        if name is None:
            return None
        return self.is_named_visible(name)

    def _mpc_line_names(self) -> list[str]:
        """Return names that use the renderer-wide MPC line-width control."""
        return ["mpc_lines"] if "mpc_lines" in self._name_to_handle else []

    def _edge_line_names(self) -> list[str]:
        """Return names that use the renderer-wide edge-line-width control."""
        edge_names = getattr(self, "_edge_geometry_names", set())
        return [
            name
            for name in self._name_to_handle
            if name in edge_names or self._name_component(name) == "edge"
        ]

    def _apply_named_visual_overrides(self, name: str, *, is_edge: bool = False) -> bool:
        """Apply renderer-wide size controls to one named material if relevant."""
        obj = self._objects.get(name)
        if obj is None:
            return False

        mat = getattr(obj, "material", None)
        if mat is None:
            return False

        updated = False

        if hasattr(mat, "thickness"):
            line_width: float | None = None
            if name == "mpc_lines":
                line_width = self._line_width
            elif self._is_trajectory_line_name(name):
                line_width = self.trajectory_line_width
            elif (
                is_edge
                or name in getattr(self, "_edge_geometry_names", set())
                or self._name_component(name) == "edge"
            ):
                line_width = self._edge_line_width

            if line_width is not None:
                try:
                    mat.thickness = float(line_width)
                    updated = True
                except Exception:
                    pass

        if hasattr(mat, "size"):
            point_size: float | None = None
            if name == "mpc_points":
                point_size = self._point_size
            elif self._is_trajectory_point_name(name):
                point_size = self.trajectory_point_size

            if point_size is not None:
                try:
                    mat.size = float(point_size)
                    updated = True
                except Exception:
                    pass

        return updated

    # Internal helpers

    def _rebuild_effect_passes(self) -> None:
        """Rebuild the pygfx effect pass list from current instance state.

        Called after AA or internal debug-pass changes and also on
        background-color changes so the bloom/EDL/AA passes all stay in sync
        with the current state.
        """
        _refresh_renderer_effect_passes(
            self._renderer,
            self._gfx,
            clear_color=self._clear_color,
            aa_mode=self._aa_mode,
            depth_pass_enabled=self._depth_pass_enabled,
        )
        self.request_redraw()

    def set_antialiasing_mode(self, mode: str) -> None:
        """Switch the interactive AA mode. Options: off, fxaa, ddaa, ppaa."""
        normalized = str(mode).lower().strip()
        if normalized not in _AA_MODES:
            logger.warning("Unknown AA mode %r, ignoring", mode)
            return
        if normalized == self._aa_mode:
            return
        self._aa_mode = normalized
        self._rebuild_effect_passes()
        self._update_status_chip_overlay()

    def get_antialiasing_mode(self) -> str:
        """Return the active post-process antialiasing mode token."""
        return self._aa_mode

    def set_depth_pass_enabled(self, enabled: bool) -> None:
        """Toggle the debug/depth visualization pass."""
        new = bool(enabled)
        if new == self._depth_pass_enabled:
            return
        self._depth_pass_enabled = new
        self._rebuild_effect_passes()
        self._update_status_chip_overlay()

    def get_depth_pass_enabled(self) -> bool:
        """Return whether the debug depth/effect pass is active."""
        return self._depth_pass_enabled

    def set_background_image(self, image_path: str) -> bool:
        """Report unsupported image backgrounds for protocol compatibility."""
        logger.debug("PygfxRenderer: set_background_image not supported")
        return False

    def show_skybox(self, show: bool) -> bool:
        """Toggle skybox visibility while keeping the solid background coherent."""
        self._skybox_visible = bool(show)
        if self._skybox_visible:
            ok = self._ibl_manager.set_skybox_visible(True, self._scene)
            if ok:
                self._remove_solid_background()
        else:
            self._ibl_manager.set_skybox_visible(False, self._scene)
            ok = self._sync_solid_background()
        if ok:
            self.request_redraw()
        return ok

    def _install_default_lights(self) -> None:
        """Attach basic lights so lit materials are visible without IBL/sun controls."""
        if self._scene is None:
            return
        gfx = self._gfx
        try:
            self._ambient_light = gfx.AmbientLight(intensity=self._base_ambient_intensity)
            self._scene.add(self._ambient_light)
        except (AttributeError, RuntimeError):
            self._ambient_light = None
        # World key/fill lights (fixed to world frame).
        try:
            self._key_light = gfx.DirectionalLight(
                color=(1.0, 1.0, 1.0), intensity=self._base_key_intensity
            )
            self._scene.add(self._key_light)
        except (AttributeError, RuntimeError):
            self._key_light = None
        if self._key_light is not None and self._shadows_enabled:
            try:
                self._key_light.cast_shadow = True
                if hasattr(self._key_light, "shadow"):
                    shadow = self._key_light.shadow
                    if hasattr(shadow, "camera"):
                        cam = shadow.camera
                        extent = self._scene_extent * 1.5
                        if hasattr(cam, "width"):
                            cam.width = extent
                        if hasattr(cam, "height"):
                            cam.height = extent
                    if hasattr(shadow, "bias"):
                        shadow.bias = 0.001
            except (AttributeError, RuntimeError) as exc:
                logger.debug("PygfxRenderer: shadow setup failed: %s", exc)
        try:
            self._fill_light = gfx.DirectionalLight(
                color=(1.0, 1.0, 1.0), intensity=self._base_fill_intensity
            )
            self._scene.add(self._fill_light)
        except (AttributeError, RuntimeError):
            self._fill_light = None
        # Camera-follow headlight to stabilize specular highlights while navigating.
        try:
            self._head_light = gfx.DirectionalLight(
                color=(1.0, 1.0, 1.0), intensity=self._headlight_intensity
            )
            self._scene.add(self._head_light)
        except (AttributeError, RuntimeError):
            self._head_light = None
        self._apply_world_light_transforms()
        previous_suppress = getattr(self, "_suppress_lighting_profile_custom", False)
        self._suppress_lighting_profile_custom = True
        try:
            self.set_ibl_intensity(self._ibl_intensity)
        finally:
            self._suppress_lighting_profile_custom = previous_suppress

    def _apply_light_intensities(self) -> None:
        """Push stored light intensity values onto live pygfx lights."""
        ambient = self._base_ambient_intensity
        key = self._base_key_intensity
        fill = self._base_fill_intensity
        head = self._headlight_intensity if self._headlight_enabled else 0.0
        for light, level in (
            (self._ambient_light, ambient),
            (self._key_light, key),
            (self._fill_light, fill),
            (self._head_light, head),
        ):
            if light is None:
                continue
            try:
                if hasattr(light, "intensity"):
                    light.intensity = float(level)
            except (AttributeError, RuntimeError):
                continue

    def _apply_world_light_transforms(self) -> None:
        """Apply stored key/fill azimuth-elevation controls to world lights."""
        self._set_directional_light_from_angles(
            light=self._key_light,
            azimuth_deg=self._key_light_azimuth_deg,
            elevation_deg=self._key_light_elevation_deg,
            radius=80.0,
        )
        self._set_directional_light_from_angles(
            light=self._fill_light,
            azimuth_deg=self._fill_light_azimuth_deg,
            elevation_deg=self._fill_light_elevation_deg,
            radius=70.0,
        )

    def _apply_shadow_flags(self, name: str, obj: Any) -> None:
        """Set cast/receive shadow flags on *obj* based on geometry name."""
        is_physical_mesh = self._is_scene_mesh_name(name) or self._is_target_mesh_name(name)
        if is_physical_mesh:
            try:
                obj.cast_shadow = True
                obj.receive_shadow = True
            except (AttributeError, RuntimeError):
                pass
        else:
            try:
                obj.cast_shadow = False
                obj.receive_shadow = False
            except (AttributeError, RuntimeError):
                pass

    def set_shadow_enabled(self, enabled: bool) -> bool:
        """Toggle shadow casting on the key light."""
        self._shadows_enabled = bool(enabled)
        if self._key_light is not None:
            try:
                self._key_light.cast_shadow = self._shadows_enabled
            except (AttributeError, RuntimeError):
                return False
        self._mark_lighting_profile_custom()
        self.request_redraw()
        return True

    def _dump_lighting_state(self) -> None:
        """Debug helper: print every per-frame lighting input for diffing.

        Gated by ``ORCHAV_PYGFX_LIGHT_DEBUG=1``.  Dumps scene.environment,
        light list with intensities, and per-mesh material env_map /
        env_map_intensity / roughness / metallic for target meshes.  Run
        --no-resume, capture frame 0 log and frame 2 log, diff them.
        """
        import sys as _sys

        try:
            scene_env = (
                getattr(self._scene, "environment", "<no-env-attr>") if self._scene else None
            )
            lines: list[str] = []
            use_scene_env = getattr(self._ibl_manager, "_use_scene_environment", "?")
            tracked_count = len(getattr(self._ibl_manager, "_tracked_materials", []) or [])
            cur_intensity = getattr(self._ibl_manager, "_current_intensity", "?")
            lines.append(
                f"[light-debug] scene.environment={type(scene_env).__name__ if scene_env is not None else None} "
                f"ibl_loaded={self._ibl_loaded} ibl_name={self._ibl_name} "
                f"use_scene_env={use_scene_env} "
                f"ibl_tracked={tracked_count} ibl_current_intensity={cur_intensity}"
            )
            if self._scene is not None:
                for child in self._scene.children:
                    cname = type(child).__name__
                    if "Light" in cname or "Background" in cname:
                        intensity = getattr(child, "intensity", "?")
                        pos = getattr(getattr(child, "local", None), "position", "?")
                        cast = getattr(child, "cast_shadow", "?")
                        lines.append(
                            f"[light-debug]   scene_child={cname} "
                            f"intensity={intensity} pos={pos} cast_shadow={cast}"
                        )
            for name in sorted(self._objects.keys()):
                if "::mesh" not in name or "target" not in name:
                    continue
                obj = self._objects[name]
                mat = getattr(obj, "material", None)
                if mat is None:
                    continue
                env_map = getattr(mat, "env_map", "?")
                env_intensity = getattr(mat, "env_map_intensity", "?")
                roughness = getattr(mat, "roughness", "?")
                metallic = getattr(mat, "metalness", getattr(mat, "metallic", "?"))
                color = getattr(mat, "color", "?")
                # Also dump geometry state.  If normals are missing/zero,
                # the mesh renders as if all faces point the same way —
                # exactly what "flat uniform pink" would look like.
                geom = getattr(obj, "geometry", None)
                normals_buf = getattr(geom, "normals", None) if geom is not None else None
                normals_data = (
                    getattr(normals_buf, "data", None) if normals_buf is not None else None
                )
                n_norm = "?"
                norm_mag = "?"
                if normals_data is not None:
                    try:
                        import numpy as _np

                        n_norm = int(normals_data.shape[0])
                        magnitudes = _np.linalg.norm(normals_data, axis=1)
                        std_xyz = normals_data.std(axis=0)
                        mean_xyz = normals_data.mean(axis=0)
                        norm_mag = (
                            f"magmean={float(magnitudes.mean()):.3f} "
                            f"mean_xyz=({float(mean_xyz[0]):.3f},{float(mean_xyz[1]):.3f},{float(mean_xyz[2]):.3f}) "
                            f"std_xyz=({float(std_xyz[0]):.3f},{float(std_xyz[1]):.3f},{float(std_xyz[2]):.3f})"
                        )
                    except Exception as _exc:  # noqa: BLE001
                        norm_mag = f"err:{_exc}"
                pos_buf = getattr(geom, "positions", None) if geom is not None else None
                pos_data = getattr(pos_buf, "data", None) if pos_buf is not None else None
                n_vert = (
                    int(pos_data.shape[0])
                    if pos_data is not None and hasattr(pos_data, "shape")
                    else "?"
                )
                lines.append(
                    f"[light-debug]   obj={name} mat={type(mat).__name__} "
                    f"env_map={type(env_map).__name__ if env_map is not None else None} "
                    f"env_int={env_intensity:.4f} rough={roughness:.4f} metal={metallic} "
                    f"color={color} n_vert={n_vert} n_norm={n_norm} norm_mag={norm_mag}"
                )
            _sys.stderr.write("\n".join(lines) + "\n")
            _sys.stderr.flush()
        except Exception as exc:
            _sys.stderr.write(f"[light-debug] dump failed: {exc}\n")
            _sys.stderr.flush()

    def _update_shadow_extent(self, extent: float) -> None:
        """Resize the shadow camera frustum to cover *extent* world units."""
        self._scene_extent = max(extent, 10.0)
        if self._key_light is None or not hasattr(self._key_light, "shadow"):
            return
        try:
            cam = self._key_light.shadow.camera
            dim = self._scene_extent * 1.5
            if hasattr(cam, "width"):
                cam.width = dim
            if hasattr(cam, "height"):
                cam.height = dim
        except (AttributeError, RuntimeError):
            pass

    @staticmethod
    def _set_directional_light_from_angles(
        *,
        light: Any,
        azimuth_deg: float,
        elevation_deg: float,
        radius: float,
    ) -> None:
        """Place a directional light from UI azimuth/elevation controls."""
        if light is None:
            return
        az = np.deg2rad(float(azimuth_deg))
        el = np.deg2rad(float(elevation_deg))
        direction = np.array(
            [
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            direction /= norm
        pos = -direction * float(radius)
        try:
            local = getattr(light, "local", None)
            if local is not None and hasattr(local, "position"):
                local.position = tuple(float(x) for x in pos)
            if hasattr(light, "look_at"):
                light.look_at((0.0, 0.0, 0.0))
        except Exception:
            return

    def _update_headlight_pose(self) -> None:
        """Keep the headlight colocated with the camera when follow mode is on."""
        if (
            not self._headlight_enabled
            or not self._headlight_follow_camera
            or self._head_light is None
            or self._camera is None
        ):
            return
        try:
            local = self._camera.local
            eye = np.asarray(local.position, dtype=np.float64).reshape(-1)[:3]
            rot = np.asarray(local.rotation, dtype=np.float64).reshape(-1)
            if rot.size < 4:
                return

            # Skip recomputation if camera hasn't moved since last frame.
            cam_key = (eye[0], eye[1], eye[2], rot[0], rot[1], rot[2], rot[3])
            if cam_key == self._last_headlight_cam_key:
                return
            self._last_headlight_cam_key = cam_key

            fwd = self._quat_forward(rot[:4])
            norm = float(np.linalg.norm(fwd))
            if norm < 1e-9:
                return
            fwd /= norm
            lookat = eye + fwd
            hl_local = getattr(self._head_light, "local", None)
            if hl_local is not None and hasattr(hl_local, "position"):
                hl_local.position = tuple(float(x) for x in eye)
            if hasattr(self._head_light, "look_at"):
                self._head_light.look_at(tuple(float(x) for x in lookat))
        except Exception:
            return

    def _focus_canvas(self) -> None:
        """Restore keyboard focus to the canvas after controller changes."""
        widget = self._canvas_widget
        try:
            if widget is not None and hasattr(widget, "setFocus"):
                widget.setFocus()
        except Exception:
            pass
        container = self._container
        try:
            if container is not None and hasattr(container, "setFocus"):
                container.setFocus()
        except Exception:
            pass
        try:
            if container is not None and hasattr(container, "activateWindow"):
                container.activateWindow()
        except Exception:
            pass
