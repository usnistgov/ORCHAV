"""UI event router for visualizer controls.

``UIController`` is the compatibility surface that most panels call when a Qt
widget changes. It translates widget state into ``AppState`` updates, cache
invalidation scopes, service calls, and renderer refresh requests. Workflow
groups with enough local policy live in subcontrollers; this file keeps the
remaining cross-panel glue and older delegate entry points stable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import QSignalBlocker, QTimer

from shared.logging import get_logger, set_log_level

from ..beamforming.visualization import MAX_BEAM_PATTERN_WORK_ITEMS
from ..renderers.protocol import renderer_capabilities
from ..scene.geometry_helpers import normalize_node_label_mode
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache
from ..services.material_entry_editor import MaterialEntryEditService
from ..services.material_mode_commands import MaterialModeCommandService
from ..services.material_modes import MaterialModeService
from ..services.metrics_service import MetricsService
from ..services.scene_service import SceneService
from ..services.trajectory_load_service import (
    TRAJECTORY_UNAVAILABLE_MESSAGE,
    TrajectoryLoadCoordinator,
    TrajectorySnapshot,
)
from ..state import (
    DEFAULT_RF_XRAY_OPACITY,
    DEFAULT_RF_XRAY_PROPERTY,
    normalize_rf_xray_mode,
    normalize_rf_xray_opacity,
    normalize_rf_xray_property,
)
from ..utils.antenna_utils import spacing_m_to_wavelengths, spacing_wavelengths_to_m
from .coverage_controller import CoverageController
from .material_ui_controller import MaterialUIController
from .menu_controller import MenuController
from .telemetry_controller import TelemetryController

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.ui_controller")

STEP_SCRUB_COALESCE_INTERVAL_MS = 33


class UIController:
    """Coordinate panel events without owning renderer or frame-pipeline data."""

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        scene_service: SceneService,
        metrics_service: MetricsService,
        coverage_service: Any,
        material_mode_service: MaterialModeService,
        material_mode_command_service: MaterialModeCommandService,
        material_entry_edit_service: MaterialEntryEditService,
        trajectory_load_coordinator: TrajectoryLoadCoordinator,
    ) -> None:
        """Bind shared services and install workflow-sized subcontrollers."""
        self.visualizer = visualizer
        self.scene_service = scene_service
        self.metrics_service = metrics_service
        self._coverage_panel_initialized = False
        self.coverage_service = coverage_service
        self.material_mode_service = material_mode_service
        self.material_mode_command_service = material_mode_command_service
        self.material_entry_edit_service = material_entry_edit_service
        self.trajectory_load_coordinator = trajectory_load_coordinator
        trajectory_load_coordinator.progress_updated.connect(self._on_trajectory_progress)
        trajectory_load_coordinator.snapshot_updated.connect(self._on_trajectory_partial_update)
        trajectory_load_coordinator.loading_complete.connect(self._on_trajectory_loaded)
        trajectory_load_coordinator.error_occurred.connect(self._on_trajectory_error)
        trajectory_load_coordinator.cleared.connect(self._on_trajectory_cleared)

        # Subcontrollers own cohesive workflow groups; this controller keeps the
        # historic public methods that panels and the root visualizer call.
        self._coverage_ctrl = CoverageController(self)
        self._menu_ctrl = MenuController(self)
        self._material_ctrl = MaterialUIController(self)
        self._telemetry_ctrl = TelemetryController(self)
        self._camera_debug_enabled: bool = os.getenv("ORCHAV_CAMERA_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._pending_tx_marker_size: Optional[float] = None
        self._pending_rx_marker_size: Optional[float] = None
        self._tx_marker_size_timer = QTimer()
        self._tx_marker_size_timer.setSingleShot(True)
        self._tx_marker_size_timer.timeout.connect(self._apply_tx_marker_size)
        self._rx_marker_size_timer = QTimer()
        self._rx_marker_size_timer.setSingleShot(True)
        self._rx_marker_size_timer.timeout.connect(self._apply_rx_marker_size)

    def _set_viewport_hud_enabled(self, enabled: bool) -> None:
        """Apply the HUD master switch when the active renderer supports it."""
        viz = self.visualizer
        state = getattr(viz, "app_state", None)
        renderer = getattr(viz, "renderer", None)
        if state is None or renderer is None or not renderer_capabilities(renderer).viewport_hud:
            return

        enabled = bool(enabled)
        if bool(getattr(state, "viewport_hud_enabled", True)) == enabled:
            return

        viz.set_state(viewport_hud_enabled=enabled)
        renderer.refresh_viewport_hud()

        ui_manager = getattr(viz, "ui_manager", None)
        render_panel = getattr(ui_manager, "panels", {}).get("render") if ui_manager else None
        sync_controls = getattr(render_panel, "_sync_viewport_hud_controls", None)
        if callable(sync_controls):
            sync_controls()

    def toggle_viewport_hud(self) -> None:
        """Toggle the HUD master switch without changing its detail settings."""
        state = getattr(self.visualizer, "app_state", None)
        if state is None:
            return
        self._set_viewport_hud_enabled(not bool(getattr(state, "viewport_hud_enabled", True)))

    def handle_viewport_hud_enabled_toggled(self, enabled: bool) -> None:
        """Apply the persistent Context HUD master switch."""
        self._set_viewport_hud_enabled(bool(enabled))

    def _invalidate_cache(self, scope: CacheInvalidationScope, *, reason: str) -> None:
        """Invalidate visualizer cache scopes after UI intent changes."""
        invalidate_visualizer_cache(self.visualizer, scope, reason=reason)

    def _object_appearance_service(self) -> Any:
        """Return the service that owns object visibility/highlight/material state."""
        return self.visualizer.object_appearance_service

    def _camera_debug(self, message: str, **fields: Any) -> None:
        """Emit camera UI diagnostics when ORCHAV_CAMERA_DEBUG is enabled."""
        if not getattr(self, "_camera_debug_enabled", False):
            return
        if fields:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            logger.info("CameraDebugUI: %s %s", message, details)
        else:
            logger.info("CameraDebugUI: %s", message)

    def toggle_metrics_window(self) -> None:
        """Delegate metrics-window toggling to the metrics service."""
        self.metrics_service.toggle_window()

    def toggle_mpc_explorer(self) -> None:
        """Open or close the lazily constructed MPC Explorer session."""
        # Keep the sizeable Qt model/query stack out of normal visualizer
        # startup and out of every frame while the pop-out is closed.
        from ..services.mpc_explorer_service import toggle_mpc_explorer

        toggle_mpc_explorer(self.visualizer)

    def refresh_scene_panels(self) -> None:
        """Refresh scene/object panels after scene-entry mutations."""
        self.populate_controls()

    def populate_controls(self) -> None:
        """Rebuild the object panel from current scene, target, TX, and RX entries."""
        viz = self.visualizer
        if (
            hasattr(viz, "ui_manager")
            and hasattr(viz.ui_manager, "panels")
            and "objects" in viz.ui_manager.panels
        ):
            total_objects = (
                len(viz.mesh_entries)
                + len(viz.target_entries)
                + len(viz.tx_entries)
                + len(viz.rx_entries)
            )
            target_count = len(viz.target_entries)
            panel = viz.ui_manager.panels["objects"]
            panel.update_object_count(
                total_objects, target_count, len(viz.tx_entries), len(viz.rx_entries)
            )

            search_text = getattr(viz, "object_search_filter", None)
            search_text = search_text.text() if search_text else ""
            group_by = getattr(viz, "group_by_combo", None)
            group_by = group_by.currentText() if group_by else "Material"

            panel.populate_object_list(
                viz.mesh_entries,
                viz.target_entries,
                viz.tx_entries,
                viz.rx_entries,
                search_text,
                group_by,
            )

    def handle_show_tx_segments_changed(self, state: bool) -> None:
        """Toggle per-TX MPC segment visibility."""
        viz = self.visualizer
        viz.show_tx_segments = bool(state)
        logger.debug(f"Show TX segments toggled to: {viz.show_tx_segments}")
        self._invalidate_cache(
            CacheInvalidationScope.MPC_RENDER_SETTINGS, reason="show_tx_segments"
        )
        viz.schedule_update()

    def _set_mpc_visibility(self, *, reason: str, **changes: bool) -> None:
        """Update the single MPC visibility value and schedule one refresh."""
        viz = self.visualizer
        visibility = replace(viz.app_state.mpc_visibility, **changes)
        if visibility == viz.app_state.mpc_visibility:
            return
        viz.set_state(mpc_visibility=visibility)
        self._invalidate_cache(CacheInvalidationScope.MPC_RENDER_SETTINGS, reason=reason)
        viz.schedule_update()

    def handle_mpc_layer_toggled(self, state: bool) -> None:
        """Enable or disable the complete MPC presentation layer."""
        self._set_mpc_visibility(reason="mpc_layer", enabled=bool(state))

    def handle_mpc_paths_toggled(self, state: bool) -> None:
        """Toggle MPC path segments while preserving bounce-point intent."""
        self._set_mpc_visibility(reason="mpc_paths", paths=bool(state))

    def handle_mpc_bounce_points_toggled(self, state: bool) -> None:
        """Toggle physical MPC bounce points while preserving path intent."""
        self._set_mpc_visibility(reason="mpc_bounce_points", bounce_points=bool(state))

    def handle_mpc_interaction_markers_toggled(self, state: bool) -> None:
        """Toggle pygfx interaction glyphs for physical MPC bounce points."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        supported = renderer is None or renderer_capabilities(renderer).mpc_type_markers
        enabled = bool(state) and supported
        viz.set_state(show_mpc_type_markers=enabled)
        if renderer is not None and supported:
            refresh_fn = getattr(renderer, "refresh_mpc_point_markers", None)
            if callable(refresh_fn):
                refresh_fn()
                return
        if hasattr(viz, "force_update_next_frame"):
            viz.force_update_next_frame = True
        viz.schedule_update()

    def handle_topk_render_toggled(self, state: bool) -> None:
        """Toggle the render-only strongest-path cap."""
        viz = self.visualizer
        enabled = bool(state)
        if enabled == viz.app_state.topk_render_enabled:
            return
        viz.set_state(topk_render_enabled=enabled)
        self._invalidate_cache(CacheInvalidationScope.MPC_RENDER_SETTINGS, reason="topk_render")
        viz.schedule_update()

    def handle_topk_render_max_paths_changed(self, value: int) -> None:
        """Update the Top-K render cap size."""
        viz = self.visualizer
        max_paths = max(1, int(value))
        if max_paths == viz.app_state.topk_render_max_paths:
            return
        viz.set_state(topk_render_max_paths=max_paths)
        self._invalidate_cache(
            CacheInvalidationScope.MPC_RENDER_SETTINGS,
            reason="topk_render_max_paths",
        )
        viz.schedule_update()

    def handle_beamforming_toggled(self, state: bool) -> None:
        """Toggle beamforming mesh overlays."""
        viz = self.visualizer
        enabled = bool(state)
        viz.set_state(show_beamforming=enabled)
        if enabled:
            # Clamp stale/programmatic settings before the first visible build,
            # not only after the user next edits a beam control.
            self._apply_beam_preview_work_budget("elevation")
        self._refresh_beam_colorbar()
        # Toggling off must still present once to remove existing meshes.
        self._invalidate_beam_patterns(force_refresh=True)

    def _invalidate_beam_patterns(self, *, force_refresh: bool = False) -> None:
        """Invalidate beam results and redraw only when their visibility needs it."""
        viz = self.visualizer
        self._invalidate_cache(CacheInvalidationScope.MPC_RENDER_SETTINGS, reason="beamforming")
        beamforming_ui = getattr(viz, "beamforming_ui_controller", None)
        show_beamforming = bool(getattr(viz.app_state, "show_beamforming", False))
        if show_beamforming:
            begin_computation = getattr(beamforming_ui, "begin_computation", None)
            if callable(begin_computation):
                begin_computation()
        else:
            apply_selector_state = getattr(beamforming_ui, "apply_selector_state", None)
            if callable(apply_selector_state):
                apply_selector_state()
        if force_refresh or show_beamforming:
            viz.schedule_update()

    def _apply_beam_preview_work_budget(self, preferred: str) -> bool:
        """Store a bounded array/sampling configuration from the current widgets."""
        viz = self.visualizer

        def _widget_value(name: str, fallback: int) -> int:
            widget = getattr(viz, name, None)
            return int(widget.value()) if widget is not None else int(fallback)

        values = {
            "rows": _widget_value("standalone_rows", viz.app_state.standalone_antenna_rows),
            "cols": _widget_value("standalone_cols", viz.app_state.standalone_antenna_cols),
            "azimuth": _widget_value(
                "beam_azimuth_spin", viz.app_state.beamforming_azimuth_samples
            ),
            "elevation": _widget_value(
                "beam_elevation_spin", viz.app_state.beamforming_elevation_samples
            ),
        }
        minimums = {"azimuth": 12, "elevation": 9}
        reduction_order = (
            [preferred, "elevation" if preferred == "azimuth" else "azimuth"]
            if preferred in minimums
            else ["elevation", "azimuth"]
        )
        adjusted = False
        for field in reduction_order:
            work_items = values["rows"] * values["cols"] * values["azimuth"] * values["elevation"]
            if work_items <= MAX_BEAM_PATTERN_WORK_ITEMS:
                break
            other = (
                values["rows"]
                * values["cols"]
                * values["elevation" if field == "azimuth" else "azimuth"]
            )
            allowed = max(minimums[field], MAX_BEAM_PATTERN_WORK_ITEMS // max(1, other))
            bounded = min(values[field], int(allowed))
            if bounded != values[field]:
                values[field] = bounded
                adjusted = True

        widget_names = {
            "rows": "standalone_rows",
            "cols": "standalone_cols",
            "azimuth": "beam_azimuth_spin",
            "elevation": "beam_elevation_spin",
        }
        for field, widget_name in widget_names.items():
            widget = getattr(viz, widget_name, None)
            if widget is not None and int(widget.value()) != values[field]:
                with QSignalBlocker(widget):
                    widget.setValue(values[field])

        note = getattr(viz, "beam_complexity_note", None)
        if note is not None:
            if adjusted:
                note.setText(
                    "Preview adjusted to "
                    f"{values['rows']} x {values['cols']} elements and "
                    f"{values['azimuth']} x {values['elevation']} samples "
                    "to stay within the memory budget"
                )
            else:
                note.setText(
                    "Preview limits: 32 x 32 elements, 180 x 91 samples, "
                    "and 8 million combined work items"
                )

        updates = {
            "standalone_antenna_rows": values["rows"],
            "standalone_antenna_cols": values["cols"],
            "beamforming_azimuth_samples": values["azimuth"],
            "beamforming_elevation_samples": values["elevation"],
        }
        changed = any(getattr(viz.app_state, key) != value for key, value in updates.items())
        if changed:
            viz.set_state(**updates)
        return changed

    def _refresh_beam_colorbar(self) -> None:
        """Sync the Antennas panel beam colorbar from current state."""
        viz = self.visualizer
        panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("beam_pattern")
        update_beam_colorbar = getattr(panel, "update_beam_colorbar", None)
        if callable(update_beam_colorbar):
            state = viz.app_state
            update_beam_colorbar(
                show_beamforming=state.show_beamforming,
                db_scale=state.beamforming_db_scale,
                dynamic_range_db=state.beamforming_dynamic_range_db,
                colormap=state.beamforming_colormap,
            )

    def handle_beamforming_resolution_azimuth_changed(self, value: int) -> None:
        """Update the beamforming azimuth samples."""
        if not self._apply_beam_preview_work_budget("azimuth"):
            return
        self._invalidate_beam_patterns()

    def handle_beamforming_resolution_elevation_changed(self, value: int) -> None:
        """Update the beamforming elevation samples."""
        if not self._apply_beam_preview_work_budget("elevation"):
            return
        self._invalidate_beam_patterns()

    def handle_beamforming_tx_scale_changed(self, value: float) -> None:
        """Adjust the TX beamforming mesh scale."""
        viz = self.visualizer
        new_value = float(value)
        if abs(new_value - viz.app_state.beamforming_tx_scale) < 1e-6:
            return
        viz.set_state(beamforming_tx_scale=new_value)
        self._invalidate_beam_patterns()

    def handle_beamforming_rx_scale_changed(self, value: float) -> None:
        """Adjust the RX beamforming mesh scale."""
        viz = self.visualizer
        new_value = float(value)
        if abs(new_value - viz.app_state.beamforming_rx_scale) < 1e-6:
            return
        viz.set_state(beamforming_rx_scale=new_value)
        self._invalidate_beam_patterns()

    def handle_standalone_mode_changed(self, checked: bool) -> None:
        """Select frame-provided or standalone beamforming parameters."""
        if not checked:
            return
        viz = self.visualizer
        mode = "frame"
        if (
            hasattr(viz, "standalone_mode_standalone")
            and viz.standalone_mode_standalone
            and viz.standalone_mode_standalone.isChecked()
        ):
            mode = "standalone"
        else:
            for optional_mode, widget in getattr(viz, "standalone_optional_modes", {}).items():
                if widget and widget.isChecked():
                    mode = optional_mode
                    break

        if mode == "frame" and not getattr(viz, "_frame_beamforming_available", False):
            # Frame mode is only meaningful when the active frame carries array
            # metadata; fall back to standalone controls otherwise.
            standalone_button = getattr(viz, "standalone_mode_standalone", None)
            frame_button = getattr(viz, "standalone_mode_frame", None)
            if frame_button is not None:
                with QSignalBlocker(frame_button):
                    frame_button.setChecked(False)
            if standalone_button is not None:
                with QSignalBlocker(standalone_button):
                    standalone_button.setChecked(True)
            if viz.app_state.standalone_beamforming_mode != "standalone":
                viz.set_state(standalone_beamforming_mode="standalone")
                self._invalidate_beam_patterns()
            return

        if mode == viz.app_state.standalone_beamforming_mode:
            return
        viz.set_state(standalone_beamforming_mode=mode)
        self._invalidate_beam_patterns()

    def handle_standalone_antenna_changed(self, value: int) -> None:
        """Update standalone array row/column counts."""
        if self._apply_beam_preview_work_budget("array"):
            self._invalidate_beam_patterns()

    def handle_standalone_frequency_changed(self, value: float) -> None:
        """Update standalone carrier frequency while preserving spacing in wavelengths."""
        viz = self.visualizer
        freq = float(value)
        old_freq = viz.app_state.standalone_carrier_frequency_ghz
        if abs(freq - old_freq) < 1e-6:
            return
        h_widget = getattr(viz, "standalone_h_spacing", None)
        v_widget = getattr(viz, "standalone_v_spacing", None)
        h_spacing_lambda = (
            h_widget.value()
            if h_widget is not None
            else spacing_m_to_wavelengths(viz.app_state.standalone_horizontal_spacing_m, old_freq)
        )
        v_spacing_lambda = (
            v_widget.value()
            if v_widget is not None
            else spacing_m_to_wavelengths(viz.app_state.standalone_vertical_spacing_m, old_freq)
        )
        current_h = spacing_wavelengths_to_m(h_spacing_lambda, freq)
        current_v = spacing_wavelengths_to_m(v_spacing_lambda, freq)
        viz.set_state(
            standalone_carrier_frequency_ghz=freq,
            standalone_horizontal_spacing_m=current_h,
            standalone_vertical_spacing_m=current_v,
        )
        if h_widget is not None:
            with QSignalBlocker(h_widget):
                h_widget.setValue(h_spacing_lambda)
        if v_widget is not None:
            with QSignalBlocker(v_widget):
                v_widget.setValue(v_spacing_lambda)
        self._invalidate_beam_patterns()

    def handle_standalone_spacing_changed(self, value: float) -> None:
        """Store standalone antenna spacing as meters from UI wavelength values."""
        viz = self.visualizer
        h_widget = getattr(viz, "standalone_h_spacing", None)
        v_widget = getattr(viz, "standalone_v_spacing", None)
        if h_widget is None and v_widget is None:
            return
        freq = viz.app_state.standalone_carrier_frequency_ghz
        h_spacing_lambda = (
            h_widget.value()
            if h_widget is not None
            else spacing_m_to_wavelengths(viz.app_state.standalone_horizontal_spacing_m, freq)
        )
        v_spacing_lambda = (
            v_widget.value()
            if v_widget is not None
            else spacing_m_to_wavelengths(viz.app_state.standalone_vertical_spacing_m, freq)
        )
        h_spacing = spacing_wavelengths_to_m(h_spacing_lambda, freq)
        v_spacing = spacing_wavelengths_to_m(v_spacing_lambda, freq)
        if (
            abs(h_spacing - viz.app_state.standalone_horizontal_spacing_m) < 1e-9
            and abs(v_spacing - viz.app_state.standalone_vertical_spacing_m) < 1e-9
        ):
            return
        viz.set_state(
            standalone_horizontal_spacing_m=h_spacing,
            standalone_vertical_spacing_m=v_spacing,
        )
        self._invalidate_beam_patterns()

    def handle_standalone_strategy_changed(self, text: str) -> None:
        """Map steering-strategy display text to beamforming state."""
        viz = self.visualizer
        strategy_map = {
            "SVD (Current MPCs)": "svd",
            "SVD Optimal": "svd",
            "LOS Steering": "los",
            "Manual Steering": "manual",
        }
        strategy = strategy_map.get(text, "svd")
        if strategy == viz.app_state.standalone_steering_strategy:
            return
        viz.set_state(standalone_steering_strategy=strategy)
        self._invalidate_beam_patterns()

    def handle_standalone_angles_changed(self, value: float) -> None:
        """Update manual standalone steering angles in degrees."""
        viz = self.visualizer
        if hasattr(viz, "standalone_azimuth") and hasattr(viz, "standalone_elevation"):
            az = viz.standalone_azimuth.value() if viz.standalone_azimuth else 0.0
            el = viz.standalone_elevation.value() if viz.standalone_elevation else 0.0
            if (
                abs(az - viz.app_state.standalone_azimuth_deg) < 1e-6
                and abs(el - viz.app_state.standalone_elevation_deg) < 1e-6
            ):
                return
            viz.set_state(
                standalone_azimuth_deg=az,
                standalone_elevation_deg=el,
            )
            self._invalidate_beam_patterns()

    # --- Beam pattern display options ------------------------------------

    def handle_beamforming_db_scale_changed(self, state: int) -> None:
        """Toggle dB-scale beam pattern display."""
        viz = self.visualizer
        enabled = state != 0
        if enabled == viz.app_state.beamforming_db_scale:
            return
        viz.set_state(beamforming_db_scale=enabled)
        self._refresh_beam_colorbar()
        self._invalidate_beam_patterns()

    def handle_beamforming_dynamic_range_changed(self, value: float) -> None:
        """Update beam colorbar dynamic range for dB-scale display."""
        viz = self.visualizer
        if abs(value - viz.app_state.beamforming_dynamic_range_db) < 0.1:
            return
        viz.set_state(beamforming_dynamic_range_db=value)
        self._refresh_beam_colorbar()
        self._invalidate_beam_patterns()

    def handle_beamforming_colormap_changed(self, value: str) -> None:
        """Update the colormap used for beamforming overlays."""
        viz = self.visualizer
        name = (value or "").strip()
        if not name or name == viz.app_state.beamforming_colormap:
            return
        viz.set_state(beamforming_colormap=name)
        self._refresh_beam_colorbar()
        self._invalidate_beam_patterns()

    def handle_beamforming_element_pattern_changed(self, value: str) -> None:
        """Set both TX and RX element patterns from the shared selector."""
        viz = self.visualizer
        name = (value or "").strip()
        if not name or name == viz.app_state.beamforming_element_pattern:
            return
        viz.set_state(
            beamforming_element_pattern=name,
            beamforming_tx_element_pattern=name,
            beamforming_rx_element_pattern=name,
        )
        self._invalidate_beam_patterns()

    def handle_beamforming_tx_element_pattern_changed(self, value: str) -> None:
        """Override the TX element pattern."""
        viz = self.visualizer
        name = (value or "").strip()
        if not name or name == viz.app_state.beamforming_tx_element_pattern:
            return
        viz.set_state(
            beamforming_tx_element_pattern=name,
            beamforming_element_pattern=name,
        )
        self._invalidate_beam_patterns()

    def handle_beamforming_rx_element_pattern_changed(self, value: str) -> None:
        """Override the RX element pattern."""
        viz = self.visualizer
        name = (value or "").strip()
        if not name or name == viz.app_state.beamforming_rx_element_pattern:
            return
        viz.set_state(beamforming_rx_element_pattern=name)
        self._invalidate_beam_patterns()

    def handle_beamforming_tx_node_changed(self, value: str) -> None:
        """Select the TX node used for beamforming overlays."""
        viz = self.visualizer
        normalized = (value or "").strip()
        if not normalized or normalized == "N/A":
            return
        if normalized == viz.app_state.beamforming_tx_node:
            return
        logger.debug("Beamforming TX selector changed -> %s", normalized)
        viz.set_state(beamforming_tx_node=normalized)
        self._invalidate_beam_patterns()

    def handle_beamforming_rx_node_changed(self, value: str) -> None:
        """Select the RX node or RX group used for beamforming overlays."""
        viz = self.visualizer
        normalized = (value or "").strip()
        if not normalized or normalized == "N/A":
            return
        normalized_value = normalized
        if normalized_value == viz.app_state.beamforming_rx_node:
            return
        logger.debug("Beamforming RX selector changed -> %s", normalized_value)
        viz.set_state(beamforming_rx_node=normalized_value)
        self._invalidate_beam_patterns()

    def handle_mpc_order_filter_changed(self, order: int, checked: bool) -> None:
        """Update the allowed MPC reflection-order set."""
        viz = self.visualizer
        allowed_orders = set(viz.app_state.mpc_allowed_orders)
        if checked:
            allowed_orders.add(order)
        else:
            allowed_orders.discard(order)

        logger.debug(
            "MPC order filter changed: order %s, checked %s, allowed orders: %s",
            order,
            checked,
            allowed_orders,
        )

        if not allowed_orders:
            logger.warning("All reflection orders are now disabled - this will show no MPCs")

        viz.set_state(mpc_allowed_orders=frozenset(allowed_orders))
        self._invalidate_cache(CacheInvalidationScope.FILTERS, reason="mpc_order_filter")
        viz.schedule_update()

    def handle_mpc_type_filter_changed(self, type_val: int, checked: bool) -> None:
        """Update the allowed MPC interaction-type set."""
        viz = self.visualizer
        allowed_types = set(viz.app_state.mpc_allowed_types)
        if checked:
            allowed_types.add(type_val)
        else:
            allowed_types.discard(type_val)

        logger.debug(
            "MPC type filter changed: type %s, checked %s, allowed types: %s",
            type_val,
            checked,
            allowed_types,
        )

        if not allowed_types:
            logger.warning("All MPC types are now disabled - this will show no MPCs")

        viz.set_state(mpc_allowed_types=frozenset(allowed_types))
        self._invalidate_cache(CacheInvalidationScope.FILTERS, reason="mpc_type_filter")
        viz.schedule_update()

    def handle_mpc_material_filter_changed(self, material_id: str, checked: bool) -> None:
        """Update MPC material filters and mirror material panel state."""
        viz = self.visualizer
        material_choices = set(getattr(viz, "_mpc_material_filter_choices", None) or set())
        if not material_choices:
            try:
                material_choices = set(
                    viz.mpc_core._get_all_environment_materials() if viz.mpc_core else []
                )
            except (ValueError, TypeError, KeyError, AttributeError):
                material_choices = set()
        if not hasattr(viz, "mpc_allowed_materials") or viz.mpc_allowed_materials is None:
            # ``None`` means no active filter yet; initialize to the currently
            # advertised material universe before applying the user's toggle.
            viz.mpc_allowed_materials = set(material_choices)
        if checked:
            viz.mpc_allowed_materials.add(material_id)
        else:
            viz.mpc_allowed_materials.discard(material_id)
        viz.mpc_allowed_materials = MaterialUIController.normalize_mpc_material_allow_list(
            viz.mpc_allowed_materials,
            material_choices,
        )
        logger.debug(
            "Material filter changed: %s -> %s; %s selected",
            material_id,
            checked,
            (
                len(material_choices)
                if viz.mpc_allowed_materials is None
                else len(viz.mpc_allowed_materials)
            ),
        )
        viz._material_filter_dirty = True
        self._invalidate_cache(CacheInvalidationScope.FILTERS, reason="mpc_material_filter")
        viz.schedule_update()

    def handle_range_filter_changed(
        self,
        delay_min_ns: Optional[float] = None,
        delay_max_ns: Optional[float] = None,
        power_min_db: Optional[float] = None,
        power_max_db: Optional[float] = None,
        aoa_az_min_deg: Optional[float] = None,
        aoa_az_max_deg: Optional[float] = None,
        aoa_el_min_deg: Optional[float] = None,
        aoa_el_max_deg: Optional[float] = None,
        aod_az_min_deg: Optional[float] = None,
        aod_az_max_deg: Optional[float] = None,
        aod_el_min_deg: Optional[float] = None,
        aod_el_max_deg: Optional[float] = None,
    ) -> None:
        """Update MPC delay, power, AoA, and AoD filters."""
        viz = self.visualizer
        viz.set_state(
            delay_filter_min_ns=delay_min_ns,
            delay_filter_max_ns=delay_max_ns,
            power_filter_min_db=power_min_db,
            power_filter_max_db=power_max_db,
            aoa_az_filter_min_deg=aoa_az_min_deg,
            aoa_az_filter_max_deg=aoa_az_max_deg,
            aoa_el_filter_min_deg=aoa_el_min_deg,
            aoa_el_filter_max_deg=aoa_el_max_deg,
            aod_az_filter_min_deg=aod_az_min_deg,
            aod_az_filter_max_deg=aod_az_max_deg,
            aod_el_filter_min_deg=aod_el_min_deg,
            aod_el_filter_max_deg=aod_el_max_deg,
        )
        logger.debug(
            "Range filter changed: delay=[%s, %s] ns, power=[%s, %s] dB",
            delay_min_ns,
            delay_max_ns,
            power_min_db,
            power_max_db,
        )
        self._invalidate_cache(CacheInvalidationScope.FILTERS, reason="range_filter")
        viz.schedule_update()
        # Aperture guides depend on the angular filter windows.
        self._update_apertures_if_visible()

    def handle_aoa_aperture_toggled(self, state: bool) -> None:
        """Toggle angle-of-arrival aperture guides."""
        viz = self.visualizer
        viz.set_state(show_aoa_aperture=bool(state))
        logger.debug(f"AOA aperture toggled: {state}")
        self._update_apertures_if_visible(force=True)

    def handle_aod_aperture_toggled(self, state: bool) -> None:
        """Toggle angle-of-departure aperture guides."""
        viz = self.visualizer
        viz.set_state(show_aod_aperture=bool(state))
        logger.debug(f"AOD aperture toggled: {state}")
        self._update_apertures_if_visible(force=True)

    def handle_aperture_radius_changed(self, value: float) -> None:
        """Update aperture guide radius in meters."""
        viz = self.visualizer
        viz.set_state(aperture_radius_m=float(value))
        logger.debug(f"Aperture radius changed: {value} m")
        self._update_apertures_if_visible()

    def handle_global_angular_reference_toggled(self, state: bool) -> None:
        """Toggle the global angular reference guide."""
        viz = self.visualizer
        viz.set_state(show_global_angular_reference=bool(state))
        logger.debug("Global angular reference toggled: %s", state)
        self._update_apertures_if_visible(force=True)

    def handle_local_angular_reference_toggled(self, state: bool) -> None:
        """Toggle per-node local angular reference guides."""
        viz = self.visualizer
        viz.set_state(show_local_angular_reference=bool(state))
        logger.debug("Local angular reference toggled: %s", state)
        self._update_apertures_if_visible(force=True)

    def _update_apertures_if_visible(self, *, force: bool = False) -> None:
        """Refresh aperture geometry when visible, or force removal/update."""
        viz = self.visualizer
        if not hasattr(viz, "aperture_service") or viz.aperture_service is None:
            return
        state = viz.app_state
        if (
            force
            or state.show_aoa_aperture
            or state.show_aod_aperture
            or getattr(state, "show_global_angular_reference", False)
            or getattr(state, "show_local_angular_reference", False)
        ):
            viz.aperture_service.update_apertures()

    def _get_mpc_panel(self):
        """Return the MPC panel owned by the panel manager, if it is mounted."""
        ui_manager = getattr(self.visualizer, "ui_manager", None)
        return getattr(ui_manager, "panels", {}).get("mpc") if ui_manager else None

    def _refresh_mpc_aperture_preview_state(self) -> None:
        """Refresh the MPC aperture controls through the available panel reference."""
        panel = self._get_mpc_panel()
        if panel is not None and hasattr(panel, "refresh_aperture_preview_state"):
            panel.refresh_aperture_preview_state()

    def update_range_filter_bounds_from_canonical(self, canon) -> None:
        """Refresh range-filter bounds from canonical MPC statistics."""
        mpc_panel = self._get_mpc_panel()
        if mpc_panel is None:
            return

        mpc_panel.update_range_filter_bounds(
            delay_min=canon.delay_min,
            delay_max=canon.delay_max,
            loss_min=canon.loss_min,
            loss_max=canon.loss_max,
            aoa_az_min=canon.aoa_az_min,
            aoa_az_max=canon.aoa_az_max,
            aoa_el_min=canon.aoa_el_min,
            aoa_el_max=canon.aoa_el_max,
            aod_az_min=canon.aod_az_min,
            aod_az_max=canon.aod_az_max,
            aod_el_min=canon.aod_el_min,
            aod_el_max=canon.aod_el_max,
        )

    def handle_target_sync_toggled(self, state: bool) -> None:
        """Toggle target-position synchronization for MPC updates."""
        viz = self.visualizer
        viz.set_state(sync_target_position=bool(state))
        if viz.vis_initialized:
            viz.update_mpc_visualization()

    def handle_log_level_changed(self, level: str) -> None:
        """Apply log-level changes requested from the UI."""
        set_log_level(level)
        logger.info(f"Log level changed to: {level}")

    def handle_preload_mode_toggled(self, state: bool) -> None:
        """Enable or disable background raw-frame preloading."""
        viz = self.visualizer
        viz.use_preload_mode = bool(state)
        if viz.use_preload_mode:
            if hasattr(viz, "preload_status_label") and viz.preload_status_label:
                viz.preload_status_label.setText("Preload: Starting...")
            QTimer.singleShot(100, viz.start_preloading)
        else:
            if hasattr(viz, "animation_service"):
                viz.animation_service.clear_preload_data(reset_cache_size=True)
            if hasattr(viz, "preload_status_label") and viz.preload_status_label:
                viz.preload_status_label.setText("Preload: Disabled (On-demand mode)")

    def handle_restart_preload(self) -> None:
        """Restart background frame preloading from the Performance panel."""
        viz = self.visualizer
        viz.use_preload_mode = True
        anim = getattr(viz, "animation_service", None)
        if anim is not None and hasattr(anim, "reset_preloading_state"):
            anim.reset_preloading_state()
        if hasattr(viz, "preload_status_label") and viz.preload_status_label:
            viz.preload_status_label.setText("Preload: Starting...")
        started = False
        start = getattr(viz, "start_preloading", None)
        if callable(start):
            started = bool(start())
        if not started:
            self._refresh_performance_panel()

    def handle_clear_performance_caches(self) -> None:
        """Clear transient frame/view-model state from the Performance panel."""
        viz = self.visualizer
        cache_service = getattr(viz, "cache_service", None)
        clear_all = getattr(cache_service, "clear_local_frame_caches", None)
        if callable(clear_all):
            clear_all(reason="performance_panel")
        else:
            anim = getattr(viz, "animation_service", None)
            clear_preload = getattr(anim, "clear_preload_data", None)
            if callable(clear_preload):
                clear_preload(reset_cache_size=True)
        if hasattr(viz, "preload_status_label") and viz.preload_status_label:
            viz.preload_status_label.setText("Preload: Frame caches cleared")
        status = getattr(viz, "_set_status_message", None)
        if callable(status):
            status("Frame caches cleared", 3000)
        self._refresh_performance_panel()

    def handle_clear_asset_caches(self) -> None:
        """Explicitly clear reusable target, texture, mesh, UV, and scene assets."""
        viz = self.visualizer
        cache_service = getattr(viz, "cache_service", None)
        clear_assets = getattr(cache_service, "clear_static_asset_caches", None)
        if not callable(clear_assets):
            return
        clear_assets(reason="performance_panel")
        status = getattr(viz, "_set_status_message", None)
        if callable(status):
            status("Asset caches cleared; the next load may be slower", 5000)
        self._refresh_performance_panel()

    def _refresh_performance_panel(self) -> None:
        """Ask the Performance panel to redraw cache/timing metrics."""
        manager = getattr(self.visualizer, "ui_manager", None)
        panel = getattr(manager, "panels", {}).get("performance") if manager is not None else None
        refresh = getattr(panel, "refresh_metrics", None)
        if callable(refresh):
            refresh()

    def handle_object_selected(self, obj_info: Dict[str, Any]) -> None:
        """Log normalized selection events from panels, picking, or shortcuts."""
        obj_type = obj_info.get("type", "unknown")
        obj_name = obj_info.get("name") or obj_info.get("entry", {}).get("name", "Unknown")
        if obj_type == "building":
            logger.debug(
                "Building selected: %s (material=%s)",
                obj_name,
                obj_info.get("entry", {}).get("material_id"),
            )
        elif obj_type == "tx":
            logger.debug("TX selected: %s (index=%s)", obj_name, obj_info.get("index"))
        elif obj_type == "rx":
            logger.debug("RX selected: %s (index=%s)", obj_name, obj_info.get("index"))
        else:
            logger.debug("Object selected: %s (%s)", obj_name, obj_type)

    def handle_labels_toggled(self, state: bool) -> None:
        """Toggle TX/RX labels through declarative node synchronization."""
        viz = self.visualizer
        viz.set_state(show_labels=bool(state))
        logger.debug(f"Labels toggled: show_labels = {viz.app_state.show_labels}")
        if viz.vis_initialized:
            if hasattr(viz, "node_service") and viz.node_service is not None:
                viz.node_service.update_label_visibility()
            viz.schedule_update()

    def handle_node_label_mode_changed(self, _index: int = 0) -> None:
        """Change TX/RX label text mode and rebuild dependent UI mirrors."""
        viz = self.visualizer
        combo = getattr(viz, "node_label_mode_combo", None)
        raw_mode = None
        if combo is not None and hasattr(combo, "currentData"):
            raw_mode = combo.currentData()
        mode = normalize_node_label_mode(raw_mode)
        if mode == getattr(getattr(viz, "app_state", None), "node_label_mode", "role"):
            return

        viz.set_state(node_label_mode=mode)
        logger.debug("Node label mode changed: %s", mode)

        if hasattr(viz, "node_service") and viz.node_service is not None:
            viz.node_service.populate_tx_rx_selections(preserve_selection=True)
            viz.node_service.refresh_comm_node_entries()
            viz.node_service.recreate_tx_rx_labels(getattr(viz, "label_font_size", 0.3))
            viz.node_service.update_node_coloring_legend()
        if hasattr(viz, "camera_controller"):
            viz.camera_controller.update_target_focus_dropdown()

        self._invalidate_cache(CacheInvalidationScope.FILTERS, reason="node_label_mode")
        viz.force_update_next_frame = True

        if getattr(viz, "vis_initialized", False):
            viz.schedule_update()

    def handle_target_labels_toggled(self, state: bool) -> None:
        """Toggle target labels through declarative node synchronization."""
        viz = self.visualizer
        viz.set_state(show_target_labels=bool(state))
        logger.debug("Target labels toggled: show_target_labels = %s", bool(state))
        if viz.vis_initialized and hasattr(viz, "node_service"):
            viz.node_service.update_target_label_visibility()
            viz.schedule_update()

    def handle_target_renamed(self, target_idx: int, new_label: str) -> None:
        """Apply a target display-label edit across state, panels, and ViewModel."""
        viz = self.visualizer
        state = viz.app_state

        labels = list(state.target_labels)
        while len(labels) <= target_idx:
            labels.append("")
        labels[target_idx] = new_label
        viz.set_state(target_labels=tuple(labels))
        if 0 <= target_idx < len(getattr(viz, "target_entries", [])):
            viz.target_entries[target_idx]["display_name"] = new_label

        logger.info("Renamed target %d to '%s'", target_idx, new_label)

        # Rename affects labels, focus dropdowns, object lists, and cached
        # ViewModels that carry display metadata.
        if hasattr(viz, "node_service"):
            viz.node_service.recreate_target_labels(getattr(viz, "label_font_size", 0.3))
        nodes_panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("nodes")
        if nodes_panel and hasattr(nodes_panel, "populate_target_dropdown"):
            nodes_panel.populate_target_dropdown()
        if hasattr(viz, "camera_controller"):
            viz.camera_controller.update_target_focus_dropdown()
        self.populate_controls()

        self._invalidate_cache(CacheInvalidationScope.LABELS, reason="target_renamed")
        viz.force_update_next_frame = True

    def handle_label_font_size_changed(self, value: float) -> None:
        """Recreate TX/RX/target labels with a new font size."""
        viz = self.visualizer
        viz.label_font_size = value
        logger.debug(f"Label font size changed: {value}")
        if viz.vis_initialized and hasattr(viz, "node_service"):
            viz.node_service.recreate_tx_rx_labels(value)
            viz.node_service.recreate_target_labels(value)

    def handle_node_renamed(self, node_type: str, node_idx: int, new_label: str) -> None:
        """Apply a TX/RX display-label edit across state, selectors, and ViewModel."""
        viz = self.visualizer
        state = viz.app_state

        if node_type == "TX":
            labels = list(state.tx_labels)
            while len(labels) <= node_idx:
                labels.append("")
            labels[node_idx] = new_label
            viz.set_state(tx_labels=tuple(labels))
            if 0 <= node_idx < len(getattr(viz, "tx_entries", [])):
                viz.tx_entries[node_idx]["display_name"] = new_label
        else:
            labels = list(state.rx_labels)
            while len(labels) <= node_idx:
                labels.append("")
            labels[node_idx] = new_label
            viz.set_state(rx_labels=tuple(labels))
            if 0 <= node_idx < len(getattr(viz, "rx_entries", [])):
                viz.rx_entries[node_idx]["display_name"] = new_label

        logger.info("Renamed %s%d to '%s'", node_type, node_idx + 1, new_label)

        # Rename affects node dropdowns, labels, focus targets, legends, and
        # cached ViewModels that carry display metadata.
        if hasattr(viz, "node_service"):
            viz.node_service.populate_tx_rx_selections(preserve_selection=True)
            viz.node_service.refresh_comm_node_entries()
            viz.node_service.recreate_tx_rx_labels(getattr(viz, "label_font_size", 0.3))
            viz.node_service.update_node_coloring_legend()
        if hasattr(viz, "camera_controller"):
            viz.camera_controller.update_target_focus_dropdown()
        self.populate_controls()

        self._invalidate_cache(CacheInvalidationScope.LABELS, reason="node_renamed")
        viz.force_update_next_frame = True

    def handle_tx_orientation_toggled(self, state: bool) -> None:
        """Toggle TX orientation frames for the current frame."""
        viz = self.visualizer
        viz.show_tx_orientation = bool(state)
        logger.debug(f"TX orientation toggled to: {viz.show_tx_orientation}")
        if viz.vis_initialized:
            self._ensure_orientation_frames()

    def handle_rx_orientation_toggled(self, state: bool) -> None:
        """Toggle RX orientation frames for the current frame."""
        viz = self.visualizer
        viz.show_rx_orientation = bool(state)
        logger.debug(f"RX orientation toggled to: {viz.show_rx_orientation}")
        if viz.vis_initialized:
            self._ensure_orientation_frames()

    def handle_target_orientation_toggled(self, state: bool) -> None:
        """Toggle target orientation frames for the current frame."""
        viz = self.visualizer
        viz.show_target_orientation = bool(state)
        logger.debug("Target orientation toggled to: %s", viz.show_target_orientation)
        if viz.vis_initialized:
            self._ensure_orientation_frames()

    def handle_orientation_scale_changed(self, value: float) -> None:
        """Apply orientation-frame scale changes immediately."""
        viz = self.visualizer
        viz.orientation_scale = float(value)
        logger.debug("Orientation scale changed to: %s", value)
        if viz.vis_initialized:
            self._ensure_orientation_frames()

    def handle_live_preview_toggled(self, state: int) -> None:
        """Enable or disable pygfx interactive TX/RX preview."""
        service = getattr(self.visualizer, "live_preview_service", None)
        set_enabled = getattr(service, "set_enabled", None)
        if callable(set_enabled):
            set_enabled(bool(state))

    def handle_live_preview_recompute(self) -> None:
        """Run the live preview solver for current edited TX/RX positions."""
        service = getattr(self.visualizer, "live_preview_service", None)
        recompute = getattr(service, "recompute_now", None)
        if callable(recompute):
            recompute()

    def handle_live_preview_reset_selected(self) -> None:
        """Reset the currently selected interactive edit."""
        service = getattr(self.visualizer, "live_preview_service", None)
        reset = getattr(service, "reset_selected_edit", None)
        if callable(reset):
            reset()

    def handle_live_preview_reset_all(self) -> None:
        """Reset all transient interactive edits."""
        service = getattr(self.visualizer, "live_preview_service", None)
        reset = getattr(service, "reset_all_edits", None)
        if callable(reset):
            reset()

    def _ensure_orientation_frames(self) -> None:
        """Ensure orientation frames; NodeService owns batching and presentation."""
        viz = self.visualizer
        step = getattr(viz, "animation_step", 0)
        step_data = viz.cache_service.get_frame(step)
        missing = step_data is None
        if missing:
            logger.debug("Orientation frames requested but step %s missing from frame cache", step)
        else:
            tx = len(step_data.get("tx_orientations", []))
            rx = len(step_data.get("rx_orientations", []))
            tgt = len(step_data.get("targets_metadata", []))
            logger.debug(
                "Orientation helper: step=%s tx=%s rx=%s targets=%s",
                step,
                tx,
                rx,
                tgt,
            )
        if missing:
            # The frame pipeline creates every orientation domain after loading
            # the missing frame; do not immediately publish the same state twice.
            viz.force_update_next_frame = True
            viz._process_frame_step(step)
            return
        viz.node_service.create_orientation_frames(step)

    def handle_coverage_toggled(self, state: bool) -> None:
        """Forward coverage visibility changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_toggled(state)

    def reset_coverage_runtime_state(self) -> None:
        """Restore controller-owned coverage state during scenario teardown."""
        return self._coverage_ctrl.reset_runtime_state()

    @property
    def coverage_controller(self) -> CoverageController:
        """Return the coverage controller for read-only workflow coordination."""
        return self._coverage_ctrl

    def handle_coverage_opacity_changed(self, value: int) -> None:
        """Forward coverage opacity changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_opacity_changed(value)

    def handle_coverage_height_changed(self, index: int) -> None:
        """Forward coverage height changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_height_changed(index)

    def handle_coverage_cache_all_clicked(self) -> None:
        """Forward coverage pre-cache requests to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_cache_all_clicked()

    def handle_coverage_interpolation_changed(self, method: str) -> None:
        """Forward coverage interpolation changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_interpolation_changed(method)

    def handle_coverage_metric_changed(self, metric_name: str) -> None:
        """Forward coverage metric changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_metric_changed(metric_name)

    def handle_coverage_threshold_changed(self, enabled: bool, value: float) -> None:
        """Forward coverage threshold changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_threshold_changed(enabled, value)

    def handle_coverage_threshold_mask_changed(self, enabled: bool) -> None:
        """Forward coverage threshold-mask changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_threshold_mask_changed(enabled)

    def handle_coverage_isolines_changed(self, enabled: bool, count: int) -> None:
        """Forward coverage isoline changes to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_isolines_changed(enabled, count)

    def handle_coverage_height_animation_play(self) -> None:
        """Forward coverage height-animation play/pause to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_height_animation_play()

    def handle_coverage_height_animation_stop(self) -> None:
        """Forward coverage height-animation stop to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_height_animation_stop()

    def handle_coverage_height_animation_speed_changed(self, speed: int) -> None:
        """Forward coverage height-animation speed to the coverage controller."""
        return self._coverage_ctrl.handle_coverage_height_animation_speed_changed(speed)

    def handle_tx_selection_changed(self, text: str) -> None:
        """Update selected TX and the default beamforming TX node."""
        viz = self.visualizer
        new_selected_tx = "all"
        if text == "All TX":
            new_selected_tx = "all"
        elif viz.tx_dropdown:
            current_index = viz.tx_dropdown.currentIndex()
            if current_index > 0:
                stored_tx_idx = viz.tx_dropdown.itemData(current_index)
                if stored_tx_idx is not None:
                    new_selected_tx = stored_tx_idx
                else:
                    try:
                        tx_idx = int(text.replace("TX", ""))
                        new_selected_tx = tx_idx - 1
                    except ValueError:
                        new_selected_tx = "all"

        state_updates = {"selected_tx": new_selected_tx}
        if new_selected_tx == "all":
            state_updates["beamforming_tx_node"] = "auto"
        else:
            try:
                state_updates["beamforming_tx_node"] = f"tx_{int(new_selected_tx) + 1}"
            except (TypeError, ValueError):
                state_updates["beamforming_tx_node"] = "auto"
        viz.beamforming_ui_controller.clear_result_metadata()
        viz.set_state(**state_updates)
        begin_computation = getattr(viz.beamforming_ui_controller, "begin_computation", None)
        if callable(begin_computation):
            begin_computation()
        else:
            viz.beamforming_ui_controller.apply_selector_state()
        self._invalidate_cache(CacheInvalidationScope.FILTERS, reason="tx_selection")
        if viz.vis_initialized:
            viz.node_service.update_tx_rx_visibility()
        viz.schedule_update()
        # TX selection affects AOD aperture guides.
        self._refresh_mpc_aperture_preview_state()
        self._update_apertures_if_visible()

    def handle_rx_selection_changed(self, text: str) -> None:
        """Update selected RX and the default beamforming RX node."""
        viz = self.visualizer
        new_selected_rx = "all"
        if text not in {"All RX", None, ""}:
            if viz.rx_dropdown:
                current_index = viz.rx_dropdown.currentIndex()
                if current_index > 0:
                    stored_rx_idx = viz.rx_dropdown.itemData(current_index)
                    if stored_rx_idx is not None:
                        new_selected_rx = stored_rx_idx
                    else:
                        try:
                            rx_idx = int(text.replace("RX", ""))
                            new_selected_rx = rx_idx - 1
                        except ValueError:
                            new_selected_rx = "all"

        state_updates = {"selected_rx": new_selected_rx}
        if new_selected_rx == "all":
            state_updates["beamforming_rx_node"] = "auto"
        else:
            try:
                state_updates["beamforming_rx_node"] = f"rx_{int(new_selected_rx) + 1}"
            except (TypeError, ValueError):
                state_updates["beamforming_rx_node"] = "auto"
        viz.beamforming_ui_controller.clear_result_metadata()
        viz.set_state(**state_updates)
        begin_computation = getattr(viz.beamforming_ui_controller, "begin_computation", None)
        if callable(begin_computation):
            begin_computation()
        else:
            viz.beamforming_ui_controller.apply_selector_state()
        self._invalidate_cache(CacheInvalidationScope.FILTERS, reason="rx_selection")
        if viz.vis_initialized:
            viz.node_service.update_tx_rx_visibility()
        viz.schedule_update()
        # RX selection affects AOA aperture guides.
        self._refresh_mpc_aperture_preview_state()
        self._update_apertures_if_visible()

    def handle_step_changed(self, value: int) -> None:
        """Queue the latest slider position while limiting frame work to about 30 Hz."""
        viz = self.visualizer
        display_index = int(value)
        viz._pending_slider_scrub_index = display_index

        # The label is lightweight preview feedback. The frame input and
        # AppState remain tied to the last frame accepted by ``update_frame``.
        step_label = getattr(viz, "step_label", None)
        if step_label is not None:
            step_label.setText(str(display_index + 1))

        timer = viz.slider_scrub_timer
        if not timer.isActive():
            timer.start(STEP_SCRUB_COALESCE_INTERVAL_MS)
        logger.debug("Queued slider scrub to frame %d", display_index)

    def handle_step_slider_released(self) -> None:
        """Commit the latest pending slider position immediately on mouse release."""
        self.visualizer.flush_pending_slider_scrub()

    def handle_frame_input_changed(self, value: int) -> None:
        """Jump to a 1-based frame number from the frame input widget."""
        viz = self.visualizer
        viz.cancel_pending_slider_scrub()
        frame_index = value - 1
        if frame_index < 0 or frame_index >= viz.total_animation_steps:
            logger.warning(f"Frame input out of range: {value} (index {frame_index})")
            return
        step = viz.resolve_animation_step(frame_index)

        logger.debug("Jumping to frame %d (step %d)", value, step)
        if hasattr(viz, "ui_manager") and "animation" in viz.ui_manager.panels:
            panel = viz.ui_manager.panels["animation"]
            if panel.is_in_online_mode():
                panel.request_frame_if_needed(step)
        viz.update_frame(step)

    def handle_playback_timing_changed(self, _value: Any = None) -> None:
        """Apply a playback policy or fixed-FPS change to active animation."""
        viz = self.visualizer
        controller = getattr(viz, "animation_controller", None)
        if getattr(viz, "animation_running", False) and controller is not None:
            controller.reset_playback_cadence()
            controller.restart_playback_timer()
            self.refresh_status_telemetry()
        logger.debug("Playback timing selection changed")

    def handle_building_labels_toggled(self, state: bool) -> None:
        """Toggle building labels through the object appearance service."""
        viz = self.visualizer
        appearance = self._object_appearance_service()
        viz.show_building_labels = bool(state)

        def _apply() -> None:
            """Publish every scene-label visibility in one renderer batch."""
            for entry in viz.mesh_entries:
                appearance.set_building_label_visibility(
                    entry,
                    viz.show_building_labels,
                    update_renderer=False,
                )

        renderer = getattr(viz, "renderer", None)
        batch_updates = getattr(renderer, "batch_updates", None)
        if callable(batch_updates):
            with batch_updates():
                _apply()
        else:
            _apply()
        if viz.vis_initialized and viz.vis is not None:
            viz.schedule_update()

    def handle_individual_building_label_toggled(self, entry: Any, state: bool) -> None:
        """Toggle one building label."""
        self._object_appearance_service().set_building_label_visibility(entry, bool(state))

    def handle_individual_building_highlight_toggled(self, entry: Any, state: bool) -> None:
        """Toggle one building highlight."""
        self._object_appearance_service().set_object_highlight(entry, bool(state))

    def handle_individual_target_highlight_toggled(self, entry: Any, state: bool) -> None:
        """Toggle one target highlight."""
        self._object_appearance_service().set_object_highlight(entry, bool(state))

    def handle_scene_toggled(self, state: bool) -> None:
        """Toggle visibility for every scene/building entry."""
        viz = self.visualizer
        logger.debug("Scene visibility toggled: %s", state)
        if not viz.mesh_entries:
            return
        appearance = self._object_appearance_service()

        def _apply():
            """Apply scene visibility changes without per-object redraws."""
            for mesh_entry in viz.mesh_entries:
                appearance.set_object_visibility(mesh_entry, bool(state), update_renderer=False)

        if viz.renderer is not None and hasattr(viz.renderer, "batch_updates"):
            with viz.renderer.batch_updates():
                _apply()
        else:
            _apply()

        if viz.renderer is not None:
            if hasattr(viz.renderer, "request_redraw"):
                viz.renderer.request_redraw()
            else:
                viz.renderer.update_renderer()

    def handle_target_toggled(self, state: bool) -> None:
        """Toggle visibility for every target entry."""
        viz = self.visualizer
        logger.debug("Target visibility toggled: %s", state)
        if not viz.target_entries:
            return
        appearance = self._object_appearance_service()

        def _apply():
            """Apply target visibility changes without per-object redraws."""
            for target_entry in viz.target_entries:
                appearance.set_object_visibility(target_entry, bool(state), update_renderer=False)

        if viz.renderer is not None and hasattr(viz.renderer, "batch_updates"):
            with viz.renderer.batch_updates():
                _apply()
        else:
            _apply()

        if viz.renderer is not None:
            if hasattr(viz.renderer, "request_redraw"):
                viz.renderer.request_redraw()
            else:
                viz.renderer.update_renderer()

    def handle_target_focus_changed(self, text: str) -> None:
        """Route focus-target changes to follow or POV camera updates."""
        logger.debug("Target focus selection changed to: %s", text)
        viz = self.visualizer
        camera = getattr(viz, "camera_controller", None)
        if camera:
            camera.remember_focus_selection()
            mode = getattr(getattr(viz, "app_state", None), "camera_mode", None)
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

        # Track controls are meaningful only when a selected entity drives view.
        track_group = getattr(viz, "track_group", None)
        if track_group:
            track_group.setVisible(mode in ("follow", "pov"))

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
                    track_group.setVisible(previous_mode in ("follow", "pov"))
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

    def handle_selection_clear(self) -> None:
        """Clear highlighted selections and selection info."""
        self.visualizer.selection_manager.clear_selections()

    def handle_object_selection_changed(self, text: str) -> None:
        """Pick a building from the object dropdown text."""
        logger.debug("Building dropdown selection changed to: %s", text)
        if text == "Select a building...":
            return
        self.visualizer.selection_manager.pick_object_by_name(text)

    def handle_view_mode_changed(self, mode: str) -> None:
        """Forward object-list display mode changes."""
        self.update_object_list_display(mode)

    def handle_clear_selections_clicked(self) -> None:
        """Clear selections triggered from the UI."""
        self.visualizer.selection_manager.clear_selections()

    def handle_label_offset_changed(self) -> None:
        """Read label-offset spinboxes and apply offsets immediately."""
        viz = self.visualizer
        if hasattr(viz, "x_offset_spinbox") and viz.x_offset_spinbox:
            viz.label_offset_x = float(viz.x_offset_spinbox.value())
        if hasattr(viz, "y_offset_spinbox") and viz.y_offset_spinbox:
            viz.label_offset_y = float(viz.y_offset_spinbox.value())
        if hasattr(viz, "z_offset_spinbox") and viz.z_offset_spinbox:
            viz.label_offset_z = float(viz.z_offset_spinbox.value())
        viz.node_service.apply_label_offsets()

    def handle_tx_marker_size_changed(self, value: float) -> None:
        """Debounce TX marker-size updates from the spinbox."""
        self._pending_tx_marker_size = float(value)
        self._tx_marker_size_timer.start(50)

    def handle_rx_marker_size_changed(self, value: float) -> None:
        """Debounce RX marker-size updates from the spinbox."""
        self._pending_rx_marker_size = float(value)
        self._rx_marker_size_timer.start(50)

    def _apply_tx_marker_size(self) -> None:
        """Apply a pending TX marker size update after debounce."""
        if self._pending_tx_marker_size is None:
            return
        viz = self.visualizer
        viz.tx_marker_size = float(self._pending_tx_marker_size)
        self._pending_tx_marker_size = None
        viz.node_service.update_tx_marker_sizes()

    def _apply_rx_marker_size(self) -> None:
        """Apply a pending RX marker size update after debounce."""
        if self._pending_rx_marker_size is None:
            return
        viz = self.visualizer
        viz.rx_marker_size = float(self._pending_rx_marker_size)
        self._pending_rx_marker_size = None
        viz.node_service.update_rx_marker_sizes()

    def handle_node_coloring_changed(self) -> None:
        """Switch node coloring mode and rerender active trajectories."""
        viz = self.visualizer
        if viz.per_node_type_rb is None or viz.individual_nodes_rb is None:
            return
        if viz.per_node_type_rb.isChecked():
            viz.node_coloring_mode = "per_type"
        else:
            viz.node_coloring_mode = "individual"
        viz.node_service.apply_node_coloring()
        viz.node_service.update_node_coloring_legend()

        # Trajectory node-color mode uses the same TX/RX color policy.
        trajectory_data = self.trajectory_load_coordinator.snapshot
        if trajectory_data is not None:
            state = viz.app_state
            if state.show_tx_trajectory:
                self._render_trajectory_if_enabled("tx", trajectory_data)
            if state.show_rx_trajectory:
                self._render_trajectory_if_enabled("rx", trajectory_data)
            if getattr(state, "show_target_trajectory", False):
                self._render_trajectory_if_enabled("target", trajectory_data)

    def handle_color_mode_changed(self) -> None:
        """Select MPC color mode and synchronize legends/colorbars."""
        viz = self.visualizer

        if viz.reflection_order_rb is None or viz.mpc_type_rb is None:
            return

        new_color_mode = "reflection_order"  # default
        show_colorbar = False
        if viz.reflection_order_rb.isChecked():
            new_color_mode = "reflection_order"
            logger.debug("Color mode changed to: Reflection Order")
        elif viz.mpc_type_rb.isChecked():
            new_color_mode = "mpc_type"
            logger.debug("Color mode changed to: MPC Type")
        elif viz.delay_rb.isChecked():
            new_color_mode = "delay"
            show_colorbar = True
            logger.debug("Color mode changed to: Delay")
        elif viz.path_loss_rb.isChecked():
            new_color_mode = "path_loss"
            show_colorbar = True
            logger.debug("Color mode changed to: Path Loss")
        elif viz.material_rb.isChecked():
            new_color_mode = "material"
            logger.debug("Color mode changed to: Material")
        elif viz.reconstruction_type_rb and viz.reconstruction_type_rb.isChecked():
            new_color_mode = "reconstruction_type"
            logger.debug("Color mode changed to: Reconstruction Type")

        if viz.colorbar_widget:
            if show_colorbar:
                viz.colorbar_widget.show()
            else:
                viz.colorbar_widget.hide()

        if hasattr(viz, "ui_manager") and viz.ui_manager:
            mpc_panel = viz.ui_manager.panels.get("mpc")
            if mpc_panel:
                mpc_panel.update_color_legend(new_color_mode)
                distinct_cb = mpc_panel.widgets.get("distinct_material_colors_cb")
                if distinct_cb:
                    distinct_cb.setVisible(new_color_mode == "material")

        viz.set_state(color_mode=new_color_mode)
        self._invalidate_cache(CacheInvalidationScope.MPC_RENDER_SETTINGS, reason="color_mode")
        if viz.vis_initialized:
            viz._process_frame_step(viz.animation_step)
        else:
            viz.schedule_update()

    def handle_distinct_material_colors_toggled(self, checked: bool) -> None:
        """Toggle distinct MPC material colors.

        Only MPC point/line colors change; building meshes keep their original
        ITU/PBR colors, which naturally creates visual separation.
        """
        viz = self.visualizer
        viz.set_state(use_distinct_material_colors=bool(checked))

        if hasattr(viz, "ui_manager") and viz.ui_manager:
            mpc_panel = viz.ui_manager.panels.get("mpc")
            if mpc_panel:
                mpc_panel.update_color_legend(viz.app_state.color_mode)

        self._invalidate_cache(
            CacheInvalidationScope.MPC_RENDER_SETTINGS,
            reason="distinct_material_colors",
        )
        if viz.vis_initialized:
            viz._process_frame_step(viz.animation_step)
        else:
            viz.schedule_update()

    def _renderer_supports_rf_xray(self) -> bool:
        """Return whether the active renderer can present RF X-Ray overlays."""
        return renderer_capabilities(getattr(self.visualizer, "renderer", None)).rf_xray_overlay

    def _sync_rf_xray_panel(self, status: str | None = None, *, active: bool = False) -> None:
        """Refresh Materials-panel RF X-Ray controls if the panel exists."""
        panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get("materials")
        if panel is None:
            return
        sync = getattr(panel, "_sync_rf_xray_controls", None)
        if callable(sync):
            sync()
        if status is not None and hasattr(panel, "set_rf_xray_status"):
            panel.set_rf_xray_status(status, active=active)

    def _schedule_rf_xray_refresh(self) -> None:
        """Schedule a renderer pass for RF X-Ray state-only changes."""
        self.visualizer.schedule_update()

    def handle_rf_xray_toggled(self, checked: bool) -> None:
        """Toggle the RF X-Ray overlay when the renderer supports it."""
        viz = self.visualizer
        enabled = bool(checked)
        if enabled and not self._renderer_supports_rf_xray():
            viz.set_state(show_rf_xray=False)
            self._sync_rf_xray_panel("RF X-Ray overlay is pygfx-only.", active=False)
            return
        if enabled == bool(getattr(viz.app_state, "show_rf_xray", False)):
            return
        viz.set_state(show_rf_xray=enabled)
        self._sync_rf_xray_panel(active=enabled)
        self._schedule_rf_xray_refresh()

    def handle_rf_xray_mode_changed(self, mode: str) -> None:
        """Change the RF X-Ray analysis mode."""
        viz = self.visualizer
        normalized = normalize_rf_xray_mode(mode)
        if normalized == getattr(viz.app_state, "rf_xray_mode", "material_map"):
            return
        viz.set_state(rf_xray_mode=normalized)
        self._sync_rf_xray_panel(active=bool(getattr(viz.app_state, "show_rf_xray", False)))
        self._schedule_rf_xray_refresh()

    def handle_rf_xray_property_changed(self, prop: str) -> None:
        """Change the RF X-Ray material property."""
        viz = self.visualizer
        normalized = normalize_rf_xray_property(prop or DEFAULT_RF_XRAY_PROPERTY)
        if normalized == getattr(viz.app_state, "rf_xray_property", DEFAULT_RF_XRAY_PROPERTY):
            return
        viz.set_state(rf_xray_property=normalized)
        self._sync_rf_xray_panel(active=bool(getattr(viz.app_state, "show_rf_xray", False)))
        self._schedule_rf_xray_refresh()

    def handle_rf_xray_opacity_changed(self, value: int) -> None:
        """Change RF X-Ray material overlay opacity from a percent slider."""
        viz = self.visualizer
        opacity = normalize_rf_xray_opacity(float(value) / 100.0)
        if opacity == normalize_rf_xray_opacity(
            getattr(viz.app_state, "rf_xray_opacity", DEFAULT_RF_XRAY_OPACITY)
        ):
            return
        viz.set_state(rf_xray_opacity=opacity)
        self._sync_rf_xray_panel(active=bool(getattr(viz.app_state, "show_rf_xray", False)))
        self._schedule_rf_xray_refresh()

    def handle_rf_xray_top_paths_toggled(self, checked: bool) -> None:
        """Toggle RF X-Ray strongest-path highlighting."""
        viz = self.visualizer
        enabled = bool(checked)
        if enabled == bool(getattr(viz.app_state, "rf_xray_show_top_paths", False)):
            return
        viz.set_state(rf_xray_show_top_paths=enabled)
        self._sync_rf_xray_panel(active=bool(getattr(viz.app_state, "show_rf_xray", False)))
        self._schedule_rf_xray_refresh()

    def handle_rf_xray_max_paths_changed(self, value: int) -> None:
        """Change the RF X-Ray strongest-path cap."""
        viz = self.visualizer
        max_paths = max(1, int(value))
        if max_paths == int(getattr(viz.app_state, "rf_xray_max_top_paths", 12)):
            return
        viz.set_state(rf_xray_max_top_paths=max_paths)
        self._sync_rf_xray_panel(active=bool(getattr(viz.app_state, "show_rf_xray", False)))
        self._schedule_rf_xray_refresh()

    def setup_menus(self, parent: Any, metrics_available: bool) -> None:
        """Forward menu-bar construction to MenuController."""
        return self._menu_ctrl.setup_menus(parent, metrics_available)

    def update_recent_menu(self) -> None:
        """Rebuild the recent files submenu."""
        return self._menu_ctrl.update_recent_menu()

    def update_recent_sessions_menu(self) -> None:
        """Rebuild the recent sessions submenu."""
        return self._menu_ctrl.update_recent_sessions_menu()

    def register_scenario_loader(self, loader: Any) -> None:
        """Inject the scenario loader service once it is available."""
        return self._menu_ctrl.register_scenario_loader(loader)

    def open_scenario_dialog(self) -> None:
        """Forward scenario-dialog handling to the menu controller."""
        return self._menu_ctrl.open_scenario_dialog()

    def add_recent_file(self, file_path: str) -> None:
        """Add a file path to the recent list and persist the change."""
        return self._menu_ctrl.add_recent_file(file_path)

    def open_recent_file(self, file_path: str) -> None:
        """Open a recent scenario or directly selected XML scene."""
        return self._menu_ctrl.open_recent_file(file_path)

    def clear_recent_files(self) -> None:
        """Clear the recent files list."""
        return self._menu_ctrl.clear_recent_files()

    def refresh_recent_files(self) -> None:
        """Refresh the recent files list and remove invalid entries."""
        return self._menu_ctrl.refresh_recent_files()

    def handle_material_color_changed(self, entry: Dict[str, Any]) -> None:
        """Forward material color edits to the material UI controller."""
        return self._material_ctrl.handle_material_color_changed(entry)

    def handle_material_id_changed(self, entry: Dict[str, Any], new_id: str) -> None:
        """Forward material-ID edits to the material UI controller."""
        return self._material_ctrl.handle_material_id_changed(entry, new_id)

    def update_colorbar(self, title: str, value_range: list[float]) -> None:
        """Forward MPC colorbar updates to the material UI controller."""
        return self._material_ctrl.update_colorbar(title, value_range)

    def update_object_list_display(self, mode: str) -> None:
        """Forward object-list display mode changes to the material UI controller."""
        return self._material_ctrl.update_object_list_display(mode)

    def populate_material_filters(self) -> None:
        """Forward MPC material filter population to the material UI controller."""
        return self._material_ctrl.populate_material_filters()

    def apply_material_modes(self, material_key: str | None = None) -> None:
        """Forward material-mode application to the material UI controller."""
        return self._material_ctrl.apply_material_modes(material_key)

    def update_frame_context(
        self, step: int, raw_frame: Optional[dict] = None, view_model: Optional[Any] = None
    ) -> None:
        """Refresh UI elements that present frame metadata."""
        return self._telemetry_ctrl.update_frame_context(step, raw_frame, view_model)

    def refresh_status_telemetry(self) -> None:
        """Update condensed performance information in the status bar."""
        return self._telemetry_ctrl.refresh_status_telemetry()

    def update_performance_display(self) -> None:
        """Recompute rolling performance stats and refresh the telemetry label."""
        return self._telemetry_ctrl.update_performance_display()

    def update_file_source_summary(self) -> None:
        """Display concise information about the active offline frame source."""
        return self._telemetry_ctrl.update_file_source_summary()

    def handle_frame_timing_update(self, step: int, elapsed_sec: float) -> None:
        """Update rolling averages and frame tooltips when pipeline timings arrive."""
        return self._telemetry_ctrl.handle_frame_timing_update(step, elapsed_sec)

    # 3D Trajectory visualization

    def handle_tx_trajectory_toggled(self, state: int) -> None:
        """Toggle TX trajectory rendering or start trajectory loading."""
        enabled = bool(state)
        viz = self.visualizer
        viz.set_state(show_tx_trajectory=enabled)
        if enabled:
            self._ensure_trajectory_loaded("tx")
        else:
            self._remove_trajectory_geometry("tx")
        self._hide_colorbar_if_all_off()

    def handle_rx_trajectory_toggled(self, state: int) -> None:
        """Toggle RX trajectory rendering or start trajectory loading."""
        enabled = bool(state)
        viz = self.visualizer
        viz.set_state(show_rx_trajectory=enabled)
        if enabled:
            self._ensure_trajectory_loaded("rx")
        else:
            self._remove_trajectory_geometry("rx")
        self._hide_colorbar_if_all_off()

    def handle_target_trajectory_toggled(self, state: int) -> None:
        """Toggle target trajectory rendering or start trajectory loading."""
        enabled = bool(state)
        viz = self.visualizer
        viz.set_state(show_target_trajectory=enabled)
        if enabled:
            self._ensure_trajectory_loaded("target")
        else:
            self._remove_trajectory_geometry("target")
        self._hide_colorbar_if_all_off()

    def handle_trajectory_line_width_changed(self, value: float) -> None:
        """Apply the authoritative Nodes-panel trajectory line width."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).trajectories:
            return
        renderer.set_trajectory_line_width(value)

        # Some renderers apply size changes only during trajectory upload.
        trajectory_data = self.trajectory_load_coordinator.snapshot
        if trajectory_data is not None:
            state = viz.app_state
            if state.show_tx_trajectory:
                self._render_trajectory_if_enabled("tx", trajectory_data)
            if state.show_rx_trajectory:
                self._render_trajectory_if_enabled("rx", trajectory_data)
            if getattr(state, "show_target_trajectory", False):
                self._render_trajectory_if_enabled("target", trajectory_data)

    def handle_trajectory_point_size_changed(self, value: float) -> None:
        """Apply trajectory point size and rerender active trajectories."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if renderer is None or not renderer_capabilities(renderer).trajectories:
            return
        renderer.set_trajectory_point_size(value)
        trajectory_data = self.trajectory_load_coordinator.snapshot
        if trajectory_data is not None:
            state = viz.app_state
            if state.show_tx_trajectory:
                self._render_trajectory_if_enabled("tx", trajectory_data)
            if state.show_rx_trajectory:
                self._render_trajectory_if_enabled("rx", trajectory_data)
            if getattr(state, "show_target_trajectory", False):
                self._render_trajectory_if_enabled("target", trajectory_data)

    def handle_trajectory_color_mode_changed(self) -> None:
        """Select trajectory color mode and refresh active trajectory geometry."""
        viz = self.visualizer
        new_mode = "node_color"
        for mode_id in ("node_color", "speed", "altitude", "time", "angular_speed"):
            rb = getattr(viz, f"trajectory_color_{mode_id}_rb", None)
            if rb and rb.isChecked():
                new_mode = mode_id
                break

        if new_mode == viz.app_state.trajectory_color_mode:
            return

        viz.set_state(trajectory_color_mode=new_mode)
        logger.debug("Trajectory color mode changed to: %s", new_mode)

        trajectory_data = self.trajectory_load_coordinator.snapshot
        if trajectory_data is None:
            self._update_trajectory_colorbar(new_mode, None)
            return
        state = viz.app_state
        if state.show_tx_trajectory:
            self._render_trajectory_if_enabled("tx", trajectory_data)
        if state.show_rx_trajectory:
            self._render_trajectory_if_enabled("rx", trajectory_data)
        if state.show_target_trajectory:
            self._render_trajectory_if_enabled("target", trajectory_data)
        self._update_trajectory_colorbar(new_mode, trajectory_data)

    def _remove_trajectory_geometry(self, kind: str) -> None:
        """Remove one trajectory kind from the renderer."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).trajectories:
            renderer.remove_trajectory(kind)

    def _hide_colorbar_if_all_off(self) -> None:
        """Hide the trajectory colorbar and markers when no trajectories are active."""
        viz = self.visualizer
        state = viz.app_state
        any_on = (
            state.show_tx_trajectory
            or state.show_rx_trajectory
            or getattr(state, "show_target_trajectory", False)
        )
        if not any_on:
            nodes_panel = None
            if hasattr(viz, "ui_manager") and viz.ui_manager:
                nodes_panel = viz.ui_manager.panels.get("nodes")
            if nodes_panel and hasattr(nodes_panel, "update_trajectory_colorbar"):
                nodes_panel.update_trajectory_colorbar("node_color")

    def _ensure_trajectory_loaded(self, kind: str) -> None:
        """Render trajectory data if cached, otherwise start the loader."""
        trajectory_data = self.trajectory_load_coordinator.snapshot
        if trajectory_data is not None:
            self._render_trajectory_if_enabled(kind, trajectory_data)
            return
        self._request_trajectory_load()

    def _request_trajectory_load(self) -> None:
        """Ask the shared coordinator to load the active file-backed source."""
        frame_source = getattr(self.visualizer, "frame_source", None)
        started = self.trajectory_load_coordinator.load(frame_source)
        if started or self.trajectory_load_coordinator.is_loading:
            self._set_trajectory_status("Loading...")

    def _on_trajectory_progress(self, loaded: int, total: int) -> None:
        """Update trajectory loading progress text."""
        self._set_trajectory_status(f"{loaded}/{total} frames")

    def _on_trajectory_partial_update(self, trajectories: TrajectorySnapshot) -> None:
        """Render a shared partial snapshot while extraction continues."""
        viz = self.visualizer
        if viz.app_state.show_tx_trajectory:
            self._render_trajectory_if_enabled("tx", trajectories)
        if viz.app_state.show_rx_trajectory:
            self._render_trajectory_if_enabled("rx", trajectories)
        if viz.app_state.show_target_trajectory:
            self._render_trajectory_if_enabled("target", trajectories)
        self._update_trajectory_colorbar(viz.app_state.trajectory_color_mode, trajectories)

    def _on_trajectory_loaded(self, trajectories: TrajectorySnapshot) -> None:
        """Render the shared final trajectory snapshot."""
        viz = self.visualizer
        total = len(trajectories.frames_loaded)
        self._set_trajectory_status(f"Loaded {total} frames")
        if viz.app_state.show_tx_trajectory:
            self._render_trajectory_if_enabled("tx", trajectories)
        if viz.app_state.show_rx_trajectory:
            self._render_trajectory_if_enabled("rx", trajectories)
        if viz.app_state.show_target_trajectory:
            self._render_trajectory_if_enabled("target", trajectories)
        self._update_trajectory_colorbar(viz.app_state.trajectory_color_mode, trajectories)

    def _on_trajectory_error(self, msg: str) -> None:
        """Report trajectory loading errors to UI and logs."""
        self._set_trajectory_status(f"Error: {msg}")
        logger.error("Trajectory loading failed: %s", msg)

    def _on_trajectory_cleared(self) -> None:
        """Remove 3D trajectory presentation when source ownership resets."""
        for kind in ("tx", "rx", "target"):
            self._remove_trajectory_geometry(kind)
        self._set_trajectory_status("Not loaded")
        self._update_trajectory_colorbar("node_color", None)

    def _render_trajectory_if_enabled(
        self,
        kind: str,
        trajectory_data: TrajectorySnapshot,
    ) -> None:
        """Render one trajectory kind if the active renderer supports it."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).trajectories:
            color_mode = viz.app_state.trajectory_color_mode
            scalar_range = self._compute_global_scalar_range(color_mode, trajectory_data)
            renderer.apply_trajectory(
                kind,
                trajectory_data,
                color_mode=color_mode,
                scalar_range=scalar_range,
            )

    def _set_trajectory_status(self, text: str) -> None:
        """Update the trajectory status label in the nodes panel."""
        label = getattr(self.visualizer, "trajectory_status_label", None)
        if label is not None:
            label.setText(text)

    @staticmethod
    def _compute_global_scalar_range(
        color_mode: str, trajectory_data: Mapping[str, Any] | None
    ) -> tuple[float, float] | None:
        """Compute global ``(vmin, vmax)`` across trajectories for a color mode.

        Returns None for ``"node_color"`` or when there is insufficient data.
        """
        import numpy as np

        if color_mode == "node_color" or trajectory_data is None:
            return None

        tracks: list[tuple[np.ndarray, np.ndarray]] = []
        for key in ("tx_positions", "rx_positions"):
            for pos_list in trajectory_data.get(key, {}).values():
                sorted_pos = sorted(pos_list, key=lambda p: p[0])
                if sorted_pos:
                    tracks.append(
                        (
                            np.asarray([[p[1], p[2], p[3]] for p in sorted_pos], dtype=np.float64),
                            np.asarray([p[0] for p in sorted_pos], dtype=np.float64),
                        )
                    )
        for pos_list in trajectory_data.get("target_positions", {}).values():
            sorted_pos = sorted(pos_list, key=lambda p: p[0])
            if sorted_pos:
                tracks.append(
                    (
                        np.asarray([[p[1], p[2], p[3]] for p in sorted_pos], dtype=np.float64),
                        np.asarray([p[0] for p in sorted_pos], dtype=np.float64),
                    )
                )

        if not tracks:
            return None

        all_points = np.concatenate([pts for pts, _frames in tracks], axis=0)
        all_frames = np.concatenate([frames for _pts, frames in tracks], axis=0)

        if color_mode == "speed":
            speed_chunks = []
            for pts, frames in tracks:
                if len(pts) < 2:
                    continue
                dp = np.diff(pts, axis=0)
                dt = np.abs(np.diff(frames))
                dt = np.where(dt < 1e-12, 1.0, dt)
                speed_chunks.append(np.linalg.norm(dp, axis=1) / dt)
            if not speed_chunks:
                return None
            speeds = np.concatenate(speed_chunks)
            return (float(speeds.min()), float(speeds.max()))
        elif color_mode == "altitude":
            return (float(all_points[:, 2].min()), float(all_points[:, 2].max()))
        elif color_mode == "angular_speed":
            angular_chunks = []
            for pts, frames in tracks:
                if len(pts) < 3:
                    continue
                dp = np.diff(pts, axis=0)
                headings = np.arctan2(dp[:, 1], dp[:, 0])
                dh = np.abs(np.diff(headings))
                dh = np.minimum(dh, 2 * np.pi - dh)
                dt = np.abs(np.diff(frames[:-1]))
                dt = np.where(dt < 1e-12, 1.0, dt)
                angular_chunks.append(dh / dt)
            if not angular_chunks:
                return (0.0, 0.0)
            ang_speeds = np.concatenate(angular_chunks)
            return (float(ang_speeds.min()), float(ang_speeds.max()))
        else:  # time
            return (float(all_frames.min()), float(all_frames.max()))

    def _update_trajectory_colorbar(
        self,
        color_mode: str,
        trajectory_data: Mapping[str, Any] | None,
    ) -> None:
        """Compute scalar range from trajectory data and update the panel colorbar."""
        viz = self.visualizer
        nodes_panel = None
        if hasattr(viz, "ui_manager") and viz.ui_manager:
            nodes_panel = viz.ui_manager.panels.get("nodes")
        if nodes_panel is None or not hasattr(nodes_panel, "update_trajectory_colorbar"):
            return

        scalar_range = self._compute_global_scalar_range(color_mode, trajectory_data)
        if scalar_range is None:
            nodes_panel.update_trajectory_colorbar(color_mode)
        else:
            nodes_panel.update_trajectory_colorbar(color_mode, scalar_range[0], scalar_range[1])

    def configure_trajectory_checkboxes(self, enabled: bool) -> None:
        """Enable trajectory controls only for file/remote-HDF5 frame sources."""
        viz = self.visualizer
        for attr in ("tx_trajectory_cb", "rx_trajectory_cb", "target_trajectory_cb"):
            widget = getattr(viz, attr, None)
            if widget is not None:
                widget.setEnabled(enabled)
                if not enabled:
                    widget.setChecked(False)
        if not enabled:
            self._set_trajectory_status(TRAJECTORY_UNAVAILABLE_MESSAGE)
