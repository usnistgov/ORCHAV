"""Coverage-map UI controller and cache-invalidation bridge.

``CoverageController`` translates coverage-panel actions into visualizer state,
coverage-service cache decisions, and frame-pipeline refreshes. The coverage
service owns HDF5 metric loading and mesh caching; this controller owns user
intent such as metric selection, height animation, threshold overlays, and
renderer-only opacity changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from PySide6.QtCore import QTimer

from ..coverage.analysis import (
    compute_coverage_threshold_summary,
    coverage_metric_valid_mask,
    default_coverage_threshold,
    format_coverage_threshold_summary,
    is_serving_tx_metric,
    supports_coverage_threshold,
)
from ..renderers.protocol import renderer_capabilities
from ..services.coverage_service import (
    DEFAULT_COVERAGE_HEIGHT_ANIMATION_SPEED,
    DEFAULT_COVERAGE_INTERPOLATION,
    DEFAULT_COVERAGE_ISOLINE_COUNT,
    DEFAULT_COVERAGE_OPACITY,
)

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..services.coverage_service import CoverageService

logger = logging.getLogger(__name__)


class CoverageController:
    """Coordinate coverage panel state with the frame pipeline and renderer."""

    def __init__(self, parent: Any) -> None:
        """Initialize user-facing coverage state owned by the UI controller."""
        self._parent = parent
        self._height_animation_timer: Optional[QTimer] = None
        self.height_animation_speed: int = 3
        self.height_animation_active: bool = False
        self.coverage_interpolation_method: str = "none"
        self.coverage_threshold_enabled: bool = False
        self.coverage_threshold_value: Optional[float] = None
        self.coverage_threshold_mask_enabled: bool = False
        self.coverage_isolines_enabled: bool = False
        self.coverage_isoline_count: int = 6

    def reset_runtime_state(self) -> None:
        """Stop playback and restore controller-owned coverage defaults."""
        self.handle_coverage_height_animation_stop()
        self.height_animation_speed = 3
        self.coverage_interpolation_method = "none"
        self.coverage_threshold_enabled = False
        self.coverage_threshold_value = None
        self.coverage_threshold_mask_enabled = False
        self.coverage_isolines_enabled = False
        self.coverage_isoline_count = 6
        viz = self.visualizer
        viz.coverage_interpolation_method = "none"
        self._sync_threshold_state_to_visualizer()

    @property
    def visualizer(self) -> OrchavVisualizer:
        """Return the visualizer owned by the parent UI controller."""
        return self._parent.visualizer

    @property
    def coverage_service(self) -> CoverageService:
        """Return the coverage service owned by the parent UI controller."""
        return self._parent.coverage_service

    @property
    def coverage_height_index(self) -> int:
        """Current coverage height index from the visualizer."""
        return getattr(self.visualizer, "coverage_height_index", 0)

    @coverage_height_index.setter
    def coverage_height_index(self, value: int) -> None:
        """Store the selected coverage-height index on the visualizer."""
        self.visualizer.coverage_height_index = value

    def handle_coverage_toggled(self, state: bool) -> None:
        """Mirror the coverage visibility toggle into app state and redraw."""
        viz = self.visualizer
        logger.debug("Coverage toggle changed to: %s", state)
        viz.set_state(show_coverage=bool(state))
        if state and getattr(viz, "coverage_data", None):
            self._refresh_coverage_visuals()
        elif getattr(viz, "coverage_data", None) and (
            bool(getattr(viz, "_scene_only_mode", False)) or not bool(getattr(viz, "ready", True))
        ):
            viz._coverage_interpolation_dirty = True
            viz.force_update_next_frame = True
            self._render_current_coverage()
        elif hasattr(viz, "schedule_update"):
            viz.schedule_update()

    def handle_coverage_opacity_changed(self, value: int) -> None:
        """Apply coverage opacity as a renderer-only visual property.

        Opacity is intentionally excluded from coverage mesh cache keys, so
        renderers that support material transparency can update it without
        rebuilding the current ViewModel.
        """
        viz = self.visualizer
        viz.coverage_opacity = value / 100.0
        logger.debug("Coverage opacity changed to: %s", viz.coverage_opacity)

        if viz.renderer is not None and renderer_capabilities(viz.renderer).transparency:
            viz.renderer.set_coverage_transparency(viz.coverage_opacity)
        else:
            viz.schedule_update()

    def handle_coverage_height_changed(self, index: int) -> None:
        """Select one coverage height slice and rebuild the current frame."""
        viz = self.visualizer
        if viz.coverage_data is None or not viz.coverage_heights:
            logger.debug("Coverage height change ignored: no coverage data loaded")
            return

        try:
            new_index = int(index)
        except (TypeError, ValueError):
            new_index = 0

        new_index = max(0, min(new_index, len(viz.coverage_heights) - 1))
        if new_index == getattr(viz.app_state, "coverage_height_index", viz.coverage_height_index):
            return
        if not self._activate_coverage_height(new_index):
            return

        viz.coverage_height_index = new_index
        viz.set_state(coverage_height_index=new_index)
        height_value = viz.coverage_heights[new_index]
        logger.debug("Coverage height index changed to %d (%.2f m)", new_index, height_value)

        if hasattr(viz, "ui_manager") and "coverage" in viz.ui_manager.panels:
            panel = viz.ui_manager.panels["coverage"]
            if hasattr(panel, "set_height_index"):
                panel.set_height_index(new_index)
        self._notify_coverage_graphs()
        self._refresh_threshold_summary()
        self._refresh_coverage_visuals()

    def handle_coverage_cache_all_clicked(self) -> None:
        """Precompute missing base meshes without changing the displayed slice."""
        viz = self.visualizer
        if not hasattr(viz, "coverage_data") or viz.coverage_data is None:
            logger.warning("No coverage data available for caching")
            return
        if not hasattr(viz, "pipeline"):
            logger.warning("Frame pipeline not available")
            return

        heights = viz.coverage_heights
        if not heights:
            logger.warning("No heights available for caching")
            return

        prewarm = getattr(viz.pipeline, "precache_coverage_heights", None)
        if not callable(prewarm):
            logger.warning("Coverage height pre-cache API is unavailable")
            return
        logger.info("Pre-caching %d coverage heights...", len(heights))
        cached_count, reused_count = prewarm()
        self.coverage_service.log_stats()
        logger.info(
            "Pre-caching complete: generated %d, reused %d",
            cached_count,
            reused_count,
        )

    def handle_coverage_interpolation_changed(self, method: str) -> None:
        """Change spatial interpolation and invalidate derived coverage meshes."""
        viz = self.visualizer
        metric_name = None
        if getattr(viz, "coverage_data", None):
            metric_name = viz.coverage_data.get("metric_name")
        if is_serving_tx_metric(metric_name):
            method = "none"
        self.coverage_interpolation_method = method
        viz.coverage_interpolation_method = method
        logger.info("Coverage spatial interpolation method changed to: %s", method)
        if hasattr(viz, "pipeline"):
            viz.pipeline.clear_coverage_cache()
        viz._coverage_interpolation_dirty = True
        self._refresh_coverage_visuals()

    def handle_coverage_metric_changed(self, metric_name: str) -> None:
        """Switch the active coverage metric layer and reset threshold state."""
        viz = self.visualizer
        if not getattr(viz, "coverage_data", None):
            return
        available = viz.coverage_data.get("available_metrics", [])
        if metric_name not in available:
            logger.warning("Coverage metric %s is not available", metric_name)
            return
        self.coverage_service.select_metric_layer(viz.coverage_data, metric_name)
        if not self._activate_coverage_height(int(getattr(viz, "coverage_height_index", 0))):
            return
        metric_name = str(viz.coverage_data.get("metric_name", metric_name))
        viz.coverage_metric_name = metric_name
        self.coverage_threshold_value = None
        if not supports_coverage_threshold(metric_name):
            self.coverage_threshold_enabled = False
            self.coverage_threshold_mask_enabled = False
            self.coverage_isolines_enabled = False
        if is_serving_tx_metric(metric_name):
            self.coverage_interpolation_method = "none"
            viz.coverage_interpolation_method = "none"
        self._sync_threshold_state_to_visualizer()
        if hasattr(viz, "pipeline"):
            viz.pipeline.clear_coverage_cache()
        viz._coverage_interpolation_dirty = True
        self._refresh_coverage_panel_status()
        logger.info("Coverage metric changed to: %s", metric_name)
        self._notify_coverage_graphs()
        self._refresh_coverage_visuals()

    def handle_coverage_threshold_changed(self, enabled: bool, value: float) -> None:
        """Update scalar threshold analysis and any visible threshold overlay."""
        metric_name = None
        if getattr(self.visualizer, "coverage_data", None):
            metric_name = self.visualizer.coverage_data.get("metric_name")
        was_enabled = self.coverage_threshold_enabled
        self.coverage_threshold_enabled = bool(enabled and supports_coverage_threshold(metric_name))
        if self.coverage_threshold_enabled and not was_enabled:
            try:
                analysis_data = self._displayed_slice_analysis_data()
                self.coverage_threshold_value = default_coverage_threshold(analysis_data, 0)
            except (TypeError, ValueError, KeyError, IndexError):
                self.coverage_threshold_value = None
        elif self.coverage_threshold_enabled:
            try:
                self.coverage_threshold_value = float(value)
            except (TypeError, ValueError):
                self.coverage_threshold_value = None
        else:
            self.coverage_threshold_value = None
        if not self.coverage_threshold_enabled:
            self.coverage_threshold_mask_enabled = False
        self._sync_threshold_state_to_visualizer()
        self._sync_analysis_state_to_panel()
        self._refresh_threshold_summary()
        self._refresh_coverage_visuals()

    def handle_coverage_threshold_mask_changed(self, enabled: bool) -> None:
        """Toggle visual dimming for coverage cells outside the threshold."""
        self.coverage_threshold_mask_enabled = bool(enabled and self.coverage_threshold_enabled)
        self._sync_threshold_state_to_visualizer()
        self._sync_analysis_state_to_panel()
        self._refresh_coverage_visuals()

    def handle_coverage_isolines_changed(self, enabled: bool, count: int) -> None:
        """Toggle auto isolines for the active scalar coverage metric."""
        metric_name = None
        if getattr(self.visualizer, "coverage_data", None):
            metric_name = self.visualizer.coverage_data.get("metric_name")
        requested_enabled = bool(enabled and supports_coverage_threshold(metric_name))
        try:
            requested_count = max(2, min(int(count), 12))
        except (TypeError, ValueError):
            requested_count = 6
        if (
            requested_enabled == self.coverage_isolines_enabled
            and requested_count == self.coverage_isoline_count
        ):
            return
        self.coverage_isolines_enabled = requested_enabled
        self.coverage_isoline_count = requested_count
        self._sync_threshold_state_to_visualizer()
        self._sync_analysis_state_to_panel()
        self._refresh_coverage_visuals()

    def restore_session_state(self, state: dict[str, Any]) -> None:
        """Restore one validated coverage-control snapshot without signal fan-out.

        Session loading is a batch operation.  Applying each saved value through
        an individual widget callback would rebuild the same coverage several
        times and would make widgets accidental state owners.  This method is
        the controller-owned restore boundary: it normalizes the snapshot,
        updates runtime intent once, and then mirrors the result to the panel.
        """
        if not isinstance(state, dict):
            return

        viz = self.visualizer
        coverage_data = getattr(viz, "coverage_data", None)
        requested_metric = state.get("metric_name")
        if requested_metric and isinstance(coverage_data, dict):
            available = coverage_data.get("available_metrics", []) or []
            if requested_metric in available:
                try:
                    self.coverage_service.select_metric_layer(
                        coverage_data,
                        str(requested_metric),
                    )
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Could not restore coverage metric %s: %s",
                        requested_metric,
                        exc,
                    )
            else:
                logger.debug("Saved coverage metric is unavailable: %s", requested_metric)

        active_metric = None
        if isinstance(coverage_data, dict):
            active_metric = coverage_data.get("metric_name")
        if not active_metric:
            active_metric = requested_metric or getattr(viz, "coverage_metric_name", None)

        try:
            opacity = float(state.get("opacity", DEFAULT_COVERAGE_OPACITY))
        except (TypeError, ValueError):
            opacity = DEFAULT_COVERAGE_OPACITY
        opacity = max(0.1, min(opacity, 1.0))

        interpolation_value = str(
            state.get("interpolation", DEFAULT_COVERAGE_INTERPOLATION)
        ).strip()
        interpolation = {
            "raw": "none",
            "nearest": "none",
            "smooth": "linear",
            "smooth+": "cubic",
        }.get(interpolation_value.lower(), interpolation_value.lower())
        if interpolation not in {"none", "linear", "cubic"}:
            interpolation = DEFAULT_COVERAGE_INTERPOLATION
        if is_serving_tx_metric(active_metric):
            interpolation = DEFAULT_COVERAGE_INTERPOLATION

        thresholdable = bool(active_metric) and supports_coverage_threshold(active_metric)
        threshold_enabled = bool(state.get("threshold_enabled", False)) and thresholdable
        try:
            raw_threshold = state.get("threshold_value")
            threshold_value = float(raw_threshold) if raw_threshold is not None else None
        except (TypeError, ValueError):
            threshold_value = None
        threshold_mask_enabled = (
            bool(state.get("threshold_mask_enabled", False)) and threshold_enabled
        )
        isolines_enabled = bool(state.get("isolines_enabled", False)) and thresholdable
        try:
            isoline_count = max(
                2,
                min(
                    int(state.get("isoline_count", DEFAULT_COVERAGE_ISOLINE_COUNT)),
                    12,
                ),
            )
        except (TypeError, ValueError):
            isoline_count = DEFAULT_COVERAGE_ISOLINE_COUNT
        try:
            height_speed = max(
                1,
                min(
                    int(
                        state.get(
                            "height_animation_speed",
                            DEFAULT_COVERAGE_HEIGHT_ANIMATION_SPEED,
                        )
                    ),
                    10,
                ),
            )
        except (TypeError, ValueError):
            height_speed = DEFAULT_COVERAGE_HEIGHT_ANIMATION_SPEED

        self.coverage_interpolation_method = interpolation
        self.coverage_threshold_enabled = threshold_enabled
        self.coverage_threshold_value = threshold_value
        self.coverage_threshold_mask_enabled = threshold_mask_enabled
        self.coverage_isolines_enabled = isolines_enabled
        self.coverage_isoline_count = isoline_count
        self.height_animation_speed = height_speed

        viz.coverage_opacity = opacity
        viz.coverage_metric_name = active_metric
        viz.coverage_interpolation_method = interpolation
        self._sync_threshold_state_to_visualizer()
        viz._coverage_interpolation_dirty = True

        self.coverage_service.clear()
        renderer = getattr(viz, "renderer", None)
        if renderer is not None and renderer_capabilities(renderer).transparency:
            renderer.set_coverage_transparency(opacity)

        panel = None
        if getattr(viz, "ui_manager", None) is not None:
            panel = getattr(viz.ui_manager, "panels", {}).get("coverage")
        if panel is not None:
            if isinstance(coverage_data, dict):
                panel.update_coverage_status(
                    True,
                    coverage_data,
                    supports_transparency=renderer_capabilities(renderer).transparency,
                )
            restore_controls = getattr(panel, "restore_session_controls", None)
            if callable(restore_controls):
                restore_controls(
                    opacity=opacity,
                    threshold_enabled=threshold_enabled,
                    threshold_value=threshold_value,
                    mask_enabled=threshold_mask_enabled,
                    isolines_enabled=isolines_enabled,
                    isoline_count=isoline_count,
                    interpolation=interpolation,
                    height_animation_speed=height_speed,
                )
        self._refresh_threshold_summary()
        self._notify_coverage_graphs()
        if isinstance(coverage_data, dict):
            viz.force_update_next_frame = True

    def _notify_coverage_graphs(self, *, render: bool = True) -> None:
        """Notify dynamic coverage figures after committed selection changes."""
        ui_manager = getattr(self.visualizer, "ui_manager", None)
        notify = getattr(ui_manager, "notify_coverage_selection_changed", None)
        if callable(notify):
            notify(render=render)

    def _sync_threshold_state_to_visualizer(self) -> None:
        """Expose coverage threshold overlay state to the frame pipeline."""
        viz = self.visualizer
        viz.coverage_threshold_enabled = self.coverage_threshold_enabled
        viz.coverage_threshold_value = self.coverage_threshold_value
        viz.coverage_threshold_mask_enabled = self.coverage_threshold_mask_enabled
        viz.coverage_isolines_enabled = self.coverage_isolines_enabled
        viz.coverage_isoline_count = self.coverage_isoline_count

    def _sync_analysis_state_to_panel(self) -> None:
        """Mirror controller-owned coverage analysis state atomically."""
        viz = self.visualizer
        panel = None
        if getattr(viz, "ui_manager", None) is not None:
            panel = getattr(viz.ui_manager, "panels", {}).get("coverage")
        if panel is None or not hasattr(panel, "set_analysis_state"):
            return
        panel.set_analysis_state(
            threshold_enabled=self.coverage_threshold_enabled,
            threshold_value=self.coverage_threshold_value,
            mask_enabled=self.coverage_threshold_mask_enabled,
            isolines_enabled=self.coverage_isolines_enabled,
            isoline_count=self.coverage_isoline_count,
            interpolation=self.coverage_interpolation_method,
        )

    def _refresh_coverage_visuals(self) -> None:
        """Regenerate the current ViewModel when threshold visuals change."""
        viz = self.visualizer
        if not getattr(viz.app_state, "show_coverage", False):
            return
        if not getattr(viz, "coverage_data", None):
            return
        viz._coverage_interpolation_dirty = True
        viz.force_update_next_frame = True
        self._render_current_coverage()

    def _render_current_coverage(self) -> None:
        """Refresh coverage through the scene-only or frame-backed pipeline."""
        viz = self.visualizer
        scene_only = bool(getattr(viz, "_scene_only_mode", False))
        ready = bool(getattr(viz, "ready", True))
        pipeline = getattr(viz, "pipeline", None)
        if scene_only or not ready:
            update_overlay = getattr(pipeline, "update_coverage_overlay", None)
            if callable(update_overlay):
                update_overlay()
            elif hasattr(viz, "schedule_update"):
                viz.schedule_update()
                return
        elif hasattr(viz, "_process_frame_step"):
            viz._process_frame_step(getattr(viz, "animation_step", 0))
        elif hasattr(viz, "schedule_update"):
            viz.schedule_update()
            return

        if hasattr(viz, "update_visualizer"):
            viz.update_visualizer()
        elif hasattr(viz, "schedule_update"):
            viz.schedule_update()

    def _refresh_coverage_panel_status(self) -> None:
        """Refresh compact panel metadata after metric or range changes."""
        viz = self.visualizer
        panel = None
        if getattr(viz, "ui_manager", None) is not None:
            panel = getattr(viz.ui_manager, "panels", {}).get("coverage")
        if panel is None or not getattr(viz, "coverage_data", None):
            return

        renderer = getattr(viz, "renderer", None)
        supports_transparency = renderer_capabilities(renderer).transparency
        panel.update_coverage_status(
            True,
            viz.coverage_data,
            supports_transparency=supports_transparency,
        )
        if hasattr(panel, "set_height_index"):
            panel.set_height_index(getattr(viz, "coverage_height_index", 0))
        self._sync_analysis_state_to_panel()
        self._refresh_threshold_summary()

    def _refresh_threshold_summary(self) -> None:
        """Refresh threshold readout for the active scalar coverage layer."""
        viz = self.visualizer
        panel = None
        if getattr(viz, "ui_manager", None) is not None:
            panel = getattr(viz.ui_manager, "panels", {}).get("coverage")
        if panel is None or not getattr(viz, "coverage_data", None):
            return

        coverage_data = viz.coverage_data
        metric_name = coverage_data.get("metric_name")
        if not supports_coverage_threshold(metric_name):
            if hasattr(panel, "set_threshold_summary"):
                panel.set_threshold_summary("Not available for this metric", active=False)
            return

        if not self.coverage_threshold_enabled:
            if hasattr(panel, "set_threshold_summary"):
                panel.set_threshold_summary("Off", active=False)
            return

        try:
            analysis_data = self._displayed_slice_analysis_data()
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            logger.debug("Coverage displayed slice unavailable: %s", exc)
            if hasattr(panel, "set_threshold_summary"):
                panel.set_threshold_summary("Unavailable", active=False)
            return

        threshold = self.coverage_threshold_value
        if threshold is None:
            # The panel and pipeline must agree on the auto-selected threshold.
            threshold = default_coverage_threshold(analysis_data, 0)
            self.coverage_threshold_value = threshold
            self._sync_threshold_state_to_visualizer()
            if hasattr(panel, "set_threshold_value"):
                panel.set_threshold_value(threshold)

        try:
            summary = compute_coverage_threshold_summary(
                analysis_data,
                height_index=0,
                threshold=threshold,
            )
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            logger.debug("Coverage threshold summary unavailable: %s", exc)
            if hasattr(panel, "set_threshold_summary"):
                panel.set_threshold_summary("Unavailable", active=False)
            return

        if hasattr(panel, "set_threshold_summary"):
            panel.set_threshold_summary(format_coverage_threshold_summary(summary), active=True)

    def _displayed_slice_analysis_data(self) -> dict[str, Any]:
        """Return analysis metadata containing the actual smoothed 2D slice."""
        viz = self.visualizer
        coverage_data = viz.coverage_data
        values = coverage_data.get("values_3d")
        if values is None:
            values = coverage_data.get("values")
        array = np.asarray(values)
        if array.ndim >= 3:
            if coverage_data.get("coverage_file"):
                height_index = 0
            else:
                height_index = max(
                    0,
                    min(int(getattr(viz, "coverage_height_index", 0)), array.shape[0] - 1),
                )
            selected = array[height_index]
        elif array.ndim == 2:
            selected = array
        else:
            selected = array.reshape(1, -1)
        analysis_values = np.where(
            coverage_metric_valid_mask(selected, coverage_data.get("metric_name")),
            selected,
            np.nan,
        )
        displayed = self.coverage_service.interpolate_values(
            analysis_values,
            self.coverage_interpolation_method,
        )
        analysis_data = dict(coverage_data)
        analysis_data["values_3d"] = np.asarray(displayed)
        analysis_data["values"] = np.asarray(displayed)
        return analysis_data

    def _activate_coverage_height(self, height_index: int) -> bool:
        """Load a file-backed height before publishing it as current UI state."""
        select_height = getattr(self.coverage_service, "select_height_layer", None)
        if not callable(select_height):
            return True
        try:
            select_height(self.visualizer.coverage_data, int(height_index))
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            logger.warning("Could not load coverage height %s: %s", height_index, exc)
            return False
        return True

    def _any_heights_uncached(self) -> bool:
        """Return True if any coverage height is missing from the mesh cache."""
        viz = self.visualizer
        if not viz.coverage_data or not viz.coverage_heights:
            return False
        interp = self.coverage_interpolation_method
        for idx in range(len(viz.coverage_heights)):
            key = self.coverage_service.compute_cache_key(viz.coverage_data, idx, interp)
            if self.coverage_service.get_mesh(key, copy=False) is None:
                return True
        return False

    def handle_coverage_height_animation_play(self) -> None:
        """Toggle height-slice animation, prewarming missing meshes first."""
        viz = self.visualizer
        if not hasattr(viz, "coverage_heights") or len(viz.coverage_heights) <= 1:
            logger.warning("Cannot start height animation - need multiple heights")
            return

        if self.height_animation_active:
            self.handle_coverage_height_animation_stop()
            return

        if self._any_heights_uncached():
            logger.info("Auto-caching heights before animation...")
            self.handle_coverage_cache_all_clicked()

        logger.info("Starting height animation...")
        self.height_animation_active = True
        if viz.ui_manager and "coverage" in viz.ui_manager.panels:
            widgets = viz.ui_manager.panels["coverage"].widgets
            play_btn = widgets.get("coverage_height_play_btn")
            stop_btn = widgets.get("coverage_height_stop_btn")
            if play_btn:
                play_btn.setText("Pause")
                play_btn.setEnabled(True)
            if stop_btn:
                stop_btn.setEnabled(True)
        self._start_height_animation()

    def handle_coverage_height_animation_stop(self) -> None:
        """Stop height-slice animation and restore button state."""
        was_active = self.height_animation_active
        if was_active:
            logger.info("Stopping height animation...")
        self.height_animation_active = False
        if self._height_animation_timer:
            self._height_animation_timer.stop()
            self._height_animation_timer.deleteLater()
            self._height_animation_timer = None
        viz = self.visualizer
        if getattr(viz, "ui_manager", None) and "coverage" in viz.ui_manager.panels:
            play_btn = viz.ui_manager.panels["coverage"].widgets.get("coverage_height_play_btn")
            stop_btn = viz.ui_manager.panels["coverage"].widgets.get("coverage_height_stop_btn")
            if play_btn:
                play_btn.setText("Play heights")
                play_btn.setEnabled(len(getattr(viz, "coverage_heights", [])) > 1)
            if stop_btn:
                stop_btn.setEnabled(False)
        if was_active:
            self._notify_coverage_graphs()

    def handle_coverage_height_animation_speed_changed(self, speed: int) -> None:
        """Update height animation speed and restart the active timer."""
        self.height_animation_speed = max(1, min(int(speed), 10))
        logger.info("Coverage height animation speed changed to: %d", self.height_animation_speed)
        if self.height_animation_active:
            self._start_height_animation()

    def _start_height_animation(self) -> None:
        """Start or restart the height animation timer."""
        if self._height_animation_timer:
            self._height_animation_timer.stop()
        interval_ms = int(2000 / self.height_animation_speed)
        self._height_animation_timer = QTimer()
        self._height_animation_timer.timeout.connect(self._height_animation_step)
        self._height_animation_timer.start(interval_ms)
        logger.debug("Height animation timer started with interval %dms", interval_ms)

    def _height_animation_step(self) -> None:
        """Advance one height slice and refresh visible coverage, if enabled."""
        if not self.height_animation_active:
            return
        viz = self.visualizer
        if not getattr(viz, "coverage_heights", None):
            self.handle_coverage_height_animation_stop()
            return
        next_index = (self.coverage_height_index + 1) % len(viz.coverage_heights)
        if not self._activate_coverage_height(next_index):
            self.handle_coverage_height_animation_stop()
            return
        self.coverage_height_index = next_index
        if viz.ui_manager and "coverage" in viz.ui_manager.panels:
            panel = viz.ui_manager.panels["coverage"]
            if hasattr(panel, "set_height_index"):
                panel.set_height_index(self.coverage_height_index)
        viz.set_state(coverage_height_index=self.coverage_height_index)
        self._notify_coverage_graphs(render=False)
        self._refresh_threshold_summary()
        self._refresh_coverage_visuals()
