"""Metrics-window service for per-frame ViewModel statistics."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QStyle

from shared.statistics import FrameStats

from ..services.base import BaseService
from ..state import DEFAULT_MPC_ALLOWED_ORDERS, DEFAULT_MPC_ALLOWED_TYPES

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..metrics.viz_metrics import MetricsWindow as MetricsWindowType

logger = logging.getLogger("orchav.metrics")

METRICS_WINDOW_AVAILABLE = find_spec("pyqtgraph") is not None
MetricsWindow: Any = None
_METRICS_WINDOW_SCREEN_MARGIN_PX = 16


def _load_metrics_window_class() -> Any:
    """Load the optional pyqtgraph window only when the user opens it."""
    global METRICS_WINDOW_AVAILABLE, MetricsWindow
    if MetricsWindow is not None:
        return MetricsWindow
    if not METRICS_WINDOW_AVAILABLE:
        return None
    try:
        from ..metrics.viz_metrics import MetricsWindow as window_class
    except ImportError:
        METRICS_WINDOW_AVAILABLE = False
        return None
    MetricsWindow = window_class
    return MetricsWindow


class MetricsService(BaseService):
    """Service responsible for managing the metrics window and statistics updates."""

    def __init__(self, visualizer: OrchavVisualizer):
        """Bind the service to the visualizer and defer metrics-window creation."""
        super().__init__()
        self.visualizer = visualizer
        self.window: Optional["MetricsWindowType"] = None

    @property
    def available(self) -> bool:
        """Return whether the optional metrics UI dependency is installed."""
        return bool(METRICS_WINDOW_AVAILABLE)

    def toggle_window(self) -> None:
        """Open or close the metrics window."""
        window_class = _load_metrics_window_class()
        if window_class is None:
            logger.warning("Metrics window not available - pyqtgraph may not be installed")
            return

        if self.window is None:
            self.window = window_class(
                self.visualizer,
                update_hz=8,
                frame_stats_provider=self._compute_frame_stats,
            )
            self.window.destroyed.connect(self._on_window_closed)
            self.visualizer.metrics_window = self.window
            self._place_window_on_visualizer_screen()
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()
            logger.info("Metrics window opened")

            # Run the force flag and pipeline update in one deferred callback so
            # the render loop cannot consume the flag between those operations.
            viz = self.visualizer
            if hasattr(viz, "force_update_next_frame") and hasattr(viz, "pipeline"):
                QTimer.singleShot(0, self._force_pipeline_rerender)
        else:
            self.window.close()
            self._on_window_closed()
            logger.info("Metrics window closed")

    def _force_pipeline_rerender(self) -> None:
        """Set force flag and run the pipeline in a single event-loop tick.

        This avoids the race where the render-loop poll clears
        force_update_next_frame before the debounced schedule_update fires.
        """
        viz = self.visualizer
        try:
            viz.force_update_next_frame = True
            step = getattr(viz, "animation_step", 0)
            viz.pipeline.update(step)
            logger.debug("Forced pipeline re-render for metrics window (step %s)", step)
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Could not force pipeline re-render: %s", exc)

    def _on_window_closed(self, *_) -> None:
        """Cleanup references when the metrics window is destroyed."""
        self.window = None
        if hasattr(self.visualizer, "metrics_window"):
            setattr(self.visualizer, "metrics_window", None)

    def _place_window_on_visualizer_screen(self) -> None:
        """Center Metrics within the available area of the visualizer's monitor."""
        if self.window is None:
            return
        screen_getter = getattr(self.visualizer, "screen", None)
        screen = screen_getter() if callable(screen_getter) else None
        if screen is None:
            return
        margin = _METRICS_WINDOW_SCREEN_MARGIN_PX
        available = screen.availableGeometry().adjusted(margin, margin, -margin, -margin)
        size = self.window.size().boundedTo(available.size())
        geometry = QStyle.alignedRect(
            Qt.LayoutDirection.LeftToRight,
            Qt.AlignmentFlag.AlignCenter,
            size,
            available,
        )
        self.window.setGeometry(geometry)

    def update_metrics(self, view_model: Any) -> None:
        """Offer the latest ViewModel to the metrics window without blocking playback."""
        if self.window is None:
            return
        if bool(getattr(self.window, "updates_paused", False)):
            return

        try:
            context = self._build_context(view_model, None)
            self.window.enqueue(view_model, None, context=context)
            logger.debug("Latest ViewModel offered to metrics window")
        except Exception as exc:  # noqa: BLE001 - metrics UI must not break frame updates.
            logger.debug("Metrics window update failed: %s", exc)

    def _compute_frame_stats(self, view_model: Any) -> FrameStats:
        """Compute selected-path statistics when the window consumes a refresh tick."""
        viz = self.visualizer
        if not (hasattr(viz, "mpc_core") and hasattr(viz.mpc_core, "stats")):
            return self._empty_frame_stats()
        try:
            processed_data = {
                "canonical_data": view_model.canonical_data,
                "metrics_visible": True,
                "path_mask": getattr(view_model, "path_mask", None),
            }
            frame_stats = viz.mpc_core.stats(processed_data)
            logger.debug("MPCCore computed stats from canonical data: %s", frame_stats)
            return frame_stats
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("MPCCore stats computation failed: %s", exc)
            return self._empty_frame_stats()

    @staticmethod
    def _empty_frame_stats() -> FrameStats:
        """Return an explicit empty result for unavailable dashboard statistics."""
        return FrameStats(
            total_paths=0,
            orders_hist={},
            delay_range_ns=None,
            path_loss_range=None,
        )

    def _build_context(self, view_model: Any, frame_stats: Optional[FrameStats]) -> dict[str, Any]:
        """Return display metadata for the metrics window status strip."""
        viz = self.visualizer
        state = getattr(viz, "app_state", None)
        selected_tx = getattr(state, "selected_tx", "all")
        selected_rx = getattr(state, "selected_rx", "all")
        step = getattr(state, "step", getattr(viz, "animation_step", None))

        visible_paths = (
            getattr(frame_stats, "total_paths", None) if frame_stats is not None else None
        )

        return {
            "step": step,
            "selected_tx": selected_tx,
            "selected_rx": selected_rx,
            "visible_paths": visible_paths,
            "total_paths": self._canonical_path_count(getattr(view_model, "canonical_data", None)),
            "filters": self._active_filter_labels(),
        }

    @staticmethod
    def _canonical_path_count(canon_data: Any) -> int:
        """Count canonical MPC paths using the first populated path-level array."""
        if canon_data is None:
            return 0
        for name in ("path_orders", "path_delays", "path_losses", "path_tx", "path_rx"):
            arr = getattr(canon_data, name, None)
            if arr is not None and getattr(arr, "size", 0) > 0:
                return int(arr.shape[0])
        starts = getattr(canon_data, "path_start_indices", None)
        if starts is not None and getattr(starts, "size", 0) > 0:
            return int(starts.shape[0])
        path_id = getattr(canon_data, "path_id", None)
        if path_id is not None and getattr(path_id, "size", 0) > 0:
            try:
                return int(max(path_id)) + 1
            except (TypeError, ValueError):
                return 0
        return 0

    def _active_filter_labels(self) -> list[str]:
        """Return compact labels for active MPC filters in the status strip."""
        state = getattr(self.visualizer, "app_state", None)
        if state is None:
            return []
        labels: list[str] = []
        if getattr(state, "mpc_allowed_orders", DEFAULT_MPC_ALLOWED_ORDERS) != (
            DEFAULT_MPC_ALLOWED_ORDERS
        ):
            labels.append("orders")
        if getattr(state, "mpc_allowed_types", DEFAULT_MPC_ALLOWED_TYPES) != (
            DEFAULT_MPC_ALLOWED_TYPES
        ):
            labels.append("types")
        if (
            getattr(state, "delay_filter_min_ns", None) is not None
            or getattr(state, "delay_filter_max_ns", None) is not None
        ):
            labels.append("delay")
        if (
            getattr(state, "loss_filter_min_db", None) is not None
            or getattr(state, "loss_filter_max_db", None) is not None
            or getattr(state, "path_loss_filter_min_db", None) is not None
            or getattr(state, "path_loss_filter_max_db", None) is not None
            or getattr(state, "power_filter_min_db", None) is not None
            or getattr(state, "power_filter_max_db", None) is not None
        ):
            labels.append("loss")
        if any(
            getattr(state, name, None) is not None
            for name in (
                "aoa_az_filter_min_deg",
                "aoa_az_filter_max_deg",
                "aoa_el_filter_min_deg",
                "aoa_el_filter_max_deg",
            )
        ):
            labels.append("AoA")
        if any(
            getattr(state, name, None) is not None
            for name in (
                "aod_az_filter_min_deg",
                "aod_az_filter_max_deg",
                "aod_el_filter_min_deg",
                "aod_el_filter_max_deg",
            )
        ):
            labels.append("AoD")
        materials = getattr(self.visualizer, "mpc_allowed_materials", None)
        if materials:
            labels.append("materials")
        return labels
