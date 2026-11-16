"""Scenario statistics controls, summaries, charts, and export helpers.

``StatisticsPanel`` streams selective MPC projections from the active provider,
prefers the scenario statistics cache when valid, renders charts lazily when
the Graphs section is visible, and exports CSV/PNG artifacts on demand. The
provider service owns background work and cache publication, so the panel does
not retain playback frames.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from shared.logging import get_logger

from ..app.plot_theme import apply_matplotlib_legend_theme, apply_matplotlib_theme
from ..app.theme import current_theme, get_theme_manager
from ..config import INTERACTION_TYPE_COLORS
from ..coverage.analysis import (
    compute_coverage_scalar_plot_data,
    compute_serving_tx_coverage_summary,
    is_serving_tx_metric,
    serving_tx_color_hex,
)
from ..services.mpc_interaction_style_service import (
    MPC_UNKNOWN_COLOR,
    mpc_interaction_label,
    mpc_interaction_sort_key,
    mpc_interaction_style,
    rgb_to_css_hex,
)
from ..services.scenario_statistics_service import (
    ScenarioStatisticsResult,
    resolve_statistics_provider,
)
from .base import BasePanel
from .collapsible_section import CollapsibleSection

logger = get_logger("orchav.statistics_panel")

_CHART_ORDER_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#9D755D",
]
_CHART_LINE_COLOR = "#1976D2"
_MAX_CDF_POINTS = 5000
_GRAPHS_STATUS_CLEAR_MS = 4000
EMPTY_VALUE = "N/A"

_GRAPH_EXPLANATIONS = {
    "delay_time_chart": (
        "Sum of retained path powers in each delay bin, without phase interference. "
        "Time bins containing multiple observations show their mean power. Blank cells "
        "have no co-valid delay/power data; they do not establish an outage. "
        "Click to inspect the nearest recorded frame. Coverage retains its own selection."
    ),
    "coverage_distribution_chart": (
        "For scalar coverage, each bar shows the share of usable cells in one value "
        "range at the selected height. For Serving TX, each bar shows the share of "
        "the full slice assigned to one transmitter or to no service."
    ),
    "coverage_success_chart": (
        "For each threshold, this curve shows the share of the full selected-height "
        "slice that meets the metric requirement. Cells without usable data do not "
        "qualify."
    ),
    "reflection_order_chart": (
        "Each retained MPC is counted once by its number of interactions. Six or more "
        "interactions are combined in the 6+ bucket; counts are not weighted by path "
        "power."
    ),
    "interaction_type_chart": (
        "Each retained MPC is counted once. A direct path is LoS; otherwise the "
        "category is its first interaction. Counts are not weighted by path power."
    ),
    "path_loss_histogram_chart": (
        "Distribution of valid path loss values across retained MPCs. Lower path loss "
        "means a stronger path; each MPC contributes one count."
    ),
    "delay_histogram_chart": (
        "Distribution of valid propagation delays across retained MPCs. Each MPC "
        "contributes one count."
    ),
    "delay_cdf_chart": (
        "At each delay, the CDF is the fraction of valid retained MPCs with an equal or "
        "smaller propagation delay."
    ),
    "path_loss_cdf_chart": (
        "At each path-loss value, the CDF is the fraction of valid retained MPCs with "
        "an equal or smaller loss."
    ),
    "pair_gain_cdf_chart": (
        "One sample represents one TX/RX pair in one frame with usable path loss. "
        "Retained path powers are summed incoherently before conversion to dB."
    ),
    "pair_delay_spread_cdf_chart": (
        "One sample represents the power-weighted RMS delay spread of one TX/RX pair "
        "in one frame, using paths with both valid delay and path loss."
    ),
    "aod_az_polar_chart": (
        "Azimuth directions in which retained MPCs leave transmitters. Each valid MPC "
        "contributes one count; the distribution is not power weighted."
    ),
    "aod_el_polar_chart": (
        "Elevation directions in which retained MPCs leave transmitters. Each valid "
        "MPC contributes one count; the distribution is not power weighted."
    ),
    "aoa_az_polar_chart": (
        "Azimuth directions from which retained MPCs arrive at receivers. Each valid "
        "MPC contributes one count; the distribution is not power weighted."
    ),
    "aoa_el_polar_chart": (
        "Elevation directions from which retained MPCs arrive at receivers. Each valid "
        "MPC contributes one count; the distribution is not power weighted."
    ),
    "mpc_evolution_chart": "Total number of retained MPCs in each scenario frame.",
    "delay_spread_trend_chart": (
        "Power-weighted RMS delay spread across all paths in each frame that have both "
        "valid delay and path loss. This is not an average of per-pair delay spreads."
    ),
    "mpc_order_evolution_hist_chart": (
        "Each active line is the exact MPC count for one interaction order in every "
        "frame. The symmetric-log vertical scale keeps rare orders visible beside "
        "dominant ones."
    ),
    "mpc_type_evolution_chart": (
        "Each retained MPC is counted once per frame. Direct paths are LoS; other paths "
        "are classified by their first interaction. All-zero categories are hidden."
    ),
    "pair_visibility_chart": (
        "Every represented TX/RX pair in each frame is classified as direct-path "
        "present, indirect-only, or no retained path. This is a channel path state, "
        "not renderer visibility."
    ),
    "strongest_path_loss_chart": (
        "Minimum valid single-path loss among all retained MPCs in each frame. Lower "
        "values indicate a stronger path."
    ),
}


def _bucket_reflection_orders(data: Dict[Any, Any]) -> dict[int, int]:
    """Return orders 0 through 5 plus one combined ``6+`` bucket."""
    result = {order: 0 for order in range(7)}
    for raw_order, raw_count in data.items():
        try:
            order = int(raw_order)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if order < 0:
            continue
        result[min(order, 6)] += count
    return result


def _interaction_type_color(interaction_type: int) -> str:
    """Return the canonical display color for one interaction type."""
    style = mpc_interaction_style(interaction_type)
    color = style.fixed_color
    if color is None and style.interaction_type is not None:
        color = INTERACTION_TYPE_COLORS.get(style.interaction_type, MPC_UNKNOWN_COLOR)
    return rgb_to_css_hex(color or MPC_UNKNOWN_COLOR)


def _constant_pair_state_message(
    frame_indices: Any,
    category_data: Dict[str, Any],
) -> Optional[str]:
    """Summarize pair-state counts when every frame has the same counts."""
    frames = np.asarray(frame_indices).reshape(-1)
    if frames.size == 0:
        return None

    categories = ("direct_path_present", "indirect_only", "no_path")
    series: list[np.ndarray] = []
    for category in categories:
        values = np.asarray(category_data.get(category, [])).reshape(-1)
        if values.size != frames.size or not np.all(values == values[0]):
            return None
        series.append(values)

    direct, indirect, no_path = (int(values[0]) for values in series)
    total = direct + indirect + no_path
    frame_count = int(frames.size)
    pair_word = "pair" if total == 1 else "pairs"
    frame_word = "frame" if frame_count == 1 else "frames"
    if total == 0:
        return f"No represented TX/RX pairs in any of the {frame_count} {frame_word}."
    if direct == total:
        return (
            f"Every frame has {total} represented TX/RX {pair_word}, and all have "
            f"a direct path ({frame_count} {frame_word})."
        )
    if indirect == total:
        return (
            f"Every frame has {total} represented TX/RX {pair_word}, and all are "
            f"indirect-only ({frame_count} {frame_word})."
        )
    if no_path == total:
        return (
            f"Every frame has {total} represented TX/RX {pair_word}, and none have "
            f"a retained path ({frame_count} {frame_word})."
        )
    return (
        f"All {frame_count} {frame_word} have the same path-state counts: "
        f"{direct} direct, {indirect} indirect-only, and {no_path} no-path "
        f"({total} represented {pair_word} per frame)."
    )


def _unique_export_path(path: Path) -> Path:
    """Return a non-existing path by appending a numeric suffix when needed."""
    if not path.exists():
        return path

    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"Could not find an unused export path for {path}")


def _log_timing(event: str, **fields: Any) -> None:
    """Emit startup/perf timings through the normal logger at DEBUG level."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if fields:
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.debug("[TIMING] %s %s", event, details)
        return
    logger.debug("[TIMING] %s", event)


class _StatsWorkerSignals(QObject):
    """Qt signals for delivering background statistics results to the main thread."""

    provider_finished = Signal(object, bool, int)
    provider_progress = Signal(int, int, int)
    provider_failed = Signal(str, int)


class _GraphVisibilityObserver(QObject):
    """Refresh expanded charts only after their tab becomes visible."""

    def __init__(self, section, callback):
        super().__init__(section)
        self.section = section
        self.callback = callback
        self.window = None
        section.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in {QEvent.Show, QEvent.Hide, QEvent.WindowStateChange}:
            window = self.section.window()
            if window is not self.window:
                self.window = window
                window.installEventFilter(self)
            QTimer.singleShot(0, self.callback)
        return False


class StatisticsPanel(BasePanel):
    """Create statistics widgets and coordinate cached/background computation."""

    def __init__(self, parent_widget: Any) -> None:
        """Initialize cache, graph, and background-worker state."""
        super().__init__(parent_widget)
        self.has_pyqtgraph = False
        self._plot_deps_checked = False
        self.stats = None
        self._distribution_group = None
        self._cdf_group = None
        self._evolution_group = None
        self._channel_evolution_group = None
        self._coverage_graphs_group = None
        self._coverage_graphs_status: Optional[QLabel] = None
        self._stats_signals = _StatsWorkerSignals()
        self._stats_signals.provider_finished.connect(self._on_provider_stats_ready)
        self._stats_signals.provider_progress.connect(self._on_provider_stats_progress)
        self._stats_signals.provider_failed.connect(self._on_provider_stats_error)
        self._statistics_provider: Optional[Any] = None
        self._provider_stats_generation: Optional[int] = None
        self._automatic_statistics = QSettings().value(
            "analysis/automaticScenarioStatistics", False, type=bool
        )
        self._cancel_stats_btn = None
        self._stats_error_section = None
        self._stats_error_text = None
        self._compute_btn: Optional[QPushButton] = None
        self._compute_status: Optional[QLabel] = None
        self._progress_bar: Optional[QProgressBar] = None
        self._export_csv_btn: Optional[QPushButton] = None
        self._export_charts_btn: Optional[QPushButton] = None
        self._polar_group = None
        self._graphs_section = None
        self._graphs_status_label: Optional[QLabel] = None
        self._graphs_panel_created = False
        self._graphs_visible = False
        self._graphs_dirty = True
        self._graphs_rendered = False
        self._coverage_graphs_dirty = True
        self._coverage_graphs_rendered = False
        self._graphs_status_clear_timer: Optional[QTimer] = None
        self._theme_manager = get_theme_manager()
        self._theme_changed_callback = self._on_theme_changed
        self._theme_manager.theme_changed.connect(self._theme_changed_callback)
        self._metrics_section = None
        self._metrics_host = None
        self._graph_tabs = None
        self._graphs_grid_visible = False
        self._graphs_grid_cb = None
        self._graphs_root = None
        self._panel_root = None
        self._chart_specs = {}
        self._whole_run_stats = None
        self._scope_bar = None
        self._selection_pair = None
        self._selection_steps = None
        self._delay_time_group = None
        self._root_widgets: dict[str, tuple[set[str], set[str]]] = {}

    def _watch_root(self, owner: str, root: QWidget, previous_keys: set[str]) -> None:
        """Associate each independently owned root with its widget references."""
        attributes = {
            name
            for name, value in vars(self).items()
            if isinstance(value, QWidget) and (value is root or root.isAncestorOf(value))
        }
        self._root_widgets[owner] = (set(self.widgets) - previous_keys, attributes)
        root.destroyed.connect(partial(self._on_root_destroyed, owner))

    def _on_root_destroyed(self, owner: str, _object: object = None) -> None:
        """Release one view while preserving its live sibling and cached results."""
        keys, attributes = self._root_widgets.pop(owner, (set(), set()))
        for key in keys:
            self.widgets.pop(key, None)
        for name in attributes:
            setattr(self, name, None)
        if owner == "panel":
            metrics_service = getattr(self.parent, "metrics_service", None)
            if self._metrics_host is not None and metrics_service is not None:
                metrics_service.close()
                self._metrics_host = None
        if owner == "graphs":
            self._graphs_panel_created = False
            self._graphs_visible = False
            self._chart_groups = ()
            self._graphs_section = None
            if self._graphs_status_clear_timer is not None:
                self._graphs_status_clear_timer.stop()
                self._graphs_status_clear_timer.deleteLater()
                self._graphs_status_clear_timer = None
        if not self._root_widgets:
            self.cleanup()

    def _on_theme_changed(self, _theme: object) -> None:
        """Recolor existing charts without rescanning the run or changing zoom."""
        from ..app.plot_theme import path_stats_style

        for root in (self._panel_root, self._graphs_root):
            if root is not None:
                root.setStyleSheet(path_stats_style(current_theme()))
        for widget in self.widgets.values():
            if hasattr(widget, "apply_theme"):
                widget.apply_theme()

    def cleanup(self) -> None:
        """Stop transient UI work and release the process-wide theme subscription."""
        metrics_service = getattr(self.parent, "metrics_service", None)
        if self._metrics_host is not None and metrics_service is not None:
            metrics_service.close()
            self._metrics_host = None
        statistics_service = self._get_scenario_statistics_service()
        if statistics_service is not None and self._statistics_provider is not None:
            statistics_service.cancel_collection()
        self._statistics_provider = None
        self._provider_stats_generation = None
        self._graphs_section = None
        if self._graphs_status_clear_timer is not None:
            self._graphs_status_clear_timer.stop()
        if self._theme_manager is not None:
            try:
                self._theme_manager.theme_changed.disconnect(self._theme_changed_callback)
            except (RuntimeError, TypeError):
                pass
            self._theme_manager = None

    def _ensure_plot_dependencies(self) -> bool:
        """Check the interactive backend; Matplotlib is loaded only for PNG export."""
        if self._plot_deps_checked:
            return self.has_pyqtgraph
        self._plot_deps_checked = True
        try:
            import pyqtgraph  # noqa: F401

            self.has_pyqtgraph = True
        except ImportError as exc:
            logger.warning("Statistics panel: pyqtgraph not available: %s", exc)
            self.missing_dependency_message = (
                "Interactive path statistics require pyqtgraph. "
                "Install with: pip install pyqtgraph"
            )
        return self.has_pyqtgraph

    @staticmethod
    def _has_values(values: Any) -> bool:
        """Return True when a sequence-like object contains at least one item."""
        if values is None:
            return False
        try:
            return len(values) > 0
        except TypeError:
            return False

    def _coverage_data(self) -> Optional[dict[str, Any]]:
        """Return active coverage metadata when a selected slice is loaded."""
        coverage_data = getattr(self.parent, "coverage_data", None)
        return coverage_data if isinstance(coverage_data, dict) else None

    def create_panel(self) -> QGroupBox:
        """Create and return the statistics panel."""
        previous_keys = set(self.widgets)
        group = self.create_group_box("")
        group.setObjectName("PathStatistics")
        group.setAttribute(Qt.WA_StyledBackground, True)
        from ..app.plot_theme import path_stats_style

        self._panel_root = group
        group.setStyleSheet(path_stats_style(current_theme()))
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignTop)

        self._metrics_section = CollapsibleSection("Live frame metrics", start_open=False)
        self._metrics_section.toggled.connect(self._on_metrics_toggled)
        layout.addWidget(self._metrics_section)
        layout.addWidget(self._create_compute_section())

        # Summary section
        layout.addWidget(self._create_summary_section())

        # Export buttons
        layout.addWidget(self._create_export_section())

        self._watch_root("panel", group, previous_keys)
        return group

    def _on_metrics_toggled(self, expanded: bool) -> None:
        """Allocate plots only after the user first expands live metrics."""
        if not expanded or self._metrics_host is not None:
            return
        service = getattr(self.parent, "metrics_service", None)
        if service is None or not service.available:
            layout = self._metrics_section.content_layout()
            if layout.count() == 0:
                layout.addWidget(QLabel("Install pyqtgraph to enable live frame metrics."))
            return
        from ..widgets.dashboard_host import DashboardHost

        self._metrics_host = DashboardHost(
            title="ORCHAV MPC Analysis Dashboard",
            settings_key="visualizer/metrics_dashboard_geometry",
        )
        self._metrics_section.content_layout().addWidget(self._metrics_host)
        self._metrics_host.returned.connect(self.show_metrics)
        service.attach_host(self._metrics_host)

    def show_metrics(self) -> None:
        """Reveal the existing analysis when it returns from a pop-out."""
        manager = getattr(self.parent, "ui_manager", None)
        if manager is not None:
            manager.show_panel("statistics")
        self._metrics_section.expand()

    def toggle_metrics(self) -> None:
        """Use the Metrics menu action to expand, collapse, or return its content."""
        if self._metrics_host is not None and self._metrics_host.is_detached():
            self._metrics_host.dock()
        elif self._metrics_section.is_expanded() and self._metrics_section.isVisible():
            self._metrics_section.collapse()
        else:
            self.show_metrics()

    def _create_missing_dependency_message(self) -> QGroupBox:
        """Create message when the interactive plotting dependency is missing."""
        group = self.create_group_box("Statistics")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        warning_label = QLabel("Warning")
        warning_label.setStyleSheet("font-size: 24px; color: #e67e22;")
        warning_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_label)

        message = QLabel(self.missing_dependency_message)
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("""
            color: #e67e22;
            padding: 10px;
            font-size: 11px;
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
        """)
        layout.addWidget(message)

        # Installation instruction
        install_label = QLabel("To enable statistics visualizations, run:\npip install pyqtgraph")
        install_label.setWordWrap(True)
        install_label.setAlignment(Qt.AlignCenter)
        install_label.setStyleSheet("font-size: 10px; padding: 5px;")
        layout.addWidget(install_label)

        return group

    def create_graphs_panel(self) -> QGroupBox:
        """Create a whole-run workspace with lazy, responsive chart families."""
        if self._graphs_root is not None:
            return self._graphs_root
        if not self._ensure_plot_dependencies():
            return self._create_missing_dependency_message()
        from ..app.plot_theme import path_stats_style

        previous_keys = set(self.widgets)
        group = self.create_group_box("")
        group.setObjectName("PathStatisticsGraphs")
        group.setStyleSheet(path_stats_style(current_theme()))
        group.setAttribute(Qt.WA_StyledBackground, True)
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._graphs_root = group
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 12, 16, 12)
        eyebrow = QLabel("PATH STATISTICS  /  RUN ANALYSIS")
        eyebrow.setObjectName("pathEyebrow")
        layout.addWidget(eyebrow)
        title = QLabel("Path statistics")
        title.setObjectName("pathTitle")
        layout.addWidget(title)
        self._graphs_scope = QLabel(
            "All retained scenario frames · independent of the playback cursor"
        )
        self._graphs_scope.setObjectName("pathScope")
        self._graphs_scope.setWordWrap(True)
        layout.addWidget(self._graphs_scope)
        status = QLabel("Open a scenario and compute statistics to explore its paths")
        status.setObjectName("pathScope")
        status.setWordWrap(True)
        self._graphs_status_label = status
        layout.addWidget(status)
        from .statistics_scope import StatisticsScopeBar

        self._scope_bar = StatisticsScopeBar()
        self._scope_bar.selection_requested.connect(self._request_statistics_scope)
        self._scope_bar.reset_requested.connect(self._reset_statistics_scope)
        layout.addWidget(self._scope_bar)
        if self._whole_run_stats is not None:
            self._scope_bar.bind(
                self._whole_run_stats, enabled=self._statistics_provider is not None
            )
        self._graphs_grid_cb = QCheckBox("Grid")
        self._graphs_grid_cb.setChecked(self._graphs_grid_visible)
        self._graphs_grid_cb.setToolTip(
            "Show grid lines on the run-statistics figures and their PNG exports"
        )
        self._graphs_grid_cb.toggled.connect(self._on_graphs_grid_toggled)
        layout.addWidget(self._graphs_grid_cb, alignment=Qt.AlignLeft)
        self._graph_tabs = QTabWidget()
        self._graph_tabs.setDocumentMode(True)
        self._graph_tabs.setMinimumHeight(560)
        layout.addWidget(self._graph_tabs, 1)
        self._distribution_group = self._create_distribution_charts_section()
        self._cdf_group = self._create_cdf_charts_section()
        self._polar_group = self._create_polar_charts_section()
        self._evolution_group = self._create_evolution_charts_section()
        self._channel_evolution_group = self._create_channel_evolution_section()
        self._coverage_graphs_group = self._create_coverage_charts_section()
        self._delay_time_group = self._chart_family(("delay_time_chart",))
        self.widgets["delay_time_chart"].frame_requested.connect(self._inspect_statistics_frame)
        self._chart_groups = (
            self._distribution_group,
            self._cdf_group,
            self._polar_group,
            self._evolution_group,
            self._channel_evolution_group,
            self._coverage_graphs_group,
            self._delay_time_group,
        )
        for name, page in zip(
            ("Distributions", "CDFs", "Angles", "Evolution", "Channels", "Coverage", "Delay–time"),
            self._chart_groups,
        ):
            scroll = QScrollArea()
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self._graph_tabs.addTab(scroll, name)
        hint = QLabel("Scroll to zoom · drag to pan · double-click to fit · hover for exact values")
        hint.setObjectName("pathScope")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._graphs_panel_created = True
        self._graphs_dirty = self._coverage_graphs_dirty = True
        if (self.stats or self._coverage_data()) and self._graphs_visible:
            self._render_statistics_graphs()
        self._watch_root("graphs", group, previous_keys)
        return group

    def _on_graphs_grid_toggled(self, visible: bool) -> None:
        """Update display guides without recomputing or invalidating statistics."""
        self._graphs_grid_visible = bool(visible)
        for widget in self.widgets.values():
            set_grid = getattr(widget, "set_grid_visible", None)
            if callable(set_grid):
                set_grid(self._graphs_grid_visible)

    def _sync_chart_tabs(self) -> None:
        """Disable unavailable families without leaving an empty selected page."""
        if self._graph_tabs is None:
            return
        for index, group in enumerate(self._chart_groups):
            self._graph_tabs.setTabEnabled(index, not group.isHidden())
        if not self._graph_tabs.isTabEnabled(self._graph_tabs.currentIndex()):
            for index in range(self._graph_tabs.count()):
                if self._graph_tabs.isTabEnabled(index):
                    self._graph_tabs.setCurrentIndex(index)
                    break

    def bind_graphs_section(self, section: Any) -> None:
        """Track Graphs section visibility so chart rendering can be lazy."""
        self._graphs_section = section
        self._graphs_visible = bool(section.is_expanded() and section.isVisible())
        self._graph_visibility_observer = _GraphVisibilityObserver(
            section, self._sync_graph_visibility
        )
        if hasattr(section, "toggled"):
            section.toggled.connect(self._on_graphs_section_toggled)
        if (
            self._graphs_visible
            and (self.stats or self._coverage_data())
            and (self._graphs_dirty or self._coverage_graphs_dirty)
        ):
            self._render_statistics_graphs()

    def _set_graphs_status(self, text: str, *, transient: bool = False) -> None:
        """Set the graph status label and optionally clear it after a short delay."""
        if self._graphs_status_label is None:
            return

        self._graphs_status_label.setText(text)
        if self._graphs_status_clear_timer is not None:
            self._graphs_status_clear_timer.stop()
        if not transient:
            return

        if self._graphs_status_clear_timer is None:
            self._graphs_status_clear_timer = QTimer(self._graphs_root)
            self._graphs_status_clear_timer.setSingleShot(True)
            self._graphs_status_clear_timer.timeout.connect(self._clear_transient_graphs_status)
        self._graphs_status_clear_timer.start(_GRAPHS_STATUS_CLEAR_MS)

    def _clear_transient_graphs_status(self) -> None:
        """Clear the transient success message when it is still current."""
        if self._graphs_status_label is None:
            return
        text_attr = getattr(self._graphs_status_label, "text", "")
        current_text = text_attr() if callable(text_attr) else text_attr
        if current_text == "Graphs updated.":
            self._graphs_status_label.setText("")

    @staticmethod
    def _graphs_render_status(*, has_angle_data: bool, multi_frame: bool) -> tuple[str, bool]:
        """Return graph-render status text and whether it should be transient."""
        hidden_notes = []
        if not has_angle_data:
            hidden_notes.append("angular charts: no angular data")
        if not multi_frame:
            hidden_notes.append("evolution charts: single frame")
        if hidden_notes:
            return f"Graphs updated. Hidden: {'; '.join(hidden_notes)}.", False
        return "Graphs updated.", True

    def _mark_graphs_dirty(self) -> None:
        """Mark graph canvases stale and refresh immediately when visible."""
        self._graphs_dirty = True
        self._coverage_graphs_dirty = True
        self._set_graphs_status("Graphs need refresh.")
        if self._graphs_visible and self._graphs_panel_created:
            self._render_statistics_graphs()

    def coverage_selection_changed(self, *, render: bool = True) -> None:
        """Invalidate figures derived from the active coverage selection."""
        self._coverage_graphs_dirty = True
        if self._export_charts_btn is not None:
            self._export_charts_btn.setEnabled(bool(self.stats or self._coverage_data()))
        if not render and self._coverage_graphs_status is not None:
            self._coverage_graphs_status.setText(
                "Height playback is active; coverage figures update when playback stops."
            )
        if render and self._graphs_visible and self._graphs_panel_created:
            self._render_coverage_graphs()

    def _on_graphs_section_toggled(self, checked: bool) -> None:
        """Render charts lazily when the Graphs section is opened."""
        self._sync_graph_visibility()

    def _sync_graph_visibility(self) -> None:
        section = self._graphs_section
        self._graphs_visible = bool(
            section is not None
            and section.is_expanded()
            and section.isVisible()
            and not section.window().isMinimized()
        )
        if (
            self._graphs_visible
            and (self.stats or self._coverage_data())
            and (self._graphs_dirty or self._coverage_graphs_dirty)
        ):
            self._render_statistics_graphs()
        if self._graphs_visible:
            for widget in self.widgets.values():
                if hasattr(widget, "render_pending") and widget.isVisible():
                    widget.render_pending()

    def _create_compute_section(self) -> QWidget:
        """Create the compute-statistics button and progress bar."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(6)

        self._compute_btn = QPushButton("Compute Statistics")
        self._compute_btn.setToolTip("Compute or refresh statistics from the active frame source")
        self._compute_btn.setEnabled(False)
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        row.addWidget(self._compute_btn)
        self._cancel_stats_btn = QPushButton("Cancel")
        self._cancel_stats_btn.setEnabled(False)
        self._cancel_stats_btn.clicked.connect(self._cancel_statistics)
        row.addWidget(self._cancel_stats_btn)

        self._compute_status = QLabel("No frames loaded")
        self._compute_status.setStyleSheet("font-size: 10px;")
        self._compute_status.setWordWrap(True)
        row.addWidget(self._compute_status, stretch=1)

        layout.addLayout(row)
        scope = QLabel(
            "Whole scenario · all recorded frames and links · display filters do not apply"
        )
        scope.setWordWrap(True)
        scope.setProperty("role", "secondary")
        layout.addWidget(scope)
        automatic = QCheckBox("Compute automatically on cache miss")
        automatic.setChecked(self._automatic_statistics)
        automatic.setToolTip(
            "Scans all recorded frames in the background. Leave off to avoid competing with playback for CPU and disk access."
        )
        automatic.toggled.connect(self._set_automatic_statistics)
        layout.addWidget(automatic)
        self._stats_error_section = CollapsibleSection("Error details", start_open=False)
        self._stats_error_text = QPlainTextEdit()
        self._stats_error_text.setReadOnly(True)
        self._stats_error_text.setMaximumHeight(100)
        self._stats_error_section.content_layout().addWidget(self._stats_error_text)
        copy = QPushButton("Copy details")
        copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self._stats_error_text.toPlainText())
        )
        self._stats_error_section.content_layout().addWidget(copy)
        self._stats_error_section.hide()
        layout.addWidget(self._stats_error_section)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumHeight(12)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m frames")
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        return widget

    def _set_automatic_statistics(self, enabled: bool) -> None:
        self._automatic_statistics = bool(enabled)
        QSettings().setValue("analysis/automaticScenarioStatistics", bool(enabled))
        if enabled and self.stats is None:
            self._start_provider_statistics(force=False)
        elif not enabled:
            self._cancel_statistics()

    def _cancel_statistics(self) -> None:
        service = self._get_scenario_statistics_service()
        if service is not None:
            service.cancel_collection()
        self._provider_stats_generation = None
        self._finish_statistics_job("Paused. Compute to refresh; existing results are retained.")

    def _finish_statistics_job(self, message: str) -> None:
        if self._cancel_stats_btn is not None:
            self._cancel_stats_btn.setEnabled(False)
        if self._compute_btn is not None:
            self._compute_btn.setEnabled(self._statistics_provider is not None)
        if self._progress_bar is not None:
            self._progress_bar.hide()
        if self._compute_status is not None:
            self._compute_status.setText(message)
            self._compute_status.setStyleSheet("")

    def set_statistics_source(self, source: Any, *, start_loading: bool = True) -> bool:
        """Bind a provider and start cache-first selective statistics loading.

        The source can be a shared ``DataProvider`` or a visualizer frame source
        exposing one as ``source.provider``. Providers without native selective
        I/O retain correct behavior through the shared full-frame projection
        fallback. ``None`` retires the source. Hidden panels can bind without
        starting a cache lookup, while remaining ready for manual computation.
        """

        provider = resolve_statistics_provider(source)
        if provider is None and source is not None:
            logger.warning("Statistics source has no shared frame provider")

        statistics_service = self._get_scenario_statistics_service()
        if statistics_service is None and provider is not None:
            logger.warning("Scenario statistics service is unavailable")
            provider = None
        if statistics_service is not None:
            statistics_service.cancel_collection()

        self._statistics_provider = provider
        self._provider_stats_generation = None
        if self._stats_error_section is not None:
            self._stats_error_section.hide()
        if self._stats_error_text is not None:
            self._stats_error_text.clear()
        if self._compute_btn is not None:
            self._compute_btn.setText("Compute Statistics")
        self._whole_run_stats = None
        self._selection_pair = self._selection_steps = None
        if self._scope_bar is not None:
            self._scope_bar.bind({}, enabled=False)
        self.stats = None
        self._chart_specs.clear()
        for widget in self.widgets.values():
            if isinstance(widget, QLabel):
                widget.setText(EMPTY_VALUE)
        for group in (
            self._distribution_group,
            self._cdf_group,
            self._polar_group,
            self._evolution_group,
            self._channel_evolution_group,
        ):
            if group is not None:
                group.hide()
        if self._delay_time_group is not None:
            self._delay_time_group.hide()
        self._graphs_dirty = True
        self._graphs_rendered = False
        self._set_graphs_status("Compute statistics to populate these graphs.")
        if self._compute_btn is not None:
            self._compute_btn.setEnabled(False)
        if self._compute_status is not None:
            self._compute_status.setText("Checking statistics cache...")
            self._compute_status.setStyleSheet("color: #2196F3; font-size: 10px;")
        if self._progress_bar is not None:
            self._progress_bar.setValue(0)
            self._progress_bar.hide()
        if self._export_csv_btn is not None:
            self._export_csv_btn.setEnabled(False)
        if self._export_charts_btn is not None:
            self._export_charts_btn.setEnabled(False)

        if provider is None:
            self._set_graphs_status("Open a recording to compute statistics.")
            self._finish_statistics_job("No statistics source.")
            return False
        if start_loading:
            self._start_provider_statistics(force=False)
        else:
            self._finish_statistics_job("Compute statistics to analyze this recording.")
        return True

    def _request_statistics_scope(self, pair, steps) -> None:
        """Collect an explicitly requested subset without replacing the run cache."""
        if pair is None and steps is None:
            self._reset_statistics_scope()
            return
        self._selection_pair, self._selection_steps = pair, steps
        self._set_graphs_status(
            "Preparing selected link/interval; charts still show the previous scope."
        )
        self._start_provider_statistics(force=False)

    def _reset_statistics_scope(self) -> None:
        service = self._get_scenario_statistics_service()
        if service is not None:
            service.cancel_collection()
        self._provider_stats_generation = None
        self._selection_pair = self._selection_steps = None
        if self._whole_run_stats is not None:
            self._finish_statistics_job("")
            self._apply_stats_result(
                self._whole_run_stats, status_text="Whole-run statistics restored"
            )

    def _inspect_statistics_frame(self, step: int) -> None:
        update = getattr(self.parent, "update_frame", None)
        if not callable(update):
            return
        controller = getattr(self.parent, "animation_controller", None)
        if getattr(self.parent, "animation_running", False) and controller is not None:
            controller.toggle_animation()
        cancel_scrub = getattr(self.parent, "cancel_pending_slider_scrub", None)
        if callable(cancel_scrub):
            cancel_scrub()
        update(step)

    def _start_provider_statistics(self, *, force: bool) -> None:
        """Start the cache-first provider job through the application service."""

        provider = self._statistics_provider
        statistics_service = self._get_scenario_statistics_service()
        if provider is None or statistics_service is None:
            return

        if self._compute_btn is not None:
            self._compute_btn.setEnabled(False)
        if self._compute_status is not None:
            self._compute_status.setText(
                "Computing statistics..." if force else "Checking statistics cache..."
            )
            self._compute_status.setStyleSheet("color: #2196F3; font-size: 10px;")
        if self._progress_bar is not None:
            self._progress_bar.setValue(0)
            self._progress_bar.hide()

        if self._cancel_stats_btn is not None:
            self._cancel_stats_btn.setEnabled(True)
        if self._stats_error_section is not None:
            self._stats_error_section.hide()
        if self._compute_btn is not None:
            self._compute_btn.setText("Compute Statistics")
        signals = self._stats_signals

        def _complete(result: ScenarioStatisticsResult) -> None:
            signals.provider_finished.emit(
                result.stats,
                result.from_cache,
                result.generation,
            )

        def _progress(current: int, total: int) -> None:
            signals.provider_progress.emit(
                current,
                total,
                statistics_service.current_generation,
            )

        def _error(exc: Exception) -> None:
            signals.provider_failed.emit(
                str(exc),
                statistics_service.current_generation,
            )

        scoped = self._selection_pair is not None or self._selection_steps is not None
        selection = {"pair": self._selection_pair, "steps": self._selection_steps} if scoped else {}
        self._provider_stats_generation = statistics_service.start_collection(
            provider,
            scenario=getattr(self.parent, "scenario", None),
            force=force,
            cache_only=not (force or self._automatic_statistics or scoped),
            **selection,
            on_progress=_progress,
            on_complete=_complete,
            on_error=_error,
        )

    def _on_compute_clicked(self) -> None:
        """Handle click on the Compute Statistics button."""
        if self._statistics_provider is None:
            return
        self._start_provider_statistics(force=True)

    def _create_export_section(self) -> QWidget:
        """Create the CSV / chart export buttons."""
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._export_csv_btn = QPushButton("Export CSV")
        self._export_csv_btn.setToolTip("Export summary statistics to a CSV file")
        self._export_csv_btn.setEnabled(False)
        self._export_csv_btn.clicked.connect(self._on_export_csv)
        row.addWidget(self._export_csv_btn)

        self._export_charts_btn = QPushButton("Export Charts")
        self._export_charts_btn.setToolTip("Save all chart figures as PNG images")
        self._export_charts_btn.setEnabled(False)
        self._export_charts_btn.clicked.connect(self._on_export_charts)
        row.addWidget(self._export_charts_btn)

        compare_btn = QPushButton("Compare links…")
        compare_btn.setToolTip("Compare named TX/RX power traces and export matched observations")
        compare_btn.clicked.connect(self._open_link_comparison)
        row.addWidget(compare_btn)
        row.addStretch()
        return widget

    def _open_link_comparison(self) -> None:
        """Load the optional comparison UI only on explicit user request."""
        from .link_comparison import LinkComparisonDialog

        dialog = getattr(self, "_link_comparison_dialog", None)
        if dialog is None:
            dialog = LinkComparisonDialog(self.parent)
            self._link_comparison_dialog = dialog
        dialog.show()
        dialog.raise_()

    def _default_export_dir(self) -> str:
        """Return the default directory for exports (scenario_dir/statistics/)."""
        viz = self.parent
        scenario_path = getattr(viz, "scenario_path", None)
        if scenario_path:
            base = Path(scenario_path).parent
        else:
            base = Path.cwd()
        return str(base / "statistics")

    _NO_STATS_MSG = (
        "Statistics have not been computed. Use the 'Compute Statistics' "
        "button to compute them before exporting."
    )

    def _on_export_csv(self) -> None:
        """Export summary statistics to a CSV file."""
        if not self.stats:
            QMessageBox.warning(None, "No Statistics", self._NO_STATS_MSG)
            return

        default_dir = self._default_export_dir()
        os.makedirs(default_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Export Statistics CSV",
            str(Path(default_dir) / "statistics.csv"),
            "CSV Files (*.csv)",
        )
        if not path:
            return

        rows: list[tuple[str, str]] = [
            (
                "Scope",
                "Selected link/interval" if self.stats.get("selection_scoped") else "Whole run",
            ),
            ("TX-RX selection", str(self.stats.get("selection_pair") or "All links")),
            ("Total MPCs", str(self.stats.get("total_mpcs", ""))),
            ("Total Frames", str(self.stats.get("total_frames", ""))),
            ("Unique TX-RX Pairs", str(self.stats.get("unique_tx_rx_pairs", ""))),
        ]

        total_frames = self.stats.get("total_frames", 0)
        if total_frames > 0:
            frames = self.stats.get("frame_indices", [])
            if len(frames):
                rows.extend(
                    (
                        ("First retained frame", str(frames[0])),
                        ("Last retained frame", str(frames[-1])),
                    )
                )
            ratio = self.stats["total_mpcs"] / total_frames
            rows.append(("MPCs per Frame", f"{ratio:.0f}"))

        ds = self.stats.get("overall_delay_spread")
        if ds is not None:
            rows.append(("Pooled Delay Spread RMS (ns)", f"{ds:.2f}"))

        pl = self.stats.get("path_loss_stats")
        if pl:
            rows.append(("Path Loss Min (dB)", f"{pl['min']:.1f}"))
            rows.append(("Path Loss Max (dB)", f"{pl['max']:.1f}"))

        cv = self.stats.get("mpc_count_variation_coeff")
        if cv is not None:
            rows.append(("MPC Count CV", f"{cv:.3f}"))

        pair_gain = self.stats.get("pair_aggregate_path_gain_stats")
        if pair_gain:
            rows.extend(
                (
                    ("Pair Aggregate Path Gain P10 (dB)", f"{pair_gain['p10']:.2f}"),
                    ("Pair Aggregate Path Gain Median (dB)", f"{pair_gain['median']:.2f}"),
                    ("Pair Aggregate Path Gain P90 (dB)", f"{pair_gain['p90']:.2f}"),
                )
            )

        pair_delay_spread = self.stats.get("pair_rms_delay_spread_stats")
        if pair_delay_spread:
            rows.extend(
                (
                    ("Pair RMS Delay Spread P10 (ns)", f"{pair_delay_spread['p10']:.2f}"),
                    (
                        "Pair RMS Delay Spread Median (ns)",
                        f"{pair_delay_spread['median']:.2f}",
                    ),
                    ("Pair RMS Delay Spread P90 (ns)", f"{pair_delay_spread['p90']:.2f}"),
                )
            )

        pair_visibility = self.stats.get("pair_visibility_summary", {})
        pair_visibility_labels = {
            "direct_path_present": "Direct Path Present Pair-Frames",
            "indirect_only": "Indirect-Only Pair-Frames",
            "no_path": "No-Path Pair-Frames",
        }
        for category, label in pair_visibility_labels.items():
            values = pair_visibility.get(category)
            if values:
                rows.append((label, str(values["count"])))
                rows.append((f"{label} (%)", f"{values['percent']:.2f}"))

        # Interaction-order distribution.
        rod = _bucket_reflection_orders(self.stats.get("reflection_order_dist", {}))
        for order, count in rod.items():
            if count == 0:
                continue
            label = f"Interaction Order {order}" if order < 6 else "Interaction Order 6+"
            rows.append((label, str(count)))

        # Initial propagation mechanism distribution.
        mtd = self.stats.get("mpc_type_dist", {})
        for raw_type in sorted(mtd, key=lambda value: mpc_interaction_sort_key(int(value))):
            interaction_type = int(raw_type)
            rows.append(
                (
                    f"Initial Propagation Mechanism {mpc_interaction_label(interaction_type)}",
                    str(mtd[raw_type]),
                )
            )

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerows(rows)
            QMessageBox.information(None, "Export Complete", f"Statistics saved to:\n{path}")
            logger.info("Statistics CSV exported to %s", path)
        except OSError as exc:
            QMessageBox.warning(None, "Export Failed", f"Could not write file:\n{exc}")
            logger.error("CSV export failed: %s", exc)

    def _on_export_charts(self) -> None:
        """Export publication PNGs without allocating hidden interactive plots."""
        if not self.stats and self._coverage_data() is None:
            QMessageBox.warning(None, "No Statistics", self._NO_STATS_MSG)
            return
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
        except ImportError:
            QMessageBox.warning(
                None,
                "PNG export unavailable",
                "PNG export requires matplotlib. Install with: pip install matplotlib",
            )
            return
        default_dir = self._default_export_dir()
        os.makedirs(default_dir, exist_ok=True)
        dir_path = QFileDialog.getExistingDirectory(
            None, "Select Directory for Chart Export", default_dir
        )
        if not dir_path:
            return
        if not self._graphs_panel_created:
            self.create_graphs_panel()
        if self._graphs_dirty or self._coverage_graphs_dirty or not self._graphs_rendered:
            self._render_statistics_graphs(force=True)
        saved = 0
        for key, spec in self._chart_specs.items():
            try:
                figure = self._create_publication_figure(**spec)
                if figure is None:
                    continue
                out_path = _unique_export_path(Path(dir_path) / f"{key}.png")
                figure.savefig(str(out_path), dpi=150, bbox_inches="tight")
                figure.clear()
                saved += 1
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                logger.error("Failed to save chart %s: %s", key, exc)
        if saved:
            QMessageBox.information(
                None, "Export Complete", f"Saved {saved} chart(s) to:\n{dir_path}"
            )
            logger.info("Exported %d charts to %s", saved, dir_path)
        else:
            QMessageBox.warning(None, "Export", "No charts available to export.")

    def _create_summary_section(self) -> QGroupBox:
        """Readable primary cards with the complete pair-channel details below."""
        group = self.create_subgroup_box("Whole-run summary")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(10)
        primary = QGridLayout()
        primary.setSpacing(8)
        definitions = (
            ("total_mpcs", "RETAINED PATHS", "Across all recorded frames"),
            ("total_frames", "FRAMES", "Whole retained scenario"),
            ("unique_pairs", "TX/RX PAIRS", "Unique represented pairs"),
            ("overall_delay_spread", "POOLED RMS DELAY", "Power weighted · co-valid paths"),
        )
        for index, (key, title, caption) in enumerate(definitions):
            card = QFrame()
            card.setObjectName("pathMetricCard")
            box = QVBoxLayout(card)
            box.setContentsMargins(10, 8, 10, 8)
            box.setSpacing(2)
            name = QLabel(title)
            name.setObjectName("pathMetricName")
            name.setWordWrap(True)
            value = QLabel(EMPTY_VALUE)
            value.setToolTip(f"{caption}. N/A: no usable data for this metric.")
            value.setObjectName("pathMetricValue")
            value.setWordWrap(True)
            self.widgets[key] = value
            note = QLabel(caption)
            note.setObjectName("pathMetricCaption")
            note.setWordWrap(True)
            box.addWidget(name)
            box.addWidget(value)
            box.addWidget(note)
            primary.addWidget(card, index // 2, index % 2)
        primary.setColumnStretch(0, 1)
        primary.setColumnStretch(1, 1)
        layout.addLayout(primary)
        details = QGridLayout()
        details.setHorizontalSpacing(12)
        details.setVerticalSpacing(6)
        rows = (
            ("mpc_frame_ratio", "Paths per frame", "Mean retained path count across all frames"),
            (
                "path_loss_range",
                "Path loss range",
                "Minimum and maximum valid single-path loss, in dB",
            ),
            (
                "mpc_count_cv",
                "Path count CV",
                "Coefficient of variation of the per-frame retained path count",
            ),
            (
                "pair_gain_percentiles",
                "Pair gain P50 [P10, P90]",
                "Aggregate path gain across represented TX/RX pair-frames with valid path loss",
            ),
            (
                "pair_delay_spread_percentiles",
                "Pair RMS delay P50 [P10, P90]",
                "Power-weighted RMS delay spread for represented pair-frames with co-valid delay and path loss",
            ),
            (
                "pair_visibility_summary",
                "Pair-frame states",
                "Every represented pair-frame is direct-path present, indirect-only, or no retained path",
            ),
        )
        for row, (key, title, help_text) in enumerate(rows):
            name = QLabel(title)
            name.setObjectName("pathMetricCaption")
            name.setWordWrap(True)
            name.setToolTip(help_text)
            value = QLabel(EMPTY_VALUE)
            value.setWordWrap(True)
            value.setToolTip(help_text)
            self.widgets[key] = value
            details.addWidget(name, row, 0)
            details.addWidget(value, row, 1)
        details.setColumnStretch(0, 2)
        details.setColumnStretch(1, 3)
        layout.addLayout(details)
        return group

    def _chart_family(self, keys) -> QWidget:
        from .statistics_plot import ResponsiveChartGrid

        cards = []
        for key in keys:
            self.widgets[key] = self._create_chart_widget(key)
            cards.append(self.widgets[key])
        return ResponsiveChartGrid(cards)

    def _create_distribution_charts_section(self) -> QWidget:
        return self._chart_family(
            (
                "reflection_order_chart",
                "interaction_type_chart",
                "path_loss_histogram_chart",
                "delay_histogram_chart",
            )
        )

    def _create_cdf_charts_section(self) -> QWidget:
        return self._chart_family(
            (
                "delay_cdf_chart",
                "path_loss_cdf_chart",
                "pair_gain_cdf_chart",
                "pair_delay_spread_cdf_chart",
            )
        )

    def _create_polar_charts_section(self) -> QWidget:
        return self._chart_family(
            ("aod_az_polar_chart", "aod_el_polar_chart", "aoa_az_polar_chart", "aoa_el_polar_chart")
        )

    def _create_evolution_charts_section(self) -> QWidget:
        return self._chart_family(
            (
                "mpc_evolution_chart",
                "delay_spread_trend_chart",
                "mpc_order_evolution_hist_chart",
                "mpc_type_evolution_chart",
            )
        )

    def _create_channel_evolution_section(self) -> QWidget:
        return self._chart_family(("pair_visibility_chart", "strongest_path_loss_chart"))

    def _create_coverage_charts_section(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 4, 0, 0)
        self._coverage_graphs_status = QLabel(
            "Coverage figures appear when a coverage map is loaded."
        )
        self._coverage_graphs_status.setObjectName("pathScope")
        self._coverage_graphs_status.setWordWrap(True)
        layout.addWidget(self._coverage_graphs_status)
        layout.addWidget(
            self._chart_family(("coverage_distribution_chart", "coverage_success_chart")), 1
        )
        group.hide()
        return group

    def _create_chart_widget(self, graph_key: str) -> QWidget:
        from .statistics_plot import StatisticsPlotCard

        card = StatisticsPlotCard(
            graph_key,
            _GRAPH_EXPLANATIONS[graph_key],
            self._format_chart_tooltip,
            self._find_nearest_chart_point,
        )
        card.set_grid_visible(self._graphs_grid_visible)
        return card

    def _create_chart(
        self, widget: QWidget, chart_type: str, data: Dict[Any, Any], title: str = ""
    ) -> None:
        """Queue scientific data on a persistent card, rendering only when visible."""
        spec = {"chart_type": chart_type, "data": data, "title": title}
        self._chart_specs[widget.key] = spec
        message = None
        if chart_type == "pair_visibility_evolution":
            message = _constant_pair_state_message(
                data.get("frame_indices", []), data.get("category_data", {})
            )
        widget.set_chart(chart_type, data, title, message)

    @staticmethod
    def _apply_matplotlib_theme(fig: Any, ax: "Axes") -> None:
        """Apply the current Qt application theme to a matplotlib figure."""
        apply_matplotlib_theme(fig, ax)

    @staticmethod
    def _apply_matplotlib_legend_theme(ax: "Axes") -> None:
        """Apply the current Qt application theme to a matplotlib legend."""
        apply_matplotlib_legend_theme(ax)

    def _format_chart_tooltip(
        self, x_val: float, y_val: float, chart_type: str, data: Dict[Any, Any]
    ) -> str:
        """Format hover tooltip text for a chart point."""
        if chart_type == "reflection_order":
            all_orders = list(range(0, 7))
            bucketed = _bucket_reflection_orders(data)
            counts = [bucketed[order] for order in all_orders]
            if isinstance(x_val, (int, float)) and 0 <= int(x_val) < len(all_orders):
                order_idx = int(x_val)
                order_label = str(all_orders[order_idx]) if all_orders[order_idx] < 6 else "6+"
                return f"Interaction order: {order_label}\nCount: {counts[order_idx]:,}"
            return f"X: {x_val:.2f}, Y: {y_val:.2f}"

        if chart_type == "interaction_type":
            interaction_types = sorted(
                (int(value) for value in data),
                key=mpc_interaction_sort_key,
            )
            index = int(round(y_val))
            if 0 <= index < len(interaction_types):
                interaction_type = interaction_types[index]
                count = int(data.get(interaction_type, data.get(str(interaction_type), 0)))
                return (
                    f"Initial mechanism: {mpc_interaction_label(interaction_type)}\n"
                    f"Count: {count:,}"
                )

        if chart_type == "line_evolution":
            value_label = str(data.get("ylabel", "Value"))
            return f"Frame: {int(x_val)}\n{value_label}: {y_val:.2f}"

        if chart_type in ("mpc_order_evolution", "mpc_type_evolution"):
            frame_indices = np.asarray(data.get("frame_indices", [])).reshape(-1)
            nearest = np.flatnonzero(frame_indices == int(x_val))
            if nearest.size:
                index = int(nearest[0])
                lines = [f"Frame: {int(x_val)}"]
                if chart_type == "mpc_order_evolution":
                    for raw_order, raw_counts in sorted(
                        data.get("order_data", {}).items(),
                        key=lambda item: int(item[0]),
                    ):
                        counts = np.asarray(raw_counts).reshape(-1)
                        if counts.size != frame_indices.size or not np.any(counts):
                            continue
                        order = int(raw_order)
                        label = str(order) if order < 6 else "6+"
                        lines.append(f"Order {label}: {int(counts[index]):,}")
                else:
                    active_types = sorted(
                        data.get("type_data", {}).items(),
                        key=lambda item: mpc_interaction_sort_key(int(item[0])),
                    )
                    for raw_type, raw_counts in active_types:
                        counts = np.asarray(raw_counts).reshape(-1)
                        if counts.size != frame_indices.size or not np.any(counts):
                            continue
                        lines.append(
                            f"{mpc_interaction_label(int(raw_type))}: " f"{int(counts[index]):,}"
                        )
                return "\n".join(lines)

        if chart_type == "pair_visibility_evolution":
            frame_indices = np.asarray(data.get("frame_indices", []))
            nearest = np.flatnonzero(frame_indices == int(x_val))
            if nearest.size:
                index = int(nearest[0])
                series = data.get("category_data", {})
                categories = (
                    series.get("direct_path_present", []),
                    series.get("indirect_only", []),
                    series.get("no_path", []),
                )
                if all(index < len(values) for values in categories):
                    direct, indirect, no_path = (int(values[index]) for values in categories)
                    return (
                        f"Frame: {int(x_val)}\n"
                        f"Direct path present: {direct}\n"
                        f"Indirect only: {indirect}\n"
                        f"No path: {no_path}"
                    )

        if chart_type in ("histogram", "angle_histogram"):
            bin_edges = data.get("_hist_edges")
            hist = data.get("_hist_counts")
            if bin_edges is not None and hist is not None and len(bin_edges) > 1:
                centers = (np.asarray(bin_edges[:-1]) + np.asarray(bin_edges[1:])) / 2
                nearest_idx = int(np.argmin(np.abs(centers - x_val)))
                if 0 <= nearest_idx < len(hist):
                    return (
                        f"Bin: {bin_edges[nearest_idx]:.2f} - "
                        f"{bin_edges[nearest_idx + 1]:.2f}\n"
                        f"Center: {centers[nearest_idx]:.2f}\n"
                        f"Count: {int(hist[nearest_idx]):,}"
                    )
            return f"Value: {x_val:.2f}\nCount: {int(y_val):,}"

        if chart_type == "polar_rose":
            bin_edges = np.asarray(data.get("_polar_hist_edges", []), dtype=np.float64)
            counts = np.asarray(data.get("_polar_hist_counts", []), dtype=np.int64)
            if bin_edges.size == counts.size + 1 and counts.size:
                centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
                circular_distance = np.abs(np.angle(np.exp(1j * (centers - x_val))))
                index = int(np.argmin(circular_distance))
                lower = float(np.rad2deg(bin_edges[index]))
                upper = float(np.rad2deg(bin_edges[index + 1]))
                return (
                    f"Azimuth bin: {lower:.0f} deg to {upper:.0f} deg\n"
                    f"Count: {int(counts[index]):,}"
                )

        if chart_type == "coverage_histogram":
            bin_edges = np.asarray(data.get("_hist_edges", []), dtype=np.float64)
            percentages = np.asarray(data.get("_hist_percentages", []), dtype=np.float64)
            counts = np.asarray(data.get("_hist_counts", []), dtype=np.int64)
            if bin_edges.size == percentages.size + 1 and percentages.size:
                centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
                index = int(np.argmin(np.abs(centers - x_val)))
                return (
                    f"Bin: {bin_edges[index]:.3g} - {bin_edges[index + 1]:.3g}\n"
                    f"Valid area: {percentages[index]:.1f}%\n"
                    f"Cells: {counts[index]:,}"
                )

        if chart_type == "coverage_success_curve":
            comparator = str(data.get("comparator", ">="))
            total = int(data.get("total_cells", 0) or 0)
            valid = int(data.get("valid_cells", 0) or 0)
            return (
                f"Threshold: {x_val:.3g}\n"
                f"Area meeting value {comparator} threshold: {y_val:.1f}%\n"
                f"Usable cells: {valid:,}/{total:,}"
            )

        if chart_type == "coverage_serving_share":
            labels = data.get("labels", [])
            cells = data.get("cells", [])
            areas = data.get("areas_m2", [])
            index = int(round(y_val))
            if 0 <= index < len(labels):
                return (
                    f"{labels[index]}\n"
                    f"Total area: {x_val:.1f}%\n"
                    f"Cells: {int(cells[index]):,}\n"
                    f"Area: {float(areas[index]):,.2f} m²"
                )

        if chart_type == "cdf":
            total = int(data.get("_cdf_total", 0) or 0)
            count = min(total, max(0, int(round(float(y_val) * total)))) if total else 0
            xlabel = str(data.get("xlabel", "Value"))
            return (
                f"{xlabel}: {x_val:.2f}\n"
                f"CDF: {float(y_val) * 100.0:.1f}%\n"
                f"Samples <= value: {count:,}/{total:,}"
            )

        return f"X: {x_val:.2f}\nY: {y_val:.2f}"

    def _find_nearest_chart_point(
        self,
        x_data: Optional[float],
        y_data: Optional[float],
        chart_type: str,
        data: Dict[Any, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find the nearest data point to a chart-space mouse position."""
        if x_data is None or y_data is None:
            return None, None

        if chart_type == "reflection_order":
            all_orders = list(range(0, 7))
            bucketed = _bucket_reflection_orders(data)
            counts = [bucketed[order] for order in all_orders]
            distances = [abs(x_data - order) for order in all_orders]
            nearest_idx = min(range(len(distances)), key=distances.__getitem__)
            if distances[nearest_idx] < 0.5:
                return all_orders[nearest_idx], counts[nearest_idx]

        elif chart_type == "interaction_type":
            interaction_types = sorted(
                (int(value) for value in data),
                key=mpc_interaction_sort_key,
            )
            nearest_idx = int(round(y_data))
            if 0 <= nearest_idx < len(interaction_types) and abs(y_data - nearest_idx) <= 0.5:
                interaction_type = interaction_types[nearest_idx]
                count = float(data.get(interaction_type, data.get(str(interaction_type), 0)))
                return count, float(nearest_idx)

        elif chart_type == "line_evolution":
            x = np.asarray(data.get("x", []))
            y = np.asarray(data.get("y", []))
            if self._has_values(x) and self._has_values(y) and len(x) == len(y):
                finite = np.isfinite(x) & np.isfinite(y)
                x, y = x[finite], y[finite]
                if not x.size:
                    return None, None
                distances = [
                    np.sqrt((x_data - xi) ** 2 + (y_data - yi) ** 2) for xi, yi in zip(x, y)
                ]
                nearest_idx = min(range(len(distances)), key=distances.__getitem__)
                x_span = max(x) - min(x) if self._has_values(x) else 1
                if distances[nearest_idx] < 0.1 * x_span:
                    return x[nearest_idx], y[nearest_idx]

        elif chart_type in ("mpc_order_evolution", "mpc_type_evolution"):
            frame_indices = np.asarray(data.get("frame_indices", []), dtype=np.float64).reshape(-1)
            if frame_indices.size:
                nearest_idx = int(np.argmin(np.abs(frame_indices - x_data)))
                frame_span = float(np.ptp(frame_indices)) if frame_indices.size > 1 else 0.0
                if abs(frame_indices[nearest_idx] - x_data) <= max(1.0, 0.02 * frame_span):
                    return float(frame_indices[nearest_idx]), 0.0

        elif chart_type == "pair_visibility_evolution":
            frame_indices = data.get("frame_indices", [])
            if self._has_values(frame_indices):
                distances = [abs(x_data - frame) for frame in frame_indices]
                nearest_idx = min(range(len(distances)), key=distances.__getitem__)
                x_span = max(frame_indices) - min(frame_indices)
                if distances[nearest_idx] <= max(1.0, 0.02 * x_span):
                    return frame_indices[nearest_idx], 0.0

        elif chart_type in ("histogram", "angle_histogram"):
            values = data.get("values", [])
            if len(values) > 0:
                hist = data.get("_hist_counts")
                bin_edges = data.get("_hist_edges")
                if hist is None or bin_edges is None:
                    bins = data.get("bins", 30)
                    hist, bin_edges = np.histogram(values, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                distances = [abs(x_data - center) for center in bin_centers]
                nearest_idx = min(range(len(distances)), key=distances.__getitem__)
                if distances[nearest_idx] < (bin_edges[1] - bin_edges[0]):
                    return bin_centers[nearest_idx], hist[nearest_idx]

        elif chart_type == "polar_rose":
            bin_edges = np.asarray(data.get("_polar_hist_edges", []), dtype=np.float64)
            counts = np.asarray(data.get("_polar_hist_counts", []), dtype=np.int64)
            if bin_edges.size == counts.size + 1 and counts.size:
                angle = ((float(x_data) + np.pi) % (2.0 * np.pi)) - np.pi
                index = int(np.searchsorted(bin_edges, angle, side="right") - 1)
                index = min(max(index, 0), counts.size - 1)
                center = float(bin_edges[index] + (bin_edges[index + 1] - bin_edges[index]) / 2.0)
                return center, float(counts[index])

        elif chart_type == "coverage_histogram":
            bin_edges = np.asarray(data.get("_hist_edges", []), dtype=np.float64)
            percentages = np.asarray(data.get("_hist_percentages", []), dtype=np.float64)
            if bin_edges.size == percentages.size + 1 and percentages.size:
                centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
                nearest_idx = int(np.argmin(np.abs(centers - x_data)))
                if bin_edges[nearest_idx] <= x_data <= bin_edges[nearest_idx + 1]:
                    return float(centers[nearest_idx]), float(percentages[nearest_idx])

        elif chart_type == "coverage_success_curve":
            thresholds = np.asarray(data.get("_curve_x", []), dtype=np.float64)
            percentages = np.asarray(data.get("_curve_y", []), dtype=np.float64)
            if thresholds.size and thresholds.size == percentages.size:
                nearest_idx = int(np.argmin(np.abs(thresholds - x_data)))
                return float(thresholds[nearest_idx]), float(percentages[nearest_idx])

        elif chart_type == "coverage_serving_share":
            percentages = np.asarray(data.get("percentages", []), dtype=np.float64)
            if percentages.size:
                nearest_idx = int(round(y_data))
                if 0 <= nearest_idx < percentages.size and abs(y_data - nearest_idx) <= 0.5:
                    return float(percentages[nearest_idx]), float(nearest_idx)

        elif chart_type == "cdf":
            plot_vals = data.get("_cdf_x")
            cdf = data.get("_cdf_y")
            if plot_vals is None or cdf is None:
                values = np.asarray(data.get("values", []), dtype=np.float64)
                if values.size == 0:
                    return None, None
                plot_vals = np.sort(values)
                cdf = np.arange(1, len(plot_vals) + 1) / len(plot_vals)
            plot_vals = np.asarray(plot_vals, dtype=np.float64)
            cdf = np.asarray(cdf, dtype=np.float64)
            if plot_vals.size and plot_vals.size == cdf.size:
                nearest_idx = int(np.argmin(np.abs(plot_vals - x_data)))
                x_span = float(plot_vals[-1] - plot_vals[0]) if plot_vals.size > 1 else 1.0
                threshold = max(1e-9, 0.03 * max(abs(x_span), 1.0))
                if abs(plot_vals[nearest_idx] - x_data) <= threshold:
                    return float(plot_vals[nearest_idx]), float(cdf[nearest_idx])

        return None, None

    def _create_publication_figure(self, chart_type: str, data: Dict[Any, Any], title: str = ""):
        """Build a headless publication figure for PNG export only."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        data = dict(data)
        fig = Figure(figsize=(7.0, 4.2), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        self._apply_matplotlib_theme(fig, ax)
        if chart_type == "pair_visibility_evolution":
            message = _constant_pair_state_message(
                data.get("frame_indices", []), data.get("category_data", {})
            )
            if message is not None:
                ax.set_axis_off()
                ax.set_title(title)
                ax.text(
                    0.5, 0.5, message, ha="center", va="center", wrap=True, transform=ax.transAxes
                )
                return fig
        if chart_type == "delay_time":
            mesh = ax.pcolormesh(
                data["x_edges"],
                data["delay_edges"],
                np.asarray(data["power_db"]).T,
                shading="flat",
                cmap="viridis",
            )
            colorbar = fig.colorbar(mesh, ax=ax)
            colorbar.set_label("Incoherent path gain (dB)", color=current_theme().text_primary)
            colorbar.ax.tick_params(colors=current_theme().text_primary)
            ax.set_xlabel(data["xlabel"])
            ax.set_ylabel("Propagation delay (ns)")
            ax.set_title(title)
            ax.grid(False)
        elif chart_type == "reflection_order":
            # Show interaction orders 0-6+ even when a bucket is empty.
            all_orders = list(range(0, 7))  # 0 through 6
            bucketed = _bucket_reflection_orders(data)
            counts = [bucketed[order] for order in all_orders]
            bar_colors = [_CHART_ORDER_COLORS[order] for order in all_orders]

            ax.bar(all_orders, counts, color=bar_colors)
            ax.set_xlabel("Interaction Order")
            ax.set_ylabel("Count")
            ax.set_xticks(all_orders)
            ax.set_xticklabels([str(o) if o < 6 else "6+" for o in all_orders])
            if title:
                ax.set_title(title, fontsize=9)

        elif chart_type == "interaction_type":
            interaction_types = sorted(
                (int(value) for value in data),
                key=mpc_interaction_sort_key,
            )
            if interaction_types:
                counts = [data.get(value, data.get(str(value), 0)) for value in interaction_types]
                labels = [mpc_interaction_label(value) for value in interaction_types]
                colors = [_interaction_type_color(value) for value in interaction_types]
                ax.barh(range(len(interaction_types)), counts, color=colors)
                ax.set_yticks(range(len(interaction_types)))
                ax.set_yticklabels(labels)
                ax.set_xlabel("Count")
                ax.set_ylabel("Initial Mechanism", fontsize=8)
                if title:
                    ax.set_title(title, fontsize=9)

        elif chart_type == "line_evolution":
            # Line plot for evolution with a fixed chart color.
            x = data.get("x", [])
            y = data.get("y", [])
            if self._has_values(x) and self._has_values(y):
                ax.plot(x, y, color=_CHART_LINE_COLOR, linewidth=1.5)
                ax.set_xlabel("Frame")
                ax.set_ylabel(data.get("ylabel", "Value"))
                if title:
                    ax.set_title(title, fontsize=9)

        elif chart_type == "mpc_order_evolution":
            frame_indices = data.get("frame_indices", [])
            order_data = data.get("order_data", {})

            if self._has_values(frame_indices) and order_data:
                active_orders = []
                for raw_order, raw_counts in order_data.items():
                    counts = np.asarray(raw_counts, dtype=np.float64).reshape(-1)
                    if counts.size != len(frame_indices) or not np.any(counts):
                        continue
                    active_orders.append((int(raw_order), counts))
                for order, counts in sorted(active_orders):
                    color_index = min(max(order, 0), len(_CHART_ORDER_COLORS) - 1)
                    ax.plot(
                        frame_indices,
                        counts,
                        color=_CHART_ORDER_COLORS[color_index],
                        linewidth=1.5,
                        label=str(order) if order < 6 else "6+",
                    )
                if active_orders:
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("MPC Count")
                    ax.set_yscale("symlog", linthresh=1.0)
                    ax.set_ylim(bottom=0.0)
                    ax.legend(
                        loc="upper right",
                        fontsize=7,
                        framealpha=0.9,
                        ncol=len(active_orders),
                        handlelength=1.0,
                        handletextpad=0.3,
                        columnspacing=0.5,
                    )
                    if title:
                        ax.set_title(title, fontsize=9)

        elif chart_type == "mpc_type_evolution":
            frame_indices = data.get("frame_indices", [])
            type_data = data.get("type_data", {})

            if self._has_values(frame_indices) and type_data:
                active_types = []
                for raw_type, raw_counts in type_data.items():
                    counts = np.asarray(raw_counts, dtype=np.float64).reshape(-1)
                    if counts.size != len(frame_indices) or not np.any(counts):
                        continue
                    active_types.append((int(raw_type), counts))
                for interaction_type, counts in sorted(
                    active_types,
                    key=lambda item: mpc_interaction_sort_key(item[0]),
                ):
                    ax.plot(
                        frame_indices,
                        counts,
                        color=_interaction_type_color(interaction_type),
                        linewidth=1.5,
                        label=mpc_interaction_label(interaction_type),
                    )

                ax.set_xlabel("Frame")
                ax.set_ylabel("MPC Count")
                if active_types:
                    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
                if title:
                    ax.set_title(title, fontsize=9)

        elif chart_type == "pair_visibility_evolution":
            frame_indices = data.get("frame_indices", [])
            category_data = data.get("category_data", {})
            if self._has_values(frame_indices) and category_data:
                categories = (
                    "direct_path_present",
                    "indirect_only",
                    "no_path",
                )
                labels = ("Direct path present", "Indirect only", "No path")
                colors = ("#4CAF50", "#FF9800", "#9E9E9E")
                all_series = [
                    np.asarray(category_data.get(category, []), dtype=np.float64).reshape(-1)
                    for category in categories
                ]
                active = [
                    (label, color, values)
                    for label, color, values in zip(labels, colors, all_series)
                    if values.size == len(frame_indices) and np.any(values)
                ]
                if active and all(len(values) == len(frame_indices) for values in all_series):
                    ax.stackplot(
                        frame_indices,
                        *(values for _label, _color, values in active),
                        labels=[label for label, _color, _values in active],
                        colors=[color for _label, color, _values in active],
                        alpha=0.9,
                        linewidth=0.0,
                    )
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Represented TX/RX Pairs")
                    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
                    if title:
                        ax.set_title(title, fontsize=9)

        elif chart_type == "histogram":
            # Histogram for distribution visualization
            values = data.get("values", [])
            xlabel = data.get("xlabel", "Value")
            ylabel = data.get("ylabel", "Count")

            if len(values) > 0:
                values = np.asarray(values, dtype=np.float64)
                bins = int(data.get("bins", 30))
                hist, bin_edges = np.histogram(values, bins=bins)
                bin_widths = np.diff(bin_edges)
                bin_centers = bin_edges[:-1] + bin_widths / 2.0
                ax.bar(
                    bin_centers,
                    hist,
                    width=bin_widths * 0.9,
                    color="#2196F3",
                    edgecolor="#1565C0",
                    linewidth=0.4,
                )
                data["_hist_counts"] = hist
                data["_hist_edges"] = bin_edges
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if title:
                    ax.set_title(title, fontsize=9)

        elif chart_type == "coverage_histogram":
            values = np.asarray(data.get("values", []), dtype=np.float64)
            if values.size > 0:
                bins: Any = int(data.get("bins", 30))
                if data.get("color_scale") == "logarithmic":
                    minimum = float(np.min(values))
                    maximum = float(np.max(values))
                    if minimum == maximum:
                        minimum *= 0.9
                        maximum *= 1.1
                    bins = np.geomspace(minimum, maximum, int(data.get("bins", 30)) + 1)
                    ax.set_xscale("log")
                counts, bin_edges = np.histogram(values, bins=bins)
                percentages = 100.0 * counts / values.size
                bin_widths = np.diff(bin_edges)
                bin_centers = bin_edges[:-1] + bin_widths / 2.0
                ax.bar(
                    bin_centers,
                    percentages,
                    width=bin_widths * 0.9,
                    color="#2196F3",
                    edgecolor="#1565C0",
                    linewidth=0.4,
                )
                data["_hist_counts"] = counts
                data["_hist_percentages"] = percentages
                data["_hist_edges"] = bin_edges
                percentile_values = data.get("percentiles", ())
                percentile_colors = ("#8E24AA", "#FF5722", "#00897B")
                for label, value, color in zip(
                    ("P10", "P50", "P90"),
                    percentile_values,
                    percentile_colors,
                ):
                    if value is not None:
                        ax.axvline(
                            float(value),
                            color=color,
                            linestyle="--",
                            linewidth=1,
                            alpha=0.85,
                            label=f"{label} {float(value):.3g}",
                        )
                ax.set_xlabel(data.get("xlabel", "Coverage value"), fontsize=8)
                ax.set_ylabel("Share of valid area (%)", fontsize=8)
                ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
            if title:
                ax.set_title(title, fontsize=9)

        elif chart_type == "coverage_success_curve":
            thresholds = np.asarray(data.get("thresholds", []), dtype=np.float64)
            percentages = np.asarray(data.get("percentages", []), dtype=np.float64)
            if thresholds.size and thresholds.size == percentages.size:
                ax.plot(thresholds, percentages, linewidth=1.6, color="#2196F3")
                if data.get("color_scale") == "logarithmic":
                    ax.set_xscale("log")
                data["_curve_x"] = thresholds
                data["_curve_y"] = percentages
                comparator = data.get("comparator", ">=")
                ax.set_xlabel(data.get("xlabel", "Coverage threshold"), fontsize=8)
                ax.set_ylabel(f"Area meeting value {comparator} threshold (%)", fontsize=8)
                ax.set_ylim(0.0, 100.0)
                ax.grid(True, alpha=0.3)
            if title:
                ax.set_title(title, fontsize=9)

        elif chart_type == "coverage_serving_share":
            labels = [str(label) for label in data.get("labels", [])]
            percentages = np.asarray(data.get("percentages", []), dtype=np.float64)
            if labels and len(labels) == percentages.size:
                positions = np.arange(len(labels))
                ax.barh(
                    positions,
                    percentages,
                    color=data.get("colors", "#2196F3"),
                )
                ax.set_yticks(positions)
                ax.set_yticklabels(labels, fontsize=7)
                ax.set_xlabel("Share of total slice area (%)", fontsize=8)
                ax.set_xlim(0.0, max(100.0, float(np.max(percentages)) * 1.05))
                ax.invert_yaxis()
                data["_bar_positions"] = positions
            if title:
                ax.set_title(title, fontsize=9)

        elif chart_type == "cdf":
            # Cumulative distribution function
            values = np.asarray(data.get("values", []), dtype=np.float64)
            xlabel = data.get("xlabel", "Value")

            if values.size > 0:
                sorted_vals = np.sort(values)
                if sorted_vals.size > _MAX_CDF_POINTS:
                    sample_idx = np.linspace(
                        0,
                        sorted_vals.size - 1,
                        _MAX_CDF_POINTS,
                        dtype=np.intp,
                    )
                    plot_vals = sorted_vals[sample_idx]
                    cdf = (sample_idx + 1) / sorted_vals.size
                else:
                    plot_vals = sorted_vals
                    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                data["_cdf_x"] = plot_vals
                data["_cdf_y"] = cdf
                data["_cdf_total"] = int(sorted_vals.size)
                ax.plot(plot_vals, cdf, linewidth=1.5, color="#2196F3")
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel("CDF", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                median_val = np.median(sorted_vals)
                ax.axvline(
                    median_val,
                    color="#FF5722",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    label=f"Median: {median_val:.1f}",
                )
                ax.legend(fontsize=7, loc="lower right")
                if title:
                    ax.set_title(title, fontsize=9)

        elif chart_type == "polar_rose":
            # Polar rose plot for angular distributions
            # Replace the rectangular axes with a polar projection
            fig.clear()
            ax = fig.add_subplot(111, projection="polar")
            self._apply_matplotlib_theme(fig, ax)
            values_deg = np.asarray(data.get("values", []), dtype=np.float64)
            n_bins = data.get("n_bins", 36)

            if values_deg.size > 0:
                values_rad = np.deg2rad(values_deg)
                bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
                hist, _ = np.histogram(values_rad, bins=bin_edges)
                data["_polar_hist_counts"] = hist
                data["_polar_hist_edges"] = bin_edges
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]
                ax.bar(
                    bin_centers,
                    hist,
                    width=bin_width * 0.9,
                    color="#2196F3",
                    alpha=0.7,
                    edgecolor="#1565C0",
                    linewidth=0.5,
                )
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
                ax.set_thetagrids(
                    [0, 45, 90, 135, 180, 225, 270, 315],
                    labels=["0", "45", "90", "135", "180", "-135", "-90", "-45"],
                )
                ax.tick_params(labelsize=6)
            if title:
                ax.set_title(title, fontsize=9, pad=12)

        elif chart_type == "angle_histogram":
            values = np.asarray(data.get("values", []), dtype=np.float64)
            xlabel = data.get("xlabel", "Angle (deg)")
            xlim = data.get("xlim")
            if values.size > 0:
                bins = int(data.get("bins", 30))
                hist_range = tuple(xlim) if xlim is not None else None
                hist, bin_edges = np.histogram(values, bins=bins, range=hist_range)
                bin_widths = np.diff(bin_edges)
                bin_centers = bin_edges[:-1] + bin_widths / 2.0
                ax.bar(
                    bin_centers,
                    hist,
                    width=bin_widths * 0.9,
                    color="#2196F3",
                    edgecolor="#1565C0",
                    linewidth=0.4,
                )
                data["_hist_counts"] = hist
                data["_hist_edges"] = bin_edges
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                if xlim is not None:
                    ax.set_xlim(*xlim)
                ax.axvline(0.0, color="#FF5722", linestyle="--", linewidth=1, alpha=0.7)
                if title:
                    ax.set_title(title, fontsize=9)

        self._apply_matplotlib_theme(fig, ax)
        self._apply_matplotlib_legend_theme(ax)
        ax.set_axisbelow(chart_type != "delay_time")
        if self._graphs_grid_visible:
            ax.grid(True, alpha=0.25)
        else:
            ax.grid(False)

        return fig

    def _on_provider_stats_progress(
        self,
        current: int,
        total: int,
        generation: int,
    ) -> None:
        """Apply progress only for the provider generation still bound here."""

        if generation != self._provider_stats_generation:
            return
        if self._progress_bar is not None:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(current)
            self._progress_bar.show()

    def _on_provider_stats_ready(
        self,
        stats: Dict[str, Any],
        from_cache: bool,
        generation: int,
    ) -> None:
        """Apply a generation-guarded result emitted by the provider service."""

        if generation != self._provider_stats_generation:
            return
        if stats is None:
            self._finish_statistics_job(
                "No cached statistics. Select Compute to scan the recorded frames."
            )
            return
        self._finish_statistics_job("")
        self._apply_stats_result(
            stats,
            status_text="Using cached statistics" if from_cache else "Done",
        )

    def _on_provider_stats_error(self, message: str, generation: int) -> None:
        """Expose provider/validation failures without mutating prior results."""

        if generation != self._provider_stats_generation:
            return
        logger.error("Scenario statistics collection failed: %s", message)
        self._finish_statistics_job("Statistics failed. Check the source, then retry.")
        if self._stats_error_section is not None:
            self._stats_error_text.setPlainText(message)
            self._stats_error_section.show()
        if self._compute_btn is not None:
            self._compute_btn.setText("Retry Statistics")
        if self._progress_bar is not None:
            self._progress_bar.hide()
        if self._compute_btn is not None:
            self._compute_btn.setEnabled(self._statistics_provider is not None)
        if self._compute_status is not None:
            self._compute_status.setText("Statistics failed. Check Error details, then retry.")
            self._compute_status.setStyleSheet("color: #F44336; font-size: 10px;")

    def _apply_stats_result(self, stats: Dict[str, Any], *, status_text: str) -> None:
        """Apply a fully prepared statistics payload to the UI."""
        t0 = time.perf_counter()
        _log_timing("StatisticsPanel._apply_stats_result.start", status=status_text)
        self.stats = stats
        if not stats.get("selection_scoped", False):
            self._whole_run_stats = stats
            if self._scope_bar is not None:
                self._scope_bar.bind(stats, enabled=self._statistics_provider is not None)
        logger.info("Statistics ready; updating UI")
        self._update_statistics()
        if not stats.get("selection_scoped", False):
            self._push_stats_to_trajectory_panel(stats)
        if self._progress_bar is not None:
            self._progress_bar.hide()
        if self._compute_btn is not None:
            self._compute_btn.setEnabled(True)
        if self._compute_status is not None:
            self._compute_status.setText(status_text)
            self._compute_status.setStyleSheet("color: #4CAF50; font-size: 10px;")
        if self._export_csv_btn is not None:
            self._export_csv_btn.setEnabled(True)
        if self._export_charts_btn is not None:
            self._export_charts_btn.setEnabled(True)
        _log_timing(
            "StatisticsPanel._apply_stats_result.end",
            elapsed_ms=f"{(time.perf_counter() - t0) * 1000:.1f}",
        )

    def _push_stats_to_trajectory_panel(self, stats: Dict[str, Any]) -> None:
        """Forward per-frame KPI data to the trajectory preview panel for coloring."""
        if not hasattr(self.parent, "ui_manager"):
            return
        panels = getattr(self.parent.ui_manager, "panels", {})
        trajectory_panel = panels.get("trajectory")
        if trajectory_panel is not None and hasattr(trajectory_panel, "set_per_frame_stats"):
            trajectory_panel.set_per_frame_stats(stats)

    def _get_scenario_statistics_service(self):
        """Return the provider-driven statistics service if available."""

        return getattr(self.parent, "scenario_statistics_service", None)

    def _update_statistics(self) -> None:
        """Update UI with collected statistics."""
        t0 = time.perf_counter()
        _log_timing("StatisticsPanel._update_statistics.start")
        if not self.stats or not self.widgets:
            _log_timing("StatisticsPanel._update_statistics.skipped")
            return

        if "total_mpcs" not in self.widgets:
            # A detached graph view can continue receiving results after the
            # independent summary/control root has been destroyed.
            self._graphs_dirty = True
            if self._graphs_visible:
                self._render_statistics_graphs()
            return

        # Hide evolution sections when there is only a single frame
        multi_frame = self.stats.get("total_frames", 0) > 1
        if self._evolution_group is not None:
            self._evolution_group.setVisible(multi_frame)
        if self._channel_evolution_group is not None:
            self._channel_evolution_group.setVisible(multi_frame)

        # ── Summary labels ──────────────────────────────────────────────
        self.widgets["total_mpcs"].setText(f"{self.stats['total_mpcs']:,}")
        self.widgets["total_frames"].setText(str(self.stats["total_frames"]))
        self.widgets["unique_pairs"].setText(str(self.stats.get("unique_tx_rx_pairs", 0)))

        if self.stats["total_frames"] > 0:
            ratio = self.stats["total_mpcs"] / self.stats["total_frames"]
            self.widgets["mpc_frame_ratio"].setText(f"{ratio:,.0f}")
        else:
            self.widgets["mpc_frame_ratio"].setText(EMPTY_VALUE)

        overall_delay_spread = self.stats.get("overall_delay_spread")
        if overall_delay_spread is not None and np.isfinite(overall_delay_spread):
            self.widgets["overall_delay_spread"].setText(f"{overall_delay_spread:.2f} ns")
        else:
            self.widgets["overall_delay_spread"].setText(EMPTY_VALUE)

        path_loss_stats = self.stats.get("path_loss_stats")
        if path_loss_stats:
            self.widgets["path_loss_range"].setText(
                f"{path_loss_stats['min']:.1f} - {path_loss_stats['max']:.1f} dB"
            )
        else:
            self.widgets["path_loss_range"].setText(EMPTY_VALUE)

        mpc_count_cv = self.stats.get("mpc_count_variation_coeff")
        if mpc_count_cv is not None and multi_frame:
            self.widgets["mpc_count_cv"].setText(f"{mpc_count_cv:.3f}")
        else:
            self.widgets["mpc_count_cv"].setText(EMPTY_VALUE)

        pair_gain = self.stats.get("pair_aggregate_path_gain_stats")
        if pair_gain:
            self.widgets["pair_gain_percentiles"].setText(
                f"{pair_gain['median']:.1f} [{pair_gain['p10']:.1f}, "
                f"{pair_gain['p90']:.1f}] dB (n={pair_gain['count']:,})"
            )
        else:
            self.widgets["pair_gain_percentiles"].setText(EMPTY_VALUE)

        pair_spread = self.stats.get("pair_rms_delay_spread_stats")
        if pair_spread:
            self.widgets["pair_delay_spread_percentiles"].setText(
                f"{pair_spread['median']:.1f} [{pair_spread['p10']:.1f}, "
                f"{pair_spread['p90']:.1f}] ns (n={pair_spread['count']:,})"
            )
        else:
            self.widgets["pair_delay_spread_percentiles"].setText(EMPTY_VALUE)

        pair_visibility = self.stats.get("pair_visibility_summary", {})
        if pair_visibility:
            direct = pair_visibility["direct_path_present"]
            indirect = pair_visibility["indirect_only"]
            no_path = pair_visibility["no_path"]
            self.widgets["pair_visibility_summary"].setText(
                f"Direct {direct['count']:,} ({direct['percent']:.1f}%)  |  "
                f"Indirect {indirect['count']:,} ({indirect['percent']:.1f}%)  |  "
                f"No path {no_path['count']:,} ({no_path['percent']:.1f}%)"
            )
        else:
            self.widgets["pair_visibility_summary"].setText(EMPTY_VALUE)

        self._graphs_dirty = True
        if self._graphs_visible:
            self._set_graphs_status("Rendering graphs...")
        else:
            self._set_graphs_status("Open this section to render graphs.")

        if self._graphs_visible and self._graphs_panel_created:
            self._render_statistics_graphs()

        _log_timing(
            "StatisticsPanel._update_statistics.summary_end",
            elapsed_ms=f"{(time.perf_counter() - t0) * 1000:.1f}",
        )
        return

    def _coverage_height_label(self, coverage_data: dict[str, Any]) -> str:
        """Return a compact label for the selected logical coverage height."""
        try:
            height_index = int(getattr(self.parent, "coverage_height_index", 0))
        except (TypeError, ValueError):
            height_index = 0
        heights = getattr(self.parent, "coverage_heights", None)
        if heights is None:
            heights = coverage_data.get("heights", [])
        try:
            if len(heights) > 0:
                index = max(0, min(height_index, len(heights) - 1))
                return f"{float(heights[index]):.2f} m"
        except (TypeError, ValueError):
            pass
        return f"height {height_index + 1}"

    def _render_coverage_graphs(self, *, force: bool = False) -> None:
        """Render figures for the already loaded coverage metric and height."""
        if not self._coverage_graphs_dirty and self._coverage_graphs_rendered:
            return
        if not self._graphs_panel_created or self._coverage_graphs_group is None:
            self._coverage_graphs_dirty = True
            return
        if not force and not self._graphs_visible:
            self._coverage_graphs_dirty = True
            return

        coverage_data = self._coverage_data()
        if coverage_data is None:
            self._coverage_graphs_group.setVisible(False)
            self._chart_specs.pop("coverage_distribution_chart", None)
            self._chart_specs.pop("coverage_success_chart", None)
            self._sync_chart_tabs()
            self._coverage_graphs_dirty = False
            self._coverage_graphs_rendered = False
            return
        if not self._ensure_plot_dependencies():
            self._coverage_graphs_status.setText(self.missing_dependency_message)
            return

        metric_name = str(coverage_data.get("metric_name", "coverage"))
        try:
            height_index = int(getattr(self.parent, "coverage_height_index", 0))
        except (TypeError, ValueError):
            height_index = 0
        height_label = self._coverage_height_label(coverage_data)
        distribution_host = self.widgets.get("coverage_distribution_chart")
        success_host = self.widgets.get("coverage_success_chart")
        if distribution_host is None or success_host is None:
            return

        try:
            if is_serving_tx_metric(metric_name):
                summary = compute_serving_tx_coverage_summary(
                    coverage_data,
                    height_index=height_index,
                )
                labels = [*summary.tx_names, "No service"]
                cells = [*summary.served_cells, summary.no_service_cells]
                percentages = [
                    *(summary.served_percent(index) for index in range(len(summary.tx_names))),
                    summary.no_service_percent,
                ]
                areas = [
                    *(summary.served_area_m2(index) for index in range(len(summary.tx_names))),
                    summary.no_service_area_m2,
                ]
                colors = [
                    *(serving_tx_color_hex(index) for index in range(len(summary.tx_names))),
                    "#7f7f7f",
                ]
                distribution_host.setVisible(True)
                success_host.setVisible(False)
                self._chart_specs.pop("coverage_success_chart", None)
                self._create_chart(
                    distribution_host,
                    "coverage_serving_share",
                    {
                        "labels": labels,
                        "percentages": percentages,
                        "cells": cells,
                        "areas_m2": areas,
                        "colors": colors,
                    },
                    f"Serving-Transmitter Area Share — {height_label}",
                )
                status = (
                    f"Raw selected slice · Serving TX · {height_label} · "
                    f"No service {summary.no_service_percent:.1f}% of total area"
                )
            else:
                plot_data = compute_coverage_scalar_plot_data(
                    coverage_data,
                    height_index=height_index,
                    max_curve_points=_MAX_CDF_POINTS,
                )
                summary = plot_data.summary
                unit_suffix = f" ({summary.unit})" if summary.unit else ""
                xlabel = f"{summary.label}{unit_suffix}"
                distribution_host.setVisible(True)
                success_host.setVisible(True)
                self._create_chart(
                    distribution_host,
                    "coverage_histogram",
                    {
                        "values": plot_data.valid_values,
                        "xlabel": xlabel,
                        "color_scale": plot_data.color_scale,
                        "percentiles": (
                            summary.percentile_10,
                            summary.percentile_50,
                            summary.percentile_90,
                        ),
                    },
                    f"Raw {summary.label} Distribution — {height_label}",
                )
                self._create_chart(
                    success_host,
                    "coverage_success_curve",
                    {
                        "thresholds": plot_data.thresholds,
                        "percentages": plot_data.qualifying_percent_total,
                        "xlabel": xlabel,
                        "comparator": plot_data.comparator,
                        "color_scale": plot_data.color_scale,
                        "total_cells": summary.total_cells,
                        "valid_cells": summary.valid_cells,
                    },
                    f"Raw {summary.label} Coverage Probability — {height_label}",
                )
                status = (
                    f"Raw selected slice · {summary.label} · {height_label} · "
                    f"Valid {summary.valid_percent:.1f}% · "
                    f"No data {summary.no_data_percent:.1f}% of total area"
                )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Could not render selected coverage figures: %s", exc)
            self._coverage_graphs_group.setVisible(True)
            self._coverage_graphs_status.setText("Coverage figures are unavailable for this slice.")
            self._coverage_graphs_dirty = False
            self._coverage_graphs_rendered = False
            return

        self._coverage_graphs_status.setText(status)
        self._coverage_graphs_group.setVisible(True)
        self._sync_chart_tabs()
        self._coverage_graphs_dirty = False
        self._coverage_graphs_rendered = True

    def _render_statistics_graphs(self, *, force: bool = False) -> None:
        """Render all graph canvases, deferring work while Graphs is collapsed."""
        t0 = time.perf_counter()
        coverage_data = self._coverage_data()
        if (not self.stats and coverage_data is None) or not self.widgets:
            return
        if not self._ensure_plot_dependencies():
            self._set_graphs_status(self.missing_dependency_message)
            return
        if not self._graphs_panel_created:
            self._graphs_dirty = True
            return
        if not force and not self._graphs_visible:
            self._graphs_dirty = True
            self._coverage_graphs_dirty = True
            return

        if not self.stats:
            self._chart_specs = {
                key: spec for key, spec in self._chart_specs.items() if key.startswith("coverage_")
            }
            if self._delay_time_group is not None:
                self._delay_time_group.hide()
            for group in (
                self._distribution_group,
                self._cdf_group,
                self._polar_group,
                self._evolution_group,
                self._channel_evolution_group,
            ):
                if group is not None:
                    group.setVisible(False)
            self._render_coverage_graphs(force=True)
            self._graphs_dirty = False
            self._graphs_rendered = self._coverage_graphs_rendered
            self._sync_chart_tabs()
            self._set_graphs_status("Coverage graphs updated.", transient=True)
            return

        if not self._graphs_dirty and self._graphs_rendered:
            self._render_coverage_graphs(force=force)
            self._sync_chart_tabs()
            return
        self._chart_specs = {
            key: spec for key, spec in self._chart_specs.items() if key.startswith("coverage_")
        }
        for key, widget in self.widgets.items():
            if (
                key.endswith("_chart")
                and not key.startswith("coverage_")
                and hasattr(widget, "clear_chart")
            ):
                widget.clear_chart()
        if hasattr(self, "_graphs_scope"):
            pair = self.stats.get("selection_pair")
            scope = f"Tx {pair[0]} → Rx {pair[1]}" if pair is not None else "all links"
            if self.stats.get("selection_scoped"):
                frames = self.stats.get("frame_indices", [])
                scope += (
                    f" · frames {frames[0]}–{frames[-1]}" if len(frames) else " · empty interval"
                )
            else:
                scope += " · whole run"
            self._graphs_scope.setText(
                f"{int(self.stats.get('total_frames', 0)):,} retained frames · "
                f"{int(self.stats.get('total_mpcs', 0)):,} paths · "
                f"{scope}, independent of playback"
            )
        heatmap = np.asarray(self.stats.get("delay_time_power_db", []))
        has_heatmap = heatmap.ndim == 2 and np.any(np.isfinite(heatmap))
        if self._delay_time_group is not None:
            self._delay_time_group.setVisible(bool(has_heatmap))
        if has_heatmap:
            from ..metrics.statistics_analysis import statistics_coordinates

            coordinates, axis = statistics_coordinates(
                self.stats["frame_indices"], self.stats.get("frame_times_s", [])
            )
            self._create_chart(
                self.widgets["delay_time_chart"],
                "delay_time",
                {
                    "power_db": heatmap,
                    "x_edges": self.stats["delay_time_x_edges"],
                    "delay_edges": self.stats["delay_time_delay_edges_ns"],
                    "observation_counts": self.stats["delay_time_observation_counts"],
                    "coordinates": coordinates,
                    "frame_indices": self.stats["frame_indices"],
                    "xlabel": axis,
                },
                "Recorded Path Power over Time and Delay",
            )
        if self._distribution_group is not None:
            self._distribution_group.setVisible(True)
        if self._cdf_group is not None:
            self._cdf_group.setVisible(True)

        # Hide evolution sections when there is only a single frame
        multi_frame = self.stats.get("total_frames", 0) > 1
        if self._evolution_group is not None:
            self._evolution_group.setVisible(multi_frame)
        if self._channel_evolution_group is not None:
            self._channel_evolution_group.setVisible(multi_frame)

        # ── Distribution charts ─────────────────────────────────────────
        reflection_order_dist = self.stats.get("reflection_order_dist", {})
        if reflection_order_dist and "reflection_order_chart" in self.widgets:
            t_chart = time.perf_counter()
            self._create_chart(
                self.widgets["reflection_order_chart"],
                "reflection_order",
                reflection_order_dist,
                "Interaction Order Distribution",
            )
            _log_timing(
                "StatisticsPanel._update_statistics.chart",
                chart="reflection_order",
                elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
            )

        mpc_type_dist = self.stats.get("mpc_type_dist", {})
        if mpc_type_dist and "interaction_type_chart" in self.widgets:
            type_data = {
                int(interaction_type): count
                for interaction_type, count in mpc_type_dist.items()
                if count > 0
            }
            if type_data:
                t_chart = time.perf_counter()
                self._create_chart(
                    self.widgets["interaction_type_chart"],
                    "interaction_type",
                    type_data,
                    "MPCs by Initial Propagation Mechanism",
                )
                _log_timing(
                    "StatisticsPanel._update_statistics.chart",
                    chart="interaction_type",
                    elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
                )

        # Path loss histogram
        path_loss_values = self.stats.get("path_loss_values", [])
        if len(path_loss_values) > 0 and "path_loss_histogram_chart" in self.widgets:
            t_chart = time.perf_counter()
            self._create_chart(
                self.widgets["path_loss_histogram_chart"],
                "histogram",
                {"values": path_loss_values, "xlabel": "Path Loss (dB)", "ylabel": "Count"},
                "Path Loss Distribution",
            )
            _log_timing(
                "StatisticsPanel._update_statistics.chart",
                chart="path_loss_histogram",
                elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
            )

        # Delay histogram
        delay_values = self.stats.get("delay_values", [])
        if len(delay_values) > 0 and "delay_histogram_chart" in self.widgets:
            t_chart = time.perf_counter()
            self._create_chart(
                self.widgets["delay_histogram_chart"],
                "histogram",
                {"values": delay_values, "xlabel": "Delay (ns)", "ylabel": "Count"},
                "Delay Distribution",
            )
            _log_timing(
                "StatisticsPanel._update_statistics.chart",
                chart="delay_histogram",
                elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
            )

        # ── CDF charts ──────────────────────────────────────────────────
        if len(delay_values) > 0 and "delay_cdf_chart" in self.widgets:
            self._create_chart(
                self.widgets["delay_cdf_chart"],
                "cdf",
                {"values": delay_values, "xlabel": "Delay (ns)"},
                "Delay CDF",
            )

        if len(path_loss_values) > 0 and "path_loss_cdf_chart" in self.widgets:
            self._create_chart(
                self.widgets["path_loss_cdf_chart"],
                "cdf",
                {"values": path_loss_values, "xlabel": "Path Loss (dB)"},
                "Path Loss CDF",
            )

        pair_gains = self.stats.get("pair_aggregate_path_gain_db_values", [])
        if len(pair_gains) > 0 and "pair_gain_cdf_chart" in self.widgets:
            self._create_chart(
                self.widgets["pair_gain_cdf_chart"],
                "cdf",
                {"values": pair_gains, "xlabel": "Aggregate Path Gain (dB)"},
                "Available Pair-Frame Aggregate Path Gain CDF",
            )

        pair_delay_spreads = self.stats.get("pair_rms_delay_spread_ns_values", [])
        if len(pair_delay_spreads) > 0 and "pair_delay_spread_cdf_chart" in self.widgets:
            self._create_chart(
                self.widgets["pair_delay_spread_cdf_chart"],
                "cdf",
                {"values": pair_delay_spreads, "xlabel": "RMS Delay Spread (ns)"},
                "Available Pair-Frame RMS Delay Spread CDF",
            )

        # ── Polar angular charts ───────────────────────────────────────
        angle_charts = [
            ("aod_az_values", "aod_az_polar_chart", "polar_rose", "AoD Azimuth", {}),
            (
                "aod_el_values",
                "aod_el_polar_chart",
                "angle_histogram",
                "AoD Elevation",
                {"xlabel": "Elevation (deg)", "xlim": (-90.0, 90.0)},
            ),
            ("aoa_az_values", "aoa_az_polar_chart", "polar_rose", "AoA Azimuth", {}),
            (
                "aoa_el_values",
                "aoa_el_polar_chart",
                "angle_histogram",
                "AoA Elevation",
                {"xlabel": "Elevation (deg)", "xlim": (-90.0, 90.0)},
            ),
        ]
        has_any_angle = False
        for data_key, widget_key, chart_type, chart_title, extra in angle_charts:
            angle_data = self.stats.get(data_key, [])
            if len(angle_data) > 0 and widget_key in self.widgets:
                has_any_angle = True
                self._create_chart(
                    self.widgets[widget_key],
                    chart_type,
                    {"values": angle_data, "n_bins": 36, **extra},
                    chart_title,
                )
        # Hide polar section if no angular data
        if hasattr(self, "_polar_group"):
            self._polar_group.setVisible(has_any_angle)

        # ── Evolution charts ────────────────────────────────────────────
        evolution = self.stats.get("frame_indices", [])
        if multi_frame and self._has_values(evolution):
            mpc_evolution = self.stats.get("mpc_evolution", [])
            if self._has_values(mpc_evolution) and "mpc_evolution_chart" in self.widgets:
                t_chart = time.perf_counter()
                self._create_chart(
                    self.widgets["mpc_evolution_chart"],
                    "line_evolution",
                    {"x": evolution, "y": mpc_evolution, "ylabel": "Total MPC Count"},
                    "Total MPCs per Frame",
                )
                _log_timing(
                    "StatisticsPanel._update_statistics.chart",
                    chart="mpc_evolution",
                    elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
                )

            delay_spread_evolution = self.stats.get("delay_spread_evolution", [])
            if (
                self._has_values(delay_spread_evolution)
                and len(delay_spread_evolution) == len(evolution)
                and "delay_spread_trend_chart" in self.widgets
            ):
                t_chart = time.perf_counter()
                self._create_chart(
                    self.widgets["delay_spread_trend_chart"],
                    "line_evolution",
                    {
                        "x": evolution,
                        "y": delay_spread_evolution,
                        "ylabel": "Pooled Path Delay Spread (RMS, ns)",
                    },
                    "Pooled Delay Spread per Frame",
                )
                _log_timing(
                    "StatisticsPanel._update_statistics.chart",
                    chart="delay_spread_trend",
                    elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
                )

            order_evolution_data = self.stats.get("reflection_order_evolution_per_frame", {})
            if order_evolution_data and "mpc_order_evolution_hist_chart" in self.widgets:
                t_chart = time.perf_counter()
                self._create_chart(
                    self.widgets["mpc_order_evolution_hist_chart"],
                    "mpc_order_evolution",
                    {"frame_indices": evolution, "order_data": order_evolution_data},
                    "Interaction Order Counts per Frame",
                )
                _log_timing(
                    "StatisticsPanel._update_statistics.chart",
                    chart="mpc_order_evolution",
                    elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
                )

            type_evolution_data = self.stats.get("mpc_type_evolution_per_frame", {})
            if type_evolution_data and "mpc_type_evolution_chart" in self.widgets:
                t_chart = time.perf_counter()
                self._create_chart(
                    self.widgets["mpc_type_evolution_chart"],
                    "mpc_type_evolution",
                    {"frame_indices": evolution, "type_data": type_evolution_data},
                    "Initial Propagation Mechanism per Frame",
                )
                _log_timing(
                    "StatisticsPanel._update_statistics.chart",
                    chart="mpc_type_evolution",
                    elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
                )

            # ── Channel Evolution charts ───────────────────────────────
            pair_visibility_evolution = self.stats.get("pair_visibility_evolution", {})
            if (
                pair_visibility_evolution
                and all(
                    len(values) == len(evolution) for values in pair_visibility_evolution.values()
                )
                and "pair_visibility_chart" in self.widgets
            ):
                t_chart = time.perf_counter()
                self._create_chart(
                    self.widgets["pair_visibility_chart"],
                    "pair_visibility_evolution",
                    {
                        "frame_indices": evolution,
                        "category_data": pair_visibility_evolution,
                    },
                    "TX/RX Pair Path-State Counts per Frame",
                )
                _log_timing(
                    "StatisticsPanel._update_statistics.chart",
                    chart="pair_visibility_evolution",
                    elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
                )

            strongest_loss_evolution = self.stats.get("strongest_single_path_loss_evolution", [])
            if (
                self._has_values(strongest_loss_evolution)
                and len(strongest_loss_evolution) == len(evolution)
                and "strongest_path_loss_chart" in self.widgets
            ):
                t_chart = time.perf_counter()
                self._create_chart(
                    self.widgets["strongest_path_loss_chart"],
                    "line_evolution",
                    {
                        "x": evolution,
                        "y": strongest_loss_evolution,
                        "ylabel": "Single-Path Loss (dB)",
                    },
                    "Strongest Single-Path Loss vs Frame",
                )
                _log_timing(
                    "StatisticsPanel._update_statistics.chart",
                    chart="strongest_path_loss",
                    elapsed_ms=f"{(time.perf_counter() - t_chart) * 1000:.1f}",
                )

        self._render_coverage_graphs(force=force)
        self._graphs_dirty = False
        self._graphs_rendered = True
        self._sync_chart_tabs()
        status_text, transient = self._graphs_render_status(
            has_angle_data=has_any_angle,
            multi_frame=multi_frame,
        )
        self._set_graphs_status(status_text, transient=transient)

        _log_timing(
            "StatisticsPanel._render_statistics_graphs.end",
            elapsed_ms=f"{(time.perf_counter() - t0) * 1000:.1f}",
        )
