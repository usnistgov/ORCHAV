"""Performance/diagnostic subsection for live-gRPC data sources.

The section reads latency statistics, buffer utilization, request history, and
connection status from ``GrpcProvider`` snapshots. It keeps short UI histories
only for graphs and tables; provider/network policy stays outside the panel.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ...app.plot_theme import apply_matplotlib_legend_theme, apply_matplotlib_theme
from ...app.theme import current_theme, get_theme_manager
from ..collapsible_section import CollapsibleSection
from ..ui_theme import compact_progress_bar_style, configure_label

logger = get_logger("orchav.performance_section")

MAX_HISTORY_POINTS = 100
MAX_CONNECTION_EVENTS = 100


def _request_latency_ms(request: Dict[str, Any]) -> Optional[float]:
    """Return the most useful request latency field as a float."""
    latency = request.get("generation_time_ms")
    if latency is None:
        latency = request.get("latency_ms")
    if latency is None:
        return None
    try:
        return float(latency)
    except (TypeError, ValueError):
        return None


def _request_is_in_progress(request: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Return whether a request should be treated as still in progress."""
    if request.get("generation_time_ms") is None and not request.get("success", False):
        return True
    if now is None:
        now = time.time()
    try:
        return (now - float(request.get("request_timestamp", 0))) < 5.0
    except (TypeError, ValueError):
        return False


def _request_status_text(request: Dict[str, Any]) -> str:
    """Return the display status for a request-history row."""
    if request.get("success", False):
        return "Success"
    if _request_is_in_progress(request):
        return "In Progress"
    return "Failed"


class _RequestHistoryModel(QAbstractTableModel):
    """Model-backed request history table for live data-source diagnostics."""

    HEADERS = ("Time", "Frame", "Type", "Status", "Latency (ms)", "Source", "Origin", "Details")

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._requests: List[Dict[str, Any]] = []

    def set_requests(self, requests: List[Dict[str, Any]]) -> None:
        """Replace the model rows with bounded request-history snapshots."""
        self.beginResetModel()
        self._requests = list(requests)
        self.endResetModel()

    def request_at(self, row: int) -> Optional[Dict[str, Any]]:
        """Return the source request at ``row`` if present."""
        if 0 <= row < len(self._requests):
            return self._requests[row]
        return None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._requests)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        req = self.request_at(index.row())
        if req is None:
            return None
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_value(req, column)

        if role == Qt.ItemDataRole.UserRole:
            return self._sort_value(req, column)

        if role == Qt.ItemDataRole.ForegroundRole:
            theme = current_theme()
            if column == 3:
                status = _request_status_text(req)
                if status == "Success":
                    return QColor(theme.success)
                if status == "In Progress":
                    return QColor(theme.warning)
                return QColor(theme.error)
            if column == 5:
                source = req.get("source", "unknown")
                if source == "buffer":
                    return QColor(theme.accent)
                if source == "server":
                    return QColor(theme.success)
                if source == "timeout":
                    return QColor(theme.warning)

        if role == Qt.ItemDataRole.TextAlignmentRole and column in (1, 4):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return None

    @staticmethod
    def _display_value(req: Dict[str, Any], column: int) -> str:
        """Return the display string for one request-history cell."""
        if column == 0:
            req_time = req.get("request_timestamp", time.time())
            return time.strftime("%H:%M:%S", time.localtime(req_time))
        if column == 1:
            return str(req.get("frame_idx", -1))
        if column == 2:
            return str(req.get("request_type", "GET_FRAME"))
        if column == 3:
            return _request_status_text(req)
        if column == 4:
            latency = _request_latency_ms(req)
            return f"{latency:.1f}" if latency is not None else "--"
        if column == 5:
            return str(req.get("source", "unknown"))
        if column == 6:
            return str(req.get("origin", "--"))
        if column == 7:
            details = req.get("details", "--")
            return str(details) if details else "--"
        return ""

    @staticmethod
    def _sort_value(req: Dict[str, Any], column: int) -> Any:
        """Return stable values for proxy sorting."""
        if column == 0:
            return float(req.get("request_timestamp", 0.0) or 0.0)
        if column == 1:
            return int(req.get("frame_idx", -1) or -1)
        if column == 3:
            status_order = {"In Progress": 0, "Failed": 1, "Success": 2}
            return status_order.get(_request_status_text(req), 99)
        if column == 4:
            latency = _request_latency_ms(req)
            return latency if latency is not None else -1.0
        return _RequestHistoryModel._display_value(req, column)


class _RequestHistoryFilterProxy(QSortFilterProxyModel):
    """Filter request-history rows by status without rebuilding table items."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._filter_value = "All"

    def set_filter_value(self, value: str) -> None:
        """Set the visible status filter."""
        self._filter_value = value
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        if not isinstance(model, _RequestHistoryModel):
            return True
        req = model.request_at(source_row)
        if req is None:
            return False
        if self._filter_value == "Success":
            return bool(req.get("success", False))
        if self._filter_value == "Failed":
            return not bool(req.get("success", False)) and not _request_is_in_progress(req)
        if self._filter_value == "In Progress":
            return _request_is_in_progress(req)
        return True


class _ConnectionHistoryModel(QAbstractTableModel):
    """Model-backed connection event history table."""

    HEADERS = ("Timestamp", "Event", "Status", "Details", "Duration")

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._events: List[tuple] = []

    def set_events(self, events: List[tuple]) -> None:
        """Replace the connection event rows."""
        self.beginResetModel()
        self._events = list(events)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._events)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if not 0 <= index.row() < len(self._events):
            return None
        timestamp, event, status, details, duration = self._events[index.row()]
        values = (
            time.strftime("%H:%M:%S", time.localtime(timestamp)),
            event,
            status,
            details,
            duration,
        )
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return str(values[column])
        if role == Qt.ItemDataRole.UserRole:
            if column == 0:
                return float(timestamp)
            return values[column]
        if role == Qt.ItemDataRole.ForegroundRole and column == 2:
            theme = current_theme()
            if status == "Success":
                return QColor(theme.success)
            if status == "Error":
                return QColor(theme.error)
        return None


def _pending_frame_idx(request: Any) -> Optional[int]:
    """Return a pending-request frame index from dataclass or dict snapshots."""
    if isinstance(request, dict):
        value = request.get("frame_idx")
    else:
        value = getattr(request, "frame_idx", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pending_request_timestamp(request: Any) -> Optional[float]:
    """Return a pending-request timestamp from dataclass or dict snapshots."""
    if isinstance(request, dict):
        value = request.get("request_timestamp")
    else:
        value = getattr(request, "request_timestamp", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pending_request_snapshots(provider: Any) -> List[Any]:
    """Return provider pending-request snapshots without touching private fields."""
    get_pending = getattr(provider, "get_pending_frame_requests", None)
    if not callable(get_pending):
        return []
    return list(get_pending())


class PerformanceSection:
    """Build live-gRPC telemetry, graph, request-history, and diagnostic views.

    Args:
        parent: The visualizer instance.
        widgets: Shared widget registry dict from DataSourcePanel.
        button_style_fn: Callable returning QPushButton stylesheet string.
    """

    def __init__(
        self,
        parent: Any,
        widgets: Dict[str, Any],
        button_style_fn: Any,
    ) -> None:
        """Initialize bounded UI histories and optional matplotlib state."""
        self.parent = parent
        self.widgets = widgets
        self._get_button_style = button_style_fn

        # Performance data history for graphs
        self.performance_history: Dict[str, List] = {
            "frame_times": [],
            "buffer_util": [],
            "latency": [],
        }
        self._last_perf_history_timestamp: float = 0.0
        self._last_perf_graph_render: float = 0.0

        # Matplotlib availability
        self._has_matplotlib: bool = False
        self._FigureCanvas: Any = None
        self._Figure: Any = None

        # Persistent Line2D objects for incremental graph updates
        self._frame_time_line: Any = None
        self._buffer_util_line: Any = None
        self._performance_graph_group: CollapsibleSection | None = None
        self._performance_graph_layout: Any = None
        self._performance_graphs_initialized = False
        self._performance_graph_placeholder: QLabel | None = None

        # Connection diagnostics state
        self.connection_events: List[tuple] = []
        self._last_connection_state: Optional[bool] = None
        self._connection_start_time: Optional[float] = None
        self._last_error_time: Optional[float] = None

        # Request history state for incremental table updates
        self._full_request_history: List[Dict[str, Any]] = []
        self._last_table_row_count: int = 0
        self._theme_manager = None
        self._theme_changed_callback = self._apply_graph_theme

    def cleanup(self) -> None:
        """Release the process-wide theme subscription."""
        if self._theme_manager is None:
            return
        try:
            self._theme_manager.theme_changed.disconnect(self._theme_changed_callback)
        except (RuntimeError, TypeError):
            pass
        self._theme_manager = None

    def create_content(self) -> QWidget:
        """Create performance section content widget.

        Returns:
            A QWidget containing telemetry, graphs, and diagnostics groups.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_telemetry_group())
        layout.addWidget(self._create_performance_graph_group())
        layout.addWidget(self._create_connection_diagnostics_group())

        return container

    # -- Frame Generation Telemetry -------------------------------------------

    def _create_telemetry_group(self) -> QGroupBox:
        """Create latency, success-rate, progress, and request-history widgets."""
        group = QGroupBox("Frame Generation Telemetry")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        avg_row = QHBoxLayout()
        avg_label = QLabel("Avg Time:")
        configure_label(avg_label, role="secondary", min_width=100)
        avg_label.setToolTip(
            "Average round-trip time (request to receive, includes network + computation)"
        )
        avg_row.addWidget(avg_label)
        self.widgets["online_avg_time"] = QLabel("--")
        avg_row.addWidget(self.widgets["online_avg_time"])
        avg_row.addStretch()
        layout.addLayout(avg_row)

        minmax_row = QHBoxLayout()
        minmax_label = QLabel("Min/Max:")
        configure_label(minmax_label, role="secondary", min_width=100)
        minmax_label.setToolTip(
            "Min/Max round-trip time (request to receive, includes network + computation)"
        )
        minmax_row.addWidget(minmax_label)
        self.widgets["online_minmax_time"] = QLabel("--")
        minmax_row.addWidget(self.widgets["online_minmax_time"])
        minmax_row.addStretch()
        layout.addLayout(minmax_row)

        success_row = QHBoxLayout()
        success_label = QLabel("Success Rate:")
        configure_label(success_label, role="secondary", min_width=100)
        success_row.addWidget(success_label)
        self.widgets["online_success_rate"] = QLabel("--")
        success_row.addWidget(self.widgets["online_success_rate"])
        success_row.addStretch()
        layout.addLayout(success_row)

        cancel_row = QHBoxLayout()
        self.widgets["cancel_request_btn"] = QPushButton("Cancel Selected Request")
        self.widgets["cancel_request_btn"].setStyleSheet(self._get_button_style())
        self.widgets["cancel_request_btn"].setEnabled(False)
        self.widgets["cancel_request_btn"].clicked.connect(self._on_cancel_request_clicked)
        self.widgets["cancel_request_btn"].setToolTip(
            "Cancel the selected in-progress frame request"
        )
        cancel_row.addWidget(self.widgets["cancel_request_btn"])
        cancel_row.addStretch()
        layout.addLayout(cancel_row)

        progress_label = QLabel("Pending Frame Requests:")
        configure_label(progress_label, role="secondary", bold=True)
        progress_label.setToolTip(
            "Requests waiting for a generator response; the protocol does not report "
            "solver completion percentage"
        )
        layout.addWidget(progress_label)

        self.widgets["frame_progress_bar"] = QProgressBar()
        self.widgets["frame_progress_bar"].setRange(0, 100)
        self.widgets["frame_progress_bar"].setValue(0)
        self.widgets["frame_progress_bar"].setFormat("No pending requests")
        self.widgets["frame_progress_bar"].setStyleSheet(compact_progress_bar_style())
        layout.addWidget(self.widgets["frame_progress_bar"])

        self.widgets["frame_progress_label"] = QLabel("")
        configure_label(self.widgets["frame_progress_label"], role="secondary", font_size=9)
        layout.addWidget(self.widgets["frame_progress_label"])

        history_section = CollapsibleSection("Request History", start_open=False)
        history_layout = history_section.content_layout()
        history_layout.setSpacing(4)
        history_layout.setContentsMargins(0, 0, 0, 0)

        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        configure_label(filter_label, role="secondary", font_size=9)
        filter_layout.addWidget(filter_label)

        self.widgets["request_history_filter"] = QComboBox()
        self.widgets["request_history_filter"].addItems(["All", "Success", "Failed", "In Progress"])
        self.widgets["request_history_filter"].setStyleSheet("font-size: 9px;")
        self.widgets["request_history_filter"].currentTextChanged.connect(
            self._filter_request_history
        )
        filter_layout.addWidget(self.widgets["request_history_filter"])
        filter_layout.addStretch()
        history_layout.addLayout(filter_layout)

        request_model = _RequestHistoryModel()
        request_proxy = _RequestHistoryFilterProxy()
        request_proxy.setSourceModel(request_model)
        request_proxy.setSortRole(Qt.ItemDataRole.UserRole)
        request_table = QTableView()
        request_table.setModel(request_proxy)
        request_table.horizontalHeader().setStretchLastSection(True)
        request_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        request_table.verticalHeader().setVisible(False)
        request_table.setMaximumHeight(200)
        request_table.setAlternatingRowColors(True)
        request_table.setSortingEnabled(True)
        request_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        request_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        request_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        request_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        request_table.setObjectName("requestHistoryTable")
        self.widgets["request_history_model"] = request_model
        self.widgets["request_history_proxy"] = request_proxy
        self.widgets["request_history_table"] = request_table
        history_layout.addWidget(self.widgets["request_history_table"])

        layout.addWidget(history_section)

        return group

    # -- Performance Graphs ---------------------------------------------------

    def _create_performance_graph_group(self) -> CollapsibleSection:
        """Create a closed shell that initializes live graphs only when expanded."""
        group = CollapsibleSection("Performance Graphs", start_open=False)
        layout = group.content_layout()
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        placeholder = QLabel("Expand to create live performance graphs")
        configure_label(placeholder, role="secondary", font_size=10)
        layout.addWidget(placeholder)
        self._performance_graph_group = group
        self._performance_graph_layout = layout
        self._performance_graph_placeholder = placeholder
        group.toggled.connect(self._on_performance_graphs_toggled)
        return group

    def _on_performance_graphs_toggled(self, expanded: bool) -> None:
        """Create graph canvases on first expansion."""
        if expanded:
            self._ensure_performance_graphs()

    def _ensure_performance_graphs(self) -> None:
        """Import Matplotlib and create live graph canvases at most once."""
        if self._performance_graphs_initialized:
            return
        self._performance_graphs_initialized = True
        layout = self._performance_graph_layout
        if layout is None:
            return

        placeholder = self._performance_graph_placeholder
        self._performance_graph_placeholder = None
        if placeholder is not None:
            layout.removeWidget(placeholder)
            placeholder.deleteLater()

        try:
            import matplotlib

            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._has_matplotlib = True
            self._FigureCanvas = FigureCanvasQTAgg
            self._Figure = Figure
        except ImportError:
            self._has_matplotlib = False
            error_label = QLabel("Matplotlib not available. Install with: pip install matplotlib")
            configure_label(error_label, role="error", font_size=10)
            layout.addWidget(error_label)
            return

        self._theme_manager = get_theme_manager()
        self._theme_manager.theme_changed.connect(self._theme_changed_callback)

        tabs = QTabWidget()

        # Frame time graph - create Line2D for incremental updates
        self.widgets["perf_graph_frame_time"] = self._create_performance_graph(
            "Frame Generation Time (ms)", "Time (ms)"
        )
        fig_ft = self.widgets["perf_graph_frame_time"].figure
        ax_ft = fig_ft.axes[0]
        theme = current_theme()
        (self._frame_time_line,) = ax_ft.plot(
            [], [], color=theme.accent, linewidth=1.5, label="Frame Time"
        )
        ax_ft.legend(fontsize=8)
        apply_matplotlib_legend_theme(ax_ft)
        tabs.addTab(self.widgets["perf_graph_frame_time"], "Frame Time")

        # Buffer utilization graph
        self.widgets["perf_graph_buffer"] = self._create_performance_graph(
            "Buffer Utilization (%)", "Utilization (%)"
        )
        fig_bu = self.widgets["perf_graph_buffer"].figure
        ax_bu = fig_bu.axes[0]
        (self._buffer_util_line,) = ax_bu.plot(
            [], [], color=theme.success, linewidth=1.5, label="Utilization"
        )
        ax_bu.set_ylim(0, 120)
        ax_bu.legend(fontsize=8)
        apply_matplotlib_legend_theme(ax_bu)
        tabs.addTab(self.widgets["perf_graph_buffer"], "Buffer")

        layout.addWidget(tabs)

    def _create_performance_graph(self, title: str, ylabel: str) -> Any:
        """Create a matplotlib graph widget with initial layout.

        Args:
            title: Graph title.
            ylabel: Y-axis label.

        Returns:
            A ``FigureCanvasQTAgg`` instance.
        """
        fig = self._Figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Elapsed (s)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        apply_matplotlib_theme(fig, ax)
        fig.tight_layout()

        canvas = self._FigureCanvas(fig)
        canvas.setMinimumHeight(200)
        canvas.setMaximumHeight(200)
        return canvas

    def _apply_graph_theme(self, *_args: Any) -> None:
        """Restyle existing performance graph canvases after a theme change."""
        theme = current_theme()
        for widget_key, line, line_color in (
            ("perf_graph_frame_time", self._frame_time_line, theme.accent),
            ("perf_graph_buffer", self._buffer_util_line, theme.success),
        ):
            canvas = self.widgets.get(widget_key)
            if canvas is None:
                continue
            fig = canvas.figure
            for ax in fig.axes:
                apply_matplotlib_theme(fig, ax)
                apply_matplotlib_legend_theme(ax)
            if line is not None:
                line.set_color(line_color)
            canvas.draw_idle()

    # -- Connection Diagnostics -----------------------------------------------

    def _create_connection_diagnostics_group(self) -> CollapsibleSection:
        """Create the connection event history widgets."""
        group = CollapsibleSection("Connection Diagnostics", start_open=False)
        layout = group.content_layout()
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        stats_label = QLabel("Network Statistics:")
        configure_label(stats_label, role="secondary", bold=True)
        layout.addWidget(stats_label)

        connection_model = _ConnectionHistoryModel()
        connection_proxy = QSortFilterProxyModel()
        connection_proxy.setSourceModel(connection_model)
        connection_proxy.setSortRole(Qt.ItemDataRole.UserRole)
        connection_table = QTableView()
        connection_table.setModel(connection_proxy)
        connection_table.horizontalHeader().setStretchLastSection(True)
        connection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        connection_table.verticalHeader().setVisible(False)
        connection_table.setMaximumHeight(150)
        connection_table.setAlternatingRowColors(True)
        connection_table.setSortingEnabled(True)
        connection_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        connection_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        connection_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        connection_table.setObjectName("connectionHistoryTable")
        self.widgets["connection_history_model"] = connection_model
        self.widgets["connection_history_proxy"] = connection_proxy
        self.widgets["connection_history_table"] = connection_table
        layout.addWidget(self.widgets["connection_history_table"])

        return group

    # -- Update ---------------------------------------------------------------

    def update(
        self,
        provider: Any,
        status: Dict[str, Any],
        buffer_status: Dict[str, Any],
        latency_stats: Dict[str, Any],
        request_history: List[Dict[str, Any]],
    ) -> None:
        """Update all performance section widgets.

        Args:
            provider: The GrpcProvider instance.
            status: Connection status dict.
            buffer_status: Buffer status dict.
            latency_stats: Latency statistics dict.
            request_history: Recent request history entries.
        """
        self._update_telemetry(latency_stats, request_history)
        self._update_request_history_table(request_history)
        self._update_cancel_button_state(provider, request_history)
        self._update_frame_progress(provider)
        self._update_performance_history(status, buffer_status, latency_stats, request_history)
        self._update_connection_diagnostics(status)

    def _update_telemetry(
        self,
        latency_stats: Dict[str, Any],
        request_history: List[Dict[str, Any]],
    ) -> None:
        """Update telemetry labels (avg/min/max time, success rate)."""
        if latency_stats.get("count", 0) > 0:
            avg_time = latency_stats.get("average_ms", 0.0)
            min_time = latency_stats.get("min_ms", 0.0)
            max_time = latency_stats.get("max_ms", 0.0)
            self.widgets["online_avg_time"].setText(f"{avg_time:.1f} ms")
            self.widgets["online_minmax_time"].setText(f"{min_time:.1f} / {max_time:.1f} ms")
        else:
            self.widgets["online_avg_time"].setText("--")
            self.widgets["online_minmax_time"].setText("--")

        if request_history:
            successful = sum(1 for r in request_history if r.get("success", False))
            total = len(request_history)
            if total > 0:
                success_rate = (successful / total) * 100
                self.widgets["online_success_rate"].setText(
                    f"{success_rate:.1f}% ({successful}/{total})"
                )
            else:
                self.widgets["online_success_rate"].setText("--")
        else:
            self.widgets["online_success_rate"].setText("--")

    def _update_request_history_table(self, request_history: List[Dict[str, Any]]) -> None:
        """Store recent request history and refresh the visible filtered table."""
        if "request_history_table" not in self.widgets:
            return
        try:
            self._full_request_history = request_history[:100]
            model = self.widgets.get("request_history_model")
            if isinstance(model, _RequestHistoryModel):
                model.set_requests(self._full_request_history)
            proxy = self.widgets.get("request_history_proxy")
            if isinstance(proxy, _RequestHistoryFilterProxy):
                proxy.invalidateFilter()
            table = self.widgets.get("request_history_table")
            if table is not None and not table.isVisible():
                return
        except (KeyError, AttributeError, ValueError) as e:
            logger.debug("Error updating request history table: %s", e)

    def _filter_request_history(self) -> None:
        """Apply the selected request-history filter to the table."""
        if "request_history_table" not in self.widgets:
            return

        try:
            filter_widget = self.widgets.get("request_history_filter")
            filter_value = filter_widget.currentText() if filter_widget else "All"

            proxy = self.widgets.get("request_history_proxy")
            if isinstance(proxy, _RequestHistoryFilterProxy):
                proxy.set_filter_value(filter_value)

        except (KeyError, AttributeError, ValueError) as e:
            logger.debug("Error filtering request history: %s", e)

    def _update_cancel_button_state(
        self,
        provider: Any,
        request_history: List[Dict[str, Any]],
    ) -> None:
        """Update cancel button enabled state based on in-progress requests."""
        try:
            in_progress_count = len(_pending_request_snapshots(provider))

            if in_progress_count == 0 and request_history:
                current_time = time.time()
                for req in request_history[:10]:
                    req_time = req.get("request_timestamp", 0)
                    if (current_time - req_time) < 30.0 and req.get("generation_time_ms") is None:
                        in_progress_count += 1
                        break

            cancel_btn = self.widgets.get("cancel_request_btn")
            if cancel_btn is not None:
                cancel_btn.setEnabled(in_progress_count > 0)
        except (KeyError, AttributeError, ValueError) as e:
            logger.debug("Error updating cancel button state: %s", e)

    def _update_frame_progress(self, provider: Any) -> None:
        """Show pending request count and age without inventing completion percent."""
        if "frame_progress_bar" not in self.widgets:
            return

        try:
            pending_requests = _pending_request_snapshots(provider)
            in_progress_frames = [
                frame_idx
                for frame_idx in (_pending_frame_idx(request) for request in pending_requests)
                if frame_idx is not None
            ]

            if in_progress_frames:
                oldest_age = 0.0
                timestamps = [
                    timestamp
                    for timestamp in (
                        _pending_request_timestamp(request) for request in pending_requests
                    )
                    if timestamp is not None
                ]
                if timestamps:
                    oldest_age = max(0.0, time.time() - min(timestamps))

                self.widgets["frame_progress_bar"].setRange(0, 0)
                self.widgets["frame_progress_bar"].setFormat(
                    f"{len(in_progress_frames)} pending - oldest {oldest_age:.1f}s"
                )

                progress_label = self.widgets.get("frame_progress_label")
                if progress_label is not None:
                    progress_label.setText(
                        "Waiting for generator responses; exact solver progress is unavailable"
                    )
            else:
                self.widgets["frame_progress_bar"].setRange(0, 1)
                self.widgets["frame_progress_bar"].setValue(0)
                self.widgets["frame_progress_bar"].setFormat("No pending requests")

                progress_label = self.widgets.get("frame_progress_label")
                if progress_label is not None:
                    progress_label.setText("")
        except (KeyError, AttributeError, ValueError) as e:
            logger.debug("Error updating frame progress: %s", e)

    # -- Performance graphs (incremental updates) -----------------------------

    def _update_performance_graphs(self) -> None:
        """Update performance graphs using incremental line.set_data()."""
        if not self._has_matplotlib:
            return
        frame_canvas = self.widgets.get("perf_graph_frame_time")
        buffer_canvas = self.widgets.get("perf_graph_buffer")
        if (frame_canvas is not None and not frame_canvas.isVisible()) and (
            buffer_canvas is not None and not buffer_canvas.isVisible()
        ):
            return

        try:
            # Frame time graph
            if (
                frame_canvas is not None
                and frame_canvas.isVisible()
                and self._frame_time_line is not None
                and self.performance_history["frame_times"]
            ):
                times, values = zip(*self.performance_history["frame_times"])
                base_time = times[0]
                rel_times = [(t - base_time) for t in times]
                self._frame_time_line.set_data(rel_times, values)

                fig = frame_canvas.figure
                ax = fig.axes[0]
                ax.relim()
                ax.autoscale_view()
                frame_canvas.draw_idle()

            # Buffer utilization graph
            if (
                buffer_canvas is not None
                and buffer_canvas.isVisible()
                and self._buffer_util_line is not None
                and self.performance_history["buffer_util"]
            ):
                times, values = zip(*self.performance_history["buffer_util"])
                base_time = times[0]
                rel_times = [(t - base_time) for t in times]
                self._buffer_util_line.set_data(rel_times, values)

                fig = buffer_canvas.figure
                ax = fig.axes[0]
                ax.relim()
                ax.autoscale_view()
                ax.set_ylim(0, max(120, max(values) * 1.1) if values else 120)
                buffer_canvas.draw_idle()

        except (ValueError, TypeError, RuntimeError) as e:
            logger.debug("Error updating performance graphs: %s", e)

    def _update_performance_history(
        self,
        status: Dict[str, Any],
        buffer_status: Dict[str, Any],
        latency_stats: Dict[str, Any],
        request_history: List[Dict[str, Any]],
    ) -> None:
        """Append bounded samples for graph rendering from provider snapshots."""
        try:
            current_time = time.time()
            updated = False

            new_entries: List[Dict[str, Any]] = []
            for req in request_history:
                req_ts = req.get("request_timestamp")
                if not req_ts:
                    continue
                if req_ts <= self._last_perf_history_timestamp:
                    break
                new_entries.append(req)

            for req in reversed(new_entries):
                req_ts = req.get("request_timestamp")
                latency = req.get("generation_time_ms") or req.get("latency_ms")
                buffer_util = req.get("buffer_utilization")

                if latency is not None:
                    self.performance_history["frame_times"].append((req_ts, latency))
                    if len(self.performance_history["frame_times"]) > MAX_HISTORY_POINTS:
                        self.performance_history["frame_times"] = self.performance_history[
                            "frame_times"
                        ][-MAX_HISTORY_POINTS:]
                    if req.get("source") == "server":
                        self.performance_history["latency"].append((req_ts, latency))
                        if len(self.performance_history["latency"]) > MAX_HISTORY_POINTS:
                            self.performance_history["latency"] = self.performance_history[
                                "latency"
                            ][-MAX_HISTORY_POINTS:]

                if buffer_util is not None:
                    utilization_pct = buffer_util * 100.0 if buffer_util <= 1.0 else buffer_util
                    self.performance_history["buffer_util"].append((req_ts, utilization_pct))
                    if len(self.performance_history["buffer_util"]) > MAX_HISTORY_POINTS:
                        self.performance_history["buffer_util"] = self.performance_history[
                            "buffer_util"
                        ][-MAX_HISTORY_POINTS:]

                self._last_perf_history_timestamp = max(self._last_perf_history_timestamp, req_ts)
                updated = True

            if not updated:
                if latency_stats.get("count", 0) > 0:
                    avg_time = latency_stats.get("average_ms", 0.0)
                    self.performance_history["frame_times"].append((current_time, avg_time))
                    if len(self.performance_history["frame_times"]) > MAX_HISTORY_POINTS:
                        self.performance_history["frame_times"] = self.performance_history[
                            "frame_times"
                        ][-MAX_HISTORY_POINTS:]
                utilization = buffer_status.get("buffer_utilization", 0.0) * 100
                self.performance_history["buffer_util"].append((current_time, utilization))
                if len(self.performance_history["buffer_util"]) > MAX_HISTORY_POINTS:
                    self.performance_history["buffer_util"] = self.performance_history[
                        "buffer_util"
                    ][-MAX_HISTORY_POINTS:]
                avg_latency = status.get("average_latency_ms", 0.0)
                if avg_latency > 0:
                    self.performance_history["latency"].append((current_time, avg_latency))
                    if len(self.performance_history["latency"]) > MAX_HISTORY_POINTS:
                        self.performance_history["latency"] = self.performance_history["latency"][
                            -MAX_HISTORY_POINTS:
                        ]

            if updated or (current_time - self._last_perf_graph_render) > 0.2:
                self._update_performance_graphs()
                self._last_perf_graph_render = current_time

        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Error updating performance history: %s", e)

    # -- Connection diagnostics -----------------------------------------------

    def _update_connection_diagnostics(self, status: Dict[str, Any]) -> None:
        """Update connection event history from status transitions and errors."""
        if "connection_history_table" not in self.widgets:
            return

        try:
            current_time = time.time()
            is_connected = status.get("connected", False)
            last_error = status.get("last_error")
            last_error_time = status.get("last_error_timestamp")

            if self._last_connection_state != is_connected:
                if is_connected:
                    self._connection_start_time = current_time
                    self._add_connection_event(
                        current_time, "Connected", "Success", "Connection established", None
                    )
                else:
                    duration = None
                    if self._connection_start_time:
                        duration = current_time - self._connection_start_time
                    self._add_connection_event(
                        current_time, "Disconnected", "Error", "Connection lost", duration
                    )
                self._last_connection_state = is_connected

            if last_error and last_error_time:
                if self._last_error_time != last_error_time:
                    self._add_connection_event(
                        last_error_time, "Error", "Error", last_error[:80], None
                    )
                    self._last_error_time = last_error_time

            self._update_connection_history_table()

        except (KeyError, AttributeError, ValueError) as e:
            logger.debug("Error updating connection diagnostics: %s", e)

    def _add_connection_event(
        self,
        timestamp: float,
        event: str,
        status: str,
        details: str,
        duration: Optional[float],
    ) -> None:
        """Add one bounded connection event row for diagnostics."""
        duration_str = f"{duration:.1f}s" if duration else "--"
        self.connection_events.append((timestamp, event, status, details, duration_str))
        if len(self.connection_events) > MAX_CONNECTION_EVENTS:
            self.connection_events = self.connection_events[-MAX_CONNECTION_EVENTS:]

    def _update_connection_history_table(self) -> None:
        """Render stored connection events when the diagnostics table is visible."""
        if "connection_history_table" not in self.widgets:
            return

        table = self.widgets["connection_history_table"]
        if not table.isVisible():
            return
        model = self.widgets.get("connection_history_model")
        if isinstance(model, _ConnectionHistoryModel):
            model.set_events(self.connection_events)

    # -- Event Handlers -------------------------------------------------------

    def _on_cancel_request_clicked(self) -> None:
        """Cancel a selected or oldest pending live-gRPC frame request."""
        try:
            from ...io.frame_sources import LiveGrpcSource

            if not isinstance(self.parent.frame_source, LiveGrpcSource):
                QMessageBox.warning(
                    self.parent,
                    "Error",
                    "Request cancellation is only available in live gRPC mode",
                )
                return

            provider = self.parent.frame_source.provider
            if not provider:
                return

            history_table = self.widgets.get("request_history_table")
            frame_idx = None

            if history_table:
                selection_model = history_table.selectionModel()
                selected_rows = selection_model.selectedRows() if selection_model else []
                proxy = self.widgets.get("request_history_proxy")
                model = self.widgets.get("request_history_model")
                if (
                    selected_rows
                    and isinstance(proxy, _RequestHistoryFilterProxy)
                    and isinstance(
                        model,
                        _RequestHistoryModel,
                    )
                ):
                    source_index = proxy.mapToSource(selected_rows[0])
                    request = model.request_at(source_index.row())
                    if request is not None:
                        try:
                            frame_idx = int(request.get("frame_idx", -1))
                        except (TypeError, ValueError):
                            pass

            if frame_idx is None:
                pending_frames = [
                    idx
                    for idx in (
                        _pending_frame_idx(request)
                        for request in _pending_request_snapshots(provider)
                    )
                    if idx is not None
                ]
                if pending_frames:
                    frame_idx = pending_frames[0]
                else:
                    QMessageBox.information(
                        self.parent,
                        "No Requests",
                        "No in-progress requests to cancel",
                    )
                    return

            cancel_request = getattr(provider, "cancel_pending_frame_request", None)
            if not callable(cancel_request):
                QMessageBox.warning(
                    self.parent,
                    "Not Supported",
                    "This live gRPC provider does not support request cancellation",
                )
                return

            if cancel_request(frame_idx):
                logger.info("Cancelled request for frame %d", frame_idx)
                QMessageBox.information(
                    self.parent,
                    "Request Cancelled",
                    f"Cancelled request for frame {frame_idx}",
                )
            else:
                QMessageBox.information(
                    self.parent,
                    "Not Found",
                    f"Frame {frame_idx} is not in the pending requests",
                )
        except (OSError, RuntimeError, KeyError, AttributeError) as e:
            logger.error("Error cancelling request: %s", e)
            QMessageBox.warning(self.parent, "Error", f"Failed to cancel request: {e}")

    def set_defaults(self) -> None:
        """Reset all telemetry widgets to default values."""
        for key in ["online_avg_time", "online_minmax_time", "online_success_rate"]:
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setText("--")
