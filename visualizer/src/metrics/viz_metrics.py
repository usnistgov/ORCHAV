"""Live per-frame MPC Analysis Dashboard (pyqtgraph).

The dashboard displays diagnostics for the current analysis path selection.
TX/RX, path type, order, delay, angle, and material filters constrain that
selection. The visualizer's strongest-path Top-K remains a 3D rendering cap and
does not discard paths from Metrics calculations or exports.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QRectF,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from PySide6.QtGui import QBrush, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger
from shared.statistics import DEFAULT_BINS

from ..app.plot_theme import apply_pyqtgraph_plot_theme, pyqtgraph_theme
from ..app.theme import current_theme, get_theme_manager
from .mpc_path_catalog import MpcPathCatalog, MpcPathCatalogError


def _default_ui_font(
    point_size: int,
    *,
    weight: QFont.Weight | None = None,
) -> QFont:
    """Return the platform's Qt UI font with dashboard-specific styling."""
    font = QFont()
    font.setPointSize(point_size)
    if weight is not None:
        font.setWeight(weight)
    application = QApplication.instance()
    if application is not None:
        font = font.resolve(application.font())
    return font


class _MaterialBreakdownModel(QAbstractTableModel):
    """Read-only model for the metrics material summary table."""

    HEADERS = (
        "Material",
        "Hits",
        "Paths",
        "Associated Path-Gain Proxy",
        "Mean Associated Loss",
        "Strongest Associated Loss",
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        """Replace material breakdown rows."""
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._rows)

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
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_value(row, column)
        if role == Qt.ItemDataRole.UserRole:
            return self._sort_value(row, column)
        if role == Qt.ItemDataRole.TextAlignmentRole and column > 0:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    @staticmethod
    def _display_value(row: dict[str, Any], column: int) -> str:
        if column == 0:
            return str(row["material"])
        if column == 1:
            return str(row["hits"])
        if column == 2:
            return str(row["path_count"])
        if column == 3:
            return MetricsWindow._format_optional_float(row["power_db"], " dB")
        if column == 4:
            return MetricsWindow._format_optional_float(row["mean_loss"], " dB")
        if column == 5:
            return MetricsWindow._format_optional_float(row["strongest_loss"], " dB")
        return ""

    @staticmethod
    def _sort_value(row: dict[str, Any], column: int) -> Any:
        if column == 0:
            return str(row["material"]).lower()
        if column == 1:
            return int(row["hits"])
        if column == 2:
            return int(row["path_count"])
        if column == 3:
            return float(row["power_db"])
        if column == 4:
            return float(row["mean_loss"])
        if column == 5:
            return float(row["strongest_loss"])
        return None


class _PairCountModel(QAbstractTableModel):
    """Read-only exact counts for represented TX/RX pairs."""

    HEADERS = ("TX", "RX", "Selected MPCs")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[int, int, int]] = []

    def set_rows(self, rows: list[tuple[int, int, int]]) -> None:
        """Replace the pair-count rows in TX-major order."""
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

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
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        tx, rx, count = self._rows[index.row()]
        values: tuple[Any, ...] = (f"TX{tx + 1}", f"RX{rx + 1}", f"{count:,}")
        if role == Qt.ItemDataRole.DisplayRole:
            return values[index.column()]
        if role == Qt.ItemDataRole.UserRole:
            return (tx, rx, count)[index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            alignment = Qt.AlignmentFlag.AlignVCenter
            alignment |= (
                Qt.AlignmentFlag.AlignRight if index.column() == 2 else Qt.AlignmentFlag.AlignCenter
            )
            return int(alignment)
        return None


class MetricsWindow(QMainWindow):
    """Per-frame MPC diagnostics window using pyqtgraph."""

    _DEFAULT_PLOT_POINT_LIMIT = 5000
    _MIN_PLOT_POINT_LIMIT = 0
    _MAX_PLOT_POINT_LIMIT = 50_000
    _MINIMUM_ADAPTIVE_COOLDOWN_MS = 16
    _CATALOG_PATH_COLUMNS = {
        ("path_tx", "tx_id"): "tx_ids",
        ("path_rx", "rx_id"): "rx_ids",
        ("path_orders", ""): "interaction_counts",
        ("path_delays", "delay"): "delays_ns",
        ("path_losses", "loss"): "path_losses_db",
        ("", "aod_az"): "aod_azimuth_deg",
        ("", "aod_el"): "aod_elevation_deg",
        ("", "aoa_az"): "aoa_azimuth_deg",
        ("", "aoa_el"): "aoa_elevation_deg",
    }
    _ORDER_COLORS = (
        "#2f7ed8",
        "#5cb85c",
        "#f0ad4e",
        "#d9534f",
        "#7b6ed6",
        "#4ec9c9",
    )
    _ORDER_FALLBACK_COLOR = "#888888"

    _ANGLE_FIELDS = {
        "aod_az": ("AoD Azimuth", "AoD Azimuth"),
        "aod_el": ("AoD Elevation", "AoD Elevation"),
        "aoa_az": ("AoA Azimuth", "AoA Azimuth"),
        "aoa_el": ("AoA Elevation", "AoA Elevation"),
    }

    def __init__(
        self,
        parent=None,
        update_hz: int = 8,
        frame_stats_provider: Callable[[Any], Any] | None = None,
    ):
        """Create the dashboard widgets and schedule periodic refreshes."""
        super().__init__(parent)
        self.setWindowTitle("ORCHAV - MPC Analysis Dashboard")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1040, 760)

        self.logger = get_logger("orchav.metrics")
        self._plot_titles: dict[pg.PlotWidget, str] = {}
        self._styled_plots: list[pg.PlotWidget] = []
        # ScatterPlotItem caches rendered symbols by QBrush identity. Reusing
        # these brushes avoids rasterizing the same order colors per point on
        # every dashboard refresh.
        self._order_brush_cache: tuple[QBrush, ...] = tuple(
            pg.mkBrush(color) for color in self._ORDER_COLORS
        )
        self._order_fallback_brush = pg.mkBrush(self._ORDER_FALLBACK_COLOR)
        self._refresh_theme_tokens()

        cw = QWidget(self)
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title = QLabel("MPC Analysis Dashboard")
        title.setFont(_default_ui_font(13, weight=QFont.Weight.Bold))
        title.setAlignment(Qt.AlignCenter)
        self.title_label = title
        root.addWidget(title)

        self.context_label = QLabel("Frame - | TX: - | RX: - | MPCs: -")
        self.context_label.setAlignment(Qt.AlignCenter)
        self.context_label.setFont(_default_ui_font(9))
        root.addWidget(self.context_label)

        toolbar = QWidget(self)
        toolbar_layout = QGridLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setHorizontalSpacing(8)
        toolbar_layout.setVerticalSpacing(4)

        self.pause_cb = QCheckBox("Pause")
        self.refresh_btn = QPushButton("Refresh")
        self.auto_range_cb = QCheckBox("Adaptive axes")
        self.auto_range_cb.setChecked(True)
        self.auto_range_cb.setToolTip(
            "Checked: Overview X/Y ranges follow the displayed frame. "
            "Unchecked: keep the current X/Y ranges until you change them manually. "
            "This does not scan other scenario frames."
        )
        self.log_y_cb = QCheckBox("Log Y")
        self.materials_tab_cb = QCheckBox("Materials")
        self.materials_tab_cb.setChecked(True)
        self.angle_combo = QComboBox()
        self.angle_combo.addItem("AoD Az", "aod_az")
        self.angle_combo.addItem("AoD El", "aod_el")
        self.angle_combo.addItem("AoA Az", "aoa_az")
        self.angle_combo.addItem("AoA El", "aoa_el")
        self.angular_map_combo = QComboBox()
        self.angular_map_combo.addItem("AoD Map", "aod")
        self.angular_map_combo.addItem("AoA Map", "aoa")
        self.pair_metric_combo = QComboBox()
        self.pair_metric_combo.addItem("Selected MPC Count", "count")
        self.pair_metric_combo.addItem("Strongest Loss", "strongest_loss")
        self.pair_metric_combo.addItem("Mean Delay", "mean_delay")
        self.plot_point_limit_spin = QSpinBox()
        self.plot_point_limit_spin.setRange(
            self._MIN_PLOT_POINT_LIMIT,
            self._MAX_PLOT_POINT_LIMIT,
        )
        self.plot_point_limit_spin.setSingleStep(500)
        self.plot_point_limit_spin.setValue(self._DEFAULT_PLOT_POINT_LIMIT)
        self.plot_point_limit_spin.setAccelerated(True)
        self.plot_point_limit_spin.setSpecialValueText("No limit")
        self.plot_point_limit_spin.setToolTip(
            "Maximum markers drawn in dense scatter/profile views. "
            "Choose No limit to draw every marker. Statistics and CSV export "
            "always use every selected path."
        )
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(0, 60)
        self.update_rate_spin.setSpecialValueText("Maximum (adaptive)")
        self.update_rate_spin.setSuffix(" Hz")
        self.update_rate_spin.setValue(max(0, min(60, int(update_hz))))
        self.update_rate_spin.setToolTip(
            "Maximum adapts to the measured computation and drawing cost so the "
            "visualizer remains responsive. Intermediate Metrics frames may be "
            "skipped, while fixed rates remain upper refresh limits."
        )
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_png_btn = QPushButton("Export PNG")

        primary_controls = (
            self.pause_cb,
            self.refresh_btn,
            self.auto_range_cb,
            self.log_y_cb,
            QLabel("Tabs:"),
            self.materials_tab_cb,
            QLabel("Angle:"),
            self.angle_combo,
            QLabel("Map:"),
            self.angular_map_combo,
        )
        for column, widget in enumerate(primary_controls):
            toolbar_layout.addWidget(widget, 0, column)
        toolbar_layout.setColumnStretch(len(primary_controls), 1)

        secondary_controls = (
            QLabel("Pair metric:"),
            self.pair_metric_combo,
            QLabel("Max plotted markers:"),
            self.plot_point_limit_spin,
            QLabel("Update rate:"),
            self.update_rate_spin,
            self.export_csv_btn,
            self.export_png_btn,
        )
        for column, widget in enumerate(secondary_controls):
            toolbar_layout.addWidget(widget, 1, column)
        root.addWidget(toolbar)

        self.pause_cb.toggled.connect(self._on_pause_toggled)
        self.refresh_btn.clicked.connect(self._manual_refresh)
        self.auto_range_cb.toggled.connect(self._mark_pending_refresh)
        self.log_y_cb.toggled.connect(self._apply_log_y)
        self.materials_tab_cb.toggled.connect(self._on_materials_tab_toggled)
        self.angle_combo.currentIndexChanged.connect(self._mark_pending_refresh)
        self.angular_map_combo.currentIndexChanged.connect(self._mark_pending_refresh)
        self.pair_metric_combo.currentIndexChanged.connect(self._mark_pending_refresh)
        self.plot_point_limit_spin.valueChanged.connect(self._mark_pending_refresh)
        self.update_rate_spin.valueChanged.connect(self._on_update_rate_changed)
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.export_png_btn.clicked.connect(self._export_png)

        pg.setConfigOptions(antialias=True)
        self.tabs = QTabWidget(self)

        overview_tab = QWidget(self.tabs)
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(6)
        grid = QGridLayout()
        grid.setSpacing(8)

        def style_plot(pw: pg.PlotWidget, title_text: str) -> None:
            """Apply the dashboard's shared pyqtgraph plot styling."""
            self._register_plot(pw, title_text)

        self.delay_plot = pg.PlotWidget()
        style_plot(self.delay_plot, "Delay Distribution")
        self.delay_bars = pg.BarGraphItem(
            x=[],
            height=[],
            width=0.8,
            brush=pg.mkBrush("#4a90d9"),
            pen=pg.mkPen("#2c5f8a", width=0.5),
        )
        self.delay_plot.addItem(self.delay_bars)
        self.delay_plot.setLabel("bottom", "Delay", units="ns", **self._label_style)
        self.delay_plot.setLabel("left", "Count", **self._label_style)
        grid.addWidget(self.delay_plot, 0, 0)

        self.angle_plot = pg.PlotWidget()
        style_plot(self.angle_plot, "AoD Azimuth Distribution")
        self.angle_bars = pg.BarGraphItem(
            x=[],
            height=[],
            width=0.8,
            brush=pg.mkBrush("#e68a3c"),
            pen=pg.mkPen("#b06020", width=0.5),
        )
        self.angle_plot.addItem(self.angle_bars)
        self.angle_plot.setLabel("bottom", "AoD Azimuth", units="deg", **self._label_style)
        self.angle_plot.setLabel("left", "Count", **self._label_style)
        grid.addWidget(self.angle_plot, 0, 1)

        self.loss_plot = pg.PlotWidget()
        style_plot(self.loss_plot, "Path Loss Distribution")
        self.loss_bars = pg.BarGraphItem(
            x=[],
            height=[],
            width=0.8,
            brush=pg.mkBrush("#d95050"),
            pen=pg.mkPen("#8b2020", width=0.5),
        )
        self.loss_plot.addItem(self.loss_bars)
        self.loss_plot.setLabel("bottom", "Path Loss", units="dB", **self._label_style)
        self.loss_plot.setLabel("left", "Count", **self._label_style)
        grid.addWidget(self.loss_plot, 1, 0)

        self.binned_pdp_plot = pg.PlotWidget()
        style_plot(
            self.binned_pdp_plot,
            "Power Delay Profile (1 ns resolution)",
        )
        self.binned_pdp_plot.setToolTip(
            "All currently selected paths are pooled. Paths within 1 ns are grouped, "
            "linear path gains are summed, and each marker is placed at the "
            "power-weighted delay. Select one TX/RX pair for a per-link profile. "
            "Phase is not used."
        )
        self.binned_pdp_stems = pg.BarGraphItem(
            x=[],
            height=[],
            width=0.01,
            brush=pg.mkBrush("#4ec9c9"),
            pen=pg.mkPen("#2a8a8a", width=0.5),
        )
        self.binned_pdp_plot.addItem(self.binned_pdp_stems)
        self.binned_pdp_dots = pg.ScatterPlotItem(
            size=6,
            brush=pg.mkBrush("#4ec9c9"),
            pen=pg.mkPen(None),
        )
        self.binned_pdp_plot.addItem(self.binned_pdp_dots)
        self.binned_pdp_plot.setLabel("bottom", "Delay", units="ns", **self._label_style)
        self.binned_pdp_plot.setLabel("left", "Summed path gain", units="dB", **self._label_style)
        grid.addWidget(self.binned_pdp_plot, 2, 0, 1, 2)

        self.order_plot = pg.PlotWidget()
        style_plot(self.order_plot, "Interaction Order Distribution")
        self.order_bars = pg.BarGraphItem(
            x=[],
            height=[],
            width=0.8,
            brush=pg.mkBrush("#5cb85c"),
            pen=pg.mkPen("#3a7a3a", width=0.5),
        )
        self.order_plot.addItem(self.order_bars)
        self.order_plot.setLabel("bottom", "Interaction Order", **self._label_style)
        self.order_plot.setLabel("left", "Count", **self._label_style)
        self.order_plot.setXRange(0, 6)
        x_ticks = [
            (0, "0"),
            (1, "1"),
            (2, "2"),
            (3, "3"),
            (4, "4"),
            (5, "5"),
            (6, "6+"),
        ]
        self.order_plot.getAxis("bottom").setTicks([x_ticks])
        grid.addWidget(self.order_plot, 1, 1)

        overview_layout.addLayout(grid)
        self.tabs.addTab(overview_tab, "Overview")
        self._overview_tab = overview_tab

        channel_tab = QWidget(self.tabs)
        channel_layout = QVBoxLayout(channel_tab)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(6)

        self.channel_summary_label = QLabel("No channel summary available")
        self.channel_summary_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.channel_summary_label.setFont(_default_ui_font(9))
        self.channel_summary_label.setWordWrap(True)
        self.channel_summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        channel_grid = QGridLayout()
        channel_grid.setSpacing(8)

        self.delay_power_plot = pg.PlotWidget()
        style_plot(self.delay_power_plot, "Delay-Power Scatter")
        self.delay_power_scatter = pg.ScatterPlotItem(size=7, pen=self._scatter_pen)
        self.delay_power_plot.addItem(self.delay_power_scatter)
        self.delay_power_plot.setLabel("bottom", "Delay", units="ns", **self._label_style)
        self.delay_power_plot.setLabel(
            "left", "Received power proxy", units="dB", **self._label_style
        )
        channel_grid.addWidget(self.delay_power_plot, 0, 0)

        self.angular_heatmap_plot = pg.PlotWidget()
        style_plot(self.angular_heatmap_plot, "AoD Polar Direction Map")
        self.angular_heatmap = pg.ImageItem()
        self.angular_heatmap_plot.addItem(self.angular_heatmap)
        self.angular_polar_scatter = pg.ScatterPlotItem(
            size=7,
            pen=self._scatter_pen,
        )
        self.angular_heatmap_plot.addItem(self.angular_polar_scatter)
        self.angular_heatmap_plot.setAspectLocked(True)
        self.angular_heatmap_plot.setLabel(
            "bottom", "cos(az) x elevation radius", **self._label_style
        )
        self.angular_heatmap_plot.setLabel(
            "left", "sin(az) x elevation radius", **self._label_style
        )
        self._angular_guide_items = []
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        for radius in (0.25, 0.5, 0.75, 1.0):
            guide = pg.PlotDataItem(
                radius * np.cos(theta),
                radius * np.sin(theta),
                pen=self._guide_pen,
            )
            self.angular_heatmap_plot.addItem(guide)
            self._angular_guide_items.append(guide)
        channel_grid.addWidget(self.angular_heatmap_plot, 0, 1)

        self.pair_count_panel = QWidget()
        pair_count_layout = QVBoxLayout(self.pair_count_panel)
        pair_count_layout.setContentsMargins(6, 6, 6, 6)
        pair_count_layout.setSpacing(4)
        self.pair_count_title_label = QLabel("Selected MPCs by TX/RX Pair")
        self.pair_count_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pair_count_title_label.setFont(_default_ui_font(10, weight=QFont.Weight.Bold))
        pair_count_layout.addWidget(self.pair_count_title_label)

        self.pair_count_model = _PairCountModel(self)
        self.pair_count_table = QTableView()
        self.pair_count_table.setModel(self.pair_count_model)
        self.pair_count_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pair_count_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.pair_count_table.setAlternatingRowColors(True)
        self.pair_count_table.verticalHeader().setVisible(False)
        self.pair_count_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.pair_count_table.setToolTip(
            "Exact selected-path count for each represented TX/RX pair."
        )
        pair_count_layout.addWidget(self.pair_count_table)

        self.pair_heatmap_plot = pg.PlotWidget()
        style_plot(self.pair_heatmap_plot, "TX/RX Pair Overview")
        self.pair_heatmap = pg.ImageItem(axisOrder="row-major")
        self.pair_heatmap_plot.addItem(self.pair_heatmap)
        self._pair_color_map = pg.colormap.get("viridis")
        self.pair_colorbar = pg.ColorBarItem(
            values=(0.0, 1.0),
            colorMap=self._pair_color_map,
            interactive=False,
            colorMapMenu=False,
            width=15,
        )
        self.pair_colorbar.setImageItem(
            self.pair_heatmap,
            insert_in=self.pair_heatmap_plot.getPlotItem(),
        )
        self.pair_colorbar.setVisible(False)
        self.pair_heatmap_plot.setAspectLocked(True)
        self.pair_heatmap_plot.setLabel("bottom", "TX", **self._label_style)
        self.pair_heatmap_plot.setLabel("left", "RX", **self._label_style)

        self.pair_stack = QStackedWidget()
        self.pair_stack.addWidget(self.pair_count_panel)
        self.pair_stack.addWidget(self.pair_heatmap_plot)
        channel_grid.addWidget(self.pair_stack, 1, 0)
        channel_grid.addWidget(self.channel_summary_label, 1, 1)
        channel_grid.setColumnStretch(0, 1)
        channel_grid.setColumnStretch(1, 1)
        channel_grid.setRowStretch(0, 3)
        channel_grid.setRowStretch(1, 2)

        channel_layout.addLayout(channel_grid, stretch=1)
        self.tabs.addTab(channel_tab, "Channel")
        self._channel_tab = channel_tab

        materials_tab = QWidget(self.tabs)
        materials_layout = QGridLayout(materials_tab)
        materials_layout.setContentsMargins(0, 0, 0, 0)
        materials_layout.setSpacing(8)

        self.material_table_model = _MaterialBreakdownModel(self)
        self.material_table_proxy = QSortFilterProxyModel(self)
        self.material_table_proxy.setSourceModel(self.material_table_model)
        self.material_table_proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.material_table = QTableView()
        self.material_table.setModel(self.material_table_proxy)
        self.material_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.material_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.material_table.setSortingEnabled(True)
        self.material_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.material_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        materials_layout.addWidget(self.material_table, 0, 0, 1, 2)

        self.material_depth_plot = pg.PlotWidget()
        style_plot(self.material_depth_plot, "Material Hits by Bounce Depth")
        self.material_depth_heatmap = pg.ImageItem(axisOrder="row-major")
        self.material_depth_plot.addItem(self.material_depth_heatmap)
        self._material_depth_color_map = pg.colormap.get("viridis")
        self.material_depth_colorbar = pg.ColorBarItem(
            values=(0.0, 1.0),
            colorMap=self._material_depth_color_map,
            interactive=False,
            colorMapMenu=False,
            width=15,
        )
        self.material_depth_colorbar.setImageItem(
            self.material_depth_heatmap,
            insert_in=self.material_depth_plot.getPlotItem(),
        )
        self.material_depth_colorbar.setLabel("right", "Hit count")
        self.material_depth_colorbar.setVisible(False)
        self.material_depth_plot.setToolTip(
            "Interaction hit count by material and bounce depth; blank cells have zero hits."
        )
        self.material_depth_plot.setLabel("bottom", "Bounce depth", **self._label_style)
        self.material_depth_plot.setLabel("left", "Material", **self._label_style)
        materials_layout.addWidget(self.material_depth_plot, 1, 0)

        self.material_power_plot = pg.PlotWidget()
        style_plot(self.material_power_plot, "Associated Path-Gain Proxy by Material")
        self.material_power_bars = pg.BarGraphItem(
            x=[],
            height=[],
            width=0.8,
            brush=pg.mkBrush("#7b6ed6"),
            pen=pg.mkPen("#4d4399", width=0.5),
        )
        self.material_power_plot.addItem(self.material_power_bars)
        self.material_power_plot.setLabel("bottom", "Material", **self._label_style)
        self.material_power_plot.setLabel(
            "left", "Associated path-gain proxy", units="dB", **self._label_style
        )
        materials_layout.addWidget(self.material_power_plot, 1, 1)
        materials_layout.setColumnStretch(0, 1)
        materials_layout.setColumnStretch(1, 1)
        materials_layout.setRowStretch(0, 1)
        materials_layout.setRowStretch(1, 3)

        self.tabs.addTab(materials_tab, "Materials")
        self._materials_tab = materials_tab
        self.tabs.currentChanged.connect(self._mark_pending_refresh)

        root.addWidget(self.tabs, stretch=1)

        self.data_status_label = QLabel("")
        self.data_status_label.setAlignment(Qt.AlignCenter)
        self.data_status_label.setFont(_default_ui_font(8))
        root.addWidget(self.data_status_label)

        self.stats_label = QLabel("No data available")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setFont(_default_ui_font(9))
        root.addWidget(self.stats_label)

        self._last_vm = None
        self._last_stats = None
        self._last_context: dict[str, Any] = {}
        self._frame_stats_provider = frame_stats_provider
        self._enqueue_revision = 0
        self._stats_revision = -1
        self._plot_messages: list[str] = []
        self._pending_refresh = False
        self._refresh_in_progress = False
        self._clock = perf_counter
        self._last_refresh_started_s: float | None = None
        self._last_refresh_finished_s: float | None = None
        self._last_refresh_duration_ms = 0.0
        self._catalog_source: Any = None
        self._catalog_instance: MpcPathCatalog | None = None
        self._catalog_attempted = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._periodic_refresh)
        self._apply_log_y()
        self._apply_theme()
        get_theme_manager().theme_changed.connect(self._apply_theme)

        self.logger.info("Metrics window initialized (selection-scoped tabbed dashboard)")

    def _refresh_theme_tokens(self) -> None:
        """Refresh cached pyqtgraph styling tokens from the app theme."""
        tokens = pyqtgraph_theme()
        self._title_style = tokens.title_style
        self._label_style = tokens.label_style
        self._plot_bg_color = tokens.background
        self._axis_pen = tokens.axis_pen
        self._axis_text_pen = tokens.axis_text_pen
        self._plot_border_pen = tokens.border_pen
        self._scatter_pen = tokens.scatter_pen
        self._guide_pen = tokens.guide_pen

    def _register_plot(self, plot: pg.PlotWidget, title_text: str) -> None:
        """Register and style a plot that should follow the application theme."""
        self._plot_titles[plot] = title_text
        if plot not in self._styled_plots:
            self._styled_plots.append(plot)
        self._style_plot(plot)

    def _style_plot(self, plot: pg.PlotWidget) -> None:
        """Apply current theme tokens to one pyqtgraph plot."""
        apply_pyqtgraph_plot_theme(
            plot,
            title=self._plot_titles.get(plot, ""),
            tokens=pyqtgraph_theme(),
        )

    def _framed_label_style(self, *, padding: int) -> str:
        """Return a theme-aware label frame used by dashboard summaries."""
        theme = current_theme()
        return (
            "QLabel { "
            f"background-color: {theme.bg_primary}; "
            f"color: {theme.text_primary}; "
            f"border: 1px solid {theme.border_primary}; "
            "border-radius: 4px; "
            f"padding: {padding}px; "
            "}"
        )

    def _apply_theme(self, *_args) -> None:
        """Apply the current application theme to the metrics dashboard."""
        self._refresh_theme_tokens()
        theme = current_theme()
        pg.setConfigOption("background", self._plot_bg_color)
        pg.setConfigOption("foreground", theme.text_primary)

        self.title_label.setStyleSheet(f"color: {theme.text_primary}; margin-bottom: 2px;")
        self.context_label.setStyleSheet(self._framed_label_style(padding=6))
        self.channel_summary_label.setStyleSheet(self._framed_label_style(padding=8))
        self.pair_count_title_label.setStyleSheet(f"color: {theme.text_primary};")
        self.data_status_label.setStyleSheet(f"color: {theme.text_secondary};")
        self.stats_label.setStyleSheet(self._framed_label_style(padding=10))

        for plot in self._styled_plots:
            self._style_plot(plot)

        for scatter_attr in ("delay_power_scatter", "angular_polar_scatter"):
            scatter = getattr(self, scatter_attr, None)
            if scatter is not None:
                scatter.setPen(self._scatter_pen)

        for guide in getattr(self, "_angular_guide_items", []):
            guide.setPen(self._guide_pen)

        self.pair_heatmap_plot.setBackground(theme.bg_tertiary)
        self.material_depth_plot.setBackground(theme.bg_tertiary)
        self._style_colorbar(self.pair_colorbar)
        self._style_colorbar(self.material_depth_colorbar)

    def _style_colorbar(self, colorbar: pg.ColorBarItem) -> None:
        """Apply dashboard axis styling while preserving a colorbar's label."""
        axis = colorbar.getAxis("right")
        axis.setPen(self._axis_pen)
        axis.setTextPen(self._axis_text_pen)
        if axis.labelText:
            axis.setLabel(
                axis.labelText,
                units=axis.labelUnits or None,
                **self._label_style,
            )

    # Public API

    @property
    def updates_paused(self) -> bool:
        """Return whether new frame statistics should remain frozen."""
        return bool(self.pause_cb.isChecked())

    def enqueue(self, view_model, frame_stats=None, context: dict[str, Any] | None = None):
        """Replace the pending dashboard input with the latest frame."""
        if view_model is None:
            return

        self._enqueue_revision += 1
        self._last_vm = view_model
        self._last_stats = frame_stats
        self._last_context = context or {}
        if frame_stats is not None or self._frame_stats_provider is None:
            self._stats_revision = self._enqueue_revision

        if self.pause_cb.isChecked():
            self.data_status_label.setText("Paused; new frame statistics are not computed.")
            return

        self._pending_refresh = True
        self._arm_refresh_timer()

    # Refresh

    def _refresh(self):
        """Compute the latest statistics and redraw only the visible tab."""
        if self._last_vm is None:
            self._clear_plots()
            return

        try:
            self._compute_latest_frame_stats()
            self._plot_messages = []
            self._update_context_label()
            self._update_summary()
            active_tab = self.tabs.currentWidget()
            if active_tab is self._overview_tab:
                self._update_delay_histogram()
                self._update_loss_histogram()
                self._update_order_distribution()
                self._update_binned_pdp()
                self._update_angle_distribution()
            elif active_tab is self._channel_tab:
                self._update_channel_summary()
                self._update_delay_power_scatter()
                self._update_angular_heatmap()
                self._update_pair_heatmap()
            elif active_tab is self._materials_tab:
                if self.materials_tab_cb.isChecked():
                    self._update_material_breakdown()
                else:
                    self._clear_material_plots()
                    self._note_plot_message("Materials tab disabled.")
            self._update_data_status()
            self.logger.debug("Metrics window refreshed")
        except (ValueError, TypeError, ZeroDivisionError, AttributeError) as exc:
            self.logger.error("Error refreshing metrics: %s", exc)
            self._clear_plots()

    def _compute_latest_frame_stats(self) -> None:
        """Compute provider-backed statistics once for the newest queued frame."""
        if self._frame_stats_provider is None or self._stats_revision == self._enqueue_revision:
            return
        try:
            self._last_stats = self._frame_stats_provider(self._last_vm)
        except Exception as exc:  # noqa: BLE001 - optional metrics must remain isolated.
            self.logger.error("Error computing dashboard statistics: %s", exc)
            self._last_stats = None
        self._stats_revision = self._enqueue_revision

    def _periodic_refresh(self):
        """Consume the newest pending frame without replaying skipped frames."""
        self._timer.stop()
        if self._last_vm is None:
            return
        if self.pause_cb.isChecked():
            self._pending_refresh = False
            return
        if not self._pending_refresh:
            return
        self._pending_refresh = False
        self._run_timed_refresh()

    def _manual_refresh(self, *_args) -> None:
        """Refresh immediately and consume any already-scheduled redraw."""
        self._timer.stop()
        self._pending_refresh = False
        self._run_timed_refresh()

    def _run_timed_refresh(self) -> None:
        """Measure one refresh so later scheduling can preserve UI time."""
        self._refresh_in_progress = True
        started_s = self._clock()
        try:
            self._refresh()
        finally:
            finished_s = self._clock()
            self._last_refresh_started_s = started_s
            self._last_refresh_finished_s = finished_s
            self._last_refresh_duration_ms = max(0.0, (finished_s - started_s) * 1000.0)
            self._refresh_in_progress = False
            self._arm_refresh_timer()

    def _arm_refresh_timer(self) -> None:
        """Schedule one refresh for the latest pending dashboard input."""
        if (
            not self._pending_refresh
            or self._last_vm is None
            or self.pause_cb.isChecked()
            or self._refresh_in_progress
            or self._timer.isActive()
        ):
            return
        self._timer.start(self._next_refresh_delay_ms())

    def _adaptive_cooldown_ms(self) -> int:
        """Return a UI breathing interval based on the preceding refresh cost."""
        return max(
            self._MINIMUM_ADAPTIVE_COOLDOWN_MS,
            math.ceil(self._last_refresh_duration_ms),
        )

    def _fixed_refresh_period_ms(self) -> int | None:
        """Return the selected fixed-rate period, or ``None`` for Maximum."""
        update_hz = int(self.update_rate_spin.value())
        if update_hz <= 0:
            return None
        return max(1, round(1000.0 / update_hz))

    def _next_refresh_delay_ms(self, *, now_s: float | None = None) -> int:
        """Return a non-catching-up delay for the current refresh policy."""
        period_ms = self._fixed_refresh_period_ms()
        if self._last_refresh_finished_s is None:
            return 0 if period_ms is None else period_ms
        if now_s is None:
            now_s = self._clock()

        elapsed_since_finish_ms = (now_s - self._last_refresh_finished_s) * 1000.0
        responsive_delay_ms = self._adaptive_cooldown_ms() - elapsed_since_finish_ms
        if period_ms is None:
            remaining_ms = responsive_delay_ms
        else:
            assert self._last_refresh_started_s is not None
            elapsed_since_start_ms = (now_s - self._last_refresh_started_s) * 1000.0
            fixed_rate_delay_ms = period_ms - elapsed_since_start_ms
            remaining_ms = max(responsive_delay_ms, fixed_rate_delay_ms)
        return max(0, math.ceil(remaining_ms))

    def _mark_pending_refresh(self, *_):
        """Schedule a coalesced refresh when dashboard inputs change."""
        if self._last_vm is None:
            return
        self._pending_refresh = True
        self._arm_refresh_timer()

    def _on_materials_tab_toggled(self, enabled: bool) -> None:
        """Refresh or clear material plots when their tab is enabled/disabled."""
        if not enabled:
            self._clear_material_plots()
            self.data_status_label.setText("Materials tab disabled.")
        self._mark_pending_refresh()

    def _on_update_rate_changed(self, _update_hz: int) -> None:
        """Apply the user-selected dashboard computation and redraw cadence."""
        self._timer.stop()
        self._arm_refresh_timer()

    def _on_pause_toggled(self, paused: bool) -> None:
        """Stop automatic updates while keeping manual refresh available."""
        if paused:
            self._pending_refresh = False
            self._timer.stop()
            self.data_status_label.setText("Paused; new frame statistics are not computed.")
            return
        self._mark_pending_refresh()

    def _apply_log_y(self, *_):
        """Apply manual log-height rendering for count-style bar plots."""
        enabled = bool(self.log_y_cb.isChecked())
        for plot in (self.delay_plot, self.loss_plot, self.angle_plot, self.order_plot):
            # BarGraphItem heights are not transformed correctly by
            # pyqtgraph's log axis mode. The axis remains linear while displaying
            # log10(count) heights manually in the count-plot update methods.
            plot.setLogMode(y=False)
            label = "log10(Count + 1)" if enabled else "Count"
            plot.setLabel("left", label, **self._label_style)
        self._mark_pending_refresh()

    # Selected-path helpers

    def _canonical_data(self):
        """Return the canonical data attached to the latest ViewModel."""
        return getattr(self._last_vm, "canonical_data", None) if self._last_vm is not None else None

    @staticmethod
    def _array_size(arr: Any) -> int:
        """Return a tolerant size for ndarray-like or sequence inputs."""
        if arr is None:
            return 0
        if hasattr(arr, "size"):
            return int(arr.size)
        try:
            return len(arr)
        except TypeError:
            return 0

    def _catalog_for(self, canon: Any) -> MpcPathCatalog | None:
        """Return one lazy catalog, or ``None`` for incomplete canonical input."""
        if canon is None:
            return None
        if canon is not self._catalog_source:
            self._catalog_source = canon
            self._catalog_instance = None
            self._catalog_attempted = False
        if not self._catalog_attempted:
            self._catalog_attempted = True
            try:
                self._catalog_instance = MpcPathCatalog(canon)
            except MpcPathCatalogError:
                # Diagnostic payloads may omit catalog-only fields. The
                # tolerant helpers below can still report their resident data.
                self._catalog_instance = None
        return self._catalog_instance

    def _path_count(self, canon) -> int:
        """Return catalog path count, inferring it when catalog data is absent."""
        if canon is None:
            return 0
        catalog = self._catalog_for(canon)
        if catalog is not None:
            return catalog.path_count
        for name in ("path_orders", "path_delays", "path_losses", "path_tx", "path_rx"):
            arr = getattr(canon, name, None)
            if self._array_size(arr) > 0:
                return int(np.asarray(arr).shape[0])
        starts = getattr(canon, "path_start_indices", None)
        if self._array_size(starts) > 0:
            return int(np.asarray(starts).shape[0])
        path_id = getattr(canon, "path_id", None)
        if self._array_size(path_id) > 0:
            return int(np.nanmax(np.asarray(path_id))) + 1
        return 0

    def _effective_path_mask(self, canon, n_paths: int) -> np.ndarray:
        """Return the visible-path mask, falling back to all paths on mismatch."""
        if n_paths <= 0:
            return np.zeros(0, dtype=bool)
        mask = getattr(self._last_vm, "path_mask", None)
        if mask is None:
            return np.ones(n_paths, dtype=bool)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[0] != n_paths:
            self._note_plot_message("Selection mask did not match path count; using all paths.")
            return np.ones(n_paths, dtype=bool)
        return mask_arr

    def _path_start_indices(self, canon, n_paths: int) -> np.ndarray:
        """Return catalog point offsets, reconstructing absent catalog data."""
        starts = np.full(n_paths, -1, dtype=np.int64)
        if canon is None or n_paths <= 0:
            return starts
        catalog = self._catalog_for(canon)
        if catalog is not None and catalog.path_count == n_paths:
            return catalog.path_start_indices
        path_starts = getattr(canon, "path_start_indices", None)
        if self._array_size(path_starts) >= n_paths:
            values = np.asarray(path_starts[:n_paths], dtype=np.int64)
            starts[: values.shape[0]] = values
            return starts

        path_id = getattr(canon, "path_id", None)
        if self._array_size(path_id) > 0:
            ids = np.asarray(path_id, dtype=np.int64)
            for path_idx in range(n_paths):
                matches = np.flatnonzero(ids == path_idx)
                if matches.size:
                    starts[path_idx] = int(matches[0])
            return starts

        points = getattr(canon, "points", None)
        lines = getattr(canon, "lines", None)
        if points is None or not hasattr(points, "shape"):
            return starts
        n_pts = int(points.shape[0])
        is_start = np.ones(n_pts, dtype=bool)
        if lines is not None and self._array_size(lines) > 0:
            line_arr = np.asarray(lines)
            if line_arr.ndim == 2 and line_arr.shape[1] >= 2:
                is_start[line_arr[:, 1].astype(np.int64)] = False
        start_values = np.flatnonzero(is_start)[:n_paths]
        starts[: start_values.shape[0]] = start_values
        return starts

    def _path_array(self, path_field: str, point_field: str, n_paths: int) -> np.ndarray:
        """Return one path column, deriving it from point data when needed."""
        canon = self._canonical_data()
        values = np.full(n_paths, np.nan, dtype=float)
        if canon is None or n_paths <= 0:
            return values

        catalog = self._catalog_for(canon)
        catalog_attr = self._CATALOG_PATH_COLUMNS.get((path_field, point_field))
        if catalog is not None and catalog.path_count == n_paths and catalog_attr is not None:
            return np.asarray(getattr(catalog, catalog_attr))

        if path_field:
            arr = getattr(canon, path_field, None)
            if self._array_size(arr) >= n_paths:
                return np.asarray(arr[:n_paths], dtype=float)

        if not point_field:
            return values

        point_arr = getattr(canon, point_field, None)
        if self._array_size(point_arr) == 0:
            return values
        point_values = np.asarray(point_arr, dtype=float)
        starts = self._path_start_indices(canon, n_paths)
        valid = (starts >= 0) & (starts < point_values.shape[0])
        values[valid] = point_values[starts[valid]]
        return values

    def _selected_path_values(self, path_field: str, point_field: str) -> np.ndarray | None:
        """Return finite values for currently visible paths."""
        canon = self._canonical_data()
        n_paths = self._path_count(canon)
        if n_paths <= 0:
            return None
        values = self._path_array(path_field, point_field, n_paths)
        mask = self._effective_path_mask(canon, n_paths)
        selected = values[mask]
        selected = selected[np.isfinite(selected)]
        return selected if selected.size > 0 else None

    def _selected_path_indices(self) -> np.ndarray:
        """Return canonical path IDs that are visible in the dashboard."""
        canon = self._canonical_data()
        n_paths = self._path_count(canon)
        if n_paths <= 0:
            return np.empty((0,), dtype=np.int64)
        mask = self._effective_path_mask(canon, n_paths)
        return np.arange(n_paths, dtype=np.int64)[mask]

    def _selected_path_data(self) -> dict[str, np.ndarray]:
        """Return all dashboard path fields filtered to visible paths."""
        canon = self._canonical_data()
        n_paths = self._path_count(canon)
        if n_paths <= 0:
            empty = np.empty((0,), dtype=float)
            return {
                "path_id": np.empty((0,), dtype=np.int64),
                "tx": empty,
                "rx": empty,
                "order": empty,
                "delay": empty,
                "loss": empty,
                "aod_az": empty,
                "aod_el": empty,
                "aoa_az": empty,
                "aoa_el": empty,
            }
        mask = self._effective_path_mask(canon, n_paths)
        return {
            "path_id": np.arange(n_paths, dtype=np.int64)[mask],
            "tx": self._path_array("path_tx", "tx_id", n_paths)[mask],
            "rx": self._path_array("path_rx", "rx_id", n_paths)[mask],
            "order": self._path_array("path_orders", "", n_paths)[mask],
            "delay": self._path_array("path_delays", "delay", n_paths)[mask],
            "loss": self._path_array("path_losses", "loss", n_paths)[mask],
            "aod_az": self._path_array("", "aod_az", n_paths)[mask],
            "aod_el": self._path_array("", "aod_el", n_paths)[mask],
            "aoa_az": self._path_array("", "aoa_az", n_paths)[mask],
            "aoa_el": self._path_array("", "aoa_el", n_paths)[mask],
        }

    def _selected_path_rows(self) -> list[dict[str, Any]]:
        """Return selected-path records for CSV export."""
        canon = self._canonical_data()
        n_paths = self._path_count(canon)
        if n_paths <= 0:
            return []
        mask = self._effective_path_mask(canon, n_paths)
        path_ids = np.arange(n_paths)
        arrays = {
            "tx": self._path_array("path_tx", "tx_id", n_paths),
            "rx": self._path_array("path_rx", "rx_id", n_paths),
            "order": self._path_array("path_orders", "", n_paths),
            "delay_ns": self._path_array("path_delays", "delay", n_paths),
            "path_loss_db": self._path_array("path_losses", "loss", n_paths),
            "aod_az_deg": self._path_array("", "aod_az", n_paths),
            "aod_el_deg": self._path_array("", "aod_el", n_paths),
            "aoa_az_deg": self._path_array("", "aoa_az", n_paths),
            "aoa_el_deg": self._path_array("", "aoa_el", n_paths),
        }
        rows: list[dict[str, Any]] = []
        for idx in path_ids[mask]:
            row: dict[str, Any] = {"path_id": int(idx)}
            for name, values in arrays.items():
                value = values[idx]
                if np.isfinite(value):
                    row[name] = int(value) if name in {"tx", "rx", "order"} else float(value)
                else:
                    row[name] = ""
            rows.append(row)
        return rows

    def _finite_path_data(self, *field_names: str) -> dict[str, np.ndarray]:
        """Return selected-path data filtered to rows finite in given fields."""
        data = self._selected_path_data()
        if not field_names:
            return data
        mask = np.ones(data["path_id"].shape[0], dtype=bool)
        for name in field_names:
            values = data.get(name)
            if values is None:
                return {key: values[:0] for key, values in data.items()}
            mask &= np.isfinite(values)
        return {key: values[mask] for key, values in data.items()}

    def _sample_plot_data(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Downsample dense scatter inputs to the dashboard point limit."""
        count = data["path_id"].shape[0]
        limit = int(self.plot_point_limit_spin.value())
        if limit == 0 or count <= limit:
            return data
        indices = np.linspace(0, count - 1, limit, dtype=np.int64)
        return {key: values[indices] for key, values in data.items()}

    def _sample_profile_for_display(
        self,
        delays: np.ndarray,
        gains_db: np.ndarray,
        *,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound profile rendering without changing the computed statistics."""
        count = int(delays.size)
        limit = int(self.plot_point_limit_spin.value())
        if limit == 0 or count <= limit:
            return delays, gains_db
        indices = np.linspace(0, count - 1, limit, dtype=np.int64)
        self._note_plot_message(
            f"{label} displays {limit:,}/{count:,} sampled markers; "
            "statistics use all selected paths."
        )
        return delays[indices], gains_db[indices]

    def _material_name(self, material_id: int) -> str:
        """Resolve a canonical material ID to a display name."""
        if material_id <= 0:
            return ""
        canon = self._canonical_data()
        catalog = self._catalog_for(canon)
        if catalog is not None:
            name = catalog.material_name(material_id)
            if name:
                return name
        else:
            name_map = getattr(canon, "material_id_to_name", None) if canon is not None else None
            if name_map:
                name = str(name_map.get(int(material_id), "")).strip()
                if name and name.lower() != "none":
                    return name
        return f"material_{int(material_id)}"

    def _path_material_sequence(
        self,
        canon: Any,
        path_id: int,
        *,
        fallback_starts: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return one catalog material sequence or a point-data slice."""
        catalog = self._catalog_for(canon)
        if catalog is not None:
            return catalog.material_sequence(path_id)

        material_ids = getattr(canon, "material_ids", None)
        if self._array_size(material_ids) == 0:
            return None
        point_material_ids = np.asarray(material_ids, dtype=np.int64)
        starts = fallback_starts
        if starts is None:
            starts = self._path_start_indices(canon, self._path_count(canon))
        if path_id < 0 or path_id >= starts.shape[0] or starts[path_id] < 0:
            return None
        start = int(starts[path_id])
        if path_id + 1 < starts.shape[0] and starts[path_id + 1] > start:
            end = int(starts[path_id + 1])
        else:
            end = int(point_material_ids.shape[0])
        return point_material_ids[start + 1 : max(start + 1, end - 1)]

    def _selected_material_breakdown(
        self,
    ) -> tuple[list[dict[str, Any]], np.ndarray, list[str], list[int]]:
        """Aggregate visible-path material hits, power proxy, and bounce depth."""
        canon = self._canonical_data()
        data = self._selected_path_data()
        selected_path_ids = data["path_id"].astype(np.int64, copy=False)
        if canon is None or selected_path_ids.size == 0:
            return [], np.empty((0, 0), dtype=float), [], []

        if getattr(canon, "material_ids", None) is None:
            return [], np.empty((0, 0), dtype=float), [], []

        n_paths = self._path_count(canon)
        catalog = self._catalog_for(canon)
        fallback_starts = None if catalog is not None else self._path_start_indices(canon, n_paths)
        path_losses = self._path_array("path_losses", "loss", n_paths)
        estimated_loss = getattr(canon, "path_loss_is_estimated", None)
        if estimated_loss is None or self._array_size(estimated_loss) < n_paths:
            estimated_loss = np.zeros((n_paths,), dtype=bool)
        else:
            estimated_loss = np.asarray(estimated_loss[:n_paths], dtype=bool)
        records: dict[str, dict[str, Any]] = {}

        for path_idx in selected_path_ids:
            interior_ids = self._path_material_sequence(
                canon,
                int(path_idx),
                fallback_starts=fallback_starts,
            )
            if interior_ids is None or interior_ids.size == 0:
                continue
            path_loss = (
                path_losses[path_idx]
                if path_idx < path_losses.shape[0] and not estimated_loss[path_idx]
                else np.nan
            )
            power = 10 ** (-float(path_loss) / 10.0) if np.isfinite(path_loss) else 0.0
            for depth, material_id in enumerate(interior_ids, start=1):
                material_id = int(material_id)
                name = self._material_name(material_id)
                if not name:
                    continue
                rec = records.setdefault(
                    name,
                    {
                        "material": name,
                        "hits": 0,
                        "paths": set(),
                        "losses": [],
                        "power_linear": 0.0,
                        "depth_counts": {},
                    },
                )
                rec["hits"] += 1
                new_path = int(path_idx) not in rec["paths"]
                rec["paths"].add(int(path_idx))
                rec["depth_counts"][depth] = rec["depth_counts"].get(depth, 0) + 1
                if new_path:
                    rec["power_linear"] += power
                if new_path and np.isfinite(path_loss):
                    rec["losses"].append(float(path_loss))

        rows: list[dict[str, Any]] = []
        for rec in records.values():
            losses = rec["losses"]
            power_linear = float(rec["power_linear"])
            rows.append(
                {
                    "material": rec["material"],
                    "hits": int(rec["hits"]),
                    "path_count": len(rec["paths"]),
                    "power_db": (
                        float(10.0 * np.log10(max(power_linear, 1e-18)))
                        if power_linear > 0.0
                        else np.nan
                    ),
                    "mean_loss": float(np.mean(losses)) if losses else np.nan,
                    "strongest_loss": float(np.min(losses)) if losses else np.nan,
                    "depth_counts": rec["depth_counts"],
                }
            )
        rows.sort(key=lambda item: (-item["hits"], item["material"]))
        top_rows = rows[:10]
        max_depth = max(
            (max(row["depth_counts"].keys()) for row in top_rows if row["depth_counts"]),
            default=0,
        )
        depths = list(range(1, max_depth + 1))
        depth_matrix = np.zeros((len(top_rows), len(depths)), dtype=float)
        for row_idx, row in enumerate(top_rows):
            for depth_idx, depth in enumerate(depths):
                depth_matrix[row_idx, depth_idx] = row["depth_counts"].get(depth, 0)
        return top_rows, depth_matrix, [row["material"] for row in top_rows], depths

    def _pair_metric_matrix(
        self, metric: str | None = None
    ) -> tuple[np.ndarray, list[int], list[int], str]:
        """Build the selected TX/RX matrix for the chosen pair metric."""
        metric = metric or self.pair_metric_combo.currentData() or "count"
        data = self._finite_path_data("tx", "rx")
        if data["path_id"].size == 0:
            return np.empty((0, 0), dtype=float), [], [], "Selected MPCs by TX/RX Pair"

        tx_values = data["tx"].astype(int)
        rx_values = data["rx"].astype(int)
        tx_labels = sorted(set(tx_values.tolist()))
        rx_labels = sorted(set(rx_values.tolist()))
        tx_index = {value: idx for idx, value in enumerate(tx_labels)}
        rx_index = {value: idx for idx, value in enumerate(rx_labels)}
        matrix = np.full((len(rx_labels), len(tx_labels)), np.nan, dtype=float)

        if metric == "count":
            matrix[:] = 0.0
            for tx, rx in zip(tx_values, rx_values):
                matrix[rx_index[int(rx)], tx_index[int(tx)]] += 1.0
            title = "Selected MPCs by TX/RX Pair"
        elif metric == "strongest_loss":
            loss = data["loss"]
            for tx, rx, path_loss in zip(tx_values, rx_values, loss):
                if not np.isfinite(path_loss):
                    continue
                row = rx_index[int(rx)]
                col = tx_index[int(tx)]
                if not np.isfinite(matrix[row, col]) or path_loss < matrix[row, col]:
                    matrix[row, col] = float(path_loss)
            title = "TX/RX Pair Strongest Path Loss (dB)"
        else:
            delay = data["delay"]
            counts = np.zeros_like(matrix)
            matrix[:] = 0.0
            for tx, rx, path_delay in zip(tx_values, rx_values, delay):
                if not np.isfinite(path_delay):
                    continue
                row = rx_index[int(rx)]
                col = tx_index[int(tx)]
                matrix[row, col] += float(path_delay)
                counts[row, col] += 1.0
            averaged = np.full_like(matrix, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                np.divide(matrix, counts, out=averaged, where=counts > 0)
            matrix = averaged
            title = "TX/RX Pair Mean Delay (ns)"
        return matrix, tx_labels, rx_labels, title

    def _channel_summary_metrics(self) -> dict[str, Any]:
        """Compute compact channel KPIs for currently visible paths."""
        data = self._finite_path_data("loss")
        canon = self._canonical_data()
        path_ids = data["path_id"].astype(np.int64, copy=False)
        measured_loss = np.ones((path_ids.size,), dtype=bool)
        if canon is not None:
            n_paths = self._path_count(canon)
            flags = getattr(canon, "path_loss_is_estimated", None)
            if flags is not None and self._array_size(flags) >= n_paths:
                measured_loss &= ~np.asarray(flags[:n_paths], dtype=bool)[path_ids]
        data = {name: values[measured_loss] for name, values in data.items()}
        losses = data["loss"]
        if losses.size == 0:
            return {}
        with np.errstate(over="ignore", under="ignore"):
            powers = 10.0 ** (-losses / 10.0)
        usable = np.isfinite(powers) & (powers > 0.0)
        if not np.any(usable):
            return {}
        data = {name: values[usable] for name, values in data.items()}
        losses = data["loss"]
        powers = powers[usable]
        aggregate_path_gain = float(np.sum(powers))
        order = data["order"]
        direct_mask = np.isfinite(order) & (order.astype(int) == 0)
        direct_gain = float(np.sum(powers[direct_mask]))
        interacted_gain = float(np.sum(powers[~direct_mask]))
        delay_valid = np.isfinite(data["delay"])
        if canon is not None:
            n_paths = self._path_count(canon)
            flags = getattr(canon, "path_delay_is_estimated", None)
            if flags is not None and self._array_size(flags) >= n_paths:
                path_ids = data["path_id"].astype(np.int64, copy=False)
                delay_valid &= ~np.asarray(flags[:n_paths], dtype=bool)[path_ids]
        if np.any(delay_valid):
            delays = data["delay"][delay_valid]
            delay_powers = powers[delay_valid]
            mean_delay = float(np.average(delays, weights=delay_powers))
            first_delay = float(np.min(delays))
            mean_delay_after_earliest = mean_delay - first_delay
        else:
            mean_delay_after_earliest = np.nan
        if direct_gain > 0.0 and interacted_gain > 0.0:
            direct_to_interacted_gain_db = float(10.0 * np.log10(direct_gain / interacted_gain))
        elif direct_gain > 0.0:
            direct_to_interacted_gain_db = np.inf
        else:
            direct_to_interacted_gain_db = np.nan
        direct_count = int(np.count_nonzero(direct_mask))
        return {
            "paths": int(losses.size),
            "direct_count": direct_count,
            "interacted_count": int(losses.size - direct_count),
            "strongest_loss": float(np.min(losses)),
            "aggregate_path_gain_db": float(10.0 * np.log10(aggregate_path_gain)),
            "mean_delay_after_earliest_ns": mean_delay_after_earliest,
            "direct_to_interacted_gain_db": direct_to_interacted_gain_db,
        }

    # Individual plot updaters

    def _histogram_range(self, values: np.ndarray) -> tuple[float, float]:
        """Return a non-degenerate finite range for plot axes."""
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        if vmax <= vmin:
            pad = max(abs(vmin) * 0.01, 0.5)
            return vmin - pad, vmax + pad
        return vmin, vmax

    def _update_histogram(
        self,
        plot: pg.PlotWidget,
        bars: pg.BarGraphItem,
        values: np.ndarray | None,
        message_label: str,
    ) -> None:
        """Update one count histogram and its axis bounds."""
        if values is None or values.size == 0:
            bars.setOpts(x=[], height=[])
            self._note_plot_message(f"No {message_label} data for current selection.")
            return
        vmin, vmax = self._histogram_range(values)
        hist, bins = np.histogram(
            values, bins=min(DEFAULT_BINS, max(1, values.size)), range=(vmin, vmax)
        )
        centers = (bins[:-1] + bins[1:]) / 2
        bin_width = float(bins[1] - bins[0]) * 0.9 if bins.size > 1 else 0.8
        display_hist = self._count_heights(hist)
        bars.setOpts(x=centers, height=display_hist, width=bin_width)
        if self._adaptive_overview_axes_enabled():
            ymax = max(float(np.max(display_hist)) * 1.2, 1.0)
            plot.setYRange(0.0, ymax)
            plot.setXRange(vmin, vmax)

    def _update_delay_histogram(self):
        """Refresh the selected-path delay histogram."""
        delays = self._selected_path_values("path_delays", "delay")
        self._update_histogram(self.delay_plot, self.delay_bars, delays, "delay")

    def _update_loss_histogram(self):
        """Refresh the selected-path path-loss histogram."""
        losses = self._selected_path_values("path_losses", "loss")
        self._update_histogram(self.loss_plot, self.loss_bars, losses, "path loss")

    def _update_order_distribution(self):
        """Refresh reflection-order counts for visible paths."""
        stats = self._last_stats
        orders_hist = getattr(stats, "orders_hist", None) if stats is not None else None
        if not orders_hist:
            orders = self._selected_path_values("path_orders", "")
            if orders is not None:
                unique, counts = np.unique(orders.astype(int), return_counts=True)
                orders_hist = dict(zip(unique.astype(int), counts.astype(int)))
        if not orders_hist:
            self.order_bars.setOpts(x=[], height=[])
            self._note_plot_message("No interaction order data for current selection.")
            return
        full_orders = list(range(7))
        full_counts = [orders_hist.get(order, 0) for order in range(6)]
        full_counts.append(
            sum(int(count) for order, count in orders_hist.items() if int(order) >= 6)
        )
        display_counts = self._count_heights(np.asarray(full_counts, dtype=float))
        self.order_bars.setOpts(x=full_orders, height=display_counts)
        if self._adaptive_overview_axes_enabled():
            ymax = max(float(np.max(display_counts)) * 1.2, 1.0)
            self.order_plot.setYRange(0.0, ymax)
            self.order_plot.setXRange(-0.5, 6.5)

    def _update_binned_pdp(self):
        """Refresh the phase-free 1 ns power-delay profile."""
        stats = self._last_stats
        if stats is None or stats.binned_power_delay_profile is None:
            self.binned_pdp_stems.setOpts(x=[], height=[])
            self.binned_pdp_dots.setData([], [])
            self._note_plot_message("No measured delay/path-gain data for the current selection.")
            return
        delays_ns, powers_db = stats.binned_power_delay_profile
        delays = np.asarray(delays_ns, dtype=float)
        powers = np.asarray(powers_db, dtype=float)
        if delays.size == 0 or powers.size == 0:
            self.binned_pdp_stems.setOpts(x=[], height=[])
            self.binned_pdp_dots.setData([], [])
            self._note_plot_message("No measured delay/path-gain data for the current selection.")
            return
        display_delays, display_powers = self._sample_profile_for_display(
            delays,
            powers,
            label="Power Delay Profile",
        )
        floor = float(np.min(powers))
        dmin, dmax = self._histogram_range(delays)
        delay_span = max(dmax - dmin, 1.0)
        stem_width = max(delay_span * 0.002, 0.01)
        stem_heights = display_powers - floor
        self.binned_pdp_stems.setOpts(
            x=display_delays, height=stem_heights, y0=floor, width=stem_width
        )
        self.binned_pdp_dots.setData(x=display_delays, y=display_powers)
        if self._adaptive_overview_axes_enabled():
            ymin, ymax = self._histogram_range(powers)
            self.binned_pdp_plot.setYRange(ymin, ymax)
            self.binned_pdp_plot.setXRange(dmin, dmax)

    def _update_angle_distribution(self):
        """Refresh the selected AoA/AoD angle histogram."""
        field = self.angle_combo.currentData() or "aod_az"
        title, label = self._ANGLE_FIELDS.get(field, self._ANGLE_FIELDS["aod_az"])
        self.angle_plot.setTitle(f"{title} Distribution", **self._title_style)
        self.angle_plot.setLabel("bottom", label, units="deg", **self._label_style)
        angles = self._selected_path_values("", field)
        self._update_histogram(self.angle_plot, self.angle_bars, angles, label)

    def _update_summary(self):
        """Refresh the overview summary text."""
        stats = self._last_stats
        if stats is None:
            self.stats_label.setText("No statistics available")
            return
        parts = [f"MPCs: {getattr(stats, 'total_paths', 0)}"]
        dr = getattr(stats, "delay_range_ns", None)
        if dr:
            parts.append(f"Delay: {dr[0]:.1f}-{dr[1]:.1f} ns")
        plr = getattr(stats, "path_loss_range", None)
        if plr:
            parts.append(f"Loss: {plr[0]:.1f}-{plr[1]:.1f} dB")
        ds = getattr(stats, "delay_spread_ns", None)
        if ds is not None:
            parts.append(f"Pooled selected-path RMS delay spread: {ds:.1f} ns")
        ang = getattr(stats, "angular_spread_deg", None)
        if ang is not None:
            parts.append(f"AoD Az. Spread: {ang:.1f} deg")
        self.stats_label.setText("  |  ".join(parts))

    def _update_channel_summary(self) -> None:
        """Refresh channel KPI summary text."""
        metrics = self._channel_summary_metrics()
        if not metrics:
            self.channel_summary_label.setText("No channel summary available")
            return
        stats = self._last_stats
        parts = [
            f"Paths: {metrics['paths']}",
            f"Direct/interacted: {metrics['direct_count']}/{metrics['interacted_count']}",
            f"Strongest loss: {metrics['strongest_loss']:.1f} dB",
            f"Aggregate path gain: {metrics['aggregate_path_gain_db']:.1f} dB",
        ]
        mean_delay = metrics["mean_delay_after_earliest_ns"]
        if np.isfinite(mean_delay):
            parts.append(f"Mean delay after earliest: {mean_delay:.1f} ns")
        else:
            parts.append("Mean delay after earliest: N/A")
        gain_ratio = metrics["direct_to_interacted_gain_db"]
        if np.isfinite(gain_ratio):
            parts.append(f"Direct/interacted gain: {gain_ratio:.1f} dB")
        elif gain_ratio == np.inf:
            parts.append("Direct/interacted gain: direct only")
        else:
            parts.append("Direct/interacted gain: N/A")
        delay_spread = getattr(stats, "delay_spread_ns", None) if stats is not None else None
        if delay_spread is not None:
            parts.append(f"Pooled selected-path RMS delay spread: {delay_spread:.1f} ns")
        angular_spread = getattr(stats, "angular_spread_deg", None) if stats is not None else None
        if angular_spread is not None:
            parts.append(f"AoD azimuth spread: {angular_spread:.1f} deg")
        self.channel_summary_label.setText("Selected-path channel summary\n\n" + "\n".join(parts))

    def _order_brushes(self, orders: np.ndarray) -> list[QBrush]:
        """Return shared scatter brushes keyed by reflection order."""
        brushes: list[QBrush] = []
        for order in orders:
            key = int(order) if np.isfinite(order) else -1
            if 0 <= key < len(self._order_brush_cache):
                brushes.append(self._order_brush_cache[key])
            else:
                brushes.append(self._order_fallback_brush)
        return brushes

    def _update_delay_power_scatter(self) -> None:
        """Refresh the delay-vs-power scatter plot."""
        data = self._finite_path_data("delay", "loss")
        if data["path_id"].size == 0:
            self.delay_power_scatter.setData([], [])
            self._note_plot_message("No delay-power data for current selection.")
            return
        plot_data = self._sample_plot_data(data)
        delays = plot_data["delay"]
        power_db = -plot_data["loss"]
        orders = plot_data["order"]
        spots = [
            {
                "pos": (float(delay), float(power)),
                "brush": brush,
                "data": int(path_id),
            }
            for delay, power, brush, path_id in zip(
                delays, power_db, self._order_brushes(orders), plot_data["path_id"]
            )
        ]
        self.delay_power_scatter.setData(spots=spots)
        if plot_data["path_id"].shape[0] < data["path_id"].shape[0]:
            self._note_plot_message(
                f"Delay-power scatter sampled {plot_data['path_id'].shape[0]}/"
                f"{data['path_id'].shape[0]} paths."
            )
        xmin, xmax = self._histogram_range(data["delay"])
        ymin, ymax = self._histogram_range(-data["loss"])
        self.delay_power_plot.setXRange(xmin, xmax)
        self.delay_power_plot.setYRange(ymin, ymax)

    def _update_angular_heatmap(self) -> None:
        """Refresh the polar AoA/AoD direction scatter plot."""
        mode = self.angular_map_combo.currentData() or "aod"
        az_field = f"{mode}_az"
        el_field = f"{mode}_el"
        data = self._finite_path_data(az_field, el_field, "order")
        az = data[az_field]
        el = data[el_field]
        title_prefix = "AoD" if mode == "aod" else "AoA"
        self.angular_heatmap_plot.setTitle(
            f"{title_prefix} Polar Direction Map", **self._title_style
        )
        if az.size == 0 or el.size == 0:
            self.angular_heatmap.clear()
            self.angular_polar_scatter.setData([], [])
            self._note_plot_message(f"No {title_prefix} polar direction data.")
            return
        self.angular_heatmap.clear()
        plot_data = self._sample_plot_data(data)
        az_rad = np.radians(plot_data[az_field])
        # Radius is an elevation projection: +90 deg at center, -90 deg on
        # the outer ring.  This preserves direction while still showing
        # whether energy is above or below the local horizon.
        radius = np.clip((90.0 - plot_data[el_field]) / 180.0, 0.0, 1.0)
        x = radius * np.cos(az_rad)
        y = radius * np.sin(az_rad)
        spots = [
            {
                "pos": (float(px), float(py)),
                "brush": brush,
                "data": int(path_id),
            }
            for px, py, brush, path_id in zip(
                x,
                y,
                self._order_brushes(plot_data["order"]),
                plot_data["path_id"],
            )
        ]
        self.angular_polar_scatter.setData(spots=spots)
        if plot_data["path_id"].shape[0] < data["path_id"].shape[0]:
            self._note_plot_message(
                f"{title_prefix} polar map sampled {plot_data['path_id'].shape[0]}/"
                f"{data['path_id'].shape[0]} paths."
            )
        self.angular_heatmap_plot.setXRange(-1.05, 1.05)
        self.angular_heatmap_plot.setYRange(-1.05, 1.05)

    def _clear_pair_view(self) -> None:
        """Clear both pair encodings and select the requested empty page."""
        self.pair_count_model.set_rows([])
        self.pair_heatmap.clear()
        self.pair_colorbar.setVisible(False)
        self.pair_heatmap_plot.getAxis("bottom").setTicks([])
        self.pair_heatmap_plot.getAxis("left").setTicks([])
        metric = self.pair_metric_combo.currentData() or "count"
        target = self.pair_count_panel if metric == "count" else self.pair_heatmap_plot
        self.pair_stack.setCurrentWidget(target)

    def _update_pair_count_table(
        self,
        matrix: np.ndarray,
        tx_labels: list[int],
        rx_labels: list[int],
    ) -> None:
        """Show exact nonzero pair counts in stable TX-major order."""
        rows = [
            (tx, rx, int(matrix[rx_idx, tx_idx]))
            for tx_idx, tx in enumerate(tx_labels)
            for rx_idx, rx in enumerate(rx_labels)
            if matrix[rx_idx, tx_idx] > 0.0
        ]
        self.pair_count_model.set_rows(rows)
        self.pair_heatmap.clear()
        self.pair_colorbar.setVisible(False)
        self.pair_stack.setCurrentWidget(self.pair_count_panel)

    def _update_pair_heatmap(self) -> None:
        """Refresh the selected TX/RX pair chart for the chosen metric."""
        matrix, tx_labels, rx_labels, title = self._pair_metric_matrix()
        self._plot_titles[self.pair_heatmap_plot] = title
        self.pair_heatmap_plot.setTitle(title, **self._title_style)
        if matrix.size == 0:
            self._clear_pair_view()
            self._note_plot_message("No TX/RX pair data for current selection.")
            return
        metric = self.pair_metric_combo.currentData() or "count"
        if metric == "count":
            self._update_pair_count_table(matrix, tx_labels, rx_labels)
            return

        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            self._clear_pair_view()
            self._note_plot_message("No finite TX/RX pair metric values.")
            return
        self.pair_count_model.set_rows([])
        self.pair_stack.setCurrentWidget(self.pair_heatmap_plot)
        levels = self._set_image_data(self.pair_heatmap, matrix)
        if levels is None:
            self._note_plot_message("No finite TX/RX pair metric values.")
            return
        if metric == "strongest_loss":
            colorbar_label = "Strongest path loss"
            colorbar_units = "dB"
            tooltip = "Lowest measured path loss per represented pair; blank means no value."
        else:
            colorbar_label = "Mean path delay"
            colorbar_units = "ns"
            tooltip = (
                "Unweighted mean of measured MPC delays per represented pair; "
                "blank means no value."
            )
        self.pair_colorbar.setLevels(levels)
        self.pair_colorbar.setLabel(
            "right",
            colorbar_label,
            units=colorbar_units,
            **self._label_style,
        )
        self._style_colorbar(self.pair_colorbar)
        self.pair_colorbar.setVisible(True)
        self.pair_heatmap_plot.setToolTip(tooltip)
        self.pair_heatmap.setRect(QRectF(-0.5, -0.5, len(tx_labels), len(rx_labels)))
        self.pair_heatmap_plot.getAxis("bottom").setTicks(
            [[(idx, f"TX{tx + 1}") for idx, tx in enumerate(tx_labels)]]
        )
        self.pair_heatmap_plot.getAxis("left").setTicks(
            [[(idx, f"RX{rx + 1}") for idx, rx in enumerate(rx_labels)]]
        )
        self.pair_heatmap_plot.setLabel("bottom", "TX", **self._label_style)
        self.pair_heatmap_plot.setLabel("left", "RX", **self._label_style)
        self.pair_heatmap_plot.setXRange(-0.5, len(tx_labels) - 0.5)
        self.pair_heatmap_plot.setYRange(-0.5, len(rx_labels) - 0.5)

    def _update_material_breakdown(self) -> None:
        """Refresh material summary table, power bars, and depth heatmap."""
        rows, depth_matrix, material_labels, depths = self._selected_material_breakdown()
        self.material_table_model.set_rows(rows)

        if not rows:
            self._clear_material_plots()
            self._note_plot_message("No material data for current selection.")
            return

        power_rows = [row for row in rows if np.isfinite(row["power_db"])]
        if power_rows:
            power_values = np.asarray([row["power_db"] for row in power_rows], dtype=float)
            x_values = np.arange(len(power_rows), dtype=float)
            value_min = float(np.min(power_values))
            value_max = float(np.max(power_values))
            value_span = value_max - value_min
            padding = max(
                value_span * 0.05,
                max(abs(value_min), abs(value_max)) * 0.01,
                0.5,
            )
            ymin = value_min - padding
            ymax = value_max + padding
            self.material_power_bars.setOpts(
                x=x_values,
                height=power_values - ymin,
                y0=ymin,
                width=0.8,
            )
            self.material_power_plot.getAxis("bottom").setTicks(
                [[(idx, row["material"]) for idx, row in enumerate(power_rows)]]
            )
            self.material_power_plot.setXRange(-0.5, len(power_rows) - 0.5)
            self.material_power_plot.setYRange(ymin, ymax)
        else:
            self.material_power_bars.setOpts(x=[], height=[], y0=0.0)
            self.material_power_plot.getAxis("bottom").setTicks([])
            self._note_plot_message("No measured path-loss data for the selected material hits.")

        if depth_matrix.size == 0:
            self.material_depth_heatmap.clear()
            self.material_depth_colorbar.setVisible(False)
            self.material_depth_plot.getAxis("bottom").setTicks([])
            self.material_depth_plot.getAxis("left").setTicks([])
            return
        display_depth = np.where(depth_matrix > 0.0, depth_matrix, np.nan)
        levels = self._set_image_data(self.material_depth_heatmap, display_depth)
        if levels is None:
            self.material_depth_colorbar.setVisible(False)
            return
        self.material_depth_colorbar.setLevels(levels)
        self.material_depth_colorbar.setVisible(True)
        self.material_depth_heatmap.setRect(
            QRectF(0.5, -0.5, max(len(depths), 1), len(material_labels))
        )
        self.material_depth_plot.getAxis("bottom").setTicks([[(d, str(d)) for d in depths]])
        self.material_depth_plot.getAxis("left").setTicks(
            [[(idx, label) for idx, label in enumerate(material_labels)]]
        )
        self.material_depth_plot.setXRange(0.5, max(len(depths), 1) + 0.5)
        self.material_depth_plot.setYRange(-0.5, len(material_labels) - 0.5)

    # Status and export

    def _update_context_label(self) -> None:
        """Refresh frame, node-selection, and filter context text."""
        context = self._last_context or {}
        canon = self._canonical_data()
        total_paths = int(context.get("total_paths") or self._path_count(canon))
        visible_paths = context.get("visible_paths")
        if visible_paths is None:
            visible_paths = int(np.count_nonzero(self._effective_path_mask(canon, total_paths)))
        filters = context.get("filters") or []
        filter_text = ", ".join(filters) if filters else "none"
        frame_text = self._format_frame_label(context.get("step"))
        self.context_label.setText(
            f"{frame_text} | TX: {self._format_node_selection(context.get('selected_tx'), 'TX')} | "
            f"RX: {self._format_node_selection(context.get('selected_rx'), 'RX')} | "
            f"MPCs: {visible_paths}/{total_paths} | Filters: {filter_text}"
        )

    @staticmethod
    def _format_frame_label(step: Any) -> str:
        """Format a zero-based frame step for the dashboard header."""
        if step is None:
            return "Frame -"
        try:
            return f"Frame {int(step) + 1}"
        except (TypeError, ValueError):
            return f"Frame {step}"

    @staticmethod
    def _format_node_selection(value: Any, kind: str) -> str:
        """Format TX/RX selection values from state or filter context."""
        if value is None or value == "":
            return "-"
        if isinstance(value, (int, np.integer)):
            return f"{kind}{int(value) + 1}"
        text = str(value)
        if text.lower() == "all":
            return f"All {kind}"
        prefix = kind.lower() + "_"
        if text.lower().startswith(prefix):
            suffix = text[len(prefix) :]
            return f"{kind}{suffix}" if suffix else text
        return text

    def _update_data_status(self) -> None:
        """Show a compact set of plot data/status messages."""
        if self._plot_messages:
            unique_messages = list(dict.fromkeys(self._plot_messages))
            self.data_status_label.setText(" | ".join(unique_messages[:3]))
        else:
            self.data_status_label.setText("")

    def _note_plot_message(self, message: str) -> None:
        """Queue a plot status message for the next status-label update."""
        if message not in self._plot_messages:
            self._plot_messages.append(message)

    def _adaptive_overview_axes_enabled(self) -> bool:
        """Return whether Overview plots should follow the displayed frame."""
        return bool(self.auto_range_cb.isChecked())

    def _count_heights(self, counts: np.ndarray) -> np.ndarray:
        """Return linear or log-scaled bar heights for count plots."""
        values = np.asarray(counts, dtype=float)
        if not self.log_y_cb.isChecked():
            return values
        return np.log10(np.maximum(values, 0.0) + 1.0)

    @staticmethod
    def _set_image_data(
        image: pg.ImageItem,
        values: np.ndarray,
    ) -> tuple[float, float] | None:
        """Set finite heatmap levels and return the displayed range."""
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            image.clear()
            return None
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            pad = max(abs(vmin) * 0.01, 1.0)
            levels = (vmin - pad, vmax + pad)
        else:
            levels = (vmin, vmax)
        image.setImage(values, levels=levels)
        return levels

    @staticmethod
    def _format_optional_float(value: Any, suffix: str = "") -> str:
        """Format optional finite numeric values for table cells."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(numeric):
            return "-"
        return f"{numeric:.1f}{suffix}"

    def _export_csv(self) -> None:
        """Export visible path rows to CSV."""
        rows = self._selected_path_rows()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export MPC dashboard data",
            "mpc_dashboard_paths.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        fieldnames = [
            "path_id",
            "tx",
            "rx",
            "order",
            "delay_ns",
            "path_loss_db",
            "aod_az_deg",
            "aod_el_deg",
            "aoa_az_deg",
            "aoa_el_deg",
        ]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.data_status_label.setText(f"Exported {len(rows)} selected paths to CSV.")

    def _export_png(self) -> None:
        """Export the dashboard widget image to PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export MPC dashboard image",
            "mpc_dashboard.png",
            "PNG Images (*.png)",
        )
        if not path:
            return
        self.grab().save(path)
        self.data_status_label.setText("Exported dashboard PNG.")

    # Housekeeping

    def _clear_channel_plots(self) -> None:
        """Clear optional channel-analysis plots."""
        self.delay_power_scatter.setData([], [])
        self.angular_heatmap.clear()
        self.angular_polar_scatter.setData([], [])
        self._clear_pair_view()
        self.channel_summary_label.setText("No channel summary available")

    def _clear_material_plots(self) -> None:
        """Clear material-analysis table and plots."""
        self.material_depth_heatmap.clear()
        self.material_depth_colorbar.setVisible(False)
        self.material_depth_plot.getAxis("bottom").setTicks([])
        self.material_depth_plot.getAxis("left").setTicks([])
        self.material_power_bars.setOpts(x=[], height=[], y0=0.0)
        self.material_power_plot.getAxis("bottom").setTicks([])
        self.material_table_model.set_rows([])

    def _clear_plots(self):
        """Clear all dashboard plots and stop pending refresh work."""
        self.delay_bars.setOpts(x=[], height=[])
        self.loss_bars.setOpts(x=[], height=[])
        self.order_bars.setOpts(x=[], height=[])
        self.binned_pdp_stems.setOpts(x=[], height=[])
        self.binned_pdp_dots.setData([], [])
        self.angle_bars.setOpts(x=[], height=[])
        self._clear_channel_plots()
        self._clear_material_plots()
        self.context_label.setText("Frame - | TX: - | RX: - | MPCs: -")
        self.data_status_label.setText("")
        self.stats_label.setText("No data available")
        self._timer.stop()
        self._pending_refresh = False

    def closeEvent(self, event):
        """Log dashboard closure before Qt destroys the window."""
        self._timer.stop()
        self._pending_refresh = False
        self.logger.info("Metrics window closing")
        super().closeEvent(event)
