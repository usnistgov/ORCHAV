"""Lazy, locally styled pyqtgraph cards for whole-run path statistics.

The statistics provider remains the owner of all scientific reductions. These
views only bin samples or transform display coordinates, never playback state.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPicture
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QToolTip, QVBoxLayout

from ..app.plot_theme import dashboard_series_color
from ..app.theme import current_theme
from ..services.mpc_interaction_style_service import (
    mpc_interaction_label,
    mpc_interaction_sort_key,
)
from ..widgets.plot_lifecycle import ManagedPlotWidget, unregister_plot_views

BLUE = "#2865b0"
INK = "#172b4d"
MUTED = "#526784"
ORDER_COLORS = (BLUE, "#00866b", "#a66b00", "#b5475a", "#7053ad", "#078596", "#64748b")
SYMBOLS = ("o", "s", "t", "d", "+", "x", "star")
DASHES = (Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine)
MAX_CDF_POINTS = 5000


CHART_NOTES = {
    "delay_time_chart": "Incoherent path-power sum per delay bin · mean over observations sharing a time bin · blank: no usable data · click to inspect",
    "reflection_order_chart": "One count per retained path · orders 6 and above are pooled",
    "interaction_type_chart": "Direct paths are LoS; other paths use their first interaction",
    "path_loss_histogram_chart": "Valid retained paths · lower loss means a stronger path",
    "delay_histogram_chart": "Valid retained paths · propagation delay in nanoseconds",
    "delay_cdf_chart": "Fraction of valid retained paths at or below each delay",
    "path_loss_cdf_chart": "Fraction of valid retained paths at or below each loss",
    "pair_gain_cdf_chart": "One sample per usable pair-frame · incoherent path-power sum",
    "pair_delay_spread_cdf_chart": "One sample per usable pair-frame · power-weighted RMS",
    "aod_az_polar_chart": "Departure azimuth · 0° east, positive counterclockwise · path counts",
    "aod_el_polar_chart": "Departure elevation · each path contributes one count",
    "aoa_az_polar_chart": "Arrival azimuth · 0° east, positive counterclockwise · path counts",
    "aoa_el_polar_chart": "Arrival elevation · each path contributes one count",
    "mpc_evolution_chart": "All retained paths in each recorded frame",
    "delay_spread_trend_chart": "Power-weighted across co-valid paths in each frame",
    "mpc_order_evolution_hist_chart": "Exact counts · symmetric log scale with linear values near zero",
    "mpc_type_evolution_chart": "Initial interaction per path · all-zero categories omitted",
    "pair_visibility_chart": "Represented pair-frame path states; renderer visibility is unrelated",
    "strongest_path_loss_chart": "Lowest valid single-path loss per frame · lower is stronger",
    "coverage_distribution_chart": "Selected height slice · hover for exact cell counts and area",
    "coverage_success_chart": "Share of the full slice meeting the threshold; no-data cells do not qualify",
}


def transform(values, scale):
    """Map scientific values to display coordinates without changing source data."""
    values = np.asarray(values, dtype=float)
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(values), np.nan)
    if scale == "symlog":
        magnitude = np.abs(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sign(values) * np.where(
                magnitude <= 1, magnitude / 0.9, 1 / 0.9 + np.log10(magnitude)
            )
    return values


def inverse(value, scale):
    if scale == "log":
        return 10.0**value
    if scale == "symlog":
        return np.sign(value) * (
            abs(value) * 0.9 if abs(value) <= 1 / 0.9 else 10.0 ** (abs(value) - 1 / 0.9)
        )
    return value


class ScientificAxis(pg.AxisItem):
    """Show true values on manually transformed bar and curve coordinates."""

    def __init__(self, orientation):
        super().__init__(orientation=orientation)
        self.display_scale = "linear"
        self.enableAutoSIPrefix(False)

    def tickValues(self, minimum, maximum, size):
        if self.display_scale == "symlog":
            limit = min(15, max(1, int(max(abs(minimum), abs(maximum))) + 1))
            values = [0.0, 1.0, *[10.0**n for n in range(1, limit + 1)]]
            values = sorted(set(values + [-x for x in values]))
            return [
                (1.0, [float(x) for x in transform(values, "symlog") if minimum <= x <= maximum])
            ]
        if self.display_scale == "log":
            return [(1.0, list(np.arange(np.ceil(minimum), np.floor(maximum) + 1)))]
        return super().tickValues(minimum, maximum, size)

    def tickStrings(self, values, scale, spacing):
        if self.display_scale != "linear":
            return [f"{inverse(v, self.display_scale):,.4g}" for v in values]
        return super().tickStrings(values, scale, spacing)


class PolarRoseItem(pg.GraphicsObject):
    """A cached vector rose; radius is the exact count in each azimuth bin."""

    def __init__(self, edges, counts):
        super().__init__()
        self._edges, self._counts = edges, counts
        self.picture = QPicture()
        self.apply_theme()

    def apply_theme(self):
        self.prepareGeometryChange()
        self.picture = QPicture()
        painter = QPainter(self.picture)
        color = QColor(dashboard_series_color(BLUE))
        painter.setPen(pg.mkPen(color, width=0.7))
        color.setAlpha(155)
        painter.setBrush(pg.mkBrush(color))
        for left, right, count in zip(self._edges[:-1], self._edges[1:], self._counts):
            if count <= 0:
                continue
            gap = (right - left) * 0.05
            angles = np.linspace(left + gap, right - gap, 12)
            path = QPainterPath()
            path.moveTo(0, 0)
            for angle in angles:
                path.lineTo(float(count * np.cos(angle)), float(count * np.sin(angle)))
            path.closeSubpath()
            painter.drawPath(path)
        painter.end()
        self.update()

    def paint(self, painter, *_args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


class ResponsiveChartGrid(QFrame):
    """Reflow cards without rebuilding plot items or their data."""

    def __init__(self, cards):
        super().__init__()
        self.cards = cards
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 8, 0, 0)
        self.grid.setSpacing(12)
        self._columns = 0
        self.reflow()

    def reflow(self):
        columns = 2 if len(self.cards) > 1 and self.width() >= 640 else 1
        if columns == self._columns:
            return
        for index, card in enumerate(self.cards):
            self.grid.addWidget(card, index // columns, index % columns)
            card.setMinimumHeight(280 if columns == 1 else 250)
        for row in range(self.grid.rowCount()):
            self.grid.setRowStretch(row, int(row < (len(self.cards) + columns - 1) // columns))
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, int(columns == 2))
        self._columns = columns

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reflow()


class StatisticsPlotCard(QFrame):
    """A persistent chart host that allocates its plot only on first visibility."""

    frame_requested = Signal(int)

    def __init__(self, key, explanation, formatter, nearest):
        super().__init__()
        self.key, self._formatter, self._nearest = key, formatter, nearest
        self.setObjectName("pathPlotCard")
        self.setToolTip(explanation)
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot = None
        self.chart_metadata = None
        self._rendered = None
        self.render_count = 0
        self._x_scale = self._y_scale = "linear"
        self._series = []
        self._theme_updates = []
        self._colorbar = None
        self._grid_visible = False
        self._grid_items = []
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(12, 10, 12, 9)
        self._box.setSpacing(5)
        self.title = QLabel(key.replace("_chart", "").replace("_", " ").title())
        self.title.setObjectName("pathPlotTitle")
        self.title.setWordWrap(True)
        self._box.addWidget(self.title)
        self.message = QLabel("No usable samples for this chart")
        self.message.setObjectName("statisticsChartMessage")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setToolTip(explanation)
        self._box.addWidget(self.message, 1)
        self.note = QLabel(CHART_NOTES[key])
        self.note.setObjectName("pathPlotNote")
        self.note.setWordWrap(True)
        self.note.setToolTip(explanation)
        self._box.addWidget(self.note)

    def sizeHint(self):
        return QSize(460, 280)

    def set_chart(self, chart_type, data, title, message=None):
        self.chart_metadata = {"chart_type": chart_type, "data": dict(data), "title": title}
        self.title.setText(title)
        self._message = message
        if self.isVisible():
            self.render_pending()

    def clear_chart(self):
        self.chart_metadata = self._rendered = None
        self._theme_updates = []
        self._grid_items = []
        self.message.setText("No usable samples for this chart")
        self.message.show()
        if self.plot is not None:
            self.plot.hide()

    def set_grid_visible(self, visible: bool) -> None:
        """Restyle guides without allocating hidden plots or recomputing chart data."""
        self._grid_visible = bool(visible)
        if self.plot is None:
            return
        polar = (
            self.chart_metadata is not None and self.chart_metadata["chart_type"] == "polar_rose"
        )
        self.plot.showGrid(
            x=self._grid_visible and not polar, y=self._grid_visible and not polar, alpha=0.18
        )
        for item in self._grid_items:
            item.setVisible(self._grid_visible)

    def showEvent(self, event):
        super().showEvent(event)
        self.render_pending()

    def _ensure_plot(self):
        if self.plot is not None:
            return
        self.plot = ManagedPlotWidget(
            axisItems={"left": ScientificAxis("left"), "bottom": ScientificAxis("bottom")}
        )
        self.plot.setBackground(current_theme().bg_secondary)
        self.plot.setMinimumHeight(145)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.plot.setMenuEnabled(False)
        self.plot.setToolTip(self.toolTip())
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.getViewBox().setDefaultPadding(0.04)
        self.plot.getViewBox().setBorder(None)
        self.plot.showGrid(x=self._grid_visible, y=self._grid_visible, alpha=0.18)
        for name in ("bottom", "left"):
            axis = self.plot.getAxis(name)
            axis.setPen(pg.mkPen(current_theme().border_primary))
            axis.setTextPen(pg.mkPen(current_theme().text_secondary))
            axis.setStyle(tickFont=QFont("DejaVu Sans", 9), maxTickLevel=0)
        item = self.plot.getPlotItem()
        item.layout.setContentsMargins(0, 0, 0, 0)
        legend = self.plot.addLegend(
            offset=None,
            labelTextSize="9pt",
            labelTextColor=current_theme().text_secondary,
            colCount=3,
            verSpacing=1,
            horSpacing=12,
        )
        item.layout.removeItem(item.getAxis("top"))
        legend.setParentItem(item)
        item.layout.addItem(legend, 1, 1)
        legend.setBrush(pg.mkBrush(current_theme().bg_secondary))
        legend.setPen(None)
        for column in range(0, 6, 2):
            legend.layout.setColumnMaximumWidth(column, 28)
            legend.layout.setColumnPreferredWidth(column, 24)
        self._box.insertWidget(1, self.plot, 1)
        self._mouse_proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=25, slot=self._hover
        )
        self.plot.scene().sigMouseClicked.connect(self._click)

    def _click(self, event):
        if event.double():
            self.plot.enableAutoRange()
        elif event.button() == Qt.LeftButton and self.chart_metadata is not None:
            if self.chart_metadata["chart_type"] != "delay_time":
                return
            view = self.plot.getViewBox()
            if not view.sceneBoundingRect().contains(event.scenePos()):
                return
            x = view.mapSceneToView(event.scenePos()).x()
            data = self.chart_metadata["data"]
            coordinates = np.asarray(data["coordinates"])
            if coordinates.size and data["x_edges"][0] <= x <= data["x_edges"][-1]:
                index = int(np.argmin(np.abs(coordinates - x)))
                self.frame_requested.emit(int(data["frame_indices"][index]))

    def _hover(self, events):
        if not self.isVisible() or self.chart_metadata is None:
            return
        position = events[0]
        view = self.plot.getViewBox()
        if not view.sceneBoundingRect().contains(position):
            self.plot.setToolTip(self.toolTip())
            QToolTip.hideText()
            return
        point = view.mapSceneToView(position)
        chart_type = self.chart_metadata["chart_type"]
        if chart_type == "delay_time":
            data = self.chart_metadata["data"]
            row = np.searchsorted(data["x_edges"], point.x(), side="right") - 1
            column = np.searchsorted(data["delay_edges"], point.y(), side="right") - 1
            power = np.asarray(data["power_db"])
            if not (0 <= row < power.shape[0] and 0 <= column < power.shape[1]):
                return
            value = power[row, column]
            text = (
                f"{data['xlabel']}: {point.x():.6g}\nDelay: {point.y():.4g} ns\n"
                + (
                    f"Summed path gain: {value:.3f} dB"
                    if np.isfinite(value)
                    else "No usable recorded power in this cell"
                )
                + f"\nContributing observations in time bin: {int(data['observation_counts'][row])}"
            )
            self.plot.setToolTip(text)
            QToolTip.showText(
                self.plot.mapToGlobal(self.plot.mapFromScene(position)), text, self.plot
            )
            return
        if chart_type == "polar_rose":
            x, y = np.arctan2(point.y(), point.x()), np.hypot(point.x(), point.y())
        else:
            x, y = inverse(point.x(), self._x_scale), inverse(point.y(), self._y_scale)
        x, y = self._nearest(x, y, chart_type, self.chart_metadata["data"])
        if x is None or y is None:
            self.plot.setToolTip(self.toolTip())
            QToolTip.hideText()
            return
        text = self._formatter(x, y, chart_type, self.chart_metadata["data"])
        self.plot.setToolTip(text)
        QToolTip.showText(self.plot.mapToGlobal(self.plot.mapFromScene(position)), text, self.plot)

    def _labels(self, x, y):
        for axis, label in (("bottom", x), ("left", y)):
            self.plot.setLabel(
                axis, label, color=current_theme().text_secondary, **{"font-size": "10pt"}
            )

    def apply_theme(self):
        """Restyle allocated items without rebinning data or resetting the view."""
        if self.plot is None:
            return
        theme = current_theme()
        self.plot.setBackground(theme.bg_secondary)
        for name in ("bottom", "left"):
            axis = self.plot.getAxis(name)
            axis.setPen(pg.mkPen(theme.border_primary))
            axis.setTextPen(pg.mkPen(theme.text_secondary))
            axis.setLabel(axis.labelText, color=theme.text_secondary, **{"font-size": "9pt"})
        legend = self.plot.getPlotItem().legend
        legend.setBrush(pg.mkBrush(theme.bg_secondary))
        legend.setLabelTextColor(theme.text_secondary)
        for _, label in legend.items:
            label.setText(label.text, color=theme.text_secondary)
        for update in self._theme_updates:
            update()
        if self._colorbar is not None:
            axis = self._colorbar.axis
            axis.setPen(pg.mkPen(theme.border_primary))
            axis.setTextPen(pg.mkPen(theme.text_secondary))
            axis.setLabel("Path gain (dB)", color=theme.text_secondary, **{"font-size": "9pt"})
        for graphic in self.plot.getPlotItem().items:
            if isinstance(graphic, pg.TextItem):
                graphic.setColor(theme.text_secondary)
            elif isinstance(graphic, PolarRoseItem):
                graphic.apply_theme()

    def _line(self, x, y, name=None, color=BLUE, index=0, fill=False):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if not x.size or x.size != y.size:
            return None
        display_x, display_y = transform(x, self._x_scale), transform(y, self._y_scale)
        finite_indices = np.flatnonzero(np.isfinite(display_x) & np.isfinite(display_y))
        if not finite_indices.size:
            return None
        self._series.append((name, x, y))
        pen = pg.mkPen(color, width=2, style=DASHES[index % len(DASHES)])
        line = self.plot.plot(
            display_x,
            display_y,
            pen=pen,
            name=name,
            connect="finite",
        )
        # A bounded marker overlay keeps long runs responsive without inventing points.
        # Scatter bounds cannot handle an all-NaN subset. Sample only real
        # points, while keeping gaps in the full-resolution curve above.
        stride = max(1, int(np.ceil(finite_indices.size / 42)))
        indices = np.unique(np.r_[finite_indices[::stride], finite_indices[-1]])
        markers = self.plot.plot(
            display_x[indices],
            display_y[indices],
            pen=None,
            symbol=SYMBOLS[index % len(SYMBOLS)],
            symbolSize=5,
            symbolPen=pg.mkPen(color),
            symbolBrush=current_theme().bg_secondary,
            connect="finite",
        )

        def recolor():
            themed = dashboard_series_color(color)
            line.setPen(pg.mkPen(themed, width=2, style=DASHES[index % len(DASHES)]))
            markers.setSymbolPen(pg.mkPen(themed))
            markers.setSymbolBrush(current_theme().bg_secondary)

        self._theme_updates.append(recolor)
        return line

    def _bars(self, x, heights, widths=0.8, colors=BLUE, horizontal=False):
        x, heights = np.asarray(x), np.asarray(heights)
        brushes = (
            [pg.mkBrush(c) for c in colors]
            if isinstance(colors, (list, tuple))
            else pg.mkBrush(colors)
        )
        kwargs = (
            {"y": x, "x0": 0, "width": heights, "height": widths}
            if horizontal
            else {"x": x, "y0": 0, "height": heights, "width": widths}
        )
        item = pg.BarGraphItem(
            **kwargs,
            brushes=brushes if isinstance(brushes, list) else None,
            brush=brushes if not isinstance(brushes, list) else None,
            pen=pg.mkPen(current_theme().bg_secondary, width=0.5),
        )
        self.plot.addItem(item)

        def recolor():
            if isinstance(colors, (list, tuple)):
                item.setOpts(brushes=[pg.mkBrush(dashboard_series_color(c)) for c in colors])
            else:
                item.setOpts(brush=pg.mkBrush(dashboard_series_color(colors)))
            item.setOpts(pen=pg.mkPen(current_theme().bg_secondary, width=0.5))

        self._theme_updates.append(recolor)

    def _reference(self, value, label, index=0):
        color = ("#7552a3", "#bf5b13", "#00866b")[index % 3]
        line = pg.InfiniteLine(
            float(transform([value], self._x_scale)[0]),
            angle=90,
            pen=pg.mkPen(color, width=1.4, style=DASHES[1 + index % 3]),
        )
        self.plot.addItem(line)
        sample = pg.PlotDataItem(pen=line.pen)
        self.plot.getPlotItem().legend.addItem(sample, label)

        def recolor():
            pen = pg.mkPen(dashboard_series_color(color), width=1.4, style=DASHES[1 + index % 3])
            line.setPen(pen)
            sample.setPen(pen)

        self._theme_updates.append(recolor)

    def render_pending(self):
        if (
            self.chart_metadata is None
            or self._rendered is self.chart_metadata
            or not self.isVisible()
            or self.window().isMinimized()
        ):
            return
        message = getattr(self, "_message", None)
        if self.chart_metadata["chart_type"] == "line_evolution":
            data = self.chart_metadata["data"]
            x, y = np.asarray(data.get("x", [])), np.asarray(data.get("y", []))
            if x.size != y.size or not np.any(np.isfinite(x) & np.isfinite(y)):
                message = "No usable samples for this chart"
        if message:
            self.message.setText(message)
            self.message.show()
            self._series = []
            self._theme_updates = []
            if self.plot is not None:
                self.plot.clear()
                self._grid_items = []
                self.plot.hide()
            self._rendered = self.chart_metadata
            return
        self._ensure_plot()
        if self._colorbar is not None:
            self.plot.getPlotItem().layout.removeItem(self._colorbar)
            unregister_plot_views(self._colorbar)
            self._colorbar.close()
            # PlotItem.close() removes its contents, but leaves the item in the scene.
            self.plot.scene().removeItem(self._colorbar)
            self._colorbar.deleteLater()
            self._colorbar = None
        self.plot.clear()
        self._grid_items = []
        self.plot.getPlotItem().legend.clear()
        self.plot.show()
        self.message.hide()
        self._series = []
        self._theme_updates = []
        self._x_scale = self._y_scale = "linear"
        for axis in ("bottom", "left"):
            self.plot.showAxis(axis)
            self.plot.getAxis(axis).setTicks(None)
            self.plot.getAxis(axis).display_scale = "linear"
        self.plot.getViewBox().setAspectLocked(False)
        self.plot.getViewBox().invertY(False)
        self.plot.showGrid(x=self._grid_visible, y=self._grid_visible, alpha=0.18)
        self.plot.enableAutoRange()
        data = self.chart_metadata["data"]
        kind = self.chart_metadata["chart_type"]
        from .statistics_panel import _bucket_reflection_orders, _interaction_type_color

        if kind == "delay_time":
            power = np.asarray(data["power_db"], dtype=np.float64)
            self._labels(data["xlabel"], "Propagation delay (ns)")
            # One image, no resampling of saved paths; NaN cells stay transparent.
            image = pg.ImageItem(axisOrder="row-major")
            image.setImage(power.T, autoLevels=False)
            x_edges, delay_edges = data["x_edges"], data["delay_edges"]
            image.setRect(
                QRectF(
                    x_edges[0],
                    delay_edges[0],
                    x_edges[-1] - x_edges[0],
                    delay_edges[-1] - delay_edges[0],
                )
            )
            self.plot.addItem(image)
            finite = power[np.isfinite(power)]
            low, high = float(finite.min()), float(finite.max())
            if low == high:
                low, high = low - 0.5, high + 0.5
            self._colorbar = pg.ColorBarItem(
                values=(low, high), colorMap=pg.colormap.get("viridis"), interactive=False, width=14
            )
            self._colorbar.setImageItem(image, insert_in=self.plot.getPlotItem())
            self.plot.setRange(
                xRange=(x_edges[0], x_edges[-1]),
                yRange=(delay_edges[0], delay_edges[-1]),
                padding=0,
            )
            self.note.setText(
                f"Delay bins: {delay_edges[1] - delay_edges[0]:.3g} ns · "
                "incoherent path-power sums, averaged within time bins · blank: no usable data · click to inspect"
            )
        elif kind == "reflection_order":
            counts = _bucket_reflection_orders(data)
            self._bars(np.arange(7), [counts[i] for i in range(7)], colors=ORDER_COLORS)
            self.plot.getAxis("bottom").setTicks(
                [[(i, str(i) if i < 6 else "6+") for i in range(7)]]
            )
            self._labels("Interaction order", "Path count")
        elif kind in {"interaction_type", "coverage_serving_share"}:
            if kind == "interaction_type":
                types = sorted(map(int, data), key=mpc_interaction_sort_key)
                values = [data.get(i, data.get(str(i), 0)) for i in types]
                labels = [mpc_interaction_label(i) for i in types]
                colors = [_interaction_type_color(i) for i in types]
                self._labels("Path count", "Initial mechanism")
            else:
                values, labels, colors = data["percentages"], data["labels"], data["colors"]
                self._labels("Share of total slice area (%)", "")
                self.plot.setXRange(0, 100, padding=0.03)
                data["_bar_positions"] = np.arange(len(labels))
            self._bars(np.arange(len(labels)), values, colors=colors, horizontal=True)
            self.plot.getAxis("left").setTicks([list(enumerate(labels))])
            self.plot.getViewBox().invertY(True)
        elif kind == "line_evolution":
            label = data.get("ylabel", "Value")
            label = {
                "Total MPC Count": "Path count",
                "Pooled Path Delay Spread (RMS, ns)": "RMS delay spread (ns)",
                "Single-Path Loss (dB)": "Path loss (dB)",
            }.get(label, label)
            self._labels("Frame", label)
            self._line(data.get("x", []), data.get("y", []))
        elif kind in {"mpc_order_evolution", "mpc_type_evolution"}:
            order = kind == "mpc_order_evolution"
            if order:
                self._y_scale = "symlog"
            x = np.asarray(data["frame_indices"])
            series = data["order_data" if order else "type_data"]
            key = (
                (lambda item: int(item[0]))
                if order
                else (lambda item: mpc_interaction_sort_key(int(item[0])))
            )
            for index, (raw, values) in enumerate(sorted(series.items(), key=key)):
                values = np.asarray(values)
                if values.size != x.size or not np.any(values):
                    continue
                number = int(raw)
                label = (
                    (str(number) if number < 6 else "6+")
                    if order
                    else mpc_interaction_label(number)
                )
                color = (
                    ORDER_COLORS[min(max(number, 0), 6)]
                    if order
                    else _interaction_type_color(number)
                )
                self._line(x, values, f"Order {label}" if order else label, color, index)
            self._labels("Frame", "Path count (symlog)" if order else "Path count")
            if order and self._series:
                maximum = max(float(np.max(values)) for _name, _x, values in self._series)
                upper = float(transform([max(1.0, maximum)], "symlog")[0])
                self.plot.setYRange(0, upper * 1.05, padding=0)
        elif kind == "pair_visibility_evolution":
            x = np.asarray(data["frame_indices"])
            lower = np.zeros(x.size)
            for index, (category, label, color) in enumerate(
                zip(
                    ("direct_path_present", "indirect_only", "no_path"),
                    ("Direct path", "Indirect only", "No path"),
                    ("#00866b", "#bf5b13", "#64748b"),
                )
            ):
                values = np.asarray(data["category_data"].get(category, []))
                if values.size != x.size or not np.any(values):
                    continue
                upper = lower + values
                low_curve = pg.PlotCurveItem(x, lower, pen=None)
                high_curve = pg.PlotCurveItem(x, upper, pen=None)
                fill_color = QColor(color)
                fill_color.setAlpha(75)
                self.plot.addItem(pg.FillBetweenItem(low_curve, high_curve, brush=fill_color))
                self._line(x, upper, label, color, index)
                lower = upper
            self._labels("Frame", "TX/RX pair count")
        elif kind in {"histogram", "angle_histogram", "coverage_histogram"}:
            values = np.asarray(data.get("values", []), dtype=float)
            values = values[np.isfinite(values)]
            bins = int(data.get("bins", 30))
            limits = data.get("xlim") if kind == "angle_histogram" else None
            if kind == "coverage_histogram" and data.get("color_scale") == "logarithmic":
                values = values[values > 0]
                if values.size:
                    lo, hi = float(values.min()), float(values.max())
                    if lo == hi:
                        lo, hi = lo * 0.9, hi * 1.1
                    bins = np.geomspace(lo, hi, bins + 1)
                    self._x_scale = "log"
            counts, edges = np.histogram(values, bins=bins, range=limits)
            data["_hist_counts"], data["_hist_edges"] = counts, edges
            heights = counts
            if kind == "coverage_histogram":
                heights = 100.0 * counts / max(1, values.size)
                data["_hist_percentages"] = heights
            display_edges = transform(edges, self._x_scale)
            self._bars(
                (display_edges[:-1] + display_edges[1:]) / 2, heights, np.diff(display_edges) * 0.92
            )
            self._labels(
                data.get("xlabel", "Value"),
                "Share of valid area (%)" if kind == "coverage_histogram" else "Path count",
            )
            if limits is not None:
                self.plot.setXRange(*limits, padding=0)
            if kind == "coverage_histogram":
                for index, (label, value) in enumerate(
                    zip(("P10", "P50", "P90"), data.get("percentiles", ()))
                ):
                    if value is not None:
                        self._reference(value, f"{label} {value:.3g}", index)
            elif kind == "angle_histogram":
                self._reference(0, "Zero elevation")
        elif kind == "cdf":
            values = np.asarray(data.get("values", []), dtype=float)
            values = np.sort(values[np.isfinite(values)])
            if values.size:
                indices = np.linspace(
                    0, values.size - 1, min(values.size, MAX_CDF_POINTS), dtype=np.intp
                )
                x, y = values[indices], (indices + 1) / values.size
                data["_cdf_x"], data["_cdf_y"], data["_cdf_total"] = x, y, int(values.size)
                self._line(x, y)
                median = float(np.median(values))
                self._reference(median, f"Median {median:.3g}")
            self._labels(data.get("xlabel", "Value"), "Cumulative fraction")
            self.plot.setYRange(0, 1.02, padding=0)
        elif kind == "coverage_success_curve":
            if data.get("color_scale") == "logarithmic":
                self._x_scale = "log"
            x, y = np.asarray(data["thresholds"]), np.asarray(data["percentages"])
            data["_curve_x"], data["_curve_y"] = x, y
            self._line(x, y)
            self._labels(
                data.get("xlabel", "Coverage threshold"),
                "Qualifying area (%)",
            )
            self.plot.setYRange(0, 100, padding=0)
        elif kind == "polar_rose":
            values = np.asarray(data.get("values", []), dtype=float)
            edges = np.linspace(-np.pi, np.pi, int(data.get("n_bins", 36)) + 1)
            counts, _ = np.histogram(np.deg2rad(values), bins=edges)
            data["_polar_hist_counts"], data["_polar_hist_edges"] = counts, edges
            self._polar(edges, counts)
        for axis, scale in (("bottom", self._x_scale), ("left", self._y_scale)):
            self.plot.getAxis(axis).display_scale = scale
            self.plot.getAxis(axis).picture = None
            self.plot.getAxis(axis).update()
        self.plot.getPlotItem().legend.setVisible(bool(self.plot.getPlotItem().legend.items))
        self.render_count += 1
        self._rendered = self.chart_metadata
        self.set_grid_visible(self._grid_visible)
        self.apply_theme()

    def _polar(self, edges, counts):
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.showGrid(x=False, y=False)
        self.plot.getViewBox().setAspectLocked(True)
        radius = max(1, int(np.max(counts)))
        angles = np.linspace(-np.pi, np.pi, 181)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            r = radius * fraction
            ring = self.plot.plot(r * np.cos(angles), r * np.sin(angles))
            self._grid_items.append(ring)
            self._theme_updates.append(
                lambda item=ring: item.setPen(pg.mkPen(current_theme().border_primary, width=1))
            )
            label = pg.TextItem(f"{r:,.3g}", color=current_theme().text_secondary, anchor=(0, 1))
            # Keep count labels between direction spokes, clear of the 45° label.
            label.setFont(QFont("DejaVu Sans", 8))
            label.setPos(r * np.cos(np.deg2rad(22.5)), r * np.sin(np.deg2rad(22.5)))
            self.plot.addItem(label, ignoreBounds=True)
        for degree in range(-135, 181, 45):
            angle = np.deg2rad(degree)
            spoke = self.plot.plot(
                [0, radius * np.cos(angle)], [0, radius * np.sin(angle)], pen=pg.mkPen("#e5ebf2")
            )
            self._grid_items.append(spoke)
            self._theme_updates.append(
                lambda item=spoke: item.setPen(pg.mkPen(current_theme().border_subtle))
            )
            label = pg.TextItem(
                f"{degree}°", color=current_theme().text_secondary, anchor=(0.5, 0.5)
            )
            label.setPos(radius * 1.13 * np.cos(angle), radius * 1.13 * np.sin(angle))
            self.plot.addItem(label, ignoreBounds=True)
        self.plot.addItem(PolarRoseItem(edges, counts))
        self.plot.setXRange(-radius * 1.3, radius * 1.3, padding=0)
        self.plot.setYRange(-radius * 1.3, radius * 1.3, padding=0)
