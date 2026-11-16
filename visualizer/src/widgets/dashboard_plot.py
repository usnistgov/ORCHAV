"""Plots with readable heights and titles that leave room for axes/colorbars."""

import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QGraphicsItem

from .plot_lifecycle import ManagedPlotWidget


class _LayoutLegend(pg.LegendItem):
    """A legend placed by the plot layout cannot be dragged over its data."""

    items_changed = Signal()

    def addItem(self, item, name):
        super().addItem(item, name)
        self.items_changed.emit()

    def removeItem(self, item):
        super().removeItem(item)
        self.items_changed.emit()

    def clear(self):
        super().clear()
        self.items_changed.emit()

    def mouseDragEvent(self, event):
        event.ignore()


class DashboardPlot(ManagedPlotWidget):
    """Keep plot titles readable in both narrow panels and floating windows."""

    def __init__(self):
        self._external_legend = None
        self._legend_layout_columns = 0
        self._minimum_plot_height = None
        self._minimum_data_height = 120
        self._live_figure = False
        self._axis_label_requests = {}
        self._empty_message = None
        self._manual_range = None
        super().__init__()
        self.getViewBox().sigRangeChangedManually.connect(self._remember_manual_range)
        # Reuse grid/tick pixels; Qt and AxisItem invalidate view/style changes.
        for name in ("left", "bottom", "right", "top"):
            self.getAxis(name).setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self._legend_timer = QTimer(self)
        self._legend_timer.setSingleShot(True)
        self._legend_timer.timeout.connect(self._reflow_legend)
        # PlotWidget installs PlotItem forwarding methods on the instance.
        self.setTitle = self._set_title
        self.setLabel = self._set_label

    def _remember_manual_range(self, *_args):
        self._manual_range = tuple(tuple(bounds) for bounds in self.viewRange())

    def has_manual_range(self) -> bool:
        """Distinguish an inspected viewport from programmatically fitted axes."""
        return self._manual_range == tuple(tuple(bounds) for bounds in self.viewRange())

    def close(self):
        if self.plotItem is None:
            return
        self._legend_timer.stop()
        super().close()

    def set_empty_message(self, text: str, *, color=None) -> None:
        """Keep a reusable message centred in the plot, including borrowed figures.

        Parenting to the ViewBox itself uses screen coordinates, excludes the
        label from data bounds and lets its anchor follow layout changes.
        Unchanged playback updates do not rebuild text or layout.
        """
        label = self._empty_message
        if label is None:
            if not text:
                return
            label = self._empty_message = pg.LabelItem(
                text, parent=self.getViewBox(), color=color, size="10pt"
            )
            label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            label.item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            label.setZValue(100)
            label.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
        elif text and (label.text != text or (color is not None and label.opts["color"] != color)):
            label.setText(text, color=color if color is not None else label.opts["color"])
        if label.isVisible() != bool(text):
            label.setVisible(bool(text))

    def _set_label(self, axis, text=None, units=None, **options):
        """Leave unchanged axis HTML and its cached paint geometry intact."""
        item = self.getPlotItem().getAxis(axis)
        request = (text, units, options)
        if (
            self._axis_label_requests.get(axis) == request
            and item.labelText == (text or "")
            and item.labelUnits == (units or "")
            and item.label.isVisible() == bool(text or units)
            and all(
                item.labelStyle.get(key) == value
                for key, value in options.items()
                if key not in {"unitPrefix", "siPrefixEnableRanges", "unitPower"}
            )
        ):
            return
        self.getPlotItem().setLabel(axis, text=text, units=units, **options)
        self._axis_label_requests[axis] = request

    def sizeHint(self):
        return QSize(360, 230)

    def setMinimumHeight(self, height):
        self._minimum_plot_height = height
        super().setMinimumHeight(height)
        if self._external_legend is not None:
            self._legend_timer.start(0)

    def set_data_minimum_height(self, height):
        """Reserve this much data area in addition to the title, axes and legend."""
        self._minimum_data_height = int(height)
        self._update_minimum_height()

    def set_live_figure(self, enabled: bool) -> None:
        """Allow a borrowed plot to shrink without losing its dashboard height."""
        self._live_figure = enabled
        self._update_minimum_height()

    def add_external_legend(self, **options):
        """Reserve a wrapping legend row between the title and the data axes."""
        plot = self.getPlotItem()
        if self._external_legend is None:
            legend = _LayoutLegend(offset=None, **options)
            plot.legend = legend
            legend.setParentItem(plot)
            plot.layout.removeItem(plot.getAxis("top"))
            self._legend_layout_columns = plot.layout.columnCount()
            plot.layout.addItem(legend, 1, 0, 1, self._legend_layout_columns)
            self._external_legend = legend
            legend.items_changed.connect(lambda: self._legend_timer.start(0))
        return self._external_legend

    def _reflow_legend(self):
        plot = self.getPlotItem()
        legend = self._external_legend
        if plot is None:
            return
        self._wrap_title()
        if legend is None or not legend.items:
            self._update_minimum_height()
            return
        # A colorbar can add layout columns after the legend is created.
        if plot.layout.columnCount() != self._legend_layout_columns:
            plot.layout.removeItem(legend)
            self._legend_layout_columns = plot.layout.columnCount()
            plot.layout.addItem(legend, 1, 0, 1, self._legend_layout_columns)
        plot_left, _, plot_right, _ = plot.layout.getContentsMargins()
        available = self.width() - plot_left - plot_right
        widths = []
        for sample, label in legend.items:
            label.item.setTextWidth(-1)
            label.resizeEvent(None)
            widths.append(sample.width() + label.item.boundingRect().width())
        left, _, right, _ = legend.layout.getContentsMargins()
        for columns in range(min(3, len(widths)), 0, -1):
            width = sum(max(widths[column::columns]) for column in range(columns))
            width += (2 * columns - 1) * legend.layout.horizontalSpacing() + left + right
            if width <= available:
                break
        legend.setColumnCount(columns)
        if columns == 1:
            for sample, label in legend.items:
                label.item.setTextWidth(
                    max(
                        40,
                        available
                        - left
                        - right
                        - sample.width()
                        - legend.layout.horizontalSpacing(),
                    )
                )
                label.resizeEvent(None)
        self._update_minimum_height()

    def _update_minimum_height(self):
        plot = self.getPlotItem()
        if plot is None:
            return
        legend = self._external_legend
        if self._minimum_plot_height is None:
            self._minimum_plot_height = self.minimumHeight()
        # Dynamic track legends need space of their own even in a short panel.
        title_height = (
            plot.titleLabel.item.boundingRect().height() if plot.titleLabel.isVisible() else 0
        )
        legend_height = (
            legend.layout.effectiveSizeHint(Qt.SizeHint.MinimumSize).height()
            if legend is not None and legend.isVisible() and legend.items
            else 0
        )
        bottom_height = plot.getAxis("bottom").minimumHeight()
        super().setMinimumHeight(
            max(
                0 if self._live_figure else self._minimum_plot_height,
                int(
                    title_height
                    + legend_height
                    + bottom_height
                    + (60 if self._live_figure else self._minimum_data_height)
                    + 8
                ),
            )
        )
        # During a wide-to-narrow resize Qt can clamp PlotItem to the old
        # unwrapped legend minimum. Reflow lowers that minimum afterwards, but
        # GraphicsView will not retry the geometry until another resize. Fit
        # the scene now so the colorbar and rightmost legend remain visible.
        # This runs only for layout changes, not unchanged playback titles.
        if plot.geometry() != self.sceneRect():
            plot.setGeometry(self.sceneRect())

    def _set_title(self, text, **style):
        plot = self.getPlotItem()
        label = plot.titleLabel
        if (
            text == label.text
            and label.isVisible()
            and all(label.opts.get(key) == value for key, value in style.items())
        ):
            # Reapplying unchanged HTML also resets the title row height and
            # invalidates axes/legend layout on every sensing playback tick.
            return
        plot.setTitle(text, **style)
        self._wrap_title()
        if self._external_legend is not None:
            self._legend_timer.start(0)
        else:
            self._update_minimum_height()

    def _wrap_title(self):
        plot = self.getPlotItem()
        if plot is not None:
            label = plot.titleLabel
            label.item.setTextWidth(-1)
            width = max(120, self.width() - 150)
            if label.item.boundingRect().width() > width:
                label.item.setTextWidth(width)
            label.resizeEvent(None)
            if label.isVisible():
                height = label.item.boundingRect().height()
                label.setMaximumHeight(height)
                plot.layout.setRowFixedHeight(0, height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Wrapping can increase our minimum height. Doing that inside Qt's
        # resize delivery leaves the viewport at the old size, clipping axes
        # above a blank strip. Reflow after Qt finishes resizing the viewport.
        if hasattr(self, "_legend_timer"):
            self._legend_timer.start(0)
