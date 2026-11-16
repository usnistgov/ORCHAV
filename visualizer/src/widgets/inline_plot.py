"""Native Qt charts sharing the application's theme and empty-data presentation."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QGridLayout, QWidget

from ..app.plot_theme import apply_pyqtgraph_plot_theme, dashboard_series_color
from ..app.theme import current_theme, get_theme_manager
from .dashboard_plot import DashboardPlot
from .plot_lifecycle import unregister_plot_views

SERIES_COLORS = ("#2865b0", "#00866b", "#a66b00", "#b5475a", "#7053ad", "#078596")


class _LegendGuide(pg.InfiniteLine):
    """Expose the line pen to pyqtgraph's standard clickable legend sample."""

    def __init__(self, **kwargs):
        self.opts = {}
        super().__init__(**kwargs)

    def setPen(self, *args, **kwargs):
        super().setPen(*args, **kwargs)
        self.opts["pen"] = self.pen


class InlinePlot(DashboardPlot):
    """A themed plot that updates styles without rebuilding data or view ranges."""

    def __init__(self, title="", xlabel="", ylabel=""):
        super().__init__()
        # PlotWidget forwards PlotItem.clear on the instance, so install our
        # cleanup after construction just like DashboardPlot's title adapter.
        self.clear = self._clear
        self._title = title
        self._themed_items = []
        self._colorbars = []
        self.setMinimumWidth(0)
        self.setLabel("bottom", xlabel)
        self.setLabel("left", ylabel)
        self.add_external_legend()
        self.apply_theme()
        get_theme_manager().theme_changed.connect(self.apply_theme)

    def close(self):
        if self.plotItem is not None:
            get_theme_manager().theme_changed.disconnect(self.apply_theme)
        super().close()

    def configure(self, *, title=None, xlabel=None, ylabel=None):
        if title is not None:
            self._title = title
        if xlabel is not None:
            self.setLabel("bottom", xlabel)
        if ylabel is not None:
            self.setLabel("left", ylabel)
        self.apply_theme()

    def _clear(self):
        self._themed_items.clear()
        for bar in self._colorbars:
            unregister_plot_views(bar)
            self.getPlotItem().layout.removeItem(bar)
            bar.setParentItem(None)
            self.scene().removeItem(bar)
        self._colorbars.clear()
        self.getPlotItem().clear()
        self.set_empty_message("")

    def apply_theme(self, *_):
        theme = current_theme()
        apply_pyqtgraph_plot_theme(self, title=self._title)
        self._external_legend.setLabelTextColor(theme.text_primary)
        for _, label in self._external_legend.items:
            # pyqtgraph's setter updates defaults but not existing label HTML.
            label.setText(label.text, color=theme.text_primary)
        self._external_legend.setBrush(pg.mkBrush(theme.bg_secondary))
        self._external_legend.setPen(pg.mkPen(theme.border_primary))
        for item, color, width, style in self._themed_items:
            resolved = dashboard_series_color(color) if color else theme.text_secondary
            if isinstance(item, pg.TextItem):
                item.setColor(resolved)
            else:
                item.setPen(pg.mkPen(resolved, width=width, style=style))
                if isinstance(item, pg.PlotDataItem) and item.opts.get("symbol"):
                    item.setSymbolBrush(resolved)
                    item.setSymbolPen(resolved)
        for bar in self._colorbars:
            bar.axis.setPen(pg.mkPen(theme.text_secondary))
            bar.axis.setTextPen(pg.mkPen(theme.text_secondary))
            bar.axis.setLabel(bar.axis.labelText, color=theme.text_secondary)
        if self._empty_message is not None:
            self.set_empty_message(
                self._empty_message.text if self._empty_message.isVisible() else "",
                color=theme.text_secondary,
            )

    def curve(
        self, x, y, *, name=None, color=SERIES_COLORS[0], symbol=None, style=Qt.SolidLine, width=1.5
    ):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        # Preserve gaps between finite samples; scatter bounds must never see
        # a nonempty all-NaN array.
        if not valid.any():
            x, y = np.array([]), np.array([])
        else:
            x, y = np.where(np.isfinite(x), x, np.nan), np.where(np.isfinite(y), y, np.nan)
        resolved = dashboard_series_color(color)
        item = self.plot(
            x,
            y,
            name=name,
            pen=pg.mkPen(resolved, width=width, style=style),
            symbol=symbol,
            symbolSize=5,
            symbolBrush=resolved,
            symbolPen=resolved,
            connect="finite",
        )
        self._themed_items.append((item, color, width, style))
        return item

    def guide(self, value, *, vertical=True, name=None, color=None, style=Qt.DashLine):
        item = _LegendGuide(
            pos=value,
            angle=90 if vertical else 0,
            movable=False,
            pen=pg.mkPen(color or current_theme().text_secondary, style=style),
        )
        self.addItem(item)
        if name:
            self._external_legend.addItem(item, name)
        self._themed_items.append((item, color, 1, style))
        return item

    def text(self, text, x, y, *, color=None, anchor=(0, 0.5)):
        item = pg.TextItem(str(text), color=color or current_theme().text_secondary, anchor=anchor)
        item.setPos(x, y)
        self.addItem(item, ignoreBounds=True)
        if color is None:
            self._themed_items.append((item, None, 1, Qt.SolidLine))
        return item

    def heatmap(self, x, y, values, *, limits, colormap="viridis", label=""):
        """Draw exact coordinate bins, including nonuniform recorded axes."""

        def edges(centres):
            centres = np.asarray(centres, dtype=float)
            if len(centres) == 1:
                return np.array([centres[0] - 0.5, centres[0] + 0.5])
            middle = (centres[:-1] + centres[1:]) / 2
            return np.r_[2 * centres[0] - middle[0], middle, 2 * centres[-1] - middle[-1]]

        cmap = pg.colormap.get(colormap) if isinstance(colormap, str) else colormap
        xedges, yedges = edges(x), edges(y)
        regular = all(
            np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-6, atol=1e-12)
            for axis in (xedges, yedges)
        )
        if regular:
            # Spectrograms usually have uniform bins. One image avoids drawing
            # a polygon per bin while retaining every saved sample for zooming.
            item = pg.ImageItem(np.asarray(values), axisOrder="row-major")
            item.setRect(
                QRectF(xedges[0], yedges[0], xedges[-1] - xedges[0], yedges[-1] - yedges[0])
            )
        else:
            xx, yy = np.meshgrid(xedges, yedges)
            item = pg.PColorMeshItem(xx, yy, np.asarray(values), colorMap=cmap, levels=limits)
        self.addItem(item)
        bar = pg.ColorBarItem(values=limits, colorMap=cmap, label=label, interactive=False)
        bar.setImageItem(item, insert_in=self.getPlotItem())
        self._colorbars.append(bar)
        self.apply_theme()
        return item


class InlinePlotGrid(QWidget):
    """A small grid of independently interactive plots, without a figure backend."""

    def __init__(self, specifications, *, columns=1, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plots = []
        for index, (title, xlabel, ylabel) in enumerate(specifications):
            plot = InlinePlot(title, xlabel, ylabel)
            layout.addWidget(plot, index // columns, index % columns)
            self.plots.append(plot)

    def clear(self):
        for plot in self.plots:
            plot.clear()
