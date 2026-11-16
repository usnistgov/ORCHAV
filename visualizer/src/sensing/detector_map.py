"""Saved detector maps with numerical or categorical color scales."""

from colorsys import hsv_to_rgb

import numpy as np
import pyqtgraph as pg

from ..widgets.dashboard_plot import DashboardPlot

_CLUSTER_COLORS = (
    "#56a9df",
    "#ef8a47",
    "#70b85a",
    "#b883d5",
    "#df6384",
    "#42bcb1",
    "#eed45c",
    "#b68d69",
    "#f29dce",
    "#9da7b4",
)


class DetectorMap(DashboardPlot):
    """Keep each stage visible without rebuilding its image for cell selections."""

    def __init__(self, title, colormap, *, categorical=False):
        super().__init__()
        self._categorical = categorical
        self.cluster_ids = ()
        self.setTitle(title)
        self.image = pg.ImageItem()
        self.addItem(self.image)
        self.colorbar = pg.ColorBarItem(
            colorMap=colormap, interactive=False, width=12, colorMapMenu=not categorical
        )
        self.colorbar.setImageItem(self.image, insert_in=self.getPlotItem())
        self.cell = pg.ScatterPlotItem(
            symbol="s", size=14, pen=pg.mkPen("#eab83c", width=2), brush=pg.mkBrush(None)
        )
        self.cell.setZValue(20)
        self.addItem(self.cell, ignoreBounds=True)
        self.slice_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#eab83c", width=2))
        self.slice_line.setZValue(18)
        self.addItem(self.slice_line, ignoreBounds=True)
        self.bin_grid = pg.PlotCurveItem(pen=pg.mkPen((170, 180, 194, 140), width=1))
        self.bin_grid.setZValue(11)
        self.addItem(self.bin_grid, ignoreBounds=True)
        self.bin_grid.hide()
        self._render_key = None

    def set_map(self, values, units, levels, rect, key):
        """Return whether the saved image changed; selection is updated separately."""
        if key is not None and key == self._render_key:
            return False
        self._render_key = key
        self.colorbar.getAxis("left").setLabel(units)
        if self._categorical:
            values, levels = self._categorical_image(values)
        if values is None:
            self.image.clear()
        else:
            self.image.setImage(values.T, autoLevels=False)
            self.colorbar.setLevels(values=levels)
            self.image.setRect(rect)
        return True

    def _categorical_image(self, values):
        """Pack visible IDs into equal color bands without changing saved labels."""
        ids = (
            np.unique(values[np.isfinite(values) & (values >= 0)]).astype(np.int64)
            if values is not None
            else np.empty(0, dtype=np.int64)
        )
        self.cluster_ids = tuple(int(value) for value in ids)
        count = len(ids)
        self.colorbar.setVisible(bool(count))
        if not count:
            return None if values is None else np.full(values.shape, np.nan), (0, 1)

        # Separate the visible categories with a qualitative palette. Cluster
        # IDs are acquisition-local labels, so colors do not identify tracks.
        colors = np.array(
            [
                (
                    pg.mkColor(_CLUSTER_COLORS[i]).getRgb()
                    if i < len(_CLUSTER_COLORS)
                    else (
                        *[
                            round(v * 255)
                            for v in hsv_to_rgb((i * 0.618033988749895) % 1, 0.65, 0.95)
                        ],
                        255,
                    )
                )
                for i in range(count)
            ],
            dtype=np.uint8,
        )
        edges = np.arange(count + 1, dtype=float) / count
        stops = np.column_stack((edges[:-1], np.nextafter(edges[1:], 0))).ravel()
        stops[-1] = 1
        self.colorbar.setColorMap(pg.ColorMap(stops, np.repeat(colors, 2, axis=0)))
        # Sparse saved IDs get equally sized bands, with their original labels
        # at the centers. Limit text density when many categories are present.
        tick_indices = np.unique(np.linspace(0, count - 1, min(count, 12), dtype=int))
        self.colorbar.axis.setTicks([[(int(i), str(ids[i])) for i in tick_indices]])
        self.colorbar.getAxis("left").setLabel("Saved cluster ID")
        display = np.full(values.shape, np.nan)
        valid = np.isfinite(values) & (values >= 0)
        display[valid] = np.searchsorted(ids, values[valid])
        return display, (-0.5, count - 0.5)

    def select_cell(self, x=None, y=None):
        if x is None or y is None:
            self.cell.clear()
            self.slice_line.hide()
        else:
            self.cell.setData([x], [y])
            self.slice_line.setPos(y)
            self.slice_line.show()
