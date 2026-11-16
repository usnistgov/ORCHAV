"""Lazy location charts from retained samples, without further propagation reads."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from shared.coverage.availability import CoverageAvailability

from ..widgets.inline_plot import InlinePlotGrid
from .analysis import (
    coverage_metric_color_scale,
    coverage_metric_label,
    coverage_metric_valid_mask,
    coverage_quantity_family,
    is_serving_tx_metric,
    serving_tx_color_hex,
)


def vertical_profile(model, indices):
    """Return the saved column with zero/unavailable entries as gaps, never floors."""
    x, y, _ = indices
    values = model.values[:, y, x]
    valid = coverage_metric_valid_mask(values, model.quantity) & ~model.zero_contributions[:, y, x]
    return np.where(valid, values, np.nan)


class CoverageCellCharts(QWidget):
    """Draw power by transmitter and the selected metric against saved height.

    The owner shares its bounded point-read cache with the table. The vertical
    column comes from the already loaded immutable metric. Appearance-only
    changes reuse the plots; shared theme updates do not rebuild their data.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.model = self.indices = self.threshold = self.data = None
        self.context_key = None
        self.canvas = None
        self._retired = False
        self.error = ""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("Power and height charts")
        self.enabled.setToolTip("Saved samples only. Missing/zero power appears as a gap.")
        self.enabled.toggled.connect(self._toggle)
        self._layout.addWidget(self.enabled)
        self.note = QLabel("Saved samples only. Missing/zero power is a gap, not a low dBm value.")
        self.note.setWordWrap(True)
        self.note.hide()
        self._layout.addWidget(self.note)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._draw)

    def _toggle(self, enabled):
        self._sync_visibility()
        if enabled:
            self._schedule()
        else:
            self.timer.stop()

    def _sync_visibility(self):
        selected = self.model is not None and self.indices is not None
        enabled = self.enabled.isChecked()
        self.note.setVisible(enabled and not selected)
        self.note.setText("Select a cell.")
        if self.canvas is not None:
            self.canvas.setVisible(enabled and selected)

    def _schedule(self):
        if self.enabled.isChecked():
            self.timer.start(0)

    def set_context(self, model, indices, threshold):
        key = (id(model), indices, threshold)
        if key != self.context_key:
            self.context_key = key
            self.model, self.indices, self.threshold = model, indices, threshold
            self._sync_visibility()
            self._schedule()

    def set_data(self, data, *, error=""):
        if data is not self.data or error != self.error:
            self.data, self.error = data, error
            self._schedule()

    def _draw(self):
        if not self.enabled.isChecked() or self.model is None or self.indices is None:
            return
        if self.canvas is None:
            self.canvas = InlinePlotGrid(
                [
                    ("Power by transmitter", "Received power (dBm)", ""),
                    ("Vertical profile", "", "Height (m)"),
                ]
            )
            self.canvas.setMinimumWidth(0)
            self._layout.addWidget(self.canvas)
            self.power_plot, self.profile_plot = self.canvas.plots
        self.canvas.clear()
        self._draw_power()
        self._draw_profile()

    def _draw_power(self):
        plot, data = self.power_plot, self.data
        if data is None:
            plot.set_empty_message(self.error or "Loading saved transmitter samples…")
            return
        values = np.asarray(data["rss_dbm"])
        valid = np.isfinite(values) & (
            np.asarray(data["availability"]) == CoverageAvailability.POSITIVE
        )
        threshold = (
            self.threshold if coverage_quantity_family(self.model.quantity) == "rss_dbm" else None
        )
        extents = np.append(values[valid], threshold) if threshold is not None else values[valid]
        low = float(extents.min()) - 8 if extents.size else -100
        high = float(extents.max()) + 12 if extents.size else -40
        rows = np.arange(len(values))
        self.power_bars = None
        if valid.any():
            self.power_bars = pg.BarGraphItem(
                y=rows[valid],
                x0=low,
                x1=values[valid],
                height=0.55,
                brushes=["#178550" if i == data["serving_tx"] else "#3073aa" for i in rows[valid]],
                pen=None,
            )
            plot.addItem(self.power_bars)
        plot.getAxis("left").setTicks(
            [
                list(
                    enumerate(
                        [
                            name + (" · serving" if i == data["serving_tx"] and valid[i] else "")
                            for i, name in enumerate(data["names"])
                        ]
                    )
                )
            ]
        )
        plot.getViewBox().invertY(True)
        plot.setXRange(low, high, padding=0)
        plot.setYRange(-0.5, max(0.5, len(rows) - 0.5), padding=0)
        for i, value in enumerate(values):
            if valid[i]:
                plot.text(f" {value:.1f}", value, i)
            else:
                reason = {0: "Unknown", 2: "Zero contribution", 3: "Invalid"}.get(
                    int(data["availability"][i]), "Unavailable"
                )
                plot.text(reason, low, i)
        if threshold is not None:
            plot.guide(threshold, name="View threshold")

    def _draw_profile(self):
        plot, model, indices = self.profile_plot, self.model, self.indices
        values, heights = vertical_profile(model, indices), model.axes[2]
        title, unit = coverage_metric_label(model.quantity)
        plot.configure(
            title=f"Vertical profile · X = {model.axes[0][indices[0]]:g} m, Y = {model.axes[1][indices[1]]:g} m",
            xlabel=f"{title} ({unit})" if unit else title,
        )
        logarithmic = coverage_metric_color_scale(model.quantity) == "logarithmic"
        plot.setLogMode(x=logarithmic, y=False)
        plot.getAxis("bottom").setTicks(None)
        if is_serving_tx_metric(model.quantity):
            valid = np.isfinite(values)
            plot.addItem(
                pg.ScatterPlotItem(
                    x=values[valid],
                    y=heights[valid],
                    size=6,
                    brush=[serving_tx_color_hex(int(v)) for v in values[valid]],
                    pen=None,
                )
            )
            plot.getAxis("bottom").setTicks([list(enumerate(model.transmitter_names))])
        else:
            self.profile_curve = plot.curve(values, heights, symbol="o", name="Saved samples")
            if self.threshold is not None and (not logarithmic or self.threshold > 0):
                plot.guide(
                    np.log10(self.threshold) if logarithmic else self.threshold,
                    name="View threshold",
                )
        height = heights[indices[2]]
        plot.guide(height, vertical=False, style=Qt.DotLine, name=f"Selected height: {height:g} m")
        if np.isfinite(values[indices[2]]):
            x = np.log10(values[indices[2]]) if logarithmic else values[indices[2]]
            plot.addItem(
                pg.ScatterPlotItem(x=[x], y=[height], size=9, brush="#26ffff", pen="#14222a")
            )
        if not np.isfinite(values).any():
            plot.set_empty_message("No data available for this profile")
        plot.enableAutoRange()

    def shutdown(self):
        """Retire lazy chart views before the source-bound Qt parent is deleted."""
        if self._retired:
            return
        self.timer.stop()
        self.model = self.data = None
        if self.canvas is not None:
            for plot in self.canvas.plots:
                plot.getViewBox().unregister()
        self._retired = True
