"""Charts of the saved full-volume evaluation."""

import numpy as np
import pyqtgraph as pg

from shared.sensing.volume.inspection import (
    DETAIL_COLORS,
    DETAIL_LABELS,
    category_counts,
    detail_grid,
)

from ..widgets.inline_plot import InlinePlotGrid


class VolumeSummaryPlots(InlinePlotGrid):
    def __init__(self):
        super().__init__(
            [
                ("Detection outcomes", "Number of cells", ""),
                ("Nearest report position error", "Error (m)", "Fraction of cells with reports"),
            ]
        )
        self.snapshot = None
        self.setMinimumHeight(400)

    def update_snapshot(self, snapshot):
        if self.snapshot is snapshot:
            return
        self.snapshot = snapshot
        self.clear()
        if snapshot is None:
            return
        counts = category_counts(detail_grid(snapshot))
        bars, errors = self.plots
        positions = np.arange(4)[::-1]
        for y, label, color in zip(positions, DETAIL_LABELS, DETAIL_COLORS, strict=True):
            count = counts[label]
            bars.addItem(
                pg.BarGraphItem(x0=0, y=[y], width=[count], height=0.65, brush=color, pen=None)
            )
            bars.text(str(count), count, y, anchor=(0, 0.5))
        bars.getAxis("left").setTicks([list(zip(positions, DETAIL_LABELS, strict=True))])
        bars.setXRange(0, max(1, max(counts.values())) * 1.18, padding=0)
        bars.setYRange(-0.6, 3.6, padding=0)
        for matched, label, color in (
            (True, "Matched", "#1accf2"),
            (False, "Missed, with reports", "#f2731a"),
        ):
            values = snapshot["nearest_error_m"][snapshot["outcomes"] == int(matched)]
            values = np.sort(values[np.isfinite(values)])
            if len(values):
                errors.curve(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    name=label,
                    color=color,
                    symbol="o",
                )
        if not np.isfinite(snapshot["nearest_error_m"]).any():
            errors.set_empty_message("No reports with a saved position")
        errors.setYRange(0, 1, padding=0.03)


class ContributionPlot(InlinePlotGrid):
    """Use the same matched-estimate attribution as the categorical volume."""

    def __init__(self):
        super().__init__([("Sensor contributions", "Number of cells", "")])
        self.data = None
        self.setMinimumHeight(230)
        self.setToolTip(
            "Sensors contributing to the currently matched estimate in each measured cell."
        )

    def update_contributions(self, data):
        if self.data is data:
            return
        self.data = data
        self.clear()
        plot = self.plots[0]
        labels, colors = data["labels"], data["colors"]
        counts = [int(np.sum(data["values"] == i)) for i in range(len(labels))]
        positions = np.arange(len(labels))[::-1]
        for y, count, color in zip(positions, counts, colors, strict=True):
            plot.addItem(
                pg.BarGraphItem(x0=0, y=[y], width=[count], height=0.65, brush=color, pen=None)
            )
            plot.text(str(count), count, y, anchor=(0, 0.5))
        plot.getAxis("left").setTicks([list(zip(positions, labels, strict=True))])
        plot.setXRange(0, max(1, max(counts)) * 1.18, padding=0)
        plot.setYRange(-0.6, len(labels) - 0.4, padding=0)
