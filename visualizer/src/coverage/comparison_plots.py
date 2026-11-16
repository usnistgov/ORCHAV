"""Interactive coverage plots over the shared, unchanged comparison results."""

import numpy as np
import pyqtgraph as pg

from ..widgets.inline_plot import SERIES_COLORS


def draw_comparison(canvas, result, *, row_index, threshold_dbm):
    canvas.clear()
    left, right = canvas.plots
    rows = result["rows"]
    row = rows[min(max(0, row_index), len(rows) - 1)]
    thresholds = np.asarray(row["thresholds_dbm"])
    left.configure(title=row["label"])
    for index, (subset, lo, hi) in enumerate(
        zip(row["subsets"], row["fraction_lower"], row["fraction_upper"], strict=True)
    ):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        lower = left.curve(thresholds, 100 * np.asarray(lo), color=color, name="+".join(subset))
        upper = left.curve(thresholds, 100 * np.asarray(hi), color=color, width=0)
        shade = pg.mkColor(color)
        shade.setAlphaF(0.15)
        left.addItem(pg.FillBetweenItem(lower, upper, brush=shade))
    left.guide(threshold_dbm)
    left.setYRange(-1, 101, padding=0)
    selected = np.flatnonzero(np.isclose(thresholds, threshold_dbm, rtol=0, atol=1e-12))
    if not len(selected):
        raise ValueError("The displayed criterion must be included in the analysis thresholds")
    index = int(selected[0])
    matrix = np.asarray([r["fraction_lower"] for r in rows])[:, :, index] * 100
    right.heatmap(
        np.arange(matrix.shape[1]),
        np.arange(matrix.shape[0]),
        matrix,
        limits=(0, 100),
        colormap=pg.ColorMap([0, 1], ["#f7fbff", "#08306b"]),
        label="%",
    )
    right.getViewBox().invertY(True)
    right.getAxis("bottom").setTicks([list(enumerate(["+".join(s) for s in row["subsets"]]))])
    right.getAxis("left").setTicks([list(enumerate([r["label"] for r in rows]))])
    for (i, j), value in np.ndenumerate(matrix):
        right.text(
            f"{value:.1f}", j, i, color="white" if value > 60 else "black", anchor=(0.5, 0.5)
        )
    bounded = any(
        np.any(
            np.asarray(r["fraction_upper"])[:, index] > np.asarray(r["fraction_lower"])[:, index]
        )
        for r in rows
    )
    right.configure(
        title=f"{'Lower-bound fraction' if bounded else 'Fraction'} at {threshold_dbm:g} dBm (%)"
    )
