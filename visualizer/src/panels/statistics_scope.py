"""Explicit link/interval selection for recorded-run path statistics."""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLayout,
    QPushButton,
    QSizePolicy,
)

from ..metrics.statistics_analysis import statistics_coordinates


class StatisticsScopeBar(QFrame):
    """Select a saved subset only on Apply; playback never changes this scope."""

    selection_requested = Signal(object, object)
    reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self._frames = np.empty(0, dtype=np.int64)
        self._coordinates = np.empty(0)
        self._columns = None
        self.pair = QComboBox()
        self.pair.addItem("All links", None)
        self.pair.setToolTip("Select an exact recorded Tx/Rx pair; device IDs are unchanged.")
        self.interval = QCheckBox("Interval")
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for spin, tip in (
            (self.start, "Inclusive interval start"),
            (self.end, "Inclusive interval end"),
        ):
            spin.setToolTip(tip)
            spin.setMinimumWidth(80)
            spin.setEnabled(False)
        self.apply = QPushButton("Apply")
        self.reset = QPushButton("Whole run")
        self.apply.setToolTip("Read saved path metrics in the background for this selection.")
        self.reset.setToolTip("Restore the already computed whole-run statistics.")
        self.grid = QGridLayout(self)
        self.grid.setSizeConstraint(QLayout.SetNoConstraint)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.interval.toggled.connect(self._interval_changed)
        self.apply.clicked.connect(self._apply)
        self.reset.clicked.connect(self.reset_requested)
        self.setEnabled(False)
        self._reflow()

    def bind(self, stats, *, enabled):
        """Populate choices from the full run, never from a filtered subset."""
        self._frames = np.asarray(stats.get("frame_indices", []), dtype=np.int64)
        self._coordinates, axis = statistics_coordinates(
            self._frames, stats.get("frame_times_s", [])
        )
        self.pair.clear()
        self.pair.addItem("All links", None)
        for tx, rx in stats.get("tx_rx_pairs", []):
            self.pair.addItem(f"Tx {tx} → Rx {rx}", (int(tx), int(rx)))
        self.interval.setChecked(False)
        if self._frames.size:
            for spin in (self.start, self.end):
                spin.setDecimals(6 if axis == "Time (s)" else 0)
                spin.setSuffix(" s" if axis == "Time (s)" else "")
                spin.setRange(float(self._coordinates.min()), float(self._coordinates.max()))
            self.start.setValue(float(self._coordinates.min()))
            self.end.setValue(float(self._coordinates.max()))
        self.interval.setText("Time" if axis == "Time (s)" else "Frames")
        self.interval.setToolTip(
            "Restrict the interval using saved timestamps."
            if axis == "Time (s)"
            else "Timestamps are unavailable; select recorded frame indices."
        )
        self.setEnabled(bool(enabled and self._frames.size))
        self._reflow()

    def _interval_changed(self, enabled):
        self.start.setEnabled(enabled)
        self.end.setEnabled(enabled)

    def _apply(self):
        steps = None
        if self.interval.isChecked():
            low, high = self.start.value(), self.end.value()
            if low > high:
                self.start.setFocus()
                return
            tolerance = 0.5 * 10 ** -self.start.decimals()
            steps = tuple(
                map(
                    int,
                    self._frames[
                        (self._coordinates >= low - tolerance)
                        & (self._coordinates <= high + tolerance)
                    ],
                )
            )
        self.selection_requested.emit(self.pair.currentData(), steps)

    def _reflow(self):
        widgets = (self.pair, self.interval, self.start, self.end, self.apply, self.reset)
        required_width = sum(
            max(widget.minimumWidth(), widget.minimumSizeHint().width()) for widget in widgets
        ) + self.grid.horizontalSpacing() * (len(widgets) - 1)
        wide = self.width() >= max(640, required_width)
        if wide == self._columns:
            return
        for widget in widgets:
            self.grid.removeWidget(widget)
        for column in range(6):
            self.grid.setColumnStretch(column, 0)
        for index, widget in enumerate(widgets):
            if wide:
                self.grid.addWidget(widget, 0, index)
            else:
                # Keep two independent fields per row; the previous overlapping
                # column spans squeezed captions and times at larger UI scales.
                positions = (
                    (0, 0, 1, 1),
                    (0, 1, 1, 1),
                    (1, 0, 1, 1),
                    (1, 1, 1, 1),
                    (2, 0, 1, 1),
                    (2, 1, 1, 1),
                )
                self.grid.addWidget(widget, *positions[index])
        self.grid.setColumnStretch(0 if wide else 1, 1)
        self._columns = wide

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()
