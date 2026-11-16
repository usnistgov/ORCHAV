"""On-demand height comparison for the current saved coverage map."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QVBoxLayout, QWidget

from shared.coverage.comparison import native_coverage_bundle, weighted_subset_summary
from shared.coverage.spatial import coverage_roi_mask
from shared.coverage.volume import coverage_file_identity

from ..widgets.form_layout import AlignedFormLayout
from ..widgets.inline_plot import InlinePlot


class CoverageHeightComparison(QWidget):
    """Read once on demand; threshold/region changes reuse the saved samples."""

    def __init__(self, path, identity, parent=None):
        super().__init__(parent)
        self.path, self.identity = path, identity
        self.axes = self.bounds = self.data = self.pending = None
        self._context_key = None
        self._executor = None
        self._retired = False
        self.summaries = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("Compare heights")
        self.enabled.setToolTip(
            "Percentage of cells meeting a received-power threshold at each saved height."
        )
        layout.addWidget(self.enabled)
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        form = AlignedFormLayout()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-250, 100)
        self.threshold.setValue(-75)
        self.threshold.setSingleStep(0.5)
        self.threshold.setSuffix(" dBm")
        self.threshold.setKeyboardTracking(False)
        form.addRow("Power ≥", self.threshold)
        self.transmitter = QComboBox()
        self.transmitter.setToolTip(
            "All transmitters: at least one must meet the threshold at a cell."
        )
        form.addRow("Transmitter", self.transmitter)
        self.form = form
        body_layout.addLayout(form)
        self.plot = InlinePlot("Coverage by height", "Height (m)", "Cells meeting threshold (%)")
        self.plot.setMinimumHeight(280)
        self.plot.setToolTip(
            "Each height uses all cell centres in View's XYZ region. Zero contributions fail the threshold. "
            "Shading bounds uncertainty from unavailable samples; it is not a confidence interval."
        )
        body_layout.addWidget(self.plot)
        layout.addWidget(self.body)
        self.body.hide()
        self.poll = QTimer(self)
        self.poll.setInterval(50)
        self.poll.timeout.connect(self._poll)
        self.enabled.toggled.connect(self._toggle)
        self.threshold.valueChanged.connect(self._draw)
        self.transmitter.currentIndexChanged.connect(self._draw)

    def set_context(self, axes, bounds):
        self.axes = axes
        key = tuple(map(tuple, bounds)) if bounds is not None else None
        self.bounds = bounds
        if key != self._context_key:
            self._context_key = key
            self._draw()

    def _toggle(self, enabled):
        self.body.setVisible(enabled)
        if not enabled:
            return
        if self.data is None and self.pending is None:
            self.plot.set_empty_message("Loading coverage…")
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="coverage-heights"
            )
            self.pending = self._executor.submit(self._read)
            self.poll.start()
        else:
            self._draw()

    def _read(self):
        if coverage_file_identity(self.path) != self.identity:
            raise ValueError("Coverage file changed; reopen the map")
        data = native_coverage_bundle(self.path)
        if coverage_file_identity(self.path) != self.identity:
            raise ValueError("Coverage file changed while reading")
        return data

    def _poll(self):
        if self.pending is None or not self.pending.done():
            return
        future, self.pending = self.pending, None
        self.poll.stop()
        self._executor.shutdown(wait=False)
        try:
            self.data = future.result()
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.plot.set_empty_message(f"Comparison unavailable: {exc}")
            return
        names = self.data["site_names"].tolist()
        self.transmitter.blockSignals(True)
        self.transmitter.clear()
        if len(names) > 1:
            self.transmitter.addItem("All transmitters", names)
        for name in names:
            self.transmitter.addItem(name, [name])
        self.transmitter.blockSignals(False)
        self.form.setRowVisible(self.transmitter, len(names) > 1)
        self._draw()

    def _draw(self):
        if not self.enabled.isChecked() or self.data is None or self.axes is None:
            return
        self.plot.clear()
        self.plot.getPlotItem().legend.hide()
        self.summaries = []
        try:
            region = coverage_roi_mask(self.axes, self.bounds).reshape(len(self.axes[2]), -1)
        except ValueError as exc:
            self.plot.set_empty_message(str(exc))
            return
        selected = np.flatnonzero(region.any(axis=1))
        names = self.data["site_names"].tolist()
        self.summaries = [
            weighted_subset_summary(
                self.data["rss_dbm"][index],
                names,
                subsets=[self.transmitter.currentData()],
                thresholds_dbm=[self.threshold.value()],
                eligible=region[index],
                availability=self.data["availability"][index],
            )
            for index in selected
        ]
        heights = self.axes[2][selected]
        lower = np.array([r["fraction_lower"][0, 0] * 100 for r in self.summaries])
        upper = np.array([r["fraction_upper"][0, 0] * 100 for r in self.summaries])
        uncertain = bool(np.any(upper > lower))
        self.plot.getPlotItem().legend.setVisible(uncertain)
        curve = self.plot.curve(heights, lower, symbol="o", name="Confirmed" if uncertain else None)
        if uncertain:
            ceiling = self.plot.curve(heights, upper, symbol="o", name="Including unknown")
            self.plot.addItem(pg.FillBetweenItem(curve, ceiling, brush=(70, 130, 210, 40)))
        self.plot.setYRange(0, 100, padding=0.03)
        self.plot.enableAutoRange(axis="x")

    def shutdown(self):
        """Retire this source's work and plot before its Qt parent is deleted."""
        if self._retired:
            return
        self.poll.stop()
        if self.pending is not None:
            self.pending.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.data = None
        # Qt parent deletion can precede Python-wrapper collection. Remove
        # this owned view before pyqtgraph's later application-exit cleanup.
        self.plot.getViewBox().unregister()
        self._retired = True
