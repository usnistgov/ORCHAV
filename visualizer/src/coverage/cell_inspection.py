"""Lazy bounded per-transmitter inspection of a selected recorded grid cell."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.coverage.hdf5 import CoverageHDF5Reader
from shared.coverage.volume import coverage_file_identity

from .cell_charts import CoverageCellCharts


class CoverageCellTable(QWidget):
    """Coalesce cell changes and reuse at most sixteen tiny cell records."""

    def __init__(self, path, identity, parent, *, tx_count=None):
        super().__init__(parent)
        self.path, self.identity = path, identity
        self.key = self.loaded = self.pending = None
        self.cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coverage-cell")
        self.timer = QTimer(self)
        self.timer.setInterval(75)
        self.timer.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)
        self.enabled = QCheckBox("Compare transmitters")
        self.enabled.setToolTip("Compare saved transmitter power and SINR at this cell.")
        self._can_compare = tx_count is None or tx_count > 1
        self.enabled.setVisible(self._can_compare)
        self.enabled.setEnabled(self._can_compare)
        self.enabled.toggled.connect(lambda *_: self.request(self.key))
        layout.addWidget(self.enabled)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Transmitter", "Power (dBm)", "SINR (dB)", "Availability"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.hide()
        layout.addWidget(self.table)
        self.charts = CoverageCellCharts(self)
        self.charts.enabled.toggled.connect(lambda *_: self.request(self.key))
        layout.addWidget(self.charts)

    def request(self, indices):
        previous = self.key
        self.key = tuple(int(i) for i in indices) if indices is not None else None
        if previous != self.key:
            self.charts.set_data(None)
        compare = self._can_compare and self.enabled.isChecked()
        self.table.setVisible(compare and self.key is not None)
        self.note.setVisible(compare and (self.key is None or self.key != self.loaded))
        if self.key is None:
            self.note.setText("Select a cell.")
        if not (compare or self.charts.enabled.isChecked()) or self.key is None:
            return
        if self.key == self.loaded and self.charts.data is self.cache.get(self.key):
            return
        if self.key in self.cache:
            self._display(self.cache[self.key], self.key)
        elif self.pending is None:
            self.pending_key = self.key
            self.note.setText("Loading cell…")
            self.table.setRowCount(0)
            self.pending = self.executor.submit(self._read, self.key)
            self.timer.start()

    def _read(self, key):
        if coverage_file_identity(self.path) != self.identity:
            raise ValueError("Coverage file changed; reopen the recording")
        data = CoverageHDF5Reader(self.path).read_cell(key)
        if coverage_file_identity(self.path) != self.identity:
            raise ValueError("Coverage file changed while reading the cell")
        return data

    def _poll(self):
        if self.pending is None or not self.pending.done():
            return
        future, key = self.pending, self.pending_key
        self.pending = None
        self.timer.stop()
        try:
            data = future.result()
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.note.setText(f"Cell comparison unavailable: {exc}")
            self.charts.set_data(None, error=f"Cell comparison unavailable: {exc}")
            return
        self.cache[key] = data
        self.cache.move_to_end(key)
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        self.request(self.key)

    def _display(self, data, key):
        self.loaded = key
        self.charts.set_data(data)
        self.cache.move_to_end(key)
        self.table.setRowCount(len(data["names"]))
        for row, name in enumerate(data["names"]):
            code = data["availability"][row]
            positive = code == 1
            values = [
                name + (" · serving" if row == data["serving_tx"] and positive else ""),
                f"{data['rss_dbm'][row]:.2f}" if positive else "N/A",
                (
                    f"{data['sinr_db'][row]:.2f}"
                    if positive and np.isfinite(data["sinr_db"][row])
                    else "N/A"
                ),
                {0: "Unknown", 1: "Finite power", 2: "Zero contribution", 3: "Invalid"}[code],
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.setMaximumHeight(min(240, 42 + 31 * len(data["names"])))
        self.note.clear()
        self.note.hide()
        self.table.setToolTip(
            "Same saved XYZ cell; SINR uses all recorded transmitters and saved noise. Display filters and smoothing do not alter these values."
        )

    def shutdown(self):
        self.timer.stop()
        self.charts.shutdown()
        if self.pending is not None:
            self.pending.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.cache.clear()
