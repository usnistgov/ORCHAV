"""Small native comparison workflow over immutable saved volume reports."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from shared.sensing.volume.comparison import (
    MAX_SOURCES,
    compare_volumes,
    export_comparison,
    load_definition,
    save_definition,
    source_definition,
    validate_definition,
)
from shared.sensing.volume.reader import SensingVolume

from ..widgets.form_layout import AlignedFormLayout
from ..widgets.inline_plot import InlinePlotGrid
from .comparison_plots import volume_comparison


class VolumeComparisonDialog(QDialog):
    """Select sources, rematch, inspect the parent scene, and export one recipe."""

    inspect_requested = Signal(str)

    def __init__(self, parent=None, *, initial_volume=None, gate=1):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("ORCHAV — Compare saved sensing volumes")
        self.resize(1100, 900)
        self.sources = []
        self.result = None
        self.pending = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="volume-comparison")
        self.destroyed.connect(partial(self.executor.shutdown, wait=False, cancel_futures=True))
        outer = QVBoxLayout(self)
        self.notice = QLabel(
            "Compare saved reports at a common 3D scoring gate. Detector outputs stay unchanged."
        )
        self.notice.setWordWrap(True)
        outer.addWidget(self.notice)
        self.sources_table = QTableWidget(0, 3)
        self.sources_table.setHorizontalHeaderLabels(
            ["Condition label (editable)", "Authored motion", "Saved volume"]
        )
        self.sources_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sources_table.setMaximumHeight(190)
        self.sources_table.horizontalHeader().setStretchLastSection(True)
        self.sources_table.itemChanged.connect(self._invalidate)
        outer.addWidget(self.sources_table)
        row = QHBoxLayout()
        for name, text, callback in (
            ("add_button", "Add volume…", self._add),
            ("remove_button", "Remove selected", self._remove),
            ("inspect_button", "Inspect selected in 3D", self._inspect),
            ("load_button", "Load comparison…", self._load),
            ("save_button", "Save comparison…", self._save),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            setattr(self, name, button)
            row.addWidget(button)
        outer.addLayout(row)
        form = AlignedFormLayout()
        self.gate = QDoubleSpinBox()
        self.gate.setRange(0.001, 10000)
        self.gate.setDecimals(3)
        self.gate.setSuffix(" m")
        self.gate.setValue(gate)
        self.gate.setKeyboardTracking(False)
        self.gate.valueChanged.connect(self._invalidate)
        form.addRow("3D position match gate", self.gate)
        self.support = QComboBox()
        self.support.addItem("Common measured cells (identical grids)", "common_measured")
        self.support.addItem("Each source's measured cells (denominators shown)", "per_source")
        self.support.currentIndexChanged.connect(self._invalidate)
        form.addRow("Evaluation support", self.support)
        outer.addLayout(form)
        row = QHBoxLayout()
        self.run_button = QPushButton("Compare saved reports")
        self.run_button.clicked.connect(self.analyze)
        row.addWidget(self.run_button)
        self.export_button = QPushButton("Export numbers and figures…")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        row.addWidget(self.export_button)
        outer.addLayout(row)
        self.status = QLabel("Choose sources, labels and a motion, then compare.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.summary = QTableWidget(0, 8)
        self.summary.setHorizontalHeaderLabels(
            [
                "Condition",
                "Locations",
                "Matched",
                "Extras",
                "Empty",
                "Out of gate",
                "RMSE (m)",
                "Median (m)",
            ]
        )
        self.summary.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary.setMaximumHeight(175)
        outer.addWidget(self.summary)
        self.canvas = InlinePlotGrid(
            [
                ("Saved reports", "Condition", "Count"),
                ("Matched position error", "Error (m)", "Conditional cumulative fraction"),
            ],
            columns=2,
        )
        outer.addWidget(self.canvas, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        if initial_volume is not None:
            self.add_volume(initial_volume)

    def add_volume(self, path, *, label=None, heading=None):
        if len(self.sources) >= MAX_SOURCES:
            raise ValueError(f"At most {MAX_SOURCES} conditions can be compared")
        reader = SensingVolume(path)
        heading = heading or reader.headings[0]["name"]
        source = source_definition(path, label=label, heading=heading)
        row = self.sources_table.rowCount()
        self.sources_table.insertRow(row)
        self.sources.append(source)
        self.sources_table.setItem(row, 0, QTableWidgetItem(source["label"]))
        combo = QComboBox()
        combo.addItems([h["name"] for h in reader.headings])
        combo.setCurrentText(heading)
        combo.currentIndexChanged.connect(self._invalidate)
        self.sources_table.setCellWidget(row, 1, combo)
        item = QTableWidgetItem(source["path"])
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.sources_table.setItem(row, 2, item)
        self.sources_table.resizeColumnsToContents()
        self._invalidate()

    def definition(self):
        sources = []
        for i, source in enumerate(self.sources):
            sources.append(
                dict(
                    source,
                    label=self.sources_table.item(i, 0).text().strip(),
                    heading=self.sources_table.cellWidget(i, 1).currentText(),
                )
            )
        return dict(
            schema_version=1,
            sources=sources,
            support=self.support.currentData(),
            position_gate_m=self.gate.value(),
        )

    def set_definition(self, definition):
        validate_definition(definition)
        for source in definition["sources"]:
            actual = source_definition(
                source["path"], label=source["label"], heading=source["heading"]
            )
            if actual["sha256"] != source["sha256"]:
                raise ValueError(f"Saved source identity changed: {source['label']}")
        self.sources.clear()
        self.sources_table.setRowCount(0)
        for source in definition["sources"]:
            self.add_volume(source["path"], label=source["label"], heading=source["heading"])
        self.gate.setValue(definition["position_gate_m"])
        self.support.setCurrentIndex(self.support.findData(definition["support"]))
        self._invalidate()

    def _invalidate(self, *_):
        self.result = None
        self.export_button.setEnabled(False)
        self.status.setText("Settings changed. Compare saved reports to refresh the result.")

    def analyze(self):
        if self.pending is not None:
            return
        definition = self.definition()
        self.result = None
        self.export_button.setEnabled(False)
        self.pending_definition = definition
        self.pending = self.executor.submit(compare_volumes, definition)
        self.run_button.setEnabled(False)
        self.status.setText("Reading saved reports and applying the evaluation gate…")

    def _poll(self):
        if self.pending is None or not self.pending.done():
            return
        pending, self.pending = self.pending, None
        self.run_button.setEnabled(True)
        try:
            result = pending.result()
            if self.pending_definition != self.definition():
                self.status.setText("Settings changed during loading. Compare again.")
                return
            self.result = result
            self.summary.setRowCount(len(result["conditions"]))
            fields = (
                "denominator",
                "matched_locations",
                "extra_reports",
                "empty_outputs",
                "out_of_gate_outputs",
                "matched_rmse_m",
                "matched_median_error_m",
            )
            for i, condition in enumerate(result["conditions"]):
                values = [condition["source"]["label"], *(condition["summary"][k] for k in fields)]
                for j, value in enumerate(values):
                    self.summary.setItem(
                        i,
                        j,
                        QTableWidgetItem(
                            "—"
                            if value is None
                            else f"{value:.3f}" if isinstance(value, float) else str(value)
                        ),
                    )
            self.summary.resizeColumnsToContents()
            volume_comparison(self.canvas, result)
            self.export_button.setEnabled(True)
            self.status.setText(
                "Complete. Pending/excluded cells are outside each denominator. Errors describe matched locations only. Use Inspect selected for pose, reporting hemisphere and full-sphere receive-array response."
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText(f"Comparison unavailable: {exc}")

    def export_to(self, directory):
        if self.result is None:
            raise ValueError("Compare the current settings before exporting")
        export_comparison(self.result, directory)
        self.status.setText(
            f"Exported exact recipe, source hashes, all cell values, CSV, PNG, PDF and SVG to {directory}"
        )

    def _add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Saved sensing volumes",
            "",
            "Volume manifests (sensing_volume.json);;JSON (*.json)",
        )
        for path in paths:
            try:
                self.add_volume(path)
            except (OSError, ValueError, KeyError) as exc:
                self.status.setText(str(exc))

    def _remove(self):
        row = self.sources_table.currentRow()
        if row >= 0:
            self.sources.pop(row)
            self.sources_table.removeRow(row)
            self._invalidate()

    def _inspect(self):
        row = self.sources_table.currentRow()
        if row >= 0:
            self.inspect_requested.emit(self.sources[row]["path"])

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load volume comparison", "", "JSON (*.json)")
        if path:
            try:
                self.set_definition(load_definition(path))
            except (OSError, ValueError, KeyError) as exc:
                self.status.setText(str(exc))

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save volume comparison", "comparison_definition.json", "JSON (*.json)"
        )
        if path:
            try:
                save_definition(Path(path), self.definition())
                self.status.setText(f"Saved source identities and comparison settings to {path}")
            except (OSError, ValueError, KeyError) as exc:
                self.status.setText(str(exc))

    def _export(self):
        directory = QFileDialog.getExistingDirectory(self, "Export comparison bundle")
        if directory:
            try:
                self.export_to(directory)
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))

    def closeEvent(self, event):
        self.timer.stop()
        if self.pending is not None:
            self.pending.cancel()
        super().closeEvent(event)
