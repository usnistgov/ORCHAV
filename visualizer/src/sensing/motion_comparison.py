"""Small native controls for reusable raw-CIR motion comparison and export."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.sensing.motion_analysis import (
    SCHEMA,
    MotionRecipe,
    analyze_comparison,
    export_comparison,
    load_definition,
    load_export,
)

from ..widgets.form_layout import AlignedFormLayout


class MotionComparisonDialog(QDialog):
    """Explicit analysis runs own results; editing controls invalidates export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ORCHAV · Motion comparison")
        self.resize(1360, 820)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motion-analysis")
        self.pending = None
        self.comparison = None
        self.canvas = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(150)
        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "Compare modeled motion signatures from retained complex CIR; source epochs are authoritative."
            )
        )
        split = QSplitter()
        root.addWidget(split, 1)
        self.controls = QWidget()
        left = QVBoxLayout(self.controls)
        left.addWidget(
            QLabel("Labeled source recordings (frames directory; optional source references):")
        )
        self.sources = QTableWidget(0, 4)
        self.sources.setMaximumHeight(210)
        self.sources.setHorizontalHeaderLabels(["Label", "Frames", "Root JSON", "Motion manifest"])
        self.sources.horizontalHeader().setStretchLastSection(True)
        self.sources.itemChanged.connect(self._invalidate)
        left.addWidget(self.sources)
        buttons = QHBoxLayout()
        for label, slot in (
            ("Add recording…", self._add_source),
            ("Remove selected", self._remove_source),
            ("Load definition…", self._load),
            ("Save definition…", self._save),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        left.addLayout(buttons)
        form = AlignedFormLayout()
        self.range_min = self._float(0, 1e6, 0)
        self.range_max = self._float(0, 1e6, 100)
        self.time_start, self.time_end = QLineEdit(), QLineEdit()
        self.time_start.setPlaceholderText("Full source interval")
        self.time_end.setPlaceholderText("Full source interval")
        self.window_samples, self.hop_samples = QSpinBox(), QSpinBox()
        self.window_samples.setRange(3, 65536)
        self.window_samples.setValue(128)
        self.hop_samples.setRange(1, 65536)
        self.hop_samples.setValue(32)
        self.window = QComboBox()
        self.window.addItems(["hann", "boxcar"])
        self.mean = QComboBox()
        self.mean.addItems(["retained", "removed", "both"])
        self.normalization = QComboBox()
        self.normalization.addItems(["absolute", "per_record_peak"])
        self.color_min, self.color_max = (
            self._float(-1000, 1000, -185),
            self._float(-1000, 1000, -115),
        )
        self.doppler_min, self.doppler_max = (
            self._float(-1e7, 1e7, -500),
            self._float(-1e7, 1e7, 500),
        )
        for label, field in (
            ("Range minimum (m)", self.range_min),
            ("Range maximum (m)", self.range_max),
            ("Start epoch (s)", self.time_start),
            ("End epoch (s)", self.time_end),
            ("Temporal window (samples)", self.window_samples),
            ("Hop (samples)", self.hop_samples),
            ("Window (symmetric Hann)", self.window),
            ("Complex mean", self.mean),
            ("Power normalization", self.normalization),
            ("Common color minimum (dB)", self.color_min),
            ("Common color maximum (dB)", self.color_max),
            ("Display Doppler minimum (Hz)", self.doppler_min),
            ("Display Doppler maximum (Hz)", self.doppler_max),
        ):
            form.addRow(label, field)
            signal = (
                field.textChanged
                if isinstance(field, QLineEdit)
                else (
                    field.currentIndexChanged
                    if isinstance(field, QComboBox)
                    else field.valueChanged
                )
            )
            signal.connect(self._invalidate)
        left.addLayout(form)
        self.analyze_button = QPushButton("Analyze raw observations")
        self.analyze_button.clicked.connect(self.analyze)
        self.export_button = QPushButton("Export spectra, metadata and figures…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        self.open_export_button = QPushButton("Inspect saved export…")
        self.open_export_button.clicked.connect(self._open_export)
        self.root_button = QPushButton("Inspect source-root motion")
        self.root_button.setEnabled(False)
        self.root_button.clicked.connect(self._show_roots)
        left.addWidget(self.analyze_button)
        left.addWidget(self.export_button)
        left.addWidget(self.open_export_button)
        left.addWidget(self.root_button)
        self.status = QLabel("Load a definition or add recordings. No detector is rerun.")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        split.addWidget(self.controls)
        split.addWidget(self.plot_container)
        split.setSizes([550, 800])

    @staticmethod
    def _float(low, high, value):
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(4)
        box.setValue(value)
        return box

    def _invalidate(self, *_args):
        self.comparison = None
        self.export_button.setEnabled(False)
        self.root_button.setEnabled(False)
        self.status.setText("Settings changed. Analyze explicitly before exporting a new result.")

    def definition(self):
        sources = []
        for row in range(self.sources.rowCount()):
            values = [
                self.sources.item(row, col).text().strip() if self.sources.item(row, col) else ""
                for col in range(4)
            ]
            sources.append(
                dict(
                    zip(("label", "frames", "root_motion", "motion_reference"), values, strict=True)
                )
            )
        return {
            "schema": SCHEMA,
            "sources": sources,
            "recipe": {
                "range_min_m": self.range_min.value(),
                "range_max_m": self.range_max.value(),
                "time_start_s": (
                    float(self.time_start.text()) if self.time_start.text().strip() else None
                ),
                "time_end_s": float(self.time_end.text()) if self.time_end.text().strip() else None,
                "window_samples": self.window_samples.value(),
                "hop_samples": self.hop_samples.value(),
                "window": self.window.currentText(),
                "mean": self.mean.currentText(),
            },
            "display": {
                "normalization": self.normalization.currentText(),
                "color_limits_db": [self.color_min.value(), self.color_max.value()],
                "doppler_limits_hz": [self.doppler_min.value(), self.doppler_max.value()],
            },
        }

    def set_definition(self, definition):
        MotionRecipe(**definition.get("recipe", {}))
        if definition.get("display", {}).get("normalization", "absolute") not in {
            "absolute",
            "per_record_peak",
        }:
            raise ValueError("Unsupported power normalization")
        self.sources.setRowCount(len(definition["sources"]))
        for row, source in enumerate(definition["sources"]):
            for col, name in enumerate(("label", "frames", "root_motion", "motion_reference")):
                item = QTableWidgetItem(source.get(name, ""))
                item.setToolTip(source.get(name, ""))
                self.sources.setItem(row, col, item)
        recipe = definition.get("recipe", {})
        for key, widget, default in (
            ("range_min_m", self.range_min, 0),
            ("range_max_m", self.range_max, 100),
            ("window_samples", self.window_samples, 128),
            ("hop_samples", self.hop_samples, 32),
        ):
            widget.setValue(recipe.get(key, default))
        self.time_start.setText(
            "" if recipe.get("time_start_s") is None else str(recipe["time_start_s"])
        )
        self.time_end.setText("" if recipe.get("time_end_s") is None else str(recipe["time_end_s"]))
        self.window.setCurrentText(recipe.get("window", "hann"))
        self.mean.setCurrentText(recipe.get("mean", "retained"))
        display = definition.get("display", {})
        self.normalization.setCurrentText(display.get("normalization", "absolute"))
        for widget, value in zip(
            (self.color_min, self.color_max, self.doppler_min, self.doppler_max),
            display.get("color_limits_db", [-185, -115])
            + display.get("doppler_limits_hz", [-500, 500]),
            strict=True,
        ):
            widget.setValue(value)

    def _add_source(self):
        path = QFileDialog.getExistingDirectory(self, "Select retained recording")
        if path:
            row = self.sources.rowCount()
            self.sources.insertRow(row)
            for col, value in enumerate((Path(path).parent.name, path, "", "")):
                self.sources.setItem(row, col, QTableWidgetItem(value))

    def _remove_source(self):
        for row in sorted({item.row() for item in self.sources.selectedItems()}, reverse=True):
            self.sources.removeRow(row)
        self._invalidate()

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load motion comparison definition", "", "JSON (*.json)"
        )
        if path:
            try:
                self.set_definition(load_definition(Path(path)))
            except (OSError, ValueError, KeyError) as exc:
                self.status.setText(str(exc))

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save motion comparison definition", "motion_definition.json", "JSON (*.json)"
        )
        if path:
            try:
                Path(path).write_text(
                    json.dumps(self.definition(), indent=2, allow_nan=False) + "\n"
                )
                self.status.setText(f"Saved comparison definition: {path}")
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))

    def analyze(self):
        if self.pending is not None:
            return
        try:
            definition = self.definition()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._invalidate()
        self.status.setText("Reading typed CIR and computing the declared recipe…")
        self.controls.setEnabled(False)
        self.pending = self.executor.submit(analyze_comparison, definition)

    def _poll(self):
        if self.pending is None or not self.pending.done():
            return
        future, self.pending = self.pending, None
        self.controls.setEnabled(True)
        try:
            self.show_comparison(future.result())
        except Exception as exc:  # UI boundary: retain an actionable refusal, never a stale export.
            self.status.setText(f"Analysis unavailable: {exc}")

    def show_comparison(self, comparison, *, portable=False):
        from .comparison_plots import motion_comparison

        canvas = motion_comparison(comparison)
        if self.canvas is not None:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        self.canvas = canvas
        self.plot_layout.addWidget(self.canvas)
        self.comparison = comparison
        self.export_button.setEnabled(True)
        self.root_button.setEnabled(any("root_displacement_m" in r for r in comparison["results"]))
        info = [
            f"{r['label']}: {r['metadata']['sample_count']} samples, {r['metadata']['window_count']} windows"
            for r in comparison["results"]
        ]
        prefix = (
            "Saved spectral export; raw data are needed for changed processing. "
            if portable
            else "Analysis complete. "
        )
        self.status.setText(prefix + "; ".join(info))

    def _show_roots(self):
        if self.comparison is None:
            return
        from .comparison_plots import root_motion

        window = QDialog(self)
        window.setWindowTitle("Source-root motion · authored reference")
        window.resize(900, 480)
        layout = QVBoxLayout(window)
        canvas = root_motion(self.comparison)
        layout.addWidget(canvas)
        window.show()

    def _export(self):
        if self.comparison is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Choose a NEW export directory", "motion_export", "Directory name (*)"
        )
        if path:
            try:
                export_comparison(self.comparison, Path(path))
                self.status.setText(
                    f"Export complete: {path} · spectra.npz, comparison.json, definition.json, PNG and PDF"
                )
            except (OSError, ValueError) as exc:
                self.status.setText(f"Export unavailable: {exc}")

    def _open_export(self):
        path = QFileDialog.getExistingDirectory(self, "Inspect a saved motion export")
        if path:
            try:
                comparison = load_export(Path(path))
                self.set_definition(comparison["definition"])
                self.show_comparison(comparison, portable=True)
            except (OSError, ValueError, KeyError) as exc:
                self.status.setText(f"Export unavailable: {exc}")

    def closeEvent(self, event):
        self.timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)
