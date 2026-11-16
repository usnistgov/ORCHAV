"""A thin native Qt view over reproducible coverage/support comparisons."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from shared.coverage.comparison import analyze_definition, file_identity, load_sample_bundle
from shared.coverage.comparison_export import export_comparison

from ..widgets.inline_plot import InlinePlotGrid
from .comparison_plots import draw_comparison


class CoverageComparisonWindow(QDialog):
    """Select saved studies/supports, inspect consistent curves and export data."""

    def __init__(
        self,
        parent=None,
        *,
        definition_path=None,
        native_path=None,
        roi_bounds=None,
        threshold_dbm=None,
    ):
        super().__init__(parent)

        self.setWindowTitle("ORCHAV — Coverage and observation-support comparison")
        self.resize(1320, 820)
        self.definition = {
            "schema_version": 1,
            "kind": "coverage_comparison",
            "thresholds_dbm": np.arange(-95, -54, dtype=float).tolist(),
            "sources": [],
        }
        self.initial_roi_bounds = roi_bounds
        self.result = None
        self._updating = False
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        for text, callback in [
            ("Load definition…", self._load_dialog),
            ("Add saved map…", lambda: self._add_dialog("native_coverage")),
            ("Add sample bundle…", lambda: self._add_dialog("sample_bundle")),
            ("Save definition…", self._save_dialog),
            ("Export figure and data…", self._export_dialog),
        ]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        controls = QHBoxLayout()
        self.sources = QComboBox()
        self.sources.currentIndexChanged.connect(self._source_changed)
        controls.addWidget(QLabel("Source"))
        controls.addWidget(self.sources)
        self.support = QComboBox()
        self.support.addItem("Per-group eligibility", "per_group")
        self.support.addItem("Fixed common columns", "common_columns")
        self.support.currentIndexChanged.connect(self._settings_changed)
        controls.addWidget(self.support)
        self.pool = QCheckBox("Pool selected groups equally")
        self.pool.toggled.connect(self._settings_changed)
        controls.addWidget(self.pool)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-250, 100)
        self.threshold.setValue(-75)
        self.threshold.setSuffix(" dBm")
        self.threshold.valueChanged.connect(self._draw)
        controls.addWidget(self.threshold)
        layout.addLayout(controls)
        self.source_label = QLineEdit()
        self.source_label.setPlaceholderText("Comparison source label")
        self.source_label.editingFinished.connect(self._settings_changed)
        layout.addWidget(self.source_label)
        self.groups = QListWidget()
        self.groups.setMaximumHeight(95)
        self.groups.itemChanged.connect(self._settings_changed)
        layout.addWidget(self.groups)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.curve = QComboBox()
        self.curve.currentIndexChanged.connect(self._draw)
        layout.addWidget(self.curve)
        self.canvas = InlinePlotGrid(
            [
                ("", "Received-power criterion (dBm)", "Eligible support passing (%)"),
                ("", "Transmitter subset", ""),
            ],
            columns=2,
        )
        layout.addWidget(self.canvas, 1)
        self.status = QLabel("Load a saved comparison, native map or portable sample bundle.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        if definition_path:
            self.load_definition(definition_path)
        elif native_path:
            self.add_source(native_path, "native_coverage")
        if threshold_dbm is not None:
            self.threshold.setValue(threshold_dbm)

    def load_definition(self, path):
        path = Path(path)
        definition = json.loads(path.read_text())
        # Resolve before saving elsewhere; source hashes still bind the original evidence.
        for source in definition.get("sources", []):
            for key in ("path", "masks_path"):
                if key in source:
                    source[key] = str((path.parent / source[key]).resolve())
        analyze_definition(definition)
        self.definition = definition
        self._populate_sources()
        self.refresh()
        display = definition.get("display", {})
        self.threshold.setValue(display.get("threshold_dbm", -75.0))
        self.curve.setCurrentIndex(min(display.get("row_index", 0), self.curve.count() - 1))

    def add_source(self, path, kind):
        path = Path(path).resolve()
        source = {
            "path": str(path),
            "sha256": file_identity(path)["sha256"],
            "kind": kind,
            "label": path.stem,
            "support": "per_group",
        }
        if kind == "native_coverage" and self.initial_roi_bounds is not None:
            source["roi_bounds_m"] = self.initial_roi_bounds
        candidate = copy.deepcopy(self.definition)
        candidate["sources"].append(source)
        analyze_definition(candidate)
        self.definition = candidate
        self._populate_sources()
        self.refresh()

    def _populate_sources(self):
        self._updating = True
        self.sources.clear()
        self.sources.addItems([s["label"] for s in self.definition["sources"]])
        self._updating = False
        self._source_changed()

    def _source_changed(self, *_):
        if self._updating or self.sources.currentIndex() < 0:
            return
        from shared.coverage.comparison import native_coverage_bundle

        self._updating = True
        try:
            source = self.definition["sources"][self.sources.currentIndex()]
            self.source_label.setText(source["label"])
            if source.get("kind", "sample_bundle") == "sample_bundle":
                data = load_sample_bundle(source["path"])
            else:
                data = native_coverage_bundle(source["path"], roi_bounds=source.get("roi_bounds_m"))
            names = data["group_names"].tolist()
            chosen = source.get("groups", names)
            self.groups.clear()
            for name in names:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if name in chosen else Qt.Unchecked)
                self.groups.addItem(item)
            self.support.setCurrentIndex(self.support.findData(source.get("support", "per_group")))
            common = self.support.model().item(self.support.findData("common_columns"))
            common.setEnabled("common_mask" in data)
            common.setToolTip(
                "Compare the same qualified XY columns across heights."
                if "common_mask" in data
                else "This source has no saved common-column eligibility mask."
            )
            self.pool.setChecked(source.get("pool_groups", False))
            self.pool.setText(
                "Pool selected groups (declared weights)"
                if "group_weights" in source
                else "Pool selected groups equally"
            )
            metadata = json.loads(str(data["metadata_json"]))
            self.description.setText(
                f"{metadata['quantity']} | Receiver: {metadata['receiver_model']}\nSupport: {metadata['support']} | Weights: {metadata['weighting']}\nSource: {source['path']}"
            )
        finally:
            self._updating = False

    def _settings_changed(self, *_):
        if self._updating or self.sources.currentIndex() < 0:
            return
        source = self.definition["sources"][self.sources.currentIndex()]
        source["label"] = self.source_label.text().strip() or "Unnamed study"
        self.sources.setItemText(self.sources.currentIndex(), source["label"])
        source["support"] = self.support.currentData()
        chosen = [
            self.groups.item(i).text()
            for i in range(self.groups.count())
            if self.groups.item(i).checkState() == Qt.Checked
        ]
        if "group_weights" in source:
            previous = source.get(
                "groups", [self.groups.item(i).text() for i in range(self.groups.count())]
            )
            by_name = dict(zip(previous, source["group_weights"], strict=True))
            if any(name not in by_name for name in chosen):
                self.result = None
                self.canvas.clear()
                self.status.setText(
                    "The added group requires an explicit weight in the saved definition."
                )
                return
            source["group_weights"] = [by_name[name] for name in chosen]
        source["groups"] = chosen
        source["pool_groups"] = self.pool.isChecked()
        self.refresh()

    def refresh(self):
        try:
            result = analyze_definition(self.definition)
        except (OSError, ValueError, KeyError) as exc:
            self.result = None
            self.canvas.clear()
            for plot in self.canvas.plots:
                plot.set_empty_message(f"Comparison unavailable: {exc}")
            self.status.setText(str(exc))
            return
        self.result = result
        old = self.curve.currentIndex()
        self.curve.blockSignals(True)
        self.curve.clear()
        self.curve.addItems([r["label"] for r in result["rows"]])
        self.curve.setCurrentIndex(max(0, min(old, len(result["rows"]) - 1)))
        self.curve.blockSignals(False)
        self._draw()
        counts = ", ".join(f"{r['label']}: {r['eligible_samples']}" for r in result["rows"])
        self.status.setText(
            "Eligible sample counts: "
            + counts
            + ". Unknown/invalid observations are bounded; empty observations fail finite thresholds. Plotting never runs propagation."
        )

    def _draw(self, *_):
        if self.result is not None:
            thresholds = np.asarray(self.definition["thresholds_dbm"])
            if not np.isclose(thresholds, self.threshold.value(), rtol=0, atol=1e-12).any():
                self.definition["thresholds_dbm"] = sorted(
                    set(self.definition["thresholds_dbm"] + [self.threshold.value()])
                )
                self.refresh()
                return
            draw_comparison(
                self.canvas,
                self.result,
                row_index=max(0, self.curve.currentIndex()),
                threshold_dbm=self.threshold.value(),
            )

    def save_definition(self, path):
        if self.result is None:
            raise ValueError("Fix the current comparison before saving")
        path = Path(path)
        definition = copy.deepcopy(self.definition)
        definition["display"] = {
            "threshold_dbm": self.threshold.value(),
            "row_index": max(0, self.curve.currentIndex()),
        }
        for source in definition["sources"]:
            for key in ("path", "masks_path"):
                if key in source:
                    source[key] = os.path.relpath(source[key], path.parent)
        path.write_text(json.dumps(definition, indent=2) + "\n")

    def export(self, path):
        if self.result is None:
            raise ValueError("Fix the current comparison before exporting")
        files = export_comparison(
            self.result,
            path,
            row_index=max(0, self.curve.currentIndex()),
            threshold_dbm=self.threshold.value(),
        )
        self.status.setText("Exported " + ", ".join(files))
        return files

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open comparison definition", "", "JSON (*.json)"
        )
        if path:
            self._action(lambda: self.load_definition(path))

    def _add_dialog(self, kind):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add saved source",
            "",
            "HDF5 (*.h5)" if kind == "native_coverage" else "Sample bundle (*.npz)",
        )
        if path:
            self._action(lambda: self.add_source(path, kind))

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save comparison definition", "", "JSON (*.json)"
        )
        if path:
            self._action(lambda: self.save_definition(path))

    def _export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure and numeric data", "", "PNG (*.png)"
        )
        if path:
            self._action(lambda: self.export(path))

    def _action(self, callback):
        try:
            callback()
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(str(exc))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", nargs="?")
    parser.add_argument("--native-map")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    window = CoverageComparisonWindow(definition_path=args.definition, native_path=args.native_map)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
