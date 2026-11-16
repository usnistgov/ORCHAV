"""Small native link-comparison dialog backed by shared CPU analysis."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from shared.statistics.link_comparison import (
    COMPARISON_SCHEMA,
    analyze_comparison,
    export_comparison,
    load_definition,
    load_link_recording,
)


class LinkComparisonDialog(QDialog):
    """Select sources/pairs, inspect matching states and export a saved recipe.

    Input recordings stay independent of playback. An explicit Analyze action
    updates plots; export always uses the visible selections. Unknown timestamps
    require an explicit frame/time binding file rather than inferred duration.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Communication link comparison")
        self.resize(1160, 830)
        self.records = []
        self.sources = []
        self.result = None
        self.canvas = None
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        for text, callback in (
            ("Load comparison…", self._load),
            ("Add native recording…", self._add_native),
            ("Add compact observation…", self._add_compact),
            ("Remove source", self._remove),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.source = QComboBox()
        self.source.setMinimumContentsLength(30)
        layout.addWidget(self.source)
        self.source_note = QLabel(
            "Load compatible observations; TX/RX selection uses recorded names."
        )
        self.source_note.setWordWrap(True)
        layout.addWidget(self.source_note)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Receiver"))
        self.receiver = QComboBox()
        controls.addWidget(self.receiver)
        self.transmitters = QListWidget()
        self.transmitters.setSelectionMode(QAbstractItemView.MultiSelection)
        self.transmitters.setMaximumHeight(85)
        controls.addWidget(QLabel("Sites (select any)"))
        controls.addWidget(self.transmitters)
        self.axis = QComboBox()
        self.axis.addItems(["time", "distance"])
        controls.addWidget(self.axis)
        self.paired = QCheckBox("Paired differences (matching states)")
        controls.addWidget(self.paired)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self._analyze)
        controls.addWidget(self.analyze_button)
        layout.addLayout(controls)
        self.plot_layout = QVBoxLayout()
        layout.addLayout(self.plot_layout, 1)
        self.status = QLabel(
            "Zero paths, missing pairs and invalid measurements are kept distinct."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        bottom = QHBoxLayout()
        self.frame = QSpinBox()
        self.frame.setRange(0, 1000000000)
        bottom.addWidget(QLabel("Recorded frame"))
        bottom.addWidget(self.frame)
        inspect = QPushButton("Inspect in active scenario")
        inspect.clicked.connect(self._inspect)
        bottom.addWidget(inspect)
        bottom.addStretch()
        self.export_button = QPushButton("Export plots, values and definition…")
        self.export_button.clicked.connect(self._export)
        bottom.addWidget(self.export_button)
        layout.addLayout(bottom)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.receiver.currentTextChanged.connect(self._mark_stale)
        self.transmitters.itemSelectionChanged.connect(self._mark_stale)
        self.axis.currentTextChanged.connect(self._mark_stale)
        self.paired.toggled.connect(self._mark_stale)

    def _mark_stale(self, *_):
        """Make it clear when controls no longer describe the displayed plot."""
        if self.result is not None:
            self.status.setText(
                "Selections changed. Analyze or Export to update the displayed comparison."
            )

    def set_definition(self, path):
        """Load a versioned definition; shared parser authenticates its sources."""
        definition, records = load_definition(path)
        self.records = records
        self.sources = [dict(s) for s in definition["sources"]]
        self._refresh()
        self.receiver.setCurrentText(definition["receiver"])
        for i in range(self.transmitters.count()):
            item = self.transmitters.item(i)
            item.setSelected(item.text() in definition["transmitters"])
        self.axis.setCurrentText(definition.get("axis", "time"))
        self.paired.setChecked(definition.get("paired", False))

    def definition(self):
        """Return all current controls, never an obsolete previously drawn recipe."""
        return dict(
            schema=COMPARISON_SCHEMA,
            sources=self.sources,
            receiver=self.receiver.currentText(),
            transmitters=[i.text() for i in self.transmitters.selectedItems()],
            axis=self.axis.currentText(),
            paired=self.paired.isChecked(),
        )

    def _refresh(self):
        self.source.clear()
        self.source.addItems([s["label"] for s in self.sources])
        self.receiver.clear()
        self.transmitters.clear()
        if self.records:
            rx = [n for n in self.records[0].rx_names if all(n in r.rx_names for r in self.records)]
            tx = [n for n in self.records[0].tx_names if all(n in r.tx_names for r in self.records)]
            self.receiver.addItems(rx)
            self.transmitters.addItems(tx)
            self.transmitters.selectAll()
        self._source_changed()

    def _source_changed(self, *_):
        i = self.source.currentIndex()
        if 0 <= i < len(self.records):
            record = self.records[i]
            self.source_note.setText(
                f"{self.sources[i]['label']} | {record.metadata.get('source', self.sources[i]['path'])}\n{record.metadata['time_reference']} | {record.metadata['quantity']}"
            )
            self.frame.setValue(int(record.arrays["frame_ids"][0]))

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load link comparison", "", "JSON (*.json)")
        if path:
            try:
                self.set_definition(path)
            except (OSError, ValueError, KeyError) as error:
                self.status.setText(str(error))

    def _add_native(self):
        path = QFileDialog.getExistingDirectory(self, "Native scenario recording directory")
        if not path:
            return
        binding = None
        try:
            record = load_link_recording(path)
        except ValueError as error:
            if "timestamps unavailable" not in str(error):
                self.status.setText(str(error))
                return
            file, _ = QFileDialog.getOpenFileName(
                self, "Explicit frame-ID to authored time JSON is required", "", "JSON (*.json)"
            )
            if not file:
                self.status.setText(str(error))
                return
            try:
                binding = json.loads(Path(file).read_text())
                record = load_link_recording(path, time_binding=binding)
            except (ValueError, OSError, KeyError) as second:
                self.status.setText(str(second))
                return
        except OSError as error:
            self.status.setText(str(error))
            return
        self._append(record, path, binding)

    def _add_compact(self):
        path, _ = QFileDialog.getOpenFileName(self, "Compact link observation", "", "NPZ (*.npz)")
        if path:
            try:
                self._append(load_link_recording(path), path)
            except (OSError, ValueError, KeyError) as error:
                self.status.setText(str(error))

    def _append(self, record, path, binding=None):
        label = Path(path).stem
        base, suffix = label, 2
        while label in [s["label"] for s in self.sources]:
            label = f"{base} ({suffix})"
            suffix += 1
        self.records.append(record)
        self.sources.append(dict(label=label, path=str(Path(path).resolve()), time_binding=binding))
        self._refresh()

    def _remove(self):
        index = self.source.currentIndex()
        if index >= 0:
            self.sources.pop(index)
            self.records.pop(index)
            self._refresh()

    def analyze(self):
        """Execute the same CPU routine as the command-line export."""
        from ..widgets.inline_plot import SERIES_COLORS, InlinePlotGrid

        self.result = analyze_comparison(self.definition(), self.records)
        if self.canvas is not None:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        result = self.result
        xlabel = "Time (s)" if result["axis"] == "time" else "Sampled traveled distance (m)"
        specs = [(f"Receiver: {result['receiver']}", xlabel, "Incoherent received power (dBm)")]
        if result["differences"]:
            specs.append(("Paired differences", xlabel, "Power difference (dB)"))
        self.canvas = InlinePlotGrid(specs)
        styles = (Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine)
        for source, trace in enumerate(result["traces"]):
            for tx, name in enumerate(result["transmitters"]):
                self.canvas.plots[0].curve(
                    trace["x"],
                    trace["values"][:, tx],
                    name=f"{trace['label']} · {name}",
                    color=SERIES_COLORS[tx % len(SERIES_COLORS)],
                    style=styles[source % 4],
                )
        for source, diff in enumerate(result["differences"]):
            for tx, name in enumerate(result["transmitters"]):
                self.canvas.plots[1].curve(
                    result["traces"][0]["x"],
                    diff["delta_db"][:, tx],
                    name=f"{diff['label']} · {name}",
                    color=SERIES_COLORS[tx % len(SERIES_COLORS)],
                    style=styles[source % 4],
                )
        if len(self.canvas.plots) > 1:
            self.canvas.plots[1].setXLink(self.canvas.plots[0])
        self.plot_layout.addWidget(self.canvas)
        self.status.setText(
            "Displayed current selections. Gaps omit zero/unknown dBm; exports retain every pair's status and frame identity."
        )
        return self.result

    def _analyze(self):
        try:
            self.analyze()
        except (ValueError, KeyError, IndexError, ImportError) as error:
            self.status.setText(str(error))

    def export_to(self, path):
        """Export visible selections and redraw so exported/displayed recipes agree."""
        self.analyze()
        result = export_comparison(self.definition(), self.records, path)
        self.status.setText(
            f"Exported PNG/PDF, CSV/NPZ, source identities and portable definition: {path}"
        )
        return result

    def _export(self):
        path = QFileDialog.getExistingDirectory(self, "Comparison export directory")
        if path:
            try:
                self.export_to(path)
            except (OSError, ValueError, KeyError, ImportError) as error:
                self.status.setText(str(error))

    def _inspect(self):
        index = self.source.currentIndex()
        if index < 0:
            return
        record = self.records[index]
        fid = self.frame.value()
        if fid not in record.arrays["frame_ids"]:
            self.status.setText("Selected frame is not present in this observation.")
            return
        parent = self.parent()
        scenario = getattr(parent, "current_scenario_path", None)
        source = record.metadata.get("source")
        tx = [i.text() for i in self.transmitters.selectedItems()]
        if (
            not source
            or not scenario
            or (
                Path(scenario).resolve().parent
                if Path(scenario).suffix.lower() in (".yaml", ".yml")
                else Path(scenario).resolve()
            )
            != Path(source).resolve()
        ):
            self.status.setText(
                f"Open {source or 'the native source from the export manifest'}/scenario.yaml in the visualizer, then inspect frame {fid}, RX {self.receiver.currentText()}, TX {', '.join(tx)}."
            )
            return
        if len(tx) != 1:
            self.status.setText("Select one transmitter for native path inspection.")
            return
        for role, name in (("tx", tx[0]), ("rx", self.receiver.currentText())):
            dropdown = getattr(parent, f"{role}_dropdown", None)
            if dropdown is not None:
                dropdown.setCurrentIndex(
                    dropdown.findData(getattr(record, f"{role}_names").index(name))
                )
        parent.update_frame(fid)
        self.status.setText(
            f"Active Paths selection: frame {fid}, {tx[0]} → {self.receiver.currentText()}."
        )
