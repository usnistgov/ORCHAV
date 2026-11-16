"""Progressive lifecycle, association and uncertainty inspection of saved states."""

import json

from PySide6.QtCore import QSignalBlocker, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets.wrapping_layout import WrappingLayout
from .tracking_presentation import TrackingSnapshot, state_uncertainty, tracking_snapshot


def _value(value):
    if value is None:
        return "Not recorded"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_value(item) for item in value)
    return str(value)


class TrackingInspector(QWidget):
    """Inspect exact saved states; indexed decisions stay on demand."""

    track_selected = Signal(int)
    evidence_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.binding = None
        self.index = None
        self.selected_id = None
        self._detail_key = None
        self._sensing = None
        self._time_s = None
        self._active = False
        self._track_choices = None
        self.snapshot = TrackingSnapshot(())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = WrappingLayout()
        self.note = QLabel()
        self.note.setWordWrap(True)
        row.addWidget(self.note)
        self.toggle = QPushButton("Track details")
        self.toggle.setCheckable(True)
        self.toggle.toggled.connect(self._render)
        row.addWidget(self.toggle)
        self.report = QPushButton("UAV evaluation report")
        self.report.clicked.connect(self._open_report)
        row.addWidget(self.report)
        layout.addLayout(row)
        self.expanded = QWidget()
        details_layout = QVBoxLayout(self.expanded)
        choices = WrappingLayout()
        self.tracks = QComboBox()
        self.tracks.currentIndexChanged.connect(self._choose_track)
        choices.addWidget(self.tracks)
        self.evidence = QPushButton("Select contributing detection")
        self.evidence.clicked.connect(self._select_evidence)
        choices.addWidget(self.evidence)
        self.uncertainty_toggle = QCheckBox("Model uncertainty")
        self.uncertainty_toggle.toggled.connect(self._show_optional)
        choices.addWidget(self.uncertainty_toggle)
        self.raw_toggle = QCheckBox("Raw saved fields")
        self.raw_toggle.toggled.connect(self._show_optional)
        choices.addWidget(self.raw_toggle)
        details_layout.addLayout(choices)
        self.state_table = QTreeWidget()
        self.state_table.setHeaderLabels(["Saved track state", "Value"])
        self.state_table.setRootIsDecorated(False)
        self.state_table.setMaximumHeight(235)
        details_layout.addWidget(self.state_table)
        self.association_note = QLabel()
        self.association_note.setWordWrap(True)
        details_layout.addWidget(self.association_note)
        self.associations = QTableWidget(0, 6)
        self.associations.setHorizontalHeaderLabels(
            ["Track", "Input / detection", "Sensor", "Outcome", "Cost / innovation²", "Probability"]
        )
        self.associations.setEditTriggers(QTableWidget.NoEditTriggers)
        self.associations.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.associations.setMaximumHeight(200)
        details_layout.addWidget(self.associations)
        self.uncertainty = QTreeWidget()
        self.uncertainty.setHeaderLabels(["Saved model uncertainty", "Standard deviation (1σ)"])
        self.uncertainty.setRootIsDecorated(False)
        self.uncertainty.setMaximumHeight(190)
        self.uncertainty.setToolTip(
            "Marginal standard deviations from the saved covariance; not measured accuracy or a confidence guarantee."
        )
        details_layout.addWidget(self.uncertainty)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(190)
        details_layout.addWidget(self.details)
        layout.addWidget(self.expanded)
        self._show_optional()
        self.expanded.hide()
        self.hide()

    def update_result(self, binding, time_s, mode, error=None, *, sensing=None, snapshot=None):
        self._active = mode == "tracking"
        if snapshot is None:
            snapshot = tracking_snapshot(sensing, saved_tracking=binding, time_s=time_s)
        if snapshot.source_key != self.snapshot.source_key:
            self.selected_id = None
        self.snapshot = snapshot
        self.binding, self._sensing, self._time_s = binding, sensing, time_s
        self.index = None
        if binding is not None and len(binding.reader.times):
            index = binding.reader.epoch_index(time_s)
            if abs(binding.reader.times[index] - time_s) < 1e-9:
                self.index = index
        available = binding is not None or bool(snapshot.states) or snapshot.complete_states
        self.setVisible(bool(error) or (available and mode == "tracking"))
        if not self._active:
            # Keep state current for linked selection without rebuilding a hidden inspector.
            self.expanded.hide()
            if error:
                self.note.setText(error)
            return
        self.toggle.setVisible(available)
        self.report.setVisible(self._report_path() is not None and self._report_path().is_file())
        self.note.setText(error or snapshot.note)
        self._render()

    def _refresh_track_choices(self):
        choices = []
        for track in self.snapshot.states:
            confirmed = track.get("confirmed")
            confirmation = (
                "confirmed"
                if confirmed is True
                else "tentative" if confirmed is False else "confirmation unknown"
            )
            choices.append(
                (
                    track["track_id"],
                    f"Track {track['track_id']} · {confirmation} · {track['status']}",
                )
            )
        with QSignalBlocker(self.tracks):
            if choices != self._track_choices:
                self.tracks.clear()
                self.tracks.addItem("Select a track", None)
                for track_id, label in choices:
                    self.tracks.addItem(label, track_id)
                self._track_choices = choices
            self.tracks.setCurrentIndex(max(0, self.tracks.findData(self.selected_id)))

    def select_track(self, track_id):
        self.selected_id = track_id
        with QSignalBlocker(self.tracks):
            self.tracks.setCurrentIndex(max(0, self.tracks.findData(self.selected_id)))
        self._render()

    def selected_track(self):
        return next((t for t in self.snapshot.states if t["track_id"] == self.selected_id), None)

    def _native_source(self):
        track = self.selected_track()
        return track.get("evidence") if track is not None else None

    def _choose_track(self):
        self.selected_id = self.tracks.currentData()
        if self.selected_id is not None:
            self.track_selected.emit(self.selected_id)
        self._render()

    def _select_evidence(self):
        native = self._native_source()
        if native is not None:
            self.evidence_requested.emit(native)

    def _show_optional(self):
        self.uncertainty.setVisible(self.uncertainty_toggle.isChecked())
        self.details.setVisible(self.raw_toggle.isChecked())

    def _render(self):
        self.expanded.setVisible(self._active and self.toggle.isChecked())
        self.evidence.setEnabled(self._native_source() is not None)
        if not self.toggle.isChecked() or not self._active:
            return
        self._refresh_track_choices()
        key = (self.snapshot.source_key, self._time_s, id(self._sensing), self.selected_id)
        if key == self._detail_key:
            return
        self._detail_key = key
        self.state_table.clear()
        self.uncertainty.clear()
        self.associations.setRowCount(0)
        track = self.selected_track()
        if track is None:
            self.state_table.addTopLevelItem(
                QTreeWidgetItem(["Selection", "Choose a saved track for state details."])
            )
        else:
            rows = [
                ("Track ID", track["track_id"]),
                ("Confirmed", track.get("confirmed")),
                ("Observation", track["status"]),
                ("State coordinates", track.get("state_frame")),
                ("Hits", track.get("hits", track.get("hit_epochs"))),
                ("Misses", track.get("misses", track.get("missed_epochs"))),
            ]
            rows.extend(zip(track.get("state_labels", ()), track.get("mean", ()), strict=True))
            for label, value in rows:
                self.state_table.addTopLevelItem(QTreeWidgetItem([label, _value(value)]))
            uncertainty = state_uncertainty(track)
            for label, value in uncertainty:
                self.uncertainty.addTopLevelItem(QTreeWidgetItem([label, _value(value)]))
            if not uncertainty:
                self.uncertainty.addTopLevelItem(QTreeWidgetItem(["Covariance", "Not recorded"]))
        try:
            detailed = tracking_snapshot(
                self._sensing,
                saved_tracking=self.binding,
                time_s=self._time_s,
                include_decisions=True,
            )
            candidates = [
                row
                for row in detailed.associations
                if self.selected_id is None or row.get("track_id") == self.selected_id
            ]
            self.associations.setRowCount(len(candidates))
            for i, row in enumerate(candidates):
                values = (
                    row.get("track_id"),
                    row.get("detection_index", row.get("detection_id")),
                    row.get("sensor_id"),
                    row.get("outcome"),
                    row.get("cost", row.get("innovation_squared")),
                    row.get("probability"),
                )
                for j, value in enumerate(values):
                    self.associations.setItem(i, j, QTableWidgetItem(_value(value)))
            self.association_note.setText(
                "Saved association decisions · native input indices precede confirmation filtering."
                if candidates
                else "No saved association candidates for this selection."
            )
            self.details.setPlainText(
                json.dumps({"track": track, "associations": candidates}, indent=2)
            )
        except (OSError, ValueError, KeyError) as exc:
            self.association_note.setText(f"Saved decisions unavailable: {exc}")
            self.details.setPlainText(str(exc))
        self.associations.setVisible(self.associations.rowCount() > 0)

    def _report_path(self):
        if self.binding is None or self.binding.reader.path is None:
            return None
        return self.binding.reader.path.parent / "uav_evaluation/index.html"

    def _open_report(self):
        if self._report_path() is not None and self._report_path().is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report_path())))
