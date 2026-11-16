"""Compact evaluation presentation, independent of plot visibility and playback."""

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .dashboard_evaluation import (
    DEFAULT_RANGE_GATE_M,
    DEFAULT_VELOCITY_GATE_M_S,
    evaluation_cache_current,
    metrics_text,
    summarize_events,
)
from .evaluation_plots import EvaluationPlots
from .evaluation_settings import EvaluationSettings
from .time_interval import TimeBoundSpinBox, contains_time


class EvaluationPanel(QWidget):
    """Score the selected event immediately; expand to inspect a saved interval."""

    history_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.evidence = None
        self.history = None
        self._current_summary = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.current = QLabel("Evaluation unavailable: no saved result")
        self.current.setWordWrap(True)
        layout.addWidget(self.current)
        self.toggle = QPushButton("Evaluation details")
        self.toggle.setCheckable(True)
        layout.addWidget(self.toggle)
        self.details = QWidget()
        grid = QGridLayout(self.details)
        grid.setContentsMargins(0, 0, 0, 0)
        self.range_gate = self._spin(DEFAULT_RANGE_GATE_M, 0.001, 10000, " m")
        self.velocity_gate = self._spin(DEFAULT_VELOCITY_GATE_M_S, 0.001, 10000, " m/s")
        grid.addWidget(QLabel("Range tolerance (±)"), 0, 0)
        grid.addWidget(self.range_gate, 0, 1)
        grid.addWidget(QLabel("Radial velocity tolerance (±)"), 1, 0)
        grid.addWidget(self.velocity_gate, 1, 1)
        self.outputs = QComboBox()
        self.outputs.addItem("All saved outputs", False)
        self.outputs.addItem("Confirmed tracks only", True)
        self.outputs.setToolTip("Choose the saved output cohort to score.")
        grid.addWidget(self.outputs, 2, 0, 1, 2)
        self.interval = QCheckBox("Selected time interval")
        grid.addWidget(self.interval, 3, 0, 1, 2)
        self.start = self._spin(0, -1e12, 1e12, " s")
        self.end = self._spin(0, -1e12, 1e12, " s")
        self.start.setDecimals(9)
        self.end.setDecimals(9)
        grid.addWidget(QLabel("From"), 4, 0)
        grid.addWidget(self.start, 4, 1)
        grid.addWidget(QLabel("To"), 5, 0)
        grid.addWidget(self.end, 5, 1)
        self.aggregate = QLabel("Expand to load recording evaluation")
        self.aggregate.setWordWrap(True)
        grid.addWidget(self.aggregate, 6, 0, 1, 2)
        self.plots = EvaluationPlots()
        grid.addWidget(self.plots, 7, 0, 1, 2)
        grid.setRowStretch(7, 1)
        self.tracks = QLabel()
        self.tracks.setWordWrap(True)
        grid.addWidget(self.tracks, 8, 0, 1, 2)
        layout.addWidget(self.details, 1)
        self.details.hide()
        self.toggle.toggled.connect(self._expand)
        for spin in (self.start, self.end):
            spin.valueChanged.connect(self._refresh)
        for spin in (self.range_gate, self.velocity_gate):
            spin.valueChanged.connect(self._commit_evaluation_settings)
        self.outputs.currentIndexChanged.connect(self._commit_evaluation_settings)
        self.interval.toggled.connect(self._refresh)
        self.evaluation_settings = None
        self.bind_evaluation_settings(EvaluationSettings(self))

    def bind_evaluation_settings(self, settings):
        """Share gates and cohort while retaining this view's local interval."""
        if self.evaluation_settings is settings:
            return
        if self.evaluation_settings is not None:
            self.evaluation_settings.changed.disconnect(self._sync_evaluation_settings)
        self.evaluation_settings = settings
        settings.changed.connect(self._sync_evaluation_settings)
        self._sync_evaluation_settings()

    def _commit_evaluation_settings(self, *_args):
        self.evaluation_settings.set_values(
            range_gate_m=self.range_gate.value(),
            velocity_gate_m_s=self.velocity_gate.value(),
            confirmed_only=bool(self.outputs.currentData()),
        )

    def _sync_evaluation_settings(self):
        settings = self.evaluation_settings
        with (
            QSignalBlocker(self.range_gate),
            QSignalBlocker(self.velocity_gate),
            QSignalBlocker(self.outputs),
        ):
            self.range_gate.setValue(settings.range_gate_m)
            self.velocity_gate.setValue(settings.velocity_gate_m_s)
            self.outputs.setCurrentIndex(self.outputs.findData(settings.confirmed_only))
        self._refresh()

    @staticmethod
    def _spin(value, minimum, maximum, suffix):
        spin = TimeBoundSpinBox() if suffix == " s" else QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _expand(self, checked):
        self.details.setVisible(checked)
        if checked:
            self.history_requested.emit()
        self._refresh()

    def set_workspace_active(self, active: bool) -> None:
        """Use the full dashboard view without a second expansion step."""
        self.toggle.hide()
        with QSignalBlocker(self.toggle):
            self.toggle.setChecked(active)
        self.details.setVisible(active)
        if active:
            self._refresh()

    def update_event(self, event, history=None):
        """Receive saved evidence; repeated playback cannot accumulate counts."""
        context = tuple(event["identity"][:3]) if event else None
        previous = tuple(self.evidence["identity"][:3]) if self.evidence else None
        changed = history is not self.history or context != previous
        self.evidence, self.history = event, history
        has_tracks = bool(
            event
            and (
                event["tracking_enabled"]
                or any(d.get("track_id") is not None for d in event["detections"])
            )
        )
        self.outputs.model().item(1).setEnabled(has_tracks)
        self.outputs.model().item(1).setToolTip(
            "" if has_tracks else "No tracking outputs were recorded for this result."
        )
        if changed and history is not None and history.records:
            with QSignalBlocker(self.start), QSignalBlocker(self.end):
                self.start.setValue(history.records[0].time_s)
                self.end.setValue(history.records[-1].time_s)
        self._refresh(aggregate=changed)

    def _default_scoring(self):
        return (
            self.range_gate.value() == DEFAULT_RANGE_GATE_M
            and self.velocity_gate.value() == DEFAULT_VELOCITY_GATE_M_S
            and not self.outputs.currentData()
        )

    def current_summary(self):
        """Return the current result's score using the committed shared choices."""
        return self._current_summary

    def _score(self, events):
        if (
            self._default_scoring()
            and len(events) == 1
            and events[0] is not None
            and "default_summary" in events[0]
            and evaluation_cache_current(events[0])
        ):
            return events[0]["default_summary"]
        return summarize_events(
            events,
            self.range_gate.value(),
            self.velocity_gate.value(),
            confirmed_only=bool(self.outputs.currentData()),
        )

    def _refresh(self, *_args, aggregate=True):
        self.start.setEnabled(self.interval.isChecked())
        self.end.setEnabled(self.interval.isChecked())
        summary = self._score([self.evidence])
        self._current_summary = summary
        gate_text = (
            f"tolerance ±{self.range_gate.value():g} m / ±{self.velocity_gate.value():g} m/s"
        )
        cohort = "Confirmed tracks" if self.outputs.currentData() else "Saved outputs"
        self.current.setText(f"Current result · {cohort} · {metrics_text(summary)} · {gate_text}")
        if self.evidence is not None:
            tracks = [d for d in self.evidence["detections"] if d.get("track_id") is not None]
            confirmed = sum(d.get("track_confirmed") is True for d in tracks)
            tentative = sum(d.get("track_confirmed") is False for d in tracks)
            self.current.setToolTip(
                f"Saved output rows: {len(self.evidence['detections'])}; confirmed {confirmed}, "
                f"tentative {tentative}. Coasting population is not recorded in this output."
            )
            self.tracks.setText(
                f"Saved track outputs: {confirmed} confirmed · {tentative} tentative · "
                "coasting population not recorded"
                if tracks or self.evidence["tracking_enabled"]
                else "Tracking: no saved track IDs"
            )
        else:
            self.tracks.clear()
        if self.toggle.isChecked():
            self.plots.update_references(self.evidence)
            self.plots.update_matches(
                self.evidence,
                summary,
                self.range_gate.value(),
                self.velocity_gate.value(),
                bool(self.outputs.currentData()),
            )
        if not aggregate or not self.toggle.isChecked():
            return
        if self.history is None:
            self.aggregate.setText("Recording evaluation is loading…")
            self.plots.update_counts({"reason": "Loading saved results…"}, "Recording")
            return
        if self.history.error:
            self.aggregate.setText(f"Recording evaluation unavailable: {self.history.error}")
            self.plots.update_counts({"reason": self.history.error}, "Recording")
            return
        records = self.history.records
        if self.interval.isChecked():
            records = tuple(
                r for r in records if contains_time(r.time_s, self.start.value(), self.end.value())
            )
        context = tuple(self.evidence["identity"][:3]) if self.evidence else None
        events = [
            r.evaluation
            for r in records
            if r.evaluation is not None and tuple(r.evaluation["identity"][:3]) == context
        ]
        if (
            self._default_scoring()
            and not self.interval.isChecked()
            and all(
                r.evaluation is None or tuple(r.evaluation["identity"][:3]) == context
                for r in records
            )
        ):
            summary = self.history.evaluation_summary
        else:
            summary = self._score(events)
        prefix = "Selected interval" if self.interval.isChecked() else "Recording"
        if self.history.truncated:
            prefix += f" (bounded to last {len(self.history.records)} loaded results)"
        self.plots.update_counts(summary, prefix)
        text = f"{prefix} · {metrics_text(summary)}"
        if not summary.get("reason"):

            def percent(value):
                return "—" if value is None else f"{100 * value:.1f}%"

            ids = summary["id_switches"]
            text += (
                f"\nPrecision {percent(summary['precision'])} · recall/coverage {percent(summary['cumulative_pd'])}"
                f" · unmatched outputs/result {summary['mean_false_alarms_per_evaluated_event']:.3f}"
                f" · ID switches {ids if ids is not None else 'unavailable'}"
                f"\n{summary['evaluated_event_count']} evaluated results · {summary['target_opportunity_count']} target opportunities"
                f" · {summary['matched_pair_count']} matched pairs"
            )
        skipped = summary.get("skipped") or {}
        if skipped:
            text += "\nExcluded results: " + "; ".join(f"{why} ({n})" for why, n in skipped.items())
        self.aggregate.setText(text)
