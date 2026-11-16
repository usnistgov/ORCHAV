"""Lazy, optional views of the same saved-decision figures used in summaries."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QDialog, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ..app.plot_theme import apply_matplotlib_legend_theme, apply_matplotlib_theme
from ..app.theme import get_theme_manager


class InspectionFigurePanel(QWidget):
    def __init__(self, title, parent=None, *, separate_window=False):
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.toggle = QPushButton(title)
        self.toggle.setCheckable(True)
        self.layout_.addWidget(self.toggle)
        self.body = QDialog(self) if separate_window else QWidget()
        if separate_window:
            self.body.setWindowTitle(title)
            self.body.resize(1150, 740)
            self.body.finished.connect(lambda _result: self.toggle.setChecked(False))
        self.body_layout = QVBoxLayout(self.body)
        self.note = QLabel("Saved evidence only; no detector or tracker rerun.")
        self.note.setWordWrap(True)
        if not separate_window:
            self.layout_.addWidget(self.body)
        self.body.hide()
        self.factory = None
        self.canvas = None
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(150)
        self.details.hide()
        self.body_layout.addWidget(self.details)
        self.body_layout.addWidget(self.note)
        self.toggle.toggled.connect(self._toggle)
        get_theme_manager().theme_changed.connect(self._apply_theme)

    def _apply_theme(self, *_):
        """Restyle the saved report preview without recomputing its evidence."""
        if self.canvas is not None:
            for axis in self.canvas.figure.axes:
                apply_matplotlib_theme(self.canvas.figure, axis, grid=False)
                apply_matplotlib_legend_theme(axis)
            self.canvas.draw_idle()

    def reset(self, *, preserve_options=False):
        """Release the figure and captured source when a recording retires."""
        self.factory = None
        self.source_key = None
        self.detail_text = ""
        self.details.clear()
        self._clear_canvas()
        if not preserve_options:
            # Retirement must not reenter the dashboard through external toggles.
            with QSignalBlocker(self.toggle):
                self.toggle.setChecked(False)
            self.body.hide()

    def _toggle(self, checked):
        self.body.setVisible(checked)
        if checked:
            self._render()
        else:
            self._clear_canvas()

    def _clear_canvas(self):
        if self.canvas is not None:
            self.body_layout.removeWidget(self.canvas)
            self.canvas.figure.clear()
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

    def set_source(self, factory, details="", *, key=None):
        changed = key is None or key != getattr(self, "source_key", None)
        self.factory = factory
        self.detail_text = details
        self.source_key = key
        if self.toggle.isChecked():
            self.details.setPlainText(details)
            self.details.setVisible(bool(details))
            if changed:
                self._render()

    def _render(self):
        self._clear_canvas()
        if self.factory is None:
            self.note.setText("Decision evidence not recorded for this result.")
            self.details.hide()
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        try:
            figure = self.factory()
        except (ValueError, KeyError, TypeError) as exc:
            self.note.setText(f"Saved decision evidence unavailable: {exc}")
            self.details.hide()
            return
        self.note.setText("Saved evidence only; no detector or tracker rerun.")
        for axis in figure.axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.set_loc("upper center")
                legend.set_bbox_to_anchor((0.5, -0.24), transform=axis.transAxes)
                legend.set_ncols(min(2, len(legend.get_texts())))
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setMinimumHeight(400)
        self.body_layout.insertWidget(0, self.canvas)
        self._apply_theme()
        self.details.setPlainText(getattr(self, "detail_text", ""))
        self.details.setVisible(bool(getattr(self, "detail_text", "")))


def tracking_decision_text(sensing):
    from shared.sensing.tracking_inspection import tracking_decisions

    try:
        record, metadata = tracking_decisions(sensing)
    except (ValueError, KeyError, TypeError) as exc:
        return f"Invalid saved decisions: {exc}"
    if record is None:
        return "Not recorded. Reprocess with --save-tracking-diagnostics into a new result."
    assignment = {a["detection_index"]: a["track_id"] for a in record["assignments"]}
    lines = ["Input D indices refer to this update before output confirmation filtering."]
    for c in record["candidates"]:
        decision = (
            "outside declared gates"
            if not c["admissible"]
            else (
                "reporting assignment"
                if assignment.get(c["detection_index"]) == c["track_id"]
                else "admissible, not assigned for reporting"
            )
        )
        cost = "unavailable" if c["cost"] is None else f"{c['cost']:.3g}"
        probability = "" if c["probability"] is None else f"; probability={c['probability']:.3g}"
        lines.append(
            f"T{c['track_id']} / D{c['detection_index']}: {decision}; cost={cost}{probability}"
        )
    if metadata.get("tracker_type") == "jpdaf":
        lines.append(
            "JPDA state updates use the recorded probabilities, not only the reporting assignment."
        )
        lines.append(f"Missed-update probabilities: {record['missed_probabilities']}")
    return "\n".join(lines)
