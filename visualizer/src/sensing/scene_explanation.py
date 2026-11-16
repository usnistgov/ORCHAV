"""Optional controls for saved-report geometry and recorded receive arrays."""

from PySide6.QtCore import QSignalBlocker, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..widgets.form_layout import AlignedFormLayout
from ..widgets.wrapping_layout import WrappingLayout
from .selection import catalog_identity


class SceneExplanationControls(QGroupBox):
    """A bounded visual sequence; it never changes scene time or saved results."""

    changed = Signal()
    restore_camera_requested = Signal()
    takeover_requested = Signal()
    pause_changed = Signal(bool)
    REVEAL_HOLD_MS = 2000

    def __init__(self, range_shell, parent=None):
        super().__init__("3D sensing explanation", parent)
        self.range_shell = range_shell
        self.context = None
        self._playing_scene = False
        self._paused = False
        self.play_generation = 0
        self.available_phases = ("range", "angle", "position")
        self.phase = QComboBox()
        for label, key in (
            ("Off", "off"),
            ("1 · Range bin", "range"),
            ("2 · Range and coarse angle", "angle"),
            ("3 · Saved position", "position"),
        ):
            self.phase.addItem(label, key)
        self.play = QPushButton("Play reveal")
        self.play.setToolTip(
            "Reveal the saved evidence with guided camera framing and fading geometry. "
            "Each step lasts two seconds. Mouse input takes control; Restore camera "
            "returns to the original view. Pause scene playback before starting."
        )
        self.timer = QTimer(self)
        self.timer.setInterval(self.REVEAL_HOLD_MS)
        self.timer.timeout.connect(self._advance)
        self.play.clicked.connect(self._play)
        layout = QVBoxLayout(self)
        row = WrappingLayout()
        row.addWidget(QLabel("Report"))
        row.addWidget(self.phase)
        row.addWidget(self.play)
        layout.addLayout(row)
        cinematic_row = WrappingLayout()
        self.pause = QPushButton("Pause reveal")
        self.takeover = QPushButton("Take control")
        self.restore_camera = QPushButton("Restore camera")
        self.pause.setEnabled(False)
        self.takeover.setEnabled(False)
        self.restore_camera.setEnabled(False)
        for widget in (self.pause, self.takeover, self.restore_camera):
            cinematic_row.addWidget(widget)
        layout.addLayout(cinematic_row)
        self.pause.clicked.connect(self._pause)
        self.takeover.clicked.connect(self._takeover)
        self.restore_camera.clicked.connect(self.restore_camera_requested.emit)
        self.phase.currentIndexChanged.connect(self._phase_changed)
        self.note = QLabel("Select a saved report, then reveal range, coarse angle and position.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        # Keep height-for-width content outside the input form: Qt can
        # underestimate a form mixing wrapped toolbars and spanning notes.
        form = AlignedFormLayout()
        layout.addLayout(form)
        self.response = QComboBox()
        for label, key in (
            ("Off", "off"),
            ("Broadside receive beam", "broadside"),
            ("Maximum across coarse beams", "scan"),
            ("Selected report's coarse beam", "beam"),
        ):
            self.response.addItem(label, key)
        self.response.currentIndexChanged.connect(self._response_changed)
        form.addRow("RX response", self.response)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(1, 1000)
        self.radius.setValue(20)
        self.radius.setSuffix(" m")
        self.radius.setToolTip("Arbitrary drawing radius, not sensing coverage or detection range.")
        self.radius.valueChanged.connect(self.changed)
        self.radius_label = QLabel("Display radius")
        form.addRow(self.radius_label, self.radius)
        self.response_note = QLabel()
        self.response_note.setWordWrap(True)
        layout.addWidget(self.response_note)
        self._response_changed()

    def _phase_changed(self):
        # Publish one complete transition. The shell's own signal otherwise
        # reenters validation before this phase's notification has completed.
        with QSignalBlocker(self.range_shell):
            self.range_shell.setChecked(False)
        self.changed.emit()

    def _response_changed(self):
        enabled = self.response.currentData() != "off"
        self.radius.setVisible(enabled)
        self.radius_label.setVisible(enabled)
        self.response_note.setVisible(enabled)
        self.changed.emit()

    def _play(self):
        if self._playing_scene:
            self.note.setText("Pause playback before starting a cinematic reveal")
            return
        if self.timer.isActive() or self._paused:
            self.stop()
            return
        self._paused = False
        self.play_generation += 1
        self.pause.setEnabled(True)
        self.takeover.setEnabled(True)
        self.play.setText("Stop reveal")
        self.timer.start()
        if self.phase.currentIndex() == 1:
            self.changed.emit()
        else:
            self.phase.setCurrentIndex(1)

    def _advance(self):
        index = self.phase.currentIndex() + 1
        while (
            index < self.phase.count() and self.phase.itemData(index) not in self.available_phases
        ):
            index += 1
        if index >= self.phase.count():
            self.timer.stop()
            self.play.setText("Play reveal")
            self.pause.setEnabled(False)
        else:
            self.phase.setCurrentIndex(index)

    def stop(self, *, notify=True):
        """Remove presentation geometry without changing the dashboard selection."""
        active = self.timer.isActive() or self.phase.currentIndex() != 0
        self.timer.stop()
        self._paused = False
        self.pause.setText("Pause reveal")
        self.pause.setEnabled(False)
        self.takeover.setEnabled(False)
        self.play.setText("Play reveal")
        with QSignalBlocker(self.phase), QSignalBlocker(self.range_shell):
            if self.phase.currentData() == "range":
                self.range_shell.setChecked(False)
            self.phase.setCurrentIndex(0)
        if active and notify:
            self.changed.emit()

    def _pause(self):
        self._paused = not self._paused
        if self._paused:
            self.timer.stop()
        else:
            self.timer.start()
        self.pause.setText("Resume reveal" if self._paused else "Pause reveal")
        self.pause_changed.emit(self._paused)

    def _takeover(self):
        self.takeover_requested.emit()
        self.stop()

    def set_playing(self, playing):
        was_playing = self._playing_scene
        self._playing_scene = bool(playing)
        if playing:
            self.stop()
            self.note.setText("Pause playback to reveal the selected saved report.")
        elif was_playing:
            self.note.setText(
                "Select a saved report, then reveal range, coarse angle and position."
            )
        self.play.setEnabled(not playing)
        self.phase.setEnabled(not playing)

    def bind_context(self, request, selection):
        """Discard a reveal on an acquisition, source, scene or selection change."""
        key = None
        if request is not None:
            sensing = request.display_frame.get("sensing") or {}
            key = (
                request.source_identity,
                request.effective_display_step,
                sensing.get("temporal", {}).get("acquisition", {}).get("acquisition_id"),
                sensing.get("temporal", {}).get("result", {}).get("processing_recipe_id"),
                request.variant,
                catalog_identity(sensing),
                selection,
            )
        if self.context is not None and key != self.context:
            self.stop(notify=False)
        self.context = key

    def hideEvent(self, event):
        self.stop()
        self.changed.emit()
        super().hideEvent(event)
