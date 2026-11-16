"""Small optional covariance controls beside the existing track overlays."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QVBoxLayout, QWidget

from ..widgets.wrapping_layout import WrappingLayout


class TrackingUncertaintyControls(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = WrappingLayout()
        self._row = row
        self._details_only = False
        self.enabled = QCheckBox("Show tracking uncertainty")
        self.enabled.setToolTip(
            "Saved Cartesian tracker model covariance at each result. Not measured error or calibrated confidence."
        )
        self.scale = QComboBox()
        self.scale.addItem("1σ axes", 1.0)
        self.scale.addItem("2σ axes", 2.0)
        self.scope = QComboBox()
        self.scope.addItem("Selected track, or all if none", True)
        self.scope.addItem("All eligible tracks", False)
        for widget in (self.enabled, self.scale, self.scope):
            row.addWidget(widget)
        layout.addLayout(row)
        self.note = QLabel("Tracking uncertainty off")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.enabled.toggled.connect(self._changed)
        self.scale.currentIndexChanged.connect(self._changed)
        self.scope.currentIndexChanged.connect(self._changed)
        self._playing = False
        self._changed()

    def embed_toggle(self, layout):
        """Place the opt-in checkbox in an existing compact tracking-flags row."""
        self._row.removeWidget(self.enabled)
        layout.addWidget(self.enabled)
        self._details_only = True
        self._changed()

    def _changed(self):
        active = self.enabled.isChecked()
        self.scale.setEnabled(active)
        self.scope.setEnabled(active)
        self.scale.setVisible(active)
        self.scope.setVisible(active)
        self.note.setVisible(active)
        if self._details_only:
            self.setVisible(active)
        self.changed.emit()

    def set_playing(self, playing):
        """Playback changes the result, not the user's overlay preference."""
        self._playing = playing
