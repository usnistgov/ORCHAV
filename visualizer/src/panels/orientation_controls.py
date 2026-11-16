"""Single-actor global/local orientation inspection controls for Scene."""

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..renderers.protocol import renderer_capabilities
from ..services.orientation_reference import orientation_actor_choices, orientation_actor_pose
from ..types.orientation_reference import (
    ORIENTATION_REFERENCE_RADIUS_BOUNDS_M,
    OrientationReferenceSettings,
)
from .ui_theme import configure_label


class OrientationControls(QWidget):
    """Keep the inspected actor independent of path and camera selections."""

    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("Actor"))
        self.actor = QComboBox()
        self.actor.setPlaceholderText("Select an actor…")
        self.actor.setToolTip(
            "Inspect one TX, RX or target without changing Context or camera focus"
        )
        row.addWidget(self.actor, 1)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.global_frame = QCheckBox("Global")
        self.global_frame.setToolTip("World-aligned angular reference at the selected actor")
        self.local_frame = QCheckBox("Local")
        self.local_frame.setToolTip("Reference rotated by the actor's yaw, pitch and roll")
        row.addWidget(self.global_frame)
        row.addWidget(self.local_frame)
        row.addStretch()
        self.radius_label = QLabel("Radius")
        row.addWidget(self.radius_label)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(*ORIENTATION_REFERENCE_RADIUS_BOUNDS_M)
        self.radius.setDecimals(2)
        self.radius.setSingleStep(0.5)
        self.radius.setSuffix(" m")
        self.radius.setToolTip("Display size of orientation guides; independent of filter sectors")
        row.addWidget(self.radius)
        layout.addLayout(row)
        self.status = configure_label(QLabel(), role="muted", word_wrap=True)
        layout.addWidget(self.status)
        self.sync_from_state()
        self.actor.currentIndexChanged.connect(self._changed)
        self.global_frame.toggled.connect(self._changed)
        self.local_frame.toggled.connect(self._changed)
        self.radius.valueChanged.connect(self._changed)

    def sync_from_state(self) -> None:
        """Refresh choices and pose without changing an open actor popup or state."""
        viz = self.visualizer
        state = getattr(viz, "app_state", None)
        settings = OrientationReferenceSettings.from_dict(
            getattr(state, "orientation_reference", None)
        )
        choices = orientation_actor_choices(viz)
        changed = self.actor.count() != len(choices) or any(
            self.actor.itemText(i) != text or self.actor.itemData(i) != key
            for i, (text, key) in enumerate(choices)
        )
        with QSignalBlocker(self.actor):
            if changed:
                self.actor.clear()
                for text, key in choices:
                    self.actor.addItem(text, key)
            index = self.actor.findData(settings.actor)
            if self.actor.currentIndex() != index:
                self.actor.setCurrentIndex(index)
        for widget, value in (
            (self.global_frame, getattr(state, "show_global_angular_reference", False)),
            (self.local_frame, getattr(state, "show_local_angular_reference", False)),
        ):
            with QSignalBlocker(widget):
                widget.setChecked(value)
        with QSignalBlocker(self.radius):
            self.radius.setValue(settings.radius_m)
        supported = renderer_capabilities(getattr(viz, "renderer", None)).angular_preview
        position, orientation = orientation_actor_pose(viz, settings.actor)
        self.actor.setEnabled(supported and bool(choices))
        self.global_frame.setEnabled(supported and position is not None)
        self.local_frame.setEnabled(supported and position is not None and orientation is not None)
        self.radius.setEnabled(supported and position is not None)
        visible = self.global_frame.isChecked() or self.local_frame.isChecked()
        self.radius.setVisible(visible)
        self.radius_label.setVisible(visible)
        if not supported:
            text = "Orientation guides are available with pygfx."
        elif not choices:
            text = "No actors available."
        elif not settings.actor:
            text = "Select one actor to compare its local frame with the global frame."
        elif position is None:
            text = "The selected actor is unavailable in this frame."
        elif orientation is None:
            text = "Local orientation is unavailable in this frame."
        else:
            yaw, pitch, roll = np.degrees(orientation)
            text = f"Yaw {yaw:.1f}° · Pitch {pitch:.1f}° · Roll {roll:.1f}°"
        self.status.setText(text)

    def _changed(self, *_args) -> None:
        """Submit one display-only orientation choice through the controller."""
        settings = OrientationReferenceSettings.from_dict(
            {"actor": self.actor.currentData(), "radius_m": self.radius.value()}
        )
        self.visualizer.ui_controller.handle_orientation_reference_changed(
            settings, self.global_frame.isChecked(), self.local_frame.isChecked()
        )
        self.sync_from_state()
