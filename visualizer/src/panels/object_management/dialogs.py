"""Live node-edit dialog used by the Object Management panel."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


class NodePropertiesDialog(QDialog):
    """Collect supported property updates for one live TX, RX, or target."""

    def __init__(self, parent: Any, entry: Dict[str, Any]) -> None:
        """Build editable fields based on the node entry capability flags."""
        super().__init__(parent)
        self.entry = entry
        self.setWindowTitle(f"Edit {entry.get('name', 'Node')}")
        self.setModal(True)
        self.resize(360, 320)

        self.position_spinboxes: Dict[str, QDoubleSpinBox] = {}
        self.orientation_spinboxes: Dict[str, QDoubleSpinBox] = {}
        self.scale_spinbox: Optional[QDoubleSpinBox] = None
        self._initial_position: tuple[float, ...] = ()
        self._initial_orientation: tuple[float, ...] = ()
        self._initial_scale: Optional[float] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        position = entry.get("position") or entry.get("current_center") or [0.0, 0.0, 0.0]
        orientation = (
            entry.get("orientation_degrees") or entry.get("orientation") or [0.0, 0.0, 0.0]
        )
        scale = entry.get("scale", 1.0)

        if entry.get("supports_position", True):
            pos_group = QGroupBox("Position (meters)")
            pos_layout = QGridLayout(pos_group)
            axes = ["X", "Y", "Z"]
            for row, axis in enumerate(axes):
                pos_layout.addWidget(QLabel(f"{axis}:"), row, 0)
                spin = QDoubleSpinBox()
                spin.setRange(-1000.0, 1000.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.1)
                spin.setValue(float(position[row]) if row < len(position) else 0.0)
                self.position_spinboxes[axis.lower()] = spin
                pos_layout.addWidget(spin, row, 1)
            layout.addWidget(pos_group)
            self._initial_position = self._position_values()

        if entry.get("supports_orientation", True):
            orient_group = QGroupBox("Orientation (degrees)")
            orient_layout = QGridLayout(orient_group)
            labels = ["Yaw", "Pitch", "Roll"]
            for row, axis in enumerate(labels):
                orient_layout.addWidget(QLabel(f"{axis}:"), row, 0)
                spin = QDoubleSpinBox()
                spin.setRange(-360.0, 360.0)
                spin.setDecimals(2)
                spin.setSingleStep(1.0)
                spin.setValue(float(orientation[row]) if row < len(orientation) else 0.0)
                self.orientation_spinboxes[axis.lower()] = spin
                orient_layout.addWidget(spin, row, 1)
            layout.addWidget(orient_group)
            self._initial_orientation = self._orientation_values()

        if entry.get("supports_scale", False):
            scale_group = QGroupBox("Scale")
            scale_layout = QHBoxLayout(scale_group)
            self.scale_spinbox = QDoubleSpinBox()
            self.scale_spinbox.setRange(0.1, 10.0)
            self.scale_spinbox.setDecimals(3)
            self.scale_spinbox.setSingleStep(0.1)
            self.scale_spinbox.setValue(float(scale))
            scale_layout.addWidget(self.scale_spinbox)
            layout.addWidget(scale_group)
            self._initial_scale = self.scale_spinbox.value()

        info_label = QLabel("Changes will trigger a gRPC update on the generator.")
        info_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        layout.addWidget(info_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_result(self) -> Dict[str, Any]:
        """Return only fields whose displayed values changed."""
        result: Dict[str, Any] = {}
        if self.position_spinboxes:
            position = self._position_values()
            if position != self._initial_position:
                result["position"] = list(position)
        if self.orientation_spinboxes:
            orientation = self._orientation_values()
            if orientation != self._initial_orientation:
                result["orientation"] = list(orientation)
        if self.scale_spinbox is not None and self.scale_spinbox.value() != self._initial_scale:
            result["scale"] = self.scale_spinbox.value()
        return result

    def _position_values(self) -> tuple[float, float, float]:
        """Return the displayed XYZ values in stable axis order."""
        return tuple(self.position_spinboxes[axis].value() for axis in ("x", "y", "z"))

    def _orientation_values(self) -> tuple[float, float, float]:
        """Return the displayed yaw, pitch, and roll values."""
        return tuple(self.orientation_spinboxes[axis].value() for axis in ("yaw", "pitch", "roll"))
