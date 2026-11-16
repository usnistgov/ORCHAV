"""Controls for the optional downloaded geographic map layer."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout, QWidget


class MapLayerControls(QWidget):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("2D OpenStreetMap")
        self.enabled.setToolTip("Show the saved map, or download the scene's map on demand.")
        self.enabled.toggled.connect(service.set_enabled)
        layout.addWidget(self.enabled)
        form = QFormLayout()
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 100)
        self.opacity.setSuffix(" %")
        self.opacity.setDecimals(0)
        self.opacity.setSingleStep(5)
        self.opacity.setKeyboardTracking(True)
        self.opacity.valueChanged.connect(lambda value: service.set_opacity(value / 100))
        self.height = QDoubleSpinBox()
        self.height.setRange(-10000, 10000)
        self.height.setSuffix(" m")
        self.height.setDecimals(2)
        self.height.setSingleStep(1.0)
        self.height.setKeyboardTracking(True)
        self.height.setToolTip(
            "Flat map height in local scene Z. Raise it above ground and roads to avoid overlap; terrain is not draped."
        )
        self.height.valueChanged.connect(service.set_height)
        form.addRow("Opacity", self.opacity)
        form.addRow("Height Z", self.height)
        layout.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        service.changed.connect(self.sync)
        self.sync()

    def sync(self):
        service = self.service
        self.setVisible(service.available)
        with (
            QSignalBlocker(self.enabled),
            QSignalBlocker(self.opacity),
            QSignalBlocker(self.height),
        ):
            self.enabled.setChecked(service.enabled)
            # Avoid reformatting the text/cursor in response to its own edit.
            opacity = round(service.opacity * 100, self.opacity.decimals())
            height = round(service.height, self.height.decimals())
            if self.opacity.value() != opacity:
                self.opacity.setValue(opacity)
            if self.height.value() != height:
                self.height.setValue(height)
        self.opacity.setEnabled(service.enabled)
        self.height.setEnabled(service.enabled)
        self.status.setText(service.status)
        self.status.setVisible(bool(service.status))
