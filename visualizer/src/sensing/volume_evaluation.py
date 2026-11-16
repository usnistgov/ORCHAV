"""Progressive matching settings, independent of the volume's displayed metric."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets.form_layout import AlignedFormLayout


class VolumeEvaluationControls(QWidget):
    """Expose gates only for matching; expose color limits only for error maps."""

    changed = Signal()
    retry_requested = Signal()

    def __init__(self, parent=None, *, collapsible=True):
        super().__init__(parent)
        self.collapsible = collapsible
        self.ready = False
        self.quantity = "outcomes"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.expand = QCheckBox("Matching criteria")
        layout.addWidget(self.expand)
        self.details = QWidget()
        form = AlignedFormLayout(self.details)
        self.basis = QComboBox()
        self.basis.addItem("3D position", "position")
        self.basis.addItem("Range + radial velocity", "range_velocity")
        form.addRow("Match using", self.basis)
        self.position_gate = self._spin(0.75, " m")
        self.range_gate = self._spin(0.75, " m")
        self.velocity_gate = self._spin(0.5, " m/s")
        self.rows = []
        for name, widget in (
            ("3D position gate", self.position_gate),
            ("Range gate", self.range_gate),
            ("Radial velocity gate", self.velocity_gate),
        ):
            label = QLabel(name)
            form.addRow(label, widget)
            self.rows.append((label, widget))
            widget.valueChanged.connect(self._changed)
        self.basis.currentIndexChanged.connect(self._changed)
        layout.addWidget(self.details)
        self.error_settings = QWidget()
        error_form = AlignedFormLayout(self.error_settings)
        error_form.setContentsMargins(0, 0, 0, 0)
        self.error_max = self._spin(2, " m")
        error_form.addRow("Error colour scale maximum", self.error_max)
        self.error_max.valueChanged.connect(self._changed)
        layout.addWidget(self.error_settings)
        self.notice = QLabel("Evaluation only; detector unchanged.")
        self.notice.setToolTip(
            "Gates rematch the saved detector reports to ground truth. "
            "They do not change CFAR, clutter removal, ray tracing or the saved reports."
        )
        layout.addWidget(self.notice)
        self.progress = QLabel()
        self.progress.setWordWrap(True)
        layout.addWidget(self.progress)
        self.retry = QPushButton("Retry loading saved reports")
        self.retry.clicked.connect(self.retry_requested)
        self.retry.hide()
        layout.addWidget(self.retry)
        self.expand.toggled.connect(self._changed)
        self._changed()

    @staticmethod
    def _spin(value, suffix):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.001, 10000)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def configure(self, metadata, state=None):
        fusion = metadata.get("kind") == "conditional_fusion_outcomes"
        radial_index = self.basis.findData("range_velocity")
        self.basis.model().item(radial_index).setEnabled(not fusion)
        self.position_gate.setValue(metadata["position_gate_m"])
        self.range_gate.setValue(
            metadata.get("evaluation_range_gate_m", metadata["position_gate_m"])
        )
        self.velocity_gate.setValue(metadata.get("evaluation_velocity_gate_m_s", 0.5))
        self.error_max.setValue(max(1, metadata["position_gate_m"] * 2))
        if state:
            self.basis.setCurrentIndex(self.basis.findData(state["basis"]))
            for name in ("position_gate", "range_gate", "velocity_gate", "error_max"):
                getattr(self, name).setValue(state[name])
            self.expand.setChecked(state["expanded"])
        if fusion:
            self.basis.setCurrentIndex(self.basis.findData("position"))

    def state(self):
        return {
            "expanded": self.expand.isChecked(),
            "basis": self.basis.currentData(),
            **{
                name: getattr(self, name).value()
                for name in ("position_gate", "range_gate", "velocity_gate", "error_max")
            },
        }

    def set_ready(self):
        self.ready = True
        self.progress.clear()
        self.notice.setToolTip(
            self.notice.toolTip()
            + " Using a saved report snapshot; opening a native trial verifies its hashes."
        )
        self._changed()

    def set_quantity(self, quantity):
        """Changing display quantity never changes gates or the saved detector."""
        if quantity != self.quantity:
            self.quantity = quantity
            self._changed()

    def _changed(self, *_):
        outcomes = self.quantity in {"outcomes", "details", "extras", "pair"}
        self.expand.setVisible(outcomes and self.collapsible)
        self.details.setVisible((outcomes and self.expand.isChecked()) or not self.collapsible)
        self.error_settings.setVisible(self.quantity == "reports")
        self.error_max.setEnabled(self.ready)
        self.notice.setVisible(self.quantity != "geometry")
        self.basis.setEnabled(self.ready)
        basis = self.basis.currentData()
        visible = (basis == "position", basis == "range_velocity", basis == "range_velocity")
        for (label, widget), show in zip(self.rows, visible, strict=True):
            label.setVisible(show)
            widget.setVisible(show)
            widget.setEnabled(self.ready)
        self.changed.emit()
