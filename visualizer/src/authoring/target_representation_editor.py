"""Native target RF controls backed by the shared scenario schema."""

from __future__ import annotations

from typing import Any, get_args

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from shared.scenarios.actors import (
    ConstantRCSSpec,
    NativeBoxGeometrySpec,
    NativeMeshGeometrySpec,
    TR38901RCSSpec,
)

from ..widgets.form_layout import AlignedFormLayout
from .domain import AuthoringActor


class TargetRepresentationEditor(QWidget):
    """Edit one RF model and its absorbing proxy without changing motion."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self.form = AlignedFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.representation = QComboBox(self)
        for label, value in (
            ("Mesh + radio material", "mesh_material"),
            ("Native constant RCS", "constant_rcs"),
            ("Native 3GPP TR 38.901", "tr38901"),
        ):
            self.representation.addItem(label, value)
        self.form.addRow("RF representation", self.representation)
        self.geometry_type = QComboBox(self)
        self.geometry_type.addItem("Absorbing box", "box")
        self.geometry_type.addItem("Absorbing static mesh", "mesh")
        self.form.addRow("Occlusion geometry", self.geometry_type)
        self.dimensions = tuple(self._spin(0.0, 1e6) for _ in range(3))
        for label, spin in zip(("Length x [m]", "Width y [m]", "Height z [m]"), self.dimensions):
            spin.setSpecialValueText("Native default")
            self.form.addRow(label, spin)
        self.mesh_path = QLineEdit(self)
        self.mesh_path.setPlaceholderText("Single .ply or .obj path")
        self.form.addRow("Mesh path", self.mesh_path)
        self.mesh_scale = self._spin(1e-9, 1e6, 1.0)
        self.form.addRow("Proxy scale", self.mesh_scale)
        self.sigma = self._spin(0.0, 1e12, 1.0)
        self.form.addRow("RCS [m²]", self.sigma)
        self.preserve_polarization = QCheckBox("Preserve polarization", self)
        self.preserve_polarization.setChecked(True)
        self.form.addRow(self.preserve_polarization)
        self.xpr = self._spin(-300.0, 300.0)
        self.form.addRow("XPR [dB]", self.xpr)
        self.object_type = QComboBox(self)
        for value in get_args(TR38901RCSSpec.model_fields["object_type"].annotation):
            self.object_type.addItem(value, value)
        self.object_type.setCurrentIndex(self.object_type.findData("vehicle-multi-sp"))
        self.form.addRow("Native preset", self.object_type)
        self.model_type = QComboBox(self)
        self.model_type.addItem("Model 1", 1)
        self.model_type.addItem("Model 2", 2)
        self.model_type.setCurrentIndex(1)
        self.form.addRow("3GPP RCS model", self.model_type)
        self.random_flags: dict[str, QCheckBox] = {}
        for key, label in (
            ("random_sigma_s", "Random RCS component"),
            ("random_xpr", "Random XPR"),
            ("random_phases", "Random phases"),
        ):
            check = QCheckBox(label, self)
            self.random_flags[key] = check
            self.form.addRow(check)
        self.notice = QLabel(
            "Native geometry blocks radio paths; its surface does not scatter. "
            "Position is the target center. Doppler models rigid translation only. "
            "Random flags do not provide time-coherent fluctuations.",
            self,
        )
        self.notice.setWordWrap(True)
        self.form.addRow(self.notice)
        self.object_type.currentIndexChanged.connect(self._preset_changed)
        for control in (self.representation, self.geometry_type, self.model_type):
            control.currentIndexChanged.connect(self._changed)
        for control in (*self.dimensions, self.mesh_scale, self.sigma, self.xpr):
            control.valueChanged.connect(self._changed)
        self.mesh_path.textChanged.connect(self._changed)
        for control in (self.preserve_polarization, *self.random_flags.values()):
            control.toggled.connect(self._changed)
        self._update_visibility()

    def _spin(self, low: float, high: float, value: float = 0.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setDecimals(12)
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    def _show(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        label = self.form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    def _update_visibility(self) -> None:
        representation = self.representation.currentData()
        native = representation != "mesh_material"
        box = self.geometry_type.currentData() == "box"
        self._show(self.geometry_type, native)
        control: QWidget
        for control in self.dimensions:
            self._show(control, native and box)
        for control in (self.mesh_path, self.mesh_scale):
            self._show(control, native and not box)
        for control in (self.sigma, self.preserve_polarization, self.xpr):
            self._show(control, representation == "constant_rcs")
        self.xpr.setEnabled(not self.preserve_polarization.isChecked())
        native_controls: tuple[QWidget, ...] = (
            self.object_type,
            self.model_type,
            *self.random_flags.values(),
        )
        for control in native_controls:
            self._show(control, representation == "tr38901")
        self.model_type.setEnabled(self.object_type.currentData() == "human")
        self.notice.setVisible(native)

    def _preset_changed(self, *_args: Any) -> None:
        if not self._syncing:
            required_model = 1 if self.object_type.currentData() == "uav-small-size" else 2
            self.model_type.setCurrentIndex(self.model_type.findData(required_model))
        self._changed()

    def _changed(self, *_args: Any) -> None:
        self._update_visibility()
        if not self._syncing:
            self.changed.emit()

    def set_actor(self, actor: AuthoringActor) -> None:
        """Load controls without creating a pending draft."""
        self._syncing = True
        try:
            self.representation.setCurrentIndex(self.representation.findData(actor.representation))
            geometry = actor.geometry
            self.geometry_type.setCurrentIndex(
                self.geometry_type.findData(geometry.type if geometry is not None else "box")
            )
            for name, spin in zip(("length_m", "width_m", "height_m"), self.dimensions):
                spin.setValue(float(getattr(geometry, name, None) or 0.0))
            self.mesh_path.setText(str(getattr(geometry, "path", "")))
            self.mesh_scale.setValue(float(getattr(geometry, "scale", 1.0)))
            self.sigma.setValue(actor.constant_rcs.sigma_m2 if actor.constant_rcs else 1.0)
            xpr = actor.constant_rcs.xpr_db if actor.constant_rcs else None
            self.preserve_polarization.setChecked(xpr is None)
            self.xpr.setValue(float(xpr or 0.0))
            preset = actor.tr38901
            self.object_type.setCurrentIndex(
                self.object_type.findData(preset.object_type if preset else "vehicle-multi-sp")
            )
            self.model_type.setCurrentIndex(
                self.model_type.findData(preset.model_type if preset else 2)
            )
            for name, control in self.random_flags.items():
                control.setChecked(bool(getattr(preset, name, False)))
        finally:
            self._syncing = False
            self._update_visibility()

    def actor_changes(self) -> dict[str, Any]:
        """Return validated native fields and clear the alternate RF models."""
        representation = str(self.representation.currentData())
        changes: dict[str, Any] = {
            "representation": representation,
            "geometry": None,
            "constant_rcs": None,
            "tr38901": None,
        }
        if representation == "mesh_material":
            return changes
        changes["target"] = None
        if self.geometry_type.currentData() == "box":
            changes["geometry"] = NativeBoxGeometrySpec(
                **{
                    key: spin.value() or None
                    for key, spin in zip(("length_m", "width_m", "height_m"), self.dimensions)
                }
            )
        else:
            changes["geometry"] = NativeMeshGeometrySpec(
                path=self.mesh_path.text().strip(), scale=self.mesh_scale.value()
            )
        if representation == "constant_rcs":
            changes["constant_rcs"] = ConstantRCSSpec(
                sigma_m2=self.sigma.value(),
                xpr_db=None if self.preserve_polarization.isChecked() else self.xpr.value(),
            )
        else:
            changes["tr38901"] = TR38901RCSSpec(
                object_type=self.object_type.currentData(),
                model_type=self.model_type.currentData(),
                **{name: control.isChecked() for name, control in self.random_flags.items()},
            )
        return changes
