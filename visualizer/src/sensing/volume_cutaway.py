"""Display-only cuts through sampled volumes; measurements remain untouched."""

import numpy as np
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..utils.volume_cut import BoxCut, cut_volume  # noqa: F401 - retained public import
from ..widgets.form_layout import AlignedFormLayout


class VolumeCutawayControls(QWidget):
    """Plane or box removal at sample-cell boundaries in world coordinates."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.axes = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.active = QCheckBox("Cutaway")
        layout.addWidget(self.active)
        self.details = QWidget()
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        self.mode.addItem("Plane cut", "plane")
        self.mode.addItem("Hide cells inside XYZ box", "box")
        details.addWidget(self.mode)
        self.plane_details = QWidget()
        form = AlignedFormLayout(self.plane_details)
        self.axis = QComboBox()
        for name in ("z", "x", "y"):
            self.axis.addItem(name.upper(), name)
        self.side = QComboBox()
        self.side.addItem("Keep ≤ position", True)
        self.side.addItem("Keep ≥ position", False)
        self.boundary = QComboBox()
        form.addRow("Axis", self.axis)
        form.addRow("Keep", self.side)
        form.addRow("Position", self.boundary)
        details.addWidget(self.plane_details)
        self.box_details = QWidget()
        box_form = AlignedFormLayout(self.box_details)
        self.box_bounds = {}
        for axis in "xyz":
            lower, upper = QComboBox(), QComboBox()
            self.box_bounds[axis] = lower, upper
            row = QWidget()
            pair = QHBoxLayout(row)
            pair.setContentsMargins(0, 0, 0, 0)
            for widget in (QLabel("Min"), lower, QLabel("Max"), upper):
                pair.addWidget(widget)
            lower.setMinimumContentsLength(6)
            upper.setMinimumContentsLength(6)
            lower.setAccessibleName(f"{axis.upper()} minimum")
            upper.setAccessibleName(f"{axis.upper()} maximum")
            box_form.addRow(axis.upper(), row)
            lower.currentIndexChanged.connect(lambda _, a=axis: self._box_changed(a, True))
            upper.currentIndexChanged.connect(lambda _, a=axis: self._box_changed(a, False))
        self.box_details.setToolTip(
            "Remove cells inside all three coordinate intervals. Values are cell "
            "boundaries in metres; choosing a corner reveals the volume interior. "
            "Scene geometry, saved data, full-volume statistics and linked slices "
            "remain unchanged."
        )
        details.addWidget(self.box_details)
        self.box_details.hide()
        self.plane_details.setToolTip(
            "Hides sample cells on one side of the selected centre. "
            "The cut surface lies halfway to the next centre. Scene geometry and "
            "full-volume TP/FP/FN are unchanged."
        )
        layout.addWidget(self.details)
        self.details.hide()
        self.active.toggled.connect(self.details.setVisible)
        self.active.toggled.connect(self.changed)
        self.axis.currentIndexChanged.connect(self._axis_changed)
        self.side.currentIndexChanged.connect(self.changed)
        self.boundary.currentIndexChanged.connect(self.changed)
        self.mode.currentIndexChanged.connect(self._mode_changed)

    def configure(self, axes, state=None):
        self.axes = axes
        state = state or {}
        self.mode.setCurrentIndex(self.mode.findData(state.get("mode", "plane")))
        for axis, (lower, upper) in self.box_bounds.items():
            centres = np.asarray(axes[axis])
            edges = np.r_[
                centres[0] - (centres[1] - centres[0]) / 2,
                (centres[:-1] + centres[1:]) / 2,
                centres[-1] + (centres[-1] - centres[-2]) / 2,
            ]
            count = len(centres)
            # A near upper corner is a useful initial cut; all six bounds remain editable.
            default = (0, max(1, count // 2)) if axis != "z" else (count // 2, count)
            saved = state.get("box_bounds", {}).get(axis)
            with QSignalBlocker(lower), QSignalBlocker(upper):
                for control, index in zip((lower, upper), default, strict=True):
                    control.clear()
                    for value in edges:
                        control.addItem(f"{value:g} m", float(value))
                    control.setCurrentIndex(index)
                if saved and len(saved) == 2:
                    indices = [int(np.argmin(np.abs(edges - value))) for value in saved]
                    if indices[0] < indices[1]:
                        lower.setCurrentIndex(indices[0])
                        upper.setCurrentIndex(indices[1])
        if state:
            self.axis.setCurrentIndex(self.axis.findData(state.get("axis", "z")))
        self._axis_changed()
        if state:
            self.side.setCurrentIndex(self.side.findData(state.get("keep_lower", True)))
            index = self.boundary.findData(state.get("coordinate"))
            if index >= 0:
                self.boundary.setCurrentIndex(index)
            self.active.setChecked(state.get("active", False))

    def _mode_changed(self, *_):
        box = self.mode.currentData() == "box"
        self.plane_details.setVisible(not box)
        self.box_details.setVisible(box)
        self.changed.emit()

    def _box_changed(self, axis, lower_changed):
        lower, upper = self.box_bounds[axis]
        if lower.count() == 0:
            return
        with QSignalBlocker(lower), QSignalBlocker(upper):
            if lower.currentIndex() >= upper.currentIndex():
                if lower_changed:
                    lower.setCurrentIndex(min(lower.currentIndex(), lower.count() - 2))
                    upper.setCurrentIndex(lower.currentIndex() + 1)
                else:
                    upper.setCurrentIndex(max(upper.currentIndex(), 1))
                    lower.setCurrentIndex(upper.currentIndex() - 1)
        self.changed.emit()

    def _axis_changed(self, *_):
        if self.axes is None:
            return
        with QSignalBlocker(self.boundary):
            self.boundary.clear()
            values = self.axes[self.axis.currentData()]
            for value in values:
                self.boundary.addItem(f"{value:g} m", float(value))
            self.boundary.setCurrentIndex(len(values) // 2)
        self.changed.emit()

    def state(self):
        return dict(
            active=self.active.isChecked(),
            axis=self.axis.currentData(),
            keep_lower=self.side.currentData(),
            coordinate=self.boundary.currentData(),
            mode=self.mode.currentData(),
            box_bounds={
                axis: [lower.currentData(), upper.currentData()]
                for axis, (lower, upper) in self.box_bounds.items()
            },
        )

    def cut(self, display):
        if display == "slice" or not self.active.isChecked() or self.axes is None:
            return None
        if self.mode.currentData() == "box":
            return BoxCut(
                tuple(
                    (lower.currentIndex(), upper.currentIndex())
                    for lower, upper in self.box_bounds.values()
                )
            )
        return self.axis.currentData(), self.boundary.currentIndex(), self.side.currentData()

    def description(self, display):
        cut = self.cut(display)
        if cut is None:
            return ""
        if isinstance(cut, BoxCut):
            bounds = "; ".join(
                f"{axis.upper()} {lower.currentData():g} to {upper.currentData():g} m"
                for axis, (lower, upper) in self.box_bounds.items()
            )
            return "Cutaway: remove box (cell boundaries), " + bounds
        axis, _, keep_lower = cut
        return f"Cutaway: {axis.upper()} {'≤' if keep_lower else '≥'} {self.boundary.currentData():g} m (centres)"
