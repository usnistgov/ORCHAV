"""Lazy optional receive-pattern controls for the existing volume scene."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QGroupBox, QLabel

from ..widgets.form_layout import AlignedFormLayout
from .volume_antenna import load_receive_pattern, response_object


class ReceivePatternControls(QGroupBox):
    """Own a bounded two-surface display cache; never run a sensing detector."""

    changed = Signal()

    def __init__(self):
        super().__init__("Receive-array response")
        self.pattern = None
        self.pending = None
        self.failed = False
        self.cache = {}
        form = AlignedFormLayout(self)
        self.mode = QComboBox()
        self.mode.addItem("Off", "off")
        self.mode.addItem("Broadside receive beam", "broadside")
        self.mode.addItem("Maximum across recorded coarse beams", "scan")
        self.mode.currentIndexChanged.connect(self.changed)
        form.addRow("Response", self.mode)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(1, 1000)
        self.radius.setValue(35)
        self.radius.setSuffix(" m")
        self.radius.setToolTip(
            "Arbitrary display radius in scene coordinates; not coverage or detection range."
        )
        self.radius.valueChanged.connect(self.changed)
        form.addRow("Display radius", self.radius)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.05, 1)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.4)
        self.opacity.valueChanged.connect(self.changed)
        form.addRow("Response opacity", self.opacity)
        self.note = QLabel(
            "Optional RX-only scalar response. Exact array metadata is read from one verified native trial when enabled."
        )
        self.note.setWordWrap(True)
        form.addRow(self.note)

    def state(self):
        return dict(
            mode=self.mode.currentData(), radius=self.radius.value(), opacity=self.opacity.value()
        )

    def restore(self, state):
        with QSignalBlocker(self):
            self.mode.setCurrentIndex(max(0, self.mode.findData(state.get("mode", "off"))))
            self.radius.setValue(state.get("radius", 35))
            self.opacity.setValue(state.get("opacity", 0.4))

    def poll(self):
        """Consume a worker result only while the owning volume window is open."""
        if self.pending is None or not self.pending.done():
            return
        future, self.pending = self.pending, None
        try:
            self.pattern = future.result()
            if not self.pattern.scan_enabled:
                self.mode.model().item(self.mode.findData("scan")).setEnabled(False)
                if self.mode.currentData() == "scan":
                    self.mode.setCurrentIndex(self.mode.findData("broadside"))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.failed = True
            self.note.setText(f"RX response unavailable: {exc}. Saved outcomes remain available.")
        self.changed.emit()

    def objects(self, reader, executor):
        """Return only this optional surface; the main overlay owns its lifetime."""
        enabled = self.mode.currentData() != "off"
        self.radius.setEnabled(enabled and not self.failed)
        self.opacity.setEnabled(enabled and not self.failed)
        if not enabled or self.failed:
            return []
        if self.pattern is None:
            if self.pending is None:
                self.note.setText(
                    "Verifying one completed native trial and loading its exact RX pose/array…"
                )
                self.pending = executor.submit(load_receive_pattern, reader)
            return []
        mode, radius, opacity = self.mode.currentData(), self.radius.value(), self.opacity.value()
        if mode == "scan" and not self.pattern.scan_enabled:
            return []
        key = (mode, radius, opacity)
        if key not in self.cache:
            try:
                obj = response_object(self.pattern, mode, radius, opacity)
            except ValueError as exc:
                self.failed = True
                self.note.setText(f"RX response unavailable: {exc}.")
                return []
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = obj
        count = len(self.pattern.beam_directions)
        label = "Broadside beam" if mode == "broadside" else f"Maximum of {count} coarse beams"
        self.mode.setItemText(
            self.mode.findData("scan"), f"Maximum across {count} recorded coarse beams"
        )
        self.note.setText(
            f"{self.pattern.rows} × {self.pattern.columns} isotropic RX elements; {label.lower()}. "
            f"{self.pattern.carrier_frequency_hz / 1e9:g} GHz; spacings "
            f"{self.pattern.horizontal_spacing:g}λ horizontal, {self.pattern.vertical_spacing:g}λ vertical.\n"
            "Purple surface: relative scalar array power, −40 to 0 dB; radius maps this clipped dB scale linearly. "
            "Display radius is arbitrary, not coverage, detectable range or probability.\n"
            "Physical front/back symmetry is retained. Native angles are reported in local +X; that is an algorithm convention, not a hard field of view. "
            "No TX pattern, polarimetric propagation or angle-refinement gain is implied."
        )
        return [self.cache[key]]

    def legend_lines(self):
        if self.pattern is None or self.failed or self.mode.currentData() == "off":
            return []
        mode = (
            "broadside"
            if self.mode.currentData() == "broadside"
            else f"max {len(self.pattern.beam_directions)} beams"
        )
        return [
            f"RX {mode}: relative power −40…0 dB",
            f"Arbitrary radius {self.radius.value():g} m; not detection coverage",
            "Front/back symmetry; angles reported in local +X",
        ]
