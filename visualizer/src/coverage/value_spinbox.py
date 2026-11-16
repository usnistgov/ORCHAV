"""Readable RF-value entry that retains the metric's original numerical units."""

import math

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox


class CoverageValueSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that can expose very small linear values accurately."""

    def __init__(self) -> None:
        super().__init__()
        self._scientific = False
        self._decibel_offset = None
        self.setKeyboardTracking(False)

    def set_metric_display(self, metric: str) -> None:
        """Edit linear power/ratios in dBm/dB; value() and signals stay native.

        Threshold masks, bookmarks, and source arrays continue to use the
        original units. A half-decibel step is relative to the current value,
        so tiny powers remain adjustable independently of the volume's range.
        """
        from .analysis import coverage_metric_label

        base = str(metric).split("/", 1)[0]
        offsets = {
            "path_gain_linear": 0,
            "serving_path_gain_linear": 0,
            "sinr_linear": 0,
            "rss_w": 30,
        }
        self._decibel_offset = offsets.get(base)
        # Leave fractional precision below the supported 1e-30 lower bound,
        # otherwise a half-dB increase at that boundary would round back to it.
        self.setDecimals(60 if self._decibel_offset is not None else 2)
        self.set_scientific_mode(False)
        unit = (
            ("dBm" if base == "rss_w" else "dB")
            if base in offsets
            else coverage_metric_label(metric)[1]
        )
        self.setSuffix(f" {unit}" if unit and unit != "index" else "")
        self.setSingleStep(0.5)

    def set_scientific_mode(self, enabled: bool) -> None:
        """Switch between ordinary fixed-point and compact scientific text."""
        self._scientific = bool(enabled)
        if enabled:
            self._decibel_offset = None
        self.lineEdit().setText(self.textFromValue(self.value()) + self.suffix())

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        """Format linear path-gain thresholds without rounding them to zero."""
        if self._decibel_offset is not None:
            if value <= 0:
                return "−∞"
            return self.locale().toString(10 * math.log10(value) + self._decibel_offset, "f", 1)
        if self._scientific:
            return f"{float(value):.6g}"
        return super().textFromValue(value)

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt override
        """Convert displayed decibels or scientific text to the native value."""
        if not self._scientific and self._decibel_offset is None:
            return super().valueFromText(text)
        try:
            value = self._parse_number(text)
        except ValueError:
            return self.value()
        if self._decibel_offset is not None:
            return self._from_decibels(value)
        return value

    def _parse_number(self, text):
        value_text = str(text).strip()
        suffix = self.suffix().strip()
        if suffix and value_text.endswith(suffix):
            value_text = value_text[: -len(suffix)].strip()
        value, ok = self.locale().toDouble(value_text)
        value = value if ok else float(value_text)
        if not math.isfinite(value):
            raise ValueError("Coverage values must be finite")
        return value

    def _from_decibels(self, value):
        exponent = (value - self._decibel_offset) / 10
        # Clamp before exponentiation, including temporarily extreme typed text.
        maximum = max(self.maximum(), 1e-300)
        minimum = max(self.minimum(), 10.0 ** -self.decimals())
        return 10 ** min(max(exponent, math.log10(minimum)), math.log10(maximum))

    def stepBy(self, steps):  # noqa: N802 - Qt override
        if self._decibel_offset is None:
            return super().stepBy(steps)
        value = max(self.value(), self.minimum(), 10.0 ** -self.decimals())
        decibels = 10 * math.log10(value) + self._decibel_offset
        self.setValue(self._from_decibels(decibels + steps * self.singleStep()))

    def validate(self, text: str, pos: int):  # noqa: N802 - Qt override
        """Accept signed, localized numbers and intermediate exponent edits."""
        if not self._scientific and self._decibel_offset is None:
            return super().validate(text, pos)
        value_text = str(text).strip()
        suffix = self.suffix().strip()
        if suffix and value_text.endswith(suffix):
            value_text = value_text[: -len(suffix)].strip()
        if value_text in {"", "+", "-", ".", "+.", "-."}:
            return QValidator.Intermediate, text, pos
        try:
            self._parse_number(value_text)
        except ValueError:
            lowered = value_text.lower()
            if lowered.endswith(("e", "e+", "e-")):
                return QValidator.Intermediate, text, pos
            return QValidator.Invalid, text, pos
        return QValidator.Acceptable, text, pos
