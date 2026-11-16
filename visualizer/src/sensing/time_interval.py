"""Inclusive physical-time bounds without changing stored epochs.

Two outward binary64 ULPs cover the independently observed sample-clock
arithmetic error. This is deliberately unrelated to the decimal precision of
controls: decimal-sized differences remain real cutoffs, at every time scale.
"""

import math

from PySide6.QtWidgets import QDoubleSpinBox


def contains_time(time_s, start, end):
    """Include endpoint roundoff of at most two representable float steps."""
    if not math.isfinite(time_s):
        return False
    low = math.nextafter(math.nextafter(start, -math.inf), -math.inf)
    high = math.nextafter(math.nextafter(end, math.inf), math.inf)
    return low <= time_s <= high


class TimeBoundSpinBox(QDoubleSpinBox):
    """Keep programmatic bounds exact until the user actually edits them.

    Qt rounds setValue to display decimals. Keeping the original separately
    prevents an untouched 1/3-second endpoint becoming a shorter interval.
    textEdited also handles entering the displayed rounded value deliberately.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._exact_bound = None
        self.valueChanged.connect(self._edited)
        self.lineEdit().textEdited.connect(self._edited)

    def _edited(self, *_args):
        self._exact_bound = None

    def setValue(self, value):
        super().setValue(value)
        self._exact_bound = max(self.minimum(), min(self.maximum(), float(value)))

    def value(self):
        return super().value() if self._exact_bound is None else self._exact_bound
