"""Shared matching choices; each view owns its own time interval."""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, Signal

from .dashboard_evaluation import DEFAULT_RANGE_GATE_M, DEFAULT_VELOCITY_GATE_M_S


class EvaluationSettings(QObject):
    """Publish committed evaluation choices without changing saved results."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.range_gate_m = DEFAULT_RANGE_GATE_M
        self.velocity_gate_m_s = DEFAULT_VELOCITY_GATE_M_S
        self.confirmed_only = False

    def set_values(self, *, range_gate_m, velocity_gate_m_s, confirmed_only):
        values = (float(range_gate_m), float(velocity_gate_m_s), bool(confirmed_only))
        if any(not math.isfinite(value) or value <= 0 for value in values[:2]):
            raise ValueError("Matching tolerances must be finite and positive")
        if values == (self.range_gate_m, self.velocity_gate_m_s, self.confirmed_only):
            return
        self.range_gate_m, self.velocity_gate_m_s, self.confirmed_only = values
        self.changed.emit()
