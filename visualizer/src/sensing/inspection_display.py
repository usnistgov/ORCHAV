"""Quantity and scale contracts shared by numerical signal inspection views."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIGNAL_COMPANIONS = {
    "cfar": (
        "CFAR / NMS / clustering",
        "Saved detector decisions",
    ),
    "clutter": (
        "Before/after clutter suppression",
        "Before / after clutter suppression",
    ),
    "inputs": (
        "Complex input samples (CIR)",
        "Saved CIR magnitude and phase",
    ),
    "frequency": (
        "Frequency response",
        "Saved CFR magnitude and phase",
    ),
}


def range_label(mode: str) -> str:
    """Name the saved coordinate without changing its values."""
    if mode == "monostatic":
        return "Range"
    if mode == "bistatic":
        return "Total path length"
    return "Recorded distance coordinate"


def power_db(power: np.ndarray) -> np.ndarray:
    """Absolute logarithm in the recorded linear units; zero remains minus infinity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(power)


def regular_axis(axis: np.ndarray) -> bool:
    """Image rectangles support only increasing, uniform coordinates."""
    delta = np.diff(axis)
    return bool(
        len(axis)
        and np.all(np.isfinite(axis))
        and (
            not len(delta)
            or (np.all(delta > 0) and np.allclose(delta, delta[0], rtol=1e-7, atol=1e-10))
        )
    )


@dataclass
class PowerScale:
    """Common comparison reference with a lock scoped to compatible quantities.

    Summaries are cached by caller-supplied scientific identity. Retaining the
    arrays also prevents recycled object addresses from validating stale data.
    The two-entry cache bounds references held during companion switches.
    """

    locked: bool = False
    _scope: tuple | None = None
    _top: float | None = None
    _cache: dict = field(default_factory=dict)

    def reset(self) -> None:
        self._scope, self._top = None, None
        self._cache.clear()

    def levels(self, powers, *, identity: tuple, scope: tuple, span: float) -> tuple[float, float]:
        if scope != self._scope:
            self._scope, self._top = scope, None
        cached = self._cache.get(identity)
        if cached is None:
            arrays = tuple(p for p in powers if p is not None)
            maxima = [float(np.max(p, where=np.isfinite(p), initial=0)) for p in arrays]
            maximum = max(maxima, default=0)
            top = float(10 * np.log10(maximum)) if maximum > 0 else 0.0
            cached = (top, arrays)
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[identity] = cached
        if self._top is None or not self.locked:
            self._top = cached[0]
        return self._top - span, self._top
