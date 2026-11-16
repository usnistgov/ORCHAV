"""Portable appearance settings for actor labels; never affect radio positions."""

import math
from dataclasses import asdict, dataclass

CALLOUT_MARKER_SIZE_BOUNDS_PX = (6, 60)


@dataclass(frozen=True)
class LabelCalloutStyle:
    """Screen-sized callouts, with the ordinary text style as the default."""

    mode: str = "plain"
    background: str = "paper"
    opacity: float = 0.85
    offset_x: float = 32.0
    offset_y: float = 42.0
    marker_size: float = 18.0
    leader: bool = True
    on_top: bool = True
    # Missing overrides inherit the shared size from older saved workspaces.
    tx_marker_size: float | None = None
    rx_marker_size: float | None = None

    def node_marker_size(self, kind: str) -> float:
        """Return the TX/RX callout marker size in pixels."""
        value = getattr(self, f"{kind}_marker_size")
        return self.marker_size if value is None else value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values) -> "LabelCalloutStyle":
        """Accept older sessions and reject malformed/nonfinite display values."""
        if isinstance(values, cls):
            return values
        defaults = cls().to_dict()
        if not isinstance(values, dict):
            return cls()
        for key, choices in (
            ("mode", ("plain", "callout", "card")),
            ("background", ("paper", "node", "white", "ink")),
        ):
            if values.get(key) in choices:
                defaults[key] = values[key]
        for key, (low, high) in {
            "opacity": (0.1, 1.0),
            "offset_x": (-500, 500),
            "offset_y": (-500, 500),
            "marker_size": CALLOUT_MARKER_SIZE_BOUNDS_PX,
            "tx_marker_size": CALLOUT_MARKER_SIZE_BOUNDS_PX,
            "rx_marker_size": CALLOUT_MARKER_SIZE_BOUNDS_PX,
        }.items():
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    defaults[key] = min(high, max(low, float(value)))
        for key in ("leader", "on_top"):
            if isinstance(values.get(key), bool):
                defaults[key] = values[key]
        return cls(**defaults)
