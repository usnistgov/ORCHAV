"""Portable selection and scale for one actor's orientation guides."""

import math
from dataclasses import asdict, dataclass

ORIENTATION_REFERENCE_RADIUS_BOUNDS_M = (0.01, 1_000_000.0)


@dataclass(frozen=True)
class OrientationReferenceSettings:
    """Select a TX/RX index or stable target ID, independently of path filters."""

    actor: str = ""
    radius_m: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values) -> "OrientationReferenceSettings":
        """Keep malformed or missing saved settings from selecting an actor."""
        if isinstance(values, cls):
            return values
        if not isinstance(values, dict):
            return cls()
        actor = values.get("actor", "")
        if not isinstance(actor, str):
            actor = ""
        kind, _, key = actor.partition(":")
        if not key or not (kind == "target" or (kind in ("tx", "rx") and key.isdecimal())):
            actor = ""
        radius = values.get("radius_m")
        if (
            not isinstance(radius, (int, float))
            or isinstance(radius, bool)
            or not math.isfinite(radius)
        ):
            radius = cls().radius_m
        low, high = ORIENTATION_REFERENCE_RADIUS_BOUNDS_M
        return cls(actor=actor, radius_m=min(high, max(low, float(radius))))
