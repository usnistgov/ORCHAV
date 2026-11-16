"""Shared street-mesh width policy for tagged geometry and declared fallbacks.

Defaults are illustrative widths, not surveyed measurements. Keeping this
policy separate also lets vegetation placement exclude the same path corridor.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

# Total widths in meters. Unknown/motor-road classes retain the historic 8 m.
DEFAULT_WIDTHS_M = {
    "footway": 2.0,
    "steps": 1.8,
    "path": 1.5,
    "cycleway": 2.5,
    "bridleway": 2.5,
    "pedestrian": 4.0,
    "service": 4.0,
}
GENERIC_WIDTH_M = 8.0
_WIDTH = re.compile(r"\s*(\+?(?:\d+(?:\.\d*)?|\.\d+))\s*(m|metres?|meters?|ft|feet)?\s*", re.I)


def _tagged_width(value: object) -> float | None:
    if isinstance(value, (list, tuple)):
        widths = [_tagged_width(item) for item in value]
        # Mixed simplified edges use the widest valid tagged segment.
        return max((w for w in widths if w is not None), default=None)
    if isinstance(value, bool) or value is None:
        return None
    match = _WIDTH.fullmatch(str(value))
    if match is None:
        return None
    width = float(match[1]) * (0.3048 if (match[2] or "").lower() in {"ft", "feet"} else 1)
    return width if math.isfinite(width) and width > 0 else None


def street_width(data: Mapping) -> tuple[float, str]:
    """Resolve total width and its provenance, retaining valid explicit tags."""
    width = _tagged_width(data.get("width"))
    if width is not None:
        return width, "osm_width"
    highway = data.get("highway", "")
    classes = highway if isinstance(highway, (list, tuple)) else [highway]
    values = [DEFAULT_WIDTHS_M.get(str(value), GENERIC_WIDTH_M) for value in classes]
    return max(values, default=GENERIC_WIDTH_M), "highway_default"
