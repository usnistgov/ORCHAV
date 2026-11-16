"""Geographic selection math for the temporary Scenario Builder ground map.

Only the human-requested 3 x 3 tile view is fetched. Coordinates are local,
latitude-scaled Web Mercator for selection; the generator still owns UTM geometry.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

RADIUS = 6378137.0


def parse_location(text: str) -> tuple[float, float, int | None]:
    """Accept latitude, longitude or an OSM #map=zoom/lat/lon permalink."""
    match = re.search(r"#map=(\d+)/([-+\d.]+)/([-+\d.]+)", text)
    if match:
        zoom, lat, lon = match.groups()
        result = float(lat), float(lon), int(zoom)
    else:
        parts = re.split(r"[,;\s]+", text.strip())
        if len(parts) != 2:
            raise ValueError("Enter latitude, longitude or an OpenStreetMap #map link.")
        result = float(parts[0]), float(parts[1]), None
    lat, lon, zoom = result
    if not all(math.isfinite(v) for v in (lat, lon)) or not (
        -80 <= lat <= 80 and -179.9 <= lon <= 179.9
    ):
        raise ValueError("Choose latitude -80 to 80 and longitude -179.9 to 179.9.")
    return result


def validate_bbox(values) -> tuple[float, float, float, float]:
    """Validate one modest non-wrapping Builder selection, in south/west/north/east order."""
    if len(values) != 4:
        raise ValueError("A region needs south, west, north and east coordinates.")
    south, west, north, east = map(float, values)
    if not all(math.isfinite(v) for v in (south, west, north, east)):
        raise ValueError("Region coordinates must be finite.")
    if not (-80 <= south < north <= 80 and -180 <= west < east <= 180):
        raise ValueError("Select a region within the map, without crossing the date line.")
    width = math.radians(east - west) * RADIUS * math.cos(math.radians((south + north) / 2))
    height = math.radians(north - south) * RADIUS
    if not (20 <= width <= 5000 and 20 <= height <= 5000):
        raise ValueError("Select a region between 20 m and 5 km on each side.")
    return south, west, north, east


@dataclass(frozen=True)
class GroundMap:
    latitude: float
    longitude: float
    zoom: int = 16

    def __post_init__(self):
        parse_location(f"{self.latitude}, {self.longitude}")
        if not 13 <= self.zoom <= 19 or int(self.zoom) != self.zoom:
            raise ValueError("Map zoom must be between 13 and 19.")

    @property
    def scale(self):
        return math.cos(math.radians(self.latitude))

    @property
    def origin(self):
        return self._mercator(self.latitude, self.longitude)

    @staticmethod
    def _mercator(lat, lon):
        return RADIUS * math.radians(lon), RADIUS * math.asinh(math.tan(math.radians(lat)))

    def world(self, lat, lon):
        x, y = self._mercator(lat, lon)
        ox, oy = self.origin
        return (x - ox) * self.scale, (y - oy) * self.scale

    def geographic(self, x, y):
        ox, oy = self.origin
        return math.degrees(math.atan(math.sinh((y / self.scale + oy) / RADIUS))), math.degrees(
            (x / self.scale + ox) / RADIUS
        )

    @property
    def tile_origin(self):
        n = 2**self.zoom
        x = (self.longitude + 180) / 360 * n
        y = (1 - math.asinh(math.tan(math.radians(self.latitude))) / math.pi) / 2 * n
        return max(0, min(n - 3, math.floor(x) - 1)), max(0, min(n - 3, math.floor(y) - 1))

    @property
    def bounds(self):
        """Local west, south, east, north edges of the nine displayed tiles."""
        tx, ty = self.tile_origin
        unit = 2 * math.pi * RADIUS / 2**self.zoom
        ox, oy = self.origin
        return (
            (tx * unit - math.pi * RADIUS - ox) * self.scale,
            (math.pi * RADIUS - (ty + 3) * unit - oy) * self.scale,
            ((tx + 3) * unit - math.pi * RADIUS - ox) * self.scale,
            (math.pi * RADIUS - ty * unit - oy) * self.scale,
        )

    def selection(self, first, last):
        west, south, east, north = self.bounds
        x1, x2 = sorted(max(west, min(east, p[0])) for p in (first, last))
        y1, y2 = sorted(max(south, min(north, p[1])) for p in (first, last))
        s, w = self.geographic(x1, y1)
        n, e = self.geographic(x2, y2)
        return validate_bbox((max(-80.0, s), max(-180.0, w), min(80.0, n), min(180.0, e)))

    def default_selection(self):
        west, south, east, north = self.bounds
        half = min(250.0, (east - west) / 4, (north - south) / 4)
        return self.selection((-half, -half), (half, half))
