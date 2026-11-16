"""Derive display north from the geographic producer's local UTM mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GeographicNorth:
    """A local horizontal direction, or the reason no compass can be shown."""

    direction_xy: tuple[float, float] | None = None
    label: str = "Geographic north at scene origin"
    reason: str = "This scene has no geographic coordinate metadata."


def geographic_north(metadata) -> GeographicNorth:
    """Resolve north once using the scene's CRS and origin, without graph data.

    OSM meshes use projected easting/northing minus ``origin_utm`` with no XY
    rotation. A short geodesic north from that origin captures map-projection
    convergence; local +Y alone would only indicate projected grid north.
    """
    if not isinstance(metadata, Mapping) or not metadata.get("utm_crs"):
        return GeographicNorth()
    origin = metadata.get("origin_utm")
    try:
        if not isinstance(origin, (tuple, list)) or len(origin) not in (2, 3):
            raise ValueError("missing origin")
        east, north = (float(value) for value in origin[:2])
        if not math.isfinite(east) or not math.isfinite(north):
            raise ValueError("nonfinite origin")
    except (TypeError, ValueError, OverflowError):
        return GeographicNorth(reason="Geographic metadata has no valid local projected origin.")
    try:
        from pyproj import CRS, Geod, Transformer
        from pyproj.exceptions import ProjError
    except ImportError:
        return GeographicNorth(
            reason="The geographic compass requires the pyproj coordinate library."
        )
    try:
        crs = CRS.from_user_input(metadata["utm_crs"])
        axes = crs.axis_info
        if (
            not crs.is_projected
            or len(axes) < 2
            or [axis.direction for axis in axes[:2]] != ["east", "north"]
            or not all(math.isclose(axis.unit_conversion_factor, 1.0) for axis in axes[:2])
        ):
            return GeographicNorth(
                reason="The scene CRS must use projected easting and northing in metres."
            )
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
            east, north, errcheck=True
        )
        if not (
            math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 < lat < 90
        ):
            raise ValueError("invalid geographic origin")
        lon_north, lat_north, _ = Geod(ellps="WGS84").fwd(lon, lat, 0, 10)
        x_north, y_north = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            lon_north, lat_north, errcheck=True
        )
        dx, dy = x_north - east, y_north - north
        length = math.hypot(dx, dy)
        if not math.isfinite(length) or length < 1e-6:
            raise ValueError("undefined geographic north")
        return GeographicNorth(direction_xy=(dx / length, dy / length), reason="")
    except (ProjError, TypeError, ValueError, OverflowError):
        return GeographicNorth(reason="The scene's geographic CRS or projected origin is invalid.")
