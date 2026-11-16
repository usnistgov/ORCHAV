"""Merge recorded trees with OSM points and bounded, explicit inferred vegetation.

All geometry uses a caller-selected metric CRS. There is no city-specific
lookup or mesh code here. Mapped tree points take precedence over deterministic
polygon filling; buildings, water, paths and nonliving inventory sites can be
excluded. Ordinary gardens/lawns are not assumed to be forests.
"""

from __future__ import annotations

import math
from collections import Counter
from hashlib import sha256

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep


class _PointIndex:
    """Small spatial hash supports incremental duplicate/spacing checks."""

    def __init__(self, points, cell_m=4.0):
        self.cell_m = cell_m
        self.cells = {}
        for x, y in points:
            self.add(x, y)

    def add(self, x, y):
        key = (math.floor(x / self.cell_m), math.floor(y / self.cell_m))
        self.cells.setdefault(key, []).append((x, y))

    def near(self, x, y, radius):
        cx, cy = math.floor(x / self.cell_m), math.floor(y / self.cell_m)
        reach = math.ceil(radius / self.cell_m)
        for i in range(cx - reach, cx + reach + 1):
            for j in range(cy - reach, cy + reach + 1):
                if any(
                    (x - a) ** 2 + (y - b) ** 2 < radius**2 for a, b in self.cells.get((i, j), ())
                ):
                    return True
        return False


def merge_vegetation(
    recorded: list[dict],
    osm_features: list[dict],
    *,
    bounds,
    exclusions,
    unavailable_sites=(),
    duplicate_radius_m=3.0,
    inferred_spacing_m=8.0,
    max_candidates=100_000,
) -> tuple[list[dict], dict]:
    """Return recorded/OSM/inferred points with stable source identities.

    Recorded entries contain id, x, y and optionally species. OSM records contain
    a shapely geometry plus normal OSM tags and id. Polygon inference is limited
    to wood/forest and explicitly botanical gardens. It is an opt-in geometry
    approximation, not a canopy observation or a foliage propagation model.
    """
    if duplicate_radius_m <= 0 or inferred_spacing_m <= 0 or max_candidates <= 0:
        raise ValueError("Vegetation spacing and candidate budget must be positive")
    result = [{**item, "source": "inventory", "inferred": False} for item in recorded]
    points = _PointIndex([(p["x"], p["y"]) for p in result])
    unavailable = _PointIndex(unavailable_sites)
    blocked = prep(exclusions)
    counters = Counter()
    features = sorted(osm_features, key=lambda f: str(f["id"]))

    def accept(x, y, record, radius):
        point = Point(x, y)
        if not bounds.covers(point) or blocked.intersects(point):
            counters["excluded_geometry"] += 1
            return
        if unavailable.near(x, y, duplicate_radius_m):
            counters["unavailable_inventory_site"] += 1
            return
        if points.near(x, y, radius):
            counters["near_existing_tree"] += 1
            return
        result.append({**record, "x": float(x), "y": float(y)})
        points.add(x, y)

    # Individual mapped trees always precede inferred area/row samples.
    for feature in features:
        geom = feature["geometry"]
        if geom.geom_type == "Point" and feature.get("natural") == "tree":
            accept(
                geom.x,
                geom.y,
                {
                    "id": str(feature["id"]),
                    "source": "osm_tree",
                    "inferred": False,
                    "species": feature.get("species") or "",
                },
                duplicate_radius_m,
            )

    candidate_count = 0
    for feature in features:
        geom = feature["geometry"]
        wooded = feature.get("natural") == "wood" or feature.get("landuse") == "forest"
        botanical = feature.get("garden:type") == "botanical"
        row = feature.get("natural") == "tree_row"
        if not (wooded or botanical or row):
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        clipped = geom.intersection(bounds)
        if clipped.is_empty:
            continue
        identity = str(feature["id"])
        seed = int.from_bytes(sha256(identity.encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        candidates = []
        if row and clipped.geom_type in {"LineString", "MultiLineString"}:
            lines = [clipped] if clipped.geom_type == "LineString" else list(clipped.geoms)
            for line in lines:
                n = math.ceil(line.length / inferred_spacing_m)
                if candidate_count + len(candidates) + n > max_candidates:
                    raise ValueError("Vegetation candidate budget exceeded")
                candidates.extend(line.interpolate(i * inferred_spacing_m) for i in range(n))
        elif clipped.geom_type in {"Polygon", "MultiPolygon"}:
            lo_x, lo_y, hi_x, hi_y = clipped.bounds
            xs = np.arange(lo_x + inferred_spacing_m / 2, hi_x, inferred_spacing_m)
            ys = np.arange(lo_y + inferred_spacing_m / 2, hi_y, inferred_spacing_m)
            if candidate_count + len(xs) * len(ys) > max_candidates:
                raise ValueError("Vegetation candidate budget exceeded")
            for y in ys:
                for x in xs:
                    jitter = rng.uniform(-0.22, 0.22, 2) * inferred_spacing_m
                    candidates.append(Point(x + jitter[0], y + jitter[1]))
        candidate_count += len(candidates)
        for index, point in enumerate(candidates):
            if not row and not clipped.covers(point):
                continue
            accept(
                point.x,
                point.y,
                {
                    "id": f"{identity}:{index}",
                    "source": "osm_row_inferred" if row else "osm_area_inferred",
                    "source_feature": identity,
                    "source_name": feature.get("name") or "",
                    "inferred": True,
                    "species": "",
                },
                inferred_spacing_m * 0.8,
            )
    return result, {
        "sources": dict(Counter(p["source"] for p in result)),
        "decisions": dict(counters),
        "inferred_candidates": candidate_count,
        "duplicate_radius_m": duplicate_radius_m,
        "inferred_spacing_m": inferred_spacing_m,
    }
