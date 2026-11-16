"""Read named local road curves without importing OSM acquisition packages.

Candidates are static scene annotations. Camera-dependent selection belongs to
the renderer; neither this loader nor its source files enter frame playback.
"""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
MIN_SEGMENT_LENGTH_M = 12.0
ANCHOR_SPACING_M = 60.0
MAX_CANDIDATES = 4000


@dataclass(frozen=True)
class StreetName:
    """One candidate anchored on a named road and its supporting curve length."""

    name: str
    position: tuple[float, float, float]
    length_m: float


def street_graph_path(scene_path: Path) -> tuple[Path, dict]:
    """Honor remapped scene sidecars, falling back to the adjacent graph."""
    metadata_path = scene_path.with_name("osm_metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    relative = metadata.get("artifacts", {}).get("street_network_graphml", "street_network.graphml")
    return (scene_path.parent / relative).resolve(), metadata


def _name(value: str) -> str:
    value = value.strip()
    if value.startswith("["):
        try:
            names = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return ""
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            return ""
        value = " / ".join(dict.fromkeys(name.strip() for name in names if name.strip()))
    return " ".join(value.split())


def load_street_names(path: Path, metadata: dict) -> tuple[StreetName, ...]:
    """Place bounded, deduplicated candidates at arc-length road midpoints.

    New graphs retain ``geometry_local``. Older graphs use local node positions
    or projected node positions plus the explicitly recorded UTM origin. Raw
    longitude/latitude is never interpreted as scene metres.
    """
    root = ET.parse(path).getroot()
    keys = {key.attrib["id"]: key.attrib.get("attr.name", "") for key in root.findall("g:key", _NS)}

    def values(element):
        return {keys.get(d.attrib["key"], ""): d.text or "" for d in element.findall("g:data", _NS)}

    origin = metadata.get("origin_utm")
    nodes = {}
    for node in root.findall("g:graph/g:node", _NS):
        data = values(node)
        try:
            if "x_local" in data and "y_local" in data:
                xy = [float(data["x_local"]), float(data["y_local"])]
            elif origin is not None and metadata.get("utm_crs"):
                xy = [float(data["x"]) - origin[0], float(data["y"]) - origin[1]]
            else:
                continue
            if np.isfinite(xy).all():
                nodes[node.attrib["id"]] = xy
        except (KeyError, TypeError, ValueError, IndexError):
            continue

    candidates, seen = [], set()
    for edge in root.findall("g:graph/g:edge", _NS):
        data = values(edge)
        name = _name(data.get("name", ""))
        if not name or data.get("tunnel", "no").lower() not in {"no", "false", "0", ""}:
            continue
        try:
            curve = (
                json.loads(data["geometry_local"])
                if data.get("geometry_local")
                else [nodes[edge.attrib["source"]], nodes[edge.attrib["target"]]]
            )
            xy = np.asarray(curve, dtype=float)
            if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 2 or not np.isfinite(xy).all():
                continue
            xy = xy[:, :2]
            distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            cumulative = np.r_[0.0, distances.cumsum()]
            length = float(cumulative[-1])
            if length < MIN_SEGMENT_LENGTH_M:
                continue
            # Generated roads sit 0.15 m above terrain. The remaining 0.02 m
            # keeps the label anchor clear of the surface, without a pole.
            z = float(data.get("bridge_z_offset", 0.0)) + 0.17
            if not np.isfinite(z):
                continue
            # More than one anchor on long curves keeps a name available in a
            # close street view. The renderer still shows each name only once.
            count = min(MAX_CANDIDATES, max(1, int(np.ceil(length / ANCHOR_SPACING_M))))
            for middle in (np.arange(count) + 0.5) * length / count:
                segment = min(
                    int(np.searchsorted(cumulative, middle, side="right")) - 1, len(xy) - 2
                )
                fraction = (middle - cumulative[segment]) / distances[segment]
                center = xy[segment] * (1 - fraction) + xy[segment + 1] * fraction
                # Reverse edges and parallel lane copies share one candidate.
                key = (name, *np.round(center / 3).astype(int), round(z))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(StreetName(name, (float(center[0]), float(center[1]), z), length))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    # Prefer long segments if a very large scene exceeds the draw budget.
    return tuple(
        sorted(candidates, key=lambda c: (-c.length_m, c.name, c.position))[:MAX_CANDIDATES]
    )
