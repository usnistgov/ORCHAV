"""Display maps in the producer's local metric frame, without changing the scene.

Only one bounded tile level covering the requested scene is acquired. The atlas
is a worker artifact; HTTP tiles live in the ordinary user cache, not the scenario.
"""

from __future__ import annotations

import json
import math
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from .map_tiles import TILE_URL, fetch_tile

RADIUS = 6378137.0
ATTRIBUTION = "© OpenStreetMap contributors"


class _AttributionText(HTMLParser):
    """Saved Builder references may contain a hyperlink, but renderers receive text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def mercator_bounds(metadata):
    """Validate georeferencing and include the producer's ground padding."""
    from .geographic_north import geographic_north

    north = geographic_north(metadata)
    if north.direction_xy is None:
        raise ValueError(north.reason)
    south, west, north, east = map(float, metadata["bbox"])
    if not (-85 < south < north < 85 and -180 <= west < east <= 180):
        raise ValueError("Map bounds must be finite and must not cross the date line.")
    scale = math.cos(math.radians((south + north) / 2))
    pad = float(metadata.get("osm_config", {}).get("ground_pad_m", 0)) / scale
    if not math.isfinite(pad) or pad < 0:
        raise ValueError("Invalid map ground padding.")
    return (
        RADIUS * math.radians(west) - pad,
        RADIUS * math.asinh(math.tan(math.radians(south))) - pad,
        RADIUS * math.radians(east) + pad,
        RADIUS * math.asinh(math.tan(math.radians(north))) + pad,
    )


def tile_layout(bounds):
    """Choose at most 16 tiles and a 1024-pixel longest side at one zoom level."""
    west, south, east, north = bounds
    half = math.pi * RADIUS
    if not (-half <= west < east <= half and -half <= south < north <= half):
        raise ValueError("Padded map bounds extend beyond Web Mercator.")
    for zoom in range(19, -1, -1):
        unit = 2 * half / 2**zoom
        x0, y0 = math.floor((west + half) / unit), math.floor((half - north) / unit)
        x1, y1 = math.ceil((east + half) / unit), math.ceil((half - south) / unit)
        if max(x1 - x0, y1 - y0) <= 4:
            return zoom, x0, y0, x1, y1
    raise ValueError("No bounded map level is available.")


def project_map(metadata, atlas_bounds, *, bounds=None, height=0.0):
    """Project a tessellated Mercator atlas into the exact OSM UTM origin.

    UV origin is the image's top-left. A grid retains meridian convergence and
    projection curvature instead of approximating the scene with a rectangle.
    Builder's saved references and downloaded viewer maps use this same path.
    """
    from pyproj import Transformer

    aw, ass, ae, an = atlas_bounds
    west, south, east, north = bounds if bounds is not None else mercator_bounds(metadata)
    west, south, east, north = max(west, aw), max(south, ass), min(east, ae), min(north, an)
    if west >= east or south >= north:
        raise ValueError("The map atlas does not intersect the scene.")
    n = 12
    x, y = np.meshgrid(np.linspace(west, east, n + 1), np.linspace(south, north, n + 1))
    x, y = x.ravel(), y.ravel()
    px, py = Transformer.from_crs(3857, metadata["utm_crs"], always_xy=True).transform(
        x, y, errcheck=True
    )
    origin = metadata["origin_utm"]
    vertices = np.column_stack((px - origin[0], py - origin[1], np.full(len(x), height)))
    uv = np.column_stack(((x - aw) / (ae - aw), 1 - (y - ass) / (an - ass)))
    triangles = []
    for row in range(n):
        for col in range(n):
            i = row * (n + 1) + col
            triangles.extend(((i, i + 1, i + n + 2), (i, i + n + 2, i + n + 1)))
    return {"vertices": vertices.tolist(), "triangles": triangles, "uvs": uv.tolist()}


def load_scene_map(config, root, cancelled, progress):
    """Worker entry point: load a saved atlas, or fetch just the requested map."""
    import requests
    from PIL import Image

    metadata = config["metadata"]
    bounds = mercator_bounds(metadata)
    zoom, x0, y0, x1, y1 = tile_layout(bounds)
    atlas = Image.new("RGB", ((x1 - x0) * 256, (y1 - y0) * 256))
    total, cached, loaded = (x1 - x0) * (y1 - y0), 0, 0
    with requests.Session() as session:
        for y in range(y0, y1):
            for x in range(x0, x1):
                if cancelled():
                    raise InterruptedError("Map loading cancelled")
                url = config.get("tile_url", TILE_URL).format(z=zoom, x=x, y=y)
                tile, reused = fetch_tile(session, url, Path(config["cache"]), cancelled)
                atlas.paste(tile, ((x - x0) * 256, (y - y0) * 256))
                cached += int(reused)
                loaded += 1
                progress({"message": f"Map tiles: {loaded}/{total} ({cached} cached)"})
    half = math.pi * RADIUS
    unit = 2 * half / 2**zoom
    geometry = project_map(
        metadata,
        (x0 * unit - half, half - y1 * unit, x1 * unit - half, half - y0 * unit),
        bounds=bounds,
    )
    geometry["attribution"] = config.get("attribution") or ATTRIBUTION
    atlas.save(root / "map.png")
    (root / "map.json").write_text(json.dumps(geometry), encoding="utf-8")
    return {"map_path": str(root / "map.json"), "loaded_tiles": loaded, "cached_tiles": cached}


def read_map(path):
    """Return neutral geometry and texture for either saved or temporary maps."""
    from PIL import Image

    from ..types.render_payloads import MeshPayload

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    uvs = np.asarray(data["uvs"], dtype=np.float32)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not len(triangles)
        or triangles.min() < 0
        or triangles.max() >= len(vertices)
        or uvs.shape != (len(vertices), 2)
        or not np.isfinite(vertices).all()
        or not np.isfinite(uvs).all()
    ):
        raise ValueError("Invalid saved map geometry.")
    texture = path.with_name("map.png")
    if not texture.is_file():
        raise ValueError("The saved map image is missing.")
    try:
        # A filename suffix does not identify the decoder. Validate the complete
        # raster before publishing its path to Builder or the scene renderer.
        with Image.open(texture, formats=["PNG"]) as image:
            image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("The saved map image must be a readable PNG.") from exc
    credit = _AttributionText()
    credit.feed(str(data.get("attribution") or ATTRIBUTION))
    return (
        MeshPayload(vertices=vertices, triangles=triangles, triangle_uvs=uvs),
        str(texture),
        "".join(credit.parts).strip() or ATTRIBUTION,
    )
