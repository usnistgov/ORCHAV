# Portions of this file are adapted from NVlabs/sionna-large-radio-maps
# (scripts/scene_builder.py), which carries the following notice:
#
#   Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#
# The NVIDIA code is itself adapted from geo2sigmap
# (https://github.com/functions-lab/geo2sigmap).
#
# See generator/osm/NOTICE.md for the full list of changes.

"""Build Mitsuba 3 / Sionna RT scenes from OpenStreetMap data.

Usage
-----
As a library::

    from generator.osm import build_scene_from_center
    scene_dir = build_scene_from_center(
        lat=48.8738, lon=2.2950, size_m=500,
        output_dir="scenarios/osm/etoile",
    )

As a CLI::

    python -m generator.osm --center 48.8738 2.2950 --size 500 \\
        -o scenarios/osm/etoile

"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import signal
import tempfile
import time
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from collections import namedtuple
from functools import wraps
from pathlib import Path

import numpy as np
import open3d as o3d
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString as _LineString
from shapely.geometry import Point, Polygon, shape
from shapely.geometry import box as shapely_box
from shapely.prepared import prep as shapely_prep
from triangle import triangulate

from .acquisition import (
    AcquisitionError,
    acquisition_scope,
    active_acquisition,
    download_file,
    merge_tags,
    select_features,
)
from .barriers import barrier_mesh, barrier_spec
from .coords import (
    bbox_from_center,
    flat_ground_mesh,
    get_utm_crs,
    reorder_localize,
    terrain_ground_mesh,
    unique_coords,
)
from .metadata import OSM_METADATA_SCHEMA_VERSION, normalize_osm_config
from .publication import staged_scene
from .street_width import street_width

logger = logging.getLogger(__name__)

_OSM_INSUFFICIENT_RESPONSE_ERROR = getattr(ox._errors, "InsufficientResponseError", None)
_OSM_NO_DATA_ERRORS = tuple(
    err for err in (_OSM_INSUFFICIENT_RESPONSE_ERROR,) if isinstance(err, type)
)
_OSM_RETRYABLE_ERRORS = tuple(
    err
    for err in (
        getattr(ox._errors, "ResponseStatusCodeError", None),
        getattr(ox._errors, "RequestsConnectionError", None),
        getattr(ox._errors, "RequestsReadTimeout", None),
        getattr(ox._errors, "RequestsTimeout", None),
    )
    if isinstance(err, type)
) + (ConnectionError, TimeoutError, OSError)
_OSM_QUERY_ERRORS = _OSM_NO_DATA_ERRORS + _OSM_RETRYABLE_ERRORS

# ITU-R P.2040 radio materials with display colors (RGB 0-1).
ITU_MATERIALS: dict[str, dict] = {
    "concrete": {
        "color": (0.54, 0.54, 0.54),
        "thickness": 0.1,
    },
    "marble": {
        "color": (0.70, 0.64, 0.49),
        "thickness": 0.1,
    },
    "metal": {
        "color": (0.22, 0.22, 0.25),
        "thickness": 0.005,
    },
    "wood": {
        "color": (0.55, 0.35, 0.17),
        "thickness": 0.05,
    },
    "wet_ground": {
        "color": (0.40, 0.30, 0.15),
        "thickness": 0.5,
    },
    "glass": {
        "color": (0.65, 0.80, 0.90),
        "thickness": 0.01,
    },
    "medium_dry_ground": {
        "color": (0.30, 0.30, 0.30),
        "thickness": 0.3,
    },
    "brick": {
        "color": (0.40, 0.11, 0.09),
        "thickness": 0.1,
    },
    "plasterboard": {
        "color": (0.80, 0.78, 0.70),
        "thickness": 0.05,
    },
}

# Mapping from building type to material.
DEFAULT_BUILDING_MATERIAL = "concrete"
DEFAULT_GROUND_MATERIAL = "wet_ground"
DEFAULT_WATER_MATERIAL = "glass"
DEFAULT_STREET_MATERIAL = "medium_dry_ground"

# Default street half-width in meters (used when OSM has no width tag).
DEFAULT_STREET_HALF_WIDTH_M = 4.0

# Mapping from OSM surface tag to ITU material for streets.
_SURFACE_MATERIAL_MAP: dict[str, str] = {
    "asphalt": "concrete",
    "concrete": "concrete",
    "concrete:lanes": "concrete",
    "concrete:plates": "concrete",
    "paving_stones": "concrete",
    "sett": "concrete",
    "cobblestone": "concrete",
    "unhewn_cobblestone": "concrete",
    "bricks": "brick",
    "metal": "metal",
    "wood": "wood",
    "compacted": "medium_dry_ground",
    "fine_gravel": "medium_dry_ground",
    "gravel": "medium_dry_ground",
    "ground": "medium_dry_ground",
    "dirt": "medium_dry_ground",
    "earth": "medium_dry_ground",
    "sand": "medium_dry_ground",
    "grass": "wet_ground",
    "mud": "wet_ground",
}

# Default floor height in meters (used when OSM tags are missing).
DEFAULT_FLOOR_HEIGHT_M = 3.5

# Skip terrain water bodies that span too much elevation to be represented as a
# single flat surface.
# Skip water bodies that span extreme elevation (e.g., waterfalls, reservoirs
# on steep slopes).  Rivers in cities should pass (the Seine spans ~15m across
# hilly Paris).  Set high enough to allow rivers but filter truly broken cases.
MAX_FLAT_WATER_ELEVATION_SPAN_M = 50.0

# Retry external OSM service queries to ride out transient Overpass failures.
OSM_QUERY_MAX_ATTEMPTS = 3
GLOBALML_INDEX_DOWNLOAD_TIMEOUT_S = 15

# Default Overpass query timeout in seconds.  Used as the [timeout:N] header on
# every Overpass query.  Big-bbox queries (Manhattan-sized buildings) routinely
# need more than the OSMnx default of 180 s on slow Overpass mirrors, so we
# err on the generous side.
DEFAULT_OSM_OVERPASS_TIMEOUT_S = 180

# Maximum distance (meters) to match an external building to an OSM centroid.
HEIGHT_MATCH_RADIUS_M = 30.0
SYNTHETIC_BUILDING_HEIGHT_SEED_SCALE = 1000
SYNTHETIC_BUILDING_HEIGHT_SEED_MODULUS = 2**31
SYNTHETIC_BUILDING_MIN_FLOORS = 1
SYNTHETIC_BUILDING_MAX_FLOORS = 15
SYNTHETIC_BUILDING_MEAN_FLOORS = 5
SYNTHETIC_BUILDING_FLOOR_STDDEV = 1
OSMNX_MAX_QUERY_AREA_SIZE_M2 = 50_000_000_000

# Cache directory for downloaded GlobalML tiles.
_GLOBALML_CACHE_DIR = os.path.join(
    os.environ.get("ORCHAV_CACHE_DIR", os.path.expanduser("~/.orchav/cache")), "globalml"
)

# GlobalML tile index URL.
_GLOBALML_INDEX_URL = (
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
)

# ---- Optional ETH Canopy Height + ESA WorldCover tile sources ---------------
# Both datasets are CC-BY 4.0.  Attribution is mandatory whenever either is
# used; see the runtime log in ``_cite_external_dataset`` and the
# ``external_data_sources`` entry in ``osm_metadata.json``.

_ETH_CANOPY_CACHE_DIR = os.path.join(
    os.environ.get("ORCHAV_CACHE_DIR", os.path.expanduser("~/.orchav/cache")), "eth_canopy"
)
# ETH publishes its canopy-height tiles via a libdrive share link (rather
# than a path-under-hostname URL).  The "share key" is the immutable
# handle tied to the 2020 v1 dataset; tile names append to a query string.
_ETH_CANOPY_SHARE_KEY = "cO8or7iOe5dT2Rt"
_ETH_CANOPY_BASE_URL = (
    f"https://libdrive.ethz.ch/index.php/s/{_ETH_CANOPY_SHARE_KEY}/download"
    "?path=%2F3deg_cogs&files="
)
_ETH_CANOPY_ATTRIBUTION = {
    "name": "ETH Global Canopy Height",
    "version": "2020 v1",
    "license": "CC-BY 4.0",
    "citation": (
        "Lang et al. 2023, Nature Ecology & Evolution 7:1778-1789 (DOI 10.3929/ethz-b-000609802)"
    ),
    "doi": "10.3929/ethz-b-000609802",
    "url": _ETH_CANOPY_BASE_URL,
}

_WORLDCOVER_CACHE_DIR = os.path.join(
    os.environ.get("ORCHAV_CACHE_DIR", os.path.expanduser("~/.orchav/cache")), "worldcover"
)
_WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
_WORLDCOVER_ATTRIBUTION = {
    "name": "ESA WorldCover",
    "version": "v200 (2021)",
    "license": "CC-BY 4.0",
    "citation": (
        "\u00a9 ESA WorldCover project 2021 / Contains modified Copernicus "
        "Sentinel data (2021) processed by ESA WorldCover consortium"
    ),
    "url": _WORLDCOVER_BASE_URL,
}

# WorldCover class codes (uint8, LCCS-aligned per ESA documentation).
_WC_TREE_COVER = 10
_WC_SHRUBLAND = 20
_WC_GRASSLAND = 30
_WC_CROPLAND = 40
_WC_BUILT_UP = 50
_WC_BARE_SPARSE = 60
_WC_SNOW_ICE = 70
_WC_PERMANENT_WATER = 80
_WC_HERBACEOUS_WETLAND = 90
_WC_MANGROVES = 95
_WC_MOSS_LICHEN = 100


class _GlobalMLHeightIndex:
    """Spatial index of Microsoft GlobalML building heights.

    Downloads a Bing-quadkey tile from the Microsoft Global ML Building
    Footprints dataset, parses buildings within the scene bbox, and builds
    a KD-tree for nearest-centroid matching.

    Data source: `Microsoft GlobalMLBuildingFootprints
    <https://github.com/microsoft/GlobalMLBuildingFootprints>`_
    (CDLA Permissive 2.0 license, ML-derived heights from satellite imagery).
    """

    def __init__(
        self,
        bbox_wgs84: tuple[float, float, float, float],
        to_utm,
    ):
        self._tree = None
        self._heights = np.array([], dtype=np.float32)

        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # Find the quadkey tile URL.
        tile_url = self._find_tile_url(center_lat, center_lon)
        if tile_url is None:
            logger.info("GlobalML: no tile found for (%.4f, %.4f)", center_lat, center_lon)
            return

        # Download tile (cached).
        tile_path = self._download_tile(tile_url)
        if tile_path is None:
            return

        # Reuse the processed spatial lookup, keyed by tile revision, bbox
        # and projection. A warm rebuild need not scan a country-sized gzip.
        tile_stat = Path(tile_path).stat()
        lookup_key = hashlib.sha256(
            json.dumps(
                {
                    "version": 1,
                    "tile": tile_path,
                    "size": tile_stat.st_size,
                    "mtime": tile_stat.st_mtime_ns,
                    "bbox": list(bbox_wgs84),
                    "projection": to_utm.definition,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:24]
        lookup_path = Path(tile_path).parent / f"lookup-{lookup_key}.npz"
        try:
            if lookup_path.exists():
                with np.load(lookup_path, allow_pickle=False) as lookup:
                    points = lookup["points"]
                    heights = lookup["heights"]
                if (
                    points.shape != (len(heights), 2)
                    or not np.isfinite(points).all()
                    or not np.isfinite(heights).all()
                ):
                    raise ValueError("Invalid GlobalML lookup arrays")
                if len(points):
                    from scipy.spatial import cKDTree

                    self._tree = cKDTree(points)
                self._heights = heights
                logger.info("GlobalML processed lookup cache HIT: %s", lookup_path.name)
                return
        except (OSError, ValueError, KeyError):
            lookup_path.unlink(missing_ok=True)

        # Parse buildings within bbox and build KD-tree.
        import gzip

        centroids_utm = []
        heights = []
        try:
            with gzip.open(tile_path, "rt") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    props = obj.get("properties", {})
                    h = props.get("height", -1)
                    if not isinstance(h, (int, float)) or h <= 0:
                        continue

                    geom = obj.get("geometry", {})
                    coords = geom.get("coordinates", [[]])
                    ring = coords[0] if coords else []
                    if len(ring) < 3:
                        continue

                    # Quick bbox filter on first vertex.
                    lon, lat = ring[0]
                    if not (min_lon - 0.01 <= lon <= max_lon + 0.01):
                        continue
                    if not (min_lat - 0.01 <= lat <= max_lat + 0.01):
                        continue

                    # Compute centroid (average of ring vertices).
                    cx = sum(p[0] for p in ring) / len(ring)
                    cy = sum(p[1] for p in ring) / len(ring)
                    if not (min_lon - 0.001 <= cx <= max_lon + 0.001):
                        continue
                    if not (min_lat - 0.001 <= cy <= max_lat + 0.001):
                        continue

                    ux, uy = to_utm.transform(cx, cy)
                    centroids_utm.append((ux, uy))
                    heights.append(h)
        except (OSError, gzip.BadGzipFile) as exc:
            logger.warning("GlobalML tile parse failed: %s", exc)
            Path(tile_path).unlink(missing_ok=True)
            raise

        try:
            fd, temporary = tempfile.mkstemp(dir=lookup_path.parent, suffix=".npz")
            os.close(fd)
            try:
                np.savez_compressed(
                    temporary,
                    points=np.asarray(centroids_utm, dtype=np.float64).reshape(-1, 2),
                    heights=np.asarray(heights, dtype=np.float32),
                )
                os.replace(temporary, lookup_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("GlobalML lookup cache write failed: %s", exc)

        if not centroids_utm:
            logger.info("GlobalML: 0 buildings with height in bbox")
            return

        from scipy.spatial import cKDTree

        pts = np.array(centroids_utm, dtype=np.float64)
        self._tree = cKDTree(pts)
        self._heights = np.array(heights, dtype=np.float32)
        logger.info(
            "GlobalML height index: %d buildings (%.0f-%.0f m)",
            len(heights),
            self._heights.min(),
            self._heights.max(),
        )

    @staticmethod
    def _latlon_to_quadkey(lat: float, lon: float, level: int = 9) -> str:
        """Convert lat/lon to Bing Maps quadkey at the given zoom level."""
        x = int((lon + 180) / 360 * (1 << level))
        sin_lat = math.sin(lat * math.pi / 180)
        y = int((0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (1 << level))
        qk = ""
        for i in range(level, 0, -1):
            d = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                d += 1
            if (y & mask) != 0:
                d += 2
            qk += str(d)
        return qk

    # Class-level cache for the index CSV (downloaded once per process).
    _cached_index: str | None = None

    @classmethod
    def _find_tile_url(cls, lat: float, lon: float) -> str | None:
        """Find the GlobalML tile URL for a location (index cached per process)."""
        import urllib.request

        qk = cls._latlon_to_quadkey(lat, lon)

        if cls._cached_index is None:
            try:
                # Cache the index file locally as well.
                index_cache = os.path.join(_GLOBALML_CACHE_DIR, "dataset-links.csv")
                os.makedirs(_GLOBALML_CACHE_DIR, exist_ok=True)
                if os.path.exists(index_cache):
                    with open(index_cache, encoding="utf-8") as f:
                        cls._cached_index = f.read()
                    logger.debug("GlobalML index loaded from cache")
                else:
                    download_file(
                        _GLOBALML_INDEX_URL, index_cache, timeout=GLOBALML_INDEX_DOWNLOAD_TIMEOUT_S
                    )
                    cls._cached_index = Path(index_cache).read_text(encoding="utf-8")
                    logger.debug("GlobalML index downloaded and cached")
            except (OSError, urllib.error.URLError) as exc:
                logger.warning("GlobalML index download failed: %s", exc)
                raise

        header = cls._cached_index.splitlines()[0].lower() if cls._cached_index else ""
        if "quadkey" not in header or "url" not in header:
            cls._cached_index = None
            Path(_GLOBALML_CACHE_DIR, "dataset-links.csv").unlink(missing_ok=True)
            raise AcquisitionError("Invalid GlobalML dataset index")
        for line in cls._cached_index.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 3 and parts[1] == qk:
                return parts[2]
        return None

    @staticmethod
    def _download_tile(url: str) -> str | None:
        """Download a GlobalML tile, caching locally."""
        import hashlib
        import urllib.request

        os.makedirs(_GLOBALML_CACHE_DIR, exist_ok=True)
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(_GLOBALML_CACHE_DIR, f"{cache_key}.csv.gz")

        if os.path.exists(cache_path):
            logger.info("GlobalML tile cached: %s", cache_path)
            return cache_path

        try:
            logger.info("Downloading GlobalML tile (this may take a minute)...")
            download_file(url, cache_path)
            size_mb = os.path.getsize(cache_path) / 1e6
            logger.info("GlobalML tile downloaded: %.0f MB → %s", size_mb, cache_path)
            return cache_path
        except (OSError, urllib.error.URLError) as exc:
            logger.warning("GlobalML tile download failed: %s", exc)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            raise

    def query(self, utm_x: float, utm_y: float) -> float | None:
        """Return the GlobalML height for the nearest building, or None."""
        if self._tree is None:
            return None
        dist, idx = self._tree.query([utm_x, utm_y])
        if dist > HEIGHT_MATCH_RADIUS_M:
            return None
        return float(self._heights[idx])


# Module-level placeholder; set per-scene in build_scene().
_height_index = None  # _GlobalMLHeightIndex instance or None

# Height source counters — reset per scene for logging.
_height_source_counts: dict[str, int] = {}


def _tile_lat_lon_origin(lon: float, lat: float, step_deg: int = 3) -> tuple[int, int]:
    """Return the SW-corner lat/lon of the 3°×3° tile containing *(lon, lat)*.

    Both ETH Canopy Height and ESA WorldCover use the same 3° SW-corner
    grid, so this helper is shared.  Tile origins are *floored*: a point
    at (48.85, 2.34) lands in the tile whose SW corner is (48°, 3°)...
    wait — lat 48 + 0.85 < 48 + 3, so yes, (48, 3°), which is named
    ``N48E003``.
    """
    import math as _math

    lat_origin = int(_math.floor(lat / step_deg) * step_deg)
    lon_origin = int(_math.floor(lon / step_deg) * step_deg)
    return lat_origin, lon_origin


def _format_tile_coord(lat_origin: int, lon_origin: int) -> str:
    """Format a 3°-tile SW corner as ``NxxEyyy`` / ``SxxWyyy`` etc."""
    lat_hem = "N" if lat_origin >= 0 else "S"
    lon_hem = "E" if lon_origin >= 0 else "W"
    return f"{lat_hem}{abs(lat_origin):02d}{lon_hem}{abs(lon_origin):03d}"


def _tile_coords_for_bbox(
    bbox_wgs84: tuple[float, float, float, float], step_deg: int = 3
) -> list[tuple[int, int]]:
    """Return every (lat_origin, lon_origin) 3° tile that intersects *bbox*.

    bbox_wgs84 is ``(min_lon, min_lat, max_lon, max_lat)``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    import math as _math

    lat_start = int(_math.floor(min_lat / step_deg) * step_deg)
    lon_start = int(_math.floor(min_lon / step_deg) * step_deg)
    # Include the upper-right tile even if max aligns exactly on a step boundary.
    lat_end = int(_math.floor((max_lat - 1e-9) / step_deg) * step_deg)
    lon_end = int(_math.floor((max_lon - 1e-9) / step_deg) * step_deg)

    tiles: list[tuple[int, int]] = []
    lat = lat_start
    while lat <= lat_end:
        lon = lon_start
        while lon <= lon_end:
            tiles.append((lat, lon))
            lon += step_deg
        lat += step_deg
    return tiles


def _download_raster_tile(
    url: str, cache_dir: str, cache_filename: str, *, label: str
) -> str | None:
    """Download a raster tile to *cache_dir/cache_filename*, returning the path.

    Atomic write via ``<name>.tmp`` + ``os.replace`` so a Ctrl-C mid-download
    can't leave a half file.  Returns None on failure (404, connection
    error, etc.) — caller decides how to react.

    **Content validation.** ``urllib.request.urlretrieve`` saves whatever
    body the server returns, regardless of status code.  For upstreams
    that 301-redirect a missing tile to an HTML landing page (e.g.
    the ETH Research Collection DOI wrapper), we'd end up with an HTML
    file masquerading as a GeoTIFF.  We verify the first few bytes are
    a TIFF magic number before accepting the download.
    """
    import urllib.error
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_path):
        logger.info("%s tile cached: %s", label, cache_filename)
        return cache_path

    tmp_path = cache_path + ".tmp"
    try:
        logger.info("Downloading %s tile %s...", label, cache_filename)
        download_file(url, tmp_path)
        # TIFF magic numbers: "II\x2a\x00" (little-endian) or "MM\x00\x2a" (big-endian).
        with open(tmp_path, "rb") as fh:
            magic = fh.read(4)
        if magic not in (b"II*\x00", b"MM\x00*"):
            logger.warning(
                "%s tile download returned non-TIFF content (magic=%r, url=%s); "
                "leaving canopy/land-cover data disabled for this scene",
                label,
                magic,
                url,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return None
        os.replace(tmp_path, cache_path)
        size_mb = os.path.getsize(cache_path) / 1e6
        logger.info("%s tile downloaded: %.1f MB → %s", label, size_mb, cache_filename)
        return cache_path
    except (OSError, urllib.error.URLError) as exc:
        logger.warning("%s tile download failed (%s): %s", label, url, exc)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return None


class _RasterTileIndex:
    """Shared helper for 3°×3° raster-tile fetchers (ETH Canopy, WorldCover).

    Both datasets follow the same layout: one Cloud-Optimized GeoTIFF per
    3° tile, named with the SW-corner lat/lon.  Subclasses override
    ``_tile_filename`` to provide the dataset-specific name pattern.

    Lookups open the tile with rasterio and sample pixels directly.  We
    keep each tile open via ``rasterio.open()`` so subsequent queries
    don't reopen the file.
    """

    LABEL = "raster"
    BASE_URL = ""
    CACHE_DIR = ""
    ATTRIBUTION: dict[str, str] = {}

    def __init__(self, bbox_wgs84: tuple[float, float, float, float]):
        self._bbox = bbox_wgs84
        self._datasets: dict[tuple[int, int], object] = {}
        self._empty = True
        self.failures = []

        tiles = _tile_coords_for_bbox(bbox_wgs84)
        if not tiles:
            return

        try:
            import rasterio  # noqa: F401 — confirm availability up-front
        except ImportError:
            self.failures.append("rasterio is not installed")
            logger.warning("%s disabled: rasterio is not installed in this venv", self.LABEL)
            return

        for lat_origin, lon_origin in tiles:
            filename = self._tile_filename(lat_origin, lon_origin)
            url = self._tile_url(filename)
            tile_path = _download_raster_tile(url, self.CACHE_DIR, filename, label=self.LABEL)
            if tile_path is None:
                self.failures.append(f"Tile unavailable: {filename}")
                continue
            try:
                import rasterio

                self._datasets[(lat_origin, lon_origin)] = rasterio.open(tile_path)
                self._empty = False
            except Exception as exc:  # pragma: no cover - corrupted tile
                self.failures.append(f"Tile invalid: {filename}: {exc}")
                Path(tile_path).unlink(missing_ok=True)
                logger.warning("%s tile open failed for %s (%s)", self.LABEL, filename, exc)

    # Subclasses must implement:
    def _tile_filename(self, lat_origin: int, lon_origin: int) -> str:
        raise NotImplementedError

    def _tile_url(self, filename: str) -> str:
        """Return the full download URL for a given tile filename.

        Default concatenation is ``BASE_URL + '/' + filename``.  Subclasses
        whose upstream uses a query-string URL (e.g. the ETH libdrive
        share) override this to append the filename to the ``files=``
        parameter instead.
        """
        return f"{self.BASE_URL}/{filename}"

    def is_available(self) -> bool:
        return not self._empty

    def _tile_for_point(self, lon: float, lat: float):
        """Return the rasterio dataset whose tile contains *(lon, lat)*."""
        key = _tile_lat_lon_origin(lon, lat)
        return self._datasets.get(key)

    def sample_point(self, lon: float, lat: float) -> float | None:
        """Return the raw pixel value at a WGS84 point, or None if outside.

        Caller must cast / interpret the value (float meters for ETH,
        uint8 class code for WorldCover).
        """
        ds = self._tile_for_point(lon, lat)
        if ds is None:
            return None
        try:
            values = list(ds.sample([(lon, lat)]))
            if not values:
                return None
            arr = values[0]
            if arr is None or len(arr) == 0:
                return None
            return float(arr[0])
        except Exception:  # pragma: no cover - rasterio sampling edge case
            return None


class _EthCanopyHeightIndex(_RasterTileIndex):
    """Spatial index of ETH Global Canopy Height raster tiles.

    Downloads the 3°×3° Cloud-Optimized GeoTIFF tiles covering the scene
    bounding box from ``share.phys.ethz.ch`` (anonymous HTTPS) and
    exposes per-point and per-polygon canopy-height lookups.

    **License.** CC-BY 4.0.  First use of this index per build emits a
    logger.info attribution line, and the scene metadata records the
    required citation string.  See ``generator/osm/NOTICE.md``.
    """

    LABEL = "ETH canopy"
    BASE_URL = _ETH_CANOPY_BASE_URL
    CACHE_DIR = _ETH_CANOPY_CACHE_DIR
    ATTRIBUTION = _ETH_CANOPY_ATTRIBUTION

    def _tile_filename(self, lat_origin: int, lon_origin: int) -> str:
        coord = _format_tile_coord(lat_origin, lon_origin)
        return f"ETH_GlobalCanopyHeight_10m_2020_{coord}_Map.tif"

    def _tile_url(self, filename: str) -> str:
        """ETH libdrive URL: append the filename to the ``files=`` query param."""
        return f"{self.BASE_URL}{filename}"

    def height_at(self, lon: float, lat: float) -> float | None:
        """Return canopy height in meters at a WGS84 point, or None.

        Zero-valued cells (non-vegetated ground) return None so the
        caller can fall back to the OSM / default height, rather than
        mistaking "no trees here" for a 0 m tree.
        """
        value = self.sample_point(lon, lat)
        if value is None or value <= 0:
            return None
        return float(value)

    def height_over_polygon(self, polygon, to_wgs84) -> float | None:
        """Return the mean non-zero canopy height over a shapely polygon.

        *polygon* is assumed to be in UTM coordinates.  *to_wgs84* is a
        pyproj Transformer going from UTM to WGS84 so we can sample the
        raster in its native CRS.  Sampling is done at the polygon
        centroid (fast) and, if the centroid is zero, the four bbox
        corners — good enough for forest-polygon averaging without
        pulling in ``rasterio.features.rasterize``.
        """
        if polygon.is_empty:
            return None
        cx_utm, cy_utm = polygon.centroid.x, polygon.centroid.y
        cx_lon, cy_lat = to_wgs84.transform(cx_utm, cy_utm)
        h = self.height_at(cx_lon, cy_lat)
        if h is not None:
            return h
        # Fallback: sample the 4 corners of the polygon bbox.
        samples: list[float] = []
        minx, miny, maxx, maxy = polygon.bounds
        for ux, uy in (
            (minx, miny),
            (maxx, miny),
            (minx, maxy),
            (maxx, maxy),
        ):
            lon, lat = to_wgs84.transform(ux, uy)
            v = self.height_at(lon, lat)
            if v is not None:
                samples.append(v)
        if not samples:
            return None
        return float(sum(samples) / len(samples))


class _WorldCoverIndex(_RasterTileIndex):
    """Spatial index of ESA WorldCover v200 land-cover raster tiles.

    Downloads 3°×3° COGs from the anonymous HTTPS virtual-host endpoint
    of the ``esa-worldcover`` S3 bucket and exposes per-point class
    lookups plus an iterator over vegetation cells for gap-fill.

    **License.** CC-BY 4.0.  See ``generator/osm/NOTICE.md``.
    """

    LABEL = "ESA WorldCover"
    BASE_URL = _WORLDCOVER_BASE_URL
    CACHE_DIR = _WORLDCOVER_CACHE_DIR
    ATTRIBUTION = _WORLDCOVER_ATTRIBUTION

    def _tile_filename(self, lat_origin: int, lon_origin: int) -> str:
        coord = _format_tile_coord(lat_origin, lon_origin)
        return f"ESA_WorldCover_10m_2021_v200_{coord}_Map.tif"

    def class_at(self, lon: float, lat: float) -> int | None:
        """Return the uint8 class code at a WGS84 point, or None."""
        value = self.sample_point(lon, lat)
        if value is None:
            return None
        return int(value)

    def iter_class_cells(
        self,
        wanted_classes: tuple[int, ...],
        to_wgs84,
        to_utm,
    ):
        """Yield (utm_x, utm_y, class_code) for every raster cell inside the
        scene bbox whose class is in *wanted_classes*.

        *to_wgs84* converts the scene bbox corners to WGS84 so we can
        clip against each tile's window; *to_utm* converts the resulting
        WGS84 cell centers back to UTM for downstream consumers (the
        vegetation gap-fill scatters trees at these points).

        The iteration is bounded to the scene bbox; we never yield cells
        outside it even if the underlying tile is larger.
        """
        if self._empty:
            return
        import numpy as _np
        import rasterio

        wanted = set(int(c) for c in wanted_classes)
        min_lon, min_lat, max_lon, max_lat = self._bbox
        for (lat_origin, lon_origin), ds in self._datasets.items():
            # Clip the scene bbox against this tile's extent.
            tile_min_lon = lon_origin
            tile_min_lat = lat_origin
            tile_max_lon = lon_origin + 3
            tile_max_lat = lat_origin + 3
            clip_min_lon = max(min_lon, tile_min_lon)
            clip_min_lat = max(min_lat, tile_min_lat)
            clip_max_lon = min(max_lon, tile_max_lon)
            clip_max_lat = min(max_lat, tile_max_lat)
            if clip_min_lon >= clip_max_lon or clip_min_lat >= clip_max_lat:
                continue
            try:
                window = rasterio.windows.from_bounds(
                    clip_min_lon,
                    clip_min_lat,
                    clip_max_lon,
                    clip_max_lat,
                    ds.transform,
                )
                data = ds.read(1, window=window)
                if data.size == 0:
                    continue
                row_off = int(window.row_off)
                col_off = int(window.col_off)
                rows, cols = _np.where(_np.isin(data, list(wanted)))
                for r, c in zip(rows, cols):
                    pix_row = row_off + int(r)
                    pix_col = col_off + int(c)
                    # Pixel center in WGS84 via the dataset transform.
                    lon, lat = rasterio.transform.xy(
                        ds.transform, pix_row, pix_col, offset="center"
                    )
                    if not (min_lon <= lon <= max_lon):
                        continue
                    if not (min_lat <= lat <= max_lat):
                        continue
                    ux, uy = to_utm.transform(lon, lat)
                    yield (float(ux), float(uy), int(data[r, c]))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "%s tile sample failed for (%d,%d): %s",
                    self.LABEL,
                    lat_origin,
                    lon_origin,
                    exc,
                )


# Per-build accounting of which external datasets contributed — written
# into ``osm_metadata.json`` as ``external_data_sources``.  Reset at the
# top of every ``build_scene`` call.
_external_data_used: list[dict] = []


def _cite_external_dataset(attribution: dict) -> None:
    """Emit an INFO log citing an external dataset and record its provenance.

    Called at most once per dataset per build: the first use of
    ``_EthCanopyHeightIndex`` or ``_WorldCoverIndex`` triggers an
    attribution log line plus a deduplicated entry in
    ``_external_data_used``.
    """
    name = attribution.get("name")
    if not name:
        return
    # Deduplicate per build.
    for entry in _external_data_used:
        if entry.get("name") == name:
            return
    _external_data_used.append(dict(attribution))
    logger.info(
        "%s used under %s — %s",
        name,
        attribution.get("license", "unknown license"),
        attribution.get("citation", "(no citation)"),
    )


def _build_default_metadata_config(
    *,
    bbox: list[float] | None,
    center: list[float] | None,
    size_m: float | None,
    width_m: float | None,
    terrain: bool,
    terrain_zoom: int,
    ground_pad_m: float,
    ground_resolution_m: float,
    osm_server: str | None,
    barriers: bool,
    vegetation: bool,
    bridges: bool,
    height_enrichment: bool,
    canopy_heights: bool,
    tree_rows: bool,
    vegetation_gap_fill: bool,
    worldcover_water_check: bool,
    green_vegetation: bool,
    tree_row_spacing_m: float,
    landuse_materials: bool,
) -> dict:
    """Build a fully-populated ``osm_config`` dict for ``osm_metadata.json``.

    Used by the ``build_scene`` and ``build_scene_from_center`` fallback
    paths when no caller-supplied ``metadata_config`` is provided (i.e.
    direct API use, not the YAML/scenario-loader path).  Forwards every
    relevant OSM feature flag through ``normalize_osm_config`` so
    the saved metadata matches what was actually built — without this,
    direct-API calls with non-default flags would write metadata that
    normalizes the flag back to its default and confuse the next
    ``should_reuse_generated_scene`` check.
    """
    raw: dict = {
        "terrain": terrain,
        "terrain_zoom": terrain_zoom,
        "ground_pad_m": ground_pad_m,
        "ground_resolution_m": ground_resolution_m,
        "osm_server": osm_server,
        "barriers": barriers,
        "vegetation": vegetation,
        "bridges": bridges,
        "height_enrichment": height_enrichment,
        "canopy_heights": canopy_heights,
        "tree_rows": tree_rows,
        "vegetation_gap_fill": vegetation_gap_fill,
        "worldcover_water_check": worldcover_water_check,
        "green_vegetation": green_vegetation,
        "tree_row_spacing_m": tree_row_spacing_m,
        "landuse_materials": landuse_materials,
    }
    if bbox is not None:
        raw["bbox"] = bbox
    if center is not None:
        raw["center"] = center
    if size_m is not None:
        raw["size_m"] = size_m
    if width_m is not None:
        raw["width_m"] = width_m
    return normalize_osm_config(raw)


def _has_tagged_height(building):
    for key in ("building:height", "height", "building:levels"):
        try:
            value = float(building.get(key))
            if math.isfinite(value) and value > 0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _building_height(building: dict, polygon: Polygon) -> float:
    """Estimate building height from OSM tags, GlobalML fallback, or random."""
    # 1) Explicit OSM height tag.
    for tag in ("building:height", "height"):
        val = building.get(tag)
        if val is not None:
            try:
                h = float(val)
                if math.isfinite(h) and h > 0:
                    _height_source_counts["osm_height"] = (
                        _height_source_counts.get("osm_height", 0) + 1
                    )
                    return h
            except (TypeError, ValueError):
                pass

    # 2) OSM building:levels.
    levels_str = building.get("building:levels")
    if levels_str is not None:
        try:
            levels = float(levels_str)
            if math.isfinite(levels) and levels > 0:
                _height_source_counts["osm_levels"] = _height_source_counts.get("osm_levels", 0) + 1
                return levels * DEFAULT_FLOOR_HEIGHT_M
        except (TypeError, ValueError):
            pass

    # 3) Microsoft GlobalML height (ML-derived from satellite imagery).
    if _height_index is not None:
        centroid = polygon.centroid
        ext_h = _height_index.query(centroid.x, centroid.y)
        if ext_h is not None:
            _height_source_counts["external"] = _height_source_counts.get("external", 0) + 1
            return ext_h

    # 4) Deterministic random fallback seeded by centroid.
    _height_source_counts["random"] = _height_source_counts.get("random", 0) + 1
    cx = polygon.centroid.x
    cy = polygon.centroid.y
    rng = np.random.RandomState(
        int(abs(cx + cy) * SYNTHETIC_BUILDING_HEIGHT_SEED_SCALE)
        % SYNTHETIC_BUILDING_HEIGHT_SEED_MODULUS
    )
    floors = max(
        SYNTHETIC_BUILDING_MIN_FLOORS,
        min(
            SYNTHETIC_BUILDING_MAX_FLOORS,
            int(
                rng.normal(
                    loc=SYNTHETIC_BUILDING_MEAN_FLOORS,
                    scale=SYNTHETIC_BUILDING_FLOOR_STDDEV,
                )
            ),
        ),
    )
    return floors * DEFAULT_FLOOR_HEIGHT_M


def _building_min_height(building: dict) -> float:
    """Return the min_height (base elevation above ground) from OSM tags.

    The Simple 3D Buildings spec uses ``min_height`` to create elevated volumes.
    For example, an arch lintel with ``min_height=18, height=32`` floats 18 m
    above ground, leaving a void beneath it.
    """
    for tag in ("building:min_height", "min_height"):
        val = building.get(tag)
        if val is not None:
            try:
                h = float(val)
                if math.isfinite(h) and h >= 0:
                    return h
            except (TypeError, ValueError):
                pass

    min_level = building.get("building:min_level")
    if min_level is not None:
        try:
            levels = float(min_level)
            if math.isfinite(levels) and levels > 0:
                return levels * DEFAULT_FLOOR_HEIGHT_M
        except (TypeError, ValueError):
            pass

    return 0.0


# OSM tag → ITU material mapping (first match wins).
_OSM_MATERIAL_TAG_MAP: dict[str, str] = {
    "marble": "marble",
    "stone": "marble",
    "brick": "brick",
    "glass": "glass",
    "metal": "metal",
    "steel": "metal",
    "aluminium": "metal",
    "aluminum": "metal",
    "wood": "wood",
    "timber": "wood",
    "concrete": "concrete",
}

_OSM_BUILDING_TYPE_MAP: dict[str, str] = {
    "industrial": "metal",
    "warehouse": "metal",
    "garage": "metal",
    "hangar": "metal",
    "church": "marble",
    "cathedral": "marble",
    "chapel": "marble",
    "temple": "marble",
    "hotel": "glass",
    "office": "glass",
    "commercial": "glass",
    "retail": "glass",
}

# OSM ``landuse=*`` value → default ITU material for any building that
# falls inside that land-use polygon and has no explicit
# ``building:material`` tag and no overriding ``_OSM_BUILDING_TYPE_MAP``
# match.  Used by the optional ``landuse_materials`` flag.
#
# Rationale: 99 % of OSM buildings have no ``building:material`` tag,
# so without this lookup nearly every untagged building defaults to
# concrete, which is unrealistic.  OSM does have rich ``landuse=*``
# polygons covering whole districts; matching a building's centroid
# against them gives a much better statistical default.
_OSM_LANDUSE_MATERIAL_MAP: dict[str, str] = {
    # Industrial / utility districts → metal-clad sheds and warehouses.
    "industrial": "metal",
    "warehouse": "metal",
    "depot": "metal",
    "port": "metal",
    "harbour": "metal",
    "garages": "metal",
    "railway": "metal",
    # Office / retail / commercial districts → glass curtain walls.
    "commercial": "glass",
    "retail": "glass",
    # Schools, universities, hospitals → masonry / brick.
    "education": "brick",
    "institutional": "brick",
    "civic": "brick",
    # Residential districts → brick (houses + low-rise apartments).
    "residential": "brick",
    # Religious complexes → marble.
    "religious": "marble",
    "cemetery": "marble",
    # Construction sites → bare concrete.
    "construction": "concrete",
    # quarry / military / farmland / forest / meadow / etc. → leave
    # unmapped so the existing default chain takes over.
}


class _LandUseMaterialIndex:
    """Spatial index of OSM ``landuse=*`` polygons → default material.

    Built once per scene from a GeoDataFrame of landuse features and used
    to give untagged buildings a smarter material default than the
    blanket ``concrete`` fallback.  Lookups are O(log N) via a shapely
    STRtree; we keep a parallel ``materials`` list aligned with the
    inserted polygons because STRtree's ``geometries`` only round-trips
    indices, not arbitrary metadata.
    """

    def __init__(self, landuse_records: list[dict] | None):
        self._tree = None
        self._polys: list = []
        self._materials: list[str] = []
        self._landuse_tags: list[str] = []
        if not landuse_records:
            return

        from shapely.strtree import STRtree

        for rec in landuse_records:
            geom = rec.get("geometry")
            if geom is None:
                continue
            try:
                shapely_geom = shape(geom) if not hasattr(geom, "geom_type") else geom
            except (AttributeError, TypeError, ValueError):
                continue
            if shapely_geom.is_empty or not shapely_geom.is_valid:
                continue
            tag_val = rec.get("landuse")
            if not isinstance(tag_val, str):
                continue
            landuse_tag = tag_val.strip().lower()
            material = _OSM_LANDUSE_MATERIAL_MAP.get(landuse_tag)
            if material is None:
                continue
            for sub in _iter_polygon_parts(shapely_geom):
                if sub.is_empty or sub.area < 1.0:
                    continue
                self._polys.append(sub)
                self._materials.append(material)
                self._landuse_tags.append(landuse_tag)

        if self._polys:
            self._tree = STRtree(self._polys)

    def is_available(self) -> bool:
        return self._tree is not None

    def material_for_point(self, x: float, y: float) -> str | None:
        """Return the landuse-derived material at a UTM point, or None.

        When multiple landuse polygons contain the same point (e.g. a small
        ``landuse=religious`` church parcel nested inside a large
        ``landuse=commercial`` district), the **smallest-area** polygon wins.
        This gives deterministic results independent of STRtree iteration
        order and favors the more-specific classification. Equal-area
        ties fall back to the normalized ``landuse`` tag so overlapping
        same-size parcels still return a stable material.
        """
        if self._tree is None:
            return None
        from shapely.geometry import Point as _Point

        pt = _Point(x, y)
        candidate_idxs = self._tree.query(pt)
        # shapely 2.x STRtree.query returns a numpy int array of indices;
        # 1.x returned Geometry objects.  Handle both.
        try:
            iter_idxs = list(candidate_idxs)
        except TypeError:
            iter_idxs = []
        best_idx: int | None = None
        best_key: tuple[float, str] | None = None
        for raw in iter_idxs:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                # shapely 1.x: raw is a Geometry — find it by identity.
                try:
                    idx = self._polys.index(raw)
                except ValueError:
                    continue
            if not (0 <= idx < len(self._polys)):
                continue
            poly = self._polys[idx]
            if not poly.contains(pt):
                continue
            candidate_key = (poly.area, self._landuse_tags[idx])
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_idx = idx
        if best_idx is None:
            return None
        return self._materials[best_idx]


def _building_material(
    building: dict,
    polygon: Polygon | None = None,
    landuse_index: "_LandUseMaterialIndex | None" = None,
) -> str:
    """Determine ITU wall material from OSM tags, with concrete fallback.

    Lookup order:

    1. Explicit ``building:material`` tag — always wins.
    2. ``building=*`` type map (industrial/office/church/etc.).
    3. **Land-use context** — if a ``_LandUseMaterialIndex`` is supplied
       and the building's centroid falls inside a known landuse polygon,
       use the landuse-derived material. Optional, gated by the
       ``landuse_materials`` flag (default ``True``).
    4. ``DEFAULT_BUILDING_MATERIAL`` (concrete).
    """
    # 1) Explicit building:material tag (highest priority).
    mat_tag = building.get("building:material")
    if isinstance(mat_tag, str):
        key = mat_tag.strip().lower()
        if key in _OSM_MATERIAL_TAG_MAP:
            return _OSM_MATERIAL_TAG_MAP[key]

    # 2) Building type.
    btype = building.get("building")
    if isinstance(btype, str):
        key = btype.strip().lower()
        if key in _OSM_BUILDING_TYPE_MAP:
            return _OSM_BUILDING_TYPE_MAP[key]

    # 3) Land-use context (optional).  The polygon arg is in UTM so the
    # centroid lookup matches the index's coordinate system.
    if landuse_index is not None and landuse_index.is_available() and polygon is not None:
        try:
            centroid = polygon.centroid
            material = landuse_index.material_for_point(float(centroid.x), float(centroid.y))
            if material is not None:
                return material
        except Exception:
            pass

    return DEFAULT_BUILDING_MATERIAL


def _roof_material(building: dict) -> str | None:
    """Return the roof-specific ITU material, or None to use wall material."""
    roof_tag = building.get("roof:material")
    if isinstance(roof_tag, str):
        key = roof_tag.strip().lower()
        if key in _OSM_MATERIAL_TAG_MAP:
            return _OSM_MATERIAL_TAG_MAP[key]
    return None


def _triangulate_polygon(
    outer_xy: list[tuple[float, float]],
    holes_xy: list[list[tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Constrained Delaunay triangulation of a polygon with holes."""

    def _edge_idxs(nv: int) -> np.ndarray:
        i = np.append(np.arange(nv), 0)
        return np.stack([i[:-1], i[1:]], axis=1)

    nv = 0
    verts_list, edges_list = [], []
    for loop in (outer_xy, *holes_xy):
        arr = np.array(loop)
        verts_list.append(arr)
        edges_list.append(nv + _edge_idxs(len(arr)))
        nv += len(arr)

    verts = np.concatenate(verts_list)
    edges = np.concatenate(edges_list)

    # De-duplicate vertices.
    verts, inverse = np.unique(verts, axis=0, return_inverse=True)
    edges = inverse[edges]

    tri_input = {"vertices": verts, "segments": edges}
    if holes_xy:
        hole_points = np.array([np.mean(h, axis=0) for h in holes_xy])
        tri_input["holes"] = hole_points

    result = triangulate(tri_input, opts="p")
    return result["vertices"], result["triangles"]


def _iter_polygon_parts(geometry_obj) -> list[Polygon]:
    """Return Polygon parts from a Polygon/MultiPolygon-like geometry."""
    if geometry_obj is None:
        return []
    try:
        geom = shape(geometry_obj)
    except (AttributeError, TypeError, ValueError):
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


def _clip_polygon_to_bbox(geometry_obj, bbox_polygon) -> list[Polygon]:
    """Return Polygon parts of *geometry_obj* clipped against *bbox_polygon*.

    Used to clip large OSM features (rivers, forests, parks) that
    Overpass returns as whole geometries even when the scene bbox only
    intersects a corner of them.  Resulting pieces may be one or more
    Polygons; GeometryCollection parts that aren't polygons are
    dropped.  Returns an empty list when the geometry is None, invalid,
    or does not overlap the bbox.
    """
    if geometry_obj is None:
        return []
    try:
        geom = shape(geometry_obj)
    except (AttributeError, TypeError, ValueError):
        return []
    try:
        clipped = geom.intersection(bbox_polygon)
    except Exception:
        # Fall back to the un-clipped parts rather than drop the feature.
        return _iter_polygon_parts(geom)
    if clipped.is_empty:
        return []
    if clipped.geom_type == "Polygon":
        return [clipped]
    if clipped.geom_type == "MultiPolygon":
        return list(clipped.geoms)
    if clipped.geom_type == "GeometryCollection":
        return [g for g in clipped.geoms if g.geom_type == "Polygon"]
    return []


def _clip_linestring_to_bbox(line, bbox_polygon) -> list:
    """Return LineString parts of *line* clipped against *bbox_polygon*.

    Used to clip long OSM street ways that run through the bbox but
    extend far beyond it.  Returns an empty list when the line is None
    or the intersection does not yield any LineString component.
    """
    if line is None or line.is_empty:
        return []
    try:
        clipped = line.intersection(bbox_polygon)
    except Exception:
        return [line]
    if clipped.is_empty:
        return []
    if clipped.geom_type == "LineString":
        return [clipped]
    if clipped.geom_type == "MultiLineString":
        return list(clipped.geoms)
    if clipped.geom_type == "GeometryCollection":
        return [g for g in clipped.geoms if g.geom_type == "LineString"]
    return []


def _log_scene_complexity(
    *,
    output_dir: str,
    mesh_files: list[str],
    mat_groups: dict[str, dict],
    street_mat_groups: dict[str, dict],
    buildings_count: int,
    graph,
    bbox: tuple[float, float, float, float],
) -> None:
    """Log a compact summary of the generated scene's size and complexity.

    Prints one INFO line with raw counts (vertices, faces, buildings, graph
    edges, on-disk PLY size) plus a WARN if any single mesh exceeds 5M verts.
    Intentionally does not estimate ray-tracing cost — that needs telemetry
    from real RT runs, not a linear extrapolation.
    """
    total_verts = 0
    total_faces = 0
    max_mesh_verts = 0
    max_mesh_label = ""
    for material, grp in mat_groups.items():
        for v in grp.get("vertices") or []:
            n_v = int(v.shape[0]) if hasattr(v, "shape") else len(v)
            total_verts += n_v
            if n_v > max_mesh_verts:
                max_mesh_verts = n_v
                max_mesh_label = f"buildings_{material}"
        for f in grp.get("faces") or []:
            n_f = int(f.shape[0]) if hasattr(f, "shape") else len(f)
            total_faces += n_f
    for material, grp in street_mat_groups.items():
        for v in grp.get("vertices") or []:
            n_v = int(v.shape[0]) if hasattr(v, "shape") else len(v)
            total_verts += n_v
            if n_v > max_mesh_verts:
                max_mesh_verts = n_v
                max_mesh_label = f"streets_{material}"
        for f in grp.get("faces") or []:
            n_f = int(f.shape[0]) if hasattr(f, "shape") else len(f)
            total_faces += n_f

    on_disk_bytes = 0
    for rel in mesh_files:
        path = os.path.join(output_dir, rel)
        try:
            on_disk_bytes += os.path.getsize(path)
        except OSError:
            pass

    # Rough bbox size in meters: 111 km per degree of latitude, cos(lat) for lon.
    min_lat, min_lon, max_lat, max_lon = bbox
    lat_center = (min_lat + max_lat) / 2
    height_m = (max_lat - min_lat) * 111_000
    width_m = (max_lon - min_lon) * 111_000 * math.cos(math.radians(lat_center))

    n_nodes = graph.number_of_nodes() if graph is not None else 0
    n_edges = graph.number_of_edges() if graph is not None else 0

    logger.info(
        "Scene complexity: %d buildings (%.2fM verts, %.2fM faces), "
        "%d graph nodes, %d street edges, "
        "bbox %.0f x %.0f m, %d mesh files, %.1f MB on disk",
        buildings_count,
        total_verts / 1e6,
        total_faces / 1e6,
        n_nodes,
        n_edges,
        width_m,
        height_m,
        len(mesh_files),
        on_disk_bytes / (1024 * 1024),
    )

    if max_mesh_verts > 5_000_000:
        logger.warning(
            "Large single mesh detected: %s has %.1fM vertices — scene may be "
            "too big for headless RT runs. Consider reducing --size.",
            max_mesh_label,
            max_mesh_verts / 1e6,
        )


def _configure_osmnx_overpass_timeout(seconds: int) -> None:
    """Set the Overpass query timeout header via the current OSMnx API.

    OSMnx 2.x builds the ``[timeout:N]`` header from
    ``settings.requests_timeout`` (see
    ``osmnx/_overpass.py:_make_overpass_settings``).  The legacy
    ``settings.timeout`` attribute from older OSMnx releases has been
    removed from the settings module entirely in 2.1.0, so we don't
    touch it — the previous compatibility shim was dead code.
    """
    ox.settings.requests_timeout = seconds


# OSM cache directory default — overridable via ORCHAV_CACHE_DIR.  The
# final directory is always ``<root>/osm/``.  Mirrors ``_GLOBALML_CACHE_DIR``.
_DEFAULT_ORCHAV_CACHE_ROOT = Path("~/.orchav/cache").expanduser()


def _default_osm_cache_dir() -> Path:
    """Return the directory used to cache raw OSM query results.

    Precedence: ``ORCHAV_CACHE_DIR`` env var > ``~/.orchav/cache``.
    """
    root = os.environ.get("ORCHAV_CACHE_DIR")
    base = Path(root).expanduser() if root else _DEFAULT_ORCHAV_CACHE_ROOT
    return base / "osm"


def _osm_refresh_from_env() -> bool:
    """Return True when the caller set ``OSM_REFRESH_CACHE=1`` in the env."""
    return os.environ.get("OSM_REFRESH_CACHE", "").strip().lower() in ("1", "true", "yes")


def _normalize_tags_for_hash(tags: dict | None) -> object:
    """Return a JSON-stable form of an OSM tag dict for hashing.

    Keys are sorted; list values are sorted; ``True`` is kept as-is.  OSM
    tag dicts are small enough that this runs in microseconds.
    """
    if tags is None:
        return None
    out: dict = {}
    for k in sorted(tags.keys()):
        v = tags[k]
        if isinstance(v, (list, tuple, set)):
            out[k] = sorted([str(x) for x in v])
        else:
            out[k] = v
    return out


def _osm_cache_filename(label: str, bbox: tuple, tags: dict | None, ext: str) -> str:
    """Build a deterministic cache filename for an OSM query.

    Hash inputs are the bounding box (rounded to 6 decimal places so floating
    point noise doesn't churn the cache), the normalized tag dict, and the
    installed osmnx version.  An osmnx upgrade therefore silently invalidates
    stale entries instead of deserializing them into a different schema.
    """
    payload = {
        "bbox": [round(float(v), 6) for v in bbox],
        "tags": _normalize_tags_for_hash(tags),
        "ox_version": getattr(ox, "__version__", "unknown"),
    }
    context = active_acquisition()
    if context and context.endpoint != "https://overpass-api.de/api":
        payload["endpoint"] = context.endpoint
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{label}_{digest}{ext}"


# Serializer contract: (to_disk, from_disk, ext) where to_disk(result, path)
# writes the result of a fetch_fn and from_disk(path) returns the same shape.
OsmSerializer = namedtuple("OsmSerializer", ["to_disk", "from_disk", "ext"])


def _gdf_to_geojson(gdf, path: Path) -> None:
    """Serialize a GeoDataFrame to GeoJSON on disk (WGS84)."""
    gdf.to_file(str(path), driver="GeoJSON", engine="pyogrio")


def _geojson_to_gdf(path: Path):
    """Deserialize a GeoJSON file into a GeoDataFrame."""
    import geopandas as gpd

    return gpd.read_file(str(path), engine="pyogrio")


def _graph_to_graphml(graph, path: Path) -> None:
    """Serialize a raw (unprojected) street network graph to GraphML."""
    ox.save_graphml(graph, str(path))


def _graphml_to_graph(path: Path):
    """Deserialize a GraphML file into the same nx.MultiDiGraph osmnx returns."""
    return ox.load_graphml(str(path))


_GDF_SERIALIZER = OsmSerializer(_gdf_to_geojson, _geojson_to_gdf, ".geojson")
_GRAPH_SERIALIZER = OsmSerializer(_graph_to_graphml, _graphml_to_graph, ".graphml")


def _cached_osm_query(
    label: str,
    *,
    bbox: tuple,
    tags: dict | None,
    fetch_fn,
    serializer: OsmSerializer,
    optional: bool,
    no_data_value,
    cache_dir: Path | None,
    refresh: bool,
    stats: dict | None = None,
):
    """Cache only successful data/empty results; leave failed layers retryable.

    Legacy empty markers are ambiguous and are revalidated once. Positive
    payloads keep their existing filenames and remain reusable.
    """
    started = time.monotonic()
    context = active_acquisition()

    def report(outcome, cache):
        if context:
            context.record(
                label,
                outcome,
                cache=cache,
                elapsed_s=time.monotonic() - started,
                endpoint=context.endpoint,
            )

    cache_file = None
    empty_marker = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / _osm_cache_filename(label, bbox, tags, serializer.ext)
        empty_marker = cache_file.with_suffix(cache_file.suffix + ".empty")
        retry_marker = cache_file.with_suffix(cache_file.suffix + ".retry")
        if not refresh and not retry_marker.exists():
            if _confirmed_empty(empty_marker):
                if stats is not None:
                    stats["hits"] = stats.get("hits", 0) + 1
                report("empty", "hit")
                return no_data_value
            if cache_file.exists():
                try:
                    result = serializer.from_disk(cache_file)
                    if stats is not None:
                        stats["hits"] = stats.get("hits", 0) + 1
                    report("data", "hit")
                    return result
                except Exception as exc:
                    logger.warning("OSM cache read failed for %s (%s); re-fetching", label, exc)

    # Preserve the old payload on interrupted refresh, but don't let it
    # suppress the next recovery attempt.
    if cache_file is not None and (cache_file.exists() or empty_marker.exists()):
        retry_marker.write_text("revalidation pending", encoding="utf-8")
    result = _run_osm_query(label, fetch_fn, optional=optional, no_data_value=no_data_value)
    if stats is not None:
        stats["misses"] = stats.get("misses", 0) + 1
    if isinstance(result, _QueryFailure):
        if stats is not None:
            stats["failures"] = stats.get("failures", 0) + 1
        if context:
            context.record(
                label,
                "failed",
                cache="miss",
                error=str(result.error),
                elapsed_s=time.monotonic() - started,
                endpoint=context.endpoint,
            )
        return no_data_value

    empty = result is None or result is no_data_value
    if cache_file is not None:
        temporary = None
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=cache_file.parent, suffix=".tmp")
            os.close(fd)
            temporary = Path(name)
            if empty:
                temporary.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "outcome": "empty",
                            "query": cache_file.name,
                            "endpoint": context.endpoint if context else None,
                        }
                    ),
                    encoding="utf-8",
                )
                os.replace(temporary, empty_marker)
                cache_file.unlink(missing_ok=True)
            else:
                serializer.to_disk(result, temporary)
                os.replace(temporary, cache_file)
                empty_marker.unlink(missing_ok=True)
            retry_marker.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("OSM cache write failed for %s (%s)", label, exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    report("empty" if empty else "data", "miss")
    return result


def _confirmed_empty(path):
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        return (
            isinstance(marker, dict)
            and marker.get("schema_version") == 2
            and marker.get("outcome") == "empty"
            and marker.get("query") == path.name.removesuffix(".empty")
        )
    except (OSError, ValueError):
        return False


class _QueryFailure:
    def __init__(self, error):
        self.error = error


def _run_osm_query(label: str, fetch_fn, *, optional: bool, no_data_value):
    """Distinguish absence from acquisition failure without nested transport retries."""
    logger.info("Querying OSM for %s...", label)
    for attempt in range(1, OSM_QUERY_MAX_ATTEMPTS + 1):
        try:
            return fetch_fn()
        except _OSM_QUERY_ERRORS as exc:
            # The bounded adapter validates raw JSON before OSMnx can call an
            # error/partial response 'no features'. Do not cache other errors
            # that happen to share OSMnx's InsufficientResponseError type.
            if (
                _OSM_NO_DATA_ERRORS
                and isinstance(exc, _OSM_NO_DATA_ERRORS)
                and str(exc).startswith(
                    (
                        "No matching features",
                        "No data elements in server response",
                        "No graph nodes within the requested polygon",
                    )
                )
            ):
                return no_data_value
            if attempt < OSM_QUERY_MAX_ATTEMPTS and not isinstance(exc, AcquisitionError):
                logger.warning(
                    "%s query failed (%s), retrying %d/%d",
                    label,
                    exc,
                    attempt,
                    OSM_QUERY_MAX_ATTEMPTS,
                )
                continue
            if optional:
                logger.warning(
                    "%s unavailable (%s); scene is partial and this layer will be retried",
                    label,
                    exc,
                )
                return _QueryFailure(exc)
            raise RuntimeError(f"{label} query failed: {exc}") from exc


_FEATURE_TAGS = {
    "buildings": {"building": True},
    "building_parts": {"building:part": True},
    "water": {"water": True, "landuse": ["reservoir", "basin", "salt_pond"], "natural": ["water"]},
    "landuse_materials": {
        "landuse": [
            "industrial",
            "warehouse",
            "depot",
            "port",
            "harbour",
            "garages",
            "railway",
            "commercial",
            "retail",
            "education",
            "institutional",
            "civic",
            "residential",
            "religious",
            "cemetery",
            "construction",
        ]
    },
    "barriers": {"barrier": ["wall", "fence", "retaining_wall", "hedge", "city_wall"]},
    "vegetation": {
        "natural": ["tree", "tree_row", "wood"],
        "landuse": ["forest"],
        "leisure": ["park"],
    },
}


class _FeatureBatch:
    """Fetch only missing feature layers together; retain independent disk entries."""

    def __init__(self, bbox, cache_dir, refresh, labels):
        self.bbox = bbox
        self.missing = {}
        for label in labels:
            tags = _FEATURE_TAGS[label]
            path = Path(cache_dir) / _osm_cache_filename(label, bbox, tags, ".geojson")
            if (
                refresh
                or path.with_suffix(".geojson.retry").exists()
                or not (path.exists() or _confirmed_empty(path.with_suffix(".geojson.empty")))
            ):
                self.missing[label] = tags
        self.pending = True
        self.values = {}

    def get(self, label, tags):
        if self.pending:
            self.pending = False
            if len(self.missing) > 1:
                logger.info("OSM combined feature acquisition: %s", ", ".join(self.missing))
                try:
                    data = ox.features.features_from_bbox(
                        bbox=self.bbox, tags=merge_tags(self.missing.values())
                    )
                    self.values = {
                        key: select_features(data, value) for key, value in self.missing.items()
                    }
                except _OSM_QUERY_ERRORS as exc:
                    if (
                        _OSM_NO_DATA_ERRORS
                        and isinstance(exc, _OSM_NO_DATA_ERRORS)
                        and str(exc).startswith("No matching features")
                    ):
                        self.values = dict.fromkeys(self.missing)
                    else:
                        logger.warning(
                            "Combined OSM acquisition failed (%s); retrying layers separately within remaining budget",
                            exc,
                        )
        if label in self.values:
            return self.values[label]
        return ox.features.features_from_bbox(bbox=self.bbox, tags=tags)


def _write_acquisition_attempt(path, context, config, outcome, error=None):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
        os.close(fd)
        temporary = Path(name)
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "outcome": outcome,
                        "error": error,
                        "osm_config": config,
                        "layers": context.layers,
                        "requests": context.requests,
                        "elapsed_s": time.monotonic() - context.started,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not save OSM acquisition attempt: %s", exc)


def _managed_build(function):
    @wraps(function)
    def wrapper(bbox, output_dir, **kwargs):
        timeout = kwargs.pop("osm_acquisition_timeout", 180.0)
        cancel = kwargs.pop("cancel", None)
        progress = kwargs.pop("progress", None)
        with acquisition_scope(
            endpoint=kwargs.get("osm_server"),
            timeout=timeout,
            request_timeout=min(float(kwargs.get("osm_overpass_timeout", 45)), float(timeout)),
            cancel=cancel,
            progress=progress,
        ) as context:
            attempt_path = Path(output_dir) / "osm_acquisition_attempt.json"
            try:
                with staged_scene(output_dir) as stage:
                    function(bbox, str(stage), **kwargs)
                    if cancel and cancel():
                        from .acquisition import AcquisitionCancelled

                        raise AcquisitionCancelled("OSM scene build cancelled")
                    logger.info(
                        "OSM preparation completed in %.3fs (%s)",
                        time.monotonic() - context.started,
                        (
                            "partial"
                            if any(v["outcome"] == "failed" for v in context.layers.values())
                            else "complete"
                        ),
                    )
            except BaseException as exc:
                _write_acquisition_attempt(
                    attempt_path, context, kwargs.get("metadata_config"), "failed", str(exc)
                )
                raise
            else:
                _write_acquisition_attempt(
                    attempt_path,
                    context,
                    kwargs.get("metadata_config"),
                    (
                        "partial"
                        if any(v["outcome"] == "failed" for v in context.layers.values())
                        else "complete"
                    ),
                )
        return str(Path(output_dir) / "scene.xml")

    return wrapper


def _edge_local_polyline(
    edge_data: dict,
    u_data: dict,
    v_data: dict,
    origin_x: float,
    origin_y: float,
) -> list[tuple[float, float]]:
    """Return a local XY polyline for a projected graph edge."""
    geometry_local = edge_data.get("geometry_local")
    if isinstance(geometry_local, str):
        try:
            coords = json.loads(geometry_local)
            return [(float(x), float(y)) for x, y in coords]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    geometry = edge_data.get("geometry")
    if geometry is not None and hasattr(geometry, "coords"):
        return [(float(x) - origin_x, float(y) - origin_y) for x, y in geometry.coords]

    return [
        (float(u_data["x"]) - origin_x, float(u_data["y"]) - origin_y),
        (float(v_data["x"]) - origin_x, float(v_data["y"]) - origin_y),
    ]


def _sample_local_terrain_heights(
    heightmap,
    origin: np.ndarray,
    origin_z: float,
    xy_local: np.ndarray,
) -> np.ndarray:
    """Sample terrain height at local XY positions and return local Z values."""
    if heightmap is None or xy_local.size == 0:
        return np.zeros(len(xy_local), dtype=np.float32)

    x_abs = xy_local[:, 0] + origin[0]
    y_abs = xy_local[:, 1] + origin[1]
    heights = heightmap.heights_from_utm_batch(x_abs, y_abs) - origin_z
    return np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _write_mitsuba_xml(
    scene_el: ET.Element,
    filepath: str,
) -> None:
    """Write a Mitsuba XML scene element to file with pretty formatting."""
    raw = ET.tostring(scene_el, encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="    ")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(pretty)


# -- Roof shape generation ----------------------------------------------------

# Default roof height when roof:height is missing but roof:shape is present.
# Bound road chords when draping them over curved terrain.
TERRAIN_STREET_MAX_SEGMENT_M = 5.0

DEFAULT_ROOF_HEIGHT_M = 4.0

# Minimum footprint dimension for straight skeleton (avoid degenerate skeletons).
MIN_SKELETON_DIM_M = 2.0


try:
    from shapely_polyskel import StraightSkeleton as _StraightSkeleton

    _HAS_POLYSKEL = True
except ImportError:
    _HAS_POLYSKEL = False


def _parse_roof_tags(building: dict) -> tuple[str, float]:
    """Extract roof shape and roof height from OSM tags.

    Returns (roof_shape, roof_height).  ``roof_shape`` is one of the S3DB
    values (``"flat"``, ``"gabled"``, ``"hipped"``, ``"pyramidal"``, etc.)
    or ``"flat"`` when no tag is present.  ``roof_height`` is in meters.
    """
    roof_shape = "flat"
    raw = building.get("roof:shape")
    if isinstance(raw, str) and raw.strip():
        roof_shape = raw.strip().lower()

    roof_height = DEFAULT_ROOF_HEIGHT_M
    for tag in ("roof:height",):
        val = building.get(tag)
        if val is not None:
            try:
                h = float(val)
                if math.isfinite(h) and h > 0:
                    roof_height = h
                    break
            except (TypeError, ValueError):
                pass

    # Derive roof height from roof:angle if available and roof:height wasn't.
    if building.get("roof:height") is None:
        angle_str = building.get("roof:angle")
        if angle_str is not None:
            try:
                angle_deg = float(angle_str)
                if 0 < angle_deg < 89:
                    # Approximate: roof_height = half_min_span * tan(angle)
                    # We don't know span here, so keep the default.
                    pass
            except (TypeError, ValueError):
                pass

    return roof_shape, roof_height


def _generate_roof_mesh(
    outer_xy: list[tuple[float, float]],
    roof_shape: str,
    roof_height: float,
    wall_top_z: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Generate a 3D roof mesh for the given footprint and shape.

    Parameters
    ----------
    outer_xy : list of (x, y)
        Footprint vertices in local coordinates (no holes).
    roof_shape : str
        S3DB roof shape value (``"hipped"``, ``"gabled"``, ``"pyramidal"``, etc.).
    roof_height : float
        Height of the roof above wall top, in meters.
    wall_top_z : float
        Z coordinate of the wall top (base of the roof).

    Returns
    -------
    (vertices_3d, faces) or None if the roof could not be generated.
    """
    if roof_height <= 0 or len(outer_xy) < 3:
        return None

    poly = Polygon(outer_xy)
    if not poly.is_valid or poly.area < 1.0:
        return None

    # Pyramidal family: cone is a pyramidal apex on a (near-)circular base.
    if roof_shape in ("pyramidal", "cone"):
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    # Gabled family: pitched / saltbox / lean_to all approximate as a
    # two-slope ridge along the longest bbox axis.  Aliases land here.
    if roof_shape in ("gabled", "pitched", "saltbox", "double_saltbox", "lean_to"):
        return _roof_gabled(outer_xy, roof_height, wall_top_z, poly)

    # Hipped family: half-hipped, mansard, side_hipped, hipped-and-gabled all
    # share the same skeleton-based mesh as plain hipped.
    if roof_shape in (
        "hipped",
        "half-hipped",
        "mansard",
        "side_hipped",
        "hipped-and-gabled",
    ):
        return _roof_hipped(outer_xy, roof_height, wall_top_z, poly)

    # Dome family: dome / onion / round → hemispherical cap.
    if roof_shape in ("dome", "onion", "round"):
        return _roof_dome(outer_xy, roof_height, wall_top_z, poly)

    # Skillion (single-slope shed) — most common shape we don't otherwise
    # model, ~2.79 % of all OSM roof:shape tags globally.
    if roof_shape == "skillion":
        return _roof_skillion(outer_xy, roof_height, wall_top_z, poly)

    # Unknown shape — skip (keep flat cap from extrusion).  The
    # SUPPORTED_ROOF_SHAPES guard in _extrude_polygon_to_group ensures
    # this path doesn't leave a vertical hole at the top.
    return None


# Roof shapes ``_generate_roof_mesh`` knows how to build.  Used by
# ``_extrude_polygon_to_group`` to decide whether shrinking the walls to
# make room for a parametric roof is safe.  Anything not in this set is
# treated as flat so the walls extrude all the way to ``height`` and we
# never leave a vertical hole at the top (the bug the Capitol dome hit
# before the dome handler + this guard existed).
SUPPORTED_ROOF_SHAPES: frozenset[str] = frozenset(
    {
        # Pyramidal family
        "pyramidal",
        "cone",
        # Gabled family
        "gabled",
        "pitched",
        "saltbox",
        "double_saltbox",
        "lean_to",
        # Hipped family
        "hipped",
        "half-hipped",
        "mansard",
        "side_hipped",
        "hipped-and-gabled",
        # Dome family
        "dome",
        "onion",
        "round",
        # Skillion (single-slope shed)
        "skillion",
    }
)


def _roof_pyramidal(
    outer_xy: list[tuple[float, float]],
    roof_height: float,
    wall_top_z: float,
    poly: Polygon,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pyramidal roof: single apex at centroid."""
    n = len(outer_xy)
    cx, cy = poly.centroid.x, poly.centroid.y
    apex_z = wall_top_z + roof_height

    verts = np.array(
        [(x, y, wall_top_z) for x, y in outer_xy] + [(cx, cy, apex_z)],
        dtype=np.float32,
    )
    faces = np.array([[n, i, (i + 1) % n] for i in range(n)], dtype=np.int32)
    return verts, faces


def _roof_dome(
    outer_xy: list[tuple[float, float]],
    roof_height: float,
    wall_top_z: float,
    poly: Polygon,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Hemispherical / onion / round roof.

    Builds a quad-meshed half-ellipsoid whose horizontal radius equals the
    polygon's mean centroid-to-vertex distance and whose vertical radius
    equals *roof_height*.  The base of the dome connects to the wall top
    via a fan of triangles to the polygon vertices, so there is no
    visible gap when the footprint isn't perfectly circular.

    For ``roof:shape=onion`` the OSM convention is a bulbous Russian-style
    cap; we approximate it as a hemisphere here because the visual
    distinction matters far less than closing the hole at the top of the
    Capitol.
    """
    n = len(outer_xy)
    if n < 3:
        return None

    cx, cy = poly.centroid.x, poly.centroid.y
    # Best-fit horizontal radius: mean distance from centroid to the
    # polygon vertices.  Works for circular and near-circular footprints
    # (rotunda, baptistery, mausoleum, etc.); for very elongated shapes
    # the dome will look more like a tall cap, which is acceptable.
    radius = float(sum(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in outer_xy) / n)
    if radius < 0.5:
        return None

    n_meridians = 16
    n_parallels = 8
    apex_z = wall_top_z + roof_height

    verts: list[tuple[float, float, float]] = [(x, y, wall_top_z) for x, y in outer_xy]

    # Generate hemisphere vertices in (n_parallels + 1) rings, going from
    # the dome base (parallel 0) up to the apex (parallel n_parallels).
    # The ring at parallel == 0 lives slightly above wall_top_z so the
    # dome doesn't intersect the wall cap; the apex is a single point.
    ring_starts: list[int] = []
    for i in range(1, n_parallels + 1):
        # phi = 0 at base (ring 1 is just above wall top), pi/2 at apex.
        phi = (math.pi / 2) * i / n_parallels
        z = wall_top_z + roof_height * math.sin(phi)
        r = radius * math.cos(phi)
        ring_starts.append(len(verts))
        for j in range(n_meridians):
            theta = 2 * math.pi * j / n_meridians
            verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta), z))
    # The very top of the apex collapses to a single point; the last
    # ring already has zero radius (cos(pi/2) = 0) so all its vertices
    # coincide.  Leave it as-is — the meshing is robust to degenerate
    # quads at the apex.

    faces: list[list[int]] = []

    # Connect the wall-top polygon ring to the first hemisphere ring as
    # a fan: for each polygon edge, find the meridian segment that's
    # closest in angle around the centroid and triangulate.
    def _theta_of(x: float, y: float) -> float:
        return math.atan2(y - cy, x - cx)

    poly_thetas = [_theta_of(x, y) for x, y in outer_xy]

    def _meridian_index_for_theta(t: float) -> int:
        # Map (-pi, pi] to [0, n_meridians)
        return int(((t + 2 * math.pi) % (2 * math.pi)) / (2 * math.pi) * n_meridians) % n_meridians

    for i in range(n):
        v0 = i
        v1 = (i + 1) % n
        m0 = _meridian_index_for_theta(poly_thetas[i])
        m1 = _meridian_index_for_theta(poly_thetas[v1])
        # First ring of hemisphere vertices.
        first_ring_offset = ring_starts[0]
        h0 = first_ring_offset + m0
        h1 = first_ring_offset + m1
        # Two triangles forming a quad bridge between the polygon edge
        # (v0, v1) and the hemisphere edge (h0, h1).  Degenerate cases
        # (m0 == m1) collapse to a single triangle.
        faces.append([v0, v1, h1])
        if m0 != m1:
            faces.append([v0, h1, h0])

    # Quads between successive hemisphere rings.
    for i in range(len(ring_starts) - 1):
        ring_a = ring_starts[i]
        ring_b = ring_starts[i + 1]
        for j in range(n_meridians):
            j2 = (j + 1) % n_meridians
            faces.append([ring_a + j, ring_a + j2, ring_b + j2])
            faces.append([ring_a + j, ring_b + j2, ring_b + j])

    # Final fan to the apex (single point at the centroid + apex_z).
    apex_idx = len(verts)
    verts.append((cx, cy, apex_z))
    last_ring = ring_starts[-1]
    for j in range(n_meridians):
        j2 = (j + 1) % n_meridians
        faces.append([last_ring + j, last_ring + j2, apex_idx])

    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


def _roof_gabled(
    outer_xy: list[tuple[float, float]],
    roof_height: float,
    wall_top_z: float,
    poly: Polygon,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Gabled roof: ridge along longest bounding-box axis."""
    n = len(outer_xy)
    minx, miny, maxx, maxy = poly.bounds
    dx, dy = maxx - minx, maxy - miny
    cx, cy = poly.centroid.x, poly.centroid.y
    ridge_z = wall_top_z + roof_height

    if dx >= dy:
        # Ridge runs east-west.
        r0, r1 = (minx, cy), (maxx, cy)
    else:
        # Ridge runs north-south.
        r0, r1 = (cx, miny), (cx, maxy)

    pts_2d = list(outer_xy) + [r0, r1]
    heights = [wall_top_z] * n + [ridge_z, ridge_z]

    verts_np = np.array(pts_2d)
    edges = np.array([[i, (i + 1) % n] for i in range(n)] + [[n, n + 1]])

    try:
        result = triangulate({"vertices": verts_np, "segments": edges}, opts="p")
    except RuntimeError:
        return None

    tri_verts = result["vertices"]
    z = np.full(len(tri_verts), wall_top_z, dtype=np.float32)
    for i in range(min(len(heights), len(tri_verts))):
        z[i] = heights[i]

    verts_3d = np.column_stack([tri_verts, z]).astype(np.float32)
    return verts_3d, result["triangles"].astype(np.int32)


def _roof_skillion(
    outer_xy: list[tuple[float, float]],
    roof_height: float,
    wall_top_z: float,
    poly: Polygon,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Skillion (single-slope shed) roof.

    A single tilted plane.  The slope runs along the longest bbox axis:
    every vertex on one bbox side stays at ``wall_top_z`` and every
    vertex on the opposite side rises to ``wall_top_z + roof_height``.
    Vertices in between are linearly interpolated by their fractional
    position along that axis.

    With ~2.79 % of all OSM ``roof:shape`` tags this is the single
    most common shape we don't otherwise model — gabled / hipped /
    pyramidal / dome together cover the rest.
    """
    n = len(outer_xy)
    if n < 3:
        return None

    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny
    if dx >= dy:
        axis_low = minx
        span = dx if dx > 1e-9 else 1e-9

        def _t_for(x: float, y: float) -> float:
            return (x - axis_low) / span

    else:
        axis_low = miny
        span = dy if dy > 1e-9 else 1e-9

        def _t_for(x: float, y: float) -> float:
            return (y - axis_low) / span

    pts_2d = np.array(outer_xy, dtype=np.float64)
    edges = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)

    try:
        result = triangulate({"vertices": pts_2d, "segments": edges}, opts="p")
    except (RuntimeError, KeyError, ValueError):
        return None

    tri_verts = result.get("vertices")
    tri_faces = result.get("triangles")
    if tri_verts is None or tri_faces is None or len(tri_verts) == 0 or len(tri_faces) == 0:
        return None

    z = np.array(
        [wall_top_z + roof_height * _t_for(float(x), float(y)) for x, y in tri_verts],
        dtype=np.float32,
    )
    verts_3d = np.column_stack([np.asarray(tri_verts, dtype=np.float32), z])
    return verts_3d, np.asarray(tri_faces, dtype=np.int32)


def _roof_hipped(
    outer_xy: list[tuple[float, float]],
    roof_height: float,
    wall_top_z: float,
    poly: Polygon,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Hipped roof using the straight skeleton algorithm.

    Falls back to pyramidal if the skeleton library is unavailable or the
    polygon is too small.  The polygon is simplified before skeleton
    computation to remove near-collinear vertices from UTM projection that
    can cause the skeleton to hang.
    """
    if not _HAS_POLYSKEL:
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    minx, miny, maxx, maxy = poly.bounds
    if (maxx - minx) < MIN_SKELETON_DIM_M or (maxy - miny) < MIN_SKELETON_DIM_M:
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    # Polygons with holes can't be handled by the skeleton.
    if len(poly.interiors) > 0:
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    # Simplify to remove near-collinear vertices from UTM projection.
    # 0.1m tolerance is invisible at building scale but eliminates the
    # degenerate split events that cause the skeleton to hang.
    skeleton_poly = poly.simplify(0.1, preserve_topology=True)
    if not skeleton_poly.is_valid or skeleton_poly.area < 1.0:
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    def _timeout_handler(signum, frame):
        raise TimeoutError("Straight skeleton computation timed out")

    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)  # 5 second timeout as safety net
        try:
            ss = _StraightSkeleton(skeleton_poly)
            arcs = ss.straight_skeleton
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (RuntimeError, ValueError, TypeError, ZeroDivisionError, TimeoutError):
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    if not arcs:
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    max_h = max(a.height for a in arcs)
    if max_h <= 0:
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)
    h_scale = roof_height / max_h

    # Collect 2D points: exterior corners + skeleton sources.
    # Use the simplified polygon since skeleton arcs reference its vertices.
    ext = list(skeleton_poly.exterior.coords)[:-1]
    n_ext = len(ext)
    heights: dict[tuple[float, float], float] = {}
    pts_2d: list[tuple[float, float]] = []

    for x, y in ext:
        pts_2d.append((x, y))
        heights[(round(x, 6), round(y, 6))] = wall_top_z

    for arc in arcs:
        x, y = arc.source.x, arc.source.y
        key = (round(x, 6), round(y, 6))
        if key not in heights:
            pts_2d.append((x, y))
            heights[key] = wall_top_z + arc.height * h_scale

    # Build constraint edges: exterior + skeleton ridges.
    edges_list = [[i, (i + 1) % n_ext] for i in range(n_ext)]
    pt_index = {(round(x, 6), round(y, 6)): i for i, (x, y) in enumerate(pts_2d)}

    for arc in arcs:
        src_key = (round(arc.source.x, 6), round(arc.source.y, 6))
        src_idx = pt_index.get(src_key)
        if src_idx is None:
            continue
        for sink in arc.sinks:
            sink_key = (round(sink.x, 6), round(sink.y, 6))
            sink_idx = pt_index.get(sink_key)
            if sink_idx is not None and src_idx != sink_idx:
                edge = (min(src_idx, sink_idx), max(src_idx, sink_idx))
                edges_list.append(list(edge))

    # De-duplicate edges.
    seen: set[tuple[int, int]] = set()
    unique_edges = []
    for e in edges_list:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    verts_np = np.array(pts_2d)
    edges_np = np.array(unique_edges)

    try:
        result = triangulate({"vertices": verts_np, "segments": edges_np}, opts="p")
    except RuntimeError:
        return _roof_pyramidal(outer_xy, roof_height, wall_top_z, poly)

    tri_verts = result["vertices"]
    z = np.full(len(tri_verts), wall_top_z, dtype=np.float32)
    for i, (x, y) in enumerate(tri_verts):
        key = (round(float(x), 6), round(float(y), 6))
        z[i] = heights.get(key, wall_top_z)

    verts_3d = np.column_stack([tri_verts, z]).astype(np.float32)
    return verts_3d, result["triangles"].astype(np.int32)


def _append_mesh_to_group(
    verts_3d: np.ndarray,
    faces: np.ndarray,
    material: str,
    mat_groups: dict[str, dict],
) -> None:
    """Append a raw (vertices, faces) mesh to a material group."""
    vt = o3d.core.Tensor(verts_3d.astype(np.float32))
    ft = o3d.core.Tensor(faces.astype(np.int64))
    if material not in mat_groups:
        mat_groups[material] = {"vertices": [], "faces": [], "vcount": 0}
    grp = mat_groups[material]
    grp["vertices"].append(vt)
    grp["faces"].append(ft + grp["vcount"])
    grp["vcount"] += vt.shape[0]


# -- Extrusion ----------------------------------------------------------------


def _drop_in_building_edges(graph, building_polys: list[Polygon]) -> int:
    """Remove edges whose polyline intersects a building footprint.

    Uses the local-coord polyline stored on each edge.  Endpoints sitting
    exactly on a wall do not count — only edges that visibly cross or sit
    inside a building are removed.  Returns the number of pruned edges.
    """
    if not building_polys:
        return 0
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    valid = [p for p in building_polys if p.is_valid and p.area > 1.0]
    if not valid:
        return 0
    union = unary_union(valid)
    prepared = shapely_prep(union)

    to_drop: list[tuple] = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        polyline_str = data.get("geometry_local")
        if not isinstance(polyline_str, str):
            continue
        try:
            coords = json.loads(polyline_str)
        except (TypeError, ValueError):
            continue
        if not coords or len(coords) < 2:
            continue
        try:
            line = LineString([(float(x), float(y)) for x, y in coords])
        except (TypeError, ValueError):
            continue
        if not prepared.intersects(line):
            continue
        # Tolerate edges that only graze a wall: require a non-trivial overlap.
        try:
            inside = line.intersection(union)
        except Exception:
            inside = None
        if inside is None or inside.is_empty:
            continue
        overlap_len = float(getattr(inside, "length", 0.0) or 0.0)
        if overlap_len < 1.0:
            continue
        to_drop.append((u, v, key))

    if to_drop:
        graph.remove_edges_from(to_drop)
        # Drop nodes that became isolated (no incident edges).
        isolated = [n for n in graph.nodes if graph.degree(n) == 0]
        if isolated:
            graph.remove_nodes_from(isolated)
    return len(to_drop)


def _prune_graph_to_bbox(graph, scene_bbox_local) -> int:
    """Drop graph nodes whose local (origin-relative) coordinates lie
    outside *scene_bbox_local*, plus their incident edges.

    Walking the node coordinates rather than the edge polylines keeps
    the routing layer simple: any node a mobility router could ever
    sample is now guaranteed to be inside the visible street mesh.
    Without this, rectangular scenes (e.g. ``scenarios/osm/champs_elysees``
    with a 2.4 × 1.0 km bbox) inherit the full disc returned by
    ``ox.graph_from_point`` and can place mobility devices on roads
    far outside the rendered ground extent.

    Returns the number of nodes removed.
    """
    if scene_bbox_local is None:
        return 0
    minx, miny, maxx, maxy = scene_bbox_local.bounds
    to_drop: list = []
    for node, data in graph.nodes(data=True):
        x_local = data.get("x_local")
        y_local = data.get("y_local")
        if x_local is None or y_local is None:
            continue
        try:
            xf = float(x_local)
            yf = float(y_local)
        except (TypeError, ValueError):
            continue
        if not (minx <= xf <= maxx and miny <= yf <= maxy):
            to_drop.append(node)
    if to_drop:
        graph.remove_nodes_from(to_drop)
    return len(to_drop)


def _polyline_length_xy(points_xy: list[tuple[float, float]]) -> float:
    """Return the Euclidean length of a 2-D polyline."""
    if len(points_xy) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points_xy[:-1], points_xy[1:]):
        total += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    return float(total)


def _sanitize_graph_edges_to_bbox(graph, scene_bbox_local) -> tuple[int, int, int]:
    """Clip or drop graph edges so saved ``geometry_local`` stays bbox-bound.

    After node pruning, a curved edge can still leave the padded scene bbox
    and re-enter it while both endpoint nodes remain inside.  Mobility uses
    ``geometry_local`` directly, so keeping such an edge would still route a
    device through invisible off-scene pavement.

    We therefore keep only edges whose intersection with the bbox is a single
    continuous LineString.  If clipping yields multiple disjoint pieces, the
    edge is dropped rather than inventing a teleport or silently stitching the
    pieces back together.  Surviving edges have ``geometry_local`` and
    ``length`` rewritten to the clipped polyline.

    Returns ``(n_clipped_edges, n_dropped_edges, n_isolated_nodes_removed)``.
    """
    if scene_bbox_local is None:
        return (0, 0, 0)

    edges_to_drop: list[tuple] = []
    n_clipped_edges = 0

    for u_node, v_node, key, data in graph.edges(keys=True, data=True):
        u_data = graph.nodes[u_node]
        v_data = graph.nodes[v_node]
        geometry_local = data.get("geometry_local")
        local_polyline: list[tuple[float, float]] = []
        if isinstance(geometry_local, str):
            try:
                coords = json.loads(geometry_local)
                local_polyline = [(float(x), float(y)) for x, y in coords]
            except (TypeError, ValueError, json.JSONDecodeError):
                local_polyline = []
        elif isinstance(geometry_local, list):
            try:
                local_polyline = [(float(x), float(y)) for x, y in geometry_local]
            except (TypeError, ValueError):
                local_polyline = []

        if len(local_polyline) < 2:
            x0 = u_data.get("x_local")
            y0 = u_data.get("y_local")
            x1 = v_data.get("x_local")
            y1 = v_data.get("y_local")
            try:
                local_polyline = [
                    (float(x0), float(y0)),
                    (float(x1), float(y1)),
                ]
            except (TypeError, ValueError):
                local_polyline = []

        if len(local_polyline) < 2:
            edges_to_drop.append((u_node, v_node, key))
            continue

        try:
            edge_line = _LineString(local_polyline)
        except Exception:
            edges_to_drop.append((u_node, v_node, key))
            continue

        clipped_lines = _clip_linestring_to_bbox(edge_line, scene_bbox_local)
        if len(clipped_lines) != 1:
            edges_to_drop.append((u_node, v_node, key))
            continue

        coords = list(clipped_lines[0].coords)
        if len(coords) < 2:
            edges_to_drop.append((u_node, v_node, key))
            continue

        clipped_polyline = [(float(x), float(y)) for x, y in coords]
        if clipped_polyline != local_polyline:
            n_clipped_edges += 1
        data["geometry_local"] = json.dumps(clipped_polyline)
        data["length"] = _polyline_length_xy(clipped_polyline)

    if edges_to_drop:
        graph.remove_edges_from(edges_to_drop)

    isolated = [node for node in graph.nodes if graph.degree(node) == 0]
    if isolated:
        graph.remove_nodes_from(isolated)

    return (n_clipped_edges, len(edges_to_drop), len(isolated))


def _edge_bridge_offset(data: dict, *, bridges_enabled: bool) -> float:
    """Return the vertical offset (m) for a street edge tagged as a bridge.

    Returns 0 when bridges are disabled or the edge has no bridge tag.
    Each OSM ``layer`` level lifts the deck by 5 m (matches the mesh).
    """
    if not bridges_enabled:
        return 0.0
    if str(data.get("bridge", "")).lower() not in ("yes", "true", "1", "viaduct"):
        return 0.0
    layer = 1
    layer_str = data.get("layer")
    if layer_str is not None:
        try:
            layer = max(1, int(float(str(layer_str))))
        except (ValueError, TypeError):
            pass
    return float(layer) * 5.0


def _build_tree_at(
    cx: float,
    cy: float,
    base_z: float,
    tree_h: float,
    canopy_r: float,
    trunk_radius: float,
    trunk_height_frac: float,
    mat_groups: dict[str, dict],
    *,
    material: str = "wood",
) -> None:
    """Append a single tree (octagonal trunk + cone canopy) to mat_groups.

    *material* selects which material group the trunk + canopy meshes are
    appended to. The green-vegetation option routes trees to ``"vegetation"``
    so the visualizer can override the PLY filename with a green PBR without
    affecting Sionna RT (which is keyed on the ITU material ID in
    ``scene.xml``, not the PLY filename).
    """
    trunk_h = tree_h * trunk_height_frac
    n_sides = 8
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    bottom_ring = np.column_stack(
        [
            cx + trunk_radius * np.cos(angles),
            cy + trunk_radius * np.sin(angles),
            np.full(n_sides, base_z),
        ]
    ).astype(np.float32)
    top_ring = bottom_ring.copy()
    top_ring[:, 2] = base_z + trunk_h
    trunk_verts = np.vstack([bottom_ring, top_ring])
    trunk_faces = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        trunk_faces.append([i, j, j + n_sides])
        trunk_faces.append([i, j + n_sides, i + n_sides])
    _append_mesh_to_group(trunk_verts, np.array(trunk_faces, dtype=np.int32), material, mat_groups)

    canopy_base_z = base_z + trunk_h
    canopy_peak_z = base_z + tree_h
    base_ring = np.column_stack(
        [
            cx + canopy_r * np.cos(angles),
            cy + canopy_r * np.sin(angles),
            np.full(n_sides, canopy_base_z),
        ]
    ).astype(np.float32)
    apex = np.array([[cx, cy, canopy_peak_z]], dtype=np.float32)
    canopy_verts = np.vstack([base_ring, apex])
    canopy_faces = []
    apex_idx = n_sides
    for i in range(n_sides):
        j = (i + 1) % n_sides
        canopy_faces.append([apex_idx, i, j])
    _append_mesh_to_group(
        canopy_verts, np.array(canopy_faces, dtype=np.int32), material, mat_groups
    )


def _iter_linestring_parts(geometry_obj):
    """Yield LineString parts from a shapely geometry object.

    Mirrors :func:`_iter_polygon_parts` but for lines.  Accepts a single
    LineString, a MultiLineString, or a GeoJSON-ish mapping with a
    ``type`` field (the shape OSM features come in after ``shape(...)``).
    """
    from shapely.geometry import LineString, MultiLineString
    from shapely.geometry import shape as _shape

    geom = geometry_obj
    if isinstance(geom, dict):
        geom = _shape(geom)

    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return []


def _sample_line_points(line, *, spacing_m: float) -> list[tuple[float, float]]:
    """Return equidistant points along a shapely LineString at *spacing_m* m.

    Includes both endpoints.  Returns an empty list if the line is
    degenerate or shorter than *spacing_m*.

    A non-positive *spacing_m* would otherwise spin the ``while d <=
    length`` advance below forever — Pydantic's ``Field(gt=0)`` on
    ``tree_row_spacing_m`` rejects bad YAML input, but we also guard
    here so any in-process caller wiring around the schema can't wedge
    scene generation.
    """
    if line is None or line.is_empty:
        return []
    if spacing_m <= 0:
        return []
    length = float(line.length)
    if length < spacing_m:
        return []
    out: list[tuple[float, float]] = []
    d = 0.0
    while d <= length:
        pt = line.interpolate(d)
        out.append((float(pt.x), float(pt.y)))
        d += spacing_m
    # Always include the final endpoint if the last sample missed it.
    end = line.interpolate(length)
    if not out or (abs(end.x - out[-1][0]) > 1e-6 or abs(end.y - out[-1][1]) > 1e-6):
        out.append((float(end.x), float(end.y)))
    return out


def _apply_worldcover_gap_fill(
    *,
    worldcover_index,
    osm_vegetation_geoms: list,
    to_utm,
    to_wgs84,
    origin,
    heightmap,
    origin_z: float,
    canopy_index,
    default_tree_height: float,
    default_crown_radius: float,
    trunk_radius: float,
    trunk_height_frac: float,
    mat_groups: dict,
    tree_material: str,
) -> int:
    """Scatter trees inside WorldCover tree_cover / shrubland cells that OSM
    doesn't already cover.  Returns the number of gap-fill trees added.

    One raster cell = roughly 10 × 10 m = 100 m².  We translate that into:
      - tree_cover (10): 1 tree per cell (~0.01 trees/m²)
      - shrubland (20): ~1 tree per 3 cells (~0.003 trees/m²)
    """
    if worldcover_index is None or not worldcover_index.is_available():
        return 0
    if not osm_vegetation_geoms:
        osm_prep = None
    else:
        try:
            from shapely.ops import unary_union

            osm_union = unary_union(osm_vegetation_geoms)
            osm_prep = shapely_prep(osm_union) if not osm_union.is_empty else None
        except Exception:
            osm_prep = None

    rng = np.random.default_rng(42)
    added = 0
    shrub_cell_counter = 0
    for ux, uy, cls in worldcover_index.iter_class_cells(
        (_WC_TREE_COVER, _WC_SHRUBLAND), to_wgs84, to_utm
    ):
        if osm_prep is not None and osm_prep.contains(Point(ux, uy)):
            continue
        if cls == _WC_SHRUBLAND:
            shrub_cell_counter += 1
            if shrub_cell_counter % 3 != 0:
                continue
        cx = ux - origin[0]
        cy = uy - origin[1]
        # Small jitter so neighbouring cells don't form a visibly regular grid.
        cx += float(rng.uniform(-3.0, 3.0))
        cy += float(rng.uniform(-3.0, 3.0))
        base_z = 0.0
        if heightmap is not None:
            try:
                base_z = float(np.array(heightmap.height_from_utm(ux, uy)).ravel()[0]) - origin_z
            except Exception:
                base_z = 0.0
        tree_h = default_tree_height
        if canopy_index is not None:
            lon, lat = to_wgs84.transform(ux, uy)
            cheight = canopy_index.height_at(lon, lat)
            if cheight is not None and 1 < cheight < 80:
                tree_h = float(cheight)
        _build_tree_at(
            cx,
            cy,
            base_z,
            tree_h,
            default_crown_radius,
            trunk_radius,
            trunk_height_frac,
            mat_groups,
            material=tree_material,
        )
        added += 1
    return added


def _scatter_trees_in_polygon(
    polygon: Polygon,
    *,
    spacing_m: float,
    jitter_frac: float = 0.35,
    seed: int = 0,
    max_trees: int = 5000,
) -> list[tuple[float, float]]:
    """Return jittered grid samples (in UTM coords) that lie inside *polygon*.

    Used to populate forest/park polygons with discrete trees instead of
    treating the polygon as one solid wood block.
    """
    minx, miny, maxx, maxy = polygon.bounds
    if not math.isfinite(minx) or maxx - minx < spacing_m or maxy - miny < spacing_m:
        return []

    rng = np.random.default_rng(seed)
    nx_steps = max(1, int(math.floor((maxx - minx) / spacing_m)))
    ny_steps = max(1, int(math.floor((maxy - miny) / spacing_m)))
    if nx_steps * ny_steps > max_trees * 4:
        # Coarsen grid to keep the candidate count bounded.
        scale = math.sqrt((nx_steps * ny_steps) / (max_trees * 4))
        spacing_m *= scale
        nx_steps = max(1, int(math.floor((maxx - minx) / spacing_m)))
        ny_steps = max(1, int(math.floor((maxy - miny) / spacing_m)))

    jitter_amp = jitter_frac * spacing_m
    prepared = shapely_prep(polygon)
    out: list[tuple[float, float]] = []
    for ix in range(nx_steps):
        for iy in range(ny_steps):
            cx = minx + (ix + 0.5) * spacing_m + float(rng.uniform(-jitter_amp, jitter_amp))
            cy = miny + (iy + 0.5) * spacing_m + float(rng.uniform(-jitter_amp, jitter_amp))
            if prepared.contains(Point(cx, cy)):
                out.append((cx, cy))
                if len(out) >= max_trees:
                    return out
    return out


def _fetch_street_graph(center: tuple[float, float], distance: float):
    """Retain optional visual surfacing without changing RF material defaults.

    Default graph extraction omits surface tags. Capture them in a visual-only
    attribute; existing caller configurations that already retain surface keep
    their previous RF behavior. Cached older graphs remain usable as unknown.
    """
    previous = ox.settings.useful_tags_way
    added_surface = "surface" not in previous
    try:
        ox.settings.useful_tags_way = list(dict.fromkeys([*previous, "surface"]))
        graph = ox.graph_from_point(center, dist=distance, network_type="all")
    finally:
        ox.settings.useful_tags_way = previous
    for _u, _v, data in graph.edges(data=True):
        if "surface" in data:
            data["_visual_surface"] = data["surface"]
            if added_surface:
                del data["surface"]
    return graph


def _extrude_polygon_to_group(
    polygon: Polygon,
    height: float,
    min_height: float,
    material: str,
    origin: np.ndarray,
    base_z: float,
    mat_groups: dict[str, dict],
    *,
    roof_shape: str = "flat",
    roof_height: float = 0.0,
    roof_material: str | None = None,
) -> bool:
    """Triangulate, extrude a polygon, and append to the material group.

    Supports ``min_height > 0`` for elevated volumes (Simple 3D Buildings spec).
    When *roof_shape* is not ``"flat"``, the walls are extruded to
    ``height - roof_height`` and a parametric roof mesh is added on top.
    When *roof_material* differs from *material*, the roof faces use a
    separate ITU material (per-face wall/roof split).
    Returns True on success, False if the polygon was skipped.
    """
    outer_xy = unique_coords(reorder_localize(polygon.exterior, origin[0], origin[1]))
    if len(outer_xy) < 3:
        return False

    holes_xy = []
    for interior in polygon.interiors:
        h = unique_coords(reorder_localize(interior, origin[0], origin[1]))
        if len(h) >= 3:
            holes_xy.append(h)

    try:
        v2d, faces = _triangulate_polygon(outer_xy, holes_xy)
    except (RuntimeError, KeyError, ValueError):
        return False

    # Determine wall and roof heights.
    effective_roof_height = 0.0
    # Only shrink walls for roof shapes we actually know how to mesh —
    # otherwise the walls get shortened, the parametric roof generator
    # returns None, and the part renders with a vertical hole at the top
    # (e.g. the Capitol dome with ``roof:shape=dome`` before this guard
    # existed).
    if roof_shape != "flat" and roof_shape in SUPPORTED_ROOF_SHAPES and roof_height > 0:
        # Clamp roof_height so walls are at least 1m tall.
        effective_roof_height = min(roof_height, max(0.0, height - min_height - 1.0))

    wall_height = height - min_height - effective_roof_height
    if wall_height <= 0:
        return False

    nv = len(v2d)
    z_base = base_z + min_height
    points = np.column_stack([v2d, np.full(nv, z_base)]).astype(np.float32)

    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(points)
    mesh_t.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
    extruded = mesh_t.extrude_linear([0, 0, wall_height])

    if material not in mat_groups:
        mat_groups[material] = {"vertices": [], "faces": [], "vcount": 0}
    grp = mat_groups[material]
    grp["vertices"].append(extruded.vertex.positions)
    grp["faces"].append(extruded.triangle.indices + grp["vcount"])
    grp["vcount"] += extruded.vertex.positions.shape[0]

    # Generate roof mesh if non-flat.
    if effective_roof_height > 0:
        wall_top_z = z_base + wall_height
        roof_result = _generate_roof_mesh(outer_xy, roof_shape, effective_roof_height, wall_top_z)
        if roof_result is not None:
            roof_verts, roof_faces = roof_result
            roof_mat = roof_material if roof_material else material
            _append_mesh_to_group(roof_verts, roof_faces, roof_mat, mat_groups)

    return True


def _match_parts_to_buildings(
    buildings_list: list[dict],
    parts_list: list[dict],
    origin: np.ndarray,
) -> dict[int, list[dict]]:
    """Match building:part features to their parent building outlines.

    Returns a mapping from building index (in *buildings_list*) to the list
    of part dicts whose centroid falls within that building's footprint.
    Parts that don't match any parent are returned under key ``-1``.
    """
    if not parts_list:
        return {}

    # Build a spatial index of parent buildings.
    building_polys: list[tuple[int, Polygon, object]] = []
    for idx, bldg in enumerate(buildings_list):
        # Only consider features tagged as proper buildings (not building:part).
        if bldg.get("building:part") and not bldg.get("building"):
            continue
        for poly in _iter_polygon_parts(bldg["geometry"]):
            building_polys.append((idx, poly, shapely_prep(poly)))

    matched: dict[int, list[dict]] = {}
    unmatched: list[dict] = []

    for part in parts_list:
        part_polys = _iter_polygon_parts(part["geometry"])
        if not part_polys:
            continue
        centroid = part_polys[0].centroid

        parent_idx = -1
        for idx, _poly, prepared in building_polys:
            if prepared.contains(centroid):
                parent_idx = idx
                break

        if parent_idx >= 0:
            matched.setdefault(parent_idx, []).append(part)
        else:
            unmatched.append(part)

    if unmatched:
        matched[-1] = unmatched

    return matched


def _terrain_base_z(heightmap, polygon: Polygon, origin_z: float) -> float:
    """Compute building base elevation from terrain at polygon centroid."""
    if heightmap is None:
        return 0.0
    centroid = polygon.centroid
    return float(np.array(heightmap.height_from_utm(centroid.x, centroid.y)).ravel()[0]) - origin_z


def _add_material(scene_el: ET.Element, name: str) -> None:
    """Add an ITU radio material BSDF to the scene XML."""
    props = ITU_MATERIALS[name]
    mat_id = f"mat-itu_{name}"
    bsdf = ET.SubElement(scene_el, "bsdf", type="diffuse", id=mat_id)
    r, g, b = props["color"]
    ET.SubElement(bsdf, "rgb", value=f"{r} {g} {b}", name="reflectance")


def _add_shape(
    scene_el: ET.Element,
    shape_id: str,
    ply_path: str,
    material_name: str,
) -> None:
    """Add a PLY shape referencing a material to the scene XML."""
    shape_el = ET.SubElement(scene_el, "shape", type="ply", id=shape_id)
    ET.SubElement(shape_el, "string", name="filename", value=ply_path)
    ET.SubElement(shape_el, "boolean", name="face_normals", value="true")
    ET.SubElement(shape_el, "ref", id=f"mat-itu_{material_name}", name="bsdf")


@_managed_build
def build_scene(
    bbox: tuple[float, float, float, float],
    output_dir: str,
    *,
    terrain: bool = False,
    terrain_zoom: int = 14,
    ground_pad_m: float = 200.0,
    ground_resolution_m: float = 50.0,
    osm_server: str | None = None,
    metadata_config: dict | None = None,
    barriers: bool = True,
    vegetation: bool = True,
    bridges: bool = True,
    height_enrichment: bool = True,
    osm_overpass_timeout: int = DEFAULT_OSM_OVERPASS_TIMEOUT_S,
    osm_cache_dir: Path | None = None,
    osm_refresh_cache: bool = False,
    # -- Optional vegetation realism flags (all default to False) -----------
    canopy_heights: bool = False,
    tree_rows: bool = False,
    vegetation_gap_fill: bool = False,
    worldcover_water_check: bool = False,
    green_vegetation: bool = True,
    tree_row_spacing_m: float = 8.0,
    # Landuse-informed material defaults (default True).
    landuse_materials: bool = True,
) -> str:
    """Generate a Mitsuba scene from OpenStreetMap data.

    Parameters
    ----------
    bbox : tuple
        ``(min_lat, min_lon, max_lat, max_lon)`` in WGS84 degrees.
    output_dir : str
        Directory to write ``scene.xml`` and ``mesh/`` into.
    terrain : bool
        Use public elevation GeoTIFFs for the ground mesh.
    terrain_zoom : int
        Slippy-map zoom level for terrain tiles.
    ground_pad_m : float
        Extra ground padding beyond the bounding box in meters.
    ground_resolution_m : float
        Ground mesh grid resolution in meters.
    osm_server : str, optional
        Custom Overpass API endpoint URL.

    Returns
    -------
    str
        Path to the generated ``scene.xml``.
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError(f"Invalid bbox: {bbox}. Expected (min_lat, min_lon, max_lat, max_lon).")

    logger.info(
        "Building OSM scene: bbox=(%.4f, %.4f, %.4f, %.4f)",
        min_lat,
        min_lon,
        max_lat,
        max_lon,
    )

    # -- Setup ----------------------------------------------------------------
    if osm_server:
        ox.settings.overpass_url = osm_server
    _configure_osmnx_overpass_timeout(osm_overpass_timeout)
    ox.settings.max_query_area_size = OSMNX_MAX_QUERY_AREA_SIZE_M2

    # Resolve the OSM cache directory.  Call sites use the cache uniformly; a
    # None cache_dir falls back to plain `_run_osm_query` (no caching).
    if osm_cache_dir is None:
        osm_cache_dir = _default_osm_cache_dir()
    osm_refresh_cache = osm_refresh_cache or _osm_refresh_from_env()
    logger.info("OSM cache dir: %s (refresh=%s)", osm_cache_dir, osm_refresh_cache)
    # Per-build cache accounting — populated by each _cached_osm_query call
    # and surfaced on osm_metadata.json so callers can tell at a glance
    # whether a rebuild actually re-fetched from Overpass.
    cache_stats: dict[str, int] = {"hits": 0, "misses": 0}
    context = active_acquisition()
    feature_labels = ["buildings", "building_parts", "water"]
    if landuse_materials:
        feature_labels.append("landuse_materials")
    if barriers:
        feature_labels.append("barriers")
    if vegetation:
        feature_labels.append("vegetation")
    feature_batch = _FeatureBatch(
        (min_lon, min_lat, max_lon, max_lat), osm_cache_dir, osm_refresh_cache, feature_labels
    )

    utm_crs = get_utm_crs(min_lon, min_lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    logger.info("UTM zone: %s", utm_crs)

    # Origin = center of bounding box in UTM.
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    origin_x, origin_y = to_utm.transform(center_lon, center_lat)
    origin = np.array([origin_x, origin_y, 0.0])

    # Scene bbox in both UTM and local (origin-relative) coordinates.
    # Used to clip OSM features that Overpass returns in full even when
    # they extend well beyond the requested bbox (long rivers, long
    # roads, continent-sized forests).  Padded by ``ground_pad_m`` on
    # every side so clipped features tuck neatly under the ground mesh's
    # padded extent instead of leaving a gap along the edge.
    _bbox_corners_ll = [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
    ]
    _bbox_corners_utm = [to_utm.transform(lon, lat) for lon, lat in _bbox_corners_ll]
    _xs_utm = [p[0] for p in _bbox_corners_utm]
    _ys_utm = [p[1] for p in _bbox_corners_utm]
    half_w_m = (max(_xs_utm) - min(_xs_utm)) / 2.0
    half_h_m = (max(_ys_utm) - min(_ys_utm)) / 2.0
    scene_clip_pad_m = float(ground_pad_m)
    scene_bbox_utm = shapely_box(
        origin_x - half_w_m - scene_clip_pad_m,
        origin_y - half_h_m - scene_clip_pad_m,
        origin_x + half_w_m + scene_clip_pad_m,
        origin_y + half_h_m + scene_clip_pad_m,
    )
    scene_bbox_local = shapely_box(
        -half_w_m - scene_clip_pad_m,
        -half_h_m - scene_clip_pad_m,
        half_w_m + scene_clip_pad_m,
        half_h_m + scene_clip_pad_m,
    )

    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    # -- Height enrichment (Microsoft GlobalML) ---------------------------------
    global _height_index, _height_source_counts, _external_data_used  # noqa: PLW0603
    _height_source_counts = {}
    _height_index = None
    _external_data_used = []
    osm_bbox_wgs84 = (min_lon, min_lat, max_lon, max_lat)

    # -- Scene XML skeleton ---------------------------------------------------
    scene_el = ET.Element("scene", version="2.1.0")
    ET.Comment(
        f"Generated from OSM: bbox=({min_lat:.4f}, {min_lon:.4f}, {max_lat:.4f}, {max_lon:.4f})"
    )

    for mat_name in ITU_MATERIALS:
        _add_material(scene_el, mat_name)

    # -- Query OSM buildings and building:part ---------------------------------
    osm_bbox = (min_lon, min_lat, max_lon, max_lat)

    # Optional landuse query feeds _LandUseMaterialIndex so
    # untagged buildings get a smarter default than blanket concrete
    # (industrial→metal, commercial→glass, residential→brick, etc.).
    landuse_index: _LandUseMaterialIndex | None = None
    if landuse_materials:
        landuse_tags = {
            "landuse": [
                "industrial",
                "warehouse",
                "depot",
                "port",
                "harbour",
                "garages",
                "railway",
                "commercial",
                "retail",
                "education",
                "institutional",
                "civic",
                "residential",
                "religious",
                "cemetery",
                "construction",
            ]
        }
        landuse_gdf = _cached_osm_query(
            "landuse_materials",
            bbox=osm_bbox,
            tags=landuse_tags,
            fetch_fn=lambda: feature_batch.get("landuse_materials", landuse_tags),
            serializer=_GDF_SERIALIZER,
            optional=True,
            no_data_value=None,
            cache_dir=osm_cache_dir,
            refresh=osm_refresh_cache,
            stats=cache_stats,
        )
        if landuse_gdf is not None and len(landuse_gdf) > 0:
            try:
                landuse_gdf = landuse_gdf.to_crs(utm_crs)
                landuse_records = landuse_gdf.to_dict("records")
                landuse_index = _LandUseMaterialIndex(landuse_records)
                if landuse_index.is_available():
                    logger.info(
                        "Landuse material index: %d polygons across %d classes",
                        len(landuse_index._polys),
                        len(set(landuse_index._materials)),
                    )
                else:
                    landuse_index = None
            except Exception as exc:
                logger.warning("Failed to build landuse material index: %s", exc)
                landuse_index = None

    buildings_tags = {"building": True}
    buildings_gdf = _cached_osm_query(
        "buildings",
        bbox=osm_bbox,
        tags=buildings_tags,
        fetch_fn=lambda: feature_batch.get("buildings", buildings_tags),
        serializer=_GDF_SERIALIZER,
        optional=False,
        no_data_value=None,
        cache_dir=osm_cache_dir,
        refresh=osm_refresh_cache,
        stats=cache_stats,
    )
    if buildings_gdf is None:
        buildings_list = []
    else:
        buildings_gdf = buildings_gdf.to_crs(utm_crs)
        buildings_list = buildings_gdf.to_dict("records")

    # Query building:part features (S3DB spec — multi-level buildings).
    parts_tags = {"building:part": True}
    parts_gdf = _cached_osm_query(
        "building_parts",
        bbox=osm_bbox,
        tags=parts_tags,
        fetch_fn=lambda: feature_batch.get("building_parts", parts_tags),
        serializer=_GDF_SERIALIZER,
        optional=True,
        no_data_value=None,
        cache_dir=osm_cache_dir,
        refresh=osm_refresh_cache,
        stats=cache_stats,
    )
    if parts_gdf is None:
        parts_list = []
    else:
        parts_gdf = parts_gdf.to_crs(utm_crs)
        parts_list = parts_gdf.to_dict("records")

    # -- Fetch and save street network for mobility ----------------------------
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    half_size = max(
        abs(max_lat - min_lat) * 111_000 / 2,
        abs(max_lon - min_lon) * 111_000 * math.cos(math.radians(center_lat)) / 2,
    )
    # The hash key carries ``network_type`` so changing the filter invalidates
    # the cache, and ``dist`` so a resized scene doesn't hit a stale graph.
    street_key_tags = {"network_type": "all", "dist_m": round(float(half_size), 1)}
    G = _cached_osm_query(
        "street_network",
        bbox=osm_bbox,
        tags=street_key_tags,
        fetch_fn=lambda: _fetch_street_graph((center_lat, center_lon), half_size),
        serializer=_GRAPH_SERIALIZER,
        optional=False,
        no_data_value=None,
        cache_dir=osm_cache_dir,
        refresh=osm_refresh_cache,
        stats=cache_stats,
    )
    if G is None:
        raise RuntimeError("Street network query returned no data")

    # -- Ground mesh ----------------------------------------------------------
    heightmap = None
    terrain_grid_meta = None
    origin_z = 0.0
    mesh_files = ["mesh/ground.ply"]
    if terrain:
        try:
            from .heightmap import HeightMap

            ext_bbox = (
                min_lon - 0.01,
                min_lat - 0.01,
                max_lon + 0.01,
                max_lat + 0.01,
            )
            heightmap = HeightMap(utm_crs, ext_bbox, zoom=terrain_zoom)
            # Compute origin elevation so the scene is z-centered.
            origin_z = float(np.array(heightmap.height_from_utm(origin[0], origin[1])).ravel()[0])
            context.record("terrain", "data")
            logger.info("Terrain heightmap loaded (origin elevation: %.1f m)", origin_z)

            ground_verts, ground_faces, terrain_grid_meta = terrain_ground_mesh(
                bbox,
                to_utm,
                origin,
                heightmap,
                origin_z,
                pad_m=ground_pad_m,
                resolution_m=ground_resolution_m,
                return_grid=True,
            )
        except (ImportError, OSError) as exc:
            context.record("terrain", "failed", error=str(exc), fallback="flat ground")
            logger.warning("Terrain elevation failed (%s), using flat ground", exc)
            heightmap = None
            ground_verts, ground_faces = flat_ground_mesh(
                bbox,
                to_utm,
                origin,
                pad_m=ground_pad_m,
                resolution_m=ground_resolution_m,
            )
    else:
        ground_verts, ground_faces = flat_ground_mesh(
            bbox,
            to_utm,
            origin,
            pad_m=ground_pad_m,
            resolution_m=ground_resolution_m,
        )

    ground_mesh = o3d.geometry.TriangleMesh()
    ground_mesh.vertices = o3d.utility.Vector3dVector(ground_verts)
    ground_mesh.triangles = o3d.utility.Vector3iVector(ground_faces)
    ground_mesh.compute_vertex_normals()
    ground_path = os.path.join(mesh_dir, "ground.ply")
    o3d.io.write_triangle_mesh(ground_path, ground_mesh)
    _add_shape(scene_el, "mesh-ground", "mesh/ground.ply", DEFAULT_GROUND_MATERIAL)
    logger.info("Ground mesh: %d vertices, %d faces", len(ground_verts), len(ground_faces))

    visual_geographic_layers = {
        "mesh/ground.ply": {
            "total_triangles": len(ground_faces),
            "layers": {"ground": [[0, len(ground_faces)]]},
        }
    }

    # -- Canopy height + land cover indices (built lazily) --------------------
    to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    canopy_index: _EthCanopyHeightIndex | None = None
    worldcover_index: _WorldCoverIndex | None = None

    if canopy_heights:
        try:
            canopy_index = _EthCanopyHeightIndex(osm_bbox_wgs84)
            context.record(
                "canopy_heights",
                "failed" if canopy_index.failures else "data",
                errors=canopy_index.failures,
            )
            if canopy_index.is_available():
                _cite_external_dataset(_ETH_CANOPY_ATTRIBUTION)
            else:
                canopy_index = None
        except (OSError, RuntimeError, ImportError) as exc:
            context.record("canopy_heights", "failed", error=str(exc))
            logger.warning("ETH canopy height index unavailable: %s", exc)
            canopy_index = None

    if vegetation_gap_fill or worldcover_water_check:
        try:
            worldcover_index = _WorldCoverIndex(osm_bbox_wgs84)
            context.record(
                "worldcover",
                "failed" if worldcover_index.failures else "data",
                errors=worldcover_index.failures,
            )
            if worldcover_index.is_available():
                _cite_external_dataset(_WORLDCOVER_ATTRIBUTION)
            else:
                worldcover_index = None
        except (OSError, RuntimeError, ImportError) as exc:
            context.record("worldcover", "failed", error=str(exc))
            logger.warning("ESA WorldCover index unavailable: %s", exc)
            worldcover_index = None

    # Download external heights only if a building/part actually needs them.
    if height_enrichment:
        needs_height = any(not _has_tagged_height(item) for item in buildings_list + parts_list)
        if needs_height:
            try:
                gml = _GlobalMLHeightIndex(osm_bbox_wgs84, to_utm)
                if gml._tree is not None:
                    _height_index = gml
                context.record(
                    "height_enrichment",
                    "data" if gml._tree is not None else "fallback",
                    reason=(
                        "GlobalML lookup"
                        if gml._tree is not None
                        else "No matching external heights"
                    ),
                )
            except (OSError, RuntimeError, ImportError) as exc:
                context.record(
                    "height_enrichment",
                    "failed",
                    error=str(exc),
                    fallback="OSM deterministic height defaults",
                )
        else:
            context.record(
                "height_enrichment",
                "not_needed",
                reason="All building and part heights have valid OSM tags",
            )

    # Match parts to parent building outlines.
    parts_by_building = _match_parts_to_buildings(buildings_list, parts_list, origin)
    n_buildings_with_parts = sum(1 for k in parts_by_building if k >= 0)
    n_orphan_parts = len(parts_by_building.get(-1, []))

    logger.info(
        "Processing %d buildings (%d with building:part, %d parts total, %d orphan parts)...",
        len(buildings_list),
        n_buildings_with_parts,
        len(parts_list),
        n_orphan_parts,
    )

    # Group extruded geometry by ITU material.
    mat_groups: dict[str, dict] = {}
    building_skip_counts = {"non_polygon": 0, "degenerate": 0, "triangulation": 0}
    part_counts = {"extruded": 0, "skipped": 0}
    # Collect building footprints in local coords for trajectory snapping.
    building_local_polys: list[Polygon] = []

    for bldg_idx, building in enumerate(buildings_list):
        parts = parts_by_building.get(bldg_idx)

        if parts:
            # This building has building:part children — extrude each part
            # individually instead of the parent outline (S3DB spec).
            # Collect the PARENT footprint for trajectory snapping buffer.
            for parent_poly in _iter_polygon_parts(building["geometry"]):
                local_xy = unique_coords(
                    reorder_localize(parent_poly.exterior, origin[0], origin[1])
                )
                if len(local_xy) >= 3:
                    building_local_polys.append(Polygon(local_xy))
            for part in parts:
                for polygon in _iter_polygon_parts(part["geometry"]):
                    height = _building_height(part, polygon)
                    min_height = _building_min_height(part)
                    material = _building_material(part, polygon, landuse_index)
                    r_shape, r_height = _parse_roof_tags(part)
                    r_mat = _roof_material(part)
                    # Parts inherit parent material if they don't have their own.
                    if material == DEFAULT_BUILDING_MATERIAL:
                        parent_polygon = next(iter(_iter_polygon_parts(building["geometry"])), None)
                        parent_material = _building_material(
                            building, parent_polygon, landuse_index
                        )
                        if parent_material != DEFAULT_BUILDING_MATERIAL:
                            material = parent_material

                    base_z = _terrain_base_z(heightmap, polygon, origin_z)
                    ok = _extrude_polygon_to_group(
                        polygon,
                        height,
                        min_height,
                        material,
                        origin,
                        base_z,
                        mat_groups,
                        roof_shape=r_shape,
                        roof_height=r_height,
                        roof_material=r_mat,
                    )
                    if ok:
                        part_counts["extruded"] += 1
                    else:
                        part_counts["skipped"] += 1
        else:
            # No parts — extrude the parent outline as before.
            polygons = _iter_polygon_parts(building["geometry"])
            if not polygons:
                building_skip_counts["non_polygon"] += 1
                continue

            r_shape, r_height = _parse_roof_tags(building)
            r_mat = _roof_material(building)
            for polygon in polygons:
                # Collect footprint for trajectory snapping buffer.
                local_xy = unique_coords(reorder_localize(polygon.exterior, origin[0], origin[1]))
                if len(local_xy) >= 3:
                    building_local_polys.append(Polygon(local_xy))
                height = _building_height(building, polygon)
                material = _building_material(building, polygon, landuse_index)
                base_z = _terrain_base_z(heightmap, polygon, origin_z)
                ok = _extrude_polygon_to_group(
                    polygon,
                    height,
                    0.0,
                    material,
                    origin,
                    base_z,
                    mat_groups,
                    roof_shape=r_shape,
                    roof_height=r_height,
                    roof_material=r_mat,
                )
                if not ok:
                    building_skip_counts["degenerate"] += 1

    # Also extrude orphan parts (building:part not matched to any parent).
    for part in parts_by_building.get(-1, []):
        for polygon in _iter_polygon_parts(part["geometry"]):
            height = _building_height(part, polygon)
            min_height = _building_min_height(part)
            material = _building_material(part, polygon, landuse_index)
            r_shape, r_height = _parse_roof_tags(part)
            r_mat = _roof_material(part)
            base_z = _terrain_base_z(heightmap, polygon, origin_z)
            ok = _extrude_polygon_to_group(
                polygon,
                height,
                min_height,
                material,
                origin,
                base_z,
                mat_groups,
                roof_shape=r_shape,
                roof_height=r_height,
                roof_material=r_mat,
            )
            if ok:
                part_counts["extruded"] += 1
            else:
                part_counts["skipped"] += 1

    skipped_total = sum(building_skip_counts.values())
    if skipped_total > 0:
        logger.info("Skipped buildings: %s", building_skip_counts)
    if part_counts["extruded"] > 0 or part_counts["skipped"] > 0:
        logger.info("Building parts: %s", part_counts)
    if _height_source_counts:
        logger.info("Height sources: %s", _height_source_counts)

    # -- Query and extrude barriers (walls, fences, hedges) --------------------
    # Visual-only face classification: buildings precede barriers/vegetation in
    # each material group. Never infer semantic buildings from a PLY filename.
    visual_building_faces = {
        f"mesh/buildings_{material}.ply": {
            "building_triangles": sum(int(f.shape[0]) for f in group["faces"]),
            # Each extrusion/roof chunk retains a distinct visual part even when
            # several buildings share one RF material mesh.
            "parts": np.column_stack(
                (
                    np.cumsum([0] + [int(f.shape[0]) for f in group["faces"][:-1]]),
                    [int(f.shape[0]) for f in group["faces"]],
                )
            ).tolist(),
        }
        for material, group in mat_groups.items()
    }

    building_face_counts = {
        material: sum(int(f.shape[0]) for f in group["faces"])
        for material, group in mat_groups.items()
    }
    barrier_features = []

    # Must run BEFORE PLY write so barriers end up in mat_groups.
    barrier_list = []
    if barriers:
        barrier_tags = {"barrier": ["wall", "fence", "retaining_wall", "hedge", "city_wall"]}
        barrier_gdf = _cached_osm_query(
            "barriers",
            bbox=osm_bbox,
            tags=barrier_tags,
            fetch_fn=lambda: feature_batch.get("barriers", barrier_tags),
            serializer=_GDF_SERIALIZER,
            optional=True,
            no_data_value=None,
            cache_dir=osm_cache_dir,
            refresh=osm_refresh_cache,
            stats=cache_stats,
        )
        if barrier_gdf is not None:
            barrier_gdf = barrier_gdf.to_crs(utm_crs)
            barrier_list = barrier_gdf.to_dict("records")
            for record, feature_id in zip(barrier_list, barrier_gdf.index):
                record["source_id"] = (
                    "/".join(map(str, feature_id))
                    if isinstance(feature_id, tuple)
                    else str(feature_id)
                )

    if barrier_list:
        logger.info("Processing %d barriers...", len(barrier_list))
        for barrier in barrier_list:
            geom = shape(barrier["geometry"]) if barrier.get("geometry") else None
            spec = barrier_spec(barrier)
            if geom is None or spec is None:
                continue
            lines = (
                [geom]
                if geom.geom_type == "LineString"
                else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
            )
            material = (
                "vegetation"
                if spec["barrier"] == "hedge" and green_vegetation
                else spec["material"]
            )
            parts = []
            for line in lines:
                xy = np.asarray(line.coords, dtype=float)[:, :2] - np.asarray(origin[:2])
                if len(xy) < 2:
                    continue
                z = (
                    _sample_local_terrain_heights(heightmap, origin, origin_z, xy)
                    if heightmap is not None
                    else np.zeros(len(xy))
                )
                vertices, faces = barrier_mesh(np.column_stack((xy, z)), spec)
                if not len(faces):
                    continue
                first = sum(int(f.shape[0]) for f in mat_groups.get(material, {}).get("faces", []))
                _append_mesh_to_group(vertices, faces, material, mat_groups)
                parts.append([first, len(faces)])
            if parts:
                mesh_path = (
                    "mesh/vegetation.ply"
                    if material == "vegetation"
                    else f"mesh/buildings_{material}.ply"
                )
                barrier_features.append(
                    {
                        "source_id": barrier["source_id"],
                        "tags": {
                            key: str(barrier[key])
                            for key in (
                                "barrier",
                                "fence_type",
                                "height",
                                "width",
                                "material",
                                "source",
                                "name",
                            )
                            if barrier.get(key) is not None and str(barrier[key]) != "nan"
                        },
                        **spec,
                        "mesh": mesh_path,
                        "triangle_ranges": parts,
                    }
                )
        logger.info("Barrier features: %d", len(barrier_features))
    barrier_end_counts = {
        material: sum(int(f.shape[0]) for f in group["faces"])
        for material, group in mat_groups.items()
    }

    # -- Query OSM water bodies -----------------------------------------------
    water_tags = {
        "water": True,
        "landuse": ["reservoir", "basin", "salt_pond"],
        "natural": ["water"],
    }
    water_gdf = _cached_osm_query(
        "water",
        bbox=osm_bbox,
        tags=water_tags,
        fetch_fn=lambda: feature_batch.get("water", water_tags),
        serializer=_GDF_SERIALIZER,
        optional=True,
        no_data_value=None,
        cache_dir=osm_cache_dir,
        refresh=osm_refresh_cache,
        stats=cache_stats,
    )
    if water_gdf is None:
        water_list = []
    else:
        water_gdf = water_gdf.to_crs(utm_crs)
        water_list = water_gdf.to_dict("records")

    # -- WorldCover water validation ------------------------------------------
    # Cross-check OSM water polygons against WorldCover's permanent_water_bodies
    # class (80).  If a substantial fraction of WorldCover water cells have no
    # corresponding OSM polygon, emit a WARN so users know the scene may be
    # missing a lake / pond / canal that OSM hasn't mapped.  No mesh extension
    # is attempted here — the warning alone closes the debugging loop.
    if worldcover_water_check and worldcover_index is not None and worldcover_index.is_available():
        try:
            from shapely.ops import unary_union as _unary_union

            osm_water_union = (
                _unary_union(
                    [
                        _shape_water
                        for _shape_water in (
                            shape(w["geometry"]) for w in water_list if w.get("geometry")
                        )
                    ]
                )
                if water_list
                else None
            )
            osm_water_prep = (
                shapely_prep(osm_water_union)
                if osm_water_union is not None and not osm_water_union.is_empty
                else None
            )
        except Exception:
            osm_water_prep = None

        missing_cells = 0
        total_water_cells = 0
        for ux, uy, _cls in worldcover_index.iter_class_cells(
            (_WC_PERMANENT_WATER,), to_wgs84, to_utm
        ):
            total_water_cells += 1
            if osm_water_prep is not None and osm_water_prep.contains(Point(ux, uy)):
                continue
            missing_cells += 1

        # Roughly 100 m² per 10 m raster cell; compare against bbox area in m².
        bbox_area_m2 = max(
            (max_lon - min_lon)
            * 111_000
            * math.cos(math.radians((min_lat + max_lat) / 2))
            * (max_lat - min_lat)
            * 111_000,
            1.0,
        )
        missing_area_m2 = missing_cells * 100.0
        if missing_cells > 0 and missing_area_m2 / bbox_area_m2 > 0.01:
            logger.warning(
                "WorldCover reports %d water cells (~%.0f m²) not present in "
                "OSM — scene may be missing a water body (total WorldCover "
                "water cells in bbox: %d)",
                missing_cells,
                missing_area_m2,
                total_water_cells,
            )

    if water_list:
        logger.info("Processing %d water bodies...", len(water_list))
        water_vertices: list[o3d.core.Tensor] = []
        water_faces_list: list[o3d.core.Tensor] = []
        water_vcount = 0
        water_skip_counts = {
            "non_polygon": 0,
            "degenerate": 0,
            "triangulation": 0,
            "terrain_span": 0,
        }

        for water in water_list:
            # Clip each water feature against the padded scene bbox so long
            # rivers (the Seine, the Thames, etc.) that Overpass returns in
            # their entirety don't leave water floating past the ground-mesh
            # edge.  The clip happens in UTM because ``water_gdf`` was
            # already re-projected above.
            polygons = _clip_polygon_to_bbox(water["geometry"], scene_bbox_utm)
            if not polygons:
                water_skip_counts["non_polygon"] += 1
                continue

            for wpoly in polygons:
                outer_xy = unique_coords(reorder_localize(wpoly.exterior, origin[0], origin[1]))
                if len(outer_xy) < 3:
                    water_skip_counts["degenerate"] += 1
                    continue

                holes_xy = []
                for interior in wpoly.interiors:
                    h = unique_coords(reorder_localize(interior, origin[0], origin[1]))
                    if len(h) >= 3:
                        holes_xy.append(h)

                try:
                    v2d, faces = _triangulate_polygon(outer_xy, holes_xy)
                except (RuntimeError, KeyError, ValueError):
                    water_skip_counts["triangulation"] += 1
                    continue

                nv = len(v2d)
                z_offset = 0.5
                if heightmap is not None:
                    sampled = _sample_local_terrain_heights(
                        heightmap,
                        origin,
                        origin_z,
                        np.asarray(v2d, dtype=np.float32),
                    )
                    if (
                        sampled.size
                        and float(sampled.max() - sampled.min()) > MAX_FLAT_WATER_ELEVATION_SPAN_M
                    ):
                        water_skip_counts["terrain_span"] += 1
                        continue
                    z_offset = float(sampled.mean()) + 0.5 if sampled.size else 0.5

                points = np.column_stack([v2d, np.full(nv, z_offset)]).astype(np.float32)

                water_vertices.append(o3d.core.Tensor(points))
                water_faces_list.append(o3d.core.Tensor(faces.astype(np.int32)) + water_vcount)
                water_vcount += nv

        if water_vcount > 0:
            if len(water_vertices) > 1:
                combined_wv = o3d.core.concatenate(water_vertices, axis=0)
                combined_wf = o3d.core.concatenate(water_faces_list, axis=0)
            else:
                combined_wv = water_vertices[0]
                combined_wf = water_faces_list[0]

            water_mesh = o3d.t.geometry.TriangleMesh()
            water_mesh.vertex.positions = combined_wv
            water_mesh.triangle.indices = combined_wf

            water_path = os.path.join(mesh_dir, "water.ply")
            o3d.t.io.write_triangle_mesh(water_path, water_mesh)
            _add_shape(scene_el, "mesh-water", "mesh/water.ply", DEFAULT_WATER_MATERIAL)
            logger.info("Water mesh: %d bodies, %d vertices", len(water_vertices), water_vcount)
            mesh_files.append("mesh/water.ply")
            count = int(combined_wf.shape[0])
            visual_geographic_layers["mesh/water.ply"] = {
                "total_triangles": count,
                "layers": {"water": [[0, count]]},
            }
        skipped_water_total = sum(water_skip_counts.values())
        if skipped_water_total > 0:
            logger.info("Skipped water features: %s", water_skip_counts)

    G_proj = ox.project_graph(G, to_crs=utm_crs)
    for _, data in G_proj.nodes(data=True):
        data["x_local"] = float(data["x"]) - origin[0]
        data["y_local"] = float(data["y"]) - origin[1]

    # Keep the full projected edge polyline in local coordinates so downstream
    # routing can follow curves instead of jumping node-to-node chords.  Also
    # tag each edge with its bridge layer offset (if any) so mobility can lift
    # vehicles onto the deck instead of leaving them at terrain height.
    for u_node, v_node, key, data in G_proj.edges(keys=True, data=True):
        u_data = G_proj.nodes[u_node]
        v_data = G_proj.nodes[v_node]
        local_polyline = _edge_local_polyline(data, u_data, v_data, origin[0], origin[1])
        data["geometry_local"] = json.dumps(local_polyline)
        offset = _edge_bridge_offset(data, bridges_enabled=bridges)
        data["bridge_z_offset"] = float(offset)
        data["is_bridge"] = "1" if offset > 0.0 else "0"

    # Drop street edges whose polyline intersects a building footprint.  OSM
    # tags many indoor parking-garage and atrium driveways as ``service``
    # roads, which routes shortest-path solvers happily walk through.  Stripping
    # them at build time gives mobility a clean graph and avoids interpolated
    # waypoints crossing through buildings even after snap_to_road.
    n_dropped = _drop_in_building_edges(G_proj, building_local_polys)
    if n_dropped:
        logger.info("Pruned %d street edges that crossed building footprints", n_dropped)

    # Prune the routing graph to the padded scene bbox so mobility never
    # samples a node outside the visible street mesh.  Without this,
    # rectangular scenes (e.g. scenarios/osm/champs_elysees) hand mobility nodes
    # near the corners of the osmnx graph_from_point disc — far outside
    # the bbox — and the device ends up walking on roads with no
    # rendered pavement.  The street-mesh clipping (which uses
    # scene_bbox_local) then becomes the geometric source of truth.
    n_graph_pruned = _prune_graph_to_bbox(G_proj, scene_bbox_local)
    if n_graph_pruned:
        logger.info("Pruned %d street nodes outside the padded scene bbox", n_graph_pruned)

    n_edges_clipped, n_edges_dropped, n_isolated_after_clip = _sanitize_graph_edges_to_bbox(
        G_proj, scene_bbox_local
    )
    if n_edges_clipped or n_edges_dropped:
        logger.info(
            "Sanitized %d street edges to the padded scene bbox (%d clipped, "
            "%d dropped, %d isolated nodes removed)",
            n_edges_clipped + n_edges_dropped,
            n_edges_clipped,
            n_edges_dropped,
            n_isolated_after_clip,
        )

    graph_path = os.path.join(output_dir, "street_network.graphml")
    ox.save_graphml(G_proj, graph_path)
    logger.info(
        "Street network: %d nodes, %d edges",
        len(G_proj.nodes),
        len(G_proj.edges),
    )

    # -- Build street mesh from network edge polylines -------------------------
    # Group street segments by surface material.
    street_mat_groups: dict[str, dict] = {}
    total_street_segments = 0
    visual_road_faces: dict[str, dict] = {}

    bridge_count = 0
    tunnel_count = 0

    for u_node, v_node, data in G_proj.edges(data=True):
        # Skip tunnels entirely (underground, no mesh needed).
        is_tunnel = str(data.get("tunnel", "")).lower() in ("yes", "true", "1")
        if is_tunnel:
            tunnel_count += 1
            continue

        u_data = G_proj.nodes[u_node]
        v_data = G_proj.nodes[v_node]
        local_polyline = _edge_local_polyline(data, u_data, v_data, origin[0], origin[1])
        if len(local_polyline) < 2:
            continue

        # Clip the edge polyline against the padded scene bbox so long
        # OSM ways (highways, boulevards) that stretch beyond the bbox
        # don't extrude street pavement past the ground mesh edge.
        try:
            edge_line = _LineString(local_polyline)
            clipped_lines = _clip_linestring_to_bbox(edge_line, scene_bbox_local)
        except Exception:
            clipped_lines = []
        if not clipped_lines:
            continue
        # Use the longest clipped piece as the representative polyline so
        # the per-segment loop below sees coherent geometry.  Multi-piece
        # results (where the edge enters and leaves the bbox) get every
        # piece processed by repeating the loop body via concatenation.
        local_polylines: list[list[tuple[float, float]]] = []
        for line in clipped_lines:
            coords = list(line.coords)
            if len(coords) >= 2:
                local_polylines.append([(float(x), float(y)) for x, y in coords])
        if not local_polylines:
            continue

        # Bridge offset was precomputed during graph projection and persisted
        # on the edge so routing/mobility see the same lift as the mesh.
        bridge_z_offset = float(data.get("bridge_z_offset", 0.0) or 0.0)
        is_bridge = bridge_z_offset > 0.0
        if is_bridge:
            bridge_count += 1

        # Pedestrian corridors need narrower defaults than motor roads.
        width_m, width_source = street_width(data)
        hw = width_m / 2

        # Determine street material from surface tag.
        # Bridges use concrete by default (deck material).
        if is_bridge:
            street_material = "concrete"
        else:
            surface = data.get("surface")
            if isinstance(surface, str):
                street_material = _SURFACE_MATERIAL_MAP.get(
                    surface.strip().lower(), DEFAULT_STREET_MATERIAL
                )
            else:
                street_material = DEFAULT_STREET_MATERIAL

        if street_material not in street_mat_groups:
            street_mat_groups[street_material] = {
                "vertices": [],
                "faces": [],
                "vcount": 0,
            }
        grp = street_mat_groups[street_material]
        segments = grp.setdefault("visual_segments", [])

        for clipped_polyline in local_polylines:
            if heightmap is not None:
                # Long street quads otherwise bridge over terrain valleys or
                # cut through hills between their endpoint height samples.
                # Keep the original graph and RF material; refine only the
                # terrain-following mesh, preserving road arc-length metadata.
                clipped_polyline = list(
                    _LineString(clipped_polyline).segmentize(TERRAIN_STREET_MAX_SEGMENT_M).coords
                )
            edge_length = sum(
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(clipped_polyline[:-1], clipped_polyline[1:])
            )
            edge_distance = 0.0
            for (x0, y0), (x1, y1) in zip(clipped_polyline[:-1], clipped_polyline[1:]):
                dx = x1 - x0
                dy = y1 - y0
                seg_len = math.sqrt(dx * dx + dy * dy)
                if seg_len < 0.1:
                    continue
                nx_dir = -dy / seg_len * hw
                ny_dir = dx / seg_len * hw

                z_base = bridge_z_offset  # 0 for normal streets, >0 for bridges
                verts = np.array(
                    [
                        [x0 + nx_dir, y0 + ny_dir, z_base],
                        [x0 - nx_dir, y0 - ny_dir, z_base],
                        [x1 - nx_dir, y1 - ny_dir, z_base],
                        [x1 + nx_dir, y1 + ny_dir, z_base],
                    ],
                    dtype=np.float32,
                )
                faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32) + grp["vcount"]

                segments.append(
                    {
                        "first_triangle": 2 * len(grp["faces"]),
                        "start_m": edge_distance,
                        "end_m": edge_distance + seg_len,
                        "length_m": edge_length,
                        "highway": str(data.get("highway", "")),
                        "surface": str(data.get("_visual_surface", data.get("surface", ""))),
                        "lanes": str(data.get("lanes", "")),
                        "oneway": str(data.get("oneway", "")).lower() in ("yes", "true", "1"),
                        "width_m": 2 * hw,
                        "width_source": width_source,
                    }
                )
                edge_distance += seg_len
                grp["vertices"].append(verts)
                grp["faces"].append(faces)
                grp["vcount"] += 4
                total_street_segments += 1

    for street_material, grp in street_mat_groups.items():
        if grp["vcount"] == 0:
            continue
        all_sv = np.concatenate(grp["vertices"], axis=0)
        # Z column already has bridge_z_offset (0 for normal streets).
        # Add terrain height + 0.15m z-fighting offset on top.
        if heightmap is not None:
            terrain_z = _sample_local_terrain_heights(heightmap, origin, origin_z, all_sv[:, :2])
            all_sv[:, 2] += terrain_z + 0.15
        else:
            all_sv[:, 2] += 0.15
        all_sf = np.concatenate(grp["faces"], axis=0)

        street_mesh = o3d.t.geometry.TriangleMesh()
        street_mesh.vertex.positions = o3d.core.Tensor(all_sv)
        street_mesh.triangle.indices = o3d.core.Tensor(all_sf)

        suffix = f"_{street_material}" if len(street_mat_groups) > 1 else ""
        ply_name = f"streets{suffix}.ply"
        o3d.t.io.write_triangle_mesh(os.path.join(mesh_dir, ply_name), street_mesh)
        _add_shape(scene_el, f"mesh-streets{suffix}", f"mesh/{ply_name}", street_material)
        mesh_files.append(f"mesh/{ply_name}")
        count = int(all_sf.shape[0])
        visual_geographic_layers[f"mesh/{ply_name}"] = {
            "total_triangles": count,
            "layers": {"roads": [[0, count]]},
        }
        visual_road_faces[f"mesh/{ply_name}"] = {
            "total_triangles": int(all_sf.shape[0]),
            "segments": grp["visual_segments"],
        }

    if total_street_segments > 0:
        mat_dist = ", ".join(
            f"{m}: {g['vcount'] // 4}" for m, g in street_mat_groups.items() if g["vcount"]
        )
        logger.info(
            "Street mesh: %d segments, %d vertices (%s)",
            total_street_segments,
            sum(g["vcount"] for g in street_mat_groups.values()),
            mat_dist,
        )
    if bridge_count > 0 or tunnel_count > 0:
        logger.info("Bridges: %d elevated, tunnels: %d skipped", bridge_count, tunnel_count)

    # -- Vegetation (trees, forests) ------------------------------------------
    # Must run BEFORE PLY write so vegetation ends up in mat_groups.
    if vegetation:
        vegetation_tags = {
            "natural": ["tree", "tree_row", "wood"],
            "landuse": ["forest"],
            "leisure": ["park"],
        }
        tree_gdf = _cached_osm_query(
            "vegetation",
            bbox=osm_bbox,
            tags=vegetation_tags,
            fetch_fn=lambda: feature_batch.get("vegetation", vegetation_tags),
            serializer=_GDF_SERIALIZER,
            optional=True,
            no_data_value=None,
            cache_dir=osm_cache_dir,
            refresh=osm_refresh_cache,
            stats=cache_stats,
        )
        if tree_gdf is not None:
            tree_gdf = tree_gdf.to_crs(utm_crs)
            tree_list = tree_gdf.to_dict("records")
        else:
            tree_list = []

        # Counters and constants live OUTSIDE the ``if tree_list:`` block so
        # the WorldCover gap-fill pass below can run on a sparse scene where
        # OSM returned zero vegetation features.
        tree_count = 0
        forest_count = 0
        tree_row_count = 0
        gap_fill_count = 0
        default_tree_height = 8.0
        default_crown_radius = 3.0
        trunk_radius = 0.3
        trunk_height_frac = 0.3  # trunk is 30% of total height
        tree_material = "vegetation" if green_vegetation else "wood"
        osm_vegetation_geoms: list = []

        if tree_list:
            logger.info("Processing %d vegetation features...", len(tree_list))

            # Vegetation meshes are routed to the ``vegetation`` material group
            # when green_vegetation=True so the visualizer can apply a green PBR
            # override keyed on filename.  Sionna RT still sees mat-itu_wood
            # because the XML material reference is emitted as such; only the
            # PLY filename differs.

            # ``tree_material`` and ``osm_vegetation_geoms`` are now declared
            # above the ``if tree_list:`` block so the WorldCover gap-fill
            # below can run on a sparse scene with zero OSM vegetation.

            def _resolve_tree_height(feature_obj, lon_lat=None) -> float:
                """Apply the canopy height fallback chain: OSM tag > ETH > default."""
                tag_val = feature_obj.get("height") if feature_obj else None
                if tag_val is not None:
                    try:
                        h = float(tag_val)
                        if math.isfinite(h) and 1 < h < 100:
                            return h
                    except (TypeError, ValueError):
                        pass
                if canopy_index is not None and lon_lat is not None:
                    lon, lat = lon_lat
                    cheight = canopy_index.height_at(lon, lat)
                    if cheight is not None and 1 < cheight < 80:
                        return float(cheight)
                return default_tree_height

            def _base_z_at(ux: float, uy: float) -> float:
                if heightmap is None:
                    return 0.0
                return float(np.array(heightmap.height_from_utm(ux, uy)).ravel()[0]) - origin_z

            for feature in tree_list:
                geom = shape(feature["geometry"]) if feature.get("geometry") else None
                if geom is None:
                    continue

                tag_type = (
                    feature.get("natural") or feature.get("landuse") or feature.get("leisure")
                )

                if geom.geom_type == "Point" and tag_type == "tree":
                    cx = geom.x - origin[0]
                    cy = geom.y - origin[1]
                    base_z = _base_z_at(geom.x, geom.y)
                    lon_lat = to_wgs84.transform(geom.x, geom.y)
                    tree_h = _resolve_tree_height(feature, lon_lat)
                    _build_tree_at(
                        cx,
                        cy,
                        base_z,
                        tree_h,
                        default_crown_radius,
                        trunk_radius,
                        trunk_height_frac,
                        mat_groups,
                        material=tree_material,
                    )
                    tree_count += 1

                elif (
                    tree_rows
                    and tag_type == "tree_row"
                    and geom.geom_type in ("LineString", "MultiLineString")
                ):
                    # natural=tree_row: sample evenly along the polyline and
                    # plant one tree per sample.  OSM spacing is implicit —
                    # we use tree_row_spacing_m (default 8 m) unless the
                    # feature carries an explicit ``spacing`` tag.
                    feat_spacing = feature.get("spacing")
                    if feat_spacing is not None:
                        try:
                            parsed_spacing = float(str(feat_spacing).split()[0])
                            if 1.0 < parsed_spacing < 50.0:
                                spacing_m = parsed_spacing
                            else:
                                spacing_m = tree_row_spacing_m
                        except (TypeError, ValueError, IndexError):
                            spacing_m = tree_row_spacing_m
                    else:
                        spacing_m = tree_row_spacing_m

                    for line in _iter_linestring_parts(feature["geometry"]):
                        pts = _sample_line_points(line, spacing_m=spacing_m)
                        if not pts:
                            continue
                        # Record the tree-row buffer for gap-fill exclusion.
                        try:
                            osm_vegetation_geoms.append(line.buffer(spacing_m))
                        except Exception:
                            pass
                        for px_utm, py_utm in pts:
                            cx = px_utm - origin[0]
                            cy = py_utm - origin[1]
                            base_z = _base_z_at(px_utm, py_utm)
                            lon_lat = to_wgs84.transform(px_utm, py_utm)
                            tree_h = _resolve_tree_height(feature, lon_lat)
                            _build_tree_at(
                                cx,
                                cy,
                                base_z,
                                tree_h,
                                default_crown_radius,
                                trunk_radius,
                                trunk_height_frac,
                                mat_groups,
                                material=tree_material,
                            )
                            tree_row_count += 1

                elif geom.geom_type in ("Polygon", "MultiPolygon") and tag_type in (
                    "wood",
                    "forest",
                    "park",
                ):
                    # Scatter discrete trees inside the polygon instead of
                    # extruding it as a solid wood prism — a solid block would
                    # over-block LOS and over-attenuate paths through the area.
                    # Clip against the padded scene bbox first so a forest that
                    # extends well beyond the bbox doesn't seed thousands of
                    # trees outside the ground mesh.
                    for poly in _clip_polygon_to_bbox(feature["geometry"], scene_bbox_utm):
                        if poly.is_empty or poly.area < 4.0:
                            continue
                        osm_vegetation_geoms.append(poly)
                        density_trees_per_m2 = 0.02 if tag_type in ("wood", "forest") else 0.005
                        spacing = float(1.0 / math.sqrt(density_trees_per_m2))
                        scattered = _scatter_trees_in_polygon(
                            poly,
                            spacing_m=spacing,
                            jitter_frac=0.35,
                            seed=tree_count + forest_count + 1,
                        )
                        if not scattered:
                            continue
                        # One canopy-height sample per polygon (averaged) is
                        # plenty; we don't need to re-sample at every scatter
                        # point.
                        polygon_h = None
                        if canopy_index is not None:
                            polygon_h = canopy_index.height_over_polygon(poly, to_wgs84)
                        forest_tree_h = (
                            polygon_h
                            if polygon_h is not None and 1 < polygon_h < 80
                            else _resolve_tree_height(feature)
                        )
                        for px_utm, py_utm in scattered:
                            cx = px_utm - origin[0]
                            cy = py_utm - origin[1]
                            base_z = _base_z_at(px_utm, py_utm)
                            _build_tree_at(
                                cx,
                                cy,
                                base_z,
                                forest_tree_h,
                                default_crown_radius,
                                trunk_radius,
                                trunk_height_frac,
                                mat_groups,
                                material=tree_material,
                            )
                            tree_count += 1
                        forest_count += 1

        # -- WorldCover vegetation gap-fill ----------------------------------
        # Runs OUTSIDE the ``if tree_list:`` block so it works on the exact
        # sparse-scene case it was designed for: OSM has zero vegetation
        # tagged but WorldCover says there are tree-cover or shrubland cells
        # in the bbox.  When OSM also returned trees, ``osm_vegetation_geoms``
        # carries the OSM polygons + tree-row buffers so gap-fill skips
        # cells OSM already covered.
        if vegetation_gap_fill and worldcover_index is not None and worldcover_index.is_available():
            gap_fill_count = _apply_worldcover_gap_fill(
                worldcover_index=worldcover_index,
                osm_vegetation_geoms=osm_vegetation_geoms,
                to_utm=to_utm,
                to_wgs84=to_wgs84,
                origin=origin,
                heightmap=heightmap,
                origin_z=origin_z,
                canopy_index=canopy_index,
                default_tree_height=default_tree_height,
                default_crown_radius=default_crown_radius,
                trunk_radius=trunk_radius,
                trunk_height_frac=trunk_height_frac,
                mat_groups=mat_groups,
                tree_material=tree_material,
            )

        if tree_count or forest_count or tree_row_count or gap_fill_count:
            logger.info(
                "Vegetation: %d trees, %d forest/park areas, %d tree_row "
                "samples, %d gap-filled from WorldCover",
                tree_count,
                forest_count,
                tree_row_count,
                gap_fill_count,
            )

    # -- Write one PLY per material group (buildings + barriers + vegetation) ---
    total_meshes = 0
    for material, grp in mat_groups.items():
        if grp["vcount"] == 0:
            continue
        n_meshes = len(grp["vertices"])
        total_meshes += n_meshes

        if n_meshes > 1:
            combined_v = o3d.core.concatenate(grp["vertices"], axis=0)
            combined_f = o3d.core.concatenate(grp["faces"], axis=0)
        else:
            combined_v = grp["vertices"][0]
            combined_f = grp["faces"][0]

        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = combined_v
        mesh.triangle.indices = combined_f

        # The "vegetation" material group gets a dedicated filename
        # (``mesh/vegetation.ply``) so the visualizer's PBR override — which
        # is keyed on PLY filename — can apply a green color without
        # touching building meshes.  Sionna RT still sees ``mat-itu_wood``
        # to preserve the existing RF approximation; this visual option
        # does not introduce a foliage-specific propagation model.
        if material == "vegetation":
            ply_name = "vegetation.ply"
            shape_id = "mesh-vegetation"
            sionna_material = "wood"
        else:
            ply_name = f"buildings_{material}.ply"
            shape_id = f"mesh-buildings-{material}"
            sionna_material = material
        o3d.t.io.write_triangle_mesh(os.path.join(mesh_dir, ply_name), mesh)
        _add_shape(scene_el, shape_id, f"mesh/{ply_name}", sionna_material)
        mesh_files.append(f"mesh/{ply_name}")
        count = int(combined_f.shape[0])
        building_end = building_face_counts.get(material, 0)
        barrier_end = barrier_end_counts.get(material, building_end)
        ranges = {}
        for layer, start, end in (
            ("buildings", 0, building_end),
            ("barriers", building_end, barrier_end),
            ("vegetation", barrier_end, count),
        ):
            if end > start:
                ranges[layer] = [[start, end - start]]
        visual_geographic_layers[f"mesh/{ply_name}"] = {"total_triangles": count, "layers": ranges}
        classification = visual_building_faces.get(f"mesh/{ply_name}")
        if classification is not None:
            classification["total_triangles"] = int(combined_f.shape[0])

    if total_meshes > 0:
        dist_str = ", ".join(
            f"{m}: {len(g['vertices'])}" for m, g in mat_groups.items() if g["vcount"]
        )
        logger.info("Scene meshes: %d total (%s)", total_meshes, dist_str)

    # Save scene metadata for downstream tools (mobility, analysis).
    # When the caller didn't pre-normalize a config (i.e. direct API
    # use, not the YAML/scenario-loader path), build one from *every*
    # relevant build-time kwarg so the saved metadata matches what was
    # actually built — see ``_build_default_metadata_config`` for the
    # full field list.
    metadata_config = metadata_config or _build_default_metadata_config(
        bbox=list(bbox),
        center=None,
        size_m=None,
        width_m=None,
        terrain=terrain,
        terrain_zoom=terrain_zoom,
        ground_pad_m=ground_pad_m,
        ground_resolution_m=ground_resolution_m,
        osm_server=osm_server,
        barriers=barriers,
        vegetation=vegetation,
        bridges=bridges,
        height_enrichment=height_enrichment,
        canopy_heights=canopy_heights,
        tree_rows=tree_rows,
        vegetation_gap_fill=vegetation_gap_fill,
        worldcover_water_check=worldcover_water_check,
        green_vegetation=green_vegetation,
        tree_row_spacing_m=tree_row_spacing_m,
        landuse_materials=landuse_materials,
    )
    # -- Compute building buffer for trajectory snapping -------------------------
    building_buffer_coords: list[list[list[float]]] = []
    if building_local_polys:
        from shapely.ops import unary_union

        try:
            valid_polys = [p for p in building_local_polys if p.is_valid and p.area > 1.0]
            if valid_polys:
                buffer_union = unary_union(valid_polys).buffer(2.0).simplify(1.0)
                if buffer_union.geom_type == "Polygon":
                    building_buffer_coords = [
                        [[float(x), float(y)] for x, y in buffer_union.exterior.coords]
                    ]
                elif buffer_union.geom_type == "MultiPolygon":
                    building_buffer_coords = [
                        [[float(x), float(y)] for x, y in poly.exterior.coords]
                        for poly in buffer_union.geoms
                    ]
                logger.info(
                    "Building buffer: %d polygons from %d footprints",
                    len(building_buffer_coords),
                    len(valid_polys),
                )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Building buffer computation failed (%s), skipping", exc)

    # Fingerprints prevent stale display selections after a source mesh changes.
    for rel_path, record in visual_geographic_layers.items():
        record["source_sha256"] = hashlib.sha256(
            (Path(output_dir) / rel_path).read_bytes()
        ).hexdigest()

    meta = {
        "schema_version": OSM_METADATA_SCHEMA_VERSION,
        "osm_config": metadata_config,
        "bbox": list(bbox),
        "origin_utm": [float(origin[0]), float(origin[1]), float(origin_z)],
        "utm_crs": str(utm_crs),
        "center_latlon": [center_lat, center_lon],
        "artifacts": {
            "scene_xml": "scene.xml",
            "street_network_graphml": "street_network.graphml",
            "mesh_files": mesh_files,
        },
        "terrain_grid": (
            {
                "enabled": True,
                **terrain_grid_meta,
            }
            if terrain_grid_meta is not None
            else {"enabled": False}
        ),
        "building_buffer": building_buffer_coords,
        "visual_building_faces": visual_building_faces,
        "visual_road_faces": visual_road_faces,
        "visual_geographic_layers": visual_geographic_layers,
        "barrier_features": barrier_features,
        "osm_cache_present": cache_stats.get("hits", 0) > 0,
        "osm_cache_stats": dict(cache_stats),
        "acquisition": {
            "complete": not any(value["outcome"] == "failed" for value in context.layers.values()),
            "layers": dict(context.layers),
            "requests": list(context.requests),
            "elapsed_s": time.monotonic() - context.started,
            "network_budget_remaining_s": max(0.0, context.remaining),
        },
        "external_data_sources": [dict(src) for src in _external_data_used],
    }
    with open(os.path.join(output_dir, "osm_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # -- Write scene XML ------------------------------------------------------
    scene_xml_path = os.path.join(output_dir, "scene.xml")
    _write_mitsuba_xml(scene_el, scene_xml_path)
    logger.info("Scene written to %s", scene_xml_path)

    _log_scene_complexity(
        output_dir=output_dir,
        mesh_files=mesh_files,
        mat_groups=mat_groups,
        street_mat_groups=street_mat_groups,
        buildings_count=len(buildings_list),
        graph=G_proj,
        bbox=bbox,
    )
    logger.info(
        "OSM cache: %d hits, %d misses (%s)",
        cache_stats.get("hits", 0),
        cache_stats.get("misses", 0),
        osm_cache_dir,
    )

    return scene_xml_path


def build_scene_from_center(
    lat: float,
    lon: float,
    size_m: float,
    output_dir: str,
    *,
    width_m: float | None = None,
    metadata_config: dict | None = None,
    **kwargs,
) -> str:
    """Convenience wrapper: build a scene centered on a GPS point.

    Parameters
    ----------
    lat, lon : float
        Center point in WGS84 degrees.
    size_m : float
        Scene side length in meters.
    output_dir : str
        Output directory.
    width_m : float, optional
        East-west dimension. If ``None``, same as *size_m*.
    **kwargs
        Passed through to :func:`build_scene`.

    Returns
    -------
    str
        Path to the generated ``scene.xml``.
    """
    bbox = bbox_from_center(lat, lon, size_m, width_m=width_m)
    logger.info(
        "Center (%.4f, %.4f), size %.0fm → bbox (%.4f, %.4f, %.4f, %.4f)",
        lat,
        lon,
        size_m,
        *bbox,
    )
    # Same rule as the build_scene fallback: when the caller hasn't
    # pre-normalized a config (direct API use), build one from every
    # relevant feature kwarg so the saved metadata matches the
    # actually-built scene.
    metadata_config = metadata_config or _build_default_metadata_config(
        bbox=None,
        center=[lat, lon],
        size_m=size_m,
        width_m=width_m,
        terrain=kwargs.get("terrain", False),
        terrain_zoom=kwargs.get("terrain_zoom", 14),
        ground_pad_m=kwargs.get("ground_pad_m", 200.0),
        ground_resolution_m=kwargs.get("ground_resolution_m", 50.0),
        osm_server=kwargs.get("osm_server"),
        barriers=kwargs.get("barriers", True),
        vegetation=kwargs.get("vegetation", True),
        bridges=kwargs.get("bridges", True),
        height_enrichment=kwargs.get("height_enrichment", True),
        canopy_heights=kwargs.get("canopy_heights", False),
        tree_rows=kwargs.get("tree_rows", False),
        vegetation_gap_fill=kwargs.get("vegetation_gap_fill", False),
        worldcover_water_check=kwargs.get("worldcover_water_check", False),
        green_vegetation=kwargs.get("green_vegetation", True),
        tree_row_spacing_m=kwargs.get("tree_row_spacing_m", 8.0),
        landuse_materials=kwargs.get("landuse_materials", True),
    )
    return build_scene(bbox, output_dir, metadata_config=metadata_config, **kwargs)
