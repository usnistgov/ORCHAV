"""Unit tests for OSM vegetation realism helpers.

These tests build tiny synthetic GeoTIFFs in tmp_path via rasterio so
the OSM fetcher classes (`_EthCanopyHeightIndex`, `_WorldCoverIndex`)
can be exercised without hitting the real ETH or ESA endpoints.  Real
network downloads live under the ``osm_network`` pytest marker and are
deselected by default.

The tests also cover pure geometry and attribution helpers:
``_iter_linestring_parts``, ``_sample_line_points``, the attribution
helper ``_cite_external_dataset``, and the green-vegetation material
routing in ``_build_tree_at``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

osmnx = pytest.importorskip("osmnx")
rasterio = pytest.importorskip("rasterio")

from shapely.geometry import (  # noqa: E402
    LineString,
    MultiLineString,
    Polygon,
)

from generator.osm import scene_builder  # noqa: E402
from generator.osm.scene_builder import (  # noqa: E402
    _WC_SHRUBLAND,
    _WC_TREE_COVER,
    _build_tree_at,
    _cite_external_dataset,
    _EthCanopyHeightIndex,
    _format_tile_coord,
    _iter_linestring_parts,
    _sample_line_points,
    _tile_coords_for_bbox,
    _WorldCoverIndex,
)

# ----------------------------------------------------------------------------
# Helpers to build synthetic GeoTIFF fixtures
# ----------------------------------------------------------------------------


def _write_tile(
    path: Path,
    data: np.ndarray,
    *,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    dtype: str = "uint8",
) -> None:
    """Write a single-band GeoTIFF covering the given WGS84 bbox."""
    from rasterio.transform import from_bounds

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, data.shape[1], data.shape[0])
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


class _FakeTransformer:
    """Minimal pyproj Transformer stand-in that treats UTM as identical to WGS84.

    We use it in tests where we don't need a real CRS projection — the
    raster tiles are written in EPSG:4326 and we pass lon/lat directly
    as UTM coords.
    """

    @staticmethod
    def transform(x, y):
        return x, y


# ----------------------------------------------------------------------------
# Line helpers
# ----------------------------------------------------------------------------


def test_iter_linestring_parts_handles_single_and_multi():
    line = LineString([(0, 0), (10, 0)])
    assert _iter_linestring_parts(line) == [line]

    multi = MultiLineString([[(0, 0), (10, 0)], [(20, 0), (30, 0)]])
    parts = _iter_linestring_parts(multi)
    assert len(parts) == 2
    assert parts[0].length == pytest.approx(10.0)


def test_iter_linestring_parts_rejects_polygons():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert _iter_linestring_parts(poly) == []


def test_sample_line_points_uniform_spacing():
    line = LineString([(0, 0), (100, 0)])
    pts = _sample_line_points(line, spacing_m=25.0)
    xs = [p[0] for p in pts]
    # Includes both endpoints plus inner samples at 0, 25, 50, 75, 100.
    assert xs == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_sample_line_points_short_line_returns_empty():
    line = LineString([(0, 0), (3, 0)])
    assert _sample_line_points(line, spacing_m=10.0) == []


def test_sample_line_points_polyline_traces_corners():
    line = LineString([(0, 0), (100, 0), (100, 50)])
    pts = _sample_line_points(line, spacing_m=50.0)
    # Total length = 150; samples at 0, 50, 100, 150 → 4 points along the path.
    assert len(pts) == 4
    # First sample on origin, last sample on end, middle samples follow the corner.
    assert pts[0] == (0.0, 0.0)
    assert pts[-1] == pytest.approx((100.0, 50.0))


# ----------------------------------------------------------------------------
# Tile-coord helpers
# ----------------------------------------------------------------------------


def test_tile_coord_format_matches_dataset_naming():
    assert _format_tile_coord(48, 0) == "N48E000"
    assert _format_tile_coord(48, 3) == "N48E003"
    assert _format_tile_coord(39, -75) == "N39W075"
    assert _format_tile_coord(-33, 18) == "S33E018"


def test_tile_coords_for_bbox_paris_single_tile():
    # Île de la Cité (min_lon, min_lat, max_lon, max_lat).
    bbox = (2.34, 48.85, 2.36, 48.86)
    tiles = _tile_coords_for_bbox(bbox)
    assert tiles == [(48, 0)]


def test_tile_coords_for_bbox_straddling_meridian():
    bbox = (2.9, 48.9, 3.1, 49.1)
    tiles = _tile_coords_for_bbox(bbox)
    # The bbox crosses both the 3° lon boundary and a 3° lat boundary.
    assert set(tiles) == {(48, 0), (48, 3)}


# ----------------------------------------------------------------------------
# _EthCanopyHeightIndex
# ----------------------------------------------------------------------------


class _StubEthIndex(_EthCanopyHeightIndex):
    """ETH canopy index that loads a pre-written tile from disk instead of HTTP."""

    def __init__(self, bbox_wgs84, tile_path: Path, tile_key: tuple[int, int]):
        # Bypass the parent __init__ — we have no network.
        self._bbox = bbox_wgs84
        self._datasets = {}
        self._empty = True
        self._datasets[tile_key] = rasterio.open(str(tile_path))
        self._empty = False


def test_eth_canopy_height_at_known_values(tmp_path: Path):
    # Build a 10×10 canopy-height tile: row i carries height (i+1)*2 m.
    data = np.zeros((10, 10), dtype=np.uint8)
    for r in range(10):
        data[r, :] = (r + 1) * 2
    tile_path = tmp_path / "ETH_GlobalCanopyHeight_10m_2020_N48E000_Map.tif"
    _write_tile(
        tile_path,
        data,
        lon_min=0.0,
        lat_min=48.0,
        lon_max=3.0,
        lat_max=51.0,
    )

    idx = _StubEthIndex((0.0, 48.0, 3.0, 51.0), tile_path, (48, 0))
    # Sample near the center of the tile.
    h = idx.height_at(1.5, 49.5)
    assert h is not None
    assert 2.0 <= h <= 20.0

    # Zero-valued cells should return None (fall-through to default).
    tile_zero_path = tmp_path / "zero.tif"
    _write_tile(
        tile_zero_path,
        np.zeros((5, 5), dtype=np.uint8),
        lon_min=0.0,
        lat_min=48.0,
        lon_max=3.0,
        lat_max=51.0,
    )
    idx_zero = _StubEthIndex((0.0, 48.0, 3.0, 51.0), tile_zero_path, (48, 0))
    assert idx_zero.height_at(1.5, 49.5) is None


def test_eth_canopy_height_over_polygon_samples_centroid(tmp_path: Path):
    data = np.full((10, 10), 15, dtype=np.uint8)
    tile_path = tmp_path / "canopy.tif"
    _write_tile(
        tile_path,
        data,
        lon_min=0.0,
        lat_min=48.0,
        lon_max=3.0,
        lat_max=51.0,
    )
    idx = _StubEthIndex((0.0, 48.0, 3.0, 51.0), tile_path, (48, 0))

    poly = Polygon([(1.0, 49.0), (2.0, 49.0), (2.0, 50.0), (1.0, 50.0)])
    h = idx.height_over_polygon(poly, _FakeTransformer())
    assert h == pytest.approx(15.0)


def test_eth_canopy_fall_through_outside_tile(tmp_path: Path):
    data = np.full((5, 5), 10, dtype=np.uint8)
    tile_path = tmp_path / "canopy.tif"
    _write_tile(
        tile_path,
        data,
        lon_min=0.0,
        lat_min=48.0,
        lon_max=3.0,
        lat_max=51.0,
    )
    idx = _StubEthIndex((0.0, 48.0, 3.0, 51.0), tile_path, (48, 0))
    # Outside the tile footprint → None.
    assert idx.height_at(10.0, 60.0) is None


# ----------------------------------------------------------------------------
# _WorldCoverIndex
# ----------------------------------------------------------------------------


class _StubWorldCoverIndex(_WorldCoverIndex):
    def __init__(self, bbox_wgs84, tile_path: Path, tile_key: tuple[int, int]):
        self._bbox = bbox_wgs84
        self._datasets = {}
        self._empty = True
        self._datasets[tile_key] = rasterio.open(str(tile_path))
        self._empty = False


def test_worldcover_class_at_reads_known_codes(tmp_path: Path):
    # 2×2 tile with four different classes.
    data = np.array(
        [
            [_WC_TREE_COVER, _WC_SHRUBLAND],
            [80, 30],  # permanent_water, grassland
        ],
        dtype=np.uint8,
    )
    tile_path = tmp_path / "worldcover.tif"
    _write_tile(
        tile_path,
        data,
        lon_min=0.0,
        lat_min=48.0,
        lon_max=2.0,
        lat_max=50.0,
    )
    idx = _StubWorldCoverIndex((0.0, 48.0, 2.0, 50.0), tile_path, (48, 0))

    # Raster rows go north → south (top row is higher lat), so the top-left
    # pixel covers roughly (0.0-1.0 lon, 49.0-50.0 lat).  Sample well inside.
    assert idx.class_at(0.5, 49.5) == _WC_TREE_COVER
    assert idx.class_at(1.5, 49.5) == _WC_SHRUBLAND
    assert idx.class_at(0.5, 48.5) == 80
    assert idx.class_at(1.5, 48.5) == 30


def test_worldcover_iter_class_cells_yields_only_wanted(tmp_path: Path):
    data = np.array(
        [
            [_WC_TREE_COVER, _WC_SHRUBLAND, 30],
            [_WC_TREE_COVER, 30, 50],
            [30, 30, 30],
        ],
        dtype=np.uint8,
    )
    tile_path = tmp_path / "worldcover.tif"
    _write_tile(
        tile_path,
        data,
        lon_min=0.0,
        lat_min=48.0,
        lon_max=3.0,
        lat_max=51.0,
    )
    idx = _StubWorldCoverIndex((0.0, 48.0, 3.0, 51.0), tile_path, (48, 0))

    cells = list(idx.iter_class_cells((_WC_TREE_COVER,), _FakeTransformer(), _FakeTransformer()))
    # Two tree_cover cells in the synthetic raster.
    assert len(cells) == 2
    for _ux, _uy, cls in cells:
        assert cls == _WC_TREE_COVER


# ----------------------------------------------------------------------------
# Attribution helper
# ----------------------------------------------------------------------------


def test_cite_external_dataset_deduplicates(caplog):
    scene_builder._external_data_used = []
    attribution = {
        "name": "Synthetic Test Dataset",
        "license": "CC-BY 4.0",
        "citation": "Test et al. 2026",
    }
    import logging

    with caplog.at_level(logging.INFO, logger="generator.osm.scene_builder"):
        _cite_external_dataset(attribution)
        _cite_external_dataset(attribution)  # duplicate — must not re-log

    # Exactly one attribution log line, exactly one recorded entry.
    name_hits = [r for r in caplog.records if "Synthetic Test Dataset" in r.getMessage()]
    assert len(name_hits) == 1
    assert [e["name"] for e in scene_builder._external_data_used] == ["Synthetic Test Dataset"]


def test_cite_external_dataset_accumulates_multiple_sources():
    scene_builder._external_data_used = []
    _cite_external_dataset({"name": "One", "license": "CC-BY 4.0", "citation": "a"})
    _cite_external_dataset({"name": "Two", "license": "CC-BY 4.0", "citation": "b"})
    names = [e["name"] for e in scene_builder._external_data_used]
    assert names == ["One", "Two"]


# ----------------------------------------------------------------------------
# Green vegetation material routing
# ----------------------------------------------------------------------------


def test_build_tree_at_routes_to_named_material_group():
    mat_groups: dict[str, dict] = {}
    _build_tree_at(
        cx=0.0,
        cy=0.0,
        base_z=0.0,
        tree_h=8.0,
        canopy_r=3.0,
        trunk_radius=0.3,
        trunk_height_frac=0.3,
        mat_groups=mat_groups,
        material="vegetation",
    )
    assert "vegetation" in mat_groups
    assert "wood" not in mat_groups
    assert mat_groups["vegetation"]["vcount"] > 0


def test_build_tree_at_default_material_unchanged():
    mat_groups: dict[str, dict] = {}
    _build_tree_at(
        cx=0.0,
        cy=0.0,
        base_z=0.0,
        tree_h=8.0,
        canopy_r=3.0,
        trunk_radius=0.3,
        trunk_height_frac=0.3,
        mat_groups=mat_groups,
    )
    # Default material remains "wood" for backward compat.
    assert "wood" in mat_groups
    assert "vegetation" not in mat_groups
