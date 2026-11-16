"""Tests for landuse-informed material defaults.

The lookup chain in ``_building_material`` is:

1. Explicit ``building:material`` tag.
2. ``building=*`` type map (industrial / office / church / etc.).
3. Land-use context via ``_LandUseMaterialIndex`` (the new layer).
4. ``DEFAULT_BUILDING_MATERIAL`` (concrete).

These tests exercise every branch and verify the new layer is correctly
gated by the polygon centroid containment check.
"""

from __future__ import annotations

import pytest

osmnx = pytest.importorskip("osmnx")

from shapely.geometry import Polygon  # noqa: E402

from generator.osm.metadata import normalize_osm_config  # noqa: E402
from generator.osm.scene_builder import (  # noqa: E402
    _OSM_LANDUSE_MATERIAL_MAP,
    DEFAULT_BUILDING_MATERIAL,
    _building_material,
    _LandUseMaterialIndex,
)

# ---------------------------------------------------------------------------
# _LandUseMaterialIndex
# ---------------------------------------------------------------------------


def _residential_record() -> dict:
    return {
        "geometry": Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)]),
        "landuse": "residential",
    }


def _industrial_record(offset: float = 200.0) -> dict:
    return {
        "geometry": Polygon(
            [
                (offset, offset),
                (offset + 100, offset),
                (offset + 100, offset + 100),
                (offset, offset + 100),
            ]
        ),
        "landuse": "industrial",
    }


def test_index_is_unavailable_when_records_empty():
    idx = _LandUseMaterialIndex([])
    assert idx.is_available() is False
    assert idx.material_for_point(0.0, 0.0) is None


def test_index_returns_residential_material_inside_polygon():
    idx = _LandUseMaterialIndex([_residential_record()])
    assert idx.is_available() is True
    assert idx.material_for_point(0.0, 0.0) == "brick"


def test_index_returns_none_outside_any_polygon():
    idx = _LandUseMaterialIndex([_residential_record()])
    # Point well outside the residential square.
    assert idx.material_for_point(500.0, 500.0) is None


def test_index_picks_correct_polygon_for_overlapping_classes():
    idx = _LandUseMaterialIndex([_residential_record(), _industrial_record()])
    assert idx.material_for_point(0.0, 0.0) == "brick"  # residential
    assert idx.material_for_point(250.0, 250.0) == "metal"  # industrial


def test_index_prefers_smallest_polygon_when_nested():
    """Nested landuse polygons: the more-specific (smaller-area) one wins.

    A small ``landuse=religious`` church parcel nested inside a large
    ``landuse=commercial`` district must return ``marble``, not ``glass``,
    regardless of STRtree iteration order.  This pins the deterministic
    precedence contract documented on ``material_for_point``.
    """
    big_commercial = {
        "geometry": Polygon([(-500, -500), (500, -500), (500, 500), (-500, 500)]),
        "landuse": "commercial",
    }
    small_religious = {
        "geometry": Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        "landuse": "religious",
    }
    # Order of insertion must not matter — try both.
    idx_a = _LandUseMaterialIndex([big_commercial, small_religious])
    idx_b = _LandUseMaterialIndex([small_religious, big_commercial])
    # Inside the small nested parcel: religious (marble) wins.
    assert idx_a.material_for_point(0.0, 0.0) == "marble"
    assert idx_b.material_for_point(0.0, 0.0) == "marble"
    # Outside the small parcel but inside the big district: commercial.
    assert idx_a.material_for_point(200.0, 200.0) == "glass"
    assert idx_b.material_for_point(200.0, 200.0) == "glass"


def test_index_equal_area_overlap_uses_stable_landuse_tie_breaker():
    """Equal-area overlaps must not depend on insertion/query order."""
    commercial = {
        "geometry": Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)]),
        "landuse": "commercial",
    }
    religious = {
        "geometry": Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)]),
        "landuse": "religious",
    }
    idx_a = _LandUseMaterialIndex([commercial, religious])
    idx_b = _LandUseMaterialIndex([religious, commercial])
    # Equal area ties fall back to normalized landuse tag order.
    assert idx_a.material_for_point(0.0, 0.0) == "glass"
    assert idx_b.material_for_point(0.0, 0.0) == "glass"


def test_index_skips_unknown_landuse_classes():
    idx = _LandUseMaterialIndex(
        [
            {
                "geometry": Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                "landuse": "quarry",  # not in _OSM_LANDUSE_MATERIAL_MAP
            }
        ]
    )
    assert idx.is_available() is False


# ---------------------------------------------------------------------------
# _building_material lookup chain
# ---------------------------------------------------------------------------


def _index_residential() -> _LandUseMaterialIndex:
    return _LandUseMaterialIndex([_residential_record()])


def test_building_material_explicit_tag_wins_over_landuse():
    idx = _index_residential()
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    material = _building_material({"building": "yes", "building:material": "glass"}, poly, idx)
    assert material == "glass"


def test_building_material_type_map_wins_over_landuse():
    idx = _index_residential()
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    # building=industrial → metal (type map), even inside residential landuse.
    assert _building_material({"building": "industrial"}, poly, idx) == "metal"


def test_building_material_falls_back_to_landuse_for_generic_building():
    idx = _index_residential()
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert _building_material({"building": "yes"}, poly, idx) == "brick"


def test_building_material_falls_back_to_default_when_outside_landuse():
    idx = _index_residential()
    # Polygon outside the residential square.
    far_poly = Polygon([(500, 500), (510, 500), (510, 510), (500, 510)])
    assert _building_material({"building": "yes"}, far_poly, idx) == DEFAULT_BUILDING_MATERIAL


def test_building_material_no_index_uses_existing_default():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    # Pre-Phase-B-2.5 behaviour: no landuse index, falls straight to default.
    assert _building_material({"building": "yes"}, poly, None) == DEFAULT_BUILDING_MATERIAL


def test_building_material_signature_back_compat():
    """The new optional args mustn't break existing single-arg callers."""
    assert _building_material({"building": "industrial"}) == "metal"
    assert _building_material({"building": "yes"}) == DEFAULT_BUILDING_MATERIAL


# ---------------------------------------------------------------------------
# Cache key + landuse map sanity
# ---------------------------------------------------------------------------


def test_normalize_osm_config_carries_landuse_materials_default_true():
    cfg = normalize_osm_config({"center": [48.85, 2.34]})
    assert cfg["landuse_materials"] is True


def test_normalize_osm_config_landuse_toggle_invalidates_cache_key():
    cfg_on = normalize_osm_config({"center": [48.85, 2.34], "landuse_materials": True})
    cfg_off = normalize_osm_config({"center": [48.85, 2.34], "landuse_materials": False})
    assert cfg_on != cfg_off


def test_default_metadata_config_preserves_non_default_landuse_flag():
    """``_build_default_metadata_config`` must round-trip OSM feature flags.

    The direct API path should persist the same normalized configuration that
    the scenario-loader path writes, including optional vegetation flags and
    landuse material defaults.
    """
    from generator.osm.scene_builder import _build_default_metadata_config

    cfg = _build_default_metadata_config(
        bbox=None,
        center=[38.8893, -77.022],
        size_m=2400.0,
        width_m=None,
        terrain=False,
        terrain_zoom=14,
        ground_pad_m=200.0,
        ground_resolution_m=50.0,
        osm_server=None,
        barriers=True,
        vegetation=True,
        bridges=True,
        height_enrichment=True,
        canopy_heights=True,
        tree_rows=True,
        vegetation_gap_fill=True,
        worldcover_water_check=True,
        green_vegetation=True,
        tree_row_spacing_m=6.5,  # non-default
        landuse_materials=False,  # non-default — the codex example
    )

    # Every non-default flag must persist in the normalized dict.
    assert cfg["landuse_materials"] is False
    assert cfg["canopy_heights"] is True
    assert cfg["tree_rows"] is True
    assert cfg["vegetation_gap_fill"] is True
    assert cfg["worldcover_water_check"] is True
    assert cfg["green_vegetation"] is True
    assert cfg["tree_row_spacing_m"] == 6.5
    # Core OSM flags also persist.
    assert cfg["barriers"] is True
    assert cfg["vegetation"] is True
    assert cfg["bridges"] is True
    assert cfg["height_enrichment"] is True

    # A second helper call with landuse_materials flipped must produce
    # a *different* dict — that's what makes the next reuse check
    # actually detect the change.
    flipped = _build_default_metadata_config(
        bbox=None,
        center=[38.8893, -77.022],
        size_m=2400.0,
        width_m=None,
        terrain=False,
        terrain_zoom=14,
        ground_pad_m=200.0,
        ground_resolution_m=50.0,
        osm_server=None,
        barriers=True,
        vegetation=True,
        bridges=True,
        height_enrichment=True,
        canopy_heights=True,
        tree_rows=True,
        vegetation_gap_fill=True,
        worldcover_water_check=True,
        green_vegetation=True,
        tree_row_spacing_m=6.5,
        landuse_materials=True,  # the only delta
    )
    assert flipped != cfg
    assert flipped["landuse_materials"] is True


def test_default_metadata_config_bbox_path_carries_flags_too():
    """The ``build_scene(bbox=..., metadata_config=None)`` path uses
    the same helper.  Verify the bbox-mode call signature also
    preserves optional vegetation flags.
    """
    from generator.osm.scene_builder import _build_default_metadata_config

    cfg = _build_default_metadata_config(
        bbox=[38.85, -77.04, 38.91, -77.00],
        center=None,
        size_m=None,
        width_m=None,
        terrain=False,
        terrain_zoom=14,
        ground_pad_m=200.0,
        ground_resolution_m=50.0,
        osm_server=None,
        barriers=True,
        vegetation=True,
        bridges=True,
        height_enrichment=True,
        canopy_heights=False,
        tree_rows=False,
        vegetation_gap_fill=False,
        worldcover_water_check=False,
        green_vegetation=False,
        tree_row_spacing_m=8.0,
        landuse_materials=False,
    )
    assert cfg["bbox"] is not None
    assert cfg["center"] is None
    assert cfg["landuse_materials"] is False


def test_landuse_material_map_covers_all_expected_classes():
    # Sanity: the map covers the high-impact OSM landuse values we
    # actually expect to drive material decisions.
    expected = {"industrial", "commercial", "retail", "residential", "religious"}
    assert expected <= set(_OSM_LANDUSE_MATERIAL_MAP.keys())
    # Every mapped value must be one of the existing ITU material slots
    # (concrete / brick / glass / metal / marble / wood) to avoid
    # unmappable material strings reaching the mesh writer.
    valid_materials = {"concrete", "brick", "glass", "metal", "marble", "wood"}
    assert set(_OSM_LANDUSE_MATERIAL_MAP.values()) <= valid_materials
