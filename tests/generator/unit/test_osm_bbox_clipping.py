"""Unit tests for the bbox-clipping helpers in scene_builder.py.

These cover the clipping behavior that prevents long OSM features
(rivers, highways, continent-sized forests) from extending past the
ground mesh in the rendered scene.
"""

from __future__ import annotations

import pytest

shapely_geometry = pytest.importorskip("shapely.geometry")
LineString = shapely_geometry.LineString
MultiPolygon = shapely_geometry.MultiPolygon
Polygon = shapely_geometry.Polygon
shapely_box = shapely_geometry.box

pytest.importorskip("open3d")
pytest.importorskip("osmnx")
pytest.importorskip("triangle")

from generator.osm.scene_builder import (
    _clip_linestring_to_bbox,
    _clip_polygon_to_bbox,
)

# ----------------------------------------------------------------------------
# Polygon clipping
# ----------------------------------------------------------------------------


def test_clip_polygon_inside_bbox_returns_unchanged():
    bbox = shapely_box(-100, -100, 100, 100)
    poly = Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])
    out = _clip_polygon_to_bbox(poly, bbox)
    assert len(out) == 1
    assert out[0].area == pytest.approx(100 * 100)


def test_clip_polygon_larger_than_bbox_clamps_to_bbox():
    bbox = shapely_box(-100, -100, 100, 100)
    huge = Polygon([(-500, -500), (500, -500), (500, 500), (-500, 500)])
    out = _clip_polygon_to_bbox(huge, bbox)
    assert len(out) == 1
    assert out[0].area == pytest.approx(200 * 200)


def test_clip_polygon_outside_bbox_returns_empty():
    bbox = shapely_box(-100, -100, 100, 100)
    far_away = Polygon([(500, 500), (600, 500), (600, 600), (500, 600)])
    out = _clip_polygon_to_bbox(far_away, bbox)
    assert out == []


def test_clip_polygon_partial_overlap_returns_clipped_piece():
    bbox = shapely_box(0, 0, 100, 100)
    overlapping = Polygon([(50, -50), (150, -50), (150, 50), (50, 50)])
    out = _clip_polygon_to_bbox(overlapping, bbox)
    assert len(out) == 1
    # Overlap region is the 50×50 square inside the bbox.
    assert out[0].area == pytest.approx(50 * 50)


def test_clip_multipolygon_iterates_each_intersected_part():
    bbox = shapely_box(0, 0, 100, 100)
    multi = MultiPolygon(
        [
            Polygon([(10, 10), (40, 10), (40, 40), (10, 40)]),
            Polygon([(60, 60), (90, 60), (90, 90), (60, 90)]),
            Polygon([(200, 200), (300, 200), (300, 300), (200, 300)]),
        ]
    )
    out = _clip_polygon_to_bbox(multi, bbox)
    # Two of the three sub-polygons are inside the bbox.
    assert len(out) == 2
    assert all(p.area == pytest.approx(30 * 30) for p in out)


def test_clip_polygon_handles_geojson_dict():
    bbox = shapely_box(-100, -100, 100, 100)
    gj = {
        "type": "Polygon",
        "coordinates": [
            [[-50, -50], [50, -50], [50, 50], [-50, 50], [-50, -50]],
        ],
    }
    out = _clip_polygon_to_bbox(gj, bbox)
    assert len(out) == 1
    assert out[0].area == pytest.approx(100 * 100)


def test_clip_polygon_none_returns_empty():
    bbox = shapely_box(-100, -100, 100, 100)
    assert _clip_polygon_to_bbox(None, bbox) == []


# ----------------------------------------------------------------------------
# LineString clipping
# ----------------------------------------------------------------------------


def test_clip_linestring_fully_inside_returns_unchanged():
    bbox = shapely_box(-100, -100, 100, 100)
    line = LineString([(-50, 0), (50, 0)])
    out = _clip_linestring_to_bbox(line, bbox)
    assert len(out) == 1
    assert out[0].length == pytest.approx(100)


def test_clip_linestring_extending_past_bbox_clips_to_edges():
    bbox = shapely_box(-100, -100, 100, 100)
    long = LineString([(-300, 0), (300, 0)])
    out = _clip_linestring_to_bbox(long, bbox)
    assert len(out) == 1
    # 600 m line clipped to 200 m (the bbox width along the line).
    assert out[0].length == pytest.approx(200)


def test_clip_linestring_outside_returns_empty():
    bbox = shapely_box(-100, -100, 100, 100)
    far = LineString([(500, 500), (600, 600)])
    assert _clip_linestring_to_bbox(far, bbox) == []


def test_clip_linestring_zigzag_through_bbox_returns_multiple_pieces():
    bbox = shapely_box(-100, -100, 100, 100)
    # Zigzag: enters bbox, exits, re-enters → should produce two pieces.
    zigzag = LineString([(-300, 0), (-50, 0), (-50, 200), (50, 200), (50, 0), (300, 0)])
    out = _clip_linestring_to_bbox(zigzag, bbox)
    assert len(out) >= 2
    # All clipped pieces must be entirely within the bbox + boundary.
    for piece in out:
        for x, y in piece.coords:
            assert -100 - 1e-9 <= x <= 100 + 1e-9
            assert -100 - 1e-9 <= y <= 100 + 1e-9


def test_clip_linestring_none_returns_empty():
    bbox = shapely_box(-100, -100, 100, 100)
    assert _clip_linestring_to_bbox(None, bbox) == []


# ----------------------------------------------------------------------------
# Combined: typical OSM use case (the Seine through Île de la Cité)
# ----------------------------------------------------------------------------


def test_clip_simulates_seine_through_paris_bbox():
    """The Seine in OSM is one long polygon spanning all of Paris.  When
    we query a 500 m bbox on Île de la Cité, Overpass returns the full
    river, which would otherwise stretch many kilometers past the
    ground plane.  The clipping helper should reduce it to a polygon
    contained within the padded bbox.
    """
    # 500 m square scene centered at (0, 0), padded by 200 m.
    half = 250 + 200
    scene_bbox = shapely_box(-half, -half, half, half)

    # Synthetic Seine-like polygon: 100 m wide, 8 km long, runs through (0, 0).
    seine = Polygon(
        [
            (-4000, -50),
            (4000, -50),
            (4000, 50),
            (-4000, 50),
        ]
    )
    out = _clip_polygon_to_bbox(seine, scene_bbox)
    assert len(out) == 1
    # Clipped should be exactly the 900 m × 100 m intersection
    # (full bbox width × river width).
    assert out[0].area == pytest.approx((2 * half) * 100)
    # Bounds are entirely inside the bbox.
    minx, miny, maxx, maxy = out[0].bounds
    assert minx >= -half - 1e-9
    assert maxx <= half + 1e-9
    assert miny >= -half - 1e-9
    assert maxy <= half + 1e-9
