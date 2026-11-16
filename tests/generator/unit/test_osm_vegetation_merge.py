"""Mixed inventories must retain evidence and never fill excluded geometry."""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="Install the osm extra to test vegetation geometry")

from shapely.geometry import Point, box  # noqa: E402

from generator.osm.street_width import street_width  # noqa: E402
from generator.osm.vegetation_merge import merge_vegetation  # noqa: E402


@pytest.mark.parametrize(
    ("data", "expected", "source"),
    [
        ({"highway": "footway"}, 2.0, "highway_default"),
        ({"highway": "steps"}, 1.8, "highway_default"),
        ({"highway": "path"}, 1.5, "highway_default"),
        ({"highway": "residential"}, 8.0, "highway_default"),
        ({"highway": ["footway", "service"]}, 4.0, "highway_default"),
        ({"highway": "footway", "width": "5 m"}, 5.0, "osm_width"),
        ({"highway": "footway", "width": ".8"}, 0.8, "osm_width"),
        ({"width": "10 ft"}, 3.048, "osm_width"),
        ({"width": ["2", "3.5"]}, 3.5, "osm_width"),
    ],
)
def test_width_policy(data, expected, source):
    width, provenance = street_width(data)
    assert width == pytest.approx(expected)
    assert provenance == source


@pytest.mark.parametrize("value", ["nan", "inf", "-2", "0", True, "wide", "3 yards"])
def test_invalid_width_uses_class_default(value):
    assert street_width({"highway": "footway", "width": value}) == (2.0, "highway_default")


def test_recorded_positions_take_precedence_and_sources_survive():
    recorded = [{"id": "city/1", "x": 10.0, "y": 10.0, "species": "Celtis"}]
    osm = [
        {"id": "node/1", "natural": "tree", "geometry": Point(11, 10)},
        {"id": "node/2", "natural": "tree", "geometry": Point(20, 10)},
        {"id": "node/3", "natural": "tree", "geometry": Point(30, 10)},
    ]
    points, info = merge_vegetation(
        recorded, osm, bounds=box(0, 0, 40, 40), exclusions=box(29, 9, 31, 11)
    )
    assert [(p["id"], p["source"]) for p in points] == [
        ("city/1", "inventory"),
        ("node/2", "osm_tree"),
    ]
    assert points[0]["species"] == "Celtis"
    assert info["decisions"]["near_existing_tree"] == 1
    assert recorded[0] == {"id": "city/1", "x": 10.0, "y": 10.0, "species": "Celtis"}


def test_fill_is_deterministic_bounded_and_avoids_paths_and_inventory():
    osm = [
        {"id": "wood", "natural": "wood", "geometry": box(-100, -100, 200, 200)},
        {"id": "point", "natural": "tree", "geometry": Point(12, 12)},
    ]
    bounds, blocked = box(0, 0, 60, 60), box(25, 0, 35, 60)
    args = dict(bounds=bounds, exclusions=blocked, unavailable_sites=[(8, 8)])
    points, _ = merge_vegetation([], osm, **args)
    again, _ = merge_vegetation([], list(reversed(osm)), **args)
    assert points == again
    assert len(points) > 10
    assert points[0]["source"] == "osm_tree"
    for p in points:
        pt = Point(p["x"], p["y"])
        assert bounds.covers(pt) and not blocked.intersects(pt)
        assert pt.distance(Point(8, 8)) >= 3
    xy = np.array([[p["x"], p["y"]] for p in points])
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 6.4
    assert all(p["inferred"] for p in points[1:])


def test_ordinary_garden_is_not_assumed_wooded():
    feature = {"id": "garden", "leisure": "garden", "geometry": box(0, 0, 50, 50)}
    args = dict(bounds=box(0, 0, 50, 50), exclusions=Point().buffer(0))
    assert merge_vegetation([], [feature], **args)[0] == []
    feature["garden:type"] = "botanical"
    assert len(merge_vegetation([], [feature], **args)[0]) > 10


def test_candidate_budget_fails_without_publishing_partial_points():
    feature = {"id": "wood", "natural": "wood", "geometry": box(0, 0, 1000, 1000)}
    with pytest.raises(ValueError, match="budget"):
        merge_vegetation(
            [],
            [feature],
            bounds=feature["geometry"],
            exclusions=Point().buffer(0),
            max_candidates=10,
        )
