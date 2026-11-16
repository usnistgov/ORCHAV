"""Regression tests for OSM vegetation and bbox clipping behavior.

Each test pins behavior that must remain stable:

1. ``tree_row_spacing_m <= 0`` used to spin the
   ``while d <= length`` advance in ``_sample_line_points`` forever.
   A defensive guard now returns ``[]`` for any non-positive spacing,
   and Pydantic ``Field(gt=0)`` validation rejects bad YAML at load
   time.
2. ``normalize_osm_config`` did not include ``tree_row_spacing_m``,
   so toggling the knob silently reused stale geometry.  The cache
   key now carries the spacing.
3. ``_apply_worldcover_gap_fill`` runs even for sparse scenes with
   zero OSM vegetation.
4. The routing graph (``street_network.graphml``) used to contain
   nodes outside the rendered street mesh in rectangular scenes;
   ``_prune_graph_to_bbox`` now drops them before save.
5. Even with inside nodes only, an edge ``geometry_local`` polyline
   could still leave and re-enter the bbox; saved graph edges are now
   clipped or dropped so mobility stays bbox-bound too.
"""

from __future__ import annotations

import pytest

osmnx = pytest.importorskip("osmnx")

import networkx as nx  # noqa: E402
from shapely.geometry import LineString  # noqa: E402
from shapely.geometry import box as shapely_box  # noqa: E402

from generator.osm.metadata import normalize_osm_config  # noqa: E402
from generator.osm.scene_builder import (  # noqa: E402
    _prune_graph_to_bbox,
    _sample_line_points,
    _sanitize_graph_edges_to_bbox,
)

# ---------------------------------------------------------------------------
# Non-positive spacing must not hang.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_m", [0.0, -0.001, -1.0, -42.0])
def test_sample_line_points_rejects_non_positive_spacing(spacing_m):
    """Non-positive spacing returns empty list instead of looping forever."""
    line = LineString([(0.0, 0.0), (100.0, 0.0)])
    out = _sample_line_points(line, spacing_m=spacing_m)
    assert out == []


def test_sample_line_points_positive_spacing_still_works():
    line = LineString([(0.0, 0.0), (100.0, 0.0)])
    out = _sample_line_points(line, spacing_m=20.0)
    assert len(out) == 6  # 0, 20, 40, 60, 80, 100


def test_scenario_model_rejects_zero_or_negative_tree_row_spacing():
    """Pydantic Field(gt=0) on tree_row_spacing_m surfaces bad YAML."""
    from pydantic import ValidationError

    from generator.osm.scenario_extension import OsmSceneConfigModel

    for bad in (0.0, -1.0, -42.0):
        with pytest.raises(ValidationError):
            OsmSceneConfigModel(
                center=[48.85, 2.34],
                size_m=500.0,
                tree_row_spacing_m=bad,
            )

    # Sanity: positive spacing accepted.
    cfg = OsmSceneConfigModel(
        center=[48.85, 2.34],
        size_m=500.0,
        tree_row_spacing_m=10.0,
    )
    assert cfg.tree_row_spacing_m == 10.0


def test_scenario_model_rejects_zero_or_negative_overpass_timeout():
    from pydantic import ValidationError

    from generator.osm.scenario_extension import OsmSceneConfigModel

    for bad in (0, -1, -180):
        with pytest.raises(ValidationError):
            OsmSceneConfigModel(
                center=[48.85, 2.34],
                size_m=500.0,
                overpass_timeout=bad,
            )


# ---------------------------------------------------------------------------
# Finding 2: tree_row_spacing_m must contribute to the cache key
# ---------------------------------------------------------------------------


def test_normalize_osm_config_includes_tree_row_spacing():
    cfg_a = {
        "center": [48.85, 2.34],
        "size_m": 500.0,
        "tree_row_spacing_m": 8.0,
    }
    cfg_b = dict(cfg_a)
    cfg_b["tree_row_spacing_m"] = 6.5
    assert normalize_osm_config(cfg_a) != normalize_osm_config(cfg_b)
    # The default is 8.0 — confirm it's actually written into the dict.
    assert normalize_osm_config({"center": [0, 0]}).get("tree_row_spacing_m") == 8.0


# ---------------------------------------------------------------------------
# Finding 4: routing graph pruning to the padded scene bbox
# ---------------------------------------------------------------------------


def _node(graph, node_id, x_local, y_local):
    graph.add_node(node_id, x_local=x_local, y_local=y_local)


def test_prune_graph_to_bbox_keeps_inside_drops_outside():
    g = nx.MultiDiGraph()
    _node(g, "A", 0.0, 0.0)  # inside
    _node(g, "B", 50.0, 50.0)  # inside
    _node(g, "C", 200.0, 50.0)  # outside (x > 100)
    _node(g, "D", -300.0, -50.0)  # outside (x < -100)
    g.add_edge("A", "B", length=70.7)
    g.add_edge("B", "C", length=150.0)  # crosses the boundary
    g.add_edge("D", "A", length=300.0)

    scene_bbox = shapely_box(-100.0, -100.0, 100.0, 100.0)
    n_dropped = _prune_graph_to_bbox(g, scene_bbox)

    assert n_dropped == 2
    assert set(g.nodes) == {"A", "B"}
    # Edges incident to dropped nodes are gone too.
    assert g.number_of_edges() == 1
    assert ("A", "B") in [(u, v) for u, v, _ in g.edges(keys=True)]


def test_prune_graph_to_bbox_no_op_when_all_nodes_inside():
    g = nx.MultiDiGraph()
    for i in range(5):
        _node(g, f"N{i}", float(i * 10), float(i * 5))
    scene_bbox = shapely_box(-1000.0, -1000.0, 1000.0, 1000.0)
    assert _prune_graph_to_bbox(g, scene_bbox) == 0
    assert len(g.nodes) == 5


def test_prune_graph_to_bbox_handles_missing_local_coords():
    g = nx.MultiDiGraph()
    g.add_node("missing")  # no x_local / y_local at all
    _node(g, "ok", 0.0, 0.0)
    scene_bbox = shapely_box(-10.0, -10.0, 10.0, 10.0)
    # Node without local coords is silently skipped (not pruned).
    assert _prune_graph_to_bbox(g, scene_bbox) == 0
    assert "missing" in g.nodes
    assert "ok" in g.nodes


def test_sanitize_graph_edges_to_bbox_drops_edge_that_leaves_and_reenters_scene():
    g = nx.MultiDiGraph()
    _node(g, "A", 0.0, 0.0)
    _node(g, "B", 10.0, 0.0)
    # Endpoints are inside, but the polyline leaves the bbox and comes back.
    g.add_edge(
        "A",
        "B",
        geometry_local="[[0.0, 0.0], [200.0, 0.0], [10.0, 0.0]]",
        length=210.0,
    )

    scene_bbox = shapely_box(-100.0, -100.0, 100.0, 100.0)
    n_clipped, n_dropped, n_isolated = _sanitize_graph_edges_to_bbox(g, scene_bbox)

    assert n_clipped == 0
    assert n_dropped == 1
    assert n_isolated == 2
    assert g.number_of_edges() == 0
    assert g.number_of_nodes() == 0
