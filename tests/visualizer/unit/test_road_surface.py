"""Marked OSM pavement has one visible surface without changing RF meshes."""

from dataclasses import replace

import numpy as np
import pytest

from visualizer.src.materials.road_surface import visible_road_payload
from visualizer.src.types.render_payloads import MeshPayload, SurfaceColorSource


def road_scene(quads, kinds=None, widths=None):
    vertices = np.asarray(quads, dtype=float).reshape(-1, 3)
    triangles = np.concatenate(
        [np.array([[0, 1, 2], [0, 2, 3]]) + 4 * i for i in range(len(quads))]
    )
    source = MeshPayload(vertices=vertices, triangles=triangles)
    segments = [
        dict(
            first_triangle=2 * i,
            start_m=0.0,
            end_m=8.0,
            length_m=8.0,
            width_m=widths[i] if widths else 2.0,
            highway=kinds[i] if kinds else "footway",
            lanes="",
            oneway=False,
        )
        for i in range(len(quads))
    ]
    return source, segments, vertices[triangles][:, :, :2].reshape(-1, 2)


def rectangle(x0, y0, x1, y1, z=0.15):
    return [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]


def area(mesh):
    p = mesh.vertices[mesh.triangles]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2


def covering_triangles(mesh, point):
    p = mesh.vertices[mesh.triangles][:, :, :2]
    edges = np.roll(p, -1, axis=1) - p
    d = point - p
    signs = edges[:, :, 0] * d[:, :, 1] - edges[:, :, 1] * d[:, :, 0]
    return np.flatnonzero(np.all(signs > 1e-10, axis=1) | np.all(signs < -1e-10, axis=1))


def visible(source, segments, uv):
    return visible_road_payload(source, segments, uv, cache_key="fixture")


def test_reciprocal_osm_quad_keeps_only_one_surface_and_original_uvs():
    q = rectangle(-3, -1, 3, 1)
    source, segments, uv = road_scene([q, q[2:] + q[:2]])
    uv[6:] += 100  # Opposite graph directions need not have matching atlas UVs.
    result = visible(source, segments, uv)
    assert len(result.triangles) == 2
    assert area(result) == pytest.approx(12)
    np.testing.assert_array_equal(result.triangle_uvs, uv[:6])
    np.testing.assert_array_equal(source.vertices, np.array([q, q[2:] + q[:2]]).reshape(-1, 3))
    assert len(source.triangles) == 4


@pytest.mark.parametrize("reverse", [False, True])
def test_junction_preserves_union_and_vehicle_road_owns_path_crossing(reverse):
    quads = [rectangle(-3, -1, 3, 1), rectangle(-0.5, -3, 0.5, 3)]
    kinds = ["primary", "footway"]
    if reverse:
        quads.reverse()
        kinds.reverse()
    source, segments, uv = road_scene(quads, kinds)
    for i, kind in enumerate(kinds):
        uv[6 * i : 6 * i + 6] = 1 if kind == "primary" else 2
    result = visible(source, segments, uv)
    assert area(result) == pytest.approx(16)
    assert np.all(result.vertices[:, 2] == 0.15)
    for x in np.arange(-3, 3, 0.31) + 0.137:
        for y in np.arange(-3, 3, 0.29) + 0.113:
            point = np.array([x, y])
            ids = covering_triangles(result, point)
            expected = (-3 < x < 3 and -1 < y < 1) or (-0.5 < x < 0.5 and -3 < y < 3)
            assert len(ids) == int(expected)
    crossing = covering_triangles(result, np.array([0.13, 0.23]))
    np.testing.assert_array_equal(result.triangle_uvs.reshape(-1, 3, 2)[crossing], 1)


def test_partial_overlap_interpolates_uvs_and_normals_without_mutating_input():
    source, segments, uv = road_scene([rectangle(0, 0, 2, 2), rectangle(1, 1, 3, 3)])
    normals = np.broadcast_to([0.0, 0.0, 1.0], source.vertices.shape).copy()
    source = replace(source, normals=normals)
    result = visible(source, segments, uv)
    assert area(result) == pytest.approx(7)
    np.testing.assert_allclose(
        result.triangle_uvs, result.vertices[result.triangles].reshape(-1, 3)[:, :2]
    )
    np.testing.assert_allclose(result.normals, np.broadcast_to([0, 0, 1], result.normals.shape))
    assert not source.vertices.flags.writeable
    assert not source.triangles.flags.writeable
    np.testing.assert_array_equal(source.normals, normals)


def test_bridge_and_nonoverlapping_quads_keep_original_buffers():
    source, segments, uv = road_scene([rectangle(0, 0, 2, 2), rectangle(0, 0, 2, 2, z=3)])
    result = visible(source, segments, uv)
    assert result.vertices is source.vertices
    assert result.triangles is source.triangles
    assert area(result) == pytest.approx(8)


def test_authored_vertex_colors_survive_clipping_and_unchanged_pieces():
    source, segments, uv = road_scene([rectangle(0, 0, 2, 2), rectangle(1, 1, 3, 3)])
    colors = np.column_stack((source.vertices[:, :2] / 3, np.full(len(source.vertices), 0.5)))
    source = replace(source, vertex_colors=colors, color_source=SurfaceColorSource.VERTEX)
    result = visible(source, segments, uv)
    assert area(result) == pytest.approx(7)
    assert result.color_source is SurfaceColorSource.VERTEX
    expected = np.column_stack((result.vertices[:, :2] / 3, np.full(len(result.vertices), 0.5)))
    np.testing.assert_allclose(result.vertex_colors, expected)
    np.testing.assert_array_equal(source.vertex_colors, colors)
    assert not source.vertex_colors.flags.writeable


def test_terrain_slopes_are_not_flattened_or_clipped():
    slope = np.array(rectangle(0, 0, 2, 2))
    slope[:, 2] += slope[:, 0] / 10
    source, segments, uv = road_scene([rectangle(0, 0, 2, 2), slope])
    result = visible(source, segments, uv)
    assert result.vertices is source.vertices
    assert result.triangles is source.triangles


def test_narrow_road_inside_wider_road_leaves_no_hole():
    source, segments, uv = road_scene([rectangle(1, 1, 2, 2), rectangle(0, 0, 3, 3)], widths=[1, 3])
    result = visible(source, segments, uv)
    assert area(result) == pytest.approx(9)
    assert len(result.triangles) == 2


def test_subtracting_an_interior_quad_preserves_every_side_of_the_ring():
    source, segments, uv = road_scene([rectangle(1, 1, 2, 2), rectangle(0, 0, 3, 3)])
    uv[:6], uv[6:] = 1, 2
    result = visible(source, segments, uv)
    assert area(result) == pytest.approx(9)
    for x in np.arange(0.137, 3, 0.31):
        for y in np.arange(0.113, 3, 0.29):
            ids = covering_triangles(result, np.array([x, y]))
            assert len(ids) == 1
            expected = 1 if 1 < x < 2 and 1 < y < 2 else 2
            np.testing.assert_array_equal(result.triangle_uvs.reshape(-1, 3, 2)[ids], expected)


def test_touching_quads_preserve_original_geometry_and_clockwise_winding():
    quads = [rectangle(0, 0, 2, 2)[::-1], rectangle(2, 0, 4, 2)[::-1]]
    source, segments, uv = road_scene(quads)
    result = visible(source, segments, uv)
    assert result.vertices is source.vertices
    assert result.triangles is source.triangles
    overlapping, segments, uv = road_scene([quads[0], rectangle(1, 1, 3, 3)[::-1]])
    result = visible(overlapping, segments, uv)
    triangles = result.vertices[result.triangles]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    assert np.all(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0)


def test_intersection_budget_rejects_pathological_overlap(monkeypatch):
    import visualizer.src.materials.road_surface as roads

    monkeypatch.setattr(roads, "_MAX_INTERSECTION_PAIRS", 0)
    source, segments, uv = road_scene([rectangle(0, 0, 2, 2), rectangle(1, 1, 3, 3)])
    with pytest.raises(ValueError, match="intersection budget"):
        visible(source, segments, uv)


def test_style_toggle_restores_exact_source(tmp_path, monkeypatch):
    from visualizer.src.materials.building_style import apply_building_style
    from visualizer.src.model import RenderObjectState

    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    q = rectangle(0, 0, 2, 2)
    source, segments, _ = road_scene([q, q])
    entry = {"mesh": RenderObjectState(id="roads", payload=source), "_osm_road_segments": segments}
    apply_building_style(entry, {}, enabled=True, eligible=True, roads=True)
    styled = entry["mesh"].payload
    assert len(styled.triangles) == 2
    apply_building_style(entry, {}, enabled=False, eligible=True, roads=True)
    assert entry["mesh"].payload is source
    apply_building_style(entry, {}, enabled=True, eligible=True, roads=True)
    assert entry["mesh"].payload is styled
