"""Architectural appearance preserves the source surface and landmark identity."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.scenarios.model import ViewDefaultsModel
from visualizer.src.materials.haussmann_geometry import facade_payload
from visualizer.src.materials.haussmann_textures import (
    COLUMNS,
    FLOORS,
    GUTTER,
    HEIGHT,
    ORIGINAL_SLOT,
    ROOF_METRES,
    ROOF_REPEATS,
    ROWS,
    STONE_SLOT,
    WIDTH,
    _bay,
    _pbr_maps,
    _repeat_roof_tile,
    _slate_surfaces,
    tile_uv,
)
from visualizer.src.model import RenderObjectState
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.services.scene_edit_service import SceneEditService
from visualizer.src.types.render_payloads import MeshPayload


def wall(width=17.3, height=20.0, angle=0.47):
    v = np.array([[0, 0, 0], [width, 0, 0], [width, 0, height], [0, 0, height]], dtype=np.float64)
    rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    return MeshPayload(
        vertices=v @ rot.T + [2.3, -1.7, 4.0], triangles=np.array([[0, 1, 2], [0, 2, 3]])
    )


def area(payload):
    t = payload.vertices[payload.triangles]
    return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2


@pytest.mark.parametrize("width,height,angle", [(17.3, 20, 0.47), (1.2, 12, -0.9), (41.1, 27, 2.2)])
def test_bay_subdivision_preserves_surface_and_winding(width, height, angle):
    source = wall(width, height, angle)
    original = source.vertices.copy()
    styled = facade_payload(source, [[0, 2]], "wall")
    assert area(styled) == pytest.approx(area(source), rel=1e-10)
    expected = np.cross(
        source.vertices[1] - source.vertices[0], source.vertices[2] - source.vertices[0]
    )
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose((styled.vertices - source.vertices[0]) @ expected, 0, atol=1e-12)
    np.testing.assert_allclose(styled.normals @ expected, 1, atol=1e-7)
    np.testing.assert_array_equal(source.vertices, original)
    assert source.triangle_uvs is None
    assert not styled.vertices.flags.writeable
    assert np.isfinite(styled.triangle_uvs).all()
    assert styled.triangle_uvs.min() >= 0 and styled.triangle_uvs.max() <= 1


def test_excluded_landmark_and_unclassified_surface_never_get_windows():
    source = replace(wall(), normals=np.tile([0.1, -0.9, 0.2], (4, 1)))
    styled = facade_payload(source, [[0, 1]], "wall", exclusions=((0, 1),))
    np.testing.assert_allclose(
        styled.triangle_uvs[:3], np.tile(tile_uv(STONE_SLOT, 0.5, 0.5), (3, 1))
    )
    np.testing.assert_allclose(
        styled.triangle_uvs[3:], np.tile(tile_uv(ORIGINAL_SLOT, 0.5, 0.5), (3, 1))
    )
    assert area(styled) == pytest.approx(area(source))
    np.testing.assert_allclose(
        styled.normals, source.normals[source.triangles].reshape(-1, 3), atol=3e-8
    )


def test_roof_uvs_repeat_at_physical_scale_without_changing_surface():
    source = wall()
    # Add an upward-facing sloped roof, connected to the top of the wall.
    vertices = np.vstack((source.vertices, source.vertices[[2, 3]] + [0, 10, 2]))
    source = replace(
        source, vertices=vertices, triangles=np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]])
    )
    styled = facade_payload(source, [[0, 4]], "roof")
    assert area(styled) == pytest.approx(area(source), rel=1e-10)
    slots = np.floor(styled.triangle_uvs[:, 0] * COLUMNS) + COLUMNS * np.floor(
        styled.triangle_uvs[:, 1] * ROWS
    )
    roof = (slots >= 18) & (slots <= 20)
    assert roof.any()
    # Decode atlas UV edge differences back into metres. Packing repeats changes
    # tile coverage, not the physical slate scale or underlying roof surface.
    roof_triangles = styled.vertices[styled.triangles][roof.reshape(-1, 3).all(axis=1)]
    roof_uv = styled.triangle_uvs[roof].reshape(-1, 3, 2)
    normal = np.cross(
        roof_triangles[0, 1] - roof_triangles[0, 0], roof_triangles[0, 2] - roof_triangles[0, 0]
    )
    normal /= np.linalg.norm(normal)
    tangent = np.cross([0, 0, 1], normal)
    tangent /= np.linalg.norm(tangent)
    uphill = np.cross(normal, tangent)
    uv_metres = (
        np.array([COLUMNS, ROWS])
        * [WIDTH, HEIGHT]
        / (np.array([WIDTH, HEIGHT]) - 2 * GUTTER)
        * ROOF_METRES
        * ROOF_REPEATS
    )
    np.testing.assert_allclose(
        np.diff(roof_uv, axis=1) * uv_metres,
        np.diff(roof_triangles @ np.column_stack([tangent, uphill]), axis=1),
        atol=3e-5,
    )


def test_large_roof_fits_without_adding_surface_or_increasing_budget():
    source = MeshPayload(
        vertices=np.array(
            [[0, 0, 0], [500, 0, 0], [500, 0, 20], [0, 0, 20], [500, 500, 20], [0, 500, 20]],
            dtype=float,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]]),
    )
    styled = facade_payload(source, [[0, 4]], "large-roof")
    assert area(styled) == pytest.approx(area(source), rel=1e-10)
    assert len(styled.vertices) < 600_000
    assert np.isfinite(styled.triangle_uvs).all()
    assert source.triangle_uvs is None


def test_roof_texture_repeats_stay_in_the_existing_atlas_footprint():
    original = _pbr_maps(*_slate_surfaces(0, (WIDTH, HEIGHT), 12), (3, 12))[0]
    packed = _repeat_roof_tile(original)
    assert packed.shape == original.shape
    inside = packed[GUTTER:-GUTTER, GUTTER:-GUTTER]
    h, w = inside.shape[0] // ROOF_REPEATS[1], inside.shape[1] // ROOF_REPEATS[0]
    for row in range(ROOF_REPEATS[1]):
        for col in range(ROOF_REPEATS[0]):
            np.testing.assert_array_equal(
                inside[:h, :w], inside[row * h : (row + 1) * h, col * w : (col + 1) * w]
            )


def test_slate_has_deterministic_surface_detail():
    slate = _pbr_maps(*_slate_surfaces(0, (WIDTH, HEIGHT), 12), (3, 12))
    repeated = _pbr_maps(*_slate_surfaces(0, (WIDTH, HEIGHT), 12), (3, 12))
    for a, b in zip(slate, repeated, strict=True):
        np.testing.assert_array_equal(a, b)
        assert np.ptp(a.astype(int)) > 15
    # The top storey is slate, not limestone.
    facade = _bay(0, 6)[0]
    assert facade[-50, 20].mean() < facade[500, 20].mean() - 60


def test_height_and_source_edits_invalidate_cached_geometry():
    source = wall()
    a = facade_payload(source, [[0, 2]], "wall")
    assert a.cache_key == facade_payload(source, [[0, 2]], "wall").cache_key
    assert (
        a.cache_key
        != facade_payload(
            replace(source, vertices=source.vertices * 1.01), [[0, 2]], "wall"
        ).cache_key
    )
    assert a.cache_key != facade_payload(source, [[0, 2]], "wall", exclusions=((0, 2),)).cache_key


def test_pbr_maps_are_deterministic_bounded_and_have_surface_contrast():
    a, n, r, ao = _bay(1, 6)
    repeated = _bay(1, 6)
    assert ao.min() >= 178 and ao.max() == 255
    assert np.any(ao < 250)
    for first, second in zip((a, n, r, ao), repeated, strict=True):
        np.testing.assert_array_equal(first, second)
        assert first.shape == (HEIGHT, WIDTH, 3)
        assert first.dtype == np.uint8
    assert COLUMNS * WIDTH <= 8192 and ROWS * HEIGHT <= 8192
    # Glass is smoother than stone; normals remain in the outward hemisphere.
    assert r.min() < 100 and r.max() > 200
    decoded = n.astype(float) / 127.5 - 1
    np.testing.assert_allclose(np.linalg.norm(decoded, axis=-1), 1, atol=0.008)
    assert (decoded[..., 2] > 0).all()
    assert len(FLOORS) * 3 <= COLUMNS * ROWS - 6


def test_theme_switch_and_texture_disable_restore_original_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path))
    source = wall()
    entry = dict(
        name="wall",
        rel_path="mesh/wall.ply",
        entry_type="mesh",
        material_type="concrete",
        mesh=RenderObjectState(id="wall", payload=source),
        _osm_building_faces=2,
        _osm_building_parts=[[0, 2]],
    )
    viz = SimpleNamespace(
        mesh_entries=[entry],
        target_entries=[],
        building_appearance="varied",
        facade_theme="haussmann",
    )
    service = MaterialPBRService(viz)
    material = service.resolve_entry_material(entry).payload
    assert material.normal_map_path and material.roughness_map_path and material.texture_path
    styled = entry["mesh"].payload
    assert len(styled.triangles) > len(source.triangles)
    service.resolve_entry_material(entry)
    assert entry["mesh"].payload is styled
    viz.facade_theme = "paris"
    service.resolve_entry_material(entry)
    np.testing.assert_array_equal(entry["mesh"].payload.vertices, source.vertices)
    np.testing.assert_array_equal(entry["mesh"].payload.triangles, source.triangles)
    viz.facade_theme = "haussmann"
    service.resolve_entry_material(entry)
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    material = service.resolve_entry_material(entry).payload
    assert entry["mesh"].payload is source
    assert material.normal_map_path is None and material.texture_path is None


def test_paris_subset_classification_is_bound_to_retained_geometry():
    root = (
        Path(__file__).resolve().parents[3]
        / "scenarios/sensing/applications/paris_opera_urban_tracking/scene"
    )
    data = json.loads((root / "building_visuals.json").read_text())
    for name, digest in data["provenance"]["mesh_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        record = data["visual_building_faces"][name]
        assert sum(n for _, n in record["parts"]) == record["total_triangles"]
        for exclusion in record["facade_exclusions"]:
            assert [exclusion["first_triangle"], exclusion["triangle_count"]] in record["parts"]
            assert exclusion["source_id"].startswith("osm:way/")
    assert sum(len(r["facade_exclusions"]) for r in data["visual_building_faces"].values()) == 16


def test_theme_is_an_opt_in_scenario_setting():
    assert ViewDefaultsModel(facade_theme="haussmann").facade_theme == "haussmann"
    assert ViewDefaultsModel().facade_theme is None


def test_source_transform_survives_style_rebuild_and_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path))
    source = wall()
    mesh = RenderObjectState(id="scene:wall::mesh", payload=source)
    entry = dict(
        name="wall",
        entry_type="mesh",
        material_type="concrete",
        mesh=mesh,
        _osm_building_faces=2,
        _osm_building_parts=[[0, 2]],
    )
    viz = SimpleNamespace(
        mesh_entries=[entry],
        target_entries=[],
        building_appearance="varied",
        facade_theme="haussmann",
    )
    materials = MaterialPBRService(viz)
    materials.resolve_entry_material(entry)
    assert len(mesh.payload.vertices) > len(source.vertices)
    editor = SceneEditService(viz)
    center = editor._ensure_original_center(entry, mesh)
    original = editor._ensure_original_vertices(entry)
    editor._apply_transform_to_entry_mesh(
        entry, mesh, original, center, {"scale": 1, "rotation": [0, 0, 0], "translate": [10, 20, 0]}
    )
    materials.resolve_entry_material(entry)
    assert area(mesh.payload) == pytest.approx(area(source))
    viz.building_appearance = "original"
    materials.resolve_entry_material(entry)
    np.testing.assert_allclose(mesh.payload.vertices, source.vertices + [10, 20, 0])
    np.testing.assert_array_equal(mesh.payload.triangles, source.triangles)
    assert mesh.payload.triangle_uvs is None
