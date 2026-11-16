"""Building appearance is optional, semantic, reversible and RF-independent."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.materials.appearance import VisualMaterialBinding, VisualMaterialSource
from visualizer.src.materials.building_style import (
    annotate_building_faces,
    apply_building_style,
    facade_pixels,
    styled_payload,
)
from visualizer.src.model import RenderObjectState
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.types.render_payloads import MeshPayload


@pytest.fixture
def entry(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path / "uv"))
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    # Vertical wall, horizontal roof, and an unrelated triangle in the same RF mesh.
    payload = MeshPayload(
        vertices=np.array(
            [[0.0, 0, 0], [6.0, 0, 0], [6.0, 0, 9], [0.0, 0, 9], [0.0, 4, 9], [6.0, 4, 9]]
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [1, 4, 5]]),
    )
    return {
        "name": "buildings_concrete",
        "rel_path": "mesh/buildings_concrete.ply",
        "entry_type": "mesh",
        "material_type": "concrete",
        "_osm_building_faces": 3,
        "mesh": RenderObjectState(id="building", payload=payload),
    }


def test_restore_and_source_geometry_unchanged(entry):
    source = entry["mesh"].payload
    props = {"color": [0.7, 0.7, 0.7], "roughness": 0.5}
    out = apply_building_style(entry, props, enabled=True, eligible=True)
    styled = entry["mesh"].payload
    assert out["texture_path"] and props == {"color": [0.7, 0.7, 0.7], "roughness": 0.5}
    assert source.triangle_uvs is None
    np.testing.assert_array_equal(source.vertices, styled.vertices)
    np.testing.assert_array_equal(source.triangles, styled.triangles)
    assert entry["material_type"] == "concrete"
    # Repeated material resolution must keep the same payload and buffers.
    apply_building_style(entry, props, enabled=True, eligible=True)
    assert entry["mesh"].payload is styled
    apply_building_style(entry, props, enabled=False, eligible=True)
    assert entry["mesh"].payload is source
    apply_building_style(entry, props, enabled=True, eligible=True)
    assert entry["mesh"].payload is styled


def test_neutral_scene_assembly_matches_owned_material_preparation(entry):
    from visualizer.src.scene.assembly import mesh_entry_to_payload
    from visualizer.src.services.material_context import MaterialContext

    context = MaterialContext(building_appearance="procedural")
    service = MaterialPBRService(context)
    neutral = {**entry, "mesh": entry["mesh"].payload}
    expected = service.resolve_entry_material(entry)
    _, payload, material = mesh_entry_to_payload(neutral, {}, material_service=service)
    assert material.texture_path == expected.payload.texture_path
    assert material.texture_path
    np.testing.assert_array_equal(payload.triangle_uvs, entry["mesh"].payload.triangle_uvs)


def test_builder_publishes_base_geometry_before_procedural_appearance(entry, tmp_path, monkeypatch):
    import visualizer.src.authoring.assets as module
    from visualizer.src.authoring.domain import SceneReference

    path = tmp_path / "scene.xml"
    path.write_text('<scene version="3.0.0"/>')
    raw = {**entry, "mesh": entry["mesh"].payload}
    monkeypatch.setattr(module, "build_scene_from_root", lambda *args: [raw])
    compiler = module.PreviewAssetCompiler(tmp_path)
    compiler.configure_appearance({"building_appearance": "procedural"}, tmp_path)
    stages = []
    result = compiler.build_scene_assets(
        SceneReference("local", "scene.xml"), path, geometry_ready=stages.append
    )
    assert len(stages) == 1
    assert stages[0][0].material.texture_path != result[0].material.texture_path
    assert result[0].material.texture_path
    assert stages[0][0].cache_key != result[0].cache_key
    compiler.target_cache.close()


@pytest.mark.parametrize(
    "mode,theme", [("procedural", "mixed"), ("varied", "paris"), ("varied", "haussmann")]
)
@pytest.mark.parametrize("inactive", ["original", "textures_off"])
def test_inactive_recipe_change_retains_prepared_geometry(
    entry, monkeypatch, mode, theme, inactive
):
    service = _style_service(entry, mode)
    viz = service.visualizer
    viz.facade_theme = theme
    source = entry["mesh"].payload
    before = service.resolve_entry_material(entry)
    styled = entry["mesh"].payload
    cached = entry["_building_style_payloads"]
    recipe = entry["_building_style_mode"]

    if inactive == "original":
        viz.building_appearance = "original"
    else:
        monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
        viz.building_appearance, viz.facade_theme = "varied", "american"
    service.resolve_entry_material(entry)
    assert entry["mesh"].payload is source
    assert "_building_style_key" not in entry
    assert entry["_building_style_payloads"] is cached
    assert entry["_building_style_mode"] == recipe

    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    viz.building_appearance, viz.facade_theme = mode, theme
    restored = service.resolve_entry_material(entry)
    # Reuse the exact immutable render geometry and its material, including maps.
    assert entry["mesh"].payload is styled
    assert entry["_building_style_payloads"] is cached
    assert restored.payload == before.payload
    assert restored.properties == before.properties


@pytest.mark.parametrize("change", ["mode", "theme", "exclusions"])
def test_active_recipe_change_rebuilds_retained_geometry(entry, change):
    entry["_osm_building_parts"] = [[0, 3]]
    source = entry["mesh"].payload
    props = {"color": [0.7, 0.7, 0.7]}
    apply_building_style(entry, props, enabled=True, eligible=True, mode="varied", theme="paris")
    old_styled = entry["mesh"].payload
    apply_building_style(entry, props, enabled=False, eligible=True, mode="original")
    mode, theme = "varied", "paris"
    if change == "mode":
        mode = "procedural"
    elif change == "theme":
        theme = "american"
    else:
        entry["_osm_facade_exclusions"] = ((0, 3),)
    fresh = {key: value for key, value in entry.items() if not key.startswith("_building_style")}
    fresh["mesh"] = RenderObjectState(id="fresh-building", payload=source)

    actual = apply_building_style(entry, props, enabled=True, eligible=True, mode=mode, theme=theme)
    expected = apply_building_style(
        fresh, props, enabled=True, eligible=True, mode=mode, theme=theme
    )
    assert entry["mesh"].payload is not old_styled
    assert entry["_building_style_payloads"][0] is source
    assert actual == expected
    for key in ("vertices", "triangles", "triangle_uvs"):
        np.testing.assert_array_equal(
            getattr(entry["mesh"].payload, key), getattr(fresh["mesh"].payload, key)
        )


@pytest.mark.parametrize("edit_while_styled", [False, True])
def test_inactive_style_still_invalidates_changed_source(entry, edit_while_styled):
    source = entry["mesh"].payload
    apply_building_style(entry, {}, enabled=True, eligible=True)
    old_styled = entry["mesh"].payload
    if not edit_while_styled:
        apply_building_style(entry, {}, enabled=False, eligible=True)
    moved = entry["mesh"].payload.vertices + [10, 20, 0]
    entry["mesh"].replace_payload(replace(entry["mesh"].payload, vertices=moved))

    apply_building_style(entry, {}, enabled=False, eligible=True, mode="original")
    changed_source = entry["mesh"].payload
    assert "_building_style_payloads" not in entry
    assert "_building_style_key" not in entry
    np.testing.assert_array_equal(changed_source.vertices, moved)
    np.testing.assert_array_equal(changed_source.triangles, source.triangles)
    assert changed_source.triangle_uvs is source.triangle_uvs

    apply_building_style(entry, {}, enabled=True, eligible=True)
    assert entry["mesh"].payload is not old_styled
    assert entry["_building_style_payloads"][0] is changed_source
    expected = styled_payload(changed_source, 3, entry["rel_path"])
    np.testing.assert_array_equal(entry["mesh"].payload.triangle_uvs, expected.triangle_uvs)
    np.testing.assert_array_equal(entry["mesh"].payload.vertices, moved)


def test_authored_material_and_texture_disable_win(entry, monkeypatch):
    source = entry["mesh"].payload
    for props, eligible in [({"texture_path": "authored.png"}, True), ({}, False)]:
        assert apply_building_style(entry, props, enabled=True, eligible=eligible) == props
        assert entry["mesh"].payload is source
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    apply_building_style(entry, {}, enabled=True, eligible=True)
    assert entry["mesh"].payload is source


def test_roofs_and_unclassified_faces_do_not_get_windows(entry):
    out = styled_payload(entry["mesh"].payload, 3, "fixture")
    uv = out.triangle_uvs.reshape(-1, 3, 2)
    np.testing.assert_allclose(uv[2], np.tile([0.905, 0.03], (3, 1)))
    np.testing.assert_allclose(uv[3], np.tile([0.97, 0.97], (3, 1)))
    # A 6 m horizontal wall spans two 3 m repeats; 9 m height spans three.
    assert np.ptp(uv[:2, :, 0]) == pytest.approx(2)
    assert np.ptp(uv[:2, :, 1]) == pytest.approx(3)


def test_uv_cache_and_geometry_invalidation(entry):
    source = entry["mesh"].payload
    first = styled_payload(source, 3, "fixture")
    second = styled_payload(source, 3, "fixture")
    np.testing.assert_array_equal(first.triangle_uvs, second.triangle_uvs)
    assert first.cache_key == second.cache_key
    changed = styled_payload(replace(source, vertices=source.vertices * 2), 3, "fixture")
    assert changed.cache_key != first.cache_key
    assert styled_payload(source, 2, "fixture").cache_key != first.cache_key


def test_annotation_requires_valid_producer_metadata(entry, tmp_path):
    source = entry["mesh"].payload
    item = {"mesh": source, "rel_path": entry["rel_path"]}
    path = tmp_path / "osm_metadata.json"
    annotate_building_faces([item], str(tmp_path / "scene.xml"))
    assert "_osm_building_faces" not in item
    path.write_text(
        json.dumps(
            {
                "visual_building_faces": {
                    entry["rel_path"]: {"building_triangles": 3, "total_triangles": 4}
                }
            }
        )
    )
    annotate_building_faces([item], str(tmp_path / "scene.xml"))
    assert item["_osm_building_faces"] == 3
    bad = {"mesh": source, "rel_path": entry["rel_path"]}
    path.write_text(
        json.dumps(
            {
                "visual_building_faces": {
                    entry["rel_path"]: {"building_triangles": 30, "total_triangles": 4}
                }
            }
        )
    )
    annotate_building_faces([bad], str(tmp_path / "scene.xml"))
    assert "_osm_building_faces" not in bad


def test_material_service_preserves_manual_override(entry):
    viz = SimpleNamespace(mesh_entries=[entry], target_entries=[], building_appearance="procedural")
    service = MaterialPBRService(viz)
    source = entry["mesh"].payload
    assert "building-facades-v2" in service.resolve_entry_material(entry).payload.texture_path
    entry["_visual_material_binding"] = VisualMaterialBinding(
        source=VisualMaterialSource.MANUAL,
        material_type="concrete",
        overrides={"color": [1.0, 0, 0]},
    )
    assert "building-facades-v2" not in (
        service.resolve_entry_material(entry).payload.texture_path or ""
    )
    assert entry["mesh"].payload is source


def _style_service(entry, mode="varied"):
    entry["_osm_building_parts"] = [[0, 3]]
    if mode == "roads":
        entry["_osm_road_segments"] = [
            {
                "first_triangle": first,
                "start_m": first * 3.0,
                "end_m": first * 3.0 + 6.0,
                "length_m": 12.0,
                "width_m": 6.0,
                "highway": "residential",
                "lanes": "2",
                "oneway": False,
            }
            for first in (0, 2)
        ]
    viz = SimpleNamespace(
        mesh_entries=[entry],
        target_entries=[],
        building_appearance=mode if mode != "roads" else "original",
        facade_theme="paris",
        road_appearance="marked" if mode == "roads" else "original",
    )
    return MaterialPBRService(viz)


@pytest.mark.parametrize("mode", ["procedural", "varied", "roads"])
@pytest.mark.parametrize(
    "prop,value",
    [
        ("roughness", 0.2),
        ("alpha", 0.55),
        ("metallic", 0.8),
        ("reflectance", 0.7),
        ("clearcoat", 0.6),
        ("emissive_intensity", 0.4),
    ],
)
def test_property_edit_preserves_style_and_unedited_values(entry, mode, prop, value):
    service = _style_service(entry, mode)
    original = entry["mesh"].payload
    before = service.get_effective_entry_properties(entry)
    styled = entry["mesh"].payload
    assert entry.get("_building_style_key")
    assert service.set_property("concrete", prop, value)
    after = service.get_effective_entry_properties(entry)
    assert entry["mesh"].payload is styled
    assert after == {**before, prop: value}
    assert styled.vertices is original.vertices and styled.triangles is original.triangles
    assert entry["material_type"] == "concrete"
    service.reset_material("concrete")
    assert service.get_effective_entry_properties(entry) == before
    assert entry["mesh"].payload is styled


def test_property_edits_survive_style_and_texture_toggles(entry, monkeypatch):
    service = _style_service(entry)
    original = entry["mesh"].payload
    before = service.resolve_entry_material(entry).payload.texture_path
    service.set_property("concrete", "roughness", 0.2)
    service.set_property("concrete", "reflectance", 0.65)
    service.visualizer.building_appearance = "original"
    props = service.get_effective_entry_properties(entry)
    assert not entry.get("_building_style_key")
    assert props["roughness"] == 0.2 and props["reflectance"] == 0.65
    assert entry["mesh"].payload is original
    service.visualizer.building_appearance = "varied"
    service.visualizer.facade_theme = "american"
    changed = service.resolve_entry_material(entry)
    assert changed.payload.texture_path != before and entry.get("_building_style_key")
    assert changed.properties["roughness"] == 0.2
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    assert service.resolve_entry_material(entry).payload.texture_path is None
    assert entry["mesh"].payload is original
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES")
    service.visualizer.facade_theme = "paris"
    assert service.resolve_entry_material(entry).payload.texture_path == before
    assert service.get_effective_entry_properties(entry)["reflectance"] == 0.65


@pytest.mark.parametrize("preset", ["Gold", "Brick"])
def test_explicit_preset_replaces_style_until_reset(entry, preset):
    service = _style_service(entry)
    before = service.resolve_entry_material(entry).payload.texture_path
    service.set_property("concrete", "roughness", 0.25)
    assert service.apply_preset(preset, "concrete")
    assert service.resolve_entry_material(entry).payload.texture_path != before
    assert not entry.get("_building_style_key")
    service.set_property("concrete", "roughness", 0.3)
    service.resolve_entry_material(entry)
    assert not entry.get("_building_style_key")
    service.reset_material("concrete")
    assert service.resolve_entry_material(entry).payload.texture_path == before


@pytest.mark.parametrize(
    "property_name,value",
    [
        ("texture_path", None),
        ("normal_map_path", None),
        ("uv_repeat_scale", [2.0, 2.0]),
        ("shader_variant", None),
    ],
)
def test_explicit_surface_property_replaces_style(entry, property_name, value):
    service = _style_service(entry)
    before = service.resolve_entry_material(entry).payload.texture_path
    service.set_property("concrete", "roughness", 0.25)
    service.set_property("concrete", property_name, value)
    service.resolve_entry_material(entry)
    assert not entry.get("_building_style_key")
    service.set_property("concrete", "alpha", 0.6)
    service.resolve_entry_material(entry)
    assert not entry.get("_building_style_key")
    service.reset_material("concrete")
    assert service.resolve_entry_material(entry).payload.texture_path == before


def test_authored_texture_and_profile_remain_authoritative_during_edits(entry, tmp_path):
    from PIL import Image

    albedo = tmp_path / "landmark.png"
    Image.new("RGB", (2, 2), "red").save(albedo)
    entry["pbr_properties"] = {"texture_path": str(albedo)}
    service = _style_service(entry)
    service.set_property("concrete", "roughness", 0.2)
    assert service.resolve_entry_material(entry).payload.texture_path == str(albedo)
    assert not entry.get("_building_style_key")
    service.reset_material("concrete")
    entry.pop("pbr_properties")
    service.set_visual_binding(
        entry,
        VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type="concrete",
            overrides={"texture_path": str(albedo)},
        ),
    )
    service.set_property("concrete", "alpha", 0.4)
    assert service.resolve_entry_material(entry).payload.texture_path == str(albedo)
    assert not entry.get("_building_style_key")


def test_mixed_group_color_edit_keeps_procedural_albedo(entry):
    entry["material_type"] = "custom_plaster"
    service = _style_service(entry)
    uniform = {"name": "unclassified", "material_type": "custom_plaster"}
    service.visualizer.mesh_entries.append(uniform)
    before = service.resolve_entry_material(entry).payload.texture_path
    styled = entry["mesh"].payload
    service.set_property("custom_plaster", "color", [0.1, 0.2, 0.3])
    assert service.resolve_entry_material(entry).payload.texture_path == before
    assert entry["mesh"].payload is styled
    assert service.resolve_entry_material(entry).payload.base_color[:3] == (1.0, 1.0, 1.0)
    assert service.resolve_entry_material(uniform).properties["color"] == (0.1, 0.2, 0.3)


def test_saved_property_edits_restore_style_without_freezing_a_generated_texture(entry, tmp_path):
    service = _style_service(entry)
    service.preset_dir = tmp_path
    before = service.resolve_entry_material(entry).payload.texture_path
    service.set_property("concrete", "roughness", 0.2)
    service.set_property("concrete", "clearcoat", 0.6)
    assert service.save_preset("facade-adjustments")
    saved = json.loads((tmp_path / "facade-adjustments.json").read_text())
    assert saved["overrides"]["concrete"] == {"roughness": 0.2, "clearcoat": 0.6}
    service.apply_preset("Gold", "concrete")
    service.resolve_entry_material(entry)
    assert not entry.get("_building_style_key")
    assert service.load_preset("facade-adjustments")
    restored = service.resolve_entry_material(entry)
    assert restored.payload.texture_path == before
    assert restored.properties["roughness"] == 0.2
    assert restored.properties["clearcoat"] == 0.6
    assert restored.properties["reflectance"] == 0.3


def test_original_procedural_artwork_has_three_distinct_palettes():
    images = [facade_pixels(v, (120, 130, 140)) for v in range(3)]
    assert all(a.shape == (256, 256, 3) for a in images)
    assert not np.array_equal(images[0], images[1])
    np.testing.assert_array_equal(images[0][5, 250], [120, 130, 140])


def test_catalog_maps_do_not_override_requested_style(entry):
    import xml.etree.ElementTree as ET

    from visualizer.src.materials.catalog import material_preset

    entry["xml_bsdf"] = ET.fromstring('<bsdf><string name="type" value="concrete"/></bsdf>')
    entry["pbr_properties"] = dict(material_preset("concrete"))
    entry["_resolved_texture_maps"] = {
        k: v for k, v in entry["pbr_properties"].items() if k.endswith("_path")
    }
    viz = SimpleNamespace(mesh_entries=[entry], target_entries=[], building_appearance="procedural")
    service = MaterialPBRService(viz)
    assert "building-facades-v2" in service.resolve_entry_material(entry).payload.texture_path
    # A source-authored map must still win, even when it names the catalog map.
    ET.SubElement(
        entry["xml_bsdf"],
        "string",
        name="texture_path",
        value=entry["pbr_properties"]["texture_path"],
    )
    assert "building-facades-v2" not in service.resolve_entry_material(entry).payload.texture_path
    assert "_building_style_key" not in entry


@pytest.mark.parametrize("material_type", ["default", "medium_dry_ground"])
def test_geographic_asphalt_defaults_do_not_hide_road_markings(entry, material_type):
    import xml.etree.ElementTree as ET

    from visualizer.src.materials.surface_defaults import geographic_surface_properties

    entry["rel_path"] = "mesh/streets.ply"
    entry["material_type"] = material_type
    entry["xml_bsdf"] = ET.fromstring('<bsdf type="radio-material" id="mat-surrogate"/>')
    props = geographic_surface_properties(entry["rel_path"], material_type)
    entry["pbr_properties"] = props
    entry["_resolved_texture_maps"] = {k: v for k, v in props.items() if k.endswith("_path")}
    service = _style_service(entry, "roads")
    marked = service.resolve_entry_material(entry).payload.texture_path
    assert "_building_style_key" in entry
    assert marked != props["texture_path"]
    assert service.resolve_entry_material(entry).payload.texture_path == marked
    service.visualizer.road_appearance = "original"
    assert service.resolve_entry_material(entry).payload.texture_path == props["texture_path"]
    service.visualizer.road_appearance = "marked"
    # Explicit authored maps still take precedence, even for a catalog file.
    ET.SubElement(entry["xml_bsdf"], "string", name="texture_path", value=props["texture_path"])
    assert service.resolve_entry_material(entry).payload.texture_path == props["texture_path"]
    assert "_building_style_key" not in entry


def test_rotated_wall_uses_world_metres(entry):
    source = entry["mesh"].payload
    angle = np.radians(31)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    rotated = replace(source, vertices=source.vertices @ rotation.T)
    out = styled_payload(rotated, 3, "rotated")
    uv = out.triangle_uvs.reshape(-1, 3, 2)
    assert np.ptp(uv[:2, :, 0]) == pytest.approx(2, abs=0.01)
    assert np.ptp(uv[:2, :, 1]) == pytest.approx(3)


def test_missing_texture_cache_restores_original(entry, monkeypatch):
    import visualizer.src.materials.building_style as module

    original = entry["mesh"].payload
    apply_building_style(entry, {}, enabled=True, eligible=True)

    def failed(*args):
        raise OSError("read-only cache")

    monkeypatch.setattr(module, "_texture", failed)
    apply_building_style(entry, {}, enabled=True, eligible=True)
    assert entry["mesh"].payload is original
    assert "_building_style_key" not in entry


def test_geometry_edit_while_styled_is_not_rolled_back(entry):
    apply_building_style(entry, {}, enabled=True, eligible=True)
    first_key = entry["_building_style_key"]
    styled = entry["mesh"].payload
    moved = styled.vertices + [10, 20, 0]
    entry["mesh"].replace_payload(replace(styled, vertices=moved))
    apply_building_style(entry, {}, enabled=True, eligible=True)
    assert entry["_building_style_key"] != first_key
    np.testing.assert_array_equal(entry["mesh"].payload.vertices, moved)
    apply_building_style(entry, {}, enabled=False, eligible=True)
    np.testing.assert_array_equal(entry["mesh"].payload.vertices, moved)
    assert entry["mesh"].payload.triangle_uvs is None


def test_varied_modes_theme_changes_and_restore(entry):
    entry["_osm_building_parts"] = [[0, 3]]
    source = entry["mesh"].payload
    props = {"color": [0.7, 0.7, 0.7]}
    for mode, theme in [
        ("varied", "mixed"),
        ("procedural", "mixed"),
        ("varied", "american"),
        ("varied", "european"),
        ("varied", "asian"),
        ("varied", "portuguese"),
        ("varied", "paris"),
    ]:
        out = apply_building_style(
            entry, props, enabled=True, eligible=True, mode=mode, theme=theme
        )
        styled = entry["mesh"].payload
        assert styled.vertices is source.vertices and styled.triangles is source.triangles
        assert out["texture_path"]
        if mode == "varied":
            assert theme in out["texture_path"]
            assert np.isfinite(styled.triangle_uvs).all()
            assert np.all((styled.triangle_uvs >= 0) & (styled.triangle_uvs <= 1))
        apply_building_style(entry, props, enabled=False, eligible=True)
        assert entry["mesh"].payload is source


def test_variety_needs_valid_partition_and_changes_with_height(entry):
    from visualizer.src.materials.scene_variation import facade_uvs, valid_parts

    assert valid_parts([[0, 1], [1, 2]], 3)
    for parts in ([[0, 4]], [[1, 2]], [[0, -1]], [[0, 2], [1, 1]], [[0, 2]]):
        assert not valid_parts(parts, 3)
    source = entry["mesh"].payload
    a = facade_uvs(source, [[0, 3]], "same")
    b = facade_uvs(replace(source, vertices=source.vertices * [1, 1, 4]), [[0, 3]], "same")
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, facade_uvs(source, [[0, 3]], "same"))
    apply_building_style(entry, {}, enabled=True, eligible=True, mode="varied")
    assert entry["_building_style_mode"][0] == "procedural"
    np.testing.assert_array_equal(entry["mesh"].payload.vertices, source.vertices)


def test_roads_follow_polyline_and_leave_footpaths_unmarked(tmp_path, monkeypatch):
    from visualizer.src.materials.scene_variation import road_uvs, valid_roads

    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path / "uv"))
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    vertices = np.array(
        [
            [0.0, -3, 0],
            [0, 3, 0],
            [10, 3, 0],
            [10, -3, 0],
            [10, -3, 0],
            [10, 3, 0],
            [20, 13, 0],
            [20, 7, 0],
        ]
    )
    source = MeshPayload(
        vertices=vertices, triangles=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    )
    segments = [
        {
            "first_triangle": 0,
            "start_m": 0.0,
            "end_m": 10.0,
            "length_m": 25.0,
            "width_m": 6.0,
            "highway": "residential",
            "lanes": "2",
            "oneway": False,
        },
        {
            "first_triangle": 2,
            "start_m": 10.0,
            "end_m": 25.0,
            "length_m": 25.0,
            "width_m": 6.0,
            "highway": "residential",
            "lanes": "2",
            "oneway": False,
        },
    ]
    assert valid_roads({"total_triangles": 4, "segments": segments}, source)
    uv = road_uvs(source, segments).reshape(-1, 3, 2)
    np.testing.assert_allclose(uv[0, 2], uv[2, 1])  # Shared right edge across bend.
    np.testing.assert_allclose(uv[1, 2], uv[2, 0])  # Shared left edge.
    foot = [{**s, "highway": "footway"} for s in segments]
    assert len(np.unique(road_uvs(source, foot), axis=0)) == 1
    bad = [{**segments[0], "length_m": float("nan")}, segments[1]]
    assert not valid_roads({"total_triangles": 4, "segments": bad}, source)
    entry = {"mesh": RenderObjectState(id="road", payload=source), "_osm_road_segments": segments}
    apply_building_style(entry, {}, enabled=True, eligible=True, roads=True)
    assert entry["mesh"].payload.vertices is source.vertices
    apply_building_style(entry, {}, enabled=False, eligible=True, roads=True)
    assert entry["mesh"].payload is source


def test_path_surfaces_require_explicit_tags_and_unpaved_roads_have_no_markings():
    from visualizer.src.materials.scene_variation import road_uvs

    source = MeshPayload(
        vertices=np.array([[0.0, 0, 0], [3, 0, 0], [3, 15, 0], [0, 15, 0]]),
        triangles=np.array([[0, 1, 2], [0, 2, 3]]),
    )
    segment = {
        "first_triangle": 0,
        "start_m": 0.0,
        "end_m": 15.0,
        "length_m": 15.0,
        "width_m": 3.0,
        "highway": "footway",
        "lanes": "",
        "oneway": False,
    }
    unknown = road_uvs(source, [segment])
    assert len(np.unique(unknown, axis=0)) == 1
    paving = road_uvs(source, [{**segment, "surface": "paving_stones"}])
    gravel = road_uvs(source, [{**segment, "surface": "gravel"}])
    assert len(np.unique(paving, axis=0)) == 4
    assert not np.array_equal(gravel, paving)
    # Known gravel remains unmarked even when it is a vehicle road.
    np.testing.assert_array_equal(
        gravel, road_uvs(source, [{**segment, "highway": "residential", "surface": "gravel"}])
    )
    np.testing.assert_array_equal(unknown, road_uvs(source, [{**segment, "surface": "unknown"}]))


@pytest.mark.parametrize(
    "mode,theme", [("procedural", "mixed"), ("varied", "european"), ("varied", "paris")]
)
def test_landmark_exclusion_protects_wall_and_roof_without_splitting_mesh(entry, mode, theme):
    from visualizer.src.materials.scene_variation import _tile_uv

    entry["_osm_building_parts"] = [[0, 1], [1, 2]]
    original = entry["mesh"].payload
    props = {"color": [0.7, 0.7, 0.7]}
    apply_building_style(entry, props, enabled=True, eligible=True, mode=mode, theme=theme)
    unprotected_key = entry["_building_style_key"]
    entry["_osm_facade_exclusions"] = ((1, 2),)
    out = apply_building_style(entry, props, enabled=True, eligible=True, mode=mode, theme=theme)
    protected = entry["mesh"].payload
    uv = protected.triangle_uvs.reshape(-1, 3, 2)
    assert entry["_building_style_key"] != unprotected_key
    assert protected.vertices is original.vertices
    assert protected.triangles is original.triangles
    assert np.ptp(uv[0], axis=0).max() > 0  # Neighbor still has a facade.
    expected = (
        (0.97, 0.97)
        if mode == "procedural"
        else _tile_uv(246 if theme == "paris" else 255, 0.5, 0.5)
    )
    np.testing.assert_allclose(uv[1:3], np.broadcast_to(expected, (2, 3, 2)))
    # All landmark corners sample one solid swatch; no windows on its roof.
    from PIL import Image

    pixels = np.asarray(Image.open(out["texture_path"]))
    x, y = (np.asarray(expected) * [pixels.shape[1], pixels.shape[0]]).astype(int)
    np.testing.assert_array_equal(
        pixels[y, x], [216, 208, 190] if theme == "paris" else [178, 178, 178]
    )
    apply_building_style(entry, props, enabled=False, eligible=True)
    assert entry["mesh"].payload is original


@pytest.mark.parametrize(
    "bad",
    [
        False,
        {},
        [{"first_triangle": 1, "triangle_count": 1, "source_id": "osm:way/1"}],
        [{"first_triangle": 0, "triangle_count": 3}],
    ],
)
def test_invalid_landmark_metadata_keeps_entire_mesh_original(entry, tmp_path, bad):
    record = {
        "building_triangles": 3,
        "total_triangles": 4,
        "parts": [[0, 3]],
        "facade_exclusions": bad,
    }
    (tmp_path / "building_visuals.json").write_text(
        json.dumps({"visual_building_faces": {entry["rel_path"]: record}})
    )
    item = {**entry, "mesh": entry["mesh"].payload}
    annotate_building_faces([item], str(tmp_path / "scene.xml"))
    assert "_osm_building_faces" not in item


def test_landmark_metadata_retains_source_association_and_part_partition(entry, tmp_path):
    parts = [[0, 1], [1, 2]]
    record = {
        "building_triangles": 3,
        "total_triangles": 4,
        "parts": parts,
        "facade_exclusions": [
            {
                "first_triangle": 1,
                "triangle_count": 2,
                "source_id": "osm:way/226413508",
                "name": "Arc de Triomphe",
            }
        ],
    }
    (tmp_path / "building_visuals.json").write_text(
        json.dumps({"visual_building_faces": {entry["rel_path"]: record}})
    )
    item = {"rel_path": entry["rel_path"], "mesh": entry["mesh"].payload}
    annotate_building_faces([item], str(tmp_path / "scene.xml"))
    assert item["_osm_facade_exclusions"] == ((1, 2),)
    assert item["_osm_building_parts"] == parts


def test_paris_uses_existing_atlas_memory_budget():
    from visualizer.src.materials.scene_variation import atlas_pixels

    paris = atlas_pixels("facades", (120, 130, 140), "paris")
    european = atlas_pixels("facades", (120, 130, 140), "european")
    assert paris.shape == european.shape == (2048, 2048, 3)
    assert paris.nbytes == european.nbytes
    assert not np.array_equal(paris, european)
