"""Authored visual surfaces survive normal scene loading without changing RF inputs."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from visualizer.src.materials.catalog import material_preset
from visualizer.src.scene.geometry_helpers import ensure_scene_mesh_render_state
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.types.render_payloads import MeshPayload, SurfaceColorSource


@pytest.fixture
def surface_scene(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path / "uv_cache"))
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(tmp_path / "scene_cache"))
    monkeypatch.setenv("ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR", str(tmp_path / "pygfx_cache"))
    monkeypatch.delenv("ORCHAV_DISABLE_SCENE_PAYLOAD_CACHE", raising=False)
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    mesh_path = tmp_path / "mesh" / "city.ply"
    mesh_path.parent.mkdir()
    vertices = np.array(
        [point for x in range(8) for point in ((x * 4, 0, 0), (x * 4 + 2, 0, 0), (x * 4, 0, 6))]
        + [(999, 999, 999)],
        dtype=float,
    )
    triangles = np.arange(24, dtype=np.int32).reshape(8, 3)
    mesh_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(vertices)}",
                "property float x",
                "property float y",
                "property float z",
                f"element face {len(triangles)}",
                "property list uchar int vertex_indices",
                "end_header",
            ]
            + [" ".join(map(str, row)) for row in vertices]
            + ["3 " + " ".join(map(str, row)) for row in triangles]
        )
        + "\n"
    )
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text(
        '<scene version="2.1.0">'
        '<bsdf id="mat-city" type="diffuse">'
        '<string name="type" value="concrete"/>'
        '<rgb name="reflectance" value="0.7 0.7 0.7"/>'
        '</bsdf><shape id="city" type="ply">'
        '<string name="filename" value="mesh/city.ply"/>'
        '<ref name="bsdf" id="mat-city"/>'
        "</shape></scene>"
    )
    xml_root = ET.parse(xml_path).getroot()
    Image.new("RGB", (4, 4), "gold").save(tmp_path / "front.png")
    source = MeshPayload(
        vertices=vertices,
        triangles=triangles,
        normals=np.tile([0.0, -1.0, 0.0], (len(vertices), 1)),
        vertex_colors=np.linspace(0, 1, len(vertices) * 3).reshape(-1, 3),
        triangle_uvs=np.arange(48, dtype=float).reshape(-1, 2) / 12,
        cache_key="source-city",
    )
    props = dict(material_preset("concrete"))
    entry = {
        "name": "city",
        "rel_path": "mesh/city.ply",
        "mesh": source,
        "material_id": "mat-city",
        "material_type": "concrete",
        "pbr_properties": props,
        "color": [0.7, 0.7, 0.7],
        "entry_type": "mesh",
        "xml_shape": xml_root.find("shape"),
        "xml_bsdf": xml_root.find("bsdf"),
        "original_vertices": vertices.copy(),
        "_osm_building_faces": 6,
        "_osm_building_parts": [[0, 2], [2, 2], [4, 2]],
        "_osm_facade_exclusions": ((2, 2),),
    }
    surface = {
        "id": "front",
        "name": "Authored facade",
        "triangle_indices": [1, 2],
        "triangle_uvs": [[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 1]],
        "material": {
            "texture_path": "front.png",
            "color": [1, 1, 1],
            "roughness": 0.8,
            "metallic": 0.0,
            "reflectance": 0.3,
        },
    }
    document = {
        "schema_version": 1,
        "meshes": {
            "mesh/city.ply": {
                "source_sha256": sha256(mesh_path.read_bytes()).hexdigest(),
                "surfaces": [surface],
            }
        },
    }
    return SimpleNamespace(
        directory=tmp_path,
        xml_path=xml_path,
        mesh_path=mesh_path,
        entry=entry,
        source=source,
        document=document,
        surface=surface,
        sidecar=tmp_path / "visual_surfaces.json",
    )


def _apply(scene):
    from visualizer.src.scene.visual_surfaces import apply_visual_surfaces

    scene.sidecar.write_text(json.dumps(scene.document))
    return apply_visual_surfaces([scene.entry], str(scene.xml_path))


def _corners(payload):
    return payload.vertices[payload.triangles]


def _faces(payload):
    # Order inside each face is deliberately retained: reversing winding must fail.
    return Counter(tuple(face.ravel()) for face in _corners(payload))


def _authored(entries):
    return next(entry for entry in entries if entry.get("texture_path"))


def test_partition_conserves_faces_uvs_and_source_files(surface_scene):
    scene = surface_scene
    before_xml = scene.xml_path.read_bytes()
    before_ply = scene.mesh_path.read_bytes()
    before_keys = set(scene.entry)
    before_props = deepcopy(scene.entry["pbr_properties"])
    entries = _apply(scene)

    assert len(entries) == 2
    actual_faces = Counter()
    for entry in entries:
        payload = entry["mesh"]
        actual_faces.update(_faces(payload))
        # Unused source vertices must not pollute selection bounds or camera fitting.
        assert len(np.unique(payload.triangles)) == len(payload.vertices)
        assert entry["material_id"] == "mat-city"
        assert entry["material_type"] == "concrete"
        assert entry["xml_shape"] is scene.entry["xml_shape"]
        assert entry["xml_bsdf"] is scene.entry["xml_bsdf"]
        assert entry["_visual_surface_partition"] is True
        assert entry["_visual_surface_key"]
    assert actual_faces == _faces(scene.source)
    assert all(count == 1 for count in actual_faces.values())
    assert len({entry["stable_mesh_id"] for entry in entries}) == 2

    authored = _authored(entries)
    complement = next(entry for entry in entries if entry is not authored)
    np.testing.assert_array_equal(_corners(authored["mesh"]), _corners(scene.source)[[1, 2]])
    np.testing.assert_array_equal(authored["mesh"].triangle_uvs, scene.surface["triangle_uvs"])
    np.testing.assert_array_equal(
        complement["mesh"].triangle_uvs.reshape(-1, 3, 2),
        scene.source.triangle_uvs.reshape(-1, 3, 2)[[0, 3, 4, 5, 6, 7]],
    )
    assert authored["mesh"].color_source is SurfaceColorSource.MATERIAL
    assert authored["mesh"].vertex_colors is None
    assert scene.entry["mesh"] is scene.source
    assert set(scene.entry) == before_keys
    assert scene.entry["pbr_properties"] == before_props
    assert scene.xml_path.read_bytes() == before_xml
    assert scene.mesh_path.read_bytes() == before_ply


def test_remainder_keeps_correct_procedural_classification(surface_scene):
    entries = _apply(surface_scene)
    authored = _authored(entries)
    complement = next(entry for entry in entries if entry is not authored)
    assert complement["_osm_building_faces"] == 4
    assert complement["_osm_building_parts"] == [[0, 1], [1, 1], [2, 2]]
    assert tuple(map(tuple, complement["_osm_facade_exclusions"])) == ((1, 1),)
    assert not any(key.startswith("_osm_") for key in authored)


def test_material_and_uv_mapping_survive_style_and_texture_toggles(surface_scene, monkeypatch):
    entries = _apply(surface_scene)
    authored = _authored(entries)
    complement = next(entry for entry in entries if entry is not authored)
    for index, entry in enumerate(entries):
        ensure_scene_mesh_render_state(entry, index)
    assert len({entry["mesh"].id for entry in entries}) == 2
    original_uvs = authored["mesh"].payload.triangle_uvs.copy()
    remainder_source = complement["mesh"].payload
    viz = SimpleNamespace(
        mesh_entries=entries,
        target_entries=[],
        building_appearance="original",
        facade_theme="paris",
    )
    service = MaterialPBRService(viz)
    expected_path = str(surface_scene.directory / "front.png")
    for style in ("original", "procedural", "varied", "original"):
        viz.building_appearance = style
        assert service.resolve_entry_material(authored).payload.texture_path == expected_path
        np.testing.assert_array_equal(authored["mesh"].payload.triangle_uvs, original_uvs)
        service.resolve_entry_material(complement)
        if style in {"procedural", "varied"}:
            assert complement.get("_building_style_key")
        else:
            assert complement["mesh"].payload is remainder_source
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    assert service.resolve_entry_material(authored).payload.texture_path is None
    np.testing.assert_array_equal(authored["mesh"].payload.triangle_uvs, original_uvs)
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES")
    assert service.resolve_entry_material(authored).payload.texture_path == expected_path


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "missing_texture",
        "overlap",
        "duplicate_face",
        "negative_face",
        "out_of_bounds_face",
        "fractional_face",
        "boolean_face",
        "bad_uv_count",
        "nonfinite_uv",
        "bad_schema",
    ],
)
def test_invalid_surface_falls_back_atomically(surface_scene, caplog, fault):
    scene = surface_scene
    record = scene.document["meshes"]["mesh/city.ply"]
    if fault == "hash":
        record["source_sha256"] = "0" * 64
    elif fault == "missing_texture":
        scene.surface["material"]["texture_path"] = "missing.png"
    elif fault == "overlap":
        second = deepcopy(scene.surface)
        second["id"] = "another_surface"
        record["surfaces"].append(second)
    elif fault == "duplicate_face":
        scene.surface["triangle_indices"] = [1, 1]
    elif fault == "negative_face":
        scene.surface["triangle_indices"] = [-1, 2]
    elif fault == "out_of_bounds_face":
        scene.surface["triangle_indices"] = [1, 8]
    elif fault == "fractional_face":
        scene.surface["triangle_indices"] = [1.5, 2]
    elif fault == "boolean_face":
        scene.surface["triangle_indices"] = [True, 2]
    elif fault == "bad_uv_count":
        scene.surface["triangle_uvs"].pop()
    elif fault == "nonfinite_uv":
        scene.surface["triangle_uvs"][0][0] = float("nan")
    elif fault == "bad_schema":
        scene.document["schema_version"] = 99
    entries = _apply(scene)
    assert len(entries) == 1
    assert entries[0]["mesh"] is scene.source
    assert "_visual_surface_partition" not in entries[0]
    assert any(record.levelno >= 30 for record in caplog.records)


def test_absent_sidecar_leaves_unrelated_scenes_unchanged(surface_scene):
    from visualizer.src.scene.visual_surfaces import apply_visual_surfaces

    scene = surface_scene
    entries = apply_visual_surfaces([scene.entry], str(scene.xml_path))
    assert len(entries) == 1
    assert entries[0] is scene.entry


def test_multiple_surfaces_can_cover_the_complete_source(surface_scene):
    scene = surface_scene
    remaining = [0, 3, 4, 5, 6, 7]
    second = deepcopy(scene.surface)
    second.update(
        id="other_faces",
        name="Other authored faces",
        triangle_indices=remaining,
        triangle_uvs=[[0, 0], [2, 0], [0, 2]] * len(remaining),
    )
    scene.document["meshes"]["mesh/city.ply"]["surfaces"].append(second)
    entries = _apply(scene)
    assert len(entries) == 2
    assert {entry["_visual_surface_id"] for entry in entries} == {"front", "other_faces"}
    actual_faces = Counter()
    for entry in entries:
        actual_faces.update(_faces(entry["mesh"]))
        assert entry["texture_path"] == str(scene.directory / "front.png")
        assert not any(key.startswith("_osm_") for key in entry)
    assert actual_faces == _faces(scene.source)


def test_bad_mapping_does_not_disable_another_valid_mesh(surface_scene):
    from visualizer.src.scene.visual_surfaces import apply_visual_surfaces

    scene = surface_scene
    second_path = scene.directory / "mesh" / "second.ply"
    second_path.write_bytes(scene.mesh_path.read_bytes())
    second_entry = {**scene.entry, "name": "second", "rel_path": "mesh/second.ply"}
    record = scene.document["meshes"]["mesh/city.ply"]
    scene.document["meshes"]["mesh/second.ply"] = deepcopy(record)
    record["source_sha256"] = "0" * 64
    scene.sidecar.write_text(json.dumps(scene.document))
    entries = apply_visual_surfaces([scene.entry, second_entry], str(scene.xml_path))
    assert len(entries) == 3
    assert entries[0] is scene.entry
    assert all(entry["_visual_surface_partition"] for entry in entries[1:])
    actual_faces = Counter()
    for entry in entries[1:]:
        actual_faces.update(_faces(entry["mesh"]))
    assert actual_faces == _faces(scene.source)


def test_scene_cache_reloads_current_mapping_without_reloading_rf_mesh(surface_scene, monkeypatch):
    from visualizer.src.scene import io as scene_io

    scene = surface_scene
    # Cache the original scene before a visual mapping is authored.
    source_root = scene_io.XMLSceneHandler.load_xml_scene(str(scene.xml_path))
    original_entries = scene_io.build_scene_from_root(source_root, str(scene.xml_path))
    assert len(original_entries) == 1
    original_faces = _faces(original_entries[0]["mesh"])
    cache_files = list((scene.directory / "scene_cache").glob("scene.*.npz"))
    assert len(cache_files) == 1
    cache_bytes = cache_files[0].read_bytes()

    def fail_source_reload(*args, **kwargs):
        raise AssertionError("Visual sidecar changes must reuse the original geometry cache")

    monkeypatch.setattr(scene_io.MeshLoader, "load_mesh", staticmethod(fail_source_reload))

    def load():
        root = scene_io.XMLSceneHandler.load_xml_scene(str(scene.xml_path))
        return scene_io.build_scene_from_root(root, str(scene.xml_path))

    scene.sidecar.write_text(json.dumps(scene.document))
    first = load()
    assert len(first) == 2
    first_key = _authored(first)["_visual_surface_key"]
    Image.new("RGB", (4, 4), "blue").save(scene.directory / "revised.png")
    scene.surface["material"]["texture_path"] = "revised.png"
    scene.sidecar.write_text(json.dumps(scene.document))
    revised = load()
    assert len(revised) == 2
    assert _authored(revised)["texture_path"] == str(scene.directory / "revised.png")
    assert _authored(revised)["_visual_surface_key"] != first_key
    assert {entry["stable_mesh_id"] for entry in first} == {
        entry["stable_mesh_id"] for entry in revised
    }
    assert cache_files[0].read_bytes() == cache_bytes
    scene.sidecar.unlink()
    restored = load()
    assert len(restored) == 1
    assert _faces(restored[0]["mesh"]) == original_faces
    assert "_visual_surface_partition" not in restored[0]


def test_cold_and_warm_loads_apply_identical_surfaces(surface_scene):
    from visualizer.src.scene import io as scene_io

    scene = surface_scene
    scene.sidecar.write_text(json.dumps(scene.document))
    cold_root = scene_io.XMLSceneHandler.load_xml_scene(str(scene.xml_path))
    cold = scene_io.build_scene_from_root(cold_root, str(scene.xml_path))
    warm_root = scene_io.XMLSceneHandler.load_xml_scene(str(scene.xml_path))
    warm = scene_io.build_scene_from_root(warm_root, str(scene.xml_path))
    assert len(cold) == len(warm) == 2
    for first, second in zip(cold, warm, strict=True):
        assert first["stable_mesh_id"] == second["stable_mesh_id"]
        assert first["_visual_surface_key"] == second["_visual_surface_key"]
        np.testing.assert_array_equal(_corners(first["mesh"]), _corners(second["mesh"]))
        np.testing.assert_array_equal(first["mesh"].triangle_uvs, second["mesh"].triangle_uvs)
        assert second["xml_shape"] is warm_root.find("shape")
        assert second["xml_bsdf"] is warm_root.find("bsdf")


def test_transformed_instances_keep_distinct_prepared_buffers_and_scene_ids(surface_scene):
    """Notebook/Builder preparation must not reuse another instance's vertices."""
    from visualizer.src.backends.pygfx_scene_helpers import mesh_payload_to_pygfx_buffers
    from visualizer.src.scene import io as scene_io

    scene = surface_scene
    scene.sidecar.write_text(json.dumps(scene.document))
    root = ET.parse(scene.xml_path).getroot()
    second_shape = deepcopy(root.find("shape"))
    second_shape.set("id", "shifted_city")
    transform = ET.SubElement(second_shape, "transform", name="to_world")
    ET.SubElement(transform, "translate", x="100", y="0", z="0")
    root.append(second_shape)
    entries = scene_io.build_scene_from_root(root, str(scene.xml_path))
    assert len(entries) == 4
    fronts = [entry for entry in entries if entry.get("_visual_surface_id") == "front"]
    first, second = [entry["mesh"] for entry in fronts]
    assert first.cache_key != second.cache_key
    np.testing.assert_allclose(second.vertices - first.vertices, [[100, 0, 0]] * 6)
    first_buffers = mesh_payload_to_pygfx_buffers(first)
    second_buffers = mesh_payload_to_pygfx_buffers(second)
    np.testing.assert_allclose(
        second_buffers["positions"] - first_buffers["positions"],
        [[100, 0, 0]] * len(first_buffers["positions"]),
    )
    np.testing.assert_array_equal(first_buffers["texcoords"], second_buffers["texcoords"])
    for index, entry in enumerate(entries):
        ensure_scene_mesh_render_state(entry, index)
    assert len({entry["mesh"].id for entry in entries}) == 4


@pytest.mark.parametrize(
    "change", ["add", "edit_uv", "remove", "texture_pixels", "restore_texture"]
)
def test_builder_refreshes_authored_surface_dependencies(surface_scene, change):
    """A reused Builder compiler must observe sidecar and authored-map changes."""
    from pathlib import Path

    from visualizer.src.authoring import PreviewAssetCompiler, SceneReference

    scene = surface_scene
    texture = scene.directory / "front.png"
    if change != "add":
        scene.sidecar.write_text(json.dumps(scene.document))
    if change == "restore_texture":
        texture.unlink()
    compiler = PreviewAssetCompiler(Path.cwd())
    reference = SceneReference("library", "example.xml")
    before = compiler.build_scene_assets(reference, scene.xml_path)
    assert len(before) == (1 if change in {"add", "restore_texture"} else 2)
    if change == "add":
        scene.sidecar.write_text(json.dumps(scene.document))
    elif change == "edit_uv":
        scene.surface["triangle_uvs"][0][0] = 0.25
        scene.sidecar.write_text(json.dumps(scene.document))
    elif change == "remove":
        scene.sidecar.unlink()
    else:
        Image.new("RGB", (4, 4), "blue").save(texture)
    after = compiler.build_scene_assets(reference, scene.xml_path)
    assert after is not before
    assert len(after) == (1 if change == "remove" else 2)
    assert {asset.cache_key for asset in after} != {asset.cache_key for asset in before}
    if change == "edit_uv":
        authored = next(asset for asset in after if asset.material.texture_path == str(texture))
        assert authored.mesh.triangle_uvs[0, 0] == pytest.approx(0.25)


@pytest.mark.parametrize("request_kind", ["canonical", "stable_snapshot"])
def test_protected_surface_edit_does_not_resolve_an_earlier_duplicate_name(
    surface_scene, request_kind
):
    from unittest.mock import Mock

    from visualizer.src.services.scene_edit_service import SceneEditService

    scene = surface_scene
    partitions = _apply(scene)
    authored = _authored(partitions)
    ordinary = {
        **scene.entry,
        "name": authored["name"],
        "stable_mesh_id": "unrelated_ordinary_mesh",
        "xml_shape": ET.fromstring('<shape id="unrelated"/>'),
    }
    entries = [ordinary, *partitions]
    for index, entry in enumerate(entries):
        ensure_scene_mesh_render_state(entry, index)
    before = [entry["mesh"].payload for entry in entries]
    xml_before = [ET.tostring(entry["xml_shape"]) for entry in entries]
    viz = SimpleNamespace(mesh_entries=entries, target_entries=[], _set_status_message=Mock())
    service = SceneEditService(viz)
    service.push_scene_updates_online = Mock()
    request = (
        authored
        if request_kind == "canonical"
        else {
            "name": authored["name"],
            "entry_type": "mesh",
            "stable_mesh_id": authored["stable_mesh_id"],
        }
    )
    service.set_object_position(request, [100.0, 200.0, 300.0])
    assert all(entry["mesh"].payload is payload for entry, payload in zip(entries, before))
    assert [ET.tostring(entry["xml_shape"]) for entry in entries] == xml_before
    service.push_scene_updates_online.assert_not_called()
    assert viz._set_status_message.called


@pytest.mark.parametrize("surface_id", ["front", "remainder"])
def test_visual_pbr_edits_preserve_xml_and_legacy_xml_color_edits_are_blocked(
    surface_scene, surface_id
):
    from visualizer.src.services.material_entry_editor import MaterialEntryEditService

    scene = surface_scene
    entries = _apply(scene)
    selected = next(entry for entry in entries if entry["_visual_surface_id"] == surface_id)
    xml_before = ET.tostring(scene.entry["xml_bsdf"])
    colors_before = [entry["color"][:] for entry in entries]
    viz = SimpleNamespace(mesh_entries=entries, target_entries=[], vis_initialized=False)
    legacy = MaterialEntryEditService()
    assert legacy.update_material_color(selected, [1.0, 0.0, 0.0], entries, []) == []
    assert [entry["color"] for entry in entries] == colors_before
    assert ET.tostring(scene.entry["xml_bsdf"]) == xml_before

    appearance = MaterialPBRService(viz)
    key = appearance.get_visual_material_key(selected)
    assert appearance.set_property(key, "roughness", 0.25)
    assert appearance.set_property(key, "color", [0.2, 0.3, 0.4])
    assert appearance.get_effective_entry_properties(selected)["roughness"] == 0.25
    assert ET.tostring(scene.entry["xml_bsdf"]) == xml_before
    assert ET.tostring(selected["xml_bsdf"]) == xml_before


def test_authored_surface_option_restores_base_material_and_custom_mapping(surface_scene):
    entries = _apply(surface_scene)
    authored = _authored(entries)
    for index, entry in enumerate(entries):
        ensure_scene_mesh_render_state(entry, index)
    viz = SimpleNamespace(
        mesh_entries=entries,
        target_entries=[],
        building_appearance="varied",
        facade_theme="paris",
        authored_surface_textures=True,
    )
    service = MaterialPBRService(viz)
    source_uvs = authored["mesh"].payload.triangle_uvs.copy()
    before = deepcopy(authored["pbr_properties"])
    custom = service.resolve_entry_material(authored).payload
    assert custom.texture_path == str(surface_scene.directory / "front.png")
    viz.authored_surface_textures = False
    base = service.resolve_entry_material(authored).payload
    assert base.texture_path == surface_scene.entry["pbr_properties"]["texture_path"]
    assert base.texture_path != custom.texture_path
    assert base.roughness == surface_scene.entry["pbr_properties"]["roughness"]
    np.testing.assert_allclose(
        service.get_entry_base_properties(authored)["color"], surface_scene.entry["color"]
    )
    np.testing.assert_array_equal(authored["mesh"].payload.triangle_uvs, source_uvs)
    assert authored["pbr_properties"] == before
    viz.authored_surface_textures = True
    assert service.resolve_entry_material(authored).payload == custom
