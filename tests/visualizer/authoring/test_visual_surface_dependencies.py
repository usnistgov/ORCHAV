"""Scenario copies retain authored surfaces and refuse nonportable texture paths."""

from __future__ import annotations

import json

import pytest
import yaml

from tests.visualizer.authoring.test_schema_first_import import (
    _mapping,
    _materialize_dependencies,
    _MergedCompiler,
)
from tests.visualizer.unit.test_visual_surfaces import surface_scene as _surface_scene
from visualizer.src.authoring.persistence import create_scenario_copy, save_document

surface_scene = _surface_scene


@pytest.fixture
def visual_scenario(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _materialize_dependencies(source)
    scene = source / "scene"
    scene.mkdir()
    (scene / "city.ply").write_bytes(b"source mesh")
    (scene / "city.xml").write_text(
        '<scene><shape type="ply"><string name="filename" value="city.ply"/></shape></scene>'
    )
    (scene / "maps").mkdir()
    (scene / "maps" / "front.png").write_bytes(b"albedo texture")
    (scene / "maps" / "normal.png").write_bytes(b"normal texture")
    material = {"texture_path": "maps/front.png", "normal_map_path": "maps/normal.png"}
    payload = {
        "schema_version": 1,
        "meshes": {
            "city.ply": {
                "source_sha256": "source fingerprint",
                "surfaces": [{"id": "front", "material": material}],
            }
        },
    }
    sidecar = scene / "visual_surfaces.json"
    sidecar.write_text(json.dumps(payload))
    mapping = _mapping()
    mapping["scene"] = {"source": "local", "id": "scene/city.xml"}
    scenario = source / "scenario.yaml"
    scenario.write_text(yaml.safe_dump(mapping))
    return source, scenario, sidecar, payload, material


def test_copy_retains_visual_sidecar_and_only_referenced_maps(visual_scenario):
    _assert_copy_retains_visual_sidecar(visual_scenario, "local")


def _assert_copy_retains_visual_sidecar(visual_scenario, scene_source):
    source, scenario, sidecar, _, _ = visual_scenario
    if scene_source == "osm":
        mapping = yaml.safe_load(scenario.read_text())
        mapping["scene"] = {
            "source": "osm",
            "osm": {"center": [48.853, 2.349], "size_m": 500.0},
        }
        scenario.write_text(yaml.safe_dump(mapping))
        (source / "osm_metadata.json").write_text(
            json.dumps({"artifacts": {"scene_xml": "scene/city.xml", "mesh_files": []}})
        )
    (sidecar.parent / "maps" / "unused.png").write_bytes(b"unrelated")
    destination = source.parent / "copy"
    document = create_scenario_copy(scenario, destination)
    save_document(document, compiler=_MergedCompiler())

    for relative in (
        "scene/city.xml",
        "scene/city.ply",
        "scene/visual_surfaces.json",
        "scene/maps/front.png",
        "scene/maps/normal.png",
    ):
        assert (destination / relative).read_bytes() == (source / relative).read_bytes()
    assert not (destination / "scene/maps/unused.png").exists()
    assert not any(dependency.problem for dependency in document.scenario.dependencies)


@pytest.mark.parametrize("path_kind", ["traversal", "absolute", "symlink", "missing"])
def test_nonportable_texture_blocks_copy_without_mutating_source(visual_scenario, path_kind):
    source, scenario, sidecar, payload, material = visual_scenario
    outside = source.parent / "paper_texture.png"
    outside.write_bytes(b"external paper asset")
    if path_kind == "traversal":
        material["texture_path"] = "../../paper_texture.png"
        expected = "parent traversal"
    elif path_kind == "absolute":
        material["texture_path"] = str(outside)
        expected = "must be relative"
    elif path_kind == "symlink":
        linked = sidecar.parent / "maps" / "external.png"
        linked.symlink_to(outside)
        material["texture_path"] = "maps/external.png"
        expected = "escapes the source scenario"
    else:
        material["texture_path"] = "maps/missing.png"
        expected = "Scenario dependency not found"
    sidecar.write_text(json.dumps(payload))
    before = sidecar.read_bytes()
    destination = source.parent / "copy"
    document = create_scenario_copy(scenario, destination)
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        save_document(document, compiler=_MergedCompiler())
    assert sidecar.read_bytes() == before
    assert outside.read_bytes() == b"external paper asset"
    assert not (destination / "scenario.yaml").exists()
    assert not (destination / "scene/visual_surfaces.json").exists()


@pytest.mark.parametrize("payload", ["{not-json", '{"schema_version": 2, "meshes": {}}'])
def test_unreadable_visual_sidecar_blocks_copy(visual_scenario, payload):
    source, scenario, sidecar, _, _ = visual_scenario
    sidecar.write_text(payload)
    destination = source.parent / "copy"
    document = create_scenario_copy(scenario, destination)
    with pytest.raises(ValueError, match="Visual surface sidecar is unreadable"):
        save_document(document, compiler=_MergedCompiler())
    assert not (destination / "scenario.yaml").exists()


def test_external_sidecar_symlink_blocks_copy(visual_scenario):
    source, scenario, sidecar, _, _ = visual_scenario
    outside = source.parent / "external_mapping.json"
    sidecar.rename(outside)
    sidecar.symlink_to(outside)
    document = create_scenario_copy(scenario, source.parent / "copy")
    with pytest.raises(ValueError, match="Visual surface sidecar escapes"):
        save_document(document, compiler=_MergedCompiler())


@pytest.mark.parametrize("metadata_name", ["building_visuals.json", "osm_metadata.json"])
def test_copied_scene_retains_partition_and_effective_geographic_classification(
    surface_scene, metadata_name
):
    _assert_copied_scene_retains_classification(surface_scene, "local", metadata_name)


def _assert_copied_scene_retains_classification(surface_scene, scene_source, metadata_name):
    from xml.etree import ElementTree as ET

    import numpy as np

    from visualizer.src.scene.io import build_scene_from_root

    scene = surface_scene
    source = scene.directory
    _materialize_dependencies(source)
    scene.sidecar.write_text(json.dumps(scene.document))
    classification = {
        "visual_building_faces": {
            "mesh/city.ply": {
                "building_triangles": 6,
                "total_triangles": 8,
                "parts": [[0, 2], [2, 2], [4, 2]],
                "facade_exclusions": [
                    {"first_triangle": 2, "triangle_count": 2, "source_id": "landmark"}
                ],
            }
        }
    }
    # The explicit building sidecar must win even when OSM metadata disagrees.
    osm = {
        "artifacts": {"scene_xml": "scene.xml", "mesh_files": ["mesh/city.ply"]},
        "visual_building_faces": {},
    }
    if metadata_name == "osm_metadata.json":
        osm.update(classification)
    else:
        (source / metadata_name).write_text(json.dumps(classification))
    (source / "osm_metadata.json").write_text(json.dumps(osm))
    mapping = _mapping()
    mapping["scene"] = (
        {"source": "local", "id": "scene.xml"}
        if scene_source == "local"
        else {"source": "osm", "osm": {"center": [48.853, 2.349], "size_m": 500.0}}
    )
    scenario = source / "scenario.yaml"
    scenario.write_text(yaml.safe_dump(mapping))
    original = build_scene_from_root(ET.parse(scene.xml_path).getroot(), str(scene.xml_path))
    destination = source / "copy"
    document = create_scenario_copy(scenario, destination)
    save_document(document, compiler=_MergedCompiler())
    copied_xml = destination / "scene.xml"
    copied = build_scene_from_root(ET.parse(copied_xml).getroot(), str(copied_xml))

    assert (destination / metadata_name).read_bytes() == (source / metadata_name).read_bytes()
    assert (destination / "osm_metadata.json").read_bytes() == (
        source / "osm_metadata.json"
    ).read_bytes()
    # Geographic partitioning can create separate unclassified and building
    # remainders. Select the building-bearing part, rather than relying on order.
    (original_remainder,) = (
        e for e in original if e["_visual_surface_id"] == "remainder" and "_osm_building_faces" in e
    )
    (copied_remainder,) = (
        e for e in copied if e["_visual_surface_id"] == "remainder" and "_osm_building_faces" in e
    )
    for key in ("_osm_building_faces", "_osm_building_parts", "_osm_facade_exclusions"):
        assert copied_remainder[key] == original_remainder[key]
    assert copied_remainder["_osm_building_faces"] == 4
    assert copied_remainder["_osm_facade_exclusions"] == ((1, 1),)
    for first, second in zip(original, copied, strict=True):
        for key in (
            "_visual_surface_id",
            "geographic_layer",
            "_osm_building_faces",
            "_osm_building_parts",
            "_osm_facade_exclusions",
        ):
            assert first.get(key) == second.get(key)
        np.testing.assert_array_equal(first["mesh"].vertices, second["mesh"].vertices)
        np.testing.assert_array_equal(first["mesh"].triangles, second["mesh"].triangles)
        np.testing.assert_array_equal(first["mesh"].triangle_uvs, second["mesh"].triangle_uvs)


@pytest.mark.parametrize("metadata_name", ["building_visuals.json", "osm_metadata.json"])
def test_external_appearance_metadata_symlink_blocks_copy(visual_scenario, metadata_name):
    source, scenario, sidecar, _, _ = visual_scenario
    outside = source.parent / "external_appearance.json"
    outside.write_text("{}")
    (sidecar.parent / metadata_name).symlink_to(outside)
    document = create_scenario_copy(scenario, source.parent / "copy")
    with pytest.raises(ValueError, match="Scene appearance metadata escapes"):
        save_document(document, compiler=_MergedCompiler())


def test_open_document_recompiles_optional_mapping_add_change_remove_restore(surface_scene):
    import numpy as np

    from visualizer.src.authoring import ScenarioCompiler
    from visualizer.src.authoring.persistence import load_for_authoring

    scene = surface_scene
    mapping = _mapping()
    mapping["scene"] = {"source": "local", "id": "scene.xml"}
    mapping["actors"]["tx"][0]["mobility"] = {"type": "stationary", "position_m": [0, 0, 1]}
    mapping["actors"]["targets"] = []
    path = scene.directory / "scenario.yaml"
    path.write_text(yaml.safe_dump(mapping))
    document = load_for_authoring(path).document
    assert document is not None
    compiler = ScenarioCompiler()

    def compile_current():
        result = compiler.compile(document.scenario, scenario_directory=scene.directory)
        assert result.valid, result.issues
        return result.scene_assets

    assert len(compile_current()) == 1
    scene.sidecar.write_text(json.dumps(scene.document))
    mapped = compile_current()
    assert len(mapped) == 2
    front = next(
        asset
        for asset in mapped
        if asset.material.texture_path == str(scene.directory / "front.png")
    )
    before_uv = front.mesh.triangle_uvs.copy()
    scene.surface["triangle_uvs"] = (np.asarray(scene.surface["triangle_uvs"]) * 0.5).tolist()
    scene.sidecar.write_text(json.dumps(scene.document))
    changed = next(
        asset
        for asset in compile_current()
        if asset.material.texture_path == str(scene.directory / "front.png")
    )
    np.testing.assert_allclose(changed.mesh.triangle_uvs, before_uv * 0.5)
    # Reload once with the mapping present: this retains the stale dependency
    # snapshot that originally blocked a later removal from the open document.
    document = load_for_authoring(path).document
    assert any(d.kind == "visual_surfaces" for d in document.scenario.dependencies)
    scene.sidecar.unlink()
    assert len(compile_current()) == 1
    save_document(document, scene.directory / "copy", compiler=compiler)
    assert not (scene.directory / "copy/visual_surfaces.json").exists()
    assert (
        len(
            compiler.compile(
                document.scenario, scenario_directory=document.path.parent
            ).scene_assets
        )
        == 1
    )
    # Restore against the same original open source, then make a portable copy.
    document = load_for_authoring(path).document
    scene.sidecar.write_text(json.dumps(scene.document))
    assert len(compile_current()) == 2
    destination = scene.directory / "restored"
    save_document(document, destination, compiler=compiler)
    assert (destination / "visual_surfaces.json").read_bytes() == scene.sidecar.read_bytes()
    assert (destination / "front.png").read_bytes() == (scene.directory / "front.png").read_bytes()


@pytest.mark.parametrize("missing", ["scene.xml", "mesh/city.ply", "front.png"])
def test_recompiled_scene_still_rejects_missing_required_assets(surface_scene, missing):
    from visualizer.src.authoring import ScenarioCompiler
    from visualizer.src.authoring.persistence import load_for_authoring

    scene = surface_scene
    mapping = _mapping()
    mapping["scene"] = {"source": "local", "id": "scene.xml"}
    mapping["actors"]["tx"][0]["mobility"] = {"type": "stationary", "position_m": [0, 0, 1]}
    mapping["actors"]["targets"] = []
    path = scene.directory / "scenario.yaml"
    path.write_text(yaml.safe_dump(mapping))
    scene.sidecar.write_text(json.dumps(scene.document))
    document = load_for_authoring(path).document
    (scene.directory / missing).unlink()
    result = ScenarioCompiler().compile(document.scenario, scenario_directory=scene.directory)
    assert not result.valid
    assert any(
        issue.code == "dependency.missing" and missing in issue.message for issue in result.issues
    )
    assert result.scene_assets == ()
