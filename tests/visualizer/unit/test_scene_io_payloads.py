"""Tests for renderer-neutral XML scene payload loading."""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from shared.geometry.transforms import UnsupportedLightweightTransformError
from visualizer.src.scene import io as scene_io
from visualizer.src.types.render_payloads import MeshPayload


@pytest.fixture(autouse=True)
def _reset_scene_payload_root_pruning() -> None:
    scene_io._PRUNED_SCENE_PAYLOAD_CACHE_ROOTS.clear()
    yield
    scene_io._PRUNED_SCENE_PAYLOAD_CACHE_ROOTS.clear()


def _write_tiny_scene(tmp_path):
    mesh_path = tmp_path / "wall.ply"
    mesh_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        ),
        encoding="utf-8",
    )
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<scene version="2.1.0">',
                '  <bsdf id="mat-wall" type="diffuse">',
                '    <rgb name="reflectance" value="0.2 0.3 0.4"/>',
                "  </bsdf>",
                '  <shape id="wall-shape" type="ply">',
                '    <string name="filename" value="wall.ply"/>',
                '    <ref name="bsdf" id="mat-wall"/>',
                '    <transform name="to_world">',
                '      <translate x="1" y="2" z="3"/>',
                "    </transform>",
                "  </shape>",
                "</scene>",
            ]
        ),
        encoding="utf-8",
    )

    return xml_path


def test_build_scene_from_root_returns_neutral_mesh_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHAV_DISABLE_SCENE_PAYLOAD_CACHE", "1")
    xml_path = _write_tiny_scene(tmp_path)

    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    entries = scene_io.build_scene_from_root(root, str(xml_path))

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry["mesh"], MeshPayload)
    assert entry["xml_shape"] is root.find("shape")
    assert entry["xml_bsdf"] is root.find("bsdf")
    assert not hasattr(entry["mesh"], "get_center")
    assert entry["_source_signature"]["path"] == str((tmp_path / "wall.ply").resolve())
    assert entry["_source_signature"]["size"] > 0
    np.testing.assert_allclose(entry["current_center"], [4.0 / 3.0, 7.0 / 3.0, 3.0])
    np.testing.assert_allclose(entry["mesh"].vertices[0], [1.0, 2.0, 3.0])


def test_build_scene_from_root_reuses_neutral_scene_payload_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ORCHAV_DISABLE_SCENE_PAYLOAD_CACHE", raising=False)
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(tmp_path / "cache"))
    xml_path = _write_tiny_scene(tmp_path)

    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    entries = scene_io.build_scene_from_root(root, str(xml_path))

    assert len(entries) == 1
    assert len(list((tmp_path / "cache").glob("scene.*.npz"))) == 1

    def fail_mesh_load(*args, **kwargs):
        raise AssertionError("cache hit should not reload source mesh")

    monkeypatch.setattr(
        scene_io.MeshLoader,
        "load_mesh",
        staticmethod(fail_mesh_load),
    )
    restored_root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    cached_entries = scene_io.build_scene_from_root(restored_root, str(xml_path))

    assert len(cached_entries) == 1
    cached_entry = cached_entries[0]
    assert isinstance(cached_entry["mesh"], MeshPayload)
    assert cached_entry["xml_shape"] is restored_root.find("shape")
    assert cached_entry["xml_bsdf"] is restored_root.find("bsdf")
    assert cached_entry["mesh"] is not entries[0]["mesh"]
    np.testing.assert_allclose(cached_entry["mesh"].vertices, entries[0]["mesh"].vertices)
    np.testing.assert_allclose(cached_entry["original_vertices"], entries[0]["original_vertices"])
    np.testing.assert_allclose(cached_entry["current_center"], entries[0]["current_center"])
    assert cached_entry["_source_signature"] == entries[0]["_source_signature"]


def test_unsupported_transform_fails_before_scene_cache_lookup(tmp_path, monkeypatch) -> None:
    xml_path = _write_tiny_scene(tmp_path)
    xml_text = xml_path.read_text(encoding="utf-8").replace(
        '<translate x="1" y="2" z="3"/>',
        '<matrix value="1 0 0 0 1 0 0 0 1"/>',
    )
    xml_path.write_text(xml_text, encoding="utf-8")
    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))

    def unexpected_fingerprint(*_args, **_kwargs):
        raise AssertionError("unsupported transforms must fail before cache lookup")

    monkeypatch.setattr(scene_io, "_scene_payload_cache_fingerprint", unexpected_fingerprint)

    with pytest.raises(UnsupportedLightweightTransformError, match="<matrix>"):
        scene_io.build_scene_from_root(root, str(xml_path))


@pytest.mark.parametrize("authored_texture", [False, True])
def test_cached_materials_follow_current_catalog_but_preserve_authored_maps(
    tmp_path, monkeypatch, authored_texture
) -> None:
    """A cache from another checkout must not turn catalog maps into authored maps."""
    from PIL import Image

    from visualizer.src.model import RenderObjectState
    from visualizer.src.services import material_pbr_service
    from visualizer.src.services.scene_service import SceneService

    monkeypatch.delenv("ORCHAV_DISABLE_SCENE_PAYLOAD_CACHE", raising=False)
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path / "uv"))
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    original_preset = scene_io.material_preset
    catalog_root = tmp_path / "checkout_a"

    def catalog_preset(name):
        props = original_preset(name)
        props["texture_path"] = str(catalog_root / "concrete.png")
        props["roughness"] = 0.3 if catalog_root.name == "checkout_a" else 0.7
        return props

    for folder in ("checkout_a", "checkout_b", "authored"):
        directory = tmp_path / folder
        directory.mkdir()
        Image.new("RGB", (2, 2), "gray").save(directory / "concrete.png")
    monkeypatch.setattr(scene_io, "material_preset", catalog_preset)
    monkeypatch.setattr(material_pbr_service, "material_preset", catalog_preset)
    xml_path = _write_tiny_scene(tmp_path)
    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    bsdf = root.find("bsdf")
    ET.SubElement(bsdf, "string", name="type", value="concrete")
    if authored_texture:
        # Same basename as the catalog, but explicitly selected by the author.
        ET.SubElement(bsdf, "string", name="texture_path", value="authored/concrete.png")
        ET.SubElement(bsdf, "float", name="roughness", value="0.23")
    first = scene_io.build_scene_from_root(root, str(xml_path))[0]
    cache_path = next((tmp_path / "cache").glob("scene.*.npz"))
    cache_bytes = cache_path.read_bytes()

    catalog_root = tmp_path / "checkout_b"
    expected = (
        tmp_path / "authored/concrete.png" if authored_texture else catalog_root / "concrete.png"
    )
    with monkeypatch.context() as cache_only:

        def fail_mesh_load(*args, **kwargs):
            raise AssertionError("material refresh must reuse cached geometry")

        cache_only.setattr(scene_io.MeshLoader, "load_mesh", staticmethod(fail_mesh_load))
        restored = scene_io.build_scene_from_root(root, str(xml_path))[0]

    assert restored["pbr_properties"]["texture_path"] == str(expected)
    assert restored["pbr_properties"]["roughness"] == (0.23 if authored_texture else 0.7)
    assert restored["xml_bsdf"] is bsdf
    assert restored["material_id"] == first["material_id"]
    np.testing.assert_array_equal(restored["mesh"].vertices, first["mesh"].vertices)
    np.testing.assert_array_equal(restored["mesh"].triangles, first["mesh"].triangles)
    assert cache_path.read_bytes() == cache_bytes

    # The same resolution must apply with and without a geometry cache.
    monkeypatch.setenv("ORCHAV_DISABLE_SCENE_PAYLOAD_CACHE", "1")
    cold = scene_io.build_scene_from_root(root, str(xml_path))[0]
    assert restored["pbr_properties"] == cold["pbr_properties"]
    assert restored["color"] == cold["color"]

    restored["mesh"] = RenderObjectState(id="building", payload=restored["mesh"])
    restored["_osm_building_faces"] = 1
    restored["_osm_building_parts"] = [[0, 1]]
    viz = SimpleNamespace(
        mesh_entries=[restored], target_entries=[], building_appearance="original"
    )
    service = material_pbr_service.MaterialPBRService(viz)
    initial = service.resolve_entry_material(restored)
    SceneService._remember_resolved_texture_maps(restored, initial.properties_copy())
    viz.building_appearance = "varied"
    styled = service.resolve_entry_material(restored)
    assert bool(restored.get("_building_style_key")) is not authored_texture
    if authored_texture:
        assert styled.payload.texture_path == str(expected)


@pytest.mark.parametrize(
    "mesh_name", ["vegetation.ply", "ground.ply", "buildings_glass.ply", "water.ply"]
)
def test_cached_geographic_materials_match_cold_load(tmp_path, monkeypatch, mesh_name) -> None:
    monkeypatch.delenv("ORCHAV_DISABLE_SCENE_PAYLOAD_CACHE", raising=False)
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(tmp_path / "cache"))
    xml_path = _write_tiny_scene(tmp_path)
    (tmp_path / "wall.ply").rename(tmp_path / mesh_name)
    xml_path.write_text(xml_path.read_text().replace("wall.ply", mesh_name))
    (tmp_path / "osm_metadata.json").write_text("{}")
    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    cold = scene_io.build_scene_from_root(root, str(xml_path))[0]
    warm = scene_io.build_scene_from_root(root, str(xml_path))[0]
    assert warm["material_type"] == cold["material_type"]
    assert warm["pbr_properties"] == cold["pbr_properties"]
    assert warm["color"] == cold["color"]


def test_scene_payload_cache_invalidates_when_source_mesh_revision_changes(
    tmp_path,
    monkeypatch,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(cache_root))
    xml_path = _write_tiny_scene(tmp_path)
    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    first = scene_io.build_scene_from_root(root, str(xml_path))
    first_signature = first[0]["_source_signature"]
    assert len(list(cache_root.glob("scene.*.npz"))) == 1

    mesh_path = tmp_path / "wall.ply"
    revised = mesh_path.read_text(encoding="utf-8").replace("1 0 0", "2 0 0")
    mesh_path.write_text(revised, encoding="utf-8")
    current = time.time_ns()
    os.utime(mesh_path, ns=(current, current))

    revised_root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    second = scene_io.build_scene_from_root(revised_root, str(xml_path))

    assert len(list(cache_root.glob("scene.*.npz"))) == 2
    assert second[0]["_source_signature"] != first_signature
    assert float(second[0]["mesh"].vertices[:, 0].max()) > float(
        first[0]["mesh"].vertices[:, 0].max()
    )


def test_corrupt_scene_payload_cache_falls_back_to_source_and_repairs_file(
    tmp_path,
    monkeypatch,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(cache_root))
    xml_path = _write_tiny_scene(tmp_path)
    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    expected = scene_io.build_scene_from_root(root, str(xml_path))
    cache_path = next(cache_root.glob("scene.*.npz"))
    cache_path.write_bytes(b"corrupt")

    restored_root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    restored = scene_io.build_scene_from_root(restored_root, str(xml_path))

    np.testing.assert_array_equal(restored[0]["mesh"].vertices, expected[0]["mesh"].vertices)
    with np.load(cache_path, allow_pickle=False) as data:
        assert str(data["schema_version"].item()) == scene_io._SCENE_PAYLOAD_CACHE_VERSION


def test_scene_payload_disk_budget_evicts_oldest_files(
    tmp_path,
    monkeypatch,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_MAX_BYTES", str(1024 * 1024))

    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first_xml = _write_tiny_scene(first_dir)
    first_root = scene_io.XMLSceneHandler.load_xml_scene(str(first_xml))
    scene_io.build_scene_from_root(first_root, str(first_xml))
    first_path = next(cache_root.glob("scene.*.npz"))
    first_size = first_path.stat().st_size
    old = time.time() - 60.0
    os.utime(first_path, (old, old))

    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_MAX_BYTES", str(first_size))
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second_xml = _write_tiny_scene(second_dir)
    second_root = scene_io.XMLSceneHandler.load_xml_scene(str(second_xml))
    scene_io.build_scene_from_root(second_root, str(second_xml))

    info = scene_io.get_scene_payload_cache_info()
    assert not first_path.exists()
    assert info["bytes"] <= first_size
    assert info["entries"] <= 1


def test_clear_scene_payload_cache_reports_owned_disk_bytes(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORCHAV_SCENE_PAYLOAD_CACHE_DIR", str(cache_root))
    xml_path = _write_tiny_scene(tmp_path)
    root = scene_io.XMLSceneHandler.load_xml_scene(str(xml_path))
    scene_io.build_scene_from_root(root, str(xml_path))
    expected_bytes = sum(path.stat().st_size for path in cache_root.glob("*.npz"))

    result = scene_io.clear_scene_payload_cache()

    assert result == {"files": 1, "bytes": expected_bytes}
    assert scene_io.get_scene_payload_cache_info()["entries"] == 0
