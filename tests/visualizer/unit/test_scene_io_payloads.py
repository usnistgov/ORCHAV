"""Tests for renderer-neutral XML scene payload loading."""

from __future__ import annotations

import os
import time

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
