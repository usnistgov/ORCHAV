from types import SimpleNamespace

import numpy as np
import pytest

from shared.geometry import cache as scene_geometry_cache
from shared.geometry import scene as scene_geometry_loader
from shared.geometry.transforms import UnsupportedLightweightTransformError


def test_get_scene_geometry_caches_by_resolved_xml_path(monkeypatch, tmp_path):
    scene_geometry_cache.clear_scene_geometry_cache()
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene/>")
    geometry = [{"mesh": object()}]
    calls = []

    def fake_load_geometry(path):
        calls.append(path)
        return geometry

    monkeypatch.setattr(scene_geometry_cache, "_load_geometry", fake_load_geometry)

    assert scene_geometry_cache.get_scene_geometry(scene_xml=xml_path) is geometry
    assert scene_geometry_cache.get_scene_geometry(scene_xml=str(xml_path)) is geometry
    assert calls == [xml_path]


def test_transform_contract_error_is_not_treated_as_optional_geometry(monkeypatch, tmp_path):
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text("<scene/>", encoding="utf-8")

    def reject_transform(_path):
        raise UnsupportedLightweightTransformError("unsupported transform")

    monkeypatch.setattr(scene_geometry_loader, "load_scene_geometry", reject_transform)

    with pytest.raises(UnsupportedLightweightTransformError, match="unsupported transform"):
        scene_geometry_cache._load_geometry(xml_path)


def test_compute_bounds_from_geometry_skips_invalid_vertices():
    scene_geometry_cache.clear_scene_geometry_cache()
    geometry = [
        {
            "mesh": SimpleNamespace(
                vertices=np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [np.nan, 10.0, 5.0],
                        [4.0, 8.0, 2.0],
                        [np.inf, 2.0, 3.0],
                    ],
                    dtype=float,
                )
            )
        }
    ]

    assert scene_geometry_cache.compute_xy_bounds_from_geometry(geometry) == (
        -2.0,
        6.0,
        -2.0,
        10.0,
    )
    assert scene_geometry_cache.compute_xyz_bounds_from_geometry(geometry) == (
        -5.0,
        9.0,
        -5.0,
        13.0,
        -1.0,
        3.0,
    )


def test_geometry_cache_stats_track_known_buffers_and_activity(monkeypatch, tmp_path):
    scene_geometry_cache.clear_scene_geometry_cache()
    baseline = scene_geometry_cache.get_scene_geometry_cache_stats()
    xml_path = tmp_path / "scene.xml"
    mesh_path = tmp_path / "wall.ply"
    xml_path.write_text("<scene/>", encoding="utf-8")
    mesh_path.write_bytes(b"mesh-v1")
    vertices = np.zeros((5, 3), dtype=np.float64)
    triangles = np.zeros((2, 3), dtype=np.int32)
    geometry = [
        {
            "mesh": SimpleNamespace(vertices=vertices, triangles=triangles),
            "source_xml": str(xml_path),
            "full_path": str(mesh_path),
        }
    ]
    monkeypatch.setattr(scene_geometry_cache, "_load_geometry", lambda _path: geometry)

    assert scene_geometry_cache.get_scene_geometry(scene_xml=xml_path) is geometry
    assert scene_geometry_cache.get_scene_geometry(scene_xml=xml_path) is geometry

    stats = scene_geometry_cache.get_scene_geometry_cache_stats()
    assert stats["entries"] == 1
    assert stats["mesh_entries"] == 1
    assert stats["current_bytes"] == vertices.nbytes + triangles.nbytes
    assert stats["peak_bytes"] >= stats["current_bytes"]
    assert stats["hits"] == baseline["hits"] + 1
    assert stats["misses"] == baseline["misses"] + 1
    assert stats["evictions"] == 0
    assert stats["revision_tracked_entries"] == 1
    assert stats["source_revision_mismatches"] == 0

    scene_geometry_cache.clear_scene_geometry_cache()
    cleared = scene_geometry_cache.get_scene_geometry_cache_stats()
    assert cleared["entries"] == 0
    assert cleared["current_bytes"] == 0
    assert cleared["peak_bytes"] >= stats["current_bytes"]
    assert cleared["clears"] == stats["clears"] + 1


def test_geometry_revision_stats_do_not_invalidate_changed_sources(monkeypatch, tmp_path):
    scene_geometry_cache.clear_scene_geometry_cache()
    xml_path = tmp_path / "scene.xml"
    mesh_path = tmp_path / "wall.ply"
    xml_path.write_text("<scene/>", encoding="utf-8")
    mesh_path.write_bytes(b"mesh-v1")
    geometry = [
        {
            "mesh": SimpleNamespace(vertices=np.zeros((1, 3), dtype=np.float64)),
            "source_xml": str(xml_path),
            "full_path": str(mesh_path),
        }
    ]
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return geometry

    monkeypatch.setattr(scene_geometry_cache, "_load_geometry", fake_load)

    assert scene_geometry_cache.get_scene_geometry(scene_xml=xml_path) is geometry
    initial_stats = scene_geometry_cache.get_scene_geometry_cache_stats()
    assert initial_stats["source_revision_mismatches"] == 0

    mesh_path.write_bytes(b"mesh-v2-with-a-different-size")
    stats = scene_geometry_cache.get_scene_geometry_cache_stats()

    assert stats["source_revision_mismatches"] == 1
    assert scene_geometry_cache.get_scene_geometry(scene_xml=xml_path) is geometry
    assert load_calls == [xml_path]
    scene_geometry_cache.clear_scene_geometry_cache()
