"""Regression tests for auto-generated UVs on textured architectural meshes."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from visualizer.src.scene import io as scene_io
from visualizer.src.scene import uv_cache_store
from visualizer.src.scene.io import _generate_box_projection_uvs


def _make_mesh(vertices: np.ndarray, triangles: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return mesh


@pytest.fixture(autouse=True)
def _isolate_uv_cache_stores():
    """Keep process-local SQLite connections from leaking between tests."""
    uv_cache_store.close_uv_cache_stores(flush=False)
    scene_io._PRUNED_UV_CACHE_ROOTS.clear()
    yield
    uv_cache_store.close_uv_cache_stores(flush=False)
    scene_io._PRUNED_UV_CACHE_ROOTS.clear()


@pytest.mark.headless
def test_generate_box_projection_uvs_keeps_shared_corners_coherent_on_nearly_planar_wall() -> None:
    """Adjacent triangles with tiny normal drift should still share one UV basis.

    This models the real-world case of imported or generated meshes whose
    wall triangles are almost coplanar but not numerically identical.
    """
    mesh = _make_mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.02, 2.0, 3.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    uvs = _generate_box_projection_uvs(mesh, scale=0.5)

    assert uvs is not None
    assert uvs.shape == (6, 2)
    np.testing.assert_allclose(uvs[0], uvs[3], atol=1e-6)
    np.testing.assert_allclose(uvs[2], uvs[4], atol=1e-6)


@pytest.mark.headless
def test_generate_box_projection_uvs_maps_vertical_extent_to_world_up() -> None:
    mesh = _make_mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 2.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    uvs = _generate_box_projection_uvs(mesh, scale=0.5)

    assert uvs is not None
    np.testing.assert_allclose(uvs[0, 0], uvs[5, 0], atol=1e-6)
    np.testing.assert_allclose(abs(uvs[5, 1] - uvs[0, 1]), 1.0, atol=1e-6)


@pytest.mark.headless
def test_generate_box_projection_uvs_fast_path_keeps_shared_corners_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _make_mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.02, 2.0, 3.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    monkeypatch.setattr(scene_io, "_PATCH_PROJECTION_TRIANGLE_LIMIT", 1)

    uvs = _generate_box_projection_uvs(mesh, scale=0.5)

    assert uvs is not None
    assert uvs.shape == (6, 2)
    np.testing.assert_allclose(uvs[0], uvs[3], atol=1e-6)
    np.testing.assert_allclose(uvs[2], uvs[4], atol=1e-6)


@pytest.mark.headless
def test_load_or_generate_box_projection_uvs_reuses_persisted_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    scenario_dir = tmp_path / "scenarios" / "example_scene"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text("schema_version: 1\n")
    source_mesh = scenario_dir / "wall.ply"
    source_mesh.write_text("ply\nformat ascii 1.0\nend_header\n")
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path / "uv_cache"))
    scene_io._PRUNED_UV_CACHE_ROOTS.clear()

    transform_state = {
        "scale": 1.0,
        "rotation": [0.0, 0.0, 0.0],
        "translate": [0.0, 0.0, 0.0],
    }
    mesh1 = _make_mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 3.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    uvs_first = scene_io.load_or_generate_box_projection_uvs(
        mesh1,
        scale=0.5,
        cache_source_path=str(source_mesh),
        transform_state=transform_state,
    )

    assert uvs_first is not None
    assert uv_cache_store.flush_uv_cache_stores() == 1
    uv_cache_store.close_uv_cache_stores(flush=False)

    mesh2 = _make_mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 3.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    def _fail_if_regenerated(*args, **kwargs):
        raise AssertionError("expected persisted UV cache hit")

    monkeypatch.setattr(scene_io, "_generate_box_projection_uvs", _fail_if_regenerated)

    uvs_second = scene_io.load_or_generate_box_projection_uvs(
        mesh2,
        scale=0.5,
        cache_source_path=str(source_mesh),
        transform_state=transform_state,
    )

    assert uvs_second is not None
    np.testing.assert_allclose(uvs_second, uvs_first, atol=1e-6)


@pytest.mark.headless
def test_load_or_generate_box_projection_uvs_uses_repo_cache_layout_with_scenario_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    scenario_dir = tmp_path / "scenarios" / "osm" / "paris_west"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text("schema_version: 1\n")
    source_mesh = scenario_dir / "mesh" / "wall.ply"
    source_mesh.parent.mkdir(parents=True)
    source_mesh.write_text("ply\nformat ascii 1.0\nend_header\n")

    cache_root = tmp_path / "cache_root"
    monkeypatch.setenv("ORCHAV_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("ORCHAV_UV_CACHE_DIR", raising=False)
    scene_io._PRUNED_UV_CACHE_ROOTS.clear()

    mesh = _make_mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 3.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    uvs = scene_io.load_or_generate_box_projection_uvs(
        mesh,
        scale=0.5,
        cache_source_path=str(source_mesh),
        transform_state={"scale": 1.0},
    )

    assert uvs is not None
    assert uv_cache_store.flush_uv_cache_stores() == 1
    sqlite_files = list((cache_root / "uvcache").glob("*.uv.sqlite3"))
    assert len(sqlite_files) == 1
    assert sqlite_files[0].name.startswith("paris_west.")
    assert not list((cache_root / "uvcache").rglob("*.npy"))


@pytest.mark.headless
def test_uv_cache_prune_removes_stale_entries_once_per_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cache_root = tmp_path / "cache_root"
    uv_root = cache_root / "uvcache"
    uv_root.mkdir(parents=True)
    old_cache_file = uv_root / "old_scene.deadbeef.uv.sqlite3"
    old_cache_file.write_bytes(b"stale")

    stale_timestamp = time.time() - (5 * 24 * 60 * 60)
    os.utime(old_cache_file, (stale_timestamp, stale_timestamp))

    monkeypatch.setenv("ORCHAV_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("ORCHAV_UV_CACHE_MAX_AGE_DAYS", "1")
    monkeypatch.delenv("ORCHAV_UV_CACHE_DIR", raising=False)
    scene_io._PRUNED_UV_CACHE_ROOTS.clear()

    root = scene_io._uv_cache_root()

    assert root == cache_root / "uvcache"
    assert not old_cache_file.exists()


@pytest.mark.headless
def test_uv_cache_batches_entries_in_one_scenario_store_and_reports_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    scenario_dir = tmp_path / "scenarios" / "munich"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text("schema_version: 1\n")
    cache_root = tmp_path / "uv_cache"
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(cache_root))

    mesh = _make_mesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 3.0]],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )
    for name in ("wall_a.ply", "wall_b.ply"):
        source = scenario_dir / name
        source.write_text("ply\nformat ascii 1.0\nend_header\n")
        assert (
            scene_io.load_or_generate_box_projection_uvs(
                mesh,
                scale=0.5,
                cache_source_path=str(source),
                transform_state={"translate": [0.0, 0.0, 0.0]},
            )
            is not None
        )

    before = scene_io.get_persistent_uv_cache_info()
    assert before["stores"] == 1
    assert before["pending_entries"] == 2

    assert uv_cache_store.flush_uv_cache_stores() == 2
    uv_cache_store.close_uv_cache_stores(flush=False)
    after = scene_io.get_persistent_uv_cache_info()

    assert after["stores"] == 1
    assert after["stored_entries"] == 2
    assert after["pending_entries"] == 0
    assert len(list(cache_root.glob("*.uv.sqlite3"))) == 1
    assert not list(cache_root.rglob("*.npy"))


@pytest.mark.headless
def test_finalize_uv_cache_closes_stores_and_enforces_bytes_when_age_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cache_root = tmp_path / "uv_cache"
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("ORCHAV_UV_CACHE_MAX_AGE_DAYS", "0")
    monkeypatch.setenv("ORCHAV_UV_CACHE_MAX_BYTES", "1")
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    for namespace in ("first", "second"):
        store = uv_cache_store.get_uv_cache_store(cache_root, namespace)
        store.put("mesh", values, source_signature=namespace)

    removed = scene_io.finalize_uv_cache_stores()

    assert removed == 2
    assert not list(cache_root.glob("*.uv.sqlite3"))
    assert not any(Path(key).is_relative_to(cache_root) for key in uv_cache_store._STORES)


@pytest.mark.headless
def test_supplied_source_signature_avoids_restat_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    scenario_dir = tmp_path / "scenarios" / "signature_scene"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text("schema_version: 1\n")
    source = scenario_dir / "wall.ply"
    source.write_text("first revision")
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path / "uv_cache"))

    stat = source.stat()
    signature = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    mesh = _make_mesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 3.0]],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )
    expected = scene_io.load_or_generate_box_projection_uvs(
        mesh,
        scale=0.5,
        cache_source_path=str(source),
        source_signature=signature,
    )
    assert expected is not None
    assert uv_cache_store.flush_uv_cache_stores() == 1
    uv_cache_store.close_uv_cache_stores(flush=False)

    source.write_text("a newer revision with a different size")

    def _fail_if_regenerated(*_args, **_kwargs):
        raise AssertionError("the authoritative source signature should reuse the entry")

    monkeypatch.setattr(scene_io, "_generate_box_projection_uvs", _fail_if_regenerated)
    restored = scene_io.load_or_generate_box_projection_uvs(
        mesh,
        scale=0.5,
        cache_source_path=str(source),
        source_signature=signature,
    )
    np.testing.assert_array_equal(restored, expected)


def test_uv_cache_store_budget_evicts_oldest_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("ORCHAV_UV_CACHE_STORE_MAX_BYTES", "24")
    store = uv_cache_store.get_uv_cache_store(tmp_path, "scene")
    first = np.arange(6, dtype=np.float32).reshape(3, 2)
    second = first + 10.0
    store.put("first", first, source_signature="first-revision")
    store.put("second", second, source_signature="second-revision")

    assert store.flush() == 2
    assert store.get("first", expected_shape=(3, 2), source_signature="first-revision") is None
    restored = store.get("second", expected_shape=(3, 2), source_signature="second-revision")
    np.testing.assert_array_equal(restored, second)


def test_clear_uv_cache_is_root_scoped_and_preserves_unowned_files(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = uv_cache_store.get_uv_cache_store(first_root, "scene")
    second = uv_cache_store.get_uv_cache_store(second_root, "scene")
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    first.put("mesh", values, source_signature="one")
    second.put("mesh", values, source_signature="two")
    first.flush()
    second.flush()
    sentinel = first_root / "keep-me.txt"
    sentinel.write_text("not cache data")
    legacy_dir = first_root / "legacy-scene"
    legacy_dir.mkdir()
    legacy_bundle = legacy_dir / "bundle-old-format.npz"
    np.savez_compressed(legacy_bundle, uv=values)

    result = uv_cache_store.clear_uv_cache(first_root)

    assert result["files"] >= 1
    assert sentinel.read_text() == "not cache data"
    assert not list(first_root.glob("*.uv.sqlite3"))
    assert not legacy_bundle.exists()
    assert not legacy_dir.exists()
    assert list(second_root.glob("*.uv.sqlite3"))
    restored = second.get("mesh", expected_shape=(3, 2), source_signature="two")
    np.testing.assert_array_equal(restored, values)
