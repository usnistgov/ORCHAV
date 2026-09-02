"""Focused lifecycle tests for reusable visualizer asset caches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from tests.visualizer.fixtures.mock_factories import make_mock_visualizer
from visualizer.src.cache import asset_cache_coordinator as coordinator
from visualizer.src.cache import pygfx_mesh_buffers as mesh_cache
from visualizer.src.materials import texture_assets
from visualizer.src.services import cache_service as cache_service_module
from visualizer.src.services.cache_service import CacheService


@pytest.fixture(autouse=True)
def _reset_process_cache_state():
    """Isolate byte-budgeted process caches and their root memoization."""
    mesh_cache.clear_pygfx_mesh_buffer_cache(memory=True, disk=False)
    mesh_cache._ROOTS.clear()
    mesh_cache._FAILED_ROOT_RETRY_AT.clear()
    mesh_cache._WRITE_RETRY_AT.clear()
    mesh_cache._PRUNED_ROOTS.clear()
    mesh_cache._TOUCHED_PATHS.clear()
    mesh_cache._DISK_LRU.clear()
    mesh_cache._DISK_LRU_BYTES.clear()
    texture_assets.clear_decoded_texture_cache()
    yield
    mesh_cache.clear_pygfx_mesh_buffer_cache(memory=True, disk=False)
    mesh_cache._ROOTS.clear()
    mesh_cache._FAILED_ROOT_RETRY_AT.clear()
    mesh_cache._WRITE_RETRY_AT.clear()
    mesh_cache._PRUNED_ROOTS.clear()
    mesh_cache._TOUCHED_PATHS.clear()
    mesh_cache._DISK_LRU.clear()
    mesh_cache._DISK_LRU_BYTES.clear()
    texture_assets.clear_decoded_texture_cache()


def _configure_mesh_cache(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    monkeypatch.setenv("ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR", str(root))
    monkeypatch.delenv("ORCHAV_DISABLE_PYGFX_MESH_BUFFER_CACHE", raising=False)
    resolved = mesh_cache.get_pygfx_mesh_buffer_cache_root()
    assert resolved is not None
    return resolved


def _fake_gfx_texture_module():
    """Return a stable fake gfx module whose textures retain copied pixels."""

    def _texture(values, *, dim):
        return SimpleNamespace(data=np.array(values, copy=True), dim=dim)

    return SimpleNamespace(Texture=_texture)


def _split_all_plan() -> dict[str, np.uint8]:
    return {"mode": mesh_cache.PYGFX_MESH_CACHE_MODE_SPLIT_ALL}


def _store_plan(cache_key: str) -> tuple[str, Path]:
    identity = mesh_cache.pygfx_mesh_cache_identity(cache_key)
    path = mesh_cache.resolve_pygfx_mesh_buffer_cache_path(
        cache_key,
        cache_identity=identity,
    )
    assert path is not None
    mesh_cache.store_pygfx_mesh_cache_plan(
        cache_key,
        cache_identity=identity,
        vertex_count=3,
        triangle_count=1,
        uv_count=3,
        plan=_split_all_plan(),
    )
    return identity, path


def test_pygfx_cache_paths_are_opaque_contained_and_collision_resistant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = _configure_mesh_cache(monkeypatch, tmp_path / "prepared")

    first = mesh_cache.resolve_pygfx_mesh_buffer_cache_path("../../outside/cache.npz")
    second = mesh_cache.resolve_pygfx_mesh_buffer_cache_path("..\\outside\\cache.npz")

    assert first is not None and second is not None
    assert first != second
    assert first.resolve(strict=False).is_relative_to(root.resolve(strict=False))
    assert second.resolve(strict=False).is_relative_to(root.resolve(strict=False))
    assert len(first.stem) == 64
    assert first.parent.name == first.stem[:2]
    assert (
        mesh_cache.resolve_pygfx_mesh_buffer_cache_path("key", cache_identity="../../escape")
        is None
    )


def test_pygfx_cache_identity_changes_with_source_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _configure_mesh_cache(monkeypatch, tmp_path / "prepared")
    source = tmp_path / "mesh.ply"
    source.write_text("first")
    first = mesh_cache.pygfx_mesh_cache_identity(str(source))

    source.write_text("second revision")
    second = mesh_cache.pygfx_mesh_cache_identity(str(source))

    assert first != second


def test_corrupt_or_stale_pygfx_plan_is_deleted_and_rebuilt_on_next_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _configure_mesh_cache(monkeypatch, tmp_path / "prepared")
    identity, path = _store_plan("mesh-one")
    assert path.is_file()

    path.write_bytes(b"not a valid npz archive")
    assert (
        mesh_cache.load_pygfx_mesh_cache_plan(
            "mesh-one",
            cache_identity=identity,
            vertex_count=3,
            triangle_count=1,
            uv_count=3,
        )
        is None
    )
    assert not path.exists()

    identity, path = _store_plan("mesh-one")
    assert (
        mesh_cache.load_pygfx_mesh_cache_plan(
            "mesh-one",
            cache_identity=identity,
            vertex_count=4,
            triangle_count=1,
            uv_count=3,
        )
        is None
    )
    assert not path.exists()
    assert mesh_cache.get_pygfx_mesh_buffer_cache_info()["metrics"]["disk_invalidations"] >= 2


def test_prepared_mesh_memory_cache_is_true_lru_and_returns_writable_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHAV_PYGFX_MESH_BUFFER_MEMORY_CACHE_MAX_BYTES", "48")
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first + 10.0
    third = first + 20.0
    mesh_cache.store_prepared_pygfx_mesh_buffers("first", {"positions": first})
    mesh_cache.store_prepared_pygfx_mesh_buffers("second", {"positions": second})

    first_hit = mesh_cache.load_prepared_pygfx_mesh_buffers("first")
    assert first_hit is not None
    assert first_hit["positions"].flags.writeable
    first_hit["positions"][0, 0] = -1.0
    mesh_cache.store_prepared_pygfx_mesh_buffers("third", {"positions": third})

    assert mesh_cache.load_prepared_pygfx_mesh_buffers("second") is None
    restored_first = mesh_cache.load_prepared_pygfx_mesh_buffers("first")
    assert restored_first is not None
    np.testing.assert_array_equal(restored_first["positions"], first)
    assert mesh_cache.load_prepared_pygfx_mesh_buffers("third") is not None
    info = mesh_cache.get_pygfx_mesh_buffer_cache_info()["memory"]
    assert info == {"entries": 2, "bytes": 48, "max_bytes": 48}


def test_pygfx_disk_budget_evicts_oldest_owned_plan_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = _configure_mesh_cache(monkeypatch, tmp_path / "prepared")
    _first_identity, first_path = _store_plan("first")
    first_size = first_path.stat().st_size
    monkeypatch.setenv("ORCHAV_PYGFX_MESH_CACHE_MAX_BYTES", str(first_size))

    _second_identity, second_path = _store_plan("second")

    assert not first_path.exists()
    assert second_path.exists()
    assert second_path.stat().st_size <= first_size
    sentinel = root / "not-an-owned-cache.npz"
    sentinel.write_text("preserve")
    result = mesh_cache.clear_pygfx_mesh_buffer_cache(memory=False, disk=True)
    assert result["disk_files"] == 1
    assert sentinel.read_text() == "preserve"


def _write_rgba(path: Path, color: tuple[int, int, int, int], *, size=(2, 2)) -> None:
    pixels = np.empty((size[1], size[0], 4), dtype=np.uint8)
    pixels[:] = color
    Image.fromarray(pixels, mode="RGBA").save(path)


def test_decoded_texture_cache_reuses_immutable_pixels_and_tracks_revisions(
    tmp_path,
) -> None:
    texture = tmp_path / "albedo.png"
    _write_rgba(texture, (10, 20, 30, 255))

    first = texture_assets.load_decoded_texture(texture)
    again = texture_assets.load_decoded_texture(texture)
    assert first is not None and again is first
    assert not first.rgba.flags.writeable

    _write_rgba(texture, (200, 100, 50, 255), size=(3, 2))
    revised = texture_assets.load_decoded_texture(texture)

    assert revised is not None and revised is not first
    assert revised.identity != first.identity
    np.testing.assert_array_equal(revised.rgba[0, 0], [200, 100, 50, 255])


def test_decoded_texture_cache_enforces_byte_lru_and_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("ORCHAV_DECODED_TEXTURE_CACHE_MAX_BYTES", "32")
    paths = [tmp_path / f"texture_{index}.png" for index in range(3)]
    for index, path in enumerate(paths):
        _write_rgba(path, (index, index, index, 255))

    first = texture_assets.load_decoded_texture(paths[0])
    second = texture_assets.load_decoded_texture(paths[1])
    assert texture_assets.load_decoded_texture(paths[0]) is first
    third = texture_assets.load_decoded_texture(paths[2])
    assert first is not None and second is not None and third is not None
    assert texture_assets.get_decoded_texture_cache_info()["entries"] == 2

    redecoded_second = texture_assets.load_decoded_texture(paths[1])
    assert redecoded_second is not None and redecoded_second is not second
    cleared = texture_assets.clear_decoded_texture_cache()
    assert cleared == {"entries": 2, "bytes": 32}
    assert texture_assets.get_decoded_texture_cache_info()["entries"] == 0


def test_notebook_native_texture_cache_restats_in_place_revisions(tmp_path) -> None:
    from visualizer.src.backends import pygfx_scene_helpers

    pygfx_scene_helpers.clear_pygfx_native_texture_cache()
    gfx = _fake_gfx_texture_module()
    texture = tmp_path / "albedo.png"
    _write_rgba(texture, (10, 20, 30, 255), size=(2, 2))
    first = pygfx_scene_helpers._load_pygfx_texture(gfx, str(texture))

    _write_rgba(texture, (200, 100, 50, 255), size=(3, 2))
    second = pygfx_scene_helpers._load_pygfx_texture(gfx, str(texture))

    assert first is not None and second is not None and second is not first
    np.testing.assert_array_equal(second.data[0, 0], [200, 100, 50, 255])
    pygfx_scene_helpers.clear_pygfx_native_texture_cache()


def test_notebook_native_texture_negative_cache_allows_new_file(tmp_path) -> None:
    from visualizer.src.backends import pygfx_scene_helpers

    pygfx_scene_helpers.clear_pygfx_native_texture_cache()
    gfx = _fake_gfx_texture_module()
    texture = tmp_path / "appears_later.png"
    assert pygfx_scene_helpers._load_pygfx_texture(gfx, str(texture)) is None

    _write_rgba(texture, (1, 2, 3, 255))
    loaded = pygfx_scene_helpers._load_pygfx_texture(gfx, str(texture))

    assert loaded is not None
    np.testing.assert_array_equal(loaded.data[0, 0], [1, 2, 3, 255])
    pygfx_scene_helpers.clear_pygfx_native_texture_cache()


def test_asset_cache_snapshot_reports_separate_storage_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visualizer.src.backends import pygfx_scene_helpers
    from visualizer.src.scene import io as scene_io

    monkeypatch.setattr(
        scene_io,
        "get_scene_payload_cache_info",
        lambda: {"entries": 2, "bytes": 200, "max_bytes": 1_000, "root": "scene"},
    )
    monkeypatch.setattr(
        scene_io,
        "get_persistent_uv_cache_info",
        lambda: {"entries": 3, "bytes": 300, "max_bytes": 2_000, "root": "uv"},
    )
    monkeypatch.setattr(
        texture_assets,
        "get_decoded_texture_cache_info",
        lambda: {"entries": 4, "bytes": 400, "max_bytes": 3_000},
    )
    monkeypatch.setattr(
        mesh_cache,
        "get_pygfx_mesh_buffer_cache_info",
        lambda: {
            "root": "prepared",
            "memory": {"entries": 5, "bytes": 500, "max_bytes": 4_000},
            "disk": {"files": 6, "bytes": 600, "max_bytes": 5_000},
        },
    )
    monkeypatch.setattr(
        pygfx_scene_helpers,
        "get_pygfx_native_texture_cache_info",
        lambda: {"entries": 7, "bytes": 700, "max_bytes": 6_000},
    )
    renderer = SimpleNamespace(
        get_native_asset_cache_info=lambda: {
            "entries": 8,
            "bytes": 800,
            "max_bytes": 7_000,
        }
    )

    snapshot = coordinator.collect_static_asset_cache_snapshot(renderer).as_dict()

    assert tuple(snapshot["layers"]) == (
        "scene_payload_disk",
        "generated_uv_disk",
        "decoded_texture_memory",
        "prepared_mesh_memory",
        "prepared_mesh_disk",
        "shared_native_textures",
        "renderer_native_textures",
    )
    assert snapshot["aggregate"]["disk"] == {
        "layers": 3,
        "entries": 11,
        "bytes": 1_100,
        "max_bytes": 8_000,
        "unknown_entry_layers": 0,
        "unknown_byte_layers": 0,
        "unbudgeted_layers": 0,
    }
    assert snapshot["aggregate"]["memory"]["entries"] == 9
    assert snapshot["aggregate"]["native"]["entries"] == 15


def test_asset_clear_respects_memory_only_mode_and_isolates_owner_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visualizer.src.backends import pygfx_scene_helpers
    from visualizer.src.scene import io as scene_io

    calls: list[str] = []
    monkeypatch.setattr(
        texture_assets,
        "clear_decoded_texture_cache",
        lambda: calls.append("decoded") or {"entries": 1},
    )
    monkeypatch.setattr(
        mesh_cache,
        "clear_pygfx_mesh_buffer_cache",
        lambda **kwargs: calls.append(f"prepared:{kwargs['disk']}") or {"disk_files": 0},
    )
    monkeypatch.setattr(
        pygfx_scene_helpers,
        "clear_pygfx_native_texture_cache",
        lambda: calls.append("shared-native") or {"entries": 1},
    )
    monkeypatch.setattr(
        scene_io,
        "clear_scene_payload_cache",
        lambda: calls.append("scene-disk") or {"files": 1},
    )
    monkeypatch.setattr(
        scene_io,
        "clear_persistent_uv_cache",
        lambda: calls.append("uv-disk") or {"files": 1},
    )

    def _renderer_failure():
        calls.append("renderer-native")
        raise RuntimeError("native device already closed")

    result = coordinator.clear_static_asset_caches(
        SimpleNamespace(clear_native_asset_cache=_renderer_failure),
        include_disk=False,
    )

    assert calls == ["decoded", "prepared:False", "shared-native", "renderer-native"]
    assert "error" in result["renderer_native_textures"]
    assert "scene_payload_disk" not in result
    assert "generated_uv_disk" not in result


def test_cache_service_keeps_frame_and_static_asset_lifecycles_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visualizer = make_mock_visualizer()
    visualizer.mpc_view_cache = {}
    visualizer.mpc_core = Mock()
    visualizer.animation_service = Mock()
    visualizer.animation_service.clear_preload_data.return_value = 0
    visualizer.renderer = SimpleNamespace(last_frame_packet=None, vis_initialized=False)
    visualizer.material_pbr_service = SimpleNamespace(invalidate_material_resolution_cache=Mock())
    visualizer.target_asset_cache = SimpleNamespace(
        clear_inactive_assets=Mock(return_value={"entries": 2, "bytes": 64, "pending": 1})
    )
    clear_assets = Mock(return_value={"decoded_texture_memory": {"entries": 1}})
    monkeypatch.setattr(cache_service_module, "clear_reusable_asset_caches", clear_assets)
    service = CacheService(visualizer)
    service.store_frame(0, {"frame": True})

    service.clear_local_frame_caches(reason="unit")
    clear_assets.assert_not_called()

    result = service.clear_static_asset_caches(reason="unit")
    clear_assets.assert_called_once_with(visualizer.renderer, include_disk=True)
    visualizer.material_pbr_service.invalidate_material_resolution_cache.assert_called_once_with()
    visualizer.target_asset_cache.clear_inactive_assets.assert_called_once_with()
    assert result["target_assets"] == {"entries": 2, "bytes": 64, "pending": 1}
