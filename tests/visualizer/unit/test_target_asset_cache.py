"""Focused tests for bounded, lazy target-asset ownership."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Lock
from unittest.mock import Mock

import numpy as np
import pytest

from visualizer.src.materials.catalog import ResolvedMaterial
from visualizer.src.materials.texture_policy import TexturePolicyResult
from visualizer.src.model import RenderObjectState
from visualizer.src.scene.target_materials import target_entry_pbr_fields
from visualizer.src.scene.target_transforms import TargetGeometryMeta
from visualizer.src.services.target_asset_cache import (
    DEFAULT_TARGET_ASSET_CACHE_ENTRIES,
    ResolvedTargetAssetSource,
    TargetAsset,
    TargetAssetCache,
    TargetAssetSource,
)
from visualizer.src.services.target_service import TargetService
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    SurfaceColorSource,
)


@pytest.fixture
def cache_factory() -> Callable[..., TargetAssetCache]:
    """Return tracked caches whose worker threads are closed after each test."""
    caches: list[TargetAssetCache] = []

    def create(**kwargs: object) -> TargetAssetCache:
        cache = TargetAssetCache(**kwargs)
        caches.append(cache)
        return cache

    yield create
    for cache in caches:
        cache.close()


def _write_source(path: Path, payload: bytes = b"target") -> None:
    path.write_bytes(payload)


def _asset(
    source: ResolvedTargetAssetSource,
    *,
    offset: float = 0.0,
    vertex_count: int = 3,
) -> TargetAsset:
    vertices = np.arange(vertex_count * 3, dtype=np.float64).reshape((-1, 3)) + offset
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    mesh = RenderObjectState(
        id=f"target:{source.target_name}:{source.mesh_filename}",
        payload=MeshPayload(vertices=vertices.copy(), triangles=triangles),
    )
    return TargetAsset(
        source=source,
        mesh=mesh,
        original_vertices=vertices.copy(),
        scaled_vertices=vertices.copy(),
        geometry_meta=TargetGeometryMeta(scaled_aabb_center=vertices.mean(axis=0)),
    )


def _registered_asset(
    cache: TargetAssetCache,
    target_name: str,
    path: Path,
    *,
    offset: float = 0.0,
    vertex_count: int = 3,
) -> TargetAsset:
    source = cache.register_source(target_name, path).resolve()
    return _asset(source, offset=offset, vertex_count=vertex_count)


def test_source_key_uses_canonical_path_and_concrete_file_revision(tmp_path: Path) -> None:
    path = tmp_path / "person.ply"
    _write_source(path, b"first")
    source = TargetAssetSource.from_path("person", path)

    first = source.resolve()
    _write_source(path, b"a different payload size")
    second = source.resolve()

    assert first.key.canonical_path == os.path.normcase(str(path.resolve()))
    assert first.key.target_name == "person"
    assert first.key.mesh_filename == "person.ply"
    assert first.key.revision != second.key.revision


def test_sequence_registration_does_not_stat_animation_frames(
    cache_factory: Callable[..., TargetAssetCache],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = cache_factory()
    paths = [tmp_path / f"frame_{index:05d}.ply" for index in range(10_000)]
    stat_calls: list[Path] = []
    original_stat = Path.stat
    animation_paths = {os.path.normcase(str(path)) for path in paths}

    def reject_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        if os.path.normcase(str(path)) in animation_paths:
            stat_calls.append(path)
            raise AssertionError("sequence registration must not stat individual frames")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", reject_stat)
    sources = cache.register_sequence("person", paths)

    assert len(sources) == len(paths)
    assert sources[-1].mesh_filename == "frame_09999.ply"
    assert stat_calls == []


def test_default_capacity_holds_a_typical_short_animation_sequence(
    cache_factory: Callable[..., TargetAssetCache],
    tmp_path: Path,
) -> None:
    cache = cache_factory()
    paths = [tmp_path / f"etoile_{index:05d}.ply" for index in range(37)]
    for index, path in enumerate(paths):
        _write_source(path)
        cache.put(_registered_asset(cache, "etoile", path, offset=float(index)))

    inventory = cache.telemetry()

    assert DEFAULT_TARGET_ASSET_CACHE_ENTRIES >= len(paths)
    assert inventory["entries"] == len(paths)
    assert inventory["evictions"] == 0
    assert inventory["bytes"] < inventory["max_bytes"]


def test_entry_and_byte_budgets_evict_the_true_lru(
    cache_factory: Callable[..., TargetAssetCache],
    tmp_path: Path,
) -> None:
    entry_cache = cache_factory(max_entries=2, max_bytes=10**9)
    paths = [tmp_path / f"entry_{name}.ply" for name in "abc"]
    for path in paths:
        _write_source(path)
    first = _registered_asset(entry_cache, "person", paths[0], offset=1.0)
    second = _registered_asset(entry_cache, "person", paths[1], offset=2.0)
    third = _registered_asset(entry_cache, "person", paths[2], offset=3.0)
    entry_cache.put(first)
    entry_cache.put(second)

    assert entry_cache.get(*first.logical_key) is first
    entry_cache.put(third)

    assert entry_cache.asset_for_logical_key(first.logical_key) is first
    assert entry_cache.asset_for_logical_key(second.logical_key) is None
    assert entry_cache.asset_for_logical_key(third.logical_key) is third
    assert entry_cache.telemetry()["evictions"] == 1

    byte_cache = cache_factory(max_entries=10, max_bytes=1)
    byte_first = _registered_asset(byte_cache, "car", paths[0], vertex_count=64)
    byte_second = _registered_asset(byte_cache, "car", paths[1], vertex_count=64)
    one_asset_budget = max(byte_first.estimated_bytes, byte_second.estimated_bytes) + 1
    byte_cache.configure(max_bytes=one_asset_budget)
    byte_cache.put(byte_first)
    byte_cache.put(byte_second)

    assert byte_cache.asset_for_logical_key(byte_first.logical_key) is None
    assert byte_cache.asset_for_logical_key(byte_second.logical_key) is byte_second
    assert byte_cache.telemetry()["bytes"] <= one_asset_budget


def test_pin_handoff_switches_when_only_one_asset_fits_the_byte_budget(
    cache_factory: Callable[..., TargetAssetCache],
    tmp_path: Path,
) -> None:
    cache = cache_factory(max_entries=10, max_bytes=10**9)
    paths = [tmp_path / f"handoff_{index}.ply" for index in range(2)]
    for path in paths:
        _write_source(path)
    current = _registered_asset(cache, "person", paths[0], vertex_count=64)
    replacement = _registered_asset(cache, "person", paths[1], vertex_count=64)
    one_asset_budget = max(current.estimated_bytes, replacement.estimated_bytes) + 1
    assert current.estimated_bytes + replacement.estimated_bytes > one_asset_budget
    cache.configure(max_bytes=one_asset_budget)
    cache.put(current)
    cache.pin(current)

    cache.pin_handoff(replacement, current)

    assert cache.asset_for_logical_key(current.logical_key) is None
    assert cache.asset_for_logical_key(replacement.logical_key) is replacement
    assert cache.telemetry()["pinned"] == 1
    assert cache.telemetry()["bytes"] <= one_asset_budget


def test_pinning_and_clear_inactive_preserve_the_presented_asset(
    cache_factory: Callable[..., TargetAssetCache],
    tmp_path: Path,
) -> None:
    cache = cache_factory(max_entries=2, max_bytes=10**9)
    paths = [tmp_path / f"frame_{index}.ply" for index in range(3)]
    for path in paths:
        _write_source(path)
    sources = cache.register_sequence("person", paths)
    assets = [_asset(source.resolve(), offset=float(index)) for index, source in enumerate(sources)]
    cache.put(assets[0])
    cache.put(assets[1])
    cache.pin(assets[0])
    cache.put(assets[2])

    assert cache.asset_for_logical_key(assets[0].logical_key) is assets[0]
    assert cache.asset_for_logical_key(assets[1].logical_key) is None
    assert cache.asset_for_logical_key(assets[2].logical_key) is assets[2]

    cleared = cache.clear_inactive_assets()

    assert cleared["entries"] == 1
    assert cache.asset_for_logical_key(assets[0].logical_key) is assets[0]
    assert cache.asset_for_logical_key(assets[2].logical_key) is None
    assert cache.source_for("person", paths[2].name) is not None
    assert cache.telemetry()["pinned"] == 1


@pytest.mark.parametrize(
    ("current", "direction", "expected"),
    [
        ("c.ply", 1, ["d.ply", "a.ply"]),
        ("a.ply", -1, ["d.ply", "c.ply"]),
    ],
)
def test_lookahead_follows_playback_direction_and_wraps(
    cache_factory: Callable[..., TargetAssetCache],
    tmp_path: Path,
    current: str,
    direction: int,
    expected: list[str],
) -> None:
    cache = cache_factory(prefetch_workers=1)
    paths = [tmp_path / f"{name}.ply" for name in "abcd"]
    for path in paths:
        _write_source(path)
    cache.register_sequence("person", paths)
    loaded: list[str] = []
    all_callbacks_finished = Event()
    release_loader = Event()

    def loader(source: ResolvedTargetAssetSource) -> TargetAsset:
        loaded.append(source.mesh_filename)
        assert release_loader.wait(timeout=5.0)
        return _asset(source)

    assert (
        cache.prefetch_after(
            "person",
            current,
            count=2,
            loader=loader,
            direction=direction,
        )
        == 2
    )
    with cache._lock:
        futures = [future for _, future in cache._pending.values()]
    remaining = len(futures)

    def finished(_: Future[TargetAsset]) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining == 0:
            all_callbacks_finished.set()

    for future in futures:
        future.add_done_callback(finished)
    release_loader.set()
    assert all_callbacks_finished.wait(timeout=5.0)

    assert loaded == expected
    assert cache.telemetry()["prefetch_completed"] == 2
    assert [
        name for name in expected if cache.asset_for_logical_key(("person", name)) is not None
    ] == expected


def test_default_prefetch_pool_prepares_independent_targets_concurrently(
    cache_factory: Callable[..., TargetAssetCache],
    tmp_path: Path,
) -> None:
    cache = cache_factory(max_entries=1)
    paths = {
        target_name: [
            tmp_path / f"{target_name}_a.ply",
            tmp_path / f"{target_name}_b.ply",
        ]
        for target_name in ("first", "second")
    }
    for target_name, sequence in paths.items():
        for path in sequence:
            _write_source(path)
        cache.register_sequence(target_name, sequence)

    started_count = 0
    started_lock = Lock()
    both_started = Event()
    release_loaders = Event()
    callbacks_finished = Event()

    def loader(source: ResolvedTargetAssetSource) -> TargetAsset:
        nonlocal started_count
        with started_lock:
            started_count += 1
            if started_count == 2:
                both_started.set()
        assert release_loaders.wait(timeout=5.0)
        return _asset(source)

    for target_name in paths:
        assert (
            cache.prefetch_after(
                target_name,
                paths[target_name][0].name,
                count=1,
                loader=loader,
            )
            == 1
        )

    assert both_started.wait(timeout=5.0)
    with cache._lock:
        futures = [future for _, future in cache._pending.values()]
    remaining = len(futures)

    def finished(_: Future[TargetAsset]) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining == 0:
            callbacks_finished.set()

    for future in futures:
        future.add_done_callback(finished)
    release_loaders.set()
    assert callbacks_finished.wait(timeout=5.0)

    telemetry = cache.telemetry()
    assert telemetry["prefetch_completed"] == 2
    assert telemetry["entries"] == telemetry["max_entries"] == 1
    assert telemetry["evictions"] == 1


def test_get_or_load_rechecks_asset_published_after_initial_miss(
    cache_factory: Callable[..., TargetAssetCache],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = cache_factory()
    path = tmp_path / "person.ply"
    _write_source(path)
    asset = _registered_asset(cache, "person", path)
    loader = Mock(side_effect=AssertionError("published prefetch must not parse twice"))

    def miss_while_prefetch_publishes(_target_name: str, _mesh_filename: str):
        cache.put(asset)
        return None

    monkeypatch.setattr(cache, "get", miss_while_prefetch_publishes)

    loaded = cache.get_or_load("person", path.name, loader)

    assert loaded is asset
    loader.assert_not_called()


def test_completed_prefetch_is_rejected_after_generation_invalidation(
    cache_factory: Callable[..., TargetAssetCache],
    tmp_path: Path,
) -> None:
    cache = cache_factory(prefetch_workers=1)
    paths = [tmp_path / "a.ply", tmp_path / "b.ply"]
    for path in paths:
        _write_source(path)
    cache.register_sequence("person", paths)
    loader_started = Event()
    release_loader = Event()
    callback_finished = Event()

    def loader(source: ResolvedTargetAssetSource) -> TargetAsset:
        loader_started.set()
        assert release_loader.wait(timeout=5.0)
        return _asset(source)

    assert (
        cache.prefetch_after(
            "person",
            "a.ply",
            count=1,
            loader=loader,
        )
        == 1
    )
    assert loader_started.wait(timeout=5.0)
    with cache._lock:
        future = next(iter(cache._pending.values()))[1]
    future.add_done_callback(lambda _: callback_finished.set())

    cleared = cache.clear_inactive_assets()
    release_loader.set()
    assert callback_finished.wait(timeout=5.0)

    assert cleared["pending"] == 1
    assert cache.asset_for_logical_key(("person", "b.ply")) is None
    assert cache.telemetry()["prefetch_completed"] == 0


def test_target_build_spec_preserves_complete_pbr_and_intrinsic_vertex_colors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "colored.ply"
    _write_source(source_path)
    source = TargetAssetSource.from_path("person", source_path).resolve()
    colors = np.asarray(
        [[index / 20.0, 0.25, 1.0 - index / 20.0] for index in range(20)],
        dtype=np.float32,
    )
    payload = MeshPayload(
        vertices=np.asarray([[float(index), 0.0, 0.0] for index in range(20)]),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        vertex_colors=colors.copy(),
    )
    material = MaterialPayload(
        base_color=(0.2, 0.3, 0.4, 0.65),
        roughness=0.23,
        metallic=0.74,
        reflectance=0.81,
        texture_path="albedo.png",
        clearcoat=0.42,
        clearcoat_roughness=0.17,
        anisotropy=0.31,
        emissive_color=(0.1, 0.2, 0.3),
        emissive_intensity=1.7,
        transmission=0.26,
        glass_thickness=0.08,
        absorption_color=(0.7, 0.8, 0.9),
        normal_map_path="normal.png",
        roughness_map_path="roughness.png",
        ao_map_path="ao.png",
        metallic_map_path="metallic.png",
        normal_map_strength=1.4,
        uv_scale_meters=3.5,
        uv_repeat_scale=(2.0, 3.0),
        shader_variant="defaultLitSSR",
    )
    texture_policy = TexturePolicyResult(
        textures_enabled=True,
        active_maps={},
        color_editable=False,
        renderer_base_color=material.base_color,
    )
    resolved = ResolvedMaterial(
        properties={"color": list(material.base_color[:3])},
        texture_policy=texture_policy,
        payload=material,
    )
    spec = TargetService._target_asset_build_spec(
        scale=1.0,
        orientation=(0.0, 0.0, 0.0),
        use_ply_position=False,
        pbr_props={},
        resolved_material=resolved,
    )
    service = object.__new__(TargetService)
    mesh = RenderObjectState(id="target:person::mesh", payload=payload)
    monkeypatch.setattr(service, "_target_mesh_handle", lambda **_: mesh)

    asset = service._build_target_asset(source, spec)

    np.testing.assert_array_equal(asset.mesh.payload.vertex_colors, colors)
    assert asset.mesh.payload.color_source is SurfaceColorSource.VERTEX
    assert asset.has_vertex_texture is True
    assert asset.mesh.material.base_color == pytest.approx((1.0, 1.0, 1.0, 0.65))
    assert asset.mesh.material.roughness == pytest.approx(0.23)
    assert asset.mesh.material.metallic == pytest.approx(0.74)
    assert asset.mesh.material.reflectance == pytest.approx(0.81)
    assert asset.mesh.material.clearcoat == pytest.approx(0.42)
    assert asset.mesh.material.emissive_intensity == pytest.approx(1.7)
    assert asset.mesh.material.transmission == pytest.approx(0.26)
    assert asset.mesh.material.texture_path is None
    assert asset.mesh.material.normal_map_path is None
    assert asset.mesh.material.roughness_map_path is None
    assert asset.mesh.material.ao_map_path is None
    assert asset.mesh.material.metallic_map_path is None

    fields = target_entry_pbr_fields(
        {
            "color": [0.2, 0.3, 0.4],
            "roughness": 0.23,
            "clearcoat": 0.42,
            "normal_map_path": "normal.png",
            "shader_variant": "defaultLitSSR",
        }
    )
    assert fields["pbr_properties"]["clearcoat"] == pytest.approx(0.42)
    assert fields["pbr_properties"]["normal_map_path"] == "normal.png"
    assert fields["pbr_properties"]["shader_variant"] == "defaultLitSSR"
