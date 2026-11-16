"""Optional skies never download implicitly or install unverified partial files."""

import hashlib
from dataclasses import replace
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.lighting import assets
from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager


@pytest.fixture
def asset(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHAV_IBL_CACHE_DIR", str(tmp_path / "user-cache"))
    data = b"small pinned test HDR payload"
    original = assets.environment_catalog()["sky_sunny"]
    return replace(original, size=len(data), sha256=hashlib.sha256(data).hexdigest()), data


def test_catalog_is_pinned_and_offline(monkeypatch):
    monkeypatch.setattr(assets, "urlopen", lambda *a, **k: pytest.fail("implicit network"))
    catalog = assets.environment_catalog()
    assert [a.label for a in catalog.values()] == ["Overcast", "Sunny", "Partly cloudy", "Sunset"]
    assert all(a.license == "CC0-1.0" and a.authors and a.size > 0 for a in catalog.values())
    assert all(a.url.endswith("_2k.hdr") and len(a.sha256) == 64 for a in catalog.values())
    assert len({a.cache_key for a in catalog.values()}) == 4


def test_download_verifies_and_reuses_offline(asset, monkeypatch):
    entry, data = asset
    calls = []

    def open_source(request, timeout):
        calls.append(request.full_url)
        assert timeout == 5
        return BytesIO(data)

    monkeypatch.setattr(assets, "urlopen", open_source)
    path = assets.download_environment(entry)
    assert path.read_bytes() == data
    assert assets.cached_environment(entry) == path
    assert assets.download_environment(entry) == path
    assert calls == [entry.url]
    assert not list(path.parent.glob("*.part"))


@pytest.mark.parametrize("bad", [b"short", b"x" * 29, b"x" * 100])
def test_bad_download_never_replaces_existing_file(asset, monkeypatch, bad):
    entry, _ = asset
    path = assets.source_path(entry)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"previous invalid bytes")
    monkeypatch.setattr(assets, "urlopen", lambda *a, **k: BytesIO(bad))
    with pytest.raises(ValueError):
        assets.download_environment(entry)
    assert path.read_bytes() == b"previous invalid bytes"
    assert not list(path.parent.glob("*.part"))


def test_cancellation_before_install_leaves_no_partial_file(asset, monkeypatch):
    entry, data = asset
    monkeypatch.setattr(assets, "urlopen", lambda *a, **k: BytesIO(data))
    cancel = Event()
    with pytest.raises(assets.DownloadCancelled):
        assets.download_environment(entry, cancelled=cancel, progress=lambda *_: cancel.set())
    assert not assets.source_path(entry).exists()
    assert not list(assets.source_path(entry).parent.glob("*.part"))


def test_cache_detects_corruption_and_recipe_changes(asset, monkeypatch):
    entry, data = asset
    monkeypatch.setattr(assets, "urlopen", lambda *a, **k: BytesIO(data))
    path = assets.download_environment(entry)
    assert assets.cached_environment(entry)
    path.write_bytes(b"x" * len(data))
    assert assets.cached_environment(entry) is None
    assert replace(entry, yaw_degrees=10).cache_key != entry.cache_key
    assert replace(entry, radiance_scale=0.2).cache_key != entry.cache_key


def test_lfs_pointer_has_actionable_error(tmp_path):
    pointer = tmp_path / "missing.hdr"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 10\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        PygfxIBLManager._read_rgbe(pointer)


def test_manager_applies_sky_preparation_and_reuses_external_cache(asset, monkeypatch, tmp_path):
    entry, data = asset
    entry = replace(entry, radiance_scale=0.25, yaw_degrees=90)
    monkeypatch.setattr(assets, "urlopen", lambda *a, **k: BytesIO(data))
    assets.download_environment(entry)
    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.lighting.environment_catalog", lambda: {entry.id: entry}
    )
    gfx = SimpleNamespace(Texture=lambda data, **kw: SimpleNamespace(data=data))
    manager = PygfxIBLManager(gfx, tmp_path / "read-only-library")
    source = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    manager._load_hdr = lambda *_: source.copy()
    captured = []

    def convert(image, face_size):
        captured.append(image.copy())
        return np.ones((6, face_size, face_size, 3), np.float32)

    manager._equirect_to_cubemap = convert
    manager._face_size = 2
    assert manager.load_ibl(entry.id) is not None
    np.testing.assert_array_equal(captured[0], np.roll(source, 1, axis=1) * 0.25)
    assert manager._cache_path(entry.id).is_relative_to(assets.ibl_cache_dir())
    assert not manager._ibl_dir.exists()
    other = PygfxIBLManager(gfx, tmp_path / "another-checkout")
    other._face_size = 2
    other._load_hdr = lambda *_: pytest.fail("cached cubemap should avoid decode")
    assert other.load_ibl(entry.id) is not None


def test_local_hdr_cache_invalidates_when_file_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHAV_IBL_CACHE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "custom.hdr"
    path.write_bytes(b"first")
    manager = PygfxIBLManager(SimpleNamespace(), tmp_path)
    before = manager._cache_path(str(path))
    path.write_bytes(b"second version")
    assert before != manager._cache_path(str(path))
