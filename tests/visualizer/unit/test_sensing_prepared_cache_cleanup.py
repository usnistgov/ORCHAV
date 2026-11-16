"""Explicit bundle cleanup respects the existing reader/preparation locks."""

import gc
import sqlite3
from threading import Event

from shared.frames.providers import Hdf5Provider
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from visualizer.src.sensing.recording_cache import RecordingCache, clear_prepared_cache
from visualizer.src.sensing.startup_cache import prepare_startup_cache


def test_cleanup_skips_active_bundle_then_removes_only_disposable_data(recording):
    root, _ = recording
    provider = Hdf5Provider(root)
    originals = {p: p.read_bytes() for p in (root / "frames").iterdir() if p.is_file()}
    try:
        prepare_startup_cache(provider, Event())
        cache = RecordingCache.for_provider(provider)
        bundle, cache_root = cache._prepared_root, cache.root
        assert bundle is not None
        assert clear_prepared_cache(cache_root)["removed_bundles"] == 0
        assert bundle.is_dir()
        del cache
        gc.collect()  # Retired service/scoped-cache cycles can conservatively retain leases.
        cleaned = clear_prepared_cache(cache_root)
        assert cleaned["removed_bundles"] == 1
        assert cleaned["removed_bytes"] > 0
        assert not bundle.exists()
        assert not list((cache_root / "recordings").glob("*.ready"))
        assert list((cache_root / "recordings").glob("*.lock"))
        assert all(p.read_bytes() == content for p, content in originals.items())
        assert prepare_startup_cache(provider, Event())
    finally:
        provider.close()


def test_cleanup_cannot_race_preparation_or_follow_unrecognized_paths(recording):
    root, _ = recording
    provider = Hdf5Provider(root)
    try:
        prepare_startup_cache(provider, Event())
        cache = RecordingCache.for_provider(provider)
        cache_root, signature, bundle = cache.root, cache.signature, cache._prepared_root
        del cache
        gc.collect()
        recordings = cache_root / "recordings"
        connection = sqlite3.connect(recordings / f"{signature[:20]}.lock")
        try:
            connection.execute("BEGIN IMMEDIATE")
            assert clear_prepared_cache(cache_root)["removed_bundles"] == 0
            assert bundle.exists()
        finally:
            connection.close()
        incomplete = recordings / "bundle_0000000000000000"
        incomplete.mkdir()
        (incomplete / "partial.npz").write_bytes(b"in progress")
        link = recordings / "bundle_1111111111111111"
        link.symlink_to(root / "frames", target_is_directory=True)
        cleaned = clear_prepared_cache(cache_root)
        assert cleaned["removed_bundles"] == 1
        assert cleaned["skipped_bundles"] == 2
        assert (incomplete / "partial.npz").read_bytes() == b"in progress"
        assert link.is_symlink()
        assert (root / "frames" / "frames_manifest.json").is_file()
    finally:
        provider.close()
