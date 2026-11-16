"""Prepared display caches preserve numeric arrays and remain disposable."""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from visualizer.src.sensing import recording_cache as module
from visualizer.src.sensing.recording_cache import RecordingCache


@pytest.fixture
def cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"chunks": ["chunk.h5"]}')
    chunk = tmp_path / "chunk.h5"
    chunk.write_bytes(b"recorded source")
    return RecordingCache((manifest, chunk), root=tmp_path / "display-cache")


def _replace_member(path, name, payload):
    with zipfile.ZipFile(path) as archive:
        members = [(entry.filename, archive.read(entry.filename)) for entry in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for member_name, content in members:
            archive.writestr(member_name, payload if member_name == name else content)


@pytest.fixture(params=("json", "numeric"))
def published_entry(cache, request):
    if request.param == "numeric":

        def save(value):
            assert cache.save_arrays("plots", {}, {"value": np.array([value])})

        def load():
            result = cache.load_arrays("plots")
            return None if result is None else result[1]["value"][0]

        local = cache._array_path("plots", ())
    else:

        def save(value):
            cache.save("overview", {"value": value})

        def load():
            result = cache.load("overview")
            return None if result is None else result["value"]

        local = cache._path("overview", ())
    save(1)
    bundle = cache.root / "recordings" / "bundle_test"
    bundle.mkdir(parents=True)
    published = bundle / local.name
    local.replace(published)
    cache._prepared_root = bundle
    return cache, local, published, save, load


@pytest.mark.parametrize("corrupt_local", (False, True))
def test_missing_or_corrupt_local_entry_falls_back_to_published_bundle(
    published_entry, corrupt_local
):
    cache, local, _published, _save, load = published_entry
    if corrupt_local:
        local.write_bytes(b"interrupted cache write")
    assert load() == 1
    assert cache.hits == 1


@pytest.mark.parametrize("corrupt_bundle", (False, True))
def test_repaired_local_entry_takes_priority_over_published_bundle(published_entry, corrupt_bundle):
    cache, _local, published, save, load = published_entry
    if corrupt_bundle:
        published.write_bytes(b"damaged published entry")
        assert load() is None
    # Also covers a structurally valid bundle whose display model needed rebuilding.
    save(2)
    assert load() == 2
    assert load() == 2
    assert cache.hits == 2


def test_semantic_numeric_admission_falls_back_before_promotion(cache, monkeypatch):
    key = ("selected",)
    assert cache.save_arrays("plots", {"scope": "partial"}, {"rd": np.ones(2)}, key)
    local = cache._array_path("plots", key)
    bundle = cache.root / "recordings" / "bundle_partial"
    bundle.mkdir(parents=True)
    local.replace(bundle / local.name)
    assert cache.save_arrays(
        "plots", {"scope": "complete"}, {"rd": np.ones(2), "rp": np.zeros(3)}, key
    )
    ordinary = cache.root
    complete_bytes = local.read_bytes()
    partial_bytes = (bundle / local.name).read_bytes()
    monkeypatch.setattr(module, "_cache_root", lambda: ordinary)
    cache.root = ordinary / "stage"
    cache._prepared_root = bundle
    cache._building = True
    seen, saved = [], []
    original_save = cache.save_arrays

    def accept(metadata, arrays):
        seen.append(metadata["scope"])
        return {"rd", "rp"}.issubset(arrays)

    def save(kind, metadata, arrays, key=()):
        saved.append(metadata["scope"])
        return original_save(kind, metadata, arrays, key)

    monkeypatch.setattr(cache, "save_arrays", save)
    metadata, arrays = cache.load_arrays("plots", key, accept=accept)
    assert metadata == {"scope": "complete"}
    assert set(arrays) == {"rd", "rp"}
    assert seen == ["partial", "complete"]
    assert saved == ["complete"]
    assert cache.hits == 1
    assert local.read_bytes() == complete_bytes
    assert (bundle / local.name).read_bytes() == partial_bytes
    assert cache._array_path("plots", key).is_file()


def test_semantic_numeric_admission_rechecks_source_after_predicate(cache):
    assert cache.save_arrays("plots", {}, {"rd": np.ones(2)})

    def changed_source(_metadata, _arrays):
        cache.paths[-1].write_bytes(b"changed recorded source")
        return True

    assert cache.load_arrays("plots", accept=changed_source) is None
    assert cache.hits == 0


def test_source_change_during_bundle_read_is_not_returned(published_entry, monkeypatch):
    cache, _local, _published, _save, load = published_entry
    checks = iter((True, False))
    monkeypatch.setattr(cache, "unchanged", lambda: next(checks))
    assert load() is None
    assert cache.hits == 0


def test_numeric_roundtrip_is_exact_and_immutable_across_reopen(cache):
    arrays = {
        "range projection": np.arange(24, dtype=np.float32).reshape(4, 6),
        "complex": np.array([1 + 2j, -3j], dtype=np.complex128),
        "mask": np.array([True, False]),
        "scalar": np.array(3, dtype=np.int16),
        "empty": np.empty((0, 2), dtype=np.float64),
        "fortran": np.asfortranarray(np.arange(12, dtype=np.uint64).reshape(3, 4)),
        "strided": np.arange(20, dtype=np.float64)[::2],
        "big_endian": np.arange(5, dtype=">f8"),
    }
    metadata = {"variant": "primary", "count": np.int64(12), "axes": ["azimuth", "range"]}
    assert cache.save_arrays("plots", metadata, arrays, key=(127, "primary"))
    reopened = RecordingCache(cache.paths, root=cache.root)
    loaded_metadata, loaded_arrays = reopened.load_arrays("plots", (127, "primary"))
    assert loaded_metadata == {"variant": "primary", "count": 12, "axes": ["azimuth", "range"]}
    assert reopened.hits == 1
    for name, expected in arrays.items():
        actual = loaded_arrays[name]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.setflags(write=True)


@pytest.mark.parametrize("source_index", [0, 1])
def test_source_change_invalidates_existing_and_reopened_cache(cache, source_index):
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    source = cache.paths[source_index]
    original = source.read_bytes()
    stamp = source.stat()
    original_change_time = module._change_time(source, stamp)
    source.write_bytes(original[:-1] + b"x")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    if source_index == 1:
        # Chunk cache identity uses change time; the manifest also hashes bytes.
        # Rapid same-size writes can share one filesystem timestamp tick.
        deadline = time.monotonic() + 2.0
        while module._change_time(source, source.stat()) == original_change_time:
            assert time.monotonic() < deadline, "Fixture change time did not advance"
            time.sleep(0.01)
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert source.stat().st_size == stamp.st_size
    assert source.stat().st_mtime_ns == stamp.st_mtime_ns
    assert source.read_bytes() != original
    assert cache.load_arrays("plots") is None
    assert not cache.save_arrays("plots", {}, {"power": np.zeros(4)})
    assert RecordingCache(cache.paths, root=cache.root).load_arrays("plots") is None


def test_numeric_cache_version_is_part_of_identity(cache, monkeypatch):
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    monkeypatch.setattr(module, "ARRAY_CACHE_VERSION", module.ARRAY_CACHE_VERSION + 1)
    assert cache.load_arrays("plots") is None


def test_fingerprint_reuses_path_spelling_without_repeated_resolution(cache, monkeypatch):
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    cache.save("overview", {"results": 1})

    def unexpected(*args, **kwargs):
        raise AssertionError("Warm fingerprint checks must not resolve every source path")

    monkeypatch.setattr(Path, "resolve", unexpected)
    assert cache.unchanged()
    assert cache.load_arrays("plots") is not None
    assert cache.load("overview") == {"results": 1}


def test_retargeted_source_symlink_invalidates_cached_data(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    first.write_bytes(b"first-source")
    second.write_bytes(b"other-source")
    source_link = tmp_path / "chunk.h5"
    source_link.symlink_to(first)
    cache = RecordingCache((manifest, source_link), tmp_path / "display-cache")
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    assert cache.load_arrays("plots") is not None
    source_link.unlink()
    source_link.symlink_to(second)
    assert not cache.unchanged()
    assert cache.load_arrays("plots") is None
    assert RecordingCache(cache.paths, cache.root).load_arrays("plots") is None


def test_truncated_archive_falls_back(cache):
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    path = cache._array_path("plots", ())
    path.write_bytes(path.read_bytes()[:100])
    assert cache.load_arrays("plots") is None


@pytest.mark.parametrize(
    "field,value", [("version", -1), ("signature", "old"), ("identity", "other")]
)
def test_mismatched_envelope_falls_back(cache, field, value):
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    path = cache._array_path("plots", ())
    with zipfile.ZipFile(path) as archive:
        envelope = json.loads(archive.read("metadata.json"))
    envelope[field] = value
    _replace_member(path, "metadata.json", json.dumps(envelope).encode())
    assert cache.load_arrays("plots") is None


def test_object_arrays_are_never_saved_or_loaded(cache):
    assert not cache.save_arrays("plots", {}, {"unsafe": np.array([{"x": 1}], dtype=object)})
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    payload = io.BytesIO()
    np.save(payload, np.array([{"x": 1}], dtype=object))
    _replace_member(cache._array_path("plots", ()), "array_0.npy", payload.getvalue())
    assert cache.load_arrays("plots") is None


def test_oversized_declared_shape_is_rejected_before_allocation(cache):
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        payload, {"descr": "<f8", "fortran_order": False, "shape": (2**50,)}
    )
    _replace_member(cache._array_path("plots", ()), "array_0.npy", payload.getvalue())
    assert cache.load_arrays("plots") is None


def test_numeric_byte_budget_and_metadata_budget(cache, monkeypatch):
    monkeypatch.setattr(module, "MAX_ARRAY_ENTRY_BYTES", 128)
    assert not cache.save_arrays("plots", {}, {"power": np.ones(100)})
    assert not list(cache.root.glob("*.npz"))
    monkeypatch.setattr(module, "MAX_ARRAY_ENTRY_BYTES", 2**20)
    monkeypatch.setattr(module, "MAX_ARRAY_METADATA_BYTES", 128)
    assert not cache.save_arrays("plots", {"label": "x" * 1000}, {"power": np.ones(4)})


def test_source_change_during_read_is_not_returned(cache, monkeypatch):
    assert cache.save_arrays("plots", {}, {"power": np.ones(4)})
    checks = iter((True, False))
    monkeypatch.setattr(cache, "unchanged", lambda: next(checks))
    assert cache.load_arrays("plots") is None
    assert cache.hits == 0


def test_interrupted_save_preserves_previous_entry_and_removes_temporary(cache, monkeypatch):
    assert cache.save_arrays("plots", {"label": "old"}, {"power": np.ones(4)})

    def fail(*args, **kwargs):
        raise OSError("interrupted write")

    monkeypatch.setattr(np, "save", fail)
    assert not cache.save_arrays("plots", {"label": "new"}, {"power": np.zeros(4)})
    assert not list(cache.root.glob("*.pending"))
    metadata, arrays = cache.load_arrays("plots")
    assert metadata["label"] == "old"
    np.testing.assert_array_equal(arrays["power"], np.ones(4))


def test_source_change_before_publish_discards_temporary(cache, monkeypatch):
    checks = iter((True, False))
    monkeypatch.setattr(cache, "unchanged", lambda: next(checks))
    assert not cache.save_arrays("plots", {}, {"power": np.ones(4)})
    assert not list(cache.root.glob("*.pending"))
    assert not list(cache.root.glob("*.npz"))


def test_result_cache_checks_its_manifest_and_source_chunk(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cache = RecordingCache((manifest, first, second), tmp_path / "cache")
    cache._frame_path = lambda step: {1: first, 2: second}.get(step)
    scoped = cache.for_frame(1)
    assert scoped.paths == (manifest, first)
    assert scoped.save_arrays("plots", {}, {"power": np.ones(4)}, (1,))

    second.write_bytes(b"changed unrelated chunk")
    assert scoped.load_arrays("plots", (1,)) is not None
    assert cache.hits == 1
    assert not cache.unchanged()
    first.write_bytes(b"changed selected chunk")
    assert scoped.load_arrays("plots", (1,)) is None
    assert not scoped.save_arrays("plots", {}, {"power": np.ones(4)}, (1,))


def test_result_cache_rejects_changes_before_scope_is_created(tmp_path):
    manifest = tmp_path / "manifest.json"
    chunk = tmp_path / "chunk.h5"
    manifest.write_text("{}")
    chunk.write_bytes(b"original")
    cache = RecordingCache((manifest, chunk), tmp_path / "cache")
    cache._frame_path = lambda _step: chunk
    chunk.write_bytes(b"changed")
    assert cache.for_frame(1) is cache
    assert not cache.unchanged()


def test_denied_cache_creation_returns_without_retrying(cache, monkeypatch):
    attempts = []

    def denied(path, flags, mode):
        attempts.append(path)
        raise PermissionError("cache directory is not writable")

    monkeypatch.setattr(module.os, "open", denied)
    assert not cache.save_arrays("plots", {}, {"power": np.ones(4)})
    assert len(attempts) == 1
    cache.save("overview", {"results": 1})
    assert len(attempts) == 2


def test_eighty_cpis_coexist_with_json_overview(cache):
    cache.save("overview", {"results": 80})
    for index in range(80):
        assert cache.save_arrays("plots", {"result": index}, {"power": np.ones(4)}, (index,))
    assert len(list(cache.root.glob("*.npz"))) == 80
    assert cache.load("overview") == {"results": 80}
    for index in (0, 40, 79):
        assert cache.load_arrays("plots", (index,))[0]["result"] == index


def test_array_count_and_shared_disk_byte_limits(cache, monkeypatch):
    monkeypatch.setattr(module, "MAX_ARRAY_ENTRIES", 2)
    cache.save("overview", {"results": 3})
    os.utime(cache._path("overview", ()), ns=(1, 1))
    for index in range(3):
        assert cache.save_arrays("plots", {}, {"power": np.ones(4)}, (index,))
        # Filesystem timestamp ties do not define a chronological order.
        stamp = (index + 1) * 1_000_000_000
        os.utime(cache._array_path("plots", (index,)), ns=(stamp, stamp))
    assert len(list(cache.root.glob("*.npz"))) == 2
    assert cache.load_arrays("plots", (0,)) is None
    assert cache.load("overview") == {"results": 3}
    newest = cache._array_path("plots", (2,))
    monkeypatch.setattr(module, "MAX_DISK_BYTES", newest.stat().st_size)
    cache._prune()
    remaining = list(cache.root.glob("*.npz")) + list(cache.root.glob("*.json.gz"))
    assert sum(path.stat().st_size for path in remaining) <= module.MAX_DISK_BYTES
    assert newest in remaining


def test_unwritable_cache_directory_is_optional(cache, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked")
    cache.root = blocked
    assert not cache.save_arrays("plots", {}, {"power": np.ones(4)})
    assert cache.load_arrays("plots") is None


@pytest.mark.parametrize("numeric", (False, True))
def test_build_reuses_only_valid_ordinary_entries(cache, tmp_path, monkeypatch, numeric):
    monkeypatch.setattr(module, "_cache_root", lambda: cache.root)
    metadata = {"result": 3}
    arrays = {"power": np.arange(4.0)}
    if numeric:
        assert cache.save_arrays("plots", metadata, arrays)
        source = cache._array_path("plots", ())
    else:
        cache.save("overview", metadata)
        source = cache._path("overview", ())
    original = source.read_bytes()
    building = RecordingCache(cache.paths, tmp_path / "staging")
    building._building = True

    if numeric:
        restored, values = building.load_arrays("plots")
        np.testing.assert_array_equal(values["power"], arrays["power"])
    else:
        restored = building.load("overview")
    assert restored == metadata
    assert source.read_bytes() == original
    assert (building.root / source.name).is_file()

    source.unlink()
    if numeric:
        assert building.load_arrays("plots")[0] == metadata
    else:
        assert building.load("overview") == metadata


@pytest.mark.parametrize("numeric", (False, True))
@pytest.mark.parametrize("invalid", ("corrupt_entry", "changed_source"))
def test_build_does_not_promote_invalid_ordinary_entries(
    cache, tmp_path, monkeypatch, numeric, invalid
):
    monkeypatch.setattr(module, "_cache_root", lambda: cache.root)
    if numeric:
        assert cache.save_arrays("plots", {}, {"power": np.arange(4.0)})
        source = cache._array_path("plots", ())
    else:
        cache.save("overview", {"result": 3})
        source = cache._path("overview", ())
    building = RecordingCache(cache.paths, tmp_path / "staging")
    building._building = True
    if invalid == "corrupt_entry":
        source.write_bytes(b"incomplete cache entry")
    else:
        cache.paths[0].write_text('{"changed": true}')

    result = building.load_arrays("plots") if numeric else building.load("overview")
    assert result is None
    assert not (building.root / source.name).exists()
