"""Exact component tiling retains the ordinary safe numeric cache boundary."""

from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import pytest

from visualizer.src.sensing import component_cache as module
from visualizer.src.sensing import recording_cache
from visualizer.src.sensing.recording_cache import RecordingCache


@pytest.fixture
def cache(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    return RecordingCache((source,), tmp_path / "cache")


def _save(cache, arrays, **kwargs):
    return module.save_component(
        cache, "overview", ("cohort",), {"label": "µ"}, arrays, protocol="exact-v1", **kwargs
    )


def _load(cache, **kwargs):
    return module.load_component(cache, "overview", ("cohort",), protocol="exact-v1", **kwargs)


def test_component_larger_than_archive_ceiling_round_trips_exactly(cache):
    # The real 64 MiB codec limit remains in force, not a patched approximation.
    values = np.arange(9 * 1024 * 1024, dtype=np.float64).reshape(9216, 1024)
    assert values.nbytes > recording_cache.MAX_ARRAY_ENTRY_BYTES
    assert _save(cache, {"power": values, "empty": np.empty((0, 3)), "scalar": np.array(-0.0)})
    parts = list(cache.root.glob("*.npz"))
    assert len(parts) > 3
    assert all(p.stat().st_size < recording_cache.MAX_ARRAY_ENTRY_BYTES for p in parts)
    metadata, arrays = _load(cache, byte_limit=values.nbytes + 8)
    assert metadata == {"label": "µ"}
    assert arrays["power"].tobytes() == values.tobytes()
    assert arrays["empty"].shape == (0, 3)
    assert np.signbit(arrays["scalar"])
    with pytest.raises(ValueError):
        arrays["power"].setflags(write=True)


def test_declared_budget_rejects_before_numeric_reads(cache, monkeypatch):
    assert _save(cache, {"x": np.arange(20, dtype=np.float32)})
    monkeypatch.setattr(
        cache, "load_arrays", lambda *_: pytest.fail("numeric read before admission")
    )
    assert _load(cache, byte_limit=79) is None


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_incomplete_part_is_a_miss(cache, damage):
    assert _save(cache, {"x": np.arange(3)})
    part = next(cache.root.glob("*.npz"))
    if damage == "missing":
        part.unlink()
    else:
        part.write_bytes(b"damaged archive")
    assert _load(cache) is None


def test_failed_replacement_keeps_previous_committed_component(cache, monkeypatch):
    original = np.array([np.nan, np.inf, -np.inf, -0.0], dtype=np.float32)
    assert _save(cache, {"x": original})
    monkeypatch.setattr(cache, "save", lambda *_: False)
    assert not _save(cache, {"x": np.arange(4, dtype=np.float32)})
    assert _load(cache)[1]["x"].tobytes() == original.tobytes()


def test_cancelled_component_does_not_commit(cache, monkeypatch):
    cancelled = Event()
    save_arrays = cache.save_arrays

    def stop_after_part(*args, **kwargs):
        result = save_arrays(*args, **kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(cache, "save_arrays", stop_after_part)
    with pytest.raises(CancelledError):
        _save(cache, {"x": np.arange(8)}, cancelled=cancelled)
    assert _load(cache) is None


def test_protocol_and_source_changes_are_misses(cache):
    assert _save(cache, {"x": np.arange(4)})
    assert module.load_component(cache, "overview", ("cohort",), protocol="exact-v2") is None
    cache.paths[0].write_text('{"changed":true}')
    assert _load(cache) is None


def test_bad_manifest_shape_is_rejected_before_numeric_reads(cache, monkeypatch):
    assert _save(cache, {"x": np.arange(4)})
    key = (module.COMPONENT_VERSION, "exact-v1", "cohort")
    manifest = cache.load("overview-component", key)
    manifest["arrays"]["x"]["shape"] = [-1]
    assert cache.save("overview-component", manifest, key)
    monkeypatch.setattr(cache, "load_arrays", lambda *_: pytest.fail("malformed shape read"))
    assert _load(cache) is None
