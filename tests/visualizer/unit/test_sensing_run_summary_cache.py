"""Persistent summaries preserve all-event totals and exact scoring settings."""

from dataclasses import fields
from threading import Event

import numpy as np
import pytest

from tests.visualizer.unit.test_sensing_run_summary_data import (
    CONTEXT,
    Service,
    event,
)
from tests.visualizer.unit.test_sensing_run_summary_data import (
    compact_evidence as compact_evidence,
)
from visualizer.src.sensing import run_summary_data as module
from visualizer.src.sensing.recording_cache import RecordingCache


@pytest.fixture
def service(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    reader = Service([event(i, detections=[] if i % 2 else None) for i in range(9)])
    reader.recording_cache = RecordingCache((source,), tmp_path / "cache")
    return reader


def _load(service, **kwargs):
    return module.load_run_summary(service, CONTEXT, Event(), max_plot_points=4, **kwargs)


def test_warm_summary_has_identical_totals_and_samples_without_source_rows(service):
    original = _load(service)
    assert original.acquisition_count == 9
    assert len(original.time_s) == 4
    service.loads.clear()
    restored = _load(service)
    assert service.loads == []
    for field in fields(original):
        expected, actual = getattr(original, field.name), getattr(restored, field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
            assert not actual.flags.writeable
        else:
            assert actual == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"range_gate": 3.0},
        {"velocity_gate": 3.0},
        {"confirmed_only": True},
        {"interval": (2.0, 6.0)},
    ],
)
def test_different_settings_never_reuse_summary(service, settings):
    _load(service)
    service.loads.clear()
    second = _load(service, **settings)
    assert service.loads
    assert second.summary["range_gate_m"] == settings.get("range_gate", module.DEFAULT_RANGE_GATE_M)


def test_corrupt_part_rebuilds_and_source_change_refuses_stale_data(service):
    original = _load(service)
    next(service.recording_cache.root.glob("*.npz")).write_bytes(b"bad")
    service.loads.clear()
    rebuilt = _load(service)
    assert service.loads and rebuilt.summary == original.summary
    service.recording_cache.paths[0].write_text('{"new":true}')
    service.loads.clear()
    assert _load(service).summary == original.summary
    assert service.loads


def test_build_requires_successful_summary_publication(service, monkeypatch):
    service.recording_cache._building = True
    monkeypatch.setattr(module, "save_component", lambda *_args, **_kwargs: False)
    with pytest.raises(OSError, match="summary"):
        _load(service)


def test_cancelled_summary_does_not_publish_partial_totals(service):
    cancelled = Event()
    reader = service.dashboard_load_row

    def stop(storage, products):
        result = reader(storage, products)
        cancelled.set()
        return result

    service.dashboard_load_row = stop
    result = module.load_run_summary(service, CONTEXT, cancelled)
    assert result.cancelled and not result.summary
    assert not service.recording_cache.root.exists()


def test_zero_detection_run_caches_all_events_without_inventing_observations(service):
    service.events = [event(i, detections=[]) for i in range(9)]
    first = _load(service)
    assert first.acquisition_count == first.evaluated_count == 9
    assert first.summary["total_tp"] == first.summary["total_fp"] == 0
    assert first.summary["total_fn"] == 9
    assert first.output_time_s.size == 0
    service.loads.clear()
    restored = _load(service)
    assert restored.summary == first.summary and service.loads == []
    np.testing.assert_array_equal(restored.event_output_count, np.zeros(4))
