"""Prepared views keep one result owner and never stall a ready foreground hit."""

from concurrent.futures import TimeoutError
from threading import Event

import numpy as np

from shared.cache_sizing import estimate_retained_bytes
from shared.frames.providers import Hdf5Provider
from tests.visualizer.unit.test_sensing_dashboard_reuse import _held_provider
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from visualizer.src.sensing import dashboard_data as module
from visualizer.src.sensing.dashboard_data import DashboardSession
from visualizer.src.sensing.prepared_angle import prepare_angular


def test_held_inspection_upgrade_keeps_one_result_cache_owner(monkeypatch):
    session = DashboardSession(_held_provider())
    try:
        original = session._load_request((1, "analysis", "auto", True, "auto"))
        owner_key = next(iter(session._request_cache))
        inspected = session._load_request((2, "diagnostics", "auto", True, "auto"))
        assert inspected.rd_result is original.rd_result
        assert inspected.replay_event.key == original.replay_event.key
        assert inspected.state_step == 2
        assert inspected.replay_event.scene_binding.current_scene_frame_id == 2
        assert list(session._request_cache) == [owner_key]
        owner, size = session._request_cache[owner_key]
        assert owner.state_step == original.state_step
        assert owner.replay_event.scene_binding == original.replay_event.scene_binding
        assert owner.detection_stage_bundle is inspected.detection_stage_bundle
        assert {"diagnostics", "analysis-inspection"} <= owner.prepared_views
        assert size == estimate_retained_bytes(owner) == session._request_cache_bytes

        def unexpected(*args, **kwargs):
            raise AssertionError("The same saved inspection must remain prepared")

        monkeypatch.setattr(module, "load_dashboard_request", unexpected)
        monkeypatch.setattr(session.service, "load_detection_stage_bundle", unexpected)
        for step in (1, 2, 1):
            repeated = session._load_request((step, "diagnostics", "auto", True, "auto"))
            assert repeated.detection_stage_bundle is inspected.detection_stage_bundle
            assert repeated.state_step == step
        assert list(session._request_cache) == [owner_key]
    finally:
        session.close()


def test_ready_request_does_not_wait_for_inflight_speculative_preparation(monkeypatch):
    session = DashboardSession(_held_provider())
    entered = Event()
    release = Event()
    background = None
    try:
        key = (0, "analysis", "auto", True, "auto")
        ready = session._cached_request(key)
        load = module.load_dashboard_request

        def slow_load(service, step, *args):
            if step == 3:
                entered.set()
                if not release.wait(5):
                    raise AssertionError("Speculative test read was not released")
            return load(service, step, *args)

        monkeypatch.setattr(module, "load_dashboard_request", slow_load)
        background = session._executor.submit(
            session._cached_request, (3, "analysis", "auto", True, "auto"), prefetch=True
        )
        assert entered.wait(5)
        foreground = session._executor.submit(session._cached_request, key)
        try:
            result = foreground.result(timeout=1)
        except TimeoutError as exc:
            raise AssertionError("A ready request waited behind speculative preparation") from exc
        assert result is ready
        detached = session._executor.submit(
            session._cached_request, (0, "live", "auto", True, "auto", ("range_doppler",))
        )
        assert detached.result(timeout=1) is ready
        assert not release.is_set()
    finally:
        release.set()
        if background is not None:
            background.result(timeout=5)
        session.close()


def test_new_pinned_angle_is_prefetched_when_current_maps_are_already_ready(recording):
    source, payloads = recording
    provider = Hdf5Provider(source)
    session = DashboardSession(provider)
    try:
        key = (0, "analysis", "auto", True, "auto")
        current = session._load_request(key)
        assert "angular" not in current.prepared_views
        session.prefetch_next((*key, ("range_doppler_angle",)))
        assert session._prefetch_pending is not None
        session._prefetch_pending.result(timeout=5)
        assert len(session._request_cache) == 1
        prepared, _size = session._request_cache[key]
        assert "angular" in prepared.prepared_views
        assert prepared.prepared_angular is not None
        np.testing.assert_array_equal(
            prepared.prepared_angular.range_azimuth_power,
            prepare_angular(payloads[0]).range_azimuth_power,
        )
        assert "range_doppler_angle_by_pair" not in prepared.display_frame["sensing"]
        assert prepared.rd_result is current.rd_result
        assert prepared.state_step == 0
    finally:
        session.close()
        provider.close()
