"""Opt-in sensing observations never become another data-loading path."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

from tests.visualizer.unit.test_sensing_dashboard_reuse import _held_provider
from visualizer.src.sensing import dashboard_data as module
from visualizer.src.sensing.dashboard_data import DashboardSession


def test_disabled_diagnostics_never_time_requests_or_collect_reader_events(monkeypatch):
    def unexpected():
        raise AssertionError("Disabled diagnostics read the timing clock")

    monkeypatch.setattr(module, "perf_counter", unexpected)
    session = DashboardSession(_held_provider())
    try:
        session.request((0, "analysis", "primary", True, "auto"))
        assert session.pending.result(timeout=5) is not None
        snapshot = session.diagnostics_snapshot()
        assert snapshot["enabled"] is False
        assert snapshot["requests"] is None
        assert all(value == 0 for value in snapshot["reader"].values())
    finally:
        session.close()


def test_diagnostics_observe_hits_evictions_and_reads_then_pause(monkeypatch):
    session = DashboardSession(_held_provider())
    session.cache_policy = replace(session.cache_policy, request_entries=1)
    try:
        session.set_diagnostics_enabled(True)
        key = (0, "analysis", "primary", True, "auto")
        session.request(key)
        assert session.pending.result(timeout=5) is not None
        session.pending = None
        session._load_request(key)
        session._load_request((3, *key[1:]))
        snapshot = session.diagnostics_snapshot()
        assert snapshot["requests"]["ram_hits"] == 1
        assert snapshot["requests"]["ram_misses"] == 2
        assert snapshot["requests"]["ram_evictions"] == 1
        assert snapshot["requests"]["completed_requests"] == 3
        assert snapshot["reader"]["index_builds"] == 1
        assert snapshot["reader"]["selected_dense_loads"] > 0
        assert snapshot["request_entries"] == snapshot["request_entry_limit"] == 1
        assert snapshot["request_bytes"] > 0
        assert snapshot["requests"]["last_queue_ms"] >= 0
        assert snapshot["requests"]["last_request_ms"] >= snapshot["requests"]["last_result_ms"]
        json.dumps(snapshot)

        session.set_diagnostics_enabled(False)

        def unexpected():
            raise AssertionError("Paused diagnostics read the timing clock")

        monkeypatch.setattr(module, "perf_counter", unexpected)
        session._load_request(key)
        paused = session.diagnostics_snapshot()
        assert paused["requests"] == snapshot["requests"]
        assert paused["reader"] == snapshot["reader"]
    finally:
        session.close()


def test_snapshot_never_waits_for_product_reader_lock(monkeypatch):
    session = DashboardSession(_held_provider())
    entered, release = Event(), Event()

    def hold_reader():
        with session._request_lock:
            entered.set()
            assert release.wait(timeout=5)

    def unexpected(*args, **kwargs):
        raise AssertionError("Diagnostic snapshot tried to load data")

    try:
        monkeypatch.setattr(session.service, "dashboard_result_rows", unexpected)
        with ThreadPoolExecutor(max_workers=2) as executor:
            blocked = executor.submit(hold_reader)
            assert entered.wait(timeout=3)
            try:
                result = executor.submit(session.diagnostics_snapshot).result(timeout=1)
                assert result["request_entries"] == 0
            finally:
                release.set()
            blocked.result(timeout=3)
    finally:
        session.close()


def test_disabling_diagnostics_stops_timing_an_inflight_request(monkeypatch):
    session = DashboardSession(_held_provider())
    entered, release = Event(), Event()
    load = session._cached_request

    def delayed(key):
        request = load(key)
        entered.set()
        assert release.wait(timeout=5)
        return request

    def unexpected():
        raise AssertionError("An in-flight request timed work after diagnostics was disabled")

    monkeypatch.setattr(session, "_cached_request", delayed)
    try:
        session.set_diagnostics_enabled(True)
        session.request((0, "analysis", "primary", True, "auto"))
        assert entered.wait(timeout=3)
        session.set_diagnostics_enabled(False)
        monkeypatch.setattr(module, "perf_counter", unexpected)
        release.set()
        assert session.pending.result(timeout=5) is not None
        assert session.diagnostics_snapshot()["requests"]["completed_requests"] == 0
    finally:
        release.set()
        session.close()
