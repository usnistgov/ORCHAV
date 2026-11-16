"""Prepared-cache lookup preserves selection without a second cold replay build."""

from copy import deepcopy

import numpy as np
import pytest

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from tests.visualizer.unit.test_sensing_dashboard_reuse import _held_provider
from tests.visualizer.unit.test_sensing_prepared_cache import _loaded_products
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from visualizer.src.sensing import prepared_results
from visualizer.src.sensing.dashboard_data import DashboardSession, load_dashboard_request


@pytest.mark.parametrize("reopen", [False, True])
def test_absent_angular_variant_never_loads_another_variants_cube(recording, monkeypatch, reopen):
    root, _payloads = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        # The recording has raw RD maps but only a primary angular cube.
        session._cached_request((0, "angular", "raw", True, "auto"))
    finally:
        session.close()
        provider.close()
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    reads = _loaded_products(provider, monkeypatch)
    try:
        if not reopen:
            session.service.dashboard_result_rows()
            monkeypatch.setattr(prepared_results, "load_prepared", lambda *_args: None)
        raw = session._cached_request((0, "angular", "raw", True, "auto"))
        assert raw.prepared_angular is None
        assert "angular" in raw.prepared_views
        assert (0, "range_doppler_angle_by_pair") not in reads
        primary = session._cached_request((0, "angular", "primary", True, "auto"))
        assert primary.prepared_angular is not None
        assert (0, "range_doppler_angle_by_pair") in reads
    finally:
        session.close()
        provider.close()


@pytest.fixture
def shifted_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = _held_provider()
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None, chunk_size=1)
    for step in sorted(source.frames):
        writer.append(source.frames[step])
    writer.finalize()
    return tmp_path


@pytest.mark.parametrize("pinned", [False, True])
def test_cold_exact_result_builds_one_replay(shifted_recording, monkeypatch, pinned):
    provider = Hdf5Provider(shifted_recording)
    session = DashboardSession(provider)
    calls = []
    original = session.service.resolve_display_sensing_frame

    def resolve(*args, **kwargs):
        calls.append(kwargs["product_names"])
        return original(*args, **kwargs)

    monkeypatch.setattr(session.service, "resolve_display_sensing_frame", resolve)
    session.set_diagnostics_enabled(True)
    try:
        request = load_dashboard_request(session.service, 0, "analysis", "primary", pinned, "auto")
        assert len(calls) == 1
        assert "range_doppler" in calls[0]
        assert request.display_frame["_sensing_display_provenance"]["storage_frame_id"] == 1
        assert request.replay_event.scene_binding.current_scene_frame_id == 0
        assert session.service.replay_telemetry_snapshot().replay_builds == 1
    finally:
        session.close()
        provider.close()


@pytest.mark.parametrize("pinned", [False, True])
def test_reopened_exact_result_uses_its_storage_chunk(shifted_recording, monkeypatch, pinned):
    provider = Hdf5Provider(shifted_recording)
    session = DashboardSession(provider)
    try:
        expected = load_dashboard_request(session.service, 0, "analysis", "primary", pinned, "auto")
    finally:
        session.close()
        provider.close()
    provider = Hdf5Provider(shifted_recording)
    session = DashboardSession(provider)
    reads = _loaded_products(provider, monkeypatch)
    try:
        actual = load_dashboard_request(session.service, 0, "analysis", "primary", pinned, "auto")
        assert reads == []
        assert actual.replay_event.to_manifest() == expected.replay_event.to_manifest()
        np.testing.assert_array_equal(
            actual.rd_result.power_linear, expected.rd_result.power_linear
        )
        assert actual.display_frame["_sensing_display_provenance"]["storage_frame_id"] == 1
    finally:
        session.close()
        provider.close()


def test_held_result_without_exact_cache_hint_uses_one_authoritative_read(
    shifted_recording, monkeypatch
):
    provider = Hdf5Provider(shifted_recording)
    session = DashboardSession(provider)
    calls = []
    original = session.service.resolve_display_sensing_frame

    def resolve(*args, **kwargs):
        calls.append(kwargs["product_names"])
        return original(*args, **kwargs)

    monkeypatch.setattr(session.service, "resolve_display_sensing_frame", resolve)
    try:
        request = load_dashboard_request(session.service, 2, "analysis", "primary", True, "auto")
        assert len(calls) == 1
        assert "range_doppler" in calls[0]
        assert request.state_step == 2
        assert request.display_step == 0
        assert request.display_from_cache
        assert request.replay_event.scene_binding.storage_frame_id == 1
        assert request.replay_event.scene_binding.current_scene_frame_id == 2
    finally:
        session.close()
        provider.close()


def test_cache_identity_mismatch_falls_back_to_selected_source(shifted_recording, monkeypatch):
    provider = Hdf5Provider(shifted_recording)
    session = DashboardSession(provider)
    try:
        expected = load_dashboard_request(session.service, 0, "analysis", "primary", False, "auto")
        original = prepared_results.load_prepared

        def mismatched(cache, kind, key):
            cached = original(cache, kind, key)
            if kind == "plots" and cached is not None:
                cached = deepcopy(cached)
                cached["identity"] = ("another result",)
                cached["rd"].power_linear.fill(123)
            return cached

        monkeypatch.setattr(prepared_results, "load_prepared", mismatched)
        actual = load_dashboard_request(session.service, 0, "analysis", "primary", False, "auto")
        np.testing.assert_array_equal(
            actual.rd_result.power_linear, expected.rd_result.power_linear
        )
        assert actual.replay_event.to_manifest() == expected.replay_event.to_manifest()
    finally:
        session.close()
        provider.close()
