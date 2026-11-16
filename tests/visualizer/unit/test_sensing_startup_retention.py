"""Upfront preparation hands its bounded RAM data to the actual dashboard."""

import json
import os
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from PySide6.QtWidgets import QVBoxLayout, QWidget

from shared.frames.contracts import FrameComponent
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_sensing_fusion_inspection import result as result
from tests.visualizer.unit.test_sensing_prepared_cache import _payload
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from tests.visualizer.unit.test_sensing_prepared_held_cache import (
    future_bound_recording as future_bound_recording,
)
from visualizer.src.sensing import startup_cache
from visualizer.src.sensing.cache_policy import SensingCachePolicy
from visualizer.src.sensing.dashboard_data import DashboardSession
from visualizer.src.sensing.prepared_results import _encode
from visualizer.src.sensing.recording_cache import RecordingCache
from visualizer.src.sensing.sensing_window import SensingWindow


def _forbid_source_products(provider, monkeypatch):
    original = provider.load_frame_projection

    def scene_pose_only(step, request):
        # Replay rebinding keeps the requested scene's own device pose/current
        # time; its canonical result arrays must remain the resident owner's.
        assert request.components <= {FrameComponent.DEVICES, FrameComponent.PROVENANCE}
        assert not request.sensing_products
        return original(step, request)

    monkeypatch.setattr(provider, "load_frame_projection", scene_pose_only)


@pytest.mark.parametrize("reopen", (False, True))
def test_figure_only_preparation_skips_scene_and_preserves_plot_handoff(
    recording, monkeypatch, reopen
):
    root, payloads = recording
    provider = Hdf5Provider(root)
    adopted = None

    def unexpected(*args, **kwargs):
        pytest.fail("Figure-only preparation started 3D sensing work")

    monkeypatch.setattr(startup_cache, "prepare_scene_recording", unexpected)
    try:
        if reopen:
            assert startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        assert startup_cache.prepare_startup_cache(
            provider, Event(), retain_session=True, include_scene=False
        ) is (not reopen)
        session = provider._prepared_sensing_dashboard
        assert session.prepared is None
        assert session.preparation_summary["requests_complete"]
        assert len(session.history.records) == 2
        cache = session.service.recording_cache
        marker = json.loads((cache._prepared_root / "complete.json").read_text())
        assert marker["scene_prepared"] is False
        assert not startup_cache._complete(cache._prepared_root, cache.signature)
        assert startup_cache._complete(cache._prepared_root, cache.signature, require_scene=False)
        scene_name = cache._path("scene", startup_cache.SCENE_CACHE_KEY).name
        assert scene_name not in marker["files"]
        adopted = DashboardSession(provider, previous=session, source_path=provider.source)
        request = adopted._load_request((1, "analysis", "primary", True, "auto"))
        np.testing.assert_array_equal(
            request.rd_result.power_linear, payloads[1]["range_doppler_power_by_pair"][0]
        )
    finally:
        if adopted is not None:
            adopted.close()
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


def test_full_prepare_extends_figure_only_bundle_without_rebuilding_its_products(
    recording, monkeypatch
):
    from visualizer.src.services.sensing_service import SensingService

    root, _ = recording
    provider = Hdf5Provider(root)
    try:
        startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        cache = RecordingCache.for_provider(provider)
        old_root = cache._prepared_root
        old_marker = (old_root / "complete.json").read_bytes()

        def unexpected(*args, **kwargs):
            pytest.fail("Extending a valid figure bundle rebuilt its saved plot products")

        monkeypatch.setattr(SensingService, "dashboard_history_products", unexpected)
        monkeypatch.setattr(SensingService, "_dashboard_products_from_validated", unexpected)
        assert startup_cache.prepare_startup_cache(provider, Event(), retain_session=True)
        session = provider._prepared_sensing_dashboard
        assert session.prepared.scene.matches_provider(provider)
        assert session.preparation_summary["requests_complete"]
        updated = session.service.recording_cache
        assert updated._prepared_root != old_root
        assert startup_cache._complete(updated._prepared_root, updated.signature)
        assert (old_root / "complete.json").read_bytes() == old_marker
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


def test_figure_only_reopen_reuses_full_bundle_without_loading_scene(recording, monkeypatch):
    root, _ = recording
    provider = Hdf5Provider(root)
    try:
        startup_cache.prepare_startup_cache(provider, Event())
        cache = RecordingCache.for_provider(provider)
        bundle = cache._prepared_root

        def unexpected(*args, **kwargs):
            pytest.fail("Figure-only reopening loaded 3D sensing")

        monkeypatch.setattr(startup_cache, "prepare_scene_recording", unexpected)
        assert not startup_cache.prepare_startup_cache(
            provider, Event(), retain_session=True, include_scene=False
        )
        session = provider._prepared_sensing_dashboard
        assert session.prepared is None
        assert session.service.recording_cache._prepared_root == bundle
        assert session.preparation_summary["requests_complete"]
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


@pytest.mark.parametrize("reopen", (False, True))
def test_prepared_dashboard_handoff_keeps_all_views_resident(recording, monkeypatch, reopen):
    root, _ = recording
    provider = Hdf5Provider(root)
    session = None
    try:
        if reopen:
            startup_cache.prepare_startup_cache(provider, Event())
            provider.close()
            provider = Hdf5Provider(root)
        startup_cache.prepare_startup_cache(provider, Event(), retain_session=True)
        prepared = provider._prepared_sensing_dashboard
        assert prepared.preparation_summary["requests_complete"]
        assert prepared.preparation_summary["retained_requests"] == 8
        assert prepared.history is not None
        history_plot_data = prepared.history.tracking_plot_data
        assert prepared.service.recording_cache._prepared_root is not None
        assert not prepared.service.recording_cache._building
        assert not hasattr(provider, "_sensing_cache_build_root")
        session = DashboardSession(provider, previous=prepared, source_path=provider.source)
        del provider._prepared_sensing_dashboard
        assert session._request_cache is prepared._request_cache
        assert session.prepared is prepared.prepared
        assert session.saved_tracking is prepared.saved_tracking
        assert session.history.tracking_plot_data is history_plot_data

        def unexpected(*args, **kwargs):
            pytest.fail("An already prepared dashboard reread source products")

        _forbid_source_products(provider, monkeypatch)
        monkeypatch.setattr(session.service, "dashboard_load_row", unexpected)
        session.set_diagnostics_enabled(True)
        for _loop in range(2):
            for step in (0, 1):
                for variant in ("primary", "raw"):
                    for view in (
                        "analysis",
                        "diagnostics",
                        "processing-clutter",
                        "angular",
                        "tracking",
                    ):
                        request = session._load_request((step, view, variant, True, "auto"))
                        assert request is not None
                        assert request.state_step == step
                        assert request.variant == variant
                    for view, products in (
                        ("live", ("range_doppler",)),
                        ("live", ("range_profile", "range_doppler")),
                        ("live", ("range_doppler_angle",)),
                        ("tracking", ("range_doppler", "range_doppler_angle")),
                        ("tracking", ()),
                    ):
                        request = session._load_request(
                            (step, view, variant, True, "auto", products)
                        )
                        assert request is not None
                        assert request.variant == variant
                        assert request.state_step == step
        diagnostics = session.diagnostics_snapshot()
        assert diagnostics["requests"]["ram_misses"] == 0
        assert diagnostics["requests"]["ram_evictions"] == 0
        assert diagnostics["reader"]["selected_dense_loads"] == 0
    finally:
        if session is not None:
            session.close()
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


def test_prepared_data_survives_pin_hide_return_and_dashboard_reopen(recording, monkeypatch, qtbot):
    root, _ = recording
    provider = Hdf5Provider(root)
    startup_cache.prepare_startup_cache(provider, Event(), retain_session=True)
    session = DashboardSession(provider, previous=provider._prepared_sensing_dashboard)
    del provider._prepared_sensing_dashboard
    window, host = SensingWindow(), QWidget()
    qtbot.addWidget(window)
    qtbot.addWidget(host)
    host.setLayout(QVBoxLayout())
    host.show()
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")

    def unexpected(*args, **kwargs):
        pytest.fail("Presentation change reread prepared products")

    _forbid_source_products(provider, monkeypatch)
    monkeypatch.setattr(session.service, "dashboard_load_row", unexpected)
    session.set_diagnostics_enabled(True)
    try:
        original = session._load_request((0, "analysis", "primary", True, "auto"))
        window.update_from_request(original)
        for plot in ("rd_plot", "rp_plot"):
            wrapper = window.acquire_live_figure(plot, host)
            host.layout().addWidget(wrapper)
            wrapper.show()
        window.hide()
        window.refresh_live_figure_visibility()
        assert window.request_view_mode() == "live"
        suffix = window.live_figure_request_suffix()
        for variant in ("primary", "raw", "primary"):
            request = session._load_request((0, "live", variant, True, "auto", *suffix))
            window.update_from_request(request)
            assert request.variant == variant
            expected = original.rd_result.power_linear * (2 if variant == "raw" else 1)
            np.testing.assert_array_equal(request.rd_result.power_linear, expected)
        for plot in tuple(window._live_figures):
            window.release_live_figure(plot)
        window.show()
        request = session._load_request((0, window.request_view_mode(), "primary", True, "auto"))
        window.update_from_request(request)
        assert request.rd_result is original.rd_result
        window.close()
        session.close()
        session = DashboardSession(provider, previous=session)
        assert (
            session._load_request((0, "analysis", "primary", True, "auto")).rd_result
            is original.rd_result
        )
        assert session.diagnostics_snapshot()["reader"]["selected_dense_loads"] == 0
    finally:
        for plot in tuple(window._live_figures):
            window.release_live_figure(plot)
        session.close()
        provider.close()


def test_preparation_reports_partial_residency_at_its_budget(recording, monkeypatch):
    root, _ = recording
    policy = replace(SensingCachePolicy.for_preparation(), request_bytes=128)
    provider = Hdf5Provider(root)
    progress = []
    try:
        startup_cache.prepare_startup_cache(
            provider,
            Event(),
            lambda *args: progress.append(args),
            retain_session=True,
            cache_policy=policy,
        )
        session = provider._prepared_sensing_dashboard
        summary = session.preparation_summary
        assert not summary["requests_complete"]
        assert summary["retained_requests"] < summary["requested_requests"]
        assert summary["request_bytes"] <= summary["request_byte_limit"] == 128
        assert progress[-1][0] == "Sensing playback partly resident"
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


@pytest.mark.parametrize("entry_limit", (1, 3))
def test_published_warming_keeps_first_playback_owners_without_eviction(
    recording, monkeypatch, entry_limit
):
    root, _ = recording
    provider = Hdf5Provider(root)
    startup_cache.prepare_startup_cache(provider, Event())
    cache = RecordingCache.for_provider(provider)
    bundle = cache._prepared_root
    original_files = {path.name: path.read_bytes() for path in bundle.iterdir()}
    provider.close()
    provider = Hdf5Provider(root)
    reference = DashboardSession(provider)
    expected = {}
    views = ("analysis", "diagnostics", "processing-clutter", "angular", "tracking")

    def numeric_models(request):
        arrays = {}
        metadata = _encode(
            (
                request.rp_result,
                request.rd_result,
                request.detection_stage_bundle,
                request.processing_inspection,
                request.prepared_angular,
            ),
            arrays,
            {},
        )
        return metadata, arrays

    try:
        for step in (0, 1):
            for variant in ("primary", "raw"):
                for view in views:
                    request = reference._cached_request((step, view, variant, True, "auto"))
                expected[(step, variant)] = numeric_models(request)
    finally:
        reference.close()

    calls = []
    original = DashboardSession._cached_request

    def observe(session, key, *, prefetch=False):
        calls.append((key, prefetch))
        return original(session, key, prefetch=prefetch)

    monkeypatch.setattr(DashboardSession, "_cached_request", observe)
    policy = replace(SensingCachePolicy.for_preparation(), request_entries=entry_limit)
    session = None
    try:
        assert not startup_cache.prepare_startup_cache(
            provider, Event(), retain_session=True, cache_policy=policy
        )
        prepared = provider._prepared_sensing_dashboard
        owners = [(0, "primary"), (0, "raw"), (1, "primary")][:entry_limit]
        assert [(key[0], key[2]) for key in prepared._request_cache] == owners
        assert calls == [
            ((step, view, variant, True, "auto"), True)
            for step, variant in owners
            for view in views
        ]
        assert prepared.preparation_summary["retained_requests"] == 2 * entry_limit
        assert prepared.preparation_summary["requested_requests"] == 8
        assert not prepared.preparation_summary["requests_complete"]
        assert prepared._cancelled.is_set()
        session = DashboardSession(provider, previous=prepared)
        del provider._prepared_sensing_dashboard

        def unexpected(*args, **kwargs):
            pytest.fail("First playback reread an already retained model")

        monkeypatch.setattr(provider, "load_frame_projection", unexpected)
        monkeypatch.setattr(session.service, "dashboard_load_row", unexpected)
        for step, variant in owners:
            for view in views:
                request = session._cached_request((step, view, variant, True, "auto"))
                assert request.state_step == step
                assert request.replay_event.scene_binding.pinned
                metadata, arrays = numeric_models(request)
                expected_metadata, expected_arrays = expected[(step, variant)]
                assert metadata == expected_metadata
                assert arrays.keys() == expected_arrays.keys()
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], expected_arrays[key])
        assert {path.name: path.read_bytes() for path in bundle.iterdir()} == original_files
    finally:
        if session is not None:
            session.close()
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


def test_published_warming_stops_when_first_map_exceeds_byte_budget(recording, monkeypatch):
    root, _ = recording
    provider = Hdf5Provider(root)
    startup_cache.prepare_startup_cache(provider, Event())
    policy = replace(SensingCachePolicy.for_preparation(), request_bytes=128)
    calls = []
    original = DashboardSession._cached_request

    def observe(session, key, *, prefetch=False):
        calls.append(key)
        return original(session, key, prefetch=prefetch)

    monkeypatch.setattr(DashboardSession, "_cached_request", observe)
    try:
        assert not startup_cache.prepare_startup_cache(
            provider, Event(), retain_session=True, cache_policy=policy
        )
        assert calls == [(0, "analysis", "primary", True, "auto")]
        prepared = provider._prepared_sensing_dashboard
        assert not prepared._request_cache
        assert prepared.preparation_summary["retained_requests"] == 0
        assert prepared.preparation_summary["requested_requests"] == 8
        assert prepared.preparation_summary["request_bytes"] == 0
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


def test_published_warming_cancels_between_views_without_publishing_session(recording, monkeypatch):
    root, _ = recording
    provider = Hdf5Provider(root)
    startup_cache.prepare_startup_cache(provider, Event())
    cache = RecordingCache.for_provider(provider)
    marker = (cache._prepared_root / "complete.json").read_bytes()
    cancelled, sessions = Event(), []
    original = DashboardSession._cached_request

    def cancel_after_first_view(session, key, *, prefetch=False):
        sessions.append(session)
        request = original(session, key, prefetch=prefetch)
        cancelled.set()
        return request

    monkeypatch.setattr(DashboardSession, "_cached_request", cancel_after_first_view)
    try:
        with pytest.raises(CancelledError):
            startup_cache.prepare_startup_cache(provider, cancelled, retain_session=True)
        assert len(sessions) == 1
        assert sessions[0]._cancelled.is_set()
        assert not hasattr(provider, "_prepared_sensing_dashboard")
        assert (cache._prepared_root / "complete.json").read_bytes() == marker
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


def test_published_warming_accepts_a_held_result_owned_by_an_earlier_scene(recording):
    root, _ = recording
    provider = Hdf5Provider(root)
    startup_cache.prepare_startup_cache(provider, Event())
    session = DashboardSession(provider)
    scenes = {step: (None, {"primary"}) for step in (0, 1, 2)}
    progress = []
    try:
        startup_cache._warm_published_requests(
            session, scenes, Event(), lambda *args: progress.append(args)
        )
        assert len(session._request_cache) == 2
        assert [row[1] for row in progress] == [0, 1, 2]
        held = session._cached_request((2, "analysis", "primary", True, "auto"))
        assert held.state_step == 2
        assert held.replay_event.scene_binding.bound_scene_frame_id == 1
        np.testing.assert_array_equal(
            held.rd_result.power_linear, _payload(1)["range_doppler_power_by_pair"][0]
        )
    finally:
        session.close()
        provider.close()


def test_published_warming_rejects_a_changed_source_before_handoff(recording):
    root, _ = recording
    provider = Hdf5Provider(root)
    startup_cache.prepare_startup_cache(provider, Event())
    cache = RecordingCache.for_provider(provider)
    marker = (cache._prepared_root / "complete.json").read_bytes()
    manifest = cache.paths[0]

    def change_source(stage, *_):
        if stage == "Preparing saved tracking":
            stat = manifest.stat()
            os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    try:
        with pytest.raises(RuntimeError, match="Recording changed while preparing resident"):
            startup_cache.prepare_startup_cache(
                provider, Event(), change_source, retain_session=True
            )
        assert not hasattr(provider, "_prepared_sensing_dashboard")
        assert (cache._prepared_root / "complete.json").read_bytes() == marker
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


@pytest.mark.parametrize("reopen", (False, True))
def test_preparation_reports_partial_when_maps_fit_but_detector_does_not(
    tmp_path, monkeypatch, reopen
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    payload = _payload(0)
    payload.pop("raw_rd")
    payload["products"].pop("raw_rd")
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    writer.append(replace(build_standard_mpc_frame(frame_idx=0), sensing=payload, timestamp_s=0.0))
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    if reopen:
        startup_cache.prepare_startup_cache(provider, Event())
        provider.close()
        provider = Hdf5Provider(tmp_path)
    probe = DashboardSession(provider)
    try:
        probe._cached_request((0, "analysis", "primary", False, "auto"))
        maps_bytes = probe._request_cache_bytes
        probe._cached_request((0, "diagnostics", "primary", False, "auto"))
        detector_bytes = probe._request_cache_bytes
        assert detector_bytes > maps_bytes
    finally:
        probe.close()
    policy = replace(
        SensingCachePolicy.for_preparation(), request_bytes=(maps_bytes + detector_bytes) // 2
    )
    progress = []
    try:
        startup_cache.prepare_startup_cache(
            provider,
            Event(),
            lambda *args: progress.append(args),
            retain_session=True,
            cache_policy=policy,
        )
        session = provider._prepared_sensing_dashboard
        owner = next(
            r for k, (r, _) in session._request_cache.items() if k[:3] == (0, "analysis", "primary")
        )
        assert "diagnostics" not in owner.prepared_views
        assert session.preparation_summary["retained_requests"] == 1
        assert session.preparation_summary["requested_requests"] == 2
        assert not session.preparation_summary["requests_complete"]
        assert progress[-1][0] == "Sensing playback partly resident"
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


def test_half_sample_owners_are_counted_and_reused_without_eager_rebinding(
    future_bound_recording, monkeypatch
):
    root, payloads = future_bound_recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    adopted = None
    try:
        for step in (0, 1):
            for variant in ("primary", "raw"):
                for view in (
                    "analysis",
                    "diagnostics",
                    "processing-clutter",
                    "angular",
                    "tracking",
                ):
                    session._cached_request((step, view, variant, False, "auto"))
        _rows, _contexts, scenes = startup_cache._selection_plan(session)
        before = tuple(session._request_cache)
        monkeypatch.setattr(
            session,
            "_cached_request",
            lambda *_args, **_kwargs: pytest.fail("handoff eagerly rebound a result"),
        )
        startup_cache._retain_playback_session(
            provider, session, None, scenes, Event(), lambda *_: None
        )
        assert tuple(session._request_cache) == before
        assert session.preparation_summary["retained_requests"] == 8
        assert session.preparation_summary["requested_requests"] == 8
        assert session.preparation_summary["requests_complete"]
        assert "Canonical" in session.preparation_summary["residency_scope"]
        adopted = DashboardSession(provider, previous=session)
        _forbid_source_products(provider, monkeypatch)
        for variant in ("primary", "raw"):
            held = adopted._cached_request((1, "analysis", variant, True, "auto"))
            assert held.display_step == 0 and held.state_step == 1
            assert held.replay_event.scene_binding.storage_frame_id == 0
            assert held.replay_event.scene_binding.pinned
            expected = payloads[0]["raw_rd" if variant == "raw" else "range_doppler_power_by_pair"][
                0
            ]
            np.testing.assert_array_equal(held.rd_result.power_linear, expected)
            assert adopted._cached_request((0, "analysis", variant, True, "auto")) is None
    finally:
        if adopted is not None:
            adopted.close()
        startup_cache.discard_prepared_dashboard(provider)
        session.close()
        provider.close()


def test_yaml_dashboard_adopts_verified_adjacent_saved_tracking(recording, result, monkeypatch):
    from shared.sensing.fusion.experiment import fingerprints

    root, _ = recording
    native = root / "native"
    native.mkdir()
    (root / "frames").rename(native / "frames")
    scenario = native / "scenario.yaml"
    scenario.write_text("{}")
    tracking = root / "tracking"
    tracking.mkdir()
    data = deepcopy(dict(result))
    data["sources"]["a"].update(source="../native", hashes=fingerprints(native / "frames"))
    (tracking / "fusion_results.json").write_text(json.dumps(data))
    provider = Hdf5Provider(native)
    session = None
    try:
        startup_cache.prepare_startup_cache(provider, Event(), retain_session=True)
        prepared = provider._prepared_sensing_dashboard
        saved = prepared.prepared.tracking
        assert saved is not None and prepared.prepared.tracking_error is None
        plot_data = saved.history.tracking_plot_data
        session = DashboardSession(provider, previous=prepared, source_path=scenario)
        del provider._prepared_sensing_dashboard
        assert session.saved_tracking is prepared.saved_tracking

        def unexpected(*args, **kwargs):
            pytest.fail("Verified adjacent tracking was loaded or hashed again")

        monkeypatch.setattr(session.saved_tracking, "load", unexpected)
        monkeypatch.setattr("shared.sensing.fusion.experiment.fingerprints", unexpected)
        for step in (0, 1, 0):
            request = session._load_request((step, "tracking", "primary", True, "auto"))
            assert request.saved_tracking is saved
            assert request.tracking_history.tracking_plot_data is plot_data
    finally:
        if session is not None:
            session.close()
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


@pytest.mark.parametrize("failure", (CancelledError, OSError))
def test_failed_build_removes_only_its_unpublished_stage(recording, monkeypatch, failure):
    root, _ = recording
    provider = Hdf5Provider(root)
    cache = RecordingCache.for_provider(provider)
    recordings = cache.root / "recordings"
    earlier = recordings / "bundle_earlier"
    earlier.mkdir(parents=True)
    (earlier / "keep").write_text("published elsewhere")

    def fail(*args, **kwargs):
        (provider._sensing_cache_build_root / "partial.npz").write_bytes(b"partial")
        raise failure()

    monkeypatch.setattr(startup_cache, "prepare_scene_recording", fail)
    try:
        with pytest.raises(failure):
            startup_cache.prepare_startup_cache(provider, Event(), retain_session=True)
        assert sorted(p.name for p in recordings.glob("bundle_*")) == ["bundle_earlier"]
        assert not hasattr(provider, "_prepared_sensing_dashboard")
        assert not hasattr(provider, "_sensing_cache_build_root")
        assert (earlier / "keep").read_text() == "published elsewhere"
    finally:
        provider.close()


def test_failed_numeric_cache_promotion_cannot_publish_complete_bundle(recording, monkeypatch):
    root, payloads = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        startup_cache.prepare_scene_recording(provider, Event())
        for variant in ("primary", "raw"):
            startup_cache.load_recording_history(
                session.service, startup_cache.result_context(payloads[0]), variant, Event()
            )
            for step in (0, 1):
                for view in ("analysis", "diagnostics", "processing-clutter", "angular"):
                    assert session._load_request((step, view, variant, False, "auto")) is not None
        ordinary = RecordingCache.for_provider(provider)
        original_files = {
            path: path.read_bytes()
            for pattern in ("*.json.gz", "*.npz")
            for path in ordinary.root.glob(pattern)
        }
        assert any(path.suffix == ".npz" for path in original_files)
    finally:
        session.close()
        provider.close()

    original_save = RecordingCache.save_arrays
    failed_promotions = []

    def refuse_staging_arrays(cache, kind, metadata, arrays, key=()):
        if cache._building:
            failed_promotions.append((kind, key))
            return False
        return original_save(cache, kind, metadata, arrays, key)

    monkeypatch.setattr(RecordingCache, "save_arrays", refuse_staging_arrays)
    provider = Hdf5Provider(root)
    try:
        with pytest.raises(OSError, match="Could not save the complete run summary"):
            startup_cache.prepare_startup_cache(provider, Event(), retain_session=True)
        assert failed_promotions
        assert not list((ordinary.root / "recordings").glob("*.ready"))
        assert not list((ordinary.root / "recordings").glob("bundle_*"))
        assert not hasattr(provider, "_prepared_sensing_dashboard")
        assert not hasattr(provider, "_sensing_cache_build_root")
        assert {
            path: path.read_bytes()
            for pattern in ("*.json.gz", "*.npz")
            for path in ordinary.root.glob(pattern)
        } == original_files
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()


@pytest.mark.parametrize("fail_repair", [False, True])
def test_legacy_partial_bundle_repairs_only_its_missing_current_products(
    recording, monkeypatch, fail_repair
):
    from visualizer.src.sensing.prepared_results import (
        PREPARATION_VERSION,
        _decode,
        load_prepared,
        save_prepared,
    )

    root, payloads = recording
    provider = Hdf5Provider(root)
    try:
        startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        cache = RecordingCache.for_provider(provider)
        old_root = cache._prepared_root
        pointer = cache.root / "recordings" / f"{cache.signature[:20]}.ready"
        scoped = cache.for_frame(0)
        value = load_prepared(scoped, "plots", (0, "primary"))
        missing = value["rp"].metadata["data_key"]
        partial = {
            **value,
            "rp": None,
            "arrays": {k: v for k, v in value["arrays"].items() if k != missing},
        }
        # Reproduce a stamp-valid legacy full bundle whose Angular-first owner
        # was never persisted after Current added its authored range profile.
        scoped.root = old_root
        scoped._building = True
        assert save_prepared(scoped, "plots", (0, "primary"), partial)
        path = scoped._array_path("plots", (PREPARATION_VERSION, 0, "primary"))
        marker = json.loads((old_root / "complete.json").read_text())
        marker.pop("plot_coverage")
        marker["artifact_stamps"][path.name] = startup_cache._artifact_stamp(path)
        (old_root / "complete.json").write_text(json.dumps(marker))
        assert not startup_cache._complete(old_root, cache.signature, require_scene=False)
        assert startup_cache._reusable_model_manifest(old_root, cache.signature) is not None
        old_files = {p.name: p.read_bytes() for p in old_root.iterdir()}
        old_pointer = pointer.read_bytes()
        calls, dense = [], []
        original_request = DashboardSession._cached_request
        original_projection = provider.load_frame_projection
        original_save = RecordingCache.save_arrays

        def request(session, key, **kwargs):
            calls.append(key)
            assert key[0] == 0 and key[2] == "primary", "A complete model was rebuilt"
            return original_request(session, key, **kwargs)

        def projection(step, request):
            result = original_projection(step, request)
            sensing = result.frame.sensing or {}
            for name in sensing.get("products", {}):
                if isinstance(sensing.get(name), np.ndarray):
                    dense.append((step, name))
            return result

        def save(cache, kind, metadata, arrays, key=()):
            if (
                fail_repair
                and cache._building
                and kind == "plots"
                and key == (PREPARATION_VERSION, 0, "primary")
            ):
                decoded = _decode(metadata["payload"], arrays, {})
                if decoded["rp"] is not None:
                    return False
            return original_save(cache, kind, metadata, arrays, key)

        monkeypatch.setattr(DashboardSession, "_cached_request", request)
        monkeypatch.setattr(provider, "load_frame_projection", projection)
        monkeypatch.setattr(RecordingCache, "save_arrays", save)
        if fail_repair:
            with pytest.raises(OSError, match="save"):
                startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
            assert pointer.read_bytes() == old_pointer
            assert sorted(p.name for p in old_root.parent.glob("bundle_*")) == [old_root.name]
        else:
            assert startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
            updated = RecordingCache.for_provider(provider)
            assert updated._prepared_root != old_root
            assert startup_cache._complete(
                updated._prepared_root, cache.signature, require_scene=False
            )
            complete = load_prepared(updated.for_frame(0), "plots", (0, "primary"))
            assert complete["rp"] is not None
            np.testing.assert_array_equal(complete["rp"].power_linear, payloads[0][missing][0])
        assert calls
        assert dense == [(0, missing)]
        assert {p.name: p.read_bytes() for p in old_root.iterdir()} == old_files
        assert not hasattr(provider, "_sensing_cache_build_root")
    finally:
        provider.close()


def test_indexed_plot_coverage_rejects_wrong_occurrence_keys_variants_and_supplied_ids(recording):
    from visualizer.src.sensing.prepared_results import load_prepared

    root, _payloads = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        startup_cache.prepare_startup_cache(provider, Event(), include_scene=False)
        sensing = dict(session.service.dashboard_result_rows())[0]["sensing"]
        value = load_prepared(session.service.recording_cache.for_frame(0), "plots", (0, "primary"))
        assert "product_id" not in sensing["products"][value["rp"].metadata["data_key"]]
        assert startup_cache._indexed_plot_complete(value, sensing, "primary")
        assert not startup_cache._indexed_plot_complete(
            {**value, "identity": ("wrong",)}, sensing, "primary"
        )
        for field, replacement in (("data_key", "wrong_key"), ("variant", "raw")):
            model = replace(value["rp"], metadata={**value["rp"].metadata, field: replacement})
            assert not startup_cache._indexed_plot_complete(
                {**value, "rp": model}, sensing, "primary"
            )
        supplied = deepcopy(sensing)
        key = value["rp"].metadata["data_key"]
        supplied["products"][key]["product_id"] = "wrong-id"
        assert not startup_cache._indexed_plot_complete(value, supplied, "primary")
        supplied["products"][key]["product_id"] = value["rp"].metadata["product_id"]
        assert startup_cache._indexed_plot_complete(value, supplied, "primary")
        arrays = {**value["arrays"], key: np.zeros((123,))}
        assert not startup_cache._indexed_plot_complete(
            {**value, "arrays": arrays}, sensing, "primary"
        )
    finally:
        session.close()
        provider.close()


@pytest.mark.parametrize("kind", ("scene", "index", "overview"))
def test_invalid_metadata_cache_with_failed_repair_cannot_publish_bundle(
    recording, monkeypatch, kind
):
    from visualizer.src.sensing.scene_recording import SCENE_CACHE_KEY

    root, payloads = recording
    provider = Hdf5Provider(root)
    ordinary = RecordingCache.for_provider(provider)
    if kind == "scene":
        key = SCENE_CACHE_KEY
        malformed = {"records": [{}], "analytical_sources": []}
    elif kind == "index":
        key = ("input-preflight-v3-rx-array",)
        malformed = [["incomplete index row"]]
    else:
        key = startup_cache.recording_history_cache_key(
            startup_cache.result_context(payloads[0]), "primary", False
        )
        malformed = {"records": [{}]}
    assert ordinary.save(kind, malformed, key)
    source = ordinary._path(kind, key)
    original_bytes = source.read_bytes()
    original_save = RecordingCache.save
    failed_repairs = []

    def refuse_repaired_metadata(cache, requested_kind, payload, key=()):
        if cache._building and requested_kind == kind and payload != malformed:
            failed_repairs.append(payload)
            return False
        return original_save(cache, requested_kind, payload, key)

    monkeypatch.setattr(RecordingCache, "save", refuse_repaired_metadata)
    try:
        with pytest.raises(OSError):
            startup_cache.prepare_startup_cache(provider, Event(), retain_session=True)
        assert len(failed_repairs) == 1
        repaired_rows = failed_repairs[0] if kind == "index" else failed_repairs[0]["records"]
        assert len(repaired_rows) == 2
        assert source.read_bytes() == original_bytes
        assert not list((ordinary.root / "recordings").glob("*.ready"))
        assert not list((ordinary.root / "recordings").glob("bundle_*"))
        assert not hasattr(provider, "_prepared_sensing_dashboard")
        assert not hasattr(provider, "_sensing_cache_build_root")
    finally:
        startup_cache.discard_prepared_dashboard(provider)
        provider.close()
