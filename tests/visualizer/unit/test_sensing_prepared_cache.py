"""Packed recordings reuse prepared figures across views and application sessions."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from shared.sensing.products import add_product_spec, materialize_product_specs
from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_sensing_panel_fallback import _typed_dashboard_payload
from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products
from visualizer.src.sensing.dashboard_data import (
    DashboardSession,
    load_recording_history,
    result_context,
)
from visualizer.src.sensing.prepared_angle import prepare_angular
from visualizer.src.sensing.processing_inspection import inspect_processing
from visualizer.src.sensing.stage_inspector import (
    DETECTION_STAGE_KINDS,
    CellSelection,
    build_detection_stage_bundle,
)
from visualizer.src.services.sensing_service import SensingService

VIEWS = ("analysis", "diagnostics", "processing-clutter", "angular")


def _payload(step=0):
    """Add real typed stages and saved before/after maps to the dashboard fixture."""
    sensing = _typed_dashboard_payload(scene_frame_id=step, sample_time_s=float(step))
    sensing["range_doppler_power_by_pair"] = np.arange(1, 33, dtype=np.float32).reshape(1, 4, 8)
    sensing["range_doppler_angle_by_pair"] = np.arange(96, dtype=np.float32).reshape(1, 1, 3, 4, 8)
    rd = sensing["products"]["range_doppler_power_by_pair"]
    rd["processing"] = {"slow_time_mean_removal_applied": True}
    stage_source = _typed_sensing()
    source_rd_id = stage_source["products"]["range_doppler_by_pair"]["product_id"]
    for key, descriptor in stage_source["products"].items():
        if descriptor["kind"] not in DETECTION_STAGE_KINDS:
            continue
        sensing[key] = np.resize(stage_source[key], (1, 4, 8))
        copied = deepcopy(descriptor)
        copied["axes"] = deepcopy(rd["axes"])
        copied["shape"] = [1, 4, 8]
        copied["lineage"]["source_product_ids"] = [
            rd["product_id"] if value == source_rd_id else value
            for value in copied["lineage"]["source_product_ids"]
        ]
        sensing["products"][key] = copied
    sensing["detections"] = [
        dict(cluster_id=0, doppler_bin=1, range_bin=2, range_m=1.0, doppler_hz=-4.0)
    ]
    _rematerialize_products(sensing)
    rd = sensing["products"]["range_doppler_power_by_pair"]
    sensing["raw_rd"] = 2 * sensing["range_doppler_power_by_pair"]
    add_product_spec(
        sensing,
        data_key="raw_rd",
        kind="range_doppler",
        variant="raw",
        axes=rd["axes"],
        representation="power_linear",
        unit="1",
        combine=rd["combine"],
        processing={"slow_time_mean_removal_applied": False},
    )
    materialize_product_specs(sensing)
    return sensing


@pytest.fixture
def recording(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    payloads = [_payload(step) for step in range(2)]
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for step, payload in enumerate(payloads):
        frame = replace(
            build_standard_mpc_frame(frame_idx=step), sensing=payload, timestamp_s=float(step)
        )
        writer.append(frame)
    writer.append(replace(build_standard_mpc_frame(frame_idx=2), timestamp_s=2.0))
    writer.finalize()
    return tmp_path, payloads


def _get(session, view, step=0, variant="primary", pin=True):
    return session._load_request((step, view, variant, pin, "auto"))


def _loaded_products(provider, monkeypatch):
    loaded = []
    original = provider.load_frame_projection

    def load(step, request):
        projection = original(step, request)
        sensing = projection.frame.sensing or {}
        loaded.extend((step, key) for key in (sensing.get("products") or {}) if key in sensing)
        return projection

    monkeypatch.setattr(provider, "load_frame_projection", load)
    return loaded


def test_simultaneous_readers_share_one_result_index(recording, monkeypatch):
    root, _ = recording
    provider = Hdf5Provider(root)
    proxy = SimpleNamespace(frame_source=SimpleNamespace(provider=provider))
    first, second = SensingService(proxy), SensingService(proxy)
    entered, release, waiting = Event(), Event(), Event()
    builds = []
    original = first._iter_sensing_metadata_candidates

    def scan(*args):
        builds.append(1)
        entered.set()
        assert release.wait(10)
        yield from original(*args)

    def progress(stage, *_):
        if stage == "Waiting for result index":
            waiting.set()

    monkeypatch.setattr(first, "_iter_sensing_metadata_candidates", scan)
    monkeypatch.setattr(second, "report_read_progress", progress)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            owner = executor.submit(first.dashboard_result_rows)
            assert entered.wait(10)
            follower = executor.submit(second.dashboard_result_rows)
            try:
                assert waiting.wait(10)
            finally:
                release.set()
            rows1, rows2 = owner.result(timeout=20), follower.result(timeout=20)
        assert len(builds) == 1
        assert len(rows1) == len(rows2) == 2
        rows1[0][1]["sensing"]["test_only"] = True
        assert "test_only" not in rows2[0][1]["sensing"]
    finally:
        release.set()
        provider.close()


def test_estimate_history_has_independent_coverage_and_reuses_its_own_cache(recording, monkeypatch):
    root, payloads = recording
    provider = Hdf5Provider(root)
    proxy = SimpleNamespace(frame_source=SimpleNamespace(provider=provider))
    context = result_context(payloads[0])
    try:
        full = load_recording_history(SensingService(proxy), context, "primary", Event())
        service = SensingService(proxy)
        reads = []
        original = service.dashboard_load_row

        def metadata_only(storage, products):
            reads.append((storage, products))
            assert products == ("config_resolved",)
            return original(storage, products)

        monkeypatch.setattr(service, "dashboard_load_row", metadata_only)
        estimates = load_recording_history(
            service, context, "primary", Event(), include_signal_products=False
        )
        assert len(reads) == 2

        def unexpected_read(*_args):
            pytest.fail("the independent estimates cache should satisfy its own repeated demand")

        monkeypatch.setattr(service, "dashboard_load_row", unexpected_read)
        cached_estimates = load_recording_history(
            service, context, "primary", Event(), include_signal_products=False
        )
        assert len(full.records) == len(estimates.records) == len(cached_estimates.records) == 2
        assert [r.detections for r in full.records] == [r.detections for r in estimates.records]
        assert [r.detections for r in estimates.records] == [
            r.detections for r in cached_estimates.records
        ]
    finally:
        provider.close()


def test_history_json_admission_preserves_cold_warm_coverage(recording, monkeypatch):
    from shared.cache_sizing import estimate_retained_bytes
    from visualizer.src.sensing import dashboard_data
    from visualizer.src.sensing.cache_policy import SensingCachePolicy

    root, payloads = recording
    provider = Hdf5Provider(root)
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    context = result_context(payloads[0])
    original_event = dashboard_data.evaluation_event
    original_admission = dashboard_data._persisted_history_record
    source_sizes, durable_sizes = [], []

    def aliased_evidence(*args):
        event = original_event(*args)
        # Source evidence can share nested sample provenance. JSON preserves
        # its values, but independently reconstructs both retained references.
        provenance = [{"sample": i, "time_s": i / 4000, "source": "saved"} for i in range(64)]
        event["input_evidence"] = event["reference_evidence"] = provenance
        return event

    def observe_admission(record):
        result = original_admission(record)
        source_sizes.append(estimate_retained_bytes(record))
        durable_sizes.append(estimate_retained_bytes(result))
        return result

    monkeypatch.setattr(dashboard_data, "evaluation_event", aliased_evidence)
    monkeypatch.setattr(dashboard_data, "_persisted_history_record", observe_admission)
    try:
        load_recording_history(service, context, "primary", Event(), False)
        assert sum(durable_sizes) > sum(source_sizes)
        policy = replace(SensingCachePolicy(), history_bytes=sum(source_sizes))
        cold = load_recording_history(
            service, context, "primary", Event(), False, cache_policy=policy
        )
        assert 0 < len(cold.records) < 2
        assert cold.truncated
        assert sum(estimate_retained_bytes(r) for r in cold.records) <= policy.history_bytes
        # A newly admitted artifact must remain usable on every subsequent read.
        monkeypatch.setattr(
            service, "dashboard_load_row", lambda *_: pytest.fail("cache reread source rows")
        )
        for _ in range(2):
            warm = load_recording_history(
                service, context, "primary", Event(), False, cache_policy=policy
            )
            assert warm.records == cold.records
            assert warm.evaluation_summary == cold.evaluation_summary
            assert warm.available_time_span == cold.available_time_span
            assert warm.available_results == cold.available_results == 2
            assert warm.truncated == cold.truncated
            assert sum(estimate_retained_bytes(r) for r in warm.records) <= policy.history_bytes
    finally:
        provider.close()


def test_startup_cache_is_complete_and_reused(recording, monkeypatch):
    from visualizer.src.sensing.prepared_results import load_prepared
    from visualizer.src.sensing.recording_cache import RecordingCache
    from visualizer.src.sensing.startup_cache import prepare_startup_cache

    root, _ = recording
    provider = Hdf5Provider(root)
    try:
        assert prepare_startup_cache(provider, Event()) is True
        cache = RecordingCache.for_provider(provider)
        assert cache is not None and cache._prepared_root is not None
        assert load_prepared(cache.for_frame(0), "plots", (0, "primary")) is not None
        monkeypatch.setattr(
            provider,
            "load_frame_projection",
            lambda *_: pytest.fail("complete startup cache should not be rebuilt"),
        )
        assert prepare_startup_cache(provider, Event()) is False
    finally:
        provider.close()


def test_startup_promotes_existing_work_into_an_independent_bundle(recording, monkeypatch):
    from shared.frames import FrameComponent
    from visualizer.src.sensing.motion_overview import SPECTRAL_FAMILIES, load_motion_overview
    from visualizer.src.sensing.recording_cache import RecordingCache
    from visualizer.src.sensing.run_summary_data import load_run_summary
    from visualizer.src.sensing.scene_recording import prepare_scene_recording
    from visualizer.src.sensing.startup_cache import prepare_startup_cache

    root, payloads = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    expected = {}
    try:
        prepare_scene_recording(provider, Event())
        context = result_context(payloads[0])
        load_run_summary(session.service, context, Event())
        for variant in ("primary", "raw"):
            load_motion_overview(
                session.service, context, variant, frozenset(SPECTRAL_FAMILIES), Event()
            )
            load_recording_history(
                session.service, context, variant, Event(), include_signal_products=False
            )
            for step in (0, 1):
                for view in VIEWS:
                    request = _get(session, view, step=step, variant=variant, pin=False)
                    expected[step, variant] = request.rd_result.power_linear.copy()
        ordinary = RecordingCache.for_provider(provider)
        cached_files = {
            path: path.read_bytes()
            for pattern in ("*.json.gz", "*.npz")
            for path in ordinary.root.glob(pattern)
        }
    finally:
        session.close()
        provider.close()

    provider = Hdf5Provider(root)
    try:
        original_load = provider.load_frame_projection

        def load_context_only(step, request):
            if FrameComponent.SENSING in request.components:
                assert request.sensing_products == frozenset({"config_resolved"})
            return original_load(step, request)

        monkeypatch.setattr(provider, "load_frame_projection", load_context_only)
        monkeypatch.setattr(
            provider,
            "iter_frame_projections",
            lambda *_: pytest.fail("Advance preparation rebuilt the cached result index"),
        )
        assert prepare_startup_cache(provider, Event())
        assert all(path.read_bytes() == contents for path, contents in cached_files.items())
        # Ordinary cache eviction must not leave a supposedly complete bundle
        # dependent on the entries it reused.
        for path in cached_files:
            path.unlink()
        session = DashboardSession(provider)
        for (step, variant), power in expected.items():
            request = _get(session, "analysis", step=step, variant=variant, pin=False)
            np.testing.assert_array_equal(request.rd_result.power_linear, power)
        assert prepare_scene_recording(provider, Event()).disk_hit
    finally:
        session.close()
        provider.close()


def test_common_views_share_one_request_and_preserve_exact_inspection(recording):
    root, payloads = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        current = _get(session, "analysis")
        detector = _get(session, "diagnostics")
        filtering = _get(session, "processing-clutter")
        angular = _get(session, "angular")
        assert len(session._request_cache) == 1
        assert current.rd_result is detector.rd_result is filtering.rd_result is angular.rd_result
        # Adding raw Filtering evidence creates a new immutable superset; the
        # primary maps remain shared and the earlier request is not mutated.
        assert "raw_rd" not in current.display_frame["sensing"]
        assert "raw_rd" in angular.display_frame["sensing"]
        assert (
            current.display_frame["sensing"]["temporal"]
            == angular.display_frame["sensing"]["temporal"]
        )
        assert "range_doppler_angle_by_pair" not in angular.display_frame["sensing"]
        np.testing.assert_array_equal(
            current.rd_result.power_linear, payloads[0]["range_doppler_power_by_pair"][0]
        )
        expected_angle = prepare_angular(payloads[0])
        np.testing.assert_array_equal(
            angular.prepared_angular.range_azimuth_power, expected_angle.range_azimuth_power
        )
        np.testing.assert_array_equal(
            angular.prepared_angular.doppler_azimuth_power, expected_angle.doppler_azimuth_power
        )
        expected_filtering = inspect_processing(payloads[0], "clutter")
        assert expected_filtering.before is not None
        np.testing.assert_array_equal(
            filtering.processing_inspection.before, expected_filtering.before
        )
        np.testing.assert_array_equal(
            filtering.processing_inspection.after, expected_filtering.after
        )
        expected_detector = build_detection_stage_bundle(payloads[0])
        assert expected_detector.available
        assert detector.detection_stage_bundle.available
        cell = CellSelection(expected_detector.event.pair_id, doppler_bin=1, range_bin=2)
        assert detector.detection_stage_bundle.inspect(cell) == expected_detector.inspect(cell)
        for actual, expected in zip(
            detector.detection_stage_bundle.products, expected_detector.products, strict=True
        ):
            np.testing.assert_array_equal(actual.values, expected.values)
        assert angular.prepared_views.issuperset(VIEWS)
        assert _get(session, "analysis").prepared_angular is angular.prepared_angular
    finally:
        session.close()
        provider.close()


def test_reopening_uses_disk_models_without_reading_dense_source_products(recording, monkeypatch):
    root, _ = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    expected = {}
    try:
        for view in VIEWS:
            expected[view] = _get(session, view)
    finally:
        session.close()
        provider.close()
    provider = Hdf5Provider(root)
    reads = _loaded_products(provider, monkeypatch)
    reopened = DashboardSession(provider)
    try:
        for view in VIEWS:
            request = _get(reopened, view)
            assert request is not None
            assert request.effective_display_step == 0
            np.testing.assert_array_equal(
                request.rd_result.power_linear, expected[view].rd_result.power_linear
            )
        assert reads == []
        assert reopened.service.recording_cache.hits >= 4
        assert request.prepared_angular is not None
        assert request.detection_stage_bundle.available
        assert request.processing_inspection.before is not None
    finally:
        reopened.close()
        provider.close()


def test_live_angle_reuses_persistent_projection_without_retaining_source_cube(
    recording, monkeypatch
):
    root, payloads = recording
    key = (0, "live", "primary", True, "auto", ("range_doppler_angle",))
    expected = prepare_angular(payloads[0])
    for reopen in (False, True):
        provider = Hdf5Provider(root)
        reads = _loaded_products(provider, monkeypatch)
        session = DashboardSession(provider)
        try:
            request = session._load_request(key)
            assert request.rd_result is None
            assert request.rp_result is None
            assert "range_doppler_angle_by_pair" not in request.display_frame["sensing"]
            np.testing.assert_array_equal(
                request.prepared_angular.range_azimuth_power, expected.range_azimuth_power
            )
            assert "angular" in request.prepared_views
            if reopen:
                assert reads == []
            else:
                assert (0, "range_doppler_angle_by_pair") in reads
        finally:
            session.close()
            provider.close()


def test_current_and_pinned_angle_share_common_request_owner(recording):
    root, _ = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        current = _get(session, "analysis")
        pinned = session._load_request(
            (0, "analysis", "primary", True, "auto", ("range_doppler_angle",))
        )
        assert len(session._request_cache) == 1
        assert pinned.rd_result is current.rd_result
        assert pinned.rp_result is current.rp_result
        assert pinned.prepared_angular is not None
        assert "range_doppler_angle_by_pair" not in pinned.display_frame["sensing"]
        assert _get(session, "angular").prepared_angular is pinned.prepared_angular
    finally:
        session.close()
        provider.close()


def test_source_change_invalidates_prepared_disk_models(recording, monkeypatch):
    root, _ = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        _get(session, "angular")
        original_signature = session.service.recording_cache.signature
        chunk = next(path for path in provider.recording_paths() if path.suffix == ".h5")
        manifest_path = provider.recording_paths()[0]
    finally:
        session.close()
        provider.close()
    with h5py.File(chunk, "r+") as source:
        source.attrs["test_source_revision"] = 1
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["chunks"]:
        if row["file"] == chunk.name:
            row["size_bytes"] = chunk.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    provider = Hdf5Provider(root)
    reads = _loaded_products(provider, monkeypatch)
    reopened = DashboardSession(provider)
    try:
        request = _get(reopened, "angular")
        assert request.prepared_angular is not None
        assert reopened.service.recording_cache.signature != original_signature
        assert (0, "range_doppler_angle_by_pair") in reads
    finally:
        reopened.close()
        provider.close()


def test_prefetch_returns_before_preparation_and_does_not_evict_visited(recording, monkeypatch):
    root, _ = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    release, entered = Event(), Event()
    try:
        visited = _get(session, "analysis")
        original_keys = tuple(session._request_cache)
        session.cache_policy = replace(
            session.cache_policy, request_bytes=session._request_cache_bytes * 3 // 2
        )
        original = session._cached_request

        def delayed(key, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("prefetch blocked its caller")
            return original(key, **kwargs)

        monkeypatch.setattr(session, "_cached_request", delayed)
        session.prefetch_next((1, "analysis", "primary", True, "auto"))
        assert entered.wait(timeout=3)
        assert not session._prefetch_pending.done()
        release.set()
        session._prefetch_pending.result(timeout=5)
        assert tuple(session._request_cache) == original_keys
        assert next(iter(session._request_cache.values()))[0].rd_result is visited.rd_result
    finally:
        release.set()
        session.close()
        provider.close()


def test_shared_cache_preserves_variant_and_held_result_provenance(recording):
    root, payloads = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    try:
        primary = _get(session, "analysis", step=1, pin=False)
        raw = _get(session, "analysis", step=1, variant="raw", pin=False)
        assert primary.variant == "primary" and raw.variant == "raw"
        np.testing.assert_array_equal(raw.rd_result.power_linear, payloads[1]["raw_rd"][0])
        np.testing.assert_array_equal(
            primary.rd_result.power_linear, payloads[1]["range_doppler_power_by_pair"][0]
        )
        assert _get(session, "analysis", step=2, pin=False) is None
        held = _get(session, "analysis", step=2, pin=True)
        assert held.state_step == 2
        assert held.display_step == 1
        assert held.display_from_cache
        assert held.rd_result is primary.rd_result
        assert held.replay_event.scene_binding.current_scene_frame_id == 2
        assert held.replay_event.scene_binding.bound_scene_frame_id == 1
        assert held.replay_event.scene_binding.pinned
        assert not primary.replay_event.scene_binding.pinned
    finally:
        session.close()
        provider.close()


def test_absent_angular_variant_finishes_loading_and_reports_unavailable(recording, qtbot):
    from visualizer.src.sensing.sensing_window import SensingWindow

    root, _ = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("angular")
    try:
        primary = _get(session, "angular")
        window.update_from_request(primary)
        assert window._angular_status == "typed primary"
        raw = _get(session, "angular", variant="raw")
        assert "angular" in raw.prepared_views
        assert raw.prepared_angular is None
        window.update_from_request(raw)
        assert window._angular_status == "no typed raw product"
    finally:
        session.close()
        provider.close()


def test_filtering_keeps_detector_models_without_updating_hidden_detector_plots(
    recording, qtbot, monkeypatch
):
    from visualizer.src.sensing.sensing_window import SensingWindow

    root, _ = recording
    provider = Hdf5Provider(root)
    session = DashboardSession(provider)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._open_result_detail("diagnostics", "cfar")
    try:
        detector = _get(session, "diagnostics")
        window.update_from_request(detector)
        assert window.cfar_rd_image.image is not None
        filtering = _get(session, "processing-clutter")
        assert filtering.detection_stage_bundle is detector.detection_stage_bundle
        assert filtering.processing_inspection.before is not None

        def hidden_update(*args, **kwargs):
            pytest.fail("Filtering updated hidden detector figures")

        with monkeypatch.context() as hidden:
            hidden.setattr(window, "_update_cfar_rd", hidden_update)
            hidden.setattr(window, "_update_cfar_rp", hidden_update)
            window._open_result_detail("diagnostics", "clutter")
            window.update_from_request(filtering)
            window._update_detection_stage_inspector_from_controls()
            assert window.processing_panel.model is filtering.processing_inspection
            window.processing_panel.range_bin.setValue(2)
            window.processing_panel.render()
            x, y = window.processing_panel.after_curve.getData()
            np.testing.assert_array_equal(x, filtering.processing_inspection.doppler)
            np.testing.assert_allclose(
                y, 10 * np.log10(np.maximum(filtering.processing_inspection.after[:, 2], 1e-30))
            )
        updates = []
        for name in ("_update_cfar_rd", "_update_cfar_rp"):
            original = getattr(window, name)

            def record(*args, _name=name, _original=original, **kwargs):
                updates.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(window, name, record)
        window._open_result_detail("diagnostics", "cfar")
        window.update_from_request(_get(session, "diagnostics"))
        assert set(updates) == {"_update_cfar_rd", "_update_cfar_rp"}
        assert window.cfar_rd_image.image is not None
        assert window.cfar_rd_plot.isVisible()
    finally:
        session.close()
        provider.close()
