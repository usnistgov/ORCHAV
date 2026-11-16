"""Persistent display preparation retains scientific selection and needs no hot I/O."""

import json
from concurrent.futures import CancelledError, Future
from copy import deepcopy
from dataclasses import asdict, replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QCheckBox, QWidget

from shared.frames.providers import Hdf5Provider
from shared.sensing.targets import build_target_catalog, build_target_state, build_target_truth
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
from tests.visualizer.unit.test_sensing_playback import _SnapshotProvider
from tests.visualizer.unit.test_sensing_prepared_cache import recording as recording
from visualizer.src.sensing import dashboard_data
from visualizer.src.sensing import scene_recording as module
from visualizer.src.sensing.dashboard_data import DashboardSession, result_context
from visualizer.src.sensing.preparation import PreparedSensing
from visualizer.src.sensing.recording_cache import RecordingCache
from visualizer.src.services.sensing_service import SensingService


def _provider(range_only=False):
    frames = {}
    for step in range(9):
        base = build_standard_mpc_frame(frame_idx=step)
        frames[step] = replace(base, timestamp_s=step * 0.25)
    for storage, bound in ((1, 0), (4, 3), (7, 6)):
        detection = {"range_m": 3.0 + bound, "snr_db": 12.0}
        if not range_only:
            detection.update(
                track_id=2,
                track_confirmed=True,
                x_m=3.0 + bound,
                y_m=1.0,
                z_m=2.0,
                track_x_m=3.25 + bound,
                track_y_m=1.5,
                track_z_m=2.5,
                track_vx_m_s=1.0,
                track_vy_m_s=2.0,
                track_vz_m_s=0.0,
            )
        frames[storage] = replace(
            frames[storage],
            sensing=build_visualizer_sensing_payload(
                {"detections": [detection]}, scene_frame_id=bound, sample_time_s=bound * 0.25
            ),
        )
    return _SnapshotProvider(frames)


def _service(provider):
    return SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))


@pytest.fixture
def disk(tmp_path, monkeypatch):
    manifest = tmp_path / "frames_manifest.json"
    manifest.write_text("{}")
    chunk = tmp_path / "frames.h5"
    chunk.write_bytes(b"source stamps")
    monkeypatch.setattr(
        RecordingCache,
        "for_provider",
        classmethod(lambda cls, provider: cls((manifest, chunk), root=tmp_path / "cache")),
    )
    return manifest, chunk


def _assert_same(expected, actual):
    if expected is None:
        assert actual is None
        return
    assert expected[0] == actual[0]
    assert [layer.name for layer in expected[1]] == [layer.name for layer in actual[1]]
    for a, b in zip(expected[1], actual[1], strict=True):
        np.testing.assert_allclose(a.payload.points, b.payload.points, rtol=1e-6, atol=1e-6)
        if a.payload.colors is not None:
            np.testing.assert_allclose(a.payload.colors, b.payload.colors, rtol=1e-6, atol=1e-7)
        if hasattr(a.payload, "lines"):
            np.testing.assert_array_equal(a.payload.lines, b.payload.lines)
        assert a.material == b.material
        # JSON arrays become lists; their semantic picking indices remain equal.
        assert (a.metadata or {}).get("replay_event_identity") == (b.metadata or {}).get(
            "replay_event_identity"
        )


@pytest.mark.parametrize("range_only", [False, True])
def test_all_views_match_validated_reader_and_reopen_without_source_reads(
    disk, monkeypatch, range_only
):
    provider = _provider(range_only)
    service = _service(provider)
    requests, expected = [], []
    for step in (6, 8, 3, 0, 2):  # Includes reverse seek and held history.
        scene = service._load_scene_pose_frame(step)
        for mode in ("detections", "tracks", "evidence", "full"):
            for full in (False, True):
                for arrows in (False, True):
                    kwargs = dict(
                        step=step,
                        mode=mode,
                        history_frames=2,
                        use_full_history=full,
                        show_velocity_arrows=arrows,
                        pin_freshest=False,
                    )
                    requests.append((scene, kwargs))
                    expected.append(service.resolve_scene_overlay_layers(scene, **kwargs))
    first = module.prepare_scene_recording(provider, Event())
    assert not first.disk_hit
    provider.requests.clear()

    def forbidden(*args, **kwargs):
        pytest.fail("Prepared playback must not read or validate source results")

    monkeypatch.setattr(provider, "load_frame_projection", forbidden)
    monkeypatch.setattr(SensingService, "_selection_replay_event", forbidden)
    service = _service(provider)
    service.prepared_scene = module.prepare_scene_recording(provider, Event())
    assert service.prepared_scene.disk_hit
    for (scene, kwargs), baseline in zip(requests, expected, strict=True):
        _assert_same(baseline, service.resolve_scene_overlay_layers(scene, **kwargs))
    assert not provider.requests


def test_cache_rebuilds_on_source_change_corruption_and_cancellation(disk):
    provider = _provider()
    first = module.prepare_scene_recording(provider, Event())
    assert not first.disk_hit
    assert module.prepare_scene_recording(provider, Event()).disk_hit
    disk[1].write_bytes(b"replacement source")
    assert not module.prepare_scene_recording(provider, Event()).disk_hit
    cache = RecordingCache.for_provider(provider)
    cache.save("scene", {"records": [{"bad": True}]}, module.SCENE_CACHE_KEY)
    assert not module.prepare_scene_recording(provider, Event()).disk_hit
    event = Event()
    event.set()
    with pytest.raises(CancelledError):
        module.prepare_scene_recording(provider, event)


def test_overview_rebuilds_unversioned_scores_and_reuses_current_cache(tmp_path, monkeypatch):
    provider = _provider()
    catalog = build_target_catalog([])
    for storage in (1, 4, 7):
        sensing = provider.frames[storage].sensing
        reference_time = sensing["temporal"]["result"]["reference_time_s"]
        sensing["target_catalog"] = catalog
        sensing["target_state"] = build_target_state(
            catalog, scene_frame_id=storage, reference_time_s=reference_time, targets=[]
        )
        sensing["truth"] = build_target_truth(catalog, targets=[])
        sensing["temporal"]["truth"] = {
            "reference_time_s": reference_time,
            "policy": "exact",
        }
        sensing["detections"][0]["velocity_m_s"] = 0.0
    context = result_context(provider.frames[1].sensing)
    monkeypatch.setattr(RecordingCache, "for_provider", classmethod(lambda cls, provider: None))
    expected = dashboard_data.load_recording_history(_service(provider), context, "auto", Event())
    assert expected.evaluation_summary["total_fp"] == 3

    manifest = tmp_path / "frames_manifest.json"
    manifest.write_text("{}")
    cache = RecordingCache((manifest,), root=tmp_path / "cache")
    # This is the persisted key format used before evaluation protocols were included.
    unversioned_key = (
        *context,
        "auto",
        dashboard_data.MAX_HISTORY_RESULTS,
        dashboard_data.MAX_HISTORY_BYTES,
        dashboard_data.SIGNAL_TRUTH_VERSION,
        dashboard_data.DEFAULT_RANGE_GATE_M,
        dashboard_data.DEFAULT_VELOCITY_GATE_M_S,
    )
    cache.save(
        "overview",
        {
            "records": [asdict(record) for record in expected.records],
            "truncated": False,
            "evaluation_summary": {**expected.evaluation_summary, "total_fp": 0},
        },
        unversioned_key,
    )
    monkeypatch.setattr(RecordingCache, "for_provider", classmethod(lambda cls, provider: cache))
    service = _service(provider)
    refreshed = dashboard_data.load_recording_history(service, context, "auto", Event())
    assert refreshed.evaluation_summary == expected.evaluation_summary
    assert cache.hits == 0

    def forbidden(*args, **kwargs):
        pytest.fail("Current-protocol overview must not reload source products")

    monkeypatch.setattr(service, "dashboard_history_products", forbidden)
    warm = dashboard_data.load_recording_history(service, context, "auto", Event())
    assert warm.evaluation_summary == expected.evaluation_summary
    assert cache.hits == 1


def test_overview_rebuilds_after_evaluation_protocol_changes(disk, monkeypatch):
    provider = _provider()
    service = _service(provider)
    context = result_context(provider.frames[1].sensing)
    dashboard_data.load_recording_history(service, context, "auto", Event())
    reads = []
    original_load = service.dashboard_history_products

    def load(storage_id, variant):
        reads.append(storage_id)
        return original_load(storage_id, variant)

    monkeypatch.setattr(service, "dashboard_history_products", load)
    monkeypatch.setattr(
        dashboard_data, "EVALUATION_CACHE_PROTOCOL", "changed-evaluation-protocol", raising=False
    )
    refreshed = dashboard_data.load_recording_history(service, context, "auto", Event())
    assert len(refreshed.records) == 3
    assert reads == [1, 4, 7]


def test_prepared_pose_edits_and_inventory_changes_use_validated_fallback(disk):
    provider = _provider()
    service = _service(provider)
    prepared = module.prepare_scene_recording(provider, Event())
    scene = service._load_scene_pose_frame(0)
    kwargs = dict(
        step=0,
        mode="detections",
        history_frames=12,
        use_full_history=True,
        show_velocity_arrows=False,
        pin_freshest=False,
    )
    moved = {**scene, "rx_positions": scene["rx_positions"] + 10}
    assert prepared.resolve(service, moved, **kwargs) is NotImplemented
    provider.snapshot = (*provider.snapshot, 100)
    assert prepared.resolve(service, scene, **kwargs) is NotImplemented


def test_zero_detections_keep_existing_exact_and_hold_selection(disk):
    provider = _provider()
    provider.frames[4] = replace(
        provider.frames[4],
        sensing=build_visualizer_sensing_payload(
            {"detections": []}, scene_frame_id=3, sample_time_s=0.75
        ),
    )
    service = _service(provider)
    prepared = module.prepare_scene_recording(provider, Event())
    assert len(prepared.records) == 3
    for mode in ("detections", "tracks", "full"):
        for step in (3, 5):
            scene = service._load_scene_pose_frame(step)
            kwargs = dict(
                step=step,
                mode=mode,
                history_frames=12,
                use_full_history=True,
                show_velocity_arrows=True,
                pin_freshest=False,
            )
            _assert_same(
                service.resolve_scene_overlay_layers(scene, **kwargs),
                prepared.resolve(service, scene, **kwargs),
            )


def test_compact_temporal_metadata_preserves_values_and_detaches_retained_fields(disk):
    provider = _provider()
    source = provider.frames[1].sensing
    expected = deepcopy(source["temporal"])
    expected["acquisition"].pop("samples")
    prepared = module.prepare_scene_recording(provider, Event())
    temporal = prepared.records[0]["sensing"]["temporal"]
    assert temporal == expected
    assert source["temporal"]["acquisition"]["samples"]

    source["temporal"]["acquisition"]["acquisition_id"] = "source edit"
    source["temporal"]["result"]["reference_time_s"] = 99.0
    assert temporal == expected
    temporal["display_binding"]["scene_frame_id"] = 99
    assert source["temporal"]["display_binding"]["scene_frame_id"] == 0


@pytest.mark.parametrize("variant", ["auto", "primary", "raw"])
def test_dashboard_loads_only_requested_history_and_reuses_it(disk, monkeypatch, variant):
    from visualizer.src.sensing.preparation import prepare_sensing

    # Compute the exact recorded history without populating the disk cache.
    reference_provider = _provider()
    sensing = reference_provider.frames[1].sensing
    with monkeypatch.context() as isolated:
        isolated.setattr(RecordingCache, "for_provider", classmethod(lambda cls, provider: None))
        expected = dashboard_data.load_recording_history(
            _service(reference_provider), result_context(sensing), variant, Event()
        )
    provider = _provider()
    provider.source = None
    prepared = prepare_sensing(provider, Event())
    session = DashboardSession(provider, prepared=prepared)
    # The panel selects a result before requesting its recording history.
    session.service.dashboard_result_rows()
    reads = []
    original = session.service.dashboard_history_products

    def load(storage_id, selected_variant):
        reads.append((storage_id, selected_variant))
        return original(storage_id, selected_variant)

    monkeypatch.setattr(session.service, "dashboard_history_products", load)
    try:
        session.request_history(sensing, variant)
        history = session.history_pending.result(timeout=5)
        assert reads == [(storage, variant) for storage in (1, 4, 7)]
        assert session.history_key == (*result_context(sensing), variant)
        for actual, reference in zip(history.records, expected.records, strict=True):
            np.testing.assert_equal(asdict(actual), asdict(reference))
        np.testing.assert_equal(history.evaluation_summary, expected.evaluation_summary)
        assert history.truncated == expected.truncated
        session.history = history
        session.history_pending = None
        session.request_history(sensing, variant)
        assert session.history_pending is None
        assert session.history is history
        assert len(reads) == 3
    finally:
        session.close()


def test_startup_prepares_scene_without_reading_dashboard_histories(disk, monkeypatch):
    from visualizer.src.sensing.preparation import prepare_sensing

    provider = _provider()
    provider.source = None

    def unexpected_dashboard_read(*args, **kwargs):
        pytest.fail("Startup must leave per-result dashboard views to on-demand loading")

    monkeypatch.setattr(DashboardSession, "_load_request", unexpected_dashboard_read)
    monkeypatch.setattr(SensingService, "dashboard_history_products", unexpected_dashboard_read)
    monkeypatch.setattr(dashboard_data, "load_recording_history", unexpected_dashboard_read)
    prepared = prepare_sensing(provider, Event())

    assert len(prepared.scene.records) == 3
    assert not hasattr(prepared, "histories")


def test_explicit_cli_prepares_components_reused_by_dashboard(recording, monkeypatch, capsys):
    from visualizer import sensing_prepare
    from visualizer.src.sensing.cache_policy import MIB, SensingCachePolicy
    from visualizer.src.sensing.motion_overview import load_motion_overview

    root, payloads = recording
    args = [
        str(root),
        "--sensing-3d",
        "off",
        "--sensing-overview-cache-mb",
        "8",
        "--sensing-history-cache-mb",
        "12",
        "--sensing-history-results",
        "32",
    ]
    assert sensing_prepare.main(args) == 0
    assert "disk cache built" in capsys.readouterr().out
    provider = Hdf5Provider(root)
    policy = SensingCachePolicy(overview_bytes=8 * MIB, history_bytes=12 * MIB, history_results=32)
    session = DashboardSession(provider, cache_policy=policy)

    def unexpected(*args, **kwargs):
        pytest.fail("Reusing the offline prepared cache reread numeric source products")

    monkeypatch.setattr(SensingService, "dashboard_spectral_products", unexpected)
    monkeypatch.setattr(SensingService, "_dashboard_products_from_validated", unexpected)
    try:
        cache = RecordingCache.for_provider(provider)
        marker = json.loads((cache._prepared_root / "complete.json").read_text())
        assert not marker["scene_prepared"]
        assert marker["coverage_policy"] == {
            "overview_bytes": 8 * MIB,
            "history_bytes": 12 * MIB,
            "history_results": 32,
        }
        assert marker["capabilities"]["summary"][0]["acquisitions"] == 2
        request = session._load_request((0, "analysis", "primary", False, "auto"))
        np.testing.assert_array_equal(
            request.rd_result.power_linear, payloads[0]["range_doppler_power_by_pair"][0]
        )
        overview = load_motion_overview(
            session.service,
            result_context(request.display_frame["sensing"]),
            "primary",
            frozenset({"summed-rd", "range-profile"}),
            Event(),
            byte_limit=policy.overview_bytes,
        )
        coverage = overview.families["summed-rd"].coverage
        assert coverage.available_results == 2
        assert coverage.state == "unavailable"
        assert "no sampled acquisition clock" in coverage.reason
        assert sensing_prepare.main(args) == 0
        assert "disk cache reused" in capsys.readouterr().out
    finally:
        session.close()
        provider.close()


@pytest.mark.parametrize("mode", ("all-ranges", "saved-stft"))
def test_motion_cli_prepares_only_its_scope_by_default(recording, monkeypatch, capsys, mode):
    from visualizer import sensing_prepare
    from visualizer.src.sensing import startup_cache

    def unrelated(*args, **kwargs):
        pytest.fail("Focused CLI preparation requested unrelated figure or 3D work")

    monkeypatch.setattr(startup_cache, "load_run_summary", unrelated)
    monkeypatch.setattr(startup_cache, "prepare_scene_recording", unrelated)
    monkeypatch.setattr(startup_cache, "load_recording_history", unrelated)
    assert (
        sensing_prepare.main(
            [str(recording[0]), "--sensing-figures", "motion", "--sensing-motion", mode]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Motion disk cache ready" in output
    assert "3D sensing deferred" in output
    provider = Hdf5Provider(recording[0])
    try:
        assert RecordingCache.for_provider(provider)._prepared_root is None
    finally:
        provider.close()


def test_offline_motion_choice_requires_motion_scope(monkeypatch):
    from visualizer import sensing_prepare

    monkeypatch.setattr(
        sensing_prepare,
        "Hdf5Provider",
        lambda *_: pytest.fail("Invalid preparation scope reached the recording reader"),
    )
    with pytest.raises(SystemExit):
        sensing_prepare.main(["unused", "--sensing-motion", "saved-stft"])


def test_startup_publishes_only_current_provider_and_cleanup_cancels(qapp, monkeypatch):
    from visualizer.src.sensing import preparation_controller as controller
    from visualizer.src.sensing.saved_tracking_scene import SavedTrackingScene

    provider = _provider()
    monkeypatch.setattr(controller, "Hdf5Provider", type(provider))
    future = Future()
    monkeypatch.setattr(
        controller,
        "ThreadPoolExecutor",
        lambda **k: SimpleNamespace(submit=lambda *a: future, shutdown=lambda **k: None),
    )
    service = _service(provider)
    viz = service.visualizer
    updates = []
    viz.schedule_update = lambda: updates.append(True)
    viz._set_status_message = lambda *a: None
    panel = SimpleNamespace(
        parent=viz,
        _container=QWidget(),
        _get_service=lambda: service,
        widgets={"show_sensing_detections": QCheckBox()},
        _closed=False,
        saved_track_scene=SavedTrackingScene(),
    )
    job = controller.SensingPreparation(panel)
    assert service.preparation_pending
    assert not panel.widgets["show_sensing_detections"].isEnabled()
    prepared = PreparedSensing(SimpleNamespace(token=module.provider_token(provider)), None, None)
    future.set_result(prepared)
    job.poll()
    assert service.prepared_sensing is prepared
    assert not service.preparation_pending and updates == [True]
    job.close()
    assert job.cancelled.is_set()


@pytest.mark.parametrize("owner", ["provider", "panel", "active"])
def test_upfront_scene_and_tracking_do_not_start_another_preparation(qapp, monkeypatch, owner):
    from visualizer.src.sensing import preparation_controller as controller
    from visualizer.src.sensing.saved_tracking_scene import SavedTrackingScene

    provider = _provider()
    monkeypatch.setattr(controller, "Hdf5Provider", type(provider))

    def unexpected(**kwargs):
        pytest.fail("Upfront 3D data must not be read by another preparation worker")

    monkeypatch.setattr(controller, "ThreadPoolExecutor", unexpected)
    scene = SimpleNamespace(
        matches_provider=lambda other: other is provider,
        token=module.provider_token(provider),
        records=[{"_sensing_display_provenance": {"bound_scene_frame_id": 3}, "timestamp_s": 0.75}],
    )
    prepared = PreparedSensing(scene, None, None)
    retained = SimpleNamespace(prepared=prepared)
    service = _service(provider)
    viz = service.visualizer
    updates, plans = [], []
    viz.schedule_update = lambda: updates.append(True)
    viz._set_status_message = lambda *args: None
    panel = SimpleNamespace(
        parent=viz,
        _container=QWidget(),
        _get_service=lambda: service,
        widgets={"show_sensing_detections": QCheckBox()},
        _closed=False,
        saved_track_scene=SavedTrackingScene(),
        playback=SimpleNamespace(prepare_for_scenario=lambda *args: plans.append(args)),
    )
    if owner == "provider":
        provider._prepared_sensing_dashboard = retained
    elif owner == "active":
        panel._dashboard_session = retained
        retained.provider = provider
    else:
        panel._dashboard_cache = retained
    job = controller.SensingPreparation(panel)
    try:
        assert service.prepared_sensing is prepared
        assert service.prepared_scene is scene
        assert not service.preparation_pending
        assert plans == [(provider, job.playback_plan)]
        assert job.playback_plan.result() == ((3,), {3: 0.75})
        assert updates == [True]
        assert job.future is job.executor is None
        assert panel.widgets["show_sensing_detections"].isEnabled()
    finally:
        job.close()


def test_playback_plan_is_published_before_saved_tracking_preparation(qapp, monkeypatch):
    from visualizer.src.sensing import preparation
    from visualizer.src.sensing import preparation_controller as controller

    provider = _provider()
    monkeypatch.setattr(controller, "Hdf5Provider", type(provider))
    monkeypatch.setattr(
        controller,
        "ThreadPoolExecutor",
        lambda **k: SimpleNamespace(submit=lambda *a: Future(), shutdown=lambda **k: None),
    )
    scene = SimpleNamespace(
        records=[{"_sensing_display_provenance": {"bound_scene_frame_id": 3}, "timestamp_s": 0.75}]
    )
    monkeypatch.setattr(preparation, "prepare_scene_recording", lambda *a: scene)
    service = _service(provider)
    service.visualizer.frame_source.provider.source = "test-recording"
    registered = []
    panel = SimpleNamespace(
        parent=service.visualizer,
        _container=QWidget(),
        _get_service=lambda: service,
        widgets={},
        playback=SimpleNamespace(prepare_for_scenario=lambda *args: registered.append(args)),
    )
    job = controller.SensingPreparation(panel)

    def load_tracking(*args):
        assert registered == [(provider, job.playback_plan)]
        assert job.playback_plan.result(timeout=0) == ((3,), {3: 0.75})
        raise ValueError("Later tracking preparation failed")

    monkeypatch.setattr(
        preparation, "SavedTrackingLoader", lambda *a: SimpleNamespace(load=load_tracking)
    )
    try:
        with pytest.raises(ValueError, match="Later tracking"):
            job._prepare()
        # An unrelated later preparation failure must not revoke the verified timeline.
        assert job.playback_plan.result() == ((3,), {3: 0.75})
    finally:
        job.close()


def test_cancel_during_preparation_never_publishes_partial_artifact(disk):
    provider = _provider()
    event = Event()

    def cancel(stage, done, total):
        if stage == "Preparing 3D sensing":
            event.set()

    with pytest.raises(CancelledError):
        module.prepare_scene_recording(provider, event, cancel)
    assert RecordingCache.for_provider(provider).load("scene", module.SCENE_CACHE_KEY) is None


def test_empty_recording_is_ready_without_io():
    prepared = module.prepare_scene_recording(_SnapshotProvider({}), Event())
    assert not prepared.records


@pytest.mark.parametrize("outcome", ["replacement", "closed", "failure"])
def test_startup_ignores_retired_completion_and_reports_failures(qapp, monkeypatch, outcome):
    from visualizer.src.sensing import preparation_controller as controller
    from visualizer.src.sensing.saved_tracking_scene import SavedTrackingScene

    provider = _provider()
    monkeypatch.setattr(controller, "Hdf5Provider", type(provider))
    future = Future()
    monkeypatch.setattr(
        controller,
        "ThreadPoolExecutor",
        lambda **k: SimpleNamespace(submit=lambda *a: future, shutdown=lambda **k: None),
    )
    service = _service(provider)
    viz = service.visualizer
    messages, updates = [], []
    viz._set_status_message = lambda message, *_: messages.append(message)
    viz.schedule_update = lambda: updates.append(True)
    panel = SimpleNamespace(
        parent=viz,
        _container=QWidget(),
        _get_service=lambda: service,
        widgets={"show_sensing_detections": QCheckBox()},
        _closed=False,
        saved_track_scene=SavedTrackingScene(),
    )
    job = controller.SensingPreparation(panel)
    try:
        if outcome == "failure":
            future.set_exception(ValueError("bad source"))
        else:
            future.set_result(object())
            if outcome == "replacement":
                viz.frame_source.provider = _provider()
            else:
                panel._closed = True
        job.poll()
        assert service.prepared_scene is None
        assert not service.preparation_pending
        if outcome == "failure":
            assert "bad source" in messages[0]
            assert panel.widgets["show_sensing_detections"].isEnabled()
        else:
            assert not messages and not updates
    finally:
        job.close()
