"""Saved dashboard results survive loops, while changed inputs invalidate reuse."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.cache_sizing import estimate_retained_bytes
from tests.visualizer.unit.test_sensing_dashboard_recording import _frame, _history, _request
from tests.visualizer.unit.test_sensing_panel_fallback import _PanelFrameProvider
from visualizer.src.sensing import dashboard_data as module
from visualizer.src.sensing.dashboard_data import DashboardSession, load_dashboard_request
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.services.sensing_service import SensingService


def unexpected(*args, **kwargs):
    raise AssertionError("Unchanged saved data must be reused")


def _compact_session(provider=None, *, family="saved-stft"):
    from visualizer.src.sensing.motion_overview import (
        MotionOverview,
        SpectralCoverage,
        SpectralHistory,
    )

    session = DashboardSession(provider or _held_provider())
    rows = session.service.dashboard_result_rows()
    context = module.result_context(rows[0][1]["sensing"])
    cohort = [
        (i, f["sensing"]["temporal"])
        for i, f in rows
        if module.result_context(f["sensing"]) == context
    ]
    times = np.array([t["result"]["reference_time_s"] for _, t in cohort])
    item = SpectralHistory(
        family,
        np.ones((len(cohort), 4)),
        times,
        np.full(len(cohort), 0.75),
        np.arange(4.0),
        "primary",
        {},
        tuple((i, t["acquisition"]["acquisition_id"]) for i, t in cohort),
        SpectralCoverage(len(cohort), len(cohort), (min(times), max(times)), "complete", 1000000),
        ("primary",),
    )
    session.overview = MotionOverview(context, "primary", {family: item}, 1000000)
    session.overview_key = (*context, "primary", (family,))
    session.overview_inventory = session.service.dashboard_inventory_token()
    return session


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("family", ["saved-stft", "summed-rd"])
def test_compact_cursor_preserves_exact_and_causal_selection_without_history(
    monkeypatch, pinned, family
):
    from tests.visualizer.unit.test_sensing_dashboard_scheduler import _warmup_provider

    session = _compact_session(_warmup_provider(), family=family)
    reference = SensingService(
        SimpleNamespace(frame_source=SimpleNamespace(provider=session.provider))
    )
    families = frozenset({family})
    try:
        assert session.history is None
        for step, timestamp in ((0, 0.0), (3, 0.749), (3, 0.75), (5, 1.25), (8, 2.0), (0, 0.0)):
            scene = reference.dashboard_scene_frame(step)
            scene["timestamp_s"] = timestamp
            expected = reference.resolve_display_sensing_frame(
                scene,
                scene_frame_id=step,
                storage_frame_id=step,
                product_names=("config_resolved",),
                pin_freshest=pinned,
            )
            with monkeypatch.context() as patch:
                patch.setattr(session.provider, "load_frame_projection", unexpected)
                patch.setattr(session.service, "dashboard_result_rows", unexpected)
                actual = session.compact_motion_cursor(
                    step, "primary", families, pin_freshest=pinned, scene_time_s=timestamp
                )
            assert actual is not None
            assert actual.scene_id == step and actual.scene_time_s == timestamp
            if expected is None:
                assert actual.result is None
            else:
                assert (actual.result.storage_id, actual.result.scene_id) == (
                    expected.storage_frame_id,
                    expected.bound_scene_frame_id,
                )
                assert (
                    actual.result.time_s
                    == expected.product_frame["sensing"]["temporal"]["result"]["reference_time_s"]
                )
            assert not hasattr(actual, "detections") and not hasattr(actual, "evaluation")
    finally:
        session.close()


def test_compact_cursor_keeps_ties_and_rejects_a_different_winning_cohort(monkeypatch):
    from tests.visualizer.unit.test_sensing_exact_selection import _select, _service

    reference, provider = _service()
    # Duplicate the winning run/pair/acquisition; the stable storage tie-break
    # must still choose exactly the ordinary resolver's occurrence.
    provider.frames[11] = provider.frames[10]
    snapshot = tuple(provider.frames)
    monkeypatch.setattr(type(provider), "frame_ids_snapshot", property(lambda _: snapshot))
    session = _compact_session(provider)
    try:
        rows = session.service.dashboard_result_rows()
        expected = _select(reference, 5, ("config_resolved",), pinned=True)
        assert expected.storage_frame_id == 11
        selected = next(f["sensing"] for i, f in rows if i == expected.storage_frame_id)
        context = module.result_context(selected)
        session.overview = replace(session.overview, context=context)
        session.overview_key = (*context, "primary", ("saved-stft",))
        cursor = session.compact_motion_cursor(
            5, "primary", frozenset({"saved-stft"}), pin_freshest=True, scene_time_s=5.0
        )
        assert cursor.result.storage_id == expected.storage_frame_id
        assert cursor.result.scene_id == expected.bound_scene_frame_id
        assert (
            session.compact_motion_cursor(
                6, "primary", frozenset({"saved-stft"}), pin_freshest=True, scene_time_s=6.0
            )
            is None
        )
    finally:
        session.close()


@pytest.mark.parametrize(
    "change",
    ["generation", "inventory", "variant", "family", "limited", "error", "pending", "closed"],
)
def test_compact_cursor_rejects_changed_source_or_unsettled_demand(change):
    from concurrent.futures import Future

    session = _compact_session()
    families = frozenset({"saved-stft"})
    variant = "primary"
    try:
        assert session.compact_motion_cursor(3, variant, families, scene_time_s=0.75) is not None
        if change == "generation":
            session.provider._generation_epoch = 99
        elif change == "inventory":
            session.provider.frames[10] = session.provider.frames[7]
            session.provider.snapshot = tuple(sorted(session.provider.frames))
        elif change == "variant":
            variant = "raw"
        elif change == "family":
            families = frozenset({"summed-rd"})
        elif change in {"limited", "error"}:
            item = session.overview.families["saved-stft"]
            session.overview = replace(
                session.overview,
                families={
                    "saved-stft": replace(item, coverage=replace(item.coverage, state=change))
                },
            )
        elif change == "pending":
            session.overview_pending = Future()
        else:
            session.close()
        assert session.compact_motion_cursor(3, variant, families, scene_time_s=0.75) is None
    finally:
        session.close()


def test_compact_cursor_accepts_valid_unavailable_optional_family():
    session = _compact_session()
    try:
        item = session.overview.families["saved-stft"]
        optional = replace(
            item,
            family="range-profile",
            coverage=replace(
                item.coverage, state="unavailable", available_results=0, loaded_results=0
            ),
        )
        session.overview = replace(
            session.overview, families={**session.overview.families, "range-profile": optional}
        )
        families = frozenset(session.overview.families)
        session.overview_key = (*session.overview.context, "primary", tuple(sorted(families)))
        assert (
            session.compact_motion_cursor(3, "primary", families, scene_time_s=0.75).result.scene_id
            == 3
        )
    finally:
        session.close()


def test_navigation_is_reused_until_inventory_or_context_changes(monkeypatch):
    session = DashboardSession(_held_provider())
    try:
        session._load_request((1, "summary", "primary", True, "auto"))
        navigation = session.navigation
        calls = []
        original = module.result_locations
        monkeypatch.setattr(
            module, "result_locations", lambda *args: calls.append(args) or original(*args)
        )
        for step in (2, 5, 8, 1):
            session._load_request((step, "summary", "primary", True, "auto"))
            assert session.navigation is navigation
        assert calls == []
        session.provider._generation_epoch = 1
        session._load_request((1, "summary", "primary", True, "auto"))
        assert len(calls) == 1
    finally:
        session.close()


@pytest.mark.parametrize("mode,family", [("auto", "summed-rd"), ("saved-stft", "saved-stft")])
@pytest.mark.parametrize("ordinary_timeline", [False, True])
def test_compact_motion_panel_moves_only_cursor_and_loads_explicit_details(
    qtbot, monkeypatch, mode, family, ordinary_timeline
):
    from PySide6.QtCore import QSignalBlocker
    from PySide6.QtWidgets import QWidget

    from tests.visualizer.unit.test_sensing_dashboard_scheduler import _warmup_provider
    from visualizer.src.panels import sensing_panel

    session = _compact_session(_warmup_provider(), family=family)
    parent = SimpleNamespace(
        animation_running=True,
        animation_step=3,
        get_available_animation_steps=lambda: list(range(9)),
        sensing_playback=SimpleNamespace(
            steps=tuple(range(9)), timestamps={i: i * 0.25 for i in range(9)}
        ),
    )
    if ordinary_timeline:
        parent.sensing_playback.steps = None
        parent.sensing_playback.timestamps = {}
        parent.cache_service = SimpleNamespace(
            get_frame=lambda step: {"timestamp_ns": step * 250000000}
        )
    panel = sensing_panel.SensingPanel(parent)
    panel._container = QWidget()
    qtbot.addWidget(panel._container)
    window = panel._sensing_window = SensingWindow()
    window.show()
    window._set_view_mode("micro-doppler")
    with QSignalBlocker(window._micro_mode_box):
        window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData(mode))
    window._table_requested = False
    window._sync_browser_visibility()
    monkeypatch.setattr(window, "overview_families", lambda: frozenset({family}))
    monkeypatch.setattr(sensing_panel, "visualizer_frame_provider", lambda _: session.provider)
    monkeypatch.setattr(panel, "_dashboard_is_visible", lambda: True)
    monkeypatch.setattr(panel, "_set_status", lambda *_: None)
    panel._dashboard_session = session
    try:
        request = session._load_request((3, "micro-doppler", "primary", True, mode))
        request = session.with_recording_data(request)
        window.update_from_request(request)
        assert session.history is None
        assert window.can_move_compact_motion_cursor(session.overview)
        with monkeypatch.context() as patch:
            patch.setattr(window, "has_visible_live_figures", lambda: True)
            assert not window.can_move_compact_motion_cursor(session.overview)
        with monkeypatch.context() as patch:
            patch.setattr(window, "_linked_selection", object())
            assert not window.can_move_compact_motion_cursor(session.overview)
        image = window.collapsed_doppler_image.image
        assert image is not None
        requests = []
        monkeypatch.setattr(session, "request", requests.append)
        monkeypatch.setattr(session.provider, "load_frame_projection", unexpected)
        monkeypatch.setattr(window.collapsed_doppler_image, "setImage", unexpected)
        monkeypatch.setattr(window.collapsed_doppler_mesh, "setData", unexpected)
        monkeypatch.setattr(window, "update_from_request", unexpected)
        for step in (4, 8, 4):
            parent.animation_step = step
            panel._refresh_dashboard_async(session.provider, step, force=False)
            assert window._compact_motion_cursor.scene_id == step
            assert window.result_time_s() == (0.75 if step == 4 else 1.5)
            assert window.collapsed_doppler_image.image is image
            assert window._last_request is request
        assert requests == []
        if ordinary_timeline:
            with monkeypatch.context() as patch:
                patch.setattr(parent.cache_service, "get_frame", lambda _: None)
                parent.animation_step = 5
                panel._refresh_dashboard_async(session.provider, 5, force=False)
                assert (
                    len(requests) == 1
                )  # No clock means ordinary loading, never an inferred time.
            requests.clear()
            parent.animation_step = 4
        assert "detections" not in window.summary_label.text().lower()
        assert not window.result_browser.isVisible()
        assert not window.detection_stage_export_button.isEnabled()
        from concurrent.futures import Future

        session.history_key = (*session.overview.context, "primary")
        session.history_pending = Future()
        session.history_pending.set_result(module.RecordingHistory(()))
        panel._poll_dashboard()
        assert window._compact_motion_cursor.scene_id == 4
        assert window._last_request is request  # Late hidden detail cannot restore an older scene.
        # Rewind into warm-up retains all overview pixels without inventing a
        # current result. Crossing back into a result restores detail controls
        # through one ordinary authoritative request.
        for step in (0, 1):
            parent.animation_step = step
            panel._refresh_dashboard_async(session.provider, step, force=False)
            assert window._compact_motion_cursor.result is None
            assert not window.collapsed_doppler_cursor.isVisible()
            assert f"no sensing result at scene {step}" in window.provenance_label.text()
            assert window.collapsed_doppler_image.image is image
        assert requests == []
        parent.animation_step = 4
        panel._refresh_dashboard_async(session.provider, 4, force=False)
        assert len(requests) == 1
        # Explicit detail demand cannot display the old retained request.
        signals = []
        window.view_changed.connect(lambda: signals.append(True))
        window._toggle_detection_table(True)
        assert signals and not window.result_browser.isVisible()
        assert not window.can_move_compact_motion_cursor(session.overview)
        panel._refresh_dashboard_async(session.provider, 4, force=True)
        assert len(requests) == 2
        window._rerender_last_request()
        assert len(signals) >= 2  # Never invokes the old full-request renderer.
    finally:
        panel.cleanup()


@pytest.mark.parametrize("variant", ["primary", "auto", "clean"])
def test_compact_cursor_normalizes_primary_variant_aliases(variant):
    session = _compact_session()
    try:
        cursor = session.compact_motion_cursor(
            3, variant, frozenset({"saved-stft"}), scene_time_s=0.75
        )
        assert cursor is not None and cursor.result.scene_id == 3
    finally:
        session.close()


def _held_provider():
    from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
    from tests.visualizer.unit.test_sensing_preparation import _provider

    provider = _provider()
    for storage, bound in ((1, 0), (4, 3), (7, 6)):
        payload = build_visualizer_sensing_payload(
            {
                "range_doppler_power_by_pair": np.ones((1, 3, 4)),
                "range_axis_m": np.arange(4.0),
                "doppler_axis_hz": np.arange(-1.0, 2.0),
                "detections": [],
            },
            scene_frame_id=bound,
            sample_time_s=bound * 0.25,
        )
        provider.frames[storage] = replace(provider.frames[storage], sensing=payload)
    return provider


@pytest.mark.parametrize("view", ["analysis", "diagnostics", "tracking"])
def test_held_results_reuse_products_but_match_fresh_scene_replay(monkeypatch, view):
    provider = _held_provider()
    current = DashboardSession(provider)
    reference = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    try:
        first = current._load_request((1, view, "auto", True, "auto"))
        assert first.replay_event is not None
        assert first.display_data_token is not None
        # Holding, seeking backwards and looping reuse one acquisition, without
        # replacing physical scene times by indices or leaking future results.
        for step in (2, 1, 2):
            with monkeypatch.context() as patch:
                patch.setattr(module, "load_dashboard_request", unexpected)
                patch.setattr(module, "estimate_retained_bytes", unexpected)
                held = current._load_request((step, view, "auto", True, "auto"))
            fresh = load_dashboard_request(reference, step, view, "auto", True, "auto")
            assert held.replay_event.to_manifest() == fresh.replay_event.to_manifest()
            assert (
                held.display_frame["_sensing_display_provenance"]
                == fresh.display_frame["_sensing_display_provenance"]
            )
            assert held.state_step == step
            assert held.state_frame["timestamp_s"] == fresh.state_frame["timestamp_s"]
            assert held.display_data_token is first.display_data_token
            assert held.rd_result is first.rd_result
        assert len(current._request_cache) == 1
        next_result = current._load_request((4, view, "auto", True, "auto"))
        assert next_result.display_data_token is not first.display_data_token
        assert next_result.replay_event.key.acquisition_id != first.replay_event.key.acquisition_id
        rewind = current._load_request((2, view, "auto", True, "auto"))
        assert rewind.display_data_token is first.display_data_token
        assert current._load_request((2, view, "auto", False, "auto")) is None
        provider._generation_epoch = getattr(provider, "_generation_epoch", 0) + 1
        changed = current._load_request((1, view, "auto", True, "auto"))
        assert changed.display_data_token is not first.display_data_token
    finally:
        current.close()


def test_current_and_motion_share_one_saved_product_cache_entry(monkeypatch):
    current = DashboardSession(_held_provider())
    try:
        first = current._load_request((0, "analysis", "auto", True, "auto"))
        retained = current._request_cache_bytes
        monkeypatch.setattr(module, "load_dashboard_request", unexpected)
        monkeypatch.setattr(current.provider, "load_frame_projection", unexpected)
        for view, micro in (("micro-doppler", "collapsed"), ("analysis", "gated")):
            request = current._load_request((0, view, "auto", True, micro))
            assert request.display_data_token is first.display_data_token
            assert request.rd_result is first.rd_result
            assert request.display_frame is first.display_frame
        assert len(current._request_cache) == 1
        assert current._request_cache_bytes == retained
    finally:
        current.close()


@pytest.mark.parametrize("initial_pin", [False, True])
def test_pin_toggle_reuses_same_result_with_exact_fresh_provenance(monkeypatch, initial_pin):
    provider = _held_provider()
    current = DashboardSession(provider)
    reference = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    try:
        first = current._load_request((0, "analysis", "auto", initial_pin, "auto"))
        for pinned in (not initial_pin, initial_pin, not initial_pin):
            with monkeypatch.context() as patch:
                patch.setattr(module, "load_dashboard_request", unexpected)
                patch.setattr(module, "estimate_retained_bytes", unexpected)
                reused = current._load_request((0, "analysis", "auto", pinned, "auto"))
            fresh = load_dashboard_request(reference, 0, "analysis", "auto", pinned, "auto")
            assert reused.replay_event.to_manifest() == fresh.replay_event.to_manifest()
            assert (
                reused.display_frame["_sensing_display_provenance"]
                == fresh.display_frame["_sensing_display_provenance"]
            )
            assert reused.display_data_token is first.display_data_token
            assert reused.rd_result is first.rd_result
        assert len(current._request_cache) == 1
        # Removing hold cannot expose the prior CPI on an unbound scene.
        assert current._load_request((2, "analysis", "auto", False, "auto")) is None
    finally:
        current.close()


@pytest.mark.parametrize("timestamp", [None, 0.1])
def test_cache_rebinding_falls_back_for_missing_or_changed_scene_clock(monkeypatch, timestamp):
    provider = _held_provider()
    current = DashboardSession(provider)
    try:
        first = current._load_request((0, "analysis", "auto", False, "auto"))
        assert first.replay_event is not None
        provider.frames[0] = replace(provider.frames[0], timestamp_s=timestamp)
        fresh_reads = []

        def fresh(*args):
            fresh_reads.append(args[1:])
            return load_dashboard_request(*args)

        monkeypatch.setattr(module, "load_dashboard_request", fresh)
        request = current._load_request((0, "analysis", "auto", True, "auto"))
        assert len(fresh_reads) == 1
        assert request.display_data_token is not first.display_data_token
        assert request.state_frame["timestamp_s"] == timestamp
        if timestamp is None:
            assert request.replay_event is None
        else:
            assert request.replay_event.scene_binding.bound_scene_time_s == timestamp
    finally:
        current.close()


def test_playback_context_preserves_plots_and_updates_provenance(qtbot, monkeypatch):
    current = DashboardSession(_held_provider())
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    try:
        first = current._load_request((1, "analysis", "auto", True, "auto"))
        window.update_from_request(first)
        held = current._load_request((2, "analysis", "auto", True, "auto"))
        monkeypatch.setattr(window, "_update_range_velocity", unexpected)
        assert window.advance_playback_context(held)
        assert window._last_request is held
        assert "state=2" in window.provenance_label.toolTip()
        assert "held result" in window.provenance_label.text()
        assert not window.advance_playback_context(replace(held, display_data_token=object()))
        assert not window.advance_playback_context(replace(held, recording_history=_history()))
        assert not window.advance_playback_context(replace(held, saved_tracking_error="changed"))
    finally:
        current.close()


def test_cached_validation_is_scoped_to_owned_payload_and_completeness(monkeypatch):
    from copy import deepcopy

    from visualizer.src.sensing import update_request

    current = DashboardSession(_held_provider())
    try:
        request = current._load_request((0, "analysis", "auto", True, "auto"))
        checked = []
        original = update_request.assert_sensing_contract

        def validate(frame, **kwargs):
            checked.append(frame)
            return original(frame, **kwargs)

        monkeypatch.setattr(update_request, "assert_sensing_contract", validate)
        request.validate_display_frame()
        replace(request, recording_history=_history()).validate_display_frame()
        assert len(checked) == 0  # The private product snapshot was validated on the worker.
        replace(request, selective_products=False).validate_display_frame()
        assert len(checked) == 1
        changed = deepcopy(request.display_frame)
        del changed["sensing"]["temporal"]
        with pytest.raises(ValueError):
            replace(request, display_frame=changed).validate_display_frame()
        assert len(checked) == 2
        caller_owned = replace(request, display_data_token=None)
        caller_owned.validate_display_frame()
        caller_owned.validate_display_frame()
        assert len(checked) == 4
    finally:
        current.close()


def test_cached_validation_token_does_not_adopt_replaced_payload(monkeypatch):
    from copy import deepcopy

    from visualizer.src.sensing import update_request

    current = DashboardSession(_held_provider())
    try:
        request = current._load_request((0, "analysis", "auto", True, "auto"))
        checked = []
        original = update_request.assert_sensing_contract

        def validate(frame, **kwargs):
            checked.append(frame)
            return original(frame, **kwargs)

        monkeypatch.setattr(update_request, "assert_sensing_contract", validate)
        request.validate_display_frame()
        replace(request, display_frame=dict(request.display_frame)).validate_display_frame()
        assert len(checked) == 0  # Rebinding only the wrapper preserves ownership.

        caller_frame = deepcopy(request.display_frame)
        caller_request = replace(request, display_frame=caller_frame)
        caller_request.validate_display_frame()
        caller_request.validate_display_frame()
        assert len(checked) == 2
        assert request.display_data_token.sensing is request.display_frame["sensing"]

        del caller_frame["sensing"]["temporal"]
        with pytest.raises(ValueError):
            caller_request.validate_display_frame()
        request.validate_display_frame()
        assert len(checked) == 3  # Caller edits cannot replace or poison cache validation.
    finally:
        current.close()


def test_progress_does_not_flash_for_fast_playback_context_updates(qtbot, monkeypatch):
    from visualizer.src.sensing import sensing_window

    now = [10.0]
    monkeypatch.setattr(sensing_window, "monotonic", lambda: now[0])
    window = SensingWindow()
    qtbot.addWidget(window)
    window.set_loading_progress("Initial read", 0, 0, active=True)
    assert not window.loading_progress.isHidden()
    window.set_loading_progress("Ready", 0, 0, active=False)
    window._last_request = _request()
    window.set_loading_progress("Loading result", 0, 0, active=True)
    assert window.loading_progress.isHidden()
    now[0] += 0.05
    window.set_loading_progress("Ready", 0, 0, active=False)
    assert window.loading_progress.isHidden()
    window.set_loading_progress("Slow read", 0, 0, active=True)
    now[0] += 0.2
    window.set_loading_progress("Slow read", 0, 0, active=True)
    assert not window.loading_progress.isHidden()


def test_tracking_uses_metadata_including_empty_detection_results(monkeypatch):
    provider = _PanelFrameProvider({0: _frame()["sensing"]})
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    monkeypatch.setattr(service, "dashboard_products", unexpected)
    request = load_dashboard_request(service, 0, "tracking", "primary", False, "auto")
    assert request is not None
    assert request.rd_result is None and request.rp_result is None
    assert not request.display_frame["sensing"].get("detections")
    assert "range_doppler_power_by_pair" not in request.display_frame["sensing"]
    assert "range_doppler" in request.available_product_kinds
    assert request.effective_display_step == 0


def test_tracking_arrays_are_reused_across_seeks_without_duplicate_truth_updates(qapp, monkeypatch):
    history = replace(
        _history(),
        records=tuple(
            replace(
                row,
                detections=(
                    {
                        "track_id": 7,
                        "world_track_position": [row.time_s, 1.0, 2.0],
                        "range_m": 3.0,
                        "velocity_m_s": 1.0,
                    },
                ),
                truth={"target_ids": ["human"], "target_ranges": [4.0], "target_speeds": [None]},
                truth_positions=(("human", (row.time_s, 2.0, 3.0)),),
            )
            for row in _history().records
        ),
    )
    prepared = history.tracking_plot_data
    from visualizer.src.sensing import tracking_plot_data

    monkeypatch.setattr(tracking_plot_data, "prepare_tracking_plot_data", unexpected)
    window = SensingWindow()
    window.show()
    window._set_view_mode("tracking")
    monkeypatch.setattr(window, "_update_ground_truth", unexpected)
    try:
        for step in (5, 1, 0, 4, 5):
            request = replace(
                _request(step, history),
                rd_result=None,
                available_product_kinds=frozenset({"range_doppler"}),
            )
            window.update_from_request(request)
            assert history.tracking_plot_data is prepared
            np.testing.assert_allclose(
                window._tracking_curves["7"].xData, np.arange(step + 1) * 0.25
            )
            np.testing.assert_allclose(window._gt_range_curves["human"].yData, 4.0)
            assert np.isnan(window._gt_speed_curves["human"].yData).all()
            motion = window._view_mode_box.findData("micro-doppler")
            assert window._view_mode_box.model().item(motion).isEnabled()
        times, points = prepared.tracks["7"].through(0.5)
        assert np.shares_memory(points, prepared.tracks["7"].values)
        assert not times.flags.writeable and not points.flags.writeable
        assert "human" in window._target_history
        assert window._current_gt_range_speed() == ([], [])
        monkeypatch.undo()
        # Changing to an incremental view must not append into immutable slices.
        window._update_ground_truth(_frame(4))
        assert not window._history_from_recording
        assert isinstance(window._detection_history["times_s"], list)
    finally:
        window.close()


@pytest.fixture
def session(monkeypatch):
    provider = SimpleNamespace(frame_ids_snapshot=(0, 1), _generation_epoch=0)
    session = DashboardSession(provider)
    reads = []

    def load(service, *key):
        reads.append(key)
        return replace(_request(key[0]), prepared_views=frozenset({"analysis", key[1]}))

    monkeypatch.setattr(module, "load_dashboard_request", load)
    monkeypatch.setattr(session.service, "dashboard_result_rows", lambda: ())
    yield session, reads
    session.close()


@pytest.mark.parametrize(
    "view", ["tracking", "analysis", "angular", "diagnostics", "micro-doppler"]
)
def test_revisiting_dashboard_result_skips_reads_and_validation(session, monkeypatch, view):
    current, reads = session
    key = (0, view, "primary", False, "auto")
    first = current._load_request(key)
    current._load_request((1, *key[1:]))
    monkeypatch.setattr(module, "load_dashboard_request", unexpected)
    repeated = current._load_request(key)
    assert repeated.display_frame is first.display_frame
    assert repeated.rd_result is first.rd_result
    assert len(reads) == 2
    assert repeated.state_step == 0


def test_cache_invalidates_for_view_options_and_provider_changes(session):
    current, reads = session
    key = (0, "analysis", "primary", False, "auto")
    current._load_request(key)
    current._load_request((0, "analysis", "raw", False, "auto"))
    current.provider._generation_epoch += 1
    current._load_request(key)
    assert len(current._request_cache) == 1
    current.provider.frame_ids_snapshot = (0, 1, 2)
    current._load_request(key)
    assert len(reads) == 4
    assert len(current._request_cache) == 1


def test_sensing_playback_prefetches_bound_results_without_changing_navigation(monkeypatch):
    current = DashboardSession(_held_provider())
    key = (0, "analysis", "auto", True, "auto")
    try:
        first = current._load_request(key)
        navigation, context = current.navigation, current._navigation_context
        for _ in range(3):
            current.prefetch_next(key)
            if current._prefetch_pending is not None:
                current._prefetch_pending.result(timeout=5)
        assert current._prefetch_pending is None
        assert current.navigation is navigation
        assert current._navigation_context == context
        assert {k[0] for k in current._request_cache} == {0, 3, 6}
        monkeypatch.setattr(module, "load_dashboard_request", unexpected)
        for step in (3, 6, 0):
            request = current._load_request((step, *key[1:]))
            assert request.state_step == step
            assert request.effective_display_step == step
            assert request.replay_event.scene_binding.current_scene_frame_id == step
        assert current._load_request(key).display_data_token is first.display_data_token
    finally:
        current.close()


def test_prefetch_respects_foreground_priority_budget_and_cancellation(session, monkeypatch):
    from concurrent.futures import Future

    current, reads = session
    key = (0, "analysis", "primary", False, "auto")
    current._load_request(key)
    current.navigation = (SimpleNamespace(scene_id=0), SimpleNamespace(scene_id=1))
    current.pending = Future()
    current.prefetch_next(key)
    assert current._prefetch_pending is None
    current.pending = None
    current.cache_policy = replace(
        current.cache_policy, request_bytes=current._request_cache_bytes + 1
    )
    current.prefetch_next(key)
    current._prefetch_pending.result(timeout=5)
    current.prefetch_next(key)
    assert current._prefetch_pending is None
    assert tuple(current._request_cache) == (key,)
    assert len(reads) == 2  # Oversized speculation is tried once and never evicts.
    current.close()
    current.prefetch_next((1, "diagnostics", *key[2:]))
    assert current._prefetch_pending is None


def test_unknown_live_inventory_does_not_reuse_requests(session):
    current, reads = session
    del current.provider.frame_ids_snapshot
    key = (0, "analysis", "primary", False, "auto")
    current._load_request(key)
    current._load_request(key)
    assert len(reads) == 2 and not current._request_cache


def test_request_cache_respects_bytes_and_reloads_evicted_results(session, monkeypatch):
    current, reads = session
    budget = int(1.5 * estimate_retained_bytes(_request()))
    current.cache_policy = replace(current.cache_policy, request_bytes=budget)
    key = (0, "analysis", "primary", False, "auto")
    for step in (0, 1, 0):
        current._load_request((step, *key[1:]))
        assert current._request_cache_bytes <= budget
    assert len(reads) == 3 and len(current._request_cache) == 1


def test_cached_display_request_does_not_freeze_separately_saved_tracks(session, monkeypatch):
    current, reads = session
    saved = [SimpleNamespace(history=_history(), reader={"sources": {"rx": {}}}, sensor_id="rx")]
    monkeypatch.setattr(current.saved_tracking, "load", lambda _: (saved[0], None))
    key = (0, "tracking", "primary", False, "auto")
    old = current._load_request(key)
    saved[0] = SimpleNamespace(history=_history(), reader={"sources": {"rx": {}}}, sensor_id="rx")
    new = current._load_request(key)
    assert old.saved_tracking is not new.saved_tracking
    assert new.saved_tracking is saved[0] and len(reads) == 1


def test_request_cache_limits_entries_and_skips_oversized_payloads(session, monkeypatch):
    current, reads = session
    current.cache_policy = replace(current.cache_policy, request_entries=1)
    key = (0, "analysis", "primary", False, "auto")
    for step in (0, 1, 0):
        current._load_request((step, *key[1:]))
    assert len(reads) == 3 and len(current._request_cache) == 1
    current.cache_policy = replace(current.cache_policy, request_bytes=1)
    for _ in range(2):
        current._load_request((1, *key[1:]))
    assert len(reads) == 5 and (1, *key[1:]) not in current._request_cache


def test_run_summary_skips_frame_preparation_but_invalidates_changed_inventory(
    session, monkeypatch
):
    current, reads = session
    frame = _frame()
    monkeypatch.setattr(current.service, "dashboard_result_rows", lambda: ((0, frame),))
    monkeypatch.setattr(current.saved_tracking, "load", unexpected)
    request = current._load_request((0, "summary", "primary", True, "auto"))
    assert request.source_identity == current.source_identity
    assert current.can_reuse_summary(request)
    assert len(reads) == 1
    # New settings are handled by the aggregate worker, not by rereading scene poses.
    current.run_summary.key = ("different gates",)
    assert current.can_reuse_summary(request)
    current.provider.frame_ids_snapshot = (0, 1, 2)
    assert not current.can_reuse_summary(request)
    current._load_request((0, "summary", "primary", True, "auto"))
    assert current.can_reuse_summary(request)
    current.provider._generation_epoch += 1
    assert not current.can_reuse_summary(request)


def test_summary_with_multiple_cohorts_keeps_normal_selection(session, monkeypatch):
    from copy import deepcopy

    current, _ = session
    a, b = _frame(), deepcopy(_frame())
    b["sensing"]["temporal"]["acquisition"]["pair_id"] = "another-pair"
    monkeypatch.setattr(current.service, "dashboard_result_rows", lambda: ((0, a), (1, b)))
    request = current._load_request((0, "summary", "primary", True, "auto"))
    assert not current.can_reuse_summary(request)


def test_summary_playback_does_not_submit_per_frame_reads(session, monkeypatch, qtbot):
    from PySide6.QtWidgets import QWidget

    from visualizer.src.panels.sensing_panel import SensingPanel

    current, reads = session
    monkeypatch.setattr(current.service, "dashboard_result_rows", lambda: ((0, _frame()),))
    request = current._load_request((0, "summary", "primary", True, "auto"))
    panel = SensingPanel(SimpleNamespace())
    panel._container = QWidget()
    qtbot.addWidget(panel._container)
    panel._dashboard_session = current
    panel._sensing_window = SimpleNamespace(
        environment=SimpleNamespace(set_scene=lambda *a: None),
        request_view_mode=lambda: "summary",
        _last_request=request,
    )
    polls = []
    monkeypatch.setattr(panel, "_poll_dashboard", lambda: polls.append(True))
    monkeypatch.setattr(panel, "_update_result_navigation", lambda: None)
    monkeypatch.setattr(current, "request", unexpected)
    for step in (0, 1, 0, 100, 2):
        panel._refresh_dashboard_async(current.provider, step, force=True)
    assert len(reads) == 1 and len(polls) == 5
    panel._dashboard_poll_timer.stop()


def test_hosted_reopen_renders_the_latest_accepted_hidden_result(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window._hosted = True
    window.show()
    window._set_view_mode("analysis")
    first, latest = _request(0), _request(4)
    window.update_from_request(first)
    window.hide()
    window._live_catalog_requested = True
    window.update_from_request(latest)
    assert window._last_request is latest and window._rendered_request is None
    window.show()
    qtbot.waitUntil(lambda: window._rendered_request is latest)
    assert "4" in window.provenance_label.text()
