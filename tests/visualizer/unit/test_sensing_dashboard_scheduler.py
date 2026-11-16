"""Dashboard completion polling respects pending work and widget lifetime."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from tests.visualizer.unit.test_sensing_dashboard_recording import _frame, _history, _request
from visualizer.src.panels import sensing_panel
from visualizer.src.sensing import sensing_window
from visualizer.src.sensing.dashboard_data import DashboardSession


def _warmup_provider():
    from tests.visualizer.unit.test_sensing_preparation import _provider

    provider = _provider()
    provider.frames[1] = replace(provider.frames[1], sensing=None)
    for storage, bound in ((4, 3), (7, 6)):
        sensing = _frame(bound)["sensing"]
        sensing["detections"] = provider.frames[storage].sensing["detections"]
        provider.frames[storage] = replace(provider.frames[storage], sensing=sensing)
    return provider


def test_motion_overview_before_first_result_preserves_scene_and_current_result(qtbot, monkeypatch):
    """Exercise normal asynchronous panel delivery across a warm-up boundary."""
    provider = _warmup_provider()
    parent = SimpleNamespace(
        animation_running=False,
        animation_step=0,
        get_available_animation_steps=lambda: list(range(9)),
    )
    panel = sensing_panel.SensingPanel(parent)
    panel._container = QWidget()
    qtbot.addWidget(panel._container)
    window = panel._sensing_window = sensing_window.SensingWindow()
    window.show()
    monkeypatch.setattr(sensing_panel, "visualizer_frame_provider", lambda _: provider)
    monkeypatch.setattr(panel, "_dashboard_is_visible", lambda: True)
    monkeypatch.setattr(panel, "_set_status", lambda *_: None)
    monkeypatch.setattr(panel, "_update_detection_label", lambda *a, **kw: None)
    session = panel._dashboard_session = DashboardSession(provider)
    panel._dashboard_poll_timer = QTimer(panel._container)
    panel._dashboard_poll_timer.setInterval(8)
    panel._dashboard_poll_timer.timeout.connect(panel._poll_dashboard)

    def show(mode, step):
        parent.animation_step = step
        window._set_view_mode(mode)
        session.request((step, mode, window.product_variant(), False, "auto"))
        panel._poll_dashboard()
        qtbot.waitUntil(
            lambda: session.pending is None
            and session.history_pending is None
            and session.overview_pending is None,
            timeout=10000,
        )

    try:
        show("micro-doppler", 3)
        gate_enabled = {
            mode: window._micro_mode_box.model()
            .item(window._micro_mode_box.findData(mode))
            .isEnabled()
            for mode in ("tracked", "gated", "fixed-bin")
        }
        show("micro-doppler", 0)
        assert parent.animation_step == 0
        assert window._last_request.recording_overview_only
        assert window._last_request.state_step == 0
        assert window._last_request.effective_display_step == 3
        assert window.next_result_button.isEnabled()
        assert window.navigation_label.text() == "2 results"
        assert session.history is None
        assert session.overview.families["summed-rd"].coverage.loaded_results == 2
        assert window.collapsed_doppler_mesh.isVisible()  # Explicit gaps between the saved CPIs.
        assert "Doppler over time" in window.collapsed_doppler_plot.plotItem.titleLabel.text
        assert "no sensing result at scene 0" in window.provenance_label.text()
        assert not window.collapsed_doppler_cursor.isVisible()
        assert not window.can_move_recording_cursor(session.history)
        assert not window.result_scroll.isVisible()
        assert not window.detection_stage_export_button.isEnabled()
        assert all(
            not window._micro_mode_box.model()
            .item(window._micro_mode_box.findData(mode))
            .isEnabled()
            for mode in gate_enabled
        )

        show("analysis", 0)
        assert window._last_request is None
        assert window._unavailable_label.isVisible()
        assert not window._plots_widget.isVisible()

        show("micro-doppler", 3)
        assert not window._last_request.recording_overview_only
        assert window._last_request.state_step == 3
        assert window.collapsed_doppler_cursor.isVisible()
        assert window.result_scroll.isVisible()
        assert {
            mode: window._micro_mode_box.model()
            .item(window._micro_mode_box.findData(mode))
            .isEnabled()
            for mode in gate_enabled
        } == gate_enabled
    finally:
        panel.cleanup()


@pytest.mark.parametrize("view", ("micro-doppler", "summary"))
def test_recording_overview_does_not_choose_between_multiple_recording_cohorts(monkeypatch, view):
    from tests.visualizer.fixtures import sensing_v3

    provider = _warmup_provider()
    other_run = "e00c0940-3e3e-4431-a161-1c1b3bc30af9"
    monkeypatch.setattr(sensing_v3, "_PROCESSING_RUN_ID", other_run)
    other = _frame(6)["sensing"]
    provider.frames[7] = replace(provider.frames[7], sensing=other)
    session = DashboardSession(provider)
    try:
        assert session._load_request((0, view, "primary", False, "auto")) is None
    finally:
        session.close()


@pytest.mark.parametrize("results_only", (False, True))
def test_summary_overview_before_first_result_is_metadata_only(qtbot, monkeypatch, results_only):
    """A real summary worker starts without presenting a future result as current."""
    from shared.extensions.sensing import assert_sensing_contract
    from tests.shared.test_sensing_contract import _incomplete_sensing
    from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
    from visualizer.src.services.sensing_service import SensingService

    provider = _warmup_provider()
    incomplete = _incomplete_sensing()
    assert_sensing_contract({"sensing": incomplete}, require_schema_version=True)
    provider.frames[0] = replace(provider.frames[0], sensing=incomplete)
    if results_only:
        for storage, bound in ((4, 3), (7, 6)):
            original = provider.frames[storage].sensing
            sensing = build_visualizer_sensing_payload(
                {
                    "config_resolved": original["config_resolved"],
                    "processing_strategy": original["processing_strategy"],
                    "detections": original["detections"],
                },
                scene_frame_id=bound,
                sample_time_s=bound * 0.25,
            )
            assert not sensing.get("products")
            provider.frames[storage] = replace(provider.frames[storage], sensing=sensing)
    parent = SimpleNamespace(
        animation_running=False,
        animation_step=0,
        get_available_animation_steps=lambda: list(range(9)),
    )
    panel = sensing_panel.SensingPanel(parent)
    panel._container = QWidget()
    qtbot.addWidget(panel._container)
    window = panel._sensing_window = sensing_window.SensingWindow()
    window.show()
    window._set_view_mode("summary")
    monkeypatch.setattr(sensing_panel, "visualizer_frame_provider", lambda _: provider)
    monkeypatch.setattr(panel, "_dashboard_is_visible", lambda: True)
    monkeypatch.setattr(panel, "_set_status", lambda *_: None)
    monkeypatch.setattr(panel, "_update_detection_label", lambda *a, **kw: None)
    session = panel._dashboard_session = DashboardSession(provider)
    panel._dashboard_poll_timer = QTimer(panel._container)
    panel._dashboard_poll_timer.setInterval(8)
    panel._dashboard_poll_timer.timeout.connect(panel._poll_dashboard)
    try:
        session.request((0, "summary", "primary", True, "auto"))
        panel._poll_dashboard()
        qtbot.waitUntil(lambda: session.pending is None, timeout=5000)
        assert window._last_request is not None
        request = window._last_request
        assert request.recording_overview_only
        assert request.state_step == parent.animation_step == 0
        assert SensingService._scene_time_s(request.state_frame) == 0.0
        assert request.effective_display_step == 3
        assert (
            request.display_frame["sensing"]["temporal"]["display_binding"]["scene_frame_id"] == 3
        )
        assert request.rp_result is None and request.rd_result is None
        assert not window.result_scroll.isVisible()
        assert not window.detection_stage_export_button.isEnabled()
        assert "no sensing result at scene 0" in window.provenance_label.text()
        qtbot.waitUntil(lambda: session.run_summary.data is not None, timeout=5000)
        assert session.run_summary.data.acquisition_count == 2
        rows = session.service.dashboard_result_rows()
        assert len(rows) == 3
        assert rows[0][1]["sensing"]["temporal"]["display_binding"]["scene_frame_id"] is None
        assert session.can_reuse_summary(request)
        assert window.run_summary_panel.isVisible()
        assert session.history is None and session.history_pending is None
        assert all(not read.all_sensing_products for _, read in provider.requests)
        assert all(
            read.sensing_products <= {"products", "config_resolved"}
            for _, read in provider.requests
        )
        assert session._load_request((0, "analysis", "primary", False, "auto")) is None
        assert session._load_request((0, "current", "primary", False, "auto")) is None
    finally:
        panel.cleanup()


def test_paused_motion_waits_for_spectral_family_and_timer_publishes_it(state, qtbot, monkeypatch):
    """Compact spectra arrive independently of rich detail; Play is unnecessary."""
    from threading import Event

    from tests.visualizer.unit.test_sensing_motion_overview import MemoryRecording
    from visualizer.src.sensing.dashboard_data import result_context
    from visualizer.src.sensing.motion_overview import load_motion_overview

    panel, session, window = state.panel, state.session, state.window
    state.parent.animation_running = False
    window.show()
    window._set_view_mode("micro-doppler")
    monkeypatch.setattr(
        window,
        "update_from_request",
        sensing_window.SensingWindow.update_from_request.__get__(window),
    )
    request = _request(64)
    window.update_from_request(request)
    full = Future()
    monkeypatch.setattr(session._executor, "submit", lambda *a, **kw: full)
    source = MemoryRecording([_frame(i) for i in range(6)])
    overview = load_motion_overview(
        source,
        result_context(request.display_frame["sensing"]),
        request.variant,
        window.overview_families(),
        Event(),
    )
    panel._poll_dashboard()
    qtbot.waitUntil(lambda: window.loading_progress.isVisible(), timeout=2000)
    assert window.collapsed_doppler_image.image is None
    assert (
        window.collapsed_doppler_plot.plotItem.titleLabel.text == "Loading saved spectral history…"
    )

    assert session.overview_pending is full
    assert window._last_request.recording_history is None
    assert window.collapsed_doppler_image.image is None
    assert panel._dashboard_poll_timer.isActive()

    full.set_result(overview)
    qtbot.waitUntil(lambda: window.collapsed_doppler_image.image is not None)
    assert window.collapsed_doppler_image.image.shape[0] == 6
    assert not state.parent.animation_running and state.parent.animation_step == 64
    assert not panel._dashboard_poll_timer.isActive()


@pytest.fixture
def state(qtbot, monkeypatch):
    provider = SimpleNamespace(_generation_epoch=0)
    provider_now = [provider]
    parent = SimpleNamespace(
        animation_running=True,
        animation_step=64,
        get_available_animation_steps=lambda: list(range(193)),
    )
    panel = sensing_panel.SensingPanel(parent)
    panel._container = QWidget()
    qtbot.addWidget(panel._container)
    panel._sensing_window = window = sensing_window.SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("analysis")
    visible = [True]
    monkeypatch.setattr(panel, "_dashboard_is_visible", lambda: visible[0])
    monkeypatch.setattr(sensing_panel, "visualizer_frame_provider", lambda _: provider_now[0])
    session = panel._dashboard_session = DashboardSession(provider)
    monkeypatch.setattr(session, "request_history", lambda *_: None)
    monkeypatch.setattr(session.service, "dashboard_inventory_token", lambda: ("inventory",))
    monkeypatch.setattr(window, "wants_recording_history", lambda: False)
    monkeypatch.setattr(panel, "_update_result_navigation", lambda: None)
    unavailable, installed = ([], [])
    monkeypatch.setattr(panel, "_set_unavailable", lambda *a, **kw: unavailable.append((a, kw)))
    monkeypatch.setattr(window, "advance_playback_context", lambda _: False)

    def install(request):
        installed.append(request)
        window._last_request = request

    monkeypatch.setattr(window, "update_from_request", install)
    panel._dashboard_poll_timer = QTimer(panel._container)
    panel._dashboard_poll_timer.setInterval(8)
    panel._dashboard_poll_timer.timeout.connect(panel._poll_dashboard)
    panel._start_auto_refresh()
    yield SimpleNamespace(
        panel=panel,
        window=window,
        session=session,
        parent=parent,
        visible=visible,
        provider=provider,
        provider_now=provider_now,
        installed=installed,
        unavailable=unavailable,
    )
    panel.cleanup()


@pytest.mark.parametrize("work", ["result", "history", "summary"])
def test_pending_work_keeps_polling_until_completion_then_stops(state, monkeypatch, work):
    panel, session, window = (state.panel, state.session, state.window)
    future = Future()
    if work == "result":
        session.pending, session.pending_key = (future, (64, "analysis", "primary", True, "auto"))
        result = _request(64)
    elif work == "history":
        session.history_pending = future
        result = _history()
    else:
        monkeypatch.setattr(window, "request_view_mode", lambda: "summary")
        window._last_request = _request(64)
        summary = session.run_summary
        summary._future = future
        summary._pending_key = summary.key = ("summary",)
        monkeypatch.setattr(summary, "request", lambda *_: None)
        monkeypatch.setattr(window.run_summary_panel, "settings", lambda: ())
        for name in ("clear", "set_summary", "set_loading"):
            monkeypatch.setattr(window.run_summary_panel, name, lambda _: None)
        result = SimpleNamespace(cancelled=False, error=None)
    panel._poll_dashboard()
    assert panel._dashboard_poll_timer.isActive()
    future.set_result(result)
    panel._poll_dashboard()
    assert not panel._dashboard_poll_timer.isActive()
    assert window.loading_progress.isHidden()
    if work == "result":
        assert state.installed == [result]
    elif work == "history":
        assert session.history is result
    else:
        assert session.run_summary.data is result


def test_idle_poll_does_not_wait_for_speculative_prefetch(state):
    state.session._prefetch_pending = Future()
    state.panel._dashboard_poll_timer.start()
    state.panel._poll_dashboard()
    assert not state.panel._dashboard_poll_timer.isActive()


@pytest.mark.parametrize("work", ["result", "history"])
def test_completed_index_progress_is_not_reused_for_another_load(state, work):
    session, window = state.session, state.window
    session.progress = ("Indexing results", 80, 80)
    if work == "result":
        session.pending = Future()
    else:
        session.history_pending = Future()
    state.panel._poll_dashboard()
    assert window.loading_progress.format() == (
        "Loading result" if work == "result" else "Loading overview"
    )
    assert window.loading_progress.maximum() == 0


def test_actual_index_progress_remains_visible(state):
    state.session.progress = ("Indexing results", 20, 80)
    state.session.pending = Future()
    state.panel._poll_dashboard()
    assert state.window.loading_progress.format() == "Indexing results · %p%"
    assert state.window.loading_progress.value() == 20
    assert state.window.loading_progress.maximum() == 80


@pytest.mark.parametrize("products", [("range_doppler_angle",), ()])
def test_prefetch_matches_foreground_live_figure_product_demand(state, monkeypatch, products):
    foreground, prefetched = [], []
    state.parent.sensing_playback = SimpleNamespace(steps=(64, 128))
    monkeypatch.setattr(state.window, "live_figure_request_suffix", lambda: (products,))
    monkeypatch.setattr(state.session, "request", foreground.append)
    monkeypatch.setattr(
        state.session, "prefetch_next", lambda key, **_options: prefetched.append(key)
    )
    state.panel._refresh_dashboard_async(state.provider, 64, force=False)
    assert foreground == prefetched == [(64, "analysis", "primary", True, "auto", products)]


def test_hide_and_resume_cursor_fast_path_rearms_residual_work(state, monkeypatch):
    panel, session, window = (state.panel, state.session, state.window)
    session.history_pending = Future()
    panel._poll_dashboard()
    assert panel._dashboard_poll_timer.isActive()
    state.visible[0] = False
    panel._poll_dashboard()
    assert not panel._dashboard_poll_timer.isActive()
    state.visible[0] = True
    session.history_key = ("clock", "run", "pair", "primary")
    monkeypatch.setattr(window, "can_move_recording_cursor", lambda _: True)
    cursor = SimpleNamespace(time_s=2.0, detections=())
    monkeypatch.setattr(session, "motion_cursor", lambda *a, **kw: cursor)
    moved = []
    monkeypatch.setattr(window, "move_recording_cursor", lambda *a: moved.append(a))
    panel._refresh_dashboard_async(state.provider, 64, force=False)
    assert moved == [(cursor, "primary")]
    assert panel._dashboard_poll_timer.isActive()
    session.history_pending.set_result(_history())
    panel._poll_dashboard()
    assert not panel._dashboard_poll_timer.isActive()


@pytest.mark.parametrize("retirement", ["source", "close"])
def test_retired_work_never_restarts_timer_or_updates_window(state, retirement):
    future = Future()
    future.set_running_or_notify_cancel()
    state.session.pending = future
    state.session.pending_key = (64, "analysis", "primary", True, "auto")
    timer = state.panel._dashboard_poll_timer
    state.panel._poll_dashboard()
    assert timer.isActive()
    if retirement == "source":
        state.provider_now[0] = SimpleNamespace()
        state.panel._poll_dashboard()
    else:
        state.panel.cleanup()
    assert not isValid(timer) or not timer.isActive()
    assert state.panel._dashboard_session is None
    future.set_result(_request(64))
    state.panel._poll_dashboard()
    assert state.installed == []


def test_refresh_coalesces_and_force_requests_without_waiting_for_periodic_timer(
    state, monkeypatch
):
    panel, session = (state.panel, state.session)
    submitted = []
    monkeypatch.setattr(
        session._executor, "submit", lambda *args: submitted.append(args) or Future()
    )
    key = (64, "analysis", "primary", True, "auto")
    panel._completed_dashboard_key = key
    panel._refresh_dashboard_async(state.provider, 64, force=False)
    assert submitted == []
    panel._refresh_dashboard_async(state.provider, 64, force=True)
    assert len(submitted) == 1
    assert panel._refresh_timer.interval() == 67
    for step in (128, 192):
        state.parent.animation_step = step
        panel._refresh_dashboard_async(state.provider, step, force=False)
    assert len(submitted) == 1
    session.pending = None
    panel._refresh_dashboard_async(state.provider, 192, force=False)
    assert len(submitted) == 2 and session.pending_key[0] == 192
    state.parent.animation_running = False
    panel._refresh_dashboard_async(state.provider, 192, force=True)
    assert panel._refresh_timer.interval() == 33


@pytest.mark.parametrize(
    "playing,current,options_match,accepted",
    [
        (True, 65, True, True),
        (True, 5, True, False),
        (False, 65, True, False),
        (True, 65, False, False),
    ],
)
def test_completed_results_keep_exact_binding_and_reject_stale_seeks(
    state, playing, current, options_match, accepted
):
    state.parent.animation_running = playing
    state.parent.animation_step = current
    request = _request(63)
    state.session.pending = Future()
    variant = "primary" if options_match else "raw"
    state.session.pending_key = (63, "analysis", variant, True, "auto")
    state.session.pending.set_result(request)
    state.panel._poll_dashboard()
    assert state.installed == ([request] if accepted else [])
    if accepted:
        assert state.window._last_request.effective_display_step == 63
        assert state.window._last_request.state_step == 63
        assert state.parent.animation_step == 65
    assert not state.panel._dashboard_poll_timer.isActive()


def test_progress_dedup_preserves_debounce_reset_and_cancel_resume(state, monkeypatch):
    window = state.window
    now = [10.0]
    monkeypatch.setattr(sensing_window, "monotonic", lambda: now[0])
    window._last_request = _request(64)
    window.set_loading_progress("Working", 3, 10, active=True)
    assert window.loading_progress.isHidden()
    now[0] += 0.2
    window.set_loading_progress("Working", 3, 10, active=True)
    assert not window.loading_progress.isHidden()
    calls = []
    original = window.loading_progress.setVisible
    monkeypatch.setattr(
        window.loading_progress, "setVisible", lambda *args: calls.append(args) or original(*args)
    )
    window.set_loading_progress("Working", 3, 10, active=True)
    assert calls == []
    window.loading_progress.hide()
    window.set_loading_progress("Working", 3, 10, active=True)
    assert not window.loading_progress.isHidden()
    window.set_loading_progress("Cancelled", 0, 0, active=False, cancelled=True)
    assert window.loading_progress.isHidden()
    assert not window.cancel_loading_button.isHidden()
    assert window.cancel_loading_button.text() == "Resume loading"
    window.reset_recording()
    window.set_loading_progress("Working", 3, 10, active=True)
    assert not window.loading_progress.isHidden()
    assert window.cancel_loading_button.text() == "Cancel loading"


def test_cancel_and_resume_use_the_existing_explicit_refresh_path(state, monkeypatch):
    panel = state.panel
    state.session.pending = Future()
    panel._poll_dashboard()
    assert panel._dashboard_poll_timer.isActive()
    panel._toggle_dashboard_loading()
    assert panel._dashboard_session is None
    assert not panel._dashboard_poll_timer.isActive()
    assert state.window.cancel_loading_button.text() == "Resume loading"
    refreshed = []
    monkeypatch.setattr(panel, "_refresh_live_plots", lambda **kwargs: refreshed.append(kwargs))
    panel._toggle_dashboard_loading()
    assert panel._loading_paused_identity is None
    assert refreshed == [{"force": True}]


def test_inactive_summary_finishes_without_polling_and_is_reused_on_return(state, monkeypatch):
    panel, session, window = state.panel, state.session, state.window
    mode = ["summary"]
    monkeypatch.setattr(window, "request_view_mode", lambda: mode[0])
    window._last_request = _request(64)
    summary = session.run_summary
    future = summary._future = Future()
    summary._pending_key = summary.key = ("summary",)
    monkeypatch.setattr(summary, "request", lambda *_: None)
    monkeypatch.setattr(session, "can_reuse_summary", lambda _: True)
    monkeypatch.setattr(window.run_summary_panel, "settings", lambda: ())
    calls = []
    for name in ("clear", "set_summary", "set_loading"):
        monkeypatch.setattr(
            window.run_summary_panel, name, lambda value, name=name: calls.append((name, value))
        )
    panel._poll_dashboard()
    assert panel._dashboard_poll_timer.isActive()
    mode[0] = "analysis"
    calls.clear()
    panel._poll_dashboard()
    assert not panel._dashboard_poll_timer.isActive()
    assert window.loading_progress.isHidden()
    assert calls == [] and state.installed == []
    # Other foreground work can still keep polling while Summary is inactive.
    session.history_pending = Future()
    panel._poll_dashboard()
    assert panel._dashboard_poll_timer.isActive()
    session.history_pending.set_result(_history())
    panel._poll_dashboard()
    assert not panel._dashboard_poll_timer.isActive()
    data = SimpleNamespace(cancelled=False, error=None)
    future.set_result(data)
    panel._poll_dashboard()
    assert summary._future is future and summary.data is None
    assert calls == [] and state.installed == []
    mode[0] = "summary"
    panel._refresh_dashboard_async(state.provider, 64, force=True)
    assert summary.data is data and summary._future is None
    assert calls == [("set_summary", data)]
    assert not panel._dashboard_poll_timer.isActive()
    assert window.loading_progress.isHidden()


@pytest.mark.parametrize("playing", [True, False])
@pytest.mark.parametrize(
    "mode",
    ["current", "analysis", "tracking", "summary", "micro-doppler", "diagnostics", "angular"],
)
def test_refresh_cadence_reserves_scene_time_only_during_play(state, monkeypatch, mode, playing):
    state.parent.animation_running = playing
    state.window._set_view_mode(mode)
    submitted = []
    monkeypatch.setattr(
        state.session._executor, "submit", lambda *args: submitted.append(args) or Future()
    )
    state.panel._refresh_dashboard_async(state.provider, 64, force=True)
    expected = 67 if playing else 33
    assert state.panel._refresh_timer.interval() == expected
    assert state.panel._dashboard_poll_timer.interval() == 8
    assert len(submitted) == 1  # force submits now, without waiting for the timer


@pytest.mark.parametrize("mode", ["current", "analysis"])
def test_expanded_current_inspection_keeps_current_cadence_and_exact_request_key(
    state, monkeypatch, mode
):
    state.window._set_view_mode(mode)
    request = _request(64)
    request.display_frame["sensing"]["detections"] = [{"range_m": 1.0, "power_linear": 1.0}]
    sensing_window.SensingWindow.update_from_request(state.window, request)
    state.window.select_detection_index(0)
    state.window.selection_tools_toggle.setChecked(True)
    assert state.window.request_view_mode() == "analysis-inspection"
    submitted = []
    monkeypatch.setattr(
        state.session._executor, "submit", lambda *args: submitted.append(args) or Future()
    )
    state.panel._refresh_dashboard_async(state.provider, 64, force=True)
    assert state.panel._refresh_timer.interval() == 67
    assert state.session.pending_key[0:2] == (64, "analysis-inspection")
    assert len(submitted) == 1
