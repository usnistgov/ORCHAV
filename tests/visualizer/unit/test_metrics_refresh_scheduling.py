"""Bounded adaptive refresh scheduling with real Qt timers and widgets."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt

pytest.importorskip("pyqtgraph")

from visualizer.src.metrics.viz_metrics import MetricsDashboard, MetricsWindow


@pytest.fixture
def dashboard(qtbot):
    window = MetricsWindow()
    qtbot.addWidget(window)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    window._refresh = lambda: None
    yield window
    window.close()


def _offer(window, step):
    window.enqueue(SimpleNamespace(canonical_data=None), context={"step": step})


@pytest.mark.parametrize("window_class", [MetricsWindow, MetricsDashboard])
def test_automatic_is_default_for_both_hosts(qtbot, window_class):
    window = window_class()
    qtbot.addWidget(window)
    assert window.update_rate_spin.value() == 0
    assert window.update_rate_spin.text() == "Automatic"
    assert window._fixed_refresh_period_ms() is None
    assert not window._timer.isActive()
    assert not window._ui_probe_timer.isActive()
    window.close()


def test_timer_lateness_increases_cooldown_and_recovers(dashboard):
    clock = [10.0]
    dashboard._clock = lambda: clock[0]
    _offer(dashboard, 0)
    clock[0] = 10.102
    dashboard._periodic_refresh()

    assert dashboard._ui_delay_ms == pytest.approx(100.0)
    assert dashboard._adaptive_cooldown_ms() >= 100
    for _ in range(20):
        dashboard._record_ui_delay(0.0, now_s=clock[0])
    assert dashboard._adaptive_cooldown_ms() == 16


def test_long_event_loop_stall_has_bounded_feedback(dashboard):
    dashboard._record_ui_delay(60_000.0, now_s=100.0)
    assert dashboard._ui_delay_ms == 250.0
    assert dashboard._adaptive_cooldown_ms() == 250


def test_small_timer_jitter_does_not_add_load(dashboard):
    dashboard._record_ui_delay(1.5, now_s=10.0)
    assert dashboard._ui_delay_ms == 0.0
    assert dashboard._ui_busy_until_s is None


def test_deferred_ui_work_reserves_time_after_the_busy_interval(dashboard):
    clock = [10.0]
    dashboard._clock = lambda: clock[0]
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    assert dashboard._ui_probe_timer.isSingleShot()
    assert dashboard._ui_probe_timer.isActive()
    _offer(dashboard, 1)

    # Refresh returned immediately, but deferred UI work held the loop until
    # 118 ms. The probe was due after 16 ms, with 2 ms of jitter tolerance.
    clock[0] = 10.118
    dashboard._measure_ui_delay()

    assert dashboard._ui_delay_ms == pytest.approx(100.0)
    assert dashboard._timer.isActive()
    assert dashboard._timer.interval() in (100, 101)
    assert dashboard._pending_refresh
    assert not dashboard._ui_probe_timer.isActive()


def test_refresh_cost_recovers_gradually(dashboard):
    clock = [10.0]
    dashboard._clock = lambda: clock[0]

    def expensive_refresh():
        clock[0] += 0.100

    dashboard._refresh = expensive_refresh
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    dashboard._refresh = lambda: None
    dashboard._manual_refresh()

    assert dashboard._last_refresh_duration_ms == 0.0
    assert dashboard._smoothed_refresh_duration_ms == pytest.approx(80.0)
    assert dashboard._adaptive_cooldown_ms() in (80, 81)


def test_fixed_rate_is_still_an_upper_limit_under_load(dashboard):
    dashboard.update_rate_spin.setValue(8)
    dashboard._last_refresh_started_s = 10.0
    dashboard._last_refresh_finished_s = 10.004
    assert dashboard._next_refresh_delay_ms(now_s=10.004) >= 120
    dashboard._record_ui_delay(202.0, now_s=10.050)
    assert dashboard._next_refresh_delay_ms(now_s=10.050) >= 200


def test_burst_replaces_input_without_restarting_deadline(dashboard):
    clock = [10.0]
    dashboard._clock = lambda: clock[0]
    dashboard.update_rate_spin.setValue(8)
    _offer(dashboard, 0)
    due_s = dashboard._refresh_timer_due_s
    for step in range(1, 100):
        clock[0] += 0.0001
        _offer(dashboard, step)
    assert dashboard._refresh_timer_due_s == due_s
    assert dashboard._last_context["step"] == 99
    assert dashboard._pending_refresh


def test_reentrant_refresh_keeps_only_the_latest_pending_input(dashboard):
    calls = []

    def refresh():
        calls.append(dashboard._last_context["step"])
        if len(calls) == 1:
            _offer(dashboard, 1)
            _offer(dashboard, 2)
            dashboard._periodic_refresh()
            dashboard._manual_refresh()

    dashboard._refresh = refresh
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    assert calls == [0]
    assert dashboard._pending_refresh
    assert dashboard._timer.isActive()
    dashboard._periodic_refresh()
    assert calls == [0, 2]
    assert not dashboard._pending_refresh


def test_freeze_cancels_both_timers_and_manual_refresh_stays_available(dashboard):
    calls = []
    dashboard._refresh = lambda: calls.append(dashboard._last_context["step"])
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    _offer(dashboard, 1)
    dashboard.pause_cb.setChecked(True)
    _offer(dashboard, 2)
    assert not dashboard._timer.isActive()
    assert not dashboard._ui_probe_timer.isActive()
    assert dashboard._ui_probe_due_s is None
    dashboard._measure_ui_delay()
    dashboard._periodic_refresh()
    assert calls == [0]
    dashboard._manual_refresh()
    assert calls == [0, 2]
    assert not dashboard._ui_probe_timer.isActive()
    dashboard.pause_cb.setChecked(False)
    assert dashboard._timer.isActive()


def test_hide_cancels_timers_until_show_and_keeps_latest(dashboard):
    dashboard.show()
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    _offer(dashboard, 1)
    dashboard.hide()
    _offer(dashboard, 2)
    dashboard._measure_ui_delay()
    dashboard._periodic_refresh()
    assert dashboard._refresh_suspended
    assert not dashboard._timer.isActive()
    assert not dashboard._ui_probe_timer.isActive()
    assert dashboard._pending_refresh
    dashboard.show()
    assert dashboard._timer.isActive()
    assert dashboard._last_context["step"] == 2


def test_close_cancels_timers_and_ignores_late_input(dashboard):
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    _offer(dashboard, 1)
    dashboard.close()
    _offer(dashboard, 2)
    dashboard._measure_ui_delay()
    dashboard._periodic_refresh()
    dashboard._manual_refresh()
    assert not dashboard._timer.isActive()
    assert not dashboard._ui_probe_timer.isActive()
    assert not dashboard._pending_refresh
    assert dashboard._last_vm is None


@pytest.mark.parametrize("action", ["hide", "close", "pause"])
def test_lifecycle_change_during_refresh_cannot_rearm_work(dashboard, action):
    dashboard.show()

    def refresh():
        _offer(dashboard, 1)
        if action == "pause":
            dashboard.pause_cb.setChecked(True)
        else:
            getattr(dashboard, action)()

    dashboard._refresh = refresh
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    assert not dashboard._timer.isActive()
    assert not dashboard._ui_probe_timer.isActive()
    assert dashboard._ui_probe_due_s is None
    if action != "close":
        assert dashboard._last_context["step"] == 1


@pytest.mark.parametrize("action", ["hide", "close", "pause"])
def test_lifecycle_change_inside_provider_discards_its_result(dashboard, action):
    dashboard.show()

    def provide(_view_model):
        if action == "pause":
            dashboard.pause_cb.setChecked(True)
        else:
            getattr(dashboard, action)()
        return SimpleNamespace(total_paths=99)

    dashboard._frame_stats_provider = provide
    _offer(dashboard, 0)
    assert dashboard._compute_latest_frame_stats() is False
    assert dashboard._last_stats is None
    assert dashboard._stats_revision != dashboard._enqueue_revision


def test_show_retries_an_uncomputed_frame_interrupted_by_hide(dashboard):
    dashboard.show()
    calls = []

    def provide(_view_model):
        calls.append(True)
        if len(calls) == 1:
            dashboard.hide()
        return SimpleNamespace(total_paths=99)

    dashboard._frame_stats_provider = provide
    dashboard._refresh = dashboard._compute_latest_frame_stats
    _offer(dashboard, 0)
    dashboard._periodic_refresh()
    assert dashboard._pending_refresh
    assert dashboard._last_stats is None
    dashboard.show()
    assert dashboard._timer.isActive()
    dashboard._periodic_refresh()
    assert calls == [True, True]
    assert dashboard._last_stats.total_paths == 99
    assert not dashboard._pending_refresh


@pytest.mark.parametrize("action", ["hide", "close"])
def test_manual_input_request_cannot_refresh_after_dismissal(dashboard, action):
    dashboard.show()
    calls = []
    dashboard._refresh = lambda: calls.append(True)
    dashboard.input_requested.connect(getattr(dashboard, action))
    _offer(dashboard, 0)
    dashboard._manual_refresh()
    assert calls == []
    assert not dashboard._timer.isActive()
    assert not dashboard._ui_probe_timer.isActive()
    if action == "hide":
        assert dashboard._pending_refresh


def test_real_event_loop_consumes_one_probe_then_stays_idle(dashboard, qtbot):
    dashboard.show()
    calls = []
    dashboard._refresh = lambda: calls.append(dashboard._last_context["step"])
    for step in range(25):
        _offer(dashboard, step)
    qtbot.waitUntil(lambda: calls == [24], timeout=2000)
    qtbot.waitUntil(lambda: not dashboard._ui_probe_timer.isActive(), timeout=2000)
    assert not dashboard._timer.isActive()
    qtbot.wait(50)
    assert calls == [24]
    assert not dashboard._ui_probe_timer.isActive()
