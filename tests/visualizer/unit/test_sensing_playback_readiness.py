"""Early transport input stays paused until the sensing timeline is installed."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QLabel, QPushButton
from shiboken6 import delete

from tests.visualizer.unit.test_animation_scrubbing import _PlaybackControls, _Timer
from tests.visualizer.unit.test_sensing_playback import _playback
from visualizer.src.controllers.animation_controller import AnimationController
from visualizer.src.controllers.telemetry_controller import TelemetryController
from visualizer.src.playback import PlaybackCadenceTracker


def _transport(qapp):
    plan, provider, updates, counts, preloads, messages = _playback(qapp)
    viz = plan.visualizer
    viz.ready = True
    viz.play_direction = 1
    viz.animation_timer = _Timer()
    viz.update_timer = _Timer()
    viz._active_poll_interval_ms = viz._idle_poll_interval_ms = 16
    viz.playback_cadence = PlaybackCadenceTracker()
    viz.status_fps_label = QLabel()
    telemetry = TelemetryController(SimpleNamespace(visualizer=viz))
    viz.ui_controller = SimpleNamespace(refresh_status_telemetry=telemetry.refresh_status_telemetry)
    controller = AnimationController(
        viz, SimpleNamespace(cache_service=None), playback_controls=_PlaybackControls()
    )
    viz.animation_controller = controller
    viz.play_btn = QPushButton()
    viz.reverse_play_btn = QPushButton()
    for button in (viz.play_btn, viz.reverse_play_btn):
        button.setCheckable(True)
    viz.play_btn.clicked.connect(lambda: controller.toggle_animation())
    viz.reverse_play_btn.clicked.connect(controller.play_backward)
    return plan, provider, controller, updates, counts, preloads, messages


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("worker_finished", [False, True])
def test_early_play_waits_for_gui_delivery_and_requires_another_click(
    qapp, reverse, worker_finished
):
    plan, provider, controller, updates, _counts, _preloads, messages = _transport(qapp)
    viz = plan.visualizer
    future = Future()
    try:
        plan.prepare_for_scenario(provider, future)
        if worker_finished:
            future.set_result(((0, 7, 23), {0: 0.0, 7: 0.064, 23: 0.201}))
        button = viz.reverse_play_btn if reverse else viz.play_btn
        button.click()

        assert plan.preparing
        assert not viz.animation_running
        assert not viz.animation_timer.isActive()
        assert not viz.animation_timer.start_intervals
        assert not viz.update_timer.start_intervals
        assert not viz.play_btn.isChecked() and not viz.reverse_play_btn.isChecked()
        assert messages == ["Preparing sensing frames… Press Play when ready."]
        assert viz.status_fps_label.text() == "Playback: preparing sensing frames"
        assert not updates

        if not worker_finished:
            future.set_result(((0, 7, 23), {0: 0.0, 7: 0.064, 23: 0.201}))
        plan._poll()
        assert not plan.preparing
        assert updates == [7]
        assert not viz.animation_running and not viz.animation_timer.isActive()
        assert viz.status_fps_label.text() == "Playback updates: paused"

        button.click()
        assert viz.animation_running and viz.animation_timer.isActive()
        assert viz.play_direction == (-1 if reverse else 1)
        assert button.isChecked()
    finally:
        controller.shutdown()
        plan.close()


def test_closed_preparation_cannot_refresh_retired_controls(qapp):
    plan, provider, controller, updates, _counts, _preloads, _messages = _transport(qapp)
    future = Future()
    plan.prepare_for_scenario(provider, future)
    plan.visualizer.play_btn.click()
    controller.shutdown()
    plan.close()
    assert not plan.preparing
    delete(plan.checkbox)
    delete(plan.visualizer.status_fps_label)
    future.set_result(((0, 7, 23), {0: 0.0, 7: 0.064, 23: 0.201}))
    plan._poll()
    assert not updates and not plan.visualizer.animation_running


def test_genuinely_empty_plan_keeps_no_frames_message_and_paused_telemetry(qapp):
    plan, _provider, controller, _updates, _counts, _preloads, messages = _transport(qapp)
    viz = plan.visualizer
    viz.get_available_animation_steps = lambda: []
    try:
        viz.play_btn.click()
        controller.handle_animation_tick()
        assert not viz.animation_running and not viz.animation_timer.isActive()
        assert messages == ["No animation frames are available"]
        assert viz.status_fps_label.text() == "Playback updates: paused"
    finally:
        controller.shutdown()
        plan.close()
