"""Focused contracts for responsive animation timeline scrubbing."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSlider, QSpinBox

from visualizer.src.app.panel_manager import UIPanelManager
from visualizer.src.controllers.animation_controller import AnimationController
from visualizer.src.controllers.ui_controller import (
    STEP_SCRUB_COALESCE_INTERVAL_MS,
    UIController,
)
from visualizer.src.playback import PlaybackMode
from visualizer.visualizer import OrchavVisualizer


class _Timer:
    def __init__(self) -> None:
        self.active = False
        self.start_intervals: list[int] = []
        self.stop_count = 0

    def isActive(self) -> bool:
        return self.active

    def start(self, interval: int) -> None:
        self.active = True
        self.start_intervals.append(int(interval))

    def stop(self) -> None:
        self.active = False
        self.stop_count += 1


class _PlaybackControls:
    def frame_stride(self) -> int:
        return 1

    def playback_mode(self) -> PlaybackMode:
        return PlaybackMode.MAXIMUM

    def fixed_playback_fps(self) -> int:
        return 30

    def loop_enabled(self) -> bool:
        return True

    def is_online_mode(self) -> bool:
        return False

    def request_frame_if_needed(self, _frame_idx: int) -> bool:
        return False

    def prefetch_enabled(self, *, playing: bool) -> bool:
        return playing


def _ui_controller(viz) -> UIController:
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    return controller


def test_slider_changes_throttle_latest_value_without_restarting_active_timer(qapp) -> None:
    label = QLabel()
    frame_input = QSpinBox()
    frame_input.setRange(1, 20)
    frame_input.setValue(1)
    timer = _Timer()
    viz = SimpleNamespace(
        step_label=label,
        frame_input=frame_input,
        slider_scrub_timer=timer,
        _pending_slider_scrub_index=None,
    )
    emitted_values: list[int] = []
    frame_input.valueChanged.connect(emitted_values.append)

    controller = _ui_controller(viz)
    controller.handle_step_changed(4)
    controller.handle_step_changed(7)

    assert viz._pending_slider_scrub_index == 7
    assert label.text() == "8"
    assert frame_input.value() == 1
    assert emitted_values == []
    assert timer.start_intervals == [STEP_SCRUB_COALESCE_INTERVAL_MS]
    assert timer.stop_count == 0


def test_slider_release_is_wired_to_immediate_flush(qapp) -> None:
    releases: list[str] = []
    slider = QSlider(Qt.Horizontal)
    controller = SimpleNamespace(
        handle_step_changed=lambda _value: None,
        handle_step_slider_released=lambda: releases.append("released"),
    )
    parent = SimpleNamespace(
        ui_controller=controller,
        animation_controller=None,
        step_slider=slider,
    )
    manager = UIPanelManager(parent, total_steps=10)
    manager._connect_event_handlers(parent)

    slider.sliderReleased.emit()

    assert releases == ["released"]


def test_flush_stops_takes_and_clears_before_committing(qapp) -> None:
    timer = _Timer()
    timer.active = True
    commits: list[int] = []
    viz = OrchavVisualizer.__new__(OrchavVisualizer)
    viz.slider_scrub_timer = timer
    viz._pending_slider_scrub_index = 6
    viz._shutdown_started = False

    def commit(value: int) -> None:
        assert timer.active is False
        assert viz._pending_slider_scrub_index is None
        commits.append(value)

    viz.animation_controller = SimpleNamespace(handle_slider_commit=commit)

    assert viz.flush_pending_slider_scrub() is True
    assert commits == [6]
    assert timer.stop_count == 1


def test_flush_drops_pending_scrub_after_shutdown_starts(qapp) -> None:
    timer = _Timer()
    timer.active = True
    commits: list[int] = []
    viz = OrchavVisualizer.__new__(OrchavVisualizer)
    viz.slider_scrub_timer = timer
    viz._pending_slider_scrub_index = 6
    viz._shutdown_started = True
    viz.animation_controller = SimpleNamespace(handle_slider_commit=commits.append)

    assert viz.flush_pending_slider_scrub() is False
    assert viz._pending_slider_scrub_index is None
    assert commits == []
    assert timer.active is False


def test_cancel_restores_timeline_to_committed_frame_without_signals(qapp) -> None:
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 9)
    slider.setValue(7)
    label = QLabel("8")
    frame_input = QSpinBox()
    frame_input.setRange(1, 10)
    frame_input.setValue(2)
    slider_events: list[int] = []
    frame_events: list[int] = []
    slider.valueChanged.connect(slider_events.append)
    frame_input.valueChanged.connect(frame_events.append)

    viz = OrchavVisualizer.__new__(OrchavVisualizer)
    viz.slider_scrub_timer = _Timer()
    viz._pending_slider_scrub_index = 7
    viz._shutdown_started = False
    viz.animation_step = 127
    viz.get_animation_step_index = lambda _step: 1
    viz.step_slider = slider
    viz.step_label = label
    viz.frame_input = frame_input

    assert viz.cancel_pending_slider_scrub() is True
    assert viz._pending_slider_scrub_index is None
    assert slider.value() == 1
    assert label.text() == "2"
    assert frame_input.value() == 2
    assert slider_events == []
    assert frame_events == []


def test_authoritative_frame_update_clears_pending_scrub_without_committing_it(qapp) -> None:
    timer = _Timer()
    timer.active = True
    commits: list[int] = []
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 9)
    frame_input = QSpinBox()
    frame_input.setRange(1, 10)
    viz = OrchavVisualizer.__new__(OrchavVisualizer)
    viz.slider_scrub_timer = timer
    viz._pending_slider_scrub_index = 7
    viz._shutdown_started = False
    viz.animation_controller = SimpleNamespace(handle_slider_commit=commits.append)
    viz.animation_step = 0
    viz.set_state = lambda **_changes: None
    viz.get_animation_step_index = lambda step: int(step)
    viz.step_slider = slider
    viz.step_label = QLabel()
    viz.frame_input = frame_input
    viz.ui_controller = SimpleNamespace(update_frame_context=lambda _step: None)
    viz.vis_initialized = False

    assert viz.update_frame(2) is False
    assert viz._pending_slider_scrub_index is None
    assert timer.active is False
    assert commits == []
    assert slider.value() == 2
    assert viz.step_label.text() == "3"
    assert frame_input.value() == 3


def test_frame_input_cancels_scrub_before_immediate_navigation() -> None:
    events: list[object] = []
    viz = SimpleNamespace(
        cancel_pending_slider_scrub=lambda: events.append("cancel"),
        total_animation_steps=5,
        resolve_animation_step=lambda index: index * 10,
        update_frame=lambda step: events.append(("frame", step)),
    )

    _ui_controller(viz).handle_frame_input_changed(3)

    assert events == ["cancel", ("frame", 20)]


def test_transport_navigation_and_play_cancel_pending_scrub() -> None:
    events: list[object] = []
    animation_timer = _Timer()
    update_timer = _Timer()
    steps = [0, 1, 2]
    viz = SimpleNamespace(
        cancel_pending_slider_scrub=lambda: events.append("cancel"),
        animation_step=0,
        total_animation_steps=len(steps),
        animation_running=False,
        animation_timer=animation_timer,
        update_timer=update_timer,
        _idle_poll_interval_ms=16,
        _active_poll_interval_ms=16,
        play_direction=1,
        frame_source=None,
        get_available_animation_steps=lambda: steps,
        get_animation_step_index=lambda step: steps.index(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )

    def update_frame(step: int) -> bool:
        viz.animation_step = step
        events.append(("frame", step))
        return True

    viz.update_frame = update_frame
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=_PlaybackControls(),
    )

    controller.next_frame()
    controller.previous_frame()
    controller.reset_animation()
    controller.toggle_animation()

    assert events == [
        "cancel",
        ("frame", 1),
        "cancel",
        ("frame", 0),
        "cancel",
        ("frame", 0),
        "cancel",
    ]
    assert viz.animation_running is True
    assert animation_timer.start_intervals == [0]
