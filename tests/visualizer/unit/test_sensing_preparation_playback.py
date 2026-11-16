"""Background sensing preparation must not hold up scene playback."""

from threading import Event
from types import SimpleNamespace

import pytest

from tests.visualizer.unit.test_services_animation import DummyPlaybackControls, DummyTimer
from visualizer.src.controllers.animation_controller import AnimationController
from visualizer.src.playback import PlaybackMode


@pytest.fixture
def playback():
    preparation = SimpleNamespace(cancelled=Event())
    service = SimpleNamespace(preparation=preparation, preparation_pending=True)
    steps = [0, 3, 6]
    updates, messages = [], []
    viz = SimpleNamespace(
        animation_running=True,
        animation_step=0,
        play_direction=1,
        animation_timer=DummyTimer(),
        update_timer=DummyTimer(),
        _idle_poll_interval_ms=100,
        _active_poll_interval_ms=10,
        total_animation_steps=len(steps),
        sensing_playback=SimpleNamespace(steps=tuple(steps)),
        extension_services={"sensing": service},
        get_available_animation_steps=lambda: steps,
        get_animation_step_index=lambda step: steps.index(step),
        _set_status_message=messages.append,
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )

    def update(step):
        viz.animation_step = step
        updates.append(step)

    viz.update_frame = update
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        timer_factory=DummyTimer,
        playback_controls=DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM),
    )
    return controller, viz, service, updates, messages


@pytest.mark.parametrize("sensing_only", [True, False])
@pytest.mark.parametrize("direction,start,expected", [(1, 0, 3), (-1, 6, 3)])
def test_scene_playback_advances_while_sensing_prepares(
    playback, sensing_only, direction, start, expected
):
    controller, viz, service, updates, messages = playback
    if not sensing_only:
        viz.sensing_playback.steps = None
    viz.play_direction = direction
    viz.animation_step = start

    controller.handle_animation_tick()

    assert service.preparation_pending
    assert updates == [expected]
    assert viz.animation_running and not controller.is_buffering
    assert messages == []
