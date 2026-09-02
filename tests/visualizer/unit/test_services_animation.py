import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication

from shared.frames.types import StandardMPCFrame
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.controllers.animation_controller import (
    AnimationController,
    _VisualizerPlaybackControls,
)
from visualizer.src.playback import PlaybackMode
from visualizer.src.services.animation_service import AnimationService
from visualizer.src.services.cache_service import CacheService


class DummyFrameSource:
    def __init__(self):
        self.loaded_steps = []

    def load_frame(self, step: int) -> StandardMPCFrame:
        self.loaded_steps.append(step)
        return build_standard_mpc_frame(frame_idx=step)


class BlockingFrameSource:
    """Frame source whose single load can outlive a preloader stop timeout."""

    def __init__(self, step: int):
        self.step = step
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def list_frames(self) -> list[int]:
        return [self.step]

    def load_frame(self, step: int) -> StandardMPCFrame:
        assert step == self.step
        self.started.set()
        self.release.wait(timeout=10.0)
        self.finished.set()
        return build_standard_mpc_frame(frame_idx=step)


class ImmediateFrameSource:
    """Frame source used to complete a replacement preload deterministically."""

    def __init__(self, step: int):
        self.step = step

    def list_frames(self) -> list[int]:
        return [self.step]

    def load_frame(self, step: int) -> StandardMPCFrame:
        assert step == self.step
        return build_standard_mpc_frame(frame_idx=step)


class DummySceneOnlyFrameSource:
    def __init__(self):
        self.loaded_steps = []

    def load_frame(self, step: int) -> dict:
        self.loaded_steps.append(step)
        raise FileNotFoundError("No frames available (scene-only mode)")


class DummyPipeline:
    def __init__(self):
        self.updated_steps = []

    def update(self, step: int) -> None:
        self.updated_steps.append(step)


class DummyTimer:
    def __init__(self):
        self.interval = None
        self.started = False
        self.stopped = False
        self.callback = None

        class Timeout:
            def __init__(self, parent):
                self._parent = parent

            def connect(self, cb):
                self._parent.callback = cb

        self.timeout = Timeout(self)

    def start(self, interval: int):
        self.started = True
        self.interval = interval

    def stop(self):
        self.stopped = True


class DummyPlaybackControls:
    def __init__(
        self,
        *,
        speed_multiplier=1.0,
        play_as_available=True,
        stride=1,
        playback_mode=PlaybackMode.FIXED_FPS,
        fixed_fps=30,
        loop=True,
        online=False,
        request_result=True,
        prefetch_enabled=True,
    ):
        self.speed_multiplier = speed_multiplier
        self.play_as_available = play_as_available
        self.stride = stride
        self.mode = playback_mode
        self.fixed_fps_value = fixed_fps
        self.loop = loop
        self.online = online
        self.request_result = request_result
        self.prefetch_enabled_value = prefetch_enabled
        self.requested_frames = []
        self.prefetch_playing = None

    def live_speed_multiplier(self) -> float:
        return self.speed_multiplier

    def live_play_as_available(self) -> bool:
        return self.play_as_available

    def frame_stride(self) -> int:
        return self.stride

    def playback_mode(self) -> PlaybackMode:
        return self.mode

    def fixed_playback_fps(self) -> int:
        return self.fixed_fps_value

    def loop_enabled(self) -> bool:
        return self.loop

    def is_online_mode(self) -> bool:
        return self.online

    def request_frame_if_needed(self, frame_idx: int) -> bool:
        self.requested_frames.append(frame_idx)
        return self.request_result

    def prefetch_enabled(self, *, playing: bool) -> bool:
        self.prefetch_playing = playing
        return self.prefetch_enabled_value


class DummyLiveFrameSource:
    def __init__(self):
        self.subscribed_callback = None
        self.unsubscribed_callback = None
        self.frames = [0, 1, 2]

    def subscribe_to_frames(self, callback):
        self.subscribed_callback = callback

    def unsubscribe_from_frames(self, callback):
        self.unsubscribed_callback = callback

    def list_frames(self):
        return self.frames


class DummyRequestFrameSource:
    def __init__(self, *, success=True):
        self.success = success
        self.requested = []

    def request_frame(self, step: int) -> bool:
        self.requested.append(step)
        return self.success


class DummyConnectionManager:
    def __init__(self, *, succeed=True):
        self.closed = False
        self.ensure_called = False
        self.succeed = succeed

    def close(self):
        self.closed = True

    def ensure_connection(self) -> bool:
        self.ensure_called = True
        return self.succeed


class DummyFlushSource:
    def __init__(self, *, has_remote=True):
        self.flushed = None
        self.cleared = False
        if has_remote:
            self.request_cache_flush = self._flush
        else:
            self.clear_buffer = self._clear

    def _flush(self, reason: str) -> None:
        self.flushed = reason

    def _clear(self) -> None:
        self.cleared = True


def make_animation_controller(viz, timer=None, playback_controls=None):
    if not hasattr(viz, "ui_controller"):
        viz.ui_controller = SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        )
    animation_service = SimpleNamespace(
        load_step=lambda step: None,
        start=lambda start_step=0: None,
        stop=lambda: None,
        advance=lambda: None,
        preload=lambda steps: None,
    )
    kwargs = {}
    if timer is not None:
        kwargs["timer_factory"] = lambda: timer
    if playback_controls is not None:
        kwargs["playback_controls"] = playback_controls
    return AnimationController(viz, animation_service, **kwargs)


@pytest.fixture
def animation_service():
    pipeline = DummyPipeline()
    source = DummyFrameSource()
    visualizer = SimpleNamespace(frame_source=source)
    visualizer.cache_service = CacheService(visualizer, max_frame_cache_size=2)
    service = AnimationService(
        pipeline, visualizer, max_cached_steps=2, cache_service=visualizer.cache_service
    )
    return service, pipeline, source


def test_animation_service_advances_and_caches(animation_service):
    service, pipeline, source = animation_service
    service.start()

    frame0 = service.advance()
    frame1 = service.advance()

    assert frame0["_source"]["frame_idx"] == 0
    assert frame1["_source"]["frame_idx"] == 1
    assert pipeline.updated_steps == [0, 1]
    assert source.loaded_steps == [0, 1]

    # Re-advance to step 2 and ensure cache size maintained at 2
    service.advance()
    assert service.cache_service.frame_cache.size == 2


def test_animation_service_preload(animation_service):
    service, _, source = animation_service
    service.preload([3, 4])
    assert source.loaded_steps == [3, 4]


def test_replacement_preload_ignores_blocked_old_worker_signals_and_cache(qapp):
    old_source = BlockingFrameSource(step=1)
    visualizer = SimpleNamespace(frame_source=old_source, animation_step=0)
    cache_service = CacheService(visualizer, max_frame_cache_size=4)
    visualizer.cache_service = cache_service
    service = AnimationService(
        DummyPipeline(),
        visualizer,
        cache_service=cache_service,
    )
    old_progress: list[str] = []
    old_complete: list[list[int]] = []
    old_steps: list[int] = []
    new_progress: list[str] = []
    new_complete: list[list[int]] = []
    new_steps: list[int] = []

    assert service.start_preloading(
        on_progress=old_progress.append,
        on_complete=lambda frames, _duration: old_complete.append(
            [step for step, _frame in frames]
        ),
        on_step_ready=old_steps.append,
    )
    assert old_source.started.wait(timeout=1.0)
    old_preloader = service._threaded_preloader
    assert old_preloader is not None

    try:
        # The blocked load outlives stop()'s bounded join. A new generation
        # must still be allowed to become the sole owner of service state.
        service.reset_preloading_state()
        assert old_preloader._thread is not None
        assert old_preloader._thread.is_alive()

        visualizer.frame_source = ImmediateFrameSource(step=2)
        assert service.start_preloading(
            on_progress=new_progress.append,
            on_complete=lambda frames, _duration: new_complete.append(
                [step for step, _frame in frames]
            ),
            on_step_ready=new_steps.append,
        )
        replacement_preloader = service._threaded_preloader
        assert replacement_preloader is not None
        assert replacement_preloader._thread is not None
        replacement_preloader._thread.join(timeout=2.0)
        assert not replacement_preloader._thread.is_alive()
        QCoreApplication.processEvents()
        assert new_complete
        replacement_end_time = service._preload_end_time

        old_source.release.set()
        assert old_source.finished.wait(timeout=1.0)
        assert old_preloader._thread is not None
        old_preloader._thread.join(timeout=2.0)
        assert not old_preloader._thread.is_alive()
        QCoreApplication.processEvents()

        assert cache_service.has_frame(2)
        assert not cache_service.has_frame(1)
        assert old_complete == []
        assert old_steps == []
        assert not any(text.startswith("Preload: Complete") for text in old_progress)
        assert new_complete == [[2]]
        assert new_steps == [2]
        assert new_progress[-1] == "Preload: Complete (1 files)"
        assert service._preload_end_time == replacement_end_time
        assert service.preloading_completed is True
    finally:
        old_source.release.set()
        service.reset_preloading_state()


def test_animation_service_scene_only_frame_source_returns_none():
    pipeline = DummyPipeline()
    source = DummySceneOnlyFrameSource()
    visualizer = SimpleNamespace(frame_source=source, _scene_only_mode=True)
    service = AnimationService(pipeline, visualizer)

    service.start()
    frame = service.advance()

    assert frame is None
    assert source.loaded_steps == [0]
    assert pipeline.updated_steps == []


def test_animation_service_stop(animation_service):
    service, pipeline, _ = animation_service
    service.start()
    service.stop()
    result = service.advance()
    assert result is None
    assert pipeline.updated_steps == []


def test_animation_controller_live_playback_start_stop():
    frame_source = DummyLiveFrameSource()
    timer = DummyTimer()
    controls = DummyPlaybackControls(speed_multiplier=2.0)
    viz = SimpleNamespace(
        frame_source=frame_source,
        animation_step=0,
        update_frame=lambda idx: None,
        next_frame=lambda: None,
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(
            load_step=lambda step: None,
            start=lambda start_step=0: None,
            stop=lambda: None,
            advance=lambda: None,
            preload=lambda steps: None,
        ),
        timer_factory=lambda: timer,
        playback_controls=controls,
    )

    controller.start_live_playback()
    assert frame_source.subscribed_callback == controller.on_new_frame_available
    assert timer.started
    assert timer.interval == 500  # 1000 / 2x

    controller.stop_live_playback()
    assert frame_source.unsubscribed_callback == controller.on_new_frame_available
    assert timer.stopped


def test_animation_controller_live_playback_step_advances_frames():
    collected = []

    def record_frame(idx: int):
        collected.append(idx)

    frame_source = DummyLiveFrameSource()
    timer = DummyTimer()
    controls = DummyPlaybackControls(play_as_available=True)
    viz = SimpleNamespace(
        frame_source=frame_source,
        animation_step=0,
        update_frame=lambda idx: record_frame(idx),
        next_frame=lambda: collected.append("next"),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(
            load_step=lambda step: None,
            start=lambda start_step=0: None,
            stop=lambda: None,
            advance=lambda: None,
            preload=lambda steps: None,
        ),
        timer_factory=lambda: timer,
        playback_controls=controls,
    )

    # Force timer creation so callback is registered
    controller.start_live_playback()
    controller._live_playback_step()
    assert collected == [1]


def test_animation_controller_request_live_frame_success():
    frame_source = DummyRequestFrameSource(success=True)
    updates = []
    viz = SimpleNamespace(
        frame_source=frame_source,
        update_frame=lambda idx: updates.append(idx),
    )
    controller = make_animation_controller(viz)
    controller.request_live_frame(5)
    assert frame_source.requested == [5]
    assert updates == [5]


def test_animation_controller_request_live_frame_failure():
    frame_source = DummyRequestFrameSource(success=False)
    updates = []
    viz = SimpleNamespace(
        frame_source=frame_source,
        update_frame=lambda idx: updates.append(idx),
    )
    controller = make_animation_controller(viz)
    controller.request_live_frame(3)
    assert frame_source.requested == [3]
    assert updates == []


def test_animation_controller_reconnect_live_stream():
    manager = DummyConnectionManager(succeed=True)
    frame_source = SimpleNamespace(connection_manager=manager)
    viz = SimpleNamespace(frame_source=frame_source)
    controller = make_animation_controller(viz)
    assert controller.reconnect_live_stream() is True
    assert manager.closed is True
    assert manager.ensure_called is True


def test_animation_controller_clear_live_buffer_prefers_remote_flush():
    frame_source = DummyFlushSource(has_remote=True)
    viz = SimpleNamespace(frame_source=frame_source)
    controller = make_animation_controller(viz)
    controller.clear_live_buffer()
    assert frame_source.flushed == "Live panel clear buffer"


def test_animation_controller_clear_live_buffer_local_fallback():
    frame_source = DummyFlushSource(has_remote=False)
    viz = SimpleNamespace(frame_source=frame_source)
    controller = make_animation_controller(viz)
    controller.clear_live_buffer()
    assert frame_source.cleared is True


def test_animation_controller_slider_commit_maps_sparse_display_index_to_step():
    available_steps = [63, 127, 191]
    updates = []
    viz = SimpleNamespace(
        resolve_animation_step=lambda value: available_steps[value],
        update_frame=lambda step: updates.append(step),
    )
    controller = make_animation_controller(viz)

    controller.handle_slider_commit(1)

    assert updates == [127]


def test_animation_controller_slider_commit_requests_online_frames_through_controls():
    available_steps = [63, 127, 191]
    updates = []
    controls = DummyPlaybackControls(online=True)
    viz = SimpleNamespace(
        resolve_animation_step=lambda value: available_steps[value],
        update_frame=lambda step: updates.append(step),
    )
    controller = make_animation_controller(viz, playback_controls=controls)

    controller.handle_slider_commit(1)

    assert controls.requested_frames == [127]
    assert updates == [127]


def test_animation_controller_toggle_uses_fixed_fps_playback_interval():
    animation_timer = DummyTimer()
    update_timer = DummyTimer()
    controls = DummyPlaybackControls(fixed_fps=20)
    cadence_resets = []
    viz = SimpleNamespace(
        play_direction=1,
        animation_running=False,
        animation_timer=animation_timer,
        update_timer=update_timer,
        _active_poll_interval_ms=12,
        _idle_poll_interval_ms=99,
        frame_source=None,
        frame_times=[],
        playback_cadence=SimpleNamespace(reset=lambda: cadence_resets.append(True)),
    )
    controller = make_animation_controller(viz, playback_controls=controls)

    controller.toggle_animation()

    assert animation_timer.started
    assert animation_timer.interval == 50
    assert update_timer.interval == 12
    assert viz.animation_running is True
    assert cadence_resets == [True]


def test_animation_controller_maximum_mode_uses_zero_delay_serial_timer():
    animation_timer = DummyTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM)
    viz = SimpleNamespace(
        play_direction=1,
        animation_running=False,
        animation_timer=animation_timer,
        update_timer=DummyTimer(),
        _active_poll_interval_ms=12,
        _idle_poll_interval_ms=99,
        frame_source=None,
    )
    controller = make_animation_controller(viz, playback_controls=controls)

    controller.toggle_animation()

    assert animation_timer.interval == 0


def test_maximum_mode_rearms_only_after_frame_transaction_completes():
    events = []

    class RecordingTimer(DummyTimer):
        def start(self, interval: int):
            super().start(interval)
            events.append(("timer", interval))

    timer = RecordingTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM, loop=False)
    available_steps = [0, 1]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        total_animation_steps=2,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )

    def update_frame(step):
        events.append(("frame_start", step))
        viz.animation_step = step
        events.append(("frame_end", step))

    viz.update_frame = update_frame
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    controller.handle_animation_tick()

    assert events == [("frame_start", 1), ("frame_end", 1), ("timer", 0)]


def test_maximum_mode_waits_for_backend_render_turn_permit():
    events = []
    permit_callbacks = []

    class RecordingTimer(DummyTimer):
        def start(self, interval: int):
            super().start(interval)
            events.append(("timer", interval))

    class DeferredRenderer:
        def defer_until_next_render_turn(self, callback):
            permit_callbacks.append(callback)
            return True

    timer = RecordingTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM, loop=False)
    available_steps = [0, 1]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        renderer=DeferredRenderer(),
        total_animation_steps=2,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )

    def update_frame(step):
        events.append(("frame", step))
        viz.animation_step = step

    viz.update_frame = update_frame
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    controller.handle_animation_tick()

    assert events == [("frame", 1)]
    assert len(permit_callbacks) == 1
    permit_callbacks[0]()
    permit_callbacks[0]()
    assert events == [("frame", 1), ("timer", 0)]


def test_timing_restart_waits_for_outstanding_renderer_turn():
    permit_callbacks = []
    timer = DummyTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM, loop=True)
    available_steps = [0, 1, 2]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        renderer=SimpleNamespace(
            defer_until_next_render_turn=lambda callback: permit_callbacks.append(callback) or True
        ),
        total_animation_steps=len(available_steps),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        update_frame=lambda step: setattr(viz, "animation_step", step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    controller.handle_animation_tick()
    assert controller._renderer_turn_pending is True
    assert timer.started is False

    controls.mode = PlaybackMode.FIXED_FPS
    controls.fixed_fps_value = 30
    controller.restart_playback_timer()

    assert timer.started is False
    assert controller._restart_after_renderer_turn is True

    permit_callbacks.pop()()

    assert controller._renderer_turn_pending is False
    assert controller._restart_after_renderer_turn is False
    assert timer.started is True
    assert timer.interval == 33


def test_direction_restart_waits_for_outstanding_renderer_turn():
    permit_callbacks = []
    timer = DummyTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM, loop=True)
    available_steps = [0, 1, 2]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        renderer=SimpleNamespace(
            defer_until_next_render_turn=lambda callback: permit_callbacks.append(callback) or True
        ),
        total_animation_steps=len(available_steps),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        update_frame=lambda step: setattr(viz, "animation_step", step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    controller.handle_animation_tick()
    controller.toggle_animation(direction=-1)

    assert viz.play_direction == -1
    assert timer.started is False
    permit_callbacks.pop()()
    assert timer.started is True
    assert timer.interval == 0


def test_failed_frame_submission_stops_playback_without_rearming():
    timer = DummyTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM, loop=True)
    available_steps = [0, 1]
    messages = []
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        total_animation_steps=len(available_steps),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        update_frame=lambda _step: False,
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
        _set_status_message=messages.append,
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    controller.handle_animation_tick()

    assert viz.animation_running is False
    assert timer.started is False
    assert timer.stopped is True
    assert messages == ["Playback stopped: frame 2 was not rendered"]


def test_shutdown_invalidates_released_backend_permit_before_renderer_close():
    permit_callbacks = []
    timer = DummyTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM, loop=False)
    available_steps = [0, 1]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        renderer=SimpleNamespace(
            defer_until_next_render_turn=lambda callback: permit_callbacks.append(callback) or True
        ),
        total_animation_steps=2,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        update_frame=lambda step: setattr(viz, "animation_step", step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    controller.handle_animation_tick()
    controller.shutdown()
    permit_callbacks.pop()()

    assert timer.started is False
    assert timer.stopped is True
    assert viz.animation_running is False


def test_fixed_fps_waits_for_backend_permit_then_uses_remaining_deadline():
    now = [0.0]
    timer_intervals = []
    permit_callbacks = []

    class RecordingTimer(DummyTimer):
        def start(self, interval: int):
            super().start(interval)
            timer_intervals.append(interval)

    controls = DummyPlaybackControls(
        playback_mode=PlaybackMode.FIXED_FPS,
        fixed_fps=60,
        loop=True,
    )
    available_steps = [0, 1, 2]
    timer = RecordingTimer()
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        renderer=SimpleNamespace(
            defer_until_next_render_turn=lambda callback: permit_callbacks.append(callback) or True
        ),
        total_animation_steps=len(available_steps),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        update_frame=lambda step: setattr(viz, "animation_step", step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
        clock=lambda: now[0],
    )
    controller.restart_playback_timer()
    assert timer_intervals == [17]

    timer_intervals.clear()
    now[0] = 0.017
    controller.handle_animation_tick()
    assert timer_intervals == []

    now[0] = 0.020
    permit_callbacks.pop()()
    assert timer_intervals == [13]


def test_fixed_60_fps_uses_absolute_deadlines_without_rounding_drift():
    now = [0.0]
    timer = DummyTimer()
    controls = DummyPlaybackControls(
        playback_mode=PlaybackMode.FIXED_FPS,
        fixed_fps=60,
        loop=True,
    )
    available_steps = [0, 1, 2, 3]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        total_animation_steps=len(available_steps),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )
    viz.update_frame = lambda step: setattr(viz, "animation_step", step)
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
        clock=lambda: now[0],
    )

    controller.restart_playback_timer()
    assert timer.interval == 17

    now[0] = 0.017
    controller.handle_animation_tick()
    assert timer.interval == 16

    now[0] = 0.033
    controller.handle_animation_tick()
    assert timer.interval == 17


def test_animation_controller_real_time_uses_scenario_duration_and_stride():
    animation_timer = DummyTimer()
    controls = DummyPlaybackControls(
        playback_mode=PlaybackMode.REAL_TIME,
        stride=2,
    )
    available_steps = list(range(50))
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=False,
        animation_timer=animation_timer,
        update_timer=DummyTimer(),
        _active_poll_interval_ms=12,
        _idle_poll_interval_ms=99,
        _frame_duration=10.0,
        frame_source=None,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
    )
    controller = make_animation_controller(viz, playback_controls=controls)

    controller.toggle_animation()

    assert animation_timer.interval == 400


def test_animation_controller_real_time_prefers_cached_frame_timestamps():
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.REAL_TIME)
    cache = SimpleNamespace(
        get_frame=lambda step: {"timestamp_ns": {0: 1_000_000_000, 1: 1_250_000_000}[step]}
    )
    animation_service = SimpleNamespace(cache_service=cache)
    available_steps = [0, 1, 2]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=DummyTimer(),
        _frame_duration=9.0,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
    )
    controller = AnimationController(
        viz,
        animation_service,
        playback_controls=controls,
    )

    assert controller.playback_timer_interval_ms() == 250


def test_animation_controller_real_time_ignores_source_provenance_timestamps():
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.REAL_TIME)
    cache = SimpleNamespace(
        get_frame=lambda step: {"_source": {"timestamp": {0: 10.0, 1: 10.125}[step]}}
    )
    animation_service = SimpleNamespace(cache_service=cache)
    available_steps = [0, 1]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=DummyTimer(),
        _frame_duration=4.0,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
    )
    controller = AnimationController(
        viz,
        animation_service,
        playback_controls=controls,
    )

    assert controller.playback_timer_interval_ms() == 2000


def test_animation_controller_online_real_time_uses_declared_scenario_frame_count():
    controls = DummyPlaybackControls(
        playback_mode=PlaybackMode.REAL_TIME,
        online=True,
    )
    available_steps = [0, 1]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=DummyTimer(),
        total_animation_steps=12,
        _frame_duration=12.0,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    assert controller.playback_timer_interval_ms() == 1000


def test_animation_controller_real_time_deadline_excludes_frame_processing_time():
    now = [0.0]
    timer = DummyTimer()
    controls = DummyPlaybackControls(
        playback_mode=PlaybackMode.REAL_TIME,
        loop=False,
    )
    available_steps = [0, 1, 2]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        total_animation_steps=len(available_steps),
        _frame_duration=3.0,
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )

    def update_frame(step):
        viz.animation_step = step
        now[0] += 0.2

    viz.update_frame = update_frame
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
        clock=lambda: now[0],
    )

    controller.restart_playback_timer()
    assert timer.interval == 1000

    now[0] = 1.0
    controller.handle_animation_tick()

    assert viz.animation_step == 1
    assert timer.interval == 800


def test_real_time_rebases_after_long_stall_with_only_one_immediate_tick():
    now = [10.0]
    timer = DummyTimer()
    viz = SimpleNamespace(
        animation_running=True,
        animation_timer=timer,
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=DummyPlaybackControls(playback_mode=PlaybackMode.REAL_TIME),
        clock=lambda: now[0],
    )
    controller._real_time_timer_interval_ms = lambda: 1000
    controller._real_time_next_deadline_s = 1.0

    controller._schedule_next_real_time_tick()
    assert timer.interval == 0
    assert controller._real_time_next_deadline_s == 10.0

    now[0] = 10.1
    controller._schedule_next_real_time_tick()
    assert timer.interval == 900
    assert controller._real_time_next_deadline_s == 11.0


def test_animation_controller_records_one_successful_pipeline_completion_per_tick():
    recorded = []
    available_steps = [0, 1]
    controls = DummyPlaybackControls(loop=False)
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=DummyTimer(),
        total_animation_steps=len(available_steps),
        playback_cadence=SimpleNamespace(record_completion=lambda: recorded.append("completed")),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=controls,
    )

    def update_frame(step):
        viz.animation_step = step
        assert controller.record_completed_playback_tick(step) is True
        assert controller.record_completed_playback_tick(step) is False

    viz.update_frame = update_frame

    controller.handle_animation_tick()

    assert recorded == ["completed"]


def test_animation_controller_does_not_record_aborted_pipeline_tick():
    recorded = []
    available_steps = [0, 1]
    viz = SimpleNamespace(
        play_direction=1,
        animation_step=0,
        animation_running=True,
        animation_timer=DummyTimer(),
        total_animation_steps=len(available_steps),
        playback_cadence=SimpleNamespace(record_completion=lambda: recorded.append("completed")),
        update_frame=lambda step: setattr(viz, "animation_step", step),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
        ),
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=None),
        playback_controls=DummyPlaybackControls(loop=False),
    )

    controller.handle_animation_tick()

    assert recorded == []


def test_maximum_playback_stops_while_buffering_and_restarts_when_ready():
    class Cache:
        ready = False

        def has_frame(self, _step):
            return self.ready

    cache = Cache()
    buffer_timer = DummyTimer()
    animation_timer = DummyTimer()
    controls = DummyPlaybackControls(playback_mode=PlaybackMode.MAXIMUM)
    updated = []
    cadence_resets = []
    status_messages = []
    viz = SimpleNamespace(
        animation_running=True,
        animation_timer=animation_timer,
        update_frame=lambda step: updated.append(step),
        ui_controller=SimpleNamespace(
            update_performance_display=lambda: None,
            refresh_status_telemetry=lambda: None,
            _telemetry_ctrl=SimpleNamespace(_scenario_summary_text="Scenario summary"),
        ),
        playback_cadence=SimpleNamespace(reset=lambda: cadence_resets.append(True)),
        _set_status_message=status_messages.append,
    )
    controller = AnimationController(
        viz,
        SimpleNamespace(cache_service=cache),
        timer_factory=lambda: buffer_timer,
        playback_controls=controls,
    )

    controller._start_buffering(4)
    assert animation_timer.stopped is True

    cache.ready = True
    controller._check_buffer_ready()

    assert updated == [4]
    assert animation_timer.interval == 0
    assert cadence_resets == [True, True]
    assert status_messages == [
        "Buffering... (frame 4 not loaded yet)",
        "Scenario summary",
    ]


def test_animation_controller_next_frame_uses_stride_over_sparse_steps():
    available_steps = [63, 127, 191, 255, 319]
    updates = []
    controls = DummyPlaybackControls(stride=2, loop=False)
    viz = SimpleNamespace(
        animation_step=63,
        total_animation_steps=len(available_steps),
        update_frame=lambda step: updates.append(step),
        get_available_animation_steps=lambda: available_steps,
        get_animation_step_index=lambda step: available_steps.index(step),
    )
    controller = make_animation_controller(viz, playback_controls=controls)

    controller.next_frame()

    assert updates == [191]


@pytest.mark.parametrize("method_name", ["next_frame", "previous_frame"])
def test_animation_navigation_is_idle_when_no_frames_are_available(method_name):
    updates = []
    statuses = []
    timer = DummyTimer()
    viz = SimpleNamespace(
        animation_step=0,
        animation_running=False,
        animation_timer=timer,
        total_animation_steps=0,
        update_frame=lambda step: updates.append(step),
        get_available_animation_steps=lambda: [],
        get_animation_step_index=lambda _step: (_ for _ in ()).throw(
            AssertionError("empty frame plans have no current index")
        ),
        _set_status_message=lambda text: statuses.append(text),
    )
    controller = make_animation_controller(viz, playback_controls=DummyPlaybackControls())

    getattr(controller, method_name)()

    assert updates == []
    assert timer.stopped is True
    assert statuses == ["No animation frames are available"]


def test_animation_tick_stops_playback_when_live_source_is_temporarily_empty():
    updates = []
    statuses = []
    timer = DummyTimer()
    viz = SimpleNamespace(
        animation_step=0,
        animation_running=True,
        animation_timer=timer,
        play_direction=1,
        total_animation_steps=0,
        update_frame=lambda step: updates.append(step),
        get_available_animation_steps=lambda: [],
        get_animation_step_index=lambda _step: (_ for _ in ()).throw(
            AssertionError("empty frame plans have no current index")
        ),
        _set_status_message=lambda text: statuses.append(text),
    )
    controller = make_animation_controller(viz, playback_controls=DummyPlaybackControls())

    controller.handle_animation_tick()

    assert updates == []
    assert viz.animation_running is False
    assert timer.stopped is True
    assert statuses == ["No animation frames are available"]


def test_animation_controller_prefetch_settings_use_playback_controls(monkeypatch):
    from visualizer.src.io import frame_sources

    class FakeLiveGrpcSource:
        def __init__(self, provider):
            self.provider = provider

    class Provider:
        def __init__(self):
            self.play_direction = None
            self.prefetch_enabled = None

        def set_play_direction(self, direction):
            self.play_direction = direction

        def set_prefetch_enabled(self, enabled):
            self.prefetch_enabled = enabled

    monkeypatch.setattr(frame_sources, "LiveGrpcSource", FakeLiveGrpcSource)
    provider = Provider()
    controls = DummyPlaybackControls(prefetch_enabled=False)
    viz = SimpleNamespace(
        frame_source=FakeLiveGrpcSource(provider),
        play_direction=-1,
        animation_running=False,
    )
    controller = make_animation_controller(viz, playback_controls=controls)

    controller.sync_prefetch_settings()

    assert controls.prefetch_playing is False
    assert provider.play_direction == -1
    assert provider.prefetch_enabled is False


def test_visualizer_prefetch_policy_reads_data_source_controls() -> None:
    class CheckBox:
        def __init__(self, checked: bool) -> None:
            self._checked = checked

        def isChecked(self) -> bool:
            return self._checked

    data_source = SimpleNamespace(
        widgets={
            "prefetch_enable": CheckBox(True),
            "prefetch_pause_when_paused": CheckBox(True),
        }
    )
    animation = SimpleNamespace(widgets={})
    viz = SimpleNamespace(
        ui_manager=SimpleNamespace(panels={"animation": animation, "data_source": data_source})
    )
    controls = _VisualizerPlaybackControls(viz)

    assert controls.prefetch_enabled(playing=True) is True
    assert controls.prefetch_enabled(playing=False) is False
