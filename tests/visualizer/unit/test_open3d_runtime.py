"""Focused tests for Open3D redraw and presentation ordering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from visualizer.src.renderers.open3d import renderer as renderer_module
from visualizer.src.renderers.open3d import runtime as runtime_module
from visualizer.src.renderers.open3d.runtime import Open3DRuntimeMixin


class _FakeVisualizer:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def post_redraw(self) -> None:
        self._events.append("post_redraw")


class _FakeTimer:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.stopped = False

    def start(self, interval_ms: int) -> None:
        self._events.append(f"timer_start:{interval_ms}")

    def isActive(self) -> bool:
        return not self.stopped

    def stop(self) -> None:
        self.stopped = True
        self._events.append("timer_stop")


def _runtime(events: list[str]) -> Open3DRuntimeMixin:
    renderer = Open3DRuntimeMixin()
    renderer._o3d_vis = _FakeVisualizer(events)
    renderer._force_scene_redraw = lambda: events.append("force_scene_dirty")
    renderer._record_redraw_request = lambda: events.append("record_redraw")
    renderer._record_event_pump = lambda: events.append("record_event_pump")
    renderer._record_draw_pump = lambda alive, **_kwargs: events.append(
        f"record_draw_pump:{bool(alive)}"
    )
    renderer._batch_mode = False
    renderer._batch_redraw_pending = False
    renderer._frame_update_in_progress = False
    renderer._deferred_render_turn_callbacks = []
    renderer._render_turn_lifecycle_generation = 0
    renderer._native_redraw_pending = False
    renderer._visibility_settle_redraw_pending = False
    renderer._render_debug_enabled = False
    renderer._gui_timer_interval_ms = 16
    return renderer


def test_nested_batch_propagates_exception_and_preserves_pending_redraw() -> None:
    events: list[str] = []
    renderer = _runtime(events)
    renderer._flush_pending_object_visibility = lambda: events.append("flush_visibility")
    renderer._submit_redraw_now = lambda: events.append("submit_redraw")

    with pytest.raises(RuntimeError, match="nested update failed"):
        with renderer.batch_updates():
            renderer.request_redraw()
            with renderer.batch_updates():
                renderer.request_redraw()
                raise RuntimeError("nested update failed")

    assert renderer._batch_mode is False
    assert renderer._batch_redraw_pending is False
    assert events == ["flush_visibility", "submit_redraw"]


def test_visibility_settle_redraw_waits_for_a_separate_native_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synchronous submit cannot consume its own visibility settle redraw."""
    events: list[str] = []
    queued_callbacks = []
    renderer = _runtime(events)
    renderer._gui_initialized = True
    renderer._gui_timer = _FakeTimer(events)
    renderer._observe_camera_state = lambda phase: events.append(phase)

    native_stack_active = False

    def run_one_tick() -> bool:
        nonlocal native_stack_active
        native_stack_active = True
        events.append("run_one_tick:start")
        native_stack_active = False
        events.append("run_one_tick:end")
        return True

    application = SimpleNamespace(run_one_tick=run_one_tick)
    monkeypatch.setattr(
        runtime_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )
    monkeypatch.setattr(runtime_module, "HAS_QTIMER", True)

    def queue_callback(delay: int, callback) -> None:  # noqa: ANN001
        assert native_stack_active is False
        events.append(f"single_shot:{delay}")
        queued_callbacks.append(callback)

    monkeypatch.setattr(
        runtime_module,
        "QTimer",
        SimpleNamespace(singleShot=queue_callback),
    )

    renderer._request_visibility_settle_redraw("first label")
    renderer._request_visibility_settle_redraw("second label")

    assert renderer._visibility_settle_redraw_pending is True
    assert len(renderer._deferred_render_turn_callbacks) == 1

    assert renderer._submit_redraw_now() is True

    assert events.count("post_redraw") == 1
    assert queued_callbacks == []
    assert renderer._visibility_settle_redraw_pending is True
    assert len(renderer._deferred_render_turn_callbacks) == 1

    renderer._tick_o3d_gui()

    native_turn_ends = [index for index, event in enumerate(events) if event == "run_one_tick:end"]
    assert len(native_turn_ends) == 2
    assert native_turn_ends[-1] < events.index("single_shot:0")
    assert events.count("post_redraw") == 1
    assert len(queued_callbacks) == 1
    assert renderer._visibility_settle_redraw_pending is True

    queued_callbacks.pop()()

    assert renderer._visibility_settle_redraw_pending is False
    assert events.count("post_redraw") == 2
    assert renderer._native_redraw_pending is True

    renderer._tick_o3d_gui()

    assert events.count("post_redraw") == 2
    assert renderer._native_redraw_pending is False
    assert queued_callbacks == []


def test_visibility_settle_redraw_rejection_clears_coalescing_state() -> None:
    events: list[str] = []
    renderer = _runtime(events)
    renderer._gui_timer = _FakeTimer(events)
    renderer._gui_timer.stop()

    renderer._request_visibility_settle_redraw("inactive native pump")

    assert renderer._visibility_settle_redraw_pending is False
    assert renderer._deferred_render_turn_callbacks == []


def test_submit_redraw_posts_before_pumping_without_claiming_present(monkeypatch) -> None:
    """A completed update queues its draw before the redraw-bearing event pump."""
    events: list[str] = []
    renderer = _runtime(events)
    application = SimpleNamespace(run_one_tick=lambda: events.append("run_one_tick") or True)
    monkeypatch.setattr(
        runtime_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )

    assert renderer._submit_redraw_now() is True
    assert events == [
        "force_scene_dirty",
        "record_redraw",
        "post_redraw",
        "record_event_pump",
        "run_one_tick",
        "record_draw_pump:True",
    ]


def test_end_frame_queues_redraw_without_pumping_or_restarting_gui_timer(monkeypatch) -> None:
    """Frame submission leaves the independently scheduled native pump armed."""
    events: list[str] = []
    renderer = _runtime(events)
    application = SimpleNamespace(run_one_tick=lambda: events.append("run_one_tick") or True)
    monkeypatch.setattr(
        runtime_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )
    renderer._set_far_clipping_plane = lambda: events.append("set_far_plane")
    renderer._frame_update_in_progress = True
    renderer._frame_redraw_pending = True
    renderer._gui_timer = _FakeTimer(events)
    renderer._gui_timer_interval_ms = 16

    assert renderer.end_frame_update() is True

    assert events == ["set_far_plane", "force_scene_dirty", "record_redraw", "post_redraw"]
    assert renderer._frame_update_in_progress is False
    assert renderer._frame_redraw_pending is False
    assert renderer._native_redraw_pending is True


def test_render_turn_runtime_error_retains_redraw_and_waiter_for_retry(monkeypatch) -> None:
    events: list[str] = []
    renderer = _runtime(events)
    renderer._gui_initialized = True
    renderer._gui_timer = _FakeTimer(events)
    renderer._native_redraw_pending = True
    renderer._observe_camera_state = lambda phase: events.append(phase)
    application = SimpleNamespace(run_one_tick=lambda: (_ for _ in ()).throw(RuntimeError("busy")))
    monkeypatch.setattr(
        runtime_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )
    monkeypatch.setattr(runtime_module, "HAS_QTIMER", False)

    assert renderer.defer_until_next_render_turn(lambda: events.append("permit")) is True
    renderer._tick_o3d_gui()

    assert "permit" not in events
    assert renderer._native_redraw_pending is True
    assert len(renderer._deferred_render_turn_callbacks) == 1
    assert renderer._gui_timer.stopped is False
    assert "record_draw_pump:False" in events

    application.run_one_tick = lambda: events.append("run_one_tick_ok") or True
    renderer._tick_o3d_gui()

    assert events[-1] == "permit"
    assert renderer._native_redraw_pending is False
    assert renderer._deferred_render_turn_callbacks == []


def test_dead_native_pump_invalidates_waiters_without_advancing(monkeypatch) -> None:
    events: list[str] = []
    renderer = _runtime(events)
    renderer._gui_initialized = True
    renderer._gui_timer = _FakeTimer(events)
    renderer._native_redraw_pending = True
    renderer._observe_camera_state = lambda phase: events.append(phase)
    application = SimpleNamespace(run_one_tick=lambda: False)
    monkeypatch.setattr(
        runtime_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )

    generation = renderer._render_turn_lifecycle_generation
    renderer._request_visibility_settle_redraw("labels")
    assert renderer.defer_until_next_render_turn(lambda: events.append("permit")) is True
    renderer._tick_o3d_gui()

    assert "permit" not in events
    assert renderer._deferred_render_turn_callbacks == []
    assert renderer._render_turn_lifecycle_generation == generation + 1
    assert renderer._native_redraw_pending is False
    assert renderer._visibility_settle_redraw_pending is False
    assert renderer._gui_timer.stopped is True


def test_render_turn_permit_releases_after_native_pump_and_outside_o3d_stack(
    monkeypatch,
) -> None:
    events: list[str] = []
    queued_callbacks = []
    renderer = _runtime(events)
    renderer._gui_initialized = True
    renderer._gui_timer = _FakeTimer(events)
    renderer._observe_camera_state = lambda phase: events.append(phase)
    application = SimpleNamespace(run_one_tick=lambda: events.append("run_one_tick") or True)
    monkeypatch.setattr(
        runtime_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )
    monkeypatch.setattr(runtime_module, "HAS_QTIMER", True)
    monkeypatch.setattr(
        runtime_module,
        "QTimer",
        SimpleNamespace(
            singleShot=lambda delay, callback: (
                events.append(f"single_shot:{delay}"),
                queued_callbacks.append(callback),
            )
        ),
    )

    assert renderer.defer_until_next_render_turn(lambda: events.append("permit")) is True
    renderer._tick_o3d_gui()

    assert events.index("run_one_tick") < events.index("single_shot:0")
    assert "permit" not in events
    assert renderer._deferred_render_turn_callbacks == []
    queued_callbacks.pop()()
    assert events[-1] == "permit"


def test_render_turn_permit_releases_all_waiters_in_registration_order(monkeypatch) -> None:
    events: list[str] = []
    queued_callbacks = []
    renderer = _runtime(events)
    renderer._gui_timer = _FakeTimer(events)
    monkeypatch.setattr(runtime_module, "HAS_QTIMER", True)
    monkeypatch.setattr(
        runtime_module,
        "QTimer",
        SimpleNamespace(singleShot=lambda _delay, callback: queued_callbacks.append(callback)),
    )

    assert renderer.defer_until_next_render_turn(lambda: events.append("first")) is True
    assert renderer.defer_until_next_render_turn(lambda: events.append("second")) is True

    renderer._release_deferred_render_turn()
    assert len(queued_callbacks) == 1
    queued_callbacks.pop()()

    assert events == ["first", "second"]


def test_queued_render_turn_permits_are_invalidated_by_renderer_lifecycle(monkeypatch) -> None:
    events: list[str] = []
    queued_callbacks = []
    renderer = _runtime(events)
    renderer._gui_timer = _FakeTimer(events)
    monkeypatch.setattr(runtime_module, "HAS_QTIMER", True)
    monkeypatch.setattr(
        runtime_module,
        "QTimer",
        SimpleNamespace(singleShot=lambda _delay, callback: queued_callbacks.append(callback)),
    )

    renderer._request_visibility_settle_redraw("stale visibility")
    assert renderer.defer_until_next_render_turn(lambda: events.append("stale")) is True
    renderer._release_deferred_render_turn()
    renderer._invalidate_native_pump("test renderer lifecycle invalidation")
    renderer._o3d_vis = None
    queued_callbacks.pop()()

    assert "stale" not in events
    assert "post_redraw" not in events
    assert renderer._visibility_settle_redraw_pending is False


def test_open3d_gui_pump_timer_uses_default_timer_to_avoid_deadline_aliasing(
    monkeypatch,
) -> None:
    events = []

    class _Signal:
        def connect(self, callback):
            events.append(("connect", callback))

    class _Timer:
        def __init__(self):
            self.timeout = _Signal()

        def start(self, interval_ms):
            events.append(("start", interval_ms))

    def callback() -> None:
        pass

    monkeypatch.setattr(renderer_module, "QTimer", _Timer)

    timer = renderer_module._create_gui_pump_timer(callback, 16)

    assert isinstance(timer, _Timer)
    assert events == [
        ("connect", callback),
        ("start", 16),
    ]


def test_open3d_runtime_stats_do_not_report_unobservable_present_success() -> None:
    class _StatsBase:
        def get_runtime_stats(self):
            return {"present_attempts": 9, "present_successes": 9}

    class _StatsRuntime(Open3DRuntimeMixin, _StatsBase):
        pass

    renderer = _StatsRuntime()
    renderer._draw_pump_attempts = 7
    renderer._draw_pump_alive = 6
    renderer._event_pump_calls = 20
    renderer._redraw_requests = 11
    renderer._benchmark_telemetry_baseline = {
        "event_pump_calls": 14,
        "redraw_requests": 7,
    }
    renderer._benchmark_frame_submissions = 4
    renderer._benchmark_redraw_pump_attempts = 5
    renderer._benchmark_redraw_pump_alive = 4
    renderer._o3d_vis = None

    stats = renderer.get_runtime_stats()

    assert stats["presentation_observable"] is False
    assert stats["present_attempts"] is None
    assert stats["present_successes"] is None
    assert stats["draw_pump_attempts"] == 7
    assert stats["draw_pump_alive"] == 6
    assert stats["benchmark_event_pump_calls"] == 6
    assert stats["benchmark_redraw_requests"] == 4
    assert stats["benchmark_frame_submissions"] == 4
    assert stats["benchmark_redraw_pump_attempts"] == 5
    assert stats["benchmark_redraw_pump_alive"] == 4


def test_benchmark_telemetry_counts_one_live_redraw_pump_per_submission(
    monkeypatch,
) -> None:
    events: list[str] = []
    renderer = _runtime(events)
    renderer._record_event_pump = lambda: setattr(
        renderer,
        "_event_pump_calls",
        renderer._event_pump_calls + 1,
    )
    renderer._record_redraw_request = lambda: setattr(
        renderer,
        "_redraw_requests",
        renderer._redraw_requests + 1,
    )
    renderer._record_draw_pump = Open3DRuntimeMixin._record_draw_pump.__get__(renderer)
    renderer._event_pump_calls = 9
    renderer._redraw_requests = 5
    renderer._draw_pump_attempts = 2
    renderer._draw_pump_alive = 2
    renderer._gui_initialized = True
    renderer._gui_timer = _FakeTimer(events)
    renderer._observe_camera_state = lambda _phase: None
    renderer._set_far_clipping_plane = lambda: None
    application = SimpleNamespace(run_one_tick=lambda: True)
    monkeypatch.setattr(
        runtime_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )
    monkeypatch.setattr(runtime_module, "HAS_QTIMER", False)

    renderer.begin_benchmark_telemetry()
    for _ in range(2):
        renderer.begin_frame_update()
        assert renderer.end_frame_update() is True
        assert renderer.defer_until_next_render_turn(lambda: None) is True
        renderer._tick_o3d_gui()

    assert renderer._benchmark_frame_submissions == 2
    assert renderer._benchmark_redraw_pump_attempts == 2
    assert renderer._benchmark_redraw_pump_alive == 2
    assert renderer._event_pump_calls == 11
    assert renderer._redraw_requests == 7
