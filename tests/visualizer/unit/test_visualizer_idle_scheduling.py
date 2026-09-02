from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from visualizer.src.model import make_text_label_state
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.object_appearance_service import ObjectAppearanceService
from visualizer.visualizer import OrchavVisualizer


class _RendererStub:
    def __init__(self, *, external_event_pump: bool) -> None:
        self.capabilities = RendererCapabilities(external_event_pump=external_event_pump)
        self.poll_calls = 0
        self.update_calls = 0

    def poll_events(self) -> None:
        self.poll_calls += 1

    def update_renderer(self) -> None:
        self.update_calls += 1


class _TimerStub:
    def __init__(self, *, active: bool = False) -> None:
        self.active = active
        self.start_calls: list[int] = []
        self.stop_calls = 0

    def isActive(self) -> bool:
        return self.active

    def start(self, interval_ms: int) -> None:
        self.active = True
        self.start_calls.append(interval_ms)

    def stop(self) -> None:
        self.active = False
        self.stop_calls += 1


def _make_viz(renderer: _RendererStub, *, force_update: bool = False):
    return SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        force_update_next_frame=force_update,
    )


def test_update_visualizer_polls_only_when_renderer_requires_it():
    renderer = _RendererStub(external_event_pump=True)
    viz = _make_viz(renderer)

    OrchavVisualizer.update_visualizer(viz)

    assert renderer.poll_calls == 1
    assert renderer.update_calls == 0


def test_update_visualizer_renders_when_force_update_is_set():
    renderer = _RendererStub(external_event_pump=False)
    viz = _make_viz(renderer, force_update=True)

    OrchavVisualizer.update_visualizer(viz)

    assert renderer.update_calls == 1
    assert viz.force_update_next_frame is False


def test_cancel_scheduled_update_stops_coalesce_timer():
    timer = _TimerStub(active=True)
    viz = SimpleNamespace(update_pending=True, update_timer_coalesce=timer)

    OrchavVisualizer.cancel_scheduled_update(viz)

    assert viz.update_pending is False
    assert timer.stop_calls == 1


def test_flush_update_reaches_pipeline_for_scene_only_coverage():
    processed_steps: list[int] = []
    viz = SimpleNamespace(
        update_pending=True,
        ready=False,
        _scene_only_mode=True,
        vis_initialized=True,
        app_state=object(),
        animation_step=0,
        _process_frame_step=lambda step: processed_steps.append(step),
    )

    OrchavVisualizer._flush_update(viz)

    assert viz.update_pending is False
    assert processed_steps == [0]


def test_flush_update_still_waits_for_frame_source_outside_scene_only_mode():
    processed_steps: list[int] = []
    viz = SimpleNamespace(
        update_pending=True,
        ready=False,
        _scene_only_mode=False,
        _process_frame_step=lambda step: processed_steps.append(step),
    )

    OrchavVisualizer._flush_update(viz)

    assert viz.update_pending is False
    assert processed_steps == []


def test_set_background_update_enabled_stops_timer():
    timer = _TimerStub(active=True)
    viz = SimpleNamespace(update_timer=timer, _idle_poll_interval_ms=16)

    OrchavVisualizer.set_background_update_enabled(viz, False)

    assert timer.stop_calls == 1
    assert timer.active is False


def test_set_background_update_enabled_restarts_timer_when_inactive():
    timer = _TimerStub(active=False)
    viz = SimpleNamespace(update_timer=timer, _idle_poll_interval_ms=16)

    OrchavVisualizer.set_background_update_enabled(viz, True)

    assert timer.start_calls == [16]
    assert timer.active is True


def _make_startup_preload_viz(*, scene_boot_logged: bool = False, timer_active: bool = False):
    timer = _TimerStub(active=timer_active)
    viz = SimpleNamespace(
        use_preload_mode=True,
        _scene_boot_logged=scene_boot_logged,
        _startup_preload_requested=False,
        _startup_preload_delay_ms=1000,
        _startup_preload_timer=timer,
        preload_runs=0,
    )

    def _schedule_startup_preload() -> None:
        OrchavVisualizer._schedule_startup_preload(viz)

    def _start_preloading() -> None:
        viz.preload_runs += 1

    viz._schedule_startup_preload = _schedule_startup_preload
    viz.start_preloading = _start_preloading
    return viz, timer


def test_request_startup_preload_waits_for_scene_boot():
    viz, timer = _make_startup_preload_viz(scene_boot_logged=False)

    OrchavVisualizer.request_startup_preload(viz, delay_ms=250)

    assert viz._startup_preload_requested is True
    assert viz._startup_preload_delay_ms == 250
    assert timer.start_calls == []


def test_on_scene_boot_completed_arms_startup_preload_timer():
    viz, timer = _make_startup_preload_viz(scene_boot_logged=False)
    viz._startup_preload_requested = True
    viz._startup_preload_delay_ms = 250

    OrchavVisualizer._on_scene_boot_completed(viz)

    assert timer.start_calls == [250]


def test_cancel_startup_preload_stops_timer():
    viz, timer = _make_startup_preload_viz(timer_active=True)
    viz._startup_preload_requested = True

    OrchavVisualizer.cancel_startup_preload(viz)

    assert viz._startup_preload_requested is False
    assert timer.stop_calls == 1


def test_run_startup_preload_calls_start_preloading_once():
    viz, _ = _make_startup_preload_viz()
    viz._startup_preload_requested = True

    OrchavVisualizer._run_startup_preload(viz)

    assert viz._startup_preload_requested is False
    assert viz.preload_runs == 1


def test_startup_timing_profile_roundtrips_and_rounds():
    viz = SimpleNamespace(
        startup_stage_timings_ms=OrderedDict(),
        startup_first_frame_timings_ms={},
        startup_detail_timings_ms=OrderedDict(),
    )

    OrchavVisualizer.reset_startup_timing_profile(viz)
    OrchavVisualizer.record_startup_stage_timing(viz, "cleanup_previous_scene_ms", 12.3456)
    OrchavVisualizer.set_startup_first_frame_timing(viz, {"apply_ms": 7.8912})
    OrchavVisualizer.set_startup_detail_timing(
        viz,
        "render_initial_scene_breakdown_ms",
        {"foo_ms": 3.2109},
    )

    profile = OrchavVisualizer.get_startup_timing_profile(viz)

    assert profile == {
        "scenario_open_stages_ms": {"cleanup_previous_scene_ms": 12.346},
        "first_frame_pipeline_ms": {"apply_ms": 7.891},
        "render_initial_scene_breakdown_ms": {"foo_ms": 3.211},
    }


def _make_preload_completion_viz(*, benchmark_active: bool, statistics_enabled: bool = True):
    callbacks: dict[str, object] = {}
    warmer_stops: list[int] = []
    detect_calls: list[int] = []
    data_source_updates: list[int] = []
    status_messages: list[str] = []

    viz = SimpleNamespace(
        animation_service=SimpleNamespace(
            start_preloading=lambda **kwargs: callbacks.update(kwargs) or True
        ),
        status_progress_bar=None,
        preload_status_label=None,
        _vm_warmer=SimpleNamespace(stop=lambda: warmer_stops.append(1), enqueue=lambda *_: None),
        _set_status_message=lambda text, *_args, **_kwargs: status_messages.append(text),
        _panel_enabled=lambda key, default=True: (
            statistics_enabled if key == "statistics" else default
        ),
        scenario_config=SimpleNamespace(
            visualizer_cfg={"panels": {"statistics": {"enabled": statistics_enabled}}}
        ),
        ui_manager=SimpleNamespace(
            panels={
                "statistics": SimpleNamespace(),
                "data_source": SimpleNamespace(
                    _update_status=lambda: data_source_updates.append(1)
                ),
            }
        ),
        pipeline=SimpleNamespace(benchmark_recorder=object() if benchmark_active else None),
        detect_mpc_frames=lambda: detect_calls.append(1),
    )
    return (
        viz,
        callbacks,
        warmer_stops,
        detect_calls,
        data_source_updates,
        status_messages,
    )


def test_start_preloading_does_not_feed_preloaded_frames_to_statistics():
    (
        viz,
        callbacks,
        warmer_stops,
        detect_calls,
        data_source_updates,
        status_messages,
    ) = _make_preload_completion_viz(benchmark_active=False)

    assert OrchavVisualizer.start_preloading(viz) is True
    callbacks["on_complete"]([(0, {})], 0.5)

    assert warmer_stops == [1]
    assert detect_calls == [1]
    assert data_source_updates == [1]
    assert status_messages == ["Preload complete: 1 frames in 0.5s"]


def test_start_preloading_skips_statistics_panel_during_benchmark():
    (
        viz,
        callbacks,
        warmer_stops,
        detect_calls,
        data_source_updates,
        status_messages,
    ) = _make_preload_completion_viz(benchmark_active=True)

    assert OrchavVisualizer.start_preloading(viz) is True
    callbacks["on_complete"]([(0, {})], 0.5)

    assert warmer_stops == [1]
    assert detect_calls == [1]
    assert data_source_updates == [1]
    assert status_messages == ["Preload complete: 1 frames in 0.5s"]


def test_start_preloading_skips_statistics_panel_when_disabled():
    (
        viz,
        callbacks,
        warmer_stops,
        detect_calls,
        data_source_updates,
        status_messages,
    ) = _make_preload_completion_viz(benchmark_active=False, statistics_enabled=False)

    assert OrchavVisualizer.start_preloading(viz) is True
    callbacks["on_complete"]([(0, {})], 0.5)

    assert warmer_stops == [1]
    assert detect_calls == [1]
    assert data_source_updates == [1]
    assert status_messages == ["Preload complete: 1 frames in 0.5s"]


def test_set_building_label_visibility_creates_labels_on_demand():
    calls: list[str] = []
    updates: list[str] = []
    entry = {"entry_type": "mesh", "show_label": False, "name": "B1", "mesh": object()}
    label = make_text_label_state("bldg_label_0", "B1", [0.5, 0.5, 0.5])

    viz = SimpleNamespace(
        vis_initialized=True,
        vis=object(),
        building_labels=[],
        mesh_entries=[entry],
        scene_service=SimpleNamespace(
            ensure_building_labels_created=lambda: viz.building_labels.append(label)
        ),
        renderer=SimpleNamespace(
            ensure_object=lambda obj: calls.append(obj.id) or True,
            update_renderer=lambda: updates.append("update"),
        ),
        _is_geometry_currently_visible=lambda _geometry: False,
    )

    ObjectAppearanceService(viz).set_building_label_visibility(entry, True)

    assert calls == ["bldg_label_0"]
    assert updates == ["update"]
