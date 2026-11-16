"""Unit tests for TelemetryController scenario/file summary text."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from visualizer.src.controllers.telemetry_controller import TelemetryController
from visualizer.src.io.frame_sources import FileSource, LiveGrpcSource
from visualizer.src.playback import PlaybackCadenceTracker


class _DummyLabel:
    def __init__(self) -> None:
        self.text = ""
        self.tooltip = ""
        self.visible = False

    def setText(self, value: str) -> None:
        self.text = value

    def setToolTip(self, value: str) -> None:
        self.tooltip = value

    def setVisible(self, value: bool) -> None:
        self.visible = value


def _cadence(*timestamps: float) -> PlaybackCadenceTracker:
    tracker = PlaybackCadenceTracker()
    for timestamp in timestamps:
        tracker.record_completion(timestamp)
    return tracker


def test_refresh_status_telemetry_uses_backend_neutral_playback_cadence():
    fps_label = _DummyLabel()
    viz = SimpleNamespace(
        status_fps_label=fps_label,
        renderer=SimpleNamespace(get_runtime_stats=lambda: {"recent_present_fps": 2.0}),
        playback_cadence=_cadence(1.0, 1.02, 1.04),
        animation_running=True,
    )

    controller = _make_controller(viz)
    controller.refresh_status_telemetry()

    assert fps_label.text == "Playback updates: 50/s"
    assert "not renderer callback or display FPS" in fps_label.tooltip


def test_refresh_status_telemetry_preserves_last_playback_rate_when_paused():
    fps_label = _DummyLabel()
    viz = SimpleNamespace(
        status_fps_label=fps_label,
        playback_cadence=_cadence(1.0, 1.04, 1.08),
        animation_running=False,
    )

    _make_controller(viz).refresh_status_telemetry()

    assert fps_label.text == "Playback updates: paused (last 25/s)"


def test_refresh_status_telemetry_preserves_sub_one_fps_precision():
    fps_label = _DummyLabel()
    viz = SimpleNamespace(
        status_fps_label=fps_label,
        playback_cadence=_cadence(1.0, 3.5, 6.0),
        animation_running=True,
    )

    _make_controller(viz).refresh_status_telemetry()

    assert fps_label.text == "Playback updates: 0.4/s"


def test_frame_timing_does_not_record_non_playback_pipeline_updates():
    fps_label = _DummyLabel()
    tracker = PlaybackCadenceTracker()
    viz = SimpleNamespace(
        status_fps_label=fps_label,
        playback_cadence=tracker,
        animation_running=True,
        frame_times=[0.01],
        last_frame_duration_ms=None,
    )
    controller = _make_controller(viz)

    controller.handle_frame_timing_update(0, 0.01)
    controller.handle_frame_timing_update(1, 0.01)

    assert tracker.frames_per_second() is None
    assert fps_label.text == "Playback updates: measuring"


def test_frame_timing_notifies_animation_controller_of_successful_pipeline():
    completed_steps = []
    fps_label = _DummyLabel()
    viz = SimpleNamespace(
        status_fps_label=fps_label,
        playback_cadence=PlaybackCadenceTracker(),
        animation_running=True,
        animation_controller=SimpleNamespace(
            record_completed_playback_tick=lambda step: completed_steps.append(step)
        ),
        frame_times=[0.01],
        last_frame_duration_ms=None,
    )

    _make_controller(viz).handle_frame_timing_update(7, 0.01)

    assert completed_steps == [7]


def _make_controller(visualizer) -> TelemetryController:
    parent = SimpleNamespace(visualizer=visualizer)
    return TelemetryController(parent)


def test_update_frame_context_allows_missing_cache_service():
    slider = _DummyLabel()
    total_steps = _DummyLabel()
    viz = SimpleNamespace(
        total_animation_steps=3,
        total_steps_label=total_steps,
        step_slider=slider,
        last_frame_duration_ms=None,
    )

    controller = _make_controller(viz)
    controller.update_frame_context(1, raw_frame={"timestamp_ns": 1234})

    assert total_steps.text == "/ 3"
    assert slider.tooltip == "Frame 2 / 3\nTimestamp: 1234 ns"
    assert viz._last_frame_tooltip_context["source_flags"] == []


@pytest.mark.parametrize(
    "frame, expected",
    [
        ({"timestamp_ns": 511000000}, "Simulation time: 0.511000 s"),
        ({"timestamp_s": 1.024}, "Simulation time: 1.024000 s"),
        ({"timestamp_ns": 0}, "Simulation time: 0.000000 s"),
        ({"timestamp_s": float("nan")}, ""),
        ({"timestamp_s": float("inf")}, ""),
        ({"timestamp_s": True}, ""),
        ({"timestamp_s": None}, ""),
        ({"timestamp": 64}, ""),
        ({}, ""),
    ],
)
def test_simulation_time_is_shown_only_with_valid_recorded_metadata(frame, expected):
    label = _DummyLabel()
    hud_times = []
    viz = SimpleNamespace(
        status_time_label=label,
        renderer=SimpleNamespace(set_simulation_time_label=hud_times.append),
    )
    controller = _make_controller(viz)
    controller.update_frame_context(123, raw_frame=frame)
    assert label.text == expected
    assert label.visible is bool(expected)
    assert viz.scene_time_text == expected
    assert hud_times == [expected]


def test_simulation_time_disappears_when_next_frame_has_no_timestamp():
    label = _DummyLabel()
    hud_times = []
    viz = SimpleNamespace(
        status_time_label=label,
        renderer=SimpleNamespace(set_simulation_time_label=hud_times.append),
    )
    controller = _make_controller(viz)
    controller.update_frame_context(0, raw_frame={"timestamp_s": 1.0})
    controller.update_frame_context(1, raw_frame={})

    assert label.text == ""
    assert not label.visible
    assert viz.scene_time_text == ""
    assert hud_times == ["Simulation time: 1.000000 s", ""]


def test_source_change_clears_time_in_status_bar_and_viewport():
    label = _DummyLabel()
    hud_times = []
    viz = SimpleNamespace(
        frame_source=object(),
        status_time_label=label,
        renderer=SimpleNamespace(set_simulation_time_label=hud_times.append),
    )
    controller = _make_controller(viz)
    controller.update_file_source_summary()
    controller.update_frame_context(0, raw_frame={"timestamp_s": 1.0})
    controller.update_file_source_summary()
    assert label.visible

    viz.frame_source = object()
    controller.update_file_source_summary()

    assert label.text == ""
    assert not label.visible
    assert viz.scene_time_text == ""
    assert hud_times == ["", "Simulation time: 1.000000 s", ""]


def test_update_frame_context_accepts_narrow_cache_service():
    slider = _DummyLabel()
    cache = SimpleNamespace(has_frame=lambda step: step == 2)
    viz = SimpleNamespace(
        total_animation_steps=4,
        total_steps_label=None,
        step_slider=slider,
        cache_service=cache,
        use_preload_mode=True,
        last_frame_duration_ms=None,
    )

    controller = _make_controller(viz)
    controller.update_frame_context(2)

    assert slider.tooltip == "Frame 3 / 4\nsource: cached"
    assert viz._last_frame_tooltip_context["source_flags"] == ["source: cached"]


def test_noop_navigation_keeps_the_displayed_scene_time():
    label = _DummyLabel()
    viz = SimpleNamespace(status_time_label=label)
    controller = _make_controller(viz)
    controller.update_frame_context(64, raw_frame={"timestamp_s": 0.256})
    controller.update_frame_context(64)
    assert label.text == "Simulation time: 0.256000 s"


def test_update_frame_context_ignores_faulty_cache_source_predicate():
    slider = _DummyLabel()

    def _raise(_step):
        raise RuntimeError("cache unavailable")

    cache = SimpleNamespace(is_override=_raise, has_frame=lambda _step: True)
    viz = SimpleNamespace(
        total_animation_steps=1,
        total_steps_label=None,
        step_slider=slider,
        cache_service=cache,
        use_preload_mode=False,
        last_frame_duration_ms=None,
    )

    controller = _make_controller(viz)
    controller.update_frame_context(0)

    assert slider.tooltip == "Frame 1 / 1\nsource: cached"


def test_online_summary_uses_scene_id_and_source():
    label = _DummyLabel()
    scenario = SimpleNamespace(
        scene_id="etoile/etoile.xml",
        scene_source="library",
        scene_spec={"id": "etoile/etoile.xml", "source": "library"},
    )
    frame_source = LiveGrpcSource("grpc://localhost:50051")
    viz = SimpleNamespace(
        status_scenario_label=label,
        scenario_config=scenario,
        frame_source=frame_source,
    )

    controller = _make_controller(viz)
    controller.update_file_source_summary()

    assert label.text == "etoile/etoile.xml (library) · Disconnected · stream inactive"
    assert label.tooltip == "grpc://localhost:50051"


def test_file_summary_uses_scene_spec_fallback_and_frame_count():
    label = _DummyLabel()
    scenario = SimpleNamespace(
        scene_id=None,
        scene_source=None,
        scene_spec={"id": "LectureRoom/LectureRoom.xml", "source": "library"},
    )
    frame_source = FileSource(Path("/tmp/scenario_root"), "frames", "h5")
    frame_source.list_frames = lambda: [0, 1, 2]  # type: ignore[method-assign]
    viz = SimpleNamespace(
        status_scenario_label=label,
        scenario_config=scenario,
        frame_source=frame_source,
    )

    controller = _make_controller(viz)
    controller.update_file_source_summary()

    assert label.text == "LectureRoom/LectureRoom.xml (library) · HDF5 · 3 frames"
    assert label.tooltip == str(Path("/tmp/scenario_root"))


def test_source_summary_refreshes_data_source_polling_policy() -> None:
    refreshes = []
    source_panel = SimpleNamespace(refresh_source_status=lambda: refreshes.append("source"))
    viz = SimpleNamespace(
        status_scenario_label=None,
        ui_manager=SimpleNamespace(panels={"data_source": source_panel}),
    )

    _make_controller(viz).update_file_source_summary()

    assert refreshes == ["source"]


def test_sparse_playback_reports_ordinal_and_source_id_separately():
    slider = _DummyLabel()
    viz = SimpleNamespace(
        total_animation_steps=3, step_slider=slider, get_animation_step_index=[63, 127, 191].index
    )
    _make_controller(viz).update_frame_context(191, raw_frame={"timestamp_ns": 191000000})
    assert slider.tooltip.startswith("Frame 3 / 3\nSource frame ID: 191")
    assert viz.scene_time_text == "Simulation time: 0.191000 s"
