"""Tests for the System > Performance diagnostics panel."""

from __future__ import annotations

from types import SimpleNamespace

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.collapsible_section import CollapsibleSection
from visualizer.src.panels.performance_panel import PerformancePanel
from visualizer.src.playback import PlaybackCadenceTracker


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = str(text)


class _Cache:
    def __init__(self) -> None:
        self.frame_cache = SimpleNamespace(size=4, max_size=12)

    def is_override(self, _step: int) -> bool:
        return False

    def is_preloaded(self, _step: int) -> bool:
        return True

    def has_frame(self, _step: int) -> bool:
        return True


def _make_parent(**overrides):
    renderer_stats = {
        "recent_present_fps": 48.0,
        "effective_utp_fps": 48.0,
        "effective_present_fps": 60.0,
        "avg_update_to_present_ms": 20.8,
        "avg_draw_ms": 4.2,
        "present_jitter_ms": 1.1,
        "frame_drop_rate": 0.025,
    }
    playback_cadence = PlaybackCadenceTracker()
    playback_cadence.record_completion(1.0)
    playback_cadence.record_completion(1.04)
    parent = SimpleNamespace(
        renderer=SimpleNamespace(get_runtime_stats=lambda: renderer_stats),
        last_frame_duration_ms=24.5,
        frame_times=[0.02, 0.03],
        playback_cadence=playback_cadence,
        animation_running=True,
        animation_step=2,
        cache_service=_Cache(),
        animation_service=SimpleNamespace(
            preload_frame_count=3,
            preloading_started=True,
            preloading_completed=False,
            preload_duration=1.25,
        ),
        total_animation_steps=10,
        mpc_view_cache={1: object(), 2: object()},
        coverage_service=SimpleNamespace(stats=lambda: {"cache_size": 1, "max_cache_size": 50}),
        use_preload_mode=True,
        frame_source=object(),
        ui_manager=SimpleNamespace(is_panel_in_active_tab=lambda key: key == "performance"),
    )
    for key, value in overrides.items():
        setattr(parent, key, value)
    return parent


def test_performance_panel_has_no_stale_mode_controls(qapp):
    panel = PerformancePanel(_make_parent())
    group = panel.create_panel()

    assert "preload_mode_cb" not in panel.widgets
    assert "target_sync_cb" not in panel.widgets
    assert "restart_preload_btn" in panel.widgets
    assert "clear_cache_btn" in panel.widgets
    assert panel.widgets["clear_cache_btn"].text() == "Clear Frames"
    assert panel.widgets["clear_asset_cache_btn"].text() == "Clear Assets"

    group.deleteLater()


def test_diagnostics_log_level_reflects_runtime_and_supports_error_levels(qapp, monkeypatch):
    monkeypatch.setattr(
        "visualizer.src.panels.performance_panel.get_current_log_level_name",
        lambda: "ERROR",
    )
    panel = PerformancePanel(_make_parent())
    group = panel.create_panel()
    combo = panel.widgets["log_level_combo"]

    assert [combo.itemText(index) for index in range(combo.count())] == [
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
    ]
    assert combo.currentText() == "ERROR"

    group.deleteLater()


def test_live_stream_hides_restart_preload_action(qapp):
    from visualizer.src.io.frame_sources import LiveGrpcSource

    panel = PerformancePanel(_make_parent(frame_source=LiveGrpcSource("grpc://unit-test")))
    group = panel.create_panel()

    panel._sync_source_actions()

    assert panel.widgets["restart_preload_btn"].isHidden()
    group.deleteLater()


def test_performance_runtime_group_labels_renderer_submit_accurately(qapp):
    panel = PerformancePanel(_make_parent())
    group = panel.create_panel()

    labels = [label.text() for label in group.findChildren(type(panel.widgets["perf_draw_value"]))]

    assert "Renderer submit:" in labels
    assert "Draw callback:" not in labels
    group.deleteLater()


def test_performance_playback_rate_preserves_sub_one_update_precision(qapp):
    cadence = PlaybackCadenceTracker()
    cadence.record_completion(1.0)
    cadence.record_completion(3.5)
    cadence.record_completion(6.0)
    panel = PerformancePanel(_make_parent(playback_cadence=cadence))

    assert panel._format_playback_updates() == "0.4 updates/s"


def test_performance_refresh_timer_runs_only_when_section_expanded(qapp):
    panel = PerformancePanel(_make_parent())
    group = panel.create_panel()
    section = CollapsibleSection("Performance", start_open=False)
    section.content_layout().addWidget(group)
    section.show()
    panel.bind_section(section)
    qapp.processEvents()

    timer = panel._refresh_timer_obj()
    assert not timer.isActive()

    section.expand()
    qapp.processEvents()
    assert timer.isActive()

    section.collapse()
    qapp.processEvents()
    assert not timer.isActive()

    timer.stop()
    section.deleteLater()


def test_performance_refresh_skips_renderer_polling_when_inactive(qapp):
    calls = []

    parent = _make_parent(
        renderer=SimpleNamespace(get_runtime_stats=lambda: calls.append("stats") or {}),
        ui_manager=SimpleNamespace(is_panel_in_active_tab=lambda _key: False),
    )
    panel = PerformancePanel(parent)
    group = panel.create_panel()
    section = CollapsibleSection("Performance", start_open=True)
    section.content_layout().addWidget(group)
    section.show()
    panel.bind_section(section)

    panel.refresh_metrics()

    assert calls == []
    assert panel.widgets["perf_playback_updates_value"].text() == "--"

    panel._refresh_timer_obj().stop()
    section.deleteLater()


def test_performance_refresh_uses_existing_runtime_and_cache_stats(qapp):
    panel = PerformancePanel(_make_parent())
    group = panel.create_panel()
    section = CollapsibleSection("Performance", start_open=True)
    section.content_layout().addWidget(group)
    section.show()
    panel.bind_section(section)

    panel.refresh_metrics()

    assert panel.widgets["perf_playback_updates_value"].text() == "25 updates/s"
    assert panel.widgets["perf_frame_ms_value"].text() == "24.5 ms"
    assert panel.widgets["perf_avg_frame_ms_value"].text() == "25.0 ms"
    assert panel.widgets["perf_update_p95_value"].text() == "40.0 ms"
    assert panel.widgets["perf_present_value"].text() == "48 FPS, 20.8 ms latency"
    assert panel.widgets["perf_draw_value"].text() == "4.2 ms"
    assert panel.widgets["perf_jitter_value"].text() == "1.1 ms, 2.5%"
    assert panel.widgets["perf_frame_source_value"].text() == "preloaded"
    assert panel.widgets["perf_preloaded_value"].text() == "3/10"
    assert panel.widgets["perf_frame_cache_value"].text() == "4/12"
    assert panel.widgets["perf_viewmodel_cache_value"].text() == "2"
    assert panel.widgets["perf_coverage_cache_value"].text() == "1/50"
    assert panel.widgets["preload_status_label"].text() == "Preload: Loading (3/10), 1.2s"

    panel._refresh_timer_obj().stop()
    section.deleteLater()


def test_performance_refresh_includes_cache_telemetry(qapp):
    cache_service = SimpleNamespace(
        get_cache_telemetry=lambda: {
            "frame_cache_size": 3,
            "frame_cache_max_size": 8,
            "frame_cache_hits": 5,
            "frame_cache_misses": 2,
            "frame_cache_evictions": 1,
            "view_model_cache_size": 4,
            "mpc_line_cache_bytes": 2 * 1024 * 1024,
            "mpc_line_cache_max_bytes": 64 * 1024 * 1024,
            "mpc_line_cache_hits": 7,
            "mpc_line_cache_misses": 3,
            "mpc_line_cache_evictions": 0,
            "target_asset_cache_entries": 2,
            "target_asset_cache_max_entries": 8,
            "target_asset_cache_bytes": 2 * 1024 * 1024,
            "target_asset_cache_max_bytes": 256 * 1024 * 1024,
            "target_asset_cache_hits": 4,
            "target_asset_cache_misses": 1,
            "target_asset_cache_evictions": 0,
            "target_asset_cache_pending": 1,
            "static_asset_cache_aggregate": {
                "memory": {"bytes": 3 * 1024 * 1024, "max_bytes": 512 * 1024 * 1024},
                "disk": {"bytes": 5 * 1024 * 1024, "max_bytes": 768 * 1024 * 1024},
                "native": {"entries": 6, "bytes": 0, "max_bytes": 0},
            },
        },
        is_override=lambda _step: False,
        is_preloaded=lambda _step: False,
        has_frame=lambda _step: True,
    )
    panel = PerformancePanel(_make_parent(cache_service=cache_service))
    group = panel.create_panel()
    section = CollapsibleSection("Performance", start_open=True)
    section.content_layout().addWidget(group)
    section.show()
    panel.bind_section(section)

    panel.refresh_metrics()

    assert panel.widgets["perf_frame_cache_value"].text() == "3/8 h5/m2/e1"
    assert panel.widgets["perf_viewmodel_cache_value"].text() == "4 line 2.0 MB/64 MB h7/m3/e0"
    assert panel.widgets["perf_target_cache_value"].text() == "2/8, 2.0 MB/256 MB h4/m1/e0 p1"
    assert panel.widgets["perf_asset_cache_value"].text() == (
        "RAM 3.0 MB/512 MB | disk 5.0 MB/768 MB | native 6"
    )

    panel._refresh_timer_obj().stop()
    section.deleteLater()


def test_restart_preload_action_resets_state_and_starts_preloading():
    events = []
    label = _Label()
    viz = SimpleNamespace(
        use_preload_mode=False,
        preload_status_label=label,
        animation_service=SimpleNamespace(reset_preloading_state=lambda: events.append("reset")),
        start_preloading=lambda: events.append("start") or True,
        ui_manager=SimpleNamespace(
            panels={
                "performance": SimpleNamespace(refresh_metrics=lambda: events.append("refresh"))
            }
        ),
    )
    controller = UIController.__new__(UIController)
    controller.visualizer = viz

    controller.handle_restart_preload()

    assert viz.use_preload_mode is True
    assert events == ["reset", "start"]
    assert label.text == "Preload: Starting..."


def test_clear_performance_caches_action_uses_cache_service():
    events = []
    label = _Label()
    viz = SimpleNamespace(
        preload_status_label=label,
        cache_service=SimpleNamespace(
            clear_local_frame_caches=lambda reason: events.append(("clear", reason))
        ),
        _set_status_message=lambda text, timeout=0: events.append(("status", text, timeout)),
        ui_manager=SimpleNamespace(
            panels={
                "performance": SimpleNamespace(refresh_metrics=lambda: events.append("refresh"))
            }
        ),
    )
    controller = UIController.__new__(UIController)
    controller.visualizer = viz

    controller.handle_clear_performance_caches()

    assert events == [
        ("clear", "performance_panel"),
        ("status", "Frame caches cleared", 3000),
        "refresh",
    ]
    assert label.text == "Preload: Frame caches cleared"


def test_clear_asset_caches_action_uses_explicit_asset_service():
    events = []
    viz = SimpleNamespace(
        cache_service=SimpleNamespace(
            clear_static_asset_caches=lambda reason: events.append(("clear-assets", reason))
        ),
        _set_status_message=lambda text, timeout=0: events.append(("status", text, timeout)),
        ui_manager=SimpleNamespace(
            panels={
                "performance": SimpleNamespace(refresh_metrics=lambda: events.append("refresh"))
            }
        ),
    )
    controller = UIController.__new__(UIController)
    controller.visualizer = viz

    controller.handle_clear_asset_caches()

    assert events == [
        ("clear-assets", "performance_panel"),
        ("status", "Asset caches cleared; the next load may be slower", 5000),
        "refresh",
    ]
