"""Playback-timing ownership tests for ``UIController``."""

from types import SimpleNamespace

from visualizer.src.controllers.ui_controller import UIController


def test_playback_timing_change_resets_timer_and_status_while_running():
    events = []
    viz = SimpleNamespace(
        animation_running=True,
        animation_controller=SimpleNamespace(
            reset_playback_cadence=lambda: events.append("reset"),
            restart_playback_timer=lambda: events.append("restart"),
        ),
    )
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    controller._telemetry_ctrl = SimpleNamespace(
        refresh_status_telemetry=lambda: events.append("refresh")
    )

    controller.handle_playback_timing_changed()

    assert events == ["reset", "restart", "refresh"]
