"""Tests for visualizer runtime-state timer construction."""

from PySide6.QtCore import Qt

from visualizer.src.app.state_bootstrap import (
    _create_animation_timer,
    _create_slider_scrub_timer,
)


def test_animation_timer_uses_precise_cadence() -> None:
    """Playback uses precise, serial timeouts instead of a busy repeating timer."""
    timer = _create_animation_timer(lambda: None)

    assert timer.timerType() == Qt.PreciseTimer
    assert timer.isSingleShot() is True


def test_slider_scrub_timer_uses_precise_single_shot_throttling() -> None:
    timer = _create_slider_scrub_timer(lambda: None)

    assert timer.timerType() == Qt.PreciseTimer
    assert timer.isSingleShot() is True
