"""Regression tests for bounded forced-frame retries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import visualizer.visualizer as visualizer_module
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.visualizer import MAX_FORCED_FRAME_RETRIES, OrchavVisualizer


def test_frame_retry_uses_backoff_and_generation_guard(monkeypatch) -> None:
    callbacks = []
    monkeypatch.setattr(
        visualizer_module,
        "QTimer",
        SimpleNamespace(singleShot=lambda delay, callback: callbacks.append((delay, callback))),
    )
    schedule_update = Mock()
    viz = SimpleNamespace(
        _shutdown_started=False,
        _frame_retry_count=0,
        _frame_retry_token=3,
        _frame_retry_pending=False,
        force_update_next_frame=False,
        schedule_update=schedule_update,
        _viewport_mode="detached",
    )

    assert OrchavVisualizer.schedule_frame_retry(viz, "not ready") is True
    assert callbacks[0][0] == 16
    assert viz._frame_retry_pending is True

    viz._frame_retry_token = 4
    callbacks[0][1]()
    schedule_update.assert_not_called()


def test_frame_retry_limit_enters_terminal_state_without_rescheduling() -> None:
    set_status = Mock()
    viz = SimpleNamespace(
        _shutdown_started=False,
        _frame_retry_count=MAX_FORCED_FRAME_RETRIES,
        _frame_retry_token=1,
        _frame_retry_pending=False,
        force_update_next_frame=True,
        _set_status_message=set_status,
        _viewport_mode="detached",
    )

    assert OrchavVisualizer.schedule_frame_retry(viz, "renderer rejected frame") is False
    assert viz.force_update_next_frame is False
    assert viz._frame_retry_pending is False
    set_status.assert_called_once()


def test_idle_redraw_timer_cannot_consume_pending_pipeline_retry(monkeypatch) -> None:
    callbacks = []
    monkeypatch.setattr(
        visualizer_module,
        "QTimer",
        SimpleNamespace(singleShot=lambda _delay, callback: callbacks.append(callback)),
    )
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        update_renderer=Mock(),
    )
    process_step = Mock(return_value=True)
    viz = SimpleNamespace(
        _shutdown_started=False,
        _frame_retry_count=0,
        _frame_retry_token=7,
        _frame_retry_pending=False,
        force_update_next_frame=True,
        schedule_update=Mock(),
        _viewport_mode="detached",
        vis_initialized=True,
        vis=object(),
        renderer=renderer,
        ready=True,
        _scene_only_mode=False,
        app_state=object(),
        animation_step=4,
        _process_frame_step=process_step,
        update_pending=True,
    )

    assert OrchavVisualizer.schedule_frame_retry(viz, "transient") is True
    callbacks[0]()
    OrchavVisualizer.update_visualizer(viz)

    renderer.update_renderer.assert_not_called()
    assert viz.force_update_next_frame is True

    OrchavVisualizer._flush_update(viz)

    process_step.assert_called_once_with(4)
    assert viz._frame_retry_pending is False
