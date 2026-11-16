"""Render failures remain observable without flooding the log or losing recovery."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

from visualizer.src.renderers.pygfx import runtime
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer


def test_render_failures_are_rate_limited_and_clear_after_recovery(monkeypatch, caplog):
    renderer = PygfxRenderer(SimpleNamespace(animation_running=False))
    renderer._initialized = True
    backend = Mock()
    renderer._renderer = backend
    renderer._scene = object()
    renderer._camera = object()
    renderer._update_camera_depth_precision = lambda: None
    renderer._update_headlight_pose = lambda: None
    renderer._first_present_at = 1.0
    renderer.set_hud_overlay = Mock()
    renderer.clear_hud_overlay = Mock()
    clock = [10.0]
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: clock[0])
    caplog.set_level(logging.WARNING, logger=runtime.__name__)
    backend.render.side_effect = RuntimeError("device submission failed")

    for _ in range(3):
        renderer._animate()
    assert renderer._render_failures == renderer._consecutive_render_failures == 3
    assert len(caplog.records) == 1 and caplog.records[0].exc_info is not None
    assert "device submission failed" in caplog.records[0].message
    renderer.set_hud_overlay.assert_called_once()
    assert renderer.set_hud_overlay.call_args.args == ("render_error",)

    clock[0] += renderer._RENDER_FAILURE_LOG_INTERVAL_S
    renderer._animate()
    assert len(caplog.records) == 2
    backend.render.side_effect = None
    renderer._animate()
    assert renderer._consecutive_render_failures == 0
    assert renderer._render_successes == 1
    renderer.clear_hud_overlay.assert_called_once_with("render_error")

    backend.render.side_effect = RuntimeError("new failure")
    renderer._animate()
    assert len(caplog.records) == 3
    assert renderer.set_hud_overlay.call_count == 2
