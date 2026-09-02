from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager
from visualizer.src.renderers.pygfx.renderer import _default_ibl_dir


def test_pygfx_default_ibl_dir_uses_repo_library_assets():
    ibl_dir = _default_ibl_dir()

    assert ibl_dir.name == "ibl"
    assert ibl_dir.parent.name == "libraries"
    assert (ibl_dir / "neutral_outdoor.hdr").is_file()
    assert "visualizer/src/libraries" not in str(ibl_dir)


def test_load_ibl_from_cached_cubemap_does_not_require_hdr_metadata(tmp_path: Path):
    created: dict[str, object] = {}

    def _texture(data, **kwargs):
        created["data"] = np.asarray(data, dtype=np.float32)
        created["kwargs"] = kwargs
        return SimpleNamespace(data=data, kwargs=kwargs)

    manager = PygfxIBLManager(SimpleNamespace(Texture=_texture), Path(tmp_path))
    manager._face_size = 4

    cubemap = np.full((6, 4, 4, 4), 0.5, dtype=np.float32)
    manager._write_cubemap_cache("neutral_outdoor", cubemap)

    texture = manager.load_ibl("neutral_outdoor")

    assert texture is not None
    np.testing.assert_allclose(created["data"], cubemap, atol=1e-3)
    assert created["kwargs"]["size"] == (4, 4, 6)
    assert manager._current_name == "neutral_outdoor"


def test_pygfx_end_frame_update_records_breakdown():
    pytest.importorskip("pygfx")

    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _CanvasStub:
        def __init__(self, renderer) -> None:
            self.calls = 0
            self.renderer = renderer

        def force_draw(self) -> None:
            self.calls += 1
            self.renderer._draw_callbacks_received += 1
            self.renderer._last_draw_callback_total_ms = 2.0
            self.renderer._last_renderer_submit_ms = 1.25

    renderer = PygfxRenderer(SimpleNamespace(animation_running=False))
    renderer._initialized = True
    renderer._canvas = _CanvasStub(renderer)
    renderer._frame_update_start = renderer._created_at

    assert renderer.end_frame_update() is True

    breakdown = renderer.get_last_end_frame_update_breakdown()
    assert breakdown["force_draw_ms"] >= 0.0
    assert breakdown["geometry_update_ms"] >= 0.0
    assert breakdown["total_ms"] >= breakdown["force_draw_ms"]
    assert breakdown["force_draw_callback_count"] == 1.0
    assert breakdown["draw_callback_total_ms"] == 2.0
    assert breakdown["renderer_submit_ms"] == 1.25
    assert breakdown["canvas_present_residual_ms"] >= 0.0


def test_pygfx_benchmark_telemetry_reports_counter_deltas():
    pytest.importorskip("pygfx")

    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace(animation_running=False))
    renderer._draw_callbacks_received = 5
    renderer._render_attempts = 5
    renderer._render_successes = 4
    renderer._redraw_requests = 7
    renderer._blocking_frame_count = 2
    renderer._blocking_force_draw_callbacks = 3
    renderer._blocking_force_draw_contaminated = 1
    renderer.begin_benchmark_telemetry()

    assert renderer._blocking_frame_count == 0
    assert renderer._blocking_force_draw_callbacks == 0
    assert renderer._blocking_force_draw_contaminated == 0

    renderer._draw_callbacks_received += 3
    renderer._render_attempts += 3
    renderer._render_successes += 2
    renderer._redraw_requests += 4

    stats = renderer.get_runtime_stats()
    assert stats["benchmark_draw_callbacks"] == 3
    assert stats["benchmark_present_attempts"] == 3
    assert stats["benchmark_present_successes"] == 2
    assert stats["benchmark_redraw_requests"] == 4
