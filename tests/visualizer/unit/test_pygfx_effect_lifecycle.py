"""Explicit capture prepares optional effects before its synchronous draw."""

from types import SimpleNamespace

import numpy as np

from visualizer.src.renderers.pygfx.capture import PygfxCaptureMixin


def test_immediate_export_prepares_effects_in_live_render_order():
    events = []
    expected = np.zeros((2, 3, 4), dtype=np.uint8)
    owner = SimpleNamespace(
        _scene=object(),
        _camera=object(),
        _update_camera_depth_precision=lambda: events.append("depth"),
        _renderer=SimpleNamespace(
            render=lambda *args, **kw: events.append("render"),
            snapshot=lambda: expected,
        ),
    )
    for name in ("dusk", "scene_guides", "compass", "section_contours"):
        setattr(owner, f"_sync_{name}", lambda name=name: events.append(name))
    result = PygfxCaptureMixin._capture_renderer_snapshot(owner)
    assert result is expected
    assert events == ["dusk", "compass", "section_contours", "depth", "scene_guides", "render"]
