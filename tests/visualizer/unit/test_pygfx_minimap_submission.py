"""Minimap fallback must flush the existing main pass, never submit it twice."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer


@pytest.mark.parametrize(
    "enabled,hud,camera,bounds,expected",
    [
        (False, True, True, True, [("main", True)]),
        (True, False, True, True, [("main", False), ("flush", True)]),
        (True, True, False, True, [("main", False), ("flush", True)]),
        (True, True, True, False, [("main", False), ("flush", True)]),
        (True, True, True, True, [("main", False), ("mini", False), ("mini", True)]),
    ],
)
def test_main_scene_submitted_once(enabled, hud, camera, bounds, expected):
    calls = []
    r = PygfxRenderer(SimpleNamespace(animation_running=False))
    r._initialized = True
    r._renderer = SimpleNamespace(
        render=lambda *args, **kwargs: calls.append(("main", kwargs["flush"])),
        flush=lambda: calls.append(("flush", True)),
    )
    r._scene = object()
    r._camera = object()
    r._update_camera_depth_precision = lambda: None
    r._update_headlight_pose = lambda: None
    r._viewport_hud_policy = lambda: SimpleNamespace(enabled=hud)
    r._first_present_at = time.perf_counter()
    r._minimap_enabled = enabled
    r._minimap_camera = object() if camera else None
    r._minimap_viewport = SimpleNamespace(
        render=lambda *args, **kwargs: calls.append(("mini", kwargs["flush"]))
    )
    r.compute_scene_bounds = lambda **kwargs: object() if bounds else None
    r._update_minimap_camera = lambda bounds: None
    r._compute_minimap_rect = lambda: (0, 0, 200, 200)
    r._ensure_minimap_gpu_overlay = lambda *args: None
    r._update_minimap_gpu_overlay_state = lambda *args: None
    r._animate()
    assert r._render_failures == 0
    assert calls == expected


@pytest.mark.parametrize("fail", [False, True])
def test_minimap_omits_main_camera_guide_cards_and_restores_visibility(fail):
    guide, hidden = SimpleNamespace(visible=True), SimpleNamespace(visible=False)
    events = []
    owner = SimpleNamespace(
        _viewport_hud_policy=lambda: SimpleNamespace(enabled=True),
        _minimap_enabled=True,
        _renderer=object(),
        _scene=object(),
        _minimap_camera=object(),
        _minimap_overlay_scene=object(),
        _minimap_overlay_camera=object(),
        compute_scene_bounds=lambda **kw: object(),
        _update_minimap_camera=lambda bounds: None,
        _compute_minimap_rect=lambda: (0, 0, 200, 200),
        _ensure_minimap_gpu_overlay=lambda *args: None,
        _update_minimap_gpu_overlay_state=lambda *args: None,
        _scene_guide_labels=lambda: (guide, hidden),
    )

    def render(scene, camera, **kwargs):
        if scene is owner._scene:
            assert not guide.visible and not hidden.visible
            if fail:
                raise RuntimeError("minimap draw failed")
        events.append(scene)

    owner._minimap_viewport = SimpleNamespace(render=render)
    if fail:
        with pytest.raises(RuntimeError, match="minimap draw failed"):
            PygfxOverlayMixin._render_minimap(owner)
    else:
        assert PygfxOverlayMixin._render_minimap(owner)
        assert events == [owner._scene, owner._minimap_overlay_scene]
    assert guide.visible and not hidden.visible


@pytest.mark.parametrize("fail", [False, True])
def test_high_resolution_minimap_keeps_guide_cards_only_in_main_view(monkeypatch, fail):
    from visualizer.src.renderers.pygfx import high_resolution

    guide, hidden = SimpleNamespace(visible=True), SimpleNamespace(visible=False)
    owner = SimpleNamespace(
        _coerce_rgb8_image=lambda image: image,
        _capture_renderer_snapshot=lambda **kw: np.zeros((4, 6, 3), np.uint8),
        _canvas=SimpleNamespace(get_logical_size=lambda: (6, 4)),
        _scene=object(),
        _camera=object(),
        _minimap_camera=object(),
        _minimap_overlay_scene=object(),
        _minimap_overlay_camera=object(),
        _minimap_enabled=True,
        _viewport_hud_policy=lambda: SimpleNamespace(enabled=True),
        compute_scene_bounds=lambda **kw: object(),
        _compute_minimap_rect=lambda: (0, 0, 2, 2),
        _alpha_composite_rgba=lambda *args, **kw: None,
        _composite_visible_qt_overlays=lambda image: image,
        _scene_guide_labels=lambda: (guide, hidden),
    )
    cameras = []

    def render(_owner, _scene, camera, *, output_size, **kw):
        cameras.append(camera)
        assert guide.visible is (camera is owner._camera)
        assert not hidden.visible
        if fail and camera is owner._minimap_camera:
            raise RuntimeError("minimap export failed")
        return np.zeros((output_size[1], output_size[0], 3), np.uint8)

    monkeypatch.setattr(high_resolution, "_render_view", render)
    if fail:
        with pytest.raises(RuntimeError, match="minimap export failed"):
            high_resolution._render_prepared_frame(owner, resolution_scale=2, include_hud=True)
    else:
        result = high_resolution._render_prepared_frame(owner, resolution_scale=2, include_hud=True)
        assert result.shape == (8, 12, 3)
    assert cameras == [owner._camera, owner._minimap_camera]
    assert guide.visible and not hidden.visible
