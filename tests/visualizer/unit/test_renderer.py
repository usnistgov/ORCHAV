import subprocess
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest


def test_renderer_registry_accepts_only_supported_ids():
    from visualizer.src.renderers.registry import canonicalize_renderer_id, renderer_choices

    assert renderer_choices() == ("pygfx", "open3d")
    assert canonicalize_renderer_id("pygfx") == "pygfx"
    assert canonicalize_renderer_id("open3d") == "open3d"
    with pytest.raises(ValueError, match="Unsupported renderer"):
        canonicalize_renderer_id("unsupported")


def test_renderer_protocol_and_base_import_without_open3d():
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "open3d" or name.startswith("open3d."):
        raise AssertionError(f"neutral renderer import loaded {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import visualizer.src.renderers.protocol  # noqa: F401
import visualizer.src.renderers.base  # noqa: F401
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_pygfx_edge_width_updates_tracked_external_edge_names():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    edge_material = SimpleNamespace(thickness=1.0)
    mpc_material = SimpleNamespace(thickness=2.0)
    pygfx_renderer._name_to_handle = {"external_geom_1": 1, "mpc_lines": 2}
    pygfx_renderer._objects = {
        "external_geom_1": SimpleNamespace(material=edge_material),
        "mpc_lines": SimpleNamespace(material=mpc_material),
    }
    pygfx_renderer._edge_geometry_names = {"external_geom_1"}
    pygfx_renderer._line_width = 2.0
    pygfx_renderer._edge_line_width = 1.0
    pygfx_renderer.trajectory_line_width = 3.0
    redraws: list[bool] = []
    pygfx_renderer.request_redraw = lambda: redraws.append(True)

    assert pygfx_renderer._edge_line_names() == ["external_geom_1"]
    assert pygfx_renderer.set_edge_line_width(6.0) is True

    assert edge_material.thickness == 6.0
    assert mpc_material.thickness == 2.0
    assert redraws == [True]


def test_pygfx_show_axes_creates_named_rgb_axis_payload():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer._initialized = True
    pygfx_renderer._line_width = 2.0
    records: list[tuple] = []
    redraws: list[bool] = []
    pygfx_renderer.compute_scene_bounds = lambda scope="visible": None
    pygfx_renderer.ensure_named_geometry = (
        lambda name, payload, material=None, visible=None, **_: records.append(
            (name, payload, material, visible)
        )
        or True
    )
    pygfx_renderer.request_redraw = lambda: redraws.append(True)

    assert pygfx_renderer.show_axes(True) is True

    name, payload, material, visible = records[0]
    assert name == PygfxRenderer.AXES_NAME
    assert visible is True
    assert payload.lines.tolist() == [[0, 1], [0, 2], [0, 3]]
    assert payload.colors.shape == (3, 4)
    assert material.shader == "unlit"
    assert material.line_width == 3.0
    assert redraws == [True]


def test_pygfx_axes_size_scales_for_large_scenes():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _BBox:
        def get_extent(self):
            return np.asarray([700.0, 705.0, 50.0], dtype=np.float64)

    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer.compute_scene_bounds = lambda scope="visible": _BBox()

    assert pygfx_renderer._default_axes_size(1.0) == pytest.approx(75.0)
    assert pygfx_renderer._default_axes_size(12.0) == pytest.approx(12.0)


def test_pygfx_show_axes_removes_named_axis_payload():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer._initialized = True
    removals: list[str] = []
    redraws: list[bool] = []
    pygfx_renderer.remove_named_geometry = lambda name: removals.append(name) or True
    pygfx_renderer.request_redraw = lambda: redraws.append(True)

    assert pygfx_renderer.show_axes(False) is True

    assert removals == [PygfxRenderer.AXES_NAME]
    assert redraws == [True]


def test_pygfx_qt_window_close_marks_rendercanvas_closed():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    canvas = SimpleNamespace(_is_closed=False, _draw_frame=object())
    pygfx_renderer._qt_window_closed = False
    pygfx_renderer._frame_update_paused = False
    pygfx_renderer._canvas = canvas
    pygfx_renderer._canvas_widget = canvas

    pygfx_renderer._mark_qt_window_closed()

    assert pygfx_renderer._qt_window_closed is True
    assert pygfx_renderer._frame_update_paused is True
    assert canvas._is_closed is True
    assert canvas._draw_frame is None
    assert canvas._rc_request_paint() is None
    assert canvas._rc_close() is None


def test_pygfx_request_draw_skips_closed_qt_window():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _Canvas:
        def __init__(self):
            self.calls = 0

        def request_draw(self):
            self.calls += 1

    canvas = _Canvas()
    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer._qt_window_closed = True
    pygfx_renderer._initialized = True
    pygfx_renderer._canvas = canvas
    pygfx_renderer._frame_update_paused = False
    pygfx_renderer._redraw_requests = 0

    assert pygfx_renderer._request_canvas_draw() is False
    assert canvas.calls == 0
    assert pygfx_renderer._redraw_requests == 0


def test_pygfx_end_frame_update_skips_closed_qt_window():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _Canvas:
        def __init__(self):
            self.force_draw_calls = 0

        def force_draw(self):
            self.force_draw_calls += 1

    canvas = _Canvas()
    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer._qt_window_closed = True
    pygfx_renderer._frame_update_paused = False
    pygfx_renderer._initialized = True
    pygfx_renderer._canvas = canvas
    pygfx_renderer._last_end_frame_update_breakdown = {"old": 1.0}
    pygfx_renderer._last_end_frame_update_breakdown_bytes = {"old": 2.0}

    assert pygfx_renderer.end_frame_update() is False

    assert canvas.force_draw_calls == 0
    assert pygfx_renderer._frame_update_paused is True
    assert pygfx_renderer._last_end_frame_update_breakdown == {}
    assert pygfx_renderer._last_end_frame_update_breakdown_bytes == {}


def test_pygfx_user_render_window_close_requests_app_close_without_marking_closed():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    close_calls: list[bool] = []
    canvas = SimpleNamespace(_is_closed=False)
    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer.visualizer = SimpleNamespace(close=lambda: close_calls.append(True))
    pygfx_renderer._qt_window_closed = False
    pygfx_renderer._qt_app_close_requested = False
    pygfx_renderer._session_generation = 0
    pygfx_renderer._canvas = canvas
    pygfx_renderer._canvas_widget = canvas
    pygfx_renderer._frame_update_paused = False
    pygfx_renderer._schedule_qt_callback = lambda callback: callback()

    pygfx_renderer._request_app_close_from_render_window()

    assert pygfx_renderer._qt_window_closed is False
    assert canvas._is_closed is False
    assert pygfx_renderer._qt_app_close_requested is True
    assert close_calls == [True]


def test_pygfx_guarded_canvas_noops_after_renderer_close():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _CanvasBase:
        def __init__(self, **_kwargs):
            self.paint_calls = 0
            self.close_calls = 0

        def _rc_request_paint(self):
            self.paint_calls += 1

        def _rc_close(self):
            self.close_calls += 1

    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer._WgpuWidget = _CanvasBase
    pygfx_renderer._qt_window_closed = False

    canvas_cls = pygfx_renderer._guarded_canvas_widget_class()
    canvas = canvas_cls()

    canvas._rc_request_paint()
    canvas._rc_close()
    pygfx_renderer._qt_window_closed = True
    canvas._rc_request_paint()
    canvas._rc_close()

    assert canvas.paint_calls == 1
    assert canvas.close_calls == 1


def test_pygfx_qt_destroyed_hook_marks_window_closed():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    class _Signal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self):
            for callback in list(self.callbacks):
                callback()

    class _Widget:
        def __init__(self):
            self.destroyed = _Signal()

    container = _Widget()
    canvas = _Widget()
    canvas._is_closed = False
    pygfx_renderer = PygfxRenderer.__new__(PygfxRenderer)
    pygfx_renderer._qt_window_closed = False
    pygfx_renderer._qt_destroyed_callbacks = []
    pygfx_renderer._qt_lifecycle_connections = []
    pygfx_renderer._session_generation = 0
    pygfx_renderer._container = container
    pygfx_renderer._canvas_widget = canvas
    pygfx_renderer._canvas = canvas
    pygfx_renderer._frame_update_paused = False

    pygfx_renderer._install_qt_lifecycle_hooks()
    container.destroyed.emit()

    assert len(pygfx_renderer._qt_destroyed_callbacks) == 2
    assert pygfx_renderer._qt_window_closed is True
    assert canvas._is_closed is True


def test_open3d_window_close_callback_requests_app_close():
    from visualizer.src.renderers.open3d.renderer import Open3DRenderer

    class _O3DWindow:
        def __init__(self):
            self.callback = None

        def set_on_close(self, callback):
            self.callback = callback

    close_calls: list[bool] = []
    window = _O3DWindow()
    open3d_renderer = Open3DRenderer.__new__(Open3DRenderer)
    open3d_renderer.visualizer = SimpleNamespace(close=lambda: close_calls.append(True))
    open3d_renderer._o3d_vis = window
    open3d_renderer._closing_programmatically = False
    open3d_renderer._schedule_app_close = lambda callback: callback()

    open3d_renderer._install_window_close_callback()

    assert window.callback is not None
    assert window.callback() is False
    assert close_calls == [True]

    open3d_renderer._closing_programmatically = True
    assert window.callback() is True


def test_open3d_close_releases_python_window_before_terminal_native_tick(monkeypatch):
    from visualizer.src.renderers.open3d import renderer as renderer_module
    from visualizer.src.renderers.open3d.renderer import Open3DRenderer

    events: list[str] = []
    open3d_renderer = Open3DRenderer(SimpleNamespace())

    class _Timer:
        def stop(self):
            events.append("timer_stop")

    class _Window:
        def close(self):
            assert open3d_renderer._o3d_vis is None
            assert open3d_renderer.vis is None
            assert open3d_renderer._closing_programmatically is True
            events.append("window_close")

    def _attach_window():
        window = _Window()
        open3d_renderer._o3d_vis = window
        open3d_renderer.vis = window
        return weakref.ref(window)

    window_ref = _attach_window()

    class _Application:
        def run_one_tick(self):
            assert window_ref() is None
            events.append("terminal_tick")
            return False

        def quit(self):
            events.append("quit")

    monkeypatch.setattr(
        renderer_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=_Application())),
    )
    open3d_renderer._gui_initialized = True
    open3d_renderer._gui_timer = _Timer()
    open3d_renderer.vis_initialized = True

    open3d_renderer.close()

    assert events == ["timer_stop", "window_close", "terminal_tick"]
    assert open3d_renderer._gui_initialized is False
    assert open3d_renderer._o3d_vis is None


def test_open3d_close_quits_once_when_other_native_windows_keep_loop_alive(monkeypatch):
    from visualizer.src.renderers.open3d import renderer as renderer_module
    from visualizer.src.renderers.open3d.renderer import Open3DRenderer

    events: list[str] = []
    tick_results = iter((True, False))
    open3d_renderer = Open3DRenderer(SimpleNamespace())
    open3d_renderer._o3d_vis = SimpleNamespace(close=lambda: events.append("window_close"))
    open3d_renderer.vis = open3d_renderer._o3d_vis
    open3d_renderer.vis_initialized = True
    open3d_renderer._gui_initialized = True
    open3d_renderer._gui_timer = None
    application = SimpleNamespace(
        run_one_tick=lambda: events.append("tick") or next(tick_results),
        quit=lambda: events.append("quit"),
    )
    monkeypatch.setattr(
        renderer_module,
        "gui",
        SimpleNamespace(Application=SimpleNamespace(instance=application)),
    )

    open3d_renderer.close()

    assert events == ["window_close", "tick", "quit", "tick"]
