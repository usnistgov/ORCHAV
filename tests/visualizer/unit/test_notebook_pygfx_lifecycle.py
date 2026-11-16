"""Notebook frame selection and playback without a GPU or a browser session."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from weakref import WeakSet

import numpy as np
import pytest

from visualizer.src.notebook import pygfx as notebook

pytestmark = pytest.mark.headless


@pytest.fixture
def viz(tmp_path):
    viewer = object.__new__(notebook.PygfxNotebookViz)
    viewer._project_root = tmp_path
    viewer._scenario = SimpleNamespace(root=tmp_path, data_spec={})
    viewer._frame_source = MagicMock()
    viewer._frame_source.list_frames.return_value = [3, 7]
    viewer._mpc_core = MagicMock()
    viewer._mpc_core.create_view_model.return_value = None
    viewer._mesh_entries = []
    viewer._texture_cache = {}
    viewer._canvases = WeakSet()
    viewer._setup_ibl = MagicMock(return_value=None)
    yield viewer
    viewer.close()


def load_frame(viz, frame):
    return viz._build_frame_data(
        frame, "reflection_order", "all", "all", True, True, True, None, None
    )


@pytest.mark.parametrize("available, requested", [([3, 7], 999), ([3, 7], 0), ([], 7)])
def test_missing_frame_fails_without_loading_a_substitute(viz, monkeypatch, available, requested):
    viz._frame_source.list_frames.return_value = available
    loader = MagicMock()
    monkeypatch.setattr(notebook, "load_notebook_visual_frame", loader)

    with pytest.raises(ValueError, match=f"Frame {requested} is not available"):
        load_frame(viz, requested)

    loader.assert_not_called()
    viz._mpc_core.create_view_model.assert_not_called()


def test_valid_sparse_frame_keeps_its_identity(viz, monkeypatch):
    payload = {"tx_positions": np.array([[1.0, 2.0, 3.0]])}
    loader = MagicMock(return_value=payload)
    monkeypatch.setattr(notebook, "load_notebook_visual_frame", loader)

    actual, _ = load_frame(viz, 7)

    assert actual is payload
    assert loader.call_args.args == (viz._frame_source, 7)
    assert viz._mpc_core.create_view_model.call_args.kwargs["step"] == 7


@pytest.mark.parametrize("closed", [False, True])
def test_default_scene_only_preview_remains_available(viz, closed):
    if closed:
        viz.close()
    else:
        viz._frame_source.list_frames.return_value = []
    assert load_frame(viz, 0) == (None, None)


@pytest.mark.parametrize("method", ["show", "widget", "animate"])
def test_missing_notebook_backend_fails_before_building_scene(viz, monkeypatch, method):
    monkeypatch.setattr(notebook, "_running_in_jupyter", lambda: True)
    monkeypatch.setattr(notebook, "JUPYTER_RFB_AVAILABLE", False)
    viz._build_pygfx_scene = MagicMock()

    with pytest.raises(ImportError, match=r"jupyter_rfb.*render\(\)"):
        getattr(viz, method)()

    viz._build_pygfx_scene.assert_not_called()
    viz._setup_ibl.assert_not_called()


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_animation_rejects_invalid_rate_before_scene_work(viz, fps):
    with pytest.raises(ValueError, match="fps must be finite and greater than zero"):
        viz.animate(fps=fps)
    viz._setup_ibl.assert_not_called()


def test_animation_checks_all_requested_frames_before_starting(viz):
    with pytest.raises(ValueError, match="not available.*999"):
        viz.animate(frames=[3, 999])
    viz._setup_ibl.assert_not_called()


class FakeCanvas:
    """Capture scheduled draws and closure while leaving scene construction real."""

    def __init__(self, **kwargs):
        self.draw_callback = None
        self.draw_requests = 0
        self.close_calls = 0
        self.closed = False

    def request_draw(self, callback=None):
        if callback is not None:
            self.draw_callback = callback
        self.draw_requests += 1

    def get_closed(self):
        return self.closed

    def close(self):
        self.close_calls += 1
        self.closed = True


@pytest.fixture
def display_backend(monkeypatch):
    if not notebook.PYGFX_AVAILABLE:
        pytest.skip("pygfx is not installed")
    jupyter = pytest.importorskip("rendercanvas.jupyter")
    monkeypatch.setattr(notebook, "_running_in_jupyter", lambda: True)
    monkeypatch.setattr(notebook, "JUPYTER_RFB_AVAILABLE", True)
    monkeypatch.setattr(jupyter, "RenderCanvas", FakeCanvas)
    renderer = MagicMock()
    from visualizer.src.renderers.pygfx import canvas

    monkeypatch.setattr(canvas, "create_wgpu_renderer", MagicMock(return_value=renderer))
    controller = MagicMock()
    monkeypatch.setattr(notebook.gfx, "OrbitController", MagicMock(return_value=controller))
    return SimpleNamespace(renderer=renderer, controller=controller)


@pytest.fixture
def playback(viz, monkeypatch, display_backend):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("time.perf_counter", lambda: clock.now)
    loader = MagicMock(
        side_effect=lambda source, frame, **kwargs: {
            "tx_positions": np.array([[float(frame), 0.0, 2.0]]),
            "rx_positions": np.array([[0.0, float(frame), 1.0]]),
        }
    )
    monkeypatch.setattr(notebook, "load_notebook_visual_frame", loader)
    # Observe label positions without initializing the font atlas.
    viz._add_node_labels = MagicMock()
    return SimpleNamespace(clock=clock, loader=loader, backend=display_backend)


def test_nonlooping_animation_shows_first_frame_and_retires_playback(viz, playback):
    canvas = viz.animate(frames=np.array([3, 7]), fps=1, loop=False)
    initial_reads = playback.loader.call_count
    playback.clock.now = 500.0  # A delayed frontend must still display frame 3.
    canvas.draw_callback()
    assert playback.loader.call_count == initial_reads

    playback.clock.now += 1.0
    canvas.draw_callback()
    assert playback.loader.call_args.args[1] == 7
    reads_at_end = playback.loader.call_count
    requests_at_end = canvas.draw_requests
    renders_at_end = playback.backend.renderer.render.call_count
    for _ in range(3):
        playback.clock.now += 2.0
        canvas.draw_callback()  # Camera interaction can still redraw the last frame.
    assert playback.loader.call_count == reads_at_end
    assert canvas.draw_requests == requests_at_end
    assert playback.backend.renderer.render.call_count == renders_at_end + 3
    assert not canvas.closed
    labels = viz._add_node_labels.call_args_list
    assert len(labels) == 2
    np.testing.assert_array_equal(labels[0].args[1], [[3.0, 0.0, 2.0]])
    np.testing.assert_array_equal(labels[1].args[1], [[7.0, 0.0, 2.0]])
    np.testing.assert_array_equal(labels[1].args[2], [[0.0, 7.0, 1.0]])


def test_looping_animation_wraps_to_the_first_selected_frame(viz, playback):
    canvas = viz.animate(frames=[7, 3], fps=1, loop=True)
    canvas.draw_callback()
    playback.clock.now += 1.0
    canvas.draw_callback()
    assert playback.loader.call_args.args[1] == 3
    playback.clock.now += 1.0
    canvas.draw_callback()
    assert playback.loader.call_args.args[1] == 7
    assert canvas.draw_requests == 4


def test_animation_keeps_its_schedule_after_a_late_draw(viz, playback):
    canvas = viz.animate(frames=[3, 7], fps=1)
    canvas.draw_callback()
    playback.clock.now += 1.2
    canvas.draw_callback()
    assert playback.loader.call_args.args[1] == 7
    playback.clock.now += 0.8
    canvas.draw_callback()
    assert playback.loader.call_args.args[1] == 3


@pytest.mark.parametrize("loop", [False, True])
def test_single_frame_animation_becomes_an_interactive_still(viz, playback, loop):
    canvas = viz.animate(frames=[7], loop=loop)
    reads = playback.loader.call_count
    canvas.draw_callback()
    playback.clock.now += 10
    canvas.draw_callback()
    assert playback.loader.call_count == reads
    assert canvas.draw_requests == 2


@pytest.mark.parametrize("method", ["show", "widget"])
def test_static_views_redraw_on_demand_and_close_with_owner(viz, playback, method):
    canvas = getattr(viz, method)(frame=3)
    canvas.draw_callback()
    assert canvas.draw_requests == 2
    canvas.draw_callback()
    assert canvas.draw_requests == 2
    playback.backend.controller.register_events.assert_called_once_with(playback.backend.renderer)
    viz.close()
    viz.close()
    assert canvas.close_calls == 1
    renders = playback.backend.renderer.render.call_count
    canvas.draw_callback()
    assert playback.backend.renderer.render.call_count == renders


@pytest.mark.parametrize(
    "method,frames",
    [("show", [3]), ("widget", [3]), ("animate", [3]), ("animate", [3, 7])],
)
def test_static_view_delivers_deferred_first_image_then_becomes_idle(
    viz, playback, monkeypatch, method, frames
):
    import rendercanvas.jupyter

    class DeferredReadbackCanvas(FakeCanvas):
        """Model a transport pull returning before GPU readback completes."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.draw_pending = False
            self.completed_image = None

        def request_draw(self, callback=None):
            super().request_draw(callback)
            self.draw_pending = True

        def pull_image(self):
            assert self.draw_pending
            self.draw_pending = False
            image = self.completed_image
            self.draw_callback()
            self.completed_image = ("rendered frame", playback.loader.call_args.args[1])
            return image

    monkeypatch.setattr(rendercanvas.jupyter, "RenderCanvas", DeferredReadbackCanvas)
    kwargs = {"frames": frames, "loop": False} if method == "animate" else {"frame": 3}
    canvas = getattr(viz, method)(**kwargs)
    assert canvas.pull_image() is None
    assert canvas.draw_pending
    playback.clock.now += 1.0
    assert canvas.pull_image() == ("rendered frame", 3)
    if len(frames) > 1:
        assert canvas.draw_pending
        assert canvas.pull_image() == ("rendered frame", 7)
    assert not canvas.draw_pending
    viz.close()
    renders = playback.backend.renderer.render.call_count
    canvas.draw_callback()
    assert playback.backend.renderer.render.call_count == renders
    assert not canvas.draw_pending


def test_reload_closes_old_views_before_reopening_frame_files(viz, playback, monkeypatch):
    from visualizer.src.io import frame_sources

    animation = viz.animate(fps=1)
    still = viz.widget(frame=3)
    old_source = viz._frame_source
    replacement = MagicMock()

    def open_replacement():
        assert animation.closed and still.closed
        old_source.close.assert_called_once()

    replacement.open.side_effect = open_replacement
    replacement.list_frames.return_value = [3, 7]
    monkeypatch.setattr(frame_sources, "FileSource", MagicMock(return_value=replacement))
    viz.reload()
    assert viz._frame_source is replacement
    reads = playback.loader.call_count
    renders = playback.backend.renderer.render.call_count
    animation.draw_callback()
    still.draw_callback()
    assert playback.loader.call_count == reads
    assert playback.backend.renderer.render.call_count == renders
    new_canvas = viz.widget(frame=3)
    new_canvas.draw_callback()
    assert playback.backend.renderer.render.call_count == renders + 1


def test_closed_canvas_cannot_advance_playback(viz, playback):
    canvas = viz.animate()
    canvas.draw_callback()
    canvas.close()
    reads = playback.loader.call_count
    renders = playback.backend.renderer.render.call_count
    playback.clock.now += 10
    canvas.draw_callback()
    assert playback.loader.call_count == reads
    assert playback.backend.renderer.render.call_count == renders


def test_renderer_initialization_failure_releases_canvas(viz, monkeypatch):
    from visualizer.src.renderers.pygfx import canvas

    original = RuntimeError("renderer unavailable")
    monkeypatch.setattr(canvas, "create_wgpu_renderer", MagicMock(side_effect=original))
    target = FakeCanvas()
    with pytest.raises(RuntimeError, match="renderer unavailable") as raised:
        viz._create_interactive_renderer(target)
    assert raised.value is original
    assert target.close_calls == 1
    assert not viz._canvases
