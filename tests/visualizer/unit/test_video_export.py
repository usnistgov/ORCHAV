import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.visualizer import OrchavVisualizer, _open_export_location


class _DummyProgressDialog:
    def __init__(self, *_args, **_kwargs):
        self.values = []
        self.closed = False

    def setWindowModality(self, *_args, **_kwargs):
        return None

    def setMinimumDuration(self, *_args, **_kwargs):
        return None

    def wasCanceled(self) -> bool:
        return False

    def setValue(self, value: int) -> None:
        self.values.append(value)

    def close(self) -> None:
        self.closed = True


class _DummyWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, image: np.ndarray) -> None:
        self.frames.append(np.asarray(image))

    def close(self) -> None:
        self.closed = True


class _PlaybackController:
    def __init__(self, *, stride: int = 1):
        self.visualizer = None
        self.stride = stride
        self.toggle_calls = []

    def get_current_stride(self) -> int:
        return self.stride

    def toggle_animation(self, direction=None) -> None:
        viz = self.visualizer
        self.toggle_calls.append((direction, bool(viz.animation_running)))
        if viz.animation_running:
            viz.animation_running = False
            return
        if direction is not None:
            viz.play_direction = int(direction)
        viz.animation_running = True


def _install_imageio_writer(monkeypatch, get_writer) -> None:
    """Install the lazily imported imageio surface used by video export."""
    imageio_package = ModuleType("imageio")
    imageio_v2 = ModuleType("imageio.v2")
    imageio_v2.get_writer = get_writer
    imageio_package.v2 = imageio_v2
    monkeypatch.setitem(sys.modules, "imageio", imageio_package)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio_v2)


def test_open_export_location_uses_argument_list_for_windows_shell_metacharacters(monkeypatch):
    output_path = r"C:\exports\report & review [final] ^ draft.mp4"
    calls = []
    monkeypatch.setattr("visualizer.visualizer.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "visualizer.visualizer.subprocess.Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    _open_export_location(output_path)

    assert calls == [((["explorer.exe", f"/select,{output_path}"],), {})]


def test_export_video_uses_current_animation_stride(monkeypatch, tmp_path):
    writer = _DummyWriter()
    progress_instances = []

    def _make_progress(*args, **kwargs):
        dialog = _DummyProgressDialog(*args, **kwargs)
        progress_instances.append(dialog)
        return dialog

    monkeypatch.setattr("visualizer.visualizer.QProgressDialog", _make_progress)
    monkeypatch.setattr(
        "visualizer.visualizer.QApplication",
        SimpleNamespace(processEvents=lambda: None),
    )
    _install_imageio_writer(monkeypatch, lambda *a, **k: writer)
    monkeypatch.setattr("visualizer.visualizer.platform.system", lambda: "Linux")
    popen_calls = []
    monkeypatch.setattr(
        "visualizer.visualizer.subprocess.Popen",
        lambda *a, **k: popen_calls.append((a, k)),
    )

    update_calls = []
    dummy_viz = SimpleNamespace(
        total_animation_steps=8,
        app_state=SimpleNamespace(step=6),
        renderer=SimpleNamespace(
            capabilities=RendererCapabilities(screenshot_export=True),
            export_screenshot_to_array=lambda resolution_scale=1.0, include_hud=False: np.zeros(
                (2, 2, 3), dtype=np.uint8
            ),
        ),
        animation_controller=SimpleNamespace(get_current_stride=lambda: 2),
        update_frame=lambda frame_idx: update_calls.append(frame_idx),
        get_available_animation_steps=lambda: list(range(8)),
    )

    ok = OrchavVisualizer.export_video(
        dummy_viz,
        output_path=str(tmp_path / "animation.gif"),
        fps=24,
        start_frame=1,
        end_frame=5,
    )

    assert ok is True
    assert update_calls == [1, 3, 5, 6]
    assert len(writer.frames) == 3
    assert writer.closed is True
    assert progress_instances
    assert progress_instances[0].values == [1, 2, 3]
    assert len(popen_calls) == 1


def test_export_video_forwards_resolution_scale(monkeypatch, tmp_path):
    writer = _DummyWriter()
    captured_scales = []

    monkeypatch.setattr("visualizer.visualizer.QProgressDialog", _DummyProgressDialog)
    monkeypatch.setattr(
        "visualizer.visualizer.QApplication",
        SimpleNamespace(processEvents=lambda: None),
    )
    _install_imageio_writer(monkeypatch, lambda *a, **k: writer)
    monkeypatch.setattr("visualizer.visualizer.platform.system", lambda: "Linux")
    monkeypatch.setattr("visualizer.visualizer.subprocess.Popen", lambda *a, **k: None)

    def _capture_frame(resolution_scale=1.0, include_hud=False):
        captured_scales.append((float(resolution_scale), bool(include_hud)))
        return np.zeros((2, 2, 3), dtype=np.uint8)

    dummy_viz = SimpleNamespace(
        total_animation_steps=3,
        app_state=SimpleNamespace(step=0),
        renderer=SimpleNamespace(
            capabilities=RendererCapabilities(screenshot_export=True),
            export_screenshot_to_array=_capture_frame,
        ),
        animation_controller=SimpleNamespace(get_current_stride=lambda: 1),
        update_frame=lambda frame_idx: None,
        get_available_animation_steps=lambda: list(range(3)),
    )

    ok = OrchavVisualizer.export_video(
        dummy_viz,
        output_path=str(tmp_path / "scaled.gif"),
        fps=24,
        start_frame=0,
        end_frame=1,
        resolution_scale=1.5,
        include_hud=True,
    )

    assert ok is True
    assert captured_scales == [(1.5, True), (1.5, True)]
    assert len(writer.frames) == 2


def test_export_video_pauses_playback_during_event_pumps_and_restores_intent(
    monkeypatch,
    tmp_path,
):
    writer = _DummyWriter()
    controller = _PlaybackController()
    holder = {}
    process_states = []
    captured_steps = []
    update_calls = []

    def _update_frame(frame_idx):
        viz = holder["viz"]
        update_calls.append(frame_idx)
        viz.app_state.step = frame_idx

    def _process_events():
        viz = holder["viz"]
        process_states.append((viz.animation_running, viz.app_state.step))
        if viz.animation_running:
            _update_frame(99)

    def _capture_frame(*, resolution_scale=1.0, include_hud=False):
        del resolution_scale, include_hud
        viz = holder["viz"]
        assert viz.animation_running is False
        captured_steps.append(viz.app_state.step)
        return np.full((2, 2, 3), viz.app_state.step, dtype=np.uint8)

    monkeypatch.setattr("visualizer.visualizer.QProgressDialog", _DummyProgressDialog)
    monkeypatch.setattr(
        "visualizer.visualizer.QApplication",
        SimpleNamespace(processEvents=_process_events),
    )
    _install_imageio_writer(monkeypatch, lambda *a, **k: writer)
    monkeypatch.setattr("visualizer.visualizer.platform.system", lambda: "Linux")
    monkeypatch.setattr("visualizer.visualizer.subprocess.Popen", lambda *a, **k: None)

    dummy_viz = SimpleNamespace(
        total_animation_steps=4,
        app_state=SimpleNamespace(step=3),
        renderer=SimpleNamespace(
            capabilities=RendererCapabilities(screenshot_export=True),
            export_screenshot_to_array=_capture_frame,
        ),
        animation_controller=controller,
        animation_running=True,
        play_direction=-1,
        update_frame=_update_frame,
        get_available_animation_steps=lambda: list(range(4)),
    )
    holder["viz"] = dummy_viz
    controller.visualizer = dummy_viz

    ok = OrchavVisualizer.export_video(
        dummy_viz,
        output_path=str(tmp_path / "playing.gif"),
        fps=24,
        start_frame=0,
        end_frame=1,
    )

    assert ok is True
    assert captured_steps == [0, 1]
    assert update_calls == [0, 1, 3]
    assert process_states
    assert all(running is False for running, _step in process_states)
    assert dummy_viz.app_state.step == 3
    assert dummy_viz.animation_running is True
    assert dummy_viz.play_direction == -1
    assert controller.toggle_calls == [(None, True), (-1, False)]


def test_export_video_reports_mp4_backend_failure_and_resets(monkeypatch, tmp_path):
    progress_instances = []

    def _make_progress(*args, **kwargs):
        dialog = _DummyProgressDialog(*args, **kwargs)
        progress_instances.append(dialog)
        return dialog

    def _raise_missing_backend(*_args, **_kwargs):
        raise RuntimeError("No ffmpeg backend available")

    monkeypatch.setattr("visualizer.visualizer.QProgressDialog", _make_progress)
    monkeypatch.setattr(
        "visualizer.visualizer.QApplication",
        SimpleNamespace(processEvents=lambda: None),
    )
    _install_imageio_writer(monkeypatch, _raise_missing_backend)

    update_calls = []
    controller = _PlaybackController()
    dummy_viz = SimpleNamespace(
        total_animation_steps=4,
        app_state=SimpleNamespace(step=2),
        renderer=SimpleNamespace(
            capabilities=RendererCapabilities(screenshot_export=True),
            export_screenshot_to_array=lambda resolution_scale=1.0, include_hud=False: np.zeros(
                (2, 2, 3), dtype=np.uint8
            ),
        ),
        animation_controller=controller,
        animation_running=True,
        play_direction=1,
        update_frame=lambda frame_idx: update_calls.append(frame_idx),
        get_available_animation_steps=lambda: list(range(4)),
    )
    controller.visualizer = dummy_viz

    ok = OrchavVisualizer.export_video(
        dummy_viz,
        output_path=str(tmp_path / "animation.mp4"),
        fps=24,
        start_frame=0,
        end_frame=2,
    )

    assert ok is False
    assert "imageio[ffmpeg]" in dummy_viz.last_video_export_error
    assert "pyav" in dummy_viz.last_video_export_error
    assert "No ffmpeg backend available" in dummy_viz.last_video_export_error
    assert update_calls == [2]
    assert progress_instances
    assert progress_instances[0].values == []
    assert progress_instances[0].closed is True
    assert dummy_viz.animation_running is True
    assert dummy_viz.play_direction == 1
    assert controller.toggle_calls == [(None, True), (1, False)]
