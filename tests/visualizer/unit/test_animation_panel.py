"""Qt tests for explicit animation playback controls."""

from visualizer.src.panels.animation_panel import AnimationControlsPanel
from visualizer.src.playback import PlaybackMode


class _MemorySettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


def test_playback_controls_default_to_maximum_with_fixed_30_fps(qtbot):
    panel = AnimationControlsPanel(None, settings=_MemorySettings())
    group = panel.create_panel()
    qtbot.addWidget(group)

    combo = panel.widgets["playback_mode_combo"]
    fps = panel.widgets["playback_fps_spinbox"]

    assert [combo.itemText(index) for index in range(combo.count())] == [
        "Real time",
        "Fixed FPS",
        "Maximum",
    ]
    assert panel.playback_mode() is PlaybackMode.MAXIMUM
    assert panel.fixed_playback_fps() == 30
    assert fps.isHidden() is True
    assert "camera-interaction" in fps.toolTip()


def test_playback_controls_persist_selected_mode_and_fixed_fps(qtbot):
    settings = _MemorySettings()
    panel = AnimationControlsPanel(None, settings=settings)
    group = panel.create_panel()
    qtbot.addWidget(group)

    combo = panel.widgets["playback_mode_combo"]
    fps = panel.widgets["playback_fps_spinbox"]
    fps.setValue(60)
    combo.setCurrentIndex(combo.findData(PlaybackMode.FIXED_FPS.value))

    assert fps.isHidden() is False

    restored = AnimationControlsPanel(None, settings=settings)
    restored_group = restored.create_panel()
    qtbot.addWidget(restored_group)

    assert restored.playback_mode() is PlaybackMode.FIXED_FPS
    assert restored.fixed_playback_fps() == 60
    assert restored.widgets["playback_fps_spinbox"].isHidden() is False


def test_real_time_mode_hides_fixed_fps_selector(qtbot):
    panel = AnimationControlsPanel(None, settings=_MemorySettings())
    group = panel.create_panel()
    qtbot.addWidget(group)

    combo = panel.widgets["playback_mode_combo"]
    combo.setCurrentIndex(combo.findData(PlaybackMode.REAL_TIME.value))

    assert panel.playback_mode() is PlaybackMode.REAL_TIME
    assert panel.widgets["playback_fps_spinbox"].isHidden() is True


def test_sparse_context_does_not_leak_into_dense_recording(qtbot):
    panel = AnimationControlsPanel(None, settings=_MemorySettings())
    group = panel.create_panel()
    qtbot.addWidget(group)
    panel.update_total_steps(3)
    panel.set_recorded_frame_context([63, 127, 191])
    assert panel.widgets["frame_input"].maximum() == 3
    label = panel.widgets["recorded_frame_context"]
    assert not label.isHidden() and "3 saved frames" in label.text()
    panel.set_recorded_frame_context(list(range(192)))
    assert label.isHidden()


def test_finite_live_endpoint_does_not_request_out_of_range(qtbot):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    calls = []
    from unittest.mock import Mock

    from shared.frames.providers import DataProvider

    provider = Mock(spec=DataProvider)
    provider.info = SimpleNamespace(total_frames=50)
    parent.frame_source = SimpleNamespace(
        provider=provider,
        list_frames=lambda: list(range(50)),
        request_frame=lambda i: calls.append(i) or True,
    )
    panel = AnimationControlsPanel(parent, settings=_MemorySettings())
    panel.is_online_mode = panel.auto_request_frames = True
    assert not panel.request_frame_if_needed(50)
    assert not calls
    assert panel.request_frame_if_needed(49)
    provider.info.total_frames = -1
    assert panel.request_frame_if_needed(50)
    assert calls == [50]
