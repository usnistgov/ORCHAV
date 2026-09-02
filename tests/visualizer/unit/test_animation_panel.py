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
