"""Portable initial views retain navigation, restoration precedence and timing."""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QComboBox, QSpinBox

from shared.scenarios.model import ViewDefaultsModel
from visualizer.src.app.scenario_presentation import (
    apply_camera_defaults,
    apply_presentation_defaults,
)
from visualizer.src.app.scenario_workflow import _apply_initial_camera
from visualizer.src.renderers.protocol import RendererCapabilities


def test_explicit_camera_defaults_do_not_touch_navigation_and_session_wins():
    camera, restored = [], []
    controller = SimpleNamespace(enabled=True)
    viz = SimpleNamespace(
        renderer=SimpleNamespace(set_camera_state=camera.append, _controller=controller),
        session_service=SimpleNamespace(_restore_camera_state=restored.append),
        record_startup_stage_timing=lambda *args: None,
    )
    defaults = {"camera_pose": {"eye": [1, 2, 3], "lookat": [0, 0, 0]}}
    apply_camera_defaults(viz, defaults)
    assert camera[-1].eye == (1, 2, 3)
    assert camera[-1].up == (0, 0, 1) and camera[-1].fov_deg == 60
    pending = {"saved": "camera"}
    _apply_initial_camera(viz, SimpleNamespace(view_defaults=defaults), pending)
    assert restored == [pending] and len(camera) == 1
    assert controller.enabled


@pytest.mark.parametrize(
    "pose",
    [
        {"eye": [0, 0, 0], "lookat": [0, 0, 0]},
        {"eye": [0, 0, 3], "lookat": [0, 0, 0], "up": [0, 0, 1]},
        {"eye": [float("nan"), 1, 3], "lookat": [0, 0, 0]},
        {"eye": [1, 2, 3], "lookat": [0, 0, 0], "fov": 180},
    ],
)
def test_degenerate_camera_rejected_before_scenario_open(pose):
    with pytest.raises(ValueError):
        ViewDefaultsModel.model_validate({"camera_pose": pose})


def test_playback_defaults_change_view_cadence_only(qtbot):
    mode, stride, fps = QComboBox(), QComboBox(), QSpinBox()
    for widget in (mode, stride, fps):
        qtbot.addWidget(widget)
    mode.addItem("Maximum", "maximum")
    mode.addItem("Fixed FPS", "fixed_fps")
    stride.setEditable(True)
    fps.setRange(1, 240)
    widgets = {"playback_mode_combo": mode, "stride_combo": stride, "playback_fps_spinbox": fps}
    viz = SimpleNamespace(
        ui_manager=SimpleNamespace(panels={"animation": SimpleNamespace(widgets=widgets)}),
        animation_running=False,
    )
    apply_presentation_defaults(viz, {"playback": {"mode": "fixed_fps", "fps": 25, "stride": 32}})
    assert (mode.currentData(), fps.value(), stride.currentText()) == ("fixed_fps", 25, "x32")
    assert not viz.animation_running


def test_scenario_dusk_is_configured_without_activation_and_cleared_for_next_source():
    from unittest.mock import Mock

    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(dusk_lighting=True),
        configure_dusk=Mock(),
        set_dusk_enabled=Mock(),
    )
    viz = SimpleNamespace(renderer=renderer)
    recipe = {"label": "Blue hour"}
    apply_presentation_defaults(viz, {"dusk": recipe})
    renderer.configure_dusk.assert_called_once_with(recipe)
    renderer.set_dusk_enabled.assert_not_called()
    apply_presentation_defaults(viz, {})
    renderer.configure_dusk.assert_called_with(None)
