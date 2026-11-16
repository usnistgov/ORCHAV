"""Demo target cycling reuses fixed-position Follow and restores user settings."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox

from tests.visualizer.unit import test_camera_presentation
from tests.visualizer.unit.test_demo_service import Appearance, _add_dusk
from tests.visualizer.unit.test_demo_service import rig as rig
from tests.visualizer.unit.test_demo_tour import install_camera_tools
from visualizer.src.controllers.camera_controller import CameraController
from visualizer.src.panels.demo_panel import DemoPanel
from visualizer.src.services.demo_service import DemoService

native_rig = test_camera_presentation.rig


def _install_targets(viz):
    install_camera_tools(viz).close()
    combo = viz.target_focus_dropdown = QComboBox(viz)
    combo.addItem("Auto", {"type": "auto"})
    for name in ("b", "a", "c"):
        combo.addItem(name, {"type": "target", "stable_target_id": name})
    combo.addItem("TX", {"type": "tx", "index": 0})
    positions = {"a": [10, 2, 1], "b": [4, 5, 2], "c": [-2, 3, 0]}
    query = SimpleNamespace(
        get_focus_position=lambda: positions.get(combo.currentData().get("stable_target_id"))
    )
    viz.camera_controller = CameraController(viz, query)
    viz.vis_initialized, viz.vis = True, object()
    viz.renderer.focus_camera = Mock()
    viz.renderer.update_follow_camera = Mock()
    tools = viz.camera_controller.presentation()
    tools.follow_heading = "chase"
    tools.follow_offset = [5, 2, 4]
    return combo, positions, tools


def test_cycles_all_targets_from_fixed_eye_and_follows_recorded_motion(rig, monkeypatch):
    viz, service, _ = rig
    combo, positions, tools = _install_targets(viz)
    monkeypatch.setattr("visualizer.src.services.demo_service.perf_counter", lambda: 100)
    eye = viz.renderer.camera.eye
    try:
        assert service.start(
            orbit=False, cycle_targets=True, target_interval=7, cycle_lighting=False
        )
        assert tools.follow_heading == "fixed" and viz.app_state.camera_mode == "follow"
        assert combo.currentText() == "b"
        for time, target in ((106.9, "b"), (107, "a"), (114, "c"), (121, "b")):
            service.tick(time)
            assert combo.currentText() == target
            assert viz.renderer.camera.eye == eye
            assert tuple(viz.renderer.camera.lookat) == tuple(positions[target])
        positions["b"] = [11, 4, 7]
        viz.camera_controller.update_follow_camera_focus()
        assert viz.renderer.camera.eye == eye
        assert viz.renderer.camera.lookat == (11, 4, 7)
        visible = viz.renderer.camera
        service.stop()
        assert combo.currentIndex() == 0 and viz.app_state.camera_mode == "overview"
        assert tools.follow_heading == "chase" and tools.follow_offset == [5, 2, 4]
        assert viz.renderer.camera == visible
    finally:
        service.stop()
        tools.close()


def test_removed_target_is_skipped_and_empty_catalog_stops_safely(rig):
    viz, service, _ = rig
    combo, _positions, tools = _install_targets(viz)
    try:
        assert service.start(orbit=False, cycle_targets=True, cycle_lighting=False, playback=False)
        combo.removeItem(2)  # a disappears; TX must never enter the target cycle.
        service.tick(service._next_target)
        assert combo.currentText() == "c"
        with QSignalBlocker(combo):
            combo.clear()
        service.tick(service._next_target)
        assert not service.active and viz.app_state.camera_mode == "overview"
    finally:
        service.stop()
        tools.close()


def test_sunset_duration_is_independent_and_finishes_before_next_look(rig, monkeypatch):
    viz, service, _ = rig
    viz.appearance_preset_service = Appearance(viz.renderer)
    state, _calls = _add_dusk(viz.renderer)
    monkeypatch.setattr("visualizer.src.services.demo_service.perf_counter", lambda: 100)
    assert service.start(
        orbit=False,
        playback=False,
        cycle_appearance=True,
        scenario_sunset=True,
        look_interval=3,
        sunset_duration=20,
    )
    assert service._next_look == 120
    service.tick(106)
    assert state["progress"] == pytest.approx(0.5)
    service.tick(111)
    assert state["progress"] == 1
    service.tick(116)
    assert state["progress"] == pytest.approx(0.5)
    service.tick(120)
    assert viz.appearance_preset_service.active_name == "Detailed"
    assert service._next_look == 123


def test_demo_controls_expose_independent_parameters_and_exclusive_camera_actions(rig, qtbot):
    viz, _service, _ = rig
    _add_dusk(viz.renderer)
    viz.target_focus_dropdown = QComboBox(viz)
    viz.target_focus_dropdown.addItem("Target", {"type": "target", "stable_target_id": "car"})
    panel = DemoPanel(viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    widget.show()
    try:
        assert not panel.widgets["demo_advanced"].is_expanded()
        panel.widgets["demo_advanced"].expand()
        panel.widgets["demo_targets"].setChecked(True)
        assert not panel.widgets["demo_orbit"].isChecked()
        assert panel.widgets["demo_target_interval"].isVisible()
        panel.widgets["demo_target_interval"].setValue(9)
        lighting = panel.widgets["demo_lighting_mode"]
        lighting.setCurrentIndex(lighting.findData("sunset"))
        panel.widgets["demo_sunset_duration"].setValue(42)
        panel.service.start = Mock()
        panel.start()
        args = panel.service.start.call_args.kwargs
        assert args["cycle_targets"] and not args["orbit"]
        assert args["target_interval"] == 9 and args["sunset_duration"] == 42
        panel.widgets["demo_orbit"].setChecked(True)
        assert not panel.widgets["demo_targets"].isChecked()
    finally:
        panel.cleanup()


def test_target_switch_uses_existing_native_smooth_transition_and_tracks_destination(
    native_rig, qtbot
):
    tools, renderer, _position = native_rig
    viz = tools.viz
    viz.ready = True
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.app_state.camera_mode = "overview"
    combo = viz.target_focus_dropdown = QComboBox()
    qtbot.addWidget(combo)
    for name in ("a", "b"):
        combo.addItem(name, {"type": "target", "stable_target_id": name})
    positions = {"a": [4, 5, 1], "b": [-3, 7, 2]}
    tools.owner.scene_query.get_focus_position = lambda: positions[combo.currentText()]
    initial = renderer.get_camera_state()
    history = list(tools.history)
    service = DemoService(viz)
    try:
        assert service.start(orbit=False, cycle_targets=True, playback=False, cycle_lighting=False)
        assert tools.transition is not None
        assert renderer.get_camera_state() == initial  # No immediate jump.
        began = tools.transition[2]
        tools.tick(began + tools.duration / 2)
        np.testing.assert_allclose(renderer.get_camera_state().eye, initial.eye)
        positions["a"] = [5, 6, 2]
        tools.owner.update_follow_camera_focus()
        np.testing.assert_allclose(tools.transition[1].lookat, positions["a"])
        tools.tick(began + tools.duration)
        np.testing.assert_allclose(renderer.get_camera_state().lookat, positions["a"])
        previous = renderer.get_camera_state()
        service.tick(service._next_target)
        assert combo.currentText() == "b" and tools.transition is not None
        assert renderer.get_camera_state() == previous
        tools.tick(tools.transition[2] + tools.duration / 2)
        visible = renderer.get_camera_state()
        np.testing.assert_allclose(visible.eye, initial.eye)
        assert not np.allclose(visible.lookat, previous.lookat)
        assert not np.allclose(visible.lookat, positions["b"])
        service.stop()
        assert tools.transition is None and not tools.timer.isActive()
        assert renderer.get_camera_state() == visible and tools.history == history
    finally:
        service.stop()
