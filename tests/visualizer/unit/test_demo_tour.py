"""Authored demo tours share camera timing and restore the user's saved shots."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.visualizer.unit import test_demo_service
from visualizer.src.panels.demo_panel import DemoPanel
from visualizer.src.services.camera_presentation import CameraPresentation
from visualizer.src.services.demo_tour import scenario_tour
from visualizer.src.types.camera_state import CameraState

rig = test_demo_service.rig


def tour_definition():
    return {
        "label": "Paris Opéra",
        "views": {
            "city": {"pose": CameraState((0, -100, 40), (0, 0, 10), (0, 0, 1), 55).to_dict()},
            "street": {"pose": CameraState((0, -40, 5), (0, 0, 10), (0, 0, 1), 65).to_dict()},
        },
        "shots": [
            {"view": "city", "hold": 3, "move": 0},
            {"view": "street", "hold": 4, "move": 2},
        ],
    }


def install_camera_tools(viz):
    viz.set_state = lambda **updates: [
        setattr(viz.app_state, key, value) for key, value in updates.items()
    ]
    viz.renderer.set_camera_input_callback = Mock()
    viz.renderer.cancel_camera_gestures = Mock()
    viz.renderer.set_camera_state = lambda pose: setattr(viz.renderer, "camera", pose) or True
    owner = SimpleNamespace(
        visualizer=viz,
        restore_pov_entity_visibility=Mock(),
        clear_pre_pov_camera_state=Mock(),
    )
    tools = CameraPresentation(owner)
    viz.camera_controller.presentation = lambda: tools
    tools.views = {"User view": {"pose": viz.renderer.camera.to_dict(), "actor": None}}
    tools.shots = [{"view": "User view", "hold": 8, "move": 2}]
    tools.history = [viz.renderer.camera]
    return tools


def test_scenario_tour_validates_and_closes_loop_without_mutating_yaml():
    raw = tour_definition()
    before = deepcopy(raw)
    tour = scenario_tour(SimpleNamespace(view_defaults={"demo_tour": raw}))
    assert raw == before
    assert tour["shots"][-1] == {"view": "city", "hold": 0, "move": 4}
    assert tour["views"]["city"]["actor"] is None
    assert scenario_tour(SimpleNamespace(view_defaults={})) is None


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t["shots"][1].update(move=float("nan")),
        lambda t: t["shots"][0].update(view="missing"),
        lambda t: t["views"]["city"]["pose"].update(eye=[1, 2]),
        lambda t: t["views"]["city"].update(actor={"type": "auto"}),
    ],
)
def test_invalid_tour_is_rejected_without_touching_camera(rig, change):
    viz, service, tools = rig
    raw = tour_definition()
    change(raw)
    viz.scenario.view_defaults = {"demo_tour": raw}
    pose = viz.renderer.camera
    assert not service.start(tour=True)
    assert viz.renderer.camera is pose
    assert not service.active and tools.autorotate is None


def test_tour_uses_existing_sequence_and_restores_user_shots_at_visible_pose(rig):
    viz, service, _fake = rig
    tools = install_camera_tools(viz)
    before = deepcopy((tools.views, tools.shots, tools.history))
    viz.scenario.view_defaults = {"demo_tour": tour_definition()}
    try:
        assert service.start(tour=True, playback=False, cycle_lighting=False)
        assert service._tour_owned and not service._orbit_owned
        assert tools.autorotate is None
        began = tools.sequence[0]
        tools.tick(began + 4)
        visible = viz.renderer.camera
        assert -100 < visible.eye[1] < -40
        assert not viz.animation_running
        service.stop()
        assert (tools.views, tools.shots, tools.history) == before
        assert viz.renderer.camera is visible
        assert tools.sequence is None and not tools.timer.isActive()
    finally:
        tools.close()


def test_camera_interrupt_pauses_tour_until_explicit_resume_or_stop(rig):
    viz, service, _fake = rig
    tools = install_camera_tools(viz)
    before = deepcopy((tools.views, tools.shots))
    viz.scenario.view_defaults = {"demo_tour": tour_definition()}
    try:
        assert service.start(tour=True, playback=False, cycle_lighting=False)
        tools.interrupt()
        service.tick()
        assert service.active and service.camera_paused
        assert service.resume_camera() and tools.sequence is not None
        service.stop()
        assert (tools.views, tools.shots) == before
    finally:
        tools.close()


def test_partial_tour_start_failure_restores_original_definitions(rig):
    viz, service, tools = rig
    tools.views, tools.shots = {"original": 1}, ["original shot"]
    tools.start_sequence = Mock(side_effect=ValueError("missing actor"))
    viz.scenario.view_defaults = {"demo_tour": tour_definition()}
    assert not service.start(tour=True)
    assert tools.views == {"original": 1} and tools.shots == ["original shot"]
    assert tools.history == ["original-history"]
    assert "missing actor" in service.status


def test_demo_panel_refreshes_tour_availability_when_reopened(rig, qtbot):
    viz, _service, _tools = rig
    panel = DemoPanel(viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        widget.show()
        assert not panel.widgets["demo_tour"].isEnabled()
        widget.hide()
        viz.scenario.view_defaults = {"demo_tour": tour_definition()}
        widget.show()
        assert panel.widgets["demo_tour"].isEnabled()
        panel.widgets["demo_tour"].setChecked(True)
        assert not panel.widgets["demo_orbit"].isChecked()
        assert panel.widgets["demo_speed"].isHidden()
        panel.widgets["demo_orbit"].setChecked(True)
        assert not panel.widgets["demo_tour"].isChecked()
    finally:
        panel.cleanup()
