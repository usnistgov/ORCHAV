"""Presentation continues while the audience explores the actual scene camera."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent, QStandardItem
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget
from shiboken6 import delete, isValid

from tests.visualizer.unit.test_demo_service import rig as rig
from tests.visualizer.unit.test_demo_target_cycle import _install_targets
from tests.visualizer.unit.test_demo_tour import install_camera_tools, tour_definition
from visualizer.src.app.viewport_workspace import VisualizationWorkspace


@pytest.mark.parametrize("event_kind", ["unrelated", "qt_item", "deleted_event"])
def test_unusable_event_wrapper_keeps_demo_and_playback_running(rig, event_kind):
    viz, service, _ = rig
    assert service.start()

    class UnexpectedItem(QStandardItem):
        def type(self):
            pytest.fail("A non-event Qt wrapper must not be inspected as an event")

    if event_kind == "qt_item":
        event = UnexpectedItem()
        assert isValid(event)
    elif event_kind == "deleted_event":
        event = QEvent(QEvent.Type.MouseButtonPress)
        delete(event)
        assert not isValid(event)
    else:
        event = object()

    assert service.eventFilter(viz, event) is False
    assert service.active and viz.animation_running
    service.stop()
    assert not service.active and not viz.animation_running


def install_workspace(viz, tmp_path):
    settings = QSettings(str(tmp_path / "demo.ini"), QSettings.IniFormat)
    workspace = VisualizationWorkspace(QWidget(), embedded=True, settings=settings)
    viz._visualization_workspace = workspace
    viz.setCentralWidget(workspace)
    workspace.viewport_host.set_state("active")
    viz.show()
    return workspace


def test_viewport_gesture_keeps_playback_and_effects_then_resumes_orbit(rig, tmp_path):
    viz, service, tools = rig
    workspace = install_workspace(viz, tmp_path)
    assert service.start(orbit_speed=4)
    service.eventFilter(workspace.viewport_host.canvas_parent, QEvent(QEvent.Type.Wheel))
    assert service.active and service.camera_paused and viz.animation_running
    assert tools.autorotate is None
    service.tick(service._next_look)
    assert viz.renderer.look_calls == 1 and service.active
    assert "Manual camera" in service.caption_text()
    assert service.resume_camera()
    assert tools.autorotate == (viz.renderer.camera.lookat, 4)
    assert not service.camera_paused
    service.stop()
    assert not viz.animation_running and tools.history == ["original-history"]


def test_tour_can_resume_and_stop_restores_original_named_views(rig, tmp_path):
    viz, service, _ = rig
    workspace = install_workspace(viz, tmp_path)
    tools = install_camera_tools(viz)
    before = deepcopy((tools.views, tools.shots, tools.history))
    viz.scenario.view_defaults = {"demo_tour": tour_definition()}
    try:
        assert service.start(tour=True, playback=False, cycle_lighting=False)
        service.eventFilter(
            workspace.viewport_host.canvas_parent, QEvent(QEvent.Type.MouseButtonPress)
        )
        service.tick()
        assert service.active and service.camera_paused and tools.sequence is None
        assert service.resume_camera() and tools.sequence is not None
        service.stop()
        assert (tools.views, tools.shots, tools.history) == before
    finally:
        tools.close()


def test_modified_viewport_gesture_keeps_demo_running(rig, tmp_path):
    viz, service, _ = rig
    workspace = install_workspace(viz, tmp_path)
    assert service.start()
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_Alt, Qt.AltModifier)
    assert not service.eventFilter(workspace.viewport_host.canvas_parent, event)
    assert service.active and viz.animation_running


def test_target_follow_yields_camera_without_cancelling_playback(rig):
    viz, service, _ = rig
    combo, _, tools = _install_targets(viz)
    try:
        assert service.start(orbit=False, cycle_targets=True, cycle_lighting=False)
        selected = combo.currentData()
        service.pause_camera()
        service.tick(service._next_target + 100)
        assert viz.app_state.camera_mode == "overview" and viz.animation_running
        assert combo.currentData() == selected
        assert service.resume_camera()
        assert viz.app_state.camera_mode == "follow"
        service.stop()
        assert combo.currentIndex() == 0 and viz.app_state.camera_mode == "overview"
        assert tools.follow_heading == "chase"
    finally:
        tools.close()


def test_clean_view_footer_resume_and_escape_restore_layout(rig, tmp_path):
    viz, service, tools = rig
    workspace = install_workspace(viz, tmp_path)
    viz.get_available_animation_steps = lambda: [0, 1]
    assert service.start(clean_view=True, captions=True)
    assert workspace.splitter.widget(0).isHidden()
    presentation = service._presentation_view
    service.pause_camera()
    resume = presentation.resume_button
    QTest.mouseClick(resume, Qt.LeftButton)
    assert service.active and not service.camera_paused and tools.autorotate is not None
    QTest.keyClick(workspace.viewport_host, Qt.Key_Escape)
    assert not service.active and not workspace.splitter.widget(0).isHidden()
    assert service._presentation_view is None


def test_replacing_scene_while_camera_paused_cleans_up(rig):
    viz, service, tools = rig
    assert service.start()
    service.pause_camera()
    viz.scenario = SimpleNamespace(data_mode="files")
    service.tick()
    assert not service.active and not service.camera_paused
    assert tools.autorotate is None and service._snapshot is None
