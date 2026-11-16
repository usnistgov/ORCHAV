"""Demo footer and clean-view restoration use the existing Qt workspace."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDockWidget, QMainWindow, QSlider, QToolBar, QWidget

from visualizer.src.app.viewport_workspace import ViewportState, VisualizationWorkspace
from visualizer.src.services.demo_presentation import DemoPresentation


class Demo(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.camera_paused = False
        self.caption_text = Mock(return_value="Current look · recorded playback · paths by delay")
        self.stop = Mock()
        self.resume_camera = Mock()


@pytest.fixture
def presentation(qtbot, qapp):
    viz = QMainWindow()
    qtbot.addWidget(viz)
    viz.resize(1200, 800)
    settings = SimpleNamespace(value=lambda _: None)
    workspace = VisualizationWorkspace(QWidget(), embedded=True, settings=settings)
    viz.setCentralWidget(workspace)
    viz._visualization_workspace = workspace
    workspace.viewport_host.set_state(ViewportState.ACTIVE)
    viz.scenario = SimpleNamespace(root=Path("/scenarios/hello_world_scripted"))
    viz.menuBar().addMenu("File")
    viz.statusBar().showMessage("Ready")
    toolbar = QToolBar("Main tools", viz)
    toolbar.setObjectName("main_tools")
    viz.addToolBar(toolbar)
    for name, floating, visible in (("visible", True, True), ("hidden", False, False)):
        dock = QDockWidget(name, viz)
        dock.setObjectName(name)
        dock.setWidget(QWidget())
        viz.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setFloating(floating)
        dock.setVisible(visible)
    viz.show()
    workspace.splitter.setSizes([350, 850])
    qapp.processEvents()
    demo = Demo()
    view = DemoPresentation(viz, demo)
    yield viz, demo, view
    view.restore()


def test_clean_view_restores_all_visibility_and_canvas_parent(presentation, qapp):
    viz, service, view = presentation
    workspace = viz._visualization_workspace
    host = workspace.viewport_host
    canvas_parent = host.canvas_parent
    controls = workspace.splitter.widget(0)
    docks = viz.findChildren(QDockWidget)
    chrome = [viz.menuWidget(), viz.statusBar(), *viz.findChildren(QToolBar)]
    before_visibility = [(widget, not widget.isHidden()) for widget in (*docks, *chrome)]
    sizes = workspace.splitter.sizes()
    original_count = host.layout().count()
    original_spacing = host.layout().spacing()
    viz.statusBar().hide()
    before_visibility = [(w, not w.isHidden()) for w, _ in before_visibility]

    view.enter(clean_view=True, captions=True)
    qapp.processEvents()
    assert controls.isHidden()
    assert all(widget.isHidden() for widget in (*docks, *chrome))
    assert host.layout().count() == original_count + 1
    assert host.layout().spacing() == 0
    assert host.canvas_parent is canvas_parent
    assert not view.footer.isHidden()
    assert view.title.text() == "Hello world scripted"
    assert view.caption.text() == service.caption_text()
    assert view.resume_button.isHidden()
    assert view.owns_widget(view.stop_button)
    assert not view.owns_widget(controls)

    view.restore()
    qapp.processEvents()
    assert not controls.isHidden()
    assert host.layout().count() == original_count
    assert host.layout().spacing() == original_spacing
    assert all((not widget.isHidden()) == visible for widget, visible in before_visibility)
    assert workspace.splitter.sizes() == sizes
    assert next(dock for dock in docks if dock.objectName() == "visible").isFloating()
    assert not view.owns_widget(controls)
    view.restore()  # Closing twice must not replace the captured arrangement.


def test_captions_can_run_without_hiding_controls_and_update_actions(presentation, qtbot):
    viz, service, view = presentation
    view.enter(clean_view=False, captions=True)
    assert not viz._visualization_workspace.splitter.widget(0).isHidden()
    assert not viz.menuWidget().isHidden()
    service.camera_paused = True
    service.caption_text.return_value = "Glow · slowed geometric MPC travel (illustration)"
    service.changed.emit()
    assert not view.resume_button.isHidden()
    assert view.caption.text() == service.caption_text()
    qtbot.mouseClick(view.resume_button, Qt.LeftButton)
    service.resume_camera.assert_called_once_with()
    qtbot.mouseClick(view.stop_button, Qt.LeftButton)
    service.stop.assert_called_once_with()
    view.restore()
    service.caption_text.reset_mock()
    service.changed.emit()
    service.caption_text.assert_not_called()


def test_clean_view_without_captions_keeps_exit_controls(presentation):
    _viz, _service, view = presentation
    view.enter(clean_view=True, captions=False)
    assert view.caption.isHidden()
    assert not view.stop_button.isHidden()


def test_compact_footer_reuses_player_mapping_and_stops_its_timer(presentation, qapp):
    viz, _service, view = presentation
    viz.step_slider = QSlider(Qt.Horizontal, viz)
    viz.step_slider.setRange(0, 79)
    viz.step_slider.setValue(5)
    viz.animation_running = True
    viz.animation_step = 640
    viz.sensing_playback = SimpleNamespace(timestamps={640: 0.32})
    viz.animation_controller = SimpleNamespace(toggle_animation=Mock())
    viz.flush_pending_slider_scrub = Mock()
    view.enter(clean_view=False, captions=True, compact=True, playback=True)
    qapp.processEvents()
    assert view.title.isHidden()
    assert view.seek_slider.value() == 5 and view.seek_slider.maximum() == 79
    assert view.time_label.text() == "0.32 s · 6/80"
    assert view.play_button.text() == "Pause"
    view.play_button.click()
    viz.animation_controller.toggle_animation.assert_called_once()
    view.seek_slider.setValue(12)
    assert viz.step_slider.value() == 12
    viz.flush_pending_slider_scrub.assert_called_once()
    timer = view._playback_timer
    view.restore()
    assert not timer.isActive()


def test_authored_tour_label_names_the_scene_without_showing_its_path(presentation):
    viz, _service, view = presentation
    viz.scenario.view_defaults = {"demo_tour": {"label": "Paris Opéra"}}
    view.enter(clean_view=False, captions=True)
    assert view.title.text() == "Paris Opéra"


def test_footer_text_is_plain_and_wraps_on_narrow_windows(presentation, qapp):
    viz, service, view = presentation
    viz.scenario.name = "<b>A scene with a long name</b>"
    service.caption_text.return_value = "Recorded paths and their delays · " * 8
    view.enter(clean_view=True, captions=True)
    viz.resize(360, 650)
    qapp.processEvents()
    assert view.title.textFormat() == Qt.PlainText
    assert view.caption.textFormat() == Qt.PlainText
    assert view.title.text() == viz.scenario.name
    assert view.caption.wordWrap()
    assert view.footer.width() <= viz.width()
    assert view.stop_button.geometry().right() < view.footer.width()


def test_unsupported_host_is_rejected_before_hiding_anything(presentation):
    viz, _service, view = presentation
    viz._visualization_workspace.viewport_host = QWidget()
    with pytest.raises(ValueError, match="embedded"):
        view.enter(clean_view=True, captions=True)
    assert view.state is None
    assert not viz.menuWidget().isHidden()
    assert not viz._visualization_workspace.splitter.widget(0).isHidden()


def test_partial_enter_failure_still_restores_controls(presentation):
    viz, service, view = presentation
    service.caption_text.side_effect = RuntimeError("caption unavailable")
    with pytest.raises(RuntimeError, match="caption unavailable"):
        view.enter(clean_view=True, captions=True)
    assert viz._visualization_workspace.splitter.widget(0).isHidden()
    view.restore()
    assert not viz._visualization_workspace.splitter.widget(0).isHidden()
    assert not viz.menuWidget().isHidden()
    assert view.footer is None


def test_scene_replacement_restores_chrome_without_resurrecting_docks(presentation):
    viz, _service, view = presentation
    view.enter(clean_view=True, captions=True)
    restore = viz.restoreState = Mock()
    view.restore(restore_docks=False)
    restore.assert_not_called()
    assert all(dock.isHidden() for dock in viz.findChildren(QDockWidget))
    assert not viz._visualization_workspace.splitter.widget(0).isHidden()
    assert not viz.menuWidget().isHidden()
