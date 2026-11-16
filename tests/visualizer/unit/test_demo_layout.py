"""Exhibition layout borrows existing docks and returns their exact ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QComboBox, QDockWidget, QMainWindow, QSplitter, QWidget

from visualizer.src.services.demo_layout import ExhibitionLayout
from visualizer.src.services.demo_service import DemoService


class Dock(QDockWidget):
    def __init__(self, window, name):
        super().__init__(name, window)
        self.setObjectName(name)
        self.setWidget(QWidget())
        self.closed = False
        self.figure_combo = QComboBox(self.widget())
        self.figure_combo.addItem("PDP", "Channel:pdp")
        window.addDockWidget(Qt.RightDockWidgetArea, self)

    def closeEvent(self, event):
        self.closed = True
        super().closeEvent(event)


class SensingOwner(QWidget):
    live_figures_changed = Signal()


@pytest.fixture
def window(qtbot):
    viz = QMainWindow()
    viz.resize(1600, 1000)
    qtbot.addWidget(viz)
    splitter = QSplitter()
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    viz.setCentralWidget(splitter)
    viz._visualization_workspace = SimpleNamespace(splitter=splitter)
    viz.renderer = SimpleNamespace()
    viz.show()
    splitter.setSizes([420, 1180])
    return viz


def test_exhibition_reuses_one_of_each_and_restores_hidden_floating_docks(window, qapp):
    target, extra_target = Dock(window, "target1"), Dock(window, "target2")
    figure, extra_figure = Dock(window, "figure1"), Dock(window, "figure2")
    target.show()
    extra_target.hide()
    figure.setFloating(True)
    figure.show()
    extra_figure.show()
    qapp.processEvents()
    window.renderer.get_target_views = lambda: (extra_target, target)
    window.renderer.open_target_view = Mock(side_effect=AssertionError("must reuse target"))
    window.live_figure_manager = SimpleNamespace(
        views=[figure, extra_figure], add=Mock(side_effect=AssertionError("must reuse figure"))
    )
    splitter = window._visualization_workspace.splitter
    before_sizes = splitter.sizes()
    controls = splitter.widget(0)
    layout = ExhibitionLayout(window)
    layout.enter()
    qapp.processEvents()
    assert controls.isHidden()
    assert target.isVisible() and figure.isVisible()
    assert not figure.isFloating()
    assert extra_target.isHidden() and extra_figure.isHidden()
    assert splitter.widget(1).width() > window.width() * 0.65
    layout.restore()
    qapp.processEvents()
    assert not controls.isHidden()
    assert figure.isFloating() and figure.isVisible()
    assert extra_figure.isVisible() and extra_target.isHidden()
    assert not any(dock.closed for dock in (target, extra_target, figure, extra_figure))
    assert splitter.sizes() == before_sizes


def test_missing_views_are_created_only_on_enter_and_removed_on_restore(window, qapp):
    original = Dock(window, "original_auxiliary")
    original.show()
    target = Dock(window, "new_target")
    figure = Dock(window, "new_figure")
    target.hide()
    figure.hide()
    window.renderer.open_target_view = Mock(return_value=target)
    owner = SensingOwner(window)
    manager = SimpleNamespace(views=[], sources={"Sensing": owner}, add=Mock(return_value=figure))
    window.live_figure_manager = manager
    layout = ExhibitionLayout(window)
    window.renderer.open_target_view.assert_not_called()
    manager.add.assert_not_called()
    layout.enter()
    assert original.isHidden()
    window.renderer.open_target_view.assert_called_once()
    manager.add.assert_called_once()
    figure.figure_combo.addItem("RD", "Sensing:rd_plot")
    owner.live_figures_changed.emit()
    assert figure.figure_combo.currentData() == "Sensing:rd_plot"
    layout.restore()
    qapp.processEvents()
    assert target.closed and figure.closed
    assert original.isVisible() and not original.closed


def test_failed_optional_view_creation_can_restore_controls_and_docks(window):
    dock = Dock(window, "existing")
    dock.show()
    window.renderer.open_target_view = Mock(side_effect=RuntimeError("canvas failed"))
    layout = ExhibitionLayout(window)
    with pytest.raises(RuntimeError, match="canvas failed"):
        layout.enter()
    layout.restore()
    assert not window._visualization_workspace.splitter.widget(0).isHidden()
    assert dock.isVisible() and not dock.closed


def test_external_viewport_is_rejected_before_mutating_layout(window):
    window._visualization_workspace.splitter.widget(1).setParent(None)
    layout = ExhibitionLayout(window)
    with pytest.raises(ValueError, match="embedded"):
        layout.enter()
    assert layout.state is None


def test_takeover_consumes_only_a_gesture_on_a_retiring_temporary_dock(window):
    window.renderer.renderer_id = "pygfx"
    window.ready = True
    window.scenario = SimpleNamespace(data_mode="files")
    window.camera_controller = SimpleNamespace(presentation=lambda: None)
    window.app_state = SimpleNamespace(camera_mode="overview")
    target = Dock(window, "temporary_target")
    figure = Dock(window, "temporary_figure")
    target.hide()
    figure.hide()
    window.renderer.open_target_view = Mock(return_value=target)
    window.live_figure_manager = SimpleNamespace(
        views=[], sources={}, add=Mock(return_value=figure)
    )
    service = DemoService(window)
    try:
        assert service.start(orbit=False, playback=False, cycle_lighting=False, exhibition=True)
        assert service.eventFilter(target.widget(), QEvent(QEvent.Type.MouseButtonPress)) is True
        assert not service.active and target.closed and figure.closed
        assert not window._visualization_workspace.splitter.widget(0).isHidden()
        assert service.eventFilter(window, QEvent(QEvent.Type.MouseButtonPress)) is False
    finally:
        service.close()
