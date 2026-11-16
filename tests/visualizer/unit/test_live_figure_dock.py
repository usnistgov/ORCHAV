"""Plot docks borrow widgets without duplicating dashboard data or timers."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget

from visualizer.src.widgets.live_figure_dock import MAX_LIVE_FIGURES, LiveFigureManager


class Dashboard(QWidget):
    live_figures_changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.plots = {key: QLabel(key, self) for key in ("pdp", "angle", "delay", "loss", "order")}
        self.borrowed = set()
        self.refreshes = 0

    def available_live_figures(self):
        return tuple((key, key.title()) for key in self.plots)

    def acquire_live_figure(self, key, parent):
        assert key not in self.borrowed
        self.borrowed.add(key)
        self.plots[key].setParent(parent)
        return self.plots[key]

    def release_live_figure(self, key):
        self.borrowed.remove(key)
        self.plots[key].setParent(self)
        self.plots[key].hide()

    def refresh_live_figure_visibility(self):
        self.refreshes += 1


@pytest.fixture
def rig(qtbot):
    viz = QMainWindow()
    viz.resize(1400, 1000)
    viz.setCentralWidget(QWidget())
    qtbot.addWidget(viz)
    owner = Dashboard(viz)
    viz.frame_source = object()
    viz.metrics_service = SimpleNamespace(ensure_live_figure_dashboard=lambda: owner)
    manager = LiveFigureManager(viz)
    viz.show()
    yield viz, owner, manager
    manager.close()


def test_dock_cap_and_release_return_same_figures(rig):
    _viz, owner, manager = rig
    views = [manager.add() for _ in range(MAX_LIVE_FIGURES)]
    assert MAX_LIVE_FIGURES == 4
    assert all(views)
    assert len(owner.borrowed) == 4
    assert manager.add() is None
    assert len(manager.views) == 4
    for view in views:
        assert view.plot is owner.plots[view.figure_key]
    first = views[0]
    key = first.figure_key
    first.close()
    assert key not in owner.borrowed
    assert owner.plots[key].parentWidget() is owner
    assert manager.add() is not None


def test_selection_moves_one_plot_and_prevents_duplicate_ownership(rig):
    _viz, owner, manager = rig
    first, second = manager.add(), manager.add()
    original_key = first.figure_key
    first.figure_combo.setCurrentIndex(first.figure_combo.findData("Channel:loss"))
    assert first.figure_key == "loss"
    assert original_key not in owner.borrowed
    taken_index = second.figure_combo.findData("Channel:loss")
    assert not second.figure_combo.model().item(taken_index).isEnabled()
    second.figure_combo.setCurrentIndex(taken_index)
    assert second.figure_key != "loss"
    assert first.plot is owner.plots["loss"]


def test_dock_hide_show_reconciles_refresh_without_data_loading(rig):
    _viz, owner, manager = rig
    view = manager.add()
    before = owner.refreshes
    view.hide()
    assert not view.plot.isVisible()
    view.show()
    assert owner.refreshes >= before + 2


def test_new_available_figures_refresh_choices_without_replacing_current_plot(rig):
    _viz, owner, manager = rig
    view = manager.add()
    plot = view.plot
    owner.plots["new"] = QLabel("New", owner)
    owner.live_figures_changed.emit()
    assert view.figure_combo.findData("Channel:new") >= 0
    assert view.plot is plot


def test_scenario_close_releases_every_plot_and_allows_reopening(rig):
    _viz, owner, manager = rig
    manager.add()
    manager.add()
    manager.close()
    assert not manager.views and not manager.sources and not owner.borrowed
    assert all(plot.parentWidget() is owner for plot in owner.plots.values())
    assert manager.add() is not None


def test_no_scene_does_not_create_a_dashboard(rig):
    viz, _owner, manager = rig
    viz.frame_source = None
    viz.metrics_service.ensure_live_figure_dashboard = lambda: pytest.fail("Should stay lazy")
    assert manager.add() is None


def test_live_figures_start_compact_and_preserve_other_window_sizes(rig, qtbot):
    _viz, _owner, manager = rig
    first = manager.add()
    assert first.isFloating()
    first.resize(310, 220)
    qtbot.waitUntil(lambda: first.height() == 220)
    before = first.size()
    second = manager.add()
    assert second.isFloating() and second.height() == 360
    assert first.size() == before


def test_borrowed_plot_can_shrink_and_restores_dashboard_height(rig, qtbot):
    from visualizer.src.widgets.dashboard_plot import DashboardPlot

    _viz, owner, manager = rig
    plot = owner.plots["pdp"] = DashboardPlot()
    plot.setParent(owner)
    plot.set_data_minimum_height(500)
    original = plot.minimumHeight()
    view = manager.add()
    assert view.plot is plot
    assert plot.minimumHeight() < 200
    plot.set_data_minimum_height(600)  # Dashboard preference may change while borrowed.
    assert plot.minimumHeight() < 200
    view.resize(350, 240)
    qtbot.waitUntil(lambda: view.height() == 240)
    view.close()
    assert plot.parentWidget() is owner
    assert plot.minimumHeight() >= original + 100
