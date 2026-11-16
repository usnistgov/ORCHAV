"""Sensing exhibition borrows actual plots and restores their owners and controls."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QMainWindow, QSplitter, QVBoxLayout, QWidget

from tests.visualizer.fixtures.sensing_v3 import materialize_visualizer_product_specs
from tests.visualizer.unit.test_live_figure_dock import Dashboard as ChannelDashboard
from tests.visualizer.unit.test_sensing_dashboard_recording import _request
from visualizer.src.sensing.runtime import state_value
from visualizer.src.sensing.sensing_window import SensingDashboard
from visualizer.src.services.demo_sensing_layout import SensingExhibitionLayout
from visualizer.src.services.demo_service import DemoService
from visualizer.src.state import create_initial_state
from visualizer.src.widgets.live_figure_dock import LiveFigureManager


@pytest.fixture
def exhibition(qtbot):
    viz = QMainWindow()
    qtbot.addWidget(viz)
    viz.resize(1280, 720)
    controls, viewport = QWidget(), QWidget()
    splitter = QSplitter()
    splitter.addWidget(controls)
    splitter.addWidget(viewport)
    viz.setCentralWidget(splitter)
    viz._visualization_workspace = SimpleNamespace(splitter=splitter, viewport_host=viewport)
    viz.app_state = create_initial_state()
    viz.set_state = lambda **changes: setattr(viz, "app_state", replace(viz.app_state, **changes))
    viz.renderer = SimpleNamespace(
        open_target_view=Mock(side_effect=AssertionError("No actor view"))
    )
    viz.frame_source = object()
    dashboard = SensingDashboard(parent=controls)
    QVBoxLayout(controls).addWidget(dashboard)
    dashboard._set_view_mode("current")
    dashboard.update_from_request(_request())
    panel = SimpleNamespace(
        widgets={key: QCheckBox(controls) for key in ("load_sensing_figures", "sensing_playback")},
        ensure_live_figure_dashboard=lambda: dashboard,
    )
    viz.ui_manager = SimpleNamespace(panels={"sensing": panel}, ensure_panel=Mock())
    channel = ChannelDashboard(viz)
    viz.metrics_service = SimpleNamespace(ensure_live_figure_dashboard=lambda: channel)
    manager = viz.live_figure_manager = LiveFigureManager(viz)
    failed = Mock()
    layout = SensingExhibitionLayout(viz, failed)
    viz.show()
    yield viz, dashboard, panel, manager, layout, failed
    layout.restore()
    manager.close()


def test_presentation_uses_two_real_figures_and_restores_display_state(exhibition, qapp):
    viz, dashboard, panel, manager, layout, failed = exhibition
    dashboard._show_gt_cb.setChecked(False)
    original = {key: state_value(viz.app_state, key) for key in layout.OVERLAYS}
    checkboxes = (
        *panel.widgets.values(),
        dashboard._show_det_cb,
        dashboard._show_gt_cb,
        dashboard._profile_detections_cb,
        dashboard._profile_full_dynamic_range_cb,
    )
    values = [c.isChecked() for c in checkboxes]
    plot_ids = id(dashboard.rd_plot), id(dashboard.rp_plot)
    layout.enter()
    qapp.processEvents()
    assert [v.choice_id for v in layout.chosen] == list(layout.FIGURES)
    assert all(v.isVisible() and not v.isFloating() for v in layout.chosen)
    assert all(c.isChecked() for c in panel.widgets.values())
    assert dashboard._profile_detections_cb.isChecked()
    assert dashboard._show_gt_cb.isChecked()
    assert dashboard._profile_full_dynamic_range_cb.isChecked()
    assert dashboard.rd_image.image is not None
    assert dashboard.rp_curve.getData()[0] is not None
    assert viz._visualization_workspace.splitter.widget(0).isHidden()
    failed.assert_not_called()
    layout.restore()
    assert not manager.views
    assert {key: state_value(viz.app_state, key) for key in layout.OVERLAYS} == original
    assert [c.isChecked() for c in checkboxes] == values
    assert (id(dashboard.rd_plot), id(dashboard.rp_plot)) == plot_ids
    dashboard.update_from_request(_request(2))
    assert dashboard._last_request.state_step == 2


def test_presentation_shows_truth_and_deep_profile_samples(exhibition, qtbot, monkeypatch):
    _viz, dashboard, _panel, _manager, layout, _failed = exhibition
    request = _request()
    request.rd_result.power_linear[:] = [1e-10, 1e-20, 1e-18]
    materialize_visualizer_product_specs(request.display_frame["sensing"])
    saved_power = request.rd_result.power_linear.copy()
    monkeypatch.setattr(dashboard, "_current_gt_range_speed", lambda: ([1.0], [0.0]))
    dashboard._show_gt_cb.setChecked(False)
    dashboard.update_from_request(request)
    profile = dashboard.rp_curve.getData()[1].copy()
    assert dashboard.rp_plot.viewRange()[1][0] > profile.min()

    layout.enter()

    def check_presentation():
        assert dashboard.rd_gt_scatter.isVisible()
        np.testing.assert_allclose(dashboard.rd_gt_scatter.getData(), [[1.0], [0.0]])
        legend = dashboard.rd_plot.getPlotItem().legend
        assert legend.getLabel(dashboard.rd_gt_scatter).isVisible()
        assert legend.getLabel(dashboard.rd_gt_scatter).text == "Ground truth"
        assert dashboard.rp_plot.viewRange()[1][0] < profile.min()

    qtbot.waitUntil(check_presentation)
    next_request = _request(1)
    next_request.rd_result.power_linear[:] = [1e-8, 1e-30, 1e-19]
    materialize_visualizer_product_specs(next_request.display_frame["sensing"])
    dashboard.update_from_request(next_request)
    profile = dashboard.rp_curve.getData()[1].copy()
    qtbot.waitUntil(check_presentation)
    layout.restore()
    assert not dashboard.rd_gt_scatter.isVisible()
    assert dashboard.rp_plot.viewRange()[1][0] > profile.min()
    np.testing.assert_array_equal(dashboard.rp_curve.getData()[1], profile)
    np.testing.assert_array_equal(request.rd_result.power_linear, saved_power)


def test_four_existing_windows_retain_choices_visibility_and_geometry(exhibition, qapp):
    viz, dashboard, panel, manager, layout, failed = exhibition
    views = [manager.add() for _ in range(4)]
    for view, choice in zip(views, ("pdp", "angle", "delay", "loss")):
        view.figure_combo.setCurrentIndex(view.figure_combo.findData(f"Channel:{choice}"))
    views[2].hide()
    qapp.processEvents()
    before = [(v.choice_id, v.isFloating(), v.isHidden(), v.geometry()) for v in views]
    layout.enter()
    assert len(manager.views) == 4
    assert [v.choice_id for v in layout.chosen] == list(layout.FIGURES)
    layout.restore()
    qapp.processEvents()
    assert manager.views == views
    assert [(v.choice_id, v.isFloating(), v.isHidden(), v.geometry()) for v in views] == before
    failed.assert_not_called()


def test_reused_figure_choice_returns_after_manual_switch(exhibition):
    _viz, _dashboard, _panel, manager, layout, _failed = exhibition
    existing = manager.add()
    assert existing.choice_id == "Sensing:rd_plot"
    layout.enter()
    existing.figure_combo.setCurrentIndex(existing.figure_combo.findData("Channel:pdp"))
    assert existing.choice_id == "Channel:pdp"
    layout.restore()
    assert existing.choice_id == "Sensing:rd_plot"
    assert manager.views == [existing]


def test_interacting_with_presentation_figure_does_not_stop_demo(exhibition):
    viz, _dashboard, _panel, _manager, layout, _failed = exhibition
    layout.enter()
    service = DemoService(viz)
    service.active = True
    service._exhibition = layout
    service.stop = Mock()
    assert service.eventFilter(layout.chosen[0].widget(), QEvent(QEvent.Wheel)) is False
    service.stop.assert_not_called()
    service.active = False


def test_plot_mouse_gesture_does_not_trigger_control_takeover(exhibition, qapp):
    viz, dashboard, _panel, _manager, layout, _failed = exhibition
    layout.enter()
    service = DemoService(viz)
    service.active = True
    service._exhibition = layout
    service.stop = Mock()
    qapp.installEventFilter(service)
    try:
        viewport = dashboard.rd_plot.viewport()
        QTest.mousePress(viewport, Qt.LeftButton, pos=viewport.rect().center())
        QTest.mouseRelease(viewport, Qt.LeftButton, pos=viewport.rect().center())
        service.stop.assert_not_called()
    finally:
        qapp.removeEventFilter(service)
        service.active = False


def test_known_missing_maps_are_rejected_before_creating_docks(exhibition):
    _viz, dashboard, _panel, manager, layout, _failed = exhibition
    dashboard._last_request = replace(_request(), available_product_kinds=frozenset())
    with pytest.raises(ValueError, match="range–velocity"):
        layout.enter()
    layout.restore()
    assert not manager.views


def test_lazy_catalog_does_not_override_user_choice_after_first_binding(exhibition, monkeypatch):
    _viz, dashboard, _panel, manager, layout, _failed = exhibition
    catalog = dashboard.available_live_figures
    monkeypatch.setattr(
        dashboard,
        "available_live_figures",
        lambda: [(key, title) for key, title in catalog() if key != "rp_plot"],
    )
    layout.enter()
    first, second = layout.chosen
    assert first.choice_id == "Sensing:rd_plot"
    assert layout.pending_figures == [(second, "Sensing:rp_plot")]
    first.figure_combo.setCurrentIndex(first.figure_combo.findData("Channel:angle"))
    assert first.choice_id == "Channel:angle"
    monkeypatch.setattr(dashboard, "available_live_figures", catalog)
    layout._select_figures()
    assert first.choice_id == "Channel:angle"
    assert second.choice_id == "Sensing:rp_plot"
    assert not layout.pending_figures
    assert not layout._timer.isActive()
