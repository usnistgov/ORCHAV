"""Metrics computation is opt-in, visibility-gated, and explicitly freezable."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QPushButton, QScrollArea, QTabWidget, QWidget
from shiboken6 import isValid

from tests.visualizer.unit.test_viz_metrics import _deep_frame_stats, _deep_view_model
from visualizer.src.panels.statistics_panel import StatisticsPanel
from visualizer.src.services.metrics_service import MetricsService
from visualizer.src.widgets.dashboard_host import DashboardHost


@pytest.fixture
def metrics(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "geometry.ini"), QSettings.IniFormat)
    monkeypatch.setattr("visualizer.src.widgets.dashboard_host.QSettings", lambda: settings)
    viz = SimpleNamespace(current_view_model=_deep_view_model(), animation_step=0)
    calls = []
    service = viz.metrics_service = MetricsService(viz)
    service._compute_frame_stats = lambda vm: calls.append(vm) or _deep_frame_stats()
    panel = StatisticsPanel(viz)
    tabs = QTabWidget()
    tabs.addTab(QWidget(), "Scene")
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel.create_panel())
    tabs.addTab(scroll, "Statistics")
    viz.ui_manager = SimpleNamespace(
        panels={"statistics": panel}, show_panel=lambda _: tabs.setCurrentIndex(1)
    )
    tabs.resize(460, 950)
    qtbot.addWidget(tabs)
    tabs.show()
    yield viz, service, panel, tabs, scroll, calls
    panel.cleanup()
    tabs.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_metrics_start_collapsed_without_allocating_or_computing(metrics):
    viz, service, panel, tabs, _, calls = metrics
    tabs.setCurrentIndex(1)
    service.update_metrics(viz.current_view_model)
    assert not panel._metrics_section.is_expanded()
    assert service.window is None
    assert panel._metrics_host is None
    assert calls == []


def test_metrics_menu_reveals_existing_analysis_from_another_tab(metrics, qtbot):
    _, service, panel, tabs, _, calls = metrics
    service.toggle_window()
    qtbot.waitUntil(lambda: len(calls) == 1)
    dashboard = service.window
    tabs.setCurrentIndex(0)
    service.toggle_window()
    assert tabs.currentIndex() == 1
    assert panel._metrics_section.is_expanded()
    assert service.window is dashboard
    service.toggle_window()
    assert not panel._metrics_section.is_expanded()


def test_expand_consumes_latest_frame_once_and_collapse_stops_pending_work(metrics, qtbot):
    viz, service, panel, _, _, calls = metrics
    panel.show_metrics()
    qtbot.waitUntil(lambda: len(calls) == 1)
    dashboard = service.window
    service.update_metrics(viz.current_view_model)
    dashboard._periodic_refresh()
    assert len(calls) == 1
    viz.current_view_model = _deep_view_model()
    service.update_metrics(viz.current_view_model)
    panel._metrics_section.collapse()
    dashboard._periodic_refresh()
    assert len(calls) == 1
    assert not dashboard._timer.isActive()
    panel.show_metrics()
    qtbot.waitUntil(lambda: len(calls) == 2)
    assert calls[-1] is viz.current_view_model
    assert service.window is dashboard


@pytest.mark.parametrize("floating", [False, True])
def test_minimize_suspends_calculations_and_restore_consumes_latest(metrics, qtbot, floating):
    viz, service, panel, tabs, _, calls = metrics
    panel.show_metrics()
    qtbot.waitUntil(lambda: len(calls) == 1)
    dashboard = service.window
    if floating:
        panel._metrics_host.detach()
        shell = panel._metrics_host.floating_window
    else:
        shell = tabs
    viz.current_view_model = _deep_view_model()
    service.update_metrics(viz.current_view_model)
    shell.showMinimized()
    dashboard._periodic_refresh()
    assert len(calls) == 1
    viz.current_view_model = _deep_view_model()
    service.update_metrics(viz.current_view_model)
    assert len(calls) == 1
    shell.showNormal()
    qtbot.waitUntil(lambda: len(calls) == 2)
    assert calls[-1] is viz.current_view_model


def test_hidden_tab_suspends_and_freeze_survives_return(metrics, qtbot):
    viz, service, panel, tabs, _, calls = metrics
    panel.show_metrics()
    qtbot.waitUntil(lambda: len(calls) == 1)
    dashboard = service.window
    tabs.setCurrentIndex(0)
    viz.current_view_model = _deep_view_model()
    service.update_metrics(viz.current_view_model)
    dashboard._periodic_refresh()
    assert len(calls) == 1
    dashboard.pause_cb.setChecked(True)
    tabs.setCurrentIndex(1)
    dashboard._periodic_refresh()
    assert len(calls) == 1
    assert dashboard.updates_paused


def test_freeze_keeps_results_and_refresh_once_uses_current_frame(metrics, qtbot):
    viz, service, panel, _, _, calls = metrics
    panel.show_metrics()
    qtbot.waitUntil(lambda: len(calls) == 1)
    dashboard = service.window
    previous = dashboard._last_vm
    dashboard.pause_cb.setChecked(True)
    viz.current_view_model = _deep_view_model()
    service.update_metrics(viz.current_view_model)
    dashboard._periodic_refresh()
    assert calls == [previous]
    assert dashboard._last_vm is previous
    dashboard.refresh_btn.click()
    assert len(calls) == 2
    assert calls[-1] is viz.current_view_model
    assert dashboard.updates_paused and not dashboard._timer.isActive()
    assert "Frozen" in dashboard.data_status_label.text()
    viz.current_view_model = _deep_view_model()
    service.update_metrics(viz.current_view_model)
    dashboard.pause_cb.setChecked(False)
    qtbot.waitUntil(lambda: len(calls) == 3)
    assert calls[-1] is viz.current_view_model


def test_popout_return_retains_results_zoom_and_freeze_and_releases_on_cleanup(metrics, qtbot):
    _, service, panel, tabs, _, calls = metrics
    panel.show_metrics()
    qtbot.waitUntil(lambda: len(calls) == 1)
    dashboard = service.window
    dashboard.pause_cb.setChecked(True)
    dashboard.delay_plot.setRange(xRange=(3, 30), yRange=(0, 5), padding=0)
    ranges = dashboard.delay_plot.viewRange()
    data = dashboard._last_stats
    dashboard.placement_button.click()
    shell = panel._metrics_host.floating_window
    tabs.setCurrentIndex(0)
    shell.close()
    assert tabs.currentIndex() == 1
    assert dashboard.parentWidget() is panel._metrics_host
    assert dashboard._last_stats is data and dashboard.updates_paused
    np.testing.assert_allclose(dashboard.delay_plot.viewRange(), ranges)
    assert len(calls) == 1
    panel.cleanup()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(dashboard) and not isValid(shell)
    assert service.window is None


@pytest.mark.parametrize("width,columns", [(380, 1), (460, 1), (1150, 2)])
def test_metrics_overview_reflows_complete_figures_at_panel_and_window_widths(
    metrics, qtbot, width, columns
):
    _, service, panel, tabs, scroll, _ = metrics
    panel.show_metrics()
    dashboard = service.window
    tabs.resize(width, 950)
    qtbot.waitUntil(dashboard.delay_plot.isVisible)
    qtbot.waitUntil(lambda: dashboard._overview_columns == columns)
    grid = dashboard._overview_grid
    for index, plot in enumerate(dashboard._overview_plots):
        assert grid.getItemPosition(grid.indexOf(plot)) == (index // columns, index % columns, 1, 1)
        assert plot.isVisible()
    assert dashboard._overview_tab.horizontalScrollBar().maximum() == 0
    assert scroll.horizontalScrollBar().maximum() == 0
    dashboard.tabs.setCurrentWidget(dashboard._channel_tab)
    qtbot.waitUntil(dashboard.direction_plots["aoa"].isVisible)
    assert dashboard.direction_plots["aod"].isVisible()
    assert not dashboard.overview_grid_cb.isVisible()
    assert scroll.horizontalScrollBar().maximum() == 0


@pytest.mark.parametrize("save_action", ["dock", "dispose"])
def test_floating_geometry_survives_host_recreation(metrics, qtbot, save_action):
    _, _, panel, _, _, _ = metrics
    panel.show_metrics()
    host = panel._metrics_host
    host.detach()
    shell = host.floating_window
    shell.setGeometry(70, 60, 610, 500)
    qtbot.waitUntil(lambda: shell.width() == 610)
    geometry = shell.geometry()
    getattr(host, save_action)()
    host._settings.sync()
    # Recreate just the placement owner with the same settings, as on next launch.
    fresh = DashboardHost(
        title="Metrics",
        settings_key=host._settings_key,
        settings=QSettings(host._settings.fileName(), QSettings.IniFormat),
    )
    qtbot.addWidget(fresh)
    dashboard = QWidget()
    dashboard.placement_button = QPushButton("Pop out", dashboard)
    fresh.set_dashboard(dashboard)
    fresh.show()
    fresh.detach()
    assert fresh.floating_window.geometry() == geometry
    fresh.dock()
    fresh.dispose()
