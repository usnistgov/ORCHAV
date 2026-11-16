"""Borrowed live plots share metrics data and only refresh visible consumers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from tests.visualizer.unit.test_viz_metrics import _deep_frame_stats, _deep_view_model
from visualizer.src.metrics.viz_metrics import MetricsDashboard
from visualizer.src.services.metrics_service import MetricsService
from visualizer.src.widgets.dashboard_host import DashboardHost


@pytest.fixture
def live_metrics(qtbot):
    dashboard = MetricsDashboard(frame_stats_provider=Mock(return_value=_deep_frame_stats()))
    dashboard.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    dashboard._hosted = True
    qtbot.addWidget(dashboard)
    hosts = []

    def pin(key):
        host = QWidget()
        qtbot.addWidget(host)
        hosts.append(host)
        layout = QVBoxLayout(host)
        plot = dashboard.acquire_live_figure(key, host)
        layout.addWidget(plot)
        host.show()
        dashboard.refresh_live_figure_visibility()
        return host, plot

    yield dashboard, pin
    dashboard.close()
    for host in hosts:
        host.close()


def test_borrow_returns_same_plot_and_reflow_preserves_dock_parent(live_metrics):
    dashboard, pin = live_metrics
    host, plot = pin("aoa_az")
    plot.setXRange(-30, 45, padding=0)
    previous_range = plot.viewRange()

    assert plot is dashboard.angle_plots["aoa_az"]
    assert dashboard.acquire_live_figure("aoa_az", host) is plot
    with pytest.raises(ValueError, match="already displayed"):
        dashboard.acquire_live_figure("aoa_az", QWidget())
    dashboard.resize(1000, 700)
    dashboard._overview_columns = 0
    dashboard._reflow_overview_plots()
    assert plot.parentWidget() is host
    assert dashboard._overview_grid.indexOf(plot) == -1

    dashboard.release_live_figure("aoa_az")

    assert not dashboard._live_figure_leases
    assert dashboard._overview_grid.indexOf(plot) >= 0
    assert plot.parentWidget() is not host
    assert np.allclose(plot.viewRange(), previous_range)


def test_hidden_dashboard_refreshes_only_selected_histogram_without_statistics(live_metrics):
    dashboard, pin = live_metrics
    pin("aoa_az")
    dashboard._update_summary = Mock(side_effect=AssertionError("hidden summary"))
    dashboard._update_material_breakdown = Mock(side_effect=AssertionError("hidden materials"))
    histogram = Mock(wraps=dashboard._update_histogram)
    dashboard._update_histogram = histogram
    dashboard.enqueue(_deep_view_model())

    dashboard._periodic_refresh()

    dashboard._frame_stats_provider.assert_not_called()
    assert histogram.call_count == 1
    assert histogram.call_args.args[0] is dashboard.angle_plots["aoa_az"]
    assert sum(dashboard.angle_bars["aoa_az"].opts["height"]) == 3
    assert len(dashboard.delay_bars.opts["height"]) == 0
    dashboard._update_summary.assert_not_called()


@pytest.mark.parametrize("key", ["loss", "pdp", "aod_map", "delay_power"])
def test_borrowed_figure_keeps_empty_message_across_data_transitions(live_metrics, qtbot, key):
    from shared.statistics import FrameStats

    dashboard, pin = live_metrics
    host, plot = pin(key)
    message = plot._empty_message
    for available in (True, False, False, True):
        vm = _deep_view_model()
        if not available:
            vm.path_mask[:] = False
        dashboard._frame_stats_provider.return_value = (
            _deep_frame_stats() if available else FrameStats(total_paths=0, orders_hist={})
        )
        dashboard.enqueue(vm)
        dashboard._periodic_refresh()
        assert plot._empty_message is message
        assert message.isVisible() != available
        assert plot.parentWidget() is host
    dashboard._clear_plots()
    for width in (360, 700):
        host.resize(width, 400)
        qtbot.waitUntil(
            lambda: abs(message.geometry().center().x() - plot.getViewBox().rect().center().x()) < 1
        )
        assert abs(message.geometry().center().y() - plot.getViewBox().rect().center().y()) < 1
    dashboard.release_live_figure(key)
    assert plot._empty_message is message and message.isVisible()


def test_two_figures_share_one_current_frame_statistics_result(live_metrics):
    dashboard, pin = live_metrics
    pin("pdp")
    pin("order")
    vm = _deep_view_model()
    dashboard.enqueue(vm)

    dashboard._periodic_refresh()
    dashboard._mark_pending_refresh()
    dashboard._periodic_refresh()

    dashboard._frame_stats_provider.assert_called_once_with(vm)
    assert len(dashboard.binned_pdp_dots.getData()[0]) == 3
    assert sum(dashboard.order_bars.opts["height"]) == 3


def test_hidden_and_minimized_figure_hosts_stop_refresh_work(live_metrics):
    dashboard, pin = live_metrics
    host, _plot = pin("pdp")
    dashboard.enqueue(_deep_view_model())
    host.hide()
    dashboard.refresh_live_figure_visibility()
    dashboard._periodic_refresh()
    assert not dashboard.has_visible_live_figures()
    assert not dashboard._timer.isActive()
    dashboard._frame_stats_provider.assert_not_called()

    host.show()
    dashboard.refresh_live_figure_visibility()
    dashboard._periodic_refresh()
    assert dashboard._frame_stats_provider.call_count == 1

    host.showMinimized()
    dashboard.enqueue(_deep_view_model())
    dashboard.refresh_live_figure_visibility()
    dashboard._periodic_refresh()
    assert dashboard._frame_stats_provider.call_count == 1
    assert not dashboard._timer.isActive()


def test_visible_dashboard_skips_hidden_borrowed_angle(live_metrics):
    dashboard, pin = live_metrics
    host, plot = pin("aoa_az")
    host.hide()
    dashboard.show()
    dashboard.refresh_live_figure_visibility()
    histogram = Mock(wraps=dashboard._update_histogram)
    dashboard._update_histogram = histogram
    dashboard.enqueue(_deep_view_model())

    dashboard._periodic_refresh()

    assert plot not in [call.args[0] for call in histogram.call_args_list]
    assert histogram.call_count == 5  # Delay, loss and the other three angles.


def test_channel_pin_refreshes_alongside_overview_without_changing_dashboard_tab(live_metrics):
    dashboard, pin = live_metrics
    pin("aod_map")
    dashboard.show()
    dashboard.enqueue(_deep_view_model())

    dashboard._periodic_refresh()

    assert dashboard.tabs.currentWidget() is dashboard._overview_tab
    assert len(dashboard.direction_scatters["aod"].getData()[0]) == 3
    assert len(dashboard.direction_scatters["aoa"].getData()[0]) == 0


def test_theme_and_grid_controls_still_reach_borrowed_plots(live_metrics):
    dashboard, pin = live_metrics
    host, plot = pin("delay")
    dashboard.overview_grid_cb.setChecked(True)
    dashboard._apply_theme()
    assert plot.parentWidget() is host
    assert plot.getAxis("bottom").grid
    assert plot.getAxis("left").grid


def test_owner_close_recovers_borrowed_figures_before_releasing_plots(live_metrics):
    dashboard, pin = live_metrics
    host, plot = pin("delay")
    dashboard.close()
    assert not dashboard._live_figure_leases
    assert plot.parentWidget() is dashboard
    assert not dashboard._timer.isActive()
    host.close()


def test_service_lazily_reuses_live_figure_owner_in_statistics_host(qtbot):
    viz = SimpleNamespace(current_view_model=_deep_view_model())
    service = MetricsService(viz)
    assert service.window is None
    service._compute_frame_stats = Mock(return_value=_deep_frame_stats())
    dashboard = service.ensure_live_figure_dashboard()
    dashboard.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dashboard)
    assert service.ensure_live_figure_dashboard() is dashboard
    assert not dashboard.isVisible()
    service._compute_frame_stats.assert_not_called()

    host = DashboardHost(title="Metrics", settings_key="metrics_live_figure_test")
    qtbot.addWidget(host)
    service.attach_host(host)
    assert service.window is dashboard
    assert host.dashboard is dashboard
    service.close()


def test_service_offers_data_to_visible_pin_and_stops_when_it_is_hidden(qtbot):
    viz = SimpleNamespace(current_view_model=_deep_view_model())
    service = MetricsService(viz)
    dashboard = service.ensure_live_figure_dashboard()
    dashboard.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dashboard)
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    layout.addWidget(dashboard.acquire_live_figure("delay", host))
    host.show()
    dashboard.refresh_live_figure_visibility()
    assert dashboard._last_vm is viz.current_view_model
    dashboard._periodic_refresh()

    host.hide()
    dashboard.refresh_live_figure_visibility()
    previous_vm = dashboard._last_vm
    viz.current_view_model = _deep_view_model()
    service.update_metrics(viz.current_view_model)
    assert dashboard._last_vm is previous_vm

    host.show()
    dashboard.refresh_live_figure_visibility()
    assert dashboard._last_vm is viz.current_view_model
    dashboard.release_live_figure("delay")
    service.close()
