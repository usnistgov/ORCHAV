"""Display-only contracts for compact live MPC dashboard surfaces."""

import warnings
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest

from tests.visualizer.unit.test_viz_metrics import _deep_frame_stats, _deep_view_model
from visualizer.src.app.plot_theme import dashboard_series_color
from visualizer.src.app.theme import current_theme
from visualizer.src.metrics.viz_metrics import MetricsWindow


@pytest.mark.parametrize("size", [(1280, 800), (1600, 1000)])
@pytest.mark.parametrize("tab", [0, 1, 2])
def test_metrics_target_sizes_keep_figures_and_values_inside_surface(qapp, size, tab):
    window = MetricsWindow()
    window.enqueue(_deep_view_model(), _deep_frame_stats())
    window.resize(*size)
    window.show()
    window.tabs.setCurrentIndex(tab)
    window._refresh()
    qapp.processEvents()
    assert (window.width(), window.height()) == size
    assert window.metric_values["paths"].text() == "3"
    assert all(card.isVisible() for card in window._metric_cards)
    for plot in window._styled_plots:
        if plot.isVisible():
            assert plot.parentWidget().rect().contains(plot.geometry())
    if tab == 1:
        upper = window.direction_plots["aod"].geometry()
        lower = window.delay_power_plot.geometry()
        assert upper.bottom() < lower.top()
        assert window.direction_plots["aod"].height() >= 260
    window.close()


def test_metrics_kpi_precision_and_clear_are_display_only(qapp):
    window = MetricsWindow()
    stats = replace(_deep_frame_stats(), delay_spread_ns=0.0125248)
    window.enqueue(_deep_view_model(), stats)
    window._refresh()
    assert window.metric_values["spread"].text() == "0.0125"
    assert window.metric_values["spread"].toolTip() == "0.0125248"
    assert stats.delay_spread_ns == 0.0125248
    window.enqueue(
        _deep_view_model(),
        replace(stats, delay_spread_ns=np.nan, path_loss_range=None, angular_spread_deg=0.0),
    )
    window._refresh()
    assert window.metric_values["spread"].text() == "N/A"
    assert window.metric_values["loss"].text() == "N/A"
    assert window.metric_values["angle"].text() == "0"
    window._clear_plots()
    assert all(label.text() == "N/A" for label in window.metric_values.values())
    assert all("no usable data" in label.toolTip() for label in window.metric_values.values())
    window.close()


def test_pdp_ignores_unavailable_samples_and_clears_previous_points(qapp):
    window = MetricsWindow()
    for powers, expected_count in (([np.nan, -80.0, np.inf], 1), ([np.nan] * 3, 0)):
        profile = (np.array([5.0, 20.0, 30.0]), np.array(powers))
        stats = replace(_deep_frame_stats(), binned_power_delay_profile=profile)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            window.enqueue(_deep_view_model(), stats)
            window._refresh()
            x, y = window.binned_pdp_dots.getData()
            assert len(x) == expected_count
            assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            assert window.binned_pdp_plot._empty_message.isVisible() == (expected_count == 0)
        np.testing.assert_array_equal(stats.binned_power_delay_profile[1], powers)
    window.close()


def test_metrics_order_legends_are_outside_axes_and_match_cached_symbols(qapp):
    window = MetricsWindow()
    window.enqueue(_deep_view_model(), _deep_frame_stats())
    window.tabs.setCurrentIndex(1)
    window._refresh()
    expected_symbols = window._order_symbols(np.array([0, 1, 2]))
    for scatter in [window.delay_power_scatter, *window.direction_scatters.values()]:
        assert [point.symbol() for point in scatter.points()] == expected_symbols
    for plot in [window.delay_power_plot, *window.direction_plots.values()]:
        item = plot.getPlotItem()
        assert item.legend.parentItem() is item
        assert item.layout.itemAt(1, 1) is item.legend
        for index, (sample, label) in enumerate(item.legend.items[:6]):
            assert pg.mkBrush(sample.item.opts["brush"]).color().name() == dashboard_series_color(
                window._ORDER_COLORS[index]
            )
            assert sample.item.opts["symbol"] == window._ORDER_SYMBOLS[index]
            assert label.opts["color"] == current_theme().text_secondary
    window.close()


@pytest.mark.parametrize(
    "orders",
    [
        np.array([], dtype=float),
        np.arange(-2, 9, dtype=np.int64),
        np.array([-np.inf, -1e300, -1.1, -0.5, 0.0, 1.9, 5.9, 6.0, 1e300, np.inf, np.nan]),
    ],
)
def test_metrics_order_symbols_preserve_scalar_mapping(qapp, orders):
    window = MetricsWindow()
    original = orders.copy()
    expected = [
        (
            window._ORDER_SYMBOLS[int(order)]
            if np.isfinite(order) and 0 <= int(order) < len(window._ORDER_SYMBOLS)
            else "star"
        )
        for order in orders
    ]
    with np.errstate(all="raise"):
        assert window._order_symbols(orders) == expected
    np.testing.assert_array_equal(orders, original)
    window.close()


def test_metrics_style_does_not_change_global_plot_colours(qapp):
    before = {key: pg.getConfigOption(key) for key in ("background", "foreground")}
    window = MetricsWindow()
    window._apply_theme()
    assert {key: pg.getConfigOption(key) for key in before} == before
    assert window.delay_plot.backgroundBrush().color().name() == current_theme().bg_secondary
    window.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_overview_grid_preserves_data_zoom_and_refresh_state(qapp, enabled):
    window = MetricsWindow()
    window.enqueue(_deep_view_model(), _deep_frame_stats())
    window._refresh()
    window._cancel_refresh_timers()
    window.pause_cb.setChecked(True)
    window.delay_plot.setRange(xRange=(5, 30), yRange=(0, 5), padding=0)
    zoom = np.asarray(window.delay_plot.viewRange()).copy()
    counts = {key: bars.opts["height"].copy() for key, bars in window.angle_bars.items()}
    revision = window._stats_revision
    window.overview_grid_cb.setChecked(enabled)
    window._apply_theme()
    assert window._stats_revision == revision
    assert not window._pending_refresh and not window._timer.isActive()
    np.testing.assert_allclose(window.delay_plot.viewRange(), zoom)
    for plot in window._overview_plots:
        assert bool(plot.getAxis("left").grid) is enabled
        assert bool(plot.getAxis("bottom").grid) is enabled
    for key, bars in window.angle_bars.items():
        np.testing.assert_array_equal(bars.opts["height"], counts[key])
    window.close()


def test_metrics_secondary_controls_preserve_values_when_reopened(qapp):
    window = MetricsWindow()
    window.show()
    window.settings_toggle.click()
    assert window.settings_drawer.isVisible()
    window.plot_point_limit_spin.setValue(1500)
    window.update_rate_spin.setValue(12)
    window.settings_toggle.click()
    window.settings_toggle.click()
    assert window.plot_point_limit_spin.value() == 1500
    assert window.update_rate_spin.value() == 12
    window.tabs.setCurrentWidget(window._channel_tab)
    assert window._channel_pair_control.isVisible()
    assert not window.overview_grid_cb.isVisible()
    window.close()


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_metrics_theme_switch_preserves_points_and_zoom(qapp, mode):
    from visualizer.src.app.theme import get_theme_manager

    manager = get_theme_manager()
    previous = manager.mode
    window = MetricsWindow()
    window.show()
    window.enqueue(_deep_view_model(), _deep_frame_stats())
    window.tabs.setCurrentIndex(1)
    window._refresh()
    window._cancel_refresh_timers()
    window.delay_power_plot.setRange(xRange=(6, 25), yRange=(-85, -65), padding=0)
    points = tuple(a.copy() for a in window.delay_power_scatter.getData())
    zoom = np.array(window.delay_power_plot.viewRange())
    try:
        manager.set_mode(mode, persist=False)
        window._apply_theme()
        qapp.processEvents()
        theme = current_theme()
        assert window.delay_power_plot.backgroundBrush().color().name() == theme.bg_secondary
        assert theme.bg_secondary in window.pair_count_panel.styleSheet()
        assert theme.text_primary in window._surface.styleSheet()
        np.testing.assert_allclose(window.delay_power_plot.viewRange(), zoom)
        for before, after in zip(points, window.delay_power_scatter.getData()):
            np.testing.assert_array_equal(after, before)
        for plot in [window.delay_power_plot, *window.direction_plots.values()]:
            for _, label in plot.getPlotItem().legend.items:
                assert theme.text_secondary in label.item.toHtml()
        assert [point.brush().color().name() for point in window.delay_power_scatter.points()] == [
            dashboard_series_color(c) for c in window._ORDER_COLORS[:3]
        ]
    finally:
        window.close()
        manager.set_mode(previous, persist=False)


def test_adaptive_axes_follow_changing_data_with_one_range_notification(qapp):
    from PySide6.QtTest import QSignalSpy

    window = MetricsWindow()
    window.show()
    window.auto_range_cb.setChecked(True)
    window._update_histogram(window.delay_plot, window.delay_bars, np.array([0.0, 1.0]), "delay")
    qapp.processEvents()
    notifications = QSignalSpy(window.delay_plot.getViewBox().sigRangeChanged)
    for values in (np.array([5.0, 10.0, 18.0]), np.array([120.0, 280.0, 500.0])):
        before = notifications.count()
        window._update_histogram(window.delay_plot, window.delay_bars, values, "delay")
        assert notifications.count() == before + 1
        bounds = window.delay_plot.viewRange()[0]
        assert bounds[0] <= values.min() and bounds[1] >= values.max()
        assert sum(window.delay_bars.opts["height"]) == len(values)
    assert window.delay_plot.getAxis("left").style["autoExpandTextSpace"]
    assert not window.delay_plot.getAxis("left").style["autoReduceTextSpace"]
    window.close()


def test_all_angle_titles_survive_theme_application(qapp):
    window = MetricsWindow()
    window.enqueue(_deep_view_model(), _deep_frame_stats())
    window._refresh()
    window._apply_theme()
    for field, (title, label) in window._ANGLE_FIELDS.items():
        plot = window.angle_plots[field]
        assert plot.getPlotItem().titleLabel.text == f"{title} Distribution"
        assert plot.getAxis("bottom").labelText == label
    window.close()
