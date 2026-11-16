"""Scientific and lifecycle contracts for the interactive Path Stats workspace."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from tests.visualizer.unit.test_statistics_cache_service import _sample_stats
from visualizer.src.app.theme import current_theme
from visualizer.src.panels.statistics_panel import StatisticsPanel
from visualizer.src.panels.statistics_plot import inverse, transform


def _dashboard(qtbot, *, show=True, width=1280, height=800):
    panel = StatisticsPanel(SimpleNamespace())
    root = panel.create_graphs_panel()
    qtbot.addWidget(root)
    root.resize(width, height)
    panel.stats = _sample_stats()
    panel._graphs_visible = True
    panel._render_statistics_graphs()
    if show:
        root.show()
        QApplication.processEvents()
    return panel, root


def test_only_visible_family_allocates_plots_and_clean_refresh_is_noop(qtbot):
    panel, root = _dashboard(qtbot, show=False)
    assert all(card.plot is None for card in panel.widgets.values())
    root.show()
    QApplication.processEvents()
    allocated = [key for key, card in panel.widgets.items() if card.plot is not None]
    assert allocated == [
        "reflection_order_chart",
        "interaction_type_chart",
        "path_loss_histogram_chart",
        "delay_histogram_chart",
    ]
    before = {key: card.render_count for key, card in panel.widgets.items()}
    panel._render_statistics_graphs()
    QApplication.processEvents()
    assert {key: card.render_count for key, card in panel.widgets.items()} == before
    panel._graph_tabs.setCurrentIndex(3)
    QApplication.processEvents()
    assert all(
        panel.widgets[key].plot is not None
        for key in (
            "mpc_evolution_chart",
            "delay_spread_trend_chart",
            "mpc_order_evolution_hist_chart",
            "mpc_type_evolution_chart",
        )
    )
    assert panel.widgets["delay_cdf_chart"].plot is None
    panel._graph_tabs.setCurrentIndex(0)
    QApplication.processEvents()
    assert panel.widgets["reflection_order_chart"].render_count == 1


def test_hidden_updates_wait_and_reuse_existing_plot(qtbot):
    panel, root = _dashboard(qtbot)
    card = panel.widgets["reflection_order_chart"]
    plot, count = card.plot, card.render_count
    root.hide()
    panel._graphs_visible = False
    panel.stats = dict(panel.stats, reflection_order_dist={0: 100})
    panel._graphs_dirty = True
    panel._render_statistics_graphs()
    assert card.render_count == count
    panel._graphs_visible = True
    panel._render_statistics_graphs()
    assert card.render_count == count
    root.show()
    QApplication.processEvents()
    assert card.plot is plot
    assert card.render_count == count + 1
    assert card.chart_metadata["data"] == {0: 100}


@pytest.mark.parametrize(
    "tab,key", [(3, "delay_spread_trend_chart"), (4, "strongest_path_loss_chart")]
)
def test_missing_evolution_data_shows_message_and_recovers_without_warnings(qtbot, tab, key):
    panel, root = _dashboard(qtbot, show=False)
    panel.stats["delay_spread_evolution"] = np.full(3, np.nan)
    panel.stats["strongest_single_path_loss_evolution"] = np.full(3, np.nan)
    panel._graphs_dirty = True
    panel._render_statistics_graphs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        root.show()
        panel._graph_tabs.setCurrentIndex(tab)
        QApplication.processEvents()
        card = panel.widgets[key]
        assert card.message.isVisible()
        assert "No usable samples" in card.message.text()
        assert card.plot is None
        for values in ([0.0, 1.0, 2.0], [np.nan] * 3, [0.0, 0.0, 0.0]):
            panel._create_chart(card, "line_evolution", {"x": [0, 1, 2], "y": values})
            QApplication.processEvents()
            root.grab()
            assert card.message.isVisible() == bool(np.all(np.isnan(values)))
            assert card.plot.isVisible() != card.message.isVisible()


def test_sparse_evolution_keeps_gaps_and_only_finite_markers(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    card = panel._create_chart_widget("delay_spread_trend_chart")
    qtbot.addWidget(card)
    card.show()
    x = np.arange(1000, dtype=float)
    y = np.full(x.size, np.nan)
    y[[1, 3, 5]] = [0.0, 2.0, 4.0]
    y[10] = np.inf
    x[5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        panel._create_chart(card, "line_evolution", {"x": x, "y": y})
        QApplication.processEvents()
        curve, markers = card.plot.listDataItems()
        np.testing.assert_array_equal(curve.xData, x)
        np.testing.assert_array_equal(curve.yData, y)
        np.testing.assert_array_equal(markers.xData, [1.0, 3.0])
        np.testing.assert_array_equal(markers.yData, [0.0, 2.0])
        assert markers.scatter.dataBounds(1) == (0.0, 2.0)
        card.grab()
    assert panel._find_nearest_chart_point(1.0, 0.0, "line_evolution", {"x": x, "y": y}) == (
        1.0,
        0.0,
    )


def test_statistics_grid_preserves_lazy_plots_data_zoom_and_rebin_counts(qtbot):
    panel, _root = _dashboard(qtbot)
    card = panel.widgets["path_loss_histogram_chart"]
    card.plot.setRange(xRange=(70, 90), yRange=(0, 5), padding=0)
    zoom = np.asarray(card.plot.viewRange()).copy()
    metadata = card.chart_metadata
    counts = {key: item.render_count for key, item in panel.widgets.items()}
    for enabled in (True, False, True):
        panel._graphs_grid_cb.setChecked(enabled)
        panel._on_theme_changed(None)
        assert bool(card.plot.getAxis("left").grid) is enabled
        assert bool(card.plot.getAxis("bottom").grid) is enabled
        assert card.chart_metadata is metadata
        assert {key: item.render_count for key, item in panel.widgets.items()} == counts
        np.testing.assert_allclose(card.plot.viewRange(), zoom)
        assert panel.widgets["aod_az_polar_chart"].plot is None
    panel._graph_tabs.setCurrentIndex(2)
    QApplication.processEvents()
    polar = panel.widgets["aod_az_polar_chart"]
    assert polar._grid_items and all(item.isVisible() for item in polar._grid_items)
    rendered = polar.render_count
    panel._graphs_grid_cb.setChecked(False)
    assert not any(item.isVisible() for item in polar._grid_items)
    assert polar.render_count == rendered


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kind", ["histogram", "polar_rose"])
def test_statistics_png_export_uses_selected_grid_without_allocating_plots(qtbot, enabled, kind):
    panel, _root = _dashboard(qtbot, show=False)
    panel._graphs_grid_cb.setChecked(enabled)
    figure = panel._create_publication_figure(kind, {"values": [0.0, 10.0, 20.0]})
    assert all(card.plot is None for card in panel.widgets.values())
    axis = figure.axes[0]
    assert all(line.get_visible() is enabled for line in axis.get_xgridlines())
    assert all(line.get_visible() is enabled for line in axis.get_ygridlines())
    figure.clear()


def test_new_run_clears_missing_samples_and_disables_unavailable_families(qtbot):
    panel, _root = _dashboard(qtbot)
    card = panel.widgets["path_loss_histogram_chart"]
    assert card.plot is not None
    panel.stats = {"total_frames": 1, "total_mpcs": 0}
    panel._graphs_dirty = True
    panel._render_statistics_graphs()
    assert card.chart_metadata is None
    assert card.plot.isHidden()
    assert card.message.isVisible()
    assert not panel._graph_tabs.isTabEnabled(2)
    assert not panel._graph_tabs.isTabEnabled(3)
    assert not panel._graph_tabs.isTabEnabled(4)
    assert "path_loss_histogram_chart" not in panel._chart_specs


@pytest.mark.parametrize("width,height", [(700, 800), (1280, 800), (1600, 1000)])
def test_four_cards_fit_actual_viewport_and_legends_are_external(qtbot, width, height):
    panel, root = _dashboard(qtbot, width=width, height=height)
    panel._graph_tabs.setCurrentIndex(3)
    QApplication.processEvents()
    assert root.size().width() == width
    assert root.size().height() == height
    scroll = panel._graph_tabs.currentWidget()
    assert panel._evolution_group._columns == 2
    assert scroll.horizontalScrollBar().maximum() == 0
    assert scroll.verticalScrollBar().maximum() == 0
    cards = panel._evolution_group.cards
    assert all(card.height() >= 250 for card in cards)
    for key in ("mpc_order_evolution_hist_chart", "mpc_type_evolution_chart"):
        plot = panel.widgets[key].plot
        legend = plot.getPlotItem().legend
        assert (
            legend.sceneBoundingRect().bottom() <= plot.getViewBox().sceneBoundingRect().top() + 1
        )
        assert all(
            label.opts["color"] == current_theme().text_secondary for _sample, label in legend.items
        )


def test_narrow_dashboard_reflows_without_replot(qtbot):
    panel, root = _dashboard(qtbot)
    counts = [card.render_count for card in panel._distribution_group.cards]
    root.resize(480, 800)
    QApplication.processEvents()
    assert panel._distribution_group._columns == 1
    assert root.width() == 480
    assert panel._graph_tabs.currentWidget().horizontalScrollBar().maximum() == 0
    assert [card.render_count for card in panel._distribution_group.cards] == counts
    root.resize(700, 800)
    QApplication.processEvents()
    assert panel._distribution_group._columns == 2
    assert panel._graph_tabs.currentWidget().horizontalScrollBar().maximum() == 0
    assert [card.render_count for card in panel._distribution_group.cards] == counts


def test_summary_keeps_every_metric_readable_at_embedded_width(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    widget.resize(480, 900)
    panel.stats = _sample_stats()
    panel._update_statistics()
    widget.show()
    QApplication.processEvents()
    assert widget.width() == 480
    for key in (
        "total_mpcs",
        "total_frames",
        "unique_pairs",
        "mpc_frame_ratio",
        "overall_delay_spread",
        "path_loss_range",
        "mpc_count_cv",
        "pair_gain_percentiles",
        "pair_delay_spread_percentiles",
        "pair_visibility_summary",
    ):
        label = panel.widgets[key]
        assert label.text() != "--"
        assert label.width() > 80


def test_export_all_figures_without_allocating_hidden_plots(qtbot, monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    panel, _root = _dashboard(qtbot, show=False)
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QFileDialog.getExistingDirectory",
        lambda *args: str(tmp_path),
    )
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QMessageBox.information", lambda *args: None
    )
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QMessageBox.warning",
        lambda *args: pytest.fail(str(args)),
    )
    panel._on_export_charts()
    assert len(list(tmp_path.glob("*.png"))) == 18
    assert all(path.stat().st_size > 1000 for path in tmp_path.glob("*.png"))
    assert all(card.plot is None for card in panel.widgets.values())


def test_large_cdf_preserves_full_sample_count_and_bounded_points(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    card = panel._create_chart_widget("delay_cdf_chart")
    qtbot.addWidget(card)
    card.show()
    values = np.arange(50001, dtype=float)
    panel._create_chart(card, "cdf", {"values": values, "xlabel": "Delay (ns)"})
    data = card.chart_metadata["data"]
    assert data["_cdf_total"] == values.size
    assert len(data["_cdf_x"]) == 5000
    assert data["_cdf_x"][-1] == 50000
    assert data["_cdf_y"][-1] == 1
    assert len(card.plot.listDataItems()[1].xData) <= 43


def test_log_coverage_bins_and_percentiles_preserve_scientific_values(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    card = panel._create_chart_widget("coverage_distribution_chart")
    qtbot.addWidget(card)
    card.show()
    values = np.geomspace(0.01, 1000, 71)
    panel._create_chart(
        card,
        "coverage_histogram",
        {
            "values": values,
            "color_scale": "logarithmic",
            "percentiles": (0.1, 1.0, 100.0),
            "bins": 10,
        },
    )
    data = card.chart_metadata["data"]
    expected_edges = np.geomspace(values.min(), values.max(), 11)
    counts, _ = np.histogram(values, bins=expected_edges)
    np.testing.assert_allclose(data["_hist_edges"], expected_edges)
    np.testing.assert_array_equal(data["_hist_counts"], counts)
    np.testing.assert_allclose(data["_hist_percentages"], 100 * counts / len(values))
    for value in (0.01, 0.1, 1.0, 100.0, 1000.0):
        assert inverse(transform([value], "log")[0], "log") == pytest.approx(value)


def test_symlog_hover_uses_original_count_and_no_plot_redraw(qtbot, monkeypatch):
    panel = StatisticsPanel(SimpleNamespace())
    card = panel._create_chart_widget("mpc_order_evolution_hist_chart")
    qtbot.addWidget(card)
    card.resize(600, 300)
    card.show()
    panel._create_chart(
        card,
        "mpc_order_evolution",
        {"frame_indices": [0, 1], "order_data": {0: [0, 1], 1: [10000, 25000]}},
    )
    QApplication.processEvents()
    point = card.plot.getViewBox().mapViewToScene(
        QPointF(1.0, float(transform([25000], "symlog")[0]))
    )
    count = card.render_count
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_plot.QToolTip.showText", lambda *args: None
    )
    card._hover((point,))
    assert "Order 1: 25,000" in card.plot.toolTip()
    assert card.render_count == count


def test_repeated_reference_charts_keep_one_legend_entry_per_percentile(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    card = panel._create_chart_widget("coverage_distribution_chart")
    qtbot.addWidget(card)
    card.show()
    for _ in range(3):
        panel._create_chart(
            card, "coverage_histogram", {"values": [1.0, 2.0, 3.0], "percentiles": (1.2, 2.0, 2.8)}
        )
        assert len(card.plot.getPlotItem().legend.items) == 3
    for _ in range(3):
        panel._create_chart(card, "cdf", {"values": [1.0, 2.0, 3.0]})
        assert len(card.plot.getPlotItem().legend.items) == 1


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_statistics_theme_preserves_data_zoom_and_lazy_plots(qtbot, mode):
    from visualizer.src.app.theme import get_theme_manager

    manager = get_theme_manager()
    previous = manager.mode
    panel, root = _dashboard(qtbot)
    panel._graph_tabs.setCurrentIndex(3)
    QApplication.processEvents()
    card = panel.widgets["mpc_evolution_chart"]
    metadata = card.chart_metadata
    series = [(name, x.copy(), y.copy()) for name, x, y in card._series]
    card.plot.setRange(xRange=(0.25, 0.75), yRange=(1, 2), padding=0)
    zoom = np.array(card.plot.viewRange())
    counts = {key: c.render_count for key, c in panel.widgets.items()}
    try:
        manager.set_mode(mode, persist=False)
        panel._on_theme_changed(None)
        QApplication.processEvents()
        theme = current_theme()
        assert card.plot.backgroundBrush().color().name() == theme.bg_secondary
        assert theme.bg_primary in root.styleSheet()
        assert card.chart_metadata is metadata
        assert {key: c.render_count for key, c in panel.widgets.items()} == counts
        assert panel.widgets["delay_cdf_chart"].plot is None
        np.testing.assert_allclose(card.plot.viewRange(), zoom)
        for (name, x, y), (actual_name, actual_x, actual_y) in zip(series, card._series):
            assert actual_name == name
            np.testing.assert_array_equal(actual_x, x)
            np.testing.assert_array_equal(actual_y, y)
        for _, label in (
            panel.widgets["mpc_order_evolution_hist_chart"].plot.getPlotItem().legend.items
        ):
            assert theme.text_secondary in label.item.toHtml()
        # A family opened after the theme switch must pick up the new palette too.
        panel._graph_tabs.setCurrentIndex(2)
        QApplication.processEvents()
        assert (
            panel.widgets["aod_az_polar_chart"].plot.backgroundBrush().color().name()
            == theme.bg_secondary
        )
    finally:
        panel.cleanup()
        root.close()
        manager.set_mode(previous, persist=False)
