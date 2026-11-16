"""Summary-only visual presentation must preserve the scientific contracts."""

import warnings
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from visualizer.src.sensing.run_summary_data import RunSummaryData
from visualizer.src.sensing.run_summary_panel import RunSummaryPanel


def _data():
    return RunSummaryData(
        ("clock", "run", "pair"),
        summary={
            "total_tp": 3,
            "total_fp": 1,
            "total_fn": 2,
            "recall": 0.6,
            "precision": 0.75,
            "cumulative_range_rmse": 0.125,
            "cumulative_velocity_rmse": 0.25,
            "id_switches": 0,
            "per_target": {"target": {"recall": 0.6, "matched_count": 3, "eligible_count": 5}},
        },
        acquisition_count=3,
        evaluated_count=3,
        time_s=np.array([0.0, 1.0, 2.0]),
        event_tp=np.array([1.0, 1.0, 1.0]),
        event_fp=np.array([0.0, 1.0, 0.0]),
        event_fn=np.array([1.0, 0.0, 1.0]),
        event_gt_count=np.array([2.0, 1.0, 2.0]),
        event_output_count=np.array([1.0, 2.0, 1.0]),
        cumulative_recall=np.array([0.5, 2 / 3, 0.6]),
        cumulative_precision=np.array([1.0, 2 / 3, 0.75]),
        range_errors_m=np.array([0.2, -0.1, 0.05]),
        radial_errors_m_s=np.array([0.4, -0.2, 0.1]),
        confirmed_count=np.array([1.0, 1.0, 1.0]),
        tentative_count=np.array([0.0, 1.0, 0.0]),
        tracking_population_available=True,
        tracking_time_s=np.array([0.0, 1.0, 2.0]),
        tracking_target_ids=("target", "target", "target"),
        tracking_track_ids=(7, 7, 7),
    )


def test_summary_with_no_outputs_keeps_precision_unavailable_without_plot_warning(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    original = _data()
    data = replace(
        original,
        summary={
            **original.summary,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 5,
            "recall": 0.0,
            "precision": None,
            "cumulative_range_rmse": None,
            "cumulative_velocity_rmse": None,
            "per_target": {"target": {"recall": 0.0, "matched_count": 0, "eligible_count": 5}},
        },
        event_tp=np.zeros(3),
        event_fp=np.zeros(3),
        event_fn=original.event_gt_count.copy(),
        event_output_count=np.zeros(3),
        cumulative_recall=np.zeros(3),
        cumulative_precision=np.full(3, np.nan),
        range_errors_m=np.array([]),
        radial_errors_m_s=np.array([]),
        tracking_population_available=False,
        tracking_time_s=np.array([]),
        tracking_target_ids=(),
        tracking_track_ids=(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        panel.set_summary(data)
        panel.resize(1000, 800)
        panel.show()
        panel.plots["rates"].getViewBox().autoRange()
        QApplication.processEvents()
        panel.grab()
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert data.summary["precision"] is None
    assert np.isnan(data.cumulative_precision).all()
    precision = next(
        item for item in panel.plots["rates"].listDataItems() if item.name() == "Precision"
    )
    x, y = precision.getData()
    assert x is None or len(x) == 0
    assert y is None or len(y) == 0


def test_summary_line_preserves_missing_intervals_between_valid_samples(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, np.nan, 0.5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        RunSummaryPanel._line(plot, x, y, "Precision", "#7552a3")
        plot.show()
        plot.getViewBox().autoRange()
        QApplication.processEvents()
        plot.grab()
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    curve = plot.listDataItems()[0]
    actual_x, actual_y = curve.getData()
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_y, y)
    assert curve.opts["connect"] == "finite"


def test_constant_track_id_axis_keeps_literal_category_without_si_multiplier(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    data = _data()
    panel.set_summary(data)
    panel.resize(1600, 1000)
    panel.show()
    panel.tabs.setCurrentIndex(2)
    QApplication.processEvents()

    plot = panel.plots["identity"]
    axis = plot.getAxis("left")
    assert axis.label.toPlainText().strip() == "Saved track ID"
    assert axis._tickLevels == [[(0, "7")]]
    x, y = plot.listDataItems()[0].getData()
    np.testing.assert_array_equal(x, data.tracking_time_s)
    np.testing.assert_array_equal(y, np.zeros(3))


@pytest.mark.parametrize("size", [(1280, 900), (1600, 1000)])
def test_summary_fits_laptop_and_desktop_without_clipped_figures(qtbot, size):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.resize(*size)
    panel.set_summary(_data())
    panel.show()
    qtbot.wait(20)
    assert (panel.width(), panel.height()) == size
    for index in range(3):
        panel.tabs.setCurrentIndex(index)
        QApplication.processEvents()
        scroll = panel.tabs.widget(index)
        assert scroll.verticalScrollBar().maximum() == 0
        assert scroll.horizontalScrollBar().maximum() == 0
        for plot in panel.plots.values():
            if plot.isVisible():
                assert plot.width() >= 260
                assert plot.height() >= 180
                point = plot.mapTo(panel, plot.rect().bottomRight())
                assert point.x() < panel.width()
                assert point.y() < panel.height()


def test_summary_does_not_change_global_theme_or_sibling(qtbot):
    app = QApplication.instance()
    stylesheet = app.styleSheet()
    background, foreground = pg.getConfigOption("background"), pg.getConfigOption("foreground")
    sibling = QWidget()
    sibling.setStyleSheet("QWidget { color: rgb(80, 90, 100); }")
    qtbot.addWidget(sibling)
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.set_summary(_data())
    panel.show()
    QApplication.processEvents()
    assert app.styleSheet() == stylesheet
    assert pg.getConfigOption("background") == background
    assert pg.getConfigOption("foreground") == foreground
    assert sibling.styleSheet() == "QWidget { color: rgb(80, 90, 100); }"


def test_summary_theme_switch_preserves_samples_plot_objects_and_zoom(qtbot, monkeypatch):
    from visualizer.src.app.theme import DARK_THEME, LIGHT_THEME
    from visualizer.src.sensing import run_summary_panel as module

    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    data = _data()
    panel.set_summary(data)
    plot = panel.plots["counts"]
    series = plot.listDataItems()[0]
    arrays = series.getData()
    plot.setRange(xRange=(0.5, 1.5), yRange=(-0.5, 2.5), padding=0)
    limits = plot.viewRange()
    colors = []
    for theme in (LIGHT_THEME, DARK_THEME, LIGHT_THEME):
        monkeypatch.setattr(module, "current_theme", lambda: theme)
        panel.apply_theme()
        assert plot.backgroundBrush().color().name() == theme.bg_secondary
        assert plot.getAxis("bottom").textPen().color().name() == theme.text_secondary
        for _, label in plot.getPlotItem().legend.items:
            assert theme.text_secondary in label.item.toHtml()
        assert plot.listDataItems()[0] is series
        assert panel._rendered is data
        assert all(a is b for a, b in zip(arrays, series.getData()))
        np.testing.assert_allclose(plot.viewRange(), limits)
        colors.append(series.opts["pen"].color().name())
    assert colors[0] == colors[2] != colors[1]


@pytest.mark.parametrize("width", [700, 1100])
def test_summary_reserves_two_visible_plot_rows(qtbot, width):
    from visualizer.src.sensing.sensing_window import SensingWindow

    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(width, 940)
    panel = window.run_summary_panel
    panel.set_summary(_data())
    window.show()
    qtbot.wait(30)
    assert panel._columns == 2
    scroll = panel.tabs.widget(0)
    assert scroll.verticalScrollBar().maximum() == 0
    assert scroll.horizontalScrollBar().maximum() == 0
    for key in ("counts", "population", "rates", "coverage"):
        card = panel.plot_cards[key]
        point = card.mapTo(scroll.viewport(), card.rect().bottomRight())
        assert point.y() <= scroll.viewport().height()
    assert panel.plots["counts"].listDataItems()[0].opts["symbolSize"] < 6


def test_summary_preserves_series_and_input_data_on_repeated_render(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    data = _data()
    saved = deepcopy(data)
    panel.set_summary(data)
    counts = {item.name(): item for item in panel.plots["counts"].listDataItems()}
    for name, expected in [
        ("TP", data.event_tp),
        ("FP", data.event_fp),
        ("FN", data.event_fn),
    ]:
        x, y = counts[name].getData()
        np.testing.assert_array_equal(x, data.time_s)
        np.testing.assert_array_equal(y, expected)
    rates = {item.name(): item for item in panel.plots["rates"].listDataItems()}
    np.testing.assert_allclose(rates["Recall"].getData()[1], data.cumulative_recall * 100)
    np.testing.assert_allclose(rates["Precision"].getData()[1], data.cumulative_precision * 100)
    errors = panel.plots["range_error"].listDataItems()[0]
    np.testing.assert_allclose(errors.getData()[0], [0.05, 0.1, 0.2])
    np.testing.assert_allclose(errors.getData()[1], [100 / 3, 200 / 3, 100])
    panel.set_summary(data)
    assert len(panel.plots["counts"].listDataItems()) == 3
    assert panel.plots["counts"].listDataItems()[0] is counts["TP"]
    for name, value in vars(data).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(saved, name))
        else:
            assert value == getattr(saved, name)


def test_summary_settings_apply_and_clear_remain_explicit(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.set_summary(_data())
    old = panel.settings()
    events = []
    panel.settings_changed.connect(lambda: events.append(panel.settings()))
    panel.interval.setChecked(True)
    panel.start.setValue(0.25)
    panel.end.setValue(1.75)
    panel.range_gate.setValue(4)
    assert panel.settings() == old
    panel.apply.click()
    assert len(events) == 1
    assert panel.settings() == (4.0, old[1], (0.25, 1.75), False)
    panel.clear("Choose a saved run")
    assert panel.data is None
    assert not panel.tabs.isTabEnabled(2)
    assert all(not plot.listDataItems() for plot in panel.plots.values())
    panel.set_summary(_data())
    assert panel.tabs.isTabEnabled(2)


@pytest.mark.parametrize(
    "data",
    [
        RunSummaryData(("c", "r", "p")),
        RunSummaryData(("c", "r", "p"), acquisition_count=3, uncertainty=("No usable truth",)),
    ],
)
@pytest.mark.parametrize("size", [(1280, 900), (1600, 1000)])
def test_empty_and_missing_truth_never_present_zero_quality(qtbot, data, size):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.resize(*size)
    panel.set_summary(data)
    panel.show()
    QApplication.processEvents()
    visible_text = " ".join(
        label.text() for label in panel.findChildren(QLabel) if label.isVisible()
    ).lower()
    assert "unavailable" in visible_text
    assert not panel.tabs.isTabEnabled(2)
    assert "0.0%" not in visible_text
    assert (panel.width(), panel.height()) == size
    for index in (0, 1):
        panel.tabs.setCurrentIndex(index)
        QApplication.processEvents()
        scroll = panel.tabs.widget(index)
        assert scroll.verticalScrollBar().maximum() == 0
        assert scroll.horizontalScrollBar().maximum() == 0


def test_visible_kpi_values_match_exact_summary_and_clear(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.resize(1280, 900)
    panel.set_summary(_data())
    panel.show()
    QApplication.processEvents()
    expected = {
        "tp": "3",
        "fp": "1",
        "fn": "2",
        "recall": "60.0%",
        "precision": "75.0%",
        "range": "0.125",
        "radial": "0.25",
    }
    assert set(panel.metric_values) == set(expected)
    for key, value in expected.items():
        label = panel.metric_values[key]
        assert label.isVisible()
        assert label.text() == value
        assert label.width() >= label.fontMetrics().horizontalAdvance(value)
    assert "m/s" in panel.metric_captions["radial"].text()
    assert "m" in panel.metric_captions["range"].text()
    panel.set_loading(("Reading saved data", 1, 4))
    assert all(label.text() == "—" for label in panel.metric_values.values())
    assert "1/4" in panel.coverage_note.text()
    panel.set_summary(RunSummaryData(("c", "r", "p"), error="unreadable recording"))
    assert "unreadable recording" in panel.coverage_note.text()
    assert all(label.text() == "—" for label in panel.metric_values.values())


def test_summary_collapsible_controls_do_not_change_scores(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.set_summary(_data())
    panel.show()
    QApplication.processEvents()
    assert not panel.controls.isVisible()
    values = {key: label.text() for key, label in panel.metric_values.items()}
    old = panel.settings()
    for _ in range(3):
        panel.settings_toggle.click()
        assert panel.controls.isVisible()
        panel.settings_toggle.click()
        assert not panel.controls.isVisible()
    assert not hasattr(panel, "details_toggle")
    assert "Clock: clock" in panel.scope.toolTip()
    assert panel.settings() == old
    assert values == {key: label.text() for key, label in panel.metric_values.items()}


def test_error_distributions_use_zero_based_axis_without_changing_samples(qtbot):
    from dataclasses import replace

    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.resize(1280, 900)
    data = replace(_data(), radial_errors_m_s=np.array([0.00376939062496, 0.00376939062498]))
    panel.set_summary(data)
    panel.show()
    panel.tabs.setCurrentIndex(1)
    QApplication.processEvents()
    for key in ("range_error", "radial_error"):
        lo, hi = panel.plots[key].getViewBox().viewRange()[0]
        assert lo <= 0 < hi
    np.testing.assert_array_equal(
        panel.plots["radial_error"].listDataItems()[0].getData()[0], data.radial_errors_m_s
    )
    panel.set_summary(replace(data, radial_errors_m_s=np.array([0.0, 0.0])))
    QApplication.processEvents()
    lo, hi = panel.plots["radial_error"].getViewBox().viewRange()[0]
    assert lo <= 0 < hi
    np.testing.assert_array_equal(
        panel.plots["radial_error"].listDataItems()[0].getData()[0], [0.0, 0.0]
    )


def test_summary_reflows_cards_settings_and_plots_in_narrow_host(qtbot):
    from PySide6.QtWidgets import QVBoxLayout

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    panel = RunSummaryPanel()
    layout.addWidget(panel)
    panel.set_summary(_data())
    host.resize(650, 850)
    host.show()
    qtbot.wait(20)
    assert host.width() == 650
    assert panel._columns == 1
    assert panel.metric_cards[-1].y() > panel.metric_cards[0].y()
    panel.settings_toggle.click()
    qtbot.wait(20)
    assert host.width() == 650
    for widget in [*panel.metric_cards, *panel.control_items]:
        assert widget.isVisible()
        assert widget.mapTo(panel, widget.rect().bottomRight()).x() < panel.width()
    for index in range(3):
        panel.tabs.setCurrentIndex(index)
        QApplication.processEvents()
        assert panel.tabs.widget(index).horizontalScrollBar().maximum() == 0
    host.resize(1300, 950)
    qtbot.wait(20)
    assert panel._columns == 2
    assert panel.metric_cards[-1].y() == panel.metric_cards[0].y()
    host.resize(650, 850)
    qtbot.wait(20)
    assert host.width() == 650
    assert panel._columns == 1
