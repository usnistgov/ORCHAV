"""Native charts preserve numerical data and reuse it during theme changes."""

import warnings
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg

from visualizer.src.app.theme import get_theme_manager
from visualizer.src.panels.data_source.performance_section import PerformanceSection
from visualizer.src.widgets.inline_plot import InlinePlot


def test_theme_switch_restyles_without_replacing_plot_data(qtbot):
    plot = InlinePlot("Example", "Time (s)", "Power (dBm)")
    qtbot.addWidget(plot)
    curve = plot.curve([0, 1, 2], [-80, np.nan, -70], name="Signal", symbol="o")
    guide = plot.guide(-75, vertical=False, name="Threshold")
    annotation = plot.text("Unavailable", 1, -75)
    plot.resize(600, 400)
    plot.show()
    x, y = curve.xData, curve.yData
    manager = get_theme_manager()
    previous = manager.mode
    try:
        for mode in ("dark", "light"):
            manager.set_mode(mode, persist=False)
            assert curve.xData is x and curve.yData is y
            assert plot.backgroundBrush().color().name() == manager.current_theme.bg_secondary
            assert annotation.color.name() == manager.current_theme.text_secondary
            assert guide.pen.color().name() == manager.current_theme.text_secondary
            for _, label in plot._external_legend.items:
                assert manager.current_theme.text_primary in label.item.toHtml()
            assert not plot.grab().isNull()
    finally:
        manager.set_mode(previous, persist=False)


def test_closed_plot_does_not_receive_theme_updates(qtbot):
    plot = InlinePlot()
    qtbot.addWidget(plot)
    plot.close()
    manager = get_theme_manager()
    with qtbot.captureExceptions() as errors:
        manager.theme_changed.emit(manager.current_theme)
    assert errors == []


def test_all_missing_series_and_colourbar_redraws_are_bounded(qtbot):
    plot = InlinePlot("Empty")
    qtbot.addWidget(plot)
    plot.show()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        curve = plot.curve([0, 1], [np.nan, np.nan], symbol="o")
        plot.set_empty_message("No data available")
        assert curve.xData is None or not len(curve.xData)
        plot.grab()
    for _ in range(3):
        plot.clear()
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        mesh = plot.heatmap([1, 3], [5, 9], values, limits=(0, 4))
        assert isinstance(mesh, pg.ImageItem)
        np.testing.assert_array_equal(mesh.image, values)
        assert len(plot._colorbars) == 1
        plot.grab()

    plot.clear()
    values = np.arange(9).reshape(3, 3)
    mesh = plot.heatmap([1, 3, 8], [5, 9, 20], values, limits=(0, 8))
    assert isinstance(mesh, pg.PColorMeshItem)
    np.testing.assert_array_equal(mesh.z, values)


def test_live_performance_curves_update_in_place(qtbot):
    section = PerformanceSection(SimpleNamespace(), {}, lambda: "")
    group = section._create_performance_graph_group()
    qtbot.addWidget(group)
    group.expand()
    group.show()
    section._ensure_performance_graphs()
    plot = section.widgets["perf_graph_frame_time"]
    qtbot.waitUntil(plot.isVisible)
    curve = section._frame_time_line
    section.performance_history["frame_times"] = [(10, 12), (11, 14)]
    section._update_performance_graphs()
    np.testing.assert_array_equal(curve.xData, [0, 1])
    np.testing.assert_array_equal(curve.yData, [12, 14])
    section.performance_history["frame_times"].append((12, 9))
    section._update_performance_graphs()
    assert section._frame_time_line is curve
    np.testing.assert_array_equal(curve.yData, [12, 14, 9])
    section.cleanup()
