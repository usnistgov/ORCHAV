"""External sensing legends reserve space as series appear and panels resize."""

import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QGraphicsItem

from visualizer.src.widgets.dashboard_plot import DashboardPlot


@pytest.fixture
def painted_plot(qtbot, monkeypatch):
    plot = DashboardPlot()
    qtbot.addWidget(plot)
    axes = {plot.getAxis(name) for name in ("left", "bottom")}
    painted = []
    original = pg.AxisItem.paint

    def paint(axis, *args):
        if axis in axes:
            painted.append(axis)
        return original(axis, *args)

    monkeypatch.setattr(pg.AxisItem, "paint", paint)
    plot.resize(600, 360)
    plot.showGrid(x=True, y=True)
    curve = plot.plot([0, 1, 2, 3], [1, 3, 2, 4], pen="orange", symbol="o")
    plot.setRange(xRange=(-1, 5), yRange=(0, 5), padding=0)
    plot.show()
    qtbot.waitUntil(lambda: all(axis.picture is not None for axis in axes))
    plot.grab()
    painted.clear()
    return plot, curve, painted


def test_data_updates_reuse_axis_pixels_without_hiding_changed_curves(painted_plot):
    plot, curve, painted = painted_plot
    previous = plot.grab().toImage()
    for values in ([2, 1, 4, 3], [1, 3, 2, 4]):
        curve.setData([0, 1, 2, 3], values)
        image = plot.grab().toImage()
        assert image != previous
        previous = image
    assert painted == []


@pytest.mark.parametrize("change", ["range", "style"])
def test_axis_changes_invalidate_pixels_and_match_uncached_rendering(painted_plot, change):
    plot, _curve, painted = painted_plot
    before = plot.grab().toImage()
    if change == "range":
        plot.setXRange(-2, 4, padding=0)
    else:
        plot.getAxis("bottom").setTextPen("red")
        plot.getAxis("bottom").setGrid(180)
    cached = plot.grab().toImage()
    assert painted and cached != before
    for name in ("left", "bottom", "right", "top"):
        plot.getAxis(name).setCacheMode(QGraphicsItem.CacheMode.NoCache)
    assert plot.grab().toImage() == cached


def test_empty_message_preserves_bounds_and_reuses_text_on_unchanged_updates(
    painted_plot, monkeypatch
):
    plot, _curve, _painted = painted_plot
    previous_range = plot.viewRange()
    plot.set_empty_message("No data available<br>for this selection", color="#526784")
    label = plot._empty_message
    monkeypatch.setattr(
        label, "setText", lambda *args, **kwargs: pytest.fail("Unchanged text rebuilt")
    )
    for _ in range(10):
        plot.set_empty_message("No data available<br>for this selection", color="#526784")
    assert plot.viewRange() == previous_range
    assert label not in plot.getViewBox().addedItems
    plot.set_empty_message("")
    assert not label.isVisible()
    plot.set_empty_message("No data available<br>for this selection", color="#526784")
    assert label.isVisible()


def test_unchanged_axis_label_preserves_paint_cache_and_automatic_units(qtbot, monkeypatch):
    plot = DashboardPlot()
    qtbot.addWidget(plot)
    plot.setLabel("bottom", "Time", units="s")
    axis = plot.getPlotItem().getAxis("bottom")
    axis.setRange(0, 0.001)
    prefix = axis.labelUnitPrefix
    calls = []
    original = axis._updateLabel

    def update_label():
        calls.append(True)
        original()

    monkeypatch.setattr(axis, "_updateLabel", update_label)
    plot.setLabel("bottom", "Time", units="s")
    assert calls == []
    assert axis.labelUnitPrefix == prefix
    plot.setLabel("bottom", "Range", units="m", color="#123456")
    assert calls and axis.labelText == "Range"
    assert "#123456" in axis.label.toHtml()
    axis.setLabel("External label")
    plot.setLabel("bottom", "Range", units="m", color="#123456")
    assert axis.labelText == "Range"


def test_dynamic_legend_wraps_without_covering_data_or_clipping_labels(qtbot):
    plot = DashboardPlot()
    qtbot.addWidget(plot)
    legend = plot.add_external_legend(labelTextSize="8pt")
    plot.setTitle("Saved range history")
    plot.resize(380, 320)
    plot.show()
    series = []
    for index in range(8):
        series.append(plot.plot([0, 1], [index, index + 1], name=f"GT moving target {index}"))

    def check():
        view = plot.getViewBox()
        assert legend.sceneBoundingRect().bottom() <= view.sceneBoundingRect().top()
        assert view.height() >= 100
        for _, label in legend.items:
            assert plot.sceneRect().contains(label.item.sceneBoundingRect())

    for width in (380, 1000, 360):
        plot.resize(width, 320)
        qtbot.waitUntil(check)
    tall = plot.minimumHeight()
    for item in series[1:]:
        plot.removeItem(item)
    qtbot.waitUntil(check)
    qtbot.waitUntil(lambda: plot.minimumHeight() < tall)
    assert len(legend.items) == 1


def test_repeated_title_preserves_layout_but_style_and_resize_still_apply(qtbot):
    plot = DashboardPlot()
    qtbot.addWidget(plot)
    plot.add_external_legend()
    plot.resize(1000, 320)
    plot.show()
    title = "A long sensing title that should wrap when the panel becomes narrow"
    plot.setTitle(title, color="#aabbcc")
    qtbot.waitUntil(lambda: not plot._legend_timer.isActive())
    wide_height = plot.getPlotItem().titleLabel.maximumHeight()
    plot.setTitle(title)
    assert not plot._legend_timer.isActive()
    assert plot.getPlotItem().titleLabel.maximumHeight() == wide_height

    plot.setTitle(title, color="#112233")
    assert "#112233" in plot.getPlotItem().titleLabel.item.toHtml()
    plot.resize(340, 320)
    qtbot.waitUntil(lambda: plot.getPlotItem().titleLabel.maximumHeight() > wide_height)
    plot.setTitle(None)
    assert not plot.getPlotItem().titleLabel.isVisible()
    plot.setTitle(title)
    assert plot.getPlotItem().titleLabel.isVisible()


def test_wrapping_title_does_not_leave_a_short_viewport_and_clipped_axes(qtbot):
    from PySide6.QtWidgets import QHBoxLayout, QTableWidget, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    layout = QHBoxLayout(host)
    plot = DashboardPlot()
    layout.addWidget(plot)
    layout.addWidget(QTableWidget(10, 4))
    plot.add_external_legend(labelTextSize="8pt")
    plot.setLabel("left", "Power (dB)")
    plot.setLabel("bottom", "Range (m)")
    for name in ("Detector input", "CFAR Threshold", "Published peaks in this bin"):
        plot.plot([0, 10, 20, 40], [-150, -50, -200, -150], name=name)
    plot.set_data_minimum_height(180)
    plot.setTitle("Input / threshold · Doppler bin 41 · 141 Hz · 0.753 m/s")
    host.resize(1267, 400)
    host.show()

    def axes_fit():
        assert plot.viewport().size() == plot.size()
        bottom = plot.getPlotItem().getAxis("bottom").sceneBoundingRect()
        visible = plot.mapToScene(plot.viewport().rect()).boundingRect()
        assert visible.contains(bottom)

    qtbot.waitUntil(axes_fit)
    # The initial viewport can fit before the queued title/legend reflow has
    # replaced the pre-show minimum. Compare settled layouts, not that transient.
    qtbot.waitUntil(lambda: not plot._legend_timer.isActive())
    previous_minimum = plot.minimumHeight()
    host.resize(1000, 320)
    qtbot.waitUntil(lambda: plot.minimumHeight() > previous_minimum)
    qtbot.waitUntil(axes_fit)
    host.resize(1267, 400)
    qtbot.waitUntil(axes_fit)


def test_narrow_negative_power_colorbar_stays_inside_viewport_after_legend_reflow(qtbot):
    import numpy as np
    import pyqtgraph as pg

    plot = DashboardPlot()
    qtbot.addWidget(plot)
    plot.resize(1000, 340)
    legend = plot.add_external_legend(labelTextSize="8pt", verSpacing=0, horSpacing=5)
    plot.setTitle(
        "Range–Velocity<br><span style='font-size: 9pt'>"
        "Range window: Hann · Doppler window: Hann</span>"
    )
    plot.setLabel("left", "Radial closing velocity (m/s)")
    plot.setLabel("bottom", "Range (m)")
    values = np.linspace(-150, -100, 100).reshape(10, 10)
    image = pg.ImageItem(values)
    plot.addItem(image)
    bar = pg.ColorBarItem(
        colorMap=pg.colormap.get("viridis"), label="Power (dB)", interactive=False, width=15
    )
    bar.setImageItem(image, insert_in=plot.getPlotItem())
    bar.setLevels(values=(-150, -100))
    for name in ("Detection", "Ground truth", "Selected saved bin"):
        plot.plot([1], [1], symbol="x", pen=None, name=name)
    plot.show()

    def all_content_fits():
        bounds = plot.mapToScene(plot.viewport().rect()).boundingRect()
        assert bounds.contains(plot.getPlotItem().sceneBoundingRect())
        assert bounds.contains(bar.getAxis("right").sceneBoundingRect())
        for sample, label in legend.items:
            assert bounds.contains(sample.sceneBoundingRect())
            assert bounds.contains(label.item.sceneBoundingRect())
        assert bar.levels() == (-150, -100)
        np.testing.assert_array_equal(image.image, values)

    qtbot.waitUntil(all_content_fits)
    for width in (378, 296, 1000, 378):
        plot.resize(width, 340)
        qtbot.waitUntil(all_content_fits)
