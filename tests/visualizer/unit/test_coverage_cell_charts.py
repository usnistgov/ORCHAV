"""Location charts keep saved units, missing samples and selected-cell identity."""

import warnings
from dataclasses import replace

import numpy as np

from tests.visualizer.unit.test_coverage_slice_analysis import model as model
from visualizer.src.coverage.cell_charts import CoverageCellCharts, vertical_profile


def test_profile_preserves_zero_and_missing_gaps_without_changing_source(model):
    before = model.values.copy()
    values = vertical_profile(model, (0, 0, 0))
    assert np.isnan(values[0])
    np.testing.assert_array_equal(values[1:], model.values[1:, 0, 0])
    assert np.isnan(vertical_profile(model, (1, 1, 2))[1])
    np.testing.assert_array_equal(model.values, before)


def test_lazy_charts_show_only_finite_power_and_raw_height_samples(qtbot, model):
    widget = CoverageCellCharts(None)
    qtbot.addWidget(widget)
    widget.set_context(model, (1, 1, 2), -80)
    data = dict(
        names=["A", "B", "C"],
        rss_dbm=[-63, -270, float("nan")],
        availability=[1, 2, 0],
        serving_tx=0,
    )
    widget.set_data(data)
    assert widget.canvas is None
    widget.enabled.setChecked(True)
    qtbot.waitUntil(lambda: widget.canvas is not None)
    assert len(widget.power_bars.opts["x1"]) == 1
    texts = [
        item.toPlainText() for item in widget.power_plot.items() if hasattr(item, "toPlainText")
    ]
    assert "Zero contribution" in texts and "Unknown" in texts
    np.testing.assert_array_equal(widget.profile_curve.xData, vertical_profile(model, (1, 1, 2)))
    np.testing.assert_array_equal(widget.profile_curve.yData, model.axes[2])
    assert any(
        label.text == "View threshold" for _, label in widget.power_plot._external_legend.items
    )
    assert any(
        label.text == "Selected height: 13 m"
        for _, label in widget.profile_plot._external_legend.items
    )
    widget.set_context(model, (1, 1, 2), -80)
    assert not widget.timer.isActive()
    loss = replace(model, quantity="path_loss_db/A", values=-model.values)
    widget.set_context(loss, (1, 1, 2), 80)
    qtbot.waitUntil(lambda: not widget.timer.isActive())
    assert not any(
        label.text == "View threshold" for _, label in widget.power_plot._external_legend.items
    )
    assert any(
        label.text == "View threshold" for _, label in widget.profile_plot._external_legend.items
    )
    assert "dB" in widget.profile_plot.getAxis("bottom").labelText
    widget.shutdown()


def test_unavailable_transmitters_do_not_create_empty_bar_geometry(qtbot, model):
    widget = CoverageCellCharts(None)
    qtbot.addWidget(widget)
    widget.set_context(model, (0, 0, 0), None)
    widget.set_data(
        dict(names=["A", "B"], rss_dbm=[np.nan, -270], availability=[0, 2], serving_tx=-1)
    )
    widget.enabled.setChecked(True)
    qtbot.waitUntil(lambda: widget.canvas is not None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert widget.power_bars is None
        widget.show()
        assert not widget.grab().isNull()
    widget.shutdown()
