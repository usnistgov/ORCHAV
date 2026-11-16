"""Real Qt range restoration must not transiently auto-range the linked X axis."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from visualizer.src.sensing import sensing_window


def drain():
    for _ in range(4):
        QApplication.processEvents()


@pytest.fixture
def dashboard(qtbot):
    window = sensing_window.SensingWindow()
    qtbot.addWidget(window)
    window.resize(1200, 950)
    window._set_view_mode("current")
    window._zoom_data_cb.setChecked(False)
    window.show()
    drain()
    return window


def install(window, axis, *, y=2.0):
    power_db = np.linspace(-150.0, -90.0, len(axis))
    power = 10.0 ** (power_db / 10.0)
    window._update_range_profile(SimpleNamespace(range_axis_m=axis, power_linear=power), "")
    edges = window._axis_edges(axis, axis_name="range")
    window.rd_image.setImage(np.tile(power_db, (3, 1)).T, autoLevels=False)
    window.rd_image.setRect(QRectF(edges[0], -1.5, edges[-1] - edges[0], 3.0))
    # Overlay X must not enlarge the saved-cell view; Y still participates in auto range.
    window.rd_det_scatter.setData([axis[-1] + 50], [y], data=[17])
    window._selected_rd_cells.setData([axis[0]], [0.0])
    window._restore_full_spectrum_axes(window.rd_plot, axis, np.arange(-1, 2))
    drain()
    return power_db


@pytest.mark.parametrize("axis", [np.arange(32) * 0.25, np.arange(-4, 5) * 0.5])
def test_repeated_saved_cells_do_not_emit_intermediate_map_or_profile_ranges(dashboard, axis):
    window = dashboard
    power_db = install(window, axis)
    expected = list(window.rd_plot.viewRange()[0])
    ranges = {"map": [], "profile": []}
    window.rd_plot.getViewBox().sigXRangeChanged.connect(
        lambda _, value: ranges["map"].append(tuple(value))
    )
    window.rp_plot.getViewBox().sigXRangeChanged.connect(
        lambda _, value: ranges["profile"].append(tuple(value))
    )
    for _ in range(3):
        window._restore_full_spectrum_axes(window.rd_plot, axis, np.arange(-1, 2))
        drain()
        np.testing.assert_allclose(window.rd_plot.viewRange()[0], expected)
        np.testing.assert_allclose(window.rp_plot.viewRange()[0], expected)
    assert ranges == {"map": [], "profile": []}
    assert window.rd_plot.getViewBox().autoRangeEnabled() == [False, True]
    np.testing.assert_array_equal(window.rp_curve.getData()[0], axis)
    np.testing.assert_allclose(window.rp_curve.getData()[1], power_db)
    assert window.rd_det_scatter.points()[0].data() == 17
    np.testing.assert_array_equal(window._selected_rd_cells.getData()[0], [axis[0]])


def test_full_range_still_tracks_y_and_changed_saved_cells(dashboard):
    window = dashboard
    axis = np.arange(16) * 0.25
    install(window, axis, y=2.0)
    first_y = window.rd_plot.viewRange()[1]
    new_axis = np.arange(24) * 0.5
    samples = install(window, new_axis, y=40.0)
    expected_x = [0.0, new_axis[-1] + 0.25]
    np.testing.assert_allclose(window.rd_plot.viewRange()[0], expected_x)
    np.testing.assert_allclose(window.rp_plot.viewRange()[0], expected_x)
    assert window.rd_plot.getViewBox().autoRangeEnabled() == [False, True]
    assert window.rd_plot.viewRange()[1][1] >= 40.0
    assert window.rd_plot.viewRange()[1][1] > first_y[1]
    np.testing.assert_allclose(window.rp_curve.getData()[1], samples)


def test_zoom_then_full_range_matches_existing_final_map_and_profile_bounds(dashboard):
    window = dashboard
    axis = np.arange(32) * 0.25
    install(window, axis)
    # Reference the pre-change final result on these same real widgets.
    window.rd_plot.getViewBox().enableAutoRange()
    window.rd_plot.setXRange(0.0, axis[-1] + 0.125, padding=0)
    drain()
    expected_map = window.rd_plot.viewRange()
    expected_profile = window.rp_plot.viewRange()
    window._zoom_data_cb.setChecked(True)
    power_db = np.full((3, len(axis)), -150.0)
    power_db[1, 5] = -50.0
    window._apply_zoom_to_data(window.rd_plot, power_db, axis, np.arange(-1, 2))
    assert window.rd_plot.getViewBox().autoRangeEnabled() == [False, False]
    window._zoom_data_cb.setChecked(False)
    window._apply_zoom_to_data(window.rd_plot, power_db, axis, np.arange(-1, 2))
    drain()
    assert window.rd_plot.getViewBox().autoRangeEnabled() == [False, True]
    np.testing.assert_allclose(window.rd_plot.viewRange(), expected_map)
    np.testing.assert_allclose(window.rp_plot.viewRange(), expected_profile)


def test_other_plots_keep_both_axes_automatic(dashboard):
    plot = dashboard.gt_range_plot
    plot.plot([1.0, 3.0], [2.0, 6.0])
    plot.getViewBox().disableAutoRange()
    dashboard._restore_full_spectrum_axes(plot, np.array([1.0, 3.0]), np.array([2.0, 6.0]))
    drain()
    assert plot.getViewBox().autoRangeEnabled() == [True, True]
    assert plot.getViewBox().viewRange()[0][0] <= 1.0
    assert plot.getViewBox().viewRange()[0][1] >= 3.0
