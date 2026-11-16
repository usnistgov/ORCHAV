"""Collapsed figure controls leave navigation, loading and live evidence usable."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, QSettings

from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products
from visualizer.src.sensing.sensing_window import SensingDashboard, SensingWindow
from visualizer.src.widgets.dashboard_host import DashboardHost, dashboard_is_visible


@pytest.fixture(autouse=True)
def preferences():
    settings = QSettings()
    keys = (
        "visualizer/sensing_figure_options_open",
        "visualizer/sensing_browser_open",
        "visualizer/sensing_heatmaps",
        "test/sensing_figure_options_geometry",
    )
    previous = {key: settings.value(key) for key in keys}
    for key in keys:
        settings.remove(key)
    yield
    for key, value in previous.items():
        settings.remove(key) if value is None else settings.setValue(key, value)


@pytest.mark.parametrize("placement", ["standalone", "embedded", "detached"])
@pytest.mark.parametrize("width", [460, 1100])
def test_small_window_reaches_figures_with_options_collapsed(qtbot, placement, width):
    host = None
    if placement == "standalone":
        window = SensingWindow()
        shell = window
    else:
        host = DashboardHost(title="Sensing", settings_key="test/sensing_figure_options_geometry")
        qtbot.addWidget(host)
        window = SensingDashboard()
        host.set_dashboard(window)
        host.show()
        if placement == "detached":
            window.placement_button.click()
            shell = host.floating_window
        else:
            shell = host
    qtbot.addWidget(window)
    shell.resize(width, 700)
    shell.show()
    window._set_view_mode("analysis")
    window.table_button.setChecked(False)
    request = _request(0, _history())
    window.update_from_request(request)
    window.set_result_navigation((0, 1), previous=True, following=True)

    def figures_reachable():
        viewport = window.evidence_scroll.viewport()
        bounds = QRect(window.rd_plot.mapTo(viewport, QPoint()), window.rd_plot.size())
        assert 0 <= bounds.top() < 65
        assert viewport.rect().contains(bounds)
        assert window.evidence_scroll.horizontalScrollBar().maximum() == 0

    qtbot.waitUntil(figures_reachable)
    assert not window.figure_options.is_expanded()
    assert not window._advanced_widget.isVisible()
    assert not window.current_result_summary.isVisible()
    for widget in (
        window.workspace_box,
        window.workspace_tabs,
        window.previous_result_button,
        window.next_result_button,
        window.navigation_label,
        window.table_button,
    ):
        assert widget.isVisible()
    assert dashboard_is_visible(window)

    image = window.rd_image.image
    window.rd_plot.setRange(xRange=(0.5, 2), yRange=(-1, 1), padding=0)
    bounds = window.rd_plot.viewRange()
    changes = []
    window.view_changed.connect(lambda: changes.append(True))
    window.figure_options._toggle.click()
    assert window._advanced_widget.isVisible()
    assert window.current_result_summary.isVisible()
    window.figure_options._toggle.click()
    qtbot.waitUntil(figures_reachable)
    assert window._last_request is request and window.rd_image.image is image
    np.testing.assert_allclose(window.rd_plot.viewRange(), bounds)
    assert not changes

    next_request = _request(1, _history())
    next_request.rd_result.power_linear *= 2
    _rematerialize_products(next_request.display_frame["sensing"])
    window.update_from_request(next_request)
    assert window._rendered_request is next_request
    np.testing.assert_allclose(
        window.rd_image.image, 10 * np.log10(next_request.rd_result.power_linear.T)
    )
    # Presentation presets still address these exact controls while hidden.
    window._profile_detections_cb.setChecked(True)
    window._show_det_cb.setChecked(False)
    assert not window.rd_det_scatter.isVisible()
    assert window._profile_detections_cb.isChecked()
    assert not window.figure_options.is_expanded()
    if host is not None:
        if host.is_detached():
            window.placement_button.click()
        host.dispose()


def test_expansion_preference_survives_new_dashboard(qtbot):
    first = SensingWindow()
    qtbot.addWidget(first)
    assert not first.figure_options.is_expanded()
    first.figure_options.expand()
    second = SensingWindow()
    qtbot.addWidget(second)
    assert second.figure_options.is_expanded()
    second.figure_options.collapse()
    third = SensingWindow()
    qtbot.addWidget(third)
    assert not third.figure_options.is_expanded()


def test_compact_header_keeps_result_navigation_above_laptop_figures(qtbot):
    host = DashboardHost(title="Sensing", settings_key="test/sensing_figure_options_geometry")
    qtbot.addWidget(host)
    window = SensingDashboard()
    host.set_dashboard(window)
    host.resize(900, 500)
    host.show()
    window._set_view_mode("current")
    window.table_button.setChecked(False)
    window.update_from_request(_request(0, _history()))
    window.set_result_navigation((0, 1), previous=True, following=True)

    def compact_navigation():
        plot_top = window.rd_plot.mapTo(window, QPoint()).y()
        assert plot_top < 145
        navigation = (
            window.workspace_box,
            window.table_button,
            window.placement_button,
            window.previous_result_button,
            window.next_result_button,
            window.navigation_label,
            window.provenance_label,
        )
        boxes = [QRect(w.mapTo(window, QPoint()), w.size()) for w in navigation]
        for widget, box in zip(navigation, boxes):
            assert widget.isVisible()
            assert window.rect().contains(box)
            assert box.bottom() < window.workspace_tabs.y()
        assert all(not a.intersects(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])

    qtbot.waitUntil(compact_navigation)
    navigated = []
    window.result_step_requested.connect(navigated.append)
    window.next_result_button.click()
    assert navigated == [1]
    host.dispose()


def test_missing_products_and_loading_remain_visible_when_options_are_hidden(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("analysis")
    window.update_from_request(
        replace(_request(), rd_result=None, available_product_kinds=frozenset())
    )
    assert window.current_product_notice.isVisible()
    assert "not recorded" in window.current_product_notice.text()
    window.set_loading_progress("Loading overview", 1, 20, active=True)
    qtbot.wait(160)
    window.set_loading_progress("Loading overview", 2, 20, active=True)
    assert window.loading_progress.isVisible()
    assert window.cancel_loading_button.isVisible()
    assert not window.figure_options.is_expanded()
    window.set_unavailable("Saved result cannot be loaded", scene_frame_id=10)
    assert window._unavailable_label.isVisible()
    assert "Saved result cannot be loaded" in window._unavailable_label.text()
