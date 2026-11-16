"""Sensing figures receive space released by the optional results browser."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, QSettings
from PySide6.QtWidgets import QTabWidget, QWidget

from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
from visualizer.src.app.panel_manager import UIPanelManager
from visualizer.src.panels.sensing_panel import SensingPanel


@pytest.fixture
def dashboard_tab(qtbot, monkeypatch):
    settings = QSettings()
    keys = (
        "layout/sections/sensing",
        "layout/sections/sensing_results",
        "visualizer/sensing_browser_open",
        "visualizer/sensing_dashboard_geometry",
        "visualizer/sensing_figure_options_open",
    )
    previous = {key: settings.value(key) for key in keys}
    for key in keys:
        settings.remove(key)
    panel = SensingPanel(SimpleNamespace(renderer=None))
    monkeypatch.setattr(panel, "_refresh_live_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr(panel, "_on_linked_selection", lambda selection: None)

    # Exercise the actual tab builder: two collapsible bodies and its trailing
    # spacer are what previously shrank the dashboard when the table closed.
    manager = object.__new__(UIPanelManager)
    manager.panels = {"sensing": panel}
    manager._runtime_extensions = {}
    manager.sections = {}
    manager._tab_map = {}
    manager._tab_widget_by_panel = {}
    manager._tab_widget = QTabWidget()
    manager._add_tab("Sensing", [("sensing", True)])
    tabs = manager._tab_widget
    qtbot.addWidget(tabs)
    tabs.resize(1300, 1340)
    tabs.show()
    qtbot.waitUntil(lambda: panel._sensing_window is not None)
    dashboard = panel._sensing_window
    dashboard._initial_view_selected = True
    dashboard._set_view_mode("analysis")
    dashboard.update_from_request(_request(0, _history()))
    _settle(qtbot, dashboard)
    yield panel, tabs, manager._tab_widget_by_panel["sensing"], dashboard
    panel.cleanup()
    tabs.close()
    for key, value in previous.items():
        settings.remove(key) if value is None else settings.setValue(key, value)


def _settle(qtbot, dashboard):
    previous, unchanged = None, 0
    plots = (dashboard.rd_plot, dashboard.rp_plot, dashboard.gt_range_plot, dashboard.gt_speed_plot)

    def stable():
        nonlocal previous, unchanged
        state = (
            dashboard.size().toTuple(),
            dashboard.evidence_scroll.viewport().size().toTuple(),
            dashboard.result_scroll.viewport().size().toTuple(),
            dashboard.selection_tools.geometry().getRect(),
            tuple(plot.geometry().getRect() for plot in plots),
        )
        idle = all(not plot._legend_timer.isActive() for plot in plots)
        unchanged = unchanged + 1 if idle and state == previous else 0
        previous = state
        return unchanged >= 4

    qtbot.waitUntil(stable)


def _assert_both_rows_fit(dashboard):
    viewport = dashboard.evidence_scroll.viewport()
    for plot in (
        dashboard.rd_plot,
        dashboard.rp_plot,
        dashboard.gt_range_plot,
        dashboard.gt_speed_plot,
    ):
        bounds = QRect(plot.mapTo(viewport, QPoint()), plot.size())
        assert viewport.rect().contains(bounds)
        # The saved axes and legends must stay within each plot's own viewport.
        item = plot.getPlotItem()
        scene = plot.sceneRect().adjusted(-1, -1, 1, 1)
        assert scene.contains(item.getAxis("bottom").sceneBoundingRect())
        assert scene.contains(item.getAxis("left").sceneBoundingRect())
    assert dashboard.evidence_scroll.verticalScrollBar().maximum() == 0


def test_hiding_table_reclaims_space_and_displays_both_figure_rows(dashboard_tab, qtbot):
    _, _, outer_scroll, dashboard = dashboard_tab
    dashboard.table_button.setChecked(True)
    _settle(qtbot, dashboard)
    dashboard_height = dashboard.height()
    before = dashboard.evidence_scroll.viewport().height()
    dashboard.table_button.setChecked(False)
    _settle(qtbot, dashboard)
    assert dashboard.height() >= dashboard_height
    assert dashboard.evidence_scroll.viewport().height() > before
    assert outer_scroll.verticalScrollBar().maximum() == 0
    _assert_both_rows_fit(dashboard)


def test_narrow_dashboard_scrolls_and_graph_height_remains_respected(dashboard_tab, qtbot):
    _, tabs, _, dashboard = dashboard_tab
    dashboard.table_button.setChecked(False)
    tabs.resize(440, 950)
    _settle(qtbot, dashboard)
    assert dashboard._compact
    assert dashboard.evidence_scroll.verticalScrollBar().maximum() > 0
    assert dashboard.evidence_scroll.horizontalScrollBar().maximum() == 0
    tabs.resize(1300, 1340)
    _settle(qtbot, dashboard)
    _assert_both_rows_fit(dashboard)
    original = dashboard.rd_plot.height()
    requested_height = max(420, original + 40)
    dashboard._graph_height_box.setValue(requested_height)
    _settle(qtbot, dashboard)
    assert dashboard.rd_plot.height() > original
    assert dashboard.rd_plot.getViewBox().height() >= requested_height
    assert dashboard.evidence_scroll.verticalScrollBar().maximum() > 0
    dashboard._graph_height_box.setValue(180)
    _settle(qtbot, dashboard)
    _assert_both_rows_fit(dashboard)


def test_table_toggle_and_popout_preserve_plot_data_zoom_and_full_axes(
    dashboard_tab, qtbot, mocker
):
    panel, _, _, dashboard = dashboard_tab
    request = dashboard._last_request
    image = dashboard.rd_image.image
    dashboard.rd_plot.setRange(xRange=(0.5, 2), yRange=(-1, 1), padding=0)
    bounds = dashboard.rd_plot.viewRange()
    render = mocker.spy(dashboard, "update_from_request")
    for _ in range(2):
        dashboard.table_button.setChecked(True)
        _settle(qtbot, dashboard)
        dashboard.table_button.setChecked(False)
        _settle(qtbot, dashboard)
        _assert_both_rows_fit(dashboard)
        panel.dashboard_host.detach()
        panel.dashboard_host.floating_window.resize(1300, 1340)
        _settle(qtbot, dashboard)
        _assert_both_rows_fit(dashboard)
        panel.dashboard_host.dock()
        _settle(qtbot, dashboard)
        _assert_both_rows_fit(dashboard)
        assert dashboard._last_request is request
        assert dashboard.rd_image.image is image
        np.testing.assert_allclose(dashboard.rd_plot.viewRange(), bounds)
        render.assert_not_called()


def test_collapsing_sections_keeps_headers_compact_and_restores_figures(
    dashboard_tab, qtbot, mocker
):
    panel, _, outer_scroll, dashboard = dashboard_tab
    dashboard.table_button.setChecked(False)
    _settle(qtbot, dashboard)
    request, image = dashboard._last_request, dashboard.rd_image.image
    render = mocker.spy(dashboard, "update_from_request")
    outer_section = outer_scroll.widget().layout().itemAt(0).widget()
    for section in (outer_section, panel.results_section):
        original_top = section.mapTo(outer_scroll.viewport(), QPoint()).y()
        section.collapse()
        _settle(qtbot, dashboard)
        assert not dashboard.isVisible()
        assert section.height() == section._toggle.sizeHint().height()
        assert section._toggle.y() == 0
        assert section.mapTo(outer_scroll.viewport(), QPoint()).y() <= original_top + 1
        section.expand()
        _settle(qtbot, dashboard)
        assert dashboard.isVisible()
        assert dashboard._last_request is request
        assert dashboard.rd_image.image is image
        _assert_both_rows_fit(dashboard)
        render.assert_not_called()


@pytest.mark.parametrize("width", [780, 440])
def test_retention_notice_has_readable_row_outside_collapsed_options(dashboard_tab, qtbot, width):
    panel, tabs, _, dashboard = dashboard_tab
    tabs.resize(width, 850)
    dashboard.figure_options.collapse()
    history = replace(
        _history(), truncated=True, available_results=30, byte_limit=4096, result_limit=6
    )
    dashboard.update_from_request(_request(0, history))

    def assert_readable():
        _settle(qtbot, dashboard)
        notice = dashboard.retention_notice
        body = dashboard.evidence_scroll.widget()
        assert notice.isVisible() and not dashboard.figure_options.is_expanded()
        assert notice.parentWidget() is body
        assert body.layout().indexOf(notice) >= 0
        assert notice.geometry().bottom() < dashboard._plots_widget.geometry().top()
        assert body.rect().contains(notice.geometry())
        assert notice.height() >= notice.heightForWidth(notice.width())
        assert notice.width() >= body.width() - 2
        assert "--sensing-history-cache-mb" in notice.text()

    assert_readable()
    panel.dashboard_host.detach()
    panel.dashboard_host.floating_window.resize(width, 850)
    assert_readable()
    panel.dashboard_host.dock()
    assert_readable()
    dashboard.update_from_request(_request(0, _history()))
    assert dashboard.retention_notice.isHidden()


@pytest.mark.parametrize("width", [780, 440])
def test_selection_details_expand_without_clipping_reveal_or_range_preview(
    dashboard_tab, qtbot, width
):
    _, tabs, _, dashboard = dashboard_tab
    tabs.resize(width, 850)
    _settle(qtbot, dashboard)
    request = replace(_request(), display_step=0, source_identity=("recording",))
    request.display_frame.update(
        timestamp_s=0.0, tx_positions=[[0, 0, 0]], rx_positions=[[0, 0, 0]]
    )
    request.display_frame["sensing"]["detections"] = [
        dict(range_m=1.0, doppler_hz=0.0, velocity_m_s=0.0, range_bin=1, doppler_bin=2)
    ]
    dashboard.update_from_request(request)
    assert dashboard.figure_options.isVisible()
    dashboard.selection_tools_toggle.setChecked(True)
    dashboard.select_detection_index(0)
    controls = dashboard.scene_explanation
    controls.phase.setCurrentIndex(3)
    controls.note.setText("Saved report position at scene 0; not an identified object centre.")
    controls.response.setCurrentIndex(2)
    controls.response_note.setText(
        "16 × 16 RX · Maximum of 193 coarse beams. Relative array power −40…0 dB; "
        "arbitrary drawing radius. Front/back lobes retained. "
        "Not received power or detection coverage."
    )
    _settle(qtbot, dashboard)

    def assert_readable():
        # Inspect the actual allocated geometry, not just Qt's requested sizes.
        assert dashboard.selection_tools.parentWidget() is dashboard.result_scroll.widget()
        for widget in (
            dashboard.cell_explanation,
            controls,
            controls.phase,
            controls.play,
            controls.note,
            controls.response,
            controls.radius,
            controls.response_note,
        ):
            assert widget.isVisible()
            assert widget.height() >= widget.minimumSizeHint().height()
            if widget.hasHeightForWidth():
                assert widget.height() >= widget.heightForWidth(widget.width())
            assert widget.parentWidget().rect().contains(QWidget.geometry(widget))
        fields = (
            controls.phase,
            controls.play,
            controls.note,
            controls.response,
            controls.radius,
            controls.response_note,
        )
        for index, field in enumerate(fields):
            for other in fields[index + 1 :]:
                assert not field.geometry().intersects(other.geometry())
        cell = dashboard.cell_explanation
        captions = (cell.values, cell.decision, cell.paths_note)
        for caption, following in zip(captions, captions[1:]):
            assert not caption.geometry().intersects(following.geometry())
        assert dashboard.result_scroll.horizontalScrollBar().maximum() == 0

    assert_readable()
    assert dashboard.result_scroll.verticalScrollBar().maximum() > 0
    scrollbar = dashboard.result_scroll.verticalScrollBar()
    scrollbar.setValue(0)
    before_scroll = dashboard.selection_tools.mapTo(dashboard.result_scroll.viewport(), QPoint())
    scrollbar.setValue(50)
    after_scroll = dashboard.selection_tools.mapTo(dashboard.result_scroll.viewport(), QPoint())
    assert before_scroll.y() - after_scroll.y() == scrollbar.value()
    dashboard.result_scroll.ensureWidgetVisible(controls.response_note)
    _settle(qtbot, dashboard)
    viewport = dashboard.result_scroll.viewport()
    assert viewport.rect().contains(
        QRect(controls.response_note.mapTo(viewport, QPoint()), controls.response_note.size())
    )
    controls.response.setCurrentIndex(0)
    _settle(qtbot, dashboard)
    assert controls.radius.isHidden() and controls.response_note.isHidden()
    controls.response.setCurrentIndex(2)
    _settle(qtbot, dashboard)
    assert_readable()
    for resized_width in (440, 1100, width):
        tabs.resize(resized_width, 850)
        _settle(qtbot, dashboard)
        assert_readable()
    dashboard.selection_tools_toggle.setChecked(False)
    _settle(qtbot, dashboard)
    dashboard.selection_tools_toggle.setChecked(True)
    _settle(qtbot, dashboard)
    assert_readable()
    assert dashboard._last_request is request
