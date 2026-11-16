"""Workspace navigation preserves current evidence without forcing dense reads."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
from tests.visualizer.unit.test_sensing_tracking_presentation import native_sensing
from visualizer.src.sensing.dashboard_data import HistoryRecord, RecordingHistory
from visualizer.src.sensing.selection import selection_scope
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.tracking_presentation import tracking_snapshot


@pytest.fixture
def dashboard(qtbot):
    settings = QSettings()
    keys = ("visualizer/sensing_browser_open", "visualizer/sensing_figure_options_open")
    previous = {key: settings.value(key) for key in keys}
    for key in keys:
        settings.setValue(key, True)
    window = SensingWindow()
    window.resize(1100, 850)
    qtbot.addWidget(window)
    window.show()
    yield window
    for key, value in previous.items():
        settings.remove(key) if value is None else settings.setValue(key, value)


def _radial_request(step=4):
    request = _request(step)
    sensing = request.display_frame["sensing"]
    sensing["config_resolved"]["track_confirmation_enabled"] = True
    sensing["config"]["track_confirmation_enabled"] = True
    sensing["detections"] = native_sensing()["detections"]
    time = sensing["temporal"]["result"]["reference_time_s"]
    identity = (*selection_scope(sensing), sensing["temporal"]["acquisition"]["acquisition_id"])
    history = RecordingHistory(
        (
            HistoryRecord(
                time_s=time,
                identity=identity,
                spectrum=None,
                doppler_hz=None,
                range_m=None,
                range_power=None,
                detections=tuple(sensing["detections"]),
                truth=None,
                truth_positions=(),
                cadence_s=0.25,
                tracking_states=tracking_snapshot(sensing).states,
                sensing_metadata=sensing,
            ),
        )
    )
    return replace(request, recording_history=history)


def test_four_destinations_shared_browser_and_optional_selection(dashboard, qtbot):
    assert [
        dashboard.workspace_box.itemText(i) for i in range(dashboard.workspace_box.count())
    ] == [
        "Run summary",
        "Current result",
        "Motion",
        "Tracking",
    ]
    assert dashboard.workspace_box.currentText() == "Run summary"
    request = _radial_request()
    for mode in (
        "summary",
        "current",
        "analysis",
        "diagnostics",
        "evaluation",
        "micro-doppler",
        "tracking",
    ):
        dashboard._set_view_mode(mode)
        dashboard.update_from_request(request)
        qtbot.wait(1)
        per_result = mode not in {"summary", "evaluation"}
        assert dashboard.result_browser.isVisible() == per_result
        assert dashboard.table_button.isVisible() == per_result
        assert dashboard.result_scroll.isVisible() == per_result
        assert dashboard.detection_table_model.rowCount() == 1
        assert dashboard._linked_selection is None
        assert not dashboard.selection_tools_toggle.isChecked()
        assert dashboard.view_description.isHidden()
        assert not dashboard.range_shell_enabled.isChecked()


def test_table_remains_fixed_while_evidence_scrolls_and_collapses(dashboard, qtbot):
    dashboard._set_view_mode("analysis")
    dashboard.update_from_request(_request())
    dashboard._graph_height_box.setValue(600)
    qtbot.wait(10)
    position = dashboard.result_browser.mapToGlobal(dashboard.result_browser.rect().topLeft())
    scrollbar = dashboard.evidence_scroll.verticalScrollBar()
    assert scrollbar.maximum() > 0
    scrollbar.setValue(scrollbar.maximum())
    qtbot.wait(1)
    assert (
        dashboard.result_browser.mapToGlobal(dashboard.result_browser.rect().topLeft()) == position
    )
    dashboard.table_button.setChecked(False)
    assert dashboard.result_browser.isHidden()
    dashboard._set_view_mode("summary")
    dashboard.table_button.setChecked(True)
    assert not dashboard.result_browser.isVisible()
    assert not dashboard.table_button.isVisible()
    dashboard._set_view_mode("analysis")
    assert dashboard.result_browser.isVisible()


def test_selection_details_follow_table_and_hide_in_run_summary(dashboard, qtbot):
    dashboard._set_view_mode("analysis")
    dashboard.update_from_request(_radial_request())
    dashboard.select_detection_index(0)
    dashboard.selection_tools_toggle.setChecked(True)
    qtbot.wait(10)
    assert dashboard.selection_tools.isVisible()
    assert dashboard.selection_tools_toggle.parentWidget() is dashboard.result_scroll.widget()
    assert dashboard.selection_tools_toggle.y() > dashboard.result_browser.geometry().bottom()
    assert dashboard.selection_tools.y() > dashboard.selection_tools_toggle.geometry().bottom()
    dashboard._set_view_mode("summary")
    assert not dashboard.result_scroll.isVisible()
    assert not dashboard.selection_tools.isVisible()
    assert dashboard.table_button.isChecked()
    dashboard._set_view_mode("analysis")
    assert dashboard.result_browser.isVisible()
    assert dashboard.selection_tools.isVisible()
    dashboard.table_button.setChecked(False)
    assert not dashboard.result_browser.isVisible()
    assert dashboard.selection_tools.isVisible()
    dashboard.set_linked_selection(None)
    assert not dashboard.result_scroll.isVisible()


def test_radial_tracking_estimates_differ_from_measurements_and_do_not_leak_future(dashboard):
    request = _radial_request()
    dashboard._set_view_mode("tracking")
    dashboard.update_from_request(request)
    assert dashboard._selected_view_mode() == "tracking"
    assert dashboard.tracking_plot.isHidden()
    curve = dashboard._track_estimate_curves[("7", 0)]
    np.testing.assert_allclose(curve.getData()[1], [10.8])
    np.testing.assert_allclose(dashboard._gt_range_det_scatter.getData()[1], [11.0])
    dashboard.select_track("7")
    assert dashboard.tracking_inspector.selected_id == 7
    assert dashboard.result_browser.track_table.selectionModel().hasSelection()
    dashboard._render_tracking_history(request.recording_history, 0)
    dashboard._apply_overlay_visibility()
    assert not curve.isVisible()
    dashboard.tracking_inspector.toggle.setChecked(True)
    dashboard.set_linked_selection(None, notify=True)
    assert dashboard.tracking_inspector.selected_track() is None


def test_motion_browser_cursor_stays_metadata_only_and_selection_materializes_correct_result(
    dashboard, monkeypatch
):
    history = _history()
    request = _request(0, history)
    dashboard._set_view_mode("micro-doppler")
    dashboard.update_from_request(request)
    assert dashboard.can_move_recording_cursor(history)
    next_request = _request(1, history)
    sensing = next_request.display_frame["sensing"]
    sensing["detections"] = [{"range_m": 1.0, "velocity_m_s": 0.0}]
    identity = (*selection_scope(sensing), sensing["temporal"]["acquisition"]["acquisition_id"])
    record = replace(
        history.records[1],
        identity=identity,
        detections=tuple(sensing["detections"]),
        sensing_metadata=sensing,
    )
    dashboard.move_recording_cursor(record, "primary")
    assert dashboard._last_request is request
    assert dashboard.detection_table_model.rowCount() == 1
    dashboard._select_browser_detection(0)
    assert dashboard._linked_selection is None
    assert not dashboard.can_move_recording_cursor(history)
    dashboard.update_from_request(next_request)
    assert dashboard._linked_selection.acquisition_id == identity[-1]
    assert dashboard._linked_selection.detection_index == 0


def test_current_overview_keeps_detail_products_lazy(dashboard, monkeypatch):
    request = _radial_request()
    dashboard._set_view_mode("current")
    monkeypatch.setattr(
        dashboard,
        "_update_detection_stage_inspector",
        lambda *_: pytest.fail("hidden detector maps loaded"),
    )
    monkeypatch.setattr(
        dashboard, "_update_angular_products", lambda *_: pytest.fail("hidden angle product loaded")
    )
    dashboard.update_from_request(request)
    assert dashboard.current_result_summary.isVisible()
    assert dashboard.wants_recording_history()
    assert not dashboard.wants_signal_history()
    assert dashboard.request_view_mode() == "analysis"
    np.testing.assert_allclose(dashboard._gt_range_det_scatter.getData()[1], [11.0])
    np.testing.assert_allclose(
        dashboard._gt_speed_det_scatter.getData()[1],
        [request.display_frame["sensing"]["detections"][0]["velocity_m_s"]],
    )
    monkeypatch.undo()
    dashboard._open_result_detail("diagnostics", "cfar")
    assert dashboard.workspace_box.currentText() == "Current result"
    assert dashboard.workspace_tabs.currentIndex() == 1
    assert dashboard.request_view_mode() == "diagnostics"


def test_only_visible_motion_requests_independent_spectral_history(dashboard):
    for mode in ("current", "tracking", "evaluation"):
        dashboard._set_view_mode(mode)
        assert not dashboard.wants_signal_history()
        assert not dashboard.overview_families()
    dashboard.apply_display_preset({"view_mode": "micro-doppler", "micro_doppler_mode": "auto"})
    assert not dashboard.wants_signal_history()
    assert not dashboard.wants_recording_history()
    assert dashboard.overview_families() == {"summed-rd", "range-profile"}
    dashboard.hide()
    assert not dashboard.overview_families()


def test_history_bounds_reuse_immutable_data_but_follow_display_choices(dashboard, monkeypatch):
    history = _history()
    dashboard._set_view_mode("current")
    dashboard.update_from_request(_request(2, history))
    calls = []
    original = dashboard._history_axis_limits

    def compute(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(dashboard, "_history_axis_limits", compute)
    dashboard._fit_history_axes(history, 0.25)
    dashboard._fit_history_axes(history, 0.5)
    assert calls == []
    full_time = dashboard.gt_range_plot.viewRange()[0]
    dashboard._zoom_data_cb.setChecked(True)
    assert calls
    dashboard._fit_history_axes(history, 0.25)
    assert dashboard.gt_range_plot.viewRange()[0][1] < full_time[1]
    previous = len(calls)
    dashboard._show_gt_cb.setChecked(not dashboard._show_gt_cb.isChecked())
    assert len(calls) > previous
    previous = len(calls)
    dashboard._show_det_cb.setChecked(not dashboard._show_det_cb.isChecked())
    assert len(calls) > previous
    previous = len(calls)
    dashboard._fit_history_axes(replace(history), 0.25)
    assert len(calls) > previous


def test_history_scatter_reuses_prefix_and_clearing_selection_restores_in_place_styles(
    dashboard, monkeypatch
):
    import pyqtgraph as pg

    request = _radial_request()
    dashboard._set_view_mode("current")
    dashboard.update_from_request(request)
    scatter = dashboard._gt_range_det_scatter
    original_pen = scatter.data["pen"][0].color().name()
    calls = []
    original = scatter.setData

    def set_data(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(scatter, "setData", set_data)
    dashboard._render_detection_history()
    assert calls == []
    dashboard.select_track(7)
    calls.clear()
    data = scatter.data
    scatter.setPen([pg.mkPen("#123456")])
    assert scatter.data is data
    dashboard.set_linked_selection(None, notify=True)
    assert calls
    assert scatter.data["pen"][0].color().name() == original_pen


def test_current_result_tabs_keep_all_evidence_in_one_workspace(dashboard, qtbot):
    dashboard.workspace_box.setCurrentIndex(dashboard.workspace_box.findData("current"))
    dashboard.update_from_request(_request(0, _history()))
    assert [
        dashboard.workspace_tabs.tabText(i) for i in range(dashboard.workspace_tabs.count())
    ] == ["Detections", "Detector", "Angle", "Inputs"]
    browser = dashboard.result_browser
    assert dashboard.current_result_summary.isVisible()
    assert all(
        plot.isVisible()
        for plot in (
            dashboard.rd_plot,
            dashboard.rp_plot,
            dashboard.gt_range_plot,
            dashboard.gt_speed_plot,
        )
    )
    assert dashboard.wants_recording_history()

    # Availability remains source-owned: exposing the tab does not invent angles.
    assert not dashboard.workspace_tabs.isTabEnabled(2)
    for index, requested in (
        (1, "diagnostics"),
        (3, "processing-inputs"),
    ):
        dashboard.workspace_tabs.setCurrentIndex(index)
        qtbot.wait(1)
        assert dashboard.workspace_box.currentData() == "current"
        assert dashboard.request_view_mode() == requested
        assert dashboard.current_result_summary.isHidden()
        assert dashboard.result_browser is browser
        assert browser.isVisible()
        assert dashboard._linked_selection is None
    dashboard.workspace_tabs.setCurrentIndex(0)
    assert dashboard.current_result_summary.isVisible()
    assert dashboard.request_view_mode() == "analysis"
    assert dashboard.gt_range_plot.isVisible() and dashboard.gt_speed_plot.isVisible()

    # Existing internal callers can still ask for the analysis reader/view token.
    dashboard._set_view_mode("analysis")
    assert dashboard.workspace_box.currentData() == "current"
    assert dashboard.workspace_tabs.currentIndex() == 0
    assert dashboard.current_result_summary.isVisible()
    assert not hasattr(dashboard.current_result_summary.cards["Detection"], "details")
    dashboard.workspace_tabs.setCurrentIndex(1)
    assert dashboard.workspace_box.currentData() == "current"
    assert dashboard.workspace_tabs.currentIndex() == 1
    assert dashboard.request_view_mode() == "diagnostics"


def test_frequency_input_choice_requires_retained_frequency_observation(dashboard):
    request = _request()
    dashboard.update_from_request(replace(request, available_product_kinds=()))
    item = dashboard.input_kind_box.model().item(dashboard.input_kind_box.findData("frequency"))
    assert not item.isEnabled()
    dashboard.update_from_request(
        replace(request, available_product_kinds=("baseband_frequency_observation",))
    )
    assert item.isEnabled()


def test_full_dashboard_expands_inspection_without_a_resize(dashboard, qtbot):
    from matplotlib.figure import Figure

    dashboard.resize(1920, 1080)
    dashboard._set_view_mode("tracking")
    dashboard.update_from_request(_radial_request())
    qtbot.wait(10)
    size = dashboard.size()
    before = dashboard.tracking_inspector.height()
    dashboard.tracking_inspector.toggle.setChecked(True)
    dashboard.tracking_inspector.raw_toggle.setChecked(True)
    dashboard.tracking_inspector.uncertainty_toggle.setChecked(True)
    dashboard.association_figure.set_source(Figure)
    dashboard.association_figure.toggle.setChecked(True)
    qtbot.waitUntil(lambda: dashboard.association_figure.height() >= 400)
    assert dashboard.tracking_inspector.height() > before
    assert dashboard.association_figure.canvas.height() >= 400
    assert dashboard.size() == size
