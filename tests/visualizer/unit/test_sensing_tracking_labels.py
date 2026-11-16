"""Tracking refreshes keep final captions without rebuilding their layout."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.visualizer.unit.test_sensing_workspace_shell import _radial_request
from visualizer.src.sensing import sensing_window


@pytest.fixture
def dashboard(qtbot):
    window = sensing_window.SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("tracking")
    window.update_from_request(_radial_request())
    return window


def test_repeated_tracking_results_do_not_relayout_unchanged_text(dashboard, monkeypatch):
    calls = []

    def observe(owner, name, kind):
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    for plot in (dashboard.tracking_plot, dashboard.gt_range_plot, dashboard.gt_speed_plot):
        observe(plot.getPlotItem(), "setTitle", "title")
        for name in ("bottom", "left"):
            observe(plot.getPlotItem().getAxis(name), "setLabel", "axis")
    for plot, scatter in (
        (dashboard.gt_range_plot, dashboard._gt_range_det_scatter),
        (dashboard.gt_speed_plot, dashboard._gt_speed_det_scatter),
    ):
        observe(plot.getPlotItem().legend.getLabel(scatter), "setText", "legend")
    dashboard.update_from_request(_radial_request(5))
    assert calls == []
    assert dashboard.gt_range_plot.getPlotItem().titleLabel.text == "Range estimates"
    assert dashboard.gt_speed_plot.getPlotItem().titleLabel.text == "Radial velocity estimates"


def test_current_tracking_switch_and_no_truth_labels_remain_correct(dashboard):
    request = _radial_request()
    for mode, title in (
        ("current", "Detection range (no persisted truth)"),
        ("tracking", "Range estimates"),
        ("current", "Detection range (no persisted truth)"),
    ):
        dashboard._set_view_mode(mode)
        dashboard.update_from_request(request)
        assert dashboard.gt_range_plot.getPlotItem().titleLabel.text == title


def test_history_absence_still_has_unavailable_titles(dashboard):
    dashboard._render_tracking_history(None, 1.0)
    assert (
        dashboard.gt_range_plot.getPlotItem().titleLabel.text == "GT Range (loading saved history…)"
    )
    assert dashboard.tracking_plot.getPlotItem().titleLabel.text == "Loading saved tracks…"


def test_saved_world_titles_and_legend_use_final_projection_wording(dashboard, monkeypatch):
    request = _radial_request()
    dashboard._last_request = replace(
        request, tracking_history=request.recording_history, saved_tracking=SimpleNamespace()
    )
    # Exercise captions directly without installing a complete world-track model.
    monkeypatch.setattr(dashboard, "_live_plot_needs_update", lambda plot: True)
    dashboard._render_tracking_history(request.recording_history, 1.0)
    assert (
        dashboard.tracking_plot.getPlotItem().titleLabel.text
        == "Saved confirmed 3D tracks · includes coasting"
    )
    assert dashboard.gt_range_plot.getPlotItem().titleLabel.text == "Projected track range"
    label = dashboard.gt_range_plot.getPlotItem().legend.getLabel(dashboard._gt_range_det_scatter)
    assert label.text == "Projected states"


def test_current_truth_visibility_title_updates_and_survives_refresh(dashboard):
    from visualizer.src.sensing.dashboard_data import RecordingHistory

    request = _radial_request()
    truth = {"target_ids": ["actor"], "target_ranges": [10.0], "target_speeds": [2.0]}
    history = RecordingHistory(
        tuple((replace(row, truth=truth) for row in request.recording_history.records))
    )
    request = replace(request, recording_history=history)
    dashboard._set_view_mode("current")
    dashboard._show_gt_cb.setChecked(True)
    dashboard.update_from_request(request)
    assert (
        dashboard.gt_range_plot.getPlotItem().titleLabel.text
        == "Range: detections and ground truth"
    )
    dashboard._show_gt_cb.setChecked(False)
    hidden = "Range: detections and ground truth (ground truth hidden)"
    assert dashboard.gt_range_plot.getPlotItem().titleLabel.text == hidden
    dashboard.update_from_request(request)
    assert dashboard.gt_range_plot.getPlotItem().titleLabel.text == hidden
    dashboard._show_gt_cb.setChecked(True)
    assert (
        dashboard.gt_range_plot.getPlotItem().titleLabel.text
        == "Range: detections and ground truth"
    )
