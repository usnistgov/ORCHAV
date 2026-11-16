"""Evaluation denominators, saved-output semantics, and dashboard presentation."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from visualizer.src.sensing.dashboard_data import HistoryRecord, RecordingHistory
from visualizer.src.sensing.dashboard_evaluation import (
    EVALUATION_CACHE_PROTOCOL,
    evaluation_event,
    summarize_events,
)
from visualizer.src.sensing.evaluation_panel import EvaluationPanel


def event(index=0, *, detections=None, truth=None, reason=None):
    return {
        "identity": ("clock", "run", "pair", f"acq{index}"),
        "time_s": float(index),
        "truth": (
            truth
            if truth is not None
            else {"target_ids": ["a", "b"], "target_ranges": [5, 10], "target_speeds": [1, 2]}
        ),
        "detections": (
            detections
            if detections is not None
            else [{"range_m": 5.1, "velocity_m_s": 1}, {"range_m": 20, "velocity_m_s": 0}]
        ),
        "tracking_enabled": False,
        "reason": reason,
    }


def test_counts_use_unique_events_and_explicit_denominators():
    first, second = event(), event(1, detections=[])
    missing = event(2, reason="Ground truth was not recorded")
    before = deepcopy([first, second, missing])
    summary = summarize_events([second, first, first, missing], 1, 1)
    assert (summary["total_tp"], summary["total_fp"], summary["total_fn"]) == (1, 1, 3)
    assert summary["evaluated_event_count"] == 2
    assert summary["target_opportunity_count"] == 4
    assert summary["precision"] == 0.5 and summary["cumulative_pd"] == 0.25
    assert summary["cumulative_range_rmse"] == pytest.approx(0.1)
    assert summary["id_switches"] is None
    assert summary["skipped"] == {"Ground truth was not recorded": 1}
    assert before == [first, second, missing]


def test_empty_truth_is_distinct_from_missing_truth_and_missing_velocity():
    empty = event(truth={"target_ids": [], "target_ranges": [], "target_speeds": []})
    assert summarize_events([empty])["total_fp"] == 2
    assert summarize_events([empty])["cumulative_pd"] is None
    missing = event(reason="Ground truth was not recorded")
    assert "not recorded" in summarize_events([missing])["reason"]
    invalid = event(detections=[{"range_m": 5, "velocity_m_s": None}])
    assert "velocity unavailable" in summarize_events([invalid])["reason"]


def test_truth_requires_result_alignment_and_valid_radial_observables():
    from tests.visualizer.unit.test_sensing_dashboard_recording import _frame

    sensing = _frame()["sensing"]
    sensing.pop("truth", None)
    assert "not recorded" in evaluation_event(sensing)["reason"]
    sensing["truth"] = {"targets": []}
    sensing["temporal"]["truth"] = {"reference_time_s": 100.0}
    assert "not aligned" in evaluation_event(sensing)["reason"]
    sensing["temporal"]["truth"]["reference_time_s"] = sensing["temporal"]["result"][
        "reference_time_s"
    ]
    assert evaluation_event(sensing)["reason"] is None


def test_track_cohort_and_identity_switches_are_explicit():
    truth = {"target_ids": ["a"], "target_ranges": [5], "target_speeds": [1]}
    events = [
        event(
            i,
            truth=truth,
            detections=[
                {"range_m": 5, "velocity_m_s": 1, "track_id": i + 1, "track_confirmed": i > 0}
            ],
        )
        for i in range(2)
    ]
    for e in events:
        e["tracking_enabled"] = True
    assert summarize_events(events)["id_switches"] == 1
    confirmed = summarize_events(events, confirmed_only=True)
    assert (confirmed["total_tp"], confirmed["total_fn"]) == (1, 1)
    assert "not recorded" in summarize_events([event()], confirmed_only=True)["reason"]
    other = deepcopy(events[1])
    other["identity"] = ("clock", "different_run", "pair", "acq2")
    assert "one clock" in summarize_events([events[0], other])["reason"]


def history(events):
    return RecordingHistory(
        tuple(
            HistoryRecord(
                time_s=e["time_s"],
                identity=e["identity"],
                spectrum=None,
                doppler_hz=None,
                range_m=None,
                range_power=None,
                detections=(),
                truth=None,
                truth_positions=(),
                cadence_s=1,
                evaluation=e,
            )
            for e in events
        )
    )


def test_panel_seeking_interval_gates_and_unavailable_reset(qtbot):
    panel = EvaluationPanel()
    qtbot.addWidget(panel)
    first, second = event(), event(1, detections=[])
    records = history([first, second])
    panel.range_gate.setValue(1)
    panel.velocity_gate.setValue(1)
    panel.toggle.setChecked(True)
    panel.update_event(first, records)
    assert "TP 1 · FP 1 · FN 1" in panel.current.text()
    expected = panel.aggregate.text()
    for e in [second, first, second, first]:
        panel.update_event(e, records)
        assert panel.aggregate.text() == expected
    panel.interval.setChecked(True)
    panel.end.setValue(0)
    assert "1 evaluated results" in panel.aggregate.text()
    panel.range_gate.setValue(0.01)
    assert "TP 0 · FP 2 · FN 2" in panel.current.text()
    panel.update_event(None)
    assert "Unavailable" in panel.current.text()


def test_live_match_plot_and_counts_share_gates_without_changing_detections(qtbot):
    import numpy as np

    panel = EvaluationPanel()
    qtbot.addWidget(panel)
    first, second = event(), event(1, detections=[])
    saved = deepcopy([first, second])
    records = history([first, second])
    panel.toggle.setChecked(True)
    panel.range_gate.setValue(1)
    panel.velocity_gate.setValue(1)
    panel.update_event(first, records)
    assert len(panel.plots.markers["tp"].getData()[0]) == 1
    assert len(panel.plots.markers["fp"].getData()[0]) == 1
    assert len(panel.plots.markers["fn"].getData()[0]) == 1
    np.testing.assert_array_equal(panel.plots.bars.opts["height"], [1, 1, 3])
    panel.range_gate.setValue(0.05)
    assert panel.plots.markers["tp"].getData()[0] is None
    assert len(panel.plots.markers["fp"].getData()[0]) == 2
    assert len(panel.plots.markers["fn"].getData()[0]) == 2
    np.testing.assert_array_equal(panel.plots.bars.opts["height"], [0, 2, 4])
    x, _ = panel.plots.gates.getData()
    assert x[1] - x[0] == pytest.approx(0.1)
    panel.interval.setChecked(True)
    panel.end.setValue(0)
    np.testing.assert_array_equal(panel.plots.bars.opts["height"], [0, 2, 2])
    assert saved == [first, second]
    panel.update_event(event(reason="Ground truth was not recorded"), history([]))
    assert "unavailable" in panel.plots.matches.getPlotItem().titleLabel.text
    assert "unavailable" in panel.plots.counts.getPlotItem().titleLabel.text
    assert not any(label.toPlainText() for label in panel.plots.count_labels)
    for item in panel.plots.markers.values():
        assert item.getData()[0] is None


def test_controls_are_contextual_and_display_toggles_do_not_change_scoring(qtbot):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _request
    from visualizer.src.sensing.sensing_window import SensingWindow

    window = SensingWindow()
    window.show()
    window._advanced_widget.show()
    request = _request()
    window.update_from_request(request)
    window._set_view_mode("analysis")
    assert window._micro_mode_box.isHidden()
    assert not window._rd_dynamic_range_box.isHidden()
    assert not window._variant_box.model().item(window._variant_box.findData("raw")).isEnabled()
    metrics = window.evaluation_panel.current.text()
    window._show_gt_cb.setChecked(not window._show_gt_cb.isChecked())
    window._rd_dynamic_range_box.setValue(70)
    window.update_from_request(replace(request, variant="raw"))
    assert window.evaluation_panel.current.text() == metrics
    window._set_view_mode("micro-doppler")
    assert not window._micro_mode_box.isHidden()
    window._set_view_mode("diagnostics")
    assert window.acquisition_timeline_plot.isHidden()
    window.timing_toggle.setChecked(True)
    assert not window.acquisition_timeline_plot.isHidden()
    window.set_unavailable("No result", scene_frame_id=10)
    assert "Unavailable" in window.evaluation_panel.current.text()
    window.close()


def test_stage_absence_is_not_a_loading_failure():
    from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
    from visualizer.src.sensing.dashboard_diagnostics import (
        cell_explanation,
        stage_unavailable_text,
    )
    from visualizer.src.sensing.stage_inspector import (
        DETECTION_STAGE_KINDS,
        build_detection_stage_bundle,
    )

    sensing = _typed_sensing()
    bundle = build_detection_stage_bundle(sensing)
    from visualizer.src.sensing.stage_inspector import CellSelection

    cell = bundle.inspect(CellSelection(bundle.event.pair_id, 1, 2))
    assert "input power" in cell_explanation(cell)
    for key, descriptor in list(sensing["products"].items()):
        if descriptor["kind"] in DETECTION_STAGE_KINDS:
            sensing.pop(key, None)
            del sensing["products"][key]
    missing = build_detection_stage_bundle(sensing)
    assert missing.reason == "detector_stages_not_recorded"
    assert "were not recorded" in stage_unavailable_text(missing, missing.reason)


def test_workspaces_group_signal_details_and_evaluation_without_instruction_text(qtbot):
    from PySide6.QtCore import QSignalBlocker

    from tests.visualizer.unit.test_sensing_dashboard_recording import _request
    from visualizer.src.sensing.sensing_window import SensingWindow

    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(1100, 700)
    window.show()
    window._set_view_mode("analysis")
    window.update_from_request(_request())
    assert [window.workspace_box.itemText(i) for i in range(window.workspace_box.count())] == [
        "Run summary",
        "Current result",
        "Motion",
        "Tracking",
    ]
    assert window.workspace_box.currentData() == "current"
    assert [window.workspace_tabs.tabText(i) for i in range(window.workspace_tabs.count())] == [
        "Detections",
        "Detector",
        "Angle",
        "Inputs",
    ]
    assert window._support_box.isHidden() and window._view_mode_box.isHidden()
    assert window.view_description.isHidden()
    assert all(p.isVisible() for p in (window.rp_plot, window.gt_range_plot, window.gt_speed_plot))
    assert not window.workspace_tabs.isTabEnabled(2)  # This fixture has no angular product.

    window._set_view_mode("micro-doppler")
    with QSignalBlocker(window.figure_options):
        window.figure_options.expand()
    assert window.workspace_box.currentData() == "micro-doppler"
    assert window.motion_controls.isHidden()
    window._advanced_motion_cb.setChecked(True)
    assert window.motion_controls.isVisible()
    assert window._micro_gate_box.isHidden()
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("gated"))
    assert window._micro_gate_box.isVisible()

    window.workspace_box.setCurrentIndex(window.workspace_box.findData("summary"))
    assert window.workspace_tabs.count() == 1
    qtbot.wait(10)
    assert window._selected_view_mode() == "summary"
    assert window.run_summary_panel.isVisible()
    assert not window.evaluation_panel.isVisible()
    assert window.rd_plot.isHidden() and window._support_box.isHidden()
    assert window.view_description.isHidden()
    window._set_view_mode("analysis")
    assert not window.evaluation_panel.isVisible()


def test_detection_histories_work_without_tracking_and_clear_while_loading(qtbot):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
    from visualizer.src.sensing.sensing_window import SensingWindow

    records = _history()
    records = replace(
        records,
        records=tuple(
            replace(r, detections=({"range_m": 3 + r.time_s, "velocity_m_s": 1},))
            for r in records.records
        ),
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("analysis")
    window._initial_view_selected = True
    assert window.wants_recording_history()
    for plot in (window.gt_range_plot, window.gt_speed_plot):
        for step in (5, 1, 4, 0):
            window.update_from_request(_request(step, records))
            assert plot.isVisible()
            assert len(window._gt_range_det_scatter.getData()[0]) == step + 1
            assert max(window._gt_range_det_scatter.getData()[0]) == step * 0.25
    window.update_from_request(_request(0))
    assert len(window._gt_range_det_scatter.getData()[0]) == 0
    assert "loading" in window.gt_speed_plot.getPlotItem().titleLabel.text
    window.close()


@pytest.mark.parametrize("mode", ["analysis", "tracking"])
def test_history_zoom_uses_full_recording_when_disabled_and_height_is_adjustable(qtbot, mode):
    from PySide6.QtCore import QSignalBlocker

    from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
    from visualizer.src.sensing.sensing_window import SensingWindow

    records = _history()
    records = replace(
        records,
        records=tuple(
            replace(
                record,
                detections=(
                    {
                        "range_m": 2 + i,
                        "velocity_m_s": 0.5 * i,
                        "track_id": 2,
                        "world_track_position": (10.0 * i, float(i), 0.0),
                    },
                ),
            )
            for i, record in enumerate(records.records)
        ),
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode(mode)
    with QSignalBlocker(window.figure_options):
        window.figure_options.expand()
    window._zoom_data_cb.setChecked(False)
    assert window._zoom_data_cb.isVisible()
    for step in (1, 3, 0):
        window.update_from_request(_request(step, records))
        x, y = window.gt_range_plot.viewRange()
        assert x[0] < 0 and x[1] > 1.25
        assert y[0] < 2 and y[1] > 7
        assert len(window._gt_range_det_scatter.getData()[0]) == step + 1
        if mode == "tracking":
            assert window.tracking_plot.viewRange()[0][1] >= 50
            assert len(window._tracking_curves["2"].getData()[0]) == step + 1
    window.update_from_request(_request(1, records))
    window._zoom_data_cb.setChecked(True)
    assert window.gt_range_plot.viewRange()[0][1] < 0.3
    assert window.gt_range_plot.viewRange()[1][1] < 3.1
    if mode == "tracking":
        # Check the actual XY view after Qt settles, not only the time histories.
        qtbot.waitUntil(lambda: window.tracking_plot.viewRange()[0][1] < 20)
        assert len(window._tracking_curves["2"].getData()[0]) == 2
    original = window.gt_range_plot.viewRange()
    assert window.figure_options.isVisible()
    window._graph_height_box.setValue(420)
    qtbot.waitUntil(lambda: window.gt_range_plot.getViewBox().height() >= 420)
    np.testing.assert_allclose(window.gt_range_plot.viewRange(), original)
    if mode == "tracking":
        assert window.tracking_plot.viewRange()[0][1] < 20


def test_threshold_slice_uses_saved_detector_values_and_selected_doppler(qtbot):
    import numpy as np

    from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
    from tests.visualizer.unit.test_sensing_stage_dashboard import (
        _replay_event_for,
        _request,
        _with_detection,
    )
    from visualizer.src.sensing.sensing_window import SensingWindow
    from visualizer.src.sensing.stage_inspector import build_detection_stage_bundle

    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    assert window.cfar_rp_plot.isVisible()
    window.stage_comparison_scope.setCurrentIndex(window.stage_comparison_scope.findData("slice"))
    window._select_inspection_coordinate(1, 2)
    for row in (2, 0, 1):
        window._detection_stage_doppler_box.setValue(row)
        x, y = window.cfar_rp_thresh.getData()
        np.testing.assert_array_equal(x, bundle.range_axis_m)
        expected = sensing["range_doppler_cfar_threshold_power_by_pair"][0, row]
        np.testing.assert_allclose(y, 10 * np.log10(np.maximum(expected, 1e-30)))
    window.update_from_request(_request(sensing, None, None))
    assert not window.cfar_rp_plot.isVisible()
    assert not window.cfar_rd_plot.isVisible()
    window.close()


def test_per_target_scores_use_global_assignment_and_keep_misses():
    shared = event(
        truth={"target_ids": ["a", "b"], "target_ranges": [5, 5.5], "target_speeds": [1, 1]},
        detections=[{"range_m": 5.1, "velocity_m_s": 1.2}],
    )
    summary = summarize_events([shared, event(1, detections=[])])
    a, b = summary["per_target"]["a"], summary["per_target"]["b"]
    assert (a["matched"], a["eligible"], a["missed"], a["coverage"]) == (1, 2, 1, 0.5)
    assert a["range_rmse"] == pytest.approx(0.1)
    assert a["velocity_rmse"] == pytest.approx(0.2)
    assert b["matched"] == 0 and b["missed"] == 2 and b["range_rmse"] is None
    assert sum(r["matched"] for r in summary["per_target"].values()) == summary["total_tp"]
    assert sum(r["missed"] for r in summary["per_target"].values()) == summary["total_fn"]


def test_defaults_reject_outside_either_tolerance_and_include_boundary():
    for r, v, expected in [(7, 3, 1), (7.001, 3, 0), (7, 3.001, 0), (3, -1, 1)]:
        evidence = event(
            truth={"target_ids": ["a"], "target_ranges": [5], "target_speeds": [1]},
            detections=[{"range_m": r, "velocity_m_s": v}],
        )
        assert summarize_events([evidence])["total_tp"] == expected


def test_prepared_request_is_reused_but_replacing_frame_invalidates_it(monkeypatch):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _request
    from visualizer.src.sensing import dashboard_evaluation

    request = _request()
    evidence = request.evaluation
    prepared = replace(request, prepared_evaluation=(request.display_frame, evidence))
    with monkeypatch.context() as patch:
        patch.setattr(
            dashboard_evaluation, "evaluation_event", lambda *args: pytest.fail("Recomputed")
        )
        assert replace(prepared, recording_history=history([])).evaluation is evidence
    new_frame = deepcopy(request.display_frame)
    new_frame["sensing"]["temporal"]["result"]["processing_run_id"] = "new-run"
    assert replace(prepared, display_frame=new_frame).evaluation["identity"][1] == "new-run"


def test_per_target_table_defaults_and_prepared_summary_survive_reopen(qtbot, monkeypatch):
    first = event()
    first["evaluation_protocol"] = EVALUATION_CACHE_PROTOCOL
    first["default_summary"] = summarize_events([first])
    recording = history([first, event(1, detections=[])])
    _ = recording.evaluation_summary
    from visualizer.src.sensing import evaluation_panel

    original = evaluation_panel.summarize_events

    def score(events, *args, **kwargs):
        if any(e is not None for e in events):
            pytest.fail("Recomputed default scores")
        return original(events, *args, **kwargs)

    monkeypatch.setattr(evaluation_panel, "summarize_events", score)
    for _ in range(2):
        panel = EvaluationPanel()
        qtbot.addWidget(panel)
        assert panel.range_gate.value() == panel.velocity_gate.value() == 2
        panel.update_event(first, recording)
        panel.toggle.setChecked(True)
        table = panel.plots.targets
        assert table.rowCount() == 2
        assert table.item(0, 1).text() == "1 / 2"
        assert table.item(0, 2).text() == "50.0%"
        assert table.item(1, 4).text() == "—"


@pytest.mark.parametrize("protocol", [None, "previous-evaluation-protocol"])
def test_missing_or_old_protocol_recomputes_prepared_summary(qtbot, monkeypatch, protocol):
    first = event()
    stale = event(detections=[{"range_m": 6.0, "velocity_m_s": 1.0}])
    first["default_summary"] = summarize_events([stale])
    assert first["default_summary"]["cumulative_range_rmse"] == pytest.approx(1.0)
    if protocol is not None:
        first["evaluation_protocol"] = protocol

    panel = EvaluationPanel()
    qtbot.addWidget(panel)
    from visualizer.src.sensing import evaluation_panel

    original = evaluation_panel.summarize_events
    recomputed = []

    def score(events, *args, **kwargs):
        result = original(events, *args, **kwargs)
        recomputed.append(result)
        return result

    monkeypatch.setattr(evaluation_panel, "summarize_events", score)
    panel.update_event(first, None)

    assert len(recomputed) == 1
    assert recomputed[0]["cumulative_range_rmse"] == pytest.approx(0.1)
    assert (recomputed[0]["total_tp"], recomputed[0]["total_fp"], recomputed[0]["total_fn"]) == (
        1,
        1,
        1,
    )
    assert first["default_summary"]["cumulative_range_rmse"] == pytest.approx(1.0)


def test_rd_markers_evaluation_and_reference_diagnostic_agree(qtbot):
    from tests.shared.test_sensing_evaluation_truth import animated_frame
    from tests.visualizer.unit.test_sensing_dashboard_recording import _frame
    from visualizer.src.sensing.sensing_window import SensingWindow

    frame = animated_frame()
    sensing = frame["sensing"]
    base = _frame()["sensing"]
    sensing["config_resolved"] = {**base["config_resolved"], **sensing["config_resolved"]}
    sensing["temporal"] = {
        **base["temporal"],
        "acquisition": {**base["temporal"]["acquisition"], **sensing["temporal"]["acquisition"]},
    }
    sensing["temporal"]["truth"] = {
        "reference_time_s": sensing["temporal"]["result"]["reference_time_s"]
    }
    sensing["detections"] = [{"range_m": 5, "velocity_m_s": 0.75}]
    # Root projection is selected even when the saved-center value fits better.
    evidence = evaluation_event(sensing, frame)
    assert evidence["default_summary"]["per_target"]["walker"]["velocity_rmse"] == 1.75
    window = SensingWindow()
    qtbot.addWidget(window)
    window._update_ground_truth(frame)
    ranges, speeds = window._current_gt_range_speed()
    assert list(ranges) == [5] and list(speeds) == [-1]
    window.evaluation_panel.plots.update_references(evidence)
    diagnostic = window.evaluation_panel.plots.reference.text()
    assert "-1.000 m/s" in diagnostic and "recorded center 0.750 m/s" in diagnostic


def test_prepared_assignments_round_trip_without_reassociation(monkeypatch):
    import json

    from tests.shared.test_sensing_evaluation_truth import animated_frame
    from visualizer.src.sensing import dashboard_evaluation as scoring

    frame = animated_frame()
    sensing = frame["sensing"]
    sensing["temporal"]["clock"] = {"clock_id": "clock"}
    sensing["temporal"]["result"] = {"processing_run_id": "run", "reference_time_s": 0}
    sensing["temporal"]["truth"] = {"reference_time_s": 0}
    sensing["temporal"]["acquisition"]["acquisition_id"] = "acq"
    sensing["detections"] = [{"range_m": 5.1, "velocity_m_s": -1}]
    saved = json.loads(json.dumps(evaluation_event(sensing, frame)))
    monkeypatch.setattr(
        scoring, "associate_detections_to_gt", lambda *a, **kw: pytest.fail("Reassociated")
    )
    _, _, match = scoring.match_event(saved, 2, 2)
    assert match.gt_to_det == {0: 0}
    assert match.true_positives == 1
