"""Matching choices are shared without merging each view's time scope."""

from copy import deepcopy

import pytest

from tests.visualizer.unit.test_sensing_dashboard_evaluation import event, history
from visualizer.src.sensing.evaluation_panel import EvaluationPanel
from visualizer.src.sensing.evaluation_settings import EvaluationSettings
from visualizer.src.sensing.run_summary_panel import RunSummaryPanel


def test_committed_gates_and_cohort_sync_both_ways_with_local_intervals(qtbot):
    settings = EvaluationSettings()
    current, summary = EvaluationPanel(), RunSummaryPanel()
    for panel in (current, summary):
        qtbot.addWidget(panel)
        panel.bind_evaluation_settings(settings)
    first = event(
        detections=[
            {"range_m": 5.1, "velocity_m_s": 1, "track_id": 1, "track_confirmed": True},
            {"range_m": 20, "velocity_m_s": 0, "track_id": 2, "track_confirmed": False},
        ]
    )
    first["tracking_enabled"] = True
    original = deepcopy(first)
    current.update_event(first, history([first]))
    current.interval.setChecked(True)
    current.start.setValue(0)
    current.end.setValue(0)

    current.range_gate.setValue(1)
    assert summary.range_gate.value() == summary.settings()[0] == 1
    assert "FP 1" in current.current.text()

    summary.range_gate.setValue(0.5)
    summary.outputs.setCurrentIndex(1)
    summary.interval.setChecked(True)
    summary.start.setValue(10)
    summary.end.setValue(20)
    assert settings.range_gate_m == 1  # Run settings are committed with Apply.
    summary.apply.click()
    assert current.range_gate.value() == 0.5
    assert current.outputs.currentData() is True
    assert "FP 0" in current.current.text()
    assert current.current_summary()["total_fp"] == 0
    assert summary.settings() == (0.5, settings.velocity_gate_m_s, (10, 20), True)
    assert (current.start.value(), current.end.value()) == (0, 0)

    current.velocity_gate.setValue(0.25)
    assert summary.velocity_gate.value() == 0.25
    assert summary.settings()[2] == (10, 20)
    assert first == original


def test_settings_rebind_disconnects_retired_model_and_emits_once(qtbot):
    first, second = EvaluationSettings(), EvaluationSettings()
    panel = EvaluationPanel()
    qtbot.addWidget(panel)
    panel.bind_evaluation_settings(first)
    panel.bind_evaluation_settings(second)
    changes = []
    second.changed.connect(lambda: changes.append(second.range_gate_m))
    first.set_values(range_gate_m=8, velocity_gate_m_s=9, confirmed_only=True)
    assert panel.range_gate.value() == second.range_gate_m
    panel.range_gate.setValue(3)
    assert changes == [3]
    second.set_values(
        range_gate_m=3, velocity_gate_m_s=second.velocity_gate_m_s, confirmed_only=False
    )
    assert changes == [3]
    with pytest.raises(ValueError, match="finite and positive"):
        second.set_values(range_gate_m=float("nan"), velocity_gate_m_s=1, confirmed_only=False)
    assert second.range_gate_m == 3


def test_untouched_time_defaults_preserve_exact_history_endpoints(qtbot):
    current = EvaluationPanel()
    qtbot.addWidget(current)
    records = [event(0), event(1)]
    records[0]["time_s"], records[1]["time_s"] = 1 / 3, 2 / 3
    current.toggle.setChecked(True)
    current.update_event(records[0], history(records))
    current.interval.setChecked(True)
    assert current.start.value() == 1 / 3 and current.end.value() == 2 / 3
    assert "2 evaluated results" in current.aggregate.text()
    # Explicitly typing the rounded shorter endpoint must replace the exact default.
    current.end.lineEdit().selectAll()
    qtbot.keyClicks(current.end.lineEdit(), "0.666666666")
    current.end.interpretText()
    current._refresh()
    assert "1 evaluated results" in current.aggregate.text()


def test_run_summary_six_decimal_bounds_keep_unedited_exact_values(qtbot):
    panel = RunSummaryPanel()
    qtbot.addWidget(panel)
    panel.start.setValue(1 / 3)
    panel.end.setValue(2 / 3)
    panel.interval.setChecked(True)
    panel.apply.click()
    assert panel.settings()[2] == (1 / 3, 2 / 3)
    panel.end.lineEdit().selectAll()
    qtbot.keyClicks(panel.end.lineEdit(), "0.666666")
    panel.end.interpretText()
    panel.apply.click()
    assert panel.settings()[2] == (1 / 3, 0.666666)
