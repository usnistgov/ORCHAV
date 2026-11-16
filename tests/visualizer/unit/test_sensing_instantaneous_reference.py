"""UI scores expose exact references and missing-velocity denominators."""

from copy import deepcopy
from threading import Event

import numpy as np
import pytest

from shared.sensing.evaluation import ASSOCIATION_PROTOCOL, MOT_PROTOCOL
from shared.sensing.evaluation_truth import extract_evaluation_truth
from tests.shared.test_sensing_instantaneous_evaluation import interval_payload, reference_payload
from tests.visualizer.unit.test_sensing_run_summary_data import CONTEXT, Service, event
from visualizer.src.sensing import run_summary_data
from visualizer.src.sensing.dashboard_evaluation import (
    evaluation_cache_current,
    evaluation_event,
    summarize_events,
)
from visualizer.src.sensing.evaluation_panel import EvaluationPanel


def test_range_only_scoring_has_no_invented_velocity_marker_and_old_cache_is_invalid(qtbot):
    sensing = interval_payload()
    sensing["detections"] = [{"range_m": 5, "velocity_m_s": -100}]
    before = deepcopy(sensing)
    saved = evaluation_event(sensing)
    assert saved["reason"] is None
    score = summarize_events([saved])
    assert score["total_tp"] == 1
    assert score["cumulative_range_rmse"] == 0
    assert score["cumulative_velocity_rmse"] is None
    assert score["velocity_excluded_matched_pair_count"] == 1
    assert not evaluation_cache_current(
        {"evaluation_protocol": f"{ASSOCIATION_PROTOCOL}/{MOT_PROTOCOL}"}
    )
    panel = EvaluationPanel()
    qtbot.addWidget(panel)
    panel.update_event(saved, None)
    panel.toggle.setChecked(True)
    assert "radial pairs 0/1" in panel.current.text()
    assert "instantaneous velocity unavailable" in panel.plots.reference.text()
    assert panel.plots.markers["gt"].getData()[0] is None
    assert panel.plots.gates.getData()[0] is None
    assert len(panel.plots.markers["tp"].getData()[0]) == 1
    assert sensing == before


def test_whole_run_velocity_rmse_uses_supported_matches_only(monkeypatch):
    monkeypatch.setattr(
        run_summary_data, "evaluation_event", lambda s, f: deepcopy(s["test_event"])
    )
    partial = event(
        0,
        truth=extract_evaluation_truth(interval_payload()),
        detections=[dict(range_m=5.1, velocity_m_s=100, track_id=1)],
    )
    complete = event(
        1,
        truth=extract_evaluation_truth(reference_payload()),
        detections=[dict(range_m=5, velocity_m_s=2.2, track_id=1)],
    )
    result = run_summary_data.load_run_summary(Service([partial, complete]), CONTEXT, Event())
    score = result.summary
    assert score["total_tp"] == 2 and score["evaluated_event_count"] == 2
    assert score["cumulative_range_rmse"] == pytest.approx(np.sqrt(0.01 / 2))
    assert score["cumulative_velocity_rmse"] == pytest.approx(0.2)
    assert score["velocity_matched_pair_count"] == 1
    assert score["velocity_excluded_matched_pair_count"] == 1
    assert score["per_target"]["cube"]["velocity_matched"] == 1
    assert score["id_switches"] is None
    assert np.count_nonzero(np.isfinite(result.radial_errors_m_s)) == 1
