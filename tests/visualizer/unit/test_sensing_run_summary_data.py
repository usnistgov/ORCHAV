"""Whole-run counts must not inherit the bounded dashboard history window."""

from copy import deepcopy
from threading import Event

import numpy as np
import pytest

from visualizer.src.sensing import run_summary_data as module
from visualizer.src.sensing.dashboard_evaluation import summarize_events

CONTEXT = ("clock", "run", "pair")


def event(index, *, time=None, detections=None, truth=None, reason=None):
    return dict(
        identity=(*CONTEXT, f"a{index}"),
        time_s=float(index if time is None else time),
        truth=(
            truth
            if truth is not None
            else dict(target_ids=["target"], target_ranges=[5.0], target_speeds=[1.0])
        ),
        detections=(
            detections
            if detections is not None
            else [dict(range_m=5.25, velocity_m_s=0.5, track_id=1, track_confirmed=True)]
        ),
        reason=reason,
        tracking_enabled=True,
    )


class Service:
    def __init__(self, events):
        self.events = events
        self.loads = []

    def dashboard_result_rows(self):
        return [(i, self.frame(e)) for i, e in enumerate(self.events)]

    @staticmethod
    def frame(e):
        c, r, p, a = e["identity"]
        return {
            "sensing": {
                "temporal": {
                    "clock": {"clock_id": c},
                    "result": {"processing_run_id": r, "reference_time_s": e["time_s"]},
                    "acquisition": {"pair_id": p, "acquisition_id": a},
                },
                "test_event": e,
            }
        }

    def dashboard_load_row(self, storage_id, products):
        assert products == ("config_resolved",), "must never request dense products"
        self.loads.append(storage_id)
        return self.frame(self.events[storage_id])


@pytest.fixture(autouse=True)
def compact_evidence(monkeypatch):
    monkeypatch.setattr(module, "evaluation_event", lambda s, f: deepcopy(s["test_event"]))


def load(events, **kwargs):
    return module.load_run_summary(
        Service(events), CONTEXT, Event(), range_gate=1.0, velocity_gate=1.0, **kwargs
    )


def test_confirmed_cohort_matches_current_result_scoring_without_mutating_outputs():
    events = [event(0), event(1)]
    events[0]["detections"][0]["track_confirmed"] = False
    original = deepcopy(events)
    expected = summarize_events(events, 1, 1, confirmed_only=True)
    result = load(events, confirmed_only=True)
    for key in ("total_tp", "total_fp", "total_fn", "cumulative_pd", "precision"):
        assert result.summary[key] == expected[key]
    assert result.summary["confirmed_only"] is True
    np.testing.assert_array_equal(result.event_output_count, [0, 1])
    np.testing.assert_array_equal(result.output_time_s, [1])
    assert events == original


def test_confirmed_cohort_missing_confirmation_is_unavailable():
    saved = event(0)
    saved["tracking_enabled"] = False
    del saved["detections"][0]["track_confirmed"]
    result = load([saved], confirmed_only=True)
    assert result.evaluated_count == 0
    assert result.summary["skipped"] == {"Track confirmation was not recorded": 1}


def test_identity_dedup_and_equal_timestamps_and_context():
    first, second = event(0), event(1, time=0)
    other = event(2)
    other["identity"] = ("clock", "other", "pair", "a2")
    result = load([first, first, second, other])
    assert result.acquisition_count == result.evaluated_count == 2
    assert result.summary["total_tp"] == 2
    assert list(result.time_s) == [0, 0]


def test_missing_truth_excluded_empty_truth_false_positive_and_rate_gap():
    events = [
        event(0),
        event(1, reason="Ground truth was not recorded"),
        event(2, truth=dict(target_ids=[], target_ranges=[], target_speeds=[])),
        event(3, detections=[]),
    ]
    result = load(events)
    assert (result.summary["total_tp"], result.summary["total_fp"], result.summary["total_fn"]) == (
        1,
        1,
        1,
    )
    assert result.evaluated_count == 3
    assert result.summary["precision"] == result.summary["recall"] == 0.5
    assert np.isnan(result.event_tp[1]) and np.isnan(result.cumulative_recall[1])
    assert result.cumulative_tp[1] == 1
    assert len(result.output_time_s) == 3  # Missing GT does not erase saved output.
    assert result.summary["skipped"] == {"Ground truth was not recorded": 1}


def test_over_history_cap_scores_every_acquisition_and_bounds_every_plot():
    result = load([event(i) for i in range(4207)], max_plot_points=32)
    assert result.acquisition_count == result.evaluated_count == 4207
    assert result.summary["total_tp"] == 4207
    assert result.cumulative_tp[-1] == 4207
    assert result.summary["cumulative_range_rmse"] == 0.25
    assert result.summary["cumulative_velocity_rmse"] == 0.5
    assert len(result.time_s) == len(result.range_errors_m) == len(result.gt_time_s) == 32
    assert result.plot_sampled
    assert result.time_s[0] == 0 and result.time_s[-1] == 4206


def test_shared_assignment_pooled_denominators_and_signed_error_pairing():
    events = [
        event(0),
        event(1, detections=[]),
        event(2, detections=[dict(range_m=20.0, velocity_m_s=0.0, track_id=3)]),
    ]
    expected = summarize_events(events, 1.0, 1.0)
    result = load(events)
    for key in (
        "total_tp",
        "total_fp",
        "total_fn",
        "cumulative_pd",
        "precision",
        "cumulative_range_rmse",
        "cumulative_velocity_rmse",
    ):
        assert result.summary[key] == expected[key]
    assert list(result.range_errors_m) == [0.25]
    assert list(result.radial_errors_m_s) == [-0.5]
    row = result.summary["per_target"]["target"]
    assert row["eligible_count"] == 3 and row["matched_count"] == 1


def test_tracking_switch_survives_miss_and_confirmation_is_not_filter():
    last = event(2)
    last["detections"][0].update(track_id=2, track_confirmed=False)
    events = [event(0), event(1, detections=[]), last]
    result = load(events)
    assert result.summary["id_switches"] == summarize_events(events, 1.0, 1.0)["id_switches"] == 1
    assert result.summary["total_tp"] == 2
    assert list(result.confirmed_count) == [1, 0, 0]
    assert list(result.tentative_count) == [0, 0, 1]
    assert result.summary["per_target"]["target"]["distinct_assigned_ids"] == 2


@pytest.mark.parametrize("ids", [[True], [None], [float("nan")], [1, 1], ["1"]])
def test_invalid_ids_withhold_identity_claim(ids):
    ds = [dict(range_m=5.0, velocity_m_s=1.0, track_id=i) for i in ids]
    result = load([event(0), event(1, detections=ds)])
    assert result.summary["id_switches"] is None
    assert result.tracking_track_ids == ()
    assert result.summary["per_target"]["target"]["distinct_assigned_ids"] is None


def test_cancellation_does_not_publish_partial_totals():
    flag = Event()
    service = Service([event(i) for i in range(4)])
    original = service.dashboard_load_row

    def cancelling(*args):
        frame = original(*args)
        flag.set()
        return frame

    service.dashboard_load_row = cancelling
    result = module.load_run_summary(service, CONTEXT, flag)
    assert result.cancelled and not result.summary and result.acquisition_count == 0
    assert len(service.loads) == 1


def test_gates_interval_and_reference_labels():
    e = event(1)
    e["truth"]["target_velocity_references"] = ["Root projection (fixed radar)"]
    result = load([event(0), e, event(2)], interval=(1.0, 1.0))
    assert result.acquisition_count == 1
    assert (
        result.summary["per_target"]["target"]["velocity_reference"]
        == "Root projection (fixed radar)"
    )
    result = module.load_run_summary(Service([e]), CONTEXT, Event(), range_gate=0.1)
    assert result.summary["total_tp"] == 0
    assert result.summary["cumulative_range_rmse"] is None
    with pytest.raises(ValueError):
        load([e], interval=(2.0, 1.0))


def test_all_missing_no_truth_is_not_zero_recall():
    result = load([event(0, reason="Ground truth was not recorded")])
    assert result.summary["recall"] is None
    assert result.summary["cumulative_range_rmse"] is None
    assert result.summary["reason"]
    assert result.evaluated_count == 0


def test_real_evaluation_event_preserves_root_velocity_reference(monkeypatch):
    from tests.shared.test_sensing_evaluation_truth import animated_frame
    from visualizer.src.sensing.dashboard_evaluation import evaluation_event

    monkeypatch.setattr(module, "evaluation_event", evaluation_event)
    frame = animated_frame()
    sensing = frame["sensing"]
    sensing["temporal"].update(
        clock={"clock_id": "clock"},
        result={"processing_run_id": "run", "reference_time_s": 0.0},
        truth={"reference_time_s": 0.0},
        display_binding={"scene_frame_id": 0},
    )
    sensing["temporal"]["acquisition"]["acquisition_id"] = "a0"
    sensing["detections"] = [dict(range_m=5.0, velocity_m_s=-1.0, track_id=1)]
    context = ("clock", "run", sensing["temporal"]["acquisition"]["pair_id"])

    class RealService:
        def dashboard_result_rows(self):
            return [(0, frame)]

        def dashboard_load_row(self, storage_id, products):
            assert products == ("config_resolved",)
            return {"sensing": sensing}

        def dashboard_scene_frame(self, scene_id):
            return frame

    result = module.load_run_summary(RealService(), context, Event())
    assert result.summary["total_tp"] == 1
    assert result.summary["cumulative_velocity_rmse"] == 0
    assert (
        result.summary["per_target"]["walker"]["velocity_reference"]
        == "Root projection (fixed radar)"
    )
    assert list(result.gt_radial_m_s) == [-1.0]


def test_equal_time_tracking_preserves_shared_inventory_tie_order():
    events = [event(9, time=0), event(2, time=0), event(1, time=0)]
    for e, track in zip(events, (1, 2, 1), strict=True):
        e["detections"][0]["track_id"] = track
    result = load(events)
    assert result.summary["id_switches"] == summarize_events(events, 1.0, 1.0)["id_switches"] == 2
    assert any("share timestamps" in text for text in result.uncertainty)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_inclusive_clock_roundoff_is_narrow_and_does_not_mutate_events(scale):
    import math

    nominal = [i * 0.064 * scale for i in range(16)]
    times = [math.nextafter(math.nextafter(t, math.inf), math.inf) for t in nominal]
    events = [event(i, time=t) for i, t in enumerate(times)]
    before = deepcopy(events)
    for low in range(16):
        for high in range(low, 16):
            result = load(events, interval=(nominal[low], nominal[high]))
            assert result.acquisition_count == high - low + 1
    assert load(events, interval=(nominal[2], nominal[6])).acquisition_count == 5
    assert load(events, interval=(nominal[2], nominal[6] - 1e-9 * scale)).acquisition_count == 4
    assert events == before
