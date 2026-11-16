"""Independent assignment enumeration and temporal metric regressions."""

import itertools
import math

import numpy as np
import pytest

from shared.sensing.evaluation import (
    _gated_linear_assignment,
    _rmse,
    associate_detections_to_gt,
    compute_mot_metrics,
    evaluate_scenario,
)
from shared.sensing.uav_evaluation import UAVProfile, evaluate_events


def exhaustive_objective(cost, valid):
    """Enumerate every partial injective row-to-column map, without LAP code."""
    best = (0, 0.0)
    for mapping in itertools.product(range(-1, cost.shape[1]), repeat=cost.shape[0]):
        chosen = [j for j in mapping if j >= 0]
        if len(set(chosen)) != len(chosen):
            continue
        if any(j >= 0 and not valid[i, j] for i, j in enumerate(mapping)):
            continue
        score = (-len(chosen), math.fsum(cost[i, j] for i, j in enumerate(mapping) if j >= 0))
        best = min(best, score)
    return best


@pytest.mark.parametrize(
    "shape",
    [
        (0, 3),
        (3, 0),
        (1, 4),
        (4, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    ],
)
def test_every_small_gate_graph_matches_exhaustive_partial_assignments(shape):
    rng = np.random.default_rng(498)
    for mask in range(2 ** math.prod(shape)):
        valid = np.array([(mask >> bit) & 1 for bit in range(math.prod(shape))], bool).reshape(
            shape
        )
        # Discrete costs create exact ties as well as unique minima.
        cost = rng.integers(0, 11, size=shape) / 10
        rows, cols = _gated_linear_assignment(cost, valid)
        expected = exhaustive_objective(cost, valid)
        assert -len(rows) == expected[0]
        assert math.fsum(cost[rows, cols]) == pytest.approx(expected[1], abs=1e-13)
        pr, pc = rng.permutation(shape[0]), rng.permutation(shape[1])
        permuted = cost[np.ix_(pr, pc)]
        rr, cc = _gated_linear_assignment(permuted, valid[np.ix_(pr, pc)])
        assert -len(rr) == expected[0]
        assert math.fsum(permuted[rr, cc]) == pytest.approx(expected[1], abs=1e-13)


def test_old_sentinel_counterexample_and_all_row_column_permutations():
    for gr in ([1, 0.1], [0.1, 1]):
        for dr in ([0, 100], [100, 0]):
            r = associate_detections_to_gt(gr, [0, 0], dr, [0, 0])
            assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 1, 1)
            assert r.range_rmse == pytest.approx(0.1)


def det(r=0, track=1):
    return dict(range_m=r, velocity_m_s=0, track_id=track)


def test_clear_continuity_retains_valid_pairs_despite_smaller_swapped_residual():
    gt = [(["a", "b"], [0, 1], [0, 0])] * 2
    r = compute_mot_metrics(gt, [[det(0, 1), det(1, 2)], [det(0.6, 1), det(0.4, 2)]])
    assert r.mota == 1
    assert r.motp == pytest.approx(0.3)
    assert r.total_id_switches == 0
    assert r.track_fragmentation == {"a": 0, "b": 0}
    assert r.distinct_track_id_count == {"a": 1, "b": 1}
    assert r.metric_protocol == "clear-mot-continuity-v1"


@pytest.mark.parametrize(
    "history, fragments, switches, distinct",
    [
        ([None, 1, 1, None], 0, 0, 1),
        ([1, None, 1, None, 1], 2, 0, 1),
        ([1, 2, 3], 0, 2, 3),
        ([1, None, None, 2], 1, 1, 2),
        ([None, None], 0, 0, 0),
    ],
)
def test_fragmentation_is_internal_miss_runs_not_distinct_ids(
    history, fragments, switches, distinct
):
    r = compute_mot_metrics(
        [(["a"], [0], [0])] * len(history), [[] if h is None else [det(track=h)] for h in history]
    )
    assert r.track_fragmentation == {"a": fragments}
    assert r.total_id_switches == switches
    assert r.distinct_track_id_count == {"a": distinct}


def test_absent_truth_does_not_invent_misses_and_large_ids_remain_exact():
    big = 2**60
    r = compute_mot_metrics(
        [(["a"], [0], [0]), ([], [], []), (["a"], [0], [0])],
        [[det(track=big)], [], [det(track=big + 1)]],
    )
    assert r.track_fragmentation == {"a": 0}
    assert r.total_id_switches == 1
    assert r.total_misses == 0


def test_mostly_lost_boundary_and_empty_metric_denominators():
    r = compute_mot_metrics([(["a"], [0], [0])] * 5, [[det()], [], [], [], []])
    assert r.mostly_lost == 0  # motmetrics uses strictly less than 20%.
    e = compute_mot_metrics([], [])
    assert e.mota is None and e.motp is None and e.track_fragmentation == {}
    s = evaluate_scenario([], [])
    assert s.cumulative_pd is None and s.cumulative_range_rmse is None
    assert s.mean_false_alarms_per_evaluated_event is None


def test_rmse_pools_matches_and_avoids_squaring_overflow():
    r = evaluate_scenario(
        [([0], [0]), ([0, 10, 20], [0, 0, 0])], [([1], [0]), ([0, 10, 20], [0, 0, 0])]
    )
    assert r.cumulative_range_rmse == 0.5
    assert _rmse([1e200, 1e200]) == pytest.approx(1e200)


def test_uav_bootstrap_cannot_split_one_run_into_independent_groups():
    rows = [
        dict(
            run_id="a",
            event_id=str(i),
            time_s=i,
            truth=[],
            outputs=[],
            independent_run_group=str(i),
        )
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="consistent independent"):
        evaluate_events(rows, UAVProfile(bootstrap_samples=100))


def test_uav_missing_truth_and_empty_truth_have_distinct_denominators():
    rows = [
        dict(run_id="a", event_id=str(i), time_s=i, truth=t, outputs=[dict(position_m=[0, 0, 0])])
        for i, t in enumerate([None, []])
    ]
    r = evaluate_events(rows)
    assert r["counts"]["evaluated_epochs"] == 1
    assert r["counts"]["fp"] == 1
    assert r["metrics"]["false_alarm_type_1"]["value"] == 1
    assert r["metrics"]["miss_probability"]["value"] is None
    assert r["skipped"] == {"truth_unavailable_or_unaligned": 1}
