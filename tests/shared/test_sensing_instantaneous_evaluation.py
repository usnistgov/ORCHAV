"""Exact temporal/kinematic references and honest missing-velocity denominators."""

import json
from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.evaluation import (
    INSTANTANEOUS_REFERENCE_PROTOCOL,
    LEGACY_REFERENCE_PROTOCOL,
    associate_detections_to_gt,
    evaluate_scenario,
)
from shared.sensing.evaluation_truth import (
    INTERVAL_REFERENCE,
    ROOT_RELATIVE_REFERENCE,
    evaluation_velocity_mask,
    extract_evaluation_truth,
    extract_recorded_evaluation_truth,
)
from shared.sensing.gt import extract_stored_gt_entry
from shared.sensing.targets import INSTANTANEOUS_VELOCITY_REFERENCE_POLICY
from tests.shared.test_sensing_evaluation_truth import animated_frame
from tests.shared.test_sensing_gt import PAIR_ID, _active, _payload


def reference_payload(
    *,
    epoch=0.128,
    tx=(0, 0, 0),
    rx=(0, 0, 0),
    position=(5, 0, 0),
    tx_velocity=(0, 0, 0),
    rx_velocity=(0, 0, 0),
    target_velocity=(-2, 0, 0),
):
    """Independent relative path derivative with a complete persisted binding."""
    tx, rx, position = (np.asarray(v, dtype=float) for v in (tx, rx, position))
    tv, rv, velocity = (
        np.asarray(v, dtype=float) for v in (tx_velocity, rx_velocity, target_velocity)
    )
    u_tx = (position - tx) / np.linalg.norm(position - tx)
    u_rx = (position - rx) / np.linalg.norm(position - rx)
    expected = -0.5 * (np.dot(u_tx, velocity - tv) + np.dot(u_rx, velocity - rv))
    target = _active(
        "cube",
        path_m=float(np.linalg.norm(position - tx) + np.linalg.norm(position - rx)),
        velocity_m_s=float(expected),
    )
    target["position_m"] = position.tolist()
    sensing = _payload(targets=[target])
    sensing["truth"]["velocity_reference_policy"] = INSTANTANEOUS_VELOCITY_REFERENCE_POLICY
    sensing["temporal"].update(
        clock={"clock_id": "clock"},
        result={"reference_time_s": epoch, "processing_run_id": "run"},
        truth={"reference_time_s": epoch},
    )
    sensing["temporal"]["acquisition"].update(acquisition_id="acquisition")
    target["pair_observables"][0]["velocity_reference_evidence"] = {
        "schema_version": 1,
        "policy": INSTANTANEOUS_VELOCITY_REFERENCE_POLICY,
        "pair_id": PAIR_ID,
        "target_id": "cube",
        "evaluation_epoch_s": epoch,
        "source": "analytic_velocity_projection",
        "instantaneous_eligible": True,
        "support": {
            "kind": "instantaneous",
            "reference_time_s": epoch,
            "start_time_s": epoch,
            "end_time_s": epoch,
        },
        "tx_position_m": tx.tolist(),
        "rx_position_m": rx.tolist(),
        "target_position_m": position.tolist(),
        "tx_velocity_m_s": tv.tolist(),
        "rx_velocity_m_s": rv.tolist(),
        "target_velocity_m_s": velocity.tolist(),
        "fallback_reason": None,
    }
    return sensing


def interval_payload():
    sensing = reference_payload()
    obs = sensing["truth"]["targets"][0]["pair_observables"][0]
    obs.update(
        monostatic_equivalent_velocity_m_s=0.6538604244203522,
        velocity_derivation_source="exact_epoch_finite_difference",
        velocity_reference_dt_s=0.064,
    )
    evidence = obs["velocity_reference_evidence"]
    evidence.update(
        source="exact_epoch_finite_difference",
        instantaneous_eligible=False,
        tx_velocity_m_s=None,
        fallback_reason="tx_velocity_unavailable",
    )
    evidence["support"] = {
        "kind": "interval_mean",
        "reference_time_s": 0.096,
        "start_time_s": 0.064,
        "end_time_s": 0.128,
    }
    return sensing


def test_standard_rd_markers_use_selected_reference_and_omit_unavailable_velocity():
    from shared.sensing.plotting.viz import _canonical_truth_points

    assert _canonical_truth_points(reference_payload()) == [
        {"target_id": "cube", "range_m": 5.0, "velocity_m_s": 2.0}
    ]
    assert _canonical_truth_points(interval_payload()) == []
    unaligned = reference_payload()
    unaligned["temporal"]["result"]["reference_time_s"] += 0.001
    assert _canonical_truth_points(unaligned) == []


def test_accelerating_exact_instantaneous_estimate_has_zero_error_and_legacy_stays_interval():
    epoch = 0.128
    position = 5 - 0.15 * np.sin(2 * np.pi * 2 * epoch)
    target_velocity = -0.15 * 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * epoch)
    perfect = -target_velocity
    sensing = reference_payload(position=(position, 0, 0), target_velocity=(target_velocity, 0, 0))
    truth = extract_evaluation_truth(sensing)
    score = associate_detections_to_gt(
        truth["target_ranges"],
        truth["target_speeds"],
        [position],
        [perfect],
        gt_velocity_valid=evaluation_velocity_mask(truth),
    )
    assert score.velocity_rmse == pytest.approx(0, abs=1e-15)
    assert score.evaluation_reference_protocol == INSTANTANEOUS_REFERENCE_PROTOCOL
    legacy = interval_payload()
    legacy["truth"].pop("velocity_reference_policy")
    legacy["truth"]["targets"][0]["pair_observables"][0].pop("velocity_reference_evidence")
    old = extract_evaluation_truth(legacy)
    assert evaluation_velocity_mask(old) is None
    assert old["target_speeds"] == [0.6538604244203522]
    assert old["evaluation_reference_protocol"] == LEGACY_REFERENCE_PROTOCOL


def test_moving_monostatic_sensors_use_bound_snapshots_not_carrier_frame():
    sensing = reference_payload(
        target_velocity=(0, 0, 0), tx_velocity=(2, 0, 0), rx_velocity=(2, 0, 0)
    )
    before = deepcopy(sensing)
    truth = extract_evaluation_truth(
        sensing, {"tx_positions": [[100, 0, 0]], "rx_positions": [[200, 0, 0]]}
    )
    assert truth["target_speeds"] == [2.0]
    assert sensing == before
    assert extract_evaluation_truth(json.loads(json.dumps(sensing)))["target_speeds"] == [2.0]


def test_general_moving_bistatic_reference_matches_independent_centered_path_derivative():
    tx, rx, target = (np.array(v, dtype=float) for v in [(0, 0, 0), (1, 2, 0), (5, 6, 1)])
    tv, rv, velocity = (
        np.array(v, dtype=float) for v in [(0.1, -0.3, 0), (-0.2, 0.4, 0.1), (-1, 0.5, 0.2)]
    )
    sensing = reference_payload(
        tx=tx, rx=rx, position=target, tx_velocity=tv, rx_velocity=rv, target_velocity=velocity
    )
    sensing["config_resolved"]["range_mode"] = "bistatic"

    def length(t):
        return np.linalg.norm(target + velocity * t - tx - tv * t) + np.linalg.norm(
            target + velocity * t - rx - rv * t
        )

    h = 1e-5
    truth = extract_evaluation_truth(sensing)
    assert truth["target_speeds"][0] == pytest.approx(
        -(length(h) - length(-h)) / (4 * h), abs=2e-10
    )
    assert truth["target_ranges"] == pytest.approx([length(0)])
    common = reference_payload(
        tx_velocity=(1, 2, 3), rx_velocity=(1, 2, 3), target_velocity=(1, 2, 3)
    )
    assert extract_evaluation_truth(common)["target_speeds"] == [0]


def test_interval_value_remains_recorded_but_is_not_an_instantaneous_error():
    sensing = interval_payload()
    truth = extract_evaluation_truth(sensing)
    assert truth["recorded_target_speeds"] == [0.6538604244203522]
    assert truth["target_speeds"] == [None]
    assert truth["target_velocity_references"] == [INTERVAL_REFERENCE]
    assert evaluation_velocity_mask(truth) == [False]
    score = associate_detections_to_gt(
        truth["target_ranges"],
        truth["target_speeds"],
        [5],
        [-100],
        gt_velocity_valid=evaluation_velocity_mask(truth),
    )
    assert (score.true_positives, score.false_positives, score.false_negatives) == (1, 0, 0)
    assert score.range_rmse == 0
    assert score.velocity_rmse is None
    assert score.velocity_matched_pair_count == 0
    assert score.velocity_excluded_matched_pair_count == 1
    assert score.velocity_unavailable_opportunity_count == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("pair_id", "b" * 64),
        ("target_id", "other"),
        ("evaluation_epoch_s", 0.127),
        ("tx_velocity_m_s", None),
        ("rx_velocity_m_s", None),
        ("target_velocity_m_s", None),
    ],
)
def test_bad_instantaneous_evidence_is_explicit_failure_not_assumed_stationary(field, value):
    sensing = reference_payload()
    sensing["truth"]["targets"][0]["pair_observables"][0]["velocity_reference_evidence"][
        field
    ] = value
    with pytest.raises(ValueError):
        extract_evaluation_truth(sensing)


def test_epoch_misalignment_is_rejected_and_loader_retains_explicit_unavailable_truth():
    sensing = reference_payload()
    sensing["temporal"]["result"]["reference_time_s"] = 0.129
    with pytest.raises(ValueError, match="exact result reference epoch"):
        extract_evaluation_truth(sensing)
    stored = extract_recorded_evaluation_truth(sensing)
    assert stored["target_ids"] == ["cube"]
    assert stored["_evaluation_unavailable_reason"]
    sensing["temporal"].pop("result")
    assert extract_recorded_evaluation_truth(sensing) == extract_stored_gt_entry(sensing)


@pytest.mark.parametrize("result_epoch", [False, True, float("nan"), float("inf"), "0", None])
def test_result_epoch_requires_a_finite_numeric_timestamp(result_epoch):
    sensing = reference_payload(epoch=0.0)
    sensing["temporal"]["result"]["reference_time_s"] = result_epoch
    with pytest.raises(ValueError, match="exact result reference epoch"):
        extract_evaluation_truth(sensing)


def test_new_frozen_root_reference_includes_sensor_motion_and_preserves_raw_center():
    from shared.sensing.plotting.viz import _canonical_truth_points

    sensing = reference_payload(
        target_velocity=(0, 0, 0), tx_velocity=(0.5, 0, 0), rx_velocity=(0.5, 0, 0)
    )
    frame = animated_frame()
    target = sensing["truth"]["targets"][0]
    target["target_id"] = "walker"
    target["animated_asset"] = frame["sensing"]["truth"]["targets"][0]["animated_asset"]
    target["pair_observables"][0]["velocity_reference_evidence"]["target_id"] = "walker"
    sensing["target_catalog"] = frame["sensing"]["target_catalog"]
    sensing["config_resolved"]["acquisition"] = {"sample_source": "coherent_one_solve"}
    truth = extract_evaluation_truth(sensing)
    assert truth["target_speeds"] == [-0.5]
    assert truth["recorded_target_speeds"] == [0.5]
    assert truth["target_velocity_references"] == [ROOT_RELATIVE_REFERENCE]
    assert _canonical_truth_points(sensing)[0]["velocity_m_s"] == -0.5


def test_pooled_velocity_denominator_excludes_range_only_matches_and_missed_support():
    result = evaluate_scenario(
        [([5, 9, 20], [None, 2, None])],
        [([5.5, 9], [100, 2.2])],
        per_event_target_ids=[["range-only", "both", "missed"]],
        per_event_velocity_valid=[[False, True, False]],
    )
    assert result.total_tp == 2 and result.total_fn == 1
    assert result.cumulative_range_rmse == pytest.approx(np.sqrt(0.25 / 2))
    assert result.cumulative_velocity_rmse == pytest.approx(0.2)
    assert result.velocity_matched_pair_count == 1
    assert result.velocity_excluded_matched_pair_count == 1
    assert result.velocity_unavailable_opportunity_count == 2
    assert result.per_target["range-only"]["velocity_rmse"] is None
    assert result.per_target["both"]["velocity_matched"] == 1
    assert result.per_target["missed"]["velocity_unavailable_opportunities"] == 1
    assert result.per_target["missed"]["matched"] == 0


@pytest.mark.parametrize("mask", [[1], [], [True, False], [[True]], [None]])
def test_velocity_mask_must_be_exact_boolean_support(mask):
    with pytest.raises(ValueError, match="boolean mask"):
        associate_detections_to_gt([5], [1], [5], [1], gt_velocity_valid=mask)


def test_empty_and_no_report_events_do_not_invent_matches_or_velocity_errors():
    empty = associate_detections_to_gt([], [], [], [], gt_velocity_valid=[])
    assert empty.true_positives == 0 and empty.velocity_rmse is None
    missed = associate_detections_to_gt([5], [None], [], [], gt_velocity_valid=[False])
    assert (missed.true_positives, missed.false_negatives) == (0, 1)
    assert missed.velocity_unavailable_opportunity_count == 1


def test_offline_join_keeps_new_range_only_event_and_rejects_misaligned_truth():
    from shared.sensing.postprocessor import _join_evaluation_events
    from tests.shared.test_sensing_postprocessor import (
        _evaluation_join_frame,
        _evaluation_join_result,
    )

    frame = _evaluation_join_frame(0, 5)
    result = _evaluation_join_result(frame, 5)
    truth = extract_evaluation_truth(interval_payload())
    frame["_gt"] = truth
    joined = _join_evaluation_events([frame], [result])
    assert len(joined.events) == 1 and joined.events[0].gt_velocity_valid == (False,)
    frame["_gt"]["_evaluation_unavailable_reason"] = "lifecycle reference epoch differs"
    blocked = _join_evaluation_events([frame], [result])
    assert blocked.events == ()
    assert len(blocked.skipped_invalid_truth_keys) == 1
