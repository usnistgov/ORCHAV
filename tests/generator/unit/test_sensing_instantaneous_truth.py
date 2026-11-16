"""Native truth references bind relative motion to the result's sensor pair."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.propagation.snapshots import velocities_to_array
from generator.core.sensing.hooks import (
    accumulate_sensing_gt,
    build_sensing_gt_channel,
    compute_frame_association,
)
from shared.extensions.sensing import canonical_sensing_pair_id
from shared.sensing.targets import INSTANTANEOUS_VELOCITY_REFERENCE_POLICY as POLICY


def _frame(time_s=2.0):
    pair_id = canonical_sensing_pair_id("tx-b", "rx-b")
    return {
        "frame_idx": 2,
        "tx_list": [SimpleNamespace(name="tx-a"), SimpleNamespace(name="tx-b")],
        "rx_list": [SimpleNamespace(name="rx-a"), SimpleNamespace(name="rx-b")],
        "tx_positions_snapshot": np.array([[-20.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "rx_positions_snapshot": np.array([[-30.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "tx_velocities_snapshot": np.array([[99.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "rx_velocities_snapshot": np.array([[99.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "target_positions_snapshot": np.array([[5.0, 0.0, 0.0]]),
        "target_velocities_snapshot": np.array([[1.0, 0.0, 0.0]]),
        "_sensing_target_ids": ["cube"],
        "sensing": {
            "pair_index": 3,
            "detections": [{"range_m": 5.0, "velocity_m_s": 1.0}],
            "temporal": {
                "acquisition": {"pair_id": pair_id, "tx_sensor_id": "tx-b", "rx_sensor_id": "rx-b"},
                "result": {"reference_time_s": time_s},
            },
        },
    }


def _entry(frame):
    return accumulate_sensing_gt(
        frame,
        "monostatic",
        1.0,
        {"cube": 2.0},
        reference_time_s=2.0,
        velocity_reference_policy=POLICY,
    )


def test_native_reference_uses_selected_pair_and_relative_motion():
    frame = _frame()
    entry = _entry(frame)
    # Both sensors close at 2 m/s while the target recedes at 1 m/s.
    # Previous range history would incorrectly give -3 m/s at the current epoch.
    assert entry["target_ranges"] == [5.0]
    assert entry["target_speeds"] == [1.0]
    result = compute_frame_association(entry)
    assert result["tp"] == 1 and result["velocity_rmse"] == 0.0
    evidence = entry["target_velocity_reference_evidence"][0]
    assert evidence["tx_position_m"] == [0.0, 0.0, 0.0]
    assert evidence["tx_velocity_m_s"] == [2.0, 0.0, 0.0]
    assert evidence["support"]["reference_time_s"] == 2.0
    channel = build_sensing_gt_channel(frame, entry)
    assert channel["velocity_reference_policy"] == POLICY
    assert channel["targets"][0]["velocity_reference_evidence"] == evidence


def test_native_missing_sensor_velocity_retains_interval_but_scores_range_only():
    frame = _frame()
    del frame["rx_velocities_snapshot"]
    entry = _entry(frame)
    assert entry["target_speeds"] == [-3.0]
    assert entry["target_velocity_reference_evidence"][0]["support"]["kind"] == "interval_mean"
    result = compute_frame_association(entry)
    assert result["tp"] == 1
    assert result["range_rmse"] == 0.0
    assert result["velocity_rmse"] is None
    assert result["velocity_excluded_matched_pair_count"] == 1


def test_native_summary_persists_range_only_and_unavailable_reference_denominators(tmp_path):
    from generator.figures.summary.sensing import maybe_generate_sensing_summary

    frame = _frame()
    del frame["rx_velocities_snapshot"]
    ranged = compute_frame_association(_entry(frame))
    unavailable = compute_frame_association(_entry(_frame(time_s=2.5)))
    maybe_generate_sensing_summary(
        {},
        [],
        None,
        SimpleNamespace(
            generator_summary={
                "enabled": True,
                "create": ["sensing"],
                "sensing": {"figures": ["detection"]},
            }
        ),
        association_history=[ranged, unavailable],
        output_root=tmp_path,
        strict=True,
    )
    metrics = json.loads((tmp_path / "sensing/detection_metrics.json").read_text())
    assert metrics["total_tp"] == 1 and metrics["mean_range_rmse_m"] == 0.0
    assert metrics["mean_velocity_rmse_m_s"] is None
    assert metrics["evaluated_frames"] == metrics["unavailable_reference_frames"] == 1
    assert metrics["velocity_matched_pair_count"] == 0
    assert metrics["velocity_excluded_matched_pair_count"] == 1
    assert metrics["velocity_error_frame_count"] == 0
    assert (tmp_path / "sensing/detection_summary.png").is_file()


def test_native_misaligned_truth_is_unavailable_not_empty_truth():
    entry = _entry(_frame(time_s=2.5))
    result = compute_frame_association(entry)
    assert result["evaluated"] is False
    assert result["skipped_reason"] == "truth_result_epoch_mismatch"
    assert result["tp"] == result["fp"] == result["fn"] == 0


def test_native_pair_binding_cannot_relabel_snapshot_rows():
    frame = _frame()
    frame["sensing"]["pair_index"] = 0
    with pytest.raises(ValueError, match="disagrees with acquisition identity"):
        _entry(frame)


@pytest.mark.parametrize("values", [[None], [(np.nan, 0.0, 0.0)], [object()]])
def test_missing_or_invalid_velocity_cannot_establish_stationary_truth(values):
    assert velocities_to_array(values, require_complete=True) is None


def test_known_zero_velocity_is_preserved_as_known():
    np.testing.assert_array_equal(
        velocities_to_array([(0.0, 0.0, 0.0)], require_complete=True), np.zeros((1, 3))
    )
    np.testing.assert_array_equal(velocities_to_array([None]), np.zeros((1, 3)))


@pytest.mark.parametrize("missing_upper", [False, True])
def test_sequential_sensor_velocity_is_interpolated_without_inventing_state(missing_upper):
    first = _frame()
    second = deepcopy(first)
    second["tx_positions_snapshot"] += 1.0
    second["rx_positions_snapshot"] += 1.0
    second["tx_velocities_snapshot"] += 2.0
    second["rx_velocities_snapshot"] += 2.0
    if missing_upper:
        del second["rx_velocities_snapshot"]
    observed = [
        {"scene_frame_id": 0, "sample_time_s": 0.0, "frame_data": first},
        {"scene_frame_id": 1, "sample_time_s": 1.0, "frame_data": second},
    ]
    midpoint, _, _, policy, epoch = SensingPipelineState._sequential_truth_state(
        observed, reference_time_s=0.5
    )
    assert policy == "linear_interpolation_between_source_scenes" and epoch == 0.5
    np.testing.assert_array_equal(
        midpoint["tx_velocities_snapshot"], first["tx_velocities_snapshot"] + 1.0
    )
    if missing_upper:
        assert "rx_velocities_snapshot" not in midpoint
    else:
        np.testing.assert_array_equal(
            midpoint["rx_velocities_snapshot"], first["rx_velocities_snapshot"] + 1.0
        )
