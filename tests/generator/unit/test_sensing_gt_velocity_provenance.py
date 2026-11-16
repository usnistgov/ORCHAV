"""Focused contracts for sensing truth velocity provenance and validity."""

from __future__ import annotations

import numpy as np
import pytest

from generator.core.sensing.hooks import (
    accumulate_sensing_gt,
    build_sensing_gt_channel,
    compute_frame_association,
)


def _frame(
    target_ids: list[str],
    positions_m: list[list[float]],
    *,
    velocities_m_s: list[list[float]] | None = None,
) -> dict:
    frame = {
        "tx_positions_snapshot": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions_snapshot": np.array([[0.0, 0.0, 0.0]]),
        "target_positions_snapshot": np.asarray(positions_m, dtype=np.float64),
        "targets_metadata": [{"name": target_id} for target_id in target_ids],
        "_sensing_target_ids": list(target_ids),
    }
    if velocities_m_s is not None:
        frame["target_velocities_snapshot"] = np.asarray(
            velocities_m_s,
            dtype=np.float64,
        )
    return frame


def test_first_epoch_without_velocity_is_null_and_traceable() -> None:
    frame = _frame(["walker"], [[3.0, 0.0, 0.0]])

    entry = accumulate_sensing_gt(
        frame,
        "bistatic",
        None,
        None,
        reference_time_s=4.25,
    )
    channel = build_sensing_gt_channel(frame, entry)

    assert entry["target_ids"] == ["walker"]
    assert entry["reference_time_s"] == pytest.approx(4.25)
    assert entry["target_speeds"] == [None]
    assert entry["target_velocity_valid"] == [False]
    assert entry["target_velocity_sources"] == ["unavailable"]
    assert channel == {
        "targets": [
            {
                "target_id": "walker",
                "range_m": 6.0,
                "path_closing_rate_m_s": None,
                "monostatic_equivalent_velocity_m_s": None,
                "bistatic_bisector_velocity_m_s": None,
                "velocity_valid": False,
                "velocity_derivation_source": "unavailable",
                "velocity_reference_dt_s": None,
                "velocity_unavailable_reason": ("no_prior_reference_epoch_or_target_velocity"),
            }
        ]
    }


def test_finite_difference_joins_previous_ranges_by_exact_authored_id() -> None:
    frame = _frame(
        ["bravo", "alpha"],
        [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    entry = accumulate_sensing_gt(
        frame,
        "bistatic",
        0.5,
        {"alpha": 5.0, "bravo": 10.0},
        reference_time_s=8.5,
    )

    assert entry["target_ids"] == ["bravo", "alpha"]
    assert entry["target_ranges"] == pytest.approx([8.0, 4.0])
    assert entry["target_path_closing_rates_m_s"] == pytest.approx([4.0, 2.0])
    assert entry["target_speeds"] == pytest.approx([2.0, 1.0])
    assert entry["target_velocity_sources"] == [
        "exact_epoch_finite_difference",
        "exact_epoch_finite_difference",
    ]
    assert entry["target_velocity_reference_dt_s"] == pytest.approx([0.5, 0.5])


def test_canonical_channel_rejects_identity_rebinding() -> None:
    frame = _frame(["authored-id"], [[3.0, 0.0, 0.0]], velocities_m_s=[[0.0, 0.0, 0.0]])
    entry = accumulate_sensing_gt(
        frame,
        "monostatic",
        None,
        None,
        reference_time_s=1.0,
    )
    entry["target_ids"] = ["different-id"]

    with pytest.raises(ValueError, match="changed after source-row binding"):
        build_sensing_gt_channel(frame, entry)


def test_association_explicitly_skips_unavailable_truth_velocity() -> None:
    result = compute_frame_association(
        {
            "target_ranges": [5.0],
            "target_speeds": [None],
            "target_velocity_valid": [False],
            "det_ranges": [5.0],
            "det_velocities": [0.0],
        }
    )

    assert result == {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "range_rmse": None,
        "velocity_rmse": None,
        "evaluated": False,
        "skipped_reason": "truth_velocity_unavailable",
        "truth_targets_skipped": 1,
        "detections_skipped": 1,
    }
