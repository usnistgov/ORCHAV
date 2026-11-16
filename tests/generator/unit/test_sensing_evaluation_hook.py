"""Focused evaluation-contract tests for generator sensing summaries."""

from generator.core.sensing.hooks import compute_frame_association


def test_generator_association_does_not_report_zero_rmse_without_matches() -> None:
    empty = compute_frame_association({})
    no_match = compute_frame_association(
        {
            "target_ranges": [10.0],
            "target_speeds": [0.0],
            "target_velocity_valid": [True],
            "det_ranges": [100.0],
            "det_velocities": [20.0],
        }
    )

    assert empty == {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "range_rmse": None,
        "velocity_rmse": None,
    }
    assert no_match["tp"] == 0
    assert no_match["fp"] == 1
    assert no_match["fn"] == 1
    assert no_match["range_rmse"] is None
    assert no_match["velocity_rmse"] is None
