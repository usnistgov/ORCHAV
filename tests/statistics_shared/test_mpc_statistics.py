"""Correctness tests for aggregate MPC statistics."""

import numpy as np
import pytest

from shared.statistics import compute_comparison_stats, compute_mpc_statistics


def test_aggregate_power_uses_received_power_db_sign() -> None:
    stats = compute_mpc_statistics(
        np.array([10.0, 20.0]),
        np.array([60.0, 60.0]),
    )

    assert stats.P_incoh == pytest.approx(2.0e-6)
    assert stats.P_incoh_db == pytest.approx(10.0 * np.log10(2.0e-6))


def test_empty_statistics_use_negative_infinity_for_zero_power() -> None:
    stats = compute_mpc_statistics(np.array([]), np.array([]))

    assert stats.P_incoh == 0.0
    assert stats.P_incoh_db == float("-inf")
    assert stats.earliest_path_delay_ns is None
    assert stats.earliest_path_loss_db is None


def test_nonfinite_paths_are_removed_as_aligned_pairs() -> None:
    stats = compute_mpc_statistics(
        np.array([1.0, np.nan, 3.0, 4.0]),
        np.array([60.0, 70.0, np.nan, 80.0]),
        aoa_az_deg=np.array([350.0, 90.0, 180.0, np.nan]),
    )

    assert stats.N_paths == 2
    assert stats.min_delay_ns == 1.0
    assert stats.max_delay_ns == 4.0
    assert stats.earliest_path_delay_ns == 1.0
    assert stats.earliest_path_loss_db == 60.0
    # Only one retained path also has a finite AoA azimuth.
    assert stats.rms_aoa_az_deg == pytest.approx(0.0, abs=1e-12)


def test_azimuth_wraps_while_elevation_remains_linear() -> None:
    stats = compute_mpc_statistics(
        np.array([1.0, 2.0]),
        np.array([60.0, 60.0]),
        aoa_az_deg=np.array([350.0, 10.0]),
        aoa_el_deg=np.array([-90.0, 90.0]),
    )

    assert stats.rms_aoa_az_deg == pytest.approx(10.0)
    assert abs(stats.mean_aoa_az_deg) < 1e-12
    assert stats.rms_aoa_el_deg == pytest.approx(90.0)
    assert stats.mean_aoa_el_deg == pytest.approx(0.0)


@pytest.mark.parametrize("threshold_db", [np.nan, np.inf, 0.1])
def test_invalid_relative_threshold_is_rejected(threshold_db: float) -> None:
    with pytest.raises(ValueError, match="threshold_db"):
        compute_mpc_statistics(
            np.array([1.0]),
            np.array([60.0]),
            threshold_db=threshold_db,
        )


def test_misaligned_path_or_angle_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="same paths"):
        compute_mpc_statistics(np.array([1.0, 2.0]), np.array([60.0]))

    with pytest.raises(ValueError, match="angle arrays"):
        compute_mpc_statistics(
            np.array([1.0, 2.0]),
            np.array([60.0, 70.0]),
            aoa_az_deg=np.array([0.0]),
        )


def test_comparison_names_earliest_path_without_los_claim() -> None:
    measured = compute_mpc_statistics(np.array([1.0]), np.array([60.0]))
    simulated = compute_mpc_statistics(np.array([2.0]), np.array([65.0]))

    comparison = compute_comparison_stats(measured, simulated)

    assert comparison["earliest_path_delay_diff_ns"] == 1.0
    assert comparison["earliest_path_loss_diff_db"] == 5.0
    assert "los_delay_diff_ns" not in comparison
