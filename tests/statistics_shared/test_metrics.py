"""
Unit tests for shared.statistics metrics module.

These tests verify the mathematical correctness of unified metric computations.
"""

import numpy as np

from shared.statistics import (
    compute_angular_spread,
    compute_binned_power_delay_profile,
    compute_delay_spread,
    compute_linear_angular_spread,
    compute_power_delay_profile,
    compute_signal_strength_distribution,
    compute_snr_distribution,
    pathloss_db_to_power_linear,
    power_linear_to_db,
)


class TestDelaySpread:
    """Tests for compute_delay_spread function."""

    def test_empty_input_returns_zero(self):
        """Empty arrays should return 0.0."""
        result = compute_delay_spread(np.array([]), np.array([]))
        assert result == 0.0

    def test_single_path_returns_zero(self):
        """Single path has no spread."""
        result = compute_delay_spread(np.array([100.0]), np.array([1.0]))
        assert result == 0.0

    def test_zero_total_power_returns_zero(self):
        """Non-empty arrays with no usable power should return 0.0."""
        result = compute_delay_spread(np.array([10.0, 20.0]), np.array([0.0, 0.0]))
        assert result == 0.0

    def test_same_delay_returns_zero(self):
        """All paths at same delay = no spread."""
        delays = np.array([50.0, 50.0, 50.0])
        powers = np.array([1.0, 2.0, 3.0])
        result = compute_delay_spread(delays, powers)
        assert np.isclose(result, 0.0)

    def test_known_values(self):
        """Test against known computed values."""
        delays = np.array([10.0, 20.0, 30.0])
        powers = np.array([1.0, 2.0, 3.0])
        result = compute_delay_spread(delays, powers)
        # Mean delay = (10*1 + 20*2 + 30*3) / 6 = 140/6 = 23.333...
        # Variance = (1*(10-23.33)^2 + 2*(20-23.33)^2 + 3*(30-23.33)^2) / 6
        #          = (177.78 + 22.22 + 133.33) / 6 = 55.56
        # RMS = sqrt(55.56) = 7.45
        assert np.isclose(result, 7.4535599, rtol=1e-5)

    def test_weights_affect_result(self):
        """Higher power at certain delays should pull the mean."""
        delays = np.array([0.0, 100.0])
        # Equal weights
        equal = compute_delay_spread(delays, np.array([1.0, 1.0]))
        # More weight on 0
        weighted_left = compute_delay_spread(delays, np.array([10.0, 1.0]))
        # More weight on 100
        weighted_right = compute_delay_spread(delays, np.array([1.0, 10.0]))

        # All should be positive
        assert equal > 0
        assert weighted_left > 0
        assert weighted_right > 0


class TestAngularSpread:
    """Tests for compute_angular_spread function."""

    def test_empty_input_returns_zero(self):
        """Empty arrays should return 0.0."""
        result = compute_angular_spread(np.array([]), np.array([]))
        assert result == 0.0

    def test_single_angle_returns_zero(self):
        """Single angle has no spread."""
        result = compute_angular_spread(np.array([45.0]), np.array([1.0]))
        assert result == 0.0

    def test_single_usable_angle_returns_exact_zero(self):
        """Invalid and zero-power records do not create numerical residue."""
        result = compute_angular_spread(
            np.array([45.0, np.nan, 90.0, 135.0]),
            np.array([1.0, 2.0, 0.0, np.nan]),
        )
        assert result == 0.0

    def test_zero_total_power_returns_zero(self):
        """Non-empty arrays with no usable power should return 0.0."""
        result = compute_angular_spread(np.array([0.0, 90.0]), np.array([0.0, 0.0]))
        assert result == 0.0

    def test_same_angle_returns_zero(self):
        """All angles the same = no spread."""
        angles = np.array([90.0, 90.0, 90.0])
        powers = np.array([1.0, 1.0, 1.0])
        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, 0.0)

    def test_wrapping_around_zero(self):
        """Test circular statistics handle 0/360 boundary."""
        # Angles around 0 degrees (350, 0, 10)
        angles = np.array([350.0, 0.0, 10.0])
        powers = np.array([1.0, 1.0, 1.0])
        result = compute_angular_spread(angles, powers)
        # Should be approximately 8.16 degrees (same as 0, 10, 20)
        assert 5.0 < result < 15.0

    def test_opposite_angles_give_90_spread(self):
        """180 degrees apart with equal weights = 90 degree spread."""
        angles = np.array([0.0, 180.0])
        powers = np.array([1.0, 1.0])
        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, 90.0)

    def test_simple_spread(self):
        """Test simple angular spread case."""
        angles = np.array([0.0, 10.0, 20.0])
        powers = np.array([1.0, 1.0, 1.0])
        result = compute_angular_spread(angles, powers)
        # Should be about 8.16 degrees
        assert np.isclose(result, 8.1649658, rtol=1e-5)


class TestLinearAngularSpread:
    """Tests for endpoint-bounded elevation spread."""

    def test_elevation_does_not_wrap_at_360_degrees(self):
        angles = np.array([-90.0, 90.0])
        powers = np.array([1.0, 1.0])

        assert compute_linear_angular_spread(angles, powers) == 90.0


class TestPowerDelayProfile:
    """Tests for compute_power_delay_profile function."""

    def test_empty_input(self):
        """Empty arrays should return empty arrays."""
        delays, powers_db = compute_power_delay_profile(np.array([]), np.array([]))
        assert len(delays) == 0
        assert len(powers_db) == 0

    def test_output_sorted_by_delay(self):
        """Output should be sorted by increasing delay."""
        delays = np.array([30.0, 10.0, 20.0])
        powers = np.array([0.1, 1.0, 0.5])
        sorted_delays, _ = compute_power_delay_profile(delays, powers)
        assert np.all(sorted_delays == [10.0, 20.0, 30.0])

    def test_power_converted_to_db(self):
        """Powers should be converted to dB."""
        delays = np.array([10.0, 20.0])
        powers = np.array([1.0, 0.1])  # 0 dB and -10 dB
        _, powers_db = compute_power_delay_profile(delays, powers)
        assert np.isclose(powers_db[0], 0.0, atol=0.01)
        assert np.isclose(powers_db[1], -10.0, atol=0.01)

    def test_weak_valid_path_is_not_clipped(self):
        """Finite positive path gain keeps its full dB dynamic range."""
        _, powers_db = compute_power_delay_profile(
            np.array([10.0]),
            np.array([1e-24]),
        )

        np.testing.assert_allclose(powers_db, np.array([-240.0]))

    def test_nonpositive_and_nonfinite_records_are_ignored(self):
        delays, powers_db = compute_power_delay_profile(
            np.array([10.0, 20.0, 30.0, np.nan]),
            np.array([1.0, 0.0, -1.0, 1.0]),
        )

        np.testing.assert_array_equal(delays, np.array([10.0]))
        np.testing.assert_array_equal(powers_db, np.array([0.0]))


class TestBinnedPowerDelayProfile:
    """Tests for the phase-free, incoherently summed PDP view."""

    def test_bin_sums_linear_power_and_uses_weighted_delay(self):
        delays, powers_db = compute_binned_power_delay_profile(
            np.array([10.0, 10.5, 20.0]),
            np.array([1.0, 0.5, 0.25]),
            coalescence_threshold_ns=1.0,
        )

        np.testing.assert_allclose(delays, np.array([10.0 + 1.0 / 6.0, 20.0]))
        np.testing.assert_allclose(powers_db, 10.0 * np.log10(np.array([1.5, 0.25])))

    def test_bin_never_spans_more_than_threshold(self):
        delays, _ = compute_binned_power_delay_profile(
            np.array([0.0, 0.9, 1.8]),
            np.ones(3),
            coalescence_threshold_ns=1.0,
        )

        np.testing.assert_allclose(delays, np.array([0.45, 1.8]))

    def test_binning_preserves_total_linear_power(self):
        powers = np.array([1.0, 0.5, 0.25, 0.125])
        _, binned_powers_db = compute_binned_power_delay_profile(
            np.array([10.0, 10.5, 20.0, 20.4]),
            powers,
            coalescence_threshold_ns=1.0,
        )

        binned_linear_power = np.sum(10.0 ** (binned_powers_db / 10.0))
        np.testing.assert_allclose(binned_linear_power, np.sum(powers))


class TestSignalStrengthDistribution:
    """Tests for compute_signal_strength_distribution function."""

    def test_empty_input(self):
        """Empty arrays should return empty arrays."""
        centers, counts = compute_signal_strength_distribution(np.array([]))
        assert len(centers) == 0
        assert len(counts) == 0

    def test_output_shape(self):
        """Should return requested number of bins."""
        powers = np.random.rand(100) * 1e-3
        centers, counts = compute_signal_strength_distribution(powers, num_bins=20)
        assert len(centers) == 20
        assert len(counts) == 20

    def test_count_sum_equals_input(self):
        """Sum of counts should equal number of input values."""
        powers = np.random.rand(100) * 1e-3
        _, counts = compute_signal_strength_distribution(powers, num_bins=20)
        assert np.sum(counts) == 100


class TestSNRDistribution:
    """Tests for compute_snr_distribution function."""

    def test_empty_input(self):
        """Empty arrays should return empty arrays."""
        centers, counts = compute_snr_distribution(np.array([]))
        assert len(centers) == 0
        assert len(counts) == 0

    def test_output_shape(self):
        """Should return requested number of bins."""
        powers = np.random.rand(100) * 1e-6
        centers, counts = compute_snr_distribution(powers, num_bins=15)
        assert len(centers) == 15
        assert len(counts) == 15


class TestUnitConversions:
    """Tests for unit conversion utility functions."""

    def test_pathloss_db_to_power_linear(self):
        """Test path loss to power conversion."""
        # 0 dB path loss = 1.0 linear power
        assert np.isclose(pathloss_db_to_power_linear(np.array([0.0]))[0], 1.0)
        # 10 dB path loss = 0.1 linear power
        assert np.isclose(pathloss_db_to_power_linear(np.array([10.0]))[0], 0.1)
        # 20 dB path loss = 0.01 linear power
        assert np.isclose(pathloss_db_to_power_linear(np.array([20.0]))[0], 0.01)

    def test_power_linear_to_db(self):
        """Test power to dB conversion."""
        # 1.0 linear = 0 dB
        assert np.isclose(power_linear_to_db(np.array([1.0]))[0], 0.0)
        # 0.1 linear = -10 dB
        assert np.isclose(power_linear_to_db(np.array([0.1]))[0], -10.0)
        # 0.01 linear = -20 dB
        assert np.isclose(power_linear_to_db(np.array([0.01]))[0], -20.0)
