"""
Regression tests for statistics unification.

These tests ensure the new unified implementations produce identical results
to the old implementations. The expected values were captured from the old
implementations BEFORE the migration.

See: capture_regression_baseline.py for how expected values were captured.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from shared.statistics import (
    compute_angular_spread,
    compute_delay_spread,
    compute_signal_strength_distribution,
)


@pytest.fixture
def expected_values():
    """Load expected values captured from old implementations."""
    baseline_file = Path(__file__).parent / "regression_expected_values.json"
    with open(baseline_file) as f:
        return json.load(f)


class TestDelaySpreadRegression:
    """Verify new delay_spread matches old implementation exactly."""

    def test_simple_increasing(self, expected_values):
        """Test simple increasing delays/powers case."""
        case = expected_values["delay_spread"]["simple_increasing"]
        delays = np.array(case["delays_ns"])
        powers = np.array(case["powers_w"])
        expected = case["expected_delay_spread"]

        result = compute_delay_spread(delays, powers)
        assert np.isclose(result, expected, rtol=1e-10), f"Expected {expected}, got {result}"

    def test_same_delay(self, expected_values):
        """Test all same delay case."""
        case = expected_values["delay_spread"]["same_delay"]
        delays = np.array(case["delays_ns"])
        powers = np.array(case["powers_w"])
        expected = case["expected_delay_spread"]

        result = compute_delay_spread(delays, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_spread_powers(self, expected_values):
        """Test spread delays with decreasing powers."""
        case = expected_values["delay_spread"]["spread_powers"]
        delays = np.array(case["delays_ns"])
        powers = np.array(case["powers_w"])
        expected = case["expected_delay_spread"]

        result = compute_delay_spread(delays, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_single_path(self, expected_values):
        """Test single path case."""
        case = expected_values["delay_spread"]["single_path"]
        delays = np.array(case["delays_ns"])
        powers = np.array(case["powers_w"])
        expected = case["expected_delay_spread"]

        result = compute_delay_spread(delays, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_empty(self, expected_values):
        """Test empty input case."""
        case = expected_values["delay_spread"]["empty"]
        delays = np.array(case["delays_ns"])
        powers = np.array(case["powers_w"])
        expected = case["expected_delay_spread"]

        result = compute_delay_spread(delays, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_large_spread(self, expected_values):
        """Test large delay spread case."""
        case = expected_values["delay_spread"]["large_spread"]
        delays = np.array(case["delays_ns"])
        powers = np.array(case["powers_w"])
        expected = case["expected_delay_spread"]

        result = compute_delay_spread(delays, powers)
        assert np.isclose(result, expected, rtol=1e-10)


class TestAngularSpreadRegression:
    """Verify new angular_spread matches old implementation exactly."""

    def test_simple(self, expected_values):
        """Test simple angular spread case."""
        case = expected_values["angular_spread"]["simple"]
        angles = np.array(case["angles_deg"])
        powers = np.array(case["powers_w"])
        expected = case["expected_angular_spread"]

        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, expected, rtol=1e-10), f"Expected {expected}, got {result}"

    def test_around_zero(self, expected_values):
        """Test angles wrapping around zero."""
        case = expected_values["angular_spread"]["around_zero"]
        angles = np.array(case["angles_deg"])
        powers = np.array(case["powers_w"])
        expected = case["expected_angular_spread"]

        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_opposite_sides(self, expected_values):
        """Test angles 180 degrees apart."""
        case = expected_values["angular_spread"]["opposite_sides"]
        angles = np.array(case["angles_deg"])
        powers = np.array(case["powers_w"])
        expected = case["expected_angular_spread"]

        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_single_angle(self, expected_values):
        """Test single angle case."""
        case = expected_values["angular_spread"]["single_angle"]
        angles = np.array(case["angles_deg"])
        powers = np.array(case["powers_w"])
        expected = case["expected_angular_spread"]

        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_empty(self, expected_values):
        """Test empty input case."""
        case = expected_values["angular_spread"]["empty"]
        angles = np.array(case["angles_deg"])
        powers = np.array(case["powers_w"])
        expected = case["expected_angular_spread"]

        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_weighted(self, expected_values):
        """Test weighted angular spread."""
        case = expected_values["angular_spread"]["weighted"]
        angles = np.array(case["angles_deg"])
        powers = np.array(case["powers_w"])
        expected = case["expected_angular_spread"]

        result = compute_angular_spread(angles, powers)
        assert np.isclose(result, expected, rtol=1e-10)


class TestHistogramRegression:
    """Verify histogram computation matches old implementation."""

    def test_uniform_distribution(self, expected_values):
        """Test uniform distribution histogram."""
        case = expected_values["histogram"]["uniform"]
        powers = np.array(case["powers_w"])
        expected_centers = np.array(case["expected_bin_centers"])
        expected_counts = np.array(case["expected_counts"])
        num_bins = case["num_bins"]

        centers, counts = compute_signal_strength_distribution(powers, num_bins=num_bins)

        # Check bin centers match
        assert len(centers) == len(
            expected_centers
        ), f"Expected {len(expected_centers)} bins, got {len(centers)}"
        np.testing.assert_array_almost_equal(
            centers, expected_centers, decimal=5, err_msg="Bin centers don't match"
        )

        # Check counts match
        np.testing.assert_array_equal(
            counts, expected_counts, err_msg="Histogram counts don't match"
        )

    def test_single_value(self, expected_values):
        """Test single value histogram."""
        case = expected_values["histogram"]["single_value"]
        powers = np.array(case["powers_w"])
        expected_counts = np.array(case["expected_counts"])
        num_bins = case["num_bins"]

        centers, counts = compute_signal_strength_distribution(powers, num_bins=num_bins)

        # Check total count is 1
        assert np.sum(counts) == 1
        # Check counts match
        np.testing.assert_array_equal(counts, expected_counts)

    def test_empty(self, expected_values):
        """Test empty input histogram."""
        case = expected_values["histogram"]["empty"]
        powers = np.array(case["powers_w"])
        num_bins = case["num_bins"]

        centers, counts = compute_signal_strength_distribution(powers, num_bins=num_bins)

        assert len(centers) == 0
        assert len(counts) == 0
