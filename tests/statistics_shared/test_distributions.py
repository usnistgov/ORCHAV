"""
Unit tests for shared.statistics distributions module.

These tests verify histogram, CDF, and other distribution utilities.
"""

import numpy as np

from shared.statistics import (
    DEFAULT_BINS,
    compute_cdf,
    compute_histogram,
    compute_percentile,
    compute_statistics_summary,
)


class TestDefaultBins:
    """Tests for DEFAULT_BINS constant."""

    def test_default_bins_is_50(self):
        """DEFAULT_BINS should be 50."""
        assert DEFAULT_BINS == 50


class TestComputeHistogram:
    """Tests for compute_histogram function."""

    def test_empty_input(self):
        """Empty arrays should return empty arrays."""
        centers, counts = compute_histogram(np.array([]))
        assert len(centers) == 0
        assert len(counts) == 0

    def test_default_bins(self):
        """Should use DEFAULT_BINS by default."""
        values = np.random.randn(1000)
        centers, counts = compute_histogram(values)
        assert len(centers) == DEFAULT_BINS
        assert len(counts) == DEFAULT_BINS

    def test_custom_bins(self):
        """Should respect custom bin count."""
        values = np.random.randn(1000)
        centers, counts = compute_histogram(values, bins=20)
        assert len(centers) == 20
        assert len(counts) == 20

    def test_count_sum_equals_input(self):
        """Sum of counts should equal number of input values."""
        values = np.random.randn(500)
        _, counts = compute_histogram(values)
        assert np.sum(counts) == 500

    def test_custom_range(self):
        """Should respect custom range."""
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        centers, _ = compute_histogram(values, bins=5, range=(0.0, 1.0))
        # Centers should be in the range [0, 1]
        assert centers[0] >= 0.0
        assert centers[-1] <= 1.0

    def test_density_normalization(self):
        """Density mode should normalize to integrate to 1."""
        values = np.random.randn(10000)
        centers, counts = compute_histogram(values, density=True)
        # For density, integral (sum * bin_width) should be close to 1
        bin_width = centers[1] - centers[0]
        integral = np.sum(counts) * bin_width
        assert np.isclose(integral, 1.0, atol=0.1)

    def test_non_finite_values_are_excluded(self):
        """Histogram counts should include only finite values."""
        values = np.array([1.0, np.nan, 2.0, np.inf, 3.0, -np.inf])
        centers, counts = compute_histogram(values, bins=3)

        assert np.all(np.isfinite(centers))
        assert np.sum(counts) == 3

    def test_only_non_finite_values_use_empty_result(self):
        """An input without finite values should behave like an empty input."""
        centers, counts = compute_histogram(np.array([np.nan, np.inf, -np.inf]))

        assert len(centers) == 0
        assert len(counts) == 0


class TestComputeCDF:
    """Tests for compute_cdf function."""

    def test_empty_input(self):
        """Empty arrays should return empty arrays."""
        x, cdf = compute_cdf(np.array([]))
        assert len(x) == 0
        assert len(cdf) == 0

    def test_cdf_ends_at_one(self):
        """CDF should end at 1.0."""
        values = np.random.randn(100)
        _, cdf = compute_cdf(values)
        assert cdf[-1] == 1.0

    def test_cdf_starts_near_zero(self):
        """CDF should start at 1/N (first percentile)."""
        values = np.random.randn(100)
        _, cdf = compute_cdf(values)
        assert cdf[0] == 1 / 100

    def test_cdf_monotonic_increasing(self):
        """CDF should be monotonically increasing."""
        values = np.random.randn(100)
        _, cdf = compute_cdf(values)
        assert np.all(np.diff(cdf) >= 0)

    def test_sorted_values(self):
        """Output x values should be sorted."""
        values = np.array([5, 2, 8, 1, 9, 3])
        x, _ = compute_cdf(values)
        assert np.all(np.diff(x) >= 0)

    def test_non_finite_values_are_excluded(self):
        """CDF values should be based only on finite observations."""
        values = np.array([3.0, np.nan, np.inf, 1.0, -np.inf, 2.0])
        x, cdf = compute_cdf(values)

        np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(cdf, np.array([1 / 3, 2 / 3, 1.0]))

    def test_only_non_finite_values_use_empty_result(self):
        """An input without finite values should behave like an empty input."""
        x, cdf = compute_cdf(np.array([np.nan, np.inf, -np.inf]))

        assert len(x) == 0
        assert len(cdf) == 0


class TestComputePercentile:
    """Tests for compute_percentile function."""

    def test_empty_input(self):
        """Empty arrays should return NaN."""
        result = compute_percentile(np.array([]), np.array([50]))
        assert np.isnan(result[0])

    def test_median_percentile(self):
        """50th percentile should be median."""
        values = np.arange(1, 101)  # 1 to 100
        result = compute_percentile(values, np.array([50]))
        assert np.isclose(result[0], 50.5)  # Median of 1-100

    def test_extreme_percentiles(self):
        """0th and 100th percentiles should be min and max."""
        values = np.array([1, 5, 10, 50, 100])
        result = compute_percentile(values, np.array([0, 100]))
        assert result[0] == 1
        assert result[1] == 100

    def test_quartiles(self):
        """Test quartile computation."""
        values = np.arange(100)
        result = compute_percentile(values, np.array([25, 50, 75]))
        assert len(result) == 3
        # 25th percentile should be around 24.75
        assert np.isclose(result[0], 24.75)

    def test_non_finite_values_are_excluded(self):
        """Percentiles should use only finite observations."""
        values = np.array([1.0, np.nan, np.inf, 2.0, -np.inf, 3.0])
        result = compute_percentile(values, np.array([0, 50, 100]))

        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_only_non_finite_values_use_empty_result(self):
        """An input without finite values should return NaN percentiles."""
        result = compute_percentile(np.array([np.nan, np.inf, -np.inf]), np.array([25, 50, 75]))

        assert np.all(np.isnan(result))


class TestComputeStatisticsSummary:
    """Tests for compute_statistics_summary function."""

    def test_empty_input(self):
        """Empty arrays should return NaN for all stats."""
        result = compute_statistics_summary(np.array([]))
        assert result["count"] == 0
        assert np.isnan(result["mean"])
        assert np.isnan(result["std"])
        assert np.isnan(result["min"])
        assert np.isnan(result["max"])

    def test_single_value(self):
        """Single value should have zero std."""
        result = compute_statistics_summary(np.array([42.0]))
        assert result["count"] == 1
        assert result["mean"] == 42.0
        assert result["std"] == 0.0
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert result["median"] == 42.0

    def test_known_values(self):
        """Test against known statistical values."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_statistics_summary(values)
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["median"] == 3.0
        assert result["q25"] == 2.0
        assert result["q75"] == 4.0

    def test_all_keys_present(self):
        """Should return all expected keys."""
        result = compute_statistics_summary(np.random.randn(100))
        expected_keys = ["count", "mean", "std", "min", "max", "median", "q25", "q75"]
        for key in expected_keys:
            assert key in result

    def test_non_finite_values_are_excluded(self):
        """Summary statistics should describe only finite observations."""
        values = np.array([1.0, np.nan, np.inf, 2.0, -np.inf, 3.0])
        result = compute_statistics_summary(values)

        assert result["count"] == 3
        assert result["mean"] == 2.0
        assert result["std"] == np.std(np.array([1.0, 2.0, 3.0]))
        assert result["min"] == 1.0
        assert result["max"] == 3.0
        assert result["median"] == 2.0
        assert result["q25"] == 1.5
        assert result["q75"] == 2.5

    def test_only_non_finite_values_use_empty_result(self):
        """An input without finite values should use the empty summary."""
        result = compute_statistics_summary(np.array([np.nan, np.inf, -np.inf]))

        assert result["count"] == 0
        assert all(np.isnan(result[key]) for key in result if key != "count")
