"""Independent beta-mixture and boundary checks for the CA-CFAR Pd reference."""

import numpy as np
import pytest
from scipy import special, stats

from shared.sensing.detection.reference import (
    ca_cfar_detection_probability_nonfluctuating_target as pd,
)


@pytest.mark.parametrize("count", [1, 8, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("snr", [0, 0.1, 1, 10, 100])
def test_noncentral_f_against_independent_poisson_beta_mixture(count, snr):
    alpha = 10.0
    k = np.arange(int(stats.poisson.ppf(1 - 1e-14, snr)) + 1) if snr else np.array([0])
    expected = np.sum(
        stats.poisson.pmf(k, snr) * special.betaincc(1 + k, count, alpha / (count + alpha))
    )
    assert pd(alpha, count, snr) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("count", [1, 8, 256, 2048])
def test_pd_monotonic_and_zero_threshold(count):
    assert pd(0, count, 0) == 1
    assert pd(0, count, 1e308) == 1
    probabilities = [pd(10, count, snr) for snr in [0, 0.1, 1, 10, 100, 1000]]
    assert np.all(np.diff(probabilities) >= 0)
    probabilities = [pd(alpha, count, 10) for alpha in [0, 0.1, 1, 10, 100, 1000]]
    assert np.all(np.diff(probabilities) <= 0)


@pytest.mark.parametrize(
    "alpha,count,snr",
    [
        (-1, 8, 1),
        (np.inf, 8, 1),
        (1, 0, 1),
        (1, 8, -1),
        (1, 8, np.inf),
        (1, 8, np.nan),
        (10, 256, 1e100),
        (10, 256, 1e308),
    ],
)
def test_invalid_and_unsupported_numeric_cases_reject_without_nan(alpha, count, snr):
    with pytest.raises(ValueError):
        pd(alpha, count, snr)
