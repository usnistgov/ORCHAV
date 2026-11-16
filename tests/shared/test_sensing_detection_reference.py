"""Independent Phase-4 references for detector and clutter-removal semantics."""

from __future__ import annotations

import numpy as np
import pytest

from shared.sensing.detection.reference import (
    ca_cfar_detection_probability_nonfluctuating_target,
    ca_cfar_false_alarm_probability,
    ca_cfar_threshold_scale,
    cfar_training_cell_count_1d,
    cfar_training_cell_count_2d,
    finite_cpi_mean_removal_power_gain,
    os_cfar_false_alarm_probability,
    os_cfar_rank_index,
    os_cfar_threshold_scale,
    relative_complex_awgn_power,
    sliding_baseline_frequency_response,
    wilson_binomial_interval,
)


def test_full_window_training_cell_counts_are_hand_computed() -> None:
    assert cfar_training_cell_count_1d(4) == 8
    # Outer 7x11 rectangle minus the 3x3 guard-plus-CUT block.
    assert cfar_training_cell_count_2d((1, 1), (2, 4)) == 68


@pytest.mark.parametrize("training_cells", [8, 24, 68])
@pytest.mark.parametrize("requested_pfa", [0.1, 0.01, 1e-4])
def test_ca_cfar_probability_and_multiplier_are_exact_inverses(
    training_cells: int,
    requested_pfa: float,
) -> None:
    multiplier = ca_cfar_threshold_scale(requested_pfa, training_cells)
    assert ca_cfar_false_alarm_probability(multiplier, training_cells) == pytest.approx(
        requested_pfa,
        rel=1e-12,
    )


def test_ca_cfar_seeded_exponential_monte_carlo_contains_declared_probability() -> None:
    rng = np.random.default_rng(20260828)
    trials = 200_000
    training_cells = 24
    requested_pfa = 1e-3
    multiplier = ca_cfar_threshold_scale(requested_pfa, training_cells)
    training = rng.exponential(size=(trials, training_cells))
    cell_under_test = rng.exponential(size=trials)
    false_alarms = int(np.count_nonzero(cell_under_test > multiplier * training.mean(axis=1)))
    lower, upper = wilson_binomial_interval(false_alarms, trials, confidence=0.999)
    assert lower <= requested_pfa <= upper


def test_ca_cfar_seeded_nonfluctuating_target_pd_matches_independent_integral() -> None:
    rng = np.random.default_rng(20260830)
    trials = 200_000
    training_cells = 24
    requested_pfa = 1e-3
    target_snr_linear = 10.0 ** (6.0 / 10.0)
    multiplier = ca_cfar_threshold_scale(requested_pfa, training_cells)
    expected_pd = ca_cfar_detection_probability_nonfluctuating_target(
        multiplier,
        training_cells,
        target_snr_linear,
    )

    training = rng.exponential(size=(trials, training_cells))
    noise = (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2.0)
    cell_under_test = np.abs(np.sqrt(target_snr_linear) + noise) ** 2
    detections = int(np.count_nonzero(cell_under_test > multiplier * training.mean(axis=1)))
    lower, upper = wilson_binomial_interval(detections, trials, confidence=0.999)
    assert lower <= expected_pd <= upper
    assert expected_pd > requested_pfa


@pytest.mark.parametrize("rank_fraction", [0.0, 0.5, 0.75, 1.0])
def test_os_cfar_probability_and_multiplier_are_exact_inverses(rank_fraction: float) -> None:
    training_cells = 24
    rank_index = os_cfar_rank_index(training_cells, rank_fraction)
    requested_pfa = 0.01
    multiplier = os_cfar_threshold_scale(requested_pfa, training_cells, rank_index)
    assert os_cfar_false_alarm_probability(
        multiplier,
        training_cells,
        rank_index,
    ) == pytest.approx(requested_pfa, rel=1e-11)


def test_os_cfar_half_ties_round_to_nearest_even_rank() -> None:
    assert os_cfar_rank_index(6, 0.5) == 2
    assert os_cfar_rank_index(8, 0.5) == 4


def test_os_cfar_seeded_exponential_monte_carlo_contains_declared_probability() -> None:
    rng = np.random.default_rng(20260829)
    trials = 120_000
    training_cells = 24
    rank_index = os_cfar_rank_index(training_cells, 0.75)
    requested_pfa = 0.01
    multiplier = os_cfar_threshold_scale(requested_pfa, training_cells, rank_index)
    training = rng.exponential(size=(trials, training_cells))
    ordered_noise = np.partition(training, rank_index, axis=1)[:, rank_index]
    cell_under_test = rng.exponential(size=trials)
    false_alarms = int(np.count_nonzero(cell_under_test > multiplier * ordered_noise))
    lower, upper = wilson_binomial_interval(false_alarms, trials, confidence=0.999)
    assert lower <= requested_pfa <= upper


def test_relative_awgn_power_is_statistical_not_thermal() -> None:
    assert relative_complex_awgn_power(2.0, 10.0) == pytest.approx(0.2)
    assert relative_complex_awgn_power(0.0, -20.0) == 0.0


@pytest.mark.parametrize("snr_db", [-1e308, 1e308])
def test_relative_awgn_power_rejects_unrepresentable_finite_snr(snr_db: float) -> None:
    with pytest.raises(ValueError, match="floating-point range"):
        relative_complex_awgn_power(1.0, snr_db)


@pytest.mark.parametrize("sample_count", [4, 16])
def test_finite_cpi_mean_removal_matches_direct_tone_energy(sample_count: int) -> None:
    frequency = 0.125
    tone = np.exp(2j * np.pi * frequency * np.arange(sample_count))
    direct_gain = np.mean(np.abs(tone - np.mean(tone)) ** 2) / np.mean(np.abs(tone) ** 2)
    assert finite_cpi_mean_removal_power_gain(frequency, sample_count) == pytest.approx(
        direct_gain,
        abs=1e-14,
    )
    assert finite_cpi_mean_removal_power_gain(0.0, sample_count) == pytest.approx(0.0)


@pytest.mark.parametrize("window", [1, 2, 8])
def test_sliding_baseline_frequency_response_matches_steady_state(window: int) -> None:
    frequency = 0.07
    indices = np.arange(128, dtype=np.float64)
    tone = np.exp(2j * np.pi * frequency * indices)
    n = 100
    direct = tone[n] - np.mean(tone[n - window : n])
    direct_response = direct / tone[n]
    assert sliding_baseline_frequency_response(frequency, window) == pytest.approx(
        direct_response,
        abs=1e-13,
    )


def test_wilson_interval_rejects_missing_denominator() -> None:
    with pytest.raises(ValueError, match="trials must be positive"):
        wilson_binomial_interval(0, 0)
