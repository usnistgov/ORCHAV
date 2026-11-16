from __future__ import annotations

import numpy as np
import pytest

from shared.sensing.detection.cfar import cfar_1d, cfar_2d, cluster_detections_2d, nms_peaks_2d
from shared.sensing.detection.reference import (
    ca_cfar_false_alarm_probability,
    ca_cfar_threshold_scale,
    os_cfar_false_alarm_probability,
    os_cfar_rank_index,
    os_cfar_threshold_scale,
    wilson_binomial_interval,
)


def test_cfar_1d_detects_peak() -> None:
    signal = np.ones(32, dtype=np.float64)
    signal[10] = 20.0
    result = cfar_1d(
        signal,
        guard_cells=1,
        reference_cells=4,
        threshold_scale=3.0,
        method="ca",
    )
    assert result.detections[10]
    assert not result.detections[5]
    assert result.eligible[10]
    assert result.training_cell_count[10] == 8


def test_cfar_1d_os_mode_uses_exact_ascending_rank() -> None:
    # The only full-window CUT is index 4. Its eight training values are 1..8.
    signal = np.array([1.0, 8.0, 3.0, 6.0, 100.0, 2.0, 7.0, 4.0, 5.0])
    rank_fraction = 0.5
    result = cfar_1d(
        signal,
        guard_cells=0,
        reference_cells=4,
        threshold_scale=1.5,
        method="os",
        os_rank=rank_fraction,
    )

    # round(0.5 * (8 - 1)) == 4, so the selected zero-based ascending
    # order statistic is the fifth value, 5.0.
    assert os_cfar_rank_index(8, rank_fraction) == 4
    assert result.eligible[4]
    assert result.training_cell_count[4] == 8
    assert result.noise[4] == 5.0
    assert result.threshold[4] == 7.5
    assert result.detections[4]


@pytest.mark.parametrize(
    ("training_values", "expected_noise"),
    [
        ([1.0, 6.0, 2.0, 5.0, 3.0, 4.0], 3.0),
        ([1.0, 8.0, 3.0, 6.0, 2.0, 7.0, 4.0, 5.0], 5.0),
    ],
)
def test_cfar_os_half_rank_uses_nearest_integer_with_ties_to_even(
    training_values: list[float],
    expected_noise: float,
) -> None:
    training_count = len(training_values)
    reference_cells = training_count // 2
    signal = np.asarray(
        [
            *training_values[:reference_cells],
            100.0,
            *training_values[reference_cells:],
        ],
        dtype=np.float64,
    )

    result = cfar_1d(
        signal,
        guard_cells=0,
        reference_cells=reference_cells,
        threshold_scale=1.0,
        method="os",
        os_rank=0.5,
    )

    center = reference_cells
    assert result.training_cell_count[center] == training_count
    assert result.noise[center] == expected_noise


def test_cfar_2d_detects_strong_full_window_cell() -> None:
    rd = np.ones((9, 9), dtype=np.float64)
    rd[4, 5] = 50.0
    result = cfar_2d(
        rd,
        guard_cells=(1, 1),
        reference_cells=(2, 2),
        threshold_scale=2.0,
        method="ca",
    )
    assert result.detections[4, 5]
    assert not result.detections[0, 0]
    assert result.eligible[4, 5]
    # Complete 7x7 outer window minus the 3x3 guard-plus-CUT block.
    assert result.training_cell_count[4, 5] == 40
    assert not result.eligible[0, 0]
    assert result.training_cell_count[0, 0] == 0


@pytest.mark.parametrize(
    ("method", "expected_noise"),
    [("ca", 4.5), ("os", 5.0)],
)
def test_cfar_2d_uses_exact_rectangular_ring_statistic(
    method: str,
    expected_noise: float,
) -> None:
    # At the center CUT, the complete 3x3 ring contains exactly values 1..8.
    power = np.array(
        [
            [1.0, 8.0, 3.0],
            [6.0, 100.0, 2.0],
            [7.0, 4.0, 5.0],
        ]
    )
    result = cfar_2d(
        power,
        guard_cells=(0, 0),
        reference_cells=(1, 1),
        threshold_scale=2.0,
        method=method,
        os_rank=0.5,
    )

    assert result.eligible[1, 1]
    assert result.training_cell_count[1, 1] == 8
    assert result.noise[1, 1] == expected_noise
    assert result.threshold[1, 1] == 2.0 * expected_noise
    assert result.detections[1, 1]


def test_cfar_2d_handles_edges_when_truncation_is_explicit() -> None:
    rd = np.ones((4, 16), dtype=np.float64)
    rd[0, 3] = 20.0
    result = cfar_2d(
        rd,
        guard_cells=(1, 1),
        reference_cells=(2, 4),
        threshold_scale=1.5,
        method="ca",
        edge_policy="truncate",
    )
    assert result.detections[0, 3]
    assert result.eligible[0, 3]
    # Clipped 4x9 outer window minus the clipped 2x3 guard block.
    assert result.training_cell_count[0, 3] == 30
    assert result.noise[0, 3] == 1.0


def test_cfar_1d_edge_policy_controls_eligibility_and_exact_count() -> None:
    signal = np.ones(9, dtype=np.float64)
    signal[0] = 10.0

    full = cfar_1d(
        signal,
        guard_cells=1,
        reference_cells=2,
        threshold_scale=2.0,
    )
    truncated = cfar_1d(
        signal,
        guard_cells=1,
        reference_cells=2,
        threshold_scale=2.0,
        edge_policy="truncate",
    )

    assert not full.eligible[0]
    assert full.training_cell_count[0] == 0
    assert not full.detections[0]
    assert truncated.eligible[0]
    assert truncated.training_cell_count[0] == 2
    assert truncated.noise[0] == 1.0
    assert truncated.threshold[0] == 2.0
    assert truncated.detections[0]


@pytest.mark.parametrize("method", ["ca", "os"])
def test_cfar_2d_edge_policy_controls_rectangular_training_ring(method: str) -> None:
    power = np.ones((5, 5), dtype=np.float64)
    power[0, 0] = 10.0

    full = cfar_2d(
        power,
        guard_cells=(0, 0),
        reference_cells=(1, 1),
        threshold_scale=2.0,
        method=method,
        os_rank=0.5,
    )
    truncated = cfar_2d(
        power,
        guard_cells=(0, 0),
        reference_cells=(1, 1),
        threshold_scale=2.0,
        method=method,
        os_rank=0.5,
        edge_policy="truncate",
    )

    assert full.eligible[2, 2]
    assert full.training_cell_count[2, 2] == 8
    assert not full.eligible[0, 0]
    assert full.training_cell_count[0, 0] == 0
    assert truncated.eligible[0, 0]
    assert truncated.training_cell_count[0, 0] == 3
    assert truncated.noise[0, 0] == 1.0
    assert truncated.detections[0, 0]


@pytest.mark.parametrize("method", ["ca", "os"])
def test_threshold_scale_is_raw_linear_multiplier_with_strict_cut_comparison(
    method: str,
) -> None:
    equal_to_threshold = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    above_threshold = equal_to_threshold.copy()
    above_threshold[1] = np.nextafter(2.0, np.inf)

    equal_result = cfar_1d(
        equal_to_threshold,
        guard_cells=0,
        reference_cells=1,
        threshold_scale=2.0,
        method=method,
        os_rank=0.5,
    )
    above_result = cfar_1d(
        above_threshold,
        guard_cells=0,
        reference_cells=1,
        threshold_scale=2.0,
        method=method,
        os_rank=0.5,
    )

    assert equal_result.noise[1] == 1.0
    assert equal_result.threshold[1] == 2.0
    assert not equal_result.detections[1]
    assert above_result.detections[1]


@pytest.mark.parametrize(
    ("power", "match"),
    [
        (np.ones((2, 2)), "expects a 1D array"),
        (np.array([True, False]), "real numeric"),
        (np.array(["1.0", "2.0"]), "real numeric"),
        (np.array([1.0 + 0.0j, 2.0 + 0.0j]), "real numeric"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, -1.0]), "nonnegative"),
    ],
)
def test_cfar_1d_rejects_invalid_power(power: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cfar_1d(
            power,
            guard_cells=0,
            reference_cells=1,
            threshold_scale=1.0,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"guard_cells": -1}, "guard_cells"),
        ({"guard_cells": 1.0}, "guard_cells"),
        ({"reference_cells": 0}, "reference_cells"),
        ({"reference_cells": True}, "reference_cells"),
        ({"threshold_scale": -1.0}, "threshold_scale"),
        ({"threshold_scale": np.nan}, "threshold_scale"),
        ({"threshold_scale": "1.0"}, "threshold_scale"),
        ({"method": "cell_averaging"}, "method"),
        ({"method": None}, "method"),
        ({"os_rank": -0.01}, "os_rank"),
        ({"os_rank": 1.01}, "os_rank"),
        ({"os_rank": "0.75"}, "os_rank"),
        ({"edge_policy": "implicit"}, "edge_policy"),
        ({"edge_policy": None}, "edge_policy"),
    ],
)
def test_cfar_1d_rejects_invalid_parameters(
    overrides: dict[str, object],
    match: str,
) -> None:
    parameters: dict[str, object] = {
        "guard_cells": 0,
        "reference_cells": 1,
        "threshold_scale": 1.0,
        "method": "ca",
        "os_rank": 0.75,
        "edge_policy": "full_window",
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=match):
        cfar_1d(np.ones(5), **parameters)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"guard_cells": [0, 0]}, "two-item"),
        ({"guard_cells": (-1, 0)}, "guard_cells"),
        ({"reference_cells": (1, 0)}, "reference_cells"),
        ({"reference_cells": (1.0, 1)}, "reference_cells"),
    ],
)
def test_cfar_2d_rejects_invalid_cell_pairs(
    overrides: dict[str, object],
    match: str,
) -> None:
    parameters: dict[str, object] = {
        "guard_cells": (0, 0),
        "reference_cells": (1, 1),
        "threshold_scale": 1.0,
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=match):
        cfar_2d(np.ones((5, 5)), **parameters)


@pytest.mark.parametrize(
    ("power", "match"),
    [
        (np.array([[1.0, np.nan]]), "finite"),
        (np.array([[1.0, -1.0]]), "nonnegative"),
        (np.array([[1.0 + 0.0j]]), "real numeric"),
    ],
)
def test_cfar_2d_rejects_invalid_power(power: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cfar_2d(
            power,
            guard_cells=(0, 0),
            reference_cells=(1, 1),
            threshold_scale=1.0,
        )


@pytest.mark.parametrize("cfar", [cfar_1d, cfar_2d])
def test_ca_cfar_rejects_float64_accumulator_overflow(cfar: object) -> None:
    huge = np.finfo(np.float64).max / 2.0
    if cfar is cfar_1d:
        with pytest.raises(ValueError, match="sums exceed finite"):
            cfar_1d(
                np.full(5, huge),
                guard_cells=0,
                reference_cells=1,
                threshold_scale=1.0,
            )
    else:
        with pytest.raises(ValueError, match="sums exceed finite"):
            cfar_2d(
                np.full((3, 3), huge),
                guard_cells=(0, 0),
                reference_cells=(1, 1),
                threshold_scale=1.0,
            )


@pytest.mark.parametrize(
    ("mask", "power", "radius", "match"),
    [
        (np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2)), (1, 1), "boolean"),
        (np.zeros((2, 2), dtype=bool), np.ones((2, 3)), (1, 1), "identical shapes"),
        (
            np.zeros((2, 2), dtype=bool),
            np.ones((2, 2), dtype=np.complex128),
            (1, 1),
            "real numeric",
        ),
        (np.zeros((2, 2), dtype=bool), np.ones((2, 2)), (-1, 1), "suppression_radius"),
    ],
)
def test_nms_validates_inputs_even_when_detection_mask_is_empty(
    mask: np.ndarray,
    power: np.ndarray,
    radius: tuple[int, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        nms_peaks_2d(mask, power, suppression_radius=radius)


def test_cluster_uses_an_actual_detected_peak_for_power_and_snr() -> None:
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 1] = True
    mask[1, 3] = True
    power = np.zeros(mask.shape, dtype=np.float64)
    power[1, 1] = 10.0
    power[1, 3] = 6.0
    snr = np.full(mask.shape, -300.0)
    snr[1, 1] = 12.0
    snr[1, 3] = 7.0
    snr_valid = np.zeros(mask.shape, dtype=bool)
    snr_valid[mask] = True

    detections = cluster_detections_2d(
        mask,
        power,
        np.arange(5, dtype=np.float64),
        np.arange(-1, 2, dtype=np.float64),
        wavelength=1.0,
        snr_map=snr,
        snr_valid_mask=snr_valid,
        cluster_range_gate_m=3.0,
        cluster_velocity_gate_m_s=1.0,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.range_bin == 1
    assert detection.doppler_bin == 1
    assert detection.power_linear == 10.0
    assert detection.snr_db == 12.0
    assert detection.centroid_range_m == pytest.approx(1.75)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"wavelength": 0.0}, "wavelength"),
        ({"cluster_range_gate_m": 0.0}, "cluster_range_gate_m"),
        ({"cluster_velocity_gate_m_s": np.nan}, "cluster_velocity_gate_m_s"),
        ({"range_axis_m": np.array([0.0, 0.0])}, "strictly increasing"),
        ({"snr_valid_mask": np.ones((2, 2), dtype=np.uint8)}, "boolean"),
    ],
)
def test_cluster_validates_scientific_inputs_before_empty_return(
    overrides: dict[str, object],
    match: str,
) -> None:
    parameters: dict[str, object] = {
        "detections_mask": np.zeros((2, 2), dtype=bool),
        "power_map": np.ones((2, 2)),
        "range_axis_m": np.array([0.0, 1.0]),
        "doppler_axis_hz": np.array([-1.0, 1.0]),
        "wavelength": 1.0,
        "snr_map": np.ones((2, 2)),
        "snr_valid_mask": np.ones((2, 2), dtype=bool),
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=match):
        cluster_detections_2d(**parameters)


def _independent_window_centers(trials: int, *, guard: int, reference: int) -> np.ndarray:
    block_size = 2 * (guard + reference) + 1
    return np.arange(trials, dtype=np.int64) * block_size + guard + reference


def test_ca_cfar_seeded_production_rate_matches_independent_analytic_reference() -> None:
    rng = np.random.default_rng(20260830)
    trials = 30_000
    guard = 1
    reference = 4
    training_cells = 2 * reference
    requested_pfa = 0.02
    threshold_scale = ca_cfar_threshold_scale(requested_pfa, training_cells)
    centers = _independent_window_centers(trials, guard=guard, reference=reference)
    block_size = 2 * (guard + reference) + 1
    power = rng.exponential(size=trials * block_size)

    result = cfar_1d(
        power,
        guard_cells=guard,
        reference_cells=reference,
        threshold_scale=threshold_scale,
        method="ca",
    )
    false_alarms = int(np.count_nonzero(result.detections[centers]))
    lower, upper = wilson_binomial_interval(false_alarms, trials, confidence=0.999)

    assert np.all(result.eligible[centers])
    assert np.all(result.training_cell_count[centers] == training_cells)
    assert ca_cfar_false_alarm_probability(
        threshold_scale,
        training_cells,
    ) == pytest.approx(requested_pfa, rel=1e-12)
    assert lower <= requested_pfa <= upper


def test_os_cfar_seeded_production_rate_matches_independent_analytic_reference() -> None:
    rng = np.random.default_rng(20260831)
    trials = 10_000
    guard = 1
    reference = 4
    training_cells = 2 * reference
    rank_fraction = 0.75
    rank_index = os_cfar_rank_index(training_cells, rank_fraction)
    requested_pfa = 0.1
    threshold_scale = os_cfar_threshold_scale(
        requested_pfa,
        training_cells,
        rank_index,
    )
    centers = _independent_window_centers(trials, guard=guard, reference=reference)
    block_size = 2 * (guard + reference) + 1
    power = rng.exponential(size=trials * block_size)

    result = cfar_1d(
        power,
        guard_cells=guard,
        reference_cells=reference,
        threshold_scale=threshold_scale,
        method="os",
        os_rank=rank_fraction,
    )
    false_alarms = int(np.count_nonzero(result.detections[centers]))
    lower, upper = wilson_binomial_interval(false_alarms, trials, confidence=0.999)

    assert np.all(result.eligible[centers])
    assert np.all(result.training_cell_count[centers] == training_cells)
    assert os_cfar_false_alarm_probability(
        threshold_scale,
        training_cells,
        rank_index,
    ) == pytest.approx(requested_pfa, rel=1e-11)
    assert lower <= requested_pfa <= upper
