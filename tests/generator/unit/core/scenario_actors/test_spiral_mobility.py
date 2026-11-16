"""Exact helix geometry, traversal semantics, and retained polyline defaults."""

import math

import numpy as np
import pytest

from generator.core.scenario_actors import PosePreparationError, Timeline, prepare_mobility
from shared.scenarios.actors import SpiralMobilitySpec


def _spec(**changes):
    return SpiralMobilitySpec(
        **{
            "center_m": (3, -2, 99),
            "radius_m": 2,
            "start_altitude_m": 5,
            "end_altitude_m": 1,
            "turns": 1.25,
            "start_angle_deg": 31,
            "interpolation": "analytic",
            **changes,
        }
    )


@pytest.mark.parametrize("clockwise", [False, True])
def test_analytic_spiral_follows_exact_curve_and_absolute_altitudes(clockwise):
    spec = _spec(clockwise=clockwise)
    timeline = Timeline(steps=138, duration_s=3.7)
    positions = np.asarray(prepare_mobility(spec, timeline).positions_m)
    fraction = np.asarray(timeline.timestamps_s) / timeline.duration_s
    direction = -1 if clockwise else 1
    complex_expected = 2 * np.exp(1j * (math.radians(31) + direction * 2.5 * math.pi * fraction))
    np.testing.assert_allclose(positions[:, 0], 3 + complex_expected.real, atol=1e-14, rtol=0)
    np.testing.assert_allclose(positions[:, 1], -2 + complex_expected.imag, atol=1e-14, rtol=0)
    np.testing.assert_allclose(positions[:, 2], 5 - 4 * fraction, atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(positions[:, :2] - (3, -2), axis=1), 2, atol=1e-15)
    assert positions[0, 2] == 5 and positions[-1, 2] == 1


def test_constant_speed_spiral_has_prefix_stable_positions_and_smooth_velocity():
    spec = _spec(traversal={"type": "constant_speed", "speed_mps": 1.1})
    short = Timeline(steps=512, duration_s=511 / 2000)
    full = Timeline(steps=8192, duration_s=8191 / 2000)
    motion = prepare_mobility(spec, full)
    positions = np.asarray(motion.positions_m)
    assert np.array_equal(prepare_mobility(spec, short).positions_m, positions[:512])
    velocities = np.asarray(motion.velocities_mps)
    np.testing.assert_allclose(np.linalg.norm(velocities, axis=1), 1.1, atol=4e-8, rtol=0)
    acceleration = np.gradient(velocities, full.timestamps_s, axis=0, edge_order=2)
    # A smooth helix has constant curvature. Polygon vertices create spikes here.
    horizontal_acceleration = np.linalg.norm(acceleration[2:-2, :2], axis=1)
    assert np.ptp(horizontal_acceleration) < 1e-7
    assert motion.has_physical_velocity


@pytest.mark.parametrize(
    ("mode", "fractions"),
    [
        ("hold", [0, 0.5, 1, 1, 1, 1, 1]),
        ("loop", [0, 0.5, 0, 0.5, 0, 0.5, 0]),
        ("ping_pong", [0, 0.5, 1, 0.5, 0, 0.5, 1]),
    ],
)
def test_analytic_spiral_preserves_after_end_traversal(mode, fractions):
    # A three metre quarter-turn arc and four metre climb: exact length 5 m.
    # This also puts the loop boundaries on exactly representable distances.
    radius = 6 / math.pi
    length = 5
    spec = _spec(
        center_m=(0, 0, 999),
        radius_m=radius,
        start_altitude_m=1,
        end_altitude_m=5,
        turns=0.25,
        start_angle_deg=0,
        traversal={"type": "constant_speed", "speed_mps": length, "after_end": mode},
    )
    positions = np.asarray(prepare_mobility(spec, Timeline(steps=7, duration_s=3)).positions_m)
    fraction = np.asarray(fractions)
    np.testing.assert_allclose(positions[:, 0], radius * np.cos(fraction * math.pi / 2), atol=2e-15)
    np.testing.assert_allclose(positions[:, 1], radius * np.sin(fraction * math.pi / 2), atol=2e-15)
    np.testing.assert_allclose(positions[:, 2], 1 + 4 * fraction, atol=2e-15)


def test_default_spiral_preserves_linear_polyline_samples():
    config = {
        "type": "spiral",
        "center_m": [1, -2, 88],
        "radius_m": 2,
        "start_altitude_m": 1,
        "end_altitude_m": 4,
        "turns": 1,
    }
    timeline = Timeline(steps=18, duration_s=3)
    default = prepare_mobility(config, timeline)
    explicit = prepare_mobility({**config, "interpolation": "linear"}, timeline)
    assert default == explicit
    assert SpiralMobilitySpec(**config).interpolation == "linear"
    # Existing default interpolates 128 straight segments per turn, not the curve.
    dense_fraction = np.linspace(0, 1, 129)
    dense = np.column_stack(
        (
            1 + 2 * np.cos(2 * np.pi * dense_fraction),
            -2 + 2 * np.sin(2 * np.pi * dense_fraction),
            1 + 3 * dense_fraction,
        )
    )
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    sampled_distance = distances[-1] * np.asarray(timeline.timestamps_s) / 3
    expected = np.column_stack(
        [np.interp(sampled_distance, distances, dense[:, axis]) for axis in range(3)]
    )
    np.testing.assert_allclose(default.positions_m, expected, rtol=0, atol=1e-14)
    analytic = prepare_mobility({**config, "interpolation": "analytic"}, timeline)
    assert np.max(np.abs(np.asarray(default.positions_m) - analytic.positions_m)) > 1e-4


def test_unvalidated_spiral_rejects_unknown_interpolation():
    spec = _spec().model_dump()
    spec["interpolation"] = "cubic"
    with pytest.raises(PosePreparationError, match="linear or analytic"):
        prepare_mobility(spec, Timeline(steps=8, duration_s=1))
