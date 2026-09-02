"""Unit tests for mobility model math — verify positions, bounds, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from generator.core.mobility import MeshGridMobility
from generator.core.mobility.drone import (
    Figure8Mobility,
    ReturnToHomeMobility,
    SpiralMobility,
)
from generator.core.mobility.group import GroupMobility
from generator.core.mobility.micro import OscillatingMobility, PendulumMobility
from generator.core.mobility.standard import (
    GaussMarkovMobility,
    ManhattanGridMobility,
    RandomWaypointMobility,
)

STEPS = 30
DURATION = 3.0


def _to_np(positions):
    """Convert tuple positions to a numpy array (N, 3)."""
    return np.asarray(positions, dtype=np.float64)


# -----------------------------------------------------------------------
# Mesh grid
# -----------------------------------------------------------------------


class TestMeshGrid:
    def test_start_corner_controls_sequential_traversal(self):
        mob = MeshGridMobility(
            x_bounds=(0.0, 1.0),
            y_bounds=(10.0, 11.0),
            z_bounds=(1.5, 1.5),
            x_steps=2,
            y_steps=2,
            z_steps=1,
            traversal_pattern="sequential",
            start_corner="top_right",
        )

        pts = _to_np(mob.get_positions((0.0, 0.0, 0.0), 4, 1.0))
        np.testing.assert_array_equal(
            pts,
            np.array(
                [
                    [1.0, 11.0, 1.5],
                    [0.0, 11.0, 1.5],
                    [1.0, 10.0, 1.5],
                    [0.0, 10.0, 1.5],
                ]
            ),
        )

    def test_start_corner_controls_snake_traversal(self):
        mob = MeshGridMobility(
            x_bounds=(0.0, 1.0),
            y_bounds=(10.0, 11.0),
            z_bounds=(1.5, 1.5),
            x_steps=2,
            y_steps=2,
            z_steps=1,
            traversal_pattern="snake",
            start_corner="top_right",
        )

        pts = _to_np(mob.get_positions((0.0, 0.0, 0.0), 4, 1.0))
        np.testing.assert_array_equal(
            pts,
            np.array(
                [
                    [1.0, 11.0, 1.5],
                    [0.0, 11.0, 1.5],
                    [0.0, 10.0, 1.5],
                    [1.0, 10.0, 1.5],
                ]
            ),
        )


# -----------------------------------------------------------------------
# Gauss-Markov
# -----------------------------------------------------------------------


class TestGaussMarkov:
    def test_correct_count(self):
        mob = GaussMarkovMobility(start_pos=(0, 0, 0), seed=42)
        pts = mob.get_positions((0, 0, 0), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_starts_at_start_pos(self):
        mob = GaussMarkovMobility(start_pos=(10, 20, 1.5), seed=42)
        pts = _to_np(mob.get_positions((10, 20, 1.5), STEPS, DURATION))
        assert pts[0, 0] == pytest.approx(10, abs=0.01)
        assert pts[0, 1] == pytest.approx(20, abs=0.01)

    def test_respects_bounds(self):
        mob = GaussMarkovMobility(
            start_pos=(0, 0, 1.5),
            bounds=(-10, 10, -10, 10, 1.5, 1.5),
            seed=42,
            mean_speed=5.0,
        )
        pts = _to_np(mob.get_positions((0, 0, 1.5), 200, 20.0))
        assert pts[:, 0].min() >= -10.01
        assert pts[:, 0].max() <= 10.01
        assert pts[:, 1].min() >= -10.01
        assert pts[:, 1].max() <= 10.01

    def test_deterministic(self):
        mob1 = GaussMarkovMobility(start_pos=(0, 0, 0), seed=99)
        mob2 = GaussMarkovMobility(start_pos=(0, 0, 0), seed=99)
        p1 = _to_np(mob1.get_positions((0, 0, 0), STEPS, DURATION))
        p2 = _to_np(mob2.get_positions((0, 0, 0), STEPS, DURATION))
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_path(self):
        mob1 = GaussMarkovMobility(start_pos=(0, 0, 0), seed=1)
        mob2 = GaussMarkovMobility(start_pos=(0, 0, 0), seed=2)
        p1 = _to_np(mob1.get_positions((0, 0, 0), STEPS, DURATION))
        p2 = _to_np(mob2.get_positions((0, 0, 0), STEPS, DURATION))
        assert not np.allclose(p1, p2)


# -----------------------------------------------------------------------
# Random Waypoint
# -----------------------------------------------------------------------


class TestRandomWaypoint:
    def test_correct_count(self):
        mob = RandomWaypointMobility(start_pos=(0, 0, 0), seed=42)
        pts = mob.get_positions((0, 0, 0), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_starts_at_start_pos(self):
        mob = RandomWaypointMobility(start_pos=(5, 5, 1.5), seed=42)
        pts = _to_np(mob.get_positions((5, 5, 1.5), STEPS, DURATION))
        assert pts[0, 0] == pytest.approx(5, abs=0.01)

    def test_deterministic(self):
        mob1 = RandomWaypointMobility(start_pos=(0, 0, 0), seed=77)
        mob2 = RandomWaypointMobility(start_pos=(0, 0, 0), seed=77)
        p1 = _to_np(mob1.get_positions((0, 0, 0), STEPS, DURATION))
        p2 = _to_np(mob2.get_positions((0, 0, 0), STEPS, DURATION))
        np.testing.assert_array_equal(p1, p2)


# -----------------------------------------------------------------------
# Manhattan Grid
# -----------------------------------------------------------------------


class TestManhattanGrid:
    def test_correct_count(self):
        mob = ManhattanGridMobility(start_pos=(0, 0, 1.5), seed=42)
        pts = mob.get_positions((0, 0, 1.5), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_fixed_altitude(self):
        mob = ManhattanGridMobility(start_pos=(0, 0, 0), altitude=2.0, seed=42)
        pts = _to_np(mob.get_positions((0, 0, 0), STEPS, DURATION))
        np.testing.assert_array_almost_equal(pts[:, 2], 2.0)

    def test_deterministic(self):
        mob1 = ManhattanGridMobility(start_pos=(0, 0, 0), seed=55)
        mob2 = ManhattanGridMobility(start_pos=(0, 0, 0), seed=55)
        p1 = _to_np(mob1.get_positions((0, 0, 0), STEPS, DURATION))
        p2 = _to_np(mob2.get_positions((0, 0, 0), STEPS, DURATION))
        np.testing.assert_array_equal(p1, p2)


# -----------------------------------------------------------------------
# Oscillating
# -----------------------------------------------------------------------


class TestOscillating:
    def test_correct_count(self):
        mob = OscillatingMobility(center=(0, 0, 0))
        pts = mob.get_positions((0, 0, 0), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_oscillates_around_center(self):
        mob = OscillatingMobility(center=(5, 0, 0), amplitude=1.0, frequency_hz=1.0, axis=(1, 0, 0))
        pts = _to_np(mob.get_positions((5, 0, 0), 100, 2.0))
        # Mean should be near center
        assert pts[:, 0].mean() == pytest.approx(5.0, abs=0.2)
        # Max displacement should be near amplitude
        assert pts[:, 0].max() <= 6.01
        assert pts[:, 0].min() >= 3.99

    def test_y_z_unchanged_for_x_axis(self):
        mob = OscillatingMobility(center=(0, 5, 10), axis=(1, 0, 0))
        pts = _to_np(mob.get_positions((0, 5, 10), STEPS, DURATION))
        np.testing.assert_array_almost_equal(pts[:, 1], 5.0)
        np.testing.assert_array_almost_equal(pts[:, 2], 10.0)


# -----------------------------------------------------------------------
# Pendulum
# -----------------------------------------------------------------------


class TestPendulum:
    def test_correct_count(self):
        mob = PendulumMobility(pivot=(0, 0, 2))
        pts = mob.get_positions((0, 0, 2), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_stays_near_pivot(self):
        mob = PendulumMobility(pivot=(0, 0, 2), length=0.5, max_angle_deg=30)
        pts = _to_np(mob.get_positions((0, 0, 2), STEPS, DURATION))
        # All points should be within length distance from pivot
        distances = np.sqrt(pts[:, 0] ** 2 + (pts[:, 2] - 2) ** 2)
        assert distances.max() <= 0.51

    def test_xz_plane(self):
        mob = PendulumMobility(pivot=(0, 5, 2), plane="xz")
        pts = _to_np(mob.get_positions((0, 5, 2), STEPS, DURATION))
        # Y should stay constant
        np.testing.assert_array_almost_equal(pts[:, 1], 5.0)


# -----------------------------------------------------------------------
# Figure-8
# -----------------------------------------------------------------------


class TestFigure8:
    def test_correct_count(self):
        mob = Figure8Mobility(center=(0, 0, 20))
        pts = mob.get_positions((0, 0, 20), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_fixed_altitude(self):
        mob = Figure8Mobility(center=(10, 10, 30), size=20)
        pts = _to_np(mob.get_positions((10, 10, 30), STEPS, DURATION))
        np.testing.assert_array_almost_equal(pts[:, 2], 30.0)

    def test_centered(self):
        mob = Figure8Mobility(center=(5, 5, 20), size=30)
        pts = _to_np(mob.get_positions((5, 5, 20), 100, 10.0))
        # Mean should be near center
        assert pts[:, 0].mean() == pytest.approx(5.0, abs=1.0)
        assert pts[:, 1].mean() == pytest.approx(5.0, abs=1.0)


# -----------------------------------------------------------------------
# Spiral
# -----------------------------------------------------------------------


class TestSpiral:
    def test_correct_count(self):
        mob = SpiralMobility(center=(0, 0, 0))
        pts = mob.get_positions((0, 0, 0), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_altitude_range(self):
        mob = SpiralMobility(center=(0, 0, 0), start_altitude=10, end_altitude=50)
        pts = _to_np(mob.get_positions((0, 0, 0), STEPS, DURATION))
        assert pts[0, 2] == pytest.approx(10, abs=0.1)
        assert pts[-1, 2] == pytest.approx(50, abs=0.1)

    def test_radius(self):
        mob = SpiralMobility(center=(0, 0, 0), radius=25)
        pts = _to_np(mob.get_positions((0, 0, 0), STEPS, DURATION))
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        np.testing.assert_array_almost_equal(radii, 25.0, decimal=1)


# -----------------------------------------------------------------------
# Return to Home
# -----------------------------------------------------------------------


class TestReturnToHome:
    def test_correct_count(self):
        mob = ReturnToHomeMobility(waypoints=[(0, 0, 20), (50, 50, 25), (100, 0, 20)])
        pts = mob.get_positions((0, 0, 20), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_starts_at_first_waypoint(self):
        mob = ReturnToHomeMobility(waypoints=[(10, 20, 30), (50, 50, 25)])
        pts = _to_np(mob.get_positions((10, 20, 30), STEPS, DURATION))
        assert pts[0, 0] == pytest.approx(10, abs=0.1)
        assert pts[0, 1] == pytest.approx(20, abs=0.1)

    def test_returns_to_home(self):
        mob = ReturnToHomeMobility(waypoints=[(0, 0, 20), (50, 0, 20)])
        pts = _to_np(mob.get_positions((0, 0, 20), 100, 10.0))
        # Last point should be back near home
        assert pts[-1, 0] == pytest.approx(0, abs=1.0)
        assert pts[-1, 1] == pytest.approx(0, abs=1.0)


# -----------------------------------------------------------------------
# Group (RPGM)
# -----------------------------------------------------------------------


class TestGroupMobility:
    def test_correct_count(self):
        from generator.core.mobility import LinearMobility

        ref = LinearMobility(start_pos=(0, 0, 0), end_pos=(10, 0, 0))
        mob = GroupMobility(reference_mobility=ref, max_deviation=2.0, seed=42)
        pts = mob.get_positions((0, 0, 0), STEPS, DURATION)
        assert len(pts) == STEPS

    def test_stays_near_reference(self):
        from generator.core.mobility import LinearMobility

        ref = LinearMobility(start_pos=(0, 0, 0), end_pos=(100, 0, 0))
        mob = GroupMobility(reference_mobility=ref, max_deviation=3.0, seed=42)
        ref_pts = _to_np(ref.get_positions((0, 0, 0), STEPS, DURATION))
        grp_pts = _to_np(mob.get_positions((0, 0, 0), STEPS, DURATION))
        deviations = np.linalg.norm(grp_pts - ref_pts, axis=1)
        assert deviations.max() <= 3.01

    def test_deterministic(self):
        from generator.core.mobility import LinearMobility

        ref = LinearMobility(start_pos=(0, 0, 0), end_pos=(10, 0, 0))
        mob1 = GroupMobility(reference_mobility=ref, seed=11)
        mob2 = GroupMobility(reference_mobility=ref, seed=11)
        p1 = _to_np(mob1.get_positions((0, 0, 0), STEPS, DURATION))
        p2 = _to_np(mob2.get_positions((0, 0, 0), STEPS, DURATION))
        np.testing.assert_array_equal(p1, p2)
