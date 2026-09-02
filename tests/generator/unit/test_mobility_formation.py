"""Tests for FormationMobility."""

from __future__ import annotations

import numpy as np

from generator.core.mobility import LinearMobility, WaypointMobility
from generator.core.mobility.formation import FormationMobility

STEPS = 30
DURATION = 3.0


def _to_np(positions):
    """Convert tuple positions to a numpy array (N, 3)."""
    return np.asarray(positions, dtype=np.float64)


class TestFormationCorrectCount:
    def test_same_step_count_as_leader(self):
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 0, 0))
        follower = FormationMobility(leader_mobility=leader, offset=(5.0, 0.0, 0.0))
        pts = follower.get_positions(None, STEPS, DURATION)
        assert len(pts) == STEPS

    def test_single_step(self):
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 0, 0))
        follower = FormationMobility(leader_mobility=leader, offset=(5.0, 0.0, 0.0))
        pts = follower.get_positions(None, 1, DURATION)
        assert len(pts) == 1


class TestOffsetRotatesWithHeading:
    def test_east_then_north(self):
        """Leader walks east then turns north. Right-offset follower should shift accordingly."""
        # East segment → right = south direction
        # North segment → right = east direction
        wps = [
            (0, 0, 0),
            (20, 0, 0),  # walking east
            (20, 20, 0),  # walking north
        ]
        leader = WaypointMobility(waypoints=wps)
        right_offset = 5.0
        follower = FormationMobility(leader_mobility=leader, offset=(right_offset, 0.0, 0.0))
        pts = follower.get_positions(None, STEPS, DURATION)
        arr = _to_np(pts)

        # In the first half (walking east, heading ~ 0), right = (0, -1, 0)
        # so follower should be south of leader (y < leader_y)
        quarter = STEPS // 4
        leader_pts = _to_np(leader.get_positions((0, 0, 0), STEPS, DURATION))

        # Check early steps: follower Y should be less than leader Y (south)
        for i in range(1, quarter):
            assert (
                arr[i, 1] < leader_pts[i, 1]
            ), f"Step {i}: follower y={arr[i, 1]:.3f} should be < leader y={leader_pts[i, 1]:.3f}"

        # Check later steps (walking north, heading ~ pi/2): right = (1, 0, 0)
        # follower should be east of leader (x > leader_x)
        three_quarter = 3 * STEPS // 4
        for i in range(three_quarter, STEPS - 1):
            assert (
                arr[i, 0] > leader_pts[i, 0]
            ), f"Step {i}: follower x={arr[i, 0]:.3f} should be > leader x={leader_pts[i, 0]:.3f}"


class TestZeroOffsetMatchesLeader:
    def test_zero_offset_matches(self):
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 10, 5))
        follower = FormationMobility(leader_mobility=leader, offset=(0.0, 0.0, 0.0))
        leader_pts = _to_np(leader.get_positions((0, 0, 0), STEPS, DURATION))
        follower_pts = _to_np(follower.get_positions(None, STEPS, DURATION))
        np.testing.assert_allclose(follower_pts, leader_pts, atol=1e-10)


class TestNoiseAddsJitter:
    def test_noise_differs_from_clean(self):
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 0, 0))
        clean = FormationMobility(leader_mobility=leader, offset=(5.0, -3.0, 0.0), noise=0.0)
        noisy = FormationMobility(
            leader_mobility=leader, offset=(5.0, -3.0, 0.0), noise=1.0, seed=42
        )
        clean_pts = _to_np(clean.get_positions(None, STEPS, DURATION))
        noisy_pts = _to_np(noisy.get_positions(None, STEPS, DURATION))
        # Should not be identical
        assert not np.allclose(clean_pts, noisy_pts, atol=1e-10)
        # But should be close (within noise bound)
        diffs = np.linalg.norm(noisy_pts - clean_pts, axis=1)
        assert np.all(diffs <= 1.0 + 1e-6), f"Max diff {diffs.max():.4f} exceeds noise=1.0"


class TestDeterministicWithSeed:
    def test_same_seed_same_result(self):
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 0, 0))
        mob1 = FormationMobility(
            leader_mobility=leader, offset=(5.0, -3.0, 0.0), noise=0.5, seed=123
        )
        mob2 = FormationMobility(
            leader_mobility=leader, offset=(5.0, -3.0, 0.0), noise=0.5, seed=123
        )
        pts1 = _to_np(mob1.get_positions(None, STEPS, DURATION))
        pts2 = _to_np(mob2.get_positions(None, STEPS, DURATION))
        np.testing.assert_allclose(pts1, pts2, atol=1e-12)

    def test_different_seed_different_result(self):
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 0, 0))
        mob1 = FormationMobility(
            leader_mobility=leader, offset=(5.0, -3.0, 0.0), noise=0.5, seed=123
        )
        mob2 = FormationMobility(
            leader_mobility=leader, offset=(5.0, -3.0, 0.0), noise=0.5, seed=456
        )
        pts1 = _to_np(mob1.get_positions(None, STEPS, DURATION))
        pts2 = _to_np(mob2.get_positions(None, STEPS, DURATION))
        assert not np.allclose(pts1, pts2, atol=1e-10)


class TestForwardOffset:
    def test_forward_offset_along_heading(self):
        """With only a forward offset, follower should be ahead of leader."""
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 0, 0))
        follower = FormationMobility(leader_mobility=leader, offset=(0.0, 5.0, 0.0))
        leader_pts = _to_np(leader.get_positions((0, 0, 0), STEPS, DURATION))
        follower_pts = _to_np(follower.get_positions(None, STEPS, DURATION))
        # Walking east (heading=0): forward=(1,0,0), so follower x > leader x
        for i in range(1, STEPS - 1):
            assert (
                follower_pts[i, 0] > leader_pts[i, 0]
            ), f"Step {i}: follower x should be ahead of leader"
            np.testing.assert_allclose(follower_pts[i, 1], leader_pts[i, 1], atol=1e-10)


class TestUpOffset:
    def test_up_offset_vertical(self):
        """Up offset should add to z coordinate regardless of heading."""
        leader = LinearMobility(start_pos=(0, 0, 0), end_pos=(30, 10, 0))
        follower = FormationMobility(leader_mobility=leader, offset=(0.0, 0.0, 3.0))
        leader_pts = _to_np(leader.get_positions((0, 0, 0), STEPS, DURATION))
        follower_pts = _to_np(follower.get_positions(None, STEPS, DURATION))
        np.testing.assert_allclose(follower_pts[:, 2], leader_pts[:, 2] + 3.0, atol=1e-10)


class TestStartPos:
    def test_start_pos_property(self):
        leader = LinearMobility(start_pos=(10, 20, 5), end_pos=(30, 20, 5))
        follower = FormationMobility(leader_mobility=leader, offset=(3.0, 2.0, 1.0))
        sp = follower.start_pos
        assert sp is not None
        # At heading 0: forward=(1,0,0), right=(0,-1,0)
        # world offset = right*3 + forward*2 + up*1 = (0*3 + 1*2, -1*3 + 0*2, 1)
        # = (2, -3, 1) relative to leader start (10, 20, 5)
        np.testing.assert_allclose(sp, (12.0, 17.0, 6.0), atol=1e-10)
