"""Tests for shared.sensing.evaluation — GT-to-detection association and MOT metrics."""

import numpy as np
import pytest

from shared.sensing.evaluation import (
    AssociationResult,
    MOTMetrics,
    associate_detections_to_gt,
    compute_mot_metrics,
    evaluate_scenario,
)


class TestAssociateDetectionsToGt:
    """Tests for the Hungarian-assignment-based GT association."""

    def test_perfect_match(self):
        """One GT and one detection at the same location."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([50.0]),
            gt_velocities=np.array([5.0]),
            det_ranges=np.array([50.0]),
            det_velocities=np.array([5.0]),
        )
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.range_rmse == pytest.approx(0.0)
        assert result.velocity_rmse == pytest.approx(0.0)
        assert result.gt_to_det[0] == 0
        assert result.det_to_gt[0] == 0

    def test_offset_within_gate(self):
        """Detection offset from GT but within default gates."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([50.0]),
            gt_velocities=np.array([5.0]),
            det_ranges=np.array([52.0]),
            det_velocities=np.array([4.5]),
        )
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.range_rmse == pytest.approx(2.0, abs=0.01)
        assert result.velocity_rmse == pytest.approx(0.5, abs=0.01)
        assert len(result.range_errors) == 1
        assert result.range_errors[0] == pytest.approx(-2.0, abs=0.01)
        assert result.velocity_errors[0] == pytest.approx(0.5, abs=0.01)

    def test_outside_gate(self):
        """Detection too far from GT — should not match."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([50.0]),
            gt_velocities=np.array([5.0]),
            det_ranges=np.array([100.0]),
            det_velocities=np.array([20.0]),
        )
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1
        assert result.gt_to_det[0] is None
        assert result.det_to_gt[0] is None
        assert result.range_rmse is None
        assert result.velocity_rmse is None

    def test_multiple_targets(self):
        """Three GTs, two detections: one matches, one is FP, two are FN."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([50.0, 100.0, 200.0]),
            gt_velocities=np.array([5.0, -3.0, 8.0]),
            det_ranges=np.array([51.0, 300.0]),
            det_velocities=np.array([5.5, 0.0]),
        )
        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 2
        # The first GT (50, 5) should match the first detection (51, 5.5)
        assert result.gt_to_det[0] == 0
        assert result.gt_to_det[1] is None
        assert result.gt_to_det[2] is None

    def test_empty_gt(self):
        """No GT targets — all detections are false positives."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([]),
            gt_velocities=np.array([]),
            det_ranges=np.array([50.0, 80.0]),
            det_velocities=np.array([3.0, -2.0]),
        )
        assert result.true_positives == 0
        assert result.false_positives == 2
        assert result.false_negatives == 0
        assert len(result.gt_to_det) == 0
        assert len(result.det_to_gt) == 2

    def test_empty_detections(self):
        """No detections — all GT are missed."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([50.0, 100.0]),
            gt_velocities=np.array([5.0, -3.0]),
            det_ranges=np.array([]),
            det_velocities=np.array([]),
        )
        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 2
        assert len(result.gt_to_det) == 2
        assert all(v is None for v in result.gt_to_det.values())

    def test_empty_both(self):
        """No GT and no detections — trivial case."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([]),
            gt_velocities=np.array([]),
            det_ranges=np.array([]),
            det_velocities=np.array([]),
        )
        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.range_rmse is None
        assert result.velocity_rmse is None

    def test_custom_gates(self):
        """Tight gates should reject matches that default gates would accept."""
        # On the inclusive default boundary (±2 m, ±2 m/s) this matches
        result_default = associate_detections_to_gt(
            gt_ranges=np.array([50.0]),
            gt_velocities=np.array([5.0]),
            det_ranges=np.array([52.0]),
            det_velocities=np.array([3.0]),
        )
        assert result_default.true_positives == 1

        # With tight gates (2m, 1 m/s) this should NOT match
        result_tight = associate_detections_to_gt(
            gt_ranges=np.array([50.0]),
            gt_velocities=np.array([5.0]),
            det_ranges=np.array([52.0]),
            det_velocities=np.array([3.0]),
            range_gate=2.0,
            velocity_gate=1.0,
        )
        assert result_tight.true_positives == 0
        assert result_tight.false_positives == 1
        assert result_tight.false_negatives == 1

    def test_result_is_dataclass(self):
        """Verify the result is a proper AssociationResult dataclass."""
        result = associate_detections_to_gt(
            gt_ranges=np.array([50.0]),
            gt_velocities=np.array([5.0]),
            det_ranges=np.array([50.0]),
            det_velocities=np.array([5.0]),
        )
        assert isinstance(result, AssociationResult)
        assert hasattr(result, "gt_to_det")
        assert hasattr(result, "det_to_gt")
        assert hasattr(result, "range_rmse")
        assert hasattr(result, "velocity_rmse")

    def test_two_targets_two_detections_optimal(self):
        """Hungarian should find optimal assignment, not greedy."""
        # GT: (10, 1) and (20, 2)
        # Det: (11, 1.5) and (19, 2.5)
        # Optimal: GT0→Det0, GT1→Det1
        result = associate_detections_to_gt(
            gt_ranges=np.array([10.0, 20.0]),
            gt_velocities=np.array([1.0, 2.0]),
            det_ranges=np.array([11.0, 19.0]),
            det_velocities=np.array([1.5, 2.5]),
        )
        assert result.true_positives == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.gt_to_det[0] == 0
        assert result.gt_to_det[1] == 1

    def test_hungarian_cost_is_dimensionless_and_gate_normalized(self):
        """Gate scaling, rather than incompatible physical units, sets assignment cost."""

        result = associate_detections_to_gt(
            gt_ranges=np.array([0.0, 10.0]),
            gt_velocities=np.array([0.0, 1.0]),
            det_ranges=np.array([0.0, 1.0]),
            det_velocities=np.array([0.25, 0.0]),
            range_gate=10.0,
            velocity_gate=1.0,
        )

        # Normalized costs choose GT0->Det1 and GT1->Det0. A raw Euclidean
        # mixture of metres and m/s would choose the opposite assignment.
        assert result.gt_to_det == {0: 1, 1: 0}
        assert result.det_to_gt == {0: 1, 1: 0}

    def test_rejects_unaligned_or_nonfinite_inputs(self):
        with pytest.raises(ValueError, match="equal length"):
            associate_detections_to_gt(
                gt_ranges=np.array([1.0, 2.0]),
                gt_velocities=np.array([0.0]),
                det_ranges=np.array([]),
                det_velocities=np.array([]),
            )
        with pytest.raises(ValueError, match="finite values"):
            associate_detections_to_gt(
                gt_ranges=np.array([np.nan]),
                gt_velocities=np.array([0.0]),
                det_ranges=np.array([]),
                det_velocities=np.array([]),
            )

    @pytest.mark.parametrize("gate", [0.0, -1.0, np.nan, np.inf])
    def test_requires_finite_positive_gates(self, gate):
        with pytest.raises(ValueError, match="finite and positive"):
            associate_detections_to_gt(
                gt_ranges=np.array([]),
                gt_velocities=np.array([]),
                det_ranges=np.array([]),
                det_velocities=np.array([]),
                range_gate=gate,
            )

    def test_scenario_evaluation_never_truncates_frame_sequences(self):
        with pytest.raises(ValueError, match="never truncates events"):
            evaluate_scenario(
                [(np.array([1.0]), np.array([0.0]))],
                [],
            )


class TestEvaluateScenario:
    """Tests for denominator and no-match semantics in pooled evaluation."""

    def test_hand_computed_counts_denominators_and_pooled_rmse(self):
        result = evaluate_scenario(
            [
                (np.array([10.0, 20.0]), np.array([0.0, 2.0])),
                (np.array([5.0]), np.array([-1.0])),
            ],
            [
                (np.array([11.0, 99.0]), np.array([0.0, 20.0])),
                (np.array([7.0]), np.array([-1.0])),
            ],
            range_gate=2.0,
            velocity_gate=1.0,
        )

        assert result.evaluated_event_count == 2
        assert result.per_event_tp == [1, 1]
        assert result.per_event_fp == [1, 0]
        assert result.per_event_fn == [1, 0]
        assert result.total_tp == 2
        assert result.total_fp == 1
        assert result.total_fn == 1
        assert result.target_opportunity_count == 3
        assert result.matched_pair_count == 2
        assert result.cumulative_pd == pytest.approx(2.0 / 3.0)
        assert result.mean_false_alarms_per_evaluated_event == pytest.approx(0.5)
        assert result.cumulative_range_rmse == pytest.approx(np.sqrt(2.5))
        assert result.cumulative_velocity_rmse == pytest.approx(0.0)

    def test_no_events_have_undefined_ratios_and_rmse(self):
        result = evaluate_scenario([], [])

        assert result.evaluated_event_count == 0
        assert result.target_opportunity_count == 0
        assert result.matched_pair_count == 0
        assert result.cumulative_pd is None
        assert result.mean_false_alarms_per_evaluated_event is None
        assert result.cumulative_range_rmse is None
        assert result.cumulative_velocity_rmse is None

    def test_detection_probability_is_undefined_without_target_opportunities(self):
        result = evaluate_scenario(
            [(np.array([]), np.array([]))],
            [(np.array([42.0, 100.0]), np.array([0.0, 1.0]))],
        )

        assert result.cumulative_pd is None
        assert result.total_fp == 2
        assert result.mean_false_alarms_per_evaluated_event == pytest.approx(2.0)
        assert result.matched_pair_count == 0
        assert result.cumulative_range_rmse is None


def _make_detection(range_m: float, velocity_m_s: float, track_id: int, **extra) -> dict:
    """Helper to create a detection dict for MOT tests."""
    d = {"range_m": range_m, "velocity_m_s": velocity_m_s, "track_id": track_id}
    d.update(extra)
    return d


def _make_gt(
    target_ids: tuple[str, ...],
    ranges_m: list[float],
    velocities_m_s: list[float],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Build one exact-ID MOT ground-truth frame."""

    return target_ids, np.asarray(ranges_m), np.asarray(velocities_m_s)


class TestComputeMotMetrics:
    """Tests for CLEAR MOT metrics computation."""

    def test_perfect_tracking(self):
        """Two targets, 5 frames, consistent track IDs -- perfect score."""
        per_frame_gt = [_make_gt(("walker.a", "walker.b"), [50.0, 100.0], [5.0, -3.0])] * 5
        per_frame_dets = [
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(100.0, -3.0, track_id=2),
            ]
        ] * 5

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert isinstance(result, MOTMetrics)
        assert result.mota == pytest.approx(1.0)
        assert result.motp == pytest.approx(0.0)
        assert result.total_id_switches == 0
        assert result.total_matches == 10
        assert result.total_misses == 0
        assert result.total_fp == 0
        assert result.total_gt == 10
        assert result.num_tracks == 2
        assert result.num_gt_targets == 2
        assert result.mostly_tracked == 2
        assert result.mostly_lost == 0
        # Each GT target should have exactly 1 track ID
        assert result.distinct_track_id_count == {"walker.a": 1, "walker.b": 1}
        assert result.track_fragmentation == {"walker.a": 0, "walker.b": 0}

    def test_id_switch(self):
        """Two targets, 3 frames, track IDs swap at frame 2 -- 2 ID switches."""
        per_frame_gt = [_make_gt(("walker.a", "walker.b"), [50.0, 100.0], [5.0, -3.0])] * 3
        per_frame_dets = [
            # Frame 0: GT0->track1, GT1->track2
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(100.0, -3.0, track_id=2),
            ],
            # Frame 1: Same as frame 0
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(100.0, -3.0, track_id=2),
            ],
            # Frame 2: Track IDs swap -- GT0 gets track2, GT1 gets track1
            [
                _make_detection(50.0, 5.0, track_id=2),
                _make_detection(100.0, -3.0, track_id=1),
            ],
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_id_switches == 2  # One switch per GT target
        # MOTA = 1 - (0 + 0 + 2) / 6 = 2/3
        assert result.mota == pytest.approx(1.0 - 2.0 / 6.0)
        assert result.total_matches == 6
        assert result.total_misses == 0
        assert result.total_fp == 0
        # Each GT target was assigned 2 distinct track IDs
        assert result.distinct_track_id_count == {"walker.a": 2, "walker.b": 2}
        assert result.track_fragmentation == {"walker.a": 0, "walker.b": 0}

    def test_target_row_reorder_does_not_create_identity_switches(self):
        per_frame_gt = [
            _make_gt(("walker.a", "walker.b"), [50.0, 100.0], [5.0, -3.0]),
            _make_gt(("walker.b", "walker.a"), [100.0, 50.0], [-3.0, 5.0]),
        ]
        per_frame_dets = [
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(100.0, -3.0, track_id=2),
            ],
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(100.0, -3.0, track_id=2),
            ],
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_id_switches == 0
        assert result.distinct_track_id_count == {"walker.a": 1, "walker.b": 1}
        assert result.track_fragmentation == {"walker.a": 0, "walker.b": 0}

    def test_crossing_and_detection_order_do_not_create_identity_switches(self):
        per_frame_gt = [
            _make_gt(("walker.a", "walker.b"), [40.0, 60.0], [5.0, -5.0]),
            _make_gt(("walker.b", "walker.a"), [51.0, 49.0], [-5.0, 5.0]),
            _make_gt(("walker.a", "walker.b"), [60.0, 40.0], [5.0, -5.0]),
        ]
        per_frame_dets = [
            [
                _make_detection(60.0, -5.0, track_id=22),
                _make_detection(40.0, 5.0, track_id=11),
            ],
            [
                _make_detection(49.0, 5.0, track_id=11),
                _make_detection(51.0, -5.0, track_id=22),
            ],
            [
                _make_detection(40.0, -5.0, track_id=22),
                _make_detection(60.0, 5.0, track_id=11),
            ],
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_matches == 6
        assert result.total_id_switches == 0
        assert result.mota == pytest.approx(1.0)
        assert result.distinct_track_id_count == {"walker.a": 1, "walker.b": 1}
        assert result.track_fragmentation == {"walker.a": 0, "walker.b": 0}

    def test_requires_unique_semantic_target_ids(self):
        detections = [[_make_detection(50.0, 5.0, track_id=1)]]
        with pytest.raises(ValueError, match="must contain target_ids"):
            compute_mot_metrics(
                [(np.array([50.0]), np.array([5.0]))],  # type: ignore[list-item]
                detections,
            )
        with pytest.raises(ValueError, match="duplicate target_ids"):
            compute_mot_metrics(
                [_make_gt(("walker.a", "walker.a"), [50.0, 60.0], [5.0, 6.0])],
                detections,
            )

    def test_missed_frames(self):
        """One target, 5 frames, detection only present in 3 frames."""
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])] * 5
        per_frame_dets = [
            [_make_detection(50.0, 5.0, track_id=1)],  # Frame 0: detected
            [],  # Frame 1: missed
            [_make_detection(50.0, 5.0, track_id=1)],  # Frame 2: detected
            [],  # Frame 3: missed
            [_make_detection(50.0, 5.0, track_id=1)],  # Frame 4: detected
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_misses == 2
        assert result.total_matches == 3
        assert result.total_fp == 0
        assert result.total_gt == 5
        # MOTA = 1 - (2 + 0 + 0) / 5 = 3/5
        assert result.mota == pytest.approx(0.6)
        # Tracked 3/5 = 60% -- not >=80%, so not mostly tracked; not <=20%, so not mostly lost
        assert result.mostly_tracked == 0
        assert result.mostly_lost == 0

    def test_reappearance_with_new_track_id_counts_as_switch(self):
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])] * 3
        per_frame_dets = [
            [_make_detection(50.0, 5.0, track_id=1)],
            [],
            [_make_detection(50.0, 5.0, track_id=2)],
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_matches == 2
        assert result.total_misses == 1
        assert result.total_id_switches == 1
        assert result.mota == pytest.approx(1.0 / 3.0)
        assert result.distinct_track_id_count == {"walker.a": 2}
        assert result.track_fragmentation == {"walker.a": 1}

    def test_fragmentation(self):
        """One target tracked by 3 different track IDs across frames."""
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])] * 5
        per_frame_dets = [
            [_make_detection(50.0, 5.0, track_id=10)],  # Frame 0
            [_make_detection(50.0, 5.0, track_id=20)],  # Frame 1: new track ID
            [_make_detection(50.0, 5.0, track_id=20)],  # Frame 2: same
            [_make_detection(50.0, 5.0, track_id=30)],  # Frame 3: new track ID
            [_make_detection(50.0, 5.0, track_id=30)],  # Frame 4: same
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.distinct_track_id_count == {"walker.a": 3}
        # ID changes without a missed opportunity are not temporal fragments.
        assert result.track_fragmentation == {"walker.a": 0}
        assert result.total_id_switches == 2  # Two switches: 10->20 and 20->30
        assert result.num_tracks == 3
        # Tracked in all frames, so mostly tracked
        assert result.mostly_tracked == 1
        assert result.mostly_lost == 0

    def test_empty_detections(self):
        """No detections at all -- MOTA should handle gracefully."""
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])] * 3
        per_frame_dets: list[list[dict]] = [[], [], []]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_gt == 3
        assert result.total_matches == 0
        assert result.total_misses == 3
        assert result.total_fp == 0
        assert result.total_id_switches == 0
        # MOTA = 1 - (3 + 0 + 0) / 3 = 0.0
        assert result.mota == pytest.approx(0.0)
        assert result.motp is None
        assert result.mostly_tracked == 0
        assert result.mostly_lost == 1  # Tracked 0% of lifespan

    def test_empty_gt(self):
        """No GT targets -- MOTA should handle gracefully."""
        per_frame_gt = [_make_gt((), [], [])] * 3
        per_frame_dets = [
            [_make_detection(50.0, 5.0, track_id=1)],
            [_make_detection(50.0, 5.0, track_id=1)],
            [],
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_gt == 0
        assert result.mota is None
        assert result.motp is None
        assert result.total_fp == 2
        assert result.num_tracks == 1
        assert result.num_gt_targets == 0

    def test_false_positive_only_track_ids_are_counted(self):
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])]
        per_frame_dets = [
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(200.0, 0.0, track_id=98),
                _make_detection(300.0, 0.0, track_id=99),
            ]
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_matches == 1
        assert result.total_fp == 2
        assert result.num_tracks == 3

    @pytest.mark.parametrize("track_id", [None, np.nan, np.inf, 1.5, "1", True])
    def test_requires_finite_integer_track_id_on_every_detection(self, track_id):
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])]
        per_frame_dets = [[_make_detection(50.0, 5.0, track_id=track_id)]]

        with pytest.raises(ValueError, match="finite integer"):
            compute_mot_metrics(per_frame_gt, per_frame_dets)

    def test_requires_track_id_even_for_false_positive_detection(self):
        per_frame_gt = [_make_gt((), [], [])]
        per_frame_dets = [[{"range_m": 200.0, "velocity_m_s": 0.0}]]

        with pytest.raises(ValueError, match="missing track_id"):
            compute_mot_metrics(per_frame_gt, per_frame_dets)

    def test_rejects_duplicate_track_ids_within_event(self):
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])]
        per_frame_dets = [
            [
                _make_detection(50.0, 5.0, track_id=7),
                _make_detection(200.0, 0.0, track_id=7),
            ]
        ]

        with pytest.raises(ValueError, match="duplicate track_id 7"):
            compute_mot_metrics(per_frame_gt, per_frame_dets)

    def test_motp_with_offset(self):
        """MOTP should reflect the mean absolute range error of matched pairs."""
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])] * 3
        per_frame_dets = [
            [_make_detection(51.0, 5.0, track_id=1)],  # error = 1.0
            [_make_detection(53.0, 5.0, track_id=1)],  # error = 3.0
            [_make_detection(52.0, 5.0, track_id=1)],  # error = 2.0
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets, range_gate=3.0)

        assert result.motp == pytest.approx(2.0)  # mean(1, 3, 2) = 2.0
        assert result.mota == pytest.approx(1.0)

    def test_false_positive_impact_on_mota(self):
        """False positives should decrease MOTA."""
        per_frame_gt = [_make_gt(("walker.a",), [50.0], [5.0])] * 3
        per_frame_dets = [
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(200.0, 0.0, track_id=99),  # FP
            ],
            [_make_detection(50.0, 5.0, track_id=1)],
            [
                _make_detection(50.0, 5.0, track_id=1),
                _make_detection(300.0, 0.0, track_id=98),  # FP
            ],
        ]

        result = compute_mot_metrics(per_frame_gt, per_frame_dets)

        assert result.total_fp == 2
        assert result.total_matches == 3
        assert result.total_misses == 0
        # MOTA = 1 - (0 + 2 + 0) / 3 = 1/3
        assert result.mota == pytest.approx(1.0 / 3.0)
