"""Unit tests for shared.sensing.tracker.

Tests M-of-N tracker, Kalman tracker, JPDAF tracker, and ghost track
correlation pruning including velocity cross-correlation and co-existence
analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.sensing.tracker import (
    CartesianKalmanTracker,
    JPDAFTracker,
    KalmanTracker,
    MofNTracker,
    TrackerConfig,
    _cartesian_prediction_matrices,
    _gate_normalized_cost,
    _identify_contained_ghost_tracks,
    _radial_prediction_matrices,
    prune_correlated_ghost_tracks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(range_m: float, velocity_m_s: float, **extra) -> dict:
    return {"range_m": range_m, "velocity_m_s": velocity_m_s, **extra}


def _make_result(frame_index: int, detections: list[dict]) -> dict:
    return {"_frame_index": frame_index, "detections": detections}


_TRACKER_FACTORIES = [
    pytest.param(MofNTracker, id="m-of-n"),
    pytest.param(KalmanTracker, id="kalman"),
    pytest.param(JPDAFTracker, id="jpdaf"),
    pytest.param(CartesianKalmanTracker, id="cartesian"),
]


class TestTrackerParameterValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"m": 0}, "positive integers"),
            ({"n": 0}, "positive integers"),
            ({"m": 3, "n": 2}, "less than or equal"),
            ({"m": 1.5}, "non-negative integer"),
            ({"max_missed_frames": -1}, "non-negative integer"),
            ({"min_track_lifetime": -1}, "non-negative integer"),
            ({"range_gate_m": 0.0}, "strictly positive"),
            ({"range_gate_m": True}, "strictly positive"),
            ({"velocity_gate_m_s": -1.0}, "strictly positive"),
            ({"xy_gate_m": np.nan}, "strictly positive"),
            ({"azimuth_gate_deg": np.inf}, "strictly positive"),
            ({"azimuth_cost_scale_deg": 0.0}, "strictly positive"),
            ({"output_tentative": 1}, "boolean"),
            ({"include_z": "yes"}, "boolean"),
        ],
    )
    def test_tracker_config_rejects_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TrackerConfig(**kwargs)

    @pytest.mark.parametrize("tracker_factory", [KalmanTracker, JPDAFTracker])
    @pytest.mark.parametrize(
        "parameter",
        ["measurement_noise_range", "measurement_noise_velocity"],
    )
    @pytest.mark.parametrize("bad_variance", [0.0, -1.0, np.nan, np.inf, True, "invalid"])
    def test_radial_measurement_variances_must_be_positive(
        self,
        tracker_factory,
        parameter,
        bad_variance,
    ):
        with pytest.raises(ValueError, match=parameter):
            tracker_factory(**{parameter: bad_variance})

    @pytest.mark.parametrize(
        "parameter, bad_value",
        [
            ("pd", 0.0),
            ("pd", 1.0),
            ("pd", np.nan),
            ("pd", True),
            ("beta_fa", 0.0),
            ("beta_fa", np.inf),
            ("gate_threshold", -1.0),
            ("gate_threshold", np.nan),
        ],
    )
    def test_jpdaf_rejects_invalid_probability_and_density_parameters(
        self,
        parameter,
        bad_value,
    ):
        with pytest.raises(ValueError, match=parameter):
            JPDAFTracker(**{parameter: bad_value})

    @pytest.mark.parametrize(
        "parameter, bad_value",
        [
            ("measurement_noise_xy", 0.0),
            ("measurement_noise_xy", np.nan),
            ("initial_velocity_variance_m2_s2", -1.0),
            ("initial_velocity_variance_m2_s2", np.inf),
        ],
    )
    def test_cartesian_variances_must_be_positive(self, parameter, bad_value):
        with pytest.raises(ValueError, match=parameter):
            CartesianKalmanTracker(**{parameter: bad_value})

    def test_cartesian_include_z_must_be_boolean(self):
        with pytest.raises(ValueError, match="include_z"):
            CartesianKalmanTracker(include_z=1)


class TestTimestampContract:
    @pytest.mark.parametrize("tracker_factory", _TRACKER_FACTORIES)
    def test_update_time_is_required_and_keyword_only(self, tracker_factory):
        tracker = tracker_factory()
        with pytest.raises(TypeError, match="update_time_s"):
            tracker.filter_detections([])
        with pytest.raises(TypeError):
            tracker.filter_detections([], 0.0)

    @pytest.mark.parametrize("tracker_factory", _TRACKER_FACTORIES)
    @pytest.mark.parametrize(
        "bad_time",
        [None, "invalid", np.nan, np.inf, -np.inf, True, np.bool_(False)],
    )
    def test_first_update_rejects_nonfinite_time(self, tracker_factory, bad_time):
        tracker = tracker_factory()
        with pytest.raises(ValueError, match="finite number"):
            tracker.filter_detections([], update_time_s=bad_time)

    @pytest.mark.parametrize("tracker_factory", _TRACKER_FACTORIES)
    def test_updates_must_be_strictly_increasing(self, tracker_factory):
        tracker = tracker_factory()
        tracker.filter_detections([], update_time_s=10.0)

        with pytest.raises(ValueError, match="strictly increasing"):
            tracker.filter_detections([], update_time_s=10.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            tracker.filter_detections([], update_time_s=9.0)

        _, metadata = tracker.filter_detections([], update_time_s=11.0)
        assert metadata["prediction_dt_s"] == pytest.approx(1.0)

    @pytest.mark.parametrize("tracker_factory", _TRACKER_FACTORIES)
    def test_elapsed_time_overflow_does_not_advance_tracker_clock(self, tracker_factory):
        tracker = tracker_factory()
        tracker.filter_detections([], update_time_s=-1.0e308)

        with pytest.raises(ValueError, match="finite elapsed interval"):
            tracker.filter_detections([], update_time_s=1.0e308)

        assert tracker._last_update_time_s == -1.0e308


class TestDetectionInputContract:
    @pytest.mark.parametrize(
        "tracker_factory, detection",
        [
            (MofNTracker, _make_detection(-0.01, 0.0)),
            (KalmanTracker, _make_detection(-0.01, 0.0)),
            (JPDAFTracker, _make_detection(-0.01, 0.0)),
            (
                CartesianKalmanTracker,
                _make_detection(-0.01, 0.0, x_m=-0.01, y_m=0.0),
            ),
        ],
    )
    def test_trackers_reject_negative_physical_range(self, tracker_factory, detection):
        tracker = tracker_factory(TrackerConfig(m=1, n=1))

        with pytest.raises(ValueError, match="range_m must be finite and non-negative"):
            tracker.filter_detections([detection], update_time_s=0.0)

    @pytest.mark.parametrize("tracker_factory", [MofNTracker, KalmanTracker, JPDAFTracker])
    @pytest.mark.parametrize(
        "bad_detection, field",
        [
            (None, "mapping"),
            ({"velocity_m_s": 0.0}, "range_m"),
            ({"range_m": 1.0}, "velocity_m_s"),
            ({"range_m": np.nan, "velocity_m_s": 0.0}, "range_m"),
            ({"range_m": 1.0, "velocity_m_s": np.inf}, "velocity_m_s"),
            ({"range_m": True, "velocity_m_s": 0.0}, "range_m"),
        ],
    )
    def test_radial_trackers_reject_incomplete_or_nonfinite_detections(
        self,
        tracker_factory,
        bad_detection,
        field,
    ):
        tracker = tracker_factory(TrackerConfig(m=1, n=1))

        with pytest.raises(ValueError, match=field):
            tracker.filter_detections([bad_detection], update_time_s=2.0)

        # Rejection is transaction-like: neither time nor lifecycle state was
        # advanced, so a corrected event at the same epoch is accepted.
        filtered, _ = tracker.filter_detections(
            [_make_detection(1.0, 0.0)],
            update_time_s=2.0,
        )
        assert filtered[0]["track_id"] == 1

    @pytest.mark.parametrize("tracker_factory", [MofNTracker, KalmanTracker, JPDAFTracker])
    @pytest.mark.parametrize(
        "bad_detection",
        [
            _make_detection(1.0, 0.0),
            _make_detection(1.0, 0.0, aoa_az_deg=None),
            _make_detection(1.0, 0.0, aoa_az_deg=np.nan),
        ],
    )
    def test_configured_azimuth_requires_an_explicit_finite_angle(
        self,
        tracker_factory,
        bad_detection,
    ):
        tracker = tracker_factory(
            TrackerConfig(m=1, n=1, azimuth_gate_deg=10.0),
        )

        with pytest.raises(ValueError, match="aoa_az_deg"):
            tracker.filter_detections([bad_detection], update_time_s=0.0)

    @pytest.mark.parametrize(
        "bad_detection, field",
        [
            (_make_detection(1.0, 0.0, y_m=0.0), "x_m"),
            (_make_detection(1.0, 0.0, x_m=0.0), "y_m"),
            (_make_detection(1.0, 0.0, x_m=np.nan, y_m=0.0), "x_m"),
        ],
    )
    def test_cartesian_tracker_requires_finite_xy(self, bad_detection, field):
        tracker = CartesianKalmanTracker(TrackerConfig(m=1, n=1))

        with pytest.raises(ValueError, match=field):
            tracker.filter_detections([bad_detection], update_time_s=0.0)

    @pytest.mark.parametrize("bad_z", [None, np.nan, np.inf, True])
    def test_3d_cartesian_tracker_requires_finite_z(self, bad_z):
        tracker = CartesianKalmanTracker(TrackerConfig(m=1, n=1), include_z=True)
        detection = _make_detection(1.0, 0.0, x_m=1.0, y_m=0.0)
        if bad_z is not None:
            detection["z_m"] = bad_z

        with pytest.raises(ValueError, match="z_m"):
            tracker.filter_detections([detection], update_time_s=0.0)


class TestTimestampedPrediction:
    def test_radial_f_and_q_match_hand_computation(self):
        transition, process_covariance = _radial_prediction_matrices(
            2.0,
            3.0,
        )

        np.testing.assert_array_equal(transition, [[1.0, -2.0], [0.0, 1.0]])
        np.testing.assert_allclose(
            process_covariance,
            [
                [8.0, -6.0],  # 3 * [dt^3/3, -dt^2/2]
                [-6.0, 6.0],  # 3 * [-dt^2/2, dt]
            ],
        )

    def test_cartesian_f_and_q_match_independent_axis_blocks(self):
        transition, process_covariance = _cartesian_prediction_matrices(
            0.5,
            4.0,
            2,
        )

        np.testing.assert_array_equal(
            transition,
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        np.testing.assert_allclose(
            process_covariance,
            [
                [1.0 / 6.0, 0.0, 0.5, 0.0],
                [0.0, 1.0 / 6.0, 0.0, 0.5],
                [0.5, 0.0, 2.0, 0.0],
                [0.0, 0.5, 0.0, 2.0],
            ],
        )

    def test_radial_h_and_r_have_explicit_measurement_units(self):
        for tracker_factory in (KalmanTracker, JPDAFTracker):
            tracker = tracker_factory(
                measurement_noise_range=0.36,
                measurement_noise_velocity=0.04,
            )
            np.testing.assert_array_equal(tracker._H, np.eye(2))
            np.testing.assert_array_equal(tracker._R, np.diag([0.36, 0.04]))

    @pytest.mark.parametrize("include_z", [False, True])
    def test_cartesian_h_r_and_initial_p_preserve_units(self, include_z):
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1),
            include_z=include_z,
            measurement_noise_xy=0.25,
            initial_velocity_variance_m2_s2=9.0,
        )
        position = (
            _make_3d_detection(1.0, 2.0, 3.0, np.hypot(1.0, 2.0), 0.0)
            if include_z
            else _make_cartesian_detection(1.0, 2.0, np.hypot(1.0, 2.0), 0.0)
        )
        tracker.filter_detections([position], update_time_s=0.0)

        dimensions = 3 if include_z else 2
        expected_h = np.zeros((dimensions, 2 * dimensions))
        expected_h[:, :dimensions] = np.eye(dimensions)
        np.testing.assert_array_equal(tracker._H, expected_h)
        np.testing.assert_array_equal(tracker._R, np.eye(dimensions) * 0.25)
        np.testing.assert_array_equal(
            tracker._tracks[1].P,
            np.diag([1.0] * dimensions + [9.0] * dimensions),
        )

    def test_gate_normalized_cost_is_dimensionless(self):
        # Half a range gate plus one quarter velocity gate: unlike a raw sum,
        # changing physical units with the same ratios leaves this unchanged.
        expected = np.sqrt(0.5**2 + 0.25**2)
        assert _gate_normalized_cost((1.0, 2.0), (0.5, 2.0)) == pytest.approx(expected)
        assert _gate_normalized_cost((100.0, 200.0), (50.0, 200.0)) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "tracker_factory, detection",
        [
            (KalmanTracker, _make_detection(10.0, 1.0)),
            (JPDAFTracker, _make_detection(10.0, 1.0)),
            (
                CartesianKalmanTracker,
                _make_detection(10.0, 1.0, x_m=10.0, y_m=0.0),
            ),
        ],
    )
    def test_elapsed_time_overflow_is_transactional(self, tracker_factory, detection):
        tracker = tracker_factory(TrackerConfig(m=1, n=1, max_missed_frames=2))
        tracker.filter_detections([detection], update_time_s=-1.0e308)
        track = tracker._tracks[1]
        state_before = track.x.copy()
        covariance_before = track.P.copy()
        misses_before = track.misses

        with pytest.raises(ValueError, match="finite elapsed interval"):
            tracker.filter_detections([], update_time_s=1.0e308)

        assert tracker._last_update_time_s == -1.0e308
        assert tracker._F is None
        assert tracker._Q is None
        np.testing.assert_array_equal(track.x, state_before)
        np.testing.assert_array_equal(track.P, covariance_before)
        assert track.misses == misses_before

    @pytest.mark.parametrize(
        "tracker_factory, detection",
        [
            (KalmanTracker, _make_detection(10.0, 1.0)),
            (JPDAFTracker, _make_detection(10.0, 1.0)),
            (
                CartesianKalmanTracker,
                _make_detection(10.0, 1.0, x_m=10.0, y_m=0.0),
            ),
        ],
    )
    def test_f_q_numeric_overflow_is_transactional(self, tracker_factory, detection):
        tracker = tracker_factory(TrackerConfig(m=1, n=1, max_missed_frames=2))
        tracker.filter_detections([detection], update_time_s=0.0)
        track = tracker._tracks[1]
        state_before = track.x.copy()
        covariance_before = track.P.copy()
        misses_before = track.misses

        with pytest.raises(ValueError, match="F/Q are outside numeric range"):
            tracker.filter_detections([], update_time_s=1.0e200)

        assert tracker._last_update_time_s == 0.0
        assert tracker._F is None
        assert tracker._Q is None
        np.testing.assert_array_equal(track.x, state_before)
        np.testing.assert_array_equal(track.P, covariance_before)
        assert track.misses == misses_before

    def test_first_radial_update_does_not_predict(self):
        tracker = KalmanTracker(TrackerConfig(m=1, n=1))
        filtered, metadata = tracker.filter_detections(
            [_make_detection(10.0, -2.0)], update_time_s=123.0
        )

        assert tracker._F is None
        assert tracker._Q is None
        np.testing.assert_allclose(tracker._tracks[1].x, [10.0, -2.0])
        assert filtered[0]["track_range_m"] == pytest.approx(10.0)
        assert metadata["prediction_dt_s"] is None

    @pytest.mark.parametrize("tracker_factory", [KalmanTracker, JPDAFTracker])
    def test_radial_prediction_uses_elapsed_time_and_physical_q(self, tracker_factory):
        tracker = tracker_factory(
            TrackerConfig(m=1, n=1, max_missed_frames=2),
            acceleration_psd_m2_s3=2.0,
        )
        tracker.filter_detections([_make_detection(10.0, 2.0)], update_time_s=4.0)
        _, metadata = tracker.filter_detections([], update_time_s=4.5)

        np.testing.assert_allclose(tracker._F, [[1.0, -0.5], [0.0, 1.0]])
        np.testing.assert_allclose(
            tracker._Q,
            2.0 * np.array([[0.5**3 / 3.0, -(0.5**2) / 2.0], [-(0.5**2) / 2.0, 0.5]]),
        )
        np.testing.assert_allclose(tracker._tracks[1].x, [9.0, 2.0])
        expected_covariance = tracker._F @ np.diag([1.0, 0.4]) @ tracker._F.T + tracker._Q
        np.testing.assert_allclose(
            tracker._tracks[1].P,
            expected_covariance,
        )
        assert metadata["prediction_dt_s"] == pytest.approx(0.5)

    def test_irregular_radial_cadence_uses_each_physical_interval(self):
        tracker = KalmanTracker(
            TrackerConfig(m=1, n=1, max_missed_frames=3),
            acceleration_psd_m2_s3=0.0,
        )
        tracker.filter_detections([_make_detection(10.0, 2.0)], update_time_s=1.0)
        tracker.filter_detections([], update_time_s=1.25)
        np.testing.assert_allclose(tracker._tracks[1].x, [9.5, 2.0])

        tracker.filter_detections([], update_time_s=2.0)
        np.testing.assert_allclose(tracker._tracks[1].x, [8.0, 2.0])

    @pytest.mark.parametrize("include_z, dimensions", [(False, 2), (True, 3)])
    def test_cartesian_prediction_uses_one_q_block_per_axis(self, include_z, dimensions):
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, max_missed_frames=2),
            include_z=include_z,
            acceleration_psd_m2_s3=1.5,
        )
        tracker.filter_detections([], update_time_s=5.0)
        tracker.filter_detections([], update_time_s=5.2)

        expected_f = np.eye(2 * dimensions)
        expected_q = np.zeros((2 * dimensions, 2 * dimensions))
        q_position = 1.5 * 0.2**3 / 3.0
        q_cross = 1.5 * 0.2**2 / 2.0
        q_velocity = 1.5 * 0.2
        for axis in range(dimensions):
            velocity_axis = dimensions + axis
            expected_f[axis, velocity_axis] = 0.2
            expected_q[axis, axis] = q_position
            expected_q[axis, velocity_axis] = q_cross
            expected_q[velocity_axis, axis] = q_cross
            expected_q[velocity_axis, velocity_axis] = q_velocity

        np.testing.assert_allclose(tracker._F, expected_f)
        np.testing.assert_allclose(tracker._Q, expected_q)

    @pytest.mark.parametrize(
        "tracker_factory, expected_q",
        [(KalmanTracker, 0.5), (JPDAFTracker, 0.5), (CartesianKalmanTracker, 1.0)],
    )
    def test_tracker_specific_default_acceleration_psd(self, tracker_factory, expected_q):
        tracker = tracker_factory()
        _, metadata = tracker.filter_detections([], update_time_s=0.0)
        assert metadata["acceleration_psd_m2_s3"] == pytest.approx(expected_q)

    @pytest.mark.parametrize(
        "tracker_factory", [KalmanTracker, JPDAFTracker, CartesianKalmanTracker]
    )
    @pytest.mark.parametrize("bad_q", [-1.0, np.nan, np.inf, True, "invalid"])
    def test_acceleration_psd_must_be_finite_and_nonnegative(self, tracker_factory, bad_q):
        with pytest.raises(ValueError, match="finite and non-negative"):
            tracker_factory(acceleration_psd_m2_s3=bad_q)


class TestIdentityStability:
    @pytest.mark.parametrize("tracker_factory", [KalmanTracker, JPDAFTracker])
    def test_track_id_survives_miss_and_reacquisition(self, tracker_factory):
        tracker = tracker_factory(
            TrackerConfig(
                m=1,
                n=1,
                max_missed_frames=2,
                range_gate_m=3.0,
                velocity_gate_m_s=1.0,
            ),
            acceleration_psd_m2_s3=0.0,
        )
        first, _ = tracker.filter_detections(
            [_make_detection(10.0, -1.0)],
            update_time_s=0.0,
        )
        original_id = first[0]["track_id"]

        tracker.filter_detections([], update_time_s=0.75)
        reacquired, metadata = tracker.filter_detections(
            [_make_detection(8.5, -1.0)],
            update_time_s=1.5,
        )

        assert reacquired[0]["track_id"] == original_id
        assert metadata["active_tracks"] == 1
        assert metadata["prediction_dt_s"] == pytest.approx(0.75)

    @pytest.mark.parametrize("tracker_factory", [KalmanTracker, JPDAFTracker])
    def test_velocity_separates_tracks_through_range_crossing(self, tracker_factory):
        tracker = tracker_factory(
            TrackerConfig(
                m=1,
                n=1,
                max_missed_frames=1,
                range_gate_m=3.0,
                velocity_gate_m_s=2.0,
            ),
            acceleration_psd_m2_s3=0.01,
            measurement_noise_range=0.05,
            measurement_noise_velocity=0.01,
        )
        initial, _ = tracker.filter_detections(
            [_make_detection(4.0, -1.0), _make_detection(6.0, 1.0)],
            update_time_s=0.0,
        )
        approaching_id = next(d["track_id"] for d in initial if d["velocity_m_s"] > 0.0)
        receding_id = next(d["track_id"] for d in initial if d["velocity_m_s"] < 0.0)

        at_crossing, _ = tracker.filter_detections(
            [_make_detection(5.0, -1.0), _make_detection(5.0, 1.0)],
            update_time_s=1.0,
        )
        after_crossing, _ = tracker.filter_detections(
            [_make_detection(6.0, -1.0), _make_detection(4.0, 1.0)],
            update_time_s=2.0,
        )

        assert next(d["track_id"] for d in at_crossing if d["velocity_m_s"] > 0.0) == (
            approaching_id
        )
        assert next(d["track_id"] for d in at_crossing if d["velocity_m_s"] < 0.0) == (receding_id)
        assert next(d["track_id"] for d in after_crossing if d["velocity_m_s"] > 0.0) == (
            approaching_id
        )
        assert next(d["track_id"] for d in after_crossing if d["velocity_m_s"] < 0.0) == (
            receding_id
        )


# ---------------------------------------------------------------------------
# MofNTracker
# ---------------------------------------------------------------------------


class TestMofNTracker:
    @pytest.mark.parametrize("lifetime,first_output", [(0, 2), (2, 2), (4, 4)])
    def test_minimum_lifetime_counts_all_hits_beyond_confirmation_window(
        self, lifetime, first_output
    ):
        tracker = MofNTracker(TrackerConfig(m=2, n=3, min_track_lifetime=lifetime))
        counts = [
            len(tracker.filter_detections([_make_detection(30.0, 0.0)], update_time_s=float(i))[0])
            for i in range(10)
        ]
        assert counts == [0] * (first_output - 1) + [1] * (11 - first_output)

    def test_misses_preserve_lifetime_but_confirmation_still_requires_recent_hits(self):
        tracker = MofNTracker(
            TrackerConfig(
                confirmation_mode="continuous",
                m=2,
                n=3,
                min_track_lifetime=4,
                max_missed_frames=3,
                record_diagnostics=True,
            )
        )
        counts, hits = [], []
        for i, present in enumerate([True, True, True, True, False, False, True, True]):
            output, metadata = tracker.filter_detections(
                [_make_detection(30.0, 0.0)] if present else [], update_time_s=float(i)
            )
            counts.append(len(output))
            hits.append(metadata["decisions"]["tracks"][0]["hits"])
        assert counts == [0, 0, 0, 1, 0, 0, 0, 1]
        assert hits == [1, 2, 3, 4, 4, 4, 5, 6]

    def test_single_detection_confirmed_m1(self):
        """With M=1, a single detection is immediately confirmed."""
        tracker = MofNTracker(TrackerConfig(m=1, n=3, range_gate_m=2.0, velocity_gate_m_s=1.0))
        dets = [_make_detection(5.0, 1.0)]
        filtered, meta = tracker.filter_detections(dets, update_time_s=12.5)
        assert len(filtered) == 1
        assert filtered[0]["track_confirmed"] is True
        assert meta["kept"] == 1
        assert meta["update_time_s"] == pytest.approx(12.5)
        assert meta["prediction_dt_s"] is None

    def test_m2_needs_two_frames(self):
        """With M=2, first frame is tentative, second confirms."""
        tracker = MofNTracker(TrackerConfig(m=2, n=3, range_gate_m=2.0, velocity_gate_m_s=1.0))

        # Frame 1: tentative
        dets1 = [_make_detection(5.0, 1.0)]
        filtered1, _ = tracker.filter_detections(dets1, update_time_s=2.0)
        assert len(filtered1) == 0  # not confirmed yet

        # Frame 2: same target → confirmed
        dets2 = [_make_detection(4.9, 1.0)]
        filtered2, _ = tracker.filter_detections(dets2, update_time_s=2.1)
        assert len(filtered2) == 1
        assert filtered2[0]["track_confirmed"] is True

    def test_track_id_assigned(self):
        tracker = MofNTracker(TrackerConfig(m=1, n=3, range_gate_m=2.0, velocity_gate_m_s=1.0))
        dets = [_make_detection(5.0, 1.0), _make_detection(8.0, -0.5)]
        filtered, _ = tracker.filter_detections(dets, update_time_s=0.0)
        ids = {d["track_id"] for d in filtered}
        assert len(ids) == 2

    def test_missed_track_deleted(self):
        tracker = MofNTracker(
            TrackerConfig(m=1, n=3, max_missed_frames=1, range_gate_m=2.0, velocity_gate_m_s=1.0)
        )
        # Frame 1: create track
        tracker.filter_detections([_make_detection(5.0, 1.0)], update_time_s=1.0)
        # Frame 2: no detection → miss
        tracker.filter_detections([], update_time_s=1.5)
        # Frame 3: no detection → track deleted (> max_missed_frames)
        _, meta = tracker.filter_detections([], update_time_s=3.0)
        assert meta["active_tracks"] == 0

    def test_azimuth_gate_starts_new_track_for_large_angle_jump(self):
        tracker = MofNTracker(
            TrackerConfig(
                m=1,
                n=3,
                range_gate_m=2.0,
                velocity_gate_m_s=1.0,
                azimuth_gate_deg=10.0,
            )
        )
        filtered1, _ = tracker.filter_detections(
            [_make_detection(5.0, 1.0, aoa_az_deg=-20.0)], update_time_s=0.0
        )
        filtered2, meta2 = tracker.filter_detections(
            [_make_detection(5.1, 1.0, aoa_az_deg=20.0)], update_time_s=0.25
        )
        assert filtered1[0]["track_id"] != filtered2[0]["track_id"]
        assert meta2["active_tracks"] == 2


# ---------------------------------------------------------------------------
# KalmanTracker
# ---------------------------------------------------------------------------


class TestKalmanTracker:
    def test_single_track_smoothing(self):
        """Kalman tracker smoothes range estimates across frames."""
        tracker = KalmanTracker(
            TrackerConfig(m=1, n=1, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        # Feed a target approaching at ~1 m/s
        ranges = [10.0, 9.0, 8.1, 7.0, 6.1]
        for time_s, r in enumerate(ranges):
            filtered, _ = tracker.filter_detections(
                [_make_detection(r, 1.0)], update_time_s=float(time_s)
            )
            assert len(filtered) == 1

        # Kalman state should be close to last measurement
        assert abs(filtered[0]["track_range_m"] - 6.1) < 0.5
        assert abs(filtered[0]["track_velocity_m_s"] - 1.0) < 0.3

    def test_two_tracks(self):
        tracker = KalmanTracker(
            TrackerConfig(m=1, n=1, range_gate_m=2.0, velocity_gate_m_s=2.0),
        )
        # Two targets at different ranges
        dets = [_make_detection(5.0, 1.0), _make_detection(10.0, -0.5)]
        filtered, meta = tracker.filter_detections(dets, update_time_s=0.0)
        assert len(filtered) == 2
        assert meta["active_tracks"] == 2

    def test_coast_through_miss(self):
        tracker = KalmanTracker(
            TrackerConfig(m=1, n=1, max_missed_frames=2, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        tracker.filter_detections([_make_detection(10.0, -1.0)], update_time_s=0.0)
        # Miss one frame — track should coast
        _, meta = tracker.filter_detections([], update_time_s=0.5)
        assert meta["active_tracks"] == 1
        # Miss second frame — still coasting
        _, meta = tracker.filter_detections([], update_time_s=1.5)
        assert meta["active_tracks"] == 1
        # Miss third frame — deleted
        _, meta = tracker.filter_detections([], update_time_s=4.0)
        assert meta["active_tracks"] == 0

    def test_azimuth_gate_blocks_wrong_association(self):
        tracker = KalmanTracker(
            TrackerConfig(
                m=1,
                n=1,
                range_gate_m=2.0,
                velocity_gate_m_s=2.0,
                azimuth_gate_deg=10.0,
            ),
        )
        filtered1, _ = tracker.filter_detections(
            [_make_detection(5.0, 1.0, aoa_az_deg=-25.0)], update_time_s=0.0
        )
        filtered2, meta2 = tracker.filter_detections(
            [_make_detection(5.1, 1.0, aoa_az_deg=25.0)], update_time_s=1.0
        )
        assert filtered1[0]["track_id"] != filtered2[0]["track_id"]
        assert meta2["active_tracks"] == 2
        assert meta2["association_azimuth_gate_deg"] == pytest.approx(10.0)

    def test_soft_azimuth_cost_preserves_identity_in_ambiguous_case(self):
        tracker = KalmanTracker(
            TrackerConfig(
                m=1,
                n=1,
                range_gate_m=2.0,
                velocity_gate_m_s=2.0,
                azimuth_cost_scale_deg=10.0,
            ),
        )
        filtered1, _ = tracker.filter_detections(
            [
                _make_detection(5.0, 1.0, aoa_az_deg=-25.0),
                _make_detection(5.0, 1.0, aoa_az_deg=25.0),
            ],
            update_time_s=0.0,
        )
        neg_track_id = filtered1[0]["track_id"]
        pos_track_id = filtered1[1]["track_id"]

        filtered2, meta2 = tracker.filter_detections(
            [
                _make_detection(5.0, 1.0, aoa_az_deg=25.0),
                _make_detection(5.0, 1.0, aoa_az_deg=-25.0),
            ],
            update_time_s=1.0,
        )
        assert filtered2[0]["track_id"] == pos_track_id
        assert filtered2[1]["track_id"] == neg_track_id
        assert meta2["association_azimuth_cost_scale_deg"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# JPDAFTracker
# ---------------------------------------------------------------------------


class TestJPDAFTracker:
    def test_covariance_mixture_uses_joseph_reference_and_remains_psd(self):
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, range_gate_m=100.0, velocity_gate_m_s=100.0),
            acceleration_psd_m2_s3=0.0,
            measurement_noise_range=0.4,
            measurement_noise_velocity=0.2,
            pd=0.8,
            beta_fa=0.05,
            gate_threshold=100.0,
        )
        tracker.filter_detections(
            [_make_detection(10.0, 1.0)],
            update_time_s=0.0,
        )

        prior_p0 = np.diag([1.6, 0.8])
        transition = np.array([[1.0, -0.5], [0.0, 1.0]])
        prior_p = transition @ prior_p0 @ transition.T
        prior_x = transition @ np.array([10.0, 1.0])
        measurement = np.array([10.25, 0.8])
        observation = np.eye(2)
        measurement_covariance = np.diag([0.4, 0.2])
        innovation = measurement - observation @ prior_x
        innovation_covariance = observation @ prior_p @ observation.T + measurement_covariance
        innovation_covariance_inv = np.linalg.inv(innovation_covariance)
        gain = prior_p @ observation.T @ innovation_covariance_inv
        mahalanobis_sq = float(innovation @ innovation_covariance_inv @ innovation)
        likelihood = np.exp(
            -0.5
            * (
                2.0 * np.log(2.0 * np.pi)
                + np.log(np.linalg.det(innovation_covariance))
                + mahalanobis_sq
            )
        )
        missed_weight = (1.0 - 0.8) * 0.05
        detected_weight = 0.8 * likelihood
        beta_missed = missed_weight / (missed_weight + detected_weight)
        beta_detected = 1.0 - beta_missed
        combined_innovation = beta_detected * innovation
        innovation_spread = beta_detected * np.outer(innovation, innovation) - np.outer(
            combined_innovation,
            combined_innovation,
        )
        identity_minus_kh = np.eye(2) - gain @ observation
        conditional_covariance = (
            identity_minus_kh @ prior_p @ identity_minus_kh.T
            + gain @ measurement_covariance @ gain.T
        )
        expected_covariance = (
            beta_missed * prior_p
            + beta_detected * conditional_covariance
            + gain @ innovation_spread @ gain.T
        )

        tracker.filter_detections(
            [_make_detection(float(measurement[0]), float(measurement[1]))],
            update_time_s=0.5,
        )
        actual_covariance = tracker._tracks[1].P

        np.testing.assert_allclose(actual_covariance, expected_covariance, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_covariance, actual_covariance.T, atol=1e-14)
        assert np.min(np.linalg.eigvalsh(actual_covariance)) >= -1e-12

    def test_single_track_smoothing(self):
        """JPDAF smoothes range estimates like Kalman."""
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        ranges = [10.0, 9.0, 8.1, 7.0, 6.1]
        for time_s, r in enumerate(ranges):
            filtered, _ = tracker.filter_detections(
                [_make_detection(r, 1.0)], update_time_s=float(time_s)
            )
            assert len(filtered) == 1

        assert abs(filtered[0]["track_range_m"] - 6.1) < 0.5
        assert abs(filtered[0]["track_velocity_m_s"] - 1.0) < 0.3

    def test_two_separate_tracks(self):
        """Two well-separated targets are tracked independently."""
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        dets = [_make_detection(5.0, 1.0), _make_detection(15.0, -0.5)]
        filtered, meta = tracker.filter_detections(dets, update_time_s=0.0)
        assert len(filtered) == 2
        assert meta["active_tracks"] == 2
        ids = {d["track_id"] for d in filtered}
        assert len(ids) == 2

    def test_reporting_assignment_never_duplicates_a_track_id(self):
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        first, _ = tracker.filter_detections(
            [_make_detection(5.0, 0.0)],
            update_time_s=0.0,
        )
        original_id = first[0]["track_id"]

        # Both detections are gated by the same existing track and therefore
        # both influence its marginal JPDA update. Only one may carry its
        # discrete track identity in the published output.
        filtered, metadata = tracker.filter_detections(
            [_make_detection(5.0, 0.0), _make_detection(5.05, 0.0)],
            update_time_s=0.1,
        )

        track_ids = [d["track_id"] for d in filtered]
        assert track_ids == [original_id]
        assert len(track_ids) == len(set(track_ids))
        assert metadata["raw"] == 2
        assert metadata["kept"] == 1

    def test_component_gates_precede_mahalanobis_assignment(self):
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, range_gate_m=0.2, velocity_gate_m_s=0.2),
            measurement_noise_range=100.0,
            measurement_noise_velocity=100.0,
            gate_threshold=100.0,
        )
        initial, _ = tracker.filter_detections(
            [_make_detection(5.0, 0.0)],
            update_time_s=0.0,
        )

        # The huge covariance would admit this in Mahalanobis space, but the
        # explicit physical range gate must reject it and start a new track.
        shifted, metadata = tracker.filter_detections(
            [_make_detection(5.5, 0.0)],
            update_time_s=0.1,
        )

        assert shifted[0]["track_id"] != initial[0]["track_id"]
        assert metadata["active_tracks"] == 2
        assert metadata["association_range_gate_m"] == pytest.approx(0.2)

    def test_crossing_tracks(self):
        """Two targets that cross in range are both maintained."""
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, max_missed_frames=2, range_gate_m=5.0, velocity_gate_m_s=3.0),
        )
        # Target A approaches from 10m, Target B recedes from 5m
        for i in range(10):
            r_a = 10.0 - 0.8 * i  # 10 -> 2.8
            r_b = 5.0 + 0.5 * i  # 5 -> 9.5
            dets = [_make_detection(r_a, 0.8), _make_detection(r_b, -0.5)]
            filtered, meta = tracker.filter_detections(dets, update_time_s=float(i))

        # Both tracks should still be active at frame 10
        assert meta["active_tracks"] >= 2

    def test_clutter_rejection(self):
        """Ghost/clutter detections don't disrupt the real track."""
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, range_gate_m=3.0, velocity_gate_m_s=2.0),
            pd=0.9,
            beta_fa=0.1,  # Higher clutter density
        )
        # Establish a track first
        for i in range(3):
            tracker.filter_detections([_make_detection(5.0 - 0.1 * i, 0.1)], update_time_s=float(i))

        # Now add a clutter detection far from the track
        dets = [
            _make_detection(4.7, 0.1),  # real target
            _make_detection(15.0, -0.5),  # clutter (outside gate)
        ]
        filtered, _ = tracker.filter_detections(dets, update_time_s=3.0)
        # The real track should still exist and be close to 4.7m
        track_ranges = [d["track_range_m"] for d in filtered if d["track_confirmed"]]
        assert any(abs(r - 4.7) < 1.0 for r in track_ranges)

    def test_coast_through_miss(self):
        """Track coasts through missed detection frame."""
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, max_missed_frames=2, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        tracker.filter_detections([_make_detection(10.0, -1.0)], update_time_s=0.0)
        # Miss one frame
        _, meta = tracker.filter_detections([], update_time_s=0.1)
        assert meta["active_tracks"] == 1
        # Miss second frame — still coasting
        _, meta = tracker.filter_detections([], update_time_s=0.3)
        assert meta["active_tracks"] == 1
        # Miss third frame — deleted
        _, meta = tracker.filter_detections([], update_time_s=1.0)
        assert meta["active_tracks"] == 0

    def test_confirmation_m2(self):
        """With M=2, first frame is tentative, second confirms."""
        tracker = JPDAFTracker(
            TrackerConfig(m=2, n=3, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        filtered1, _ = tracker.filter_detections([_make_detection(5.0, 1.0)], update_time_s=0.0)
        assert len(filtered1) == 0  # not confirmed yet

        filtered2, _ = tracker.filter_detections([_make_detection(4.9, 1.0)], update_time_s=0.1)
        assert len(filtered2) == 1
        assert filtered2[0]["track_confirmed"] is True

    def test_output_format(self):
        """Output dicts contain required track fields."""
        tracker = JPDAFTracker(
            TrackerConfig(m=1, n=1, range_gate_m=3.0, velocity_gate_m_s=2.0),
        )
        filtered, meta = tracker.filter_detections([_make_detection(5.0, 1.0)], update_time_s=0.0)
        assert len(filtered) == 1
        d = filtered[0]
        assert "track_id" in d
        assert "track_confirmed" in d
        assert "track_range_m" in d
        assert "track_velocity_m_s" in d
        assert isinstance(d["track_id"], int)
        assert meta["tracker_type"] == "jpdaf"

    def test_azimuth_gate_blocks_wrong_association(self):
        tracker = JPDAFTracker(
            TrackerConfig(
                m=1,
                n=1,
                range_gate_m=2.0,
                velocity_gate_m_s=2.0,
                azimuth_gate_deg=10.0,
            ),
        )
        filtered1, _ = tracker.filter_detections(
            [_make_detection(5.0, 1.0, aoa_az_deg=-25.0)], update_time_s=0.0
        )
        filtered2, meta2 = tracker.filter_detections(
            [_make_detection(5.1, 1.0, aoa_az_deg=25.0)], update_time_s=0.5
        )
        assert filtered1[0]["track_id"] != filtered2[0]["track_id"]
        assert meta2["active_tracks"] == 2
        assert meta2["association_azimuth_gate_deg"] == pytest.approx(10.0)

    def test_soft_azimuth_cost_preserves_identity_in_ambiguous_case(self):
        tracker = JPDAFTracker(
            TrackerConfig(
                m=1,
                n=1,
                range_gate_m=2.0,
                velocity_gate_m_s=2.0,
                azimuth_cost_scale_deg=10.0,
            ),
        )
        filtered1, _ = tracker.filter_detections(
            [
                _make_detection(5.0, 1.0, aoa_az_deg=-25.0),
                _make_detection(5.0, 1.0, aoa_az_deg=25.0),
            ],
            update_time_s=0.0,
        )
        neg_track_id = filtered1[0]["track_id"]
        pos_track_id = filtered1[1]["track_id"]

        filtered2, meta2 = tracker.filter_detections(
            [
                _make_detection(5.0, 1.0, aoa_az_deg=25.0),
                _make_detection(5.0, 1.0, aoa_az_deg=-25.0),
            ],
            update_time_s=0.5,
        )
        assert filtered2[0]["track_id"] == pos_track_id
        assert filtered2[1]["track_id"] == neg_track_id
        assert meta2["association_azimuth_cost_scale_deg"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CartesianKalmanTracker
# ---------------------------------------------------------------------------


def _make_cartesian_detection(
    x_m: float, y_m: float, range_m: float, velocity_m_s: float, **extra
) -> dict:
    return {"x_m": x_m, "y_m": y_m, "range_m": range_m, "velocity_m_s": velocity_m_s, **extra}


class TestCartesianKalmanTracker:
    def test_single_track_smoothing(self):
        """Cartesian tracker smoothes x/y estimates across frames."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        # Target walking along +x at ~1 m/s, constant y=2
        positions = [(0.0, 2.0), (1.0, 2.0), (2.1, 2.0), (3.0, 2.0), (4.1, 2.0)]
        for time_s, (x, y) in enumerate(positions):
            r = float(np.hypot(x, y))
            filtered, _ = tracker.filter_detections(
                [_make_cartesian_detection(x, y, r, 0.5)],
                update_time_s=float(time_s),
            )
            assert len(filtered) == 1

        # Kalman state should be close to last measurement
        assert abs(filtered[0]["track_x_m"] - 4.1) < 0.5
        assert abs(filtered[0]["track_y_m"] - 2.0) < 0.5
        # Velocity should converge toward (1, 0)
        assert abs(filtered[0]["track_vx_m_s"] - 1.0) < 0.5
        assert abs(filtered[0]["track_vy_m_s"]) < 0.3

    def test_two_tracks(self):
        """Two targets at different positions get distinct track IDs."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=2.0, velocity_gate_m_s=5.0),
        )
        dets = [
            _make_cartesian_detection(1.0, 3.0, np.hypot(1, 3), 0.5),
            _make_cartesian_detection(8.0, -2.0, np.hypot(8, -2), -0.3),
        ]
        filtered, meta = tracker.filter_detections(dets, update_time_s=0.0)
        assert len(filtered) == 2
        assert meta["active_tracks"] == 2
        assert filtered[0]["track_id"] != filtered[1]["track_id"]

    def test_association_normalizes_position_and_velocity_by_their_gates(self):
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=10.0, velocity_gate_m_s=20.0),
            acceleration_psd_m2_s3=0.0,
        )
        initial, _ = tracker.filter_detections(
            [
                _make_cartesian_detection(0.0, 0.0, 0.0, 0.0),
                _make_cartesian_detection(1.0, 0.0, 1.0, 10.0),
            ],
            update_time_s=0.0,
        )
        slow_id = initial[0]["track_id"]
        fast_id = initial[1]["track_id"]

        # Position alone favors crossed identities. Gate-normalized velocity
        # residuals make the physically consistent global assignment cheaper.
        associated, _ = tracker.filter_detections(
            [
                _make_cartesian_detection(0.9, 0.0, 0.9, 0.0),
                _make_cartesian_detection(0.1, 0.0, 0.1, 10.0),
            ],
            update_time_s=0.1,
        )

        assert associated[0]["track_id"] == slow_id
        assert associated[1]["track_id"] == fast_id

    def test_coast_through_miss(self):
        """Track coasts through missed frames and is deleted after max_missed_frames."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, max_missed_frames=2, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        tracker.filter_detections(
            [_make_cartesian_detection(5.0, 3.0, 5.83, 1.0)], update_time_s=0.0
        )
        # Miss 1 — coast
        _, meta = tracker.filter_detections([], update_time_s=0.1)
        assert meta["active_tracks"] == 1
        # Miss 2 — still coasting
        _, meta = tracker.filter_detections([], update_time_s=0.5)
        assert meta["active_tracks"] == 1
        # Miss 3 — deleted
        _, meta = tracker.filter_detections([], update_time_s=1.5)
        assert meta["active_tracks"] == 0

    def test_xy_gate_blocks_far_detection(self):
        """Detection outside xy_gate starts a new track instead of associating."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=2.0, velocity_gate_m_s=5.0),
        )
        tracker.filter_detections(
            [_make_cartesian_detection(5.0, 3.0, 5.83, 1.0)], update_time_s=0.0
        )
        # Next detection 5m away in x — should NOT associate
        filtered, meta = tracker.filter_detections(
            [_make_cartesian_detection(10.0, 3.0, 10.44, 1.0)],
            update_time_s=1.0,
        )
        assert meta["active_tracks"] == 2

    def test_velocity_gate_secondary(self):
        """Velocity gate prevents wrong association even when x/y is close."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=1.0),
        )
        tracker.filter_detections(
            [_make_cartesian_detection(5.0, 3.0, 5.83, 1.0)], update_time_s=0.0
        )
        # Same position but wildly different velocity — should be new track
        filtered, meta = tracker.filter_detections(
            [_make_cartesian_detection(5.1, 3.0, 5.93, -3.0)],
            update_time_s=0.1,
        )
        assert meta["active_tracks"] == 2

    def test_output_fields(self):
        """Verify all expected output fields are present."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        filtered, _ = tracker.filter_detections(
            [_make_cartesian_detection(5.0, 3.0, 5.83, 1.0)], update_time_s=0.0
        )
        out = filtered[0]
        assert "track_id" in out
        assert "track_confirmed" in out
        assert "track_x_m" in out
        assert "track_y_m" in out
        assert "track_vx_m_s" in out
        assert "track_vy_m_s" in out
        assert "track_range_m" in out
        assert "track_velocity_m_s" in out

    def test_rejects_missing_or_nan_xy(self):
        """Cartesian detections require explicit finite x/y measurements."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        det = _make_detection(5.0, 1.0)  # no x_m/y_m
        with pytest.raises(ValueError, match="x_m"):
            tracker.filter_detections([det], update_time_s=0.0)

        det_nan = _make_detection(5.0, 1.0, x_m=float("nan"), y_m=float("nan"))
        with pytest.raises(ValueError, match="x_m"):
            tracker.filter_detections([det_nan], update_time_s=0.0)

    def test_backward_compat_range(self):
        """track_range_m equals hypot(track_x_m, track_y_m)."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        filtered, _ = tracker.filter_detections(
            [_make_cartesian_detection(3.0, 4.0, 5.0, 1.0)], update_time_s=0.0
        )
        out = filtered[0]
        expected_range = np.hypot(out["track_x_m"], out["track_y_m"])
        assert out["track_range_m"] == pytest.approx(expected_range, abs=1e-6)

    def test_confirmation_m2(self):
        """With M=2, first frame is tentative, second confirms."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=2, n=3, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        filtered1, _ = tracker.filter_detections(
            [_make_cartesian_detection(5.0, 3.0, 5.83, 1.0)], update_time_s=0.0
        )
        assert len(filtered1) == 0  # not confirmed yet

        filtered2, _ = tracker.filter_detections(
            [_make_cartesian_detection(5.1, 3.0, 5.93, 1.0)], update_time_s=0.1
        )
        assert len(filtered2) == 1
        assert filtered2[0]["track_confirmed"] is True

    def test_metadata_type(self):
        """Metadata reports tracker_type as 'cartesian'."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        _, meta = tracker.filter_detections(
            [_make_cartesian_detection(5.0, 3.0, 5.83, 1.0)], update_time_s=0.0
        )
        assert meta["tracker_type"] == "cartesian"
        assert meta["association_xy_gate_m"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Ghost track correlation pruning
# ---------------------------------------------------------------------------


class TestPruneCorrelatedGhostTracks:
    def _build_correlated_results(
        self,
        n_frames: int = 20,
        parent_range_start: float = 5.0,
        ghost_offset: float = 7.0,
        speed: float = -0.5,
    ) -> list[dict]:
        """Build synthetic results with a parent track and a correlated ghost."""
        results = []
        for i in range(n_frames):
            r_parent = parent_range_start + speed * i * 0.1
            r_ghost = r_parent + ghost_offset
            results.append(
                _make_result(
                    i,
                    [
                        {
                            "range_m": r_parent,
                            "velocity_m_s": speed,
                            "track_id": 1,
                            "track_range_m": r_parent,
                            "track_velocity_m_s": speed,
                        },
                        {
                            "range_m": r_ghost,
                            "velocity_m_s": speed,
                            "track_id": 2,
                            "track_range_m": r_ghost,
                            "track_velocity_m_s": speed,
                        },
                    ],
                )
            )
        return results

    def test_correlated_ghost_pruned(self):
        """A ghost track with correlated range history is pruned."""
        results = self._build_correlated_results(n_frames=20)
        pruned, meta = prune_correlated_ghost_tracks(
            results, correlation_threshold=0.85, min_overlap_frames=5
        )
        assert 2 in meta["ghost_tracks"]
        # Ghost detections removed
        for r in pruned:
            for d in r["detections"]:
                assert d["track_id"] != 2

    def test_min_overlap_respected(self):
        """Ghost is NOT pruned by correlation if overlap is below threshold."""
        results = self._build_correlated_results(n_frames=3)
        _, meta = prune_correlated_ghost_tracks(
            results,
            correlation_threshold=0.85,
            min_overlap_frames=5,
            enable_coexistence=False,  # Only test correlation pass
        )
        assert len(meta["ghost_tracks"]) == 0

    def test_uncorrelated_tracks_preserved(self):
        """Two real targets with clearly different velocities are not pruned."""
        results = []
        for i in range(20):
            results.append(
                _make_result(
                    i,
                    [
                        {
                            "range_m": 5.0 - 0.1 * i,
                            "velocity_m_s": -1.0,
                            "track_id": 1,
                            "track_range_m": 5.0 - 0.1 * i,
                            "track_velocity_m_s": -1.0,
                        },
                        {
                            "range_m": 8.0 + 0.2 * i,
                            "velocity_m_s": 8.0,
                            "track_id": 2,
                            "track_range_m": 8.0 + 0.2 * i,
                            "track_velocity_m_s": 8.0,
                        },
                    ],
                )
            )
        _, meta = prune_correlated_ghost_tracks(
            results, correlation_threshold=0.85, min_overlap_frames=5
        )
        # Track 2 has very different velocity (8.0 vs -1.0) — outside gate, not pruned
        assert len(meta["ghost_tracks"]) == 0

    def test_velocity_correlation_catches_ghost(self):
        """Ghost detected via velocity correlation even when range is noisy."""
        rng = np.random.RandomState(42)
        results = []
        for i in range(30):
            v_parent = -1.0 + 0.1 * np.sin(i * 0.3)
            v_ghost = v_parent  # perfectly correlated velocity
            # Range is noisy so range correlation is weak
            r_parent = 5.0 + rng.normal(0, 0.5)
            r_ghost = 12.0 + rng.normal(0, 0.5)
            results.append(
                _make_result(
                    i,
                    [
                        {
                            "range_m": r_parent,
                            "velocity_m_s": v_parent,
                            "track_id": 1,
                            "track_range_m": r_parent,
                            "track_velocity_m_s": v_parent,
                        },
                        {
                            "range_m": r_ghost,
                            "velocity_m_s": v_ghost,
                            "track_id": 2,
                            "track_range_m": r_ghost,
                            "track_velocity_m_s": v_ghost,
                        },
                    ],
                )
            )
        _, meta = prune_correlated_ghost_tracks(
            results, correlation_threshold=0.85, min_overlap_frames=5
        )
        # Ghost should be caught by velocity correlation
        assert 2 in meta["ghost_tracks"]


# ---------------------------------------------------------------------------
# Co-existence analysis
# ---------------------------------------------------------------------------


class TestCoexistenceAnalysis:
    def test_contained_ghost_identified(self):
        """A short-lived far-range track contained within a parent is flagged."""
        tracks = {
            1: {
                "frames": list(range(0, 30)),
                "ranges": [5.0] * 30,
                "velocities": [-1.0] * 30,
            },
            2: {
                "frames": list(range(5, 15)),
                "ranges": [12.0] * 10,
                "velocities": [-1.0] * 10,
            },
        }
        ghosts = _identify_contained_ghost_tracks(
            tracks, range_ratio_threshold=1.3, containment_ratio=0.8
        )
        assert 2 in ghosts

    def test_longer_track_not_pruned(self):
        """A track at longer range but NOT contained in another is safe."""
        tracks = {
            1: {
                "frames": list(range(0, 10)),
                "ranges": [5.0] * 10,
                "velocities": [-1.0] * 10,
            },
            2: {
                "frames": list(range(15, 30)),
                "ranges": [12.0] * 15,
                "velocities": [-1.0] * 15,
            },
        }
        ghosts = _identify_contained_ghost_tracks(tracks)
        assert len(ghosts) == 0

    def test_similar_range_not_pruned(self):
        """A contained track at similar range is NOT flagged (real target)."""
        tracks = {
            1: {
                "frames": list(range(0, 30)),
                "ranges": [5.0] * 30,
                "velocities": [-1.0] * 30,
            },
            2: {
                "frames": list(range(5, 15)),
                "ranges": [6.0] * 10,  # Only 1.2x range, below 1.3 threshold
                "velocities": [-1.0] * 10,
            },
        }
        ghosts = _identify_contained_ghost_tracks(
            tracks, range_ratio_threshold=1.3, containment_ratio=0.8
        )
        assert len(ghosts) == 0

    def test_integrated_in_prune_function(self):
        """Co-existence is applied as second pass in prune_correlated_ghost_tracks."""
        # Build results where Track 2 is contained in Track 1 but with
        # independent noise (so Pearson correlation is low)
        rng = np.random.RandomState(123)
        results = []
        for i in range(30):
            dets = [
                {
                    "range_m": 5.0 + rng.normal(0, 0.3),
                    "velocity_m_s": -1.0 + rng.normal(0, 0.1),
                    "track_id": 1,
                    "track_range_m": 5.0,
                    "track_velocity_m_s": -1.0,
                },
            ]
            # Ghost only present in frames 5-15
            if 5 <= i < 15:
                dets.append(
                    {
                        "range_m": 14.0 + rng.normal(0, 0.3),
                        "velocity_m_s": -0.8 + rng.normal(0, 0.1),
                        "track_id": 2,
                        "track_range_m": 14.0,
                        "track_velocity_m_s": -0.8,
                    }
                )
            results.append(_make_result(i, dets))

        _, meta = prune_correlated_ghost_tracks(
            results,
            correlation_threshold=0.85,
            min_overlap_frames=5,
            enable_coexistence=True,
        )
        # Track 2 should be caught by co-existence (contained, far range)
        assert 2 in meta["ghost_tracks"] or 2 in meta.get("coexistence_ghosts", [])


# ---------------------------------------------------------------------------
# CartesianKalmanTracker 3D (include_z=True)
# ---------------------------------------------------------------------------


def _make_3d_detection(
    x_m: float, y_m: float, z_m: float, range_m: float, velocity_m_s: float, **extra
) -> dict:
    return {
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        "range_m": range_m,
        "velocity_m_s": velocity_m_s,
        **extra,
    }


class TestCartesianKalmanTracker3D:
    def test_3d_single_track(self):
        """3D tracker smoothes x/y/z estimates across frames."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
            include_z=True,
        )
        # Target moving in 3D: +x at 1 m/s, constant y=2, z rising at 0.5 m/s
        positions = [
            (0.0, 2.0, 1.0),
            (1.0, 2.0, 1.5),
            (2.1, 2.0, 2.0),
            (3.0, 2.0, 2.5),
            (4.1, 2.0, 3.0),
        ]
        for time_s, (x, y, z) in enumerate(positions):
            r = float(np.hypot(x, y))
            filtered, _ = tracker.filter_detections(
                [_make_3d_detection(x, y, z, r, 0.5)],
                update_time_s=float(time_s),
            )
            assert len(filtered) == 1

        out = filtered[0]
        assert abs(out["track_x_m"] - 4.1) < 0.5
        assert abs(out["track_y_m"] - 2.0) < 0.5
        assert abs(out["track_z_m"] - 3.0) < 0.5
        assert "track_vz_m_s" in out

    def test_3d_rejects_nan_z(self):
        """Detection with finite x/y but NaN z fails closed in 3D mode."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
            include_z=True,
        )
        det = _make_3d_detection(5.0, 3.0, float("nan"), 5.83, 1.0)
        with pytest.raises(ValueError, match="z_m"):
            tracker.filter_detections([det], update_time_s=0.0)

    def test_3d_output_fields(self):
        """Verify track_z_m, track_vz_m_s are present in 3D output."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
            include_z=True,
        )
        filtered, _ = tracker.filter_detections(
            [_make_3d_detection(5.0, 3.0, 2.0, 5.83, 1.0)], update_time_s=0.0
        )
        out = filtered[0]
        assert "track_id" in out
        assert "track_confirmed" in out
        assert "track_x_m" in out
        assert "track_y_m" in out
        assert "track_z_m" in out
        assert "track_vx_m_s" in out
        assert "track_vy_m_s" in out
        assert "track_vz_m_s" in out
        assert "track_range_m" in out
        assert "track_velocity_m_s" in out

    def test_3d_metadata_type(self):
        """Metadata reports tracker_type as 'cartesian3d'."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
            include_z=True,
        )
        _, meta = tracker.filter_detections(
            [_make_3d_detection(5.0, 3.0, 2.0, 5.83, 1.0)], update_time_s=0.0
        )
        assert meta["tracker_type"] == "cartesian3d"

    def test_3d_gate(self):
        """3D Euclidean gate blocks far detections in z."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=2.0, velocity_gate_m_s=5.0),
            include_z=True,
        )
        # Establish a track at (5, 3, 1)
        tracker.filter_detections([_make_3d_detection(5.0, 3.0, 1.0, 5.83, 1.0)], update_time_s=0.0)
        # Next detection at (5, 3, 10) — 9m away in z, well outside 2m gate
        filtered, meta = tracker.filter_detections(
            [_make_3d_detection(5.0, 3.0, 10.0, 5.83, 1.0)], update_time_s=1.0
        )
        assert meta["active_tracks"] == 2  # New track created, not associated

    def test_3d_backward_compat_range(self):
        """track_range_m equals horizontal hypot(x, y) — NOT 3D range."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
            include_z=True,
        )
        filtered, _ = tracker.filter_detections(
            [_make_3d_detection(3.0, 4.0, 10.0, 5.0, 1.0)], update_time_s=0.0
        )
        out = filtered[0]
        expected_range = np.hypot(out["track_x_m"], out["track_y_m"])
        assert out["track_range_m"] == pytest.approx(expected_range, abs=1e-6)
        # Confirm it is NOT the 3D distance
        range_3d = np.sqrt(out["track_x_m"] ** 2 + out["track_y_m"] ** 2 + out["track_z_m"] ** 2)
        assert out["track_range_m"] != pytest.approx(range_3d, abs=0.1)

    def test_3d_two_tracks(self):
        """Two targets at different 3D positions get distinct track IDs."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=2.0, velocity_gate_m_s=5.0),
            include_z=True,
        )
        dets = [
            _make_3d_detection(1.0, 3.0, 0.5, np.hypot(1, 3), 0.5),
            _make_3d_detection(8.0, -2.0, 5.0, np.hypot(8, -2), -0.3),
        ]
        filtered, meta = tracker.filter_detections(dets, update_time_s=0.0)
        assert len(filtered) == 2
        assert meta["active_tracks"] == 2
        assert filtered[0]["track_id"] != filtered[1]["track_id"]

    def test_2d_mode_unchanged(self):
        """When include_z=False (default), behavior is unchanged from 2D mode."""
        tracker = CartesianKalmanTracker(
            TrackerConfig(m=1, n=1, xy_gate_m=3.0, velocity_gate_m_s=5.0),
        )
        filtered, meta = tracker.filter_detections(
            [_make_cartesian_detection(5.0, 3.0, 5.83, 1.0)], update_time_s=0.0
        )
        assert len(filtered) == 1
        assert meta["tracker_type"] == "cartesian"
        out = filtered[0]
        assert "track_z_m" not in out
        assert "track_vz_m_s" not in out
