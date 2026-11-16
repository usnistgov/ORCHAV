"""Bistatic phase/path geometry and independent coordinate-transformed KF oracle."""

import numpy as np
import pytest

from shared.sensing.tracker import (
    JPDAFTracker,
    KalmanTracker,
    TrackerConfig,
    _radial_prediction_matrices,
)


def geometry(t, off_axis=0.0):
    """Separate TX/RX, moving target; analytic baseband phase derivative."""
    tx, rx = np.array([-5.0, 0.0]), np.array([5.0, 0.0])
    target = np.array([100.0 - 3.0 * t, off_axis])
    speed = np.array([-3.0, 0.0])
    legs = [target - sensor for sensor in (tx, rx)]
    length = sum(np.linalg.norm(leg) for leg in legs)
    length_dot = sum(np.dot(leg / np.linalg.norm(leg), speed) for leg in legs)
    wavelength = 0.125
    # phi=-2*pi*L/lambda; f_D=phi_dot/(2*pi), velocity=lambda*f_D/2.
    phase_dot = -2 * np.pi * length_dot / wavelength
    doppler_hz = phase_dot / (2 * np.pi)
    return np.array([length, wavelength * doppler_hz / 2])


@pytest.mark.parametrize("bad", [0, -1, True, np.nan, np.inf])
def test_range_rate_scale_validation(bad):
    with pytest.raises(ValueError, match="range_rate_scale"):
        TrackerConfig(range_rate_scale=bad)


@pytest.mark.parametrize("dt", [0.01, 0.5, 8.0])
def test_bistatic_fq_matches_path_rate_coordinate_transform_and_semigroup(dt):
    # Independent conventional CV state [L, L_dot] driven by path acceleration.
    q_equivalent = 0.7
    transform = np.diag([1.0, -0.5])
    f_path = np.array([[1.0, dt], [0.0, 1.0]])
    q_path = 4 * q_equivalent * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    f, q = _radial_prediction_matrices(dt, q_equivalent, 2)
    np.testing.assert_allclose(f, transform @ f_path @ np.linalg.inv(transform))
    np.testing.assert_allclose(q, transform @ q_path @ transform.T)
    f1, q1 = _radial_prediction_matrices(dt * 0.3, q_equivalent, 2)
    f2, q2 = _radial_prediction_matrices(dt * 0.7, q_equivalent, 2)
    np.testing.assert_allclose(f, f2 @ f1)
    np.testing.assert_allclose(q, f2 @ q1 @ f2.T + q2)
    assert np.linalg.eigvalsh(q).min() > 0


@pytest.mark.parametrize("tracker_type", [KalmanTracker, JPDAFTracker])
def test_exact_phase_trajectory_preserves_identity_across_gap_and_miss(tracker_type):
    tracker = tracker_type(
        TrackerConfig(
            m=1,
            n=1,
            range_rate_scale=2,
            range_gate_m=0.1,
            velocity_gate_m_s=0.1,
            max_missed_frames=2,
        ),
        acceleration_psd_m2_s3=0,
    )
    for t in [0.0, 0.1, 0.7, 2.0, 4.5, 8.0]:
        z = geometry(t)
        detections = [] if t == 2.0 else [dict(range_m=z[0], velocity_m_s=z[1])]
        result, metadata = tracker.filter_detections(detections, update_time_s=t)
        assert len(tracker._tracks) == 1
        track = tracker._tracks[1]
        np.testing.assert_allclose(track.x, z, atol=1e-10)
        assert np.linalg.eigvalsh(track.P).min() > 0
        assert metadata["range_rate_scale"] == 2
        if detections:
            assert result[0]["track_id"] == 1


def test_noisy_off_axis_geometry_matches_independent_path_rate_kalman():
    rng = np.random.default_rng(435)
    r_equivalent = np.diag([0.04, 0.0025])
    q_equivalent = 0.1
    # Reference uses conventional receding path velocity; R and Q transform too.
    to_path = np.diag([1.0, -2.0])
    to_equivalent = np.linalg.inv(to_path)
    r_path = to_path @ r_equivalent @ to_path.T
    tracker = KalmanTracker(
        TrackerConfig(
            m=1, n=1, range_rate_scale=2, range_gate_m=5, velocity_gate_m_s=2, max_missed_frames=2
        ),
        acceleration_psd_m2_s3=q_equivalent,
        measurement_noise_range=r_equivalent[0, 0],
        measurement_noise_velocity=r_equivalent[1, 1],
    )
    previous = None
    for t in [0.0, 0.2, 0.8, 1.4, 3.0, 5.5, 7.0]:
        z = geometry(t, off_axis=20) + rng.multivariate_normal(np.zeros(2), r_equivalent)
        missing = t == 1.4
        if previous is None:
            x, p = to_path @ z, 4 * r_path
        else:
            dt = t - previous
            f = np.array([[1.0, dt], [0.0, 1.0]])
            q = 4 * q_equivalent * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
            x, p = f @ x, f @ p @ f.T + q
            if not missing:
                k = np.linalg.solve(p + r_path, p.T).T
                x = x + k @ (to_path @ z - x)
                p = (np.eye(2) - k) @ p @ (np.eye(2) - k).T + k @ r_path @ k.T
        tracker.filter_detections(
            [] if missing else [dict(range_m=z[0], velocity_m_s=z[1])], update_time_s=t
        )
        assert list(tracker._tracks) == [1]
        actual = tracker._tracks[1]
        np.testing.assert_allclose(actual.x, to_equivalent @ x, atol=1e-12)
        np.testing.assert_allclose(actual.P, to_equivalent @ p @ to_equivalent.T, atol=1e-12)
        assert np.linalg.eigvalsh(actual.P).min() > 0
        previous = t
