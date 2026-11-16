"""Saved 3D model covariance is never inferred or rebound to a nearby epoch."""

from copy import deepcopy

import numpy as np
import pytest

from visualizer.src.sensing.tracking_uncertainty import covariance_ellipsoid, native_epoch_valid


def state():
    return {
        "track_id": 7,
        "status": "coasting",
        "confirmed": False,
        "position_frame": "world",
        "state_labels": ["x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"],
        "mean": [1, 2, 3, 4, 5, 6],
        "covariance": np.diag([4, 9, 16, 1, 2, 3]).tolist(),
    }


def test_saved_principal_axes_scale_status_and_immutable_ownership():
    saved = state()
    original = deepcopy(saved)
    one, reason = covariance_ellipsoid(saved)
    two, _ = covariance_ellipsoid(saved, sigma_scale=2)
    assert reason == ""
    np.testing.assert_allclose(one.center, [1, 2, 3])
    np.testing.assert_allclose(one.radii, [2, 3, 4])
    np.testing.assert_allclose(two.radii, 2 * one.radii)
    assert one.track_id == 7 and one.status == "coasting"
    assert saved == original
    saved["mean"][0] = 99
    assert one.center[0] == 1
    for array in (one.center, one.basis, one.radii, one.covariance):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_reordered_labels_are_not_fixed_matrix_indices():
    saved = state()
    order = [3, 2, 5, 0, 4, 1]
    saved["mean"] = np.asarray(saved["mean"])[order]
    saved["covariance"] = np.asarray(saved["covariance"])[np.ix_(order, order)]
    saved["state_labels"] = [saved["state_labels"][i] for i in order]
    result, _ = covariance_ellipsoid(saved)
    np.testing.assert_allclose(result.center, [1, 2, 3])
    np.testing.assert_allclose(result.covariance, np.diag([4, 9, 16]))


def test_off_diagonal_covariance_principal_directions_and_receiver_rotation():
    saved = state()
    saved["position_frame"] = "receiver local"
    saved["covariance"][0][1] = saved["covariance"][1][0] = 2
    # General proper rotation exercises z mixing, not merely planar yaw.
    axis = np.array([1, 2, 3]) / np.sqrt(14)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(0.7) * cross + (1 - np.cos(0.7)) * cross @ cross
    origin = np.array([10, -20, 30])
    result, reason = covariance_ellipsoid(
        saved, receiver_position=origin, receiver_rotation=rotation
    )
    assert reason == ""
    expected = rotation @ np.asarray(saved["covariance"])[:3, :3] @ rotation.T
    np.testing.assert_allclose(result.center, origin + rotation @ [1, 2, 3])
    np.testing.assert_allclose(result.covariance, expected)
    np.testing.assert_allclose(result.basis @ np.diag(result.radii**2) @ result.basis.T, expected)
    assert np.linalg.det(result.basis) == pytest.approx(1)


def test_world_covariance_is_not_rotated_twice():
    result, _ = covariance_ellipsoid(
        state(), receiver_position=[10, 20, 30], receiver_rotation=np.zeros((3, 3))
    )
    np.testing.assert_allclose(result.center, [1, 2, 3])
    np.testing.assert_allclose(result.covariance, np.diag([4, 9, 16]))


@pytest.mark.parametrize("bad", [None, [], [[1]], "unknown", [[1j] * 6] * 6])
def test_missing_malformed_or_complex_covariance_is_unavailable(bad):
    saved = state()
    saved["covariance"] = bad
    result, reason = covariance_ellipsoid(saved)
    assert result is None and reason


@pytest.mark.parametrize(
    "mode",
    [
        "nan",
        "asymmetric",
        "negative",
        "invalid_velocity_block",
        "infinite_mean",
        "duplicate_labels",
        "unknown_frame",
    ],
)
def test_invalid_state_fails_closed(mode):
    saved = state()
    if mode == "nan":
        saved["covariance"][0][0] = np.nan
    elif mode == "asymmetric":
        saved["covariance"][0][1] = 0.1
    elif mode == "negative":
        saved["covariance"][0][0] = -0.01
    elif mode == "invalid_velocity_block":
        saved["covariance"][5][5] = -1
    elif mode == "infinite_mean":
        saved["mean"][2] = np.inf
    elif mode == "duplicate_labels":
        saved["state_labels"].append("x_m")
    else:
        saved["position_frame"] = "unknown"
    result, reason = covariance_ellipsoid(saved)
    assert result is None and reason


@pytest.mark.parametrize(
    "labels", [["range_m", "closing_velocity_m_s"], ["x_m", "y_m", "vx_m_s", "vy_m_s"]]
)
def test_radial_and_2d_states_never_gain_synthetic_z(labels):
    saved = state()
    saved.update(state_labels=labels, mean=[1] * len(labels), covariance=np.eye(len(labels)))
    saved["world_position_m"] = [1, 2, 3]
    result, reason = covariance_ellipsoid(saved)
    assert result is None and "Cartesian" in reason


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, 1j, [1], True])
def test_invalid_scale(scale):
    result, reason = covariance_ellipsoid(state(), sigma_scale=scale)
    assert result is None and reason


@pytest.mark.parametrize(
    "pose",
    [
        ({},),
        ({"receiver_position": [1, 2, 3]},),
        ({"receiver_position": [1, 2], "receiver_rotation": np.eye(3)},),
        ({"receiver_position": [1, 2, 3], "receiver_rotation": np.diag([1, 1, -1])},),
        ({"receiver_position": [1, 2, 3], "receiver_rotation": np.eye(3) * 2},),
    ],
)
def test_local_frame_requires_complete_proper_pose(pose):
    saved = state()
    saved["position_frame"] = "receiver local"
    result, reason = covariance_ellipsoid(saved, **pose[0])
    assert result is None and reason


def test_singular_psd_and_roundoff_negative_are_not_inflated():
    saved = state()
    saved["covariance"][0][0] = -1e-12
    result, _ = covariance_ellipsoid(saved)
    assert result.radii[0] == 0
    assert result.radii[1] == 3
    saved["covariance"] = np.zeros((6, 6))
    result, _ = covariance_ellipsoid(saved)
    np.testing.assert_array_equal(result.radii, [0, 0, 0])


def native():
    return {
        "temporal": {"result": {"reference_time_s": 12.5}},
        "detection_filter_metadata": {
            "tracker": {"decisions": {"schema_version": 1, "update_time_s": 12.5}}
        },
    }


def test_epoch_gate_does_not_select_nearest_or_change_source():
    sensing = native()
    original = deepcopy(sensing)
    assert native_epoch_valid(sensing, 12.5) == (True, "")
    assert not native_epoch_valid(sensing, 12.5001)[0]
    assert not native_epoch_valid(sensing, 12.4)[0]
    assert sensing == original
    sensing["detection_filter_metadata"]["tracker"]["decisions"]["update_time_s"] = 12.4
    assert not native_epoch_valid(sensing, 12.5)[0]


@pytest.mark.parametrize(
    "data,time_s",
    [
        (None, 12.5),
        ({}, 12.5),
        (native(), np.nan),
        (native(), None),
        (native(), [12.5]),
        (native(), 1j),
    ],
)
def test_missing_or_nonfinite_epoch_data(data, time_s):
    valid, reason = native_epoch_valid(data, time_s)
    assert not valid and reason
