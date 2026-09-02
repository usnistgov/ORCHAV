"""Tests for beamforming standalone weight computation and steering."""

from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.beamforming import standalone as standalone_module
from visualizer.src.beamforming.standalone import (
    compute_los_angles,
    compute_standalone_beamforming,
    compute_steering_weights,
    compute_svd_beamforming,
    construct_channel_matrix_from_mpc,
)


def _vector_alignment(left: np.ndarray, right: np.ndarray) -> float:
    """Return normalized inner-product magnitude for phase-invariant comparison."""
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(abs(np.vdot(left / left_norm, right / right_norm)))


def _reference_mpc_channel(
    *,
    num_rows_tx: int,
    num_cols_tx: int,
    num_rows_rx: int,
    num_cols_rx: int,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    carrier_frequency_hz: float,
    tx_orientation: tuple[float, float, float],
    rx_orientation: tuple[float, float, float],
    mpc_paths: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Evaluate MPC contributions with the original scalar outer-product loop."""
    k = 2.0 * np.pi / (standalone_module._C / carrier_frequency_hz)
    tx_positions = standalone_module.get_element_positions(
        num_rows_tx,
        num_cols_tx,
        horizontal_spacing_m,
        vertical_spacing_m,
    )
    rx_positions = standalone_module.get_element_positions(
        num_rows_rx,
        num_cols_rx,
        horizontal_spacing_m,
        vertical_spacing_m,
    )
    channel = np.zeros((len(rx_positions), len(tx_positions)), dtype=np.complex128)

    for path_vertices, path_loss_db in mpc_paths:
        vertices = np.asarray(path_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 2 or not np.isfinite(path_loss_db):
            continue
        path_length = float(np.sum(np.linalg.norm(np.diff(vertices, axis=0), axis=1)))
        if path_length <= 0.0:
            continue
        path_gain = 10.0 ** (-abs(path_loss_db) / 20.0) * np.exp(-1j * k * path_length)
        tx_response = standalone_module._array_response_for_world_direction(
            tx_positions,
            k,
            vertices[1] - vertices[0],
            tx_orientation,
        )
        rx_response = standalone_module._array_response_for_world_direction(
            rx_positions,
            k,
            vertices[-2] - vertices[-1],
            rx_orientation,
        )
        channel += np.outer(rx_response, np.conj(tx_response)) * path_gain

    return channel


# ---------------------------------------------------------------------------
# compute_steering_weights
# ---------------------------------------------------------------------------


class TestSteeringWeights:
    """Verify phase-based steering weight computation."""

    def test_unit_norm(self):
        w, pos, gain = compute_steering_weights(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )
        assert abs(np.linalg.norm(w) - 1.0) < 1e-10

    def test_element_count(self):
        w, pos, _ = compute_steering_weights(
            num_rows=4,
            num_cols=8,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            azimuth_deg=45.0,
            elevation_deg=10.0,
        )
        assert w.shape == (32,)
        assert pos.shape == (32, 3)

    def test_boresight_gain(self):
        """Boresight steering should give maximum gain equal to num_elements."""
        n = 16
        w, pos, gain = compute_steering_weights(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )
        # Gain should be close to N (for normalized weights, |sum(w*phase)|^2 = N for boresight)
        assert gain > 0.9 * n

    def test_positive_frequency_required(self):
        with pytest.raises(ValueError):
            compute_steering_weights(1, 1, 0.005, 0.005, -1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# compute_los_angles
# ---------------------------------------------------------------------------


class TestLOSAngles:
    def test_along_x_axis(self):
        az, el = compute_los_angles((0, 0, 0), (10, 0, 0))
        assert abs(az - 0.0) < 0.1
        assert abs(el - 0.0) < 0.1

    def test_along_y_axis(self):
        az, el = compute_los_angles((0, 0, 0), (0, 10, 0))
        assert abs(az - 90.0) < 0.1
        assert abs(el - 0.0) < 0.1

    def test_straight_up(self):
        az, el = compute_los_angles((0, 0, 0), (0, 0, 10))
        assert abs(el - 90.0) < 0.1

    def test_coincident_positions(self):
        az, el = compute_los_angles((5, 5, 5), (5, 5, 5))
        assert az == 0.0
        assert el == 0.0


# ---------------------------------------------------------------------------
# compute_standalone_beamforming
# ---------------------------------------------------------------------------


class TestStandaloneBeamforming:
    """Verify the high-level standalone beamforming API."""

    def test_manual_steering(self):
        result = compute_standalone_beamforming(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            steering_strategy="manual",
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )
        assert result is not None
        assert "pairs" in result
        assert len(result["pairs"]) == 1
        pair = result["pairs"][0]
        assert pair["tx"]["role"] == "tx"
        assert pair["rx"]["role"] == "rx"
        assert pair["tx"]["weights"].shape[1] == 2  # [real, imag]

    def test_los_steering(self):
        result = compute_standalone_beamforming(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            steering_strategy="los",
        )
        assert result is not None
        # LOS strategy should point TX toward RX
        pair = result["pairs"][0]
        az = pair["tx"]["steering_azimuth_deg"]
        # TX at origin, RX at (50, 0, 1) => azimuth near 0 degrees
        assert abs(az) < 5.0

    def test_los_steering_uses_device_local_frame(self):
        result = compute_standalone_beamforming(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 0),
            rx_position=(10, 0, 0),
            tx_orientation=(np.pi / 2.0, 0.0, 0.0),
            rx_orientation=(0.0, 0.0, 0.0),
            steering_strategy="los",
        )
        assert result is not None

        pair = result["pairs"][0]
        tx_weights = pair["tx"]["weights"][:, 0] + 1j * pair["tx"]["weights"][:, 1]
        expected_weights, _, _ = compute_steering_weights(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            azimuth_deg=-90.0,
            elevation_deg=0.0,
        )

        assert abs(pair["tx"]["steering_azimuth_deg"] + 90.0) < 0.1
        assert _vector_alignment(tx_weights, expected_weights) > 1.0 - 1e-10

    def test_svd_steering(self):
        result = compute_standalone_beamforming(
            num_rows=2,
            num_cols=2,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            steering_strategy="svd",
        )
        assert result is not None

    def test_manual_requires_angles(self):
        result = compute_standalone_beamforming(
            num_rows=2,
            num_cols=2,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 0),
            rx_position=(10, 0, 0),
            steering_strategy="manual",
            azimuth_deg=None,
            elevation_deg=None,
        )
        assert result is None

    def test_unknown_strategy_returns_none(self):
        result = compute_standalone_beamforming(
            num_rows=2,
            num_cols=2,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 0),
            rx_position=(10, 0, 0),
            steering_strategy="unknown_strategy",
        )
        assert result is None


# ---------------------------------------------------------------------------
# construct_channel_matrix_from_mpc
# ---------------------------------------------------------------------------


class TestChannelMatrix:
    """Verify channel matrix construction."""

    def test_shape(self):
        H = construct_channel_matrix_from_mpc(
            num_rows_tx=2,
            num_cols_tx=2,
            num_rows_rx=4,
            num_cols_rx=4,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            tx_orientation=(0, 0, 0),
            rx_orientation=(0, 0, 0),
        )
        assert H.shape == (16, 4)  # (rx_elements, tx_elements)
        assert H.dtype == np.complex128

    def test_los_channel_nonzero(self):
        H = construct_channel_matrix_from_mpc(
            num_rows_tx=2,
            num_cols_tx=2,
            num_rows_rx=2,
            num_cols_rx=2,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            tx_orientation=(0, 0, 0),
            rx_orientation=(0, 0, 0),
        )
        assert np.abs(H).sum() > 0

    def test_with_mpc_paths(self):
        # Two-segment path: TX -> bounce -> RX
        path_verts = np.array([[0, 0, 10], [25, 5, 5], [50, 0, 1]], dtype=np.float64)
        mpc_paths = [(path_verts, -80.0)]  # path_loss_db
        H = construct_channel_matrix_from_mpc(
            num_rows_tx=2,
            num_cols_tx=2,
            num_rows_rx=2,
            num_cols_rx=2,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            tx_orientation=(0, 0, 0),
            rx_orientation=(0, 0, 0),
            mpc_paths=mpc_paths,
        )
        assert H.shape == (4, 4)
        assert np.abs(H).sum() > 0

    def test_batched_mpc_channel_matches_scalar_outer_product_reference(self):
        tx_position = (0.0, 0.0, 10.0)
        rx_position = (50.0, 0.0, 1.0)
        mpc_paths = [
            (
                np.array(
                    [tx_position, (12.0, 7.0, 4.0), (31.0, -2.0, 3.0), rx_position],
                    dtype=np.float64,
                ),
                83.5,
            ),
            (np.array([tx_position, rx_position], dtype=np.float64), -91.25),
            (
                np.array(
                    [tx_position, tx_position, (20.0, 3.0, 2.0), rx_position],
                    dtype=np.float64,
                ),
                105.0,
            ),
            (np.array([tx_position], dtype=np.float64), 70.0),
            (np.array([tx_position, rx_position], dtype=np.float64), np.inf),
        ]
        common = {
            "num_rows_tx": 3,
            "num_cols_tx": 2,
            "num_rows_rx": 2,
            "num_cols_rx": 2,
            "horizontal_spacing_m": 0.00536,
            "vertical_spacing_m": 0.0048,
            "carrier_frequency_hz": 28e9,
            "tx_orientation": (0.35, -0.2, 0.1),
            "rx_orientation": (-0.4, 0.15, -0.05),
            "mpc_paths": mpc_paths,
        }

        expected = _reference_mpc_channel(**common)
        actual = construct_channel_matrix_from_mpc(
            **common,
            tx_position=tx_position,
            rx_position=rx_position,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_batched_mpc_channel_is_stable_across_memory_batches(self, monkeypatch):
        tx_position = (0.0, 0.0, 0.0)
        rx_position = (20.0, 0.0, 0.0)
        mpc_paths = [
            (
                np.array([tx_position, (5.0, float(index), 2.0), (15.0, -index, 1.0), rx_position]),
                80.0 + index,
            )
            for index in range(1, 9)
        ]
        kwargs = {
            "num_rows_tx": 2,
            "num_cols_tx": 2,
            "num_rows_rx": 2,
            "num_cols_rx": 2,
            "horizontal_spacing_m": 0.005,
            "vertical_spacing_m": 0.005,
            "carrier_frequency_hz": 28e9,
            "tx_position": tx_position,
            "rx_position": rx_position,
            "tx_orientation": (0.1, 0.2, -0.1),
            "rx_orientation": (-0.2, 0.1, 0.2),
            "mpc_paths": mpc_paths,
        }

        monkeypatch.setattr(standalone_module, "_MAX_MPC_RESPONSE_VALUES_PER_BATCH", 1_000_000)
        single_batch = construct_channel_matrix_from_mpc(**kwargs)
        monkeypatch.setattr(standalone_module, "_MAX_MPC_RESPONSE_VALUES_PER_BATCH", 8)
        one_path_batches = construct_channel_matrix_from_mpc(**kwargs)

        np.testing.assert_allclose(one_path_batches, single_batch, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# compute_svd_beamforming
# ---------------------------------------------------------------------------


class TestSVDBeamforming:
    def test_output_shapes(self):
        tx_w, rx_w, gain = compute_svd_beamforming(
            num_rows_tx=2,
            num_cols_tx=2,
            num_rows_rx=2,
            num_cols_rx=2,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            tx_orientation=(0, 0, 0),
            rx_orientation=(0, 0, 0),
        )
        assert tx_w.shape == (4,)
        assert rx_w.shape == (4,)
        assert gain > 0

    def test_weights_normalized(self):
        tx_w, rx_w, _ = compute_svd_beamforming(
            num_rows_tx=4,
            num_cols_tx=4,
            num_rows_rx=4,
            num_cols_rx=4,
            horizontal_spacing_m=0.005,
            vertical_spacing_m=0.005,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            tx_orientation=(0, 0, 0),
            rx_orientation=(0, 0, 0),
        )
        assert abs(np.linalg.norm(tx_w) - 1.0) < 1e-10
        assert abs(np.linalg.norm(rx_w) - 1.0) < 1e-10

    def test_single_path_mpc_matches_synthetic_los(self):
        direct_path = np.array([[0.0, 0.0, 10.0], [50.0, 0.0, 1.0]], dtype=np.float64)
        los_tx, los_rx, _ = compute_svd_beamforming(
            num_rows_tx=4,
            num_cols_tx=4,
            num_rows_rx=4,
            num_cols_rx=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            tx_orientation=(0.0, 0.0, 0.0),
            rx_orientation=(0.0, 0.0, 0.0),
        )
        mpc_tx, mpc_rx, _ = compute_svd_beamforming(
            num_rows_tx=4,
            num_cols_tx=4,
            num_rows_rx=4,
            num_cols_rx=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 10),
            rx_position=(50, 0, 1),
            tx_orientation=(0.0, 0.0, 0.0),
            rx_orientation=(0.0, 0.0, 0.0),
            mpc_paths=[(direct_path, -95.0)],
        )

        assert _vector_alignment(los_tx, mpc_tx) > 1.0 - 1e-10
        assert _vector_alignment(los_rx, mpc_rx) > 1.0 - 1e-10

    def test_svd_matches_los_for_oriented_arrays(self):
        los_result = compute_standalone_beamforming(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 0),
            rx_position=(10, 0, 0),
            tx_orientation=(np.pi / 2.0, 0.0, 0.0),
            rx_orientation=(-np.pi / 2.0, 0.0, 0.0),
            steering_strategy="los",
        )
        svd_result = compute_standalone_beamforming(
            num_rows=4,
            num_cols=4,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0, 0, 0),
            rx_position=(10, 0, 0),
            tx_orientation=(np.pi / 2.0, 0.0, 0.0),
            rx_orientation=(-np.pi / 2.0, 0.0, 0.0),
            steering_strategy="svd",
        )

        assert los_result is not None
        assert svd_result is not None

        los_pair = los_result["pairs"][0]
        svd_pair = svd_result["pairs"][0]
        los_tx = los_pair["tx"]["weights"][:, 0] + 1j * los_pair["tx"]["weights"][:, 1]
        los_rx = los_pair["rx"]["weights"][:, 0] + 1j * los_pair["rx"]["weights"][:, 1]
        svd_tx = svd_pair["tx"]["weights"][:, 0] + 1j * svd_pair["tx"]["weights"][:, 1]
        svd_rx = svd_pair["rx"]["weights"][:, 0] + 1j * svd_pair["rx"]["weights"][:, 1]

        assert _vector_alignment(los_tx, svd_tx) > 1.0 - 1e-10
        assert _vector_alignment(los_rx, svd_rx) > 1.0 - 1e-10
