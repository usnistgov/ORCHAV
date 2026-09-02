"""Standalone beamforming computation for visualization without frame data.

This module provides functions to compute beamforming weights independently
of ray-tracing results, allowing users to explore beam patterns interactively.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from shared.logging import get_logger

from ..utils.antenna_utils import get_element_positions

logger = get_logger(__name__)

_C = 299_792_458.0  # speed of light (m/s)

# Bound the two complex path-by-element response matrices used while assembling
# an MPC channel.  The channel matrix itself is necessarily RX-by-TX, but this
# keeps the additional batch workspace to roughly 16 MiB for complex128 data.
_MAX_MPC_RESPONSE_VALUES_PER_BATCH = 1_000_000


def compute_steering_weights(
    num_rows: int,
    num_cols: int,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    carrier_frequency_hz: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute phase-based steering weights for a planar antenna array.

    Args:
        num_rows: Number of rows in the antenna array
        num_cols: Number of columns in the antenna array
        horizontal_spacing_m: Element spacing along columns (meters)
        vertical_spacing_m: Element spacing along rows (meters)
        carrier_frequency_hz: Carrier frequency in Hz
        azimuth_deg: Azimuth steering angle in degrees (-180 to 180)
        elevation_deg: Elevation steering angle in degrees (-90 to 90)

    Returns:
        Tuple of (weights, element_positions, beam_gain):
        - weights: Complex array of shape (num_elements,)
        - element_positions: Array of shape (num_elements, 3) in meters
        - beam_gain: Linear beam gain
    """
    wavelength = _C / float(carrier_frequency_hz)
    if wavelength <= 0:
        raise ValueError("carrier_frequency_hz must be positive")

    k = 2.0 * math.pi / wavelength
    theta = math.radians(90.0 - elevation_deg)
    phi = math.radians(azimuth_deg)

    positions = get_element_positions(num_rows, num_cols, horizontal_spacing_m, vertical_spacing_m)

    phase = np.exp(
        1j
        * k
        * (
            positions[:, 0] * math.sin(theta) * math.cos(phi)
            + positions[:, 1] * math.sin(theta) * math.sin(phi)
            + positions[:, 2] * math.cos(theta)
        )
    )

    # Steering weights are conjugate of phase (to steer toward that direction)
    weights = np.conjugate(phase)
    norm = np.linalg.norm(weights)
    if norm > 0:
        weights = weights / norm
    else:
        weights = weights / math.sqrt(weights.size)

    gain = np.abs(np.sum(weights * phase)) ** 2

    return weights, positions, float(gain)


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return rotation matrix for yaw(Z), pitch(Y), roll(X) convention."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _normalize_direction(direction: np.ndarray) -> Optional[np.ndarray]:
    """Return a normalized 3-vector or ``None`` when the norm is too small."""
    arr = np.asarray(direction, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    arr = arr[:3]
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return None
    return arr / norm


def _coerce_orientation_radians(
    orientation: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return ``(yaw, pitch, roll)`` orientation in radians with safe padding."""
    arr = np.asarray(orientation, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    return tuple(float(value) for value in arr[:3])


def _world_to_local_direction(
    direction: np.ndarray, orientation_radians: tuple[float, float, float]
) -> Optional[np.ndarray]:
    """Transform a world-frame direction into the array local frame."""
    unit_direction = _normalize_direction(direction)
    if unit_direction is None:
        return None
    rotation = _rotation_matrix(*_coerce_orientation_radians(orientation_radians))
    local_direction = rotation.T @ unit_direction
    return _normalize_direction(local_direction)


def _direction_to_angles(direction: np.ndarray) -> tuple[float, float]:
    """Convert a direction vector into azimuth/elevation degrees."""
    unit_direction = _normalize_direction(direction)
    if unit_direction is None:
        return 0.0, 0.0

    azimuth_deg = math.degrees(math.atan2(unit_direction[1], unit_direction[0]))
    elevation_deg = math.degrees(math.asin(np.clip(unit_direction[2], -1.0, 1.0)))
    return float(azimuth_deg), float(elevation_deg)


def _array_response_for_world_direction(
    element_positions: np.ndarray,
    k: float,
    world_direction: np.ndarray,
    orientation_radians: tuple[float, float, float],
) -> np.ndarray:
    """Return array response for a world-frame propagation/steering direction."""
    local_direction = _world_to_local_direction(world_direction, orientation_radians)
    if local_direction is None:
        return np.ones(len(element_positions), dtype=np.complex128)
    return np.exp(1j * k * (element_positions @ local_direction))


def _array_responses_for_world_directions(
    element_positions: np.ndarray,
    k: float,
    world_directions: np.ndarray,
    array_rotation: np.ndarray,
) -> np.ndarray:
    """Return array responses for a batch of world-frame directions.

    ``array_rotation`` maps local coordinates to world coordinates, so row-wise
    world directions are transformed to the local frame with ``directions @ R``.
    Zero-length directions retain the scalar helper's all-ones response.
    """
    directions = np.asarray(world_directions, dtype=np.float64).reshape(-1, 3)
    direction_norms = np.linalg.norm(directions, axis=1)
    zero_directions = direction_norms < 1e-12
    safe_direction_norms = np.where(zero_directions, 1.0, direction_norms)
    unit_directions = directions / safe_direction_norms[:, np.newaxis]

    local_directions = unit_directions @ array_rotation
    local_norms = np.linalg.norm(local_directions, axis=1)
    zero_local_directions = local_norms < 1e-12
    safe_local_norms = np.where(zero_local_directions, 1.0, local_norms)
    local_directions = local_directions / safe_local_norms[:, np.newaxis]

    responses = np.exp(1j * k * (local_directions @ element_positions.T))
    zero_response = zero_directions | zero_local_directions
    if np.any(zero_response):
        responses[zero_response] = 1.0 + 0.0j
    return responses


def compute_los_angles(
    tx_position: tuple[float, float, float],
    rx_position: tuple[float, float, float],
    orientation_radians: Optional[tuple[float, float, float]] = None,
) -> tuple[float, float]:
    """Compute azimuth/elevation angles from TX to RX.

    When ``orientation_radians`` is provided, the returned angles are expressed
    in the device local frame rather than the world frame.

    Args:
        tx_position: TX position (x, y, z) in meters
        rx_position: RX position (x, y, z) in meters
        orientation_radians: Optional device orientation ``(yaw, pitch, roll)``
            in radians for local-frame angles.

    Returns:
        Tuple of ``(azimuth_deg, elevation_deg)`` in degrees
    """
    direction = np.array(
        [
            rx_position[0] - tx_position[0],
            rx_position[1] - tx_position[1],
            rx_position[2] - tx_position[2],
        ],
        dtype=np.float64,
    )

    if orientation_radians is not None:
        local_direction = _world_to_local_direction(direction, orientation_radians)
        if local_direction is None:
            return 0.0, 0.0
        return _direction_to_angles(local_direction)

    return _direction_to_angles(direction)


def construct_channel_matrix_from_mpc(
    num_rows_tx: int,
    num_cols_tx: int,
    num_rows_rx: int,
    num_cols_rx: int,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    carrier_frequency_hz: float,
    tx_position: tuple[float, float, float],
    rx_position: tuple[float, float, float],
    tx_orientation: tuple[float, float, float],
    rx_orientation: tuple[float, float, float],
    mpc_paths: Optional[list[tuple[np.ndarray, float]]] = None,
) -> np.ndarray:
    """Construct channel matrix from MPC paths or synthetic LOS path.

    Args:
        num_rows_tx, num_cols_tx: TX array dimensions
        num_rows_rx, num_cols_rx: RX array dimensions
        horizontal_spacing_m, vertical_spacing_m: Element spacing
        carrier_frequency_hz: Carrier frequency
        tx_position, rx_position: TX/RX positions
        tx_orientation, rx_orientation: TX/RX orientations (radians)
        mpc_paths: List of (path_vertices, path_loss_db) tuples, or None for synthetic LOS

    Returns:
        Channel matrix H of shape (num_rx_elements, num_tx_elements)
    """
    wavelength = _C / float(carrier_frequency_hz)
    k = 2.0 * math.pi / wavelength

    num_tx_elements = num_rows_tx * num_cols_tx
    num_rx_elements = num_rows_rx * num_cols_rx

    tx_elem_pos = get_element_positions(
        num_rows_tx, num_cols_tx, horizontal_spacing_m, vertical_spacing_m
    )
    rx_elem_pos = get_element_positions(
        num_rows_rx, num_cols_rx, horizontal_spacing_m, vertical_spacing_m
    )

    H = np.zeros((num_rx_elements, num_tx_elements), dtype=np.complex128)

    if mpc_paths is None or len(mpc_paths) == 0:
        direction = np.array(
            [
                rx_position[0] - tx_position[0],
                rx_position[1] - tx_position[1],
                rx_position[2] - tx_position[2],
            ],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(direction))
        if distance > 1e-6:
            tx_response = _array_response_for_world_direction(
                tx_elem_pos,
                k,
                direction,
                tx_orientation,
            )
            rx_response = _array_response_for_world_direction(
                rx_elem_pos,
                k,
                -direction,
                rx_orientation,
            )
            path_gain = np.exp(-1j * k * distance) / (4 * math.pi * distance)
            H = np.outer(rx_response, np.conj(tx_response)) * path_gain
        return H

    tx_directions: list[np.ndarray] = []
    rx_directions: list[np.ndarray] = []
    path_gains: list[complex] = []

    for path_vertices, path_loss_db in mpc_paths:
        vertices = np.asarray(path_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 2:
            continue

        if not np.isfinite(path_loss_db):
            continue

        segments = np.diff(vertices, axis=0)
        path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        if path_length <= 0.0:
            continue

        path_gain_linear = 10.0 ** (-abs(path_loss_db) / 20.0)
        path_phase = np.exp(-1j * k * path_length)
        path_gains.append(complex(path_gain_linear * path_phase))

        # Match the scalar response helper's padding/truncation for unusual
        # vertex arrays while keeping the common three-coordinate path fast.
        tx_direction = np.asarray(vertices[1] - vertices[0], dtype=np.float64).reshape(-1)
        rx_direction = np.asarray(vertices[-2] - vertices[-1], dtype=np.float64).reshape(-1)
        if tx_direction.size < 3:
            tx_direction = np.pad(tx_direction, (0, 3 - tx_direction.size))
        if rx_direction.size < 3:
            rx_direction = np.pad(rx_direction, (0, 3 - rx_direction.size))
        tx_directions.append(tx_direction[:3])
        rx_directions.append(rx_direction[:3])

    if not path_gains:
        return H

    tx_directions_array = np.asarray(tx_directions, dtype=np.float64)
    rx_directions_array = np.asarray(rx_directions, dtype=np.float64)
    path_gains_array = np.asarray(path_gains, dtype=np.complex128)
    tx_rotation = _rotation_matrix(*_coerce_orientation_radians(tx_orientation))
    rx_rotation = _rotation_matrix(*_coerce_orientation_radians(rx_orientation))

    response_values_per_path = max(1, num_tx_elements + num_rx_elements)
    batch_size = max(
        1,
        _MAX_MPC_RESPONSE_VALUES_PER_BATCH // response_values_per_path,
    )

    for start in range(0, len(path_gains_array), batch_size):
        stop = min(start + batch_size, len(path_gains_array))
        tx_responses = _array_responses_for_world_directions(
            tx_elem_pos,
            k,
            tx_directions_array[start:stop],
            tx_rotation,
        )
        rx_responses = _array_responses_for_world_directions(
            rx_elem_pos,
            k,
            rx_directions_array[start:stop],
            rx_rotation,
        )

        # Matrix multiplication evaluates the sum of weighted outer products
        # without allocating a path-by-RX-by-TX tensor.
        weighted_rx = rx_responses.T * path_gains_array[start:stop]
        H += weighted_rx @ np.conjugate(tx_responses)

    return H


def compute_svd_beamforming(
    num_rows_tx: int,
    num_cols_tx: int,
    num_rows_rx: int,
    num_cols_rx: int,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    carrier_frequency_hz: float,
    tx_position: tuple[float, float, float],
    rx_position: tuple[float, float, float],
    tx_orientation: tuple[float, float, float],
    rx_orientation: tuple[float, float, float],
    mpc_paths: Optional[list[tuple[np.ndarray, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute optimal beamforming weights using SVD of channel matrix.

    Returns:
        Tuple of (tx_weights, rx_weights, optimal_gain)
    """
    # Construct channel matrix
    H = construct_channel_matrix_from_mpc(
        num_rows_tx,
        num_cols_tx,
        num_rows_rx,
        num_cols_rx,
        horizontal_spacing_m,
        vertical_spacing_m,
        carrier_frequency_hz,
        tx_position,
        rx_position,
        tx_orientation,
        rx_orientation,
        mpc_paths,
    )

    # Sanitise: NaN/Inf entries cause SVD non-convergence
    if not np.all(np.isfinite(H)):
        logger.warning("Channel matrix contains NaN/Inf — replacing with zeros")
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    # H = U Σ V^H; the leading singular vectors maximize pair gain.
    try:
        U, s, Vh = np.linalg.svd(H, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("SVD did not converge — falling back to LOS steering")
        # Fall back: compute LOS steering weights instead
        az, el = compute_los_angles(tx_position, rx_position, tx_orientation)
        tx_weights, _, tx_gain = compute_steering_weights(
            num_rows_tx,
            num_cols_tx,
            horizontal_spacing_m,
            vertical_spacing_m,
            carrier_frequency_hz,
            az,
            el,
        )
        rx_az, rx_el = compute_los_angles(rx_position, tx_position, rx_orientation)
        rx_weights, _, rx_gain = compute_steering_weights(
            num_rows_rx,
            num_cols_rx,
            horizontal_spacing_m,
            vertical_spacing_m,
            carrier_frequency_hz,
            rx_az,
            rx_el,
        )
        return tx_weights, rx_weights, float(tx_gain)

    # Optimal weights are first singular vectors
    # The SVD gives us optimal beamforming vectors
    # For transmission: Vh[0, :] is the optimal TX beamforming vector
    # For reception: U[:, 0] is the optimal RX beamforming vector
    # These are already in the correct form for the channel matrix
    # Visualization uses steering weights: the conjugate of the array response.
    # Since H uses array responses, the SVD vectors are also array responses
    # Conjugation converts the array response into steering weights.
    tx_weights = Vh[0, :]  # Already conjugated because Vh = V^H
    rx_weights = U[:, 0].conj()  # Conjugate to get steering weights

    # Normalize weights
    tx_norm = np.linalg.norm(tx_weights)
    if tx_norm > 0:
        tx_weights = tx_weights / tx_norm
    else:
        tx_weights = tx_weights / math.sqrt(len(tx_weights))

    rx_norm = np.linalg.norm(rx_weights)
    if rx_norm > 0:
        rx_weights = rx_weights / rx_norm
    else:
        rx_weights = rx_weights / math.sqrt(len(rx_weights))

    # Optimal gain is first singular value squared
    optimal_gain = float(np.abs(s[0]) ** 2) if len(s) > 0 else 0.0

    return tx_weights, rx_weights, optimal_gain


def compute_standalone_beamforming(
    num_rows: int,
    num_cols: int,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    carrier_frequency_hz: float,
    tx_position: tuple[float, float, float],
    rx_position: tuple[float, float, float],
    tx_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    steering_strategy: str = "manual",
    azimuth_deg: Optional[float] = None,
    elevation_deg: Optional[float] = None,
    mpc_paths: Optional[list[tuple[np.ndarray, float]]] = None,
) -> Optional[dict]:
    """Compute standalone beamforming weights for TX and RX arrays.

    Args:
        num_rows: Number of rows in antenna arrays
        num_cols: Number of columns in antenna arrays
        horizontal_spacing_m: Element spacing along columns (meters)
        vertical_spacing_m: Element spacing along rows (meters)
        carrier_frequency_hz: Carrier frequency in Hz
        tx_position: TX position (x, y, z) in meters
        rx_position: RX position (x, y, z) in meters
        tx_orientation: TX orientation (yaw, pitch, roll) in radians
        rx_orientation: RX orientation (yaw, pitch, roll) in radians
        steering_strategy: "manual", "los", or "svd"
        azimuth_deg: Manual azimuth angle (required for "manual" strategy)
        elevation_deg: Manual elevation angle (required for "manual" strategy)

    Returns:
        Dictionary with beamforming data structure compatible with frame-based beamforming,
        or None if computation fails
    """
    try:
        # Determine steering angles based on strategy
        if steering_strategy == "los":
            az, el = compute_los_angles(tx_position, rx_position, tx_orientation)
            tx_azimuth = az
            tx_elevation = el
            # RX steers back toward TX - compute angles from RX to TX
            rx_az, rx_el = compute_los_angles(rx_position, tx_position, rx_orientation)
            rx_azimuth = rx_az
            rx_elevation = rx_el
        elif steering_strategy == "manual":
            if azimuth_deg is None or elevation_deg is None:
                logger.warning("Manual steering requires azimuth_deg and elevation_deg")
                return None
            tx_azimuth = float(azimuth_deg)
            tx_elevation = float(elevation_deg)
            # For manual, RX uses same angles (can be customized later)
            rx_azimuth = float(azimuth_deg)
            rx_elevation = float(elevation_deg)
        elif steering_strategy == "svd":
            # SVD-based optimal beamforming
            tx_weights, rx_weights, optimal_gain = compute_svd_beamforming(
                num_rows_tx=num_rows,
                num_cols_tx=num_cols,
                num_rows_rx=num_rows,
                num_cols_rx=num_cols,
                horizontal_spacing_m=horizontal_spacing_m,
                vertical_spacing_m=vertical_spacing_m,
                carrier_frequency_hz=carrier_frequency_hz,
                tx_position=tx_position,
                rx_position=rx_position,
                tx_orientation=tx_orientation,
                rx_orientation=rx_orientation,
                mpc_paths=mpc_paths,
            )
            tx_positions = get_element_positions(
                num_rows, num_cols, horizontal_spacing_m, vertical_spacing_m
            )
            rx_positions = tx_positions.copy()
            tx_gain = optimal_gain
            rx_gain = optimal_gain
            # SVD provides no explicit angles, so LOS supplies the display direction.
            tx_azimuth, tx_elevation = compute_los_angles(
                tx_position,
                rx_position,
                tx_orientation,
            )
            rx_azimuth, rx_elevation = compute_los_angles(
                rx_position,
                tx_position,
                rx_orientation,
            )
        else:
            logger.warning(f"Unknown steering strategy: {steering_strategy}")
            return None

        if steering_strategy != "svd":
            tx_weights, tx_positions, tx_gain = compute_steering_weights(
                num_rows=num_rows,
                num_cols=num_cols,
                horizontal_spacing_m=horizontal_spacing_m,
                vertical_spacing_m=vertical_spacing_m,
                carrier_frequency_hz=carrier_frequency_hz,
                azimuth_deg=tx_azimuth,
                elevation_deg=tx_elevation,
            )

            rx_weights, rx_positions, rx_gain = compute_steering_weights(
                num_rows=num_rows,
                num_cols=num_cols,
                horizontal_spacing_m=horizontal_spacing_m,
                vertical_spacing_m=vertical_spacing_m,
                carrier_frequency_hz=carrier_frequency_hz,
                azimuth_deg=rx_azimuth,
                elevation_deg=rx_elevation,
            )

        # Frame payloads store complex weights as trailing ``[real, imag]`` pairs.
        tx_weights_stacked = np.stack((tx_weights.real, tx_weights.imag), axis=-1).astype(
            np.float32
        )
        rx_weights_stacked = np.stack((rx_weights.real, rx_weights.imag), axis=-1).astype(
            np.float32
        )

        beamforming_data = {
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx_name": "standalone_tx",
                    "rx_name": "standalone_rx",
                    "tx": {
                        "device_name": "standalone_tx",
                        "device_index": 0,
                        "role": "tx",
                        "array_rows": num_rows,
                        "array_cols": num_cols,
                        "horizontal_spacing_m": float(horizontal_spacing_m),
                        "vertical_spacing_m": float(vertical_spacing_m),
                        "carrier_frequency_hz": float(carrier_frequency_hz),
                        "steering_azimuth_deg": tx_azimuth,
                        "steering_elevation_deg": tx_elevation,
                        "sector": tx_azimuth,
                        "strategy": steering_strategy,
                        "weights": tx_weights_stacked,
                        "element_positions": tx_positions.astype(np.float32),
                        "beam_gain": tx_gain,
                    },
                    "rx": {
                        "device_name": "standalone_rx",
                        "device_index": 0,
                        "role": "rx",
                        "array_rows": num_rows,
                        "array_cols": num_cols,
                        "horizontal_spacing_m": float(horizontal_spacing_m),
                        "vertical_spacing_m": float(vertical_spacing_m),
                        "carrier_frequency_hz": float(carrier_frequency_hz),
                        "steering_azimuth_deg": rx_azimuth,
                        "steering_elevation_deg": rx_elevation,
                        "sector": rx_azimuth,
                        "strategy": steering_strategy,
                        "weights": rx_weights_stacked,
                        "element_positions": rx_positions.astype(np.float32),
                        "beam_gain": rx_gain,
                    },
                }
            ]
        }

        return beamforming_data

    except (ValueError, TypeError) as exc:
        logger.exception("Standalone beamforming computation failed: %s", exc)
        return None
