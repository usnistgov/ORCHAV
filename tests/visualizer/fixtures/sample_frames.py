"""
Sample frame data generators for testing.

This module provides functions to create synthetic frame data that can be used
in tests without requiring actual simulation data files.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


def create_sample_mpc_data(
    n_mpcs: int = 100, n_tx: int = 1, n_rx: int = 1, seed: Optional[int] = 42
) -> Dict[str, np.ndarray]:
    """Create synthetic MPC data for testing.

    Args:
        n_mpcs: Number of multi-path components to generate
        n_tx: Number of transmitters
        n_rx: Number of receivers
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing MPC data with keys matching Sionna RT format
    """
    if seed is not None:
        np.random.seed(seed)

    return {
        # Delays (in seconds)
        "tau": np.random.rand(n_tx, n_rx, n_mpcs) * 1e-6,
        # TX angles (elevation and azimuth in radians)
        "theta_t": np.random.rand(n_tx, n_rx, n_mpcs) * np.pi,
        "phi_t": np.random.rand(n_tx, n_rx, n_mpcs) * 2 * np.pi,
        # RX angles (elevation and azimuth in radians)
        "theta_r": np.random.rand(n_tx, n_rx, n_mpcs) * np.pi,
        "phi_r": np.random.rand(n_tx, n_rx, n_mpcs) * 2 * np.pi,
        # Channel coefficients (complex 2x2 matrices)
        "a": (
            np.random.randn(n_tx, n_rx, n_mpcs, 2, 2)
            + 1j * np.random.randn(n_tx, n_rx, n_mpcs, 2, 2)
        ),
        # Interaction types (1=specular, 2=diffuse, 4=refraction, 8=diffraction)
        "types": np.random.choice([1, 2, 4, 8], size=(n_tx, n_rx, n_mpcs)),
        # Number of interactions per path
        "num_paths": np.random.randint(0, 5, size=(n_tx, n_rx, n_mpcs)),
    }


def create_sample_frame_data(
    n_mpcs: int = 100,
    n_tx: int = 1,
    n_rx: int = 1,
    tx_position: Optional[np.ndarray] = None,
    rx_position: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> Dict:
    """Create complete sample frame data.

    Args:
        n_mpcs: Number of multi-path components
        n_tx: Number of transmitters
        n_rx: Number of receivers
        tx_position: TX position [x, y, z], defaults to [0, 0, 1.5]
        rx_position: RX position [x, y, z], defaults to [10, 0, 1.5]
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing complete frame data
    """
    if tx_position is None:
        tx_position = np.array([[0.0, 0.0, 1.5]])
    if rx_position is None:
        rx_position = np.array([[10.0, 0.0, 1.5]])

    mpc_data = create_sample_mpc_data(n_mpcs, n_tx, n_rx, seed)

    return {
        "mpc_data": mpc_data,
        "tx_positions": tx_position,
        "rx_positions": rx_position,
        "tx_orientations": np.array([[0.0, 0.0, 0.0]] * n_tx),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]] * n_rx),
        "timestamp": 0.0,
    }


def save_sample_frame_hdf5(
    filepath: Path,
    n_frames: int = 10,
    n_mpcs: int = 100,
    n_tx: int = 1,
    n_rx: int = 1,
    seed: Optional[int] = 42,
) -> None:
    """Save sample frame data to HDF5 file.

    Args:
        filepath: Path to save HDF5 file
        n_frames: Number of frames to generate
        n_mpcs: Number of MPCs per frame
        n_tx: Number of transmitters
        n_rx: Number of receivers
        seed: Random seed for reproducibility
    """
    with h5py.File(filepath, "w") as f:
        # Create groups
        frames_group = f.create_group("frames")

        for frame_idx in range(n_frames):
            frame_seed = seed + frame_idx if seed is not None else None
            frame_data = create_sample_frame_data(n_mpcs, n_tx, n_rx, seed=frame_seed)

            # Create frame group
            frame_group = frames_group.create_group(f"frame_{frame_idx}")

            # Save MPC data
            mpc_group = frame_group.create_group("mpc_data")
            for key, value in frame_data["mpc_data"].items():
                mpc_group.create_dataset(key, data=value)

            # Save positions and orientations
            frame_group.create_dataset("tx_positions", data=frame_data["tx_positions"])
            frame_group.create_dataset("rx_positions", data=frame_data["rx_positions"])
            frame_group.create_dataset("tx_orientations", data=frame_data["tx_orientations"])
            frame_group.create_dataset("rx_orientations", data=frame_data["rx_orientations"])
            frame_group.create_dataset("timestamp", data=frame_data["timestamp"])

        # Save metadata
        f.attrs["n_frames"] = n_frames
        f.attrs["n_tx"] = n_tx
        f.attrs["n_rx"] = n_rx


def create_sample_bounce_points(
    n_bounces: int = 50,
    bounds: Tuple[float, float, float] = (20.0, 20.0, 10.0),
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Create sample bounce point positions.

    Args:
        n_bounces: Number of bounce points
        bounds: (x_max, y_max, z_max) bounds for positions
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_bounces, 3) with bounce point positions
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.rand(n_bounces, 3) * np.array(bounds)


def create_empty_frame_data() -> Dict:
    """Create empty frame data for edge case testing.

    Returns:
        Dictionary with empty arrays in correct format
    """
    return {
        "mpc_data": {
            "tau": np.array([]).reshape(0, 0, 0),
            "theta_t": np.array([]).reshape(0, 0, 0),
            "phi_t": np.array([]).reshape(0, 0, 0),
            "theta_r": np.array([]).reshape(0, 0, 0),
            "phi_r": np.array([]).reshape(0, 0, 0),
            "a": np.array([]).reshape(0, 0, 0, 2, 2),
            "types": np.array([]).reshape(0, 0, 0),
            "num_paths": np.array([]).reshape(0, 0, 0),
        },
        "tx_positions": np.array([]).reshape(0, 3),
        "rx_positions": np.array([]).reshape(0, 3),
        "tx_orientations": np.array([]).reshape(0, 3),
        "rx_orientations": np.array([]).reshape(0, 3),
        "timestamp": 0.0,
    }
