"""Synthetic coverage map data for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def create_sample_coverage_data(
    nx: int = 4,
    ny: int = 5,
    heights: Iterable[float] | None = None,
    metric_name: str = "RSRP (dBm)",
    seed: int = 7,
) -> dict:
    """Return a deterministic synthetic coverage dataset."""
    if heights is None:
        heights = (1.5, 3.0)
    heights = tuple(float(h) for h in heights)

    rng = np.random.default_rng(seed)
    grid_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grid_spacing = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    grid_shape = np.array([nx, ny, len(heights)], dtype=np.int32)

    values_3d = rng.uniform(-110.0, -70.0, size=(len(heights), nx, ny)).astype(np.float32)
    value_min = float(values_3d.min())
    value_max = float(values_3d.max())

    points = []
    for h_idx, height in enumerate(heights):
        for i in range(nx):
            for j in range(ny):
                x = grid_origin[0] + i * grid_spacing[0]
                y = grid_origin[1] + j * grid_spacing[1]
                points.append([x, y, height])
    points = np.array(points, dtype=np.float32)

    tx_positions = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    rx_positions = np.array([[5.0, 0.0, 1.5]], dtype=np.float32)

    metadata = {
        "scenario": "synthetic_lobby",
        "description": "Deterministic sample coverage data for unit tests",
    }

    return {
        "grid_origin": grid_origin,
        "grid_spacing": grid_spacing,
        "grid_shape": grid_shape,
        "points": points,
        "values": values_3d.reshape(len(heights), -1),
        "values_3d": values_3d,
        "value_min": value_min,
        "value_max": value_max,
        "metric_name": metric_name,
        "heights": np.array(heights, dtype=np.float32),
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "metadata": metadata,
    }


def save_sample_coverage_hdf5(path: Path, coverage_data: dict | None = None) -> Path:
    """Write the provided (or freshly generated) coverage data to an HDF5 file."""
    if coverage_data is None:
        coverage_data = create_sample_coverage_data()

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("grid_origin", data=coverage_data["grid_origin"])
        f.create_dataset("grid_spacing", data=coverage_data["grid_spacing"])
        f.create_dataset("grid_shape", data=coverage_data["grid_shape"])
        f.create_dataset("points", data=coverage_data["points"])
        f.create_dataset("values", data=coverage_data["values"])
        f.create_dataset("tx_positions", data=coverage_data["tx_positions"])
        f.create_dataset("rx_positions", data=coverage_data["rx_positions"])

        f.attrs["metric_name"] = coverage_data["metric_name"]
        f.attrs["value_min"] = coverage_data["value_min"]
        f.attrs["value_max"] = coverage_data["value_max"]
        for key, value in coverage_data["metadata"].items():
            f.attrs[f"metadata_{key}"] = value

    return path
