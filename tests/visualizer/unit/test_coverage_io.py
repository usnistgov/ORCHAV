"""Coverage I/O tests using synthetic data."""

from __future__ import annotations

import h5py
import numpy as np

from tests.visualizer.fixtures.sample_coverage_maps import (
    create_sample_coverage_data,
    save_sample_coverage_hdf5,
)


def test_save_sample_coverage_hdf5(tmp_path):
    data = create_sample_coverage_data()
    output = tmp_path / "coverage_map.hdf5"

    save_sample_coverage_hdf5(output, data)
    assert output.exists()

    with h5py.File(output, "r") as f:
        np.testing.assert_array_equal(f["grid_origin"][:], data["grid_origin"])
        np.testing.assert_array_equal(f["grid_shape"][:], data["grid_shape"])
        np.testing.assert_array_equal(f["values"][:], data["values"])
        assert f.attrs["metric_name"] == data["metric_name"]
        assert f.attrs["value_min"] == data["value_min"]
        assert f.attrs["value_max"] == data["value_max"]
        assert f.attrs["metadata_scenario"] == data["metadata"]["scenario"]


def test_sample_points_cover_all_cells():
    data = create_sample_coverage_data(nx=3, ny=2, heights=(1.0,))
    expected_points = 3 * 2 * 1
    assert len(data["points"]) == expected_points

    xs = np.unique(data["points"][:, 0])
    ys = np.unique(data["points"][:, 1])
    assert xs.size == 3
    assert ys.size == 2
