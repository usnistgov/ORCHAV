"""Saved RF cells retain their coordinates and scale in standalone 3D exports."""

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from generator.figures.coverage.spatial import (
    _coverage_cell_triangles,
    create_coverage_3d_figure,
)
from generator.figures.scene import drawing
from generator.io.storage.coverage_writer import save_coverage_hdf5


def test_cells_keep_anisotropic_spacing_height_and_missing_data_holes():
    values = np.ma.masked_invalid([[1.0, np.nan], [3.0, 4.0]])
    faces, samples = _coverage_cell_triangles(values, np.array([10, 20, 0]), np.array([2, 5]), 30)
    assert faces.shape == (6, 3, 3)
    np.testing.assert_array_equal(samples, [1, 1, 3, 3, 4, 4])
    np.testing.assert_array_equal(faces[:, :, 2], 30)
    np.testing.assert_array_equal(faces.min(axis=(0, 1)), [10, 20, 30])
    np.testing.assert_array_equal(faces.max(axis=(0, 1)), [14, 30, 30])
    centers = faces.mean(axis=1)
    assert not np.any((centers[:, 0] > 12) & (centers[:, 1] < 25))


@pytest.fixture
def coverage_file(tmp_path):
    path = tmp_path / "coverage.h5"
    save_coverage_hdf5(
        {
            "grid_origin": np.array([0, 0, 1.5]),
            "grid_spacing": np.array([2, 5]),
            "grid_shape": np.array([2, 2, 2]),
            "heights": np.array([1.5, 30.0]),
            "path_gain_linear": np.array([1e-8, 0, 1e-9, 1e-10, 1e-9, 1e-10, 1e-11, 0]).reshape(
                1, 2, 1, 2, 2
            ),
            "derived": {},
            "metric_name": "best_path_loss_db",
            "tx_positions": np.array([[1, 2, 5.0]]),
            "rx_positions": np.empty((0, 3)),
            "tx_names": ["TX1"],
            "rx_names": [],
            "tx_power_dbm": np.array([20.0]),
            "value_min": 80.0,
            "value_max": 110.0,
            "metadata": {"tx_mode": "best_server"},
        },
        path,
        compression=None,
    )
    return path


@pytest.mark.parametrize("metric", ["best_path_loss_db", "path_gain_linear/TX1", "serving_tx"])
def test_export_is_headless_preserves_source_and_uses_native_geometry(
    tmp_path, coverage_file, monkeypatch, metric
):
    def no_raster(*args, **kwargs):
        pytest.fail("3D export must not use the legacy pixel rasterizer")

    monkeypatch.setattr(drawing, "create_rasterized_floor_plan", no_raster)
    geometry = [
        {
            "name": "ground",
            "mesh": SimpleNamespace(
                vertices=np.array([[0, 0, 0], [4, 0, 0], [4, 10, 0], [0, 10, 0]]),
                triangles=np.array([[0, 1, 2], [0, 2, 3]]),
            ),
        }
    ]
    monkeypatch.setattr("shared.geometry.cache.get_scene_geometry", lambda **kwargs: geometry)
    saved = []
    original_save = Figure.savefig

    def capture(fig, *args, **kwargs):
        saved.append(fig)
        return original_save(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    before = coverage_file.read_bytes()
    destination = tmp_path / "figures" / "slice.png"
    assert (
        create_coverage_3d_figure(
            coverage_file, destination, scenario_context=object(), metric=metric
        )
        == destination
    )
    assert destination.read_bytes().startswith(b"\x89PNG")
    assert coverage_file.read_bytes() == before
    ax, colorbar_ax = saved[0].axes
    # The ground is behind the above-ground coverage cells and TX markers.
    assert ax.collections[0].get_zorder() < ax.collections[1].get_zorder()
    if metric == "best_path_loss_db":
        # The second height's 110 dB maximum participates in the shared scale.
        assert colorbar_ax.get_ylim() == pytest.approx((80, 110))
    if metric == "serving_tx":
        assert [t.get_text() for t in colorbar_ax.get_yticklabels()] == ["No service", "TX1"]
    drawing.clear_scene_figure_caches()


def test_invalid_height_does_not_write_a_figure(tmp_path, coverage_file):
    destination = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="height_index"):
        create_coverage_3d_figure(coverage_file, destination, height_index=2)
    assert not destination.exists()


def test_empty_scalar_slice_exports_without_nan_bounds_warnings(
    tmp_path, coverage_file, monkeypatch
):
    import warnings

    import h5py

    with h5py.File(coverage_file, "r+") as data:
        data["values/path_gain_linear"][...] = 0
    saved = []
    original_save = Figure.savefig

    def capture(fig, *args, **kwargs):
        saved.append(fig)
        return original_save(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        output = create_coverage_3d_figure(coverage_file, tmp_path / "empty.png")
    assert output.is_file()
    assert any("No coverage data" in text.get_text() for text in saved[0].axes[0].texts)
