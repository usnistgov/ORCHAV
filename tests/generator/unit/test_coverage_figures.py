from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import LogNorm

from generator.figures.coverage import (
    _coverage_extent_from_grid,
    _load_coverage,
    create_coverage_comparison_figure,
    create_coverage_distribution_figure,
    create_coverage_height_evolution_animation,
    create_coverage_metric_guide,
    create_coverage_statistics_plot,
    create_coverage_visualization,
    serving_tx_class_labels,
)
from generator.figures.coverage.figures import (
    _compute_clim_for_arrays,
    _coverage_height_output_path,
    _coverage_metric_style,
    _coverage_scalar_norm,
    _resolve_metric_layer,
)
from generator.io.storage.coverage_writer import save_coverage_hdf5
from shared.coverage.schema import COVERAGE_HDF5_SCHEMA_VERSION, COVERAGE_HDF5_STORAGE_LAYOUT


def test_coverage_figures_write_multi_height_figures(tmp_path: Path):
    # Build a tiny schema-v2 coverage HDF5 with two heights.
    nx, ny, nz = 3, 2, 2
    values = np.linspace(1.0, 2.0, nx * ny * nz).astype(np.float32).reshape(1, nz, ny, nx)
    grid_origin = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    grid_spacing = np.array([1.0, 1.0], dtype=np.float32)  # XY only
    grid_shape = np.array([nx, ny, nz], dtype=np.int32)
    heights = np.array([1.0, 2.0], dtype=np.float32)
    out_cov = tmp_path / "cov.h5"
    save_coverage_hdf5(
        {
            "grid_origin": grid_origin,
            "grid_spacing": grid_spacing,
            "grid_shape": grid_shape,
            "heights": heights,
            "path_gain_linear": np.ones((1, nz, 1, ny, nx), dtype=np.float32),
            "derived": {"best_path_loss_db": values},
            "metric_name": "best_path_loss_db",
            "tx_positions": np.empty((0, 3), dtype=np.float32),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": [],
            "rx_names": [],
            "tx_power_dbm": np.empty((0,), dtype=np.float32),
            "value_min": float(np.nanmin(values)),
            "value_max": float(np.nanmax(values)),
            "metadata": {"tx_mode": "best_server"},
        },
        out_cov,
        compression=None,
    )

    out_fig = tmp_path / "cov.png"
    generated = create_coverage_visualization(out_cov, out_fig, interpolation="bilinear")

    assert [path.name for path in generated] == [
        "cov_height-01_1m.png",
        "cov_height-02_2m.png",
    ]
    assert all(path.exists() for path in generated)


@pytest.mark.parametrize(
    ("height_index", "height_m", "expected_name"),
    [
        (0, 1.5, "coverage_maps_height-01_1.5m.png"),
        (1, 30.0, "coverage_maps_height-02_30m.png"),
        (2, -2.25, "coverage_maps_height-03_-2.25m.png"),
        (3, 1.5, "coverage_maps_height-04_1.5m.png"),
        (4, np.float32(1.2), "coverage_maps_height-05_1.2m.png"),
        (5, np.float32(0.1), "coverage_maps_height-06_0.1m.png"),
    ],
)
def test_coverage_height_output_path_is_readable_and_collision_safe(
    height_index: int,
    height_m: float,
    expected_name: str,
):
    output = _coverage_height_output_path(
        Path("coverage_maps.png"),
        height_index,
        height_m,
    )

    assert output.name == expected_name


def test_coverage_hdf5_writer_rejects_npz_suffix(tmp_path: Path):
    with pytest.raises(ValueError, match=r"end in \.h5 or \.hdf5"):
        save_coverage_hdf5({}, tmp_path / "coverage_maps.npz")


def test_coverage_figures_reject_npz_file(tmp_path: Path):
    coverage_file = tmp_path / "coverage_maps.npz"
    coverage_file.write_bytes(b"not a coverage file")

    with pytest.raises(ValueError, match=rf"schema-v{COVERAGE_HDF5_SCHEMA_VERSION} HDF5"):
        _load_coverage(coverage_file)


def test_coverage_metric_guide_writes_selected_metric_layers(tmp_path: Path):
    nx, ny, nz, ntx = 4, 3, 1, 2
    path_loss = np.stack(
        [
            np.linspace(90.0, 120.0, nx * ny).reshape(ny, nx),
            np.linspace(95.0, 125.0, nx * ny).reshape(ny, nx),
        ],
        axis=0,
    ).astype(np.float32)
    best_path_loss = np.min(path_loss, axis=0)
    serving_tx = np.argmin(path_loss, axis=0).astype(np.int16)

    coverage_file = tmp_path / "coverage_maps.h5"
    path_gain = np.power(10.0, -path_loss / 10.0).reshape(1, nz, ntx, ny, nx)
    save_coverage_hdf5(
        {
            "grid_origin": np.array([0.0, 0.0, 1.5], dtype=np.float32),
            "grid_spacing": np.array([5.0, 5.0], dtype=np.float32),
            "grid_shape": np.array([nx, ny, nz], dtype=np.int32),
            "heights": np.array([1.5], dtype=np.float32),
            "path_gain_linear": path_gain.astype(np.float32),
            "derived": {"serving_tx": serving_tx.reshape(1, nz, ny, nx)},
            "metric_name": "best_path_loss_db",
            "tx_positions": np.array(
                [[-10.0, 0.0, 35.0], [10.0, 0.0, 35.0]],
                dtype=np.float32,
            ),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": ["WestSector", "EastSector"],
            "rx_names": [],
            "tx_power_dbm": np.zeros(ntx, dtype=np.float32),
            "value_min": float(best_path_loss.min()),
            "value_max": float(best_path_loss.max()),
            "metadata": {
                "metrics_store": ["path_gain_linear"],
                "metrics_derived": ["path_loss_db", "serving_tx"],
                "noise_power_w": 1e-12,
            },
        },
        coverage_file,
        compression=None,
    )

    generated = create_coverage_metric_guide(
        coverage_file,
        tmp_path / "coverage_metrics.png",
        metrics=[
            "path_loss_db/WestSector",
            "path_loss_db/EastSector",
            "best_path_loss_db",
            "serving_path_gain_linear",
            "sinr_db/WestSector",
            "sinr_db/EastSector",
            "serving_tx",
        ],
        columns=2,
    )

    assert [path.name for path in generated] == ["coverage_metrics_height-01_1.5m.png"]
    assert generated[0].exists()


def test_selected_tx_sinr_ignores_materialized_best_server_layer():
    path_gain = np.asarray([[[[[1e-9]], [[4e-9]]]]], dtype=np.float32)
    coverage_layers = {
        "derived": {"sinr_db": np.full((1, 1, 1, 1), 999.0, dtype=np.float32)},
        "values": {"path_gain_linear": path_gain},
        "available_metrics": ["sinr_db", "sinr_db/East"],
        "tx_power_dbm": np.asarray([0.0, 3.0], dtype=np.float32),
        "tx_names": ["West", "East"],
        "noise_power_w": 1e-12,
    }

    layer = _resolve_metric_layer("sinr_db/East", coverage_layers)

    west_power = path_gain[0, 0, 0, 0, 0] * 1e-3
    east_power = path_gain[0, 0, 1, 0, 0] * 10.0 ** ((3.0 - 30.0) / 10.0)
    expected = 10.0 * np.log10(east_power / (west_power + 1e-12))
    assert layer["selector"] == "East"
    np.testing.assert_allclose(layer["data"], [[[expected]]], rtol=1e-6)


@pytest.mark.parametrize(
    "metric_name",
    ["path_gain_linear", "serving_path_gain_linear", "rss_w", "sinr_linear"],
)
def test_positive_linear_metric_guides_use_logarithmic_normalization(metric_name):
    style = _coverage_metric_style(metric_name)
    clim = _compute_clim_for_arrays(
        [np.asarray([[0.0, 1.0e-12, 1.0e-6, np.nan]])],
        positive_only=bool(style.get("logarithmic")),
    )

    norm = _coverage_scalar_norm(style, clim)

    assert isinstance(norm, LogNorm)
    assert norm.vmin == pytest.approx(1.0e-12)
    assert norm.vmax == pytest.approx(1.0e-6)


def _write_positive_linear_coverage_file(path: Path) -> None:
    path_gain = np.asarray(
        [
            [
                [[[0.0, 1.0e-12], [1.0e-9, 1.0e-6]]],
                [[[np.nan, 1.0e-11], [1.0e-8, 1.0e-7]]],
            ]
        ],
        dtype=np.float32,
    )
    save_coverage_hdf5(
        {
            "grid_origin": np.array([0.0, 0.0, 1.5], dtype=np.float32),
            "grid_spacing": np.array([5.0, 5.0], dtype=np.float32),
            "grid_shape": np.array([2, 2, 2], dtype=np.int32),
            "heights": np.array([1.5, 10.0], dtype=np.float32),
            "path_gain_linear": path_gain,
            "derived": {},
            "metric_name": "path_gain_linear/BaseStation",
            "tx_positions": np.array([[0.0, 0.0, 30.0]], dtype=np.float32),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": ["BaseStation"],
            "rx_names": [],
            "tx_power_dbm": np.array([0.0], dtype=np.float32),
            "value_min": 0.0,
            "value_max": 1.0e-6,
            "metadata": {"metrics_store": ["path_gain_linear"]},
        },
        path,
        compression=None,
    )


def test_metric_guides_share_one_color_scale_across_all_heights(
    tmp_path: Path,
    monkeypatch,
):
    from matplotlib.axes import Axes

    coverage_file = tmp_path / "coverage_maps.h5"
    _write_positive_linear_coverage_file(coverage_file)
    observed_norms = []
    original_imshow = Axes.imshow

    def observed_imshow(axis, values, *args, **kwargs):
        observed_norms.append(kwargs.get("norm"))
        return original_imshow(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", observed_imshow)

    generated = create_coverage_metric_guide(
        coverage_file,
        tmp_path / "coverage_metrics.png",
        metrics=["path_gain_linear/BaseStation"],
    )

    assert [path.name for path in generated] == [
        "coverage_metrics_height-01_1.5m.png",
        "coverage_metrics_height-02_10m.png",
    ]
    assert len(observed_norms) == 2
    assert all(isinstance(norm, LogNorm) for norm in observed_norms)
    assert all(norm.vmin == pytest.approx(1.0e-12) for norm in observed_norms)
    assert all(norm.vmax == pytest.approx(1.0e-6) for norm in observed_norms)


@pytest.mark.parametrize(
    "figure_kind",
    ["quick-look", "metric-guide", "comparison", "height-animation"],
)
def test_positive_linear_public_map_figures_use_logarithmic_normalization(
    tmp_path: Path,
    monkeypatch,
    figure_kind: str,
):
    from matplotlib.axes import Axes

    coverage_file = tmp_path / "coverage_maps.h5"
    _write_positive_linear_coverage_file(coverage_file)
    observed_norms = []
    observed_colormaps = []
    original_imshow = Axes.imshow

    def observed_imshow(axis, values, *args, **kwargs):
        observed_norms.append(kwargs.get("norm"))
        observed_colormaps.append(kwargs.get("cmap"))
        return original_imshow(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", observed_imshow)
    output_path = (
        tmp_path / f"{figure_kind}.{'gif' if figure_kind == 'height-animation' else 'png'}"
    )
    if figure_kind == "quick-look":
        create_coverage_visualization(coverage_file, output_path)
    elif figure_kind == "metric-guide":
        create_coverage_metric_guide(
            coverage_file,
            output_path,
            metrics=["path_gain_linear/BaseStation"],
        )
    elif figure_kind == "comparison":
        create_coverage_comparison_figure(coverage_file, output_path)
    else:
        import matplotlib.animation as mpl_animation

        class SynchronousAnimation:
            def __init__(self, _figure, update, *, frames, **_kwargs):
                for frame in range(frames):
                    update(frame)

            def save(self, path, **_kwargs):
                Path(path).touch()

        monkeypatch.setattr(mpl_animation, "FuncAnimation", SynchronousAnimation)
        create_coverage_height_evolution_animation(
            coverage_file,
            output_path,
            duration_ms=100,
        )

    assert observed_norms
    assert all(isinstance(norm, LogNorm) for norm in observed_norms)
    assert all(norm.vmin == pytest.approx(1.0e-12) for norm in observed_norms)
    assert all(norm.vmax == pytest.approx(1.0e-6) for norm in observed_norms)
    assert observed_colormaps == ["RdYlGn"] * len(observed_colormaps)


def test_serving_tx_labels_include_no_service_and_tx_names():
    assert serving_tx_class_labels(["BaseStation1", "BaseStation2"]) == [
        "No service",
        "BaseStation1",
        "BaseStation2",
    ]


def test_two_height_v2_comparison_distribution_and_statistics(tmp_path: Path):
    nx, ny, nz, ntx = 3, 2, 2, 2
    path_loss = np.stack(
        [
            np.linspace(90.0, 110.0, nx * ny).reshape(ny, nx),
            np.linspace(95.0, 115.0, nx * ny).reshape(ny, nx),
        ],
        axis=0,
    ).astype(np.float32)
    path_gain = np.stack([path_loss, path_loss + 3.0], axis=1)
    path_gain = np.power(10.0, -path_gain / 10.0).reshape(1, nz, ntx, ny, nx)
    serving_tx = np.zeros((nz, ny, nx), dtype=np.int16)
    serving_tx[0, 0, 0] = -1
    coverage_file = tmp_path / "coverage_maps.h5"
    save_coverage_hdf5(
        {
            "grid_origin": np.array([0.0, 0.0, 1.5], dtype=np.float32),
            "grid_spacing": np.array([5.0, 5.0], dtype=np.float32),
            "grid_shape": np.array([nx, ny, nz], dtype=np.int32),
            "heights": np.array([1.5, 10.0], dtype=np.float32),
            "path_gain_linear": path_gain.astype(np.float32),
            "derived": {"serving_tx": serving_tx.reshape(1, nz, ny, nx)},
            "metric_name": "best_path_loss_db",
            "tx_positions": np.array(
                [[-10.0, 0.0, 35.0], [10.0, 0.0, 35.0]],
                dtype=np.float32,
            ),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": ["BaseStation1", "BaseStation2"],
            "rx_names": [],
            "tx_power_dbm": np.zeros(ntx, dtype=np.float32),
            "value_min": float(path_loss.min()),
            "value_max": float(path_loss.max()),
            "metadata": {
                "metrics_store": ["path_gain_linear"],
                "metrics_derived": ["path_loss_db", "serving_tx"],
                "noise_power_w": 1e-12,
            },
        },
        coverage_file,
        compression=None,
    )

    comparison = create_coverage_comparison_figure(
        coverage_file,
        tmp_path / "coverage_comparison.png",
    )
    stats = create_coverage_statistics_plot(
        coverage_file,
        tmp_path / "coverage_statistics.png",
    )
    distributions = create_coverage_distribution_figure(
        coverage_file,
        tmp_path / "coverage_distributions.png",
        metrics=["serving_tx"],
    )

    assert comparison.exists()
    assert stats.exists()
    assert [path.name for path in distributions] == [
        "coverage_distributions_height-01_1.5m.png",
        "coverage_distributions_height-02_10m.png",
    ]
    assert all(path.exists() for path in distributions)


def test_coverage_extent_uses_v2_hdf5_spacing(tmp_path: Path):
    import h5py

    coverage_file = tmp_path / "coverage_maps.h5"
    with h5py.File(coverage_file, "w") as f:
        f.attrs["coverage_schema_version"] = COVERAGE_HDF5_SCHEMA_VERSION
        f.attrs["coverage_storage_layout"] = COVERAGE_HDF5_STORAGE_LAYOUT
        grid = f.create_group("grid")
        grid.create_dataset("spacing_xy", data=np.array([5.0, 2.0]))

    extent = _coverage_extent_from_grid(
        np.array([10.0, 20.0, 1.5]),
        np.array([4, 3, 1]),
        coverage_file,
    )

    assert extent == (10.0, 30.0, 20.0, 26.0)
