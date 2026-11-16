"""Angular summary views respect physical bounds without changing exact bins."""

import numpy as np
import pytest

from shared.sensing.plotting import viz as sensing_viz
from tests.generator.unit.test_sensing_summary_fig import _typed_angular_payload


@pytest.mark.parametrize("x_is_angle,y_is_angle", [(False, True), (True, True), (False, False)])
def test_physical_view_bounds_preserve_extrapolated_mesh_edges(x_is_angle, y_is_angle):
    x = np.array([-90.0, -45.0, 0.0, 45.0])
    y = np.array([-90.0, 0.0, 90.0])
    power = np.arange(1.0, 13.0).reshape(3, 4)
    original = power.copy()
    fig, ax = sensing_viz.plt.subplots()
    try:
        artist = sensing_viz._plot_angular_power_mesh(
            ax,
            power,
            x,
            y,
            x_name="x",
            y_name="y",
            cmap="viridis",
            x_is_angle=x_is_angle,
            y_is_angle=y_is_angle,
        )
        coordinates = artist.get_coordinates()
        np.testing.assert_array_equal(coordinates[0, :, 0], [-112.5, -67.5, -22.5, 22.5, 67.5])
        np.testing.assert_array_equal(coordinates[:, 0, 1], [-135.0, -45.0, 45.0, 135.0])
        np.testing.assert_allclose(artist.get_array(), 10 * np.log10(power))
        np.testing.assert_array_equal(power, original)
        assert ax.get_xlim() == ((-90, 90) if x_is_angle else (-112.5, 67.5))
        assert ax.get_ylim() == ((-90, 90) if y_is_angle else (-135, 135))
    finally:
        sensing_viz.plt.close(fig)


@pytest.mark.parametrize(
    "function,truth_argument,angle_axes",
    [
        ("create_range_azimuth_figure", True, [(0, "y")]),
        ("create_doppler_azimuth_figure", True, [(0, "y")]),
        ("create_azimuth_elevation_figure", False, [(0, "x"), (0, "y")]),
        ("create_range_elevation_figure", False, [(0, "y")]),
        ("create_angular_overview_figure", True, [(1, "y"), (2, "y")]),
    ],
)
def test_all_standard_angular_views_bound_degree_axes(
    tmp_path, monkeypatch, function, truth_argument, angle_axes
):
    sensing = _typed_angular_payload()
    sensing["azimuth_broadside_angle_deg"] = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    sensing["elevation_broadside_angle_deg"] = np.array([-90.0, 90.0])
    figures = []
    subplots = sensing_viz.plt.subplots

    def capture(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(sensing_viz.plt, "subplots", capture)
    args = [sensing, None] if truth_argument else [sensing]
    getattr(sensing_viz, function)(*args, tmp_path / "angular.png")
    for index, dimension in angle_axes:
        assert getattr(figures[0].axes[index], f"get_{dimension}lim")() == (-90, 90)
    if function == "create_angular_overview_figure":
        assert figures[0].axes[0].get_ylim() != (-90, 90)
