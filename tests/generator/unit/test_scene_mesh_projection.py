"""Native mesh backdrops retain topology without the pixel floor-plan pass."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection

from generator.figures.coverage.figures import _add_scene_overlay_to_axis
from generator.figures.scene import drawing


@pytest.fixture
def scene(monkeypatch):
    drawing.clear_scene_figure_caches()

    def reject_raster(*_args, **_kwargs):
        pytest.fail("Native scene drawing invoked the old pixel rasterizer")

    monkeypatch.setattr(drawing, "create_rasterized_floor_plan", reject_raster)
    # Two disconnected quads and a lower ground plane. The central gap must
    # remain open, and shared coplanar triangle diagonals must not become edges.
    geometry = []
    for xmin, xmax, z in ((0, 1, 2), (2, 3, 3), (0, 3, 0)):
        geometry.append(
            {
                "name": "concrete",
                "mesh": SimpleNamespace(
                    vertices=np.array(
                        [[xmin, 0, z], [xmax, 0, z], [xmax, 1, z], [xmin, 1, z]], dtype=float
                    ),
                    triangles=np.array([[0, 1, 2], [0, 2, 3]]),
                ),
            }
        )
    yield geometry
    plt.close("all")
    drawing.clear_scene_figure_caches()


@pytest.mark.parametrize("mode", ["mesh", "auto"])
def test_top_view_draws_exact_faces_and_creases_without_filling_the_gap(scene, mode):
    fig, ax = plt.subplots()
    drawing.plot_scene_geometry_2d(ax, scene, rendering_mode=mode)
    polygons = next(c for c in ax.collections if isinstance(c, PolyCollection))
    outlines = next(c for c in ax.collections if isinstance(c, LineCollection))
    assert len(polygons.get_paths()) == 6
    assert len(outlines.get_segments()) == 12
    # Height ordering draws the ground first, then the two independent roofs.
    for path in polygons.get_paths()[2:]:
        x = path.vertices[:, 0]
        assert x.max() <= 1 or x.min() >= 2
    assert not ax.images
    fig.canvas.draw()


@pytest.mark.parametrize("mode", ["mesh", "auto", "hybrid", "city"])
def test_three_dimensional_backdrops_avoid_the_pixel_rasterizer(scene, mode):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    drawing.plot_scene_geometry_3d(ax, scene, rendering_mode=mode)
    assert ax.collections
    fig.canvas.draw()


def test_prepared_geometry_is_reused_across_panels_and_cleared(scene):
    first = drawing._prepared_mesh_scene(scene)
    second = drawing._prepared_mesh_scene(scene)
    assert first is second
    assert drawing.get_scene_figure_cache_stats()["vector_entries"] == 1
    other = list(scene[:1])
    assert len(drawing._prepared_mesh_scene(other).faces) == 2
    drawing.clear_scene_figure_caches()
    assert drawing.get_scene_figure_cache_stats()["current_bytes"] == 0


@pytest.mark.parametrize("style", ["fill", "outline"])
def test_coverage_overlay_preserves_values_bounds_and_color_limits(scene, monkeypatch, style):
    monkeypatch.setattr("shared.geometry.cache.get_scene_geometry", lambda **_kwargs: scene)
    fig, ax = plt.subplots()
    values = np.array([[10.0, 20.0], [30.0, np.nan]])
    image = ax.imshow(values, extent=(0.5, 2.5, 0.25, 0.75), vmin=0, vmax=50, origin="lower")
    bounds = ax.get_xlim(), ax.get_ylim()
    _add_scene_overlay_to_axis(ax, image, object(), 0.35, style)
    assert (ax.get_xlim(), ax.get_ylim()) == bounds
    assert image.get_clim() == (0, 50)
    np.testing.assert_array_equal(np.asarray(image.get_array()), values)
    assert len(ax.images) == 1
    outlines = [c for c in ax.collections if isinstance(c, LineCollection)]
    polygons = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(outlines) == 1
    assert len(polygons) == (1 if style == "fill" else 0)
    assert outlines[0].get_alpha() == 0.35
    assert (outlines[0].get_zorder() > image.get_zorder()) == (style == "outline")
    fig.canvas.draw()
