"""Standalone 3D coverage slices from saved RF data and native scene meshes."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .figures import (
    _coverage_scalar_plot_policy,
    _load_v2_coverage_layers,
    _masked_coverage_scalar_values,
    _masked_serving_tx,
    _resolve_metric_layer,
    _serving_tx_cmap_norm,
    _serving_tx_ticks,
    serving_tx_class_labels,
)


def _coverage_cell_triangles(values, origin, spacing, height):
    """Keep each valid cell's exact XY footprint and constant saved value."""
    y, x = np.nonzero(~np.ma.getmaskarray(values))
    xy = origin[:2] + np.column_stack((x, y)) * spacing[:2]
    corners = np.zeros((len(x), 4, 3), dtype=float)
    corners[:, :, :2] = xy[:, None, :] + np.array(
        [[0, 0], [spacing[0], 0], [spacing[0], spacing[1]], [0, spacing[1]]]
    )
    corners[:, :, 2] = height
    faces = corners[:, [[0, 1, 2], [0, 2, 3]], :].reshape(-1, 3, 3)
    return faces, np.repeat(np.asarray(values)[y, x], 2)


def create_coverage_3d_figure(
    coverage_file: Path,
    output_path: Path,
    *,
    scenario_context=None,
    metric: str = "best_path_loss_db",
    height_index: int = 0,
    elevation: float = 40.0,
    azimuth: float = -60.0,
) -> Path:
    """Export one orthographic 3D slice without a window or radio-map solve.

    Color limits cover every saved height, matching the 2D metric figures.
    The scene and RF cells share a polygon collection for face depth sorting.
    Missing scalar values leave holes; categorical no-service cells retain
    their documented color. Coordinates and heights are used without scaling.
    """
    layers = _load_v2_coverage_layers(Path(coverage_file))
    resolved = _resolve_metric_layer(metric, layers)
    values = resolved["data"]
    if not 0 <= height_index < len(values):
        raise ValueError(f"height_index must be between 0 and {len(values) - 1}")
    if not np.isfinite([elevation, azimuth]).all():
        raise ValueError("Camera elevation and azimuth must be finite")
    height = float(layers["heights"][height_index])
    categorical = resolved["base_name"] == "serving_tx"
    if categorical:
        cmap, norm = _serving_tx_cmap_norm(len(layers["tx_names"]))
        selected = _masked_serving_tx(values[height_index], len(layers["tx_names"]))
    else:
        style, limits, norm = _coverage_scalar_plot_policy(metric, values)
        cmap = style["cmap"]
        norm = norm or Normalize(*(limits if limits is not None else (0, 1)))
        selected = _masked_coverage_scalar_values(values[height_index], style)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    faces, samples = _coverage_cell_triangles(
        selected, layers["grid_origin"], layers["grid_spacing"], height
    )
    colors = mapper.to_rgba(samples)
    bounds = []
    ground_faces, ground_colors = np.empty((0, 3, 3)), np.empty((0, 3))
    if scenario_context is not None:
        from shared.geometry.cache import get_scene_geometry

        from ..scene.drawing import _prepared_mesh_scene

        geometry = get_scene_geometry(scenario_context=scenario_context)
        if not geometry:
            raise ValueError("No scene geometry is available for the 3D coverage figure")
        scene = _prepared_mesh_scene(geometry)
        # Large triangles at the scene's lowest plane must stay behind the
        # above-ground slice. Centroid sorting alone can paint them over it.
        ground = np.zeros(len(scene.faces), dtype=bool)
        if len(scene.bounds) and scene.bounds[0, 2] <= height:
            ground = np.isclose(scene.faces[:, :, 2], scene.bounds[0, 2], atol=1e-8, rtol=0).all(
                axis=1
            )
        ground_faces, ground_colors = scene.faces[ground], scene.colors[ground]
        faces = np.concatenate((scene.faces[~ground], faces))
        scene_colors = scene.colors[~ground]
        colors = np.concatenate(
            (np.column_stack((scene_colors, np.ones(len(scene_colors)))), colors)
        )
        bounds.extend(scene.bounds)

    nx, ny, _ = layers["grid_shape"]
    x0, y0 = layers["grid_origin"][:2]
    dx, dy = layers["grid_spacing"][:2]
    bounds.extend(((x0, y0, height), (x0 + nx * dx, y0 + ny * dy, height)))
    positions = layers["tx_positions"]
    positions = positions.reshape(-1, 3)
    finite_positions = positions[np.isfinite(positions).all(axis=1)]
    bounds.extend(finite_positions)
    lo, hi = np.min(bounds, axis=0), np.max(bounds, axis=0)
    span = hi - lo
    # A single flat slice still needs a nonzero camera box; never move the data.
    span = np.maximum(span, max(float(span.max()) * 0.02, 1.0))
    center = (lo + hi) / 2
    lo, hi = center - span * 0.52, center + span * 0.52

    fig = Figure(figsize=(12, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection="3d", proj_type="ortho", computed_zorder=False)
    if len(ground_faces):
        ax.add_collection3d(
            Poly3DCollection(
                ground_faces,
                facecolors=ground_colors,
                edgecolors="none",
                antialiased=False,
                zorder=0,
            )
        )
    if len(faces):
        ax.add_collection3d(
            Poly3DCollection(
                faces, facecolors=colors, edgecolors="none", antialiased=False, zorder=2
            )
        )
    if not len(samples):
        ax.text2D(
            0.5,
            0.9,
            "No coverage data available at this height",
            transform=ax.transAxes,
            ha="center",
        )
    for name, position in zip(layers["tx_names"], positions):
        if np.isfinite(position).all():
            ax.scatter(
                *position,
                marker="^",
                s=75,
                color="#007aff",
                edgecolors="white",
                depthshade=False,
                zorder=4,
            )
            ax.text(*position, f"  {name}", fontsize=9, zorder=5)
    ax.set(
        xlim=(lo[0], hi[0]),
        ylim=(lo[1], hi[1]),
        zlim=(lo[2], hi[2]),
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
    )
    ax.set_box_aspect(span)
    ax.set_zticks(np.unique([np.min(bounds, axis=0)[2], np.max(bounds, axis=0)[2]]))
    ax.grid(False)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_title(f"{resolved['title']} — coverage at z = {height:g} m", pad=20)
    colorbar = fig.colorbar(mapper, ax=ax, shrink=0.65, pad=0.10)
    if categorical:
        colorbar.set_ticks(_serving_tx_ticks(len(layers["tx_names"])))
        colorbar.set_ticklabels(serving_tx_class_labels(layers["tx_names"]))
    colorbar.set_label(resolved["label"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    return output_path
