"""
Material visualization utilities.

Provides reusable plotting functions for material hit analysis,
including stacked histograms and evolution plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.frames.sionna_metadata import SIONNA_INVALID_OBJECT_ID
from shared.logging import get_logger

logger = get_logger(__name__)


def plot_material_evolution_stacked(
    material_counts_per_step: List[Dict[str, int]],
    step_labels: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Material Hit Evolution",
    xlabel: str = "Step",
    ylabel: str = "Hit Count",
    top_k: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    colormap: str = "tab20",
    show_legend: bool = True,
    return_figure: bool = False,
) -> Optional[Any]:
    """
    Create a stacked bar chart showing material distribution evolution across steps.

    This is useful for visualizing how material hits change across:
    - Different RX positions (position optimization)
    - Different frames in a trajectory
    - Different simulation configurations

    Args:
        material_counts_per_step: List of dicts, each mapping material_name -> hit_count
        step_labels: Labels for x-axis (defaults to step indices)
        output_path: Where to save figure (optional, None = don't save)
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        top_k: Only show top K materials by total count (others grouped as "Other")
        figsize: Figure size in inches
        colormap: Matplotlib colormap name
        show_legend: Whether to show legend
        return_figure: If True, return (fig, ax) instead of closing

    Returns:
        (fig, ax) if return_figure=True, else None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return None

    if not material_counts_per_step:
        logger.warning("No material data to plot")
        return None

    # Aggregate totals to find top-k materials
    total_counts: Dict[str, int] = {}
    for step_counts in material_counts_per_step:
        for mat, count in step_counts.items():
            total_counts[mat] = total_counts.get(mat, 0) + count

    # Sort by total count and get top-k
    sorted_materials = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    top_materials = [m[0] for m in sorted_materials[:top_k]]

    # Build data matrix: rows = materials, cols = steps
    n_steps = len(material_counts_per_step)
    data_matrix = np.zeros((len(top_materials) + 1, n_steps))  # +1 for "Other"

    for step_idx, step_counts in enumerate(material_counts_per_step):
        other_count = 0
        for mat, count in step_counts.items():
            if mat in top_materials:
                mat_idx = top_materials.index(mat)
                data_matrix[mat_idx, step_idx] = count
            else:
                other_count += count
        data_matrix[-1, step_idx] = other_count  # "Other" row

    # Remove "Other" row if all zeros
    if np.sum(data_matrix[-1, :]) == 0:
        data_matrix = data_matrix[:-1, :]
        labels = top_materials
    else:
        labels = top_materials + ["Other"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # X positions
    x = np.arange(n_steps)
    if step_labels is None:
        step_labels = [str(i) for i in range(n_steps)]

    # Get colors
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    # Plot stacked bars
    bottom = np.zeros(n_steps)
    bars = []
    for i, (mat_data, label, color) in enumerate(zip(data_matrix, labels, colors)):
        bar = ax.bar(
            x, mat_data, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5
        )
        bars.append(bar)
        bottom += mat_data

    # Styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=45, ha="right")

    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved material evolution plot to %s", output_path)

    if return_figure:
        return fig, ax
    else:
        plt.close()
        return None


def plot_material_by_bounce_depth(
    material_counts_by_depth: Dict[int, Dict[str, int]],
    output_path: Optional[Path] = None,
    title: str = "Material Hits by Bounce Depth",
    top_k: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    colormap: str = "tab20",
    show_percentages: bool = True,
    return_figure: bool = False,
) -> Optional[Any]:
    """
    Create a stacked bar chart showing which materials are hit at each bounce depth.

    This shows the "path through materials" - at depth 1 (first bounce),
    which materials are hit most? At depth 2, 3, etc.

    Args:
        material_counts_by_depth: Dict mapping depth (1,2,3...) to {material: count}
        output_path: Where to save figure (optional)
        title: Figure title
        top_k: Only show top K materials
        figsize: Figure size
        colormap: Matplotlib colormap
        show_percentages: Show percentage labels on bars
        return_figure: If True, return (fig, ax)

    Returns:
        (fig, ax) if return_figure=True, else None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return None

    if not material_counts_by_depth:
        logger.warning("No material depth data to plot")
        return None

    # Get sorted depths
    depths = sorted(material_counts_by_depth.keys())

    # Aggregate totals to find top-k materials
    total_counts: Dict[str, int] = {}
    for depth_counts in material_counts_by_depth.values():
        for mat, count in depth_counts.items():
            total_counts[mat] = total_counts.get(mat, 0) + count

    sorted_materials = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    top_materials = [m[0] for m in sorted_materials[:top_k]]

    # Build data matrix: rows = materials, cols = depths
    n_depths = len(depths)
    data_matrix = np.zeros((len(top_materials) + 1, n_depths))

    for depth_idx, depth in enumerate(depths):
        depth_counts = material_counts_by_depth.get(depth, {})
        other_count = 0
        for mat, count in depth_counts.items():
            if mat in top_materials:
                mat_idx = top_materials.index(mat)
                data_matrix[mat_idx, depth_idx] = count
            else:
                other_count += count
        data_matrix[-1, depth_idx] = other_count

    # Remove "Other" if empty
    if np.sum(data_matrix[-1, :]) == 0:
        data_matrix = data_matrix[:-1, :]
        labels = top_materials
    else:
        labels = top_materials + ["Other"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_depths)
    depth_labels = [f"Depth {d}" for d in depths]

    # Colors
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    # Plot stacked bars
    bottom = np.zeros(n_depths)
    for mat_data, label, color in zip(data_matrix, labels, colors):
        ax.bar(
            x, mat_data, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5
        )
        bottom += mat_data

    # Add percentage labels if requested
    if show_percentages:
        totals = np.sum(data_matrix, axis=0)
        for depth_idx in range(n_depths):
            if totals[depth_idx] > 0:
                cumulative = 0
                for mat_idx, mat_data in enumerate(data_matrix):
                    count = mat_data[depth_idx]
                    if count > 0:
                        pct = 100 * count / totals[depth_idx]
                        if pct >= 5:  # Only show if >= 5%
                            y_pos = cumulative + count / 2
                            ax.text(
                                depth_idx,
                                y_pos,
                                f"{pct:.0f}%",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="white",
                                fontweight="bold",
                            )
                    cumulative += count

    # Styling
    ax.set_xlabel("Bounce Depth")
    ax.set_ylabel("Hit Count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(depth_labels)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved material by depth plot to %s", output_path)

    if return_figure:
        return fig, ax
    else:
        plt.close()
        return None


def compute_material_counts_by_depth(
    obj_ids: np.ndarray,
    interactions: np.ndarray,
    valid: np.ndarray,
    id_to_material: Dict[int, str],
) -> Dict[int, Dict[str, int]]:
    """
    Compute material hit counts organized by bounce depth.

    This is a utility function to extract material-by-depth data from
    Sionna RT path tracing results.

    Args:
        obj_ids: Object IDs array from paths.objects (max_depth, num_rx, num_tx, num_paths)
        interactions: Interaction types array from paths.interactions
        valid: Valid path mask from paths.valid (num_rx, num_tx, num_paths)
        id_to_material: Mapping from object ID to material name

    Returns:
        Dict mapping depth (1-indexed) to {material_name: hit_count}
    """
    max_depth = obj_ids.shape[0]
    num_rx = obj_ids.shape[1] if len(obj_ids.shape) > 1 else 1
    num_tx = obj_ids.shape[2] if len(obj_ids.shape) > 2 else 1
    num_paths = obj_ids.shape[3] if len(obj_ids.shape) > 3 else obj_ids.shape[-1]

    counts_by_depth: Dict[int, Dict[str, int]] = {}

    for depth_idx in range(max_depth):
        depth = depth_idx + 1  # 1-indexed
        depth_counts: Dict[str, int] = {}

        for rx_idx in range(num_rx):
            for tx_idx in range(num_tx):
                for path_idx in range(num_paths):
                    if not valid[rx_idx, tx_idx, path_idx]:
                        continue

                    interaction = interactions[depth_idx, rx_idx, tx_idx, path_idx]
                    obj_id = obj_ids[depth_idx, rx_idx, tx_idx, path_idx]

                    if interaction > 0 and obj_id != SIONNA_INVALID_OBJECT_ID:
                        mat_name = id_to_material.get(obj_id, f"unknown_{obj_id}")
                        depth_counts[mat_name] = depth_counts.get(mat_name, 0) + 1

        if depth_counts:
            counts_by_depth[depth] = depth_counts

    return counts_by_depth
