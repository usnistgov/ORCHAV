"""Plotting helpers for shared statistics outputs.

The functions here render derived statistics such as material counts and bounce
depth distributions. Callers provide already-computed data or frame-derived
arrays; this package does not read frame files directly.
"""

from .material_plots import (
    compute_material_counts_by_depth,
    plot_material_by_bounce_depth,
    plot_material_evolution_stacked,
)

__all__ = [
    "plot_material_evolution_stacked",
    "plot_material_by_bounce_depth",
    "compute_material_counts_by_depth",
]
