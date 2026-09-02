"""Matplotlib scene-geometry drawing helpers for generator summary figures.

Scene helpers render cached ORCHAV scene geometry as 2D floor plans, vector
overlays, 3D backdrops, or city-scale blocks. They are shared by generator
summary figures, coverage overlays, and diagnostic comparison plots.
"""

from .drawing import (
    create_rasterized_floor_plan,
    plot_scene_geometry_2d,
    plot_scene_geometry_3d,
)

__all__ = [
    "create_rasterized_floor_plan",
    "plot_scene_geometry_2d",
    "plot_scene_geometry_3d",
]
