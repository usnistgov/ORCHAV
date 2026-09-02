"""Floor-plan rasterization helper facade for scene summary figures.

The concrete implementation sits in ``scene.drawing`` because rasterized 2D,
3D floor-plan, hybrid, and city modes all share the same cache and material
classification rules.
"""

from .drawing import (
    _cached_rasterized_floor_plan,
    _summary_cache_path,
    create_rasterized_floor_plan,
)

__all__ = [
    "_cached_rasterized_floor_plan",
    "_summary_cache_path",
    "create_rasterized_floor_plan",
]
