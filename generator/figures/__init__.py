"""Static figure generation used by generator pipeline outputs.

This package is not the interactive visualizer. It owns file-backed Matplotlib
artifacts created after generation: scenario summary figures, coverage-map
figures, scene overlays, and motion/orientation plots. The stable entry points
are exported lazily so importing ``generator.figures`` does not immediately pull
in Matplotlib, h5py, or scene-geometry helpers.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coverage import (
        create_coverage_distribution_figure,
        create_coverage_metric_guide,
        create_coverage_visualization,
    )
    from .scene import (
        create_rasterized_floor_plan,
        plot_scene_geometry_2d,
        plot_scene_geometry_3d,
    )
    from .summary import maybe_generate_generator_summary

__all__ = [
    "create_coverage_distribution_figure",
    "create_coverage_metric_guide",
    "create_coverage_visualization",
    "create_rasterized_floor_plan",
    "maybe_generate_generator_summary",
    "plot_scene_geometry_2d",
    "plot_scene_geometry_3d",
]


def __getattr__(name: str):
    """Load optional plotting subpackages only when their exported symbols are used."""
    if name in {
        "create_coverage_distribution_figure",
        "create_coverage_metric_guide",
        "create_coverage_visualization",
    }:
        from . import coverage

        return getattr(coverage, name)
    if name in {
        "create_rasterized_floor_plan",
        "plot_scene_geometry_2d",
        "plot_scene_geometry_3d",
    }:
        from . import scene

        return getattr(scene, name)
    if name == "maybe_generate_generator_summary":
        from .summary import maybe_generate_generator_summary

        return maybe_generate_generator_summary
    raise AttributeError(name)
