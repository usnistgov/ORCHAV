"""Coverage-map readers and static figure writers for generator outputs.

The functions here consume schema-v2 coverage HDF5 files from
``generator.io.storage.coverage_writer``. They resolve compact stored coverage
data into plotting layers, then write quick-look maps, metric guides,
histogram/CDF summaries, height comparisons, animations, and statistics plots.
"""

from .figures import (
    _coverage_extent_from_grid,
    _load_coverage,
    create_coverage_comparison_figure,
    create_coverage_distribution_figure,
    create_coverage_height_evolution_animation,
    create_coverage_metric_guide,
    create_coverage_statistics_plot,
    create_coverage_visualization,
    normalize_coverage_height_stack,
    serving_tx_class_labels,
)

__all__ = [
    "_coverage_extent_from_grid",
    "_load_coverage",
    "create_coverage_comparison_figure",
    "create_coverage_distribution_figure",
    "create_coverage_height_evolution_animation",
    "create_coverage_metric_guide",
    "create_coverage_statistics_plot",
    "create_coverage_visualization",
    "normalize_coverage_height_stack",
    "serving_tx_class_labels",
]
