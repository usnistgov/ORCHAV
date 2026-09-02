"""Focused facade for coverage HDF5 loading and metric resolution.

The helpers share schema-v2 coverage semantics with the plotting code in
``coverage.figures``. This module provides a reader-oriented namespace for
those entry points.
"""

from .figures import (
    _coverage_extent_from_grid,
    _coverage_grid_extent,
    _load_coverage,
    _load_v2_coverage_layers,
    _load_v2_tx_metadata,
    _resolve_metric_layer,
    _resolve_tx_index,
    normalize_coverage_height_stack,
)

__all__ = [
    "_coverage_extent_from_grid",
    "_coverage_grid_extent",
    "_load_coverage",
    "_load_v2_coverage_layers",
    "_load_v2_tx_metadata",
    "_resolve_metric_layer",
    "_resolve_tx_index",
    "normalize_coverage_height_stack",
]
