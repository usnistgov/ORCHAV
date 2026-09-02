"""Coverage map computation package.

This package is the generator-side implementation of scenario ``coverage``
blocks. ``CoverageService`` normalizes YAML/scenario settings into a
``CoverageConfig`` and then calls :func:`compute_coverage_map`, the package-level
entry point. The result is a schema-v2 coverage payload for the HDF5 writer.

Start with ``solver.py`` to follow the execution path: it resolves Sionna RT
``RadioMapSolver`` arguments, solves one radio-map plane per requested height,
derives display metrics, and assembles the output payload. The support modules
keep policy pieces separate: ``bounds.py`` resolves grid extents, ``quality.py``
maps ORCHAV quality presets and overrides to solver arguments, and ``metrics.py``
holds RF unit conversions used by derived coverage layers.
"""

from .bounds import CoverageBoundsError
from .solver import compute_coverage_map

__all__ = [
    "CoverageBoundsError",
    "compute_coverage_map",
]
