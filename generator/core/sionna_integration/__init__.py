"""Boundary helpers for Sionna, Mitsuba, Dr.Jit, and version differences.

Most generator code should use project-facing values: positions as tuple-like
objects, orientations in degrees, and solver settings in ORCHAV config terms.
This package is where those values are converted or checked against the
external Sionna/Mitsuba runtime.

``adapters.py`` contains assignment-boundary conversions such as
``point3f`` and orientation degrees to engine radians.  ``solver.py`` contains
PathSolver capability checks and Dr.Jit seeding helpers.  Imports are lazy so
packages that only need type names or config models do not initialize the Sionna
stack unnecessarily.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters import (
        EngineOrientation3,
        orientation_deg_to_engine_radians,
        orientation_to_point3f,
        orientation_to_point3f_with_engine_radians,
        point3f,
    )
    from .solver import (
        PATH_SOLVER_AUTODIFF_COMPATIBLE,
        PATH_SOLVER_SUPPORTS_DIFFRACTION,
        PATH_SOLVER_SUPPORTS_DIFFRACTION_LIT_REGION,
        PATH_SOLVER_SUPPORTS_EDGE_DIFFRACTION,
        SIONNA_VERSION,
        path_solver_supports_arg,
        set_drjit_seed,
        version_greater_equal,
    )

_LAZY_IMPORTS = {
    "EngineOrientation3": (".adapters", "EngineOrientation3"),
    "orientation_deg_to_engine_radians": (".adapters", "orientation_deg_to_engine_radians"),
    "orientation_to_point3f": (".adapters", "orientation_to_point3f"),
    "orientation_to_point3f_with_engine_radians": (
        ".adapters",
        "orientation_to_point3f_with_engine_radians",
    ),
    "point3f": (".adapters", "point3f"),
    "PATH_SOLVER_AUTODIFF_COMPATIBLE": (".solver", "PATH_SOLVER_AUTODIFF_COMPATIBLE"),
    "PATH_SOLVER_SUPPORTS_DIFFRACTION": (".solver", "PATH_SOLVER_SUPPORTS_DIFFRACTION"),
    "PATH_SOLVER_SUPPORTS_DIFFRACTION_LIT_REGION": (
        ".solver",
        "PATH_SOLVER_SUPPORTS_DIFFRACTION_LIT_REGION",
    ),
    "PATH_SOLVER_SUPPORTS_EDGE_DIFFRACTION": (
        ".solver",
        "PATH_SOLVER_SUPPORTS_EDGE_DIFFRACTION",
    ),
    "SIONNA_VERSION": (".solver", "SIONNA_VERSION"),
    "path_solver_supports_arg": (".solver", "path_solver_supports_arg"),
    "set_drjit_seed": (".solver", "set_drjit_seed"),
    "version_greater_equal": (".solver", "version_greater_equal"),
}

__all__ = [
    "EngineOrientation3",
    "PATH_SOLVER_AUTODIFF_COMPATIBLE",
    "PATH_SOLVER_SUPPORTS_DIFFRACTION",
    "PATH_SOLVER_SUPPORTS_DIFFRACTION_LIT_REGION",
    "PATH_SOLVER_SUPPORTS_EDGE_DIFFRACTION",
    "SIONNA_VERSION",
    "orientation_deg_to_engine_radians",
    "orientation_to_point3f",
    "orientation_to_point3f_with_engine_radians",
    "path_solver_supports_arg",
    "point3f",
    "set_drjit_seed",
    "version_greater_equal",
]


def __getattr__(name: str) -> Any:
    """Expose integration helpers lazily to keep heavy imports out of package import."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    attr = getattr(import_module(module_name, package=__name__), attr_name)
    globals()[name] = attr
    return attr
