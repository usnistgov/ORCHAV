"""Prepared runtime objects used by propagation and streaming.

The runtime package is deliberately small: ``SimulationObjects`` is the
propagation-facing bundle of live scene objects, prepared actor state, solver
settings, and scenario context.  ``build_on_demand_objects`` creates that bundle
for callers that need an on-demand streaming context outside the default file
pipeline.

This package should not parse YAML or decide output routing.  Configuration is
resolved before runtime objects are built, and pipeline code decides whether the
objects are used for offline output or live requests.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .on_demand import build_on_demand_objects
    from .simulation_objects import SimulationObjects

_LAZY_IMPORTS = {
    "SimulationObjects": (".simulation_objects", "SimulationObjects"),
    "build_on_demand_objects": (".on_demand", "build_on_demand_objects"),
}

__all__ = ["SimulationObjects", "build_on_demand_objects"]


def __getattr__(name: str) -> Any:
    """Import runtime helpers lazily to avoid initializing Sionna on package import."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    attr = getattr(import_module(module_name, package=__name__), attr_name)
    globals()[name] = attr
    return attr
