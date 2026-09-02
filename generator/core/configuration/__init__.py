"""Boundary between shared scenario models and generator runtime services.

Start here when a reader wants to understand the normalized configuration used
by the core generator.  ``loader.py`` adapts YAML/scenario objects into a
``SimulationConfig``.  ``models.py`` defines the normalized run-local
dataclasses consumed by services, pipeline, coverage, and propagation.
``presets.py`` keeps static solver quality and scene identifiers out of runtime
logic.

This package owns configuration data and validation.  It does not build Sionna
objects, prepare actor state, compute frames, or write outputs. Branch-heavy
optional-field handling is expected here because several configuration sections
may be omitted.
"""

from .loader import load_simulation_config
from .models import (
    AntennaConfig,
    CoverageConfig,
    ReceiverConfig,
    SensingConfig,
    SimulationConfig,
    TransmitterConfig,
    build_simulation_config,
)
from .presets import AVAILABLE_SCENES, QUALITY_PRESETS

__all__ = [
    "AntennaConfig",
    "AVAILABLE_SCENES",
    "CoverageConfig",
    "QUALITY_PRESETS",
    "ReceiverConfig",
    "SensingConfig",
    "SimulationConfig",
    "TransmitterConfig",
    "build_simulation_config",
    "load_simulation_config",
]
