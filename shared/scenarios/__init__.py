"""Public scenario models, loading, validation, and path policy.

This package is the shared schema authority for scenario files. It turns
``scenario.yaml`` into immutable actor specifications or a path-resolved
``ScenarioConfiguration`` that applications can adapt into runtime objects.
"""

from .actors import *  # noqa: F403 - re-export the documented scenario spec surface
from .actors import __all__ as _actor_exports
from .loader import ScenarioConfiguration, load_scenario, load_scenario_configuration
from .model import ScenarioModel
from .yaml import load_scenario_yaml, suggest_typo_fix, validate_scenario_data

__all__ = [
    *_actor_exports,
    "ScenarioModel",
    "ScenarioConfiguration",
    "load_scenario",
    "load_scenario_configuration",
    "load_scenario_yaml",
    "suggest_typo_fix",
    "validate_scenario_data",
]
