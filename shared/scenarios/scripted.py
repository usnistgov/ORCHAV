"""Load the explicit actor factory shared by a scenario driver and local preview.

An optional ``actors.py`` beside ``scenario.yaml`` supplies
``build_actors(scenario) -> ActorsSpec``. Keep this module limited to actor
construction: generation remains the responsibility of ``generate.py``.
"""

from __future__ import annotations

import runpy

from .actors import ActorsSpec
from .loader import ScenarioConfiguration

ACTOR_FACTORY_FILENAME = "actors.py"


def load_scripted_actors(scenario: ScenarioConfiguration) -> ActorsSpec:
    """Build immutable actors from the scenario's explicit Python factory.

    The factory receives the validated configuration, including its resolved
    root and timeline. It must not generate frames or mutate that configuration.
    Loading by path keeps separate scenarios' factories independent.
    """
    path = scenario.root / ACTOR_FACTORY_FILENAME
    namespace = runpy.run_path(str(path), run_name="_orchav_actor_factory")
    factory = namespace.get("build_actors")
    if not callable(factory):
        raise ValueError(f"{path} must define build_actors(scenario)")
    actors = factory(scenario)
    if not isinstance(actors, ActorsSpec):
        raise TypeError(f"{path}: build_actors(scenario) must return ActorsSpec")
    return actors
