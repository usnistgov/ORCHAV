"""Run-scoped generator service layer.

The pipeline uses these classes as the generator core boundary.  For the normal
pipeline path, read them in this order: ``SceneService`` builds live Sionna
objects, ``ActorStateService`` prepares generator-side actor state, and
``RayTracingService`` computes frames from both.  Each service owns one
short-lived part of a scenario run:

* ``SceneService`` loads the Sionna scene and creates live TX/RX/target objects.
* ``ActorStateService`` prepares internal actor-state caches; it does not mutate
  the Sionna scene by itself.
* ``RayTracingService`` binds scene objects, prepared actor state, and solver
  settings into the propagation runtime.
* ``CoverageService`` normalizes coverage YAML and delegates coverage outputs.

Exports are lazy so importing this package does not initialize Sionna or
Mitsuba until a concrete service is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .actor_state_service import ActorStateService
    from .coverage_service import CoverageService
    from .raytracing_service import RayTracingService
    from .scene_service import SceneService

__all__ = [
    "CoverageService",
    "ActorStateService",
    "RayTracingService",
    "SceneService",
]


def __getattr__(name: str) -> Any:
    if name == "CoverageService":
        from .coverage_service import CoverageService

        return CoverageService
    if name == "RayTracingService":
        from .raytracing_service import RayTracingService

        return RayTracingService
    if name == "SceneService":
        from .scene_service import SceneService

        return SceneService
    if name == "ActorStateService":
        from .actor_state_service import ActorStateService

        return ActorStateService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
