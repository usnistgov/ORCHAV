"""Trajectory objects for the role-specific Python generator API.

These convenience classes back callers that construct transmitter, receiver,
or target runtime configs directly. Schema actor scenarios are prepared through
``generator.core.scenario_actors`` instead. The common ``MobilityPattern``
contract stores control values first, then ``prepare()`` materializes one plain
``(x, y, z)`` position per scene step.

Start with ``base.py`` for that lifecycle, ``basic.py`` for stationary/linear/
circular/waypoint primitives, ``grid.py`` and ``random_area.py`` for sampled
spatial sweeps, and the specialized modules for drone, formation, and
academic mobility models.
"""

from __future__ import annotations

from .base import MobilityPattern
from .basic import CircularMobility, LinearMobility, StationaryMobility, WaypointMobility
from .drone import DroneMobility, Figure8Mobility, ReturnToHomeMobility, SpiralMobility
from .environment import IndoorMeshGridMobility, OutdoorMeshGridMobility
from .formation import FormationMobility
from .grid import MeshGridMobility
from .random_area import RandomBoxMobility

__all__ = [
    "MobilityPattern",
    "LinearMobility",
    "CircularMobility",
    "WaypointMobility",
    "StationaryMobility",
    "MeshGridMobility",
    "OutdoorMeshGridMobility",
    "IndoorMeshGridMobility",
    "RandomBoxMobility",
    "DroneMobility",
    "Figure8Mobility",
    "SpiralMobility",
    "ReturnToHomeMobility",
    "FormationMobility",
]
