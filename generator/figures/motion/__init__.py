"""Motion and orientation series helpers for static generator figures.

This subpackage is deliberately data-only: it samples ``ActorStateManager``
cached state and converts Mitsuba/vector-like points into numpy series consumed
by the plotting functions in ``generator_summary_fig``.
"""

from .series import (
    collect_orientation_data_from_actor_state_manager,
    collect_velocity_data_from_actor_state_manager,
    orientations_to_angular_velocities,
    point_to_xy,
    point_to_xyz,
    positions_to_tuples,
    positions_to_velocities,
    prepare_actor_state_data,
)

__all__ = [
    "collect_orientation_data_from_actor_state_manager",
    "collect_velocity_data_from_actor_state_manager",
    "orientations_to_angular_velocities",
    "point_to_xy",
    "point_to_xyz",
    "positions_to_tuples",
    "positions_to_velocities",
    "prepare_actor_state_data",
]
