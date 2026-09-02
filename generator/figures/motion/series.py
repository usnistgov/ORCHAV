"""Actor-state data helpers for generator summary figures.

These helpers are intentionally independent of Matplotlib. They normalize
Mitsuba/vector-like points, cached actor states, and numpy arrays into simple
series consumed by the speed, orientation, and angular-velocity figure writers.
"""

from typing import Any

import numpy as np

from generator.core.utils import to_float


def point_to_xy(point: Any) -> tuple[float, float]:
    """Return an ``(x, y)`` tuple from a point-like object or sequence."""
    if hasattr(point, "x") and hasattr(point, "y"):
        return to_float(point.x), to_float(point.y)
    return to_float(point[0]), to_float(point[1])


def point_to_xyz(point: Any) -> tuple[float, float, float]:
    """Return an ``(x, y, z)`` tuple from a point-like object or sequence."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return to_float(point.x), to_float(point.y), to_float(point.z)
    return to_float(point[0]), to_float(point[1]), to_float(point[2])


def positions_to_tuples(positions: list[Any]) -> list[tuple[float, float, float]]:
    """Convert a sequence of point-like positions to float triples."""
    return [point_to_xyz(position) for position in positions]


def positions_to_velocities(
    positions: list[Any],
    *,
    duration_s: float | None = None,
) -> np.ndarray | list[Any]:
    """Return per-step finite-difference velocities for sampled positions.

    The first sample reuses the first forward difference so plotted series have
    the same length as the position series. When ``duration_s`` is known, values
    are converted from displacement per step to meters per second.
    """
    if not positions:
        return []

    arr = np.asarray([point_to_xyz(position) for position in positions], dtype=np.float64)
    if len(arr) == 1:
        return np.zeros((1, 3), dtype=np.float64)

    velocities = np.zeros_like(arr)
    velocities[1:] = arr[1:] - arr[:-1]
    velocities[0] = arr[1] - arr[0]

    if duration_s is not None and duration_s > 0.0:
        dt = duration_s / float(len(arr) - 1)
        if dt > 0.0:
            velocities = velocities / dt
    return velocities


def orientations_to_angular_velocities(orientations: Any) -> np.ndarray:
    """Return adjacent yaw/pitch/roll differences for each output step.

    Orientation samples are expected in degrees following ORCHAV's yaw/pitch/roll
    conventions. The result is measured in degrees per output step and is not a
    time-scaled angular velocity.
    """
    if len(orientations) <= 1:
        return np.zeros((len(orientations), 3), dtype=np.float64)
    orientation_array = np.asarray(orientations, dtype=np.float64)
    angular_velocities = np.zeros_like(orientation_array)
    angular_velocities[1:] = orientation_array[1:] - orientation_array[:-1]
    angular_velocities[0] = orientation_array[1] - orientation_array[0]
    return angular_velocities


def prepare_actor_state_data(actor_state_manager: Any) -> dict[str, Any]:
    """Prepare cached actor-state samples for all summary figure generators.

    Calling ``prepare_cached()`` here means requested summary figures and the
    later ray-tracing loop read the same TX/RX/target state timeline.
    """
    actor_state_cache = actor_state_manager.prepare_cached()
    tx_positions = actor_state_cache.tx_positions
    rx_positions = actor_state_cache.rx_positions
    tgt_positions = actor_state_cache.target_positions
    tx_orientations = actor_state_cache.tx_orientations
    rx_orientations = actor_state_cache.rx_orientations
    tgt_orientations = actor_state_cache.target_orientations

    all_positions: list[tuple[float, float, float]] = []
    for positions in tx_positions:
        all_positions.extend(positions_to_tuples(positions))
    for positions in rx_positions:
        all_positions.extend(positions_to_tuples(positions))
    for positions in tgt_positions:
        all_positions.extend(positions_to_tuples(positions))

    return {
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "tgt_positions": tgt_positions,
        "tx_orientations": tx_orientations,
        "rx_orientations": rx_orientations,
        "tgt_orientations": tgt_orientations,
        "all_positions": all_positions,
    }


def collect_velocity_data_from_actor_state_manager(
    actor_state_manager: Any,
    simulation_config: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collect named TX, RX, and target velocity series from an actor-state manager."""
    actor_state_data = prepare_actor_state_data(actor_state_manager)
    duration = None
    if simulation_config is not None:
        duration = float(getattr(simulation_config, "duration", 0.0) or 0.0)

    velocity_data: dict[str, list[dict[str, Any]]] = {"tx": [], "rx": [], "targets": []}

    for index, positions in enumerate(actor_state_data["tx_positions"]):
        velocity_data["tx"].append(
            {
                "name": actor_state_manager.tx_configs[index].name,
                "velocities": positions_to_velocities(positions, duration_s=duration),
            }
        )
    for index, positions in enumerate(actor_state_data["rx_positions"]):
        velocity_data["rx"].append(
            {
                "name": actor_state_manager.rx_configs[index].name,
                "velocities": positions_to_velocities(positions, duration_s=duration),
            }
        )
    for index, positions in enumerate(actor_state_data["tgt_positions"]):
        velocity_data["targets"].append(
            {
                "name": actor_state_manager.target_managers[index].config.name,
                "velocities": positions_to_velocities(positions, duration_s=duration),
            }
        )

    return velocity_data


def collect_orientation_data_from_actor_state_manager(
    actor_state_manager: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Collect named TX, RX, and target orientation series from an actor-state manager."""
    actor_state_data = prepare_actor_state_data(actor_state_manager)
    orientation_data: dict[str, list[dict[str, Any]]] = {"tx": [], "rx": [], "targets": []}

    for index, orientations in enumerate(actor_state_data["tx_orientations"]):
        orientation_data["tx"].append(
            {
                "name": actor_state_manager.tx_configs[index].name,
                "orientations": orientations,
                "angular_velocities": orientations_to_angular_velocities(orientations),
            }
        )
    for index, orientations in enumerate(actor_state_data["rx_orientations"]):
        orientation_data["rx"].append(
            {
                "name": actor_state_manager.rx_configs[index].name,
                "orientations": orientations,
                "angular_velocities": orientations_to_angular_velocities(orientations),
            }
        )
    for index, orientations in enumerate(actor_state_data["tgt_orientations"]):
        orientation_data["targets"].append(
            {
                "name": actor_state_manager.target_managers[index].config.name,
                "orientations": orientations,
                "angular_velocities": orientations_to_angular_velocities(orientations),
            }
        )

    return orientation_data
