"""Run Sionna RT propagation steps and prepare one frame of path data.

``SimulationObjects.actor_state_manager`` owns the prepared per-step TX, RX,
and target state. This module asks it for one frame, lets live overrides adjust
that frame-local state, writes the result onto live Sionna scene objects, runs
the path solver, and stores numeric metadata snapshots in ``frame_data``.

Those metadata snapshots, such as ``tx_positions_snapshot``, are not
``FrozenPaths``. They record which actor state produced the frame.
``FrozenPaths`` is created later only when path buffers are frozen for cache
storage.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from shared.logging import get_logger

from ..materials.path_metadata import materials_per_bounce
from ..mobility.base import Position3
from ..orientation.base import Orientation3
from ..runtime import SimulationObjects
from ..sionna_integration import (
    PATH_SOLVER_SUPPORTS_DIFFRACTION,
    PATH_SOLVER_SUPPORTS_DIFFRACTION_LIT_REGION,
    PATH_SOLVER_SUPPORTS_EDGE_DIFFRACTION,
)
from ..utils import point_to_tuple
from .actor_state_application import (
    apply_target_scale_overrides,
    apply_target_state_to_scene,
    apply_transceiver_state_to_scene,
)
from .diagnostics import log_mpc_statistics
from .live_overrides import apply_live_overrides
from .snapshots import orientations_to_array, positions_to_array, velocities_to_array

logger = get_logger(__name__)


def compute_ray_tracing_step(
    simulation: SimulationObjects,
    frame_idx: int,
    live_overrides: Sequence[Any] | None = None,
    *,
    overrides: Sequence[Any] | None = None,
) -> dict[str, Any] | None:
    """Compute one propagation frame from the current simulation state.

    The actor-state manager provides the requested TX/RX/target positions and
    orientations. This function applies them to the live Sionna scene before
    solving paths, then returns the path result plus immutable numeric snapshots
    for storage, streaming, and visualization code.
    """
    if live_overrides is None and overrides is not None:
        live_overrides = overrides
    elif live_overrides is not None and overrides is not None:
        logger.warning("Both live_overrides and overrides supplied; using live_overrides")

    actor_state_manager = simulation.actor_state_manager
    if actor_state_manager is None:
        logger.error("Simulation is missing an actor-state manager")
        return None

    state_at_step = getattr(actor_state_manager, "state_at_step", None)
    if state_at_step is None:
        logger.error("Actor-state manager does not provide the required state_at_step method")
        return None

    logger.debug(
        "[COMPUTE_FRAME] Reading actor state for frame_idx=%s (display=%s)",
        frame_idx,
        frame_idx + 1,
    )
    actor_state = state_at_step(frame_idx)
    if actor_state.target_positions:
        logger.debug(
            "[COMPUTE_FRAME] Actor state returned %d target positions for frame_idx=%d",
            len(actor_state.target_positions),
            frame_idx,
        )

    # These lists are intentionally mutable. Live overrides edit only this
    # frame's requested state before assignment to Sionna objects; they do not
    # mutate the actor-state manager's prepared timeline.
    tx_pos_step: list[Position3 | None] = list(actor_state.tx_positions or [])
    rx_pos_step: list[Position3 | None] = list(actor_state.rx_positions or [])
    tgt_pos_step: list[Position3 | None] = list(actor_state.target_positions or [])
    tx_ori_step: list[Orientation3 | None] = list(actor_state.tx_orientations or [])
    rx_ori_step: list[Orientation3 | None] = list(actor_state.rx_orientations or [])
    tgt_ori_step: list[Orientation3 | None] = list(actor_state.target_orientations or [])

    applied_live_overrides = apply_live_overrides(
        live_overrides,
        simulation.tx_configs,
        simulation.rx_configs,
        simulation.target_configs,
        tx_pos_step,
        rx_pos_step,
        tgt_pos_step,
        tx_ori_step,
        rx_ori_step,
        tgt_ori_step,
    )

    step_velocities = None
    try:
        if hasattr(actor_state_manager, "compute_velocities"):
            step_velocities = actor_state_manager.compute_velocities(frame_idx)
    except (TypeError, ValueError, AttributeError):
        step_velocities = None

    tx_velocities = step_velocities.tx if step_velocities else None
    rx_velocities = step_velocities.rx if step_velocities else None
    tgt_velocities = step_velocities.target if step_velocities else None

    # TX/RX assignment boundary: plain Python positions/orientations become
    # Sionna-compatible Point3f values and engine radians before solving paths.
    apply_transceiver_state_to_scene(simulation.tx_list, tx_pos_step, tx_ori_step, tx_velocities)
    apply_transceiver_state_to_scene(simulation.rx_list, rx_pos_step, rx_ori_step, rx_velocities)
    logger.debug(
        "[COMPUTE_FRAME] Updating target states for frame_idx=%s (display=%s) with "
        "%d positions, %d orientations",
        frame_idx,
        frame_idx + 1,
        len(tgt_pos_step),
        len(tgt_ori_step),
    )
    apply_target_state_to_scene(
        simulation.target_managers,
        tgt_pos_step,
        tgt_ori_step,
        frame_idx,
        step_velocities=tgt_velocities,
        simulation_config=simulation.simulation_config,
    )
    apply_target_scale_overrides(
        simulation.target_managers,
        applied_live_overrides.get("target"),
    )

    settings = dict(simulation.settings)
    logger.debug(
        "Computing frame %s with %d TX and %d RX devices",
        frame_idx,
        len(simulation.tx_list),
        len(simulation.rx_list),
    )
    logger.debug("RT settings: %s", settings)

    solver_kwargs = {
        "scene": simulation.scene,
        "max_depth": settings["max_depth"],
        "max_num_paths_per_src": settings.get("max_num_paths_per_src", 100000),
        "samples_per_src": settings.get("samples_per_src", 100000),
        "los": settings.get("los", True),
        "specular_reflection": settings.get("specular_reflection", True),
        "diffuse_reflection": settings.get("diffuse_reflection", True),
        "refraction": settings.get("refraction", True),
        "synthetic_array": settings.get("synthetic_array", True),
        "seed": settings.get("seed", 42),
    }
    if PATH_SOLVER_SUPPORTS_DIFFRACTION:
        solver_kwargs["diffraction"] = settings.get("diffraction", False)
    if PATH_SOLVER_SUPPORTS_EDGE_DIFFRACTION:
        solver_kwargs["edge_diffraction"] = settings.get("edge_diffraction", False)
    if PATH_SOLVER_SUPPORTS_DIFFRACTION_LIT_REGION:
        solver_kwargs["diffraction_lit_region"] = settings.get("diffraction_lit_region", True)

    paths = simulation.path_solver(**solver_kwargs)
    log_mpc_statistics(paths, frame_idx)

    material_mapping = materials_per_bounce(simulation.scene, paths)

    frame_data: dict[str, Any] = {
        "tx_list": simulation.tx_list,
        "rx_list": simulation.rx_list,
        "paths": paths,
        "target_objects": [tm.target_object for tm in simulation.target_managers],
        "target_managers": simulation.target_managers,
        "material_mapping": material_mapping,
        "material_properties": simulation.material_properties,
        "frame_idx": frame_idx,
    }

    if simulation.simulation_config is not None:
        frame_data["simulation_config"] = simulation.simulation_config
    if simulation.scenario_configuration is not None:
        frame_data["scenario_configuration"] = simulation.scenario_configuration

    logger.debug(
        "[COMPUTE_FRAME] Creating snapshots for frame_idx=%s: "
        "TX positions count=%d, RX positions count=%d",
        frame_idx,
        len(tx_pos_step),
        len(rx_pos_step),
    )
    _log_positions_before_snapshot("TX", tx_pos_step)
    _log_positions_before_snapshot("RX", rx_pos_step)

    # Store immutable numeric copies of the state that produced this frame.
    # Live scene objects are handles reused by later frames, so cache/streaming
    # consumers should read these arrays instead of object attributes.
    frame_data["tx_positions_snapshot"] = positions_to_array(tx_pos_step)
    frame_data["rx_positions_snapshot"] = positions_to_array(rx_pos_step)
    frame_data["target_positions_snapshot"] = positions_to_array(tgt_pos_step)

    target_velocity_snapshot = velocities_to_array(tgt_velocities)
    if target_velocity_snapshot is not None:
        frame_data["target_velocities_snapshot"] = target_velocity_snapshot

    logger.debug(
        "[COMPUTE_FRAME] Snapshots created: TX shape=%s, RX shape=%s",
        frame_data["tx_positions_snapshot"].shape,
        frame_data["rx_positions_snapshot"].shape,
    )
    frame_data["tx_orientations_snapshot"] = orientations_to_array(tx_ori_step)
    frame_data["rx_orientations_snapshot"] = orientations_to_array(rx_ori_step)
    frame_data["target_orientations_snapshot"] = orientations_to_array(tgt_ori_step)

    return frame_data


def _log_positions_before_snapshot(label: str, positions: Sequence[Any]) -> None:
    if not positions or not logger.isEnabledFor(logging.DEBUG):
        return
    values = []
    for position in positions:
        if position is None:
            values.append((0.0, 0.0, 0.0))
            continue
        try:
            values.append(point_to_tuple(position, error_type=ValueError))
        except (TypeError, ValueError, AttributeError, IndexError):
            values.append("<invalid>")
    logger.debug("[COMPUTE_FRAME] %s positions before snapshot: %s", label, values)
