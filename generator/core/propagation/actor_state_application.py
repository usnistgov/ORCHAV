"""Apply per-frame actor state to live Sionna scene objects.

The actor-state manager returns ordinary Python values for the frame being solved.
This module is the mutation boundary that writes those values onto live Sionna
TX/RX objects and target managers before the path solver runs.

Per-frame actor state is one indexed slice from
``SimulationObjects.actor_state_manager``: desired positions, orientations, and
velocities for a single frame. It is unrelated to ``FrozenPaths``, which is a
CPU copy of solver output used later for caching.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from shared.logging import get_logger

from ..exceptions import ComputationError
from ..mobility.base import Position3
from ..orientation.base import Orientation3
from ..sionna_integration import orientation_to_point3f, point3f
from ..target.mesh import mesh_call_count_for_step, mesh_update_step_interval
from ..utils import point_to_tuple

logger = get_logger(__name__)


def apply_transceiver_state_to_scene(
    devices: list[Any],
    step_positions: Sequence[Position3 | None],
    step_orientations: Sequence[Orientation3 | None],
    step_velocities: Sequence[Position3 | None] | None = None,
) -> None:
    """Apply one frame of TX/RX state to live Sionna device objects.

    ``step_positions``, ``step_orientations``, and ``step_velocities`` are
    ordered like ``devices``. A ``None`` entry means "leave the current runtime
    value alone" for that field. Positions and velocities are converted to
    Sionna ``Point3f`` objects; orientations are project-facing
    ``(yaw, pitch, roll)`` degrees converted to engine radians.
    """
    for idx, device in enumerate(devices):
        if idx < len(step_positions) and step_positions[idx] is not None:
            device.position = point3f(step_positions[idx])
        if idx < len(step_orientations):
            orientation = step_orientations[idx]
            if orientation is not None:
                device.orientation = orientation_to_point3f(orientation)
        if step_velocities is not None and idx < len(step_velocities):
            velocity = step_velocities[idx]
            if velocity is not None:
                device.velocity = point3f(velocity)


def compute_mesh_update_step_interval(simulation_config: Any | None) -> int | None:
    """Return the requested mesh cadence in simulation steps, if configured."""
    if simulation_config is None:
        return None

    return mesh_update_step_interval(
        duration=getattr(simulation_config, "duration", None),
        num_steps=getattr(simulation_config, "num_steps", None),
        mesh_update_interval_s=getattr(simulation_config, "mesh_update_interval_s", None),
    )


def should_update_mesh_for_step(
    frame_idx: int,
    simulation_config: Any | None,
) -> bool:
    """Return whether mesh switching is allowed on this step.

    Single-step propagation follows the configured per-step cadence. Multi-step
    coherent propagation quantizes mesh updates to acquisition boundaries so
    geometry remains frozen inside each propagation interval, including when a
    run contains only one partial interval.
    """
    mesh_step_interval = compute_mesh_update_step_interval(simulation_config)
    if simulation_config is None:
        return True

    cir_steps = max(1, int(getattr(simulation_config, "cir_time_steps", 1) or 1))
    start_step = int(getattr(simulation_config, "start_step", 0) or 0)
    relative_step = max(0, int(frame_idx) - start_step)
    if cir_steps <= 1:
        if mesh_step_interval is None:
            return True
        return (relative_step % mesh_step_interval) == 0

    if (relative_step % cir_steps) != 0:
        return False

    if mesh_step_interval is None:
        return True

    acquisition_index = relative_step // cir_steps
    acquisitions_per_mesh = max(1, math.ceil(mesh_step_interval / cir_steps))
    return (acquisition_index % acquisitions_per_mesh) == 0


def apply_target_scale_overrides(
    target_managers: list[Any],
    target_override_entries: Mapping[str, Any] | None,
) -> None:
    """Apply each target's configured scale or its temporary live override.

    Scale is not part of the TX/RX assignment path, so live override handling
    defers it until target managers are available. Reapplying the configured
    scale when no override is present prevents one live session from changing
    the scene observed by the next session.
    """
    for manager in target_managers:
        cfg_name = getattr(manager.config, "name", None)
        if not cfg_name:
            continue
        entry = None
        if target_override_entries:
            entry = target_override_entries.get(cfg_name)
            if entry is None:
                entry = next(
                    (
                        value
                        for key, value in target_override_entries.items()
                        if key.lower() == cfg_name.lower()
                    ),
                    None,
                )
        scale_value = getattr(entry, "scale", None) if entry else None
        if scale_value is None and isinstance(entry, dict):
            scale_value = entry.get("scale")
        if scale_value is None:
            scale_value = getattr(manager.config, "scale", 1.0)
        try:
            manager.apply_scale_snapshot(float(scale_value))
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning(
                "Could not apply runtime scale %s to target %s: %s",
                scale_value,
                cfg_name,
                exc,
            )


def apply_target_state_to_scene(
    target_managers: list[Any],
    step_positions: Sequence[Position3 | None],
    step_orientations: Sequence[Orientation3 | None],
    frame_idx: int,
    step_velocities: Sequence[Position3 | None] | None = None,
    simulation_config: Any | None = None,
) -> None:
    """Apply one frame of target state to target managers and scene objects.

    Targets have more state than TX/RX devices: their scene object may be
    replaced when a mesh sequence advances, and some meshes provide their own
    position from the PLY geometry. This helper keeps those policies together so
    ``compute_ray_tracing_step`` can simply apply the frame's actor-state
    before solving paths.
    """
    should_update_mesh = should_update_mesh_for_step(frame_idx, simulation_config)
    if frame_idx == 0 and simulation_config is not None:
        mesh_interval = getattr(simulation_config, "mesh_update_interval_s", None)
        cir_steps = max(1, int(getattr(simulation_config, "cir_time_steps", 1) or 1))
        if mesh_interval is not None:
            num_steps = getattr(simulation_config, "num_steps", None)
            duration = getattr(simulation_config, "duration", None)
            if num_steps and duration and duration > 0:
                dt = duration / max(int(num_steps), 1)
                step_interval = compute_mesh_update_step_interval(simulation_config) or 1
                if cir_steps > 1:
                    acquisitions_per_mesh = max(1, math.ceil(step_interval / cir_steps))
                    effective_steps = acquisitions_per_mesh * cir_steps
                    logger.debug(
                        "Coherent mesh update interval: %.4f s -> nominal every %d steps "
                        "(dt=%.4f s), quantized to every %d acquisition(s) / %d steps",
                        mesh_interval,
                        step_interval,
                        dt,
                        acquisitions_per_mesh,
                        effective_steps,
                    )
                else:
                    logger.debug(
                        "Mesh update interval: %.4f s -> every %d steps (dt=%.4f s)",
                        mesh_interval,
                        step_interval,
                        dt,
                    )

    for idx, target_manager in enumerate(target_managers):
        if idx < len(step_positions):
            position_value = step_positions[idx]
            if (
                position_value is not None
                and getattr(target_manager.config, "mobility", None) is not None
                and not getattr(target_manager.config, "use_ply_position", False)
            ):
                # PLY-position targets are positioned by the mesh sequence
                # itself; applying the mobility position here would double-move
                # the object relative to its mesh-derived center.
                target_name = getattr(target_manager.config, "name", f"target_{idx}")
                try:
                    target_manager.apply_position_snapshot(
                        point_to_tuple(position_value, error_type=ValueError)
                    )
                except Exception as exc:  # noqa: BLE001 - external setter needs frame context
                    raise ComputationError(
                        f"Target {target_name!r} state update failed at frame {frame_idx} "
                        f"while setting position: {exc}"
                    ) from exc

        velocity_tuple = None
        if step_velocities is not None and idx < len(step_velocities):
            velocity_tuple = step_velocities[idx]

        orientation_tuple = None
        if idx < len(step_orientations):
            orientation_tuple = step_orientations[idx]

        if should_update_mesh and getattr(target_manager.config, "switch_meshes", True):
            mesh_step_interval = compute_mesh_update_step_interval(simulation_config) or 1
            # The expected count is derived from simulation time because coherent
            # acquisition can intentionally skip intermediate mesh updates.
            expected_call_count = mesh_call_count_for_step(frame_idx, mesh_step_interval)
            target_manager.update_mesh_for_frame(
                frame_idx,
                expected_call_count=expected_call_count,
            )
            logger.debug(
                "Mesh update at step %d -> mesh_idx=%s (target '%s')",
                frame_idx,
                getattr(target_manager, "current_mesh_idx", None),
                getattr(target_manager.config, "name", f"target_{idx}"),
            )

        if velocity_tuple is not None:
            # Mesh switching can replace the underlying target object, so write
            # velocity after any update to keep the current scene object in sync.
            target_obj = getattr(target_manager, "target_object", None)
            target_name = getattr(target_manager.config, "name", f"target_{idx}")
            if target_obj is None:
                raise ComputationError(
                    f"Target {target_name!r} velocity update failed at frame "
                    f"{frame_idx}: no active scene object exists"
                )
            try:
                target_obj.velocity = point3f(velocity_tuple)
            except Exception as exc:  # noqa: BLE001 - external setter needs frame context
                message = (
                    f"Target {target_name!r} velocity update failed at frame " f"{frame_idx}: {exc}"
                )
                raise ComputationError(message) from exc

        if orientation_tuple is not None:
            target_name = getattr(target_manager.config, "name", f"target_{idx}")
            try:
                target_manager.apply_orientation_snapshot(orientation_tuple)
            except Exception as exc:  # noqa: BLE001 - external setter needs frame context
                raise ComputationError(
                    f"Target {target_name!r} state update failed at frame {frame_idx} "
                    f"while setting orientation: {exc}"
                ) from exc
