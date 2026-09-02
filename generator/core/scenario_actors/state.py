#!/usr/bin/env python3
"""Prepared timeline state for transmitters, receivers, and targets.

``ActorStateManager`` assembles positions, orientations, and velocities for the
complete timeline. Mobility and orientation providers generate each series;
this module caches those series and returns one indexed snapshot per frame.
"""

from dataclasses import dataclass
from typing import Any

from shared.logging import get_logger
from shared.scenarios.actors import OrientationSpec

from ..configuration import ReceiverConfig, TransmitterConfig
from ..mobility.base import Position3
from ..orientation.base import (
    Orientation3,
    PreparedOrientationSource,
    orientation_to_tuple,
)
from ..target.mesh import positions_from_ply_aabb
from ..utils import point_to_tuple
from .mobility import prepare_sampled_mobility
from .orientation import apply_asset_alignment, prepare_orientation
from .quaternion import Quaternion
from .types import PreparedMobility, PreparedOrientation, Timeline

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ActorStateCache:
    """Prepared state for all actors over the full simulation timeline.

    Each field is grouped by actor role and then by actor index. For example,
    ``tx_positions[0][step]`` is the first transmitter's position at ``step``.
    """

    tx_positions: list[list[Position3]]
    rx_positions: list[list[Position3]]
    target_positions: list[list[Position3]]
    tx_orientations: list[list[Orientation3]]
    rx_orientations: list[list[Orientation3]]
    target_orientations: list[list[Orientation3]]


@dataclass(frozen=True, slots=True)
class ActorStateSnapshot:
    """One-step view of the prepared actor-state cache.

    The view answers "where is every TX, RX, and target, and how is it
    oriented, at this step?" The list ordering matches the original config
    ordering: all TX entries, all RX entries, and all target managers. Targets
    that exist but have no prepared position use ``None`` so the target list
    still lines up with target metadata and scene objects.
    """

    tx_positions: list[Position3]
    rx_positions: list[Position3]
    target_positions: list[Position3 | None]
    tx_orientations: list[Orientation3]
    rx_orientations: list[Orientation3]
    target_orientations: list[Orientation3]


@dataclass(frozen=True, slots=True)
class ActorStateVelocities:
    """Velocities for every actor at one simulation step."""

    tx: list[Position3 | None]
    rx: list[Position3 | None]
    target: list[Position3 | None]


class ActorStateManager:
    """Assemble cached actor state for propagation and visualization.

    The manager prepares all TX, RX, and target positions/orientations for the
    simulation steps, including orientations that depend on other actors.
    Streaming mode changes ray-tracing execution, while both file and streaming
    paths read from the same prepared actor state.

    Args:
        tx_configs: Transmitter configurations with mobility and orientation.
        rx_configs: Receiver configurations with mobility and orientation.
        target_managers: Target managers with mobility and orientation.
        steps: Total number of simulation time steps.
        duration: Total simulation duration in seconds.
        motion_mode: Either ``"cached"`` for offline output or ``"step"`` for
            streamed ray tracing.
    """

    def __init__(
        self,
        tx_configs: list[TransmitterConfig],
        rx_configs: list[ReceiverConfig],
        target_managers: list,
        steps: int,
        duration: float,
        motion_mode: str = "cached",
        mesh_update_interval_s: float | None = None,
    ):
        self.tx_configs = tx_configs
        self.rx_configs = rx_configs
        self.target_managers = target_managers
        self.steps = int(steps)
        self.duration = float(duration)
        self.motion_mode = motion_mode.lower()
        self.mesh_update_interval_s = mesh_update_interval_s
        self._tx_positions_cache: list[list[Position3]] = []
        self._rx_positions_cache: list[list[Position3]] = []
        self._tgt_positions_cache: list[list[Position3]] = []
        self._tx_orientations_cache: list[list[Orientation3]] = []
        self._rx_orientations_cache: list[list[Orientation3]] = []
        self._tgt_orientations_cache: list[list[Orientation3]] = []
        self._tx_velocities_cache: list[list[Position3]] = []
        self._rx_velocities_cache: list[list[Position3]] = []
        self._tgt_velocities_cache: list[list[Position3]] = []
        self._actor_mobility_lookup: dict[str, PreparedMobility] = {}
        self._state_cache: ActorStateCache | None = None
        self._timeline = Timeline(self.steps, self.duration)

    def is_streaming_state(self) -> bool:
        """Return whether propagation is being driven by streaming mode."""
        return self.motion_mode == "step"

    def _clamp_step(self, step: int, caller: str) -> int:
        """Clamp a requested step to the prepared timeline and warn on misuse."""
        requested_step = int(step)
        last_step = max(self.steps - 1, 0)
        if requested_step < 0 or requested_step > last_step:
            clamped_step = min(max(requested_step, 0), last_step)
            logger.warning(
                "%s requested step %d outside prepared range [0, %d]; using step %d",
                caller,
                requested_step,
                last_step,
                clamped_step,
            )
            return clamped_step
        return requested_step

    def _prepare_orientation_series(
        self,
        orientation: OrientationSpec | PreparedOrientationSource | None,
        mobility: PreparedMobility,
        *,
        path: str,
        asset_alignment_yaw_deg: float | None = None,
    ) -> list[Orientation3]:
        """Prepare one schema model or external source through canonical quaternions."""

        if orientation is None:
            return []

        alignment_applied = False
        if isinstance(orientation, PreparedOrientationSource):
            orientation.prepare(
                self.steps,
                self.duration,
                context={"self_positions": list(mobility.positions_m)},
            )
            samples = tuple(
                orientation_to_tuple(value, f"{path}[{index}]")
                for index, value in enumerate(orientation.orientations())
            )
            if len(samples) != self.steps:
                raise ValueError(
                    f"{path} returned {len(samples)} samples for {self.steps} timeline steps"
                )
            prepared = PreparedOrientation(
                tuple(Quaternion.from_euler_deg(*sample) for sample in samples),
                asset_alignment_applied=bool(
                    getattr(orientation, "asset_alignment_applied", False)
                ),
            )
            alignment_applied = prepared.asset_alignment_applied
        else:
            prepared = prepare_orientation(
                orientation,
                self._timeline,
                mobility,
                references=self._actor_mobility_lookup,
                path=path,
            )

        if (
            asset_alignment_yaw_deg is not None
            and abs(float(asset_alignment_yaw_deg)) > 1e-12
            and not alignment_applied
        ):
            prepared = apply_asset_alignment(
                prepared,
                (float(asset_alignment_yaw_deg), 0.0, 0.0),
                path=f"{path}.asset_alignment",
            )
        return list(prepared.euler_deg)

    @staticmethod
    def _normalize_positions(positions: list[Any], label: str) -> list[Position3]:
        """Normalize prepared provider output to canonical Position3 tuples."""
        normalized: list[Position3] = []
        for idx, position in enumerate(positions):
            try:
                normalized.append(point_to_tuple(position))
            except ValueError as exc:
                raise ValueError(
                    f"{label} position at step {idx} must be a numeric Position3"
                ) from exc
        return normalized

    def _validate_unique_actor_names(self) -> None:
        """Require one unambiguous name across scripted actor roles."""

        labels_and_names: list[tuple[str, str]] = [
            (f"TX[{index}]", str(config.name)) for index, config in enumerate(self.tx_configs)
        ]
        labels_and_names.extend(
            (f"RX[{index}]", str(config.name)) for index, config in enumerate(self.rx_configs)
        )
        labels_and_names.extend(
            (
                f"target[{index}]",
                str(getattr(manager.config, "name", f"target_{index}")),
            )
            for index, manager in enumerate(self.target_managers)
        )
        first_labels: dict[str, str] = {}
        for label, name in labels_and_names:
            first_label = first_labels.get(name)
            if first_label is not None:
                raise ValueError(
                    f"Actor name {name!r} is duplicated by {first_label} and {label}; "
                    "canonical actor references require globally unique names"
                )
            first_labels[name] = label

    def prepare_cached(self) -> ActorStateCache:
        """
        Pre-compute and cache all actor positions and orientations.

        This is the only place in this manager where mobility/orientation
        providers are prepared. Per-frame access methods must only index this
        cached state.

        Returns:
            Prepared timeline cache with named position and orientation arrays.
        """
        self._validate_unique_actor_names()

        # Mobility classes own position generation; this manager owns assembling
        # those prepared results into a shared actor-state cache.
        self._tx_positions_cache = []
        self._tx_velocities_cache = []
        for tx_cfg in self.tx_configs:
            tx_cfg.mobility.prepare(self.steps, self.duration, tx_cfg.initial_position)
            positions = self._normalize_positions(
                tx_cfg.mobility.prepared_positions(),
                f"TX {tx_cfg.name!r}",
            )
            self._tx_positions_cache.append(positions)
            self._tx_velocities_cache.append(
                self._prepared_velocities(tx_cfg.mobility, f"TX {tx_cfg.name!r}")
            )

        self._rx_positions_cache = []
        self._rx_velocities_cache = []
        for rx_cfg in self.rx_configs:
            rx_cfg.mobility.prepare(self.steps, self.duration, rx_cfg.initial_position)
            positions = self._normalize_positions(
                rx_cfg.mobility.prepared_positions(),
                f"RX {rx_cfg.name!r}",
            )
            self._rx_positions_cache.append(positions)
            self._rx_velocities_cache.append(
                self._prepared_velocities(rx_cfg.mobility, f"RX {rx_cfg.name!r}")
            )

        self._tgt_positions_cache = []
        self._tgt_velocities_cache = []
        for idx, tm in enumerate(self.target_managers):
            target_cfg = tm.config
            target_name = getattr(target_cfg, "name", f"target_{idx}")
            if target_cfg.use_ply_position and getattr(tm, "meshes", None):
                # Target mesh helpers own mesh-derived positions; keep this
                # prepare-time so per-frame state reads never touch mesh files.
                positions = positions_from_ply_aabb(
                    tm.meshes,
                    self.steps,
                    self.duration,
                    getattr(target_cfg, "mesh_start_index", 0),
                    getattr(target_cfg, "mesh_frame_stride", 1),
                    self.mesh_update_interval_s,
                    getattr(target_cfg, "mesh_end_behavior", "loop"),
                )
                positions = self._normalize_positions(positions, f"Target {target_name!r}")
                self._tgt_positions_cache.append(positions)
                self._tgt_velocities_cache.append([])
            elif target_cfg.mobility is not None and not target_cfg.use_ply_position:
                target_cfg.mobility.prepare(self.steps, self.duration, target_cfg.initial_position)
                positions = self._normalize_positions(
                    target_cfg.mobility.prepared_positions(),
                    f"Target {target_name!r}",
                )
                self._tgt_positions_cache.append(positions)
                self._tgt_velocities_cache.append(
                    self._prepared_velocities(
                        target_cfg.mobility,
                        f"Target {target_name!r}",
                    )
                )
            else:
                # Some targets only provide mesh/material metadata. Preserve the
                # target slot so per-frame state aligns with target manager ordering.
                self._tgt_positions_cache.append([])
                self._tgt_velocities_cache.append([])

        # Scripted mobility providers already sampled their trajectories. Convert
        # those samples once so every orientation model uses the canonical
        # velocity, stable-forward, and actor-reference semantics.
        self._actor_mobility_lookup = {}
        for tx_cfg, positions in zip(self.tx_configs, self._tx_positions_cache):
            self._actor_mobility_lookup[tx_cfg.name] = prepare_sampled_mobility(
                tuple(positions),
                self._timeline,
                physical_velocity=self._has_physical_motion(tx_cfg.mobility),
                path=f"transmitters.{tx_cfg.name}.mobility",
            )
        for rx_cfg, positions in zip(self.rx_configs, self._rx_positions_cache):
            self._actor_mobility_lookup[rx_cfg.name] = prepare_sampled_mobility(
                tuple(positions),
                self._timeline,
                physical_velocity=self._has_physical_motion(rx_cfg.mobility),
                path=f"receivers.{rx_cfg.name}.mobility",
            )
        for index, (target_manager, positions) in enumerate(
            zip(self.target_managers, self._tgt_positions_cache)
        ):
            if not positions:
                continue
            target_name = getattr(target_manager.config, "name", f"target_{index}")
            self._actor_mobility_lookup[target_name] = prepare_sampled_mobility(
                tuple(positions),
                self._timeline,
                physical_velocity=self._has_physical_motion(
                    getattr(target_manager.config, "mobility", None)
                ),
                path=f"targets.{target_name}.mobility",
            )

        # Schema models own orientation intent; the quaternion kernel evaluates
        # every role against the same named actor mobility map.
        self._tx_orientations_cache = []
        for idx, tx_cfg in enumerate(self.tx_configs):
            self._tx_orientations_cache.append(
                self._prepare_orientation_series(
                    tx_cfg.orientation,
                    self._actor_mobility_lookup[tx_cfg.name],
                    path=f"transmitters.{tx_cfg.name}.orientation",
                )
            )

        self._rx_orientations_cache = []
        for idx, rx_cfg in enumerate(self.rx_configs):
            self._rx_orientations_cache.append(
                self._prepare_orientation_series(
                    rx_cfg.orientation,
                    self._actor_mobility_lookup[rx_cfg.name],
                    path=f"receivers.{rx_cfg.name}.orientation",
                )
            )

        self._tgt_orientations_cache = []
        for idx, tm in enumerate(self.target_managers):
            target_cfg = tm.config
            target_name = getattr(target_cfg, "name", f"target_{idx}")
            mobility = self._actor_mobility_lookup.get(target_name)
            if mobility is None:
                self._tgt_orientations_cache.append([])
                continue
            self._tgt_orientations_cache.append(
                self._prepare_orientation_series(
                    target_cfg.orientation,
                    mobility,
                    path=f"targets.{target_name}.orientation",
                    asset_alignment_yaw_deg=getattr(
                        target_cfg,
                        "asset_front_yaw_offset_deg",
                        None,
                    ),
                )
            )

        self._state_cache = ActorStateCache(
            tx_positions=self._tx_positions_cache,
            rx_positions=self._rx_positions_cache,
            target_positions=self._tgt_positions_cache,
            tx_orientations=self._tx_orientations_cache,
            rx_orientations=self._rx_orientations_cache,
            target_orientations=self._tgt_orientations_cache,
        )
        return self._state_cache

    def _prepared_velocities(self, mobility: Any, label: str) -> list[Position3]:
        """Read canonical kernel velocities when a mobility adapter provides them."""

        if not self._has_physical_motion(mobility):
            return [(0.0, 0.0, 0.0)] * self.steps
        provider = getattr(mobility, "prepared_velocities", None)
        if not callable(provider):
            return []
        velocities = self._normalize_positions(provider(), f"{label} velocity")
        if len(velocities) != self.steps:
            raise ValueError(
                f"{label} prepared velocity count {len(velocities)} does not match "
                f"timeline steps {self.steps}"
            )
        return velocities

    @staticmethod
    def _has_physical_motion(mobility: object | None) -> bool:
        """Return whether sampled displacements represent physical motion."""

        return bool(getattr(mobility, "has_physical_motion", True))

    def compute_velocities(self, step: int) -> ActorStateVelocities:
        """Compute actor velocities from prepared or cached position series.

        Ray tracing assigns these velocities to Sionna RT scene objects before
        solving a frame. Sionna uses them for coherent CIR Doppler behavior and
        downstream frame metadata stores target velocities for sensing ground
        truth. Computing velocities from the same prepared positions used by
        ``state_at_step()`` keeps geometry, Doppler inputs, and exported metadata
        consistent.

        For step 0, velocity uses a forward difference when a next position is
        available.
        For step i > 0: v[i] = (pos[i] - pos[i-1]) / dt.

        Args:
            step: Time step index (0 to steps-1).

        Returns:
            Named velocity lists for TX, RX, and targets. Entries are velocity
            tuples (vx, vy, vz) in m/s, or None for actors without cached positions.
        """
        if self._state_cache is None:
            self.prepare_cached()

        dt = self.duration / max(self.steps - 1, 1) if self.steps > 1 else 1.0
        i = self._clamp_step(step, "compute_velocities")

        def _velocity_for(
            positions_cache: list[list[Position3]],
            canonical_cache: list[list[Position3]],
        ) -> list[Position3 | None]:
            result: list[Position3 | None] = []
            for actor_index, positions in enumerate(positions_cache):
                if not positions or i >= len(positions):
                    result.append(None)
                    continue
                if actor_index < len(canonical_cache):
                    canonical = canonical_cache[actor_index]
                    if canonical and i < len(canonical):
                        result.append(canonical[i])
                        continue
                if i == 0:
                    # Forward difference at step 0: use (pos[1] - pos[0]) / dt
                    # so Sionna CIR gets the correct initial velocity for
                    # phase-coherent Doppler evolution from the first CPI.
                    if len(positions) > 1:
                        x0, y0, z0 = positions[0]
                        x1, y1, z1 = positions[1]
                        result.append(((x1 - x0) / dt, (y1 - y0) / dt, (z1 - z0) / dt))
                    else:
                        result.append((0.0, 0.0, 0.0))
                    continue
                cx, cy, cz = positions[i]
                px, py, pz = positions[i - 1]
                result.append(((cx - px) / dt, (cy - py) / dt, (cz - pz) / dt))
            return result

        return ActorStateVelocities(
            tx=_velocity_for(self._tx_positions_cache, self._tx_velocities_cache),
            rx=_velocity_for(self._rx_positions_cache, self._rx_velocities_cache),
            target=_velocity_for(self._tgt_positions_cache, self._tgt_velocities_cache),
        )

    def state_at_step(self, step: int) -> ActorStateSnapshot:
        """
        Return a one-step slice of the prepared actor-state cache.

        The returned state contains the TX, RX, and target positions and
        orientations to apply for one frame before ray tracing. This method is on
        the propagation hot path, so it must remain simple cache indexing: no
        mesh I/O, mobility preparation, trajectory generation, or orientation
        preparation.

        Args:
            step: Time step index (0 to steps-1)

        Returns:
            Named actor state for the specified time step. Out-of-range
            requests are clamped to the nearest prepared step.
        """
        if self._state_cache is None:
            self.prepare_cached()
        i = self._clamp_step(step, "state_at_step")

        tx_pos_step = [pos[i] for pos in self._tx_positions_cache]
        rx_pos_step = [pos[i] for pos in self._rx_positions_cache]
        tgt_pos_step = []
        for pos in self._tgt_positions_cache:
            if pos and i < len(pos):
                tgt_pos_step.append(pos[i])
            else:
                # Targets without prepared positions still exist as scene
                # objects; ``None`` keeps the per-step state shape stable.
                tgt_pos_step.append(None)

        tx_ori_step = []
        for arr in self._tx_orientations_cache:
            # A missing orientation means "no rotation override" for the scene
            # assignment layer, represented as a zero yaw/pitch/roll tuple.
            tx_ori_step.append(arr[i] if arr and i < len(arr) else (0.0, 0.0, 0.0))
        rx_ori_step = []
        for arr in self._rx_orientations_cache:
            rx_ori_step.append(arr[i] if arr and i < len(arr) else (0.0, 0.0, 0.0))
        tgt_ori_step = []
        for arr in self._tgt_orientations_cache:
            tgt_ori_step.append(arr[i] if arr and i < len(arr) else (0.0, 0.0, 0.0))

        return ActorStateSnapshot(
            tx_positions=tx_pos_step,
            rx_positions=rx_pos_step,
            target_positions=tgt_pos_step,
            tx_orientations=tx_ori_step,
            rx_orientations=rx_ori_step,
            target_orientations=tgt_ori_step,
        )
