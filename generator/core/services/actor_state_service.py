"""Actor-state preparation service for generator pipelines.

``ActorStateService`` turns parsed TX/RX configs and target managers into an
``ActorStateManager`` plus cached per-step arrays. Coverage explicitly applies
its initial snapshot; propagation applies the same indexed state before each
frame solve. Neither operation advances the prepared timeline.
"""

from typing import TYPE_CHECKING

from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from ..propagation.actor_state_application import (
    apply_target_state_to_scene,
    apply_transceiver_state_to_scene,
)
from ..scenario_actors.state import ActorStateCache, ActorStateManager
from ..target import TargetConfig, TargetManager
from .base import BaseService

if TYPE_CHECKING:
    from .scene_service import SceneService


class ActorStateService(BaseService):
    """Prepare actor-state caches shared by offline and streaming pipelines."""

    def __init__(self, simulation_config: SimulationConfig):
        super().__init__(simulation_config)
        self.actor_state_manager: ActorStateManager | None = None
        self.max_steps_expanded: bool = False

    def normalize_scene_steps(
        self,
        tx_configs: list[TransmitterConfig],
        rx_configs: list[ReceiverConfig],
        target_configs: list[TargetConfig] | None = None,
    ) -> None:
        """Ensure ``num_steps`` covers mobility-driven frame requirements.

        Mesh-grid mobility can request one frame per grid point.  TX, RX, and
        targets share one simulation clock, so any actor can raise the global
        step count before state arrays are generated.
        """
        max_required_steps = self.simulation_config.num_steps
        all_configs = list(tx_configs) + list(rx_configs) + list(target_configs or [])
        for cfg in all_configs:
            mob = getattr(cfg, "mobility", None)
            if mob is not None and getattr(mob, "auto_expand_scene_steps", False):
                total_points = getattr(mob, "total_points", None)
                if isinstance(total_points, int) and total_points > max_required_steps:
                    max_required_steps = total_points

        if max_required_steps > self.simulation_config.num_steps:
            self.logger.info(
                "Expanding scene steps from %s to %s to match mesh grid points",
                self.simulation_config.num_steps,
                max_required_steps,
            )
            self.simulation_config.num_steps = max_required_steps
            self.max_steps_expanded = True

    def prepare_actor_state(
        self,
        tx_configs: list[TransmitterConfig],
        rx_configs: list[ReceiverConfig],
        target_managers: list[TargetManager],
        motion_mode: str = "cached",
    ) -> tuple[ActorStateManager, ActorStateCache]:
        """Initialize the actor-state manager and prepare cached state.

        The cache is an internal generator representation of positions and
        orientations over time.  It is passed to ``RayTracingService`` so frame
        computation can reuse prepared arrays instead of re-parsing mobility
        definitions.

        Returns:
            Tuple of (ActorStateManager, prepared actor-state cache).
        """
        steps = self.simulation_config.num_steps
        duration = self.simulation_config.duration

        mesh_interval = getattr(self.simulation_config, "mesh_update_interval_s", None)
        # The mesh update interval is part of target state policy.  It is
        # captured with actor-state preparation, then enforced later when
        # propagation applies target state to the live scene.
        self.actor_state_manager = ActorStateManager(
            tx_configs,
            rx_configs,
            target_managers,
            steps,
            duration,
            motion_mode,
            mesh_update_interval_s=mesh_interval,
            sample_rate_hz=getattr(self.simulation_config, "scene_sample_rate_hz", None),
        )

        cache = self.actor_state_manager.prepare_cached()
        return self.actor_state_manager, cache

    def apply_initial_state(self, scene_service: "SceneService") -> None:
        """Apply canonical time-zero pose before coverage and topology output.

        Read the prepared cache without advancing mobility or rebuilding it.
        The same target application boundary as frame propagation preserves
        animated-body alignment and absolute mesh-frame ownership.
        """
        manager = self.actor_state_manager
        if manager is None:
            raise ValueError("Actor state must be prepared before initial application")
        state = manager.state_at_step(0)
        velocities = manager.compute_velocities(0)
        apply_transceiver_state_to_scene(
            scene_service.tx_list, state.tx_positions, state.tx_orientations, velocities.tx
        )
        apply_transceiver_state_to_scene(
            scene_service.rx_list, state.rx_positions, state.rx_orientations, velocities.rx
        )
        apply_target_state_to_scene(
            scene_service.target_managers,
            state.target_positions,
            state.target_orientations,
            0,
            step_velocities=velocities.target,
            simulation_config=self.simulation_config,
        )
        # A mesh-backed target may replace its live object during application.
        scene_service.target_objects = [
            target.target_object for target in scene_service.target_managers
        ]

    def cleanup(self) -> None:
        """Release the cached actor-state manager after a pipeline run."""
        self.actor_state_manager = None
