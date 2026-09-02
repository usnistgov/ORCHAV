"""Actor-state preparation service for generator pipelines.

``ActorStateService`` turns parsed TX/RX configs and target managers into an
``ActorStateManager`` plus cached per-step arrays. It does not write positions
or orientations into Sionna objects.  Propagation reads per-step state from the
manager later and applies it to the live scene just before solving a frame.
"""

from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from ..scenario_actors.state import ActorStateCache, ActorStateManager
from ..target import TargetConfig, TargetManager
from .base import BaseService


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
        )

        cache = self.actor_state_manager.prepare_cached()
        return self.actor_state_manager, cache

    def cleanup(self) -> None:
        """Release the cached actor-state manager after a pipeline run."""
        self.actor_state_manager = None
