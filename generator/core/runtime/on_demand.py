"""Build reusable runtime contexts outside the default file pipeline.

The live gRPC server can receive requests before a full offline pipeline has
prepared ``SimulationObjects``.  This helper builds the same kind of runtime
bundle directly from parsed configs: it creates the scene, prepares actor state,
applies ray-tracing settings, and returns the container propagation expects.
"""

from typing import Any, Dict, List, Optional

from sionna.rt import PathSolver

from shared.logging import get_logger

from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from ..scenario_actors.state import ActorStateManager
from ..services.scene_service import SceneService
from .simulation_objects import SimulationObjects

logger = get_logger(__name__)


def build_on_demand_objects(
    tx_configs: List[TransmitterConfig],
    rx_configs: List[ReceiverConfig],
    target_configs: List,
    simulation_config: SimulationConfig,
    *,
    scenario_configuration: Optional[Any] = None,
    motion_mode: str = "step",
    raytracing_settings: Optional[Dict[str, Any]] = None,
) -> SimulationObjects:
    """Create ``SimulationObjects`` ready for step-by-step frame requests.

    This is not the default offline route.  It is an on-demand construction path
    for streaming/server code that needs a reusable context without running the
    file-output pipeline first.
    """

    if not simulation_config:
        raise ValueError("Simulation configuration is required for on-demand objects")
    if not tx_configs or not rx_configs:
        raise ValueError("Both TX and RX configurations are required for on-demand objects")

    logger.debug(
        "Building on-demand objects (tx=%d, rx=%d, targets=%d, motion_mode=%s)",
        len(tx_configs),
        len(rx_configs),
        len(target_configs or []),
        motion_mode,
    )

    scene_service = SceneService(simulation_config)
    scene, tx_list, rx_list, target_managers, _ = scene_service.build_scene(
        tx_configs,
        rx_configs,
        target_configs,
    )

    # On-demand contexts still use prepared actor state. The difference from
    # the file pipeline is who drives frame requests: a live caller asks for
    # individual steps instead of the pipeline iterating through all outputs.
    actor_state_manager = ActorStateManager(
        tx_configs,
        rx_configs,
        target_managers,
        simulation_config.num_steps,
        simulation_config.duration,
        motion_mode,
    )
    actor_state_cache = actor_state_manager.prepare_cached()

    settings = dict(simulation_config.get_quality_profile())
    if raytracing_settings:
        # Accept only solver keys understood by the propagation path.  This
        # keeps ad hoc request dictionaries from becoming implicit config API.
        for key in [
            "max_depth",
            "samples_per_src",
            "max_num_paths_per_src",
            "los",
            "specular_reflection",
            "diffuse_reflection",
            "refraction",
            "diffraction",
            "seed",
            "synthetic_array",
        ]:
            if key in raytracing_settings:
                settings[key] = raytracing_settings[key]

    context = SimulationObjects(
        scene=scene,
        tx_list=tx_list,
        rx_list=rx_list,
        target_managers=target_managers,
        actor_state_manager=actor_state_manager,
        settings=settings,
        path_solver=PathSolver(),
        tx_configs=tx_configs,
        rx_configs=rx_configs,
        target_configs=target_configs,
        motion_mode=motion_mode,
        tx_positions_cache=actor_state_cache.tx_positions,
        rx_positions_cache=actor_state_cache.rx_positions,
        tgt_positions_cache=actor_state_cache.target_positions,
        tx_orientations_cache=actor_state_cache.tx_orientations,
        rx_orientations_cache=actor_state_cache.rx_orientations,
        tgt_orientations_cache=actor_state_cache.target_orientations,
        simulation_config=simulation_config,
        scenario_configuration=scenario_configuration,
    )

    logger.info("On-demand objects prepared: scene and timelines cached")
    return context
