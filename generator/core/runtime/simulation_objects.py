"""Propagation-facing runtime container."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SimulationObjects:
    """All objects needed to compute or reuse propagation frames.

    This object is the handoff from services to propagation.  It contains live
    Sionna/Mitsuba scene objects, generator-side actor-state managers and
    caches, solver settings, and scenario context.  It is intentionally a data
    container: propagation modules own frame mutation/solving, while services
    own construction and cleanup.
    """

    # Live Sionna/Mitsuba objects that propagation mutates before each solve.
    scene: Any
    tx_list: List[Any]
    rx_list: List[Any]
    target_managers: List[Any]

    # Generator-side state and solver settings used to drive each frame.
    actor_state_manager: Any
    settings: Dict[str, Any]
    path_solver: Any

    # Original parsed configs are retained for metadata, rebuilding, and
    # streaming/on-demand requests that need names or YAML-derived policy.
    tx_configs: List[Any]
    rx_configs: List[Any]
    target_configs: List[Any]
    motion_mode: str

    # Prepared per-step arrays from ActorStateManager.prepare_cached().
    tx_positions_cache: Optional[List[Any]] = None
    rx_positions_cache: Optional[List[Any]] = None
    tgt_positions_cache: Optional[List[Any]] = None
    tx_orientations_cache: Optional[List[Any]] = None
    rx_orientations_cache: Optional[List[Any]] = None
    tgt_orientations_cache: Optional[List[Any]] = None
    simulation_config: Optional[Any] = None
    scenario_configuration: Optional[Any] = None
    material_properties: Optional[dict[str, Any]] = None
