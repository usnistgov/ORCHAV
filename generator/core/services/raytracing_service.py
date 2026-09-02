"""Ray-tracing service boundary between prepared state and propagation.

``RayTracingService`` does not load scenes and does not build actor-state caches.
It receives live Sionna objects from ``SceneService`` and prepared actor state
from ``ActorStateService``, binds them into ``SimulationObjects``, and delegates
per-frame scene mutation plus path solving to ``generator.core.propagation``.

The service also owns run-local solver objects and optional per-frame processors
that need access to live Sionna path methods before paths are frozen into CPU
arrays for saving and caching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sionna.rt import PathSolver

from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from ..materials.properties import collect_scene_radio_material_properties
from ..propagation import (
    apply_target_state_to_scene,
    compute_ray_tracing_step,
    freeze_frame_paths,
)
from ..propagation.snapshots import (
    orientations_to_array,
    positions_to_array,
    velocities_to_array,
)
from ..runtime import SimulationObjects
from ..scenario_actors.state import ActorStateCache
from .actor_state_service import ActorStateService
from .base import BaseService
from .scene_service import SceneService

_UNSUPPORTED_SENSING_GENERATION = (
    "Sensing generation is reserved for a future ORCHAV extension; this public "
    "release can read existing HDF5 frames with optional sensing extension "
    "payloads, but it does not generate them."
)


# Only these scenario-level quality keys are allowed to override the selected
# preset. Unknown keys stay out of the solver settings so the solver receives
# only supported configuration.
RAYTRACING_QUALITY_OVERRIDE_KEYS = frozenset(
    {
        "max_depth",
        "samples_per_src",
        "max_num_paths_per_src",
        "los",
        "specular_reflection",
        "diffuse_reflection",
        "refraction",
        "diffraction",
        "edge_diffraction",
        "diffraction_lit_region",
        "seed",
        "synthetic_array",
    }
)


class RayTracingService(BaseService):
    """Prepare propagation runtime and compute ray-tracing frames."""

    def __init__(self, simulation_config: SimulationConfig):
        super().__init__(simulation_config)
        self.path_solver = PathSolver()
        self.simulation_objects: SimulationObjects | None = None
        self.sensing_processor: Any | None = None

    def prepare_simulation(
        self,
        scene_service: SceneService,
        actor_state_service: ActorStateService,
        tx_configs: list[TransmitterConfig],
        rx_configs: list[ReceiverConfig],
        target_configs: list[Any],
        scenario_configuration: Any | None = None,
        motion_mode: str = "cached",
        actor_state_cache: ActorStateCache | None = None,
    ) -> None:
        """Prepare the ``SimulationObjects`` container used by propagation.

        This is the handoff point from orchestration into frame computation:
        scene objects are already live, actor state is already prepared, and
        quality settings are resolved into one run-local settings dict.
        """
        # ``get_quality_profile`` returns the selected preset as a mutable dict
        # for this run.  Scenario custom keys are applied to that run-local copy.
        settings = self.simulation_config.get_quality_profile()
        rt_cfg = getattr(scenario_configuration, "raytracing", {}) or {}
        rt_custom = (rt_cfg.get("quality", {}) or {}).get("custom", {}) or {}

        for k, v in rt_custom.items():
            if k in RAYTRACING_QUALITY_OVERRIDE_KEYS:
                settings[k] = v

        # SimulationObjects is the propagation-facing bundle.  It intentionally
        # carries both live Sionna scene objects and prepared actor-state arrays
        # so compute_ray_tracing_step can update scene state immediately before
        # invoking the solver.
        self.simulation_objects = SimulationObjects(
            scene=scene_service.scene,
            tx_list=scene_service.tx_list,
            rx_list=scene_service.rx_list,
            target_managers=scene_service.target_managers,
            actor_state_manager=actor_state_service.actor_state_manager,
            settings=settings,
            path_solver=self.path_solver,
            tx_configs=tx_configs,
            rx_configs=rx_configs,
            target_configs=target_configs,
            motion_mode=motion_mode,
            tx_positions_cache=actor_state_cache.tx_positions if actor_state_cache else None,
            rx_positions_cache=actor_state_cache.rx_positions if actor_state_cache else None,
            tgt_positions_cache=actor_state_cache.target_positions if actor_state_cache else None,
            tx_orientations_cache=(
                actor_state_cache.tx_orientations if actor_state_cache else None
            ),
            rx_orientations_cache=(
                actor_state_cache.rx_orientations if actor_state_cache else None
            ),
            tgt_orientations_cache=(
                actor_state_cache.target_orientations if actor_state_cache else None
            ),
            simulation_config=self.simulation_config,
            scenario_configuration=scenario_configuration,
            material_properties=collect_scene_radio_material_properties(scene_service.scene),
        )

        sensing_config = getattr(self.simulation_config, "sensing", None)
        sensing_enabled = (
            getattr(sensing_config, "enabled", False) is True if sensing_config else False
        )
        if sensing_enabled:
            raise RuntimeError(_UNSUPPORTED_SENSING_GENERATION)
        self.sensing_processor = None
        self.logger.debug(
            "Sensing processor not created (enabled=%s)",
            sensing_enabled,
        )

    def compute_step_cached(
        self, frame_idx: int, cached_frame_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Return cached paths with target state from the requested output step.

        Called on non-RT steps when ``cir_time_steps > 1``.  The paths,
        material mapping, and TX/RX snapshots remain tied to the acquisition
        step. Target objects and numeric target snapshots describe
        ``frame_idx`` so consumers can distinguish output-time scene state from
        the ray-tracing acquisition recorded in provenance.

        Args:
            frame_idx: Current simulation step index.
            cached_frame_data: Frame dict from the last real RT call.

        Returns:
            Shallow copy of *cached_frame_data* with current target state.
        """
        if self.simulation_objects is None:
            raise RuntimeError("Simulation objects are not initialized for cached computation")

        actor_state_manager = self.simulation_objects.actor_state_manager
        if actor_state_manager is None:
            raise RuntimeError("Actor-state manager is not initialized for cached computation")

        state_at_step = getattr(actor_state_manager, "state_at_step", None)
        if not callable(state_at_step):
            raise RuntimeError("Actor-state manager does not provide state_at_step")

        source_frame_idx = int(
            cached_frame_data.get(
                "_cached_rt_source_frame_idx",
                cached_frame_data.get("frame_idx", frame_idx),
            )
        )
        frame = dict(cached_frame_data)
        frame["frame_idx"] = frame_idx
        frame["_coherent_cached_frame"] = True
        frame["_cached_rt_source_frame_idx"] = source_frame_idx
        # Sensing data is attached later only when this output closes the
        # coherent interval; it is not a fresh result of the cached step.
        frame["sensing"] = None

        actor_state = state_at_step(frame_idx)
        target_positions = list(actor_state.target_positions or [])
        target_orientations = list(actor_state.target_orientations or [])

        target_velocities = None
        compute_velocities = getattr(actor_state_manager, "compute_velocities", None)
        if callable(compute_velocities):
            velocities = compute_velocities(frame_idx)
            target_velocities = velocities.target if velocities is not None else None

        apply_target_state_to_scene(
            self.simulation_objects.target_managers,
            target_positions,
            target_orientations,
            frame_idx,
            step_velocities=target_velocities,
            simulation_config=self.simulation_config,
        )

        frame["target_objects"] = [
            manager.target_object for manager in self.simulation_objects.target_managers
        ]
        frame["target_positions_snapshot"] = positions_to_array(target_positions)
        frame["target_orientations_snapshot"] = orientations_to_array(target_orientations)
        velocity_snapshot = velocities_to_array(target_velocities)
        if velocity_snapshot is None:
            frame.pop("target_velocities_snapshot", None)
        else:
            frame["target_velocities_snapshot"] = velocity_snapshot

        return frame

    def cleanup(self) -> None:
        """Release prepared simulation and optional processor state."""
        self.simulation_objects = None
        self.sensing_processor = None

    def compute_step(
        self,
        frame_idx: int,
        live_overrides: list[Any] | None = None,
        *,
        overrides: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Compute a single ray tracing frame.

        ``compute_ray_tracing_step`` applies the requested per-step actor state
        to the live Sionna objects, invokes the path solver, and returns the
        frame dictionary. This wrapper invokes an installed per-frame processor
        when present, then freezes live path buffers before downstream file or
        cache code sees them.

        When ``cir_time_steps > 1``, the method calls ``paths.cir()`` with
        ``num_time_steps`` and ``sampling_frequency`` to produce a coherent CIR
        stack for an installed processor that requires coherent samples.
        """
        if self.simulation_objects is None:
            self.logger.error("Simulation objects not initialized. Call prepare_simulation first.")
            return None

        if live_overrides is None and overrides is not None:
            live_overrides = overrides
        elif live_overrides is not None and overrides is not None:
            self.logger.warning("Both live_overrides and overrides supplied; using live_overrides")

        frame_data = compute_ray_tracing_step(
            self.simulation_objects,
            frame_idx,
            live_overrides=live_overrides,
        )
        if frame_data is None:
            return None

        # Optional processing must run before paths are frozen because it may
        # need backend methods such as paths.cir().
        if self.sensing_processor:
            try:
                paths = frame_data["paths"]
                cir_stack = self._compute_cir_stack(paths)

                sensing_result = self.sensing_processor.process(
                    self.simulation_objects,
                    frame_idx,
                    paths=paths,
                    cir_stack=cir_stack,
                )
                if sensing_result:
                    frame_data["sensing"] = sensing_result
            except Exception:
                self.logger.exception("Sensing computation failed for frame %d", frame_idx)
                return None

        # This is the CPU ownership boundary for path data. Sionna/Mitsuba path
        # buffers can be live backend allocations; after sensing has consumed
        # backend methods such as paths.cir(), downstream save/cache/summary code
        # should only see stable CPU arrays.
        freeze_frame_paths(frame_data)
        return frame_data

    def _compute_cir_stack(self, paths: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute coherent CIR stack when cir_time_steps > 1.

        This runs before ``freeze_frame_paths``.  Frozen paths are CPU
        data containers and intentionally do not expose Sionna's ``cir`` method.

        Returns:
            ``(a, tau)`` NumPy arrays if coherent mode is active, else ``None``.
        """
        n_steps = self.simulation_config.cir_time_steps
        if n_steps <= 1:
            return None

        fs = self.simulation_config.cir_sampling_frequency_hz
        if fs is None:
            self.logger.warning(
                "cir_time_steps=%d but cir_sampling_frequency_hz is None; "
                "falling back to per-frame mode",
                n_steps,
            )
            return None

        # FrozenPaths values come from cached/frozen frames, not live Sionna
        # path objects, so they cannot generate a new coherent CIR stack.
        if getattr(paths, "_is_frozen_paths", False):
            self.logger.debug("Paths are frozen CPU data; cannot compute multi-step CIR")
            return None

        try:
            a, tau = paths.cir(
                num_time_steps=n_steps,
                sampling_frequency=fs,
                normalize_delays=False,
            )
        except TypeError:
            # Some supported Sionna runtimes expose cir() without the coherent-
            # time keyword, so coherent sampling is a runtime capability.
            self.logger.warning(
                "paths.cir() does not accept num_time_steps; " "falling back to per-frame mode"
            )
            return None

        # Convert backend arrays before they cross into file/cache code.
        if hasattr(a, "numpy"):
            a = a.numpy()
        if hasattr(tau, "numpy"):
            tau = tau.numpy()

        a = np.asarray(a)
        tau = np.asarray(tau)

        # Some backends represent complex values as stacked real/imag channels.
        if a.dtype == np.float32 and a.shape[0] == 2:
            a = a[0] + 1j * a[1]

        self.logger.debug(
            "Computed coherent CIR stack: a.shape=%s, tau.shape=%s",
            a.shape,
            tau.shape,
        )
        return (a, tau)
