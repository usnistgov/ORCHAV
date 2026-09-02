"""Default offline pipeline route that writes generated outputs to disk.

``perform_pipeline`` reaches this module whenever the scenario is not routed to
live gRPC streaming. In practice, that means the default ``output_mode='local'``
and any explicit file/offline mode. Streaming mode is handled separately in
``streaming.py``.

This route is not specific to one feature. It handles the normal file-backed
cases: ray-tracing frame generation, coverage-only output, optional material
tuning, and per-frame extension metadata when an installed feature supplies a
pipeline state for it.

Service boundaries used here:

* ``SceneService`` loads the Sionna scene and creates TX/RX/target objects.
* ``ActorStateService`` prepares the ``ActorStateManager`` per-step cache.
* ``CoverageService`` normalizes coverage config and writes coverage outputs.
* ``RayTracingService`` prepares ``SimulationObjects`` and computes frames.

The offline pipeline owns output ordering, HDF5 finalization, progress callbacks,
and fixed-cadence reuse of ray-tracing frames between acquisition steps.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from shared.frames.frame_set_writer import FrameSetWriter
from shared.logging import get_logger

from ...io.storage.coverage_publication import (
    CoveragePublication,
    CoveragePublicationError,
)
from ...io.storage.hdf5_frame_output import HDF5FrameOutputStrategy
from ...io.storage.summary_publication import SummaryPublication, SummaryPublicationError
from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from ..services.actor_state_service import ActorStateService
from ..services.coverage_service import CoverageService
from ..services.raytracing_service import RayTracingService
from ..services.scene_service import SceneService
from .context import PipelineContext
from .extensions import get_material_tuning_adapter
from .progress import ProgressInfo, StderrProgress, format_duration
from .schedule import FixedIntervalScheduler
from .sensing import SensingPipelineState

logger = get_logger(__name__)


def _write_topology_frame(
    output_manager: HDF5FrameOutputStrategy, scene_service: SceneService
) -> str:
    """Write one zero-path frame so visual tools can open non-RT outputs."""
    output_manager.save_frame_data(
        0,
        {
            "tx_list": scene_service.tx_list,
            "rx_list": scene_service.rx_list,
            "target_objects": scene_service.target_objects,
            "target_managers": scene_service.target_managers,
            "_coherent_cached_frame": True,
        },
    )
    return output_manager.finalize()


def _complete_summary_publication(
    publication: SummaryPublication,
    failure: Exception | None,
    *,
    frames_committed: bool,
) -> None:
    """Finalize staged summary output and distinguish failures after frame commit."""
    if failure is not None:
        publication.fail()
        if frames_committed:
            raise SummaryPublicationError(
                "Frames committed; summary failed. The prior summary was retained "
                "and will be retried."
            ) from failure
        raise SummaryPublicationError(
            "Summary generation failed; the prior summary was retained and will be retried."
        ) from failure

    try:
        publication.finalize()
    except Exception as exc:
        if frames_committed:
            raise SummaryPublicationError(
                "Frames committed; summary publication failed. The prior summary " "was retained."
            ) from exc
        raise


def _complete_coverage_publication(
    publication: CoveragePublication,
    output_manager: HDF5FrameOutputStrategy,
) -> Path | None:
    """Commit coverage only when this run published a frame manifest."""
    manifest = output_manager.published_manifest
    if manifest is None:
        publication.abort()
        return None
    try:
        return publication.finalize(manifest)
    except Exception as exc:
        raise CoveragePublicationError(
            "Frames committed; coverage publication failed. A mismatched prior "
            "coverage map will be ignored by the visualizer."
        ) from exc


def perform_offline_pipeline(
    tx_configs: list[TransmitterConfig],
    rx_configs: list[ReceiverConfig],
    target_configs: list,
    simulation_config: SimulationConfig,
    scenario_configuration=None,
    *,
    frame_set_writer: FrameSetWriter | None = None,
    on_step_complete: Callable[[ProgressInfo], None] | None = None,
    show_progress: bool = True,
) -> str | None:
    """Run the offline generator pipeline and finalize file-backed outputs."""

    logger.info("Starting offline pipeline orchestration")

    if scenario_configuration is not None:
        HDF5FrameOutputStrategy.require_canonical_scenario_frames(
            scenario_configuration,
            scenario_root=Path(scenario_configuration.root),
        )

    with PipelineContext(simulation_config) as ctx, ExitStack() as output_lifecycle:
        # Services are run-scoped. The pipeline coordinates their order, while
        # each service owns its own mutable state and cleanup.
        scene_service = ctx.get_service(SceneService)
        actor_state_service = ctx.get_service(ActorStateService)
        raytracing_service = ctx.get_service(RayTracingService)
        coverage_service = ctx.get_service(CoverageService)

        # Step normalization must happen before actor-state preparation because
        # some mobility providers require more output steps than the YAML value.
        actor_state_service.normalize_scene_steps(tx_configs, rx_configs, target_configs)

        rt_cfg = getattr(scenario_configuration, "raytracing", {}) or {}
        rt_enabled = bool(rt_cfg.get("enabled", False))

        if rt_cfg.get("geometry_only", False):
            # Geometry-only still needs a minimal path-solver call so downstream
            # frame/topology consumers see the scene entities in a normal frame.
            rt_enabled = True
            custom = rt_cfg.setdefault("quality", {}).setdefault("custom", {})
            custom.setdefault("samples_per_src", 1)
            custom.setdefault("max_depth", 1)
            custom.setdefault("max_num_paths_per_src", 1)
            custom.setdefault("los", True)
            custom.setdefault("specular_reflection", False)
            custom.setdefault("diffuse_reflection", False)
            custom.setdefault("refraction", False)
            custom.setdefault("diffraction", False)
            logger.info("Geometry-only mode: RT quality overridden to minimum")

        cov_cfg = getattr(scenario_configuration, "coverage_cfg", {}) or {}
        cov_enabled = bool(cov_cfg.get("enabled", False))
        canonical_frame_output = frame_set_writer is None
        if cov_enabled and not canonical_frame_output:
            raise ValueError(
                "Coverage publication is available only for canonical scenario frames; "
                "derived frame-set runs must disable coverage"
            )

        material_tuning_adapter = get_material_tuning_adapter()
        material_tuning_cfg = material_tuning_adapter.build_config(scenario_configuration)
        material_tuning_enabled = bool(getattr(material_tuning_cfg, "enabled", False))
        if cov_enabled and material_tuning_enabled:
            raise ValueError(
                "Coverage and material tuning cannot be enabled together: coverage "
                "must use the same finalized scene state as its committed frames"
            )

        if not cov_enabled and not rt_enabled and not material_tuning_enabled:
            logger.info("Neither coverage nor raytracing enabled; skipping computation.")

        output_manager: HDF5FrameOutputStrategy | None = None
        coverage_publication: CoveragePublication | None = None
        if cov_enabled or rt_enabled or material_tuning_enabled:
            # Validate the managed output destination before scene loading,
            # summary writes, or ray tracing. A bad path or unowned directory
            # should fail quickly and before this run creates any output.
            output_manager = HDF5FrameOutputStrategy(
                simulation_config,
                scenario_configuration,
                frame_set_writer=frame_set_writer,
            )
            output_lifecycle.callback(output_manager.abort)
            # Hold the destination lock for the complete expensive run. This
            # makes a concurrent generator or stale lock fail before scene
            # construction rather than after the first ray-tracing solve.
            output_manager.begin()
            if canonical_frame_output:
                coverage_publication = CoveragePublication(
                    scenario_configuration,
                    generation_id=output_manager.generation_id,
                )
                output_lifecycle.callback(coverage_publication.abort)

        summary_publication = SummaryPublication(scenario_configuration)
        output_lifecycle.callback(summary_publication.abort)
        summary_publication.begin()
        summary_failure: Exception | None = None

        scene_service.build_scene(tx_configs, rx_configs, target_configs)

        # Actor-state preparation depends on target managers created by the
        # scene service, especially for mesh-backed targets.
        _actor_state_manager, actor_state_cache = actor_state_service.prepare_actor_state(
            tx_configs, rx_configs, scene_service.target_managers, motion_mode="cached"
        )

        if summary_publication.active:
            try:
                from ...figures.summary import maybe_generate_generator_summary

                generated = maybe_generate_generator_summary(
                    tx_configs,
                    rx_configs,
                    scene_service.target_managers,
                    simulation_config,
                    scenario_context=scenario_configuration,
                    path_policy=None,
                    output_root=summary_publication.staging_directory,
                    strict=True,
                )
                if generated:
                    for path in generated:
                        logger.info("Staged summary figure: %s", path)
            except Exception as exc:
                summary_failure = exc
                summary_publication.fail()
                logger.error("Summary generation failed; continuing authoritative work: %s", exc)

        if (
            not rt_enabled
            and summary_publication.active
            and summary_publication.sensing_requested
            and summary_failure is None
        ):
            summary_failure = RuntimeError(
                "Sensing summary was requested, but ray tracing is disabled"
            )
            summary_publication.fail()
            logger.error("Sensing summary generation failed: %s", summary_failure)

        if not cov_enabled and not rt_enabled and not material_tuning_enabled:
            logger.info("Neither coverage nor raytracing enabled; summary generation completed.")
            _complete_summary_publication(
                summary_publication,
                summary_failure,
                frames_committed=False,
            )
            return None

        assert output_manager is not None
        logger.info("Output mode: %s", simulation_config.output_mode)

        output_file = coverage_service.compute_coverage(
            scene_service,
            scenario_configuration,
            publication=coverage_publication,
        )
        if cov_enabled and summary_publication.active and summary_failure is None:
            figure_cfg = (cov_cfg.get("save", {}) or {}).get("figure", {}) or {}
            if bool(figure_cfg.get("enabled", False)):
                if output_file is None:
                    summary_failure = RuntimeError(
                        "Coverage figures require coverage.save.data.enabled: true"
                    )
                    summary_publication.fail()
                else:
                    try:
                        coverage_figures = coverage_service.generate_summary_figures(
                            Path(output_file),
                            scenario_configuration,
                            summary_root=summary_publication.staging_directory,
                            strict=True,
                        )
                        for path in coverage_figures:
                            logger.info("Staged coverage summary figure: %s", path)
                    except Exception as exc:
                        summary_failure = exc
                        summary_publication.fail()
                        logger.error(
                            "Coverage summary generation failed; continuing authoritative work: %s",
                            exc,
                        )
        if output_file and not rt_enabled:
            # Coverage-only runs still write a lightweight topology frame so
            # tools can inspect device and target placement from the output set.
            assert coverage_publication is not None
            logger.info("Coverage map completed")
            frame_result = _write_topology_frame(output_manager, scene_service)
            published_coverage = _complete_coverage_publication(
                coverage_publication,
                output_manager,
            )
            logger.info("Coverage topology frame completed: %s", frame_result)
            _complete_summary_publication(
                summary_publication,
                summary_failure,
                frames_committed=True,
            )
            return str(published_coverage) if published_coverage is not None else output_file

        if cov_enabled and not rt_enabled and not material_tuning_enabled:
            _complete_summary_publication(
                summary_publication,
                summary_failure,
                frames_committed=False,
            )
            return None

        need_simulation = rt_enabled or material_tuning_enabled
        if need_simulation:
            # RayTracingService binds scene objects, actor-state manager, path
            # solver settings, and scenario context into SimulationObjects.
            raytracing_service.prepare_simulation(
                scene_service,
                actor_state_service,
                tx_configs,
                rx_configs,
                target_configs,
                scenario_configuration=scenario_configuration,
                motion_mode="cached",
                actor_state_cache=actor_state_cache,
            )

        if material_tuning_enabled:
            try:
                # Material tuning mutates scene material parameters. Rebuild the
                # scene afterwards so subsequent RT frames use the tuned values
                # through the normal SceneService/RayTracingService boundary.
                material_tuning_adapter.run(
                    raytracing_service.simulation_objects,
                    material_tuning_cfg,
                )
                logger.info("Reloading scene after material tuning...")
                scene_service.build_scene(tx_configs, rx_configs, target_configs)
                if need_simulation:
                    raytracing_service.prepare_simulation(
                        scene_service,
                        actor_state_service,
                        tx_configs,
                        rx_configs,
                        target_configs,
                        scenario_configuration=scenario_configuration,
                        motion_mode="cached",
                        actor_state_cache=actor_state_cache,
                    )
            except (ValueError, RuntimeError, OSError) as exc:
                logger.error("Material tuning failed: %s", exc)
                raise

        if material_tuning_enabled and not rt_enabled:
            logger.info("Material tuning completed (raytracing not enabled)")
            output_manager.finalize()
            if coverage_publication is not None:
                _complete_coverage_publication(coverage_publication, output_manager)
            _complete_summary_publication(
                summary_publication,
                summary_failure,
                frames_committed=True,
            )
            return None

        if rt_enabled:
            first_step = int(simulation_config.start_step)
            last_step = int(simulation_config.num_steps) - 1
            total_steps = last_step - first_step + 1
            if first_step > 0:
                logger.info(
                    "Computing fresh partial frame set: %d steps (%d through %d); "
                    "earlier frames are not copied or merged",
                    total_steps,
                    first_step,
                    last_step,
                )
            else:
                logger.info("Computing %d steps", total_steps)
            progress = StderrProgress(first_step=first_step, total_steps=total_steps)
            # ``start_step`` selects a suffix of the existing scenario timeline;
            # it does not stretch that suffix across the full duration. Preserve
            # the same per-step cadence as a complete run so sensing metadata and
            # derived velocities remain comparable.
            dt = (
                float(simulation_config.duration) / max(int(simulation_config.num_steps) - 1, 1)
                if simulation_config.duration and simulation_config.num_steps > 1
                else 1.0
            )

            cir_steps = simulation_config.cir_time_steps or 1
            sensing_state = SensingPipelineState.from_config(
                scenario_configuration,
                frame_dt_s=dt,
                gt_dt_s=dt * cir_steps,
                cir_steps=cir_steps,
                first_step=first_step,
            )
            last_rt_frame_data: dict[str, Any] | None = None

            if cir_steps > 1:
                n_rt_calls = math.ceil(total_steps / cir_steps)
                logger.info(
                    "RT gating enabled: ray-tracing every %d steps "
                    "(%d RT calls for %d output frames)",
                    cir_steps,
                    n_rt_calls,
                    total_steps,
                )

            step_scheduler = FixedIntervalScheduler(
                first_step=first_step,
                last_step=last_step,
                interval_length=cir_steps,
            )
            try:
                for i in range(first_step, last_step + 1):
                    step_start = time.time()
                    schedule_event = step_scheduler.event_for_step(i)
                    is_rt_step = schedule_event.is_acquisition_step

                    if is_rt_step:
                        # Acquisition steps perform a real path solve and cache
                        # the resulting frame for possible reuse below.
                        logger.debug("Step %d/%d (RT)", i - first_step + 1, total_steps)
                        frame_data = raytracing_service.compute_step(i)
                        if frame_data is None:
                            raise RuntimeError(f"Failed to compute frame {i}")
                        last_rt_frame_data = frame_data
                    else:
                        # Non-acquisition steps reuse the last solved paths but
                        # refresh target metadata so every output frame reflects
                        # the correct actor-state timeline step.
                        logger.debug("Step %d/%d (cached)", i - first_step + 1, total_steps)
                        if last_rt_frame_data is None:
                            raise RuntimeError(f"No cached RT frame is available for frame {i}")
                        frame_data = raytracing_service.compute_step_cached(i, last_rt_frame_data)

                    sensing_state.process_frame(
                        frame_data,
                        step_idx=i,
                        is_rt_step=is_rt_step,
                        schedule_event=schedule_event,
                    )

                    output_manager.save_frame_data(i, frame_data)
                    if show_progress:
                        progress.update(i)

                    if on_step_complete is not None:
                        now = time.time()
                        info = ProgressInfo(
                            step=i,
                            total_steps=total_steps,
                            elapsed_s=now - progress.start_time,
                            step_duration_s=now - step_start,
                        )
                        on_step_complete(info)
            finally:
                if show_progress:
                    progress.newline()

            output_result = output_manager.finalize()
            if coverage_publication is not None:
                _complete_coverage_publication(coverage_publication, output_manager)
            if summary_publication.active and summary_failure is None:
                try:
                    sensing_state.generate_summary(
                        scenario_configuration,
                        output_root=summary_publication.staging_directory,
                        strict=True,
                    )
                except Exception as exc:
                    summary_failure = exc
                    summary_publication.fail()
                    logger.error("Sensing summary generation failed after frame commit: %s", exc)
            _complete_summary_publication(
                summary_publication,
                summary_failure,
                frames_committed=True,
            )

            elapsed = time.time() - progress.start_time

            n_tx = len(tx_configs)
            n_rx = len(rx_configs)
            n_pairs = n_tx * n_rx
            if show_progress:
                try:
                    sys.stderr.write(
                        f"Done. {total_steps} steps, {n_tx} TX, {n_rx} RX"
                        f" ({n_pairs} pair{'s' if n_pairs != 1 else ''})"
                        f" in {format_duration(elapsed)}."
                        f" Output: {output_result}\n"
                    )
                    sys.stderr.flush()
                except OSError:
                    pass

            logger.info("Output result: %s", output_result)
            logger.info("Ray tracing completed")
            return output_result

    return None
