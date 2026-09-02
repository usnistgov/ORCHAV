#!/usr/bin/env python3
"""Dispatch generator summary figures requested by scenario configuration.

This module is the scenario-facing routing layer. It reads
``scenario_context.generator_summary``, creates one cached ``ActorStateManager``
timeline, resolves output directories/options, and delegates actual Matplotlib
drawing to ``generator_summary_fig``.
"""

from pathlib import Path

from shared.geometry.cache import get_scene_geometry
from shared.logging import get_logger

from ...core.scenario_actors.state import ActorStateManager
from ..generator_summary_fig import (
    create_2d_scene_summary_figures,
    create_3d_scene_summary_figures,
    create_angular_velocity_summary_figures,
    create_orientation_summary_figures,
    create_speed_summary_figures,
)
from ..motion import collect_orientation_data_from_actor_state_manager

logger = get_logger(__name__)

GENERATOR_SUMMARY_PRODUCTS = frozenset(
    {"scene2d", "scene3d", "speed", "orientation", "angular_velocity"}
)


def maybe_generate_generator_summary(
    tx_configs: list,
    rx_configs: list,
    target_managers: list,
    simulation_config,
    scenario_context,
    path_policy=None,
    *,
    output_root: Path | None = None,
    strict: bool = False,
) -> list[Path] | None:
    """Generate requested scenario summary figures when enabled.

    Returns generated file paths. Direct callers retain best-effort behavior;
    the generator pipeline passes ``strict=True`` while building its private
    summary staging tree so a partial tree is never published.
    """
    gs = getattr(scenario_context, "generator_summary", None)
    if not gs or not gs.get("enabled", False):
        return None

    requested = set(gs.get("create", []) or [])
    requested_generator_products = requested & GENERATOR_SUMMARY_PRODUCTS
    if not requested_generator_products:
        logger.debug("Generator scene and motion summaries were not selected")
        return None

    out_spec = gs.get("output", {}) or {}
    logger.info("Generator summary requested: %s", sorted(requested_generator_products))
    logger.info(f"Output spec: {out_spec}")

    # One actor-state manager supplies consistent samples to every figure.
    try:
        actor_state_manager = ActorStateManager(
            tx_configs=tx_configs,
            rx_configs=rx_configs,
            target_managers=target_managers,
            steps=simulation_config.num_steps,
            duration=simulation_config.duration,
            motion_mode="cached",
        )
        logger.info(
            f"Created ActorStateManager with {len(tx_configs)} TX, {len(rx_configs)} RX, {len(target_managers)} targets"
        )
    except (ValueError, TypeError, RuntimeError, AttributeError) as e:
        logger.error(f"Failed to create ActorStateManager: {e}")
        if strict:
            raise RuntimeError("Generator summary actor-state preparation failed") from e
        return None
    out_dir = output_root if output_root is not None else out_spec.get("dir", None)
    out_fmt = (out_spec.get("format", "png") or "png").lower()
    out_dirs = (
        {
            "topology": output_root / "topology",
            "velocity": output_root / "velocity",
            "angular": output_root / "angular",
        }
        if output_root is not None
        else (out_spec.get("dirs", {}) or {})
    )

    generated: list[Path] = []
    failures: list[tuple[str, Exception]] = []

    def _figure_failed(label: str, exc: Exception) -> None:
        logger.warning("%s summary generation failed: %s", label, exc)
        if strict:
            failures.append((label, exc))

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    scene_geometry = None
    if requested & {"scene2d", "scene3d"}:
        try:
            scene_geometry = get_scene_geometry(scenario_context=scenario_context)
            if scene_geometry:
                logger.info(f"Loaded scene geometry with {len(scene_geometry)} meshes")
            else:
                logger.debug("Scene geometry not available (could be None or empty)")
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug(f"Could not load scene geometry: {e}")

    # Extract visualization options from scenario config (with defaults)
    viz_config = gs.get("visualization", {}) or {}
    scene2d_mode = viz_config.get("scene2d_mode", "rasterized")
    scene2d_resolution = viz_config.get("scene2d_resolution", 0.05)
    scene2d_material_legend = viz_config.get("scene2d_material_legend", False)
    actor_label_mode = viz_config.get("actor_label_mode", "role")
    scene3d_mode = viz_config.get("scene3d_mode", "floor_plan")
    scene3d_alpha = viz_config.get("scene3d_alpha", 0.3)
    scene3d_z_exaggeration = viz_config.get("scene3d_z_exaggeration", None)
    scene3d_camera = viz_config.get("scene3d_camera", None)
    scene3d_bounds = viz_config.get("scene3d_bounds", "union")
    scene3d_limits = viz_config.get("scene3d_limits", None)

    logger.info(f"[VIZ CONFIG] Raw viz_config dict: {viz_config}")
    logger.info(
        f"[VIZ CONFIG] Extracted: scene2d_mode={scene2d_mode!r}, scene2d_resolution={scene2d_resolution}, "
        f"scene2d_material_legend={scene2d_material_legend}, "
        f"actor_label_mode={actor_label_mode!r}, "
        f"scene3d_mode={scene3d_mode!r}, scene3d_alpha={scene3d_alpha}, "
        f"scene3d_z_exaggeration={scene3d_z_exaggeration!r}, scene3d_bounds={scene3d_bounds!r}"
    )

    # Scene 2D
    if "scene2d" in requested:
        logger.info("Generating scene2d visualization...")
        try:
            if out_dir:
                topo_dir = Path(out_dirs.get("topology", out_dir)) if out_dirs else Path(out_dir)
                topo_dir.mkdir(parents=True, exist_ok=True)
                dst = topo_dir / f"scene_2d.{out_fmt}"
                p = create_2d_scene_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    output_path=dst,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    scene_geometry=scene_geometry,
                    rendering_mode=scene2d_mode,
                    resolution=scene2d_resolution,
                    show_material_legend=scene2d_material_legend,
                    actor_label_mode=actor_label_mode,
                )
                generated.append(Path(p))
            else:
                p = create_2d_scene_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    scene_geometry=scene_geometry,
                    rendering_mode=scene2d_mode,
                    resolution=scene2d_resolution,
                    show_material_legend=scene2d_material_legend,
                    actor_label_mode=actor_label_mode,
                )
                generated.append(Path(p))
                logger.info(f"Scene2D visualization generated: {p}")
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
        ) as e:  # defensive catch: plot resilience
            _figure_failed("Scene2D", e)

    # Scene 3D
    if "scene3d" in requested:
        try:
            if out_dir:
                topo_dir = Path(out_dirs.get("topology", out_dir)) if out_dirs else Path(out_dir)
                topo_dir.mkdir(parents=True, exist_ok=True)
                dst = topo_dir / f"scene_3d.{out_fmt}"
                p = create_3d_scene_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    output_path=dst,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    scene_geometry=scene_geometry,
                    rendering_mode=scene3d_mode,
                    alpha=scene3d_alpha,
                    z_exaggeration=scene3d_z_exaggeration,
                    camera=scene3d_camera,
                    bounds_mode=scene3d_bounds,
                    explicit_bounds=scene3d_limits,
                    actor_label_mode=actor_label_mode,
                )
                generated.append(Path(p))
            else:
                p = create_3d_scene_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    scene_geometry=scene_geometry,
                    rendering_mode=scene3d_mode,
                    alpha=scene3d_alpha,
                    z_exaggeration=scene3d_z_exaggeration,
                    camera=scene3d_camera,
                    bounds_mode=scene3d_bounds,
                    explicit_bounds=scene3d_limits,
                    actor_label_mode=actor_label_mode,
                )
                generated.append(Path(p))
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
        ) as e:  # defensive catch: plot resilience
            _figure_failed("Scene3D", e)

    # Speed
    if "speed" in requested:
        try:
            if out_dir:
                vel_dir = Path(out_dirs.get("velocity", out_dir)) if out_dirs else Path(out_dir)
                vel_dir.mkdir(parents=True, exist_ok=True)
                dst = vel_dir / f"speed_evolution.{out_fmt}"
                p = create_speed_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    output_path=dst,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                )
                if p:
                    generated.append(Path(p))
            else:
                p = create_speed_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                )
                if p:
                    generated.append(Path(p))
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
        ) as e:  # defensive catch: plot resilience
            _figure_failed("Speed", e)

    # Orientation and angular velocity share the same raw orientation timeline.
    # Collect it once so both plots use identical finite differences.
    orientation_data_cache = None
    need_orientation = ("orientation" in requested) or ("angular_velocity" in requested)
    if need_orientation:
        try:
            orientation_data_cache = collect_orientation_data_from_actor_state_manager(
                actor_state_manager
            )
        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            _figure_failed("Orientation data collection", e)
            orientation_data_cache = None

    # Orientation
    if "orientation" in requested:
        try:
            # Pass a path in the summary directory to force extension and dir,
            # device-specific filenames will be applied inside.
            if out_dir:
                ang_dir = Path(out_dirs.get("angular", out_dir)) if out_dirs else Path(out_dir)
                ang_dir.mkdir(parents=True, exist_ok=True)
                stub = ang_dir / f"orientation_stub.{out_fmt}"
                p = create_orientation_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    output_path=stub,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    orientation_data=orientation_data_cache,
                )
                # Collect expected outputs in summary dir
                for name in [
                    "tx_orientation_evolution",
                    "rx_orientation_evolution",
                    "target_orientation_evolution",
                ]:
                    cand = ang_dir / f"{name}.{out_fmt}"
                    if cand.exists():
                        generated.append(cand)
            else:
                p = create_orientation_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    orientation_data=orientation_data_cache,
                )
                if p:
                    generated.append(Path(p))
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
        ) as e:  # defensive catch: plot resilience
            _figure_failed("Orientation", e)

    # Angular velocity
    if "angular_velocity" in requested:
        try:
            if out_dir:
                ang_dir = Path(out_dirs.get("angular", out_dir)) if out_dirs else Path(out_dir)
                ang_dir.mkdir(parents=True, exist_ok=True)
                stub = ang_dir / f"angular_velocity_stub.{out_fmt}"
                p = create_angular_velocity_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    output_path=stub,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    orientation_data=orientation_data_cache,
                )
                for name in [
                    "tx_angular_velocity_evolution",
                    "rx_angular_velocity_evolution",
                    "target_angular_velocity_evolution",
                ]:
                    cand = ang_dir / f"{name}.{out_fmt}"
                    if cand.exists():
                        generated.append(cand)
            else:
                p = create_angular_velocity_summary_figures(
                    tx_configs,
                    rx_configs,
                    target_managers,
                    simulation_config,
                    actor_state_manager,
                    path_policy=path_policy,
                    scenario_context=scenario_context,
                    orientation_data=orientation_data_cache,
                )
                if p:
                    generated.append(Path(p))
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
        ) as e:  # defensive catch: plot resilience
            _figure_failed("Angular velocity", e)

    logger.info(f"Summary generation completed. Generated {len(generated)} files: {generated}")
    if failures:
        labels = ", ".join(label for label, _exc in failures)
        raise RuntimeError(f"Generator summary failed for: {labels}") from failures[0][1]
    return generated
