#!/usr/bin/env python3
"""Static generator summary figures for scene and motion diagnostics.

This module owns the concrete Matplotlib figure writers. Dispatch from scenario
configuration happens in ``generator.figures.summary.generator``; data-series
conversion happens in ``generator.figures.motion``; scene drawing is delegated
to ``generator.figures.scene``.
"""

import matplotlib

SUMMARY_FIGURE_DPI = 120

# Force a non-interactive backend to avoid GUI overhead during batch generation.
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

from matplotlib import font_manager, rcParams

# Pin fonts up-front so matplotlib skips expensive findfont scans on every figure.
rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 10,
        "figure.dpi": SUMMARY_FIGURE_DPI,
    }
)

# Warm the font cache once so subsequent plots don't trigger repeated scans.
try:
    font_manager.findfont("DejaVu Sans", fallback_to_default=True)
except (ValueError, OSError):
    pass

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from generator.core.scenario_actors.state import ActorStateManager
from shared.logging import get_logger

from .motion import (
    collect_orientation_data_from_actor_state_manager,
    collect_velocity_data_from_actor_state_manager,
    point_to_xy,
    point_to_xyz,
    prepare_actor_state_data,
)

logger = get_logger(__name__)
DeviceSeries = dict[str, Any]


def _format_step_summary(simulation_config) -> str:
    """Return compact step/duration text for summary figure titles."""
    num_steps = int(getattr(simulation_config, "num_steps", 0) or 0)
    step_label = "step" if num_steps == 1 else "steps"
    if num_steps <= 1:
        return f"{num_steps} {step_label}"
    return f"{num_steps} {step_label}, {simulation_config.duration}s duration"


def _format_scene_label(simulation_config, scenario_context=None) -> str:
    """Return a reader-friendly scene label for summary figure titles."""
    scene_label = getattr(simulation_config, "scene_name", None) or "scene"
    if scenario_context is not None:
        scene_label = getattr(scenario_context, "scene_id", None) or scene_label

    scene_path = Path(str(scene_label))
    if scene_path.suffix.lower() == ".xml":
        return scene_path.stem
    if scene_path.name and (scene_path.is_absolute() or "/" in str(scene_label)):
        return scene_path.name
    return str(scene_label)


def _format_info_box(simulation_config) -> str:
    """Return summary metadata without duration for single-step snapshots."""
    lines = [f"Steps: {simulation_config.num_steps}"]
    if int(getattr(simulation_config, "num_steps", 0) or 0) > 1:
        lines.append(f"Duration: {simulation_config.duration}s")
    lines.append(f"Quality: {simulation_config.quality}")
    return "\n".join(lines)


def _positions_are_stationary(positions, *, atol: float = 1e-9) -> bool:
    """Return true when all sampled positions collapse to one physical point."""
    if len(positions) <= 1:
        return True
    position_array = np.asarray(positions, dtype=float)
    return bool(np.allclose(position_array, position_array[0], rtol=0.0, atol=atol))


def _coerce_xyz_tuple(value: Any) -> tuple[float, float, float] | None:
    """Return a strict XYZ tuple for marker placement, or ``None`` if invalid."""
    try:
        x, y, z = point_to_xyz(value)
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    return (float(x), float(y), float(z))


def _scatter(ax: Any, *args: Any, **kwargs: Any) -> Any:
    """Call Matplotlib scatter without inheriting incomplete 3D stub types."""
    return getattr(ax, "scatter")(*args, **kwargs)


def _resolve_target_marker_position(target) -> Optional[Tuple[float, float, float]]:
    """Return a position to use for static target markers in summary plots."""
    get_current_position = getattr(target, "get_current_position", None)
    if callable(get_current_position):
        try:
            current_position = get_current_position()
        except (AttributeError, TypeError, ValueError):
            current_position = None
        coerced_current_position = _coerce_xyz_tuple(current_position)
        if coerced_current_position is not None:
            return coerced_current_position

    try:
        initial_position = getattr(target.config, "initial_position", None)
    except (AttributeError, TypeError, ValueError):
        initial_position = None

    if initial_position is None:
        return None
    return _coerce_xyz_tuple(initial_position)


def _annotate_marker_label(ax, label: str, xy: Tuple[float, float], role: str) -> None:
    """Place compact marker labels while keeping edge labels inside the axes."""
    preferred_offsets = {
        "tx": [(0, 12, "center", "bottom"), (8, 8, "left", "bottom"), (-8, 8, "right", "bottom")],
        "rx": [(0, -14, "center", "top"), (8, -10, "left", "top"), (-8, -10, "right", "top")],
        "target": [(6, 6, "left", "bottom"), (8, -10, "left", "top"), (-8, 8, "right", "bottom")],
    }
    candidates = list(preferred_offsets.get(role, [(6, 6, "left", "bottom")]))
    candidates.extend(
        [
            (14, 14, "left", "bottom"),
            (-14, 14, "right", "bottom"),
            (14, -14, "left", "top"),
            (-14, -14, "right", "top"),
            (0, 22, "center", "bottom"),
            (0, -24, "center", "top"),
        ]
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = max(abs(xlim[1] - xlim[0]), 1e-9)
    y_span = max(abs(ylim[1] - ylim[0]), 1e-9)
    x_rel = (xy[0] - xlim[0]) / x_span
    y_rel = (xy[1] - ylim[0]) / y_span
    edge_margin = 0.12

    if x_rel < edge_margin:
        candidates = [(8, dy, "left", va) for _, dy, _, va in candidates]
    elif x_rel > 1.0 - edge_margin:
        candidates = [(-8, dy, "right", va) for _, dy, _, va in candidates]

    if y_rel < edge_margin:
        candidates = [(dx, 12, ha, "bottom") for dx, _, ha, _ in candidates]
    elif y_rel > 1.0 - edge_margin:
        candidates = [(dx, -14, ha, "top") for dx, _, ha, _ in candidates]

    placed = getattr(ax, "_orchav_summary_label_boxes", [])
    dpi_scale = ax.figure.dpi / 72.0
    anchor = ax.transData.transform(xy)
    font_px = 10.0 * dpi_scale
    width_px = max(32.0, len(label) * font_px * 0.6)
    height_px = font_px * 1.6

    def _candidate_box(dx: float, dy: float, ha: str, va: str) -> Tuple[float, float, float, float]:
        cx = anchor[0] + dx * dpi_scale
        cy = anchor[1] + dy * dpi_scale
        if ha == "center":
            x0 = cx - width_px / 2.0
        elif ha == "right":
            x0 = cx - width_px
        else:
            x0 = cx
        if va == "center":
            y0 = cy - height_px / 2.0
        elif va == "top":
            y0 = cy - height_px
        else:
            y0 = cy
        return (x0, y0, x0 + width_px, y0 + height_px)

    def _overlaps(box: Tuple[float, float, float, float]) -> bool:
        pad = 4.0
        return any(
            not (
                box[2] + pad < other[0]
                or box[0] - pad > other[2]
                or box[3] + pad < other[1]
                or box[1] - pad > other[3]
            )
            for other in placed
        )

    dx, dy, ha, va = candidates[0]
    chosen_box = _candidate_box(dx, dy, ha, va)
    for candidate in candidates:
        candidate_box = _candidate_box(*candidate)
        if not _overlaps(candidate_box):
            dx, dy, ha, va = candidate
            chosen_box = candidate_box
            break

    ax.annotate(
        label,
        xy,
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
    )
    placed.append(chosen_box)
    ax._orchav_summary_label_boxes = placed


def _summary_actor_label(role: str, index: int, config: Any, label_mode: str = "role") -> str:
    """Return an actor label for summary layout figures."""
    normalized = str(label_mode or "role").strip().lower()
    if normalized in {"name", "actor_name", "actor-name", "names"}:
        name = getattr(config, "name", None)
        if name:
            return str(name)
    return f"{role.upper()}{index + 1}"


def _get_actor_positions_from_state_data(
    actor_state_data: dict[str, Any],
    actor_type: str,
    actor_index: int,
) -> list[Any]:
    """Return cached positions for one TX, RX, or target."""
    positions_by_type = {
        "tx": actor_state_data["tx_positions"],
        "rx": actor_state_data["rx_positions"],
        "target": actor_state_data["tgt_positions"],
    }
    positions = positions_by_type.get(actor_type, [])
    if actor_index < len(positions):
        return positions[actor_index]
    return []


def _set_nonnegative_series_ylim(ax, values: List[np.ndarray], *, min_top: float = 0.1) -> None:
    """Set compact y-limits for nonnegative series such as speed."""
    finite_chunks = []
    for value in values:
        arr = np.asarray(value, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_chunks.append(arr)
    if not finite_chunks:
        return
    all_values = np.concatenate(finite_chunks)
    top = max(float(np.max(all_values)) * 1.08, min_top)
    ax.set_ylim(0.0, top)


def _set_angle_series_ylim(ax, values: List[np.ndarray], *, min_span: float = 5.0) -> None:
    """Set readable y-limits for angle series without forcing a 360 deg range."""
    finite_chunks = []
    for value in values:
        arr = np.asarray(value, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_chunks.append(arr)
    if not finite_chunks:
        return
    all_values = np.concatenate(finite_chunks)
    low = float(np.min(all_values))
    high = float(np.max(all_values))
    span = high - low
    if span < min_span:
        center = (low + high) / 2.0
        span = min_span
        low = center - span / 2.0
        high = center + span / 2.0
    else:
        pad = max(span * 0.08, 1.0)
        low -= pad
        high += pad
    ax.set_ylim(low, high)


from shared.geometry.cache import (
    compute_xy_bounds_from_geometry,
    compute_xyz_bounds_from_geometry,
    get_scene_geometry,
)


def create_2d_scene_summary_figures(
    tx_configs: List,
    rx_configs: List,
    target_configs: List,
    simulation_config,
    actor_state_manager: ActorStateManager,
    output_path: Optional[Path] = None,
    path_policy=None,
    scenario_context=None,
    scene_geometry=None,
    rendering_mode: str = "rasterized",
    resolution: float = 0.05,
    show_material_legend: bool = False,
    actor_label_mode: str = "role",
) -> Path:
    """Create a top-down scene summary with trajectories and optional geometry.

    ``rendering_mode`` selects rasterized, vector, or auto scene drawing.
    Bounds use the union of scene geometry and sampled actor positions so
    stationary actors outside compact meshes remain visible.
    """

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=SUMMARY_FIGURE_DPI)

    # Set up the plot
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    scene_label = _format_scene_label(simulation_config, scenario_context)
    ax.set_title(f"Scene Layout: {scene_label}\n{_format_step_summary(simulation_config)}")

    # Determine plot bounds from the union of scene geometry and device positions.
    # Compact local meshes can otherwise clip static TX/RX markers outside the mesh.
    scene_bounds = None
    geom_to_plot = None
    # Try provided geometry first
    if scene_geometry:
        scene_bounds = compute_xy_bounds_from_geometry(scene_geometry)
        geom_to_plot = scene_geometry
    # If not provided, try cached loader from scenario
    if scene_bounds is None and scenario_context is not None:
        try:
            cached_geom = get_scene_geometry(scenario_context=scenario_context)
            if cached_geom:
                b = compute_xy_bounds_from_geometry(cached_geom)
                if b:
                    scene_bounds = b
                    # Use cached geometry for plotting as well
                    geom_to_plot = cached_geom
        except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as e:
            logger.debug(f"Geometry cache lookup failed: {e}")

    actor_state_data = prepare_actor_state_data(actor_state_manager)

    all_positions = actor_state_data["all_positions"]
    device_bounds = None
    if all_positions:
        positions = np.array(all_positions)
        device_bounds = (
            positions[:, 0].min(),
            positions[:, 0].max(),
            positions[:, 1].min(),
            positions[:, 1].max(),
        )

    # Unpack union bounds if available, otherwise use defaults
    if scene_bounds:
        if device_bounds:
            x_min = min(scene_bounds[0], device_bounds[0])
            x_max = max(scene_bounds[1], device_bounds[1])
            y_min = min(scene_bounds[2], device_bounds[2])
            y_max = max(scene_bounds[3], device_bounds[3])
        else:
            x_min, x_max, y_min, y_max = scene_bounds
        logger.info(
            f"Using scene/device bounds: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]"
        )
    elif device_bounds:
        x_min, x_max, y_min, y_max = device_bounds
        logger.info(
            f"Using device position bounds: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]"
        )
    else:
        logger.debug("No scene bounds available, falling back to device positions")
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        logger.info("Using default bounds: X[-10, 10], Y[-10, 10]")

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = max(x_span * 0.05, 2.0)
    y_pad = max(y_span * 0.05, 2.0)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot scene geometry if available (provided or cached)
    if geom_to_plot:
        from .scene import plot_scene_geometry_2d

        plot_scene_geometry_2d(
            ax,
            geom_to_plot,
            rendering_mode=rendering_mode,
            resolution=resolution,
            show_material_legend=show_material_legend,
        )

    legend_labels_seen = set()

    def _legend_label(label: str) -> str:
        if label in legend_labels_seen:
            return "_nolegend_"
        legend_labels_seen.add(label)
        return label

    # Plot TX actors using the same prepared state consumed by propagation.
    for i, tx in enumerate(tx_configs):
        positions_list = _get_actor_positions_from_state_data(actor_state_data, "tx", i)
        positions = [point_to_xy(p) for p in positions_list]
        if positions:
            start_pos = (positions[0][0], positions[0][1])
            is_stationary = _positions_are_stationary(positions)
            if is_stationary:
                ax.plot(
                    start_pos[0],
                    start_pos[1],
                    "ro",
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor="red",
                    markeredgewidth=2,
                    alpha=0.8,
                    zorder=4,
                    label=_legend_label("TX Stationary"),
                )
            else:
                ax.plot(
                    start_pos[0],
                    start_pos[1],
                    "r^",
                    markersize=12,
                    zorder=4,
                    label=_legend_label("TX Start"),
                )
            _annotate_marker_label(
                ax,
                _summary_actor_label("tx", i, tx, actor_label_mode),
                start_pos,
                "tx",
            )

            if not is_stationary:
                end_pos = (positions[-1][0], positions[-1][1])
                ax.plot(
                    end_pos[0],
                    end_pos[1],
                    "rv",
                    markersize=12,
                    zorder=4,
                    label=_legend_label("TX End"),
                )

                # Draw trajectory through all positions
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                markevery = max(1, len(positions) // 24)
                ax.plot(
                    x_coords,
                    y_coords,
                    "r-",
                    linewidth=2,
                    marker=".",
                    markersize=5,
                    markevery=markevery,
                    alpha=0.7,
                    zorder=2,
                    label=_legend_label("TX Trajectory"),
                )

                # Add arrow at the end to show direction
                if len(positions) >= 2:
                    ax.annotate(
                        "",
                        xy=end_pos[:2],
                        xytext=positions[-2][:2],
                        arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    )

    # Plot RX actors using the same prepared state consumed by propagation.
    for i, rx in enumerate(rx_configs):
        positions_list = _get_actor_positions_from_state_data(actor_state_data, "rx", i)
        positions = [point_to_xy(p) for p in positions_list]
        if positions:
            start_pos = (positions[0][0], positions[0][1])
            is_stationary = _positions_are_stationary(positions)
            if is_stationary:
                ax.plot(
                    start_pos[0],
                    start_pos[1],
                    "bo",
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor="blue",
                    markeredgewidth=2,
                    alpha=0.8,
                    zorder=4,
                    label=_legend_label("RX Stationary"),
                )
            else:
                ax.plot(
                    start_pos[0],
                    start_pos[1],
                    "b^",
                    markersize=12,
                    zorder=4,
                    label=_legend_label("RX Start"),
                )
            _annotate_marker_label(
                ax,
                _summary_actor_label("rx", i, rx, actor_label_mode),
                start_pos,
                "rx",
            )

            if not is_stationary:
                end_pos = (positions[-1][0], positions[-1][1])
                ax.plot(
                    end_pos[0],
                    end_pos[1],
                    "bv",
                    markersize=12,
                    zorder=4,
                    label=_legend_label("RX End"),
                )

                # Draw trajectory through all positions
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                markevery = max(1, len(positions) // 24)
                ax.plot(
                    x_coords,
                    y_coords,
                    "b-",
                    linewidth=2,
                    marker=".",
                    markersize=5,
                    markevery=markevery,
                    alpha=0.7,
                    zorder=2,
                    label=_legend_label("RX Trajectory"),
                )

                # Add arrow at the end to show direction
                if len(positions) >= 2:
                    ax.annotate(
                        "",
                        xy=end_pos[:2],
                        xytext=positions[-2][:2],
                        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                    )

    # Plot targets using the same prepared state consumed by propagation.
    for i, target in enumerate(target_configs):
        positions_list = _get_actor_positions_from_state_data(actor_state_data, "target", i)
        positions = [point_to_xy(p) for p in positions_list]
        if positions:
            start_pos = (positions[0][0], positions[0][1])
            is_stationary = _positions_are_stationary(positions)
            if is_stationary:
                ax.plot(
                    start_pos[0],
                    start_pos[1],
                    "go",
                    markersize=5,
                    markerfacecolor="none",
                    markeredgecolor="green",
                    markeredgewidth=2,
                    alpha=0.8,
                    zorder=4,
                    label=_legend_label("Target Stationary"),
                )
            else:
                ax.plot(
                    start_pos[0],
                    start_pos[1],
                    "g^",
                    markersize=10,
                    zorder=4,
                    label=_legend_label("Target Start"),
                )

            _annotate_marker_label(ax, f"{target.config.name}", start_pos, "target")

            if not is_stationary:
                end_pos = (positions[-1][0], positions[-1][1])
                ax.plot(
                    end_pos[0],
                    end_pos[1],
                    "gv",
                    markersize=10,
                    zorder=4,
                    label=_legend_label("Target End"),
                )

                # Draw trajectory through all positions
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                markevery = max(1, len(positions) // 24)
                ax.plot(
                    x_coords,
                    y_coords,
                    "g--",
                    linewidth=2,
                    marker=".",
                    markersize=4,
                    markevery=markevery,
                    alpha=0.7,
                    zorder=2,
                    label=_legend_label("Target Trajectory"),
                )

                # Add arrow at the end to show direction
                if len(positions) >= 2:
                    ax.annotate(
                        "",
                        xy=end_pos[:2],
                        xytext=positions[-2][:2],
                        arrowprops=dict(arrowstyle="->", color="green", lw=2, alpha=0.7),
                    )
        else:
            marker_position = _resolve_target_marker_position(target)
            if marker_position is None:
                continue
            start_pos = point_to_xy(marker_position)
            ax.plot(
                start_pos[0],
                start_pos[1],
                "go",
                markersize=5,
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=2,
                alpha=0.8,
                label=_legend_label("Target Stationary"),
            )
            _annotate_marker_label(ax, f"{target.config.name}", start_pos, "target")

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2, fontsize=8, framealpha=0.9)

    # Add info box
    info_text = _format_info_box(simulation_config)
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Save the plot
    if output_path is None:
        # Use scenario context to determine output directory
        if scenario_context and hasattr(scenario_context, "root"):
            output_path = scenario_context.root / "summary2DView.png"
        elif path_policy:
            output_dir = path_policy.project_root / "output" / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"scene_layout_{simulation_config.scene_name}.png"
        else:
            # Fallback to current directory
            output_path = Path(f"scene_layout_{simulation_config.scene_name}.png")

    # Ensure output directory exists
    assert output_path is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Scene visualization saved: {output_path}")
    return output_path


def create_3d_scene_summary_figures(
    tx_configs: List,
    rx_configs: List,
    target_configs: List,
    simulation_config,
    actor_state_manager: ActorStateManager,
    output_path: Optional[Path] = None,
    path_policy=None,
    scenario_context=None,
    scene_geometry=None,
    rendering_mode: str = "floor_plan",
    alpha: float = 0.3,
    z_exaggeration: Any = None,
    camera: Optional[Dict[str, float]] = None,
    bounds_mode: str = "union",
    explicit_bounds: Optional[Any] = None,
    actor_label_mode: str = "role",
) -> Path:
    """Create a 3D scene summary with trajectories and optional geometry.

    ``rendering_mode`` selects floor-plan, mesh, wireframe, hybrid, or
    city-scale drawing. Bounds can use scene geometry, device trajectories,
    their union, or explicit six-value limits.
    """

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10), dpi=SUMMARY_FIGURE_DPI)
    ax = fig.add_subplot(111, projection="3d")

    # Set viewing angle for better visualization of both ground and vertical structures.
    default_camera = (
        {"elev": 32.0, "azim": -55.0}
        if rendering_mode == "city"
        else {
            "elev": 25.0,
            "azim": 45.0,
        }
    )
    camera_cfg = {**default_camera, **(camera or {})}
    ax.view_init(elev=float(camera_cfg["elev"]), azim=float(camera_cfg["azim"]))

    # Set up the plot
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    ax.set_zlabel("Z Position (meters)")
    scene_label = _format_scene_label(simulation_config, scenario_context)
    ax.set_title(f"3D Scene Layout: {scene_label}\n{_format_step_summary(simulation_config)}")

    # Determine plot bounds by combining scene geometry and device positions
    geom_to_plot = None
    geom_bounds_xyz = None
    try:
        if scene_geometry:
            geom_to_plot = scene_geometry
        elif scenario_context is not None:
            cached_geom = get_scene_geometry(scenario_context=scenario_context)
            if cached_geom:
                geom_to_plot = cached_geom
        if geom_to_plot:
            b = compute_xyz_bounds_from_geometry(geom_to_plot)
            if b:
                geom_bounds_xyz = b
                logger.info(
                    f"Geometry bounds: X[{b[0]:.2f},{b[1]:.2f}] Y[{b[2]:.2f},{b[3]:.2f}] Z[{b[4]:.2f},{b[5]:.2f}]"
                )
    except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as e:
        logger.debug(f"3D geometry bounds failed: {e}")

    # Always also consider device positions for bounds (use all available positions)
    actor_state_data = prepare_actor_state_data(actor_state_manager)

    x_min_p = y_min_p = z_min_p = None
    x_max_p = y_max_p = z_max_p = None
    try:
        all_positions = actor_state_data["all_positions"]
        if all_positions:
            positions = np.array(all_positions)
            x_min_p, x_max_p = positions[:, 0].min(), positions[:, 0].max()
            y_min_p, y_max_p = positions[:, 1].min(), positions[:, 1].max()
            z_min_p, z_max_p = positions[:, 2].min(), positions[:, 2].max()
            logger.info(
                f"Device bounds: X[{x_min_p:.2f},{x_max_p:.2f}] Y[{y_min_p:.2f},{y_max_p:.2f}] Z[{z_min_p:.2f},{z_max_p:.2f}]"
            )
    except (ValueError, TypeError, IndexError, KeyError) as e:
        logger.debug(f"3D device bounds failed: {e}")

    # Merge bounds (union of geometry and positions) with padding
    def _merge_bounds(geom_b, pos_b):
        if geom_b is None and pos_b is None:
            return (-10, 10, -10, 10, 0, 5)
        if geom_b is None:
            return pos_b
        if pos_b is None:
            return geom_b
        return (
            min(geom_b[0], pos_b[0]),
            max(geom_b[1], pos_b[1]),
            min(geom_b[2], pos_b[2]),
            max(geom_b[3], pos_b[3]),
            min(geom_b[4], pos_b[4]),
            max(geom_b[5], pos_b[5]),
        )

    def _parse_explicit_bounds(bounds_spec):
        if bounds_spec is None:
            return None
        if isinstance(bounds_spec, (list, tuple)) and len(bounds_spec) == 6:
            return tuple(float(v) for v in bounds_spec)
        if isinstance(bounds_spec, dict):
            try:
                if all(key in bounds_spec for key in ("x", "y", "z")):
                    x0, x1 = bounds_spec["x"]
                    y0, y1 = bounds_spec["y"]
                    z0, z1 = bounds_spec["z"]
                    return (float(x0), float(x1), float(y0), float(y1), float(z0), float(z1))
                return (
                    float(bounds_spec["x_min"]),
                    float(bounds_spec["x_max"]),
                    float(bounds_spec["y_min"]),
                    float(bounds_spec["y_max"]),
                    float(bounds_spec["z_min"]),
                    float(bounds_spec["z_max"]),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Invalid scene3d explicit bounds ignored: %s", bounds_spec)
        return None

    def _select_bounds(geom_b, pos_b, mode, bounds_spec):
        explicit = _parse_explicit_bounds(bounds_spec)
        if explicit is not None:
            return explicit
        normalized = str(mode or "union").lower()
        if normalized in {"scene", "geometry"}:
            return geom_b or pos_b or (-10, 10, -10, 10, 0, 5)
        if normalized in {"actors", "actor", "trajectory", "trajectories"}:
            return pos_b or geom_b or (-10, 10, -10, 10, 0, 5)
        return _merge_bounds(geom_b, pos_b)

    pos_bounds_xyz = None
    if x_min_p is not None:
        assert x_max_p is not None
        assert y_min_p is not None
        assert y_max_p is not None
        assert z_min_p is not None
        assert z_max_p is not None
        # Apply small padding to device bounds
        pad_x = max((x_max_p - x_min_p) * 0.05, 1.0)
        pad_y = max((y_max_p - y_min_p) * 0.05, 1.0)
        pad_z = max((z_max_p - z_min_p) * 0.05, 0.5)
        pos_bounds_xyz = (
            x_min_p - pad_x,
            x_max_p + pad_x,
            y_min_p - pad_y,
            y_max_p + pad_y,
            z_min_p - pad_z,
            z_max_p + pad_z,
        )

    x_min, x_max, y_min, y_max, z_min, z_max = _select_bounds(
        geom_bounds_xyz, pos_bounds_xyz, bounds_mode, explicit_bounds
    )

    # Preserve physical XY scale while allowing optional z exaggeration for
    # city-scale scenes where true building heights would be visually tiny.
    def _resolve_z_scale(x_range, y_range, z_range):
        if z_range <= 0:
            return 1.0
        requested = z_exaggeration
        if requested is None:
            requested = "auto" if rendering_mode == "city" else 1.0
        if isinstance(requested, str) and requested.lower() == "auto":
            max_xy = max(x_range, y_range, 1e-6)
            target_visual_fraction = 0.24
            scale = (max_xy * target_visual_fraction) / z_range
            return float(np.clip(scale, 1.0, 10.0))
        try:
            return max(float(requested), 1.0)
        except (TypeError, ValueError):
            logger.warning("Invalid scene3d_z_exaggeration ignored: %s", requested)
            return 1.0

    def _set_natural_aspect_3d(ax, x_min, x_max, y_min, y_max, z_min, z_max):
        """Set aspect ratio that preserves relative dimensions with perspective."""
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        visual_z_scale = _resolve_z_scale(x_range, y_range, z_range)

        # Add small padding
        x_pad = x_range * 0.05
        y_pad = y_range * 0.05
        z_pad = z_range * 0.05

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_zlim(z_min - z_pad, z_max + z_pad)

        # Set box aspect to actual data proportions for realistic perspective
        # This preserves true scale ratios (e.g., 50m tall buildings in 850m wide scene)
        try:
            max_range = max(x_range, y_range, z_range)
            if max_range > 0:
                visual_z_range = min(z_range * visual_z_scale, max_range)
                aspect = [x_range / max_range, y_range / max_range, visual_z_range / max_range]
                ax.set_box_aspect(aspect)
        except (AttributeError, ValueError):
            pass
        return visual_z_scale

    # Plot scene geometry (scatter) if available
    if geom_to_plot:
        from .scene import plot_scene_geometry_3d

        plot_scene_geometry_3d(ax, geom_to_plot, rendering_mode=rendering_mode, alpha=alpha)

    effective_z_scale = _set_natural_aspect_3d(ax, x_min, x_max, y_min, y_max, z_min, z_max)
    if effective_z_scale > 1.01:
        ax.set_zlabel(f"Z Position (meters; visual x{effective_z_scale:.1f})")

    legend_labels_seen = set()

    def _legend_label(label: str) -> str:
        if label in legend_labels_seen:
            return "_nolegend_"
        legend_labels_seen.add(label)
        return label

    # Plot TX actors using the same prepared state consumed by propagation.
    for i, tx in enumerate(tx_configs):
        positions_list = _get_actor_positions_from_state_data(actor_state_data, "tx", i)
        positions = [point_to_xyz(p) for p in positions_list]
        if positions:
            start_pos = positions[0]
            is_stationary = _positions_are_stationary(positions)
            if is_stationary:
                _scatter(
                    ax,
                    start_pos[0],
                    start_pos[1],
                    start_pos[2],
                    s=50,
                    marker="o",
                    facecolors="none",
                    edgecolors="red",
                    linewidth=2,
                    alpha=0.8,
                    label=_legend_label("TX Stationary"),
                )
            else:
                _scatter(
                    ax,
                    start_pos[0],
                    start_pos[1],
                    start_pos[2],
                    c="red",
                    s=100,
                    marker="^",
                    label=_legend_label("TX Start"),
                )
            ax.text(
                start_pos[0],
                start_pos[1],
                start_pos[2],
                _summary_actor_label("tx", i, tx, actor_label_mode),
                fontsize=10,
                fontweight="bold",
            )

            if not is_stationary:
                end_pos = positions[-1]
                _scatter(
                    ax,
                    end_pos[0],
                    end_pos[1],
                    end_pos[2],
                    c="red",
                    s=100,
                    marker="v",
                    label=_legend_label("TX End"),
                )

                # Draw trajectory through all positions
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                z_coords = [pos[2] for pos in positions]
                markevery = max(1, len(positions) // 24)
                ax.plot(
                    x_coords,
                    y_coords,
                    z_coords,
                    "r-",
                    linewidth=2,
                    marker=".",
                    markersize=5,
                    markevery=markevery,
                    alpha=0.7,
                    label=_legend_label("TX Trajectory"),
                )

                # Add visible but unlabeled step samples along the trajectory.
                _scatter(
                    ax,
                    x_coords,
                    y_coords,
                    z_coords,
                    c="red",
                    s=12,
                    marker=".",
                    alpha=0.8,
                )

    # Plot RX actors using the same prepared state consumed by propagation.
    for i, rx in enumerate(rx_configs):
        positions_list = _get_actor_positions_from_state_data(actor_state_data, "rx", i)
        positions = [point_to_xyz(p) for p in positions_list]
        if positions:
            start_pos = positions[0]
            is_stationary = _positions_are_stationary(positions)
            if is_stationary:
                _scatter(
                    ax,
                    start_pos[0],
                    start_pos[1],
                    start_pos[2],
                    s=50,
                    marker="o",
                    facecolors="none",
                    edgecolors="blue",
                    linewidth=2,
                    alpha=0.8,
                    label=_legend_label("RX Stationary"),
                )
            else:
                _scatter(
                    ax,
                    start_pos[0],
                    start_pos[1],
                    start_pos[2],
                    c="blue",
                    s=100,
                    marker="^",
                    label=_legend_label("RX Start"),
                )
            ax.text(
                start_pos[0],
                start_pos[1],
                start_pos[2],
                _summary_actor_label("rx", i, rx, actor_label_mode),
                fontsize=10,
                fontweight="bold",
            )

            if not is_stationary:
                end_pos = positions[-1]
                _scatter(
                    ax,
                    end_pos[0],
                    end_pos[1],
                    end_pos[2],
                    c="blue",
                    s=100,
                    marker="v",
                    label=_legend_label("RX End"),
                )

                # Draw trajectory through all positions
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                z_coords = [pos[2] for pos in positions]
                markevery = max(1, len(positions) // 24)
                ax.plot(
                    x_coords,
                    y_coords,
                    z_coords,
                    "b-",
                    linewidth=2,
                    marker=".",
                    markersize=5,
                    markevery=markevery,
                    alpha=0.7,
                    label=_legend_label("RX Trajectory"),
                )

                # Add visible but unlabeled step samples along the trajectory.
                _scatter(
                    ax,
                    x_coords,
                    y_coords,
                    z_coords,
                    c="blue",
                    s=12,
                    marker=".",
                    alpha=0.8,
                )

    # Plot targets using the same prepared state consumed by propagation.
    for i, target in enumerate(target_configs):
        positions_list = _get_actor_positions_from_state_data(actor_state_data, "target", i)
        positions = [point_to_xyz(p) for p in positions_list]
        if positions:
            start_pos = positions[0]
            is_stationary = _positions_are_stationary(positions)
            if is_stationary:
                _scatter(
                    ax,
                    start_pos[0],
                    start_pos[1],
                    start_pos[2],
                    s=40,
                    marker="o",
                    facecolors="none",
                    edgecolors="green",
                    linewidth=2,
                    alpha=0.8,
                    label=_legend_label("Target Stationary"),
                )
            else:
                _scatter(
                    ax,
                    start_pos[0],
                    start_pos[1],
                    start_pos[2],
                    c="green",
                    s=80,
                    marker="^",
                    label=_legend_label("Target Start"),
                )

            # Add target label
            ax.text(
                start_pos[0],
                start_pos[1],
                start_pos[2],
                f"{target.config.name}",
                fontsize=9,
                fontweight="bold",
            )

            if not is_stationary:
                end_pos = positions[-1]
                _scatter(
                    ax,
                    end_pos[0],
                    end_pos[1],
                    end_pos[2],
                    c="green",
                    s=80,
                    marker="v",
                    label=_legend_label("Target End"),
                )

                # Draw trajectory through all positions
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                z_coords = [pos[2] for pos in positions]
                markevery = max(1, len(positions) // 24)
                ax.plot(
                    x_coords,
                    y_coords,
                    z_coords,
                    "g--",
                    linewidth=2,
                    marker=".",
                    markersize=4,
                    markevery=markevery,
                    alpha=0.7,
                    label=_legend_label("Target Trajectory"),
                )

                # Add visible but unlabeled step samples along the trajectory.
                _scatter(
                    ax,
                    x_coords,
                    y_coords,
                    z_coords,
                    c="green",
                    s=10,
                    marker=".",
                    alpha=0.8,
                )
        else:
            marker_position = _resolve_target_marker_position(target)
            if marker_position is None:
                continue
            start_pos = point_to_xyz(marker_position)
            _scatter(
                ax,
                start_pos[0],
                start_pos[1],
                start_pos[2],
                s=40,
                marker="o",
                facecolors="none",
                edgecolors="green",
                linewidth=2,
                alpha=0.8,
                label=_legend_label("Target Stationary"),
            )
            ax.text(
                start_pos[0],
                start_pos[1],
                start_pos[2],
                f"{target.config.name}",
                fontsize=9,
                fontweight="bold",
            )

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2, fontsize=8, framealpha=0.9)

    # Add info box
    info_text = _format_info_box(simulation_config)
    if effective_z_scale > 1.01:
        info_text = f"{info_text}\nVisual Z scale: x{effective_z_scale:.1f}"
    ax.text2D(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Save the plot
    if output_path is None:
        # Use scenario context to determine output directory
        if scenario_context and hasattr(scenario_context, "root"):
            output_path = scenario_context.root / "summary3DView.png"
        elif path_policy:
            output_dir = path_policy.project_root / "output" / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"scene_3d_{simulation_config.scene_name}.png"
        else:
            # Fallback to current directory
            output_path = Path(f"scene_3d_{simulation_config.scene_name}.png")

    # Ensure output directory exists
    assert output_path is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"3D Scene visualization saved: {output_path}")
    return output_path


def create_orientation_summary_figures(
    tx_configs: List,
    rx_configs: List,
    target_configs: List,
    simulation_config,
    actor_state_manager: ActorStateManager,
    output_path: Optional[Path] = None,
    path_policy=None,
    scenario_context=None,
    wrap_angles: bool = False,
    show_markers: bool = True,
    orientation_data: dict[str, list[DeviceSeries]] | None = None,
) -> Path | None:
    """Create role-split yaw, pitch, and roll evolution figures.

    One file is written for each nonempty TX, RX, or target role. Each file has
    a combined Euler-angle panel plus individual yaw, pitch, and roll panels.
    ``output_path`` supplies the destination directory and extension; the
    writer always assigns the role-specific filename. Precomputed
    ``orientation_data`` may be supplied so multiple summary products use the
    same prepared actor timeline.

    Args:
        tx_configs: List of TransmitterConfig objects
        rx_configs: List of ReceiverConfig objects
        target_configs: List of TargetConfig objects
        simulation_config: SceneConfig object with simulation parameters
        actor_state_manager: Prepared actor-state manager.
        output_path: Optional destination and format anchor for the role files
        path_policy: Path policy for determining output location
        scenario_context: Scenario context for determining output location
        wrap_angles: If True, wrap angles to [0, 360] degrees for display. If False, show natural/unwrapped angle range.
        show_markers: If True, draw point markers along lines (not added as separate legend entries).
        orientation_data: Optional pre-computed orientation data to avoid recomputation.

    Returns:
        First generated path in TX, RX, target order, or ``None`` when all
        roles are empty.
    """

    prepare_actor_state_data(actor_state_manager)

    # Collect orientation data (or reuse provided)
    if orientation_data is None:
        orientation_data = collect_orientation_data_from_actor_state_manager(actor_state_manager)

    # Create time axis
    time_steps = np.arange(simulation_config.num_steps)

    # Determine forced output directory/extension if provided
    force_dir = None
    force_ext = None
    if output_path is not None:
        p = Path(output_path)
        if p.suffix:
            force_dir = p.parent
            force_ext = p.suffix  # includes dot
        else:
            force_dir = p
            force_ext = ".png"

    # Generate separate figures for each device type
    generated_files: list[Path] = []

    # TX Orientation Figure
    if orientation_data["tx"]:
        tx_file = _create_device_orientation_figure(
            orientation_data["tx"],
            "TX",
            "Transmitters",
            "red",
            time_steps,
            simulation_config,
            path_policy,
            scenario_context,
            wrap_angles,
            show_markers,
            force_dir,
            force_ext,
        )
        generated_files.append(tx_file)

    # RX Orientation Figure
    if orientation_data["rx"]:
        rx_file = _create_device_orientation_figure(
            orientation_data["rx"],
            "RX",
            "Receivers",
            "blue",
            time_steps,
            simulation_config,
            path_policy,
            scenario_context,
            wrap_angles,
            show_markers,
            force_dir,
            force_ext,
        )
        generated_files.append(rx_file)

    # Target Orientation Figure
    if orientation_data["targets"]:
        target_file = _create_device_orientation_figure(
            orientation_data["targets"],
            "Target",
            "Targets",
            "green",
            time_steps,
            simulation_config,
            path_policy,
            scenario_context,
            wrap_angles,
            show_markers,
            force_dir,
            force_ext,
        )
        generated_files.append(target_file)

    # Return the first generated file as the main output
    return generated_files[0] if generated_files else None


def _create_device_orientation_figure(
    device_data: list[DeviceSeries],
    device_type: str,
    device_name: str,
    base_color: str,
    time_steps: np.ndarray,
    simulation_config,
    path_policy,
    scenario_context,
    wrap_angles: bool = False,
    show_markers: bool = True,
    force_dir: Optional[Path] = None,
    force_ext: Optional[str] = None,
) -> Path:
    """
    Create a single orientation figure for a specific device type.

    Args:
        device_data: List of device orientation data
        device_type: Device type ('TX', 'RX', 'Target')
        device_name: Display name for the device type
        base_color: Base color for the device type
        time_steps: Time step array
        simulation_config: Scene configuration
        path_policy: Path policy for output location
        scenario_context: Scenario context for output location
        wrap_angles: If True, wrap angles to [0, 360] degrees for display. If False, show natural/unwrapped angle range.
        show_markers: If True, draw point markers along lines.

    Returns:
        Path to the generated figure
    """

    # Create figure with 2 rows: main plot + 3 individual angle plots.
    fig = plt.figure(figsize=(18, 8.5), dpi=SUMMARY_FIGURE_DPI)
    scene_label = _format_scene_label(simulation_config, scenario_context)
    fig.suptitle(
        f"{device_name} Orientation Evolution: {scene_label}\n{_format_step_summary(simulation_config)}",
        fontsize=16,
        fontweight="bold",
    )

    # Main plot: All Euler angles together (top row, spans full width)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
    ax_main = fig.add_subplot(grid[0, :])

    # Individual angle plots (bottom row)
    ax_yaw = fig.add_subplot(grid[1, 0])
    ax_pitch = fig.add_subplot(grid[1, 1])
    ax_roll = fig.add_subplot(grid[1, 2])

    # Set up plot titles and axes
    ax_main.set_title(f"{device_name} Euler Angles Over Time", fontsize=14, fontweight="bold")
    ax_main.set_xlabel("Time Steps")
    ax_main.set_ylabel("Angle (degrees)")
    if wrap_angles:
        ax_main.set_ylim(0, 360)
    ax_main.grid(True, alpha=0.3)

    # No angular velocity plot in orientation figures

    # Individual angle plots
    ax_yaw.set_title("Yaw Over Time", fontsize=12, fontweight="bold")
    ax_yaw.set_xlabel("Time Steps")
    ax_yaw.set_ylabel("Yaw (degrees)")
    if wrap_angles:
        ax_yaw.set_ylim(0, 360)
    ax_yaw.grid(True, alpha=0.3)

    ax_pitch.set_title("Pitch Over Time", fontsize=12, fontweight="bold")
    ax_pitch.set_xlabel("Time Steps")
    ax_pitch.set_ylabel("Pitch (degrees)")
    if wrap_angles:
        ax_pitch.set_ylim(0, 360)
    ax_pitch.grid(True, alpha=0.3)

    ax_roll.set_title("Roll Over Time", fontsize=12, fontweight="bold")
    ax_roll.set_xlabel("Time Steps")
    ax_roll.set_ylabel("Roll (degrees)")
    if wrap_angles:
        ax_roll.set_ylim(0, 360)
    ax_roll.grid(True, alpha=0.3)

    # Define highly divergent color palette for multiple devices of same type
    if base_color == "red":
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    elif base_color == "blue":
        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    else:  # green
        colors = ["green", "red", "blue", "orange", "purple", "brown", "pink", "gray"]

    # Marker density: at most ~50 markers across timeline
    mark_step = max(1, int(len(time_steps) / 50))
    marker_kwargs = {"marker": "o", "markersize": 3, "markevery": mark_step} if show_markers else {}
    angle_series = {"main": [], "yaw": [], "pitch": [], "roll": []}

    # Plot data for each device
    logger.info(
        f"Processing {len(device_data)} {device_name} devices for orientation visualization"
    )
    for i, device_entry in enumerate(device_data):
        orientations = np.array(device_entry["orientations"])
        angular_velocities = np.array(device_entry["angular_velocities"])

        logger.info(
            f"Processing {device_type} {device_entry['name']}: orientations={len(orientations)}, angular_velocities={len(angular_velocities)}"
        )

        if len(orientations) > 0:
            # Apply angle wrapping if requested
            if wrap_angles:
                yaw_wrapped = np.mod(orientations[:, 0], 360)
                pitch_wrapped = np.mod(orientations[:, 1], 360)
                roll_wrapped = np.mod(orientations[:, 2], 360)
            else:
                yaw_wrapped = orientations[:, 0]
                pitch_wrapped = orientations[:, 1]
                roll_wrapped = orientations[:, 2]

            angle_series["main"].extend([yaw_wrapped, pitch_wrapped, roll_wrapped])
            angle_series["yaw"].append(yaw_wrapped)
            angle_series["pitch"].append(pitch_wrapped)
            angle_series["roll"].append(roll_wrapped)

            # Choose color for this device
            color = colors[i % len(colors)]

            # Main plot: All angles together
            ax_main.plot(
                time_steps,
                yaw_wrapped,
                color=color,
                linestyle="-",
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}: Yaw',
            )
            ax_main.plot(
                time_steps,
                pitch_wrapped,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}: Pitch',
            )
            ax_main.plot(
                time_steps,
                roll_wrapped,
                color=color,
                linestyle=":",
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}: Roll',
            )

            # Individual angle plots
            ax_yaw.plot(
                time_steps,
                yaw_wrapped,
                color=color,
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}',
            )
            ax_pitch.plot(
                time_steps,
                pitch_wrapped,
                color=color,
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}',
            )
            ax_roll.plot(
                time_steps,
                roll_wrapped,
                color=color,
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}',
            )

            # Overlay markers without extra legend entries
            if show_markers:
                ax_main.plot(
                    time_steps,
                    yaw_wrapped,
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_main.plot(
                    time_steps,
                    pitch_wrapped,
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_main.plot(
                    time_steps,
                    roll_wrapped,
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_yaw.plot(
                    time_steps,
                    yaw_wrapped,
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_pitch.plot(
                    time_steps,
                    pitch_wrapped,
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_roll.plot(
                    time_steps,
                    roll_wrapped,
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )

        # No angular velocity plotting in orientation figures

    if not wrap_angles:
        _set_angle_series_ylim(ax_main, angle_series["main"])
        _set_angle_series_ylim(ax_yaw, angle_series["yaw"])
        _set_angle_series_ylim(ax_pitch, angle_series["pitch"])
        _set_angle_series_ylim(ax_roll, angle_series["roll"])

    # Add legends outside or inside the plot area so they do not cover x-axis labels.
    ax_main.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8)
    ax_yaw.legend(loc="best", fontsize=8)
    ax_pitch.legend(loc="best", fontsize=8)
    ax_roll.legend(loc="best", fontsize=8)

    # Determine output path
    if force_dir is not None and force_ext is not None:
        output_path = Path(force_dir) / f"{device_type.lower()}_orientation_evolution{force_ext}"
    else:
        if scenario_context and hasattr(scenario_context, "root"):
            output_path = scenario_context.root / f"{device_type.lower()}_orientation_evolution.png"
        elif path_policy:
            output_dir = path_policy.project_root / "output" / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir
                / f"{device_type.lower()}_orientation_evolution_{simulation_config.scene_name}.png"
            )
        else:
            output_path = Path(
                f"{device_type.lower()}_orientation_evolution_{simulation_config.scene_name}.png"
            )

    # Ensure output directory exists
    assert output_path is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.06, right=0.84)
    plt.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"{device_name} orientation visualization saved: {output_path}")
    return output_path


def create_angular_velocity_summary_figures(
    tx_configs: List,
    rx_configs: List,
    target_configs: List,
    simulation_config,
    actor_state_manager: ActorStateManager,
    output_path: Optional[Path] = None,
    path_policy=None,
    scenario_context=None,
    show_markers: bool = True,
    orientation_data: dict[str, list[DeviceSeries]] | None = None,
) -> Path | None:
    """Create per-output-step Euler-angle figures for TX, RX, and targets.

    This function generates orientation-change figures showing:
    - Yaw, pitch, and roll differences between adjacent output steps
    - Individual component plots for detailed analysis
    - Separate figures for each device type (TX, RX, Targets)
    - Computed from orientation differences using ``ActorStateManager``

    These values are differences in degrees per output step. They are not
    time-scaled angular velocities.

    The function uses ``ActorStateManager`` for consistent actor-state handling
    and supports marker display options for enhanced visualization.

    Args:
        tx_configs: List of TransmitterConfig objects
        rx_configs: List of ReceiverConfig objects
        target_configs: List of TargetConfig objects
        simulation_config: SceneConfig object with simulation parameters
        actor_state_manager: Prepared actor-state manager.
        output_path: Optional output path for the image file
        path_policy: Path policy for determining output location
        scenario_context: Scenario context for determining output location
        show_markers: If True, draw point markers along lines for better visibility
        orientation_data: Optional pre-computed orientation data to avoid recomputation

    Returns:
        Path to the first generated visualization image
    """

    prepare_actor_state_data(actor_state_manager)

    # Collect orientation data (or reuse provided)
    if orientation_data is None:
        orientation_data = collect_orientation_data_from_actor_state_manager(actor_state_manager)

    # Create the output-step axis.
    time_steps = np.arange(simulation_config.num_steps)

    # Determine forced output directory/extension if provided
    force_dir = None
    force_ext = None
    if output_path is not None:
        p = Path(output_path)
        if p.suffix:
            force_dir = p.parent
            force_ext = p.suffix
        else:
            force_dir = p
            force_ext = ".png"

    # Generate separate figures for each device type
    generated_files: list[Path] = []

    # TX Euler-angle change figure.
    if orientation_data["tx"]:
        tx_file = _create_device_angular_velocity_figure(
            orientation_data["tx"],
            "TX",
            "Transmitters",
            "red",
            time_steps,
            simulation_config,
            path_policy,
            scenario_context,
            show_markers,
            force_dir,
            force_ext,
        )
        generated_files.append(tx_file)

    # RX Euler-angle change figure.
    if orientation_data["rx"]:
        rx_file = _create_device_angular_velocity_figure(
            orientation_data["rx"],
            "RX",
            "Receivers",
            "blue",
            time_steps,
            simulation_config,
            path_policy,
            scenario_context,
            show_markers,
            force_dir,
            force_ext,
        )
        generated_files.append(rx_file)

    # Target Euler-angle change figure.
    if orientation_data["targets"]:
        target_file = _create_device_angular_velocity_figure(
            orientation_data["targets"],
            "Target",
            "Targets",
            "green",
            time_steps,
            simulation_config,
            path_policy,
            scenario_context,
            show_markers,
            force_dir,
            force_ext,
        )
        generated_files.append(target_file)

    # Return the first generated file as the main output
    return generated_files[0] if generated_files else None


def _create_device_angular_velocity_figure(
    device_data: list[DeviceSeries],
    device_type: str,
    device_name: str,
    base_color: str,
    time_steps: np.ndarray,
    simulation_config,
    path_policy,
    scenario_context,
    show_markers: bool = True,
    force_dir: Optional[Path] = None,
    force_ext: Optional[str] = None,
) -> Path:
    """Create one per-output-step Euler-angle figure for a device type.

    Args:
        device_data: List of device orientation data
        device_type: Device type ('TX', 'RX', 'Target')
        device_name: Display name for the device type
        base_color: Base color for the device type
        time_steps: Time step array
        simulation_config: Scene configuration
        path_policy: Path policy for output location
        scenario_context: Scenario context for output location

    Returns:
        Path to the generated figure
    """

    # Create one combined plot and three individual Euler-component plots.
    fig = plt.figure(figsize=(18, 8.5), dpi=SUMMARY_FIGURE_DPI)
    scene_label = _format_scene_label(simulation_config, scenario_context)
    fig.suptitle(
        f"{device_name} Euler-Angle Change per Output Step: {scene_label}\n"
        f"{_format_step_summary(simulation_config)}",
        fontsize=16,
        fontweight="bold",
    )

    # Main plot: all Euler-angle changes together (top row, spans full width).
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
    ax_main = fig.add_subplot(grid[0, :])

    # Individual yaw, pitch, and roll plots (bottom row).
    ax_wx = fig.add_subplot(grid[1, 0])
    ax_wy = fig.add_subplot(grid[1, 1])
    ax_wz = fig.add_subplot(grid[1, 2])

    # Set up plot titles and axes
    ax_main.set_title(
        f"{device_name} Euler-Angle Change per Output Step",
        fontsize=14,
        fontweight="bold",
    )
    ax_main.set_xlabel("Output Step")
    ax_main.set_ylabel("Euler-Angle Change (deg/output step)")
    ax_main.grid(True, alpha=0.3)

    # Individual Euler-component plots.
    ax_wx.set_title("Yaw Change per Output Step", fontsize=12, fontweight="bold")
    ax_wx.set_xlabel("Output Step")
    ax_wx.set_ylabel("Yaw Change (deg/output step)")
    ax_wx.grid(True, alpha=0.3)

    ax_wy.set_title("Pitch Change per Output Step", fontsize=12, fontweight="bold")
    ax_wy.set_xlabel("Output Step")
    ax_wy.set_ylabel("Pitch Change (deg/output step)")
    ax_wy.grid(True, alpha=0.3)

    ax_wz.set_title("Roll Change per Output Step", fontsize=12, fontweight="bold")
    ax_wz.set_xlabel("Output Step")
    ax_wz.set_ylabel("Roll Change (deg/output step)")
    ax_wz.grid(True, alpha=0.3)

    # Define highly divergent color palette for multiple devices of same type
    if base_color == "red":
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    elif base_color == "blue":
        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    else:  # green
        colors = ["green", "red", "blue", "orange", "purple", "brown", "pink", "gray"]

    # Marker density: at most ~50 markers across timeline
    mark_step = max(1, int(len(time_steps) / 50))
    marker_kwargs = {"marker": "o", "markersize": 3, "markevery": mark_step} if show_markers else {}
    velocity_series = {"main": [], "wx": [], "wy": [], "wz": []}

    # Plot data for each device.
    logger.info(
        f"Processing {len(device_data)} {device_name} devices for Euler-angle change visualization"
    )
    for i, device_entry in enumerate(device_data):
        angular_velocities = np.array(device_entry["angular_velocities"])

        logger.info(
            f"Processing {device_type} {device_entry['name']}: angular_velocities={len(angular_velocities)}"
        )

        if len(angular_velocities) > 0:
            velocity_series["main"].extend(
                [
                    angular_velocities[:, 0],
                    angular_velocities[:, 1],
                    angular_velocities[:, 2],
                ]
            )
            velocity_series["wx"].append(angular_velocities[:, 0])
            velocity_series["wy"].append(angular_velocities[:, 1])
            velocity_series["wz"].append(angular_velocities[:, 2])

            # Choose color for this device
            color = colors[i % len(colors)]

            # Main plot: all Euler-angle changes together.
            ax_main.plot(
                time_steps,
                angular_velocities[:, 0],
                color=color,
                linestyle="-",
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}: yaw',
            )
            ax_main.plot(
                time_steps,
                angular_velocities[:, 1],
                color=color,
                linestyle="--",
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}: pitch',
            )
            ax_main.plot(
                time_steps,
                angular_velocities[:, 2],
                color=color,
                linestyle=":",
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}: roll',
            )

            # Individual Euler-component plots.
            ax_wx.plot(
                time_steps,
                angular_velocities[:, 0],
                color=color,
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}',
            )
            ax_wy.plot(
                time_steps,
                angular_velocities[:, 1],
                color=color,
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}',
            )
            ax_wz.plot(
                time_steps,
                angular_velocities[:, 2],
                color=color,
                linewidth=2,
                label=f'{device_type} {device_entry["name"]}',
            )

            if show_markers:
                ax_main.plot(
                    time_steps,
                    angular_velocities[:, 0],
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_main.plot(
                    time_steps,
                    angular_velocities[:, 1],
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_main.plot(
                    time_steps,
                    angular_velocities[:, 2],
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_wx.plot(
                    time_steps,
                    angular_velocities[:, 0],
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_wy.plot(
                    time_steps,
                    angular_velocities[:, 1],
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )
                ax_wz.plot(
                    time_steps,
                    angular_velocities[:, 2],
                    color=color,
                    linestyle="None",
                    label="_nolegend_",
                    **marker_kwargs,
                )

    _set_angle_series_ylim(ax_main, velocity_series["main"])
    _set_angle_series_ylim(ax_wx, velocity_series["wx"])
    _set_angle_series_ylim(ax_wy, velocity_series["wy"])
    _set_angle_series_ylim(ax_wz, velocity_series["wz"])

    # Add legends outside or inside the plot area so they do not cover x-axis labels.
    ax_main.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8)
    ax_wx.legend(loc="best", fontsize=8)
    ax_wy.legend(loc="best", fontsize=8)
    ax_wz.legend(loc="best", fontsize=8)

    # Determine output path
    if force_dir is not None and force_ext is not None:
        output_path = (
            Path(force_dir) / f"{device_type.lower()}_angular_velocity_evolution{force_ext}"
        )
    else:
        if scenario_context and hasattr(scenario_context, "root"):
            output_path = (
                scenario_context.root / f"{device_type.lower()}_angular_velocity_evolution.png"
            )
        elif path_policy:
            output_dir = path_policy.project_root / "output" / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir
                / f"{device_type.lower()}_angular_velocity_evolution_{simulation_config.scene_name}.png"
            )
        else:
            output_path = Path(
                f"{device_type.lower()}_angular_velocity_evolution_{simulation_config.scene_name}.png"
            )

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.06, right=0.84)
    plt.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"{device_name} Euler-angle change visualization saved: {output_path}")
    return output_path


def create_speed_summary_figures(
    tx_configs: List,
    rx_configs: List,
    target_configs: List,
    simulation_config,
    actor_state_manager: ActorStateManager,
    output_path: Optional[Path] = None,
    path_policy=None,
    scenario_context=None,
    show_markers: bool = True,
) -> Path:
    """
    Create speed visualization with a single figure showing all devices.

    This function generates a comprehensive speed analysis figure showing:
    - Speed evolution over time for all devices in a single plot
    - Separate subplots for TX, RX, and Target devices
    - Speed computed from actor trajectories prepared by ``ActorStateManager``
    - Color-coded and line-styled device differentiation

    The function uses ``ActorStateManager`` for consistent actor-state handling
    and provides both overview and detailed device-specific analysis.

    Args:
        tx_configs: List of TransmitterConfig objects
        rx_configs: List of ReceiverConfig objects
        target_configs: List of TargetConfig objects
        simulation_config: SceneConfig object with simulation parameters
        actor_state_manager: Prepared actor-state manager.
        output_path: Optional output path for the image file
        path_policy: Path policy for determining output location
        scenario_context: Scenario context for determining output location
        show_markers: If True, draw point markers along lines for better visibility

    Returns:
        Path to the generated visualization image
    """

    prepare_actor_state_data(actor_state_manager)

    velocity_data = collect_velocity_data_from_actor_state_manager(
        actor_state_manager, simulation_config
    )

    # Create time axis
    time_steps = np.arange(simulation_config.num_steps)

    # Create single figure with all devices
    return _create_combined_speed_figure(
        velocity_data,
        time_steps,
        simulation_config,
        path_policy,
        scenario_context,
        output_path=output_path,
        show_markers=show_markers,
    )


def _create_combined_speed_figure(
    velocity_data: dict,
    time_steps: np.ndarray,
    simulation_config,
    path_policy,
    scenario_context,
    output_path: Optional[Path] = None,
    show_markers: bool = True,
) -> Path:
    """
    Create a single speed figure with all devices.
    Top row: All devices together, Bottom row: TX only, RX only, Targets only.

    Args:
        velocity_data: Dictionary with 'tx', 'rx', 'targets' velocity data
        time_steps: Time step array
        simulation_config: Scene configuration
        path_policy: Path policy for output location
        scenario_context: Scenario context for output location

    Returns:
        Path to the generated figure
    """

    # Create figure with 2 rows: all devices + separate device type plots.
    fig = plt.figure(figsize=(18, 8.5), dpi=SUMMARY_FIGURE_DPI)
    scene_label = _format_scene_label(simulation_config, scenario_context)
    fig.suptitle(
        f"Speed Evolution: {scene_label}\n{_format_step_summary(simulation_config)}",
        fontsize=16,
        fontweight="bold",
    )

    # Main plot: All devices together (top row, spans full width)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
    ax_main = fig.add_subplot(grid[0, :])

    # Individual device type plots (bottom row)
    ax_tx = fig.add_subplot(grid[1, 0])
    ax_rx = fig.add_subplot(grid[1, 1])
    ax_targets = fig.add_subplot(grid[1, 2])

    # Set up plot titles and axes
    ax_main.set_title("All Devices Speed Over Time", fontsize=14, fontweight="bold")
    ax_main.set_xlabel("Time Steps")
    ax_main.set_ylabel("Speed (m/s)")
    ax_main.grid(True, alpha=0.3)

    # Individual device type plots
    ax_tx.set_title("TX Speed Over Time", fontsize=12, fontweight="bold")
    ax_tx.set_xlabel("Time Steps")
    ax_tx.set_ylabel("Speed (m/s)")
    ax_tx.grid(True, alpha=0.3)

    ax_rx.set_title("RX Speed Over Time", fontsize=12, fontweight="bold")
    ax_rx.set_xlabel("Time Steps")
    ax_rx.set_ylabel("Speed (m/s)")
    ax_rx.grid(True, alpha=0.3)

    ax_targets.set_title("Targets Speed Over Time", fontsize=12, fontweight="bold")
    ax_targets.set_xlabel("Time Steps")
    ax_targets.set_ylabel("Speed (m/s)")
    ax_targets.grid(True, alpha=0.3)

    # Define highly divergent colors for all devices (not grouped by type)
    all_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "cyan",
        "magenta",
        "lime",
        "navy",
    ]

    # Define line styles for device types
    tx_linestyle = "-"  # solid line
    rx_linestyle = "--"  # dashed line
    target_linestyle = ":"  # dotted line

    # Keep track of color index across all devices
    color_index = 0

    # Marker density: at most ~50 markers across timeline
    mark_step = max(1, int(len(time_steps) / 50))
    marker_kwargs = {"marker": "o", "markersize": 3, "markevery": mark_step} if show_markers else {}
    speed_series = {"main": [], "tx": [], "rx": [], "targets": []}

    # Plot TX devices
    if velocity_data["tx"]:
        for i, device_data in enumerate(velocity_data["tx"]):
            velocities = np.array(device_data["velocities"])
            if len(velocities) > 0:
                speed = np.sqrt(np.sum(velocities**2, axis=1))
                speed_series["main"].append(speed)
                speed_series["tx"].append(speed)
                color = all_colors[color_index % len(all_colors)]
                color_index += 1

                # Main plot
                ax_main.plot(
                    time_steps,
                    speed,
                    color=color,
                    linestyle=tx_linestyle,
                    linewidth=2,
                    label=f'TX {device_data["name"]}',
                )
                if show_markers:
                    ax_main.plot(
                        time_steps,
                        speed,
                        color=color,
                        linestyle="None",
                        label="_nolegend_",
                        **marker_kwargs,
                    )

                # TX plot
                ax_tx.plot(
                    time_steps,
                    speed,
                    color=color,
                    linestyle=tx_linestyle,
                    linewidth=2,
                    label=f'TX {device_data["name"]}',
                )
                if show_markers:
                    ax_tx.plot(
                        time_steps,
                        speed,
                        color=color,
                        linestyle="None",
                        label="_nolegend_",
                        **marker_kwargs,
                    )

    # Plot RX devices
    if velocity_data["rx"]:
        for i, device_data in enumerate(velocity_data["rx"]):
            velocities = np.array(device_data["velocities"])
            if len(velocities) > 0:
                speed = np.sqrt(np.sum(velocities**2, axis=1))
                speed_series["main"].append(speed)
                speed_series["rx"].append(speed)
                color = all_colors[color_index % len(all_colors)]
                color_index += 1

                # Main plot
                ax_main.plot(
                    time_steps,
                    speed,
                    color=color,
                    linestyle=rx_linestyle,
                    linewidth=2,
                    label=f'RX {device_data["name"]}',
                )
                if show_markers:
                    ax_main.plot(
                        time_steps,
                        speed,
                        color=color,
                        linestyle="None",
                        label="_nolegend_",
                        **marker_kwargs,
                    )

                # RX plot
                ax_rx.plot(
                    time_steps,
                    speed,
                    color=color,
                    linestyle=rx_linestyle,
                    linewidth=2,
                    label=f'RX {device_data["name"]}',
                )
                if show_markers:
                    ax_rx.plot(
                        time_steps,
                        speed,
                        color=color,
                        linestyle="None",
                        label="_nolegend_",
                        **marker_kwargs,
                    )

    # Plot Target devices
    if velocity_data["targets"]:
        for i, device_data in enumerate(velocity_data["targets"]):
            velocities = np.array(device_data["velocities"])
            if len(velocities) > 0:
                speed = np.sqrt(np.sum(velocities**2, axis=1))
                speed_series["main"].append(speed)
                speed_series["targets"].append(speed)
                color = all_colors[color_index % len(all_colors)]
                color_index += 1

                # Main plot
                ax_main.plot(
                    time_steps,
                    speed,
                    color=color,
                    linestyle=target_linestyle,
                    linewidth=2,
                    label=f'Target {device_data["name"]}',
                )
                if show_markers:
                    ax_main.plot(
                        time_steps,
                        speed,
                        color=color,
                        linestyle="None",
                        label="_nolegend_",
                        **marker_kwargs,
                    )

                # Targets plot
                ax_targets.plot(
                    time_steps,
                    speed,
                    color=color,
                    linestyle=target_linestyle,
                    linewidth=2,
                    label=f'Target {device_data["name"]}',
                )
                if show_markers:
                    ax_targets.plot(
                        time_steps,
                        speed,
                        color=color,
                        linestyle="None",
                        label="_nolegend_",
                        **marker_kwargs,
                    )

    def _maybe_add_legend(ax, **kwargs):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**kwargs)

    _set_nonnegative_series_ylim(ax_main, speed_series["main"])
    _set_nonnegative_series_ylim(ax_tx, speed_series["tx"])
    _set_nonnegative_series_ylim(ax_rx, speed_series["rx"])
    if speed_series["targets"]:
        _set_nonnegative_series_ylim(ax_targets, speed_series["targets"])
    else:
        ax_targets.text(
            0.5,
            0.5,
            "No targets",
            transform=ax_targets.transAxes,
            ha="center",
            va="center",
            color="0.35",
            fontsize=11,
        )
        ax_targets.set_xticks([])
        ax_targets.set_yticks([])

    # Add legends only when there is at least one labelled artist
    _maybe_add_legend(ax_main, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8)
    _maybe_add_legend(ax_tx, loc="best", fontsize=8)
    _maybe_add_legend(ax_rx, loc="best", fontsize=8)
    _maybe_add_legend(ax_targets, loc="best", fontsize=8)

    # Determine output path
    if output_path is None:
        if scenario_context and hasattr(scenario_context, "root"):
            output_path = scenario_context.root / "speed_evolution.png"
        elif path_policy:
            output_dir = path_policy.project_root / "output" / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"speed_evolution_{simulation_config.scene_name}.png"
        else:
            output_path = Path(f"speed_evolution_{simulation_config.scene_name}.png")

    # Ensure output directory exists
    assert output_path is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.06, right=0.84)
    plt.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Combined speed visualization saved: {output_path}")
    return output_path
