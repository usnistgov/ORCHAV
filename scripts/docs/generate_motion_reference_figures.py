#!/usr/bin/env python3
"""Regenerate the schema-v2 mobility and orientation reference figures.

This maintainer utility evaluates the same renderer-neutral pose-preparation
kernels used by generation and Scenario Builder preview. The two resource-backed
panels illustrate deterministic local inputs without making documentation
generation depend on external fixture files.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import TypeAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.core.scenario_actors import (  # noqa: E402
    PreparedMobility,
    PreparedOrientation,
    Timeline,
    derive_group_member_mobility,
    prepare_mobility,
    prepare_orientation,
)
from shared.scenarios.actors import OrientationSpec, StandaloneMobilitySpec  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "assets" / "generator" / "mobility_orientation"
DPI = 120
FIGSIZE = (5.6, 4.2)
PATH_COLOR = "#264653"
START_COLOR = "#1f77b4"
END_COLOR = "#d1495b"
ACCENT_COLOR = "#e76f51"
SECONDARY_COLOR = "#2a9d8f"
GRID_COLOR = "#8d99ae"

_MOBILITY_ADAPTER = TypeAdapter(StandaloneMobilitySpec)
_ORIENTATION_ADAPTER = TypeAdapter(OrientationSpec)


def _finish(fig: plt.Figure, name: str) -> None:
    """Save one exact 4:3 PNG without content-dependent cropping."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def _new_2d(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, color=GRID_COLOR, alpha=0.25)
    return fig, ax


def _new_3d(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=25, azim=-55)
    return fig, ax


def _positions(mobility: PreparedMobility) -> npt.NDArray[np.float64]:
    return np.asarray(mobility.positions_m, dtype=np.float64)


def _prepare(
    spec: Mapping[str, object],
    timeline: Timeline,
) -> PreparedMobility:
    validated = _MOBILITY_ADAPTER.validate_python(spec)
    return prepare_mobility(validated, timeline)


def _prepare_orientation(
    spec: Mapping[str, object],
    timeline: Timeline,
    mobility: PreparedMobility,
    *,
    references: Mapping[str, PreparedMobility] | None = None,
) -> PreparedOrientation:
    validated = _ORIENTATION_ADAPTER.validate_python(spec)
    return prepare_orientation(validated, timeline, mobility, references=references)


def _set_equal_2d(ax: plt.Axes, first_label: str = "X (m)", second_label: str = "Y (m)") -> None:
    ax.set_xlabel(first_label)
    ax.set_ylabel(second_label)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.14)


def _mark_start_end_2d(ax: plt.Axes, xy: npt.NDArray[np.float64]) -> None:
    ax.scatter(*xy[0], c=START_COLOR, s=42, marker="o", label="start", zorder=4)
    if not np.allclose(xy[0], xy[-1]):
        ax.scatter(*xy[-1], c=END_COLOR, s=46, marker="s", label="end", zorder=4)


def _mark_start_end_3d(ax: plt.Axes, xyz: npt.NDArray[np.float64]) -> None:
    ax.scatter(*xyz[0], c=START_COLOR, s=42, marker="o", label="start", depthshade=False)
    if not np.allclose(xyz[0], xyz[-1]):
        ax.scatter(*xyz[-1], c=END_COLOR, s=46, marker="s", label="end", depthshade=False)


def _plot_path_2d(
    name: str,
    title: str,
    positions: npt.NDArray[np.float64],
    *,
    axes: tuple[int, int] = (0, 1),
    labels: tuple[str, str] = ("X (m)", "Y (m)"),
    sample_markers: bool = False,
) -> None:
    xy = positions[:, axes]
    fig, ax = _new_2d(title)
    ax.plot(xy[:, 0], xy[:, 1], color=PATH_COLOR, linewidth=2.1, zorder=2)
    if sample_markers:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=np.arange(len(xy)),
            cmap="viridis",
            s=22,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    _mark_start_end_2d(ax, xy)
    _set_equal_2d(ax, *labels)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, name)


def _plot_path_3d(
    name: str,
    title: str,
    positions: npt.NDArray[np.float64],
    *,
    sample_markers: bool = False,
) -> None:
    fig, ax = _new_3d(title)
    ax.plot(*positions.T, color=PATH_COLOR, linewidth=2.1, zorder=2)
    if sample_markers:
        ax.scatter(
            *positions.T,
            c=np.arange(len(positions)),
            cmap="viridis",
            s=19,
            depthshade=False,
        )
    _mark_start_end_3d(ax, positions)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best", fontsize=8)
    _finish(fig, name)


def _plot_independent_samples(
    name: str,
    title: str,
    positions: npt.NDArray[np.float64],
) -> None:
    fig, ax = _new_3d(title)
    sample_numbers = np.arange(len(positions))
    points = ax.scatter(
        *positions.T,
        c=sample_numbers,
        cmap="viridis",
        s=28,
        depthshade=False,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    colorbar = fig.colorbar(points, ax=ax, shrink=0.64, pad=0.08)
    colorbar.set_label("sample")
    ax.text2D(0.04, 0.04, "independent observations", transform=ax.transAxes, fontsize=8)
    _finish(fig, name)


def _forward_axis(angles_deg: npt.ArrayLike) -> npt.NDArray[np.float64]:
    yaw, pitch, _roll = np.radians(np.asarray(angles_deg, dtype=np.float64))
    return np.asarray(
        (
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
        ),
        dtype=np.float64,
    )


def _plot_angle_series(
    name: str,
    title: str,
    timeline: Timeline,
    orientation: PreparedOrientation,
    *,
    stepped: bool = False,
) -> None:
    fig, ax = _new_2d(title)
    angles = np.asarray(orientation.euler_deg, dtype=np.float64)
    drawstyle = "steps-post" if stepped else "default"
    for index, (label, color) in enumerate(
        (("yaw", PATH_COLOR), ("pitch", ACCENT_COLOR), ("roll", SECONDARY_COLOR))
    ):
        ax.plot(
            timeline.timestamps_s,
            angles[:, index],
            linewidth=2,
            color=color,
            label=label,
            drawstyle=drawstyle,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.legend(loc="best", fontsize=8)
    _finish(fig, name)


def generate_mobility_figures() -> None:
    """Generate one current example for every schema-v2 mobility type."""

    standard = Timeline(steps=81, duration_s=8.0)

    stationary = _prepare(
        {"type": "stationary", "position_m": (0.0, 0.0, 1.5)},
        standard,
    )
    fig, ax = _new_2d("stationary — fixed position")
    position = _positions(stationary)[0, :2]
    ax.scatter(*position, c=START_COLOR, s=64, zorder=3)
    ax.annotate(
        "same position\nat every sample",
        xy=position,
        xytext=(18, 18),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    _set_equal_2d(ax)
    _finish(fig, "mobility_stationary")

    linear = _prepare(
        {
            "type": "linear",
            "start_m": (-20.0, -5.0, 1.5),
            "end_m": (20.0, 5.0, 1.5),
        },
        standard,
    )
    _plot_path_2d("mobility_linear", "linear — straight path", _positions(linear))

    waypoint = _prepare(
        {
            "type": "waypoint",
            "points_m": (
                (-20.0, -10.0, 1.5),
                (-7.0, 9.0, 1.5),
                (8.0, -5.0, 1.5),
                (20.0, 8.0, 1.5),
            ),
        },
        standard,
    )
    _plot_path_2d("mobility_waypoint", "waypoint — ordered points", _positions(waypoint))

    circular = _prepare(
        {
            "type": "circular",
            "center_m": (0.0, 0.0, 12.0),
            "radius_m": 8.0,
            "start_angle_deg": 90.0,
            "clockwise": True,
            "turns": 1.0,
        },
        standard,
    )
    _plot_path_2d("mobility_circular", "circular — horizontal orbit", _positions(circular))

    survey = _prepare(
        {
            "type": "survey",
            "origin_m": (-20.0, -10.0, 25.0),
            "width_m": 40.0,
            "height_m": 20.0,
            "row_spacing_m": 5.0,
            "heading_deg": 30.0,
        },
        Timeline(steps=161, duration_s=16.0),
    )
    _plot_path_2d("mobility_survey", "survey — lawnmower route", _positions(survey))

    grid_scan = _prepare(
        {
            "type": "grid_scan",
            "x_bounds_m": (-10.0, 10.0),
            "y_bounds_m": (-10.0, 10.0),
            "z_bounds_m": (2.0, 6.0),
            "x_steps": 3,
            "y_steps": 3,
            "z_steps": 2,
            "traversal_pattern": "snake",
        },
        Timeline(steps=181, duration_s=18.0),
    )
    _plot_path_3d(
        "mobility_grid_scan",
        "grid_scan — 3D snake traversal",
        _positions(grid_scan),
        sample_markers=True,
    )

    oscillating_timeline = Timeline(steps=121, duration_s=2.0)
    oscillating = _prepare(
        {
            "type": "oscillating",
            "center_m": (0.0, 0.0, 5.0),
            "axis": (0.0, 0.0, 1.0),
            "amplitude_m": 0.5,
            "frequency_hz": 1.0,
        },
        oscillating_timeline,
    )
    fig, ax = _new_2d("oscillating — sinusoidal displacement")
    ax.plot(
        oscillating_timeline.timestamps_s,
        _positions(oscillating)[:, 2],
        color=PATH_COLOR,
        linewidth=2.1,
    )
    ax.axhline(5.0, color=GRID_COLOR, linestyle="--", linewidth=1, label="center")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z position (m)")
    ax.legend(fontsize=8)
    _finish(fig, "mobility_oscillating")

    pendulum = _prepare(
        {
            "type": "pendulum",
            "pivot_m": (0.0, 0.0, 5.0),
            "length_m": 2.0,
            "max_angle_deg": 30.0,
            "frequency_hz": 0.5,
            "plane": "xz",
        },
        Timeline(steps=161, duration_s=4.0),
    )
    _plot_path_2d(
        "mobility_pendulum",
        "pendulum — bounded periodic arc",
        _positions(pendulum),
        axes=(0, 2),
        labels=("X (m)", "Z (m)"),
    )

    figure8 = _prepare(
        {
            "type": "figure8",
            "center_m": (0.0, 0.0, 8.0),
            "size_m": 20.0,
            "plane": "xy",
            "turns": 1.0,
        },
        Timeline(steps=161, duration_s=8.0),
    )
    _plot_path_2d("mobility_figure8", "figure8 — one horizontal loop", _positions(figure8))

    spiral = _prepare(
        {
            "type": "spiral",
            "center_m": (0.0, 0.0, 0.0),
            "radius_m": 10.0,
            "start_altitude_m": 5.0,
            "end_altitude_m": 25.0,
            "turns": 2.0,
        },
        Timeline(steps=161, duration_s=10.0),
    )
    _plot_path_3d("mobility_spiral", "spiral — helical climb", _positions(spiral))

    random_sampling = _prepare(
        {
            "type": "random_sampling",
            "x_bounds_m": (-15.0, 15.0),
            "y_bounds_m": (-10.0, 10.0),
            "z_bounds_m": (1.0, 8.0),
            "initial_position_m": (0.0, 0.0, 2.0),
            "seed": 7,
            "sampling": "uniform",
        },
        Timeline(steps=36, duration_s=3.5),
    )
    _plot_independent_samples(
        "mobility_random_sampling",
        "random_sampling — no connecting path",
        _positions(random_sampling),
    )

    sampled_positions = (
        (-10.0, -4.0, 1.5),
        (-7.0, -1.0, 1.7),
        (-3.0, 3.0, 1.9),
        (1.0, 5.0, 2.3),
        (5.0, 2.0, 2.0),
        (8.0, -2.0, 1.8),
        (11.0, -4.0, 1.5),
    )
    sampled = _prepare(
        {"type": "sampled", "positions_m": sampled_positions},
        Timeline(steps=len(sampled_positions), duration_s=6.0),
    )
    _plot_path_3d(
        "mobility_sampled",
        "sampled — exact YAML positions",
        _positions(sampled),
        sample_markers=True,
    )

    gauss_markov = _prepare(
        {
            "type": "gauss_markov",
            "initial_position_m": (-8.0, -5.0, 1.5),
            "x_bounds_m": (-15.0, 15.0),
            "y_bounds_m": (-10.0, 10.0),
            "z_bounds_m": (1.5, 1.5),
            "alpha": 0.82,
            "mean_speed_mps": 1.4,
            "mean_direction_deg": 25.0,
            "speed_std_mps": 0.5,
            "direction_std_deg": 28.0,
            "seed": 11,
        },
        Timeline(steps=181, duration_s=30.0),
    )
    _plot_path_2d(
        "mobility_gauss_markov",
        "gauss_markov — correlated XY motion",
        _positions(gauss_markov),
    )

    random_waypoint = _prepare(
        {
            "type": "random_waypoint",
            "initial_position_m": (0.0, 0.0, 1.5),
            "x_bounds_m": (-15.0, 15.0),
            "y_bounds_m": (-10.0, 10.0),
            "z_bounds_m": (1.5, 4.0),
            "speed_range_mps": (1.0, 2.0),
            "pause_range_s": (0.0, 1.0),
            "seed": 42,
        },
        Timeline(steps=241, duration_s=40.0),
    )
    _plot_path_3d(
        "mobility_random_waypoint",
        "random_waypoint — seeded destinations",
        _positions(random_waypoint),
    )

    manhattan = _prepare(
        {
            "type": "manhattan_grid",
            "origin_xy_m": (-15.0, -10.0),
            "block_size_m": 10.0,
            "grid_width": 4,
            "grid_height": 3,
            "altitude_m": 1.5,
            "turn_probability": 0.6,
            "speed_range_mps": (2.0, 3.0),
            "pause_range_s": (0.0, 0.5),
            "seed": 13,
        },
        Timeline(steps=241, duration_s=40.0),
    )
    _plot_path_2d(
        "mobility_manhattan_grid",
        "manhattan_grid — seeded street turns",
        _positions(manhattan),
    )

    network_nodes = {
        "road_start": (-20.0, -10.0),
        "west_junction": (-5.0, -10.0),
        "center": (-5.0, 5.0),
        "east_junction": (12.0, 5.0),
        "road_end": (12.0, 18.0),
        "side_road": (22.0, -8.0),
    }
    network_edges = (
        ("road_start", "west_junction"),
        ("west_junction", "center"),
        ("center", "east_junction"),
        ("east_junction", "road_end"),
        ("west_junction", "side_road"),
        ("side_road", "east_junction"),
    )
    route_nodes = (
        network_nodes["road_start"],
        network_nodes["west_junction"],
        network_nodes["center"],
        network_nodes["east_junction"],
        network_nodes["road_end"],
    )
    # A network_route loader resolves a cached graph to a local-meter polyline;
    # a waypoint evaluator then gives this panel the same path sampling behavior.
    network_route = _prepare(
        {
            "type": "waypoint",
            "points_m": tuple((x_coord, y_coord, 1.5) for x_coord, y_coord in route_nodes),
        },
        Timeline(steps=101, duration_s=10.0),
    )
    fig, ax = _new_2d("network_route — cached local graph")
    for start, end in network_edges:
        edge = np.asarray((network_nodes[start], network_nodes[end]))
        ax.plot(*edge.T, color=GRID_COLOR, linewidth=2.2, alpha=0.45, zorder=1)
    route_xy = _positions(network_route)[:, :2]
    ax.plot(*route_xy.T, color=PATH_COLOR, linewidth=3.0, label="selected route", zorder=2)
    _mark_start_end_2d(ax, route_xy)
    _set_equal_2d(ax)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, "mobility_network_route")

    mesh_samples = np.asarray(
        (
            (-8.0, -5.0, 1.0),
            (-5.0, -1.0, 1.5),
            (-1.0, 4.0, 2.5),
            (4.0, 6.0, 3.0),
            (8.0, 2.0, 2.2),
            (10.0, -3.0, 1.6),
        ),
        dtype=np.float64,
    )
    # mesh_sequence loads these positions from a local file before applying its
    # selected linear or step interpolation. This panel depicts the linear case.
    mesh_sequence = _prepare(
        {"type": "waypoint", "points_m": tuple(map(tuple, mesh_samples))},
        Timeline(steps=81, duration_s=8.0),
    )
    fig, ax = _new_3d("mesh_sequence — positions loaded from file")
    prepared_positions = _positions(mesh_sequence)
    ax.plot(*prepared_positions.T, color=PATH_COLOR, linewidth=2.1, label="prepared path")
    ax.scatter(
        *mesh_samples.T,
        color=ACCENT_COLOR,
        marker="D",
        s=34,
        label="file samples",
        depthshade=False,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best", fontsize=8)
    _finish(fig, "mobility_mesh_sequence")

    group_timeline = Timeline(steps=121, duration_s=12.0)
    group = _prepare(
        {
            "type": "waypoint",
            "points_m": (
                (-25.0, -8.0, 2.0),
                (-8.0, 7.0, 2.0),
                (8.0, 12.0, 2.0),
                (25.0, -4.0, 2.0),
            ),
            "interpolation": "catmull_rom",
        },
        group_timeline,
    )
    member = derive_group_member_mobility(
        group,
        (3.0, -5.0, 0.0),
        group_timeline,
    )
    fig, ax = _new_2d("group_member — heading-frame offset")
    group_xy = _positions(group)[:, :2]
    member_xy = _positions(member)[:, :2]
    ax.plot(*group_xy.T, color=PATH_COLOR, linewidth=2.2, label="group reference")
    ax.plot(*member_xy.T, color=ACCENT_COLOR, linewidth=2.2, label="group member")
    for index in (15, 45, 75, 105):
        ax.plot(
            (group_xy[index, 0], member_xy[index, 0]),
            (group_xy[index, 1], member_xy[index, 1]),
            color=GRID_COLOR,
            linestyle=":",
            linewidth=1,
        )
    _set_equal_2d(ax)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, "mobility_group_member")


def generate_orientation_figures() -> None:
    """Generate one current example for every schema-v2 orientation type."""

    fixed_timeline = Timeline(steps=1, duration_s=0.0)
    fixed_mobility = _prepare(
        {"type": "stationary", "position_m": (0.0, 0.0, 0.0)},
        fixed_timeline,
    )
    fixed = _prepare_orientation(
        {"type": "fixed", "yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        fixed_timeline,
        fixed_mobility,
    )
    fixed_forward = _forward_axis(fixed.euler_deg[0])[:2]
    fig, ax = _new_2d("fixed — constant heading")
    ax.scatter(0.0, 0.0, c=START_COLOR, s=55, zorder=3, label="actor")
    ax.arrow(
        0.0,
        0.0,
        3.0 * fixed_forward[0],
        3.0 * fixed_forward[1],
        width=0.045,
        head_width=0.32,
        length_includes_head=True,
        color=ACCENT_COLOR,
        zorder=2,
    )
    ax.text(-2.8, 0.35, "yaw 180°", fontsize=8)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-3.0, 3.0)
    _set_equal_2d(ax)
    ax.legend(fontsize=8)
    _finish(fig, "orientation_fixed")

    keyframe_timeline = Timeline(steps=121, duration_s=6.0)
    keyframe_mobility = _prepare(
        {"type": "stationary", "position_m": (0.0, 0.0, 0.0)},
        keyframe_timeline,
    )
    keyframes = _prepare_orientation(
        {
            "type": "keyframes",
            "keyframes": (
                {"time_s": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
                {"time_s": 3.0, "yaw_deg": 90.0, "pitch_deg": -15.0, "roll_deg": 0.0},
                {"time_s": 6.0, "yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            ),
        },
        keyframe_timeline,
        keyframe_mobility,
    )
    _plot_angle_series(
        "orientation_keyframes",
        "keyframes — quaternion SLERP",
        keyframe_timeline,
        keyframes,
    )

    align_timeline = Timeline(steps=101, duration_s=10.0)
    align_mobility = _prepare(
        {
            "type": "waypoint",
            "points_m": (
                (-20.0, -8.0, 1.5),
                (-7.0, 9.0, 1.5),
                (8.0, -5.0, 1.5),
                (20.0, 8.0, 1.5),
            ),
            "interpolation": "catmull_rom",
        },
        align_timeline,
    )
    align_motion = _prepare_orientation(
        {"type": "align_motion", "allow_pitch": False},
        align_timeline,
        align_mobility,
    )
    align_positions = _positions(align_mobility)[:, :2]
    fig, ax = _new_2d("align_motion — faces along velocity")
    ax.plot(*align_positions.T, color=PATH_COLOR, linewidth=2.2, label="actor path")
    align_angles = np.asarray(align_motion.euler_deg)
    for index in range(8, len(align_positions), 15):
        forward = _forward_axis(align_angles[index])[:2]
        ax.arrow(
            *align_positions[index],
            *(3.4 * forward),
            width=0.06,
            head_width=0.65,
            length_includes_head=True,
            color=ACCENT_COLOR,
            zorder=3,
        )
    _set_equal_2d(ax)
    ax.legend(fontsize=8)
    _finish(fig, "orientation_align_motion")

    look_timeline = Timeline(steps=81, duration_s=8.0)
    owner = _prepare(
        {"type": "stationary", "position_m": (0.0, 0.0, 2.0)},
        look_timeline,
    )
    target = _prepare(
        {
            "type": "linear",
            "start_m": (-10.0, -7.0, 5.0),
            "end_m": (10.0, 8.0, 10.0),
        },
        look_timeline,
    )
    look_at = _prepare_orientation(
        {"type": "look_at", "actor": "moving_target", "allow_pitch": True},
        look_timeline,
        owner,
        references={"moving_target": target},
    )
    fig, ax = _new_3d("look_at — tracks another actor")
    target_positions = _positions(target)
    owner_position = _positions(owner)[0]
    ax.plot(*target_positions.T, color=SECONDARY_COLOR, linewidth=2.2, label="target path")
    ax.scatter(*owner_position, color=START_COLOR, s=48, label="tracking actor", depthshade=False)
    look_angles = np.asarray(look_at.euler_deg)
    for index in (0, 20, 40, 60, 80):
        forward = _forward_axis(look_angles[index])
        ax.quiver(
            *owner_position,
            *(7.0 * forward),
            color=ACCENT_COLOR,
            linewidth=1.4,
            arrow_length_ratio=0.14,
            alpha=0.75,
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best", fontsize=8)
    _finish(fig, "orientation_look_at")

    spin_timeline = Timeline(steps=121, duration_s=6.0)
    spin_mobility = _prepare(
        {"type": "stationary", "position_m": (0.0, 0.0, 0.0)},
        spin_timeline,
    )
    spin = _prepare_orientation(
        {
            "type": "spin",
            "axis": "yaw",
            "rate_deg_s": 30.0,
            "pitch_deg": -5.0,
        },
        spin_timeline,
        spin_mobility,
    )
    _plot_angle_series(
        "orientation_spin",
        "spin — yaw at 30°/s",
        spin_timeline,
        spin,
    )

    random_timeline = Timeline(steps=61, duration_s=3.0)
    random_mobility = _prepare(
        {"type": "stationary", "position_m": (0.0, 0.0, 0.0)},
        random_timeline,
    )
    random = _prepare_orientation(
        {
            "type": "random",
            "seed": 7,
            "yaw_range_deg": (-45.0, 45.0),
            "pitch_range_deg": (-10.0, 10.0),
            "roll_range_deg": (0.0, 0.0),
            "update_interval_s": 0.5,
        },
        random_timeline,
        random_mobility,
    )
    _plot_angle_series(
        "orientation_random",
        "random — seeded 0.5 s holds",
        random_timeline,
        random,
        stepped=True,
    )


def main() -> None:
    generate_mobility_figures()
    generate_orientation_figures()
    print(f"Generated motion reference figures in {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
