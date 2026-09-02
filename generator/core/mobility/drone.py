#!/usr/bin/env python3
"""Drone and aerial trajectory mobility patterns.

This module provides both a configurable multi-pattern ``DroneMobility`` and
named UAV trajectory primitives such as figure-8, spiral, and return-to-home.
All trajectories are sampled directly onto the scene timeline; they do not model
flight-control dynamics or obstacle avoidance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.logging import get_logger

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3, _numeric_quintuple

logger = get_logger(__name__)

try:
    from scipy.interpolate import CubicSpline

    SCIPY_AVAILABLE = True
except ImportError:
    CubicSpline = None
    SCIPY_AVAILABLE = False

# Fixed survey lattice density used before selecting/repeating samples for the
# requested scene step count.
SURVEY_GRID_AXIS_POINTS = 10


class DroneMobility(MobilityPattern):
    """
    Drone UAV mobility pattern for 3D flight trajectories.

    This class generates positions for drone movement with various flight patterns
    including waypoint navigation, survey patterns, orbital flight, and hovering.
    """

    yaml_type = "drone"

    def __init__(
        self,
        pattern: str = "waypoint",
        waypoints: list[Any] | None = None,
        survey_bounds: tuple[float, float, float, float, float] | None = None,
        orbital_center: Any | None = None,
        orbital_radius: float | None = None,
        hover_position: Any | None = None,
        target_speed_mps: float | None = None,
        start_pos: Any | None = None,
    ):
        """
        Initialize drone mobility.

        Args:
            pattern: Flight pattern type ('waypoint', 'survey', 'orbital', 'hover')
            waypoints: Sequence of 3D waypoints for waypoint pattern
            survey_bounds: (x_min, x_max, y_min, y_max, altitude) for survey pattern
            orbital_center: Center point for orbital pattern
            orbital_radius: Radius for orbital pattern
            hover_position: Fixed position for hover pattern
            target_speed_mps: Target speed in m/s (optional)
            start_pos: Starting position (optional)
        """
        super().__init__()

        if pattern not in ["waypoint", "survey", "orbital", "hover"]:
            raise ValueError(
                f"pattern must be one of: 'waypoint', 'survey', 'orbital', 'hover', got '{pattern}'"
            )

        self.pattern = pattern
        self.waypoints = waypoints
        self.survey_bounds = survey_bounds
        self.orbital_center = orbital_center
        self.orbital_radius = orbital_radius
        self.hover_position = hover_position
        self.target_speed_mps = target_speed_mps
        self.start_pos = start_pos

        if pattern == "waypoint" and not waypoints:
            raise ValueError("waypoints required for waypoint pattern")
        if pattern == "survey" and not survey_bounds:
            raise ValueError("survey_bounds required for survey pattern")
        if pattern == "orbital" and (not orbital_center or orbital_radius is None):
            raise ValueError("orbital_center and orbital_radius required for orbital pattern")
        if pattern == "hover" and not hover_position:
            raise ValueError("hover_position required for hover pattern")

        if start_pos is not None:
            self.start_pos = start_pos
        elif pattern == "waypoint" and waypoints:
            self.start_pos = waypoints[0]
        elif pattern == "orbital" and orbital_center:
            self.start_pos = orbital_center
        elif pattern == "hover" and hover_position:
            self.start_pos = hover_position
        else:
            self.start_pos = (0.0, 0.0, 10.0)

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "DroneMobility":
        """Build drone mobility from a YAML mobility block."""
        kwargs: dict[str, Any] = {"pattern": mobility_cfg.get("pattern", "waypoint")}
        if position is not None:
            kwargs["start_pos"] = position
        if mobility_cfg.get("speed_mps") is not None:
            kwargs["target_speed_mps"] = float(mobility_cfg["speed_mps"])
        if mobility_cfg.get("waypoints"):
            kwargs["waypoints"] = [point_to_tuple(wp) for wp in mobility_cfg["waypoints"]]
        if mobility_cfg.get("orbital_center"):
            kwargs["orbital_center"] = point_to_tuple(mobility_cfg["orbital_center"])
        if mobility_cfg.get("orbital_radius") is not None:
            kwargs["orbital_radius"] = float(mobility_cfg["orbital_radius"])
        if mobility_cfg.get("survey_bounds"):
            kwargs["survey_bounds"] = _numeric_quintuple(
                mobility_cfg["survey_bounds"], "survey_bounds"
            )
        if mobility_cfg.get("hover_position"):
            kwargs["hover_position"] = point_to_tuple(mobility_cfg["hover_position"])
        return cls(**kwargs)

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Get the speed for drone mobility."""
        if self.target_speed_mps is not None:
            return self.target_speed_mps
        return super().get_speed(start_pos, scene_steps, scene_duration)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate drone trajectory based on flight pattern."""
        if self.pattern == "waypoint":
            return self._generate_waypoint_trajectory(scene_steps, scene_duration)
        elif self.pattern == "survey":
            return self._generate_survey_trajectory(scene_steps, scene_duration)
        elif self.pattern == "orbital":
            return self._generate_orbital_trajectory(scene_steps, scene_duration)
        elif self.pattern == "hover":
            return self._generate_hover_trajectory(scene_steps, scene_duration)
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def _generate_waypoint_trajectory(
        self, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate smooth trajectory through waypoints with curved turns."""
        if not self.waypoints:
            raise ValueError("No waypoints provided for waypoint pattern")

        waypoint_arrays = [
            np.asarray(point_to_tuple(wp), dtype=np.float64) for wp in self.waypoints
        ]

        if len(waypoint_arrays) < 2:
            wp = waypoint_arrays[0]
            return [(float(wp[0]), float(wp[1]), float(wp[2]))] * scene_steps

        return self._generate_smooth_trajectory(waypoint_arrays, scene_steps)

    def _generate_smooth_trajectory(
        self, waypoints: list[np.ndarray], num_points: int
    ) -> list[Position3]:
        """Generate smooth trajectory using cubic spline interpolation."""

        if SCIPY_AVAILABLE and CubicSpline is not None:
            t_waypoints = np.linspace(0, 1, len(waypoints))
            t_smooth = np.linspace(0, 1, num_points)

            x_coords = np.array([wp[0] for wp in waypoints])
            y_coords = np.array([wp[1] for wp in waypoints])
            z_coords = np.array([wp[2] for wp in waypoints])

            cs_x = CubicSpline(t_waypoints, x_coords, bc_type="natural")
            cs_y = CubicSpline(t_waypoints, y_coords, bc_type="natural")
            cs_z = CubicSpline(t_waypoints, z_coords, bc_type="natural")

            x_smooth = cs_x(t_smooth)
            y_smooth = cs_y(t_smooth)
            z_smooth = cs_z(t_smooth)
        else:
            # SciPy is optional; linear interpolation preserves the waypoint
            # trajectory contract when cubic splines are unavailable.
            logger.warning("Using linear interpolation fallback - install scipy for smooth curves")
            positions = []
            for i in range(num_points):
                if num_points == 1:
                    t = 0.0
                else:
                    t = i / (num_points - 1)

                total_segments = len(waypoints) - 1
                segment_index = int(t * total_segments)
                segment_t = (t * total_segments) - segment_index

                if segment_index >= total_segments:
                    segment_index = total_segments - 1
                    segment_t = 1.0

                start_wp = waypoints[segment_index]
                end_wp = waypoints[segment_index + 1]

                pos = start_wp + segment_t * (end_wp - start_wp)
                positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

            return positions

        positions = []
        for i in range(num_points):
            positions.append((float(x_smooth[i]), float(y_smooth[i]), float(z_smooth[i])))

        return positions

    def _generate_survey_trajectory(
        self, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate lawnmower survey pattern."""
        if not self.survey_bounds:
            raise ValueError("No survey bounds provided for survey pattern")

        x_min, x_max, y_min, y_max, altitude = self.survey_bounds

        x_coords = np.linspace(x_min, x_max, SURVEY_GRID_AXIS_POINTS)
        y_coords = np.linspace(y_min, y_max, SURVEY_GRID_AXIS_POINTS)

        survey_points = []
        for i, y in enumerate(y_coords):
            if i % 2 == 0:
                for x in x_coords:
                    survey_points.append((x, y, altitude))
            else:
                for x in reversed(x_coords):
                    survey_points.append((x, y, altitude))

        if scene_steps <= len(survey_points):
            selected_points = survey_points[:scene_steps]
        else:
            selected_points = []
            for i in range(scene_steps):
                selected_points.append(survey_points[i % len(survey_points)])

        return [(float(point[0]), float(point[1]), float(point[2])) for point in selected_points]

    def _generate_orbital_trajectory(
        self, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate circular orbital pattern."""
        if not self.orbital_center or self.orbital_radius is None:
            raise ValueError("Orbital center and radius required for orbital pattern")

        center = np.asarray(point_to_tuple(self.orbital_center), dtype=np.float64)

        positions = []
        for i in range(scene_steps):
            if scene_steps == 1:
                t = 0.0
            else:
                t = i / (scene_steps - 1)

            angle = 2 * np.pi * t

            x = center[0] + self.orbital_radius * np.cos(angle)
            y = center[1] + self.orbital_radius * np.sin(angle)
            z = center[2]

            positions.append((float(x), float(y), float(z)))

        return positions

    def _generate_hover_trajectory(
        self, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate hover pattern (stationary at fixed position)."""
        if not self.hover_position:
            raise ValueError("Hover position required for hover pattern")

        pos = point_to_tuple(self.hover_position)

        return [pos] * scene_steps

    def get_flight_info(self) -> dict[str, Any]:
        """Get information about the drone flight configuration."""
        info = {
            "pattern": self.pattern,
            "target_speed_mps": self.target_speed_mps,
            "start_pos": self.start_pos,
        }

        if self.pattern == "waypoint":
            info["waypoints"] = self.waypoints
            info["num_waypoints"] = len(self.waypoints) if self.waypoints else 0
        elif self.pattern == "survey":
            info["survey_bounds"] = self.survey_bounds
        elif self.pattern == "orbital":
            info["orbital_center"] = self.orbital_center
            info["orbital_radius"] = self.orbital_radius
        elif self.pattern == "hover":
            info["hover_position"] = self.hover_position

        return info


class Figure8Mobility(MobilityPattern):
    """Lemniscate (figure-8) curve.  Classic UAV loitering pattern.

    The path follows a figure-8 (lemniscate of Bernoulli) centered at
    ``center`` with half-width ``size``.
    """

    yaml_type = "figure8"

    def __init__(
        self,
        center: Any = (0, 0, 20),
        size: float = 30.0,
        target_speed_mps: float | None = None,
    ):
        super().__init__()
        self.center = center
        self.size = float(size)
        self.target_speed_mps = target_speed_mps
        self.start_pos = center

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "Figure8Mobility":
        """Build figure-8 mobility from a YAML mobility block."""
        center = position or point_to_tuple(mobility_cfg.get("center", [0, 0, 20]))
        return cls(
            center=center,
            size=float(mobility_cfg.get("size", 30)),
            target_speed_mps=(
                float(mobility_cfg["speed_mps"]) if mobility_cfg.get("speed_mps") else None
            ),
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the configured figure-8 target speed, if provided."""
        if self.target_speed_mps is not None:
            return self.target_speed_mps
        return super().get_speed(start_pos, scene_steps, scene_duration)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate positions along the centered figure-8 loiter path."""
        cx, cy, cz = point_to_tuple(self.center)

        t = np.linspace(0, 2 * np.pi, scene_steps, endpoint=False)
        s = self.size
        # Lemniscate parametric form
        denom = 1 + np.sin(t) ** 2
        x = s * np.cos(t) / denom
        y = s * np.sin(t) * np.cos(t) / denom

        return [(cx + float(x[i]), cy + float(y[i]), cz) for i in range(scene_steps)]


class SpiralMobility(MobilityPattern):
    """Helical spiral — drone ascending or descending.

    Traces a helix from ``start_altitude`` to ``end_altitude`` with the
    given ``radius`` and number of ``turns``.
    """

    yaml_type = "spiral"

    def __init__(
        self,
        center: Any = (0, 0, 0),
        radius: float = 20.0,
        start_altitude: float = 5.0,
        end_altitude: float = 50.0,
        turns: float = 3.0,
        target_speed_mps: float | None = None,
    ):
        super().__init__()
        self.center = center
        self.radius = float(radius)
        self.start_altitude = float(start_altitude)
        self.end_altitude = float(end_altitude)
        self.turns = float(turns)
        self.target_speed_mps = target_speed_mps
        self.start_pos = center

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "SpiralMobility":
        """Build spiral mobility from a YAML mobility block."""
        center = position or point_to_tuple(mobility_cfg.get("center", [0, 0, 0]))
        return cls(
            center=center,
            radius=float(mobility_cfg.get("radius", 20)),
            start_altitude=float(mobility_cfg.get("start_altitude", 5)),
            end_altitude=float(mobility_cfg.get("end_altitude", 50)),
            turns=float(mobility_cfg.get("turns", 3)),
            target_speed_mps=(
                float(mobility_cfg["speed_mps"]) if mobility_cfg.get("speed_mps") else None
            ),
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the configured spiral target speed, if provided."""
        if self.target_speed_mps is not None:
            return self.target_speed_mps
        return super().get_speed(start_pos, scene_steps, scene_duration)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate positions along the helical spiral path."""
        cx, cy, _cz = point_to_tuple(self.center)

        t = np.linspace(0, 1, scene_steps)
        angles = 2 * np.pi * self.turns * t
        z = self.start_altitude + (self.end_altitude - self.start_altitude) * t
        x = cx + self.radius * np.cos(angles)
        y = cy + self.radius * np.sin(angles)

        return [(float(x[i]), float(y[i]), float(z[i])) for i in range(scene_steps)]


class ReturnToHomeMobility(MobilityPattern):
    """Waypoint mission with automatic return to start position.

    Travels through waypoints, then returns to the first waypoint in a
    straight line.  The return leg uses the last portion of the timeline.
    """

    yaml_type = "return_to_home"

    def __init__(
        self,
        waypoints: list[Any] | None = None,
        target_speed_mps: float | None = None,
        return_fraction: float = 0.2,
    ):
        super().__init__()
        self.waypoints = waypoints or []
        self.target_speed_mps = target_speed_mps
        self.return_fraction = float(return_fraction)
        if self.waypoints:
            self.start_pos = self.waypoints[0]

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "ReturnToHomeMobility":
        """Build return-to-home mobility from a YAML mobility block."""
        waypoints = mobility_cfg.get("waypoints")
        if not waypoints:
            raise ValueError(f"Device {entry['name']!r}: waypoints required for return_to_home")
        return cls(
            waypoints=[point_to_tuple(wp) for wp in waypoints],
            target_speed_mps=(
                float(mobility_cfg["speed_mps"]) if mobility_cfg.get("speed_mps") else None
            ),
            return_fraction=float(mobility_cfg.get("return_fraction", 0.2)),
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the configured mission speed, if provided."""
        if self.target_speed_mps is not None:
            return self.target_speed_mps
        return super().get_speed(start_pos, scene_steps, scene_duration)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate mission positions followed by a straight return leg."""
        if not self.waypoints or len(self.waypoints) < 2:
            wp = self.waypoints[0] if self.waypoints else start_pos
            pt = point_to_tuple(wp)
            return [pt] * scene_steps

        return_steps = max(1, int(scene_steps * self.return_fraction))
        mission_steps = scene_steps - return_steps

        wps = [np.asarray(point_to_tuple(wp), dtype=np.float64) for wp in self.waypoints]

        # Mission leg: linear interpolation through waypoints
        positions = []
        for i in range(mission_steps):
            t = i / max(mission_steps - 1, 1)
            seg_count = len(wps) - 1
            seg_idx = min(int(t * seg_count), seg_count - 1)
            seg_t = (t * seg_count) - seg_idx
            pos = wps[seg_idx] + seg_t * (wps[seg_idx + 1] - wps[seg_idx])
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

        # Return leg: straight line from last mission pos to home (first waypoint)
        last = wps[-1] if positions else wps[0]
        home = wps[0]
        for i in range(return_steps):
            t = (i + 1) / return_steps
            pos = last + t * (home - last)
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

        return positions
