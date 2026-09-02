"""Basic deterministic mobility primitives.

These are the patterns most scenarios should start from: fixed positions,
straight-line interpolation, circular motion, and waypoint interpolation. Each
class stores YAML control points and lets ``MobilityPattern.prepare()`` decide
the final per-step sample count.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3


class LinearMobility(MobilityPattern):
    """Linear movement between two points with optional speed control."""

    yaml_type = "linear"

    def __init__(
        self,
        start_pos: Any,
        end_pos: Any,
        target_speed_mps: float | None = None,
    ):
        """
        Initialize linear mobility.

        Args:
            start_pos: Starting position (x, y, z)
            end_pos: Ending position (x, y, z)
            target_speed_mps: Target speed in meters per second (optional)
        """
        super().__init__()
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.target_speed_mps = target_speed_mps

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "LinearMobility":
        """Build linear mobility from a YAML mobility block."""
        if position is None:
            raise ValueError(f"Device {entry['name']!r}: position required as start for linear")
        end_raw = mobility_cfg.get("end_position")
        if end_raw is None:
            raise ValueError(f"Device {entry['name']!r}: end_position required for linear")
        speed = mobility_cfg.get("speed_mps")
        return cls(
            start_pos=position,
            end_pos=point_to_tuple(end_raw),
            target_speed_mps=float(speed) if speed is not None else None,
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Get the speed for linear mobility."""
        if self.target_speed_mps is not None:
            return self.target_speed_mps
        return super().get_speed(start_pos, scene_steps, scene_duration)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate linear trajectory between start and end positions."""
        start_vec = np.asarray(point_to_tuple(self.start_pos), dtype=np.float64)
        end_vec = np.asarray(point_to_tuple(self.end_pos), dtype=np.float64)

        total_distance = np.linalg.norm(end_vec - start_vec)
        if scene_steps <= 1 or total_distance == 0.0:
            pos = (float(start_vec[0]), float(start_vec[1]), float(start_vec[2]))
            return [pos] * max(scene_steps, 1)

        t = np.linspace(0.0, 1.0, scene_steps, dtype=np.float64)

        if self.target_speed_mps is not None and scene_duration > 0 and self.target_speed_mps > 0:
            movement_time = total_distance / self.target_speed_mps
            if movement_time > scene_duration:
                # Respect a configured speed cap by ending partway along the
                # segment when the scene duration is too short.
                scale = scene_duration / movement_time
                t = np.clip(t * scale, 0.0, 1.0)

        delta = end_vec - start_vec
        path = start_vec + np.outer(t, delta)

        return [(float(p[0]), float(p[1]), float(p[2])) for p in path]


class CircularMobility(MobilityPattern):
    """Circular movement around a center point."""

    yaml_type = "circular"

    def __init__(
        self,
        center: Any,
        radius: float,
        start_angle: float = 0,
        clockwise: bool = True,
    ):
        """
        Initialize circular mobility.

        Args:
            center: Center point of the circle (x, y, z)
            radius: Radius of the circle in meters
            start_angle: Starting angle in radians (default: 0)
            clockwise: If True, rotate clockwise; if False, counterclockwise
        """
        super().__init__()
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.clockwise = clockwise

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "CircularMobility":
        """Build circular mobility from a YAML mobility block."""
        center_raw = mobility_cfg.get("center")
        if center_raw is None:
            raise ValueError(f"Device {entry['name']!r}: center required for circular")
        radius = mobility_cfg.get("radius")
        if radius is None:
            raise ValueError(f"Device {entry['name']!r}: radius required for circular")
        return cls(
            center=point_to_tuple(center_raw),
            radius=float(radius),
            start_angle=float(mobility_cfg.get("start_angle", 0)),
            clockwise=bool(mobility_cfg.get("clockwise", True)),
        )

    @property
    def start_pos(self) -> Position3:
        """Return the circle's initial point for runtime config normalization."""
        cx, cy, cz = point_to_tuple(self.center)
        x = cx + self.radius * np.cos(self.start_angle)
        y = cy + self.radius * np.sin(self.start_angle)
        z = cz
        return (float(x), float(y), float(z))

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Calculate tangential speed for circular motion."""
        if scene_duration > 0:
            angular_velocity = (2 * np.pi) / scene_duration
            return float(self.radius * angular_velocity)
        return 0.0

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate circular trajectory around center point."""
        center_vec = np.asarray(point_to_tuple(self.center), dtype=np.float64)

        if scene_steps <= 1 or scene_duration <= 0:
            x = center_vec[0] + self.radius * np.cos(self.start_angle)
            y = center_vec[1] + self.radius * np.sin(self.start_angle)
            z = center_vec[2]
            return [(float(x), float(y), float(z))] * max(scene_steps, 1)

        angular_velocity = (2 * np.pi) / scene_duration
        if not self.clockwise:
            angular_velocity = -angular_velocity

        times = np.linspace(0.0, scene_duration, scene_steps, dtype=np.float64)
        angles = self.start_angle + angular_velocity * times

        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        xs = center_vec[0] + self.radius * cos_vals
        ys = center_vec[1] + self.radius * sin_vals
        zs = np.full(scene_steps, center_vec[2], dtype=np.float64)

        return [(float(xs[i]), float(ys[i]), float(zs[i])) for i in range(scene_steps)]


class WaypointMobility(MobilityPattern):
    """Movement through a series of waypoints."""

    yaml_type = "waypoint"

    def __init__(self, waypoints: list[Any]):
        """
        Initialize waypoint mobility.

        Args:
            waypoints: Sequence of waypoints [(x1,y1,z1), (x2,y2,z2), ...]
        """
        super().__init__()
        self.waypoints = waypoints

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "WaypointMobility":
        """Build waypoint mobility from a YAML mobility block.

        This stores only the waypoint control points from YAML. The full
        per-step trajectory is interpolated later by ``prepare()`` through
        ``get_positions()``.
        """
        wps_raw = mobility_cfg.get("waypoints")
        if not wps_raw:
            raise ValueError(f"Device {entry['name']!r}: waypoints required for waypoint mobility")
        return cls(waypoints=[point_to_tuple(wp) for wp in wps_raw])

    def initial_position_for_config(
        self,
        declared_position: tuple[float, ...] | None = None,
    ) -> tuple[float, ...] | None:
        """Use the first waypoint as the initial config position."""
        if not self.waypoints:
            return declared_position
        return point_to_tuple(self.waypoints[0])

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Calculate AVERAGE speed through all waypoints."""
        if len(self.waypoints) < 2 or scene_duration <= 0:
            return 0.0

        total_distance = 0.0
        waypoint_arrays = [
            np.asarray(point_to_tuple(wp), dtype=np.float64) for wp in self.waypoints
        ]

        for i in range(1, len(waypoint_arrays)):
            start_wp = waypoint_arrays[i - 1]
            end_wp = waypoint_arrays[i]
            distance = float(np.linalg.norm(end_wp - start_wp))
            total_distance += distance

        return float(total_distance / scene_duration)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate a trajectory through waypoints with equal time per segment."""
        waypoint_arrays = [
            np.asarray(point_to_tuple(wp), dtype=np.float64) for wp in self.waypoints
        ]

        if len(waypoint_arrays) < 2:
            wp = waypoint_arrays[0]
            return [(float(wp[0]), float(wp[1]), float(wp[2]))] * scene_steps

        positions = []
        for i in range(scene_steps):
            if scene_steps == 1:
                t = 0.0
            else:
                t = i / (scene_steps - 1)

            total_segments = len(waypoint_arrays) - 1
            segment_index = int(t * total_segments)
            segment_t = (t * total_segments) - segment_index

            if segment_index >= total_segments:
                segment_index = total_segments - 1
                segment_t = 1.0

            start_wp = waypoint_arrays[segment_index]
            end_wp = waypoint_arrays[segment_index + 1]

            pos = start_wp + segment_t * (end_wp - start_wp)
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

        return positions


class StationaryMobility(MobilityPattern):
    """Device remains at a fixed position."""

    yaml_type = "stationary"

    def __init__(self, start_pos: Any):
        """
        Initialize stationary mobility.

        Args:
            start_pos: Fixed position (x, y, z)
        """
        super().__init__()
        self.start_pos = start_pos

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "StationaryMobility":
        """Build stationary mobility from a YAML mobility block."""
        if position is None:
            raise ValueError(f"Device {entry['name']!r}: position required for stationary mobility")
        return cls(start_pos=position)

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Stationary objects have zero speed."""
        return 0.0

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate stationary trajectory (same position for all steps)."""
        pos = point_to_tuple(self.start_pos)
        return [pos for _ in range(scene_steps)]
