"""Standard stochastic mobility models for wireless simulation studies.

These classes implement common literature models with deterministic seeds for
reproducible experiments. They are scenario-level approximations of user/device
motion, not physics-integrated dynamics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3, _numeric_pair, _numeric_six


class GaussMarkovMobility(MobilityPattern):
    """Gauss-Markov correlated random mobility.

    Produces smooth random trajectories with tunable temporal correlation.
    Standard model in MANET and UAV wireless research.

    The memory factor ``alpha`` controls smoothness:
    - alpha=0 → purely random (like random walk)
    - alpha=1 → constant velocity (like linear)
    - alpha=0.75 → typical smooth random motion
    """

    yaml_type = "gauss_markov"

    def __init__(
        self,
        start_pos: Any = (0, 0, 0),
        alpha: float = 0.75,
        mean_speed: float = 1.0,
        mean_direction: float = 0.0,
        speed_variance: float = 0.5,
        direction_variance: float = 0.5,
        bounds: tuple[float, float, float, float, float, float] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.start_pos = start_pos
        self.alpha = float(alpha)
        self.mean_speed = float(mean_speed)
        self.mean_direction = float(mean_direction)
        self.speed_variance = float(speed_variance)
        self.direction_variance = float(direction_variance)
        self.bounds = bounds
        self.seed = seed

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "GaussMarkovMobility":
        """Build Gauss-Markov mobility from a YAML mobility block."""
        return cls(
            start_pos=position or (0, 0, 0),
            alpha=float(mobility_cfg.get("alpha", 0.75)),
            mean_speed=float(mobility_cfg.get("mean_speed", 1.0)),
            mean_direction=float(mobility_cfg.get("mean_direction", 0.0)),
            speed_variance=float(mobility_cfg.get("speed_variance", 0.5)),
            direction_variance=float(mobility_cfg.get("direction_variance", 0.5)),
            bounds=(
                _numeric_six(mobility_cfg["bounds"], "bounds")
                if mobility_cfg.get("bounds")
                else None
            ),
            seed=int(mobility_cfg["seed"]) if mobility_cfg.get("seed") is not None else None,
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the configured mean speed."""
        return self.mean_speed

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate a correlated random trajectory from the configured start position."""
        rng = np.random.RandomState(self.seed)
        pos = np.asarray(point_to_tuple(self.start_pos), dtype=np.float64)

        dt = scene_duration / max(scene_steps - 1, 1)
        a = self.alpha
        speed = self.mean_speed
        direction = self.mean_direction

        positions = []
        for _ in range(scene_steps):
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
            speed = (
                a * speed
                + (1 - a) * self.mean_speed
                + math.sqrt(1 - a * a) * rng.normal(0, self.speed_variance)
            )
            speed = max(0, speed)
            direction = (
                a * direction
                + (1 - a) * self.mean_direction
                + math.sqrt(1 - a * a) * rng.normal(0, self.direction_variance)
            )
            dx = speed * math.cos(direction) * dt
            dy = speed * math.sin(direction) * dt
            pos = pos + np.array([dx, dy, 0.0])

            if self.bounds is not None:
                # Bounds are hard clamps, not reflective walls, matching the
                # simple Gauss-Markov model used by scenario sweeps.
                pos[0] = np.clip(pos[0], self.bounds[0], self.bounds[1])
                pos[1] = np.clip(pos[1], self.bounds[2], self.bounds[3])
                pos[2] = np.clip(pos[2], self.bounds[4], self.bounds[5])

        return positions


class RandomWaypointMobility(MobilityPattern):
    """Random Waypoint with pause — the most cited mobility model in wireless lit.

    Pick random destination within bounds, move at random speed, pause,
    pick next destination, repeat.
    """

    yaml_type = "random_waypoint"

    def __init__(
        self,
        start_pos: Any = (0, 0, 0),
        bounds: tuple[float, float, float, float, float, float] = (
            -50,
            50,
            -50,
            50,
            0,
            2,
        ),
        speed_range: tuple[float, float] = (0.5, 2.0),
        pause_range: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
    ):
        super().__init__()
        self.start_pos = start_pos
        self.bounds = bounds
        self.speed_range = speed_range
        self.pause_range = pause_range
        self.seed = seed

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "RandomWaypointMobility":
        """Build random waypoint mobility from a YAML mobility block."""
        return cls(
            start_pos=position or (0, 0, 0),
            bounds=(
                _numeric_six(mobility_cfg["bounds"], "bounds")
                if mobility_cfg.get("bounds")
                else (-50, 50, -50, 50, 0, 2)
            ),
            speed_range=_numeric_pair(mobility_cfg.get("speed_range", [0.5, 2.0]), "speed_range"),
            pause_range=_numeric_pair(mobility_cfg.get("pause_range", [0.0, 1.0]), "pause_range"),
            seed=int(mobility_cfg["seed"]) if mobility_cfg.get("seed") is not None else None,
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the midpoint of the configured speed range."""
        return float(sum(self.speed_range) / 2)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate random waypoint positions with optional pauses."""
        rng = np.random.RandomState(self.seed)
        pos = np.asarray(point_to_tuple(self.start_pos), dtype=np.float64)

        dt = scene_duration / max(scene_steps - 1, 1)
        b = self.bounds

        positions = []
        target = None
        pause_remaining = 0.0

        for _ in range(scene_steps):
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

            if pause_remaining > 0:
                pause_remaining -= dt
                continue

            if target is None or np.linalg.norm(target - pos) < 0.01:
                target = np.array(
                    [
                        rng.uniform(b[0], b[1]),
                        rng.uniform(b[2], b[3]),
                        rng.uniform(b[4], b[5]),
                    ]
                )
                pause_remaining = rng.uniform(*self.pause_range)
                continue

            speed = rng.uniform(*self.speed_range)
            direction = target - pos
            dist = np.linalg.norm(direction)
            if dist > 0:
                step_dist = min(speed * dt, dist)
                pos = pos + (direction / dist) * step_dist

        return positions


class ManhattanGridMobility(MobilityPattern):
    """Movement along a street grid with probabilistic turns at intersections.

    Standard urban mobility model for mmWave/V2X scenarios.
    """

    yaml_type = "manhattan_grid"

    def __init__(
        self,
        start_pos: Any = (0, 0, 0),
        grid_origin: tuple[float, float] = (-100, -100),
        block_size: float = 50.0,
        grid_width: int = 4,
        grid_height: int = 4,
        altitude: float = 1.5,
        turn_probability: float = 0.5,
        speed_range: tuple[float, float] = (0.5, 2.0),
        seed: int | None = None,
    ):
        super().__init__()
        self.start_pos = start_pos
        self.grid_origin = grid_origin
        self.block_size = float(block_size)
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.altitude = float(altitude)
        self.turn_probability = float(turn_probability)
        self.speed_range = speed_range
        self.seed = seed

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "ManhattanGridMobility":
        """Build Manhattan-grid mobility from a YAML mobility block."""
        return cls(
            start_pos=position or (0, 0, 0),
            grid_origin=_numeric_pair(mobility_cfg.get("grid_origin", [-100, -100]), "grid_origin"),
            block_size=float(mobility_cfg.get("block_size", 50)),
            grid_width=int(mobility_cfg.get("grid_width", 4)),
            grid_height=int(mobility_cfg.get("grid_height", 4)),
            altitude=float(mobility_cfg.get("altitude", 1.5)),
            turn_probability=float(mobility_cfg.get("turn_probability", 0.5)),
            speed_range=_numeric_pair(mobility_cfg.get("speed_range", [0.5, 2.0]), "speed_range"),
            seed=int(mobility_cfg["seed"]) if mobility_cfg.get("seed") is not None else None,
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Return the midpoint of the configured speed range."""
        return float(sum(self.speed_range) / 2)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate positions constrained to Manhattan-style grid streets."""
        rng = np.random.RandomState(self.seed)
        start_x, start_y, _start_z = point_to_tuple(self.start_pos)
        pos = np.array([start_x, start_y], dtype=np.float64)

        dt = scene_duration / max(scene_steps - 1, 1)
        ox, oy = self.grid_origin
        bs = self.block_size
        gw, gh = self.grid_width, self.grid_height

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        d_idx = rng.randint(0, 4)
        dx, dy = directions[d_idx]

        positions = []
        for _ in range(scene_steps):
            positions.append((float(pos[0]), float(pos[1]), self.altitude))

            speed = rng.uniform(*self.speed_range)
            step_dist = speed * dt
            pos[0] += dx * step_dist
            pos[1] += dy * step_dist

            x_max = ox + gw * bs
            y_max = oy + gh * bs
            pos[0] = np.clip(pos[0], ox, x_max)
            pos[1] = np.clip(pos[1], oy, y_max)

            rel_x = (pos[0] - ox) % bs
            rel_y = (pos[1] - oy) % bs
            near_intersection = (
                min(rel_x, bs - rel_x) < step_dist * 1.5
                and min(rel_y, bs - rel_y) < step_dist * 1.5
            )
            if near_intersection and rng.random() < self.turn_probability:
                if dx != 0:
                    dx, dy = 0, rng.choice([-1, 1])
                else:
                    dx, dy = rng.choice([-1, 1]), 0

            if pos[0] <= ox or pos[0] >= x_max:
                dx = -dx
            if pos[1] <= oy or pos[1] >= y_max:
                dy = -dy

        return positions
