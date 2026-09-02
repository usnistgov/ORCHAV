"""Random-area mobility patterns.

``RandomBoxMobility`` samples independent positions inside a configured 3D box.
It is intended for stochastic placement/sweep scenarios rather than physically
continuous motion; callers that need continuous paths should use waypoint,
drone or standard academic mobility models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.logging import get_logger

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3, _numeric_pair

logger = get_logger(__name__)

# Bound rejection sampling so impossible min_distance constraints do not hang.
MAX_MIN_DISTANCE_ATTEMPTS = 100


class RandomBoxMobility(MobilityPattern):
    """
    Mobility pattern that generates random positions within a 3D bounding box.

    This class creates random positions within specified bounds and provides
    various sampling strategies for different types of random movement.
    """

    yaml_type = "random_box"
    has_physical_motion = False

    def __init__(
        self,
        x_bounds: tuple[float, float] = (0.0, 10.0),
        y_bounds: tuple[float, float] = (0.0, 10.0),
        z_bounds: tuple[float, float] = (0.0, 5.0),
        sampling_strategy: str = "uniform",
        seed: int | None = None,
        target_speed_mps: float | None = None,
        start_pos: Any | None = None,
        min_distance: float | None = None,
    ):
        """
        Initialize random box mobility.

        Args:
            x_bounds: (min_x, max_x) bounds for X dimension
            y_bounds: (min_y, max_y) bounds for Y dimension
            z_bounds: (min_z, max_z) bounds for Z dimension
            sampling_strategy: How to sample positions ('uniform', 'gaussian', 'gaussian_center')
            seed: Random seed for reproducibility (optional)
            target_speed_mps: Target speed in m/s (optional)
            start_pos: Starting position (optional, defaults to random position)
            min_distance: Minimum distance between consecutive positions (optional)
        """
        super().__init__()

        if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1] or z_bounds[0] >= z_bounds[1]:
            raise ValueError("Bounds must have min < max")

        if sampling_strategy not in ["uniform", "gaussian", "gaussian_center"]:
            raise ValueError(
                "sampling_strategy must be one of: 'uniform', 'gaussian', 'gaussian_center'"
            )

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        self.target_speed_mps = target_speed_mps
        self.min_distance = min_distance

        self.rng = np.random.RandomState(seed)

        self.box_width = x_bounds[1] - x_bounds[0]
        self.box_height = y_bounds[1] - y_bounds[0]
        self.box_depth = z_bounds[1] - z_bounds[0]
        self.box_center = (
            (x_bounds[0] + x_bounds[1]) / 2,
            (y_bounds[0] + y_bounds[1]) / 2,
            (z_bounds[0] + z_bounds[1]) / 2,
        )

        if start_pos is not None:
            self.start_pos = point_to_tuple(start_pos)
        else:
            self.start_pos = self._generate_random_position()

        logger.info("RandomBox mobility initialized")
        logger.debug(
            "Bounds X%s Y%s Z%s, size %.1fx%.1fx%.1f, sampling=%s, seed=%s",
            x_bounds,
            y_bounds,
            z_bounds,
            self.box_width,
            self.box_height,
            self.box_depth,
            sampling_strategy,
            seed,
        )

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "RandomBoxMobility":
        """Build random-box mobility from a YAML mobility block."""
        speed = mobility_cfg.get("speed_mps")
        min_distance = mobility_cfg.get("min_distance")
        return cls(
            x_bounds=_numeric_pair(mobility_cfg["x_bounds"], "x_bounds"),
            y_bounds=_numeric_pair(mobility_cfg["y_bounds"], "y_bounds"),
            z_bounds=_numeric_pair(mobility_cfg["z_bounds"], "z_bounds"),
            sampling_strategy=str(mobility_cfg.get("sampling_strategy", "uniform")),
            seed=int(mobility_cfg.get("seed", 42)),
            target_speed_mps=float(speed) if speed is not None else None,
            start_pos=position,
            min_distance=float(min_distance) if min_distance is not None else None,
        )

    def _generate_random_position(self) -> tuple[float, float, float]:
        """Generate a single random position within the bounding box."""
        if self.sampling_strategy == "uniform":
            x = self.rng.uniform(self.x_bounds[0], self.x_bounds[1])
            y = self.rng.uniform(self.y_bounds[0], self.y_bounds[1])
            z = self.rng.uniform(self.z_bounds[0], self.z_bounds[1])

        elif self.sampling_strategy == "gaussian":
            x = self.rng.normal(self.box_center[0], self.box_width / 6)
            y = self.rng.normal(self.box_center[1], self.box_height / 6)
            z = self.rng.normal(self.box_center[2], self.box_depth / 6)

            x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
            y = np.clip(y, self.y_bounds[0], self.y_bounds[1])
            z = np.clip(z, self.z_bounds[0], self.z_bounds[1])

        elif self.sampling_strategy == "gaussian_center":
            x = self.rng.normal(self.box_center[0], self.box_width / 12)
            y = self.rng.normal(self.box_center[1], self.box_height / 12)
            z = self.rng.normal(self.box_center[2], self.box_depth / 12)

            x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
            y = np.clip(y, self.y_bounds[0], self.y_bounds[1])
            z = np.clip(z, self.z_bounds[0], self.z_bounds[1])
        else:
            x = self.rng.uniform(self.x_bounds[0], self.x_bounds[1])
            y = self.rng.uniform(self.y_bounds[0], self.y_bounds[1])
            z = self.rng.uniform(self.z_bounds[0], self.z_bounds[1])

        return (float(x), float(y), float(z))

    def _generate_random_positions(self, num_positions: int) -> list[tuple[float, float, float]]:
        """Generate multiple random positions with optional minimum distance constraint."""
        positions = []

        for i in range(num_positions):
            max_attempts = MAX_MIN_DISTANCE_ATTEMPTS
            attempts = 0
            fallback_pos: Position3 | None = None

            while attempts < max_attempts:
                new_pos = self._generate_random_position()
                fallback_pos = new_pos

                if self.min_distance is None or len(positions) == 0:
                    positions.append(new_pos)
                    break

                too_close = False
                for prev_pos in positions:
                    distance = np.linalg.norm(np.array(new_pos) - np.array(prev_pos))
                    if distance < self.min_distance:
                        too_close = True
                        break

                if not too_close:
                    positions.append(new_pos)
                    break

                attempts += 1

            if attempts >= max_attempts:
                # Preserve output length even when the requested spacing is too
                # strict for the box/step count. The warning marks degraded sampling.
                positions.append(
                    fallback_pos if fallback_pos is not None else self._generate_random_position()
                )
                logger.warning(
                    "Could not satisfy min_distance=%.2f for position %d", self.min_distance, i
                )

        return positions

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Get the speed for random box mobility."""
        if self.target_speed_mps is not None:
            return self.target_speed_mps

        if scene_duration <= 0:
            return 0.0

        avg_distance = (self.box_width + self.box_height + self.box_depth) / 3
        time_per_step = scene_duration / scene_steps if scene_steps > 0 else 0
        if time_per_step > 0:
            return float(avg_distance / time_per_step)
        return 0.0

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate random positions within the bounding box."""
        return self._generate_random_positions(scene_steps)

    def get_box_info(self) -> dict[str, Any]:
        """Get information about the bounding box."""
        return {
            "bounds": {"x": self.x_bounds, "y": self.y_bounds, "z": self.z_bounds},
            "dimensions": (self.box_width, self.box_height, self.box_depth),
            "center": self.box_center,
            "sampling_strategy": self.sampling_strategy,
            "seed": self.seed,
            "min_distance": self.min_distance,
            "volume": self.box_width * self.box_height * self.box_depth,
        }
