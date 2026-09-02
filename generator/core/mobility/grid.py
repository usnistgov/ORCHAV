"""Grid-based mobility patterns for generator devices and targets.

``MeshGridMobility`` is useful for deterministic spatial sweeps: it generates a
regular 2D/3D grid, orders the points with a traversal pattern, and repeats the
order if the scene has more steps than grid points. Timeline services may expand
scene steps when ``auto_expand_scene_steps`` is enabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.logging import get_logger

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3, _numeric_pair

logger = get_logger(__name__)

# Keep random traversal deterministic unless the YAML/schema grows an explicit
# traversal seed.
DEFAULT_RANDOM_TRAVERSAL_SEED = 42


class MeshGridMobility(MobilityPattern):
    """
    Mobility pattern that generates positions in a 2D or 3D meshgrid.

    This class creates a regular grid of positions within specified bounds
    and provides various traversal patterns to move through the grid points.
    """

    yaml_type = "mesh_grid"

    def __init__(
        self,
        x_bounds: tuple[float, float] = (0.0, 10.0),
        y_bounds: tuple[float, float] = (0.0, 10.0),
        z_bounds: tuple[float, float] = (0.0, 5.0),
        x_steps: int = 5,
        y_steps: int = 5,
        z_steps: int = 1,
        traversal_pattern: str = "snake",
        start_corner: str = "bottom_left",
        target_speed_mps: float | None = None,
        start_pos: Any | None = None,
        auto_expand_scene_steps: bool = False,
    ):
        """
        Initialize meshgrid mobility.

        Args:
            x_bounds: (min_x, max_x) bounds for X dimension
            y_bounds: (min_y, max_y) bounds for Y dimension
            z_bounds: (min_z, max_z) bounds for Z dimension
            x_steps: Number of grid points in X direction
            y_steps: Number of grid points in Y direction
            z_steps: Number of grid points in Z direction (1 for 2D grid)
            traversal_pattern: How to traverse the grid ('snake', 'spiral', 'sequential', 'random')
            start_corner: Starting corner for traversal ('bottom_left', 'bottom_right', 'top_left', 'top_right')
            target_speed_mps: Target speed in m/s (optional)
            start_pos: Starting position (optional, defaults to first grid point)
            auto_expand_scene_steps: If True, consumers may increase scene steps
                                     to cover all grid points (no repetition)
        """
        super().__init__()

        if x_steps < 1 or y_steps < 1 or z_steps < 1:
            raise ValueError("All step counts must be >= 1")

        if x_bounds[0] > x_bounds[1] or y_bounds[0] > y_bounds[1] or z_bounds[0] > z_bounds[1]:
            raise ValueError("Bounds must have min <= max")

        if traversal_pattern not in ["snake", "spiral", "sequential", "random"]:
            raise ValueError(
                "traversal_pattern must be one of: 'snake', 'spiral', 'sequential', 'random'"
            )

        if start_corner not in ["bottom_left", "bottom_right", "top_left", "top_right"]:
            raise ValueError(
                "start_corner must be one of: 'bottom_left', 'bottom_right', 'top_left', 'top_right'"
            )

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.x_steps = x_steps
        self.y_steps = y_steps
        self.z_steps = z_steps
        self.traversal_pattern = traversal_pattern
        self.start_corner = start_corner
        self.target_speed_mps = target_speed_mps
        self.auto_expand_scene_steps = bool(auto_expand_scene_steps)

        self.grid_points = self._generate_grid_points()
        self.total_points = len(self.grid_points)
        self._traversal_order = self._get_traversal_order()

        if start_pos is not None:
            self.start_pos = point_to_tuple(start_pos)
        else:
            traversal_order = self._get_traversal_order()
            first_idx = traversal_order[0]
            self.start_pos = self.grid_points[first_idx]

        logger.info(
            "MeshGrid: %d points, dims=%dx%dx%d, bounds X%s Y%s Z%s, traversal=%s/%s",
            self.total_points,
            x_steps,
            y_steps,
            z_steps,
            x_bounds,
            y_bounds,
            z_bounds,
            traversal_pattern,
            start_corner,
        )

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "MeshGridMobility":
        """Build mesh-grid mobility from a YAML mobility block."""
        speed = mobility_cfg.get("speed_mps")
        return cls(
            x_bounds=_bounds_from_yaml(mobility_cfg, "x_bounds", (0.0, 10.0)),
            y_bounds=_bounds_from_yaml(mobility_cfg, "y_bounds", (0.0, 10.0)),
            z_bounds=_bounds_from_yaml(mobility_cfg, "z_bounds", (0.0, 5.0)),
            x_steps=int(mobility_cfg.get("x_steps", 5)),
            y_steps=int(mobility_cfg.get("y_steps", 5)),
            z_steps=int(mobility_cfg.get("z_steps", 1)),
            traversal_pattern=str(mobility_cfg.get("traversal_pattern", "snake")),
            start_corner=str(mobility_cfg.get("start_corner", "bottom_left")),
            target_speed_mps=float(speed) if speed is not None else None,
            start_pos=position,
            auto_expand_scene_steps=bool(mobility_cfg.get("auto_expand_scene_steps", False)),
        )

    def _generate_grid_points(self) -> list[tuple[float, float, float]]:
        """Generate all grid points in the specified bounds."""
        x_coords = np.linspace(self.x_bounds[0], self.x_bounds[1], self.x_steps)
        y_coords = np.linspace(self.y_bounds[0], self.y_bounds[1], self.y_steps)
        z_coords = np.linspace(self.z_bounds[0], self.z_bounds[1], self.z_steps)

        if self.z_steps == 1:
            X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
            Z = np.full_like(X, z_coords[0])
        else:
            X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

        points = []
        for i in range(X.size):
            points.append((float(X.flat[i]), float(Y.flat[i]), float(Z.flat[i])))

        return points

    def _grid_index(self, x_idx: int, y_idx: int, z_idx: int = 0) -> int:
        """Return the flattened point index for x/y/z grid coordinates."""
        return x_idx * self.y_steps * self.z_steps + y_idx * self.z_steps + z_idx

    def _corner_ranges(self) -> tuple[list[int], list[int]]:
        """Return x and y index order implied by ``start_corner``."""
        x_range = list(range(self.x_steps))
        y_range = list(range(self.y_steps))
        if self.start_corner.endswith("right"):
            x_range.reverse()
        if self.start_corner.startswith("top"):
            y_range.reverse()
        return x_range, y_range

    def _get_traversal_order(self) -> list[int]:
        """Get the order in which to traverse grid points based on pattern."""
        if hasattr(self, "_traversal_order") and self._traversal_order is not None:
            return self._traversal_order

        if self.traversal_pattern == "sequential":
            return self._sequential_traversal()
        elif self.traversal_pattern == "random":
            order = list(range(self.total_points))
            rng = np.random.RandomState(DEFAULT_RANDOM_TRAVERSAL_SEED)
            rng.shuffle(order)
            return order
        elif self.traversal_pattern == "snake":
            return self._snake_traversal()
        elif self.traversal_pattern == "spiral":
            return self._spiral_traversal()
        else:
            return list(range(self.total_points))

    def _sequential_traversal(self) -> list[int]:
        """Generate row-by-row traversal from the configured start corner."""
        order = []
        x_range, y_range = self._corner_ranges()
        for z_idx in range(self.z_steps):
            for y_idx in y_range:
                for x_idx in x_range:
                    order.append(self._grid_index(x_idx, y_idx, z_idx))
        return order

    def _snake_traversal(self) -> list[int]:
        """Generate zigzag row traversal from the configured start corner."""
        order = []
        x_range, y_range = self._corner_ranges()
        reverse_x_range = list(reversed(x_range))

        for z_idx in range(self.z_steps):
            for row_idx, y_idx in enumerate(y_range):
                row_x_range = x_range if row_idx % 2 == 0 else reverse_x_range
                for x_idx in row_x_range:
                    order.append(self._grid_index(x_idx, y_idx, z_idx))

        return order

    def _spiral_traversal(self) -> list[int]:
        """Generate spiral traversal pattern (spiral outward from center)."""
        if self.z_steps == 1:
            return self._spiral_2d()
        else:
            order = []
            for z_idx in range(self.z_steps):
                order.extend(self._spiral_2d(z_idx))
            return order

    def _spiral_2d(self, z_idx: int = 0) -> list[int]:
        """Generate 2D spiral traversal."""
        center_x = self.x_steps // 2
        center_y = self.y_steps // 2

        visited = set()
        order = []

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        direction_idx = 0

        x, y = center_x, center_y
        steps_in_direction = 1
        steps_taken = 0

        while len(visited) < self.x_steps * self.y_steps:
            if 0 <= x < self.x_steps and 0 <= y < self.y_steps and (x, y) not in visited:
                visited.add((x, y))
                order.append(self._grid_index(x, y, z_idx))

            dx, dy = directions[direction_idx]
            x += dx
            y += dy
            steps_taken += 1

            if steps_taken >= steps_in_direction:
                steps_taken = 0
                direction_idx = (direction_idx + 1) % 4
                if direction_idx % 2 == 0:
                    steps_in_direction += 1

        return order

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Get the speed for meshgrid mobility."""
        if self.target_speed_mps is not None:
            return self.target_speed_mps

        if scene_duration <= 0 or len(self.grid_points) < 2:
            return 0.0

        traversal_order = self._get_traversal_order()
        total_distance = 0.0

        for i in range(1, len(traversal_order)):
            idx1 = traversal_order[i - 1]
            idx2 = traversal_order[i]
            pos1 = np.array(self.grid_points[idx1])
            pos2 = np.array(self.grid_points[idx2])
            total_distance += float(np.linalg.norm(pos2 - pos1))

        return float(total_distance / scene_duration) if scene_duration > 0 else 0.0

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate positions by traversing the meshgrid."""
        if not self.grid_points:
            raise ValueError("No grid points generated")

        traversal_order = self._get_traversal_order()

        if scene_steps <= self.total_points:
            selected_indices = traversal_order[:scene_steps]
        else:
            selected_indices = []
            for i in range(scene_steps):
                selected_indices.append(traversal_order[i % self.total_points])

        return [self.grid_points[idx] for idx in selected_indices]

    def get_grid_info(self) -> dict[str, Any]:
        """Get information about the generated grid."""
        return {
            "total_points": self.total_points,
            "dimensions": (self.x_steps, self.y_steps, self.z_steps),
            "bounds": {"x": self.x_bounds, "y": self.y_bounds, "z": self.z_bounds},
            "traversal_pattern": self.traversal_pattern,
            "start_corner": self.start_corner,
            "is_2d": self.z_steps == 1,
            "grid_spacing": {
                "x": (
                    (self.x_bounds[1] - self.x_bounds[0]) / (self.x_steps - 1)
                    if self.x_steps > 1
                    else 0
                ),
                "y": (
                    (self.y_bounds[1] - self.y_bounds[0]) / (self.y_steps - 1)
                    if self.y_steps > 1
                    else 0
                ),
                "z": (
                    (self.z_bounds[1] - self.z_bounds[0]) / (self.z_steps - 1)
                    if self.z_steps > 1
                    else 0
                ),
            },
        }


def _bounds_from_yaml(
    mobility_cfg: dict[str, Any], key: str, default: tuple[float, float]
) -> tuple[float, float]:
    """Parse a two-value YAML bounds field."""
    return _numeric_pair(mobility_cfg.get(key, default), key)
