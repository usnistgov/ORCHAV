"""Group mobility model (Reference Point Group Mobility).

Members follow a reference trajectory with bounded random deviations. The group
reference can be any other mobility pattern, so this module composes with the
rest of the package instead of defining its own path generator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3


class GroupMobility(MobilityPattern):
    """Reference Point Group Mobility (RPGM).

    Wraps any other mobility pattern as the group reference point.
    Each position is perturbed by a random deviation within ``max_deviation``
    meters.  For a single device, this adds jitter to the reference trajectory.
    For creating multiple group members, the caller should instantiate N
    GroupMobility objects with different seeds sharing the same reference.
    """

    yaml_type = "group"

    def __init__(
        self,
        reference_mobility: MobilityPattern,
        max_deviation: float = 3.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.reference_mobility = reference_mobility
        self.max_deviation = float(max_deviation)
        self.seed = seed
        reference_start = getattr(reference_mobility, "start_pos", None)
        if reference_start is not None:
            self.start_pos = point_to_tuple(reference_start)

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
        *,
        reference_mobility: MobilityPattern | None = None,
    ) -> "GroupMobility":
        """Build group mobility from YAML after the reference mobility is resolved."""
        if reference_mobility is None:
            raise ValueError(f"Device {entry['name']!r}: reference mobility required for group")
        return cls(
            reference_mobility=reference_mobility,
            max_deviation=float(mobility_cfg.get("max_deviation", 3.0)),
            seed=int(mobility_cfg["seed"]) if mobility_cfg.get("seed") is not None else None,
        )

    def get_speed(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> float:
        """Use the reference trajectory speed for the group member."""
        return self.reference_mobility.get_speed(start_pos, scene_steps, scene_duration)

    def get_positions(
        self, start_pos: Position3 | None, scene_steps: int, scene_duration: float
    ) -> list[Position3]:
        """Generate reference positions with bounded random member deviation."""
        ref_positions = self.reference_mobility.get_positions(
            start_pos, scene_steps, scene_duration
        )
        rng = np.random.RandomState(self.seed)

        positions = []
        for ref_pos in ref_positions:
            rx, ry, rz = point_to_tuple(ref_pos)

            # Draw Gaussian jitter, then clamp to a sphere so max_deviation is a
            # hard per-step displacement bound.
            offset = rng.normal(0, self.max_deviation / 3, size=3)
            norm = np.linalg.norm(offset)
            if norm > self.max_deviation:
                offset = offset / norm * self.max_deviation

            positions.append(
                (
                    rx + float(offset[0]),
                    ry + float(offset[1]),
                    rz + float(offset[2]),
                )
            )

        return positions
