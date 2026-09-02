"""Formation mobility -- follow a leader with a fixed spatial offset.

The offset is specified in the leader's local frame (right, forward, up),
so the formation rotates as the leader changes heading.
YAML construction is deferred until all named mobility objects exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils import point_to_tuple
from .base import MobilityPattern, Position3, _numeric_triple


@dataclass
class DeferredFormationMobility:
    """Formation YAML placeholder resolved after all named devices exist."""

    leader_name: str
    offset: tuple[float, float, float]
    noise: float
    seed: int | None

    yaml_type = "formation"

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "DeferredFormationMobility":
        """Build a deferred formation reference from a YAML mobility block."""
        leader_name = mobility_cfg.get("leader")
        if not leader_name:
            raise ValueError(f"Device {entry['name']!r}: leader required for formation mobility")
        return cls(
            leader_name=str(leader_name),
            offset=_numeric_triple(mobility_cfg.get("offset", [5.0, -3.0, 0.0]), "offset"),
            noise=float(mobility_cfg.get("noise", 0.0)),
            seed=int(mobility_cfg["seed"]) if mobility_cfg.get("seed") is not None else None,
        )


class FormationMobility(MobilityPattern):
    """Follow a leader with a fixed offset in the leader's heading frame.

    Args:
        leader_mobility: The leader's mobility pattern (resolved by factory).
        offset: ``(right, forward, up)`` in the leader's local frame.
            Right is perpendicular to heading, forward is along heading,
            up is vertical.
        noise: Gaussian jitter in meters (0 = exact formation).
        seed: RNG seed for reproducible noise.
    """

    yaml_type = "formation"

    def __init__(
        self,
        leader_mobility: MobilityPattern,
        offset: tuple[float, float, float] = (5.0, -3.0, 0.0),
        noise: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.leader_mobility = leader_mobility
        self.offset = (float(offset[0]), float(offset[1]), float(offset[2]))
        self.noise = float(noise)
        self.seed = seed

    @property
    def start_pos(self):
        """Derive initial position from leader start + offset at heading 0."""
        leader_start = getattr(self.leader_mobility, "start_pos", None)
        if leader_start is None:
            # WaypointMobility and similar patterns expose the initial point via
            # control points instead of a start_pos attribute.
            wps = getattr(self.leader_mobility, "waypoints", None)
            if wps and len(wps) > 0:
                leader_start = wps[0]
            else:
                return None
        lx, ly, lz = point_to_tuple(leader_start)
        # At heading 0: forward=(1,0,0), right=(0,-1,0)
        return (
            lx + self.offset[1],
            ly - self.offset[0],
            lz + self.offset[2],
        )

    def get_positions(
        self,
        start_pos: Position3 | None,
        scene_steps: int,
        scene_duration: float,
    ) -> list[Position3]:
        """Generate follower positions from the leader path and local offset."""
        leader_start = getattr(self.leader_mobility, "start_pos", start_pos or (0, 0, 0))
        raw = self.leader_mobility.get_positions(leader_start, scene_steps, scene_duration)

        pts = [point_to_tuple(p) for p in raw]

        leader = np.array(pts, dtype=np.float64)
        n = len(leader)

        if n == 0:
            return []

        # Derive follower heading from the leader's finite-difference direction.
        if n == 1:
            headings = np.zeros(1)
        else:
            dx = np.diff(leader[:, 0])
            dy = np.diff(leader[:, 1])
            seg_headings = np.arctan2(dy, dx)
            # Last step reuses heading from previous segment
            headings = np.empty(n)
            headings[:-1] = seg_headings
            headings[-1] = seg_headings[-1]

        # Rotate local offset into world frame per step:
        # forward=(cos(h), sin(h), 0), right=(sin(h), -cos(h), 0), up=(0, 0, 1).
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)

        r, f, u = self.offset
        wx = sin_h * r + cos_h * f
        wy = -cos_h * r + sin_h * f
        wz = np.full(n, u)

        follower = leader + np.column_stack([wx, wy, wz])

        # Add bounded Gaussian jitter if requested.
        if self.noise > 0:
            rng = np.random.RandomState(self.seed)
            for i in range(n):
                jitter = rng.normal(0, self.noise / 3, size=3)
                norm = np.linalg.norm(jitter)
                if norm > self.noise:
                    jitter = jitter / norm * self.noise
                follower[i] += jitter

        return [
            (float(follower[i, 0]), float(follower[i, 1]), float(follower[i, 2])) for i in range(n)
        ]
