"""Immutable data produced by the actor pose kernel."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, SupportsFloat, SupportsIndex

from .quaternion import Quaternion

Position3 = tuple[float, float, float]
Velocity3 = tuple[float, float, float]
ActorRole = Literal["tx", "rx", "target"]


def _finite_vec3(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a numeric three-vector")
    try:
        parsed = tuple(_as_float(component, name) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric three-vector") from exc
    if len(parsed) != 3 or not all(math.isfinite(component) for component in parsed):
        raise ValueError(f"{name} must be a finite numeric three-vector")
    return (parsed[0], parsed[1], parsed[2])


def _as_float(value: object, name: str) -> float:
    if not isinstance(value, (str, bytes, SupportsFloat, SupportsIndex)):
        raise TypeError(f"{name} must contain numeric values")
    return float(value)


@dataclass(frozen=True, slots=True)
class Timeline:
    """An endpoint-inclusive simulation timeline."""

    steps: int
    duration_s: float
    timestamps_s: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        steps = int(self.steps)
        duration = float(self.duration_s)
        if steps < 1:
            raise ValueError("timeline.steps must be at least 1")
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("timeline.duration_s must be finite and non-negative")
        if steps == 1:
            timestamps: tuple[float, ...] = (0.0,)
        else:
            interval = duration / (steps - 1)
            timestamps = tuple(interval * step for step in range(steps - 1)) + (duration,)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "timestamps_s", timestamps)


@dataclass(frozen=True, slots=True)
class PreparedMobility:
    """Exact-length immutable positions, velocities, and stable path headings."""

    positions_m: tuple[Position3, ...]
    velocities_mps: tuple[Velocity3, ...]
    forward_vectors: tuple[Position3, ...]
    has_physical_velocity: bool = True

    def __post_init__(self) -> None:
        positions = tuple(_finite_vec3(item, "position") for item in self.positions_m)
        velocities = tuple(_finite_vec3(item, "velocity") for item in self.velocities_mps)
        forwards = tuple(_finite_vec3(item, "forward vector") for item in self.forward_vectors)
        if not positions:
            raise ValueError("prepared mobility must contain at least one sample")
        if len(velocities) != len(positions) or len(forwards) != len(positions):
            raise ValueError("prepared mobility series must have identical lengths")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        object.__setattr__(self, "forward_vectors", forwards)

    @property
    def steps(self) -> int:
        return len(self.positions_m)


@dataclass(frozen=True, slots=True)
class PreparedOrientation:
    """Exact-length immutable orientation quaternions."""

    quaternions: tuple[Quaternion, ...]
    asset_alignment_applied: bool = False

    def __post_init__(self) -> None:
        quaternions = tuple(self.quaternions)
        if not quaternions:
            raise ValueError("prepared orientation must contain at least one sample")
        if not all(isinstance(item, Quaternion) for item in quaternions):
            raise TypeError("prepared orientation values must be Quaternion instances")
        object.__setattr__(self, "quaternions", quaternions)

    @property
    def euler_deg(self) -> tuple[tuple[float, float, float], ...]:
        """Expose Euler degrees only for UI/engine boundary adapters."""

        return tuple(quaternion.to_euler_deg() for quaternion in self.quaternions)


@dataclass(frozen=True, slots=True)
class PreparedActorPose:
    """All prepared pose data for one globally named actor."""

    name: str
    role: ActorRole
    mobility: PreparedMobility
    orientation: PreparedOrientation

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("actor name must not be empty")
        if self.role not in ("tx", "rx", "target"):
            raise ValueError(f"unsupported actor role {self.role!r}")
        if self.mobility.steps != len(self.orientation.quaternions):
            raise ValueError("actor mobility and orientation lengths differ")

    @property
    def positions_m(self) -> tuple[Position3, ...]:
        return self.mobility.positions_m

    @property
    def velocities_mps(self) -> tuple[Velocity3, ...]:
        return self.mobility.velocities_mps


@dataclass(frozen=True, slots=True)
class PreparedGroupPose:
    """One shared group path, evaluated once before member offsets."""

    name: str
    mobility: PreparedMobility


@dataclass(frozen=True, slots=True)
class PreparedScenario:
    """Renderer-neutral prepared poses for an entire scenario."""

    timeline: Timeline
    actors: tuple[PreparedActorPose, ...]
    groups: tuple[PreparedGroupPose, ...] = ()

    def __post_init__(self) -> None:
        actors = tuple(self.actors)
        groups = tuple(self.groups)
        names = [actor.name for actor in actors]
        if len(names) != len(set(names)):
            raise ValueError("prepared actor names must be globally unique")
        if any(actor.mobility.steps != self.timeline.steps for actor in actors):
            raise ValueError("prepared actor sample count does not match timeline")
        if any(group.mobility.steps != self.timeline.steps for group in groups):
            raise ValueError("prepared group sample count does not match timeline")
        object.__setattr__(self, "actors", actors)
        object.__setattr__(self, "groups", groups)

    def actor(self, name: str) -> PreparedActorPose:
        """Resolve one actor by its serialized name."""

        for actor in self.actors:
            if actor.name == name:
                return actor
        raise KeyError(name)

    def actors_for_role(self, role: ActorRole) -> tuple[PreparedActorPose, ...]:
        """Return actors in source order for one immutable role."""

        return tuple(actor for actor in self.actors if actor.role == role)
