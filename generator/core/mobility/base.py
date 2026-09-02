"""Base lifecycle and shared parsing helpers for generator mobility patterns.

All mobility implementations follow the same two-stage contract. YAML/device
factory code first builds a lightweight object that stores control points,
bounds, speeds, seeds, or file paths. Later, timeline preparation calls
``prepare()`` with the final scene step count and duration, and the pattern
caches exactly one normalized ``Position3`` tuple per step.
"""

from __future__ import annotations

from typing import Any, ClassVar, TypeAlias

import numpy as np

from shared.logging import get_logger

from ..utils import point_to_tuple

logger = get_logger(__name__)

# Position terminology in mobility code:
# - Raw YAML/config values can be lists, tuples, numpy arrays, or tensor-like
#   point objects supplied by scripted callers and backend adapters.
# - Actual per-step coordinates are normalized to Position3. Mobility classes
#   must return/cache only this plain Python type from get_positions()/prepare().
# - Sionna/Mitsuba Point3f is an engine-boundary type, not a mobility type. It is
#   created later when propagation or target-scene code assigns positions into
#   the RT scene.
Position3: TypeAlias = tuple[float, float, float]


def _numeric_tuple(value: Any) -> tuple[float, ...]:
    """Parse non-position numeric config sequences.

    Some options are variable-length, while others have a fixed shape. Use the
    fixed-size helpers below when the receiving field is annotated as a 2-, 3-,
    5-, or 6-tuple; use point_to_tuple() for actual coordinates.
    """
    if isinstance(value, (str, bytes)):
        raise ValueError("numeric config sequence must not be a string")
    try:
        return tuple(float(v) for v in value)
    except TypeError as exc:
        raise TypeError("numeric config value must be an iterable sequence") from exc
    except ValueError as exc:
        raise ValueError("numeric config sequence must contain only numeric values") from exc


def _numeric_pair(value: Any, name: str = "value") -> tuple[float, float]:
    """Parse a numeric config sequence that must contain exactly two values."""
    parsed = _numeric_tuple(value)
    if len(parsed) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return (parsed[0], parsed[1])


def _numeric_triple(value: Any, name: str = "value") -> tuple[float, float, float]:
    """Parse a numeric config sequence that must contain exactly three values."""
    parsed = _numeric_tuple(value)
    if len(parsed) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (parsed[0], parsed[1], parsed[2])


def _numeric_quintuple(value: Any, name: str = "value") -> tuple[float, float, float, float, float]:
    """Parse a numeric config sequence that must contain exactly five values."""
    parsed = _numeric_tuple(value)
    if len(parsed) != 5:
        raise ValueError(f"{name} must contain exactly five values")
    return (parsed[0], parsed[1], parsed[2], parsed[3], parsed[4])


def _numeric_six(
    value: Any, name: str = "value"
) -> tuple[float, float, float, float, float, float]:
    """Parse a numeric config sequence that must contain exactly six values."""
    parsed = _numeric_tuple(value)
    if len(parsed) != 6:
        raise ValueError(f"{name} must contain exactly six values")
    return (parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], parsed[5])


class MobilityPattern:
    """Base class for all mobility patterns."""

    yaml_type: ClassVar[str | None] = None
    # Independent-sampling providers set this false so sample deltas are not
    # interpreted as velocity by orientation, Doppler, or exported state.
    has_physical_motion: ClassVar[bool] = True

    def __init__(self) -> None:
        """Initialize prepared trajectory cache fields."""
        self._prepared_positions: list[Position3] | None = None
        self._prepared_steps: int | None = None
        self._prepared_duration: float | None = None

    @classmethod
    def from_yaml(
        cls,
        entry: dict[str, Any],
        mobility_cfg: dict[str, Any],
        position: tuple[float, ...] | None,
        context: dict[str, Any] | None = None,
    ) -> "MobilityPattern":
        """Build this mobility object from its scenario YAML block.

        This method should only parse configuration and store control values
        such as endpoints, waypoints, bounds, or speeds. It should not compute
        positions yet because the final step count and duration are only known
        later when ``prepare()`` is called.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement from_yaml()")

    def initial_position_for_config(
        self,
        declared_position: tuple[float, ...] | None = None,
    ) -> tuple[float, ...] | None:
        """Return the position the device/target config should store before prepare().

        Patterns with their own ``start_pos`` keep that ownership local and
        return ``None`` so config validation does not duplicate the same value.
        """
        if hasattr(self, "start_pos"):
            return None
        return declared_position

    def get_positions(
        self,
        start_pos: Position3 | None,
        scene_steps: int,
        scene_duration: float,
    ) -> list[Position3]:
        """Get positions for each animation step as plain ``(x, y, z)`` tuples."""
        raise NotImplementedError("Subclasses must implement get_positions")

    def get_speed(
        self,
        start_pos: Position3 | None,
        scene_steps: int,
        scene_duration: float,
    ) -> float:
        """Calculate the average movement speed in meters per second."""
        positions = self.get_positions(start_pos, scene_steps, scene_duration)
        if len(positions) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(positions)):
            pos1 = np.asarray(point_to_tuple(positions[i - 1]), dtype=np.float64)
            pos2 = np.asarray(point_to_tuple(positions[i]), dtype=np.float64)
            total_distance += float(np.linalg.norm(pos2 - pos1))

        return float(total_distance / scene_duration) if scene_duration > 0 else 0.0

    def prepare(
        self,
        scene_steps: int,
        scene_duration: float,
        start_pos: Position3 | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Compute and cache positions once for the given timeline.

        This is the point where a parsed mobility object becomes a concrete
        per-step trajectory. Actor-state preparation calls this after YAML
        loading, when the final scene step count and duration are available.
        """
        scene_steps = int(scene_steps)
        scene_duration = float(scene_duration)
        if scene_steps < 1:
            raise ValueError(f"Mobility {self.__class__.__name__} requires at least one scene step")

        if (
            self._prepared_positions is not None
            and self._prepared_steps == scene_steps
            and self._prepared_duration == scene_duration
        ):
            logger.info(
                "Skipping mobility preparation for %s - already prepared with same parameters",
                self.__class__.__name__,
            )
            return

        logger.info(
            "Preparing mobility %s with start_pos=%s, scene_steps=%d, scene_duration=%s",
            self.__class__.__name__,
            start_pos,
            scene_steps,
            scene_duration,
        )
        positions = self.get_positions(start_pos, scene_steps, scene_duration)
        if len(positions) != scene_steps:
            # Normalize provider length here so every downstream actor-state
            # array remains aligned to the requested timeline.
            if len(positions) > 0 and len(positions) < scene_steps:
                last = positions[-1]
                positions = list(positions) + [last] * (scene_steps - len(positions))
            elif len(positions) > scene_steps:
                positions = positions[:scene_steps]
            else:
                raise ValueError(f"Mobility {self.__class__.__name__} returned no positions")

        self._prepared_positions = [point_to_tuple(p) for p in positions]
        self._prepared_steps = scene_steps
        self._prepared_duration = scene_duration
        logger.debug(
            "Prepared mobility %s with %d positions",
            self.__class__.__name__,
            len(self._prepared_positions),
        )

    def prepared_positions(self) -> list[Position3]:
        """Return the prepared positions list."""
        if self._prepared_positions is None:
            raise RuntimeError(
                f"Mobility {self.__class__.__name__} not prepared. Call prepare() first."
            )
        return self._prepared_positions

    def get_position(self, step: int) -> tuple[float, float, float]:
        """Get the prepared position for a specific step."""
        if not hasattr(self, "_prepared_positions") or self._prepared_positions is None:
            raise RuntimeError(
                f"Mobility {self.__class__.__name__} not prepared. Call prepare() first."
            )

        requested_step = int(step)
        last_step = len(self._prepared_positions) - 1
        if requested_step < 0 or requested_step > last_step:
            # Consumers occasionally ask for a boundary-adjacent frame during
            # animation/export bookkeeping; clamp rather than failing late.
            clamped_step = min(max(requested_step, 0), last_step)
            logger.warning(
                "Requested step %d is outside prepared range [0, %d] for %s; using step %d",
                requested_step,
                last_step,
                self.__class__.__name__,
                clamped_step,
            )
            requested_step = clamped_step

        return self._prepared_positions[requested_step]
