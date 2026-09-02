"""Orientation boundary types for scripted generator integrations.

Shared scenario orientation specifications and the quaternion evaluator in
``generator.core.scenario_actors`` own all orientation semantics. This module
only normalizes degree-valued triples at Sionna-facing boundaries and defines
the small protocol implemented by already-prepared or measurement-backed
sources.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np

Orientation3: TypeAlias = tuple[float, float, float]


def orientation_to_tuple(value: Any, name: str = "orientation") -> Orientation3:
    """Convert a yaw/pitch/roll input into one finite three-value tuple."""

    if value is None:
        raise ValueError(f"{name} cannot be None")

    try:
        if hasattr(value, "numpy"):
            array = np.asarray(value.numpy(), dtype=np.float64).reshape(-1)
        else:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot convert {name} to yaw/pitch/roll tuple") from exc

    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite values")

    return (float(array[0]), float(array[1]), float(array[2]))


@runtime_checkable
class PreparedOrientationSource(Protocol):
    """Protocol for data sources that materialize canonical degree samples.

    Schema-backed scripted actors should pass a shared ``OrientationSpec``
    directly. This protocol exists only for sources whose samples come from an
    external dataset or from an already-prepared canonical scenario.
    """

    def prepare(
        self,
        steps: int,
        duration: float,
        context: object | None = None,
    ) -> None:
        """Prepare exactly ``steps`` samples for the requested timeline."""

    def orientations(self) -> Iterable[Orientation3]:
        """Return the prepared yaw/pitch/roll degree samples."""
