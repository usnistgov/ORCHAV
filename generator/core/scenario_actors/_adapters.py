"""Small structural adapters between the pose kernel and shared schema types.

The shared schema owns configuration classes. This package uses
attribute/Mapping access instead of defining a second set of schema models, so
the pure evaluator also accepts small structural test specifications.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, SupportsFloat, SupportsIndex

_MISSING = object()


def value(
    obj: object,
    name: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """Read a field from a Pydantic model, dataclass, or mapping."""

    if isinstance(obj, Mapping) and name in obj:
        return obj[name]
    if hasattr(obj, name):
        return getattr(obj, name)
    if default is not _MISSING:
        return default
    raise KeyError(name)


def discriminator(spec: object) -> str:
    """Return a normalized ``type`` discriminator."""

    raw = value(spec, "type")
    if isinstance(raw, Enum):
        raw = raw.value
    return str(raw).strip().lower()


def sequence(value_to_check: object, *, name: str) -> tuple[object, ...]:
    """Normalize a non-string sequence."""

    if isinstance(value_to_check, (str, bytes)) or not isinstance(value_to_check, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value_to_check)


def number(value_to_check: object, *, name: str) -> float:
    """Normalize a scalar accepted by Python's numeric conversion protocol."""

    if not isinstance(value_to_check, (str, bytes, SupportsFloat, SupportsIndex)):
        raise TypeError(f"{name} must be numeric")
    return float(value_to_check)


def vec3(value_to_check: object, *, name: str) -> tuple[float, float, float]:
    """Normalize schema vector representations to a finite float triple."""

    root = getattr(value_to_check, "root", _MISSING)
    if root is not _MISSING:
        value_to_check = root

    raw: tuple[object, ...]
    if isinstance(value_to_check, Mapping):
        if all(axis in value_to_check for axis in ("x", "y", "z")):
            raw = (
                value_to_check["x"],
                value_to_check["y"],
                value_to_check["z"],
            )
        elif all(axis in value_to_check for axis in ("right", "forward", "up")):
            raw = (
                value_to_check["right"],
                value_to_check["forward"],
                value_to_check["up"],
            )
        else:
            raise ValueError(f"{name} must contain three coordinates")
    elif all(hasattr(value_to_check, axis) for axis in ("x", "y", "z")):
        raw = tuple(getattr(value_to_check, axis) for axis in ("x", "y", "z"))
    elif all(hasattr(value_to_check, axis) for axis in ("right", "forward", "up")):
        raw = tuple(getattr(value_to_check, axis) for axis in ("right", "forward", "up"))
    else:
        raw = sequence(value_to_check, name=name)

    if len(raw) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    parsed = tuple(number(component, name=name) for component in raw)
    if not all(_is_finite(component) for component in parsed):
        raise ValueError(f"{name} must contain only finite coordinates")
    return (parsed[0], parsed[1], parsed[2])


def numeric_pair(value_to_check: object, *, name: str) -> tuple[float, float]:
    """Normalize a two-value numeric sequence."""

    raw = sequence(value_to_check, name=name)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    parsed = tuple(number(component, name=name) for component in raw)
    if not all(_is_finite(component) for component in parsed):
        raise ValueError(f"{name} must contain only finite values")
    return (parsed[0], parsed[1])


def _is_finite(number: float) -> bool:
    return number == number and number not in (float("inf"), float("-inf"))
