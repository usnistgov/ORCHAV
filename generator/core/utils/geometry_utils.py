"""Geometry conversion helpers shared at generator package boundaries.

Scenario YAML, mobility/orientation code, Sionna/Mitsuba assignment objects,
and NumPy/tensor outputs all represent positions slightly differently. This
module normalizes those values into plain Python ``(x, y, z)`` tuples before
they cross into configuration validation, frame writing, or scene assignment.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .tensor_utils import to_float


def _strict_float(value: Any) -> float:
    """Convert one coordinate to ``float`` and fail if it is not scalar-like."""
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    try:
        if hasattr(value, "numpy"):
            arr = np.asarray(value.numpy(), dtype=np.float64).reshape(-1)
        else:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f"Expected scalar coordinate, got {arr.size} values")
        return float(to_float(arr[0]))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot convert {type(value)!r} to float") from exc


def point_to_tuple(
    value: Any,
    *,
    component_converter: Callable[[Any], float] = _strict_float,
    attribute_converter: Callable[[Any], float] | None = None,
    error_type: type[Exception] = ValueError,
) -> tuple[float, float, float]:
    """Normalize a supported 3D point representation to ``(x, y, z)``.

    Accepted inputs are objects with ``x/y/z`` attributes, three-value Python
    sequences, NumPy arrays, and tensor-like objects exposing ``.numpy()``.
    Unlike ``to_scalar``, this helper is strict by default: a bad coordinate or
    wrong-length vector raises ``error_type`` because silently moving a device
    or target to the origin would hide configuration mistakes.
    """
    if value is None:
        raise error_type("Cannot convert None to (x, y, z) tuple")

    attribute_converter = attribute_converter or component_converter

    try:
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
            # Engine point classes expose coordinates as attributes; callers can
            # opt into a separate converter when those attributes need fallback
            # behavior that sequence inputs should not receive.
            return (
                attribute_converter(value.x),
                attribute_converter(value.y),
                attribute_converter(value.z),
            )

        if hasattr(value, "numpy"):
            arr = np.asarray(value.numpy()).reshape(-1)
        else:
            arr = np.asarray(value).reshape(-1)

        if arr.size != 3:
            raise ValueError(f"Expected exactly 3 coordinates, got {arr.size}")
        return (
            component_converter(arr[0]),
            component_converter(arr[1]),
            component_converter(arr[2]),
        )
    except (TypeError, ValueError, IndexError, AttributeError) as exc:
        raise error_type(f"Cannot convert {type(value)!r} to (x, y, z) tuple") from exc
