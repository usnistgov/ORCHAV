"""Conservative retained-payload sizing for bounded in-memory caches.

The estimator counts stable Python containers, dataclass fields, and unique
array or byte backing storage. It deliberately does not inspect arbitrary
object attributes or call conversion methods on engine-owned objects. Callers
must therefore pass any known buffers held behind an opaque wrapper as
additional roots.

The result describes known retained payload, not complete process RSS. Python,
native-library, allocator, and device-memory overhead can still exist outside
the counted objects.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def _backing_owner(value: np.ndarray | memoryview) -> Any:
    """Return the ultimate known owner of an array-like buffer."""

    owner: Any = value
    visited: set[int] = set()
    while id(owner) not in visited:
        visited.add(id(owner))
        if isinstance(owner, np.ndarray):
            base = owner.base
            if isinstance(base, (np.ndarray, memoryview, bytes, bytearray)):
                owner = base
                continue
        elif isinstance(owner, memoryview):
            base = owner.obj
            if isinstance(base, (np.ndarray, memoryview, bytes, bytearray)):
                owner = base
                continue
        break
    return owner


def _backing_nbytes(owner: Any, fallback: np.ndarray | memoryview) -> int:
    if isinstance(owner, np.ndarray):
        return int(owner.nbytes)
    if isinstance(owner, memoryview):
        return int(owner.nbytes)
    if isinstance(owner, (bytes, bytearray)):
        return len(owner)
    return int(fallback.nbytes)


def estimate_retained_bytes(*values: Any) -> int:
    """Return known bytes retained by one or more cache payload roots.

    Shared objects and shared NumPy/memoryview backing allocations are counted
    once across all roots. Recursion is limited to built-in containers and
    dataclass fields so sizing cannot unexpectedly traverse a solver or GUI
    object graph.
    """

    seen_objects: set[int] = set()
    seen_backings: set[int] = set()

    def visit(value: Any) -> int:
        if value is None:
            return 0

        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)

        if isinstance(value, (np.ndarray, memoryview)):
            owner = _backing_owner(value)
            owner_id = id(owner)
            if owner_id in seen_backings:
                return 0
            seen_backings.add(owner_id)
            return _backing_nbytes(owner, value)

        if isinstance(value, (bytes, bytearray)):
            if object_id in seen_backings:
                return 0
            seen_backings.add(object_id)
            return sys.getsizeof(value)

        if isinstance(value, str):
            return sys.getsizeof(value)

        if isinstance(value, Mapping):
            return sys.getsizeof(value) + sum(
                visit(key) + visit(item) for key, item in value.items()
            )

        if isinstance(value, (tuple, list, set, frozenset)):
            return sys.getsizeof(value) + sum(visit(item) for item in value)

        if not isinstance(value, type) and is_dataclass(value):
            return sys.getsizeof(value) + sum(
                visit(getattr(value, field.name)) for field in fields(value)
            )

        try:
            return sys.getsizeof(value)
        except TypeError:
            return 0

    return sum(visit(value) for value in values)


__all__ = ["estimate_retained_bytes"]
