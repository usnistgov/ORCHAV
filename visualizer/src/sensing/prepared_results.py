"""Persist the fixed numeric models used by sensing figures.

The codec accepts only the display models listed here. It stores numeric arrays
separately from JSON metadata and never imports types named by a cache file.
RecordingCache owns source validation, atomic writes and the disk budget.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from math import prod
from typing import Any

import numpy as np

from shared.sensing.signal_processing import RangeDopplerResult, RangeProfileResult
from shared.sensing.stage_inspection import (
    DetectionLink,
    DetectionStageBundle,
    DetectionStageEvent,
    DetectionStageProduct,
    InspectionDisposition,
    LineageNode,
    LineageTrace,
    ProductTimeSpan,
)

from .prepared_angle import PreparedAngle
from .processing_inspection import ProcessingInspection
from .recording_cache import RecordingCache

PREPARATION_VERSION = 2
_MODELS = {
    model.__name__: model
    for model in (
        RangeProfileResult,
        RangeDopplerResult,
        PreparedAngle,
        ProcessingInspection,
        DetectionStageBundle,
        DetectionStageProduct,
        DetectionStageEvent,
        ProductTimeSpan,
        LineageNode,
        LineageTrace,
        DetectionLink,
    )
}
_CONTAINERS = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}


def _array_storage_key(array: np.ndarray) -> tuple:
    """Share only complete contiguous views of the same immutable byte buffer."""
    owner = array
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    if isinstance(owner, memoryview):
        owner = owner.obj
    if (
        isinstance(owner, bytes)
        and array.nbytes == len(owner)
        and (array.flags.c_contiguous or array.flags.f_contiguous)
    ):
        return ("bytes", id(owner), array.dtype.str, array.nbytes)
    return ("array", id(array))


def _encode(value: Any, arrays: dict[str, np.ndarray], identities: dict[tuple, str]) -> Any:
    if isinstance(value, np.ndarray):
        storage = _array_storage_key(value)
        name = identities.get(storage)
        if name is None:
            name = str(len(arrays))
            identities[storage] = name
            arrays[name] = value
        return {
            "type": "array",
            "value": {
                "name": name,
                "shape": list(value.shape),
                "order": "F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C",
            },
        }
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is InspectionDisposition:
        return {"type": "InspectionDisposition", "value": value.value}
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "value": [
                [_encode(key, arrays, identities), _encode(item, arrays, identities)]
                for key, item in value.items()
            ],
        }
    kind = type(value).__name__
    if _CONTAINERS.get(kind) is type(value):
        return {"type": kind, "value": [_encode(item, arrays, identities) for item in value]}
    if _MODELS.get(kind) is type(value):
        return {
            "type": kind,
            "value": {
                field.name: _encode(getattr(value, field.name), arrays, identities)
                for field in fields(value)
            },
        }
    raise TypeError(f"Unsupported prepared sensing value: {type(value).__name__}")


def _decode(value: Any, arrays: dict[str, np.ndarray], views: dict[tuple, np.ndarray]) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise ValueError("Invalid prepared sensing value")
    kind, payload = value["type"], value["value"]
    if kind == "array":
        if not isinstance(payload, dict) or set(payload) != {"name", "shape", "order"}:
            raise ValueError("Invalid prepared sensing array reference")
        name, shape, order = payload["name"], payload["shape"], payload["order"]
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or order not in {"C", "F"}
        ):
            raise ValueError("Invalid prepared sensing array shape or order")
        array = arrays[name]
        if prod(shape) != array.size:
            raise ValueError("Prepared sensing array shape does not match its storage")
        key = (name, tuple(shape), order)
        if key not in views:
            views[key] = array.ravel(order="K").reshape(shape, order=order)
        return views[key]
    if kind == "InspectionDisposition":
        return InspectionDisposition(payload)
    if kind == "dict" and isinstance(payload, list):
        return {_decode(key, arrays, views): _decode(item, arrays, views) for key, item in payload}
    if kind in _CONTAINERS and isinstance(payload, list):
        return _CONTAINERS[kind](_decode(item, arrays, views) for item in payload)
    model = _MODELS.get(kind)
    if model is None or not isinstance(payload, dict):
        raise ValueError("Unknown prepared sensing model")
    if set(payload) != {field.name for field in fields(model)}:
        raise ValueError("Prepared sensing model fields do not match this version")
    return model(**{name: _decode(item, arrays, views) for name, item in payload.items()})


def load_prepared(
    cache: RecordingCache | None,
    kind: str,
    key: tuple,
    *,
    accept: Callable[[Any], bool] | None = None,
) -> Any | None:
    """Restore known display models, or let the caller rebuild missing data."""
    if cache is None:
        return None
    if accept is not None:
        selected = None

        def accepts_candidate(metadata, arrays):
            nonlocal selected
            try:
                value = _decode(metadata["payload"], arrays, {})
                if not accept(value):
                    return False
            except (KeyError, TypeError, ValueError, RecursionError):
                return False
            selected = value
            return True

        loaded = cache.load_arrays(kind, (PREPARATION_VERSION, *key), accept=accepts_candidate)
        return selected if loaded is not None else None
    loaded = cache.load_arrays(kind, (PREPARATION_VERSION, *key))
    if loaded is None:
        return None
    metadata, arrays = loaded
    try:
        return _decode(metadata["payload"], arrays, {})
    except (KeyError, TypeError, ValueError, RecursionError):
        return None


def save_prepared(cache: RecordingCache | None, kind: str, key: tuple, value: Any) -> bool:
    """Store display models, sharing whole views of immutable array buffers."""
    if cache is None:
        return False
    arrays: dict[str, np.ndarray] = {}
    try:
        metadata = {"payload": _encode(value, arrays, {})}
    except (TypeError, ValueError, RecursionError):
        return False
    return cache.save_arrays(kind, metadata, arrays, (PREPARATION_VERSION, *key))
