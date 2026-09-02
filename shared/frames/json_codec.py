"""JSON codec for small frame metadata containing NumPy values.

HDF5 v2 keeps large numeric products in dedicated datasets. Small, irregular
metadata such as provenance, beamforming configuration, and target properties
still benefits from JSON, but those dictionaries occasionally contain NumPy
scalars or arrays. This codec preserves such values without pickle.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

_NDARRAY_MARKER = "__orchav_ndarray__"
_COMPLEX_MARKER = "__orchav_complex__"


def _encode_value(value: Any) -> Any:
    """Return a JSON-compatible representation of *value*."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _NDARRAY_MARKER: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return {
            _COMPLEX_MARKER: True,
            "real": float(value.real),
            "imag": float(value.imag),
        }
    if isinstance(value, Mapping):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _decode_value(value: Any) -> Any:
    """Restore NumPy values produced by :func:`_encode_value`."""
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get(_NDARRAY_MARKER) is True:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(size) for size in value["shape"])
        raw = base64.b64decode(value["data"], validate=True)
        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected_size:
            raise ValueError(
                "Encoded ndarray byte count does not match its dtype and shape "
                f"({len(raw)} != {expected_size})"
            )
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if value.get(_COMPLEX_MARKER) is True:
        return complex(float(value["real"]), float(value["imag"]))
    return {str(key): _decode_value(item) for key, item in value.items()}


def dumps_frame_json(value: Any) -> str:
    """Serialize small frame metadata to compact deterministic JSON."""
    return json.dumps(
        _encode_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    )


def loads_frame_json(raw: str | bytes) -> Any:
    """Deserialize metadata written by :func:`dumps_frame_json`."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return _decode_value(json.loads(raw))


__all__ = ["dumps_frame_json", "loads_frame_json"]
