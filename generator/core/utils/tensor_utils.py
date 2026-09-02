"""Tensor and scalar extraction helpers for generator runtime backends.

Sionna/Mitsuba calls can return TensorFlow tensors, DrJit arrays, Mitsuba scalar
types, NumPy arrays, or nested Python sequences depending on the runtime path
and installed backend versions. These helpers keep that normalization local so
coverage, runtime analysis, frame writing, and visualization summaries can work with
plain NumPy arrays and Python scalars.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

Scalar = int | float


def to_numpy(value: Any, flatten: bool = False) -> np.ndarray:
    """
    Convert tensor-like values to a NumPy array.

    Args:
        value: Input value, such as a TensorFlow tensor, DrJit array, Mitsuba
            scalar/vector, Python sequence, or NumPy array.
        flatten: If True, flatten the result to 1D.

    Returns:
        NumPy array representation of the input.

    Examples:
        >>> to_numpy(mi.Float(1.5))
        array([1.5])
        >>> to_numpy([1.0, 2.0, 3.0])
        array([1., 2., 3.])
    """
    if isinstance(value, (list, tuple)):
        # Backend APIs sometimes expose compound values as Python sequences of
        # tensor-like parts. Normalize each part first so callers receive one
        # contiguous NumPy view instead of an object array.
        arrays = [np.atleast_1d(to_numpy(v)) for v in value]
        result = np.concatenate(arrays, axis=0) if arrays else np.empty((0,))
    elif hasattr(value, "numpy"):
        # TensorFlow tensors and DrJit arrays both advertise a numpy bridge.
        result = np.asarray(value.numpy())
    else:
        result = np.asarray(value)

    if flatten:
        result = result.reshape(-1)

    return result


def to_scalar(value: Any, dtype: Callable[[Any], Scalar] = float) -> Scalar:
    """
    Convert a tensor-like value to a Python scalar.

    For arrays and tensor-like values, the first flattened element is the scalar
    value. This matches call sites that unwrap one-element backend tensors, but
    it is intentionally not a strict coordinate validator.

    Args:
        value: Input value to convert.
        dtype: Target Python type (float or int). Default is float.

    Returns:
        Python scalar of the specified type. Values that cannot be converted
        fall back to ``dtype(0)``. This permissive policy is for optional
        backend scalar extraction; required coordinates use strict point
        validation instead.

    Examples:
        >>> to_scalar(mi.Float(1.5))
        1.5
        >>> to_scalar(np.array([42]), dtype=int)
        42
    """
    try:
        return dtype(value)
    except (TypeError, ValueError):
        pass

    # TensorFlow and DrJit values usually take this path without importing
    # either backend into the utility module.
    try:
        if hasattr(value, "numpy"):
            arr = value.numpy()
            if hasattr(arr, "size") and arr.size:
                return dtype(arr.reshape(-1)[0])
    except (TypeError, ValueError, AttributeError):
        pass

    # NumPy conversion handles NumPy scalar types plus backend values that
    # implement the array protocol.
    try:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size:
            return dtype(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        pass

    # Some backend scalar reprs look like "[1.0]" even when direct conversion is
    # unavailable. Keep this permissive path for existing runtime compatibility.
    try:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].split(",")[0]
            return dtype(inner)
    except (TypeError, ValueError):
        pass

    # Required coordinates use point_to_tuple(), which raises instead of
    # applying this permissive optional-scalar default.
    return dtype(0)


def to_float(value: Any) -> float:
    """Convert a tensor-like scalar to ``float`` using the standard fallback."""
    return float(to_scalar(value, float))
