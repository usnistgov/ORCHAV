"""Versioned exact numeric components split across bounded safe cache archives.

The manifest is published last. Each part uses RecordingCache's ordinary source,
dtype, shape and archive validation; partitioning never raises its size ceiling.
Consumers still validate their own scientific metadata and coverage contract.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from math import prod
from uuid import uuid4

import numpy as np

from . import recording_cache

COMPONENT_VERSION = 1


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()


def _identity(protocol, key):
    return (COMPONENT_VERSION, protocol, *key)


def _part_key(identity, name, part):
    return (*identity, name, part)


def save_component(cache, kind, key, metadata, arrays, *, protocol, cancelled=None):
    """Save exact numeric parts and commit their manifest only after readback."""
    if cache is None:
        return False
    _check_cancelled(cancelled)
    identity = _identity(protocol, key)
    generation = uuid4().hex
    part_identity = (*identity, generation)
    try:
        if not isinstance(metadata, dict) or not isinstance(arrays, dict):
            return False
        descriptors = {}
        paths = []
        limit = recording_cache.MAX_ARRAY_ENTRY_BYTES
        # Leave ample room for ZIP/NPY headers and the small part envelope.
        payload_limit = limit - min(65536, limit // 2)
        for name, array in arrays.items():
            _check_cancelled(cancelled)
            if (
                not isinstance(name, str)
                or not isinstance(array, np.ndarray)
                or array.dtype.kind not in "biufc"
                or array.dtype.itemsize > payload_limit
            ):
                return False
            flat = np.ravel(array, order="C")
            count = max(1, payload_limit // array.dtype.itemsize)
            parts = max(1, (flat.size + count - 1) // count)
            descriptors[name] = {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "parts": parts,
                "part_elements": count,
                "nbytes": array.nbytes,
            }
            for part in range(parts):
                _check_cancelled(cancelled)
                values = flat[part * count : (part + 1) * count]
                part_key = _part_key(part_identity, name, part)
                part_metadata = {"name": name, "part": part, "parts": parts}
                if not cache.save_arrays(
                    f"{kind}-part", part_metadata, {"values": values}, part_key
                ):
                    return False
                checked = cache.load_arrays(f"{kind}-part", part_key)
                if (
                    checked is None
                    or checked[0] != part_metadata
                    or set(checked[1]) != {"values"}
                    or checked[1]["values"].dtype != values.dtype
                    or checked[1]["values"].shape != values.shape
                    or checked[1]["values"].tobytes() != values.tobytes()
                ):
                    return False
                paths.append(cache._array_path(f"{kind}-part", part_key))
        _check_cancelled(cancelled)
        if not all(path.is_file() for path in paths):
            return False  # An ordinary disk quota cannot retain this component.
        return cache.save(
            f"{kind}-component",
            {"metadata": metadata, "arrays": descriptors, "generation": generation},
            identity,
        )
    except (TypeError, ValueError, OverflowError):
        return False


def load_component(cache, kind, key, *, protocol, byte_limit=None, cancelled=None):
    """Read one complete component, checking aggregate admission before parts.

    ``byte_limit`` bounds declared numeric payload, not metadata, temporary
    assembly or process RSS. The caller accounts for its full retained owner.
    Missing, stale, malformed or incomplete components are ordinary cache misses.
    """
    if cache is None:
        return None
    _check_cancelled(cancelled)
    identity = _identity(protocol, key)
    manifest = cache.load(f"{kind}-component", identity)
    if manifest is None:
        return None
    try:
        metadata, descriptors = manifest["metadata"], manifest["arrays"]
        if not isinstance(metadata, dict) or not isinstance(descriptors, dict):
            return None
        generation = manifest["generation"]
        if (
            not isinstance(generation, str)
            or len(generation) != 32
            or any(c not in "0123456789abcdef" for c in generation)
        ):
            return None
        part_identity = (*identity, generation)
        total = 0
        plans = {}
        for name, descriptor in descriptors.items():
            if not isinstance(name, str) or not isinstance(descriptor, dict):
                return None
            shape, dtype = descriptor["shape"], np.dtype(descriptor["dtype"])
            parts, count = descriptor["parts"], descriptor["part_elements"]
            if (
                not isinstance(shape, list)
                or any(type(v) is not int or v < 0 for v in shape)
                or dtype.kind not in "biufc"
                or type(parts) is not int
                or type(count) is not int
                or count <= 0
                or count * dtype.itemsize > recording_cache.MAX_ARRAY_ENTRY_BYTES
                or parts != max(1, (prod(shape) + count - 1) // count)
                or descriptor["nbytes"] != prod(shape) * dtype.itemsize
            ):
                return None
            total += descriptor["nbytes"]
            plans[name] = (shape, dtype, parts, count)
        if byte_limit is not None and total > byte_limit:
            return None
        arrays = {}
        for name, (shape, dtype, parts, count) in plans.items():
            chunks = []
            for part in range(parts):
                _check_cancelled(cancelled)
                loaded = cache.load_arrays(f"{kind}-part", _part_key(part_identity, name, part))
                if loaded is None or loaded[0] != {"name": name, "part": part, "parts": parts}:
                    return None
                if set(loaded[1]) != {"values"}:
                    return None
                values = loaded[1]["values"]
                length = min(count, max(0, prod(shape) - part * count))
                if values.dtype != dtype or values.shape != (length,):
                    return None
                chunks.append(values.tobytes())
            arrays[name] = np.frombuffer(b"".join(chunks), dtype=dtype).reshape(shape)
        _check_cancelled(cancelled)
        if not cache.unchanged():
            return None
        return metadata, arrays
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
