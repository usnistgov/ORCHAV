"""CPU-frozen copies of Sionna RT path buffers.

``FrozenPaths`` is about solver output, not actor state. It is used after a
frame has been computed, when the frame may be cached or shared beyond the
life of the live DrJit/Sionna buffers returned by the path solver.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

# Keep the subset of Sionna path attributes consumed by frame conversion,
# metrics, and visualization paths. Each copied value is wrapped below so code
# that expects ``paths.attr.numpy()`` continues to work on frozen frames.
_FROZEN_PATH_ATTRS = (
    "valid",
    "tau",
    "a",
    "phi_t",
    "theta_t",
    "phi_r",
    "theta_r",
    "doppler",
    "vertices",
    "interactions",
    "objects",
)


class _ArrayHolder:
    """Wrap a NumPy array with the ``.numpy()`` interface path consumers expect."""

    __slots__ = ("_array",)

    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)

    def numpy(self) -> np.ndarray:
        return self._array

    def __array__(self, dtype=None):
        return np.asarray(self._array, dtype=dtype)


class FrozenPaths:
    """CPU-only representation of a Sionna RT PathsBuffer.

    Sionna path results are live backend objects backed by DrJit allocations.
    They are fine while a frame is being computed, but cached ORCHAV frames can
    be held for many requested frames or sent to multiple consumers after the
    solver call has returned. Keeping those live buffers in the cache pins GPU
    allocator memory and couples cached data to backend object lifetime.

    Freezing copies only the path attributes consumed by frame conversion,
    metrics, and visualization into CPU NumPy arrays. The wrapper keeps the
    ``paths.attr.numpy()`` access pattern working for downstream code, but this
    object is intentionally not a full live ``Paths`` replacement. Callers that
    need backend methods such as ``paths.cir(...)`` must run them before freezing.
    """

    __slots__ = ("total_paths", "_is_frozen_paths", "__dict__")

    def __init__(self, source: Any) -> None:
        self._is_frozen_paths = True
        for attr in _FROZEN_PATH_ATTRS:
            array = _extract_numpy_array(getattr(source, attr, None))
            if array is None:
                continue
            setattr(self, attr, _ArrayHolder(array))
        total_paths = getattr(source, "total_paths", None)
        if total_paths is not None:
            try:
                self.total_paths = int(total_paths)
            except (TypeError, ValueError):
                self.total_paths = total_paths
        else:
            self.total_paths = None

    def iter_arrays(self) -> Iterator[np.ndarray]:
        """Yield the NumPy buffers retained by this frozen path result."""

        for attr in _FROZEN_PATH_ATTRS:
            holder = getattr(self, attr, None)
            if isinstance(holder, _ArrayHolder):
                yield holder.numpy()


def _extract_numpy_array(value: Any) -> np.ndarray | None:
    """Return a CPU NumPy array for one path attribute, or ``None`` if absent."""
    if value is None:
        return None
    try:
        if hasattr(value, "numpy"):
            value = value.numpy()
    except (RuntimeError, TypeError, ValueError):
        return None
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


def freeze_frame_paths(frame_data: dict[str, Any], flush_gpu: bool = True) -> None:
    """Replace live Sionna path buffers with CPU-frozen data in ``frame_data``.

    This is an in-place cache preparation step. It should run only after sensing,
    CIR construction, or any other code that needs the live Sionna ``paths``
    object has finished. After that point the frame cache only needs stable array
    data, and moving those arrays to CPU prevents cached frames from retaining
    large DrJit/GPU allocations across subsequent frame requests.
    """
    paths = frame_data.get("paths")
    if paths is None or isinstance(paths, FrozenPaths):
        return
    frame_data["paths"] = FrozenPaths(paths)

    if flush_gpu:
        try:
            import drjit as dr

            # Once path arrays have been copied to CPU memory, cached frames no
            # longer need DrJit's allocator to hold onto the original buffers.
            dr.flush_malloc_cache()
        except (ImportError, RuntimeError):
            pass
