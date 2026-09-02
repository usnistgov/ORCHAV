"""Build immutable frame metadata arrays from per-frame actor state.

The arrays produced here are stored in ``frame_data`` under keys such as
``tx_positions_snapshot`` and ``target_orientations_snapshot``. They are a
numeric record of actor state attached to an output frame, not live Sionna
objects and not CPU copies of path buffers. A coherent cached frame can retain
acquisition-time TX/RX snapshots while carrying output-time target snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from shared.logging import get_logger

from ..orientation.base import orientation_to_tuple
from ..sionna_integration import orientation_deg_to_engine_radians
from ..utils import point_to_tuple

logger = get_logger(__name__)


def positions_to_array(positions: Sequence[Any]) -> np.ndarray:
    """Convert per-entity positions into a stable ``(N, 3)`` float array.

    Missing or invalid positions become ``(0, 0, 0)`` so frame payloads keep a
    predictable shape. An empty input returns ``(0, 3)`` rather than a flat empty
    array because downstream frame converters expect coordinate columns.
    """
    snapshot = []
    for idx, position in enumerate(positions):
        if position is None:
            logger.debug("[positions_to_array] Position %d is None, using (0,0,0)", idx)
            snapshot.append((0.0, 0.0, 0.0))
            continue
        try:
            pos_tuple = point_to_tuple(position, error_type=ValueError)
            snapshot.append(pos_tuple)
            logger.debug("[positions_to_array] Position %d: %s", idx, pos_tuple)
        except (TypeError, ValueError, IndexError, AttributeError) as exc:
            logger.warning(
                "[positions_to_array] Failed to convert position %d: %s, using (0,0,0)",
                idx,
                exc,
            )
            snapshot.append((0.0, 0.0, 0.0))
    if not snapshot:
        return np.empty((0, 3), dtype=np.float64)
    result = np.asarray(snapshot, dtype=np.float64).reshape((-1, 3))
    logger.debug(
        "[positions_to_array] Created array with shape %s: %s",
        result.shape,
        result.tolist(),
    )
    return result


def orientations_to_array(orientations: Sequence[Any]) -> np.ndarray:
    """Convert project-facing degree orientations into engine-radian arrays."""
    snapshot = []
    for orientation in orientations:
        if orientation is None:
            snapshot.append((0.0, 0.0, 0.0))
            continue
        try:
            snapshot.append(orientation_deg_to_engine_radians(orientation_to_tuple(orientation)))
        except (TypeError, ValueError, IndexError):
            snapshot.append((0.0, 0.0, 0.0))
    if not snapshot:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(snapshot, dtype=np.float64).reshape((-1, 3))


def velocities_to_array(velocities: Sequence[Any] | None) -> np.ndarray | None:
    """Convert optional per-entity velocities into a stable float array.

    ``None`` means the frame had no velocity channel to store. Individual
    missing velocities become zeros so the array still lines up with the target
    ordering.
    """
    if not velocities:
        return None
    snapshot = []
    for velocity in velocities:
        if velocity is None:
            snapshot.append((0.0, 0.0, 0.0))
            continue
        try:
            snapshot.append(point_to_tuple(velocity, error_type=ValueError))
        except (TypeError, ValueError, IndexError, AttributeError):
            snapshot.append((0.0, 0.0, 0.0))
    return np.asarray(snapshot, dtype=np.float64).reshape((-1, 3))
