"""Validated sparse native sensing identities within the frame provenance envelope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .sionna_metadata import SIONNA_INTERACTION_SENSING


def validate_sensing_identity(
    payload: Mapping[str, Any],
    interactions: np.ndarray,
    targets_metadata: Sequence[Mapping[str, Any]],
) -> None:
    """Reject missing, misaligned, or ambiguous sensing target/point references.

    The versioned extension uses packed bounce indices and the frame's target
    catalog. It never overloads a scene-object/material ID with a target ID.
    Ordinary frames without this extension retain their existing contract.
    """
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported sionna_sensing identity schema")
    arrays = []
    for field in ("bounce_indices", "target_indices", "scattering_point_indices"):
        values = payload.get(field)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"sionna_sensing.{field} must be an integer sequence")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values
        ):
            raise ValueError(f"sionna_sensing.{field} must contain integers")
        arrays.append(np.asarray(values, dtype=np.int64))
    bounces, targets, points = arrays
    if len(bounces) != len(targets) or len(bounces) != len(points):
        raise ValueError("sionna_sensing identity arrays have different lengths")
    expected = np.flatnonzero(np.asarray(interactions) == SIONNA_INTERACTION_SENSING)
    if not np.array_equal(bounces, expected):
        raise ValueError("sionna_sensing must identify every retained sensing bounce exactly once")
    for target, point in zip(targets, points, strict=True):
        if target < 0 or target >= len(targets_metadata):
            raise ValueError("sionna_sensing references an unknown target")
        metadata = targets_metadata[int(target)]
        if metadata.get("representation") not in {"constant_rcs", "tr38901"}:
            raise ValueError("sionna_sensing references a non-native target")
        identifiers = metadata.get("scattering_point_ids", ())
        if point < 0 or int(point) not in identifiers:
            raise ValueError("sionna_sensing references an unknown scattering point")


def sensing_bounce_identity(frame: Any, bounce_index: int) -> tuple[int, int] | None:
    """Return a retained (frame target index, scattering point index) reference."""
    payload = (frame.provenance or {}).get("sionna_sensing")
    if payload is None:
        if frame.interactions[int(bounce_index)] == SIONNA_INTERACTION_SENSING:
            raise ValueError("Sensing bounce requires retained target/point identity")
        return None
    validate_sensing_identity(payload, frame.interactions, frame.targets_metadata)
    try:
        index = payload["bounce_indices"].index(int(bounce_index))
    except ValueError:
        return None
    return int(payload["target_indices"][index]), int(payload["scattering_point_indices"][index])
