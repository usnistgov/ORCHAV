"""Map live or frozen Sionna target IDs onto retained packed bounce identities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from shared.frames.sensing_identity import validate_sensing_identity
from shared.frames.sionna_metadata import SIONNA_INTERACTION_SENSING


def pack_sensing_identity(
    paths: Any,
    raw_interactions: np.ndarray,
    retained_indices: Sequence[np.ndarray],
    packed_interactions: np.ndarray,
    targets_metadata: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Preserve identity after validity, power and Top-K path filtering."""
    has_sensing = np.any(packed_interactions == SIONNA_INTERACTION_SENSING)
    if provenance is None:
        if has_sensing:
            raise ValueError("Sensing paths require the solve-time native target catalog")
        return None
    payload = dict(provenance)
    payload.update(bounce_indices=[], target_indices=[], scattering_point_indices=[])
    if has_sensing:
        try:
            objects = np.asarray(paths.objects.numpy())
            primitives = np.asarray(paths.primitives.numpy())
        except AttributeError as exc:
            raise ValueError("Sensing paths require retained objects and primitives") from exc
        if objects.shape != raw_interactions.shape or primitives.shape != raw_interactions.shape:
            raise ValueError("Sensing identity tensors must align with interactions")
        catalog = payload.get("target_catalog_frame_indices", ())
        _, num_rx, num_tx, _ = raw_interactions.shape
        bounce_index = 0
        pair_index = 0
        for tx_index in range(num_tx):
            for rx_index in range(num_rx):
                for path_index in retained_indices[pair_index]:
                    for depth in np.flatnonzero(
                        raw_interactions[:, rx_index, tx_index, path_index] > 0
                    ):
                        if (
                            raw_interactions[depth, rx_index, tx_index, path_index]
                            == SIONNA_INTERACTION_SENSING
                        ):
                            target = int(objects[depth, rx_index, tx_index, path_index])
                            point = int(primitives[depth, rx_index, tx_index, path_index])
                            if target < 0 or target >= len(catalog):
                                raise ValueError(
                                    "Sensing path references an unknown solve-time target"
                                )
                            payload["bounce_indices"].append(bounce_index)
                            payload["target_indices"].append(int(catalog[target]))
                            payload["scattering_point_indices"].append(point)
                        bounce_index += 1
                pair_index += 1
    validate_sensing_identity(payload, packed_interactions, targets_metadata)
    return payload
