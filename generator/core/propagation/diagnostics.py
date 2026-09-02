"""Debug-only diagnostics for propagation path solver output.

These helpers inspect live path results for logging only. They must not mutate
``frame_data`` or force expensive analysis unless debug logging is enabled.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shared.frames.sionna_metadata import (
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_LABELS,
    SIONNA_INTERACTION_LOS,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
)
from shared.logging import get_logger

logger = get_logger(__name__)


def log_mpc_statistics(paths: Any, frame_idx: int) -> None:
    """Log aggregate path validity and interaction counts from Sionna RT results."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    try:
        valid = paths.valid.numpy().astype(bool, copy=False)
        interactions = paths.interactions.numpy().astype(np.int32, copy=False)

        _num_rx, _num_tx, num_paths = valid.shape
        total_possible = int(valid.size)
        total_valid = int(np.count_nonzero(valid))
        total_invalid = total_possible - total_valid

        interaction_counts = {
            SIONNA_INTERACTION_LOS: 0,
            SIONNA_INTERACTION_SPECULAR: 0,
            SIONNA_INTERACTION_DIFFUSE: 0,
            SIONNA_INTERACTION_REFRACTION: 0,
            SIONNA_INTERACTION_DIFFRACTION: 0,
        }

        if total_valid > 0:
            # Move bounce axis to last position: [rx, tx, path, bounce]
            flat_interactions = np.moveaxis(interactions, 0, -1).reshape(-1, interactions.shape[0])
            flat_valid = valid.reshape(-1)
            valid_interactions = flat_interactions[flat_valid]

            los_paths = int(np.count_nonzero(np.all(valid_interactions <= 0, axis=1)))
            interaction_counts[SIONNA_INTERACTION_LOS] += los_paths

            active = valid_interactions[valid_interactions > 0]
            if active.size > 0:
                unique_vals, counts = np.unique(active, return_counts=True)
                for value, count in zip(unique_vals.tolist(), counts.tolist()):
                    if value in interaction_counts:
                        interaction_counts[int(value)] += int(count)

        valid_pct = (total_valid / total_possible * 100.0) if total_possible else 0.0
        logger.debug(
            "Frame %d MPC stats: %d paths/pair, %d valid (%.1f%%), %d invalid",
            frame_idx,
            num_paths,
            total_valid,
            valid_pct,
            total_invalid,
        )

        active_interactions = [
            f"{SIONNA_INTERACTION_LABELS.get(t, f'Type{t}')}:{c}"
            for t, c in interaction_counts.items()
            if c > 0
        ]
        if active_interactions:
            logger.debug("Interactions: %s", ", ".join(active_interactions))

    except (TypeError, ValueError, AttributeError, IndexError) as exc:
        logger.debug("Failed to analyze MPC statistics: %s", exc)
