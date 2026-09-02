"""Minimal provider projection used by trajectory-history scans."""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.frames import FrameComponent, FrameReadRequest
from shared.frames.provider_base import DataProvider

TRAJECTORY_READ_REQUEST = FrameReadRequest(
    components=frozenset(
        {
            FrameComponent.DEVICES,
            FrameComponent.TARGETS,
        }
    )
)
"""Only frame columns consumed by ``TrajectoryLoadService``."""


def try_load_packed_trajectory_frame(
    provider: DataProvider | None,
    step: int,
) -> dict[str, Any] | None:
    """Return position-only data or request a non-provider source fallback."""
    if provider is None:
        return None
    try:
        projection = provider.load_frame_projection(step, TRAJECTORY_READ_REQUEST)
    except NotImplementedError:
        return None
    frame = projection.frame
    tx_positions = np.asarray(frame.tx_positions)
    rx_positions = np.asarray(frame.rx_positions)
    target_positions = np.asarray(frame.target_positions_m)
    return {
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "target_pos": target_positions,
        "targets_metadata": [dict(item) for item in frame.targets_metadata or ()],
        "num_tx": len(tx_positions),
        "num_rx": len(rx_positions),
        "num_targets": len(target_positions),
    }


__all__ = ["TRAJECTORY_READ_REQUEST", "try_load_packed_trajectory_frame"]
