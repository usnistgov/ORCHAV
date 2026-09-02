"""Build notebook visual payloads through the shared provider contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..io.frame_sources import FrameSource
from ..io.packed_frame_payload import (
    frame_source_provider,
    projection_to_visual_frame,
    visual_frame_read_request,
)


def load_notebook_visual_frame(
    frame_source: FrameSource,
    step: int,
    *,
    tx_positions: Sequence[Sequence[float]] | None = None,
    rx_positions: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Load one frame projection and return the notebook visual payload.

    Position overrides replace node-marker coordinates in the fresh payload.
    Canonical path geometry remains the geometry stored for the frame.
    """
    provider = frame_source_provider(frame_source)
    if provider is None:
        raise RuntimeError("Notebook frame source does not expose a DataProvider")

    projection = provider.load_frame_projection(step, visual_frame_read_request())
    payload = projection_to_visual_frame(projection)
    if tx_positions is not None:
        payload["tx_positions"] = np.asarray(tx_positions, dtype=np.float32)
    if rx_positions is not None:
        payload["rx_positions"] = np.asarray(rx_positions, dtype=np.float32)
    return payload


__all__ = ["load_notebook_visual_frame"]
