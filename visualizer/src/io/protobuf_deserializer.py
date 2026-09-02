"""Protobuf deserialization utilities for gRPC frame data.

Pure data-transform functions that convert protobuf messages into complete
compact ``StandardMPCFrame`` objects. These functions are stateless and are
called by :class:`GrpcProvider`.
"""

import logging
from typing import Any, Optional

from shared.frames.protobuf import standard_mpc_frame_from_proto
from shared.frames.types import StandardMPCFrame

logger = logging.getLogger(__name__)

__all__ = ["frame_from_proto"]


def frame_from_proto(
    frame_pb: Any,
    frame_idx: int,
) -> Optional[StandardMPCFrame]:
    """Convert a protobuf ``FrameData`` message into a ``StandardMPCFrame``.

    Args:
        frame_pb: The protobuf frame message.
        frame_idx: Zero-based frame index.
    Returns:
        A validated ``StandardMPCFrame``, or ``None`` on failure.
    """
    try:
        if not frame_pb.HasField("standard_mpc_frame"):
            raise ValueError("FrameData is missing required standard_mpc_frame payload")
        frame = standard_mpc_frame_from_proto(frame_pb.standard_mpc_frame)
        if frame.frame_index != frame_idx:
            raise ValueError(
                f"Encoded frame index {frame.frame_index} does not match "
                f"response frame {frame_idx}"
            )
    except ValueError as exc:
        logger.error("Failed to unpack frame %s from gRPC provider: %s", frame_idx, exc)
        return None

    return frame
