"""Validation and inspection helpers for complete canonical MPC frames.

Construction validates ``StandardMPCFrame`` eagerly. The explicit validation
helper is intended for package boundaries that receive a frame object from
foreign or dynamically typed code; writers and ordinary consumers can trust a
successfully constructed frame without repeating the array scan.
"""

from __future__ import annotations

import numpy as np

from .types import (
    FRAME_FORMAT_VERSION,
    PATH_METRIC_ARRAY_FIELDS,
    StandardMPCFrame,
)

FRAME_SCHEMA_VERSION = FRAME_FORMAT_VERSION


def validate_standard_mpc_frame(
    frame: object,
    *,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate a complete ``StandardMPCFrame`` and return any errors.

    Args:
        frame: Candidate canonical frame. Mapping-like objects are not accepted.
        raise_on_error: Raise the first ``ValueError`` instead of returning it.

    Returns:
        An empty list for a valid frame, otherwise one validation message.
    """

    if not isinstance(frame, StandardMPCFrame):
        message = "frame must be a complete StandardMPCFrame instance"
        if raise_on_error:
            raise ValueError(message)
        return [message]
    try:
        frame.validate()
    except ValueError as exc:
        if raise_on_error:
            raise
        return [str(exc)]
    return []


def is_valid_standard_mpc_frame(frame: object) -> bool:
    """Return whether ``frame`` satisfies the complete canonical contract."""

    return not validate_standard_mpc_frame(frame, raise_on_error=False)


def count_frame_mpcs(frame: StandardMPCFrame) -> int:
    """Return the number of physical propagation paths in ``frame``."""

    if not isinstance(frame, StandardMPCFrame):
        raise TypeError("frame must be a complete StandardMPCFrame instance")
    num_paths = frame.num_paths
    assert num_paths is not None
    return num_paths


def summarize_frame(frame: StandardMPCFrame) -> str:
    """Return a compact human-readable summary of a complete frame."""

    if not isinstance(frame, StandardMPCFrame):
        raise TypeError("frame must be a complete StandardMPCFrame instance")
    num_pairs = frame.num_pairs
    num_paths = frame.num_paths
    assert num_pairs is not None
    assert num_paths is not None
    lines = [
        f"TX: {frame.num_tx}, RX: {frame.num_rx}, Targets: {frame.num_targets}",
        f"Pairs: {num_pairs}",
        f"Total MPCs: {num_paths}",
    ]

    paths_per_pair = np.diff(frame.pair_path_offsets)
    if num_pairs <= 10:
        counts = ", ".join(str(int(value)) for value in paths_per_pair)
        lines.append(f"Paths per pair: [{counts}]")
    else:
        lines.append(f"Paths per pair: [{num_pairs} pairs, see data]")

    present_metrics = [
        PATH_METRIC_ARRAY_FIELDS[metric]
        for metric in frame.path_metrics
        if np.any(frame.metric_is_valid(metric))
    ]
    lines.append(f"Metrics: {', '.join(present_metrics)}" if present_metrics else "Metrics: none")
    if frame.beamforming:
        lines.append("Beamforming: yes")
    if frame.sensing:
        lines.append("Frame extension: sensing")
    return "\n".join(lines)


__all__ = [
    "FRAME_SCHEMA_VERSION",
    "count_frame_mpcs",
    "is_valid_standard_mpc_frame",
    "summarize_frame",
    "validate_standard_mpc_frame",
]
