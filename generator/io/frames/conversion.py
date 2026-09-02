"""Raw-to-canonical frame normalization for generator output.

In this module, a *raw frame* is the dictionary returned by live generation:
``RayTracingService.compute_step`` or a dispatcher cache. It is generator-owned
state, not a stable interchange format. Typical keys include ``paths``,
``tx_list``, ``rx_list``, target objects/managers, material mapping, optional
sensing blocks, and streaming pose snapshots.

``standard_mpc_frame_from_raw`` is the common boundary for file mode and gRPC
mode. It turns that raw dictionary into ``shared.frames.types.StandardMPCFrame``,
binds its pose snapshots and acquisition provenance, and returns one validated
compact value to storage or transport.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.frames.types import StandardMPCFrame

__all__ = ["standard_mpc_frame_from_raw"]


def standard_mpc_frame_from_raw(
    frame_data: dict[str, Any],
    frame_idx: int,
    *,
    source_provider: str,
    simulation_config: Any | None = None,
    path_filter_config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    bandwidth_hz: float | None = None,
    timestamp: float | None = None,
) -> StandardMPCFrame:
    """Normalize raw generator frame data to a canonical ``StandardMPCFrame``.

    Args:
        frame_data: Raw generator dictionary containing live Sionna paths and
            scene objects, or a coherent cached-frame marker.
        frame_idx: Zero-based output-frame index.
        source_provider: Provenance label such as ``"generator_file"`` or
            ``"generator_grpc"``.
        simulation_config: Optional simulation settings used while extracting
            per-path metrics.
        path_filter_config: Optional path-loss/Top-K filter applied before the
            frame is serialized.
        output_dir: Optional diagnostic-output directory for path filtering.
        bandwidth_hz: Optional bandwidth used for path-filter diagnostics.
        timestamp: Optional provenance timestamp.
    Returns:
        Validated ``StandardMPCFrame`` ready for HDF5 writing or protobuf
        serialization.
    """
    if "tx_list" not in frame_data or "rx_list" not in frame_data:
        raise ValueError("Raw frame data requires TX/RX object lists")

    from .builder import process_cached_frame_data, process_frame_data

    tx_list = frame_data.get("tx_list", [])
    rx_list = frame_data.get("rx_list", [])
    target_objects = frame_data.get("target_objects", [])
    target_managers = frame_data.get("target_managers", [])
    sensing_data = frame_data.get("sensing")
    simulation_config = simulation_config or frame_data.get("simulation_config")

    source = {"provider": source_provider, "frame_idx": int(frame_idx)}
    source_rt_frame_idx = frame_data.get("_cached_rt_source_frame_idx")
    if source_rt_frame_idx is None and frame_data.get("paths") is not None:
        source_rt_frame_idx = frame_data.get("frame_idx", frame_idx)
    if source_rt_frame_idx is not None:
        source["source_rt_frame_idx"] = int(source_rt_frame_idx)
    if timestamp is not None:
        source["timestamp"] = float(timestamp)
    common_kwargs = {
        "sensing_data": sensing_data,
        "snapshot_overrides": frame_data,
        "provenance": source,
        "timestamp_s": timestamp,
        "beamforming": frame_data.get("beamforming"),
        "recomputed_from_stored_positions": bool(
            frame_data.get("_recomputed_from_stored_positions", False)
        ),
    }

    if bool(frame_data.get("_coherent_cached_frame")) and sensing_data is None:
        standard_frame = process_cached_frame_data(
            frame_idx,
            tx_list,
            rx_list,
            target_objects,
            target_managers,
            **common_kwargs,
        )
    else:
        paths = frame_data.get("paths")
        if paths is None:
            raise ValueError("Raw frame data requires a 'paths' object")
        standard_frame = process_frame_data(
            frame_idx,
            tx_list,
            rx_list,
            paths,
            target_objects,
            target_managers,
            simulation_config,
            frame_data.get("material_mapping"),
            path_filter_config=path_filter_config,
            output_dir=output_dir,
            bandwidth_hz=bandwidth_hz,
            **common_kwargs,
        )

    return standard_frame
