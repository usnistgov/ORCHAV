"""Shared protobuf codec for the canonical ``StandardMPCFrame`` wire format.

The codec translates between the shared in-memory frame contract and protobuf
messages. Transport ownership stays with the caller: generator gRPC servers,
remote-HDF5 services, and clients decide when to send or receive messages.

Wire version 3 carries the compact core arrays plus target, beamforming, and
provenance metadata. Arrays use their canonical fixed dtypes as contiguous raw
bytes, and decoders recover shapes from the frame contract. Sensing payloads
are not part of this wire format; serialization rejects them rather than
silently dropping derived products. Decoding requires an exact wire-version
match.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from shared.frames.json_codec import dumps_frame_json, loads_frame_json
from shared.frames.types import StandardMPCFrame
from shared.protos import visualizer_pb2 as _visualizer_pb2

visualizer_pb2: Any = _visualizer_pb2

# The compact canonical transport requires an exact version match.
STANDARD_MPC_PROTOBUF_VERSION = 3


def _encode_fixed_array(values: np.ndarray, dtype: np.dtype[Any] | type[np.generic]) -> bytes:
    """Encode a fixed-contract canonical array without an NPZ wrapper."""

    expected = np.dtype(dtype)
    array = np.asarray(values)
    if array.dtype != expected:
        raise ValueError(f"Expected {expected} array, got {array.dtype}")
    return np.ascontiguousarray(array).tobytes(order="C")


def _decode_fixed_vector(
    payload: bytes,
    dtype: np.dtype[Any] | type[np.generic],
    name: str,
) -> np.ndarray:
    """Decode one required fixed-dtype vector and own its writable buffer."""

    expected = np.dtype(dtype)
    if len(payload) % expected.itemsize:
        raise ValueError(f"{name} byte count is not aligned to dtype {expected}")
    return np.frombuffer(payload, dtype=expected).copy()


def _decode_fixed_matrix(
    payload: bytes,
    dtype: np.dtype[Any] | type[np.generic],
    columns: int,
    name: str,
) -> np.ndarray:
    """Decode one required fixed-width matrix."""

    values = _decode_fixed_vector(payload, dtype, name)
    if values.size % columns:
        raise ValueError(f"{name} element count is not divisible by {columns}")
    return values.reshape((-1, columns))


def _decode_json_field(raw: str, name: str) -> Any:
    """Decode one required NumPy-aware JSON metadata field."""

    if not raw:
        raise ValueError(f"{name} must contain a JSON value")
    try:
        return loads_frame_json(raw)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"{name} contains invalid frame metadata JSON") from exc


def _decode_optional_mapping(raw: str, name: str) -> Mapping[str, Any] | None:
    """Decode a metadata value whose canonical type is mapping or ``None``."""

    value = _decode_json_field(raw, name)
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"{name} must decode to a mapping or None")
    return value


def _decode_targets_metadata(raw: str) -> tuple[Mapping[str, Any], ...]:
    """Decode the target-aligned metadata sequence."""

    value = _decode_json_field(raw, "targets_metadata_json")
    if not isinstance(value, list):
        raise ValueError("targets_metadata_json must decode to a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError("targets_metadata_json items must decode to mappings")
    return tuple(value)


def standard_mpc_frame_to_proto(frame: StandardMPCFrame) -> Any:
    """Serialize one complete compact frame using protobuf wire version 3.

    Raises:
        ValueError: If the frame carries a sensing payload unsupported by this
            transport.
    """

    if not isinstance(frame, StandardMPCFrame):
        raise TypeError("frame must be a complete StandardMPCFrame")
    if frame.sensing is not None:
        raise ValueError("StandardMPCFrame protobuf transport does not support sensing payloads")

    standard_frame_kwargs: dict[str, Any] = {
        "tx_rx_pairs": _encode_fixed_array(frame.tx_rx_pairs, np.int32),
        "pair_path_offsets": _encode_fixed_array(frame.pair_path_offsets, np.int64),
        "bounce_offsets": _encode_fixed_array(frame.bounce_offsets, np.int64),
        "tx_positions": _encode_fixed_array(frame.tx_positions, np.float64),
        "rx_positions": _encode_fixed_array(frame.rx_positions, np.float64),
        "tx_orientations": _encode_fixed_array(frame.tx_orientations, np.float64),
        "rx_orientations": _encode_fixed_array(frame.rx_orientations, np.float64),
        "tx_names": list(frame.tx_names),
        "rx_names": list(frame.rx_names),
        "bounce_xyz_m": _encode_fixed_array(frame.bounce_xyz_m, np.float32),
        "interactions": _encode_fixed_array(frame.interactions, np.uint8),
        "material_ids": _encode_fixed_array(frame.material_ids, np.uint16),
        "material_names": list(frame.material_names),
        "material_itu_types": list(frame.material_itu_types),
        "delays_ns": _encode_fixed_array(frame.delays_ns, np.float32),
        "path_loss_db": _encode_fixed_array(frame.path_loss_db, np.float32),
        "aoa_az_deg": _encode_fixed_array(frame.aoa_az_deg, np.float32),
        "aoa_el_deg": _encode_fixed_array(frame.aoa_el_deg, np.float32),
        "aod_az_deg": _encode_fixed_array(frame.aod_az_deg, np.float32),
        "aod_el_deg": _encode_fixed_array(frame.aod_el_deg, np.float32),
        "metric_valid_bits": _encode_fixed_array(frame.metric_valid_bits, np.uint8),
        "target_positions_m": _encode_fixed_array(frame.target_positions_m, np.float64),
        "targets_metadata_json": dumps_frame_json(frame.targets_metadata),
        "provenance_json": dumps_frame_json(frame.provenance),
        "frame_idx": frame.frame_index,
        "beamforming_json": dumps_frame_json(frame.beamforming),
        "recomputed_from_stored_positions": bool(frame.recomputed_from_stored_positions),
        "wire_format_version": STANDARD_MPC_PROTOBUF_VERSION,
    }
    if frame.timestamp_s is not None:
        standard_frame_kwargs["timestamp_s"] = frame.timestamp_s

    return visualizer_pb2.StandardMPCFrame(**standard_frame_kwargs)


def frame_data_from_standard_mpc_frame(frame: StandardMPCFrame) -> Any:
    """Wrap a complete canonical frame in a FrameData protobuf message."""
    return visualizer_pb2.FrameData(standard_mpc_frame=standard_mpc_frame_to_proto(frame))


def standard_mpc_frame_from_proto(frame_pb: Any) -> StandardMPCFrame:
    """Deserialize and validate one compact protobuf frame."""
    wire_format_version = int(frame_pb.wire_format_version)
    if wire_format_version != STANDARD_MPC_PROTOBUF_VERSION:
        raise ValueError(
            "Unsupported StandardMPCFrame protobuf wire version "
            f"{wire_format_version}; expected {STANDARD_MPC_PROTOBUF_VERSION}"
        )

    target_positions = _decode_fixed_matrix(
        frame_pb.target_positions_m, np.float64, 3, "target_positions_m"
    )
    targets_metadata = _decode_targets_metadata(frame_pb.targets_metadata_json)
    beamforming = _decode_optional_mapping(frame_pb.beamforming_json, "beamforming_json")
    provenance = _decode_optional_mapping(frame_pb.provenance_json, "provenance_json")
    timestamp_s = float(frame_pb.timestamp_s) if frame_pb.HasField("timestamp_s") else None
    return StandardMPCFrame(
        frame_index=int(frame_pb.frame_idx),
        tx_rx_pairs=_decode_fixed_matrix(frame_pb.tx_rx_pairs, np.int32, 2, "tx_rx_pairs"),
        pair_path_offsets=_decode_fixed_vector(
            frame_pb.pair_path_offsets, np.int64, "pair_path_offsets"
        ),
        bounce_offsets=_decode_fixed_vector(frame_pb.bounce_offsets, np.int64, "bounce_offsets"),
        tx_positions=_decode_fixed_matrix(frame_pb.tx_positions, np.float64, 3, "tx_positions"),
        rx_positions=_decode_fixed_matrix(frame_pb.rx_positions, np.float64, 3, "rx_positions"),
        tx_orientations=_decode_fixed_matrix(
            frame_pb.tx_orientations, np.float64, 3, "tx_orientations"
        ),
        rx_orientations=_decode_fixed_matrix(
            frame_pb.rx_orientations, np.float64, 3, "rx_orientations"
        ),
        tx_names=tuple(frame_pb.tx_names),
        rx_names=tuple(frame_pb.rx_names),
        bounce_xyz_m=_decode_fixed_matrix(frame_pb.bounce_xyz_m, np.float32, 3, "bounce_xyz_m"),
        interactions=_decode_fixed_vector(frame_pb.interactions, np.uint8, "interactions"),
        material_ids=_decode_fixed_vector(frame_pb.material_ids, np.uint16, "material_ids"),
        material_names=tuple(frame_pb.material_names),
        material_itu_types=tuple(frame_pb.material_itu_types),
        delays_ns=_decode_fixed_vector(frame_pb.delays_ns, np.float32, "delays_ns"),
        path_loss_db=_decode_fixed_vector(frame_pb.path_loss_db, np.float32, "path_loss_db"),
        aoa_az_deg=_decode_fixed_vector(frame_pb.aoa_az_deg, np.float32, "aoa_az_deg"),
        aoa_el_deg=_decode_fixed_vector(frame_pb.aoa_el_deg, np.float32, "aoa_el_deg"),
        aod_az_deg=_decode_fixed_vector(frame_pb.aod_az_deg, np.float32, "aod_az_deg"),
        aod_el_deg=_decode_fixed_vector(frame_pb.aod_el_deg, np.float32, "aod_el_deg"),
        metric_valid_bits=_decode_fixed_vector(
            frame_pb.metric_valid_bits, np.uint8, "metric_valid_bits"
        ),
        target_positions_m=target_positions,
        targets_metadata=targets_metadata,
        timestamp_s=timestamp_s,
        beamforming=beamforming,
        provenance=provenance,
        recomputed_from_stored_positions=bool(frame_pb.recomputed_from_stored_positions),
    )
