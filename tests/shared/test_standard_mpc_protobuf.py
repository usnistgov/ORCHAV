from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.protobuf import (
    STANDARD_MPC_PROTOBUF_VERSION,
    standard_mpc_frame_from_proto,
    standard_mpc_frame_to_proto,
)
from shared.frames.types import StandardMPCFrame
from shared.protos import visualizer_pb2


def _standard_frame() -> StandardMPCFrame:
    return standard_mpc_frame_from_pair_data(
        frame_index=4,
        timestamp_s=123.5,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.5]], dtype=np.float64),
        rx_positions=np.asarray([[5.0, 0.0, 1.5]], dtype=np.float64),
        tx_orientations=np.asarray([[0.0, 0.1, 0.2]], dtype=np.float64),
        rx_orientations=np.asarray([[0.3, 0.4, 0.5]], dtype=np.float64),
        tx_names=("tx-main",),
        rx_names=("rx-main",),
        vertices_by_pair=[np.asarray([[[1.0, 1.0, 1.0]]], dtype=np.float32)],
        interactions_by_pair=[np.asarray([[1]], dtype=np.int32)],
        path_lengths_by_pair=[np.asarray([1], dtype=np.int64)],
        material_names_by_pair=[np.asarray([["concrete"]], dtype=object)],
        material_itu_types_by_pair=[np.asarray([["itu_concrete"]], dtype=object)],
        metrics_by_pair={
            "delays_ns": [np.asarray([12.5], dtype=np.float32)],
            "path_loss_db": [np.asarray([83.0], dtype=np.float32)],
            "aoa_az_deg": [np.asarray([10.0], dtype=np.float32)],
            "aoa_el_deg": [np.asarray([1.0], dtype=np.float32)],
            "aod_az_deg": [np.asarray([20.0], dtype=np.float32)],
            "aod_el_deg": [np.asarray([2.0], dtype=np.float32)],
        },
        target_positions_m=np.asarray([[7.0, 8.0, 9.0]], dtype=np.float64),
        targets_metadata=(
            {
                "name": "target-0",
                "orientation": [0.0, 0.0, 1.57],
                "mesh_file": "walker_000.ply",
                "mesh_directory": "libraries/targets/nist_human_walking",
                "mesh_index": 2,
                "scale": 1.25,
                "material_type": "itu_concrete",
                "material_id": 17,
                "mobility_type": "sampled",
                "use_ply_position": False,
                "current_position": [7.0, 8.0, 9.0],
                "external_properties": {
                    "track_covariance": np.arange(9, dtype=np.float32).reshape(3, 3),
                    "class_id": np.int16(12),
                },
            },
        ),
        beamforming={
            "mode": "adaptive",
            "quality": np.asarray([0.25, 0.75], dtype=np.float16),
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx_name": "tx-main",
                    "rx_name": "rx-main",
                    "tx": {
                        "device_name": "tx0",
                        "device_index": 0,
                        "role": "tx",
                        "array_rows": 1,
                        "array_cols": 2,
                        "weights": np.asarray([1.0 + 0.0j, 0.5 + 0.5j]),
                        "element_positions": np.asarray([[0.0, 0.0, 0.0]]),
                        "solver_details": {"iterations": 3},
                    },
                    "rx": {
                        "device_name": "rx0",
                        "device_index": 0,
                        "role": "rx",
                        "array_rows": 1,
                        "array_cols": 1,
                        "weights": np.asarray([1.0 + 0.0j]),
                        "element_positions": np.asarray([[0.0, 0.0, 0.0]]),
                    },
                }
            ],
        },
        provenance=MappingProxyType(
            {
                "producer": "unit",
                "quality_codes": np.asarray([1, 2, 3], dtype=np.uint16),
            }
        ),
        recomputed_from_stored_positions=True,
    )


def test_standard_mpc_frame_protobuf_round_trip_preserves_supported_fields() -> None:
    frame = _standard_frame()

    proto = standard_mpc_frame_to_proto(frame)
    wire = proto.SerializeToString()
    decoded = standard_mpc_frame_from_proto(visualizer_pb2.StandardMPCFrame.FromString(wire))

    assert proto.wire_format_version == STANDARD_MPC_PROTOBUF_VERSION
    assert decoded.frame_index == frame.frame_index
    assert decoded.timestamp_s == frame.timestamp_s
    for field_name in (
        "tx_rx_pairs",
        "pair_path_offsets",
        "bounce_offsets",
        "tx_positions",
        "rx_positions",
        "tx_orientations",
        "rx_orientations",
        "bounce_xyz_m",
        "interactions",
        "material_ids",
        "delays_ns",
        "path_loss_db",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
        "metric_valid_bits",
        "target_positions_m",
    ):
        expected_array = getattr(frame, field_name)
        actual_array = getattr(decoded, field_name)
        np.testing.assert_array_equal(actual_array, expected_array)
        assert actual_array.dtype == expected_array.dtype
    assert decoded.material_names == frame.material_names
    assert decoded.material_itu_types == frame.material_itu_types
    assert decoded.tx_names == ("tx-main",)
    assert decoded.rx_names == ("rx-main",)
    assert decoded.targets_metadata[0]["name"] == "target-0"
    assert decoded.targets_metadata[0]["mesh_file"] == "walker_000.ply"
    assert decoded.targets_metadata[0]["material_id"] == 17
    assert decoded.targets_metadata[0]["mobility_type"] == "sampled"
    assert decoded.targets_metadata[0]["current_position"] == [7.0, 8.0, 9.0]
    np.testing.assert_array_equal(
        decoded.targets_metadata[0]["external_properties"]["track_covariance"],
        frame.targets_metadata[0]["external_properties"]["track_covariance"],
    )
    assert decoded.targets_metadata[0]["external_properties"]["track_covariance"].dtype == np.dtype(
        np.float32
    )
    assert decoded.targets_metadata[0]["external_properties"]["class_id"] == 12
    assert decoded.beamforming is not None
    assert decoded.beamforming["mode"] == "adaptive"
    np.testing.assert_array_equal(decoded.beamforming["quality"], frame.beamforming["quality"])
    assert decoded.beamforming["quality"].dtype == np.dtype(np.float16)
    assert decoded.beamforming["pairs"][0]["tx"]["device_name"] == "tx0"
    assert decoded.beamforming["pairs"][0]["tx_name"] == "tx-main"
    assert decoded.beamforming["pairs"][0]["rx_name"] == "rx-main"
    np.testing.assert_allclose(
        decoded.beamforming["pairs"][0]["tx"]["weights"],
        frame.beamforming["pairs"][0]["tx"]["weights"],
    )
    assert decoded.beamforming["pairs"][0]["tx"]["weights"].dtype == np.dtype(np.complex128)
    assert decoded.beamforming["pairs"][0]["tx"]["solver_details"] == {"iterations": 3}
    assert decoded.recomputed_from_stored_positions is True
    assert decoded.provenance is not None
    assert decoded.provenance["producer"] == "unit"
    np.testing.assert_array_equal(
        decoded.provenance["quality_codes"], frame.provenance["quality_codes"]
    )
    assert decoded.provenance["quality_codes"].dtype == np.dtype(np.uint16)


def test_standard_mpc_frame_protobuf_preserves_optional_mapping_states() -> None:
    frame = replace(
        _standard_frame(),
        timestamp_s=None,
        beamforming={},
        provenance=None,
    )

    proto = standard_mpc_frame_to_proto(frame)
    decoded = standard_mpc_frame_from_proto(proto)

    assert not proto.HasField("timestamp_s")
    assert decoded.timestamp_s is None
    assert decoded.beamforming == {}
    assert decoded.provenance is None


def test_standard_mpc_frame_rejects_an_unsupported_wire_version() -> None:
    proto = standard_mpc_frame_to_proto(_standard_frame())
    proto.wire_format_version = STANDARD_MPC_PROTOBUF_VERSION - 1

    with pytest.raises(ValueError, match="protobuf wire version"):
        standard_mpc_frame_from_proto(proto)


def test_standard_mpc_frame_protobuf_rejects_sensing_payloads() -> None:
    frame = replace(_standard_frame(), sensing={"range_doppler": np.ones((2, 2))})

    with pytest.raises(ValueError, match="does not support sensing"):
        standard_mpc_frame_to_proto(frame)


def test_standard_mpc_frame_rejects_a_misaligned_byte_field() -> None:
    proto = standard_mpc_frame_to_proto(_standard_frame())
    proto.pair_path_offsets = b"not-int64"

    with pytest.raises(ValueError, match="pair_path_offsets byte count"):
        standard_mpc_frame_from_proto(proto)


def test_standard_mpc_frame_rejects_an_incomplete_material_catalog() -> None:
    proto = standard_mpc_frame_to_proto(_standard_frame())
    proto.ClearField("material_itu_types")

    with pytest.raises(ValueError, match="equal lengths"):
        standard_mpc_frame_from_proto(proto)


def test_standard_mpc_frame_rejects_out_of_range_material_ids() -> None:
    proto = standard_mpc_frame_to_proto(_standard_frame())
    proto.material_ids = np.asarray([2], dtype=np.uint16).tobytes()

    with pytest.raises(ValueError, match="references an unknown material"):
        standard_mpc_frame_from_proto(proto)


def test_standard_mpc_frame_rejects_missing_interactions() -> None:
    proto = standard_mpc_frame_to_proto(_standard_frame())
    proto.ClearField("interactions")

    with pytest.raises(ValueError, match="interactions length"):
        standard_mpc_frame_from_proto(proto)


def test_standard_mpc_frame_rejects_invalid_metadata_json() -> None:
    proto = standard_mpc_frame_to_proto(_standard_frame())
    proto.beamforming_json = "[]"

    with pytest.raises(ValueError, match="beamforming_json must decode"):
        standard_mpc_frame_from_proto(proto)


def test_generated_standard_frame_descriptor_contains_compact_fields() -> None:
    fields = visualizer_pb2.StandardMPCFrame.DESCRIPTOR.fields_by_name

    assert {
        "tx_rx_pairs",
        "pair_path_offsets",
        "bounce_offsets",
        "bounce_xyz_m",
        "interactions",
        "material_ids",
        "material_names",
        "material_itu_types",
        "delays_ns",
        "path_loss_db",
        "metric_valid_bits",
        "tx_names",
        "rx_names",
        "wire_format_version",
        "targets_metadata_json",
        "beamforming_json",
        "provenance_json",
    }.issubset(fields)
    assert "targets_metadata" not in fields
    assert "beamforming" not in fields
    assert "TargetMetadata" not in visualizer_pb2.DESCRIPTOR.message_types_by_name
    assert "BeamformingFrame" not in visualizer_pb2.DESCRIPTOR.message_types_by_name
