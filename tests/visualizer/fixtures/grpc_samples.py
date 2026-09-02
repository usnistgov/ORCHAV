"""Helper builders for synthetic gRPC canonical-frame payloads."""

from __future__ import annotations

import numpy as np

from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.protobuf import standard_mpc_frame_to_proto
from shared.protos import visualizer_pb2


def build_standard_frame_pb(*, frame_index: int = 0) -> tuple[
    visualizer_pb2.StandardMPCFrame,
    dict[str, np.ndarray],
]:
    """Create a compact protobuf frame with target and beamforming state."""

    tx_positions = np.array([[0.0, 0.0, 1.5]], dtype=np.float64)
    rx_positions = np.array([[10.0, 0.0, 1.5]], dtype=np.float64)
    tx_orientations = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    rx_orientations = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    target_positions = np.array([[1.0, 2.0, 1.0]], dtype=np.float64)
    tx_rx_pairs = np.array([[0, 0]], dtype=np.int32)
    bounce_xyz_m = np.array([[[2.0, 0.0, 1.5]]], dtype=np.float32)
    interactions = np.array([[1]], dtype=np.int32)

    frame = standard_mpc_frame_from_pair_data(
        frame_index=frame_index,
        timestamp_s=123.0,
        tx_rx_pairs=tx_rx_pairs,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_orientations=tx_orientations,
        rx_orientations=rx_orientations,
        tx_names=("tx-0",),
        rx_names=("rx-0",),
        vertices_by_pair=[bounce_xyz_m],
        interactions_by_pair=[interactions],
        path_lengths_by_pair=[np.array([1], dtype=np.int64)],
        material_names_by_pair=[np.array([["mat-default"]], dtype=object)],
        material_itu_types_by_pair=[np.array([["A"]], dtype=object)],
        target_positions_m=target_positions,
        targets_metadata=(
            {
                "name": "target-0",
                "orientation": [0.0, 0.0, 0.0],
                "mesh_name": "target_mesh.ply",
                "mesh_directory": "libraries/targets/default",
                "mesh_index": 0,
                "scale": 1.0,
                "material_type": "default",
                "use_ply_position": True,
                "current_position": [1.0, 2.0, 1.0],
            },
        ),
        beamforming={
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx": {
                        "device_name": "tx0",
                        "device_index": 0,
                        "role": "tx",
                        "array_rows": 1,
                        "array_cols": 1,
                        "weights": np.array([1.0], dtype=np.float32),
                        "element_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    },
                    "rx": {
                        "device_name": "rx0",
                        "device_index": 0,
                        "role": "rx",
                        "array_rows": 1,
                        "array_cols": 1,
                        "weights": np.array([1.0], dtype=np.float32),
                        "element_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    },
                }
            ]
        },
        provenance={"provider": "generator_grpc"},
        recomputed_from_stored_positions=True,
    )
    expected = {
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "tx_rx_pairs": tx_rx_pairs,
        "target_positions_m": target_positions,
        "bounce_xyz_m": frame.bounce_xyz_m,
        "interactions": frame.interactions,
        "material_ids": frame.material_ids,
    }
    return standard_mpc_frame_to_proto(frame), expected
