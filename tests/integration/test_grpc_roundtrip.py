"""Integration-style tests for the gRPC provider StandardMPCFrame path."""

from __future__ import annotations

import numpy as np

from shared.protos import visualizer_pb2
from tests.visualizer.fixtures.grpc_samples import build_standard_frame_pb
from visualizer.src.io.grpc_provider import GrpcProvider


def test_grpc_standard_frame_roundtrip_preserves_source_and_values():
    provider = GrpcProvider("grpc://integration-test")
    pb_frame, expected = build_standard_frame_pb(frame_index=7)
    frame_data = visualizer_pb2.FrameData(standard_mpc_frame=pb_frame)

    frame = provider._frame_from_proto(frame_data, frame_idx=7)

    assert frame is not None
    assert frame.provenance is not None
    assert frame.provenance == {"provider": "generator_grpc"}
    assert frame.frame_index == 7
    np.testing.assert_allclose(frame.tx_positions, expected["tx_positions"])
    np.testing.assert_allclose(frame.rx_positions, expected["rx_positions"])
    np.testing.assert_allclose(frame.tx_rx_pairs, expected["tx_rx_pairs"])
    assert frame.targets_metadata[0]["name"] == "target-0"
