"""Real-network regression for dense live canonical-frame transport."""

from __future__ import annotations

import socket
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from generator.io.grpc.cache import GeneratorFrameCache
from generator.io.grpc.live_server import run_generator_server
from shared.frames.protobuf import frame_data_from_standard_mpc_frame
from shared.frames.types import StandardMPCFrame
from shared.grpc_transport import GRPC_MAX_MESSAGE_BYTES
from shared.protos import visualizer_pb2
from visualizer.src.io.grpc_provider import GrpcProvider

_GRPC_DEFAULT_MESSAGE_BYTES = 4 * 1024 * 1024
_DENSE_PATH_COUNT = 100_000


def _free_loopback_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _dense_frame() -> StandardMPCFrame:
    path_count = _DENSE_PATH_COUNT
    path_axis = np.arange(path_count, dtype=np.float32)
    bounce_xyz_m = np.empty((path_count, 3), dtype=np.float32)
    bounce_xyz_m[:, 0] = path_axis * np.float32(0.001)
    bounce_xyz_m[:, 1] = np.remainder(path_axis, np.float32(101.0))
    bounce_xyz_m[:, 2] = np.float32(1.5)

    return StandardMPCFrame(
        frame_index=0,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.asarray([0, path_count], dtype=np.int64),
        bounce_offsets=np.arange(path_count + 1, dtype=np.int64),
        tx_positions=np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64),
        rx_positions=np.asarray([[50.0, 0.0, 1.5]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        tx_names=("tx-0",),
        rx_names=("rx-0",),
        bounce_xyz_m=bounce_xyz_m,
        interactions=np.ones((path_count,), dtype=np.uint8),
        material_ids=np.ones((path_count,), dtype=np.uint16),
        material_names=("", "mat-ground"),
        material_itu_types=("", "itu_concrete"),
        delays_ns=path_axis * np.float32(0.01),
        path_loss_db=np.float32(60.0) + path_axis * np.float32(0.0001),
        aoa_az_deg=np.remainder(path_axis, np.float32(360.0)),
        aoa_el_deg=np.remainder(path_axis, np.float32(180.0)) - np.float32(90.0),
        aod_az_deg=np.remainder(path_axis * np.float32(2.0), np.float32(360.0)),
        aod_el_deg=np.remainder(path_axis * np.float32(0.5), np.float32(180.0)) - np.float32(90.0),
        metric_valid_bits=np.full((path_count,), 0x3F, dtype=np.uint8),
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        provenance={"provider": "generator_grpc", "fixture": "100k-live-roundtrip"},
    )


def test_100k_path_frame_above_grpc_default_round_trips_over_live_provider() -> None:
    frame = _dense_frame()
    frame_data = frame_data_from_standard_mpc_frame(frame)
    response_size = visualizer_pb2.FrameResponse(
        frame_data=frame_data,
        frame_idx=0,
    ).ByteSize()
    assert _GRPC_DEFAULT_MESSAGE_BYTES < response_size < GRPC_MAX_MESSAGE_BYTES

    try:
        port = _free_loopback_port()
    except (OSError, PermissionError) as exc:
        pytest.skip(f"Loopback sockets are unavailable: {exc}")

    cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, {"frame_idx": 0, "paths": None})
    simulation = SimpleNamespace(simulation_config=SimpleNamespace(num_steps=1, duration=1.0))
    raytracing_service = SimpleNamespace(simulation_objects=simulation)
    server = None
    generator_service = None
    provider = GrpcProvider(f"grpc://127.0.0.1:{port}", buffer_size=1)

    try:
        server, generator_service, _returned_cache = run_generator_server(
            port,
            {
                "num_steps": 1,
                "duration": 1.0,
                "services": {"raytracing_service": raytracing_service},
                "configs": {},
            },
            cache,
            start_in_background=True,
        )
        generator_service._convert_to_protobuf_frame = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: frame_data
        )

        provider.open()
        loaded = provider.load_frame(0)

        assert loaded is not None
        assert loaded.num_paths == _DENSE_PATH_COUNT
        assert loaded.num_bounces == _DENSE_PATH_COUNT
        assert loaded.provenance == frame.provenance
        np.testing.assert_array_equal(loaded.pair_path_offsets, frame.pair_path_offsets)
        np.testing.assert_allclose(loaded.bounce_xyz_m[[0, -1]], frame.bounce_xyz_m[[0, -1]])
        np.testing.assert_allclose(loaded.delays_ns[[0, -1]], frame.delays_ns[[0, -1]])
    finally:
        provider.close()
        if server is not None:
            server.stop(0).wait(timeout=5.0)
        if generator_service is not None:
            generator_service.close()
