from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import grpc
import numpy as np
import pytest

from generator.io.frames.conversion import standard_mpc_frame_from_raw
from generator.io.grpc.cache import GeneratorFrameCache
from generator.io.grpc.client import VisualizerGRPCClient
from generator.io.grpc.live_server import GeneratorService
from shared.protos import visualizer_pb2


class _ArrayHolder:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array)

    def numpy(self) -> np.ndarray:
        return self._array


class _Vec3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=dtype)


def _raw_frame() -> dict[str, Any]:
    tx = SimpleNamespace(
        name="tx0",
        position=_Vec3(9.0, 9.0, 9.0),
        orientation=_Vec3(0.9, 0.9, 0.9),
    )
    rx = SimpleNamespace(
        name="rx0",
        position=_Vec3(8.0, 8.0, 8.0),
        orientation=_Vec3(0.8, 0.8, 0.8),
    )
    target = SimpleNamespace(
        position=_Vec3(7.0, 7.0, 7.0),
        orientation=_Vec3(0.7, 0.7, 0.7),
    )
    target_manager = SimpleNamespace(
        target_object=target,
        config=SimpleNamespace(
            name="walker",
            scale=1.25,
            material_type="itu_concrete",
            use_ply_position=False,
            initial_position=[0.0, 0.0, 0.0],
            mobility=None,
        ),
        current_mesh_idx=0,
        meshes=["walker_000.ply"],
        relative_mesh_directory="libraries/targets/nist_human_walking",
    )

    valid = np.asarray([[[True, False]]], dtype=bool)
    vertices = np.zeros((2, 1, 1, 2, 3), dtype=np.float64)
    vertices[0, 0, 0, 0, :] = [0.5, 0.25, 1.5]
    interactions = np.zeros((2, 1, 1, 2), dtype=np.int32)
    interactions[0, 0, 0, 0] = 1
    paths = SimpleNamespace(
        valid=_ArrayHolder(valid),
        vertices=_ArrayHolder(vertices),
        interactions=_ArrayHolder(interactions),
    )

    return {
        "frame_idx": 3,
        "tx_list": [tx],
        "rx_list": [rx],
        "target_objects": [target],
        "target_managers": [target_manager],
        "paths": paths,
        "material_mapping": {
            (0, 0): {
                0: [{"name": "concrete", "itu_type": "itu_concrete"}],
            }
        },
        "tx_positions_snapshot": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        "rx_positions_snapshot": np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
        "target_positions_snapshot": np.asarray([[7.0, 8.0, 9.0]], dtype=np.float64),
        "tx_orientations_snapshot": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        "rx_orientations_snapshot": np.asarray([[0.4, 0.5, 0.6]], dtype=np.float64),
        "target_orientations_snapshot": np.asarray([[0.7, 0.8, 0.9]], dtype=np.float64),
    }


def test_live_grpc_serializes_the_same_standard_frame_as_file_mode() -> None:
    raw_frame = _raw_frame()
    raw_frame["_cached_rt_source_frame_idx"] = 1
    simulation_config = SimpleNamespace(export_path_metrics=False)

    standard_frame = standard_mpc_frame_from_raw(
        raw_frame,
        3,
        source_provider="test",
        simulation_config=simulation_config,
    )
    service = GeneratorService(
        GeneratorFrameCache(max_frames=1),
        {"services": {}, "configs": {"simulation_config": simulation_config}},
    )
    proto = service._convert_to_protobuf_frame(
        raw_frame,
        frame_idx=3,
    ).standard_mpc_frame
    decoded = VisualizerGRPCClient()._convert_from_protobuf_frame(
        visualizer_pb2.FrameData(standard_mpc_frame=proto),
        expected_frame_idx=3,
    )

    np.testing.assert_allclose(decoded.tx_positions, standard_frame.tx_positions)
    np.testing.assert_allclose(decoded.rx_positions, standard_frame.rx_positions)
    np.testing.assert_allclose(decoded.target_positions_m, standard_frame.target_positions_m)
    np.testing.assert_array_equal(decoded.tx_rx_pairs, standard_frame.tx_rx_pairs)
    np.testing.assert_array_equal(decoded.pair_path_offsets, standard_frame.pair_path_offsets)
    np.testing.assert_array_equal(decoded.bounce_offsets, standard_frame.bounce_offsets)
    np.testing.assert_allclose(decoded.bounce_xyz_m, standard_frame.bounce_xyz_m)
    np.testing.assert_array_equal(decoded.interactions, standard_frame.interactions)
    np.testing.assert_array_equal(decoded.material_ids, standard_frame.material_ids)
    assert decoded.material_names == standard_frame.material_names
    assert decoded.material_itu_types == standard_frame.material_itu_types
    assert decoded.targets_metadata[0]["name"] == "walker"
    assert decoded.targets_metadata[0]["current_position"] == [7.0, 8.0, 9.0]
    assert standard_frame.provenance == {
        "provider": "test",
        "frame_idx": 3,
        "source_rt_frame_idx": 1,
    }
    assert decoded.provenance == {
        "provider": "generator_grpc",
        "frame_idx": 3,
        "source_rt_frame_idx": 1,
    }


def test_diagnostic_client_decodes_standard_mpc_frame_payload() -> None:
    raw_frame = _raw_frame()
    simulation_config = SimpleNamespace(export_path_metrics=False)
    standard_frame = standard_mpc_frame_from_raw(
        raw_frame,
        3,
        source_provider="test",
        simulation_config=simulation_config,
    )
    service = GeneratorService(
        GeneratorFrameCache(max_frames=1),
        {"services": {}, "configs": {"simulation_config": simulation_config}},
    )
    proto = service._convert_to_protobuf_frame(
        raw_frame,
        frame_idx=3,
    ).standard_mpc_frame

    decoded = VisualizerGRPCClient()._convert_from_protobuf_frame(
        visualizer_pb2.FrameData(standard_mpc_frame=proto),
        expected_frame_idx=3,
    )

    np.testing.assert_allclose(decoded.tx_positions, standard_frame.tx_positions)
    np.testing.assert_allclose(decoded.rx_positions, standard_frame.rx_positions)
    np.testing.assert_allclose(decoded.target_positions_m, standard_frame.target_positions_m)
    np.testing.assert_array_equal(decoded.interactions, standard_frame.interactions)
    assert decoded.targets_metadata[0]["name"] == "walker"
    assert decoded.targets_metadata[0]["current_position"] == [7.0, 8.0, 9.0]


def test_diagnostic_client_requires_standard_frame_data_payload() -> None:
    with pytest.raises(ValueError, match="standard_mpc_frame"):
        VisualizerGRPCClient()._convert_from_protobuf_frame(
            visualizer_pb2.FrameData(),
            expected_frame_idx=0,
        )


def test_diagnostic_client_applies_timeout_to_finite_stream_and_unary_calls() -> None:
    calls = []

    class _FiniteCall:
        def __iter__(self):
            return iter(())

        def cancel(self) -> None:
            calls.append(("cancel",))

    class _Stub:
        def StreamFrames(self, requests, *, timeout):
            calls.append(("stream", list(requests), timeout))
            return _FiniteCall()

        def GetFrameInfo(self, _request, *, timeout):
            calls.append(("frame_info", timeout))
            return visualizer_pb2.GetFrameInfoResponse(success=True)

        def GetGeneratorStatus(self, _request, *, timeout):
            calls.append(("status", timeout))
            return visualizer_pb2.GetGeneratorStatusResponse(success=True)

    client = VisualizerGRPCClient(timeout=3.5)
    client.channel = object()
    client.stub = _Stub()
    client.connected = True

    assert list(client._invoke_stream(["request"])) == []
    assert client.get_frame_info() is not None
    assert client.get_generator_status() is not None
    assert calls == [
        ("stream", ["request"], 3.5),
        ("cancel",),
        ("frame_info", 3.5),
        ("status", 3.5),
    ]


def test_diagnostic_client_connection_timeout_returns_false(monkeypatch) -> None:
    channel = MagicMock()
    ready = MagicMock()
    ready.result.side_effect = grpc.FutureTimeoutError()
    channel_factory = MagicMock(return_value=channel)
    monkeypatch.setattr(
        "generator.io.grpc.client.grpc.insecure_channel",
        channel_factory,
    )
    monkeypatch.setattr(
        "generator.io.grpc.client.grpc.channel_ready_future",
        MagicMock(return_value=ready),
    )
    client = VisualizerGRPCClient(timeout=0.01)

    assert client.connect() is False
    assert client.is_connected() is False
    channel.close.assert_called_once_with()
    assert "compression" not in channel_factory.call_args.kwargs


def test_diagnostic_client_context_rejects_failed_connection(monkeypatch) -> None:
    client = VisualizerGRPCClient("127.0.0.1:59999")
    monkeypatch.setattr(client, "connect", MagicMock(return_value=False))

    with pytest.raises(ConnectionError, match="127.0.0.1:59999"):
        client.__enter__()
