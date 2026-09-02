"""Unit tests for gRPC service health/info endpoints."""

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import grpc
import numpy as np

from generator.io.grpc.live_server import GeneratorFrameCache, GeneratorService
from shared.protos import visualizer_pb2


def _frame(num_paths: int = 1) -> Dict[str, Any]:
    """Create a minimal frame compatible with GeneratorFrameCache."""

    class _Paths:
        def __init__(self, n: int):
            self.valid = type("Valid", (), {"numpy": lambda self: [True] * n})()
            self.vertices = type("Verts", (), {"numpy": lambda self: []})()
            self.interactions = type("Ints", (), {"numpy": lambda self: []})()

    return {
        "frame_idx": 0,
        "paths": _Paths(num_paths),
        "tx_positions": [[0.0, 0.0, 0.0]],
        "rx_positions": [[1.0, 1.0, 1.0]],
    }


def _service_with_cache(
    cache: GeneratorFrameCache, generator_config: Dict[str, Any] | None = None
) -> GeneratorService:
    generator_config = generator_config or {"services": {}, "configs": {}}
    return GeneratorService(cache, generator_config)


class TestGrpcHealthEndpoints:
    def test_health_check_reports_cache_stats(self):
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0, max_size_bytes=0)
        cache.add_frame(0, _frame())
        cache.update_stats(duration=12.5, frame_rate=10.0)

        svc = _service_with_cache(cache)
        resp = svc.HealthCheck(None, None)

        assert resp.healthy is True
        assert resp.is_ready is True
        assert resp.is_streaming is False
        assert resp.frames_generated == svc.frames_generated
        assert resp.frames_failed == svc.frames_failed
        assert resp.uptime_seconds >= 0.0

    def test_get_frame_info_reports_available_frames(self):
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0, max_size_bytes=0)
        cache.add_frame(2, _frame())
        cache.add_frame(5, _frame())
        cache.update_stats(duration=20.0, frame_rate=5.0)

        svc = _service_with_cache(cache)
        resp = svc.GetFrameInfo(None, None)

        assert resp.success is True
        assert list(resp.available_frames) == [2, 5]
        assert resp.total_frames == cache.total_frames
        assert resp.duration == cache.duration
        assert resp.frame_rate == cache.frame_rate
        assert resp.generation_epoch == svc.generation_epoch

    def test_get_frame_info_reports_scenario_total_before_frames_are_generated(self):
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0, max_size_bytes=0)

        class _SimConfig:
            num_steps = 7
            duration = 3.5

        svc = _service_with_cache(
            cache,
            {
                "num_steps": 7,
                "duration": 3.5,
                "services": {},
                "configs": {"simulation_config": _SimConfig()},
            },
        )
        resp = svc.GetFrameInfo(None, None)

        assert resp.success is True
        assert list(resp.available_frames) == []
        assert resp.total_frames == 7
        assert resp.duration == 3.5
        assert resp.frame_rate == 2.0
        assert resp.generation_epoch == 0

    def test_grpc_sensing_messages_removed_from_transport(self):
        assert "SensingFrame" not in visualizer_pb2.DESCRIPTOR.message_types_by_name
        assert "SensingDetection" not in visualizer_pb2.DESCRIPTOR.message_types_by_name
        assert "sensing_frame" not in visualizer_pb2.FrameData.DESCRIPTOR.fields_by_name

    def test_live_grpc_frame_serialization_encodes_compact_arrays(self):
        cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0)
        svc = _service_with_cache(cache)

        class _ArrayWrapper:
            def __init__(self, data):
                self._data = data

            def numpy(self):
                return self._data

        paths = SimpleNamespace(
            valid=_ArrayWrapper(np.asarray([[[True]]], dtype=bool)),
            vertices=_ArrayWrapper(np.zeros((2, 1, 1, 1, 3), dtype=np.float64)),
            interactions=_ArrayWrapper(np.ones((2, 1, 1, 1), dtype=np.int32)),
        )
        frame = {
            "tx_list": [
                SimpleNamespace(
                    position=np.asarray([0.0, 0.0, 1.0]),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )
            ],
            "rx_list": [
                SimpleNamespace(
                    position=np.asarray([1.0, 0.0, 1.0]),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )
            ],
            "target_objects": [],
            "target_managers": [],
            "paths": paths,
            "material_mapping": {},
            "tx_positions_snapshot": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            "rx_positions_snapshot": np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
            "target_positions_snapshot": np.empty((0, 3), dtype=np.float64),
            "tx_orientations_snapshot": np.zeros((1, 3), dtype=np.float64),
            "rx_orientations_snapshot": np.zeros((1, 3), dtype=np.float64),
            "target_orientations_snapshot": np.empty((0, 3), dtype=np.float64),
        }
        frame_pb = svc._convert_to_protobuf_frame(
            frame,
            frame_idx=0,
        )

        assert frame_pb.HasField("standard_mpc_frame")
        standard = frame_pb.standard_mpc_frame
        assert standard.frame_idx == 0
        assert standard.tx_positions
        assert standard.rx_positions

    def test_stream_frames_client_disconnect_is_not_error_response(self):
        cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0)
        simulation = SimpleNamespace(simulation_config=SimpleNamespace(num_steps=0, duration=0.0))
        svc = _service_with_cache(
            cache,
            {
                "services": {"raytracing_service": SimpleNamespace(simulation_objects=simulation)},
                "configs": {},
            },
        )

        class _CancelledRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.CANCELLED

        class _CancelledIterator:
            def __iter__(self):
                return self

            def __next__(self):
                raise _CancelledRpcError()

        class _InactiveContext:
            def is_active(self):
                return False

        assert list(svc.StreamFrames(_CancelledIterator(), _InactiveContext())) == []
        assert svc.is_streaming is False

    def test_stream_frames_passes_override_commands_to_raytracing(self, monkeypatch):
        cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0)
        simulation = SimpleNamespace(
            simulation_config=SimpleNamespace(num_steps=1, duration=1.0),
            target_managers=[],
        )
        raytracing_service = SimpleNamespace(
            simulation_objects=simulation,
            compute_step=MagicMock(return_value={"frame_idx": 0, "paths": None}),
        )
        svc = _service_with_cache(
            cache,
            {
                "services": {"raytracing_service": raytracing_service},
                "configs": {"tx_configs": [SimpleNamespace(name="tx_1")]},
            },
        )
        monkeypatch.setattr(
            svc,
            "_convert_to_protobuf_frame",
            lambda *_args, **_kwargs: visualizer_pb2.FrameData(),
        )

        requests = iter(
            [
                visualizer_pb2.FrameRequest(
                    override_cmd=visualizer_pb2.NodeOverrideList(
                        overrides=[
                            visualizer_pb2.NodeOverride(
                                name="tx_1",
                                type=visualizer_pb2.NODE_TYPE_TX,
                                x=1.0,
                                y=2.0,
                                z=3.0,
                                orientation=[10.0, 20.0, 30.0],
                            )
                        ]
                    )
                ),
                visualizer_pb2.FrameRequest(
                    get_frame=visualizer_pb2.StreamFrameCommand(frame_idx=0)
                ),
            ]
        )

        class _ActiveContext:
            def is_active(self):
                return True

        responses = list(svc.StreamFrames(requests, _ActiveContext()))

        assert len(responses) == 2
        assert responses[0].WhichOneof("response_type") == "frame_data"
        assert responses[1].WhichOneof("response_type") == "eof"
        raytracing_service.compute_step.assert_called_once()
        call_kwargs = raytracing_service.compute_step.call_args.kwargs
        assert "live_overrides" in call_kwargs
        assert call_kwargs["live_overrides"] == [
            {
                "name": "tx_1",
                "type": "tx",
                "position": (1.0, 2.0, 3.0),
                "orientation": (10.0, 20.0, 30.0),
            }
        ]

    def test_get_cache_status_reports_limits_and_evictions(self):
        cache = GeneratorFrameCache(max_frames=3, ttl_seconds=0.5, max_size_bytes=1234)
        cache.add_frame(0, _frame())
        cache.add_frame(1, _frame())
        cache.add_frame(2, _frame())
        cache.remove_frame(1)
        cache.get_frame(2)
        cache.get_frame(99)

        svc = _service_with_cache(cache)
        resp = svc.GetCacheStatus(None, None)

        assert resp.cached_frames == len(cache.frames)
        assert resp.max_frames == 3
        assert resp.ttl_seconds == 0.5
        assert resp.max_size_bytes == 1234
        assert resp.evictions_count >= 0
        assert resp.evictions_size >= 0
        assert resp.evictions_ttl >= 0
        assert resp.peak_size_bytes >= resp.total_size_bytes
        assert resp.cache_hits == 1
        assert resp.cache_misses == 1
        assert resp.oversized_bypasses >= 0

    def test_health_check_reports_streaming_state(self):
        cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0)
        svc = _service_with_cache(cache)
        svc.is_streaming = True

        resp = svc.HealthCheck(None, None)
        assert resp.healthy is True
        assert resp.is_streaming is True
