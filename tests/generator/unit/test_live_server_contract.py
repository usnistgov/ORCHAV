"""Focused contract tests for the mutable live gRPC generator."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import pytest

from generator.core.propagation import apply_target_scale_overrides, normalize_live_overrides
from generator.io.grpc import live_server
from generator.io.grpc.cache import GeneratorFrameCache
from shared.grpc_transport import (
    DEFAULT_GRPC_BIND_HOST,
    GRPC_MAX_MESSAGE_BYTES,
    GRPC_MESSAGE_OPTIONS,
)
from shared.protos import visualizer_pb2

_RAYTRACING_FIELDS = (
    "max_depth",
    "samples_per_src",
    "max_num_paths_per_src",
    "los",
    "specular_reflection",
    "diffuse_reflection",
    "refraction",
    "diffraction",
    "synthetic_array",
    "seed",
)


class _ActiveContext:
    def is_active(self) -> bool:
        return True


class _ControllerBusyError(Exception):
    pass


class _AbortingContext(_ActiveContext):
    def __init__(self) -> None:
        self.code = None
        self.details = ""

    def abort(self, code, details):
        self.code = code
        self.details = details
        raise _ControllerBusyError(details)


def _service(
    *,
    cache: GeneratorFrameCache | None = None,
    settings: dict[str, object] | None = None,
    total_steps: int = 1,
    frame_provider=None,
) -> tuple[live_server.GeneratorService, MagicMock]:
    frame_cache = cache or GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    simulation = SimpleNamespace(
        simulation_config=SimpleNamespace(num_steps=total_steps, duration=float(total_steps)),
        settings={} if settings is None else settings,
    )
    raytracing_service = SimpleNamespace(
        simulation_objects=simulation,
        compute_step=MagicMock(return_value={"frame_idx": 0, "paths": None}),
    )
    service = live_server.GeneratorService(
        frame_cache,
        {
            "services": {"raytracing_service": raytracing_service},
            "configs": {
                "tx_configs": [SimpleNamespace(name="tx_1")],
                "rx_configs": [SimpleNamespace(name="rx_1")],
                "target_configs": [SimpleNamespace(name="target_1")],
            },
        },
        frame_provider=frame_provider,
    )
    service._convert_to_protobuf_frame = MagicMock(  # type: ignore[method-assign]
        return_value=visualizer_pb2.FrameData()
    )
    return service, raytracing_service.compute_step


def _get_frame_request(frame_idx: int = 0):
    return visualizer_pb2.FrameRequest(
        get_frame=visualizer_pb2.StreamFrameCommand(frame_idx=frame_idx)
    )


def _raytracing_settings(config) -> dict[str, object]:
    return {field: getattr(config, field) for field in _RAYTRACING_FIELDS}


def _raytracing_config(**overrides):
    values = {
        "max_depth": 4,
        "samples_per_src": 100_000,
        "max_num_paths_per_src": 100_000,
        "los": True,
        "seed": 42,
    }
    values.update(overrides)
    return visualizer_pb2.RaytracingConfig(**values)


def test_live_server_uses_loopback_and_shared_message_policy(monkeypatch) -> None:
    server = MagicMock()
    server.add_insecure_port.return_value = 43123
    server_factory = MagicMock(return_value=server)
    monkeypatch.setattr(live_server.grpc, "server", server_factory)
    monkeypatch.setattr(live_server, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        live_server.visualizer_pb2_grpc,
        "add_GeneratorServiceServicer_to_server",
        MagicMock(),
    )

    cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0)
    returned_server, _service_instance, returned_cache = live_server.run_generator_server(
        43123,
        {"services": {}, "configs": {}},
        cache,
        start_in_background=True,
    )

    assert returned_server is server
    assert returned_cache is cache
    server.add_insecure_port.assert_called_once_with(f"{DEFAULT_GRPC_BIND_HOST}:43123")
    server.start.assert_called_once_with()
    assert server_factory.call_args.kwargs["options"] == GRPC_MESSAGE_OPTIONS
    assert "compression" not in server_factory.call_args.kwargs


def test_live_server_warns_for_explicit_non_loopback_bind(monkeypatch, caplog) -> None:
    server = MagicMock()
    server.add_insecure_port.return_value = 43124
    monkeypatch.setattr(live_server.grpc, "server", MagicMock(return_value=server))
    monkeypatch.setattr(live_server, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        live_server.visualizer_pb2_grpc,
        "add_GeneratorServiceServicer_to_server",
        MagicMock(),
    )

    with caplog.at_level(logging.WARNING):
        live_server.run_generator_server(
            43124,
            {"services": {}, "configs": {}},
            GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0),
            bind_host="0.0.0.0",
            start_in_background=True,
        )

    assert "trusted network" in caplog.text
    assert "no authentication or TLS" in caplog.text


def test_live_server_rejects_failed_bind_with_endpoint(monkeypatch) -> None:
    server = MagicMock()
    server.add_insecure_port.return_value = 0
    monkeypatch.setattr(live_server.grpc, "server", MagicMock(return_value=server))
    monkeypatch.setattr(live_server, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        live_server.visualizer_pb2_grpc,
        "add_GeneratorServiceServicer_to_server",
        MagicMock(),
    )

    with pytest.raises(RuntimeError, match=r"127\.0\.0\.1:43125"):
        live_server.run_generator_server(
            43125,
            {"services": {}, "configs": {}},
            GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0),
            start_in_background=True,
        )

    server.start.assert_not_called()


def test_only_one_controlling_stream_can_be_active() -> None:
    service, _compute_step = _service(total_steps=0)
    first_request = visualizer_pb2.FrameRequest(
        parameter_update=visualizer_pb2.ParameterUpdate(
            raytracing_config=_raytracing_config(),
        )
    )
    first_stream = service.StreamFrames(iter([first_request]), _ActiveContext())

    first_response = next(first_stream)
    assert first_response.WhichOneof("response_type") == "param_update_response"
    assert service.is_streaming is True

    second_context = _AbortingContext()
    with pytest.raises(_ControllerBusyError, match="active controlling client"):
        list(service.StreamFrames(iter(()), second_context))
    assert second_context.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert service.is_streaming is True

    first_stream.close()
    assert service.is_streaming is False
    assert list(service.StreamFrames(iter(()), _ActiveContext())) == []


def test_second_controller_cannot_duplicate_an_active_solver_call() -> None:
    service, compute_step = _service(total_steps=1)
    computation_started = threading.Event()
    allow_completion = threading.Event()

    def blocked_compute(*_args, **_kwargs):
        computation_started.set()
        assert allow_completion.wait(timeout=2.0)
        return {"frame_idx": 0, "paths": None}

    compute_step.side_effect = blocked_compute
    first_stream = service.StreamFrames(iter([_get_frame_request()]), _ActiveContext())
    first_responses: list[object] = []

    thread = threading.Thread(target=lambda: first_responses.append(next(first_stream)))
    thread.start()
    assert computation_started.wait(timeout=1.0)

    second_context = _AbortingContext()
    with pytest.raises(_ControllerBusyError, match="active controlling client"):
        list(service.StreamFrames(iter([_get_frame_request()]), second_context))

    allow_completion.set()
    thread.join(timeout=2.0)
    first_stream.close()

    assert thread.is_alive() is False
    assert compute_step.call_count == 1
    assert len(first_responses) == 1
    assert first_responses[0].WhichOneof("response_type") == "frame_data"


def test_oversized_prefetch_is_sent_once_without_cache_admission() -> None:
    cache = GeneratorFrameCache(max_frames=2, ttl_seconds=0, max_size_bytes=64)
    service, compute_step = _service(cache=cache, total_steps=1)
    requests = [
        visualizer_pb2.FrameRequest(
            flow_control=visualizer_pb2.FlowControlSignal(
                buffer_capacity=0,
                server_prefetch_limit=1,
                current_display_frame_idx=-1,
            )
        ),
        visualizer_pb2.FrameRequest(
            flow_control=visualizer_pb2.FlowControlSignal(
                buffer_capacity=1,
                server_prefetch_limit=1,
                current_display_frame_idx=-1,
            )
        ),
    ]

    responses = list(service.StreamFrames(iter(requests), _ActiveContext()))
    frame_responses = [
        response for response in responses if response.WhichOneof("response_type") == "frame_data"
    ]

    assert compute_step.call_count == 1
    assert len(frame_responses) == 1
    assert cache.get_available_frames() == []
    assert cache.get_cache_stats()["oversized_bypasses"] == 1


def test_service_close_waits_until_the_controlling_stream_releases() -> None:
    service, _compute_step = _service()
    assert service._claim_controller() is True
    closed = threading.Event()

    def close_service() -> None:
        service.close()
        closed.set()

    thread = threading.Thread(target=close_service)
    thread.start()
    time.sleep(0.02)

    assert closed.is_set() is False

    service._release_controller()
    thread.join(timeout=1.0)

    assert closed.is_set() is True


def test_changed_override_invalidates_cached_frame_once_before_recompute() -> None:
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, {"frame_idx": 0, "source": "stale"})
    service, compute_step = _service(cache=cache)
    override = visualizer_pb2.FrameRequest(
        override_cmd=visualizer_pb2.NodeOverrideList(
            overrides=[
                visualizer_pb2.NodeOverride(
                    name="rx_1",
                    type=visualizer_pb2.NODE_TYPE_RX,
                    x=2.0,
                    y=0.0,
                    z=1.0,
                    orientation=[0.0, 0.0, 0.0],
                )
            ]
        )
    )

    responses = list(
        service.StreamFrames(
            iter([override, override, _get_frame_request()]),
            _ActiveContext(),
        )
    )

    frame_response = responses[0]
    assert frame_response.WhichOneof("response_type") == "frame_data"
    assert frame_response.generation_epoch == 1
    compute_step.assert_called_once()
    assert compute_step.call_args.kwargs["live_overrides"] == [
        {
            "name": "rx_1",
            "type": "rx",
            "position": (2.0, 0.0, 1.0),
            "orientation": (0.0, 0.0, 0.0),
        }
    ]
    # Releasing the stream removes frames computed under its temporary override.
    assert service.generation_epoch == 2
    assert cache.get_frame(0) is None


def test_changed_parameters_invalidate_even_without_flush_request() -> None:
    config = _raytracing_config()
    settings = _raytracing_settings(config)
    settings["max_depth"] = 2
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, {"frame_idx": 0})
    service, _compute_step = _service(cache=cache, settings=settings)

    response = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        parameter_update=visualizer_pb2.ParameterUpdate(
                            raytracing_config=config,
                            flush_cache=False,
                        )
                    )
                ]
            ),
            _ActiveContext(),
        )
    )[0].param_update_response

    assert response.success is True
    assert response.cache_flushed is True
    assert response.generation_epoch == 1
    assert cache.get_frame(0) is None


def test_identical_parameters_without_flush_keep_cached_frame() -> None:
    config = _raytracing_config()
    settings = _raytracing_settings(config)
    cached_frame = {"frame_idx": 0}
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, cached_frame)
    service, _compute_step = _service(cache=cache, settings=settings)

    response = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        parameter_update=visualizer_pb2.ParameterUpdate(
                            raytracing_config=config,
                            flush_cache=False,
                        )
                    )
                ]
            ),
            _ActiveContext(),
        )
    )[0].param_update_response

    assert response.cache_flushed is False
    assert response.generation_epoch == 0
    assert cache.get_frame(0) is cached_frame


def test_out_of_range_parameter_is_rejected_without_mutating_cache() -> None:
    current_config = _raytracing_config()
    invalid_config = _raytracing_config(samples_per_src=100_000_001)
    cached_frame = {"frame_idx": 0}
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, cached_frame)
    service, _compute_step = _service(
        cache=cache,
        settings=_raytracing_settings(current_config),
    )

    responses = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        parameter_update=visualizer_pb2.ParameterUpdate(
                            raytracing_config=invalid_config,
                            flush_cache=False,
                        )
                    ),
                    _get_frame_request(),
                ]
            ),
            _ActiveContext(),
        )
    )
    response = responses[0].param_update_response

    assert response.success is False
    assert "samples_per_src" in response.message
    assert responses[1].WhichOneof("response_type") == "frame_data"
    assert service.generation_epoch == 0
    assert cache.get_frame(0) is cached_frame


def test_preset_label_without_settings_is_rejected_without_ending_stream() -> None:
    cached_frame = {"frame_idx": 0}
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, cached_frame)
    service, _compute_step = _service(cache=cache)

    responses = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        parameter_update=visualizer_pb2.ParameterUpdate(
                            preset="high",
                            flush_cache=False,
                        )
                    ),
                    _get_frame_request(),
                ]
            ),
            _ActiveContext(),
        )
    )

    assert responses[0].param_update_response.success is False
    assert "requires raytracing_config" in responses[0].param_update_response.message
    assert responses[1].WhichOneof("response_type") == "frame_data"
    assert service.generation_epoch == 0
    assert cache.get_frame(0) is cached_frame


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            visualizer_pb2.ObjectUpdate(
                object_name="rx_1",
                object_type=visualizer_pb2.NODE_TYPE_RX,
                has_position=True,
                x=float("nan"),
            ),
            "finite",
        ),
        (
            visualizer_pb2.ObjectUpdate(
                object_name="rx_1",
                object_type=visualizer_pb2.NODE_TYPE_RX,
                has_orientation=True,
                orientation=[1.0, 2.0],
            ),
            "exactly 3",
        ),
        (
            visualizer_pb2.ObjectUpdate(
                object_name="rx_1",
                object_type=visualizer_pb2.NODE_TYPE_RX,
                has_orientation=True,
                orientation=[1.0, 2.0, 3.0, 4.0],
            ),
            "exactly 3",
        ),
        (
            visualizer_pb2.ObjectUpdate(
                object_name="target_1",
                object_type=visualizer_pb2.NODE_TYPE_TARGET,
                has_scale=True,
                scale=0.0,
            ),
            "finite and positive",
        ),
        (
            visualizer_pb2.ObjectUpdate(
                object_name="rx_1",
                object_type=visualizer_pb2.NODE_TYPE_RX,
                has_scale=True,
                scale=1.0,
            ),
            "only for targets",
        ),
    ],
)
def test_invalid_node_update_is_rejected_without_ending_stream(update, message) -> None:
    cached_frame = {"frame_idx": 0}
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, cached_frame)
    service, _compute_step = _service(cache=cache)

    responses = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(object_update=update),
                    _get_frame_request(),
                ]
            ),
            _ActiveContext(),
        )
    )

    assert responses[0].object_update_response.success is False
    assert message in responses[0].object_update_response.message
    assert responses[1].WhichOneof("response_type") == "frame_data"
    assert service.generation_epoch == 0
    assert cache.get_frame(0) is cached_frame


@pytest.mark.parametrize(
    "override",
    [
        visualizer_pb2.NodeOverride(
            name="rx_1",
            type=visualizer_pb2.NODE_TYPE_RX,
            x=float("inf"),
        ),
        visualizer_pb2.NodeOverride(
            name="rx_1",
            type=visualizer_pb2.NODE_TYPE_RX,
            orientation=[1.0, 2.0],
        ),
        visualizer_pb2.NodeOverride(
            name="rx_typo",
            type=visualizer_pb2.NODE_TYPE_RX,
        ),
    ],
)
def test_invalid_override_command_is_rejected_without_ending_stream(override) -> None:
    cached_frame = {"frame_idx": 0}
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, cached_frame)
    service, _compute_step = _service(cache=cache)

    responses = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        override_cmd=visualizer_pb2.NodeOverrideList(overrides=[override])
                    ),
                    _get_frame_request(),
                ]
            ),
            _ActiveContext(),
        )
    )

    assert responses[0].error.code == "INVALID_OVERRIDE"
    assert responses[1].WhichOneof("response_type") == "frame_data"
    assert service.generation_epoch == 0
    assert cache.get_frame(0) is cached_frame


def test_parameter_change_does_not_relabel_dispatcher_frame_with_new_epoch() -> None:
    current_config = _raytracing_config(max_depth=2)
    changed_config = _raytracing_config(max_depth=4)
    dispatcher = MagicMock()
    dispatcher.wait_for_frame.return_value = {"frame_idx": 0, "source": "old"}
    dispatcher.get_stored_positions.return_value = None
    dispatcher.get_frame.return_value = {"frame_idx": 0, "source": "old"}
    service, compute_step = _service(
        settings=_raytracing_settings(current_config),
        frame_provider=dispatcher,
    )

    responses = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        parameter_update=visualizer_pb2.ParameterUpdate(
                            raytracing_config=changed_config,
                            flush_cache=False,
                        )
                    ),
                    _get_frame_request(),
                ]
            ),
            _ActiveContext(),
        )
    )

    assert responses[0].param_update_response.cache_flushed is True
    assert responses[1].WhichOneof("response_type") == "frame_data"
    dispatcher.wait_for_frame.assert_not_called()
    compute_step.assert_called_once()
    assert service._external_frames_compatible is False


def test_changed_node_invalidates_even_without_flush_request() -> None:
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, {"frame_idx": 0})
    service, _compute_step = _service(cache=cache)

    response = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        object_update=visualizer_pb2.ObjectUpdate(
                            object_name="rx_1",
                            object_type=visualizer_pb2.NODE_TYPE_RX,
                            has_position=True,
                            x=2.0,
                            y=0.0,
                            z=1.0,
                            flush_cache=False,
                        )
                    )
                ]
            ),
            _ActiveContext(),
        )
    )[0].object_update_response

    assert response.success is True
    assert response.cache_flushed is True
    assert response.recomputation_triggered is True
    assert response.generation_epoch == 1
    # The temporary node state expires with the controlling stream.
    assert service.generation_epoch == 2
    assert cache.get_frame(0) is None


def test_unknown_node_update_fails_without_advancing_revision() -> None:
    cached_frame = {"frame_idx": 0}
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    cache.add_frame(0, cached_frame)
    service, _compute_step = _service(cache=cache)

    response = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        object_update=visualizer_pb2.ObjectUpdate(
                            object_name="rx_typo",
                            object_type=visualizer_pb2.NODE_TYPE_RX,
                            has_position=True,
                            x=2.0,
                            y=0.0,
                            z=1.0,
                            flush_cache=False,
                        )
                    )
                ]
            ),
            _ActiveContext(),
        )
    )[0].object_update_response

    assert response.success is False
    assert "Unknown RX actor" in response.message
    assert service.generation_epoch == 0
    assert cache.get_frame(0) is cached_frame


def test_target_scale_override_does_not_leak_to_the_next_controller() -> None:
    service, compute_step = _service(total_steps=1)
    manager = SimpleNamespace(
        config=SimpleNamespace(name="target_1", scale=1.0),
        apply_scale_snapshot=MagicMock(),
    )
    service.raytracing_service.simulation_objects.target_managers = [manager]
    applied_scales: list[float] = []

    def compute_with_runtime_scale(frame_idx, live_overrides=None):
        normalized = normalize_live_overrides(live_overrides)
        apply_target_scale_overrides([manager], normalized["target"])
        applied_scales.append(manager.apply_scale_snapshot.call_args.args[0])
        return {"frame_idx": frame_idx, "paths": None}

    compute_step.side_effect = compute_with_runtime_scale
    first_responses = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        object_update=visualizer_pb2.ObjectUpdate(
                            object_name="target_1",
                            object_type=visualizer_pb2.NODE_TYPE_TARGET,
                            has_scale=True,
                            scale=2.0,
                        )
                    ),
                    _get_frame_request(),
                    visualizer_pb2.FrameRequest(override_cmd=visualizer_pb2.NodeOverrideList()),
                ]
            ),
            _ActiveContext(),
        )
    )
    second_responses = list(service.StreamFrames(iter([_get_frame_request()]), _ActiveContext()))

    assert first_responses[0].object_update_response.success is True
    assert first_responses[1].WhichOneof("response_type") == "frame_data"
    assert second_responses[0].WhichOneof("response_type") == "frame_data"
    assert applied_scales == [2.0, 1.0]
    assert [entry.args[0] for entry in manager.apply_scale_snapshot.call_args_list] == [
        2.0,
        1.0,
        1.0,
    ]
    assert manager.config.scale == 1.0


@pytest.mark.parametrize("field_name", ["buffer_capacity", "server_prefetch_limit"])
def test_oversized_flow_window_is_rejected_without_computing(field_name: str) -> None:
    cache = GeneratorFrameCache(max_frames=4, ttl_seconds=0, max_size_bytes=0)
    service, compute_step = _service(cache=cache, total_steps=100)
    flow_values = {field_name: cache.max_frames + 1}

    responses = list(
        service.StreamFrames(
            iter(
                [
                    visualizer_pb2.FrameRequest(
                        flow_control=visualizer_pb2.FlowControlSignal(**flow_values)
                    )
                ]
            ),
            _ActiveContext(),
        )
    )

    assert len(responses) == 1
    assert responses[0].WhichOneof("response_type") == "error"
    assert responses[0].error.code == "INVALID_FLOW_CONTROL"
    assert field_name in responses[0].error.message
    compute_step.assert_not_called()
    assert cache.get_cache_stats()["cached_frames"] == 0


def test_oversized_live_frame_returns_protocol_error(monkeypatch) -> None:
    service, _compute_step = _service()
    monkeypatch.setattr(
        visualizer_pb2.FrameResponse,
        "ByteSize",
        lambda _response: GRPC_MAX_MESSAGE_BYTES + 1,
    )

    responses = list(service.StreamFrames(iter([_get_frame_request()]), _ActiveContext()))

    assert len(responses) == 1
    assert responses[0].WhichOneof("response_type") == "error"
    assert responses[0].error.code == "FRAME_TOO_LARGE"
    assert str(GRPC_MAX_MESSAGE_BYTES) in responses[0].error.message


def test_conversion_failure_is_not_reported_as_an_empty_frame() -> None:
    service, _compute_step = _service()
    service._convert_to_protobuf_frame.side_effect = OverflowError("cannot encode")

    responses = list(service.StreamFrames(iter([_get_frame_request()]), _ActiveContext()))

    assert len(responses) == 1
    assert responses[0].WhichOneof("response_type") == "error"
    assert responses[0].error.code == "ENCODING_FAILED"
    assert "cannot encode" in responses[0].error.message
