"""Unit tests for GrpcProvider protobuf unpacking."""

from __future__ import annotations

import logging
import queue
import threading
from unittest.mock import MagicMock

import grpc
import numpy as np
import pytest
from google.protobuf import descriptor_pb2

from shared.protos import visualizer_pb2
from tests.visualizer.fixtures.grpc_samples import build_standard_frame_pb
from visualizer.src.io.grpc_provider import (
    GrpcConnectionManager,
    GrpcProvider,
    LiveGrpcControllerBusyError,
)
from visualizer.src.io.protobuf_deserializer import frame_from_proto
from visualizer.src.metrics.scenario_statistics import SCENARIO_STATISTICS_REQUEST


class _UnavailableRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "connection refused"


class _DeadlineRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "metadata request timed out"


def test_frame_from_proto_preserves_canonical_provenance():
    pb_frame, expected = build_standard_frame_pb(frame_index=5)
    frame_data = visualizer_pb2.FrameData(standard_mpc_frame=pb_frame)

    frame = frame_from_proto(frame_data, frame_idx=5)

    assert frame is not None
    assert frame.provenance is not None
    assert frame.provenance == {"provider": "generator_grpc"}
    assert frame.frame_index == 5
    np.testing.assert_allclose(frame.tx_rx_pairs, expected["tx_rx_pairs"])


def test_frame_from_proto_requires_standard_mpc_frame():
    frame = frame_from_proto(visualizer_pb2.FrameData(), frame_idx=5)

    assert frame is None


def test_frame_from_proto_rejects_an_envelope_index_mismatch():
    pb_frame, _ = build_standard_frame_pb(frame_index=4)
    frame_data = visualizer_pb2.FrameData(standard_mpc_frame=pb_frame)

    assert frame_from_proto(frame_data, frame_idx=5) is None


def test_live_grpc_frame_range_comes_from_server_metadata(monkeypatch):
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)

    monkeypatch.setattr(
        provider,
        "_get_frame_info",
        lambda: {
            "total_frames": 4,
            "duration": 1.0,
            "frame_rate": 4.0,
            "available_frames": [],
        },
    )

    assert provider.list_frames() == [0, 1, 2, 3]
    assert provider.has_frame(0) is True
    assert provider.has_frame(3) is True
    assert provider.has_frame(4) is False


def test_live_grpc_projection_scan_does_not_move_playback(monkeypatch):
    provider = GrpcProvider("grpc://unit-test", buffer_size=25)
    provider._last_displayed_frame = 2
    provider.playback_state.current_frame = 2
    provider.streaming_buffer.set_display_position(2)
    pb_frame, _expected = build_standard_frame_pb(frame_index=32)
    frame = frame_from_proto(
        visualizer_pb2.FrameData(standard_mpc_frame=pb_frame),
        frame_idx=32,
    )
    assert frame is not None
    load = MagicMock(return_value=frame)
    monkeypatch.setattr(provider, "_load_frame", load)

    projection = next(provider.iter_frame_projections([32], SCENARIO_STATISTICS_REQUEST))

    assert projection.frame.frame_index == 32
    load.assert_called_once_with(32, origin="projection", affects_playback=False)
    assert provider._last_displayed_frame == 2
    assert provider.playback_state.current_frame == 2
    assert provider.streaming_buffer.current_display == 2


def test_live_grpc_single_projection_uses_normal_playback_load(monkeypatch):
    provider = GrpcProvider("grpc://unit-test", buffer_size=25)
    pb_frame, _expected = build_standard_frame_pb(frame_index=3)
    frame = frame_from_proto(
        visualizer_pb2.FrameData(standard_mpc_frame=pb_frame),
        frame_idx=3,
    )
    assert frame is not None
    load = MagicMock(return_value=frame)
    monkeypatch.setattr(provider, "load_frame", load)

    projection = provider.load_frame_projection(3, SCENARIO_STATISTICS_REQUEST)

    assert projection.frame.frame_index == 3
    load.assert_called_once_with(3)


def test_live_grpc_playback_seek_keeps_pending_projection_request() -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=25)
    projection_waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    playback_waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    with provider._pending_lock:
        provider._pending_frames[32] = [projection_waiter]
        provider._pending_frames[33] = [playback_waiter]
        provider._non_playback_pending_frames.add(32)
    with provider._request_history_lock:
        provider._pending_requests[32] = 100.0
        provider._pending_request_origins[32] = "projection"
        provider._pending_requests[33] = 100.0
        provider._pending_request_origins[33] = "user"

    provider._cancel_distant_pending_requests(current_display=2)

    with provider._pending_lock:
        assert provider._pending_frames == {32: [projection_waiter]}
        assert provider._non_playback_pending_frames == {32}
    assert projection_waiter.empty()
    assert playback_waiter.get_nowait() is None


def test_live_grpc_shared_projection_response_outside_playback_window_reaches_waiter() -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=25)
    provider.streaming_buffer.set_display_position(2)
    waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    decoded_frame = object()
    provider._frame_from_proto = MagicMock(return_value=decoded_frame)
    with provider._pending_lock:
        provider._pending_frames[32] = [waiter]
        provider._non_playback_pending_frames.add(32)
    with provider._request_history_lock:
        provider._pending_requests[32] = 100.0
        # A playback request may have reached the transport before a projection
        # scan joined it; the explicit non-playback marker owns seek immunity.
        provider._pending_request_origins[32] = "user"

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            frame_data=visualizer_pb2.FrameData(),
            frame_idx=32,
        )
    )

    assert waiter.get_nowait() is decoded_frame
    assert provider.streaming_buffer.get_frame(32) is None
    assert provider.streaming_buffer.current_display == 2
    assert provider._non_playback_pending_frames == set()


def test_live_grpc_projection_scan_survives_concurrent_playback_seek(monkeypatch) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=25)
    provider._last_displayed_frame = 2
    provider.playback_state.current_frame = 2
    provider.streaming_buffer.set_display_position(2)
    request_started = threading.Event()
    monkeypatch.setattr(provider, "_transport_available", lambda: True)
    monkeypatch.setattr(provider, "_start_stream_loop", lambda: None)
    monkeypatch.setattr(provider, "_enqueue_get_frame", lambda _step: request_started.set())
    monkeypatch.setattr(provider, "_maybe_prefetch", lambda *_args: None)
    results = []
    errors = []

    def _collect_projection() -> None:
        try:
            results.append(next(provider.iter_frame_projections([32], SCENARIO_STATISTICS_REQUEST)))
        except Exception as exc:  # test thread must surface failures to the assertion thread
            errors.append(exc)

    worker = threading.Thread(target=_collect_projection)
    worker.start()
    assert request_started.wait(2.0)

    provider._cancel_distant_pending_requests(current_display=2)
    pb_frame, _expected = build_standard_frame_pb(frame_index=32)
    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            frame_data=visualizer_pb2.FrameData(standard_mpc_frame=pb_frame),
            frame_idx=32,
        )
    )
    worker.join(2.0)

    assert worker.is_alive() is False
    assert errors == []
    assert len(results) == 1
    assert results[0].frame.frame_index == 32
    assert provider._last_displayed_frame == 2
    assert provider.playback_state.current_frame == 2
    assert provider.streaming_buffer.current_display == 2


def test_live_grpc_playback_joins_matching_projection_request(monkeypatch) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=25)
    request_started = threading.Event()
    monkeypatch.setattr(provider, "_transport_available", lambda: True)
    monkeypatch.setattr(provider, "_start_stream_loop", lambda: None)
    monkeypatch.setattr(provider, "_enqueue_get_frame", lambda _step: request_started.set())
    monkeypatch.setattr(provider, "_maybe_prefetch", lambda *_args: None)
    projections = []

    def _collect_projection() -> None:
        projections.append(next(provider.iter_frame_projections([32], SCENARIO_STATISTICS_REQUEST)))

    worker = threading.Thread(target=_collect_projection)
    worker.start()
    assert request_started.wait(2.0)

    playback_finished = threading.Event()
    playback_result = []

    def _load_playback() -> None:
        playback_result.append(provider.load_frame(32))
        playback_finished.set()

    playback = threading.Thread(target=_load_playback)
    playback.start()
    pb_frame, _expected = build_standard_frame_pb(frame_index=32)
    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            frame_data=visualizer_pb2.FrameData(standard_mpc_frame=pb_frame),
            frame_idx=32,
        )
    )
    worker.join(2.0)
    playback.join(2.0)

    assert playback_finished.is_set()
    assert len(projections) == 1
    assert len(playback_result) == 1
    assert playback_result[0] is not None
    assert provider.get_pending_frame_requests() == []
    assert provider._non_playback_pending_frames == set()


def test_live_grpc_unavailable_endpoint_raises_concise_connection_error(monkeypatch):
    class _Stub:
        status_calls = 0

        def __init__(self, _channel):
            pass

        def GetGeneratorStatus(self, _request, timeout=None):
            type(self).status_calls += 1
            raise _UnavailableRpcError()

    monkeypatch.setattr(
        "visualizer.src.io.grpc_provider.grpc.insecure_channel", lambda *_, **__: MagicMock()
    )
    monkeypatch.setattr(
        "visualizer.src.io.grpc_provider.visualizer_pb2_grpc.GeneratorServiceStub", _Stub
    )
    provider = GrpcProvider("grpc://localhost:59999", buffer_size=2)

    with pytest.raises(ConnectionError) as exc_info:
        provider.open()

    message = str(exc_info.value)
    assert "Live gRPC server is unavailable at grpc://localhost:59999" in message
    assert "UNAVAILABLE: connection refused" in message
    assert "_InactiveRpcError" not in message
    assert "debug_error_string" not in message
    assert _Stub.status_calls == 1


@pytest.mark.parametrize(
    ("endpoint", "expected_address"),
    [
        ("  grpc://localhost:50051  ", "localhost:50051"),
        ("  localhost:50051  ", "localhost:50051"),
        ("  grpc://[::1]:50051  ", "[::1]:50051"),
    ],
)
def test_live_grpc_connection_normalizes_validated_endpoint_before_channel(
    monkeypatch,
    endpoint,
    expected_address,
) -> None:
    class _Stub:
        def __init__(self, _channel):
            pass

        def GetGeneratorStatus(self, _request, timeout=None):
            return visualizer_pb2.GetGeneratorStatusResponse(
                success=True,
                message="ready",
                is_ready=True,
                is_streaming=False,
            )

    channel = MagicMock()
    channel_factory = MagicMock(return_value=channel)
    monkeypatch.setattr("visualizer.src.io.grpc_provider.grpc.insecure_channel", channel_factory)
    monkeypatch.setattr(
        "visualizer.src.io.grpc_provider.visualizer_pb2_grpc.GeneratorServiceStub", _Stub
    )
    manager = GrpcConnectionManager(endpoint)

    assert manager.ensure_connection() is True
    assert channel_factory.call_args.args[0] == expected_address

    manager.close()


def test_live_grpc_busy_controller_is_rejected_before_stream_start(monkeypatch):
    class _Stub:
        status_calls = 0
        stream_calls = 0

        def __init__(self, _channel):
            pass

        def GetGeneratorStatus(self, _request, timeout=None):
            type(self).status_calls += 1
            return visualizer_pb2.GetGeneratorStatusResponse(
                success=True,
                message="Generator status retrieved successfully",
                is_ready=True,
                is_streaming=True,
            )

        def StreamFrames(self, _requests):
            type(self).stream_calls += 1
            return iter(())

    channel = MagicMock()
    channel_factory = MagicMock(return_value=channel)
    monkeypatch.setattr("visualizer.src.io.grpc_provider.grpc.insecure_channel", channel_factory)
    monkeypatch.setattr(
        "visualizer.src.io.grpc_provider.visualizer_pb2_grpc.GeneratorServiceStub", _Stub
    )
    provider = GrpcProvider("grpc://localhost:50052", buffer_size=2)

    with pytest.raises(LiveGrpcControllerBusyError, match="active controlling visualizer"):
        provider.open()

    assert _Stub.status_calls == 1
    assert _Stub.stream_calls == 0
    assert provider._stream_thread is None
    assert provider.connection_manager.is_connected is False
    channel.close.assert_called_once_with()
    assert "compression" not in channel_factory.call_args.kwargs


def test_live_grpc_frame_info_deadline_is_reported_without_ending_stream() -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    provider.connection_manager.stub = MagicMock()
    provider.connection_manager.stub.GetFrameInfo.side_effect = _DeadlineRpcError()
    provider.connection_manager.is_connected = True
    provider.connection_manager.channel = MagicMock()
    provider._disconnect_reason = None
    provider._shutdown.clear()

    assert provider._get_frame_info() is None
    assert provider.connection_manager.is_connected is True
    assert provider._disconnect_reason is None
    assert provider._last_error == (
        "GetFrameInfo failed: DEADLINE_EXCEEDED: metadata request timed out"
    )


def test_live_grpc_frame_info_unavailable_ends_stream_until_reopen() -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    provider.connection_manager.stub = MagicMock()
    provider.connection_manager.stub.GetFrameInfo.side_effect = _UnavailableRpcError()
    provider.connection_manager.is_connected = True
    provider.connection_manager.channel = MagicMock()
    provider._disconnect_reason = None
    provider._shutdown.clear()

    assert provider._get_frame_info() is None
    assert provider.connection_manager.is_connected is False
    assert provider._shutdown.is_set()
    assert provider._disconnect_reason == ("GetFrameInfo failed: UNAVAILABLE: connection refused")


def test_live_grpc_disconnect_is_terminal_and_releases_all_waiters() -> None:
    class _DisconnectingStub:
        def __init__(self) -> None:
            self.stream_calls = 0

        def StreamFrames(self, _requests):
            self.stream_calls += 1
            raise _UnavailableRpcError()

    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    stub = _DisconnectingStub()
    provider.connection_manager.stub = stub
    provider.connection_manager.is_connected = True
    provider._disconnect_reason = None
    provider._shutdown.clear()

    frame_waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    joined_frame_waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    parameter_waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    object_waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    with provider._pending_lock:
        provider._pending_frames[4] = [frame_waiter, joined_frame_waiter]
    with provider._request_history_lock:
        provider._pending_requests[4] = 100.0
        provider._pending_request_origins[4] = "user"
    with provider._param_update_waiter_lock:
        provider._param_update_waiter = parameter_waiter
    with provider._object_update_waiter_lock:
        provider._object_update_waiter = object_waiter

    provider._run_stream_loop()

    assert stub.stream_calls == 1
    assert provider.connection_manager.is_connected is False
    assert (
        provider._disconnect_reason
        == "Live gRPC stream disconnected: UNAVAILABLE: connection refused"
    )
    assert frame_waiter.get_nowait() is None
    assert joined_frame_waiter.get_nowait() is None
    assert parameter_waiter.get_nowait().success is False
    assert object_waiter.get_nowait().success is False
    assert provider.get_pending_frame_requests() == []
    assert provider.prefetch_frame(5) is False
    assert stub.stream_calls == 1


def test_live_grpc_close_refuses_reopen_while_old_stream_is_blocked() -> None:
    class _BlockedStreamThread:
        def __init__(self) -> None:
            self.join_calls = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:
            self.join_calls.append(timeout)

    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    blocked_thread = _BlockedStreamThread()
    request_queue = provider._request_queue
    channel = MagicMock()
    provider._stream_thread = blocked_thread
    provider.connection_manager.channel = channel
    provider.connection_manager.stub = MagicMock()
    provider.connection_manager.is_connected = True
    provider._disconnect_reason = None
    provider._shutdown.clear()

    provider.close()

    channel.close.assert_called_once_with()
    assert blocked_thread.join_calls == [2.0]
    assert provider._stream_thread is blocked_thread
    assert provider._request_queue is request_queue
    with pytest.raises(RuntimeError, match="previous stream is stopping"):
        provider.open()


def test_live_grpc_orderly_close_does_not_log_pending_load_as_failure(
    monkeypatch,
    caplog,
) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    request_started = threading.Event()
    results = []
    monkeypatch.setattr(provider, "_transport_available", lambda: True)
    monkeypatch.setattr(provider, "_start_stream_loop", lambda: None)
    monkeypatch.setattr(provider, "_enqueue_get_frame", lambda _step: request_started.set())
    monkeypatch.setattr(provider, "_maybe_prefetch", lambda *_args: None)

    worker = threading.Thread(target=lambda: results.append(provider.load_frame(8)))
    with caplog.at_level(logging.ERROR):
        worker.start()
        assert request_started.wait(2.0)
        provider.close()
        worker.join(2.0)

    assert worker.is_alive() is False
    assert results == [None]
    assert "Failed to load frame 8" not in caplog.text
    assert provider.get_pending_frame_requests() == []


def test_live_grpc_reopen_resets_epoch_and_accepts_lower_epoch_frames(monkeypatch) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    provider._generation_epoch = 9
    provider._stale_frames_dropped = 4
    provider.frame_info_cache = {"total_frames": 99}
    provider.cache_timestamp = 123.0
    monkeypatch.setattr(provider.connection_manager, "ensure_connection", lambda: True)
    monkeypatch.setattr(provider, "_start_stream_loop", lambda: None)

    provider.open()

    pb_frame, _expected = build_standard_frame_pb(frame_index=0)
    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            frame_data=visualizer_pb2.FrameData(standard_mpc_frame=pb_frame),
            frame_idx=0,
            generation_epoch=0,
        )
    )

    assert provider._generation_epoch == 0
    assert provider._stale_frames_dropped == 0
    assert provider.frame_info_cache is None
    assert provider.cache_timestamp == 0.0
    assert provider.streaming_buffer.get_frame(0) is not None


@pytest.mark.parametrize(
    "response",
    [
        visualizer_pb2.FrameResponse(
            param_update_response=visualizer_pb2.ParameterUpdateResponse(
                success=True,
                cache_flushed=True,
                generation_epoch=3,
            )
        ),
        visualizer_pb2.FrameResponse(
            object_update_response=visualizer_pb2.ObjectUpdateResponse(
                success=True,
                object_name="RX1",
                cache_flushed=True,
                generation_epoch=3,
            )
        ),
    ],
)
def test_server_cache_flush_ack_discards_stale_local_frames_without_client_preclear(
    response,
) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=50)
    provider.streaming_buffer.add_frame(7, MagicMock())
    provider._last_displayed_frame = 7
    provider._displayed_frames.add(7)
    provider._discarded_frames.add(8)
    waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    with provider._pending_lock:
        provider._pending_frames[9] = [waiter]
    with provider._request_history_lock:
        provider._pending_requests[9] = 100.0
        provider._pending_request_origins[9] = "user"

    provider._handle_stream_response(response)

    assert provider.streaming_buffer.get_available_frames() == []
    assert provider._last_displayed_frame == -1
    assert provider._displayed_frames == set()
    assert provider._discarded_frames == set()
    assert provider.get_pending_frame_requests() == []
    assert waiter.get_nowait() is None
    resume = provider._request_queue.get_nowait()
    assert resume.WhichOneof("request_type") == "flow_control"
    assert resume.flow_control.buffer_capacity == 5
    assert resume.flow_control.server_prefetch_limit == 5
    assert resume.flow_control.current_display_frame_idx == -1
    with pytest.raises(queue.Empty):
        provider._request_queue.get_nowait()


def test_cache_flush_pauses_before_request_and_resumes_with_bounded_prefetch() -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=50)
    provider.streaming_buffer.add_frame(7, MagicMock())
    provider._last_displayed_frame = 7

    provider.request_cache_flush(reason="test")

    pause = provider._request_queue.get_nowait()
    request = provider._request_queue.get_nowait()
    assert pause.WhichOneof("request_type") == "flow_control"
    assert pause.flow_control.buffer_capacity == 0
    assert pause.flow_control.server_prefetch_limit == 0
    assert request.WhichOneof("request_type") == "cache_flush"
    assert request.cache_flush.reason == "test"
    with pytest.raises(queue.Empty):
        provider._request_queue.get_nowait()

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            cache_flush_response=visualizer_pb2.CacheFlushResponse(
                success=True,
                client_cache_flushed=True,
                server_cache_flushed=True,
                generation_epoch=4,
            )
        )
    )

    resume = provider._request_queue.get_nowait()
    assert resume.WhichOneof("request_type") == "flow_control"
    assert resume.flow_control.buffer_capacity == 5
    assert resume.flow_control.server_prefetch_limit == 5
    with pytest.raises(queue.Empty):
        provider._request_queue.get_nowait()


def test_xml_update_timeout_keeps_operation_in_flight_until_delayed_ack(monkeypatch) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    provider.OBJECT_UPDATE_TIMEOUT = 0.001
    monkeypatch.setattr(provider, "_transport_available", lambda: True)
    monkeypatch.setattr(provider, "_start_stream_loop", lambda: None)
    monkeypatch.setattr(
        "visualizer.src.io.grpc_provider.XMLSceneHandler.serialize_xml_scene",
        lambda _root: "<scene version='3.0.0'/>",
    )

    assert provider.update_object_via_xml(object()) is False
    with provider._object_update_waiter_lock:
        pending_waiter = provider._object_update_waiter
    assert pending_waiter is not None
    assert provider.update_object_via_xml(object()) is False
    assert provider._request_queue.get_nowait().WhichOneof("request_type") == "object_update"
    with pytest.raises(queue.Empty):
        provider._request_queue.get_nowait()

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            object_update_response=visualizer_pb2.ObjectUpdateResponse(
                success=True,
                object_name="scene_modified.xml",
                cache_flushed=False,
            )
        )
    )

    with provider._object_update_waiter_lock:
        assert provider._object_update_waiter is None
    assert pending_waiter.get_nowait().success is True


def test_parameter_timeout_keeps_operation_in_flight_until_delayed_ack(monkeypatch) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    provider.REQUEST_TIMEOUT = 0.001
    monkeypatch.setattr(provider, "_transport_available", lambda: True)

    assert provider.update_raytracing_params("custom", {}, flush_cache=False) is None
    with provider._param_update_waiter_lock:
        pending_waiter = provider._param_update_waiter
    assert pending_waiter is not None
    assert provider.update_raytracing_params("custom", {}, flush_cache=False) is None
    assert provider._request_queue.get_nowait().WhichOneof("request_type") == "parameter_update"
    with pytest.raises(queue.Empty):
        provider._request_queue.get_nowait()

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            param_update_response=visualizer_pb2.ParameterUpdateResponse(
                success=True,
                cache_flushed=False,
            )
        )
    )

    with provider._param_update_waiter_lock:
        assert provider._param_update_waiter is None
    assert pending_waiter.get_nowait().success is True


def test_prefetch_lookahead_updates_client_policy_and_server_flow_control() -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=50)
    provider._last_displayed_frame = 7

    provider.set_prefetch_lookahead(12)

    assert provider.streaming_buffer.lookahead == 12
    request = provider._request_queue.get_nowait()
    assert request.WhichOneof("request_type") == "flow_control"
    assert request.flow_control.current_display_frame_idx == 7
    assert request.flow_control.server_prefetch_limit == 12
    with pytest.raises(queue.Empty):
        provider._request_queue.get_nowait()


def test_node_timeout_keeps_operation_in_flight_until_delayed_ack(monkeypatch) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    provider.REQUEST_TIMEOUT = 0.001
    monkeypatch.setattr(provider, "_transport_available", lambda: True)
    monkeypatch.setattr(provider, "_start_stream_loop", lambda: None)

    assert (
        provider.update_node_properties(
            node_type="rx",
            node_name="rx_1",
            position=[1.0, 2.0, 3.0],
        )
        is False
    )
    with provider._object_update_waiter_lock:
        pending_waiter = provider._object_update_waiter
    assert pending_waiter is not None
    assert (
        provider.update_node_properties(
            node_type="rx",
            node_name="rx_1",
            position=[4.0, 5.0, 6.0],
        )
        is False
    )
    assert provider._request_queue.get_nowait().WhichOneof("request_type") == "object_update"
    with pytest.raises(queue.Empty):
        provider._request_queue.get_nowait()

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            object_update_response=visualizer_pb2.ObjectUpdateResponse(
                success=True,
                object_name="rx_1",
                cache_flushed=False,
            )
        )
    )

    with provider._object_update_waiter_lock:
        assert provider._object_update_waiter is None
    assert pending_waiter.get_nowait().success is True


@pytest.mark.parametrize(
    ("position", "orientation"),
    [
        ([1.0, 2.0], None),
        ([1.0, float("nan"), 3.0], None),
        (None, [10.0, 20.0]),
        (None, [10.0, float("inf"), 30.0]),
    ],
)
def test_live_node_update_rejects_short_or_nonfinite_vectors(
    position,
    orientation,
) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)

    assert (
        provider.update_node_properties(
            node_type="rx",
            node_name="RX1",
            position=position,
            orientation=orientation,
        )
        is False
    )


@pytest.mark.parametrize(
    ("node_type", "scale"),
    [
        ("target", 0.0),
        ("target", float("nan")),
        ("rx", 1.0),
    ],
)
def test_live_node_update_rejects_invalid_scale(node_type, scale) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)

    assert (
        provider.update_node_properties(
            node_type=node_type,
            node_name=f"{node_type}_1",
            scale=scale,
        )
        is False
    )


def test_live_grpc_pending_request_snapshot_and_cancel(monkeypatch):
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    monkeypatch.setattr("visualizer.src.io.grpc_provider.time.time", lambda: 200.0)

    with provider._pending_lock:
        provider._pending_frames[7] = [waiter]
        provider._pending_frames[9] = [None]
    with provider._request_history_lock:
        provider._pending_requests[7] = 190.0
        provider._pending_request_origins[7] = "user"
        provider._pending_requests[9] = 195.0
        provider._pending_request_origins[9] = "prefetch_forward"

    snapshots = provider.get_pending_frame_requests()

    assert [request.frame_idx for request in snapshots] == [7, 9]
    assert snapshots[0].request_timestamp == 190.0
    assert snapshots[0].origin == "user"
    assert snapshots[0].has_waiter is True
    assert snapshots[0].age_seconds == 10.0
    assert snapshots[1].origin == "prefetch_forward"
    assert snapshots[1].has_waiter is False

    assert provider.cancel_pending_frame_request(7) is True
    assert waiter.get_nowait() is None
    assert provider.cancel_pending_frame_request(7) is False

    assert provider.cancel_pending_frame_requests() == [9]
    assert provider.get_pending_frame_requests() == []
    with provider._request_history_lock:
        assert provider._pending_requests == {}
        assert provider._pending_request_origins == {}


def test_live_grpc_drops_stale_epoch_frames():
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    provider._generation_epoch = 2
    waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    with provider._pending_lock:
        provider._pending_frames[1] = [waiter]

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            frame_data=visualizer_pb2.FrameData(
                standard_mpc_frame=visualizer_pb2.StandardMPCFrame(frame_idx=1)
            ),
            frame_idx=1,
            generation_epoch=1,
        )
    )

    assert provider._stale_frames_dropped == 1
    assert provider.streaming_buffer.get_frame(1) is None
    assert waiter.get_nowait() is None
    assert provider.get_pending_frame_requests() == []


@pytest.mark.parametrize("error_code", ["INVALID_OVERRIDE", "INVALID_FLOW_CONTROL"])
def test_command_error_does_not_consume_a_pending_frame_waiter(error_code) -> None:
    provider = GrpcProvider("grpc://unit-test", buffer_size=2)
    waiter: "queue.Queue[object]" = queue.Queue(maxsize=1)
    decoded_frame = object()
    provider._frame_from_proto = MagicMock(return_value=decoded_frame)
    with provider._pending_lock:
        provider._pending_frames[0] = [waiter]
    with provider._request_history_lock:
        provider._pending_requests[0] = 100.0
        provider._pending_request_origins[0] = "user"

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            error=visualizer_pb2.ErrorDetails(
                code=error_code,
                message="invalid command",
            )
        )
    )

    assert waiter.empty()
    with provider._pending_lock:
        assert provider._pending_frames[0] == [waiter]
    with provider._request_history_lock:
        assert 0 in provider._pending_requests

    provider._handle_stream_response(
        visualizer_pb2.FrameResponse(
            frame_data=visualizer_pb2.FrameData(),
            frame_idx=0,
        )
    )

    assert waiter.get_nowait() is decoded_frame
    with provider._pending_lock:
        assert 0 not in provider._pending_frames
    with provider._request_history_lock:
        assert 0 not in provider._pending_requests


def test_live_grpc_protocol_omits_client_path_and_unused_file_stream() -> None:
    object_update = visualizer_pb2.ObjectUpdate.DESCRIPTOR
    stream_frame_command = visualizer_pb2.StreamFrameCommand.DESCRIPTOR
    frame_file_service = visualizer_pb2.DESCRIPTOR.services_by_name["FrameFileService"]
    file_descriptor = descriptor_pb2.FileDescriptorProto.FromString(
        visualizer_pb2.DESCRIPTOR.serialized_pb
    )
    object_update_proto = next(
        message for message in file_descriptor.message_type if message.name == "ObjectUpdate"
    )
    stream_frame_command_proto = next(
        message for message in file_descriptor.message_type if message.name == "StreamFrameCommand"
    )
    connection_metrics_proto = next(
        message for message in file_descriptor.message_type if message.name == "ConnectionMetrics"
    )

    assert "scene_xml_path" not in object_update.fields_by_name
    assert "has_material" not in object_update.fields_by_name
    assert "material_type" not in object_update.fields_by_name
    assert set(object_update_proto.reserved_name) == {
        "has_material",
        "material_type",
        "scene_xml_path",
    }
    assert {(item.start, item.end) for item in object_update_proto.reserved_range} == {
        (11, 13),
        (15, 16),
    }
    assert "include_paths" not in stream_frame_command.fields_by_name
    assert "include_metadata" not in stream_frame_command.fields_by_name
    assert set(stream_frame_command_proto.reserved_name) == {
        "include_paths",
        "include_metadata",
    }
    assert {(item.start, item.end) for item in stream_frame_command_proto.reserved_range} == {
        (2, 4)
    }
    assert "reconnection_count" not in visualizer_pb2.ConnectionMetrics.DESCRIPTOR.fields_by_name
    assert list(connection_metrics_proto.reserved_name) == ["reconnection_count"]
    assert [(item.start, item.end) for item in connection_metrics_proto.reserved_range] == [(2, 3)]
    assert "StreamPreGeneratedFrames" not in frame_file_service.methods_by_name
    assert "StreamPreGeneratedFramesRequest" not in visualizer_pb2.DESCRIPTOR.message_types_by_name
