from __future__ import annotations

import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest

from shared.cache_sizing import estimate_retained_bytes
from shared.frames.protobuf import STANDARD_MPC_PROTOBUF_VERSION
from shared.frames.remote_hdf5 import RemoteHdf5Provider
from shared.grpc_transport import GRPC_MESSAGE_OPTIONS
from shared.protos import visualizer_pb2
from tests.visualizer.fixtures.grpc_samples import build_standard_frame_pb


class _UnavailableRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "server unavailable"


def _metadata_response(frame_set_id: str = "frame-set") -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        message="ok",
        total_frames=1,
        num_tx=1,
        num_rx=1,
        num_targets=0,
        scene_name="scene",
        source_directory="/frames",
        is_bulk_format=True,
        frame_set_id=frame_set_id,
        manifest_schema_version=2,
        first_frame_idx=0,
        last_frame_idx=0,
        chunk_size=100,
        total_files=1,
        snapshot_valid=True,
        snapshot_error="",
        git_sha="",
        quality_profile_json="",
        material_properties_json="",
    )


def _frame_list_response(frame_set_id: str = "frame-set") -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        message="ok",
        frame_indices=[0],
        total_count=1,
        frame_set_id=frame_set_id,
    )


def test_pre_generated_frame_request_has_no_sensing_flag():
    assert (
        "include_sensing" not in visualizer_pb2.PreGeneratedFrameRequest.DESCRIPTOR.fields_by_name
    )


def test_load_frame_uses_snapshot_id_and_no_sensing_request_field():
    provider = RemoteHdf5Provider("localhost:50052", rpc_timeout=7.5)
    provider._connected = True
    provider._metadata = {"frame_set_id": "abc123"}
    provider._stub = MagicMock()
    provider._stub.GetPreGeneratedFrame.return_value = SimpleNamespace(
        success=True,
        message="ok",
        frame_data=visualizer_pb2.StandardMPCFrame(frame_idx=3),
        frame_idx=3,
        load_time_ms=0.0,
        frame_set_id="abc123",
    )
    provider._convert_protobuf_to_frame = MagicMock(return_value={"frame": 3})

    assert provider.load_frame(3) == {"frame": 3}

    request = provider._stub.GetPreGeneratedFrame.call_args[0][0]
    assert request.frame_idx == 3
    assert "include_sensing" not in request.DESCRIPTOR.fields_by_name
    assert provider._stub.GetPreGeneratedFrame.call_args.kwargs == {"timeout": 7.5}


def test_remote_cache_tracks_exact_bytes_hits_and_count_evictions() -> None:
    provider = RemoteHdf5Provider("localhost:50052", cache_size=1)
    provider._metadata = {"frame_set_id": "frame-set"}
    provider._stub = MagicMock()
    frame_pbs = {
        frame_idx: build_standard_frame_pb(frame_index=frame_idx)[0] for frame_idx in (0, 1)
    }

    def get_frame(request, *, timeout):
        assert timeout == provider.rpc_timeout
        return SimpleNamespace(
            success=True,
            message="ok",
            frame_idx=request.frame_idx,
            frame_data=frame_pbs[request.frame_idx],
            frame_set_id="frame-set",
            load_time_ms=0.0,
        )

    provider._stub.GetPreGeneratedFrame.side_effect = get_frame

    frame_0 = provider.load_frame(0)
    assert provider.load_frame(0) is frame_0
    frame_1 = provider.load_frame(1)

    frame_0_bytes = estimate_retained_bytes(frame_0)
    frame_1_bytes = estimate_retained_bytes(frame_1)
    stats = provider.cache_stats

    assert stats["frame_set_id"] == "frame-set"
    assert stats["cached_entries"] == 1
    assert stats["max_entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["current_bytes"] == frame_1_bytes
    assert stats["peak_bytes"] == frame_0_bytes + frame_1_bytes
    assert stats["evictions"] == 1
    assert provider.cached_frame_indices == [1]


def test_remote_cache_replacement_updates_bytes_without_an_eviction() -> None:
    provider = RemoteHdf5Provider("localhost:50052", cache_size=1)
    frame_pb, _ = build_standard_frame_pb(frame_index=0)
    first = provider._convert_protobuf_to_frame(frame_pb)
    replacement = replace(first, provenance={"payload": "x" * 2048})
    first_bytes = estimate_retained_bytes(first)
    replacement_bytes = estimate_retained_bytes(replacement)

    with provider._cache_lock:
        provider._store_frame_locked(0, first, first_bytes)
        provider._store_frame_locked(0, replacement, replacement_bytes)

    stats = provider.cache_stats
    assert stats["current_bytes"] == replacement_bytes
    assert stats["peak_bytes"] == max(first_bytes, replacement_bytes)
    assert stats["evictions"] == 0
    assert provider.cached_frame_count == 1


def test_open_uses_the_shared_message_limits():
    provider = RemoteHdf5Provider("localhost:50052")
    provider._fetch_metadata = MagicMock()
    channel = MagicMock()
    ready = MagicMock()

    with (
        patch("shared.frames.remote_hdf5.grpc.insecure_channel", return_value=channel) as create,
        patch("shared.frames.remote_hdf5.grpc.channel_ready_future", return_value=ready),
        patch("shared.frames.remote_hdf5.visualizer_pb2_grpc.FrameFileServiceStub"),
    ):
        provider.open()

    options = create.call_args.kwargs["options"]
    assert all(option in options for option in GRPC_MESSAGE_OPTIONS)
    ready.result.assert_called_once_with(timeout=provider.connect_timeout)


def test_close_cancels_channel_before_waiting_for_prefetch_worker():
    provider = RemoteHdf5Provider("localhost:50052")
    frame_pb, _ = build_standard_frame_pb(frame_index=0)
    frame = provider._convert_protobuf_to_frame(frame_pb)
    retained_bytes = estimate_retained_bytes(frame)
    with provider._cache_lock:
        provider._store_frame_locked(0, frame, retained_bytes)
    events: list[str] = []
    channel = MagicMock()
    channel.close.side_effect = lambda: events.append("channel_closed")
    worker = MagicMock()
    worker.is_alive.side_effect = [True, False]

    def join(*, timeout):
        assert timeout > 0
        events.append("worker_joined")

    worker.join.side_effect = join
    provider._channel = channel
    provider._stub = MagicMock()
    provider._connected = True
    provider._prefetch_thread = worker

    provider.close()

    assert events == ["channel_closed", "worker_joined"]
    assert provider._prefetch_thread is None
    assert provider.cache_stats["current_bytes"] == 0
    assert provider.cache_stats["peak_bytes"] == retained_bytes


def test_open_rejects_while_previous_prefetch_worker_is_still_stopping():
    provider = RemoteHdf5Provider("localhost:50052")
    worker = MagicMock()
    worker.is_alive.return_value = True
    provider._prefetch_thread = worker

    with pytest.raises(RuntimeError, match="previous prefetch worker"):
        provider.open()


def test_reopen_discards_cache_written_after_the_previous_close():
    provider = RemoteHdf5Provider("localhost:50052")
    frame_pb, _ = build_standard_frame_pb(frame_index=4)
    frame = provider._convert_protobuf_to_frame(frame_pb)
    retained_bytes = estimate_retained_bytes(frame)
    with provider._cache_lock:
        provider._store_frame_locked(4, frame, retained_bytes)
    provider._fetch_metadata = MagicMock()
    channel = MagicMock()
    ready = MagicMock()

    with (
        patch("shared.frames.remote_hdf5.grpc.insecure_channel", return_value=channel),
        patch("shared.frames.remote_hdf5.grpc.channel_ready_future", return_value=ready),
        patch("shared.frames.remote_hdf5.visualizer_pb2_grpc.FrameFileServiceStub"),
    ):
        provider.open()

    assert provider.cached_frame_count == 0
    assert provider.cache_stats["current_bytes"] == 0
    assert provider.cache_stats["peak_bytes"] == retained_bytes


def test_metadata_requests_use_the_configured_unary_deadline():
    provider = RemoteHdf5Provider("localhost:50052", rpc_timeout=4.25)
    provider._stub = MagicMock()
    provider._stub.GetFileServerMetadata.return_value = SimpleNamespace(
        success=True,
        message="ok",
        total_frames=1,
        num_tx=1,
        num_rx=1,
        num_targets=0,
        scene_name="scene",
        source_directory="/frames",
        is_bulk_format=True,
        frame_set_id="frame-set",
        manifest_schema_version=2,
        first_frame_idx=0,
        last_frame_idx=0,
        chunk_size=100,
        total_files=1,
        snapshot_valid=True,
        snapshot_error="",
        git_sha="",
        quality_profile_json="",
        material_properties_json="",
    )
    provider._stub.ListAvailableFrames.return_value = SimpleNamespace(
        success=True,
        message="ok",
        frame_indices=[0],
        total_count=1,
        frame_set_id="frame-set",
    )

    provider._fetch_metadata()

    assert provider._stub.GetFileServerMetadata.call_args.kwargs == {"timeout": 4.25}
    assert provider._stub.ListAvailableFrames.call_args.kwargs == {"timeout": 4.25}


def test_convert_protobuf_to_frame_uses_shared_standard_codec():
    pb_frame, expected = build_standard_frame_pb()
    provider = RemoteHdf5Provider("localhost:50052")

    frame = provider._convert_protobuf_to_frame(pb_frame)

    np.testing.assert_allclose(frame.tx_positions, expected["tx_positions"])
    assert frame.beamforming is not None
    assert frame.beamforming["pairs"][0]["tx"]["device_name"] == "tx0"
    assert frame.recomputed_from_stored_positions is True
    assert frame.provenance is not None
    assert frame.provenance == {"provider": "generator_grpc"}


def test_convert_protobuf_to_frame_validates_decoded_schema():
    provider = RemoteHdf5Provider("localhost:50052")
    malformed = visualizer_pb2.StandardMPCFrame(
        frame_idx=3,
        wire_format_version=STANDARD_MPC_PROTOBUF_VERSION,
        targets_metadata_json="[]",
        beamforming_json="null",
        provenance_json="null",
    )

    with pytest.raises(ValueError, match="pair_path_offsets must contain"):
        provider._convert_protobuf_to_frame(malformed)


def test_convert_protobuf_to_frame_rejects_the_wrong_expected_index():
    pb_frame, _ = build_standard_frame_pb(frame_index=4)
    provider = RemoteHdf5Provider("localhost:50052")

    with pytest.raises(ValueError, match="does not match requested frame 3"):
        provider._convert_protobuf_to_frame(pb_frame, expected_frame_index=3)


def test_load_frame_rejects_a_response_for_another_index():
    provider = RemoteHdf5Provider("localhost:50052")
    provider._metadata = {"frame_set_id": "frame-set"}
    provider._stub = MagicMock()
    provider._stub.GetPreGeneratedFrame.return_value = SimpleNamespace(
        success=True,
        message="ok",
        frame_idx=4,
        frame_data=visualizer_pb2.StandardMPCFrame(),
        frame_set_id="frame-set",
        load_time_ms=0.0,
    )

    with pytest.raises(ValueError, match="does not match requested frame 3"):
        provider.load_frame(3)


def test_constructor_validates_remote_hdf5_config():
    with pytest.raises(ValueError, match="cache_size"):
        RemoteHdf5Provider("localhost:50052", cache_size=0)
    with pytest.raises(ValueError, match="connect_timeout"):
        RemoteHdf5Provider("localhost:50052", connect_timeout=0)
    with pytest.raises(ValueError, match="rpc_timeout"):
        RemoteHdf5Provider("localhost:50052", rpc_timeout=0)
    with pytest.raises(ValueError, match="frame_index_ttl_s"):
        RemoteHdf5Provider("localhost:50052", frame_index_ttl_s=-1)
    with pytest.raises(ValueError, match="between 1 and 65535"):
        RemoteHdf5Provider("localhost:0")


def test_transport_error_requires_explicit_close_and_reopen() -> None:
    provider = RemoteHdf5Provider("localhost:50052")
    channel = MagicMock()
    stub = MagicMock()
    stub.GetPreGeneratedFrame.side_effect = _UnavailableRpcError()
    provider._channel = channel
    provider._stub = stub
    provider._connected = True
    provider._metadata = {"frame_set_id": "old"}
    provider._frame_indices = [0]
    provider._frame_index_set = {0}
    provider._frame_cache[1] = {"stale": True}

    with pytest.raises(ConnectionError, match="gRPC error loading frame 0"):
        provider.load_frame(0)

    assert provider.is_connected is False
    assert provider._stub is None
    assert provider.metadata == {}
    assert provider.list_frames() == []
    assert provider.cached_frame_count == 0
    channel.close.assert_called_once_with()
    assert stub.GetPreGeneratedFrame.call_count == 1

    with pytest.raises(ConnectionError, match="Not connected"):
        provider.load_frame(0)
    assert stub.GetPreGeneratedFrame.call_count == 1


def test_frame_set_change_while_loading_requires_explicit_reopen() -> None:
    provider = RemoteHdf5Provider("localhost:50052")
    channel = MagicMock()
    stub = MagicMock()
    stub.GetPreGeneratedFrame.return_value = SimpleNamespace(
        success=True,
        message="ok",
        frame_idx=0,
        frame_data=visualizer_pb2.StandardMPCFrame(),
        frame_set_id="new",
        load_time_ms=0.0,
    )
    provider._channel = channel
    provider._stub = stub
    provider._connected = True
    provider._metadata = {"frame_set_id": "old"}
    provider._frame_indices = [0]
    provider._frame_index_set = {0}
    provider._frame_cache[1] = {"stale": True}

    with pytest.raises(ConnectionError, match="frame identity changed"):
        provider.load_frame(0)

    assert provider.is_connected is False
    assert provider.metadata == {}
    assert provider.list_frames() == []
    assert provider.cached_frame_count == 0
    channel.close.assert_called_once_with()
    assert stub.GetPreGeneratedFrame.call_count == 1

    with pytest.raises(ConnectionError, match="Not connected"):
        provider.load_frame(0)
    assert stub.GetPreGeneratedFrame.call_count == 1


def test_frame_from_previous_snapshot_is_not_admitted_after_refresh() -> None:
    provider = RemoteHdf5Provider("localhost:50052")
    channel = MagicMock()
    stub = MagicMock()
    stub.GetPreGeneratedFrame.return_value = SimpleNamespace(
        success=True,
        message="ok",
        frame_idx=0,
        frame_data=visualizer_pb2.StandardMPCFrame(frame_idx=0),
        frame_set_id="old",
        load_time_ms=0.0,
    )
    provider._channel = channel
    provider._stub = stub
    provider._connected = True
    provider._metadata = {"frame_set_id": "old"}
    provider._frame_indices = [0]
    provider._frame_index_set = {0}

    def decode_after_snapshot_change(*_args, **_kwargs):
        with provider._cache_lock:
            provider._metadata = {"frame_set_id": "new"}
            provider._frame_indices = [0]
            provider._frame_index_set = {0}
            provider._frame_cache.clear()
        return {"old": True}

    provider._convert_protobuf_to_frame = decode_after_snapshot_change

    with pytest.raises(ConnectionError, match="changed before frame admission"):
        provider.load_frame(0)

    assert provider.is_connected is False
    assert provider.metadata == {}
    assert provider.cached_frame_count == 0
    channel.close.assert_called_once_with()


def test_frame_response_requires_a_frame_set_identity() -> None:
    provider = RemoteHdf5Provider("localhost:50052")
    channel = MagicMock()
    stub = MagicMock()
    stub.GetPreGeneratedFrame.return_value = SimpleNamespace(
        success=True,
        message="ok",
        frame_idx=0,
        frame_data=visualizer_pb2.StandardMPCFrame(frame_idx=0),
        frame_set_id="",
        load_time_ms=0.0,
    )
    provider._channel = channel
    provider._stub = stub
    provider._connected = True
    provider._metadata = {"frame_set_id": "old"}

    with pytest.raises(ConnectionError, match="frame identity changed"):
        provider.load_frame(0)

    assert provider.is_connected is False
    assert provider.cached_frame_count == 0
    channel.close.assert_called_once_with()


def test_has_frame_uses_cached_set_membership():
    provider = RemoteHdf5Provider("localhost:50052")
    provider._connected = False
    provider._frame_indices = [1, 4, 7]
    provider._frame_index_set = {1, 4, 7}

    assert provider.has_frame(4) is True
    assert provider.has_frame(2) is False


def test_provider_info_exposes_remote_frame_set_identity():
    provider = RemoteHdf5Provider("localhost:50052")
    provider._metadata = {"frame_set_id": "snapshot-123"}
    provider._frame_indices = [0, 1]
    provider._frame_index_set = {0, 1}

    assert provider.info.frame_set_id == "snapshot-123"


def test_refresh_invokes_metadata_fetch_when_connected():
    provider = RemoteHdf5Provider("localhost:50052")
    provider._connected = True
    provider._fetch_metadata = MagicMock()

    provider.refresh()

    provider._fetch_metadata.assert_called_once()


def test_frame_index_ttl_triggers_refresh():
    provider = RemoteHdf5Provider("localhost:50052", frame_index_ttl_s=0.1)
    provider._connected = True
    provider._frame_indices = [0]
    provider._frame_index_set = {0}
    provider._frame_index_last_refresh_s = time.monotonic() - 1.0
    provider._fetch_metadata = MagicMock()

    provider.has_frame(0)

    provider._fetch_metadata.assert_called_once()


def test_metadata_frame_set_change_clears_local_cache():
    provider = RemoteHdf5Provider("localhost:50052")
    provider._stub = MagicMock()
    provider._metadata = {"frame_set_id": "old"}
    frame_pb, _ = build_standard_frame_pb(frame_index=1)
    frame = provider._convert_protobuf_to_frame(frame_pb)
    retained_bytes = estimate_retained_bytes(frame)
    with provider._cache_lock:
        provider._store_frame_locked(1, frame, retained_bytes)
    provider._stub.GetFileServerMetadata.return_value = SimpleNamespace(
        success=True,
        message="ok",
        total_frames=1,
        num_tx=1,
        num_rx=1,
        num_targets=0,
        scene_name="scene",
        source_directory="/frames",
        is_bulk_format=True,
        frame_set_id="new",
        manifest_schema_version=1,
        first_frame_idx=0,
        last_frame_idx=0,
        chunk_size=100,
        total_files=1,
        snapshot_valid=True,
        snapshot_error="",
        git_sha="",
        quality_profile_json="",
        material_properties_json=json.dumps(
            {
                "schema_version": 1,
                "properties": {
                    "itu_concrete": {
                        "relative_permittivity": 5.31,
                    }
                },
            }
        ),
    )
    provider._stub.ListAvailableFrames.return_value = SimpleNamespace(
        success=True,
        message="ok",
        frame_indices=[0],
        total_count=1,
        frame_set_id="new",
    )

    provider._fetch_metadata()

    assert provider.frame_set_id == "new"
    assert provider.cached_frame_count == 0
    assert provider.cache_stats["current_bytes"] == 0
    assert provider.cache_stats["peak_bytes"] == retained_bytes
    assert provider.metadata["material_properties"] == {
        "schema_version": 1,
        "properties": {
            "itu_concrete": {
                "relative_permittivity": 5.31,
            }
        },
    }


def test_metadata_and_frame_list_mismatch_ends_the_remote_session() -> None:
    provider = RemoteHdf5Provider("localhost:50052")
    channel = MagicMock()
    provider._channel = channel
    provider._stub = MagicMock()
    provider._stub.GetFileServerMetadata.return_value = _metadata_response("new")
    provider._stub.ListAvailableFrames.return_value = _frame_list_response("other")
    provider._connected = True
    provider._metadata = {"frame_set_id": "old", "scene_name": "old-scene"}
    provider._frame_indices = [9]
    provider._frame_index_set = {9}
    provider._frame_cache[9] = {"old": True}

    with pytest.raises(ConnectionError, match="different frame sets"):
        provider._fetch_metadata()

    assert provider.is_connected is False
    assert provider.metadata == {}
    assert provider.list_frames() == []
    assert provider.cached_frame_count == 0
    channel.close.assert_called_once_with()


def test_failed_frame_list_does_not_commit_partial_metadata() -> None:
    provider = RemoteHdf5Provider("localhost:50052")
    channel = MagicMock()
    provider._channel = channel
    provider._stub = MagicMock()
    provider._stub.GetFileServerMetadata.return_value = _metadata_response("new")
    provider._stub.ListAvailableFrames.return_value = SimpleNamespace(
        success=False,
        message="list failed",
    )
    provider._connected = True
    provider._metadata = {"frame_set_id": "old", "scene_name": "old-scene"}
    provider._frame_indices = [9]
    provider._frame_index_set = {9}
    provider._frame_cache[9] = {"old": True}

    with pytest.raises(ConnectionError, match="Failed to list frames"):
        provider._fetch_metadata()

    assert provider.is_connected is False
    assert provider.metadata == {}
    assert provider.list_frames() == []
    assert provider.cached_frame_count == 0
    channel.close.assert_called_once_with()
