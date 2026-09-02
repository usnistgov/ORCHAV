from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from generator.io.grpc.file_server import (
    FrameFileServicer,
    _create_server,
    serve,
)
from shared.frames.contracts import PathMetric
from shared.frames.manifest import (
    FrameChunkManifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.protobuf import (
    standard_mpc_frame_from_proto,
    standard_mpc_frame_to_proto,
)
from shared.grpc_transport import (
    DEFAULT_GRPC_BIND_HOST,
    DEFAULT_GRPC_SHUTDOWN_GRACE_S,
    GRPC_MAX_MESSAGE_BYTES,
    GRPC_MESSAGE_OPTIONS,
    is_loopback_grpc_host,
)
from shared.protos import visualizer_pb2


def _standard_frame(
    *,
    frame_index: int = 0,
    beamforming=None,
    provenance=None,
    recomputed: bool = False,
):
    """Build one complete compact frame for file-server tests."""

    return standard_mpc_frame_from_pair_data(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        vertices_by_pair=[np.zeros((1, 1, 3), dtype=np.float32)],
        interactions_by_pair=[np.ones((1, 1), dtype=np.uint8)],
        path_lengths_by_pair=[np.asarray([1], dtype=np.int64)],
        material_names_by_pair=[np.asarray([["concrete"]])],
        material_itu_types_by_pair=[np.asarray([["itu_concrete"]])],
        metrics_by_pair={PathMetric.DELAY_NS: [np.asarray([3.0], dtype=np.float32)]},
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        beamforming=beamforming,
        provenance=provenance,
        recomputed_from_stored_positions=recomputed,
    )


def _write_manifest(
    frames_dir,
    *,
    frame_set_id: str = "manifest-frame-set",
    frame_ids: tuple[int, ...] = (0,),
    segmentation: dict | None = None,
    provenance: dict | None = None,
):
    frame_file = frames_dir / "mpc_frames_00000-00000.h5"
    if not frame_file.exists():
        frame_file.write_bytes(b"frame")
    manifest = manifest_from_chunks(
        generation_id="7205954c-b6bf-4514-905f-12b4d4aec855",
        frame_set_id=frame_set_id,
        chunks=(
            FrameChunkManifest(
                file=frame_file.name,
                frame_ids=frame_ids,
                size_bytes=frame_file.stat().st_size,
                uncompressed_bytes=0,
                topology_id="test-topology",
                sensing_layout_id="test-sensing",
            ),
        ),
        compression={"algorithm": "lzf"},
        segmentation=segmentation or {"effective_frame_limit": 100},
        provenance=provenance or {},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return manifest


def _mock_provider(
    *,
    frame_set_id: str = "manifest-frame-set",
    frame_ids: tuple[int, ...] = (0,),
):
    provider = MagicMock()
    provider.info = SimpleNamespace(frame_set_id=frame_set_id)
    provider.list_frames.return_value = list(frame_ids)
    return provider


def test_get_pre_generated_frame_uses_proto_cache(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_manifest(frames_dir)
    provider = _mock_provider()
    provider.has_frame.return_value = True
    provider.load_frame.return_value = _standard_frame()

    with patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider):
        servicer = FrameFileServicer(tmp_path, proto_cache_size=8)

    frame_pb = standard_mpc_frame_to_proto(_standard_frame())
    servicer._convert_frame_to_protobuf = MagicMock(return_value=frame_pb)

    request = visualizer_pb2.PreGeneratedFrameRequest(frame_idx=0)
    response_a = servicer.GetPreGeneratedFrame(request, context=None)
    response_b = servicer.GetPreGeneratedFrame(request, context=None)

    assert response_a.success is True
    assert response_b.success is True
    assert response_a.frame_set_id == servicer.frame_set_id
    assert provider.load_frame.call_count == 1
    servicer._convert_frame_to_protobuf.assert_called_once()


def test_proto_cache_tracks_bytes_replacements_hits_and_evictions(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_manifest(frames_dir)
    provider = _mock_provider()

    with patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider):
        servicer = FrameFileServicer(tmp_path, proto_cache_size=1)

    first = standard_mpc_frame_to_proto(_standard_frame(frame_index=0, provenance={"version": 1}))
    replacement = standard_mpc_frame_to_proto(
        _standard_frame(
            frame_index=0,
            provenance={"payload": "x" * 2048},
        )
    )
    second = standard_mpc_frame_to_proto(_standard_frame(frame_index=1))

    servicer._store_cached_proto(0, first)
    assert servicer.proto_cache_stats["current_bytes"] == first.ByteSize()

    servicer._store_cached_proto(0, replacement)
    replacement_stats = servicer.proto_cache_stats
    assert replacement_stats["current_bytes"] == replacement.ByteSize()
    assert replacement_stats["evictions"] == 0

    assert servicer._get_cached_proto(99) is None
    assert servicer._get_cached_proto(0) is replacement
    servicer._store_cached_proto(1, second)

    stats = servicer.proto_cache_stats
    assert stats["frame_set_id"] == "manifest-frame-set"
    assert stats["cached_entries"] == 1
    assert stats["max_entries"] == 1
    assert stats["current_bytes"] == second.ByteSize()
    assert stats["peak_bytes"] == replacement.ByteSize() + second.ByteSize()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["evictions"] == 1
    assert servicer._get_cached_proto(0) is None


def test_metadata_exposes_frame_set_manifest_fields(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_manifest(
        frames_dir,
        provenance={
            "quality_profile": {"max_depth": 2},
            "material_properties": {
                "schema_version": 1,
                "properties": {
                    "itu_concrete": {
                        "relative_permittivity": 5.31,
                    }
                },
            },
            "git_sha": "abc",
        },
    )

    provider = _mock_provider()
    provider.has_frame.return_value = True
    provider.is_bulk = True
    provider.load_frame.return_value = _standard_frame()

    with patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider):
        servicer = FrameFileServicer(tmp_path, proto_cache_size=8)

    response = servicer.GetFileServerMetadata(None, None)

    assert response.success is True
    assert response.frame_set_id == "manifest-frame-set"
    assert response.manifest_schema_version == 2
    assert response.first_frame_idx == 0
    assert response.last_frame_idx == 0
    assert response.chunk_size == 100
    assert response.total_files == 1
    assert response.snapshot_valid is True
    assert response.git_sha == "abc"
    assert json.loads(response.quality_profile_json) == {"max_depth": 2}
    assert json.loads(response.material_properties_json) == {
        "schema_version": 1,
        "properties": {
            "itu_concrete": {
                "relative_permittivity": 5.31,
            }
        },
    }


def test_persisted_manifest_frame_set_id_is_stable_across_server_restarts(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_manifest(frames_dir, frame_set_id="persisted-frame-set")
    provider = _mock_provider(frame_set_id="persisted-frame-set")

    with patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider):
        first_servicer = FrameFileServicer(tmp_path)
        second_servicer = FrameFileServicer(tmp_path)

    assert first_servicer.frame_set_id == "persisted-frame-set"
    assert second_servicer.frame_set_id == first_servicer.frame_set_id


def test_changed_snapshot_rejects_frame_list(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_file = frames_dir / "mpc_frames_00000-00000.h5"
    frame_file.write_bytes(b"frame")
    _write_manifest(frames_dir)

    provider = _mock_provider()
    provider.has_frame.return_value = True
    provider.is_bulk = True
    provider.load_frame.return_value = {"num_tx": 1, "num_rx": 1, "num_targets": 0}

    with patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider):
        servicer = FrameFileServicer(tmp_path, proto_cache_size=8)

    frame_file.write_bytes(b"changed")
    response = servicer.ListAvailableFrames(None, None)

    assert response.success is False
    assert "Snapshot changed" in response.message
    assert response.frame_set_id == servicer.frame_set_id


def test_snapshot_change_during_frame_load_rejects_mixed_identity(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_file = frames_dir / "mpc_frames_00000-00000.h5"
    _write_manifest(frames_dir)

    provider = _mock_provider()
    provider.has_frame.return_value = True

    def replace_snapshot(_frame_idx):
        frame_file.write_bytes(b"replacement-frame-bytes")
        return _standard_frame()

    provider.load_frame.side_effect = replace_snapshot

    with patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider):
        servicer = FrameFileServicer(tmp_path, proto_cache_size=8)

    response = servicer.GetPreGeneratedFrame(
        visualizer_pb2.PreGeneratedFrameRequest(frame_idx=0),
        context=None,
    )

    assert response.success is False
    assert "Snapshot changed while loading frame 0" in response.message
    assert response.frame_set_id == servicer.frame_set_id
    assert not response.HasField("frame_data")
    assert servicer._get_cached_proto(0) is None


def test_snapshot_change_during_metadata_load_does_not_cache_zero_counts(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_file = frames_dir / "mpc_frames_00000-00000.h5"
    _write_manifest(frames_dir)

    provider = _mock_provider()
    provider.has_frame.return_value = True
    provider.is_bulk = True

    def replace_snapshot(_frame_idx):
        frame_file.write_bytes(b"replacement-frame-bytes")
        return _standard_frame()

    provider.load_frame.side_effect = replace_snapshot

    with patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider):
        servicer = FrameFileServicer(tmp_path, proto_cache_size=8)

    response = servicer.GetFileServerMetadata(None, None)

    assert response.success is False
    assert "Snapshot changed while loading metadata" in response.message
    assert servicer._metadata_cache is None


def test_startup_rejects_a_frame_set_that_changes_during_open(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_manifest(frames_dir, frame_set_id="initial-frame-set", frame_ids=(0,))

    provider = _mock_provider(frame_set_id="initial-frame-set", frame_ids=(0,))
    original_capture = FrameFileServicer._capture_snapshot_signatures
    capture_count = 0

    def replace_after_first_capture(servicer, manifest=None):
        nonlocal capture_count
        result = original_capture(servicer, manifest)
        capture_count += 1
        if capture_count == 1:
            frame_file = frames_dir / "mpc_frames_00000-00000.h5"
            frame_file.write_bytes(b"replacement")
            _write_manifest(frames_dir, frame_set_id="replacement-frame-set", frame_ids=(1,))
        return result

    with (
        patch("generator.io.grpc.file_server.Hdf5Provider", return_value=provider),
        patch.object(
            FrameFileServicer,
            "_capture_snapshot_signatures",
            replace_after_first_capture,
        ),
        pytest.raises(RuntimeError, match="changed while the remote server was starting"),
    ):
        FrameFileServicer(tmp_path)

    provider.close.assert_called_once_with()


def test_file_server_serializes_full_standard_mpc_frame_contract():
    servicer = FrameFileServicer.__new__(FrameFileServicer)
    frame = _standard_frame(
        frame_index=7,
        beamforming={
            "mode": "adaptive",
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx": {
                        "device_name": "tx0",
                        "weights": np.asarray([1.0 + 0.0j]),
                        "element_positions": np.zeros((1, 3), dtype=np.float64),
                    },
                    "rx": {
                        "device_name": "rx0",
                        "weights": np.asarray([1.0 + 0.0j]),
                        "element_positions": np.zeros((1, 3), dtype=np.float64),
                    },
                }
            ],
        },
        provenance={"producer": "unit"},
        recomputed=True,
    )

    proto = servicer._convert_frame_to_protobuf(frame, frame_idx=7)
    decoded = standard_mpc_frame_from_proto(proto)

    assert decoded.provenance == {"producer": "unit"}
    assert decoded.frame_index == 7
    np.testing.assert_allclose(decoded.tx_positions, frame.tx_positions)
    np.testing.assert_allclose(decoded.delays_ns, [3.0])
    assert decoded.beamforming is not None
    assert decoded.beamforming["mode"] == "adaptive"
    assert decoded.beamforming["pairs"][0]["tx"]["device_name"] == "tx0"
    assert proto.recomputed_from_stored_positions is True


def test_file_server_rejects_a_loaded_frame_with_the_wrong_index():
    servicer = FrameFileServicer.__new__(FrameFileServicer)

    with pytest.raises(ValueError, match="does not match requested frame 7"):
        servicer._convert_frame_to_protobuf(_standard_frame(frame_index=6), frame_idx=7)


@pytest.mark.parametrize(
    ("encoded_size", "expected_success"),
    [
        (GRPC_MAX_MESSAGE_BYTES, True),
        (GRPC_MAX_MESSAGE_BYTES + 1, False),
    ],
)
def test_frame_response_enforces_the_complete_message_size_boundary(
    encoded_size,
    expected_success,
):
    servicer = FrameFileServicer.__new__(FrameFileServicer)
    servicer.frame_set_id = "frame-set"
    frame_pb = standard_mpc_frame_to_proto(_standard_frame())

    with patch.object(
        visualizer_pb2.PreGeneratedFrameResponse,
        "ByteSize",
        return_value=encoded_size,
    ):
        response = servicer._bounded_frame_response(
            frame_idx=0,
            frame_pb=frame_pb,
            load_time_ms=1.0,
        )

    assert response.success is expected_success
    assert response.frame_idx == 0
    assert response.frame_set_id == "frame-set"
    if expected_success:
        assert response.HasField("frame_data")
    else:
        assert not response.HasField("frame_data")
        assert str(GRPC_MAX_MESSAGE_BYTES) in response.message


def test_oversized_frame_is_not_admitted_to_protobuf_cache() -> None:
    servicer = FrameFileServicer.__new__(FrameFileServicer)
    servicer.frame_set_id = "frame-set"
    servicer.provider = MagicMock()
    servicer.provider.has_frame.return_value = True
    servicer.provider.load_frame.return_value = _standard_frame()
    servicer._validate_snapshot = MagicMock(return_value=True)
    servicer._get_cached_proto = MagicMock(return_value=None)
    servicer._store_cached_proto = MagicMock()
    servicer._convert_frame_to_protobuf = MagicMock(
        return_value=standard_mpc_frame_to_proto(_standard_frame())
    )

    with patch.object(
        visualizer_pb2.PreGeneratedFrameResponse,
        "ByteSize",
        return_value=GRPC_MAX_MESSAGE_BYTES + 1,
    ):
        response = servicer.GetPreGeneratedFrame(
            visualizer_pb2.PreGeneratedFrameRequest(frame_idx=0),
            MagicMock(),
        )

    assert response.success is False
    servicer._store_cached_proto.assert_not_called()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.42.0.3", "::1", "[::1]"])
def test_loopback_bind_hosts_are_recognized(host):
    assert is_loopback_grpc_host(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "lab-host"])
def test_non_loopback_bind_hosts_are_recognized(host):
    assert is_loopback_grpc_host(host) is False


def test_create_server_uses_loopback_and_shared_message_limits(tmp_path):
    grpc_server = MagicMock()
    grpc_server.add_insecure_port.return_value = 50052
    servicer = MagicMock()

    with (
        patch("generator.io.grpc.file_server.grpc.server", return_value=grpc_server) as create,
        patch("generator.io.grpc.file_server.FrameFileServicer", return_value=servicer),
        patch(
            "generator.io.grpc.file_server.visualizer_pb2_grpc."
            "add_FrameFileServiceServicer_to_server"
        ) as register,
    ):
        built_server, built_servicer, endpoint = _create_server(
            tmp_path,
            bind_host=DEFAULT_GRPC_BIND_HOST,
            port=50052,
            max_workers=3,
        )

    assert built_server is grpc_server
    assert built_servicer is servicer
    assert endpoint == "127.0.0.1:50052"
    assert create.call_args.kwargs["options"] == GRPC_MESSAGE_OPTIONS
    grpc_server.add_insecure_port.assert_called_once_with("127.0.0.1:50052")
    register.assert_called_once_with(servicer, grpc_server)


def test_create_server_reports_the_endpoint_when_bind_fails(tmp_path):
    grpc_server = MagicMock()
    grpc_server.add_insecure_port.return_value = 0
    servicer = MagicMock()

    with (
        patch("generator.io.grpc.file_server.grpc.server", return_value=grpc_server),
        patch("generator.io.grpc.file_server.FrameFileServicer", return_value=servicer),
        pytest.raises(RuntimeError, match=r"127\.0\.0\.1:51234"),
    ):
        _create_server(
            tmp_path,
            bind_host="127.0.0.1",
            port=51234,
            max_workers=1,
        )
    servicer.provider.close.assert_called_once_with()


def test_serve_stops_the_server_and_closes_the_provider(tmp_path):
    grpc_server = MagicMock()
    grpc_server.wait_for_termination.side_effect = KeyboardInterrupt
    stopped = MagicMock()
    grpc_server.stop.return_value = stopped
    servicer = MagicMock()
    servicer._frame_indices = [0]

    with patch(
        "generator.io.grpc.file_server._create_server",
        return_value=(grpc_server, servicer, "127.0.0.1:50052"),
    ):
        serve(str(tmp_path))

    grpc_server.stop.assert_called_once_with(grace=DEFAULT_GRPC_SHUTDOWN_GRACE_S)
    stopped.wait.assert_called_once_with()
    servicer.provider.close.assert_called_once_with()


def test_serve_warns_for_an_explicit_non_loopback_bind(tmp_path):
    grpc_server = MagicMock()
    grpc_server.wait_for_termination.side_effect = KeyboardInterrupt
    servicer = MagicMock()
    servicer._frame_indices = [0]

    with (
        patch(
            "generator.io.grpc.file_server._create_server",
            return_value=(grpc_server, servicer, "0.0.0.0:50052"),
        ),
        patch("generator.io.grpc.file_server.logger.warning") as warning,
    ):
        serve(str(tmp_path), bind_host="0.0.0.0")

    warning.assert_called_once()
    assert "trusted network" in warning.call_args.args[0]
