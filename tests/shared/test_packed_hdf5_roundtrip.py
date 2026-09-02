"""End-to-end semantic and selective-I/O tests for packed HDF5 v2."""

from __future__ import annotations

from collections import defaultdict
from concurrent import futures
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import grpc
import h5py
import numpy as np
import pytest

import shared.frames.packed_hdf5 as packed_reader_module
from generator.io.grpc.file_server import FrameFileServicer, _create_server
from shared.frames import project_standard_mpc_frame
from shared.frames.contracts import (
    PATH_METRIC_ORDER,
    FrameComponent,
    FrameReadRequest,
    PathMetric,
)
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5 import PackedHDF5Error, PackedHDF5Reader
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.frames.protobuf import (
    STANDARD_MPC_PROTOBUF_VERSION,
    standard_mpc_frame_from_proto,
    standard_mpc_frame_to_proto,
)
from shared.frames.remote_hdf5 import RemoteHdf5Provider
from shared.protos import visualizer_pb2, visualizer_pb2_grpc
from tests.visualizer.fixtures.semantic_mpc import (
    assert_canonical_matches_semantics,
    build_standard_mpc_frame,
    semantic_frame_sequence,
)
from visualizer.src.io.packed_frame_payload import (
    projection_to_visual_frame,
    visual_frame_read_request,
)
from visualizer.src.metrics.packed_canon import canonical_from_projection


def _write_semantic_frame_set(
    root: Path,
    *,
    include_sensing: bool = True,
) -> tuple[Path, tuple[Any, ...]]:
    frames_dir = root / "frames"
    specs = semantic_frame_sequence()
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id="semantic-generation",
        compression="lzf",
    )
    for frame_index, spec in enumerate(specs):
        frame = build_standard_mpc_frame(spec, frame_idx=frame_index)
        writer.append(frame if include_sensing else replace(frame, sensing=None))
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="semantic-generation",
        frame_set_id="semantic-frame-set",
        chunks=[chunk],
        compression={"codec": "lzf", "shuffle": True},
        segmentation={"max_frames": 100, "max_uncompressed_bytes": 256 * 1024**2},
        provenance={"fixture": "semantic-mpc"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return frames_dir, specs


def _write_dense_transport_frame_set(root: Path, *, path_count: int = 100_000) -> None:
    """Write one valid frame whose protobuf payload exceeds gRPC's 4 MiB default."""
    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        vertices_by_pair=[np.zeros((path_count, 1, 3), dtype=np.float32)],
        interactions_by_pair=[np.ones((path_count, 1), dtype=np.uint8)],
        path_lengths_by_pair=[np.ones((path_count,), dtype=np.int64)],
        material_names_by_pair=[np.full((path_count, 1), "ground")],
        material_itu_types_by_pair=[np.full((path_count, 1), "itu_ground")],
        metrics_by_pair={
            PathMetric.DELAY_NS: [np.linspace(1.0, 2.0, path_count, dtype=np.float32)]
        },
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
    )
    assert standard_mpc_frame_to_proto(frame).ByteSize() > 4 * 1024**2

    frames_dir = root / "frames"
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id="dense-transport-generation",
        compression="lzf",
    )
    writer.append(frame)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="dense-transport-generation",
        frame_set_id="dense-transport-frame-set",
        chunks=[chunk],
        compression={"codec": "lzf", "shuffle": True},
        segmentation={"max_frames": 1},
        provenance={"fixture": "dense-transport"},
        created_utc="2026-08-06T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)


def test_writer_reader_round_trip_preserves_visual_mpc_semantics(tmp_path: Path) -> None:
    frames_dir, specs = _write_semantic_frame_set(tmp_path)
    reader = PackedHDF5Reader(frames_dir)
    try:
        reader.validate_all_chunks()
        for frame_index, spec in enumerate(specs):
            loaded = reader.load_standard_frame(frame_index)
            projection = project_standard_mpc_frame(
                loaded,
                visual_frame_read_request(),
            )
            canonical = canonical_from_projection(projection, points_dtype=np.float32)
            assert_canonical_matches_semantics(canonical, spec)
    finally:
        reader.close()


def test_coherent_output_state_and_acquisition_identity_reach_visual_payload(
    tmp_path: Path,
) -> None:
    source = build_standard_mpc_frame("baseline", frame_idx=0)
    output_target = np.asarray([[70.0, 71.0, 72.0]], dtype=np.float64)
    frame = replace(
        source,
        frame_index=7,
        target_positions_m=output_target,
        targets_metadata=(
            {
                "name": "moving_target",
                "current_position": output_target[0].tolist(),
                "orientation": [0.5, 0.0, -0.25],
            },
        ),
        provenance={
            "provider": "generator_file",
            "frame_idx": 7,
            "source_rt_frame_idx": 0,
        },
    )
    frames_dir = tmp_path / "frames"
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id="coherent-generation",
        compression="lzf",
    )
    writer.append(frame)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="coherent-generation",
        frame_set_id="coherent-frame-set",
        chunks=[chunk],
        compression={"codec": "lzf", "shuffle": True},
        segmentation={"max_frames": 1},
        provenance={"fixture": "coherent-output-state"},
        created_utc="2026-08-05T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)

    reader = PackedHDF5Reader(frames_dir)
    try:
        projection = reader.load_projection(7, visual_frame_read_request())
        payload = projection_to_visual_frame(projection)
    finally:
        reader.close()

    assert payload["_source"] == {
        "provider": "generator_file",
        "frame_idx": 7,
        "source_rt_frame_idx": 0,
    }
    np.testing.assert_allclose(payload["tx_positions"], source.tx_positions)
    np.testing.assert_allclose(payload["target_pos"], output_target)
    assert payload["targets_metadata"][0]["current_position"] == [70.0, 71.0, 72.0]
    assert_canonical_matches_semantics(payload["canonical_data"], semantic_frame_sequence()[0])


def test_complete_reads_preserve_none_and_empty_optional_mappings(tmp_path: Path) -> None:
    """Round-trip the distinct canonical states for optional metadata maps."""

    frames_dir = tmp_path / "frames"
    source = build_standard_mpc_frame(semantic_frame_sequence()[0], frame_idx=0)
    frames = (
        replace(source, frame_index=0, sensing=None, beamforming=None, provenance=None),
        replace(source, frame_index=1, sensing={}, beamforming={}, provenance={}),
    )
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id="optional-mapping-generation",
        compression=None,
    )
    for frame in frames:
        writer.append(frame)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="optional-mapping-generation",
        frame_set_id="optional-mapping-frame-set",
        chunks=[chunk],
        compression={"codec": None, "shuffle": False},
        segmentation={"max_frames": 2},
        provenance={"fixture": "optional-mappings"},
        created_utc="2026-08-04T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)

    reader = PackedHDF5Reader(frames_dir)
    try:
        absent = reader.load_standard_frame(0)
        empty = reader.load_standard_frame(1)
    finally:
        reader.close()

    assert (absent.sensing, absent.beamforming, absent.provenance) == (None, None, None)
    assert (empty.sensing, empty.beamforming, empty.provenance) == ({}, {}, {})


def test_remote_hdf5_preserves_compact_materials_and_device_names(tmp_path: Path) -> None:
    frames_dir, specs = _write_semantic_frame_set(tmp_path, include_sensing=False)
    reader = PackedHDF5Reader(frames_dir)
    servicer = FrameFileServicer(tmp_path)
    try:
        local_standard = reader.load_standard_frame(0)
        response = servicer.GetPreGeneratedFrame(
            visualizer_pb2.PreGeneratedFrameRequest(frame_idx=0),
            context=None,
        )

        assert response.success is True
        wire_frame = visualizer_pb2.StandardMPCFrame.FromString(
            response.frame_data.SerializeToString()
        )
        assert wire_frame.wire_format_version == STANDARD_MPC_PROTOBUF_VERSION
        decoded = standard_mpc_frame_from_proto(wire_frame)

        assert decoded.tx_names == local_standard.tx_names
        assert decoded.rx_names == local_standard.rx_names
        np.testing.assert_array_equal(decoded.material_ids, local_standard.material_ids)
        assert decoded.material_names == local_standard.material_names
        assert decoded.material_itu_types == local_standard.material_itu_types

        projection = project_standard_mpc_frame(
            decoded,
            visual_frame_read_request(),
        )
        canonical = canonical_from_projection(projection, points_dtype=np.float32)
        assert_canonical_matches_semantics(canonical, specs[0])
    finally:
        reader.close()
        servicer.provider.close()


@pytest.mark.optional_socket
def test_remote_hdf5_loopback_serves_a_packed_frame_set(tmp_path: Path) -> None:
    """Exercise the real generated service stub and remote provider together."""
    _frames_dir, specs = _write_semantic_frame_set(tmp_path, include_sensing=False)
    servicer = FrameFileServicer(tmp_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    visualizer_pb2_grpc.add_FrameFileServiceServicer_to_server(servicer, server)
    try:
        port = server.add_insecure_port("127.0.0.1:0")
        if port == 0:
            pytest.skip("gRPC loopback port allocation is unavailable")
        server.start()

        provider = RemoteHdf5Provider(f"127.0.0.1:{port}", connect_timeout=5.0)
        try:
            provider.open()
            decoded = provider.load_frame(0)
            expected = build_standard_mpc_frame(specs[0], frame_idx=0)

            assert provider.list_frames() == list(range(len(specs)))
            assert decoded.tx_names == expected.tx_names
            assert decoded.rx_names == expected.rx_names
            assert decoded.material_names == expected.material_names
            assert decoded.material_itu_types == expected.material_itu_types
            np.testing.assert_array_equal(decoded.material_ids, expected.material_ids)
        finally:
            provider.close()
    except (OSError, PermissionError) as exc:
        pytest.skip(f"gRPC loopback is unavailable: {exc}")
    finally:
        server.stop(0).wait()
        servicer.provider.close()


@pytest.mark.optional_socket
def test_frame_file_server_releases_its_bound_port_on_shutdown(tmp_path: Path) -> None:
    """A stopped file server can be restarted immediately on the same endpoint."""
    _write_semantic_frame_set(tmp_path, include_sensing=False)
    first_server = None
    first_servicer = None
    second_server = None
    second_servicer = None
    first_started = False
    second_started = False
    try:
        first_server, first_servicer, endpoint = _create_server(
            tmp_path,
            bind_host="127.0.0.1",
            port=0,
            max_workers=1,
        )
        port = int(endpoint.rsplit(":", 1)[1])
        first_server.start()
        first_started = True
        first_server.stop(0).wait()
        first_started = False
        first_servicer.provider.close()
        first_server = None
        first_servicer = None

        second_server, second_servicer, second_endpoint = _create_server(
            tmp_path,
            bind_host="127.0.0.1",
            port=port,
            max_workers=1,
        )
        second_server.start()
        second_started = True
        assert second_endpoint == endpoint
    except (OSError, PermissionError) as exc:
        pytest.skip(f"gRPC loopback port allocation is unavailable: {exc}")
    except RuntimeError as exc:
        if "Could not bind gRPC server" not in str(exc):
            raise
        pytest.skip(f"gRPC loopback port allocation is unavailable: {exc}")
    finally:
        if first_server is not None and first_started:
            first_server.stop(0).wait()
        if first_servicer is not None:
            first_servicer.provider.close()
        if second_server is not None and second_started:
            second_server.stop(0).wait()
        if second_servicer is not None:
            second_servicer.provider.close()


@pytest.mark.optional_socket
def test_remote_hdf5_round_trips_a_frame_above_four_mib(tmp_path: Path) -> None:
    """The shared 64 MiB policy carries a frame rejected by gRPC defaults."""
    _write_dense_transport_frame_set(tmp_path)
    server = None
    servicer = None
    provider = None
    started = False
    try:
        server, servicer, endpoint = _create_server(
            tmp_path,
            bind_host="127.0.0.1",
            port=0,
            max_workers=2,
        )
        server.start()
        started = True
        provider = RemoteHdf5Provider(endpoint, connect_timeout=5.0)
        provider.open()

        decoded = provider.load_frame(0)

        assert decoded.num_paths == 100_000
        assert decoded.frame_index == 0
        assert decoded.provenance is None
    except (OSError, PermissionError) as exc:
        pytest.skip(f"gRPC loopback is unavailable: {exc}")
    finally:
        if provider is not None:
            provider.close()
        if server is not None and started:
            server.stop(0).wait()
        if servicer is not None:
            servicer.provider.close()


class _CountingDataset:
    """Proxy one h5py dataset and record every materialized selection."""

    def __init__(
        self,
        dataset: h5py.Dataset,
        path: str,
        reads: dict[str, list[Any]],
    ) -> None:
        self._dataset = dataset
        self._path = path
        self._reads = reads

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def __getitem__(self, selection: Any) -> Any:
        self._reads[self._path].append(selection)
        return self._dataset[selection]


def test_delay_projection_does_not_touch_geometry_or_other_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    reads: dict[str, list[Any]] = defaultdict(list)
    original_dataset = packed_reader_module._dataset

    def counting_dataset(root: h5py.Group | h5py.File, path: str) -> _CountingDataset:
        return _CountingDataset(original_dataset(root, path), path, reads)

    monkeypatch.setattr(packed_reader_module, "_dataset", counting_dataset)
    reader = PackedHDF5Reader(frames_dir)
    try:
        first_projection = reader.load_projection(
            0,
            FrameReadRequest.for_metrics({PathMetric.DELAY_NS}),
        )
        projection = reader.load_projection(
            0,
            FrameReadRequest.for_metrics({PathMetric.DELAY_NS}),
        )
    finally:
        reader.close()

    assert first_projection.frame.delays_ns is not None
    assert projection.frame.delays_ns is not None
    assert projection.frame.path_loss_db is None
    assert set(reads) == {
        "frames/id",
        "index/frame_pair_path_offsets",
        "paths/metric_valid_bits",
        "paths/delay_ns",
        "static/tx_rx_pairs",
    }
    assert not any(path.startswith("bounces/") for path in reads)
    assert "paths/path_loss_db" not in reads
    assert not any(path.startswith("targets/") for path in reads)
    assert not any(path.startswith("sensing/") for path in reads)
    assert len(reads["static/tx_rx_pairs"]) == 1
    assert len(reads["frames/id"]) == 1
    assert all(
        len(selections) == 2
        for path, selections in reads.items()
        if path not in {"frames/id", "static/tx_rx_pairs"}
    )


@pytest.mark.parametrize("batch", [False, True])
def test_device_projection_does_not_read_or_return_path_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch: bool,
) -> None:
    """Keep device-only reads independent from TX/RX pair topology."""

    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    reads: dict[str, list[Any]] = defaultdict(list)
    original_dataset = packed_reader_module._dataset

    def counting_dataset(root: h5py.Group | h5py.File, path: str) -> _CountingDataset:
        return _CountingDataset(original_dataset(root, path), path, reads)

    monkeypatch.setattr(packed_reader_module, "_dataset", counting_dataset)
    reader = PackedHDF5Reader(frames_dir)
    request = FrameReadRequest(components=frozenset({FrameComponent.DEVICES}))
    try:
        projections = (
            list(reader.iter_projections([0, 1], request))
            if batch
            else [reader.load_projection(0, request)]
        )
    finally:
        reader.close()

    assert all(projection.frame.tx_positions is not None for projection in projections)
    assert all(projection.frame.tx_rx_pairs is None for projection in projections)
    assert "static/tx_rx_pairs" not in reads
    assert "index/frame_pair_path_offsets" not in reads
    assert not any(path.startswith("paths/") for path in reads)
    assert not any(path.startswith("bounces/") for path in reads)


def test_file_identity_is_validated_once_per_pooled_open_handle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    reader = PackedHDF5Reader(frames_dir)
    identity_checks = 0
    original_check = reader._check_file_identity

    def counting_check(h5, chunk, *, generation_id: str) -> None:
        nonlocal identity_checks
        identity_checks += 1
        original_check(h5, chunk, generation_id=generation_id)

    monkeypatch.setattr(reader, "_check_file_identity", counting_check)
    request = FrameReadRequest.for_metrics({PathMetric.DELAY_NS})
    try:
        reader.load_projection(0, request)
        reader.load_projection(1, request)
    finally:
        reader.close()

    assert identity_checks == 1


def test_file_identity_rejects_frame_ids_that_disagree_with_manifest(
    tmp_path: Path,
) -> None:
    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    chunk_path = next(frames_dir.glob("mpc_frames_*.h5"))
    with h5py.File(chunk_path, "r+") as h5:
        h5["frames/id"][0] = 99

    reader = PackedHDF5Reader(frames_dir, verify_manifest_files=False)
    try:
        with pytest.raises(
            PackedHDF5Error,
            match="frame IDs do not match the published manifest",
        ):
            reader.load_standard_frame(0)
    finally:
        reader.close()


def test_repeated_full_reads_reuse_immutable_static_catalog_io(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    reads: dict[str, list[Any]] = defaultdict(list)
    original_dataset = packed_reader_module._dataset

    def counting_dataset(root: h5py.Group | h5py.File, path: str) -> _CountingDataset:
        return _CountingDataset(original_dataset(root, path), path, reads)

    monkeypatch.setattr(packed_reader_module, "_dataset", counting_dataset)
    reader = PackedHDF5Reader(frames_dir)
    try:
        first = reader.load_standard_frame(0)
        second = reader.load_standard_frame(0)
    finally:
        reader.close()

    for path in (
        "static/tx_rx_pairs",
        "static/tx_names",
        "static/rx_names",
        "static/materials/name",
        "static/materials/itu_type",
    ):
        assert len(reads[path]) == 1
    assert len(reads["frames/id"]) == 1
    assert len(reads["paths/bounce_offsets"]) == 2
    assert len(reads["index/frame_pair_path_offsets"]) == 2
    assert len(reads["bounces/xyz_m"]) == 2
    assert first.material_names is second.material_names
    assert first.tx_rx_pairs is not second.tx_rx_pairs


def test_projection_iteration_leases_a_chunk_once_and_preserves_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    reader = PackedHDF5Reader(frames_dir)
    lease_calls: list[Path] = []
    original_lease = reader._handles.lease

    @contextmanager
    def counting_lease(path: str | Path):
        lease_calls.append(Path(path))
        with original_lease(path) as handle:
            yield handle

    monkeypatch.setattr(reader._handles, "lease", counting_lease)
    try:
        projections = list(
            reader.iter_projections(
                [3, 0, 2, 1],
                FrameReadRequest.for_metrics({PathMetric.DELAY_NS}),
            )
        )
    finally:
        reader.close()

    assert [item.frame.frame_index for item in projections] == [3, 0, 2, 1]
    assert len(lease_calls) == 1


def test_statistics_projection_scan_reads_each_chunk_column_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    reads: dict[str, list[Any]] = defaultdict(list)
    original_dataset = packed_reader_module._dataset

    def counting_dataset(root: h5py.Group | h5py.File, path: str) -> _CountingDataset:
        return _CountingDataset(original_dataset(root, path), path, reads)

    monkeypatch.setattr(packed_reader_module, "_dataset", counting_dataset)
    reader = PackedHDF5Reader(frames_dir)
    request = FrameReadRequest.for_metrics(
        PATH_METRIC_ORDER,
        include_interactions=True,
    )
    try:
        projections = list(reader.iter_projections([0, 1, 2, 3], request))
    finally:
        reader.close()

    assert [item.frame.frame_index for item in projections] == [0, 1, 2, 3]
    assert all(len(selections) == 1 for selections in reads.values())
    assert set(reads) == {
        "frames/id",
        "static/tx_rx_pairs",
        "index/frame_pair_path_offsets",
        "paths/bounce_offsets",
        "bounces/interaction",
        "paths/metric_valid_bits",
        *(f"paths/{metric.value}" for metric in PATH_METRIC_ORDER),
    }
    assert "bounces/xyz_m" not in reads


def test_deep_validation_rejects_metric_values_that_disagree_with_validity(
    tmp_path: Path,
) -> None:
    frames_dir, _specs = _write_semantic_frame_set(tmp_path)
    reader = PackedHDF5Reader(frames_dir)
    chunk_path = next(frames_dir.glob("mpc_frames_*.h5"))
    with h5py.File(chunk_path, "r+") as h5:
        h5["paths/delay_ns"][0] = np.nan

    try:
        with pytest.raises(PackedHDF5Error, match="disagrees with metric validity"):
            reader.validate_all_chunks()
    finally:
        reader.close()
