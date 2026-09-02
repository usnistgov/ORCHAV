"""Manifest-first integration tests for the packed HDF5 v2 format handler."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from shared.frames.contracts import (
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
    PATH_METRIC_VALIDITY_BITS,
    FrameReadRequest,
    PathMetric,
)
from shared.frames.hdf5 import HDF5FormatHandler
from shared.frames.manifest import (
    FrameChunkManifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from shared.frames.types import StandardMPCFrame


def _write_minimal_packed_frame_set(
    root: Path,
    *,
    frame_id: int = 4,
    frames_subdir: str = "frames",
) -> Path:
    frames_dir = root / frames_subdir
    frames_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = frames_dir / f"mpc_frames_{frame_id:05d}-{frame_id:05d}.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    all_metric_bits = np.uint8(sum(PATH_METRIC_VALIDITY_BITS.values()))

    with h5py.File(chunk_path, "w") as h5:
        h5.attrs["file_kind"] = "mpc_frames"
        h5.attrs["schema_version"] = MPC_HDF5_SCHEMA_VERSION
        h5.attrs["storage_layout"] = MPC_HDF5_LAYOUT
        h5.attrs["packed_frame_version"] = PACKED_MPC_FRAME_VERSION
        h5.attrs["generation_id"] = "test-generation"
        h5.attrs["publication_state"] = "complete"
        h5.attrs["num_frames"] = 1
        h5.attrs["start_frame"] = frame_id
        h5.attrs["end_frame"] = frame_id
        h5.attrs["topology_id"] = "one-pair"
        h5.attrs["sensing_layout_id"] = "none"

        h5.create_dataset("static/tx_rx_pairs", data=np.array([[0, 0]], dtype=np.int32))
        h5.create_dataset("static/tx_names", data=np.array(["TX"], dtype=string_dtype))
        h5.create_dataset("static/rx_names", data=np.array(["RX"], dtype=string_dtype))
        h5.create_dataset(
            "static/materials/name",
            data=np.array(["", "concrete"], dtype=string_dtype),
        )
        h5.create_dataset(
            "static/materials/itu_type",
            data=np.array(["", "itu_concrete"], dtype=string_dtype),
        )

        h5.create_dataset("frames/id", data=np.array([frame_id], dtype=np.int64))
        h5.create_dataset("frames/timestamp_s", data=np.array([1.25], dtype=np.float64))
        h5.create_dataset("frames/recomputed", data=np.array([0], dtype=np.uint8))
        h5.create_dataset(
            "frames/source_json",
            data=np.array([json.dumps({"scenario": "packed-test"})], dtype=string_dtype),
        )
        h5.create_dataset(
            "frames/beamforming_json",
            data=np.array(["{}"], dtype=string_dtype),
        )
        h5.create_dataset(
            "frames/sensing_metadata_json",
            data=np.array(["{}"], dtype=string_dtype),
        )
        h5.create_dataset(
            "frames/tx_position_m",
            data=np.array([[[0.0, 0.0, 1.0]]], dtype=np.float64),
        )
        h5.create_dataset(
            "frames/rx_position_m",
            data=np.array([[[10.0, 0.0, 1.0]]], dtype=np.float64),
        )
        h5.create_dataset(
            "frames/tx_orientation_rad",
            data=np.zeros((1, 1, 3), dtype=np.float64),
        )
        h5.create_dataset(
            "frames/rx_orientation_rad",
            data=np.zeros((1, 1, 3), dtype=np.float64),
        )

        h5.create_dataset(
            "index/frame_pair_path_offsets",
            data=np.array([[0, 1]], dtype=np.int64),
        )
        h5.create_dataset(
            "index/frame_target_offsets",
            data=np.array([0, 0], dtype=np.int64),
        )
        h5.create_dataset("paths/bounce_offsets", data=np.array([0, 1], dtype=np.int64))
        h5.create_dataset(
            "paths/metric_valid_bits",
            data=np.array([all_metric_bits], dtype=np.uint8),
        )
        for name, value in (
            ("delay_ns", 12.5),
            ("path_loss_db", 80.0),
            ("aoa_az_deg", 10.0),
            ("aoa_el_deg", 2.0),
            ("aod_az_deg", 190.0),
            ("aod_el_deg", -2.0),
        ):
            h5.create_dataset(f"paths/{name}", data=np.array([value], dtype=np.float32))

        h5.create_dataset(
            "bounces/xyz_m",
            data=np.array([[5.0, 1.0, 1.0]], dtype=np.float32),
        )
        h5.create_dataset("bounces/interaction", data=np.array([1], dtype=np.uint8))
        h5.create_dataset("bounces/material_id", data=np.array([1], dtype=np.uint16))
        h5.create_dataset(
            "targets/position_m",
            shape=(0, 3),
            dtype=np.float64,
        )
        h5.create_dataset(
            "targets/metadata_json",
            shape=(0,),
            dtype=string_dtype,
        )

    chunk = FrameChunkManifest(
        file=chunk_path.name,
        frame_ids=(frame_id,),
        size_bytes=chunk_path.stat().st_size,
        uncompressed_bytes=0,
        topology_id="one-pair",
        sensing_layout_id="none",
    )
    manifest = manifest_from_chunks(
        generation_id="test-generation",
        frame_set_id="test-frame-set",
        chunks=[chunk],
        compression={"codec": "none"},
        segmentation={"max_frames": 1},
        provenance={"test": True},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return chunk_path


def test_v2_handler_startup_uses_manifest_without_opening_hdf5(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunk_path = _write_minimal_packed_frame_set(tmp_path)
    frames_dir = tmp_path / "frames"
    # Only frames_manifest.json is authoritative for this format handler.
    (frames_dir / "frames_index.json").write_text("{not valid json", encoding="utf-8")

    def fail_if_opened(*_args, **_kwargs):
        raise AssertionError("handler startup must not open an HDF5 chunk")

    monkeypatch.setattr(h5py, "File", fail_if_opened)
    handler = HDF5FormatHandler(tmp_path)

    assert handler.can_handle()
    assert handler.is_bulk
    assert handler.bulk_files == [chunk_path]
    assert handler.list_frames() == [4]
    assert handler.has_frame(4)
    assert not handler.has_frame(5)
    assert handler._packed_reader is not None
    assert handler._packed_reader.open_handle_count == 0
    handler.close()


def test_v2_handler_loads_a_valid_standard_frame(tmp_path: Path) -> None:
    _write_minimal_packed_frame_set(tmp_path)
    handler = HDF5FormatHandler(tmp_path)
    try:
        frame = handler.load_frame(4)
        assert isinstance(frame, StandardMPCFrame)
        frame.validate()
        np.testing.assert_array_equal(frame.tx_rx_pairs, [[0, 0]])
        np.testing.assert_allclose(frame.bounce_xyz_m[0], [5.0, 1.0, 1.0])
        np.testing.assert_allclose(frame.delays_ns, [12.5])
        assert frame.material_names == ("", "concrete")
        assert frame.provenance == {"scenario": "packed-test"}
    finally:
        handler.close()


def test_v2_handler_delegates_selective_projection_reads(tmp_path: Path) -> None:
    _write_minimal_packed_frame_set(tmp_path)
    handler = HDF5FormatHandler(tmp_path)
    request = FrameReadRequest(metrics=frozenset({PathMetric.DELAY_NS}))
    try:
        projection = handler.load_frame_projection(4, request)
        assert projection.loaded_path_metrics == frozenset({PathMetric.DELAY_NS})
        assert projection.frame.delays_ns is not None
        assert projection.frame.path_loss_db is None
        assert projection.frame.bounce_xyz_m is None

        projections = list(handler.iter_frame_projections([4], request))
        assert [projection.frame.frame_index for projection in projections] == [4]
    finally:
        handler.close()
