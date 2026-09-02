"""Focused tests for packed HDF5 reader primitives and canonical reads."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from typing import Any

import h5py
import numpy as np
import pytest

import shared.frames.packed_hdf5 as packed_reader_module
from shared.frames.contracts import (
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
    PATH_METRIC_VALIDITY_BITS,
    PathMetric,
)
from shared.frames.manifest import (
    FrameChunkManifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from shared.frames.packed_hdf5 import (
    HDF5HandleLRU,
    PackedHDF5Error,
    PackedHDF5Reader,
)
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.frames.types import StandardMPCFrame


def _write_finalized_identity_frame_set(
    root: Path,
    *,
    generation_id: str = "generation",
    attribute_overrides: dict[str, Any] | None = None,
    omitted_attributes: frozenset[str] = frozenset(),
    stored_frame_ids: np.ndarray | None = None,
) -> Path:
    """Write the metadata surface used by cheap publication validation."""
    chunk_path = root / "mpc_frames_00000-00000.h5"
    attributes: dict[str, Any] = {
        "file_kind": "mpc_frames",
        "schema_version": MPC_HDF5_SCHEMA_VERSION,
        "storage_layout": MPC_HDF5_LAYOUT,
        "packed_frame_version": PACKED_MPC_FRAME_VERSION,
        "generation_id": generation_id,
        "publication_state": "complete",
        "num_frames": 1,
        "start_frame": 0,
        "end_frame": 0,
        "topology_id": "topology",
        "sensing_layout_id": "sensing",
    }
    if attribute_overrides is not None:
        attributes.update(attribute_overrides)
    with h5py.File(chunk_path, "w") as h5:
        for name, value in attributes.items():
            if name not in omitted_attributes:
                h5.attrs[name] = value
        h5.create_dataset(
            "frames/id",
            data=(np.array([0], dtype=np.int64) if stored_frame_ids is None else stored_frame_ids),
        )

    chunk = FrameChunkManifest(
        file=chunk_path.name,
        frame_ids=(0,),
        size_bytes=chunk_path.stat().st_size,
        uncompressed_bytes=0,
        topology_id="topology",
        sensing_layout_id="sensing",
    )
    manifest = manifest_from_chunks(
        generation_id=generation_id,
        frame_set_id="frame-set",
        chunks=[chunk],
        compression={},
        segmentation={},
        provenance={},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(root, manifest)
    return chunk_path


def _write_real_packed_frame_set(root: Path) -> None:
    generation_id = "real-structure-generation"
    writer = PackedMPCChunkWriter(
        root,
        generation_id=generation_id,
        compression=None,
    )
    writer.append(replace(_complete_standard_frame(), frame_index=0))
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id=generation_id,
        frame_set_id="real-structure-frame-set",
        chunks=[chunk],
        compression={"algorithm": "none"},
        segmentation={"policy": "test"},
        provenance={"fixture": "real-packed-writer"},
        created_utc="2026-07-30T00:00:00+00:00",
    )
    write_frame_manifest_atomic(root, manifest)


def _write_target_frame_set(root: Path) -> Path:
    """Write a later output frame whose target occupies physical metadata row zero."""
    target_position = np.array([[7.0, 8.0, 9.0]], dtype=np.float64)
    frames = (
        replace(_complete_standard_frame(), frame_index=7),
        replace(
            _complete_standard_frame(),
            frame_index=11,
            target_positions_m=target_position,
            targets_metadata=(
                {
                    "name": "vehicle",
                    "current_position": target_position[0].tolist(),
                },
            ),
        ),
    )
    writer = PackedMPCChunkWriter(
        root,
        generation_id="target-metadata-generation",
        compression=None,
    )
    for frame in frames:
        writer.append(frame)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="target-metadata-generation",
        frame_set_id="target-metadata-frame-set",
        chunks=[chunk],
        compression={"algorithm": "none"},
        segmentation={"policy": "test"},
        provenance={"fixture": "target-metadata"},
        created_utc="2026-08-05T00:00:00+00:00",
    )
    write_frame_manifest_atomic(root, manifest)
    return root / chunk.file


def _complete_standard_frame() -> StandardMPCFrame:
    all_metric_bits = np.uint8(sum(PATH_METRIC_VALIDITY_BITS.values()))
    metric_validity = np.full((3,), all_metric_bits, dtype=np.uint8)
    metric_validity[2] &= np.uint8(~PATH_METRIC_VALIDITY_BITS[PathMetric.AOA_AZ_DEG] & 0xFF)
    return StandardMPCFrame(
        frame_index=7,
        timestamp_s=12.5,
        tx_rx_pairs=np.array([[1, 0], [0, 0]], dtype=np.int32),
        pair_path_offsets=np.array([0, 1, 3], dtype=np.int64),
        bounce_offsets=np.array([0, 0, 1, 3], dtype=np.int64),
        tx_positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        rx_positions=np.array([[10.0, 0.0, 0.0]], dtype=np.float64),
        tx_orientations=np.zeros((2, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        tx_names=("TX-A", "TX-B"),
        rx_names=("RX-A",),
        bounce_xyz_m=np.array(
            [[2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        interactions=np.array([1, 2, 37], dtype=np.uint8),
        material_ids=np.array([1, 2, 1], dtype=np.uint16),
        material_names=("", "concrete", "glass"),
        material_itu_types=("", "itu_concrete", "itu_glass"),
        delays_ns=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        path_loss_db=np.array([40.0, 50.0, 60.0], dtype=np.float32),
        aoa_az_deg=np.array([0.0, 90.0, np.nan], dtype=np.float32),
        aoa_el_deg=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        aod_az_deg=np.array([180.0, 270.0, 0.0], dtype=np.float32),
        aod_el_deg=np.array([3.0, 4.0, 5.0], dtype=np.float32),
        metric_valid_bits=metric_validity,
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        sensing=None,
        beamforming=None,
        provenance={"provider": "test"},
    )


def test_full_read_constructs_complete_canonical_frame_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not hasattr(packed_reader_module, "packed_to_standard_frame")
    source = _complete_standard_frame()
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="direct-canonical-read",
        compression=None,
    )
    writer.append(source)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="direct-canonical-read",
        frame_set_id="direct-canonical-read-set",
        chunks=[chunk],
        compression={"algorithm": "none"},
        segmentation={"policy": "test"},
        provenance={},
        created_utc="2026-08-04T00:00:00+00:00",
    )
    write_frame_manifest_atomic(tmp_path, manifest)

    def reject_projection(**_fields: Any) -> None:
        raise AssertionError("complete reads must not construct a projected frame")

    monkeypatch.setattr(packed_reader_module, "ProjectedMPCFrame", reject_projection)

    reader = PackedHDF5Reader(tmp_path)
    try:
        loaded = reader.load_standard_frame(7)
    finally:
        reader.close()

    assert isinstance(loaded, StandardMPCFrame)
    np.testing.assert_array_equal(loaded.tx_rx_pairs, source.tx_rx_pairs)
    np.testing.assert_array_equal(loaded.pair_path_offsets, [0, 1, 3])
    np.testing.assert_array_equal(loaded.bounce_offsets, [0, 0, 1, 3])
    np.testing.assert_array_equal(loaded.material_ids, source.material_ids)
    np.testing.assert_array_equal(loaded.delays_ns, source.delays_ns)
    assert loaded.metric_is_valid(PathMetric.AOA_AZ_DEG).tolist() == [True, True, False]
    assert loaded.provenance == source.provenance
    assert loaded.sensing is None
    assert loaded.beamforming is None


def test_writer_reader_round_trip_preserves_material_id_256(tmp_path: Path) -> None:
    material_names = ("", *(f"material-{index}" for index in range(1, 257)))
    material_itu_types = ("", *(f"itu-{index}" for index in range(1, 257)))
    source = replace(
        _complete_standard_frame(),
        frame_index=0,
        material_ids=np.array([256, 1, 256], dtype=np.uint16),
        material_names=material_names,
        material_itu_types=material_itu_types,
    )
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="uint16-material-generation",
        compression=None,
    )
    writer.append(source)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="uint16-material-generation",
        frame_set_id="uint16-material-frame-set",
        chunks=[chunk],
        compression={"algorithm": "none"},
        segmentation={"policy": "test"},
        provenance={"fixture": "uint16-material"},
        created_utc="2026-08-05T00:00:00+00:00",
    )
    write_frame_manifest_atomic(tmp_path, manifest)

    reader = PackedHDF5Reader(tmp_path)
    try:
        reader.validate_all_chunks()
        loaded = reader.load_standard_frame(0)
    finally:
        reader.close()

    np.testing.assert_array_equal(loaded.material_ids, [256, 1, 256])
    assert loaded.material_ids.dtype == np.dtype(np.uint16)
    assert loaded.material_names[256] == "material-256"
    assert loaded.material_itu_types[256] == "itu-256"


@pytest.mark.parametrize("operation", ["load", "deep_validation"])
def test_malformed_target_json_reports_physical_and_output_location(
    tmp_path: Path,
    operation: str,
) -> None:
    chunk_path = _write_target_frame_set(tmp_path)
    reader = PackedHDF5Reader(tmp_path)
    with h5py.File(chunk_path, "r+") as h5:
        h5["targets/metadata_json"][0] = "{"

    try:
        with pytest.raises(PackedHDF5Error) as captured:
            if operation == "load":
                reader.load_standard_frame(11)
            else:
                reader.validate_all_chunks()
    finally:
        reader.close()

    message = str(captured.value)
    assert chunk_path.name in message
    assert "output frame 11" in message
    assert "/targets/metadata_json" in message
    assert "target row 0" in message


def test_handle_lru_never_exceeds_bound_and_closes_evictions(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("value", data=np.array([index], dtype=np.int32))
        paths.append(path)

    pool = HDF5HandleLRU(max_open=2)
    observed_handles: list[h5py.File] = []
    for index, path in enumerate(paths):
        with pool.lease(path) as h5:
            observed_handles.append(h5)
            assert int(h5["value"][0]) == index
            assert pool.open_count <= 2

    assert observed_handles[0].id.valid == 0
    assert observed_handles[-1].id.valid == 1
    pool.close()
    assert observed_handles[-1].id.valid == 0


def test_handle_lru_caches_objects_until_that_file_is_evicted(tmp_path: Path) -> None:
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    for index, path in enumerate((first_path, second_path)):
        with h5py.File(path, "w") as h5:
            group = h5.create_group("nested")
            group.create_dataset("value", data=np.array([index], dtype=np.int32))

    pool = HDF5HandleLRU(max_open=1)
    with pool.lease(first_path) as first:
        first_group = first.get("nested")
        first_dataset = first.get("nested/value")
        assert first.get("nested") is first_group
        assert first.get("nested/value") is first_dataset
        assert first["nested/value"] is first_dataset
        first.identity_validated = True

    with pool.lease(second_path) as second:
        assert int(second["nested/value"][0]) == 1

    assert first_dataset.id.valid == 0
    with pool.lease(first_path) as reopened:
        assert reopened.identity_validated is False
        assert reopened.get("nested/value") is not first_dataset
    pool.close()


def test_handle_lru_serializes_concurrent_leases_of_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "shared.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("value", data=np.array([7], dtype=np.int32))

    pool = HDF5HandleLRU(max_open=2)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    errors: list[BaseException] = []

    def hold_first_lease() -> None:
        try:
            with pool.lease(path) as h5:
                assert int(h5["value"][0]) == 7
                first_entered.set()
                assert release_first.wait(timeout=2.0)
        except BaseException as exc:  # pragma: no cover - diagnostic handoff
            errors.append(exc)

    def take_second_lease() -> None:
        try:
            with pool.lease(path) as h5:
                second_entered.set()
                assert int(h5["value"][0]) == 7
        except BaseException as exc:  # pragma: no cover - diagnostic handoff
            errors.append(exc)

    first = Thread(target=hold_first_lease)
    second = Thread(target=take_second_lease)
    first.start()
    assert first_entered.wait(timeout=2.0)
    second.start()
    assert not second_entered.wait(timeout=0.1)

    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert errors == []
    pool.close()


def test_reader_construction_uses_manifest_without_opening_hdf5(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunk_path = tmp_path / "mpc_frames_00000-00000.h5"
    chunk_path.write_bytes(b"not opened during startup")
    chunk = FrameChunkManifest(
        file=chunk_path.name,
        frame_ids=(0,),
        size_bytes=chunk_path.stat().st_size,
        uncompressed_bytes=0,
        topology_id="topology",
        sensing_layout_id="sensing",
    )
    manifest = manifest_from_chunks(
        generation_id="generation",
        frame_set_id="frame-set",
        chunks=[chunk],
        compression={},
        segmentation={},
        provenance={},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(tmp_path, manifest)

    def fail_if_opened(*_args, **_kwargs):
        raise AssertionError("provider startup must not open an HDF5 chunk")

    monkeypatch.setattr(h5py, "File", fail_if_opened)
    reader = PackedHDF5Reader(tmp_path)

    assert reader.frame_ids == [0]
    assert reader.open_handle_count == 0
    reader.close()


def test_chunk_identity_validation_accepts_complete_published_metadata(tmp_path: Path) -> None:
    _write_finalized_identity_frame_set(tmp_path)
    reader = PackedHDF5Reader(tmp_path)
    try:
        reader.validate_chunk_identities()
        assert reader.open_handle_count == 1
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("packed_frame_version", PACKED_MPC_FRAME_VERSION + 1),
        ("publication_state", "partial"),
        ("num_frames", 2),
        ("start_frame", 1),
        ("end_frame", 1),
        ("topology_id", "other-topology"),
        ("sensing_layout_id", "other-sensing"),
    ],
)
def test_chunk_identity_validation_rejects_manifest_attribute_disagreement(
    tmp_path: Path,
    attribute: str,
    value: object,
) -> None:
    _write_finalized_identity_frame_set(
        tmp_path,
        attribute_overrides={attribute: value},
    )
    reader = PackedHDF5Reader(tmp_path)
    try:
        with pytest.raises(PackedHDF5Error, match=attribute):
            reader.validate_chunk_identities()
    finally:
        reader.close()


def test_chunk_identity_validation_requires_finalization_attributes(tmp_path: Path) -> None:
    _write_finalized_identity_frame_set(
        tmp_path,
        omitted_attributes=frozenset({"publication_state"}),
    )
    reader = PackedHDF5Reader(tmp_path)
    try:
        with pytest.raises(PackedHDF5Error, match="publication_state"):
            reader.validate_chunk_identities()
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("stored_frame_ids", "message"),
    [
        (np.array([0], dtype=np.int32), "dtype int64"),
        (np.array([0, 1], dtype=np.int64), r"shape \(1,\)"),
        (np.array([1], dtype=np.int64), "do not match"),
    ],
)
def test_chunk_identity_validation_rejects_invalid_frame_id_dataset(
    tmp_path: Path,
    stored_frame_ids: np.ndarray,
    message: str,
) -> None:
    _write_finalized_identity_frame_set(
        tmp_path,
        stored_frame_ids=stored_frame_ids,
    )
    reader = PackedHDF5Reader(tmp_path)
    try:
        with pytest.raises(PackedHDF5Error, match=message):
            reader.validate_chunk_identities()
    finally:
        reader.close()


def test_chunk_identity_validation_reads_only_frame_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_finalized_identity_frame_set(tmp_path)
    reads: list[tuple[str, Any]] = []
    original_dataset = packed_reader_module._dataset

    class CountingDataset:
        def __init__(self, dataset: h5py.Dataset, path: str) -> None:
            self._dataset = dataset
            self._path = path

        def __getattr__(self, name: str) -> Any:
            return getattr(self._dataset, name)

        def __getitem__(self, selection: Any) -> Any:
            reads.append((self._path, selection))
            return self._dataset[selection]

    def counting_dataset(root: Any, path: str) -> CountingDataset:
        return CountingDataset(original_dataset(root, path), path)

    monkeypatch.setattr(packed_reader_module, "_dataset", counting_dataset)
    reader = PackedHDF5Reader(tmp_path)
    try:
        reader.validate_chunk_identities()
    finally:
        reader.close()

    assert [path for path, _selection in reads] == ["frames/id"]


def test_chunk_structure_validation_rejects_identity_only_chunk(tmp_path: Path) -> None:
    _write_finalized_identity_frame_set(tmp_path)
    reader = PackedHDF5Reader(tmp_path)
    try:
        with pytest.raises(PackedHDF5Error, match=r"missing HDF5 group /static"):
            reader.validate_chunk_structures()
    finally:
        reader.close()


def test_chunk_structure_validation_does_not_read_large_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_real_packed_frame_set(tmp_path)
    reads: list[tuple[str, Any]] = []
    original_dataset = packed_reader_module._dataset

    class CountingDataset:
        def __init__(self, dataset: h5py.Dataset, path: str) -> None:
            self._dataset = dataset
            self._path = path

        def __getattr__(self, name: str) -> Any:
            return getattr(self._dataset, name)

        def __getitem__(self, selection: Any) -> Any:
            reads.append((self._path, selection))
            return self._dataset[selection]

    def counting_dataset(root: Any, path: str) -> CountingDataset:
        return CountingDataset(original_dataset(root, path), path)

    monkeypatch.setattr(packed_reader_module, "_dataset", counting_dataset)
    reader = PackedHDF5Reader(tmp_path)
    try:
        reader.validate_chunk_identities()
        reads.clear()
        reader.validate_chunk_structures()
    finally:
        reader.close()

    read_paths = {path for path, _selection in reads}
    assert "index/frame_pair_path_offsets" in read_paths
    assert "paths/bounce_offsets" in read_paths
    assert "index/frame_target_offsets" in read_paths

    forbidden_payload_reads = {
        "paths/metric_valid_bits",
        *(f"paths/{name}" for name in packed_reader_module._METRIC_DATASETS.values()),
        "bounces/xyz_m",
        "bounces/interaction",
        "bounces/material_id",
        "frames/tx_position_m",
        "frames/tx_orientation_rad",
        "frames/rx_position_m",
        "frames/rx_orientation_rad",
        "targets/position_m",
    }
    assert read_paths.isdisjoint(forbidden_payload_reads)


def test_same_size_chunk_from_another_generation_is_rejected(tmp_path: Path) -> None:
    published_dir = tmp_path / "published"
    other_dir = tmp_path / "other"
    published_dir.mkdir()
    other_dir.mkdir()
    published_chunk = _write_finalized_identity_frame_set(
        published_dir,
        generation_id="generation-a",
    )
    other_chunk = _write_finalized_identity_frame_set(
        other_dir,
        generation_id="generation-b",
    )
    assert published_chunk.stat().st_size == other_chunk.stat().st_size
    shutil.copyfile(other_chunk, published_chunk)

    reader = PackedHDF5Reader(published_dir)
    try:
        with pytest.raises(PackedHDF5Error, match="generation_id"):
            reader.validate_chunk_identities()
    finally:
        reader.close()


def test_normal_frame_load_rejects_unfinalized_chunk(tmp_path: Path) -> None:
    _write_finalized_identity_frame_set(
        tmp_path,
        attribute_overrides={"publication_state": "partial"},
    )
    reader = PackedHDF5Reader(tmp_path)
    try:
        with pytest.raises(PackedHDF5Error, match="publication_state"):
            reader.load_standard_frame(0)
    finally:
        reader.close()
