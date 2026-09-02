"""Focused contract tests for the packed MPC HDF5 v2 chunk writer."""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from shared.frames import packed_hdf5_writer
from shared.frames.contracts import (
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PATH_METRIC_VALIDITY_BITS,
    PathMetric,
)
from shared.frames.json_codec import loads_frame_json
from shared.frames.packed_hdf5_writer import (
    PackedMPCChunkBoundaryError,
    PackedMPCChunkWriter,
    estimate_packed_frame_bytes,
    write_packed_mpc_frame_chunk,
)
from shared.frames.types import StandardMPCFrame


def _frame(frame_idx: int, *, sensing: dict | None = None) -> StandardMPCFrame:
    metric_valid_bits = np.full(
        (3,),
        PATH_METRIC_VALIDITY_BITS[PathMetric.DELAY_NS]
        | PATH_METRIC_VALIDITY_BITS[PathMetric.AOA_AZ_DEG],
        dtype=np.uint8,
    )
    metric_valid_bits[[0, 2]] |= PATH_METRIC_VALIDITY_BITS[PathMetric.PATH_LOSS_DB]
    unavailable = np.full((3,), np.nan, dtype=np.float32)
    return StandardMPCFrame(
        frame_index=frame_idx,
        tx_rx_pairs=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        pair_path_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        bounce_offsets=np.asarray([0, 0, 2, 3], dtype=np.int64),
        tx_positions=np.asarray([[0.0 + frame_idx, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray(
            [[10.0, 0.0, 1.0], [20.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((2, 3), dtype=np.float64),
        tx_names=("tx-main",),
        rx_names=("rx-left", "rx-right"),
        bounce_xyz_m=np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        ),
        interactions=np.asarray([1, 8, 2], dtype=np.uint8),
        material_ids=np.asarray([1, 2, 1], dtype=np.uint16),
        material_names=("", "brick", "glass"),
        material_itu_types=("", "concrete", "glass"),
        delays_ns=np.asarray([0.0, 12.0, 20.0], dtype=np.float32),
        path_loss_db=np.asarray([80.0, np.nan, 95.0], dtype=np.float32),
        aoa_az_deg=np.asarray([-180.0, 45.0, 180.0], dtype=np.float32),
        aoa_el_deg=unavailable.copy(),
        aod_az_deg=unavailable.copy(),
        aod_el_deg=unavailable.copy(),
        metric_valid_bits=metric_valid_bits,
        target_positions_m=np.asarray([[3.0 + frame_idx, 4.0, 0.0]], dtype=np.float64),
        targets_metadata=(
            {
                "name": "vehicle",
                "speed_mps": np.float32(2.5),
            },
        ),
        beamforming={"mode": "test", "weights": np.asarray([1.0, 2.0])},
        sensing=sensing,
        timestamp_s=1.5 + frame_idx,
        recomputed_from_stored_positions=bool(frame_idx),
        provenance={
            "provider": "packed-writer-test",
            "frame_idx": frame_idx,
            "timestamp": 1.5 + frame_idx,
        },
    )


def _strings(dataset: h5py.Dataset) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[:]]


def test_compact_frame_is_ready_for_direct_storage() -> None:
    packed = _frame(4)

    assert not hasattr(packed_hdf5_writer, "pack_standard_mpc_frame")
    np.testing.assert_array_equal(packed.pair_path_offsets, [0, 2, 3])
    np.testing.assert_array_equal(packed.bounce_offsets, [0, 0, 2, 3])
    np.testing.assert_array_equal(
        packed.bounce_xyz_m,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    )
    np.testing.assert_array_equal(packed.interactions, [1, 8, 2])
    assert packed.material_names == ("", "brick", "glass")
    assert packed.material_itu_types == ("", "concrete", "glass")
    np.testing.assert_array_equal(packed.material_ids, [1, 2, 1])

    np.testing.assert_array_equal(packed.delays_ns, [0.0, 12.0, 20.0])
    np.testing.assert_array_equal(
        packed.metric_is_valid(PathMetric.DELAY_NS),
        [True, True, True],
    )
    np.testing.assert_array_equal(
        packed.metric_is_valid(PathMetric.PATH_LOSS_DB),
        [True, False, True],
    )
    assert np.isnan(packed.path_loss_db[1])
    assert np.all(np.isnan(packed.aod_el_deg))
    assert np.all(
        (packed.metric_valid_bits & PATH_METRIC_VALIDITY_BITS[PathMetric.AOD_EL_DEG]) == 0
    )
    assert estimate_packed_frame_bytes(_frame(4)) > packed.bounce_xyz_m.nbytes


def test_writer_rejects_noncanonical_mapping_input(tmp_path: Path) -> None:
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-canonical-contract",
    )
    try:
        with pytest.raises(TypeError, match="complete StandardMPCFrame"):
            writer.append({"frame_index": 0})  # type: ignore[arg-type]
        assert writer.frame_ids == ()
    finally:
        writer.discard()


def test_prepare_does_not_repeat_complete_frame_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(0)

    def reject_second_validation(_frame: StandardMPCFrame) -> None:
        raise AssertionError("complete frames validate once at construction")

    monkeypatch.setattr(StandardMPCFrame, "validate", reject_second_validation)
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-single-validation",
    )
    try:
        prepared = writer.prepare(frame)
        assert prepared.packed is frame
    finally:
        writer.discard()


def test_append_does_not_retain_the_canonical_frame(tmp_path: Path) -> None:
    """Release caller-owned frame buffers as soon as an append completes."""

    frame = _frame(0)
    geometry_ref = weakref.ref(frame.bounce_xyz_m)
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-frame-lifetime",
    )
    try:
        writer.append(frame)
        del frame
        gc.collect()
        assert geometry_ref() is None
    finally:
        writer.discard()


def test_writer_appends_global_offsets_sensing_and_material_catalog(
    tmp_path: Path,
) -> None:
    first = _frame(
        0,
        sensing={
            "range_profile": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "cir": np.asarray(
                [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
                dtype=np.complex64,
            ),
            "config": {"bandwidth_hz": np.float64(100e6)},
            "config_resolved": {"bandwidth_hz": np.float64(100e6), "fft_size": 128},
        },
    )
    second = _frame(
        1,
        sensing={
            "range_doppler_map": np.arange(4, dtype=np.float32).reshape((2, 2)),
            "cir": np.asarray([[9.0j], [10.0j], [11.0j]], dtype=np.complex64),
            "config": {"bandwidth_hz": 100e6},
            "config_resolved": {"bandwidth_hz": 100e6, "fft_size": 128},
        },
    )

    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-test-generation",
        partial_name=".writer-test.h5.partial",
    )
    assert writer.partial_path.exists()
    assert not (tmp_path / "mpc_frames_00000-00001.h5").exists()
    writer.append(first)
    writer.append(second)
    assert writer.current_frame_ids == (0, 1)
    assert writer.current_frame_count == 2
    assert writer.uncompressed_bytes > 0

    manifest = writer.finalize_to_range_name()
    output_path = tmp_path / manifest.file
    assert manifest.file == "mpc_frames_00000-00001.h5"
    assert manifest.frame_ids == (0, 1)
    assert manifest.size_bytes == output_path.stat().st_size
    assert manifest.uncompressed_bytes == writer.uncompressed_bytes
    assert not writer.partial_path.exists()

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["schema_version"] == MPC_HDF5_SCHEMA_VERSION
        assert handle.attrs["storage_layout"] == MPC_HDF5_LAYOUT
        assert handle.attrs["file_kind"] == "mpc_frames"
        assert handle.attrs["generation_id"] == "writer-test-generation"
        assert handle.attrs["publication_state"] == "complete"
        np.testing.assert_array_equal(handle["frames/id"][:], [0, 1])
        np.testing.assert_allclose(handle["frames/timestamp_s"][:], [1.5, 2.5])
        np.testing.assert_array_equal(handle["frames/recomputed"][:], [0, 1])
        assert handle["frames/tx_position_m"].attrs["units"] == "m"
        assert handle["frames/tx_orientation_rad"].attrs["units"] == "rad"
        assert len(handle["frames/tx_position_m"].dims[0]) == 1
        assert handle["frames/tx_position_m"].dims[0][0].name == "/frames/id"

        np.testing.assert_array_equal(
            handle["index/frame_pair_path_offsets"][:],
            [[0, 2, 3], [3, 5, 6]],
        )
        np.testing.assert_array_equal(
            handle["paths/bounce_offsets"][:],
            [0, 0, 2, 3, 3, 5, 6],
        )
        assert handle["paths/bounce_offsets"].chunks[0] > 1
        np.testing.assert_array_equal(
            handle["index/frame_target_offsets"][:],
            [0, 1, 2],
        )
        assert handle["index/frame_target_offsets"].chunks[0] > 1
        np.testing.assert_array_equal(
            handle["bounces/xyz_m"][:3],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )
        assert handle["bounces/xyz_m"].attrs["units"] == "m"
        assert handle["bounces/interaction"].dtype == np.dtype(np.uint8)
        assert handle["bounces/material_id"].dtype == np.dtype(np.uint16)
        assert _strings(handle["static/materials/name"]) == ["", "brick", "glass"]
        assert _strings(handle["static/materials/itu_type"]) == [
            "",
            "concrete",
            "glass",
        ]

        for metric in PathMetric:
            assert handle[f"paths/{metric.value}"].shape == (6,)
            assert handle[f"paths/{metric.value}"].dtype == np.dtype(np.float32)
        assert handle["paths/metric_valid_bits"].shape == (6,)

        np.testing.assert_array_equal(
            handle["sensing/fixed/range_profile/present"][:],
            [1, 0],
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/range_profile/data"][0],
            [1.0, 2.0, 3.0],
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/range_doppler_map/present"][:],
            [0, 1],
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/range_doppler_map/data"][1],
            np.arange(4, dtype=np.float32).reshape((2, 2)),
        )
        np.testing.assert_array_equal(
            handle["sensing/ragged/cir/offsets"][:],
            [0, 4, 7],
        )
        assert handle["sensing/ragged/cir/offsets"].chunks[0] > 1
        np.testing.assert_array_equal(
            handle["sensing/ragged/cir/shapes"][:],
            [[2, 2], [3, 1]],
        )
        np.testing.assert_array_equal(
            handle["sensing/ragged/cir/present"][:],
            [1, 1],
        )
        np.testing.assert_allclose(
            handle["sensing/ragged/cir/values"][:],
            np.concatenate(
                [
                    first.sensing["cir"].reshape(-1),
                    second.sensing["cir"].reshape(-1),
                ]
            ),
        )
        assert handle["sensing/ragged/cir/values"].dtype == np.dtype(np.complex64)
        metadata = loads_frame_json(handle["frames/sensing_metadata_json"][0])
        assert metadata == {}
        assert loads_frame_json(handle["static/sensing_config_json"][()]) == {
            "config": {"bandwidth_hz": 100e6},
            "config_resolved": {"bandwidth_hz": 100e6, "fft_size": 128},
        }
        assert loads_frame_json(handle["targets/metadata_json"][0])["name"] == "vehicle"


def test_writer_rebases_independent_material_catalogs_without_repacking(
    tmp_path: Path,
) -> None:
    first = _frame(0)
    second = replace(
        _frame(1),
        material_ids=np.asarray([2, 1, 2], dtype=np.uint16),
        material_names=("", "glass", "brick"),
        material_itu_types=("", "glass", "concrete"),
    )
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-material-rebase",
    )

    first_prepared = writer.prepare(first)
    assert first_prepared.packed is first
    assert first_prepared.material_ids is first.material_ids
    writer.append_prepared(first_prepared)

    second_prepared = writer.prepare(second)
    assert second_prepared.packed is second
    assert second_prepared.material_ids is not second.material_ids
    np.testing.assert_array_equal(second_prepared.material_ids, [1, 2, 1])
    writer.append_prepared(second_prepared)
    chunk = writer.finalize_to_range_name()

    with h5py.File(tmp_path / chunk.file, "r") as handle:
        assert _strings(handle["static/materials/name"]) == ["", "brick", "glass"]
        np.testing.assert_array_equal(
            handle["bounces/material_id"][:],
            [1, 2, 1, 1, 2, 1],
        )


def test_incompatible_topology_and_fixed_sensing_shape_require_new_chunk(
    tmp_path: Path,
) -> None:
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-test-generation",
    )
    writer.append(
        _frame(
            0,
            sensing={
                "range_profile": np.ones((3,), dtype=np.float32),
                "cir": np.ones((2, 2), dtype=np.complex64),
            },
        )
    )

    changed_topology = replace(_frame(1), rx_names=("renamed", "rx-right"))
    with pytest.raises(PackedMPCChunkBoundaryError, match="topology"):
        writer.append(changed_topology)
    assert writer.frame_ids == (0,)

    changed_sensing_shape = _frame(
        1,
        sensing={"range_profile": np.ones((4,), dtype=np.float32)},
    )
    with pytest.raises(PackedMPCChunkBoundaryError, match="shape or dtype"):
        writer.append(changed_sensing_shape)
    assert writer.frame_ids == (0,)

    changed_cir_dtype = _frame(
        1,
        sensing={
            "range_profile": np.ones((3,), dtype=np.float32),
            "cir": np.ones((2, 2), dtype=np.complex128),
        },
    )
    with pytest.raises(PackedMPCChunkBoundaryError, match="cir dtype"):
        writer.append(changed_cir_dtype)
    assert writer.frame_ids == (0,)

    with pytest.raises(ValueError, match="strictly increasing"):
        writer.append(_frame(0))
    writer.discard()
    assert not writer.partial_path.exists()


def test_convenience_writer_supports_gzip_four(tmp_path: Path) -> None:
    chunk = write_packed_mpc_frame_chunk(
        tmp_path,
        [_frame(7)],
        generation_id="writer-test-generation",
        compression="gzip-4",
    )

    assert chunk.frame_ids == (7,)
    assert not (tmp_path / "frames_manifest.json").exists()
    assert "write_packed_mpc_frame_chunk" not in packed_hdf5_writer.__all__
    with h5py.File(tmp_path / chunk.file, "r") as handle:
        dataset = handle["paths/delay_ns"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 4
        assert handle["sensing/ragged/cir/shapes"].shape == (1, 0)
        assert handle["sensing/ragged/cir/values"].shape == (0,)
        assert handle["sensing/ragged/cir/values"].dtype == np.dtype(np.complex64)


def test_unserializable_metadata_does_not_partially_append(tmp_path: Path) -> None:
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-test-generation",
    )
    writer.append(_frame(0))
    invalid = replace(
        _frame(1),
        provenance={"unserializable": object()},
    )

    with pytest.raises(TypeError, match="not JSON serializable"):
        writer.append(invalid)
    assert writer.frame_ids == (0,)

    partial_path = writer.partial_path
    writer.close()
    with h5py.File(partial_path, "r") as handle:
        np.testing.assert_array_equal(handle["frames/id"][:], [0])
        assert handle["frames/source_json"].shape == (1,)
        assert handle["index/frame_pair_path_offsets"].shape == (1, 3)
        assert handle["paths/delay_ns"].shape == (3,)
    writer.discard()
