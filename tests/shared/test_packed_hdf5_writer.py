"""Focused contract tests for the packed MPC HDF5 v2 chunk writer."""

from __future__ import annotations

import gc
import weakref
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid5

import h5py
import numpy as np
import pytest

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    UnsupportedSensingSchemaError,
    build_sensing_temporal_contract,
    canonical_processing_recipe_id,
    canonical_sensing_hash,
    finalize_sensing_v4_publication,
)
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
from shared.sensing.products import (
    SENSING_PRODUCT_CONTRACT_VERSION,
    TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)

_SENSING_GENERATION_ID = "9dced48a-334f-48c5-a359-e39071a90e73"


def _writer_execution_config() -> dict:
    """Return one complete coherent native-v4 execution envelope."""

    return {
        "contract_version": 1,
        "fft_size_range": 128,
        "fft_size_doppler": 2,
        "sampling_frequency": 100e6,
        "prf_hz": 1000.0,
        "carrier_freq_hz": 3.5e9,
        "range_mode": "monostatic",
        "processing_mode": "coherent_cpi",
        "processing_recipe": {
            "window_range": "hann",
            "window_doppler": "hann",
        },
        "acquisition": {
            "sample_source": "coherent_one_solve",
            "rt_solve_stride_scene_steps": 2,
            "sample_period_s": 0.001,
            "sample_rate_hz": 1000.0,
            "cpi_length_samples": 2,
            "cpi_hop_samples": 2,
            "overlap_samples": 0,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "sionna_intra_solve_phase_coherent",
        },
        "processing": {
            "mode": "coherent_cpi",
            "strategy": "coherent_cpi_fft",
            "fft_size_doppler": 2,
            "result_stride_samples": 2,
            "product_semantics": "cir_point_bin_slow_time_fft",
            "evidence_status": "explicit_writer_fixture",
        },
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "tx_device_count": 1,
            "rx_device_count": 1,
            "tx_elements": 1,
            "rx_elements": 2,
            "rx_array_processing": "planar_fft",
        },
    }


def _schema_v4_sensing(frame_idx: int, sensing: dict | None) -> dict | None:
    """Add valid temporal identity to sensing payloads used by codec tests."""
    if not sensing:
        return sensing
    if sensing.get("schema_version") == SENSING_SCHEMA_VERSION:
        return {
            **sensing,
            "product_contract_version": sensing.get(
                "product_contract_version", SENSING_PRODUCT_CONTRACT_VERSION
            ),
            "products": sensing.get("products", {}),
        }
    config_resolved = _writer_execution_config()
    sample_time_s = 1.5 + frame_idx
    sample_times_s = (sample_time_s - 0.001, sample_time_s)
    acquisition_id = str(uuid5(UUID(_SENSING_GENERATION_ID), f"writer-test:{frame_idx}"))
    temporal = build_sensing_temporal_contract(
        clock_id=_SENSING_GENERATION_ID,
        acquisition_id=acquisition_id,
        acquisition_config_id=canonical_sensing_hash(config_resolved),
        pair_id=canonical_sensing_hash({"tx_sensor_id": "writer-tx", "rx_sensor_id": "writer-rx"}),
        tx_sensor_id="writer-tx",
        rx_sensor_id="writer-rx",
        samples=[
            {
                "sample_index": sample_index,
                "sample_time_s": sample_epoch_s,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": _SENSING_GENERATION_ID,
                    "scene_frame_id": frame_idx,
                    "sample_offset": sample_index,
                },
            }
            for sample_index, sample_epoch_s in enumerate(sample_times_s)
        ],
        processing_run_id=_SENSING_GENERATION_ID,
        processing_recipe_id=canonical_processing_recipe_id(config_resolved["processing_recipe"]),
        reference_time_s=0.5 * (sample_times_s[0] + sample_times_s[-1]),
        reference_time_policy="cpi_center",
        display_scene_frame_id=frame_idx,
        display_policy="nearest_at_or_after_reference",
        display_reason="The frame carries the scene state at the final CPI sample.",
    )
    result = {
        **sensing,
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "config": deepcopy(config_resolved),
        "config_resolved": config_resolved,
        "product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
        "products": {},
    }
    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    cir = result.get("cir_by_pair_rx_element")
    if isinstance(cir, np.ndarray):
        element_count = int(cir.shape[1])
        range_count = int(cir.shape[2])
        result["rx_element_axis"] = [
            {"element_index": index, "row": 0, "column": index} for index in range(element_count)
        ]
        result["cir_range_axis_m"] = np.arange(range_count, dtype=np.float64)
        add_product_spec(
            result,
            data_key="cir_by_pair_rx_element",
            kind="delay_domain_cir",
            axes=(
                pair_axis,
                catalog_axis("rx_element", size=element_count, catalog_key="rx_element_axis"),
                dense_axis(
                    "range",
                    size=range_count,
                    unit="m",
                    coordinate_key="cir_range_axis_m",
                ),
            ),
            representation="complex_voltage",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
        )
    profile = result.get("range_profile_by_pair")
    if isinstance(profile, np.ndarray):
        range_count = int(profile.shape[1])
        result["range_profile_axis_m"] = np.arange(range_count, dtype=np.float64)
        add_product_spec(
            result,
            data_key="range_profile_by_pair",
            kind="range_profile",
            axes=(
                pair_axis,
                dense_axis(
                    "range",
                    size=range_count,
                    unit="m",
                    coordinate_key="range_profile_axis_m",
                ),
            ),
            representation="power_linear",
            unit="1",
            temporal_support=TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
        )
    range_doppler = result.get("range_doppler_by_pair")
    if isinstance(range_doppler, np.ndarray):
        doppler_count = int(range_doppler.shape[1])
        range_count = int(range_doppler.shape[2])
        result["range_doppler_axis_hz"] = np.arange(doppler_count, dtype=np.float64)
        result["range_doppler_range_axis_m"] = np.arange(range_count, dtype=np.float64)
        add_product_spec(
            result,
            data_key="range_doppler_by_pair",
            kind="range_doppler",
            axes=(
                pair_axis,
                dense_axis(
                    "doppler_frequency",
                    size=doppler_count,
                    unit="Hz",
                    coordinate_key="range_doppler_axis_hz",
                ),
                dense_axis(
                    "range",
                    size=range_count,
                    unit="m",
                    coordinate_key="range_doppler_range_axis_m",
                ),
            ),
            representation="power_linear",
            unit="1",
        )
    materialize_product_specs(result)
    finalize_sensing_v4_publication(result)
    return result


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
        sensing=_schema_v4_sensing(frame_idx, sensing),
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


def test_writer_rejects_old_sensing_without_mutating_the_chunk(tmp_path: Path) -> None:
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-schema-boundary",
    )
    try:
        legacy = replace(_frame(0), sensing={"schema_version": 2, "range_profile": [1.0]})
        with pytest.raises(UnsupportedSensingSchemaError, match="Regenerate this sensing dataset"):
            writer.append(legacy)
        assert writer.frame_ids == ()
        assert writer.frame_count == 0
    finally:
        writer.discard()


def test_writer_rejects_undescribed_v4_numeric_payload_without_mutating_chunk(
    tmp_path: Path,
) -> None:
    source = _frame(
        0,
        sensing={"range_profile_by_pair": np.ones((1, 3), dtype=np.float32)},
    )
    invalid_sensing = dict(source.sensing or {})
    invalid_sensing["unpublished_debug_array"] = np.ones((2,), dtype=np.float32)
    invalid = replace(source, sensing=invalid_sensing)
    writer = PackedMPCChunkWriter(
        tmp_path,
        generation_id="writer-described-product-boundary",
    )
    try:
        with pytest.raises(
            ValueError,
            match="forbids undeclared numeric payloads: unpublished_debug_array",
        ):
            writer.append(invalid)
        assert writer.frame_ids == ()
        assert writer.frame_count == 0
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
            "range_profile_by_pair": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            "cir_by_pair_rx_element": np.asarray(
                [[[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]]],
                dtype=np.complex64,
            ),
        },
    )
    second = _frame(
        1,
        sensing={
            "range_doppler_by_pair": np.arange(4, dtype=np.float32).reshape((1, 2, 2)),
            "cir_by_pair_rx_element": np.asarray(
                [[[9.0j, 10.0j], [11.0j, 12.0j]]], dtype=np.complex64
            ),
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
            handle["sensing/fixed/range_profile_by_pair/present"][:],
            [1, 0],
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/range_profile_by_pair/data"][0],
            [[1.0, 2.0, 3.0]],
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/range_doppler_by_pair/present"][:],
            [0, 1],
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/range_doppler_by_pair/data"][1],
            np.arange(4, dtype=np.float32).reshape((1, 2, 2)),
        )
        np.testing.assert_array_equal(
            handle["sensing/fixed/cir_by_pair_rx_element/data"][:],
            np.stack(
                [
                    first.sensing["cir_by_pair_rx_element"],
                    second.sensing["cir_by_pair_rx_element"],
                ]
            ),
        )
        assert handle["sensing/fixed/cir_by_pair_rx_element/data"].dtype == np.dtype(np.complex64)
        metadata = loads_frame_json(handle["frames/sensing_metadata_json"][0])
        assert metadata["schema_version"] == SENSING_SCHEMA_VERSION
        assert metadata["temporal"]["clock"]["clock_id"] == _SENSING_GENERATION_ID
        assert metadata["temporal"]["acquisition"]["samples"] == [
            {
                "sample_index": 0,
                "sample_time_s": 1.499,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": _SENSING_GENERATION_ID,
                    "scene_frame_id": 0,
                    "sample_offset": 0,
                },
            },
            {
                "sample_index": 1,
                "sample_time_s": 1.5,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": _SENSING_GENERATION_ID,
                    "scene_frame_id": 0,
                    "sample_offset": 1,
                },
            },
        ]
        expected_config = _writer_execution_config()
        assert loads_frame_json(handle["static/sensing_config_json"][()]) == {
            "config": expected_config,
            "config_resolved": expected_config,
        }
        assert loads_frame_json(handle["targets/metadata_json"][0])["name"] == "vehicle"


@pytest.mark.parametrize("prior_frames", [0, 3])
def test_large_sensing_products_append_without_rewriting_other_frames(
    tmp_path: Path, prior_frames: int
) -> None:
    """One large product must occupy its own chunks, including late readiness."""

    values = np.arange(128 * 256, dtype=np.float64).reshape(1, 128, 256)
    frames = [
        _frame(index, sensing={"range_profile_by_pair": np.ones((1, 256))})
        for index in range(prior_frames)
    ]
    frames.extend(
        _frame(
            prior_frames + offset,
            sensing={"range_doppler_by_pair": values + offset},
        )
        for offset in range(3)
    )
    chunk = write_packed_mpc_frame_chunk(
        tmp_path, frames, generation_id="writer-independent-sensing-frames"
    )

    with h5py.File(tmp_path / chunk.file, "r") as handle:
        group = handle["sensing/fixed/range_doppler_by_pair"]
        data = group["data"]
        assert data.chunks is not None
        # No compressed chunk may span adjacent large sensing frames: that
        # causes every append to repeatedly recompress preceding products.
        assert data.chunks[0] == 1
        np.testing.assert_array_equal(group["present"][:], [0] * prior_frames + [1] * 3)
        for index in range(prior_frames):
            np.testing.assert_array_equal(data[index], np.zeros_like(values))
        for offset in range(3):
            np.testing.assert_array_equal(data[prior_frames + offset], values + offset)


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
                "range_profile_by_pair": np.ones((1, 3), dtype=np.float32),
                "cir_by_pair_rx_element": np.ones((1, 2, 2), dtype=np.complex64),
            },
        )
    )

    changed_topology = replace(_frame(1), rx_names=("renamed", "rx-right"))
    with pytest.raises(PackedMPCChunkBoundaryError, match="topology"):
        writer.append(changed_topology)
    assert writer.frame_ids == (0,)

    changed_sensing_shape = _frame(
        1,
        sensing={"range_profile_by_pair": np.ones((1, 4), dtype=np.float32)},
    )
    with pytest.raises(PackedMPCChunkBoundaryError, match="shape or dtype"):
        writer.append(changed_sensing_shape)
    assert writer.frame_ids == (0,)

    changed_cir_dtype = _frame(
        1,
        sensing={
            "range_profile_by_pair": np.ones((1, 3), dtype=np.float32),
            "cir_by_pair_rx_element": np.ones((1, 2, 2), dtype=np.complex128),
        },
    )
    with pytest.raises(PackedMPCChunkBoundaryError, match="shape or dtype"):
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
