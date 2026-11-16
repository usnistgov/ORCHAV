"""Focused tests for packed HDF5 reader primitives and canonical reads."""

from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from typing import Any
from uuid import UUID, uuid5

import h5py
import numpy as np
import pytest

import shared.frames.packed_hdf5 as packed_reader_module
from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    UnsupportedSensingSchemaError,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
    finalize_sensing_v4_publication,
)
from shared.frames.contracts import (
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
    PATH_METRIC_VALIDITY_BITS,
    FrameComponent,
    FrameReadRequest,
    PathMetric,
)
from shared.frames.json_codec import dumps_frame_json, loads_frame_json
from shared.frames.manifest import (
    FrameChunkManifest,
    load_frame_manifest,
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
from shared.sensing.products import (
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)

_SENSING_GENERATION_ID = "fe98d1a6-148f-4af5-8d12-7863b31b6d6b"


def _reader_execution_config() -> dict[str, Any]:
    """Return the complete analytic-oracle envelope used by reader fixtures."""

    return {
        "contract_version": 1,
        "fft_size_range": 3,
        "fft_size_doppler": 2,
        "sampling_frequency": 1.0e9,
        "prf_hz": 1000.0,
        "carrier_freq_hz": 28e9,
        "range_mode": "bistatic",
        "processing_mode": "analytic_path_doppler_oracle",
        "processing_recipe": {
            "window_range": "rectangular",
            "window_doppler": "rectangular",
        },
        "acquisition": {
            "sample_source": "analytic_path_snapshot",
            "rt_solve_stride_scene_steps": 1,
            "cpi_length_samples": 1,
            "cpi_hop_samples": 1,
            "overlap_samples": 0,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "analytic_path_snapshot_phase",
        },
        "processing": {
            "mode": "analytic_path_doppler_oracle",
            "strategy": "analytic_path_doppler_oracle",
            "fft_size_doppler": 2,
            "result_stride_samples": 1,
            "product_semantics": "analytic_path_doppler_projection",
            "evidence_status": "packed_reader_fixture",
        },
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "tx_device_count": 1,
            "rx_device_count": 1,
            "tx_elements": 1,
            "rx_elements": 1,
            "rx_array_processing": "siso",
        },
    }


def _v4_sensing(
    frame_id: int, sample_time_s: float, *, window_doppler: str = "rectangular"
) -> dict[str, Any]:
    config_resolved = _reader_execution_config()
    config_resolved["processing_recipe"]["window_doppler"] = window_doppler
    temporal = build_sensing_temporal_contract(
        clock_id=_SENSING_GENERATION_ID,
        acquisition_id=str(uuid5(UUID(_SENSING_GENERATION_ID), f"packed-reader:{frame_id}")),
        acquisition_config_id=canonical_sensing_hash(config_resolved),
        pair_id=canonical_sensing_hash({"tx_sensor_id": "reader-tx", "rx_sensor_id": "reader-rx"}),
        tx_sensor_id="reader-tx",
        rx_sensor_id="reader-rx",
        samples=[
            {
                "sample_index": 0,
                "sample_time_s": sample_time_s,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": _SENSING_GENERATION_ID,
                    "scene_frame_id": frame_id,
                    "sample_offset": 0,
                },
            }
        ],
        processing_run_id=_SENSING_GENERATION_ID,
        processing_recipe_id=canonical_sensing_hash(config_resolved["processing_recipe"]),
        reference_time_s=sample_time_s,
        reference_time_policy="sample_time",
        display_scene_frame_id=frame_id,
        display_policy="exact",
        display_reason="Single-sample reader fixture.",
    )
    sensing: dict[str, Any] = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "processing_strategy": "analytic_path_doppler_oracle",
        "temporal": temporal,
        "config_resolved": config_resolved,
        "config": deepcopy(config_resolved),
        "doppler_axis_hz": np.array([-1.0, 1.0], dtype=np.float64),
        "range_axis_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        "range_doppler_by_pair": np.arange(6, dtype=np.float32).reshape((1, 2, 3)),
    }
    add_product_spec(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler",
        axes=(
            catalog_axis("pair", size=1, catalog_key="pair_axis"),
            dense_axis(
                "doppler_frequency",
                size=2,
                unit="Hz",
                coordinate_key="doppler_axis_hz",
            ),
            dense_axis(
                "range",
                size=3,
                unit="m",
                coordinate_key="range_axis_m",
            ),
        ),
        representation="power_linear",
        unit="1",
    )
    materialize_product_specs(sensing)
    finalize_sensing_v4_publication(sensing)
    return sensing


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


def _write_v4_sensing_frame_set(
    root: Path, *, sensing: dict[str, Any] | None = None
) -> tuple[StandardMPCFrame, Path]:
    source = replace(
        _complete_standard_frame(),
        frame_index=0,
        timestamp_s=0.125,
        sensing=_v4_sensing(0, 0.125) if sensing is None else sensing,
    )
    writer = PackedMPCChunkWriter(
        root,
        generation_id=_SENSING_GENERATION_ID,
        compression=None,
    )
    writer.append(source)
    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id=_SENSING_GENERATION_ID,
        frame_set_id="v4-sensing-frame-set",
        chunks=[chunk],
        compression={"algorithm": "none"},
        segmentation={"policy": "test"},
        provenance={"fixture": "v4-sensing"},
        created_utc="2026-08-26T00:00:00+00:00",
    )
    write_frame_manifest_atomic(root, manifest)
    return source, root / chunk.file


@pytest.mark.parametrize("malformed", [False, True])
def test_batched_sensing_descriptors_stay_selective_and_validate(tmp_path, monkeypatch, malformed):
    writer = PackedMPCChunkWriter(tmp_path, generation_id=_SENSING_GENERATION_ID)
    for step in range(3):
        writer.append(
            replace(
                _complete_standard_frame(),
                frame_index=step,
                sensing=_v4_sensing(step, float(step)),
            )
        )
    chunk = writer.finalize_to_range_name()
    write_frame_manifest_atomic(
        tmp_path,
        manifest_from_chunks(
            generation_id=_SENSING_GENERATION_ID,
            frame_set_id="batch-sensing",
            chunks=[chunk],
            compression={},
            segmentation={},
            provenance={},
            created_utc="2026-08-26T00:00:00+00:00",
        ),
    )
    if malformed:
        with h5py.File(tmp_path / chunk.file, "r+") as h5:
            h5["frames/sensing_metadata_json"][1] = "{"
    reader = PackedHDF5Reader(tmp_path)

    original_product = reader._read_sensing_product

    def forbid_dense(h5, product, row):
        assert product == "products", "Descriptor scans must not load numerical sensing products"
        return original_product(h5, product, row)

    monkeypatch.setattr(reader, "_read_sensing_product", forbid_dense)
    request = FrameReadRequest(
        components=frozenset({FrameComponent.SENSING}),
        sensing_products=frozenset({"products"}),
    )
    try:
        if malformed:
            with pytest.raises(ValueError):
                list(reader.iter_projections([0, 1, 2], request))
        else:
            expected = [reader.load_projection(step, request) for step in range(3)]
            actual = list(reader.iter_projections([0, 1, 2], request))
            for single, batched in zip(expected, actual, strict=True):
                assert dumps_frame_json(single.frame.sensing) == dumps_frame_json(
                    batched.frame.sensing
                )
    finally:
        reader.close()


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


def test_sensing_v4_round_trips_without_changing_generic_hdf5_layout(
    tmp_path: Path,
) -> None:
    source, chunk_path = _write_v4_sensing_frame_set(tmp_path)

    with h5py.File(chunk_path, "r") as h5:
        assert h5.attrs["schema_version"] == MPC_HDF5_SCHEMA_VERSION
        assert h5.attrs["storage_layout"] == MPC_HDF5_LAYOUT
        stored_metadata = loads_frame_json(h5["frames/sensing_metadata_json"][0])
        assert stored_metadata["schema_version"] == SENSING_SCHEMA_VERSION
        assert stored_metadata["temporal"] == source.sensing["temporal"]

    reader = PackedHDF5Reader(tmp_path)
    try:
        loaded = reader.load_standard_frame(0)
    finally:
        reader.close()

    assert loaded.sensing is not None
    assert loaded.sensing["schema_version"] == SENSING_SCHEMA_VERSION
    assert loaded.sensing["temporal"] == source.sensing["temporal"]
    np.testing.assert_array_equal(
        loaded.sensing["range_doppler_by_pair"],
        source.sensing["range_doppler_by_pair"],
    )


def _read_sensing_for_integrity(reader: PackedHDF5Reader, mode: str) -> None:
    request = FrameReadRequest(components=frozenset({FrameComponent.SENSING}))
    if mode == "full":
        reader.load_standard_frame(0)
    elif mode == "projection":
        reader.load_projection(0, request)
    elif mode == "iterator":
        list(reader.iter_projections([0], request))
    elif mode == "deep":
        reader.validate_all_chunks()
    else:
        raise AssertionError(mode)


@pytest.mark.parametrize("mode", ["full", "projection", "iterator", "deep"])
@pytest.mark.parametrize("mutation", ["carrier", "recipe"])
def test_same_size_static_sensing_mutation_is_rejected(
    tmp_path: Path, mode: str, mutation: str
) -> None:
    sensing = _v4_sensing(0, 0.125, window_doppler="hann")
    _source, chunk_path = _write_v4_sensing_frame_set(tmp_path, sensing=sensing)
    original = chunk_path.read_bytes()
    manifest = (tmp_path / "frames_manifest.json").read_bytes()
    with h5py.File(chunk_path, "r") as h5:
        raw = bytes(h5["static/sensing_config_json"][()])
    config = loads_frame_json(raw)
    for mirror in ("config", "config_resolved"):
        if mutation == "carrier":
            config[mirror]["carrier_freq_hz"] = 14e9
        else:
            config[mirror]["processing_recipe"]["window_doppler"] = "none"
    changed = dumps_frame_json(config).encode("utf-8")
    assert len(changed) == len(raw)
    assert original.count(raw) == 1
    chunk_path.write_bytes(original.replace(raw, changed))
    assert chunk_path.stat().st_size == len(original)
    assert (tmp_path / "frames_manifest.json").read_bytes() == manifest

    reader = PackedHDF5Reader(tmp_path)
    try:
        reader.load_projection(
            0, FrameReadRequest(components=frozenset({FrameComponent.PROVENANCE}))
        )
        with pytest.raises(PackedHDF5Error, match="sensing_layout_id.*static sensing payload"):
            _read_sensing_for_integrity(reader, mode)
    finally:
        reader.close()


@pytest.mark.parametrize("mode", ["full", "projection", "iterator", "deep"])
def test_persisted_recipe_identity_must_match_resolved_recipe(tmp_path: Path, mode: str) -> None:
    _source, chunk_path = _write_v4_sensing_frame_set(tmp_path)
    original = chunk_path.read_bytes()
    with h5py.File(chunk_path, "r") as h5:
        raw = bytes(h5["frames/sensing_metadata_json"][0])
    metadata = loads_frame_json(raw)
    metadata["temporal"]["result"]["processing_recipe_id"] = "0" * 64
    changed = dumps_frame_json(metadata).encode("utf-8")
    assert len(changed) == len(raw)
    assert original.count(raw) == 1
    chunk_path.write_bytes(original.replace(raw, changed))

    reader = PackedHDF5Reader(tmp_path)
    try:
        with pytest.raises(ValueError, match="processing_recipe_id must match"):
            _read_sensing_for_integrity(reader, mode)
    finally:
        reader.close()


@pytest.mark.parametrize("mode", ["full", "projection", "iterator", "deep"])
@pytest.mark.parametrize("conflicting", [False, True])
def test_inline_sensing_config_cannot_override_verified_static_metadata(
    tmp_path: Path, mode: str, conflicting: bool
) -> None:
    sensing = _v4_sensing(0, 0.125)
    sensing["padding"] = "x" * 6000
    _source, chunk_path = _write_v4_sensing_frame_set(tmp_path, sensing=sensing)
    original = chunk_path.read_bytes()
    with h5py.File(chunk_path, "r") as h5:
        raw = bytes(h5["frames/sensing_metadata_json"][0])
        static = loads_frame_json(h5["static/sensing_config_json"][()])
    metadata = loads_frame_json(raw)
    for key in ("config", "config_resolved"):
        metadata[key] = deepcopy(static[key])
        if conflicting:
            metadata[key]["carrier_freq_hz"] = 14e9
    added_bytes = len(dumps_frame_json(metadata).encode("utf-8")) - len(raw)
    metadata["padding"] = metadata["padding"][: 6000 - added_bytes]
    changed = dumps_frame_json(metadata).encode("utf-8")
    assert len(changed) == len(raw)
    assert original.count(raw) == 1
    chunk_path.write_bytes(original.replace(raw, changed))

    reader = PackedHDF5Reader(tmp_path)
    try:
        if conflicting:
            with pytest.raises(PackedHDF5Error, match="conflicts with verified static metadata"):
                _read_sensing_for_integrity(reader, mode)
        else:
            _read_sensing_for_integrity(reader, mode)
    finally:
        reader.close()


def test_static_sensing_validation_is_cached_and_keeps_numeric_products_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_v4_sensing_frame_set(tmp_path)
    calls = []
    load = packed_reader_module._load_static_sensing_config
    read = h5py.Dataset.__getitem__

    def counted_load(h5):
        calls.append(h5.filename)
        return load(h5)

    def forbid_numeric_products(dataset, selection):
        assert not dataset.name.startswith(("/sensing/fixed/", "/sensing/ragged/"))
        return read(dataset, selection)

    monkeypatch.setattr(packed_reader_module, "_load_static_sensing_config", counted_load)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbid_numeric_products)
    reader = PackedHDF5Reader(tmp_path)
    try:
        reader.load_projection(
            0, FrameReadRequest(components=frozenset({FrameComponent.PROVENANCE}))
        )
        assert not calls
        request = FrameReadRequest(
            components=frozenset({FrameComponent.SENSING}),
            sensing_products=frozenset({"products"}),
        )
        for _ in range(3):
            reader.load_projection(0, request)
        assert len(calls) == 1
    finally:
        reader.close()


def test_absent_cir_layout_rejects_scalar_values_without_reading_them(tmp_path: Path) -> None:
    _write_real_packed_frame_set(tmp_path)
    chunk = load_frame_manifest(tmp_path).chunks[0]
    with h5py.File(tmp_path / chunk.file, "r+") as h5:
        values = h5["sensing/ragged/cir/values"]
        values.resize((1,))
        values[0] = 1.0
        h5["sensing/ragged/cir/offsets"][:] = [0, 1]
        h5["sensing/ragged/cir/present"][:] = [1]
        assert h5["sensing/ragged/cir/shapes"].shape == (1, 0)
        with pytest.raises(PackedHDF5Error, match="absent CIR layout.*empty values"):
            packed_reader_module._load_static_sensing_config(h5)
        with pytest.raises(PackedHDF5Error, match="absent CIR layout.*empty values"):
            packed_reader_module.validate_packed_chunk(h5, chunk)


def test_pre_v4_sensing_is_rejected_only_when_sensing_is_requested(tmp_path: Path) -> None:
    _source, chunk_path = _write_v4_sensing_frame_set(tmp_path)
    with h5py.File(chunk_path, "r+") as h5:
        metadata = loads_frame_json(h5["frames/sensing_metadata_json"][0])
        metadata["schema_version"] = 2
        h5["frames/sensing_metadata_json"][0] = dumps_frame_json(metadata)

    reader = PackedHDF5Reader(tmp_path)
    try:
        generic_full = reader.load_standard_frame(0)
        assert generic_full.sensing is not None
        assert generic_full.sensing["schema_version"] == 2

        generic = reader.load_projection(
            0,
            FrameReadRequest(components=frozenset({FrameComponent.PROVENANCE})),
        )
        assert generic.frame.frame_index == 0
        assert generic.frame.sensing is None

        with pytest.raises(
            UnsupportedSensingSchemaError,
            match=r"Regenerate this sensing dataset.*generic packed-HDF5 container",
        ):
            reader.load_projection(
                0,
                FrameReadRequest(components=frozenset({FrameComponent.SENSING})),
            )
    finally:
        reader.close()


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


def test_handle_pool_close_retires_active_reads_without_invalidating_them(tmp_path):
    path = tmp_path / "active.h5"
    with h5py.File(path, "w") as h5:
        h5["value"] = [7]
    pool = HDF5HandleLRU(max_open=2)
    with pool.lease(path) as handle:
        pool.close()
        pool.close()  # A repeated close must not invalidate the surviving lease.
        assert int(handle["value"][0]) == 7
        assert pool.open_count == 1
        with pytest.raises(RuntimeError, match="pool is closed"):
            with pool.lease(path):
                pytest.fail("A retired pool accepted a new read")
    assert pool.open_count == 0
    assert not handle.id.valid


def test_handle_pool_close_wakes_capacity_waiter_and_releases_idle_handles(tmp_path):
    paths = [tmp_path / f"close_{i}.h5" for i in range(3)]
    for path in paths:
        with h5py.File(path, "w") as h5:
            h5["value"] = [3]
    pool = HDF5HandleLRU(max_open=1)
    waiting = Event()
    done = Event()
    errors = []

    def wait_for_capacity():
        waiting.set()
        try:
            with pool.lease(paths[1]):
                errors.append("unexpected lease")
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            done.set()

    worker = Thread(target=wait_for_capacity)
    try:
        with pool.lease(paths[0]) as handle:
            worker.start()
            assert waiting.wait(1)
            assert not done.wait(0.05)
            pool.close()
            assert done.wait(1)
            assert int(handle["value"][0]) == 3
    finally:
        worker.join(timeout=1)
        pool.close()
    assert not worker.is_alive()
    assert errors == ["HDF5 handle pool is closed"]
    assert pool.open_count == 0

    pool = HDF5HandleLRU(max_open=2)
    with pool.lease(paths[0]) as idle:
        pass
    with pool.lease(paths[2]) as active:
        pool.close()
        assert not idle.id.valid
        assert active.id.valid
    assert not active.id.valid
