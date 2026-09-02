import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from generator.io.frames.builder import process_cached_frame_data
from generator.io.storage.hdf5_frame_output import HDF5FrameOutputStrategy
from shared.extensions.sensing import SENSING_DENSE_HDF5_MIN_SCHEMA_VERSION
from shared.frames.contracts import MPC_HDF5_LAYOUT, MPC_HDF5_SCHEMA_VERSION, PathMetric
from shared.frames.hdf5 import HDF5FormatHandler
from shared.frames.manifest import (
    FRAMES_MANIFEST_FILENAME,
    load_frame_manifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import (
    PackedMPCChunkWriter,
    estimate_packed_frame_bytes,
    write_packed_mpc_frame_chunk,
)
from shared.frames.types import StandardMPCFrame


def create_mock_frame_data(frame_idx):
    return {
        "frame_idx": frame_idx,
        "timestamp": float(frame_idx),
        "tx_list": [],
        "rx_list": [],
        "paths": MagicMock(),
        "target_objects": [],
        "target_managers": [],
        "material_mapping": None,
    }


def _mock_standard_frame(frame_idx, num_mpcs=10, max_bounces=3) -> StandardMPCFrame:
    """Return a minimal StandardMPCFrame accepted by the packed-v2 writer."""
    path_lengths = np.full(
        (num_mpcs,),
        0 if max_bounces == 0 else 1,
        dtype=np.int32,
    )
    empty_metric = np.zeros(num_mpcs, dtype=np.float32)
    return standard_mpc_frame_from_pair_data(
        frame_index=frame_idx,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 30.0]], dtype=np.float64),
        rx_positions=np.asarray([[50.0, 50.0, 1.5]], dtype=np.float64),
        vertices_by_pair=[np.zeros((num_mpcs, max_bounces, 3), dtype=np.float32)],
        interactions_by_pair=[np.ones((num_mpcs, max_bounces), dtype=np.int32)],
        path_lengths_by_pair=[path_lengths],
        metrics_by_pair={metric: [empty_metric] for metric in PathMetric},
        provenance={"provider": "test", "frame_idx": frame_idx},
    )


def _write_packed_frame_set(frames_dir, frames, *, compression=None):
    """Write one provider-readable packed-v2 test chunk and manifest."""
    frames = list(frames)
    chunk = write_packed_mpc_frame_chunk(
        frames_dir,
        frames,
        generation_id="incremental-bulk-test-generation",
        compression=compression,
    )
    compression_filter = "none" if compression is None else compression
    manifest = manifest_from_chunks(
        generation_id="incremental-bulk-test-generation",
        frame_set_id="incremental-bulk-test-frame-set",
        chunks=[chunk],
        compression={
            "configured": compression,
            "filter": compression_filter,
            "shuffle": compression is not None,
        },
        segmentation={"max_frames": len(frames)},
        provenance={"fixture": "incremental-bulk"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return frames_dir / chunk.file


def _make_pose(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


@pytest.fixture
def mock_config(tmp_path):
    sim_config = MagicMock()

    scenario_config = MagicMock()
    scenario_config.root = tmp_path
    scenario_config.project_root = tmp_path
    scenario_config.frames_directory = "frames"
    scenario_config.frames_dir = tmp_path / "frames"
    scenario_config.chunk_size = 100
    scenario_config.compression = "gzip"

    return sim_config, scenario_config


def test_zero_chunk_size_is_rejected(mock_config):
    """chunk_size must be positive for indexed frame chunks."""
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 0

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        HDF5FrameOutputStrategy(sim_config, scenario_config)


def test_chunked_output(mock_config, mocker):
    """Configured frame bounds finalize private chunks without a frame list."""
    mock_normalize = mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw"
    )

    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 2

    mock_normalize.side_effect = lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx)

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)

    # Frame 0
    strategy.save_frame_data(0, create_mock_frame_data(0))
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_count == 1

    # Frame 1 -> Should flush (size 2 reached)
    strategy.save_frame_data(1, create_mock_frame_data(1))
    assert strategy._frame_set_writer._writer is None
    assert len(strategy.generated_chunks) == 1
    assert strategy._frame_set_writer.staging_directory is not None
    assert (strategy._frame_set_writer.staging_directory / "mpc_frames_00000-00001.h5").exists()
    assert not scenario_config.frames_dir.exists()

    # Frame 2
    strategy.save_frame_data(2, create_mock_frame_data(2))
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_count == 1

    # Frame 3 -> Should flush
    strategy.save_frame_data(3, create_mock_frame_data(3))
    assert strategy._frame_set_writer._writer is None
    assert len(strategy.generated_chunks) == 2
    assert (strategy._frame_set_writer.staging_directory / "mpc_frames_00002-00003.h5").exists()
    assert not scenario_config.frames_dir.exists()

    # Frame 4 (remains in buffer)
    strategy.save_frame_data(4, create_mock_frame_data(4))
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_count == 1

    # Finalize -> flush remainder
    strategy.finalize()
    assert strategy._frame_set_writer._writer is None
    assert len(strategy.generated_chunks) == 3
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert (scenario_config.frames_dir / "mpc_frames_00000-00001.h5").exists()
    assert (scenario_config.frames_dir / "mpc_frames_00002-00003.h5").exists()
    assert (scenario_config.frames_dir / "mpc_frames_00004-00004.h5").exists()

    manifest = load_frame_manifest(scenario_config.frames_dir)
    assert manifest.frame_ids == (0, 1, 2, 3, 4)
    assert [chunk.count for chunk in manifest.chunks] == [2, 2, 1]
    assert manifest.segmentation["effective_frame_limit"] == 2


def test_configured_frame_limit_is_hard_capped_at_one_hundred(mock_config):
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1000

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)

    assert strategy.configured_chunk_size == 1000
    assert strategy.chunk_size == 100


def test_estimated_byte_cap_rotates_before_appending_next_frame(mock_config, mocker):
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 10
    frames = [_mock_standard_frame(index, num_mpcs=2) for index in range(2)]
    frame_bytes = estimate_packed_frame_bytes(frames[0])
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.MAX_UNCOMPRESSED_BYTES_PER_CHUNK",
        frame_bytes * 2 - 1,
    )
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=frames,
    )

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))
    assert strategy._frame_set_writer._writer is not None
    strategy.save_frame_data(1, create_mock_frame_data(1))

    assert [chunk.frame_ids for chunk in strategy.generated_chunks] == [(0,)]
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_ids == (1,)

    strategy.finalize()
    manifest = load_frame_manifest(scenario_config.frames_dir)
    assert [chunk.frame_ids for chunk in manifest.chunks] == [(0,), (1,)]
    assert manifest.segmentation["uncompressed_byte_limit"] == frame_bytes * 2 - 1


@pytest.mark.parametrize(
    "boundary_kind",
    ["topology", "fixed_sensing", "cir", "sensing_config"],
)
def test_layout_changes_rotate_and_retry_in_a_clean_chunk(
    mock_config,
    mocker,
    boundary_kind,
):
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 10
    first = _mock_standard_frame(0, num_mpcs=1)
    second = _mock_standard_frame(1, num_mpcs=1)

    if boundary_kind == "topology":
        first = replace(first, rx_names=("rx-a",))
        second = replace(second, rx_names=("rx-b",))
    elif boundary_kind == "fixed_sensing":
        first = replace(first, sensing={"range_profile": np.ones((3,), dtype=np.float32)})
        second = replace(second, sensing={"range_profile": np.ones((4,), dtype=np.float32)})
    elif boundary_kind == "cir":
        first = replace(first, sensing={"cir": np.ones((2, 2), dtype=np.complex64)})
        second = replace(second, sensing={"cir": np.ones((2, 2), dtype=np.complex128)})
    else:
        first = replace(first, sensing={"config": {"bandwidth_hz": 1.0}})
        second = replace(second, sensing={"config": {"bandwidth_hz": 2.0}})

    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=[first, second],
    )
    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)

    strategy.save_frame_data(0, create_mock_frame_data(0))
    strategy.save_frame_data(1, create_mock_frame_data(1))

    assert [chunk.frame_ids for chunk in strategy.generated_chunks] == [(0,)]
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_ids == (1,)
    strategy.finalize()

    manifest = load_frame_manifest(scenario_config.frames_dir)
    assert [chunk.frame_ids for chunk in manifest.chunks] == [(0,), (1,)]
    assert manifest.segmentation["boundary_rotations"] == 1


def test_hdf5_compression_passthrough(mock_config, mocker):
    """Ensure configured compression is applied and advertised."""
    mock_normalize = mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw"
    )

    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1
    scenario_config.compression = "lzf"

    mock_normalize.side_effect = lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx)

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))
    strategy.finalize()

    import h5py

    chunk_file = scenario_config.frames_dir / "mpc_frames_00000-00000.h5"
    with h5py.File(chunk_file, "r") as h5:
        assert h5["paths/delay_ns"].compression == "lzf"
        assert "canonical_points" not in h5
        assert "canonical_path_offsets" not in h5
    manifest = load_frame_manifest(scenario_config.frames_dir)
    assert manifest.compression["filter"] == "lzf"


def test_frame_manifest_records_effective_quality_profile(mock_config, mocker):
    """Manifest should include preset+custom merged RT quality settings."""
    mock_normalize = mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw"
    )
    mock_normalize.side_effect = lambda _data, frame_idx, **_kwargs: _mock_standard_frame(
        frame_idx, num_mpcs=1
    )

    sim_config, scenario_config = mock_config
    sim_config.get_quality_profile.return_value = {
        "max_depth": 3,
        "samples_per_src": 1000000,
        "max_num_paths_per_src": 500000,
        "diffuse_reflection": False,
    }
    sim_config.output_mode = "local"
    sim_config.quality = "low"
    scenario_config.raytracing = {
        "quality": {
            "custom": {
                "diffuse_reflection": True,
                "max_num_paths_per_src": 20000,
            }
        }
    }

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))
    strategy.finalize()

    manifest = load_frame_manifest(scenario_config.frames_dir)
    provenance = manifest.provenance
    assert provenance["quality_profile"]["samples_per_src"] == 1000000
    assert provenance["quality_profile"]["max_num_paths_per_src"] == 20000
    assert provenance["quality_profile"]["diffuse_reflection"] is True
    assert provenance["quality_profile_base"]["max_num_paths_per_src"] == 500000
    assert provenance["quality_profile_base"]["diffuse_reflection"] is False

    # Verify the output file was created and is readable
    import h5py

    chunk_file = scenario_config.frames_dir / "mpc_frames_00000-00000.h5"
    assert chunk_file.exists()
    with h5py.File(chunk_file, "r") as f:
        assert f.attrs["storage_layout"] == MPC_HDF5_LAYOUT
        assert int(f.attrs["schema_version"]) == MPC_HDF5_SCHEMA_VERSION
        assert "bounces/xyz_m" in f


def test_process_cached_frame_data_emits_zero_path_payloads():
    tx = SimpleNamespace(position=(0.0, 0.0, 1.5), orientation=_make_pose(0.0, 0.0, 0.0))
    rx = SimpleNamespace(position=(1.0, 0.0, 1.5), orientation=_make_pose(0.1, 0.2, 0.3))
    target_obj = SimpleNamespace(
        position=_make_pose(2.0, 3.0, 0.9),
        orientation=_make_pose(0.0, 0.0, 0.0),
    )
    target_cfg = SimpleNamespace(
        name="walker",
        initial_position=(2.0, 3.0, 0.9),
        scale=1.0,
        material_type="metal",
        use_ply_position=False,
        mobility=None,
    )
    target_manager = SimpleNamespace(
        target_object=target_obj,
        current_mesh_idx=0,
        meshes=["/tmp/walker_000.ply"],
        relative_mesh_directory="meshes",
        config=target_cfg,
    )

    frame = process_cached_frame_data(
        frame_idx=17,
        tx_list=[tx],
        rx_list=[rx],
        target_objects=[target_obj],
        target_managers=[target_manager],
        sensing_data={"rd_fresh": True},
    )

    assert frame.tx_rx_pairs.shape == (1, 2)
    assert frame.pair_path_offsets.tolist() == [0, 0]
    assert frame.bounce_offsets.tolist() == [0]
    assert frame.bounce_xyz_m.shape == (0, 3)
    assert frame.interactions.shape == (0,)
    assert frame.material_ids.shape == (0,)
    assert frame.targets_metadata[0]["name"] == "walker"
    assert frame.sensing == {"rd_fresh": True}


def test_cached_coherent_frame_without_sensing_uses_lightweight_serializer(mock_config, mocker):
    mock_process = mocker.patch("generator.io.frames.builder.process_frame_data")
    mock_cached = mocker.patch("generator.io.frames.builder.process_cached_frame_data")

    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1
    mock_cached.return_value = _mock_standard_frame(frame_idx=0, num_mpcs=0, max_bounces=0)

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    frame = create_mock_frame_data(0)
    frame["_coherent_cached_frame"] = True

    strategy.save_frame_data(0, frame)

    mock_cached.assert_called_once()
    mock_process.assert_not_called()


def test_cached_coherent_frame_with_sensing_keeps_full_serializer(mock_config, mocker):
    mock_process = mocker.patch("generator.io.frames.builder.process_frame_data")
    mock_cached = mocker.patch("generator.io.frames.builder.process_cached_frame_data")

    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1
    mock_process.return_value = _mock_standard_frame(frame_idx=0, num_mpcs=1, max_bounces=1)

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    frame = create_mock_frame_data(0)
    frame["_coherent_cached_frame"] = True
    frame["sensing"] = {"rd_fresh": True}

    strategy.save_frame_data(0, frame)

    mock_process.assert_called_once()
    mock_cached.assert_not_called()


def test_chunk_size_one(mock_config, mocker):
    """Test chunk_size=1: one file per frame."""
    mock_normalize = mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw"
    )

    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1

    mock_normalize.side_effect = lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx)

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)

    assert strategy.chunk_size == 1

    # Save 3 frames - each should immediately flush
    for i in range(3):
        strategy.save_frame_data(i, create_mock_frame_data(i))
        assert strategy._frame_set_writer._writer is None
        assert len(strategy.generated_chunks) == i + 1
        # Individual files exist only in this run's private staging directory.
        assert strategy._frame_set_writer.staging_directory is not None
        expected_file = (
            strategy._frame_set_writer.staging_directory / f"mpc_frames_{i:05d}-{i:05d}.h5"
        )
        assert expected_file.exists(), f"Expected {expected_file} to exist"
        assert not scenario_config.frames_dir.exists()

    # Finalize
    strategy.finalize()

    manifest = load_frame_manifest(scenario_config.frames_dir)
    assert manifest.frame_ids == (0, 1, 2)
    assert [chunk.count for chunk in manifest.chunks] == [1, 1, 1]


def test_finalize_replaces_the_complete_frame_set(mock_config, mocker):
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx),
    )
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1
    stale_chunk = _write_packed_frame_set(
        scenario_config.frames_dir,
        [_mock_standard_frame(99)],
    )
    stale_bytes = stale_chunk.read_bytes()

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))

    assert stale_chunk.read_bytes() == stale_bytes
    assert not (scenario_config.frames_dir / "mpc_frames_00000-00000.h5").exists()

    strategy.finalize()

    assert not stale_chunk.exists()
    assert (scenario_config.frames_dir / "mpc_frames_00000-00000.h5").is_file()
    assert (scenario_config.frames_dir / FRAMES_MANIFEST_FILENAME).is_file()
    assert not (scenario_config.frames_dir / "frames_index.json").exists()
    assert not (scenario_config.frames_dir / "run_manifest.json").exists()


def test_locked_live_directory_keeps_previous_frames_byte_exact(mock_config, mocker):
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx),
    )
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1
    old_chunk = _write_packed_frame_set(
        scenario_config.frames_dir,
        [_mock_standard_frame(0)],
    )
    old_bytes = old_chunk.read_bytes()

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))
    real_replace = os.replace

    def reject_live_directory_rename(source, destination):
        if Path(source) == scenario_config.frames_dir:
            raise PermissionError("the visualizer has an HDF5 file open")
        return real_replace(source, destination)

    mocker.patch(
        "shared.frames.frame_set_writer.os.replace",
        side_effect=reject_live_directory_rename,
    )

    with pytest.raises(PermissionError, match="visualizer"):
        strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert old_chunk.stat().st_size == len(old_bytes)
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer._backup_dir is not None
    assert not strategy._frame_set_writer._backup_dir.exists()


def test_failed_staged_promotion_rolls_previous_frames_back(mock_config, mocker):
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx),
    )
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1
    old_chunk = _write_packed_frame_set(
        scenario_config.frames_dir,
        [_mock_standard_frame(0)],
    )
    old_bytes = old_chunk.read_bytes()

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))
    assert strategy._frame_set_writer.staging_directory is not None
    real_replace = os.replace

    def reject_staged_directory_rename(source, destination):
        if (
            Path(source) == strategy._frame_set_writer.staging_directory
            and Path(destination) == scenario_config.frames_dir
        ):
            raise PermissionError("could not expose staged frame set")
        return real_replace(source, destination)

    mocker.patch(
        "shared.frames.frame_set_writer.os.replace",
        side_effect=reject_staged_directory_rename,
    )

    with pytest.raises(PermissionError, match="staged frame set"):
        strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer._backup_dir is not None
    assert not strategy._frame_set_writer._backup_dir.exists()


def test_failed_authoritative_manifest_write_discards_staging_and_keeps_live_bytes(
    mock_config,
    mocker,
):
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx),
    )
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 1
    old_chunk = _write_packed_frame_set(
        scenario_config.frames_dir,
        [_mock_standard_frame(0)],
    )
    old_bytes = old_chunk.read_bytes()

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))
    mocker.patch(
        "shared.frames.frame_set_writer.write_frame_manifest_atomic",
        side_effect=OSError("injected manifest failure"),
    )

    with pytest.raises(OSError, match="injected manifest failure"):
        strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer.state == "aborted"


def test_zero_frame_finalize_does_not_replace_previous_frames(mock_config):
    sim_config, scenario_config = mock_config
    old_chunk = _write_packed_frame_set(
        scenario_config.frames_dir,
        [_mock_standard_frame(0)],
    )
    old_bytes = old_chunk.read_bytes()

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()


def test_abort_discards_active_partial_and_all_staging(mock_config, mocker):
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=lambda _data, frame_idx, **_kwargs: _mock_standard_frame(frame_idx),
    )
    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 10
    old_chunk = _write_packed_frame_set(
        scenario_config.frames_dir,
        [_mock_standard_frame(0)],
    )
    old_bytes = old_chunk.read_bytes()

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)
    strategy.save_frame_data(0, create_mock_frame_data(0))
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.partial_path.exists()

    strategy.abort()

    assert old_chunk.read_bytes() == old_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer.state == "aborted"


def test_variable_mpc_counts(mock_config, mocker):
    """Frames with different MPC counts should use ragged path offsets."""
    mock_normalize = mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw"
    )

    sim_config, scenario_config = mock_config
    scenario_config.chunk_size = 4

    # Each frame has a different number of MPCs
    mpc_counts = [8, 12, 5, 10]
    mock_normalize.side_effect = lambda _data, frame_idx, **_kwargs: _mock_standard_frame(
        frame_idx, num_mpcs=mpc_counts[frame_idx]
    )

    strategy = HDF5FrameOutputStrategy(sim_config, scenario_config)

    for i in range(4):
        strategy.save_frame_data(i, create_mock_frame_data(i))

    strategy.finalize()

    import h5py

    out_file = scenario_config.frames_dir / "mpc_frames_00000-00003.h5"
    assert out_file.exists()
    with h5py.File(out_file, "r") as f:
        assert f.attrs["storage_layout"] == MPC_HDF5_LAYOUT
        assert f["bounces/xyz_m"].shape == (sum(mpc_counts), 3)

        offsets = f["index/frame_pair_path_offsets"][:, [0, 1]]
        np.testing.assert_array_equal(
            offsets,
            np.array([[0, 8], [8, 20], [20, 25], [25, 35]]),
        )

        # Frame 2 has five paths and no padded path rows.
        start, end = offsets[2]
        bounce_offsets = f["paths/bounce_offsets"][start : end + 1]
        assert bounce_offsets.shape == (6,)
        assert np.all(np.diff(bounce_offsets) == 1)


def test_packed_writer_supports_los_only_pairs(tmp_path):
    """LOS-only frames should write even when the packed bounce axis is empty."""
    frame = _mock_standard_frame(frame_idx=0, num_mpcs=6, max_bounces=0)
    frame = replace(
        frame,
        delays_ns=np.linspace(1.0, 6.0, 6, dtype=np.float32),
        path_loss_db=np.linspace(-10.0, -15.0, 6, dtype=np.float32),
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_packed_frame_set(frames_dir, [frame], compression="lzf")

    handler = HDF5FormatHandler(tmp_path, frames_subdir="frames")
    loaded = handler.load_frame(0)

    assert loaded.num_paths == 6
    assert loaded.bounce_xyz_m.shape == (0, 3)
    assert loaded.interactions.shape == (0,)
    np.testing.assert_array_equal(loaded.bounce_offsets, np.zeros(7, dtype=np.int64))
    np.testing.assert_allclose(
        loaded.delays_ns,
        np.linspace(1.0, 6.0, 6, dtype=np.float32),
        atol=1e-6,
    )


def test_packed_writer_exposes_frame_ids_and_writes_manifest(tmp_path):
    frames_dir = tmp_path / "frames"
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id="incremental-bulk-test-generation",
        compression=None,
        partial_name=".incremental-bulk-test.h5.partial",
    )
    writer.append(_mock_standard_frame(4))
    writer.append(_mock_standard_frame(5))

    assert writer.frame_ids == (4, 5)
    assert writer.frame_count == 2

    chunk = writer.finalize_to_range_name()
    manifest = manifest_from_chunks(
        generation_id="incremental-bulk-test-generation",
        frame_set_id="incremental-bulk-test-frame-set",
        chunks=[chunk],
        compression={"configured": None, "filter": "none", "shuffle": False},
        segmentation={"max_frames": 2},
        provenance={"fixture": "incremental-bulk"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)

    loaded_manifest = load_frame_manifest(frames_dir)
    assert loaded_manifest.frame_ids == (4, 5)
    assert len(loaded_manifest.chunks) == 1
    assert loaded_manifest.chunks[0].file == "mpc_frames_00004-00005.h5"
    assert loaded_manifest.chunks[0].frame_ids == (4, 5)


def test_packed_round_trip_full_fidelity(tmp_path):
    """Verify all StandardMPCFrame fields survive the packed-v2 round-trip."""
    rng = np.random.RandomState(42)
    num_frames = 3
    frames = []
    for i in range(num_frames):
        nm = 5 + i * 2  # variable MPC counts: 5, 7, 9
        nb = 3
        # Keep the two named material columns physical while retaining a
        # variable third-bounce axis to exercise packed ragged reconstruction.
        path_lengths = rng.randint(2, nb + 1, nm, dtype=np.int32)
        vertices = np.full((nm, nb, 3), np.nan, dtype=np.float32)
        interactions = np.full((nm, nb), -1, dtype=np.int32)
        for path_index, path_length in enumerate(path_lengths):
            vertices[path_index, :path_length] = rng.randn(path_length, 3)
            interactions[path_index, :path_length] = rng.randint(
                1,
                4,
                path_length,
                dtype=np.int32,
            )
        frames.append(
            standard_mpc_frame_from_pair_data(
                frame_index=i,
                tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
                tx_positions=rng.randn(1, 3) + i * 10,
                rx_positions=rng.randn(1, 3) + i * 20,
                tx_orientations=rng.randn(1, 3),
                rx_orientations=rng.randn(1, 3),
                vertices_by_pair=[vertices],
                interactions_by_pair=[interactions],
                path_lengths_by_pair=[path_lengths],
                material_names_by_pair=[
                    np.asarray([["concrete", "glass", ""] for _ in range(nm)], dtype=object)
                ],
                material_itu_types_by_pair=[
                    np.asarray(
                        [["itu_concrete", "itu_glass", ""] for _ in range(nm)],
                        dtype=object,
                    )
                ],
                metrics_by_pair={
                    PathMetric.DELAY_NS: [rng.rand(nm).astype(np.float32) * 100],
                    PathMetric.PATH_LOSS_DB: [rng.rand(nm).astype(np.float32) * -80],
                    PathMetric.AOA_AZ_DEG: [rng.rand(nm).astype(np.float32) * 360],
                    PathMetric.AOA_EL_DEG: [rng.rand(nm).astype(np.float32) * 180 - 90],
                    PathMetric.AOD_AZ_DEG: [rng.rand(nm).astype(np.float32) * 360],
                    PathMetric.AOD_EL_DEG: [rng.rand(nm).astype(np.float32) * 180 - 90],
                },
                target_positions_m=rng.randn(2, 3),
                targets_metadata=[
                    {"name": f"target_{j}", "velocity": rng.randn(3).tolist()} for j in range(2)
                ],
                sensing={
                    "config": {"bandwidth": 100e6, "num_samples": 256},
                    "range_profile": rng.randn(256).astype(np.float32),
                    "cir": (rng.randn(1, 1, 1, 64) + 1j * rng.randn(1, 1, 1, 64)).astype(
                        np.complex64
                    ),
                },
                provenance={
                    "provider": "test",
                    "frame_idx": i,
                    "source_rt_frame_idx": max(0, i - 1),
                },
            )
        )

    # Write
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_packed_frame_set(frames_dir, frames, compression=None)

    # Read back
    handler = HDF5FormatHandler(tmp_path, frames_subdir="frames")
    assert handler.list_frames() == [0, 1, 2]

    for i, orig in enumerate(frames):
        loaded = handler.load_frame(i)
        for field in (
            "tx_positions",
            "rx_positions",
            "tx_orientations",
            "rx_orientations",
            "tx_rx_pairs",
            "pair_path_offsets",
            "bounce_offsets",
            "bounce_xyz_m",
            "interactions",
            "material_ids",
            "metric_valid_bits",
            "target_positions_m",
            "delays_ns",
            "path_loss_db",
            "aoa_az_deg",
            "aoa_el_deg",
            "aod_az_deg",
            "aod_el_deg",
        ):
            np.testing.assert_allclose(
                getattr(loaded, field),
                getattr(orig, field),
                atol=1e-6,
                equal_nan=True,
                err_msg=f"Canonical field {field} mismatch at frame {i}",
            )
        assert loaded.material_names == orig.material_names
        assert loaded.material_itu_types == orig.material_itu_types
        assert loaded.provenance == orig.provenance
        assert loaded.num_targets == 2
        assert len(loaded.targets_metadata) == 2
        assert loaded.targets_metadata[0]["name"] == "target_0"
        assert loaded.sensing is not None
        assert orig.sensing is not None
        assert loaded.sensing["config"]["bandwidth"] == 100e6
        np.testing.assert_allclose(
            loaded.sensing["range_profile"],
            orig.sensing["range_profile"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            loaded.sensing["cir"],
            orig.sensing["cir"],
            atol=1e-6,
        )


def test_packed_schema_keeps_sensing_json_metadata_only(tmp_path):
    import h5py

    frame = _mock_standard_frame(frame_idx=0, num_mpcs=2, max_bounces=2)
    frame = replace(
        frame,
        sensing={
            "schema_version": SENSING_DENSE_HDF5_MIN_SCHEMA_VERSION,
            "processing_strategy": "snapshot_direct_rd",
            "range_profile": np.linspace(0.0, 1.0, 8, dtype=np.float32),
            "cir": np.ones((1, 1, 1, 8), dtype=np.complex64),
        },
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    chunk_path = _write_packed_frame_set(frames_dir, [frame], compression=None)

    with h5py.File(chunk_path, "r") as h5:
        raw = h5["frames/sensing_metadata_json"][0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        assert payload["schema_version"] == SENSING_DENSE_HDF5_MIN_SCHEMA_VERSION
        assert "range_profile" not in payload
        assert "cir" not in payload
        assert "sensing/fixed/range_profile/data" in h5
        assert "sensing/ragged/cir/values" in h5
