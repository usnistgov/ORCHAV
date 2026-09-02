from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import numpy as np
import pytest

from generator.io.storage.hdf5_frame_output import (
    MAX_FRAMES_PER_CHUNK,
    HDF5FrameOutputStrategy,
)
from shared.frames.contracts import PathMetric
from shared.frames.directory_ownership import (
    WINDOWS_TRANSACTIONAL_PATH_LIMIT,
    FrameDirectoryChangedError,
    FrameDirectoryLockError,
    FrameDirectorySafetyError,
    ManagedFrameDirectoryLock,
    capture_frame_directory,
    destination_lock_path,
    preflight_windows_transaction_paths,
)
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.manifest import (
    FRAMES_MANIFEST_FILENAME,
    load_frame_manifest,
    write_frame_manifest_atomic,
)
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import estimate_packed_frame_bytes
from shared.frames.types import StandardMPCFrame


def _strategy(
    frames_dir: Path,
    *,
    chunk_size: int = 1,
    compression: str | None = None,
) -> HDF5FrameOutputStrategy:
    return HDF5FrameOutputStrategy(
        SimpleNamespace(
            output_mode="file",
            quality="custom",
            get_quality_profile=lambda: {"max_depth": 1},
        ),
        SimpleNamespace(
            root=frames_dir.parent,
            project_root=frames_dir.parent,
            frames_directory=frames_dir.name,
            frames_dir=frames_dir,
            chunk_size=chunk_size,
            compression=compression,
            raytracing={},
            scene={"id": "test_scene", "source": "local"},
        ),
    )


def _manifest_for(strategy: HDF5FrameOutputStrategy):
    return strategy._frame_set_writer._build_manifest(strategy._provenance())


def _frame(frame_idx: int) -> StandardMPCFrame:
    return standard_mpc_frame_from_pair_data(
        frame_index=frame_idx,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
        vertices_by_pair=[np.empty((1, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.empty((1, 0), dtype=np.int32)],
        path_lengths_by_pair=[np.zeros((1,), dtype=np.int32)],
        metrics_by_pair={
            PathMetric.DELAY_NS: [np.asarray([1.0], dtype=np.float32)],
            PathMetric.PATH_LOSS_DB: [np.asarray([2.0], dtype=np.float32)],
            PathMetric.AOA_AZ_DEG: [np.asarray([3.0], dtype=np.float32)],
            PathMetric.AOA_EL_DEG: [np.asarray([4.0], dtype=np.float32)],
            PathMetric.AOD_AZ_DEG: [np.asarray([5.0], dtype=np.float32)],
            PathMetric.AOD_EL_DEG: [np.asarray([6.0], dtype=np.float32)],
        },
        provenance={"provider": "test", "frame_idx": frame_idx},
    )


def _raw(frame_idx: int) -> dict:
    return {"frame_idx": frame_idx}


def _write_previous_frame_set(frames_dir: Path, *, frame_idx: int = 0) -> Path:
    previous = _strategy(frames_dir)
    previous.save_standard_frame(_frame(frame_idx))
    previous.finalize()
    return frames_dir / f"mpc_frames_{frame_idx:05d}-{frame_idx:05d}.h5"


def test_manifest_provenance_includes_resolved_material_properties(tmp_path):
    strategy = _strategy(tmp_path / "frames")
    strategy.material_properties = {
        "schema_version": 1,
        "source": "sionna.rt.Scene.radio_materials",
        "properties": {
            "itu_concrete": {
                "relative_permittivity": 5.2,
                "conductivity": 0.03,
            }
        },
    }

    manifest = _manifest_for(strategy)

    assert manifest.provenance["material_properties"] == strategy.material_properties
    assert str(UUID(manifest.generation_id)) == manifest.generation_id
    assert manifest.frame_set_id


def test_manifest_git_sha_comes_from_loaded_generator_tree(tmp_path, mocker):
    resolve_identity = mocker.patch(
        "generator.io.storage.hdf5_frame_output.loaded_source_identity",
        return_value=SimpleNamespace(
            source_root=Path(r"C:\private\checkout"),
            version="0.1.0",
            git_sha="actual-generator-sha",
        ),
    )

    strategy = _strategy(tmp_path / "frames")
    manifest = _manifest_for(strategy)

    resolve_identity.assert_called_once_with("generator")
    assert manifest.provenance["git_sha"] == "actual-generator-sha"
    assert "source_root" not in manifest.provenance
    assert r"C:\private\checkout" not in str(manifest.provenance)


def test_equal_empty_layouts_receive_distinct_frame_set_ids(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_strategy = _strategy(first_root / "frames")
    second_strategy = _strategy(second_root / "frames")
    first_manifest = _manifest_for(first_strategy)
    second_manifest = _manifest_for(second_strategy)

    assert first_manifest.chunks == second_manifest.chunks == ()
    assert first_manifest.generation_id != second_manifest.generation_id
    assert first_manifest.frame_set_id != second_manifest.frame_set_id


def test_proxy_like_provenance_is_not_treated_as_a_numpy_scalar(tmp_path):
    simulation_config = MagicMock()
    simulation_config.output_mode = MagicMock(name="output_mode_proxy")
    simulation_config.quality = "custom"
    simulation_config.get_quality_profile.return_value = {}
    strategy = HDF5FrameOutputStrategy(
        simulation_config,
        SimpleNamespace(
            root=tmp_path,
            project_root=tmp_path,
            frames_directory="frames",
            frames_dir=tmp_path / "frames",
            chunk_size=1,
            compression=None,
            raytracing={},
            scene={"id": "test_scene", "source": "local"},
        ),
    )

    manifest = _manifest_for(strategy)

    assert manifest.provenance["output_mode"] == str(simulation_config.output_mode)


def test_only_authoritative_manifest_is_published_at_successful_finalization(
    tmp_path,
    mocker,
):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir)
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        return_value=_frame(0),
    )

    strategy.save_frame_data(0, {})
    assert strategy.published_manifest is None
    assert strategy._frame_set_writer.staging_directory is not None
    assert not (strategy._frame_set_writer.staging_directory / FRAMES_MANIFEST_FILENAME).exists()
    assert not frames_dir.exists()

    strategy.finalize()

    manifest = load_frame_manifest(frames_dir)
    assert strategy.generation_id == manifest.generation_id
    assert strategy.published_manifest == manifest
    assert manifest.frame_ids == (0,)
    assert manifest.compression["filter"] == "none"
    assert manifest.segmentation["effective_frame_limit"] == 1
    assert sorted(path.name for path in frames_dir.iterdir()) == [
        FRAMES_MANIFEST_FILENAME,
        "mpc_frames_00000-00000.h5",
    ]
    assert not (frames_dir / "frames_index.json").exists()
    assert not (frames_dir / "run_manifest.json").exists()


def test_raw_normalization_sidecar_runs_inside_locked_staging(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir)

    def normalize_with_diagnostic(_data, frame_idx, **_kwargs):
        assert strategy._frame_set_writer._directory_lock is not None
        assert strategy._frame_set_writer._directory_lock.acquired
        assert strategy._frame_set_writer.staging_directory is not None
        assert strategy._frame_set_writer.staging_directory.is_dir()
        (strategy._frame_set_writer.staging_directory / "path_filter_diagnostic.png").write_bytes(
            b"generated during normalization"
        )
        return _frame(frame_idx)

    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=normalize_with_diagnostic,
    )

    strategy.save_frame_data(0, {})
    strategy.finalize()

    assert load_frame_manifest(frames_dir).frame_ids == (0,)
    assert (frames_dir / "path_filter_diagnostic.png").read_bytes() == (
        b"generated during normalization"
    )
    assert not destination_lock_path(frames_dir).exists()


def test_raw_normalization_interrupt_cleans_owned_staging_and_lock(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir)
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=KeyboardInterrupt("interrupt normalization"),
    )

    with pytest.raises(KeyboardInterrupt, match="interrupt normalization"):
        strategy.save_frame_data(0, {})

    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not strategy._frame_set_writer._staging_owned
    assert not destination_lock_path(frames_dir).exists()
    assert strategy._frame_set_writer.state == "aborted"


def test_standard_frame_interrupt_cleans_owned_staging_and_lock(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir)
    mocker.patch(
        "shared.frames.packed_hdf5_writer.PackedMPCChunkWriter.prepare",
        side_effect=SystemExit("interrupt standard append"),
    )

    with pytest.raises(SystemExit, match="interrupt standard append"):
        strategy.save_standard_frame(_frame(0))

    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not strategy._frame_set_writer._staging_owned
    assert not destination_lock_path(frames_dir).exists()
    assert strategy._frame_set_writer.state == "aborted"


def test_begin_interrupt_after_staging_creation_releases_visible_lock(
    tmp_path,
    mocker,
):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir)
    create_staging = strategy._frame_set_writer._create_owned_staging_directory

    def interrupt_after_staging_creation() -> None:
        create_staging()
        raise KeyboardInterrupt("interrupt after staging creation")

    mocker.patch.object(
        strategy._frame_set_writer,
        "_create_owned_staging_directory",
        side_effect=interrupt_after_staging_creation,
    )

    with pytest.raises(KeyboardInterrupt, match="after staging creation"):
        strategy.begin()

    assert strategy._frame_set_writer.state == "aborted"
    assert strategy._frame_set_writer._directory_lock is None
    assert not destination_lock_path(frames_dir).exists()
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not strategy._frame_set_writer._staging_owned


def test_standard_frame_entry_point_bypasses_raw_generator_normalization(
    tmp_path,
    mocker,
):
    frames_dir = tmp_path / "frames"
    normalize = mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
    )
    strategy = _strategy(frames_dir, chunk_size=10)

    strategy.save_standard_frame(_frame(7))
    strategy.finalize()

    normalize.assert_not_called()
    manifest = load_frame_manifest(frames_dir)
    assert manifest.frame_ids == (7,)
    assert manifest.chunks[0].file == "mpc_frames_00007-00007.h5"


def test_explicit_create_new_writer_targets_separate_absent_frame_set(tmp_path):
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    analysis_root = tmp_path / "analysis-run"
    analysis_root.mkdir()
    analysis_frames = analysis_root / "frames"
    writer = FrameSetWriter.create_new(
        analysis_frames,
        chunk_size=1,
        compression=None,
    )
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=tmp_path,
        # The injected create-new writer owns publication, so this reader
        # selection does not choose the destination.
        frames_directory="imported_frames",
        frames_dir=scenario_root / "imported_frames",
        chunk_size=10,
        compression="lzf",
        raytracing={},
        scene={"id": "test_scene", "source": "local"},
    )
    strategy = HDF5FrameOutputStrategy(
        SimpleNamespace(
            output_mode="file",
            quality="custom",
            get_quality_profile=lambda: {"max_depth": 1},
        ),
        configuration,
        frame_set_writer=writer,
    )

    strategy.save_standard_frame(_frame(3))
    strategy.finalize()

    assert load_frame_manifest(analysis_frames).frame_ids == (3,)
    assert not (scenario_root / "imported_frames").exists()


def test_frame_limit_is_bounded_and_rotates_incrementally(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(
        frames_dir,
        chunk_size=MAX_FRAMES_PER_CHUNK + 50,
    )
    assert strategy.chunk_size == MAX_FRAMES_PER_CHUNK

    mocker.patch(
        "generator.io.storage.hdf5_frame_output.MAX_FRAMES_PER_CHUNK",
        2,
    )
    strategy = _strategy(frames_dir, chunk_size=10)
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=lambda _data, frame_idx, **_kwargs: _frame(frame_idx),
    )

    for frame_idx in range(3):
        strategy.save_frame_data(frame_idx, _raw(frame_idx))

    assert [chunk.frame_ids for chunk in strategy.generated_chunks] == [(0, 1)]
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_ids == (2,)
    strategy.finalize()
    assert [chunk.frame_ids for chunk in load_frame_manifest(frames_dir).chunks] == [
        (0, 1),
        (2,),
    ]


def test_byte_limit_rotates_before_the_next_frame_is_appended(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    first = _frame(0)
    second = _frame(1)
    frame_bytes = estimate_packed_frame_bytes(first)
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.MAX_UNCOMPRESSED_BYTES_PER_CHUNK",
        frame_bytes * 2 - 1,
    )
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        side_effect=[first, second],
    )
    strategy = _strategy(frames_dir, chunk_size=10)

    strategy.save_frame_data(0, _raw(0))
    strategy.save_frame_data(1, _raw(1))

    assert [chunk.frame_ids for chunk in strategy.generated_chunks] == [(0,)]
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_ids == (1,)
    strategy.finalize()
    manifest = load_frame_manifest(frames_dir)
    assert [chunk.frame_ids for chunk in manifest.chunks] == [(0,), (1,)]
    assert manifest.segmentation["uncompressed_byte_limit"] == frame_bytes * 2 - 1


@pytest.mark.parametrize(
    "boundary_kind",
    ["topology", "fixed_sensing", "cir", "sensing_config"],
)
def test_physical_layout_boundary_rotates_and_retries_cleanly(
    tmp_path,
    mocker,
    boundary_kind,
):
    first = _frame(0)
    second = _frame(1)
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
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir, chunk_size=10)
    strategy.save_frame_data(0, _raw(0))
    strategy.save_frame_data(1, _raw(1))

    assert [chunk.frame_ids for chunk in strategy.generated_chunks] == [(0,)]
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.frame_ids == (1,)
    strategy.finalize()
    manifest = load_frame_manifest(frames_dir)
    assert [chunk.frame_ids for chunk in manifest.chunks] == [(0,), (1,)]
    assert manifest.segmentation["boundary_rotations"] == 1


def test_manifest_failure_aborts_staging_and_keeps_live_set_byte_exact(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        return_value=_frame(0),
    )
    strategy = _strategy(frames_dir)
    strategy.save_frame_data(0, _raw(0))
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


def test_promotion_failure_rolls_back_previous_set_byte_exact(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        return_value=_frame(0),
    )
    strategy = _strategy(frames_dir)
    strategy.save_frame_data(0, _raw(0))
    real_replace = os.replace

    def reject_staged_promotion(source, destination):
        if (
            Path(source) == strategy._frame_set_writer.staging_directory
            and Path(destination) == frames_dir
        ):
            raise PermissionError("injected promotion failure")
        return real_replace(source, destination)

    mocker.patch(
        "shared.frames.frame_set_writer.os.replace",
        side_effect=reject_staged_promotion,
    )

    with pytest.raises(PermissionError, match="injected promotion failure"):
        strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer._backup_dir is not None
    assert not strategy._frame_set_writer._backup_dir.exists()


def test_interrupted_staged_promotion_restores_previous_set(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(1))
    real_replace = os.replace

    def interrupt_staged_promotion(source, destination):
        if (
            Path(source) == strategy._frame_set_writer.staging_directory
            and Path(destination) == frames_dir
        ):
            raise KeyboardInterrupt("injected interruption")
        return real_replace(source, destination)

    mocker.patch(
        "shared.frames.frame_set_writer.os.replace",
        side_effect=interrupt_staged_promotion,
    )

    with pytest.raises(KeyboardInterrupt, match="injected interruption"):
        strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer._backup_dir is not None
    assert not strategy._frame_set_writer._backup_dir.exists()
    assert not destination_lock_path(frames_dir).exists()


def test_interruption_delivered_after_rename_still_restores_previous_set(
    tmp_path,
    mocker,
):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(1))
    real_replace = os.replace

    def interrupt_after_staged_promotion(source, destination):
        result = real_replace(source, destination)
        if (
            Path(source) == strategy._frame_set_writer.staging_directory
            and Path(destination) == frames_dir
        ):
            raise KeyboardInterrupt("interrupted after atomic rename")
        return result

    mocker.patch(
        "shared.frames.frame_set_writer.os.replace",
        side_effect=interrupt_after_staged_promotion,
    )

    with pytest.raises(KeyboardInterrupt, match="after atomic rename"):
        strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer._backup_dir is not None
    assert not strategy._frame_set_writer._backup_dir.exists()


def test_interruption_after_first_publication_restores_absent_state(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(0))
    real_replace = os.replace

    def interrupt_after_staged_promotion(source, destination):
        result = real_replace(source, destination)
        if (
            Path(source) == strategy._frame_set_writer.staging_directory
            and Path(destination) == frames_dir
        ):
            raise KeyboardInterrupt("interrupted first publication")
        return result

    mocker.patch(
        "shared.frames.frame_set_writer.os.replace",
        side_effect=interrupt_after_staged_promotion,
    )

    with pytest.raises(KeyboardInterrupt, match="first publication"):
        strategy.finalize()

    assert not frames_dir.exists()
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()


def test_interrupt_during_post_commit_backup_cleanup_keeps_successful_publication(
    tmp_path,
    mocker,
):
    frames_dir = tmp_path / "frames"
    _write_previous_frame_set(frames_dir, frame_idx=0)
    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(1))
    mocker.patch.object(
        strategy._frame_set_writer,
        "_remove_directory",
        side_effect=KeyboardInterrupt("interrupt backup cleanup"),
    )

    result = strategy.finalize()

    assert "Saved 1 frames" in result
    assert strategy._frame_set_writer.state == "finalized"
    assert load_frame_manifest(frames_dir).frame_ids == (1,)
    assert strategy._frame_set_writer._backup_dir is not None
    assert load_frame_manifest(strategy._frame_set_writer._backup_dir).frame_ids == (0,)
    assert not destination_lock_path(frames_dir).exists()


def test_post_commit_lock_release_interruption_keeps_successful_publication(
    tmp_path,
    mocker,
    caplog,
):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(0))
    directory_lock = strategy._frame_set_writer._directory_lock
    assert directory_lock is not None
    real_release = directory_lock.release

    def interrupt_after_release() -> None:
        real_release()
        raise KeyboardInterrupt("interrupt after lock release")

    mocker.patch.object(
        directory_lock,
        "release",
        side_effect=interrupt_after_release,
    )

    result = strategy.finalize()

    assert "Saved 1 frames" in result
    assert strategy._frame_set_writer.state == "finalized"
    assert strategy._frame_set_writer._directory_lock is None
    assert load_frame_manifest(frames_dir).frame_ids == (0,)
    assert not destination_lock_path(frames_dir).exists()
    assert "Could not release the HDF5 publication lock" in caplog.text


def test_abort_discards_active_partial_and_preserves_live_set(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    mocker.patch(
        "generator.io.storage.hdf5_frame_output.standard_mpc_frame_from_raw",
        return_value=_frame(0),
    )
    strategy = _strategy(frames_dir, chunk_size=10)
    strategy.save_frame_data(0, _raw(0))
    assert strategy._frame_set_writer._writer is not None
    assert strategy._frame_set_writer._writer.partial_path.exists()

    strategy.abort()

    assert old_chunk.read_bytes() == old_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()


def test_abort_continues_cleanup_after_discard_interruption(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir, chunk_size=10)
    strategy.save_standard_frame(_frame(0))
    writer = strategy._frame_set_writer._writer
    assert writer is not None
    real_discard = writer.discard

    def interrupt_after_discard() -> None:
        real_discard()
        raise KeyboardInterrupt("interrupt after partial discard")

    mocker.patch.object(writer, "discard", side_effect=interrupt_after_discard)

    strategy.abort()

    assert strategy._frame_set_writer.state == "aborted"
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not strategy._frame_set_writer._staging_owned
    assert not destination_lock_path(frames_dir).exists()


def test_abort_rmtree_interruption_does_not_mask_active_failure(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir, chunk_size=10)
    writer = strategy._frame_set_writer._ensure_writer()
    mocker.patch.object(
        writer,
        "prepare",
        side_effect=ValueError("primary frame failure"),
    )
    mocker.patch.object(
        strategy._frame_set_writer,
        "_remove_directory",
        side_effect=KeyboardInterrupt("interrupt staged cleanup"),
    )

    with pytest.raises(ValueError, match="primary frame failure"):
        strategy.save_standard_frame(_frame(0))

    assert strategy._frame_set_writer.state == "aborted"
    assert strategy._frame_set_writer.staging_directory is not None
    assert strategy._frame_set_writer.staging_directory.exists()
    assert strategy._frame_set_writer._staging_owned
    assert not destination_lock_path(frames_dir).exists()


def test_empty_finalize_leaves_existing_frame_set_unchanged(tmp_path):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    old_manifest_bytes = (frames_dir / FRAMES_MANIFEST_FILENAME).read_bytes()
    strategy = _strategy(frames_dir, chunk_size=10)

    strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert (frames_dir / FRAMES_MANIFEST_FILENAME).read_bytes() == old_manifest_bytes
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()


def test_provenance_interrupt_aborts_staging_and_releases_lock(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    strategy = _strategy(frames_dir, chunk_size=10)
    strategy.save_standard_frame(_frame(0))
    mocker.patch.object(
        strategy,
        "_provenance",
        side_effect=KeyboardInterrupt("interrupt provenance"),
    )

    with pytest.raises(KeyboardInterrupt, match="interrupt provenance"):
        strategy.finalize()

    assert strategy._frame_set_writer.state == "aborted"
    assert not frames_dir.exists()
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not destination_lock_path(frames_dir).exists()


def test_nonempty_unowned_destination_is_refused_before_mutation(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    unrelated = frames_dir / "notes.txt"
    unrelated.write_text("not generated output", encoding="utf-8")

    with pytest.raises(FrameDirectorySafetyError, match="not a valid HDF5 v2 frame set"):
        _strategy(frames_dir)

    assert unrelated.read_text(encoding="utf-8") == "not generated output"
    assert not list(tmp_path.glob(".orchav-*"))


def test_manifest_owned_destination_rejects_advertised_chunk_directory(tmp_path):
    frames_dir = tmp_path / "frames"
    chunk = _write_previous_frame_set(frames_dir)
    chunk.unlink()
    chunk.mkdir()

    with pytest.raises(FrameDirectorySafetyError, match="closed, flat file inventory"):
        _strategy(frames_dir)

    assert chunk.is_dir()
    assert not list(tmp_path.glob(".orchav-*"))


def test_manifest_owned_destination_rejects_statistics_cache_inside_frames(tmp_path):
    frames_dir = tmp_path / "frames"
    _write_previous_frame_set(frames_dir)
    cache = frames_dir / "scenario_stats_cache.npz"
    cache.write_bytes(b"derived from old generation")

    with pytest.raises(FrameDirectorySafetyError, match="closed frame-set inventory"):
        _strategy(frames_dir)

    assert load_frame_manifest(frames_dir).frame_ids == (0,)
    assert cache.read_bytes() == b"derived from old generation"
    assert not list(tmp_path.glob(".orchav-*"))


def test_manifest_owned_destination_rejects_nested_content(tmp_path):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    derived_dir = frames_dir / "derived"
    derived_dir.mkdir()
    cache = derived_dir / "cache.bin"
    cache.write_bytes(b"old")

    with pytest.raises(FrameDirectorySafetyError, match="closed, flat file inventory"):
        _strategy(frames_dir)

    assert old_chunk.read_bytes() == old_bytes
    assert cache.read_bytes() == b"old"
    assert not destination_lock_path(frames_dir).exists()


def test_known_generated_diagnostic_sidecar_remains_managed(tmp_path):
    frames_dir = tmp_path / "frames"
    _write_previous_frame_set(frames_dir)
    diagnostic = frames_dir / "path_filter_diagnostic.png"
    diagnostic.write_bytes(b"generated diagnostic")

    replacement = _strategy(frames_dir)
    replacement.save_standard_frame(_frame(1))
    replacement.finalize()

    assert load_frame_manifest(frames_dir).frame_ids == (1,)
    assert not diagnostic.exists()


def test_valid_managed_replacement_preserves_sibling_file(tmp_path):
    scenario_root = tmp_path / "scenario"
    frames_dir = scenario_root / "frames"
    scenario_root.mkdir()
    sentinel = scenario_root / "keep-me.txt"
    sentinel.write_text("scenario asset", encoding="utf-8")
    _write_previous_frame_set(frames_dir, frame_idx=7)

    replacement = _strategy(frames_dir)
    replacement.save_standard_frame(_frame(8))
    replacement.finalize()

    assert sentinel.read_text(encoding="utf-8") == "scenario asset"
    assert load_frame_manifest(frames_dir).frame_ids == (8,)


def test_absent_destination_with_existing_external_parent_is_supported(tmp_path):
    frames_dir = tmp_path / "external" / "nested" / "frames"
    frames_dir.parent.mkdir(parents=True)

    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(0))
    strategy.finalize()

    assert load_frame_manifest(frames_dir).frame_ids == (0,)
    assert not list(frames_dir.parent.glob(".orchav-*"))


def test_existing_empty_destination_is_supported(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(0))
    strategy.finalize()

    assert load_frame_manifest(frames_dir).frame_ids == (0,)


def test_zero_frame_manifest_does_not_prove_destructive_ownership(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    strategy = _strategy(frames_dir)
    empty_manifest = _manifest_for(strategy)
    write_frame_manifest_atomic(frames_dir, empty_manifest)
    sentinel = frames_dir / "must-survive.txt"
    sentinel.write_text("unowned content", encoding="utf-8")

    with pytest.raises(FrameDirectorySafetyError, match="contains no published frames"):
        capture_frame_directory(frames_dir)

    assert sentinel.read_text(encoding="utf-8") == "unowned content"
    assert load_frame_manifest(frames_dir).frame_ids == ()


def test_generator_requires_an_existing_scenario_root(tmp_path):
    frames_dir = tmp_path / "external" / "nested" / "frames"
    assert not frames_dir.parent.exists()

    with pytest.raises(FrameDirectorySafetyError, match="Could not resolve scenario root"):
        _strategy(frames_dir)

    assert not (tmp_path / "external").exists()


@pytest.mark.parametrize(
    "destination_kind",
    ["project", "home", "filesystem"],
)
def test_noncanonical_absolute_destinations_are_refused_before_mutation(
    tmp_path,
    destination_kind,
):
    scenario_root = tmp_path / "scenario"
    project_root = tmp_path / "project"
    scenario_root.mkdir()
    project_root.mkdir()
    if destination_kind == "project":
        frames_dir = project_root
    elif destination_kind == "home":
        frames_dir = Path.home()
    else:
        frames_dir = Path(Path.home().anchor)
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=project_root,
        frames_directory=str(frames_dir),
        frames_dir=frames_dir,
        chunk_size=1,
        compression=None,
        raytracing={},
        scene={"id": "test_scene", "source": "local"},
    )

    with pytest.raises(ValueError, match="fixed at <scenario>/frames"):
        HDF5FrameOutputStrategy(SimpleNamespace(), configuration)

    assert not list(tmp_path.glob(".orchav-*"))


def test_manifest_owned_destination_capture_needs_no_scenario_topology(tmp_path):
    destination = tmp_path / "imported-frames"
    writer = FrameSetWriter.create_new(destination, compression=None)
    writer.append(_frame(0))
    writer.finalize()

    snapshot = capture_frame_directory(destination)

    assert snapshot.destination == destination.absolute()
    assert snapshot.state == "managed"
    snapshot.revalidate()


def test_scenario_root_destination_is_refused_before_mutation(tmp_path):
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    sentinel = scenario_root / "scenario.yaml"
    sentinel.write_text("scene: {}", encoding="utf-8")
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=tmp_path,
        frames_directory=".",
        frames_dir=scenario_root,
        chunk_size=1,
        compression=None,
        raytracing={},
        scene={"id": "test_scene", "source": "local"},
    )

    with pytest.raises(ValueError, match="fixed at <scenario>/frames"):
        HDF5FrameOutputStrategy(SimpleNamespace(), configuration)

    assert sentinel.is_file()
    assert not list(tmp_path.glob(".orchav-*"))


def test_parent_traversal_is_refused_before_mutation(tmp_path):
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    frames_dir = scenario_root / ".." / "external" / "frames"
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=tmp_path,
        frames_directory="../external/frames",
        frames_dir=frames_dir,
        chunk_size=1,
        compression=None,
        raytracing={},
        scene={"id": "test_scene", "source": "local"},
    )

    with pytest.raises(ValueError, match="fixed at <scenario>/frames"):
        HDF5FrameOutputStrategy(SimpleNamespace(), configuration)

    assert not (tmp_path / "external").exists()


def test_exact_destination_symlink_is_refused_before_mutation(tmp_path):
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    frames_dir = scenario_root / "frames"
    try:
        frames_dir.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=tmp_path / "project",
        frames_directory="frames",
        frames_dir=frames_dir,
        chunk_size=1,
        compression=None,
        raytracing={},
        scene={"id": "test_scene", "source": "local"},
    )

    with pytest.raises(FrameDirectorySafetyError, match="symbolic link"):
        HDF5FrameOutputStrategy(SimpleNamespace(), configuration)

    assert not any(real_parent.iterdir())


def test_exact_destination_junction_is_refused_before_mutation(tmp_path, mocker):
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    frames_dir = scenario_root / "frames"
    frames_dir.mkdir()
    mocker.patch(
        "shared.frames.directory_ownership._is_junction",
        side_effect=lambda path: Path(path) == frames_dir,
    )
    configuration = SimpleNamespace(
        root=scenario_root,
        project_root=tmp_path / "project",
        frames_directory="frames",
        frames_dir=frames_dir,
        chunk_size=1,
        compression=None,
        raytracing={},
        scene={"id": "test_scene", "source": "local"},
    )

    with pytest.raises(FrameDirectorySafetyError, match="junction"):
        HDF5FrameOutputStrategy(SimpleNamespace(), configuration)


def test_destination_mount_point_is_refused_before_mutation(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    mocker.patch(
        "shared.frames.directory_ownership._is_mount_point",
        side_effect=lambda path: Path(path) == frames_dir,
    )

    with pytest.raises(FrameDirectorySafetyError, match="mount point"):
        _strategy(frames_dir)

    assert frames_dir.is_dir()
    assert not list(frames_dir.iterdir())
    assert not list(tmp_path.glob(".orchav-*"))


def test_exact_destination_reparse_point_is_refused_before_mutation(
    tmp_path,
    mocker,
):
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    frames_dir = scenario_root / "frames"
    frames_dir.mkdir()
    mocker.patch(
        "shared.frames.directory_ownership._is_reparse_point",
        return_value=True,
    )

    with pytest.raises(FrameDirectorySafetyError, match="Windows reparse point"):
        _strategy(frames_dir)

    assert frames_dir.is_dir()
    assert not any(frames_dir.iterdir())
    assert not list(tmp_path.glob(".orchav-*"))


def test_concurrent_destination_change_is_detected_before_promotion(tmp_path):
    frames_dir = tmp_path / "frames"
    old_chunk = _write_previous_frame_set(frames_dir)
    old_bytes = old_chunk.read_bytes()
    strategy = _strategy(frames_dir)
    strategy.save_standard_frame(_frame(1))
    concurrent_file = frames_dir / "concurrent.txt"
    concurrent_file.write_text("other process", encoding="utf-8")

    with pytest.raises(FrameDirectoryChangedError, match="changed during generation"):
        strategy.finalize()

    assert old_chunk.read_bytes() == old_bytes
    assert concurrent_file.read_text(encoding="utf-8") == "other process"
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not destination_lock_path(frames_dir).exists()


def test_destination_lock_blocks_a_second_writer_and_is_released(tmp_path):
    frames_dir = tmp_path / "frames"
    first = _strategy(frames_dir)
    second = _strategy(frames_dir)
    first.save_standard_frame(_frame(0))

    with pytest.raises(FrameDirectoryLockError, match="already locked"):
        second.save_standard_frame(_frame(0))

    lock_path = destination_lock_path(frames_dir)
    assert lock_path.is_file()
    first.abort()
    assert not lock_path.exists()


def test_begin_holds_destination_lock_before_the_first_frame(tmp_path):
    frames_dir = tmp_path / "frames"
    first = _strategy(frames_dir)
    second = _strategy(frames_dir)

    first.begin()

    assert destination_lock_path(frames_dir).is_file()
    assert first._frame_set_writer.staging_directory is not None
    assert first._frame_set_writer.staging_directory.is_dir()
    with pytest.raises(FrameDirectoryLockError, match="already locked"):
        second.begin()

    first.abort()
    assert not destination_lock_path(frames_dir).exists()
    assert not first._frame_set_writer.staging_directory.exists()


def test_zero_output_finalization_releases_an_acquired_lock(tmp_path):
    strategy = _strategy(tmp_path / "frames")
    strategy._frame_set_writer._begin_directory_transaction()
    lock_path = destination_lock_path(strategy._frame_set_writer.destination)
    assert lock_path.is_file()

    strategy.finalize()

    assert not lock_path.exists()


def test_replaced_lock_is_never_removed_as_if_still_owned(tmp_path):
    frames_dir = tmp_path / "frames"
    lock = ManagedFrameDirectoryLock(frames_dir)
    lock.acquire()
    lock.path.unlink()
    lock.path.write_text('{"owner_token": "foreign"}\n', encoding="utf-8")

    with pytest.raises(FrameDirectoryLockError, match="ownership changed"):
        lock.release()

    assert lock.path.read_text(encoding="utf-8") == '{"owner_token": "foreign"}\n'


def test_failed_lock_fsync_removes_only_its_just_created_lock(tmp_path, mocker):
    frames_dir = tmp_path / "frames"
    lock = ManagedFrameDirectoryLock(frames_dir)
    mocker.patch(
        "shared.frames.directory_ownership.os.fsync",
        side_effect=OSError("injected fsync failure"),
    )

    with pytest.raises(FrameDirectoryLockError, match="injected fsync failure"):
        lock.acquire()

    assert not lock.acquired
    assert not lock.path.exists()
    assert not lock.path.is_symlink()


def test_preflight_includes_final_destination_manifest_and_max_id_chunk(
    tmp_path,
    mocker,
):
    frames_dir = tmp_path / "frames"
    preflight = mocker.patch("shared.frames.frame_set_writer.preflight_windows_transaction_paths")

    _strategy(frames_dir)

    probes = {Path(path) for call in preflight.call_args_list for path in call.args[0]}
    assert frames_dir in probes
    assert frames_dir / FRAMES_MANIFEST_FILENAME in probes
    assert frames_dir / "mpc_frames_2147483647-2147483647.h5" in probes


@pytest.mark.skipif(os.name != "nt", reason="Windows path budget only")
def test_long_scenario_root_preflights_all_transaction_paths(tmp_path):
    target_final_length = WINDOWS_TRANSACTIONAL_PATH_LIMIT - 5
    frames_suffix_length = len(os.sep + "frames")
    name_length = target_final_length - len(str(tmp_path)) - 1 - frames_suffix_length
    if not 1 <= name_length <= 240:
        pytest.skip("pytest temporary path cannot isolate the final-path budget")
    scenario_root = tmp_path / ("f" * name_length)
    scenario_root.mkdir()
    frames_dir = scenario_root / "frames"
    assert len(str(frames_dir)) <= WINDOWS_TRANSACTIONAL_PATH_LIMIT

    with pytest.raises(
        FrameDirectorySafetyError,
        match="safe Windows budget",
    ):
        _strategy(frames_dir)

    assert not frames_dir.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows path budget only")
def test_windows_preflight_counts_non_bmp_utf16_code_units(tmp_path):
    prefix_units = len((str(tmp_path) + os.sep).encode("utf-16-le")) // 2
    emoji_count = ((WINDOWS_TRANSACTIONAL_PATH_LIMIT - prefix_units) // 2) + 1
    if emoji_count <= 0:
        pytest.skip("pytest temporary path already exceeds the Windows budget")
    candidate = tmp_path / ("\U0001f4be" * emoji_count)
    assert len(str(candidate)) <= WINDOWS_TRANSACTIONAL_PATH_LIMIT
    assert len(str(candidate).encode("utf-16-le")) // 2 > WINDOWS_TRANSACTIONAL_PATH_LIMIT

    with pytest.raises(FrameDirectorySafetyError, match="UTF-16 code units"):
        preflight_windows_transaction_paths([candidate])


def test_recognized_orphan_transaction_is_reported_but_never_deleted(tmp_path):
    orphan = tmp_path / ".orchav-s-AAAAAAAAAAAAAAAAAAAAAA"
    orphan.mkdir()
    marker = orphan / "partial-data"
    marker.write_bytes(b"leave untouched")

    strategy = _strategy(tmp_path / "frames")
    strategy.save_standard_frame(_frame(0))
    strategy.finalize()

    assert marker.read_bytes() == b"leave untouched"


def test_exact_preexisting_private_staging_collision_is_never_deleted(tmp_path):
    strategy = _strategy(tmp_path / "frames")
    assert strategy._frame_set_writer.staging_directory is not None
    strategy._frame_set_writer.staging_directory.mkdir()
    marker = strategy._frame_set_writer.staging_directory / "foreign-marker"
    marker.write_bytes(b"not owned by this run")

    with pytest.raises(FileExistsError, match="left untouched"):
        strategy.save_standard_frame(_frame(0))

    assert marker.read_bytes() == b"not owned by this run"
    assert not strategy._frame_set_writer._staging_owned
    assert not destination_lock_path(strategy._frame_set_writer.destination).exists()
    assert strategy._frame_set_writer.state == "aborted"


def test_dangling_private_staging_collision_is_never_deleted(tmp_path):
    strategy = _strategy(tmp_path / "frames")
    assert strategy._frame_set_writer.staging_directory is not None
    missing_target = tmp_path / "missing-private-staging-target"
    try:
        strategy._frame_set_writer.staging_directory.symlink_to(
            missing_target,
            target_is_directory=True,
        )
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    assert strategy._frame_set_writer.staging_directory.is_symlink()
    assert not strategy._frame_set_writer.staging_directory.exists()

    with pytest.raises(FileExistsError, match="left untouched"):
        strategy.save_standard_frame(_frame(0))

    assert strategy._frame_set_writer.staging_directory.is_symlink()
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not strategy._frame_set_writer._staging_owned
    assert not destination_lock_path(strategy._frame_set_writer.destination).exists()


def test_lexical_private_collision_is_detected_without_following_target(
    tmp_path,
    monkeypatch,
):
    strategy = _strategy(tmp_path / "frames")
    assert strategy._frame_set_writer.staging_directory is not None
    real_lstat = Path.lstat

    def expose_unfollowed_entry(path):
        if path == strategy._frame_set_writer.staging_directory:
            return SimpleNamespace()
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", expose_unfollowed_entry)

    with pytest.raises(FileExistsError, match="left untouched"):
        strategy.save_standard_frame(_frame(0))

    assert not strategy._frame_set_writer._staging_owned
    assert strategy._frame_set_writer.state == "aborted"


def test_dangling_private_backup_collision_before_promotion_is_never_deleted(
    tmp_path,
):
    strategy = _strategy(tmp_path / "frames")
    strategy.save_standard_frame(_frame(0))
    assert strategy._frame_set_writer._backup_dir is not None
    missing_target = tmp_path / "missing-private-backup-target"
    try:
        strategy._frame_set_writer._backup_dir.symlink_to(
            missing_target,
            target_is_directory=True,
        )
    except (NotImplementedError, OSError) as exc:
        strategy.abort()
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    assert strategy._frame_set_writer._backup_dir.is_symlink()
    assert not strategy._frame_set_writer._backup_dir.exists()

    with pytest.raises(FileExistsError, match="rollback path already exists"):
        strategy.finalize()

    assert strategy._frame_set_writer._backup_dir.is_symlink()
    assert not strategy._frame_set_writer._backup_dir.exists()
    assert strategy._frame_set_writer.staging_directory is not None
    assert not strategy._frame_set_writer.staging_directory.exists()
    assert not destination_lock_path(strategy._frame_set_writer.destination).exists()


def test_abort_before_begin_never_removes_external_private_path(tmp_path):
    strategy = _strategy(tmp_path / "frames")
    assert strategy._frame_set_writer.staging_directory is not None
    strategy._frame_set_writer.staging_directory.mkdir()
    marker = strategy._frame_set_writer.staging_directory / "external-marker"
    marker.write_bytes(b"appeared after capture")

    strategy.abort()

    assert marker.read_bytes() == b"appeared after capture"
    assert not strategy._frame_set_writer._staging_owned
