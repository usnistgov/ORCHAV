from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from shared.frames.contracts import PathMetric
from shared.frames.directory_ownership import (
    WINDOWS_TRANSACTIONAL_PATH_LIMIT,
    destination_lock_path,
)
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.manifest import FRAMES_MANIFEST_FILENAME, load_frame_manifest
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.types import StandardMPCFrame


def _frame(frame_index: int) -> StandardMPCFrame:
    """Build one LoS frame for frame-set transaction tests."""

    metric_values = {
        metric: [np.asarray([value], dtype=np.float32)]
        for metric, value in zip(PathMetric, range(1, 7), strict=True)
    }
    return standard_mpc_frame_from_pair_data(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        vertices_by_pair=[np.empty((1, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.empty((1, 0), dtype=np.uint8)],
        path_lengths_by_pair=[np.zeros((1,), dtype=np.int64)],
        metrics_by_pair=metric_values,
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        provenance={
            "provider": "frame-set-writer-test",
            "frame_idx": frame_index,
        },
    )


def _publish_scenario_frame(scenario_root: Path, frame_index: int = 0) -> Path:
    writer = FrameSetWriter.for_scenario(scenario_root, compression=None)
    writer.append(_frame(frame_index))
    manifest = writer.finalize(provenance={"producer": "test"})
    assert manifest is not None
    return scenario_root / "frames"


def test_for_scenario_derives_frames_and_returns_published_manifest(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()

    writer = FrameSetWriter.for_scenario(
        scenario_root,
        chunk_size=2,
        compression=None,
    )
    writer.append(_frame(4))
    manifest = writer.finalize(provenance={"producer": "external-test"})

    assert manifest is not None
    assert manifest.frame_ids == (4,)
    assert manifest.provenance == {"producer": "external-test"}
    assert load_frame_manifest(scenario_root / "frames") == manifest
    assert writer.state == "finalized"
    assert not destination_lock_path(scenario_root / "frames").exists()


def test_create_new_publishes_only_to_an_absent_destination(tmp_path: Path) -> None:
    destination = tmp_path / "imported-frames"

    writer = FrameSetWriter.create_new(destination, compression=None)
    writer.append(_frame(2))
    manifest = writer.finalize()

    assert manifest is not None
    assert manifest.frame_ids == (2,)
    assert load_frame_manifest(destination).frame_ids == (2,)


@pytest.mark.parametrize("existing_kind", ["empty", "unmanaged", "managed"])
def test_create_new_refuses_every_existing_destination(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    destination = tmp_path / "existing"
    if existing_kind == "empty":
        destination.mkdir()
    elif existing_kind == "unmanaged":
        destination.mkdir()
        (destination / "notes.txt").write_text("external", encoding="utf-8")
    else:
        scenario_root = tmp_path / "scenario"
        scenario_root.mkdir()
        destination = _publish_scenario_frame(scenario_root)

    before = sorted(path.name for path in destination.iterdir())
    with pytest.raises(FileExistsError, match="requires an absent destination"):
        FrameSetWriter.create_new(destination)

    assert sorted(path.name for path in destination.iterdir()) == before


def test_create_new_never_moves_a_destination_that_appears_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "imported-frames"
    writer = FrameSetWriter.create_new(destination, compression=None)

    marker = destination / "external.txt"
    real_mkdir = Path.mkdir

    def create_collision_before_claim(path: Path, *args, **kwargs) -> None:
        if path == destination and not destination.exists():
            real_mkdir(path)
            marker.write_text("external", encoding="utf-8")
        real_mkdir(path, *args, **kwargs)

    # The collision occurs inside the exclusive mkdir claim, after snapshot
    # revalidation and immediately before the writer attempts its own mkdir.
    monkeypatch.setattr(Path, "mkdir", create_collision_before_claim)

    with pytest.raises(FileExistsError, match="appeared during writing"):
        writer.append(_frame(0))

    assert marker.read_text(encoding="utf-8") == "external"
    assert not writer._backup_dir.exists()
    assert writer.state == "aborted"
    assert writer.staging_directory == destination
    assert writer.staging_directory.is_dir()
    assert not destination_lock_path(destination).exists()


def test_create_new_publication_failure_removes_exact_owned_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shared.frames.frame_set_writer as frame_set_writer_module

    destination = tmp_path / "imported-frames"
    writer = FrameSetWriter.create_new(destination, compression=None)
    writer.append(_frame(0))

    def fail_manifest(_destination: Path, _manifest) -> None:
        raise OSError("injected create-new manifest failure")

    monkeypatch.setattr(
        frame_set_writer_module,
        "write_frame_manifest_atomic",
        fail_manifest,
    )

    with pytest.raises(OSError, match="injected create-new manifest failure"):
        writer.finalize()

    assert writer.state == "aborted"
    assert not destination.exists()
    assert not writer.staging_directory.exists()
    assert not destination_lock_path(destination).exists()


def test_create_new_interrupt_after_manifest_commit_keeps_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shared.frames.frame_set_writer as frame_set_writer_module

    destination = tmp_path / "imported-frames"
    writer = FrameSetWriter.create_new(destination, compression=None)
    writer.append(_frame(5))
    real_write_manifest = frame_set_writer_module.write_frame_manifest_atomic

    def write_then_interrupt(frames_dir: Path, manifest) -> None:
        real_write_manifest(frames_dir, manifest)
        raise KeyboardInterrupt("interrupt after create-new manifest commit")

    monkeypatch.setattr(
        frame_set_writer_module,
        "write_frame_manifest_atomic",
        write_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="after create-new manifest commit"):
        writer.finalize()

    assert writer.state == "finalized"
    assert load_frame_manifest(destination).frame_ids == (5,)
    assert not destination_lock_path(destination).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows path budget only")
def test_create_new_preflight_uses_only_real_assembly_paths(tmp_path: Path) -> None:
    destination_name = "frames"
    final_chunk_name = "mpc_frames_2147483647-2147483647.h5"
    target_final_length = WINDOWS_TRANSACTIONAL_PATH_LIMIT - 5
    padding_length = (
        target_final_length
        - len(str(tmp_path))
        - 1
        - len(destination_name)
        - 1
        - len(final_chunk_name)
    )
    if not 1 <= padding_length <= 240:
        pytest.skip("pytest temporary path cannot isolate create-new preflight")

    parent = tmp_path / ("p" * padding_length)
    parent.mkdir()
    destination = parent / destination_name
    final_probe = destination / final_chunk_name
    obsolete_sibling_probe = (
        parent / ".orchav-s-AAAAAAAAAAAAAAAAAAAAAA" / ".p-AAAAAAAAAAAAAAAAAAAAAA.h5.partial"
    )
    assert len(str(final_probe)) <= WINDOWS_TRANSACTIONAL_PATH_LIMIT
    assert len(str(obsolete_sibling_probe)) > WINDOWS_TRANSACTIONAL_PATH_LIMIT

    writer = FrameSetWriter.create_new(destination, compression=None)

    assert writer.staging_directory == destination
    writer.abort()


def test_context_exit_without_explicit_finalize_aborts(tmp_path: Path) -> None:
    destination = tmp_path / "frames"

    with FrameSetWriter.create_new(destination, compression=None) as writer:
        writer.append(_frame(0))

    assert writer.state == "aborted"
    assert not destination.exists()
    assert not writer.staging_directory.exists()
    assert not destination_lock_path(destination).exists()


def test_context_exit_after_explicit_finalize_keeps_publication(tmp_path: Path) -> None:
    destination = tmp_path / "frames"

    with FrameSetWriter.create_new(destination, compression=None) as writer:
        writer.append(_frame(0))
        manifest = writer.finalize()

    assert manifest is not None
    assert writer.state == "finalized"
    assert load_frame_manifest(destination).frame_ids == (0,)


def test_context_catches_base_exception_and_discards_owned_output(tmp_path: Path) -> None:
    destination = tmp_path / "frames"

    with pytest.raises(KeyboardInterrupt, match="producer interrupted"):
        with FrameSetWriter.create_new(destination, compression=None) as writer:
            writer.append(_frame(0))
            raise KeyboardInterrupt("producer interrupted")

    assert writer.state == "aborted"
    assert not destination.exists()
    assert not destination_lock_path(destination).exists()


def test_empty_finalize_preserves_existing_set_and_returns_none(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    destination = _publish_scenario_frame(scenario_root, frame_index=3)
    before = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}

    writer = FrameSetWriter.for_scenario(scenario_root, compression=None)
    assert writer.finalize() is None

    assert writer.state == "finalized"
    assert {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    } == before


def test_empty_create_new_finalize_keeps_destination_absent(tmp_path: Path) -> None:
    destination = tmp_path / "frames"
    writer = FrameSetWriter.create_new(destination)

    assert writer.finalize() is None

    assert writer.state == "finalized"
    assert not destination.exists()


def test_post_promotion_interrupt_keeps_committed_state_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    destination = _publish_scenario_frame(scenario_root, frame_index=0)
    writer = FrameSetWriter.for_scenario(scenario_root, compression=None)
    writer.append(_frame(1))
    promote = writer._promote_staged_frames

    def interrupt_after_promotion() -> None:
        promote()
        raise KeyboardInterrupt("after promotion")

    monkeypatch.setattr(writer, "_promote_staged_frames", interrupt_after_promotion)

    with pytest.raises(KeyboardInterrupt, match="after promotion"):
        writer.finalize()

    assert writer.state == "finalized"
    assert load_frame_manifest(destination).frame_ids == (1,)
    assert not destination_lock_path(destination).exists()


def test_terminal_operations_do_not_reopen_writer(tmp_path: Path) -> None:
    destination = tmp_path / "frames"
    writer = FrameSetWriter.create_new(destination, compression=None)
    writer.append(_frame(0))
    writer.finalize()

    with pytest.raises(RuntimeError, match="finalized"):
        writer.append(_frame(1))
    with pytest.raises(RuntimeError, match="finalized"):
        writer.finalize()
    writer.abort()
    assert writer.state == "finalized"
    assert (destination / FRAMES_MANIFEST_FILENAME).is_file()

    aborted_destination = tmp_path / "aborted"
    aborted = FrameSetWriter.create_new(aborted_destination)
    aborted.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        aborted.append(_frame(0))
    with pytest.raises(RuntimeError, match="aborted"):
        aborted.finalize()
