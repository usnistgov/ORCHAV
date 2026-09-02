"""Tests for the authoritative HDF5 v2 frame manifest."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shared.frames.manifest import (
    FRAMES_MANIFEST_FILENAME,
    FrameChunkManifest,
    FrameManifestError,
    load_frame_manifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)


def _manifest(tmp_path: Path):
    chunk_path = tmp_path / "mpc_frames_00000-00002.h5"
    chunk_path.write_bytes(b"chunk")
    chunk = FrameChunkManifest(
        file=chunk_path.name,
        frame_ids=(0, 2),
        size_bytes=chunk_path.stat().st_size,
        uncompressed_bytes=123,
        topology_id="topology",
        sensing_layout_id="sensing",
    )
    return manifest_from_chunks(
        generation_id="generation",
        frame_set_id="frame-set",
        chunks=[chunk],
        compression={"profile": "balanced", "algorithm": "lzf"},
        segmentation={"max_frames": 100},
        provenance={"source": "test"},
        created_utc="2026-07-29T00:00:00+00:00",
    )


def test_manifest_round_trip_and_file_verification(tmp_path: Path) -> None:
    expected = _manifest(tmp_path)

    path = write_frame_manifest_atomic(tmp_path, expected)
    loaded = load_frame_manifest(tmp_path)

    assert path.name == FRAMES_MANIFEST_FILENAME
    assert loaded == expected
    assert loaded.frame_locations()[2][1] == 1


def test_manifest_rejects_file_size_change_without_hdf5_open(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    write_frame_manifest_atomic(tmp_path, manifest)
    (tmp_path / manifest.chunks[0].file).write_bytes(b"changed")

    with pytest.raises(FrameManifestError, match="size does not match"):
        load_frame_manifest(tmp_path)


def test_manifest_rejects_top_level_frame_order_disagreement(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    raw = manifest.to_dict()
    raw["frame_ids"] = [0]
    (tmp_path / FRAMES_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(FrameManifestError, match="exactly equal"):
        load_frame_manifest(tmp_path, verify_files=False)


def test_manifest_rejects_path_traversal_filename(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    raw = manifest.to_dict()
    raw["chunks"][0]["file"] = "../outside.h5"
    (tmp_path / FRAMES_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(FrameManifestError, match="plain relative filename"):
        load_frame_manifest(tmp_path, verify_files=False)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("manifest_version",), "2", "manifest_version must be an integer"),
        (("schema_version",), 2.9, "schema_version must be an integer"),
        (("frame_ids", 0), 0.0, "frame_ids item must be an integer"),
        (
            ("chunks", 0, "size_bytes"),
            "5",
            r"chunks\[0\]\.size_bytes must be an integer",
        ),
    ],
)
def test_manifest_rejects_non_integer_json_scalars(
    tmp_path: Path,
    field_path: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    manifest = _manifest(tmp_path)
    raw = manifest.to_dict()
    target: Any = raw
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = value
    (tmp_path / FRAMES_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(FrameManifestError, match=message):
        load_frame_manifest(tmp_path, verify_files=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_id", None, "generation_id must be a string"),
        ("generation_id", "", "generation_id must be non-empty"),
        ("frame_set_id", 42, "frame_set_id must be a string"),
        ("frame_set_id", "   ", "frame_set_id must be non-empty"),
        ("created_utc", None, "created_utc must be a string"),
        ("created_utc", "", "created_utc must be non-empty"),
    ],
)
def test_manifest_rejects_invalid_required_strings(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = _manifest(tmp_path)
    raw = manifest.to_dict()
    raw[field] = value
    (tmp_path / FRAMES_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(FrameManifestError, match=message):
        load_frame_manifest(tmp_path, verify_files=False)


@pytest.mark.parametrize("field", ["compression", "segmentation", "provenance"])
@pytest.mark.parametrize("value", [None, [], "not-an-object", 1])
def test_manifest_rejects_non_object_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest = _manifest(tmp_path)
    raw = manifest.to_dict()
    raw[field] = value
    (tmp_path / FRAMES_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(FrameManifestError, match=rf"{field} must be a JSON object"):
        load_frame_manifest(tmp_path, verify_files=False)


def test_manifest_rejects_advertised_chunk_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    chunk_path = frames_dir / "mpc_frames_00000-00000.h5"
    chunk_path.write_bytes(b"external chunk")

    manifest = manifest_from_chunks(
        generation_id="generation",
        frame_set_id="frame-set",
        chunks=[
            FrameChunkManifest(
                file=chunk_path.name,
                frame_ids=(0,),
                size_bytes=chunk_path.stat().st_size,
                uncompressed_bytes=0,
                topology_id="topology",
                sensing_layout_id="sensing",
            )
        ],
        compression={},
        segmentation={},
        provenance={},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    real_lstat = Path.lstat

    def lstat_as_symlink(path: Path):
        if path == chunk_path:
            current = real_lstat(path)
            return SimpleNamespace(
                st_mode=stat.S_IFLNK | 0o777,
                st_size=current.st_size,
            )
        return real_lstat(path)

    monkeypatch.setattr(
        Path,
        "lstat",
        lstat_as_symlink,
    )

    with pytest.raises(
        FrameManifestError,
        match="real regular file, not a symbolic link",
    ):
        load_frame_manifest(frames_dir)


def test_manifest_rejects_advertised_chunk_directory(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    write_frame_manifest_atomic(tmp_path, manifest)
    chunk_path = tmp_path / manifest.chunks[0].file
    chunk_path.unlink()
    chunk_path.mkdir()

    with pytest.raises(FrameManifestError, match="must be a real regular file"):
        load_frame_manifest(tmp_path)


def test_manifest_rejects_advertised_chunk_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    write_frame_manifest_atomic(tmp_path, manifest)
    chunk_path = tmp_path / manifest.chunks[0].file
    real_is_junction = getattr(Path, "is_junction", None)

    def is_junction(path: Path) -> bool:
        if path == chunk_path:
            return True
        return bool(real_is_junction is not None and real_is_junction(path))

    monkeypatch.setattr(Path, "is_junction", is_junction, raising=False)

    with pytest.raises(FrameManifestError, match="real regular file, not a junction"):
        load_frame_manifest(tmp_path)
