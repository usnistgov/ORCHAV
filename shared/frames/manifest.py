"""Authoritative manifest for packed HDF5 frame sets.

Provider startup reads this JSON file without opening HDF5 chunks. It contains
the exact ordered frame identifiers in each chunk and enough file metadata to
reject incomplete or mixed-generation directories cheaply.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

from .contracts import (
    MPC_FRAME_MANIFEST_VERSION,
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
)

FRAMES_MANIFEST_FILENAME = "frames_manifest.json"


class FrameManifestError(ValueError):
    """Raised when a frame manifest or its advertised files are inconsistent."""


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    # Values reach this parser through ``json.loads``, so accepting strings or
    # floats would weaken the on-disk schema and can silently truncate values
    # such as ``2.9`` to ``2``.
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrameManifestError(f"{label} must be an integer")
    if value < minimum:
        raise FrameManifestError(f"{label} must be at least {minimum}")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    """Return a required JSON string whose content is not only whitespace."""
    if not isinstance(value, str):
        raise FrameManifestError(f"{label} must be a string")
    if not value.strip():
        raise FrameManifestError(f"{label} must be non-empty")
    return value


def _require_object(value: Any, label: str) -> dict[str, Any]:
    """Return a required JSON object without silently replacing bad input."""
    if not isinstance(value, Mapping):
        raise FrameManifestError(f"{label} must be a JSON object")
    return dict(value)


def _frame_ids(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise FrameManifestError(f"{label} must be a JSON list")
    result = tuple(_require_int(item, f"{label} item") for item in value)
    if any(right <= left for left, right in zip(result, result[1:])):
        raise FrameManifestError(f"{label} must be strictly increasing")
    return result


@dataclass(frozen=True, slots=True)
class FrameChunkManifest:
    """Inventory entry for one immutable HDF5 chunk.

    ``frame_ids`` is the HDF5 frame-axis row order. ``topology_id`` and
    ``sensing_layout_id`` identify the two layouts that must remain fixed
    inside the chunk, while byte counts support publication checks and storage
    reporting without opening the file.
    """

    file: str
    frame_ids: tuple[int, ...]
    size_bytes: int
    uncompressed_bytes: int
    topology_id: str
    sensing_layout_id: str

    @property
    def start_frame(self) -> int:
        return self.frame_ids[0]

    @property
    def end_frame(self) -> int:
        return self.frame_ids[-1]

    @property
    def count(self) -> int:
        return len(self.frame_ids)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], index: int) -> "FrameChunkManifest":
        label = f"chunks[{index}]"
        file_name = raw.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise FrameManifestError(f"{label}.file must be a non-empty string")
        if PurePath(file_name).name != file_name or Path(file_name).is_absolute():
            raise FrameManifestError(f"{label}.file must be a plain relative filename")
        if not file_name.endswith(".h5"):
            raise FrameManifestError(f"{label}.file must end in .h5")

        ids = _frame_ids(raw.get("frame_ids"), f"{label}.frame_ids")
        if not ids:
            raise FrameManifestError(f"{label}.frame_ids cannot be empty")

        size_bytes = _require_int(raw.get("size_bytes"), f"{label}.size_bytes", minimum=1)
        uncompressed_bytes = _require_int(
            raw.get("uncompressed_bytes", 0),
            f"{label}.uncompressed_bytes",
        )
        topology_id = raw.get("topology_id", "")
        sensing_layout_id = raw.get("sensing_layout_id", "")
        if not isinstance(topology_id, str) or not isinstance(sensing_layout_id, str):
            raise FrameManifestError(f"{label}.topology_id and sensing_layout_id must be strings")

        advertised_count = raw.get("count")
        if advertised_count is not None and _require_int(advertised_count, f"{label}.count") != len(
            ids
        ):
            raise FrameManifestError(f"{label}.count does not match frame_ids")
        if (
            raw.get("start_frame") is not None
            and _require_int(raw["start_frame"], f"{label}.start_frame") != ids[0]
        ):
            raise FrameManifestError(f"{label}.start_frame does not match frame_ids")
        if (
            raw.get("end_frame") is not None
            and _require_int(raw["end_frame"], f"{label}.end_frame") != ids[-1]
        ):
            raise FrameManifestError(f"{label}.end_frame does not match frame_ids")

        return cls(
            file=file_name,
            frame_ids=ids,
            size_bytes=size_bytes,
            uncompressed_bytes=uncompressed_bytes,
            topology_id=topology_id,
            sensing_layout_id=sensing_layout_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation."""
        return {
            "file": self.file,
            "frame_ids": list(self.frame_ids),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "count": self.count,
            "size_bytes": self.size_bytes,
            "uncompressed_bytes": self.uncompressed_bytes,
            "topology_id": self.topology_id,
            "sensing_layout_id": self.sensing_layout_id,
        }


@dataclass(frozen=True, slots=True)
class FrameSetManifest:
    """Authoritative identity and ordered inventory for one published frame set.

    ``generation_id`` identifies the writer run and ``frame_set_id`` pins the
    published snapshot used by local and remote providers. The top-level
    ``frame_ids`` deliberately duplicates the concatenated per-chunk IDs so a
    provider can enumerate frames without opening HDF5; validation rejects any
    disagreement between the two indexes.
    """

    generation_id: str
    frame_set_id: str
    frame_ids: tuple[int, ...]
    chunks: tuple[FrameChunkManifest, ...]
    compression: Mapping[str, Any]
    segmentation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    created_utc: str
    manifest_version: int = MPC_FRAME_MANIFEST_VERSION
    schema_version: int = MPC_HDF5_SCHEMA_VERSION
    storage_layout: str = MPC_HDF5_LAYOUT

    def __post_init__(self) -> None:
        if self.manifest_version != MPC_FRAME_MANIFEST_VERSION:
            raise FrameManifestError(
                f"Unsupported manifest_version={self.manifest_version}; "
                f"expected {MPC_FRAME_MANIFEST_VERSION}"
            )
        if self.schema_version != MPC_HDF5_SCHEMA_VERSION:
            raise FrameManifestError(
                f"Unsupported schema_version={self.schema_version}; "
                f"expected {MPC_HDF5_SCHEMA_VERSION}"
            )
        if self.storage_layout != MPC_HDF5_LAYOUT:
            raise FrameManifestError(
                f"Unsupported storage_layout={self.storage_layout!r}; "
                f"expected {MPC_HDF5_LAYOUT!r}"
            )
        for name, string_value in (
            ("generation_id", self.generation_id),
            ("frame_set_id", self.frame_set_id),
            ("created_utc", self.created_utc),
        ):
            if not isinstance(string_value, str) or not string_value.strip():
                raise FrameManifestError(f"{name} must be a non-empty string")
        for name, mapping_value in (
            ("compression", self.compression),
            ("segmentation", self.segmentation),
            ("provenance", self.provenance),
        ):
            if not isinstance(mapping_value, Mapping):
                raise FrameManifestError(f"{name} must be a mapping")

        concatenated = tuple(frame_id for chunk in self.chunks for frame_id in chunk.frame_ids)
        if concatenated != self.frame_ids:
            raise FrameManifestError(
                "Top-level frame_ids must exactly equal the ordered chunk frame_ids"
            )
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise FrameManifestError("Frame IDs cannot occur in more than one chunk")
        if any(right <= left for left, right in zip(self.frame_ids, self.frame_ids[1:])):
            raise FrameManifestError("Top-level frame_ids must be strictly increasing")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FrameSetManifest":
        """Parse and validate a manifest JSON mapping."""
        chunks_raw = raw.get("chunks")
        if not isinstance(chunks_raw, list):
            raise FrameManifestError("chunks must be a JSON list")
        chunks = tuple(
            FrameChunkManifest.from_dict(item, index)
            for index, item in enumerate(chunks_raw)
            if isinstance(item, Mapping)
        )
        if len(chunks) != len(chunks_raw):
            raise FrameManifestError("Every chunks entry must be a JSON object")

        frame_ids = _frame_ids(raw.get("frame_ids"), "frame_ids")
        manifest = cls(
            manifest_version=_require_int(raw.get("manifest_version"), "manifest_version"),
            schema_version=_require_int(raw.get("schema_version"), "schema_version"),
            storage_layout=str(raw.get("storage_layout", "")),
            generation_id=_require_nonempty_string(raw.get("generation_id"), "generation_id"),
            frame_set_id=_require_nonempty_string(raw.get("frame_set_id"), "frame_set_id"),
            frame_ids=frame_ids,
            chunks=chunks,
            compression=_require_object(raw.get("compression"), "compression"),
            segmentation=_require_object(raw.get("segmentation"), "segmentation"),
            provenance=_require_object(raw.get("provenance"), "provenance"),
            created_utc=_require_nonempty_string(raw.get("created_utc"), "created_utc"),
        )
        if raw.get("total_frames") is not None and _require_int(
            raw["total_frames"], "total_frames"
        ) != len(frame_ids):
            raise FrameManifestError("total_frames does not match frame_ids")
        if raw.get("total_files") is not None and _require_int(
            raw["total_files"], "total_files"
        ) != len(chunks):
            raise FrameManifestError("total_files does not match chunks")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest mapping."""
        return {
            "manifest_version": self.manifest_version,
            "schema_version": self.schema_version,
            "storage_layout": self.storage_layout,
            "generation_id": self.generation_id,
            "frame_set_id": self.frame_set_id,
            "created_utc": self.created_utc,
            "total_frames": len(self.frame_ids),
            "total_files": len(self.chunks),
            "frame_ids": list(self.frame_ids),
            "compression": dict(self.compression),
            "segmentation": dict(self.segmentation),
            "provenance": dict(self.provenance),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    def frame_locations(self) -> dict[int, tuple[FrameChunkManifest, int]]:
        """Map each frame ID to its chunk and row."""
        return {
            frame_id: (chunk, row)
            for chunk in self.chunks
            for row, frame_id in enumerate(chunk.frame_ids)
        }


def load_frame_manifest(
    frames_dir: str | Path,
    *,
    verify_files: bool = True,
) -> FrameSetManifest:
    """Load the authoritative manifest without opening any HDF5 file.

    With ``verify_files=True``, require the exact advertised chunk filenames,
    real regular files rather than links or junctions, and matching byte sizes.
    """
    root = Path(frames_dir)
    path = root / FRAMES_MANIFEST_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FrameManifestError(f"{FRAMES_MANIFEST_FILENAME} is required") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise FrameManifestError(f"Could not read {FRAMES_MANIFEST_FILENAME}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise FrameManifestError(f"{FRAMES_MANIFEST_FILENAME} must contain a JSON object")
    manifest = FrameSetManifest.from_dict(raw)

    if not verify_files:
        return manifest

    expected_names = {chunk.file for chunk in manifest.chunks}
    actual_names = {path.name for path in root.glob("mpc_frames_*.h5")}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise FrameManifestError("Manifest/chunk file set mismatch (" + "; ".join(details) + ")")

    for chunk in manifest.chunks:
        path = root / chunk.file
        try:
            chunk_stat = path.lstat()
        except OSError as exc:
            raise FrameManifestError(f"Could not lstat {chunk.file}: {exc}") from exc
        if stat.S_ISLNK(chunk_stat.st_mode):
            raise FrameManifestError(
                f"{chunk.file} must be a real regular file, not a symbolic link"
            )
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None:
            try:
                if is_junction():
                    raise FrameManifestError(
                        f"{chunk.file} must be a real regular file, not a junction"
                    )
            except OSError as exc:
                raise FrameManifestError(
                    f"Could not inspect {chunk.file} for a junction: {exc}"
                ) from exc
        if not stat.S_ISREG(chunk_stat.st_mode):
            raise FrameManifestError(f"{chunk.file} must be a real regular file")
        actual_size = chunk_stat.st_size
        if actual_size != chunk.size_bytes:
            raise FrameManifestError(
                f"{chunk.file} size does not match manifest "
                f"({actual_size} != {chunk.size_bytes})"
            )
    return manifest


def write_frame_manifest_atomic(
    frames_dir: str | Path,
    manifest: FrameSetManifest,
) -> Path:
    """Publish a complete manifest through a sibling temporary file."""
    root = Path(frames_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / FRAMES_MANIFEST_FILENAME
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def manifest_from_chunks(
    *,
    generation_id: str,
    frame_set_id: str,
    chunks: Sequence[FrameChunkManifest],
    compression: Mapping[str, Any],
    segmentation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    created_utc: str,
) -> FrameSetManifest:
    """Build a validated frame-set manifest from finalized chunk entries."""
    chunk_tuple = tuple(chunks)
    return FrameSetManifest(
        generation_id=generation_id,
        frame_set_id=frame_set_id,
        frame_ids=tuple(frame_id for chunk in chunk_tuple for frame_id in chunk.frame_ids),
        chunks=chunk_tuple,
        compression=dict(compression),
        segmentation=dict(segmentation),
        provenance=dict(provenance),
        created_utc=created_utc,
    )


__all__ = [
    "FRAMES_MANIFEST_FILENAME",
    "FrameChunkManifest",
    "FrameManifestError",
    "FrameSetManifest",
    "load_frame_manifest",
    "manifest_from_chunks",
    "write_frame_manifest_atomic",
]
