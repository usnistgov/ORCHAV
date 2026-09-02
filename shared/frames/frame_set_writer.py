"""Transactional publication of complete packed HDF5 v2 frame sets.

``FrameSetWriter`` is the producer-neutral HDF5-v2 write boundary for complete
ORCHAV frame runs. Producers append normalized :class:`StandardMPCFrame`
values to a private assembly directory. For scenario replacement, explicit
finalization atomically promotes a complete sibling directory.
``create_new()`` instead claims the absent destination itself and writes the
manifest last; no existing data can be replaced, and an unpublished directory
is visibly incomplete.

The physical HDF5 codec remains deliberately narrower: it owns one chunk,
while this class owns chunk rotation, run identity, locking, rollback, and the
complete frame-set commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from shared.logging import get_logger

from .directory_ownership import (
    FrameDirectorySafetyError,
    FrameDirectorySnapshot,
    ManagedFrameDirectoryLock,
    capture_frame_directory,
    compact_uuid_token,
    destination_lock_path,
    discover_private_frame_transactions,
    preflight_windows_transaction_paths,
)
from .manifest import (
    FRAMES_MANIFEST_FILENAME,
    FrameChunkManifest,
    FrameSetManifest,
    load_frame_manifest,
    manifest_from_chunks,
    write_frame_manifest_atomic,
)
from .packed_hdf5_writer import (
    PackedMPCChunkBoundaryError,
    PackedMPCChunkWriter,
    PreparedPackedMPCFrame,
)
from .types import StandardMPCFrame

logger = get_logger(__name__)

DEFAULT_FRAME_CHUNK_SIZE = 100
DEFAULT_FRAME_COMPRESSION = "lzf"
MAX_FRAMES_PER_CHUNK = 100
MAX_UNCOMPRESSED_BYTES_PER_CHUNK = 256 * 1024 * 1024
_CANONICAL_FINAL_CHUNK_PROBE = "mpc_frames_2147483647-2147483647.h5"
_CONSTRUCTION_TOKEN = object()

_DirectoryIdentity = tuple[int, int, int]


def _directory_identity(value: os.stat_result) -> _DirectoryIdentity:
    return (stat.S_IFMT(value.st_mode), int(value.st_dev), int(value.st_ino))


def _lexical_entry_exists(path: Path) -> bool:
    """Return whether the exact directory entry exists without following it."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise FrameDirectorySafetyError(
            f"Could not inspect private HDF5 transaction path {path}: {exc}"
        ) from exc
    return True


def _final_publication_path_probes(frames_dir: Path) -> tuple[Path, ...]:
    """Return final paths whose configured basename can exceed staging paths."""
    return (
        frames_dir,
        frames_dir / FRAMES_MANIFEST_FILENAME,
        frames_dir / _CANONICAL_FINAL_CHUNK_PROBE,
    )


def _preflight_destination_before_capture(
    destination: Path,
    *,
    replace_existing: bool,
) -> None:
    """Preserve actionable Windows path errors before parent validation."""
    parent = destination.parent
    if replace_existing:
        assembly = parent / ".orchav-s-AAAAAAAAAAAAAAAAAAAAAA"
        transaction_paths = [
            assembly,
            parent / ".orchav-b-AAAAAAAAAAAAAAAAAAAAAA",
        ]
    else:
        assembly = destination
        transaction_paths = [destination, destination_lock_path(destination)]
    preflight_windows_transaction_paths(
        [
            *transaction_paths,
            assembly / ".p-AAAAAAAAAAAAAAAAAAAAAA.h5.partial",
            assembly / ".frames_manifest.json.2147483647.tmp",
            *_final_publication_path_probes(destination),
        ]
    )


class FrameSetWriter:
    """Append normalized frames and explicitly publish one complete frame set."""

    def __init__(
        self,
        destination: Path,
        *,
        _construction_token: object,
        snapshot: FrameDirectorySnapshot | None,
        replace_existing: bool,
        chunk_size: int,
        compression: str | None,
        max_frames_per_chunk: int,
        max_uncompressed_bytes: int,
    ) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("FrameSetWriter must be created with for_scenario() or create_new()")
        configured_chunk_size = int(chunk_size)
        if configured_chunk_size <= 0:
            raise ValueError("HDF5 frame chunk_size must be a positive integer")
        if max_frames_per_chunk <= 0:
            raise ValueError("HDF5 maximum frames per chunk must be positive")
        if max_uncompressed_bytes <= 0:
            raise ValueError("HDF5 uncompressed byte limit must be positive")

        self.destination = destination
        self.configured_chunk_size = configured_chunk_size
        self.chunk_size = min(configured_chunk_size, max_frames_per_chunk)
        self.max_frames_per_chunk = int(max_frames_per_chunk)
        self.max_uncompressed_bytes = int(max_uncompressed_bytes)
        self.compression = compression
        self._compression_manifest = self._describe_compression(compression)
        self.generated_chunks: list[FrameChunkManifest] = []
        self.chunk_counter = 0

        self._replace_existing = replace_existing
        self._directory_snapshot = snapshot
        self._directory_lock: ManagedFrameDirectoryLock | None = None
        self._writer: PackedMPCChunkWriter | None = None
        self._last_frame_id: int | None = None
        self._boundary_rotations = 0
        self._state = "open"
        self._generation_id = str(uuid4())
        self._transaction_id = compact_uuid_token()
        parent = destination.parent
        self._staging_dir = (
            parent / f".orchav-s-{self._transaction_id}" if replace_existing else destination
        )
        self._backup_dir = parent / f".orchav-b-{self._transaction_id}"
        self._staging_owned = False
        self._create_claim_identity: _DirectoryIdentity | None = None

        self._preflight_transaction_paths()
        self._report_existing_private_transactions()

    @classmethod
    def for_scenario(
        cls,
        scenario_root: str | os.PathLike[str],
        *,
        chunk_size: int = DEFAULT_FRAME_CHUNK_SIZE,
        compression: str | None = DEFAULT_FRAME_COMPRESSION,
    ) -> "FrameSetWriter":
        """Create a replacement-capable writer for ``<scenario>/frames``.

        Existing nonempty output must be a valid manifest-owned frame set.
        The scenario directory must already exist as a real directory.
        """
        requested_root = Path(scenario_root).expanduser()
        try:
            root = requested_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FrameDirectorySafetyError(
                f"Could not resolve scenario root {requested_root}: {exc}"
            ) from exc
        if not root.is_dir():
            raise FrameDirectorySafetyError(f"Scenario root must be an existing directory: {root}")
        destination = root / "frames"
        _preflight_destination_before_capture(destination, replace_existing=True)
        snapshot = capture_frame_directory(destination)
        return cls(
            snapshot.destination,
            _construction_token=_CONSTRUCTION_TOKEN,
            snapshot=snapshot,
            replace_existing=True,
            chunk_size=chunk_size,
            compression=compression,
            max_frames_per_chunk=MAX_FRAMES_PER_CHUNK,
            max_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES_PER_CHUNK,
        )

    @classmethod
    def create_new(
        cls,
        destination: str | os.PathLike[str],
        *,
        chunk_size: int = DEFAULT_FRAME_CHUNK_SIZE,
        compression: str | None = DEFAULT_FRAME_COMPRESSION,
    ) -> "FrameSetWriter":
        """Create a writer for an absent destination that is never replaced.

        The destination's parent must already exist. This factory is intended
        for importers and derived-data tools that choose a new output root.
        The first :meth:`begin` or :meth:`append` exclusively creates that
        directory. Callers must not mutate it until finalize or abort returns;
        readers treat it as incomplete until the manifest is written last.
        """
        requested = Path(destination).expanduser()
        requested_absolute = requested.absolute()
        try:
            parent = requested_absolute.parent.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FrameDirectorySafetyError(
                "FrameSetWriter.create_new() requires an existing parent directory: "
                f"{requested_absolute.parent}"
            ) from exc
        if not parent.is_dir():
            raise FrameDirectorySafetyError(
                "FrameSetWriter.create_new() requires an existing parent directory: " f"{parent}"
            )
        requested = parent / requested_absolute.name
        _preflight_destination_before_capture(requested, replace_existing=False)
        if _lexical_entry_exists(requested):
            raise FileExistsError(
                f"FrameSetWriter.create_new() requires an absent destination: {requested}"
            )
        return cls(
            requested,
            _construction_token=_CONSTRUCTION_TOKEN,
            snapshot=None,
            replace_existing=False,
            chunk_size=chunk_size,
            compression=compression,
            max_frames_per_chunk=MAX_FRAMES_PER_CHUNK,
            max_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES_PER_CHUNK,
        )

    @property
    def staging_directory(self) -> Path:
        """Assembly directory available for producer diagnostic sidecars.

        Replacement writers use a private sibling. ``create_new()`` uses its
        exclusively claimed, still-unpublished destination until the manifest
        is written last.
        """
        return self._staging_dir

    @property
    def generation_id(self) -> str:
        """Return the identity shared by every artifact from this write."""
        return self._generation_id

    @property
    def state(self) -> str:
        """Return the lifecycle state for diagnostics and tests."""
        return self._state

    @staticmethod
    def _describe_compression(compression: Any) -> dict[str, Any]:
        if compression is None:
            return {"configured": None, "filter": "none", "shuffle": False}
        if not isinstance(compression, str):
            raise ValueError("HDF5 compression must be a string or None")
        normalized = compression.strip().lower()
        if normalized in {"none", "fast"}:
            return {
                "configured": normalized,
                "filter": "none",
                "shuffle": False,
            }
        if normalized in {"lzf", "balanced"}:
            return {
                "configured": normalized,
                "filter": "lzf",
                "shuffle": True,
            }
        if normalized in {"gzip", "gzip-4", "compact"}:
            return {
                "configured": normalized,
                "filter": "gzip",
                "level": 4,
                "shuffle": True,
            }
        raise ValueError(
            "compression must be one of None, 'none', 'lzf'/'balanced', "
            "or 'gzip'/'gzip-4'/'compact'"
        )

    def _preflight_transaction_paths(self) -> None:
        partial_probe = self._staging_dir / ".p-AAAAAAAAAAAAAAAAAAAAAA.h5.partial"
        manifest_probe = self._staging_dir / ".frames_manifest.json.2147483647.tmp"
        transaction_paths = [self._staging_dir]
        if self._replace_existing:
            transaction_paths.append(self._backup_dir)
        preflight_windows_transaction_paths(
            [
                *transaction_paths,
                partial_probe,
                manifest_probe,
                destination_lock_path(self.destination),
                *_final_publication_path_probes(self.destination),
            ]
        )

    def _report_existing_private_transactions(self) -> None:
        artifacts = discover_private_frame_transactions(self.destination.parent)
        if not artifacts:
            return
        details = ", ".join(f"{artifact.kind}={artifact.path.name}" for artifact in artifacts)
        logger.warning(
            "Found possible active or interrupted ORCHAV frame transactions "
            "beside %s; they were left untouched (%s)",
            self.destination,
            details,
        )

    def _require_open(self, operation: str) -> None:
        if self._state != "open":
            raise RuntimeError(f"cannot {operation} after frame-set output is {self._state}")

    def _create_owned_staging_directory(self) -> None:
        """Create and record this writer's assembly directory."""
        try:
            self._staging_dir.mkdir()
        except FileExistsError as exc:
            if not self._replace_existing:
                raise FileExistsError(
                    "FrameSetWriter.create_new() destination appeared during "
                    f"writing and was left untouched: {self.destination}"
                ) from exc
            raise
        self._staging_owned = True
        if not self._replace_existing:
            try:
                value = self._staging_dir.lstat()
            except BaseException:
                # Without an exact identity, cleanup must leave the directory
                # rather than infer ownership from its expected name.
                raise
            if not stat.S_ISDIR(value.st_mode):
                raise RuntimeError(
                    f"create_new did not claim a real directory: {self._staging_dir}"
                )
            self._create_claim_identity = _directory_identity(value)

    def _begin_directory_transaction(self) -> None:
        """Acquire the destination lock before creating private run output."""
        if self._directory_lock is not None:
            return

        if self._directory_snapshot is not None:
            self._directory_snapshot.revalidate()
        directory_lock = ManagedFrameDirectoryLock(
            self.destination,
            owner_token=compact_uuid_token(),
        )
        # Publish the lock object before acquisition so interruption cleanup can
        # always find an acquired physical lock.
        self._directory_lock = directory_lock
        try:
            directory_lock.acquire()
            if self._directory_snapshot is not None:
                self._directory_snapshot.revalidate()
            if self._replace_existing:
                for private_path in (self._staging_dir, self._backup_dir):
                    if _lexical_entry_exists(private_path):
                        raise FileExistsError(
                            "Private HDF5 transaction path already exists; it was "
                            f"left untouched: {private_path}"
                        )
            self._create_owned_staging_directory()
        except BaseException:
            self.abort()
            raise

    def begin(self) -> None:
        """Reserve the destination before expensive producer work begins."""
        self._require_open("begin frame-set output")
        self._begin_directory_transaction()

    def _release_directory_lock(self) -> None:
        directory_lock = self._directory_lock
        if directory_lock is None:
            return
        try:
            directory_lock.release()
        except BaseException as exc:
            logger.warning("Could not release the HDF5 publication lock: %s", exc)
        finally:
            self._directory_lock = None

    def _new_writer(self) -> PackedMPCChunkWriter:
        self._begin_directory_transaction()
        return PackedMPCChunkWriter(
            self._staging_dir,
            generation_id=self._generation_id,
            compression=self.compression,
            partial_name=f".p-{compact_uuid_token()}.h5.partial",
        )

    def _ensure_writer(self) -> PackedMPCChunkWriter:
        if self._writer is None:
            self._writer = self._new_writer()
        return self._writer

    def _finalize_active_chunk(self) -> None:
        writer = self._writer
        if writer is None:
            return
        if writer.frame_count == 0:
            writer.discard()
            self._writer = None
            return
        chunk = writer.finalize_to_range_name()
        self._writer = None
        self.generated_chunks.append(chunk)
        self.chunk_counter += 1

    def _would_exceed_chunk(self, prepared: PreparedPackedMPCFrame) -> bool:
        writer = self._writer
        if writer is None or writer.frame_count == 0:
            return False
        return bool(
            writer.frame_count >= self.chunk_size
            or writer.uncompressed_bytes + prepared.estimated_uncompressed_bytes
            > self.max_uncompressed_bytes
        )

    def _append_with_boundary_rotation(
        self,
        frame: StandardMPCFrame,
        prepared: PreparedPackedMPCFrame,
    ) -> PreparedPackedMPCFrame:
        writer = self._ensure_writer()
        try:
            writer.append_prepared(prepared)
            return prepared
        except PackedMPCChunkBoundaryError:
            if writer.frame_count == 0:
                raise
            self._boundary_rotations += 1
            self._finalize_active_chunk()
            writer = self._ensure_writer()
            prepared = writer.prepare(frame)
            writer.append_prepared(prepared)
            return prepared

    def append(self, frame: StandardMPCFrame) -> None:
        """Append one complete normalized frame in increasing ID order."""
        self._require_open("append a frame")
        try:
            writer = self._ensure_writer()
            prepared = writer.prepare(frame)
            frame_id = int(prepared.packed.frame_index)
            if self._last_frame_id is not None and frame_id <= self._last_frame_id:
                raise ValueError("Frame IDs must be appended in strictly increasing order")

            if self._would_exceed_chunk(prepared):
                self._finalize_active_chunk()
                writer = self._ensure_writer()
                prepared = writer.prepare(frame)

            prepared = self._append_with_boundary_rotation(frame, prepared)
            self._last_frame_id = int(prepared.packed.frame_index)

            active_writer = self._writer
            assert active_writer is not None
            if (
                active_writer.frame_count >= self.chunk_size
                or active_writer.uncompressed_bytes >= self.max_uncompressed_bytes
            ):
                self._finalize_active_chunk()
        except BaseException:
            self.abort()
            raise

    @staticmethod
    def _remove_directory(path: Path) -> None:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass

    def _remove_owned_staging_directory(self) -> None:
        """Remove only the assembly directory this writer still owns."""
        if not self._staging_owned:
            return
        if not self._replace_existing:
            identity = self._create_claim_identity
            try:
                current_identity = _directory_identity(self._staging_dir.lstat())
            except (FileNotFoundError, OSError):
                current_identity = None
            if identity is None or current_identity != identity:
                logger.warning(
                    "Could not prove ownership of incomplete create_new "
                    "destination %s; it was left untouched for inspection",
                    self._staging_dir,
                )
                return
        self._remove_directory(self._staging_dir)
        self._staging_owned = False
        self._create_claim_identity = None

    def abort(self) -> None:
        """Discard the partial chunk and all private output owned by this run."""
        if self._state != "open":
            return
        writer = self._writer
        self._writer = None
        try:
            if writer is not None:
                try:
                    writer.discard()
                except BaseException as exc:
                    logger.warning(
                        "Could not discard partial HDF5 chunk %s: %s",
                        writer.partial_path,
                        exc,
                    )
            if self._staging_owned:
                try:
                    self._remove_owned_staging_directory()
                except BaseException as exc:
                    logger.warning(
                        "Could not remove staged HDF5 output %s: %s",
                        self._staging_dir,
                        exc,
                    )
        finally:
            try:
                self._release_directory_lock()
            except BaseException as exc:
                logger.warning(
                    "Could not release the HDF5 publication lock during abort: %s",
                    exc,
                )
            finally:
                self._state = "aborted"

    def _segmentation(self) -> dict[str, Any]:
        return {
            "policy": "bounded_append",
            "configured_frame_limit": self.configured_chunk_size,
            "effective_frame_limit": self.chunk_size,
            "hard_frame_limit": self.max_frames_per_chunk,
            "uncompressed_byte_limit": self.max_uncompressed_bytes,
            "oversized_single_frame": "one_frame_chunk",
            "boundary_rotations": self._boundary_rotations,
        }

    def _build_manifest(
        self,
        provenance: Mapping[str, Any] | None = None,
    ) -> FrameSetManifest:
        created_utc = datetime.now(timezone.utc).isoformat()
        provenance_dict = dict(provenance or {})
        segmentation = self._segmentation()
        identity = {
            "generation_id": self._generation_id,
            "created_utc": created_utc,
            "chunks": [chunk.to_dict() for chunk in self.generated_chunks],
            "compression": self._compression_manifest,
            "segmentation": segmentation,
            "provenance": provenance_dict,
        }
        frame_set_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return manifest_from_chunks(
            generation_id=self._generation_id,
            frame_set_id=frame_set_id,
            chunks=self.generated_chunks,
            compression=self._compression_manifest,
            segmentation=segmentation,
            provenance=provenance_dict,
            created_utc=created_utc,
        )

    def _promote_staged_frames(self) -> None:
        final = self.destination
        staged = self._staging_dir
        backup = self._backup_dir
        moved_previous = False
        attempting_staged_promotion = False
        publication_committed = False

        if self._directory_lock is None or not self._directory_lock.acquired:
            raise RuntimeError("HDF5 frame publication requires the destination lock")
        if not self._replace_existing:
            raise RuntimeError("create_new publication does not replace a directory")
        if self._directory_snapshot is None:
            raise RuntimeError("scenario replacement requires a captured frame directory")
        self._directory_snapshot.revalidate()
        if _lexical_entry_exists(backup):
            raise FileExistsError(f"rollback path already exists: {backup}")

        try:
            if _lexical_entry_exists(final):
                os.replace(final, backup)
                moved_previous = True
            attempting_staged_promotion = True
            os.replace(staged, final)
            self._staging_owned = False
            publication_committed = True
            self._state = "finalized"
        except BaseException as promotion_error:
            if self._state == "finalized":
                # An interruption after the commit-state assignment must not
                # roll back an already published set or relabel it aborted.
                raise
            try:
                if _lexical_entry_exists(backup):
                    if _lexical_entry_exists(final):
                        os.replace(final, staged)
                        self._staging_owned = True
                    os.replace(backup, final)
                elif (
                    attempting_staged_promotion
                    and _lexical_entry_exists(final)
                    and not _lexical_entry_exists(staged)
                ):
                    os.replace(final, staged)
                    self._staging_owned = True
            except BaseException as rollback_error:
                if _lexical_entry_exists(backup):
                    raise RuntimeError(
                        f"{promotion_error}; previous frames remain at {backup} "
                        f"because rollback failed: {rollback_error}"
                    ) from promotion_error
                failed_location = final if _lexical_entry_exists(final) else staged
                raise RuntimeError(
                    f"{promotion_error}; failed publication remains at "
                    f"{failed_location} because rollback failed: {rollback_error}"
                ) from promotion_error
            raise

        assert publication_committed
        if moved_previous:
            try:
                self._remove_directory(backup)
            except BaseException as exc:
                logger.warning(
                    "Published the new frame set, but could not remove the "
                    "prior-frame backup %s: %s",
                    backup,
                    exc,
                )

    def finalize(
        self,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> FrameSetManifest | None:
        """Explicitly publish the complete set, or preserve output when empty."""
        self._require_open("finalize frame-set output")
        try:
            self._finalize_active_chunk()
            if not self.generated_chunks:
                if self._staging_owned:
                    self._remove_owned_staging_directory()
                self._state = "finalized"
                self._release_directory_lock()
                return None

            manifest = self._build_manifest(provenance)
            try:
                write_frame_manifest_atomic(self._staging_dir, manifest)
            except BaseException:
                if not self._replace_existing:
                    try:
                        publication_committed = load_frame_manifest(self.destination) == manifest
                    except Exception:
                        publication_committed = False
                    if publication_committed:
                        self._state = "finalized"
                        self._staging_owned = False
                        self._create_claim_identity = None
                raise

            if self._replace_existing:
                self._promote_staged_frames()
            else:
                # The manifest is the create-new commit marker. Until this
                # assignment the exclusively claimed directory is disposable.
                self._state = "finalized"
                self._staging_owned = False
                self._create_claim_identity = None
        except BaseException:
            if self._state == "finalized":
                # Promotion committed before the interruption. Preserve that
                # terminal state while still releasing the cooperating lock.
                self._release_directory_lock()
            else:
                self.abort()
            raise

        # Promotion is the commit point. Cleanup cannot turn the installed set
        # into a failed generation.
        self._release_directory_lock()
        return manifest

    def __enter__(self) -> "FrameSetWriter":
        self.begin()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        # A normal context exit is not a commit signal. Callers must make the
        # publication boundary explicit with finalize().
        self.abort()


__all__ = [
    "DEFAULT_FRAME_CHUNK_SIZE",
    "DEFAULT_FRAME_COMPRESSION",
    "FrameSetWriter",
    "MAX_FRAMES_PER_CHUNK",
    "MAX_UNCOMPRESSED_BYTES_PER_CHUNK",
]
