"""Publication safety for complete scenario HDF5 frame sets.

``FrameSetWriter`` replaces the fixed ``<scenario>/frames`` directory as one
unit. Before a run starts, this module accepts an absent or empty destination,
or verifies that a nonempty destination is a complete packed-HDF5-v2 frame set.
Snapshot revalidation prevents publication when that destination changes while
the replacement is being generated.

The scenario root is resolved by the caller. This module checks the exact
``frames`` entry for indirection and assumes an ordinary local filesystem.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .manifest import FRAMES_MANIFEST_FILENAME, FrameManifestError, load_frame_manifest

WINDOWS_TRANSACTIONAL_PATH_LIMIT = 259

_PRIVATE_TRANSACTION_RE = re.compile(r"^\.orchav-(?P<kind>[sbl])-(?P<token>[A-Za-z0-9_-]{22})$")
_ALLOWED_FRAME_SIDECARS = frozenset({"path_filter_diagnostic.png"})


class FrameDirectorySafetyError(ValueError):
    """Raised before an unsafe or unowned frame destination is mutated."""


class FrameDirectoryChangedError(RuntimeError):
    """Raised when a captured frame destination changed before promotion."""


class FrameDirectoryLockError(RuntimeError):
    """Raised when a destination-scoped publication lock cannot be used safely."""


@dataclass(frozen=True, slots=True)
class _PathIdentity:
    mode: int
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(
        cls,
        value: os.stat_result,
        *,
        include_content_metadata: bool = True,
    ) -> "_PathIdentity":
        return cls(
            mode=stat.S_IFMT(value.st_mode),
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size) if include_content_metadata else 0,
            mtime_ns=int(value.st_mtime_ns) if include_content_metadata else 0,
        )

    def identifies_same_object(self, other: "_PathIdentity") -> bool:
        """Compare stable filesystem identity while allowing content changes."""
        return (
            self.mode,
            self.device,
            self.inode,
        ) == (
            other.mode,
            other.device,
            other.inode,
        )


@dataclass(frozen=True, slots=True)
class _DirectoryEntryFingerprint:
    relative_path: str
    identity: _PathIdentity


@dataclass(frozen=True, slots=True)
class _DirectoryFingerprint:
    identity: _PathIdentity
    entries: tuple[_DirectoryEntryFingerprint, ...]
    manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class FrameDirectorySnapshot:
    """Immutable ownership and filesystem state captured before generation.

    Call :meth:`revalidate` while holding a
    :class:`ManagedFrameDirectoryLock` and immediately before replacing the
    destination.
    """

    destination: Path
    state: Literal["absent", "empty", "managed"]
    _fingerprint: _DirectoryFingerprint | None

    def revalidate(self) -> None:
        """Fail if ownership, path safety, or the closed inventory changed."""
        revalidate_frame_directory(self)


@dataclass(frozen=True, slots=True)
class PrivateFrameTransactionArtifact:
    """One strictly named private transaction path left untouched by discovery."""

    path: Path
    kind: Literal["staging", "backup", "lock"]


def compact_uuid_token() -> str:
    """Return all 128 UUID bits in a shorter filesystem-safe representation."""
    return base64.urlsafe_b64encode(uuid4().bytes).decode("ascii").rstrip("=")


def _absolute_without_resolving(path: Path) -> Path:
    try:
        return path.expanduser().absolute()
    except (OSError, RuntimeError) as exc:
        raise FrameDirectorySafetyError(
            f"Could not make output path absolute: {path}: {exc}"
        ) from exc


def _is_junction(path: Path) -> bool:
    predicate = getattr(path, "is_junction", None)
    if predicate is None:
        return False
    try:
        return bool(predicate())
    except OSError as exc:
        raise FrameDirectorySafetyError(f"Could not inspect path junction {path}: {exc}") from exc


def _is_mount_point(path: Path) -> bool:
    try:
        return path.is_mount()
    except OSError as exc:
        raise FrameDirectorySafetyError(
            f"Could not inspect path mount point {path}: {exc}"
        ) from exc


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_flag)


def _resolved(path: Path, label: str) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise FrameDirectorySafetyError(f"Could not resolve {label} {path}: {exc}") from exc


def _normalized_destination(destination: Path) -> Path:
    """Return an absolute entry path without resolving that final entry."""

    result = _absolute_without_resolving(destination)
    parent = result.parent
    if not parent.is_dir():
        raise FrameDirectorySafetyError(
            "The frame directory parent must already exist: "
            f"{parent}. Create the scenario or analysis parent first."
        )
    return result


def _indirect_kind(path: Path, value: os.stat_result) -> str | None:
    """Describe indirection at the exact managed entry, if present."""

    if stat.S_ISLNK(value.st_mode):
        return "symbolic link"
    if _is_junction(path):
        return "junction"
    if _is_reparse_point(value):
        return "Windows reparse point"
    if path != Path(path.anchor) and _is_mount_point(path):
        return "mount point"
    return None


def _fingerprint_directory(path: Path) -> _DirectoryFingerprint | None:
    try:
        root_stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FrameDirectorySafetyError(
            f"Could not inspect managed frame directory {path}: {exc}"
        ) from exc

    indirect_kind = _indirect_kind(path, root_stat)
    if indirect_kind is not None:
        raise FrameDirectorySafetyError(
            f"Managed frame entry must not be a {indirect_kind}: {path}"
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        raise FrameDirectorySafetyError(
            f"Managed frame destination exists but is not a directory: {path}"
        )

    entries: list[_DirectoryEntryFingerprint] = []
    try:
        with os.scandir(path) as iterator:
            for entry in iterator:
                entry_path = Path(entry.path)
                entry_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise FrameDirectorySafetyError(
                        "Managed frame directories have a closed, flat file inventory; "
                        f"unexpected entry: {entry_path}"
                    )
                entries.append(
                    _DirectoryEntryFingerprint(
                        relative_path=entry.name,
                        identity=_PathIdentity.from_stat(entry_stat),
                    )
                )
    except FileNotFoundError as exc:
        raise FrameDirectoryChangedError(
            f"Managed frame directory changed while it was inspected: {path}"
        ) from exc
    except OSError as exc:
        raise FrameDirectorySafetyError(
            f"Could not enumerate managed frame directory {path}: {exc}"
        ) from exc

    entries.sort(key=lambda item: item.relative_path)
    manifest_path = path / FRAMES_MANIFEST_FILENAME
    manifest_sha256 = None
    if any(item.relative_path == FRAMES_MANIFEST_FILENAME for item in entries):
        try:
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise FrameDirectorySafetyError(
                f"Could not fingerprint {manifest_path}: {exc}"
            ) from exc
    return _DirectoryFingerprint(
        identity=_PathIdentity.from_stat(
            root_stat,
            include_content_metadata=False,
        ),
        entries=tuple(entries),
        manifest_sha256=manifest_sha256,
    )


def _capture_stable_state(
    destination: Path,
) -> tuple[Literal["absent", "empty", "managed"], _DirectoryFingerprint | None]:
    before = _fingerprint_directory(destination)
    if before is None:
        after = _fingerprint_directory(destination)
        if after is not None:
            raise FrameDirectoryChangedError(
                f"Managed frame destination appeared while it was inspected: {destination}"
            )
        return "absent", None

    if not before.entries:
        state: Literal["empty", "managed"] = "empty"
    else:
        try:
            manifest = load_frame_manifest(destination, verify_files=True)
        except FrameManifestError as exc:
            raise FrameDirectorySafetyError(
                f"Refusing to replace nonempty directory {destination}: it is not "
                f"a valid HDF5 v2 frame set ({exc})"
            ) from exc
        if not manifest.chunks or not manifest.frame_ids:
            raise FrameDirectorySafetyError(
                f"Refusing to replace nonempty directory {destination}: its "
                "HDF5 v2 manifest contains no published frames and therefore "
                "does not prove generator ownership"
            )
        actual_names = {entry.relative_path for entry in before.entries}
        expected_names = {FRAMES_MANIFEST_FILENAME}
        expected_names.update(chunk.file for chunk in manifest.chunks)
        unexpected = sorted(actual_names - expected_names - _ALLOWED_FRAME_SIDECARS)
        if unexpected:
            raise FrameDirectorySafetyError(
                f"Refusing to replace {destination}: its closed frame-set inventory "
                "contains unrecognized entries (" + ", ".join(unexpected) + ")"
            )
        state = "managed"

    after = _fingerprint_directory(destination)
    if after != before:
        raise FrameDirectoryChangedError(
            f"Managed frame directory changed while ownership was inspected: {destination}"
        )
    return state, after


def capture_frame_directory(
    destination: str | os.PathLike[str],
) -> FrameDirectorySnapshot:
    """Validate and capture one fixed frame entry before output mutation.

    Existing empty directories and absent destinations are valid.  An existing
    nonempty directory is valid only when ``frames_manifest.json`` advertises
    at least one frame and every advertised HDF5 v2 chunk passes shared
    validation.  A syntactically valid zero-frame manifest is not ownership
    proof because successful writers never publish one.  A qualifying manifest
    proves ownership of the complete directory. Only the manifest, its listed
    chunks, and the optional path-filter diagnostic are recognized; everything
    else is refused rather than guessed to be disposable.
    """

    normalized_destination = _normalized_destination(Path(destination))
    state, fingerprint = _capture_stable_state(normalized_destination)
    return FrameDirectorySnapshot(
        destination=normalized_destination,
        state=state,
        _fingerprint=fingerprint,
    )


def revalidate_frame_directory(snapshot: FrameDirectorySnapshot) -> None:
    """Recheck a captured destination and fail on any ownership/state change."""
    try:
        state, fingerprint = _capture_stable_state(snapshot.destination)
    except (FrameDirectorySafetyError, FrameDirectoryChangedError) as exc:
        raise FrameDirectoryChangedError(
            "Managed frame destination changed during generation or failed "
            f"safety revalidation: {exc}"
        ) from exc
    if state != snapshot.state or fingerprint != snapshot._fingerprint:
        raise FrameDirectoryChangedError(
            "Managed frame destination changed during generation; the new frame "
            f"set was not published: {snapshot.destination}"
        )


def destination_lock_path(destination: str | os.PathLike[str]) -> Path:
    """Return the compact sibling lock path for one resolved destination."""
    resolved = _resolved(Path(destination), "publication destination")
    identity = os.path.normcase(str(resolved)).encode("utf-8")
    token = base64.urlsafe_b64encode(hashlib.sha256(identity).digest()[:16]).decode("ascii")
    return resolved.parent / f".orchav-l-{token.rstrip('=')}"


class ManagedFrameDirectoryLock:
    """Exclusive output-publication lock with token-safe cleanup.

    A stale lock is never removed automatically.  :meth:`release` verifies
    both filesystem identity and the private owner token before unlinking, so
    a replaced or foreign lock survives for manual inspection.
    """

    def __init__(
        self,
        destination: str | os.PathLike[str],
        *,
        owner_token: str | None = None,
    ) -> None:
        self.destination = _resolved(Path(destination), "publication destination")
        self.path = destination_lock_path(self.destination)
        self.owner_token = owner_token or compact_uuid_token()
        self._identity: _PathIdentity | None = None
        self._acquired = False

    @property
    def acquired(self) -> bool:
        """Whether this object currently owns its exact lock file."""
        return self._acquired

    def acquire(self) -> None:
        """Create the lock exclusively or fail without changing an old lock."""
        if self._acquired:
            return
        if (
            not self.path.parent.is_dir()
            or self.path.parent.is_symlink()
            or _is_junction(self.path.parent)
        ):
            raise FrameDirectoryLockError(
                "Cannot lock publication destination because its parent is absent or "
                f"not a real directory: {self.path.parent}"
            )
        payload = {
            "kind": "orchav_directory_publication_lock",
            "version": 1,
            "owner_token": self.owner_token,
            "destination": str(self.destination),
            "pid": os.getpid(),
        }
        serialized_payload = json.dumps(payload, sort_keys=True) + "\n"
        try:
            handle = self.path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise FrameDirectoryLockError(
                f"Publication destination is already locked: {self.path}. Another "
                "writer may be active; stale locks require manual inspection."
            ) from exc
        except OSError as exc:
            raise FrameDirectoryLockError(
                f"Could not create publication lock {self.path}: {exc}"
            ) from exc

        created_identity: _PathIdentity | None = None
        try:
            with handle:
                created_identity = _PathIdentity.from_stat(self.path.lstat())
                written = handle.write(serialized_payload)
                if written != len(serialized_payload):
                    raise OSError("short write while creating the publication lock")
                handle.flush()
                os.fsync(handle.fileno())
                identity = _PathIdentity.from_stat(os.fstat(handle.fileno()))
                if not created_identity.identifies_same_object(identity):
                    raise FrameDirectoryLockError(
                        "Publication lock changed while it was created and "
                        f"was left untouched: {self.path}"
                    )
        except BaseException as exc:
            cleanup_issue = self._cleanup_failed_acquire(
                created_identity,
                serialized_payload.encode("utf-8"),
            )
            if cleanup_issue is not None:
                add_note = getattr(exc, "add_note", None)
                if add_note is not None:
                    add_note(cleanup_issue)
            if not isinstance(exc, OSError):
                raise
            raise FrameDirectoryLockError(
                f"Could not create publication lock {self.path}: {exc}"
            ) from exc
        self._identity = identity
        self._acquired = True

    def _cleanup_failed_acquire(
        self,
        created_identity: _PathIdentity | None,
        expected_payload: bytes,
    ) -> str | None:
        """Remove only the exact, unmodified-prefix lock this acquire created."""
        if created_identity is None:
            return (
                "The failed lock creation left its path untouched because no "
                f"filesystem identity was captured: {self.path}"
            )
        if created_identity.mode != stat.S_IFREG:
            return (
                "The failed lock creation found a non-regular replacement and "
                f"left it untouched: {self.path}"
            )
        try:
            current_identity = _PathIdentity.from_stat(self.path.lstat())
            current_payload = self.path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            return (
                "The failed lock creation could not safely inspect its path and "
                f"left it untouched: {self.path}: {exc}"
            )
        if not created_identity.identifies_same_object(current_identity):
            return (
                "The failed lock creation found a replaced path and left it "
                f"untouched: {self.path}"
            )
        if not expected_payload.startswith(current_payload):
            return (
                "The failed lock creation found changed content and left it "
                f"untouched: {self.path}"
            )
        try:
            self.path.unlink()
        except OSError as exc:
            return (
                "The failed lock creation could not remove its owned path and "
                f"left it for inspection: {self.path}: {exc}"
            )
        return None

    def release(self) -> None:
        """Remove only the unchanged lock carrying this object's owner token."""
        if not self._acquired:
            return
        assert self._identity is not None
        try:
            current_identity = _PathIdentity.from_stat(self.path.lstat())
        except (FileNotFoundError, OSError) as exc:
            raise FrameDirectoryLockError(
                f"Publication lock changed before cleanup and was left untouched: {self.path}"
            ) from exc
        if current_identity != self._identity:
            raise FrameDirectoryLockError(
                f"Publication lock ownership changed and was left untouched: {self.path}"
            )
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FrameDirectoryLockError(
                f"Publication lock content changed and was left untouched: {self.path}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("owner_token") != self.owner_token:
            raise FrameDirectoryLockError(
                f"Publication lock ownership changed and was left untouched: {self.path}"
            )
        try:
            self.path.unlink()
        except OSError as exc:
            raise FrameDirectoryLockError(
                f"Could not remove owned publication lock {self.path}: {exc}"
            ) from exc
        self._acquired = False
        self._identity = None

    def __enter__(self) -> "ManagedFrameDirectoryLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        try:
            self.release()
        except FrameDirectoryLockError as release_error:
            if exc is None:
                raise
            add_note = getattr(exc, "add_note", None)
            if add_note is not None:
                add_note(str(release_error))


def discover_private_frame_transactions(
    parent: str | os.PathLike[str],
) -> tuple[PrivateFrameTransactionArtifact, ...]:
    """Report strictly named ORCHAV transaction artifacts without deleting any.

    The result may include an active transaction.  Names that merely resemble
    ORCHAV paths, symbolic links, junctions, and unexpected filesystem objects
    are deliberately ignored.
    """

    root = Path(parent)
    try:
        unsafe_root = not root.is_dir() or root.is_symlink() or _is_junction(root)
    except (OSError, FrameDirectorySafetyError):
        return ()
    if unsafe_root:
        return ()
    artifacts: list[PrivateFrameTransactionArtifact] = []
    try:
        children = tuple(root.iterdir())
    except OSError:
        return ()
    kinds: dict[str, Literal["staging", "backup", "lock"]] = {
        "s": "staging",
        "b": "backup",
        "l": "lock",
    }
    for child in children:
        match = _PRIVATE_TRANSACTION_RE.fullmatch(child.name)
        if match is None:
            continue
        try:
            if child.is_symlink() or _is_junction(child):
                continue
        except (OSError, FrameDirectorySafetyError):
            continue
        kind = kinds[match.group("kind")]
        if kind == "lock":
            if not child.is_file():
                continue
        elif not child.is_dir():
            continue
        artifacts.append(PrivateFrameTransactionArtifact(child, kind))
    return tuple(sorted(artifacts, key=lambda item: item.path.name))


def preflight_windows_transaction_paths(
    paths: tuple[Path, ...] | list[Path],
    *,
    limit: int = WINDOWS_TRANSACTIONAL_PATH_LIMIT,
) -> None:
    """Reject a publication path beyond the conservative Windows budget."""
    if os.name != "nt":
        return
    if limit <= 0:
        raise ValueError("Windows transaction path limit must be positive")
    absolute_paths = tuple(_absolute_without_resolving(Path(path)) for path in paths)
    if not absolute_paths:
        return
    measured_paths = tuple(
        (
            path,
            len(str(path).encode("utf-16-le", errors="surrogatepass")) // 2,
        )
        for path in absolute_paths
    )
    longest, length = max(measured_paths, key=lambda item: item[1])
    if length > limit:
        raise FrameDirectorySafetyError(
            f"Publication path uses {length} UTF-16 code units, "
            f"exceeding the safe Windows budget of {limit}: {longest}. "
            "Shorten the scenario path or the explicitly chosen create-new "
            "destination."
        )


__all__ = [
    "WINDOWS_TRANSACTIONAL_PATH_LIMIT",
    "FrameDirectoryChangedError",
    "FrameDirectoryLockError",
    "FrameDirectorySafetyError",
    "FrameDirectorySnapshot",
    "ManagedFrameDirectoryLock",
    "PrivateFrameTransactionArtifact",
    "capture_frame_directory",
    "compact_uuid_token",
    "destination_lock_path",
    "discover_private_frame_transactions",
    "preflight_windows_transaction_paths",
    "revalidate_frame_directory",
]
