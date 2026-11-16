"""Disposable, bounded disk cache for derived sensing display data.

Only local manifest-backed recordings qualify. Recording-wide identities include
the manifest and every advertised chunk; per-result identities cover the manifest
and the selected chunk. Source stamps include file identity, size and modification
and change times. Cache reads never deserialize executable objects. The source
recording remains the authority and is never written here.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import zipfile
from collections.abc import Callable, Mapping
from math import prod
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np

from shared.frames.providers import Hdf5Provider

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _WindowsFileBasicInfo(ctypes.Structure):
        _fields_ = [
            ("creation_time", ctypes.c_longlong),
            ("access_time", ctypes.c_longlong),
            ("write_time", ctypes.c_longlong),
            ("change_time", ctypes.c_longlong),
            ("attributes", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


def _change_time(path: Path, stat: os.stat_result) -> int:
    """Use Windows change time, rather than its creation-time st_ctime."""
    if os.name != "nt":
        return stat.st_ctime_ns
    handle = _kernel32.CreateFileW(str(path), 0x80, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = _WindowsFileBasicInfo()
        if not _kernel32.GetFileInformationByHandleEx(
            handle, 0, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.change_time
    finally:
        _kernel32.CloseHandle(handle)


logger = logging.getLogger(__name__)
CACHE_VERSION = 1
MAX_ENTRY_BYTES = 256 * 1024 * 1024
MAX_DISK_BYTES = 512 * 1024 * 1024
MAX_ENTRIES = 24
ARRAY_CACHE_VERSION = 1
MAX_ARRAY_ENTRY_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ENTRIES = 4096
MAX_ARRAY_METADATA_BYTES = 4 * 1024 * 1024
MAX_ARRAYS_PER_ENTRY = 512


def _cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "orchav" / "sensing"


def _prepared_lease(root: Path, signature: str) -> sqlite3.Connection:
    """Hold a shared read lock on the existing preparation lock file.

    BEGIN IMMEDIATE preparation can coexist with readers. Explicit cleanup
    requires BEGIN EXCLUSIVE, so it cannot remove a bundle any reader uses.
    No tables or cache database are needed. Never unlink these lock files.
    """
    connection = sqlite3.connect(
        root / "recordings" / f"{signature[:20]}.lock", timeout=0, check_same_thread=False
    )
    try:
        connection.execute("BEGIN")
        connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
    except BaseException:
        connection.close()
        raise
    return connection


def clear_prepared_cache(root: Path | None = None) -> dict[str, int]:
    """Remove inactive complete bundles only, under their preparation locks.

    Active readers/builders and unrecognized or incomplete directories are
    skipped. Source recordings and the ordinary bounded cache are untouched.
    """
    recordings = (root or _cache_root()) / "recordings"
    result = {"removed_bundles": 0, "removed_bytes": 0, "skipped_bundles": 0}
    for bundle in recordings.glob("bundle_*"):
        connection = None
        try:
            if bundle.is_symlink() or not re.fullmatch(r"bundle_[0-9a-f]{16}", bundle.name):
                raise ValueError("Unrecognized bundle directory")
            metadata = json.loads((bundle / "complete.json").read_text(encoding="utf-8"))
            signature = metadata["signature"]
            if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
                raise ValueError("Invalid recording signature")
            connection = sqlite3.connect(recordings / f"{signature[:20]}.lock", timeout=0)
            connection.execute("BEGIN EXCLUSIVE")
            # Check the whole directory, not paths supplied by cache metadata.
            files = list(bundle.iterdir())
            if any(
                p.is_symlink()
                or not p.is_file()
                or (
                    p.name != "complete.json"
                    and not re.fullmatch(r"[0-9a-f]{64}\.(npz|json\.gz)", p.name)
                )
                for p in files
            ):
                raise ValueError("Unrecognized bundle contents")
            size = sum(p.stat().st_size for p in files)
            pointer = recordings / f"{signature[:20]}.ready"
            if pointer.is_file() and pointer.read_text(encoding="utf-8") == bundle.name:
                pointer.unlink()
            shutil.rmtree(bundle)
            result["removed_bundles"] += 1
            result["removed_bytes"] += size
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            result["skipped_bundles"] += 1
        finally:
            if connection is not None:
                connection.close()
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported display-cache value: {type(value).__name__}")


def _numeric_metadata_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Numeric cache metadata must contain JSON values; pass arrays separately")


def _temporary_file(root: Path):
    """Create an exclusive temporary file without retrying denied access on Windows."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(10):
        path = root / f"{uuid4().hex}.pending"
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        try:
            return path, os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
    raise FileExistsError("Could not allocate a sensing cache temporary file")


def prepared_cache_root(root: Path, signature: str) -> Path | None:
    """Resolve a published bundle without trusting paths from its pointer."""
    try:
        recordings = root / "recordings"
        name = (recordings / f"{signature[:20]}.ready").read_text(encoding="utf-8")
        if Path(name).name != name or not name.startswith("bundle_"):
            return None
        bundle = recordings / name
        marker = json.loads((bundle / "complete.json").read_text(encoding="utf-8"))
        return bundle if marker.get("signature") == signature else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _read_numeric_array(payload: bytes) -> np.ndarray:
    """Validate NPY dimensions before exposing an immutable, byte-backed view."""
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported numeric cache array version")
    if dtype.kind not in "biufc" or any(dimension < 0 for dimension in shape):
        raise ValueError("Numeric cache arrays require real, complex or boolean values")
    size = prod(shape) * dtype.itemsize
    offset = stream.tell()
    if size > MAX_ARRAY_ENTRY_BYTES or offset + size != len(payload):
        raise ValueError("Numeric cache array dimensions do not match its payload")
    return np.frombuffer(payload, dtype=dtype, offset=offset).reshape(
        shape, order="F" if fortran else "C"
    )


class RecordingCache:
    """Cache compact metadata and plot summaries for one immutable recording."""

    def __init__(self, paths: tuple[Path, ...], root: Path | None = None) -> None:
        # Keep the authored path for repeated stats so replacing a symlink is
        # observed, while resolving its identity spelling only once per source.
        self.paths = tuple(path.absolute() for path in paths)
        self._identity_paths = tuple(str(path.resolve()) for path in self.paths)
        self.root = root or _cache_root()
        self._baseline_stamps = None
        self._baseline_manifest = None
        self.signature = self._fingerprint()
        self.hits = 0
        self._frame_path = None
        self._scoped: dict[Path, RecordingCache] = {}
        self._scoped_lock = RLock()
        self._parent = None
        self._prepared_root = None
        self._building = False
        self._prepared_lease = None

    @classmethod
    def for_provider(cls, provider: Any) -> RecordingCache | None:
        """Leave live/remote/custom providers on their ordinary validated reader."""
        if not isinstance(provider, Hdf5Provider):
            return None
        try:
            build_root = getattr(provider, "_sensing_cache_build_root", None)
            cache = cls(provider.recording_paths(), root=build_root)
            cache._building = build_root is not None
            if cache._building:
                # Extending a figure-only bundle can reuse its validated
                # products while the builder's recording lock prevents cleanup.
                cache._prepared_root = prepared_cache_root(_cache_root(), cache.signature)
            if not cache._building and (cache.root / "recordings").is_dir():
                try:
                    cache._prepared_lease = _prepared_lease(cache.root, cache.signature)
                    cache._prepared_root = prepared_cache_root(cache.root, cache.signature)
                except (OSError, sqlite3.Error):
                    # Cleanup may own the lock; ordinary reads remain available.
                    cache._prepared_root = None
            cache._frame_path = provider.recording_frame_path
            return cache
        except (OSError, ValueError):
            return None

    def __del__(self):
        lease = getattr(self, "_prepared_lease", None)
        if lease is not None:
            lease.close()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(f"orchav-sensing-display:{CACHE_VERSION}".encode())
        stamps = []
        manifest = None
        for index, path in enumerate(self.paths):
            stat = path.stat()
            stamp = (
                self._identity_paths[index],
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                _change_time(path, stat),
            )
            stamps.append(stamp)
            digest.update(repr(stamp).encode())
            if index == 0:
                manifest = path.read_bytes()
                digest.update(manifest)
        if self._baseline_stamps is None:
            self._baseline_stamps = tuple(stamps)
            self._baseline_manifest = hashlib.sha256(manifest or b"").digest()
        return digest.hexdigest()

    def for_frame(self, frame_id: int | str) -> RecordingCache:
        """Scope a per-result model to its manifest and authoritative chunk.

        Recording-wide indexes and histories continue to validate every chunk.
        A scoped cache is accepted only when both of its initial source stamps
        match the parent recording's original snapshot.
        """
        if self._frame_path is None or not isinstance(frame_id, int) or isinstance(frame_id, bool):
            return self
        path = self._frame_path(frame_id)
        if path is None:
            return self
        path = path.absolute()
        with self._scoped_lock:
            cached = self._scoped.get(path)
            if cached is not None:
                return cached
            try:
                index = self.paths.index(path)
                scoped = RecordingCache((self.paths[0], path), root=self.root)
                if (
                    scoped._baseline_stamps
                    != (self._baseline_stamps[0], self._baseline_stamps[index])
                    or scoped._baseline_manifest != self._baseline_manifest
                ):
                    return self
            except (OSError, ValueError):
                return self
            scoped._parent = self
            scoped._building = self._building
            scoped._prepared_root = self._prepared_root
            self._scoped[path] = scoped
            return scoped

    def _record_hit(self) -> None:
        self.hits += 1
        if self._parent is not None:
            self._parent.hits += 1

    def unchanged(self) -> bool:
        try:
            return self.signature == self._fingerprint()
        except OSError:
            return False

    def _path(self, kind: str, key: tuple) -> Path:
        identity = hashlib.sha256(json.dumps((self.signature, kind, key)).encode()).hexdigest()
        return self.root / f"{identity}.json.gz"

    def _read_paths(self, path: Path) -> tuple[Path, ...]:
        """Prefer local entries, then reuse matching prepared or ordinary data."""
        if self._building:
            # A new bundle can reuse earlier on-demand work. Each reader still
            # checks the source identity and promotes the validated payload
            # into this bundle so publication never depends on the LRU cache.
            prepared = (self._prepared_root / path.name,) if self._prepared_root is not None else ()
            return (path, *prepared, _cache_root() / path.name)
        if self._prepared_root is not None:
            return path, self._prepared_root / path.name
        return (path,)

    def load(self, kind: str, key: tuple = ()) -> Any | None:
        """Return a derived payload only if its source still matches exactly."""
        try:
            if not self.unchanged():
                return None
            paths = self._read_paths(self._path(kind, key))
        except (OSError, ValueError, TypeError):
            return None
        for path in paths:
            try:
                with gzip.open(path, "rb") as stream:
                    payload = stream.read(MAX_ENTRY_BYTES + 1)
                if len(payload) > MAX_ENTRY_BYTES:
                    continue
                data = json.loads(payload)
                if data.get("signature") != self.signature or data.get("version") != CACHE_VERSION:
                    continue
                result = data["payload"]
                if not self.unchanged():
                    return None
                if self._building and path.parent != self.root and not self.save(kind, result, key):
                    return None
                self._record_hit()
                return result
            except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError):
                continue
        return None

    def save(self, kind: str, payload: Any, key: tuple = ()) -> bool:
        """Publish atomically; interrupted or source-changing jobs leave no valid entry."""
        temporary = None
        try:
            if not self.unchanged():
                return False
            data = json.dumps(
                {"version": CACHE_VERSION, "signature": self.signature, "payload": payload},
                default=_json_value,
                separators=(",", ":"),
            ).encode()
            if len(data) > MAX_ENTRY_BYTES:
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            temporary, stream = _temporary_file(self.root)
            with stream:
                with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1) as compressed:
                    compressed.write(data)
            if not self.unchanged():
                return False
            os.replace(temporary, self._path(kind, key))
            self._prune()
            return True
        except (OSError, ValueError, TypeError):
            logger.debug("Sensing display cache unavailable; using recording reads", exc_info=True)
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove interrupted display cache", exc_info=True)

    def _array_path(self, kind: str, key: tuple) -> Path:
        identity = hashlib.sha256(
            json.dumps((self.signature, ARRAY_CACHE_VERSION, kind, key)).encode()
        ).hexdigest()
        return self.root / f"{identity}.npz"

    def load_arrays(
        self,
        kind: str,
        key: tuple = (),
        *,
        accept: Callable[[dict, dict[str, np.ndarray]], bool] | None = None,
    ) -> tuple[dict, dict[str, np.ndarray]] | None:
        """Read prepared numeric display data without pickle or dense JSON decoding.

        The container and decoded byte sizes are bounded before array creation.
        Returned arrays share immutable byte buffers, including after reshaping.
        Missing, stale and damaged entries all fall back to the recording reader.
        An optional semantic predicate can reject a valid but insufficient
        candidate before promotion, allowing the next normal cache path to serve it.
        """
        try:
            if not self.unchanged():
                return None
            paths = self._read_paths(self._array_path(kind, key))
        except (OSError, ValueError, TypeError):
            return None
        for path in paths:
            try:
                if path.stat().st_size > MAX_ARRAY_ENTRY_BYTES:
                    continue
                with zipfile.ZipFile(path) as archive:
                    entries = archive.infolist()
                    if (
                        len(entries) > MAX_ARRAYS_PER_ENTRY + 1
                        or sum(entry.file_size for entry in entries) > MAX_ARRAY_ENTRY_BYTES
                        or any(entry.compress_type != zipfile.ZIP_STORED for entry in entries)
                        or any(entry.flag_bits & 1 for entry in entries)
                        or archive.getinfo("metadata.json").file_size > MAX_ARRAY_METADATA_BYTES
                    ):
                        continue
                    envelope = json.loads(archive.read("metadata.json"))
                    if (
                        envelope.get("signature") != self.signature
                        or envelope.get("version") != ARRAY_CACHE_VERSION
                        or envelope.get("identity") != path.stem
                        or not isinstance(envelope.get("metadata"), dict)
                    ):
                        continue
                    names = envelope["arrays"]
                    if (
                        not isinstance(names, list)
                        or not all(isinstance(name, str) for name in names)
                        or len(set(names)) != len(names)
                        or [entry.filename for entry in entries]
                        != ["metadata.json", *(f"array_{index}.npy" for index in range(len(names)))]
                    ):
                        continue
                    arrays = {
                        name: _read_numeric_array(archive.read(f"array_{index}.npy"))
                        for index, name in enumerate(names)
                    }
                if accept is not None and not accept(envelope["metadata"], arrays):
                    continue
                if not self.unchanged():
                    return None
                if (
                    self._building
                    and path.parent != self.root
                    and not self.save_arrays(kind, envelope["metadata"], arrays, key)
                ):
                    return None
                self._record_hit()
                return envelope["metadata"], arrays
            except (
                OSError,
                EOFError,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                OverflowError,
                zipfile.BadZipFile,
            ):
                continue
        return None

    def save_arrays(
        self,
        kind: str,
        metadata: dict,
        arrays: Mapping[str, np.ndarray],
        key: tuple = (),
    ) -> bool:
        """Atomically retain bounded plot arrays under the shared disk-cache budget."""
        temporary = None
        try:
            if (
                not self.unchanged()
                or not isinstance(metadata, dict)
                or len(arrays) > MAX_ARRAYS_PER_ENTRY
                or any(
                    not isinstance(name, str)
                    or not isinstance(array, np.ndarray)
                    or array.dtype.kind not in "biufc"
                    for name, array in arrays.items()
                )
                or sum(array.nbytes for array in arrays.values()) > MAX_ARRAY_ENTRY_BYTES
            ):
                return False
            path = self._array_path(kind, key)
            encoded = json.dumps(
                {
                    "signature": self.signature,
                    "version": ARRAY_CACHE_VERSION,
                    "identity": path.stem,
                    "metadata": metadata,
                    "arrays": list(arrays),
                },
                default=_numeric_metadata_value,
                separators=(",", ":"),
            ).encode()
            if len(encoded) > MAX_ARRAY_METADATA_BYTES:
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            temporary, stream = _temporary_file(self.root)
            with stream:
                with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr("metadata.json", encoded)
                    for index, array in enumerate(arrays.values()):
                        with archive.open(f"array_{index}.npy", "w") as member:
                            np.save(member, array, allow_pickle=False)
            if temporary.stat().st_size > MAX_ARRAY_ENTRY_BYTES or not self.unchanged():
                return False
            os.replace(temporary, path)
            self._prune()
            return True
        except (OSError, ValueError, TypeError, OverflowError):
            logger.debug("Prepared sensing cache unavailable; using recording reads", exc_info=True)
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove interrupted numeric cache", exc_info=True)

    def _prune(self) -> None:
        if self._building:
            return
        entries = sorted(
            (*self.root.glob("*.json.gz"), *self.root.glob("*.npz")),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        size = 0
        counts = {".gz": 0, ".npz": 0}
        for path in entries:
            count_limit = MAX_ARRAY_ENTRIES if path.suffix == ".npz" else MAX_ENTRIES
            counts[path.suffix] += 1
            file_size = path.stat().st_size
            if counts[path.suffix] > count_limit or size + file_size > MAX_DISK_BYTES:
                path.unlink(missing_ok=True)
            else:
                size += file_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove inactive prepared sensing bundles.")
    parser.add_argument("--clear-prepared", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(clear_prepared_cache(), indent=2))  # noqa: T201 - explicit CLI report
