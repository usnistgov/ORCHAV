"""Safe lifecycle owner for persistent and process-local pygfx mesh caches.

Logical keys are opaque. They are hashed into contained paths and augmented
with source-file revision data when a key directly names a file. This module
owns storage, validation, byte budgets, and telemetry; the backend adapter
only applies the validated seam-splitting plan to a neutral mesh payload.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from shared.logging import get_logger

from ..diagnostics.cache_telemetry import (
    record_cache_event,
    reset_cache_telemetry,
    set_cache_inventory,
)

logger = get_logger("orchav.pygfx_mesh_cache")

PYGFX_MESH_CACHE_MODE_SPLIT_ALL = np.uint8(1)
PYGFX_MESH_CACHE_MODE_REINDEX = np.uint8(2)

_DEFAULT_CACHE_ROOT = Path("~/.orchav/cache").expanduser()
_CACHE_SUBDIR = "pygfx_mesh_buffers"
_SCHEMA_VERSION = 3
_LAYOUT_SUBDIR = f"v{_SCHEMA_VERSION}"
_KEY_VERSION = 1
_DEFAULT_MAX_AGE_DAYS = 14
_DEFAULT_MEMORY_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_DISK_MAX_BYTES = 256 * 1024 * 1024
_ROOT_FAILURE_RETRY_SECONDS = 30.0
_MEMORY_NAMESPACE = "pygfx_mesh_buffers.memory"
_DISK_NAMESPACE = "pygfx_mesh_buffers.disk"
_LOCK = threading.RLock()
_MISSING = object()
_ROOTS: dict[tuple[str | None, str | None], Path] = {}
_FAILED_ROOT_RETRY_AT: dict[tuple[str | None, str | None], float] = {}
_WRITE_RETRY_AT: dict[str, float] = {}
_PRUNED_ROOTS: set[str] = set()
_TOUCHED_PATHS: set[str] = set()
_DISK_LRU: dict[str, OrderedDict[str, int]] = {}
_DISK_LRU_BYTES: dict[str, int] = {}


def _buffer_nbytes(buffers: Mapping[str, np.ndarray]) -> int:
    return sum(int(np.asarray(value).nbytes) for value in buffers.values())


def _copy_buffers(buffers: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for name, value in buffers.items():
        array = np.asarray(value)
        copied[name] = array.copy() if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)
    return copied


def _cache_owned_buffers(
    buffers: Mapping[str, np.ndarray],
    *,
    take_ownership: bool,
) -> dict[str, np.ndarray]:
    """Return read-only arrays owned by the cache after the call."""
    stored = dict(buffers) if take_ownership else _copy_buffers(buffers)
    for name, value in tuple(stored.items()):
        array = np.asarray(value)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
            stored[name] = array
        array.setflags(write=False)
    return stored


class _PreparedBufferLru(OrderedDict[str, dict[str, np.ndarray]]):
    """Ordered mapping with exact O(1) owned-byte accounting."""

    def __init__(self) -> None:
        super().__init__()
        self.nbytes = 0

    def __setitem__(self, key: str, value: dict[str, np.ndarray]) -> None:
        if key in self:
            self.nbytes -= _buffer_nbytes(super().pop(key))
        super().__setitem__(key, value)
        self.nbytes += _buffer_nbytes(value)

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = super().pop(key)
        self.nbytes -= _buffer_nbytes(value)
        return value

    def popitem(self, last: bool = True) -> tuple[str, dict[str, np.ndarray]]:
        key, value = super().popitem(last=last)
        self.nbytes -= _buffer_nbytes(value)
        return key, value

    def clear(self) -> None:
        super().clear()
        self.nbytes = 0


# Compatibility alias for focused tests that explicitly clear process state.
_PYGFX_MESH_BUFFER_MEMORY_CACHE = _PreparedBufferLru()
_EVENT_HOOK: Callable[[str, Mapping[str, Any]], None] | None = None
_METRICS: dict[str, int] = {
    "memory_hits": 0,
    "memory_misses": 0,
    "memory_stores": 0,
    "memory_invalidations": 0,
    "memory_evictions": 0,
    "memory_oversize_rejections": 0,
    "disk_hits": 0,
    "disk_misses": 0,
    "disk_invalidations": 0,
    "disk_read_failures": 0,
    "disk_writes": 0,
    "disk_write_failures": 0,
    "disk_pruned_files": 0,
    "disk_budget_evictions": 0,
    "root_failures": 0,
}


def _emit(
    event: str,
    *,
    metric: str | None = None,
    count: int = 1,
    **details: Any,
) -> None:
    with _LOCK:
        if metric is not None:
            _METRICS[metric] += int(count)
        hook = _EVENT_HOOK

    namespace = _MEMORY_NAMESPACE if event.startswith("memory_") else _DISK_NAMESPACE
    telemetry_event = event.split("_", 1)[1] if "_" in event else event
    try:
        record_cache_event(
            namespace,
            telemetry_event,
            count=count,
            elapsed_ms=(
                float(details["elapsed_ms"]) if details.get("elapsed_ms") is not None else None
            ),
            byte_count=(int(details["bytes"]) if details.get("bytes") is not None else None),
        )
        if event in {
            "memory_store",
            "memory_invalidation",
            "memory_eviction",
            "memory_clear",
        }:
            with _LOCK:
                set_cache_inventory(
                    namespace,
                    entries=len(_PYGFX_MESH_BUFFER_MEMORY_CACHE),
                    byte_count=_PYGFX_MESH_BUFFER_MEMORY_CACHE.nbytes,
                )
        elif details.get("root") is not None:
            set_cache_inventory(namespace, root=str(details["root"]))
    except Exception:
        logger.debug("Failed to publish pygfx mesh cache telemetry", exc_info=True)

    if hook is not None:
        try:
            hook(event, dict(details))
        except Exception:
            logger.debug("pygfx mesh cache event hook failed", exc_info=True)


def set_pygfx_mesh_buffer_cache_event_hook(
    hook: Callable[[str, Mapping[str, Any]], None] | None,
) -> Callable[[str, Mapping[str, Any]], None] | None:
    """Install an observational cache hook and return its predecessor."""
    if hook is not None and not callable(hook):
        raise TypeError("pygfx mesh cache event hook must be callable or None")
    global _EVENT_HOOK
    with _LOCK:
        previous = _EVENT_HOOK
        _EVENT_HOOK = hook
    return previous


def reset_pygfx_mesh_buffer_cache_metrics() -> None:
    """Reset local and renderer-neutral cache telemetry."""
    with _LOCK:
        for name in _METRICS:
            _METRICS[name] = 0
    reset_cache_telemetry(_MEMORY_NAMESPACE)
    reset_cache_telemetry(_DISK_NAMESPACE)


def _memory_max_bytes() -> int:
    raw_bytes = os.environ.get("ORCHAV_PYGFX_MESH_BUFFER_MEMORY_CACHE_MAX_BYTES")
    raw_megabytes = os.environ.get("ORCHAV_PYGFX_MESH_BUFFER_MEMORY_CACHE_MAX_MB")
    try:
        if raw_bytes is not None:
            return max(0, int(float(raw_bytes)))
        if raw_megabytes is not None:
            return max(0, int(float(raw_megabytes) * 1024 * 1024))
    except (OverflowError, TypeError, ValueError):
        pass
    return _DEFAULT_MEMORY_MAX_BYTES


def _max_age_days() -> int:
    raw = os.environ.get("ORCHAV_PYGFX_MESH_CACHE_MAX_AGE_DAYS")
    try:
        return _DEFAULT_MAX_AGE_DAYS if raw is None else max(0, int(float(raw)))
    except (OverflowError, TypeError, ValueError):
        return _DEFAULT_MAX_AGE_DAYS


def _disk_max_bytes() -> int:
    raw_bytes = os.environ.get("ORCHAV_PYGFX_MESH_CACHE_MAX_BYTES") or os.environ.get(
        "ORCHAV_PYGFX_MESH_BUFFER_DISK_CACHE_MAX_BYTES"
    )
    raw_megabytes = os.environ.get("ORCHAV_PYGFX_MESH_CACHE_MAX_MB") or os.environ.get(
        "ORCHAV_PYGFX_MESH_BUFFER_DISK_CACHE_MAX_MB"
    )
    try:
        if raw_bytes is not None:
            return max(0, int(float(raw_bytes)))
        if raw_megabytes is not None:
            return max(0, int(float(raw_megabytes) * 1024 * 1024))
    except (OverflowError, TypeError, ValueError):
        pass
    return _DEFAULT_DISK_MAX_BYTES


def _is_hex(value: str, *, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def _owned_shards(root: Path) -> Iterator[Path]:
    """Yield hash-shard directories from this cache's opaque layout."""
    try:
        shards = tuple(root.iterdir())
    except OSError:
        return
    for shard in shards:
        try:
            if shard.is_symlink() or not shard.is_dir() or not _is_hex(shard.name, length=2):
                continue
        except OSError:
            continue
        yield shard


def _owned_files(root: Path) -> Iterator[Path]:
    """Yield completed files from this cache's opaque hashed layout."""
    for shard in _owned_shards(root):
        try:
            children = tuple(shard.iterdir())
        except OSError:
            continue
        for path in children:
            try:
                is_file = path.is_file() and not path.is_symlink()
            except OSError:
                is_file = False
            if (
                is_file
                and path.suffix == ".npz"
                and _is_hex(path.stem, length=64)
                and path.stem.startswith(shard.name)
            ):
                yield path


def _owned_temporary_files(root: Path) -> Iterator[Path]:
    """Yield interrupted atomic-write files that are safe to remove."""
    for shard in _owned_shards(root):
        try:
            children = tuple(shard.iterdir())
        except OSError:
            continue
        for path in children:
            name = path.name
            identity = name[1:65] if name.startswith(".") else ""
            try:
                is_file = path.is_file() and not path.is_symlink()
            except OSError:
                is_file = False
            if (
                is_file
                and name.endswith(".tmp.npz")
                and _is_hex(identity, length=64)
                and identity.startswith(shard.name)
            ):
                yield path


def _prune_root(root: Path) -> None:
    root_key = str(root)
    with _LOCK:
        if root_key in _PRUNED_ROOTS:
            return
        _PRUNED_ROOTS.add(root_key)
    max_age_days = _max_age_days()
    cutoff = time.time() - (max_age_days * 24 * 60 * 60) if max_age_days > 0 else None
    age_removed = 0
    owned_directories: set[Path] = set()
    completed_paths = tuple(_owned_files(root))
    completed_path_set = set(completed_paths)
    temporary_paths = tuple(_owned_temporary_files(root))
    survivors: list[tuple[float, Path, int]] = []
    for path in (*completed_paths, *temporary_paths):
        owned_directories.add(path.parent)
        try:
            path_stat = path.stat()
            if cutoff is not None and path_stat.st_mtime < cutoff:
                path.unlink()
                age_removed += 1
                continue
            if path in completed_path_set:
                survivors.append((path_stat.st_mtime, path, int(path_stat.st_size)))
        except OSError:
            continue

    budget_removed = 0
    budget_removed_bytes = 0
    max_bytes = _disk_max_bytes()
    current_bytes = sum(size for _mtime, _path, size in survivors)
    survivors.sort(key=lambda item: item[0])
    retained: list[tuple[float, Path, int]] = []
    for item in survivors:
        _mtime, path, size = item
        if current_bytes > max_bytes:
            try:
                path.unlink()
            except OSError:
                retained.append(item)
                continue
            current_bytes -= size
            budget_removed += 1
            budget_removed_bytes += size
        else:
            retained.append(item)
    survivors = retained

    with _LOCK:
        _DISK_LRU[root_key] = OrderedDict((str(path), size) for _mtime, path, size in survivors)
        _DISK_LRU_BYTES[root_key] = current_bytes
    for directory in owned_directories:
        try:
            directory.rmdir()
        except OSError:
            continue
    if age_removed:
        _emit(
            "disk_prune",
            metric="disk_pruned_files",
            count=age_removed,
            files=age_removed,
            root=str(root),
        )
    if budget_removed:
        _emit(
            "disk_budget_eviction",
            metric="disk_budget_evictions",
            count=budget_removed,
            files=budget_removed,
            bytes=budget_removed_bytes,
            root=str(root),
        )


def _record_disk_hit(path: Path) -> None:
    """Move one valid file to the newest position in the process disk LRU."""
    root_key = str(path.parent.parent)
    path_key = str(path)
    with _LOCK:
        lru = _DISK_LRU.get(root_key)
        if lru is not None and path_key in lru:
            lru.move_to_end(path_key)


def _record_disk_write(path: Path, size: int) -> None:
    """Register a write and enforce the configured disk-byte budget."""
    root_key = str(path.parent.parent)
    path_key = str(path)
    max_bytes = _disk_max_bytes()
    evicted_files = 0
    evicted_bytes = 0
    with _LOCK:
        lru = _DISK_LRU.setdefault(root_key, OrderedDict())
        current_bytes = _DISK_LRU_BYTES.get(root_key, 0)
        previous_size = lru.pop(path_key, None)
        if previous_size is not None:
            current_bytes -= previous_size
        lru[path_key] = size
        current_bytes += size
        while lru and current_bytes > max_bytes:
            stale_key, stale_size = lru.popitem(last=False)
            try:
                Path(stale_key).unlink(missing_ok=True)
            except OSError:
                # Keep failed removals accounted so the next write retries.
                lru[stale_key] = stale_size
                lru.move_to_end(stale_key, last=False)
                break
            current_bytes -= stale_size
            evicted_files += 1
            evicted_bytes += stale_size
        _DISK_LRU_BYTES[root_key] = current_bytes
    if evicted_files:
        _emit(
            "disk_budget_eviction",
            metric="disk_budget_evictions",
            count=evicted_files,
            files=evicted_files,
            bytes=evicted_bytes,
            root=root_key,
        )


def _cache_root() -> Path | None:
    if os.environ.get("ORCHAV_DISABLE_PYGFX_MESH_BUFFER_CACHE") == "1":
        return None
    configured = os.environ.get("ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR")
    base_root = os.environ.get("ORCHAV_CACHE_DIR")
    key = (configured, base_root)
    with _LOCK:
        cached = _ROOTS.get(key)
        if cached is not None:
            _prune_root(cached)
            return cached
        retry_at = _FAILED_ROOT_RETRY_AT.get(key)
        if retry_at is not None and retry_at > time.monotonic():
            return None
        _FAILED_ROOT_RETRY_AT.pop(key, None)

    root = _DEFAULT_CACHE_ROOT / _CACHE_SUBDIR
    try:
        if configured:
            root = Path(configured).expanduser()
        elif base_root:
            root = Path(base_root).expanduser() / _CACHE_SUBDIR
        root /= _LAYOUT_SUBDIR
        root = root.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError):
        _emit(
            "root_failure",
            metric="root_failures",
            root=str(configured or base_root or root),
        )
        with _LOCK:
            _FAILED_ROOT_RETRY_AT[key] = time.monotonic() + _ROOT_FAILURE_RETRY_SECONDS
        logger.debug("Failed to prepare pygfx mesh cache root %s", root, exc_info=True)
        return None
    _prune_root(root)
    with _LOCK:
        _ROOTS[key] = root
        _FAILED_ROOT_RETRY_AT.pop(key, None)
    return root


def get_pygfx_mesh_buffer_cache_root() -> Path | None:
    """Return the persistent cache root, if enabled and available."""
    return _cache_root()


def _source_version(cache_key: str) -> tuple[str, int, int, int] | None:
    try:
        source = Path(cache_key).expanduser()
        if not source.is_absolute() and not source.suffix:
            return None
        source_stat = source.stat()
        if not stat.S_ISREG(source_stat.st_mode):
            return None
        return (
            str(source.resolve(strict=False)),
            int(source_stat.st_size),
            int(source_stat.st_mtime_ns),
            int(source_stat.st_ctime_ns),
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def pygfx_mesh_cache_identity(cache_key: str) -> str:
    """Return a source/version-aware SHA-256 identity for an opaque key."""
    parts = [
        f"key-version={_KEY_VERSION}",
        f"schema-version={_SCHEMA_VERSION}",
        f"key={cache_key}",
    ]
    source_version = _source_version(cache_key)
    if source_version is not None:
        path, size, mtime_ns, ctime_ns = source_version
        parts.extend(
            (
                f"source={path}",
                f"source-size={size}",
                f"source-mtime-ns={mtime_ns}",
                f"source-ctime-ns={ctime_ns}",
            )
        )
    material = "\0".join(parts).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(material).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def resolve_pygfx_mesh_buffer_cache_path(
    cache_key: str,
    *,
    cache_identity: str | None = None,
) -> Path | None:
    """Return a contained opaque path for a logical cache key."""
    if not cache_key:
        return None
    root = _cache_root()
    if root is None:
        return None
    identity = cache_identity or pygfx_mesh_cache_identity(cache_key)
    if not _is_hex(identity, length=64):
        return None
    path = root / identity[:2] / f"{identity}.npz"
    if not _is_within(path, root):
        _emit(
            "root_failure",
            metric="root_failures",
            root=str(root),
            candidate=str(path),
        )
        return None
    return path


def load_prepared_pygfx_mesh_buffers(memory_key: str) -> dict[str, np.ndarray] | None:
    """Return a detached prepared-buffer memory-cache hit."""
    with _LOCK:
        cached = _PYGFX_MESH_BUFFER_MEMORY_CACHE.get(memory_key)
        if cached is not None:
            _PYGFX_MESH_BUFFER_MEMORY_CACHE.move_to_end(memory_key)
    if cached is None:
        _emit("memory_miss", metric="memory_misses", key=memory_key)
        return None
    _emit("memory_hit", metric="memory_hits", key=memory_key)
    return _copy_buffers(cached)


def invalidate_prepared_pygfx_mesh_buffers(memory_key: str) -> None:
    """Discard one prepared-buffer entry after failed payload validation."""
    with _LOCK:
        removed = _PYGFX_MESH_BUFFER_MEMORY_CACHE.pop(memory_key, None)
    if removed is not None:
        _emit(
            "memory_invalidation",
            metric="memory_invalidations",
            key=memory_key,
        )


def store_prepared_pygfx_mesh_buffers(
    memory_key: str,
    buffers: Mapping[str, np.ndarray],
    *,
    take_ownership: bool = False,
) -> None:
    """Store prepared buffers under a byte-budgeted true LRU policy.

    ``take_ownership`` removes one cold-path copy. The caller must never expose
    the transferred arrays to pygfx; cache hits remain detached writable copies
    because backend buffer objects may mutate their numpy inputs.
    """
    entry_bytes = _buffer_nbytes(buffers)
    max_bytes = _memory_max_bytes()
    if max_bytes <= 0 or entry_bytes > max_bytes:
        _emit(
            "memory_oversize_rejection",
            metric="memory_oversize_rejections",
            key=memory_key,
            bytes=entry_bytes,
            max_bytes=max_bytes,
        )
        return

    stored = _cache_owned_buffers(buffers, take_ownership=take_ownership)
    evicted_entries = 0
    evicted_bytes = 0
    with _LOCK:
        _PYGFX_MESH_BUFFER_MEMORY_CACHE[memory_key] = stored
        while _PYGFX_MESH_BUFFER_MEMORY_CACHE.nbytes > max_bytes:
            _, evicted = _PYGFX_MESH_BUFFER_MEMORY_CACHE.popitem(last=False)
            evicted_entries += 1
            evicted_bytes += _buffer_nbytes(evicted)
    _emit(
        "memory_store",
        metric="memory_stores",
        key=memory_key,
        bytes=entry_bytes,
    )
    if evicted_entries:
        _emit(
            "memory_eviction",
            metric="memory_evictions",
            count=evicted_entries,
            entries=evicted_entries,
            bytes=evicted_bytes,
        )


class _InvalidCache(ValueError):
    pass


def _scalar(data: Any, key: str) -> Any:
    values = np.asarray(data[key])
    if values.size != 1:
        raise _InvalidCache(f"{key} is not scalar")
    return values.reshape(-1)[0].item()


def _discard_invalid(path: Path, *, reason: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Failed to remove invalid pygfx mesh cache %s", path)
    with _LOCK:
        _TOUCHED_PATHS.discard(str(path))
    _emit(
        "disk_invalidation",
        metric="disk_invalidations",
        path=str(path),
        reason=reason,
    )


def _touch_on_hit(path: Path) -> None:
    key = str(path)
    with _LOCK:
        if key in _TOUCHED_PATHS:
            _record_disk_hit(path)
            return
    try:
        path.touch(exist_ok=True)
    except OSError:
        return
    with _LOCK:
        _TOUCHED_PATHS.add(key)
    _record_disk_hit(path)


def load_pygfx_mesh_cache_plan(
    cache_key: str,
    *,
    cache_identity: str,
    vertex_count: int,
    triangle_count: int,
    uv_count: int,
) -> dict[str, Any] | None:
    """Load and validate a persisted seam-splitting plan."""
    started = time.perf_counter()
    path = resolve_pygfx_mesh_buffer_cache_path(
        cache_key,
        cache_identity=cache_identity,
    )
    if path is None:
        return None
    try:
        exists = path.is_file() and not path.is_symlink()
    except OSError:
        exists = False
    if not exists:
        _emit(
            "disk_miss",
            metric="disk_misses",
            path=str(path),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            if int(_scalar(data, "schema_version")) != _SCHEMA_VERSION:
                raise _InvalidCache("schema version mismatch")
            if str(_scalar(data, "cache_identity")) != cache_identity:
                raise _InvalidCache("identity mismatch")
            for name, expected in (
                ("vertex_count", vertex_count),
                ("triangle_count", triangle_count),
                ("uv_count", uv_count),
            ):
                if int(_scalar(data, name)) != int(expected):
                    raise _InvalidCache(f"{name} changed")

            mode = np.uint8(_scalar(data, "mode"))
            plan: dict[str, Any] = {"mode": mode}
            if mode == PYGFX_MESH_CACHE_MODE_REINDEX:
                unique = np.asarray(data["unique_corner_indices"])
                indices = np.asarray(data["indices"])
                if (
                    unique.ndim != 1
                    or unique.dtype.kind not in "iu"
                    or indices.shape != (triangle_count, 3)
                    or indices.dtype.kind not in "iu"
                ):
                    raise _InvalidCache("invalid reindex arrays")
                unique = np.ascontiguousarray(unique, dtype=np.int32)
                indices = np.ascontiguousarray(indices, dtype=np.int32)
                if unique.size and (int(unique.min()) < 0 or int(unique.max()) >= uv_count):
                    raise _InvalidCache("corner index out of range")
                if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(unique)):
                    raise _InvalidCache("mesh index out of range")
                plan["unique_corner_indices"] = unique
                plan["indices"] = indices
            elif mode != PYGFX_MESH_CACHE_MODE_SPLIT_ALL:
                raise _InvalidCache("unknown mode")

        _touch_on_hit(path)
        _emit(
            "disk_hit",
            metric="disk_hits",
            path=str(path),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return plan
    except FileNotFoundError:
        _emit("disk_miss", metric="disk_misses", path=str(path))
    except _InvalidCache as exc:
        _discard_invalid(path, reason=str(exc))
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        _discard_invalid(path, reason=type(exc).__name__)
    except OSError:
        _emit(
            "disk_read_failure",
            metric="disk_read_failures",
            path=str(path),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        logger.debug("Failed to read pygfx mesh cache %s", path, exc_info=True)
    return None


def store_pygfx_mesh_cache_plan(
    cache_key: str,
    *,
    cache_identity: str,
    vertex_count: int,
    triangle_count: int,
    uv_count: int,
    plan: Mapping[str, Any],
) -> None:
    """Atomically persist one seam-splitting plan."""
    path = resolve_pygfx_mesh_buffer_cache_path(
        cache_key,
        cache_identity=cache_identity,
    )
    if path is None:
        return
    root_key = str(path.parent.parent)
    with _LOCK:
        retry_at = _WRITE_RETRY_AT.get(root_key)
        if retry_at is not None and retry_at > time.monotonic():
            return
        _WRITE_RETRY_AT.pop(root_key, None)
    save: dict[str, np.ndarray] = {
        "schema_version": np.asarray([_SCHEMA_VERSION], dtype=np.uint8),
        "cache_identity": np.asarray([cache_identity], dtype="<U64"),
        "mode": np.asarray([plan["mode"]], dtype=np.uint8),
        "vertex_count": np.asarray([vertex_count], dtype=np.int64),
        "triangle_count": np.asarray([triangle_count], dtype=np.int64),
        "uv_count": np.asarray([uv_count], dtype=np.int64),
    }
    for name in ("indices", "unique_corner_indices"):
        if plan.get(name) is not None:
            save[name] = np.ascontiguousarray(plan[name], dtype=np.int32)

    started = time.perf_counter()
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink() or not _is_within(path, path.parent.parent):
            raise OSError("pygfx mesh cache shard escaped its root")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{cache_identity}.",
            suffix=".tmp.npz",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez(temporary, **save)
        os.replace(temporary, path)
        temporary = None
        with _LOCK:
            _WRITE_RETRY_AT.pop(root_key, None)
        try:
            written_bytes = int(path.stat().st_size)
        except OSError:
            written_bytes = 0
        _record_disk_write(path, written_bytes)
        _emit(
            "disk_write",
            metric="disk_writes",
            path=str(path),
            root=str(path.parent.parent),
            bytes=written_bytes,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
    except (OSError, ValueError):
        with _LOCK:
            _WRITE_RETRY_AT[root_key] = time.monotonic() + _ROOT_FAILURE_RETRY_SECONDS
        _emit(
            "disk_write_failure",
            metric="disk_write_failures",
            path=str(path),
            root=str(path.parent.parent),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        logger.debug("Failed to store pygfx mesh cache %s", path, exc_info=True)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def get_pygfx_mesh_buffer_cache_info() -> dict[str, Any]:
    """Return location, budgets, occupancy, and local counters."""
    root = _cache_root()
    disk_files = 0
    disk_bytes = 0
    if root is not None:
        for path in _owned_files(root):
            try:
                disk_files += 1
                disk_bytes += int(path.stat().st_size)
            except OSError:
                continue
    with _LOCK:
        memory_entries = len(_PYGFX_MESH_BUFFER_MEMORY_CACHE)
        memory_bytes = _PYGFX_MESH_BUFFER_MEMORY_CACHE.nbytes
        metrics = dict(_METRICS)
    set_cache_inventory(
        _MEMORY_NAMESPACE,
        entries=memory_entries,
        byte_count=memory_bytes,
    )
    set_cache_inventory(
        _DISK_NAMESPACE,
        entries=disk_files,
        byte_count=disk_bytes,
        root=str(root) if root is not None else None,
    )
    return {
        "enabled": root is not None,
        "root": str(root) if root is not None else None,
        "memory": {
            "entries": memory_entries,
            "bytes": memory_bytes,
            "max_bytes": _memory_max_bytes(),
        },
        "disk": {
            "files": disk_files,
            "bytes": disk_bytes,
            "max_bytes": _disk_max_bytes(),
            "max_age_days": _max_age_days(),
        },
        "metrics": metrics,
    }


def clear_pygfx_mesh_buffer_cache(
    *,
    memory: bool = True,
    disk: bool = False,
    reset_metrics: bool = False,
) -> dict[str, int]:
    """Clear selected layers; disk deletion is safe-layout-only."""
    memory_entries = 0
    memory_bytes = 0
    disk_files = 0
    disk_bytes = 0
    if memory:
        with _LOCK:
            memory_entries = len(_PYGFX_MESH_BUFFER_MEMORY_CACHE)
            memory_bytes = _PYGFX_MESH_BUFFER_MEMORY_CACHE.nbytes
            _PYGFX_MESH_BUFFER_MEMORY_CACHE.clear()
        _emit("memory_clear", entries=memory_entries, bytes=memory_bytes)

    if disk:
        root = _cache_root()
        if root is not None:
            with _LOCK:
                _WRITE_RETRY_AT.pop(str(root), None)
            owned_paths = (
                *tuple(_owned_files(root)),
                *tuple(_owned_temporary_files(root)),
            )
            owned_directories = {path.parent for path in owned_paths}
            for path in owned_paths:
                try:
                    size = int(path.stat().st_size)
                    path.unlink()
                    disk_files += 1
                    disk_bytes += size
                    with _LOCK:
                        _TOUCHED_PATHS.discard(str(path))
                except OSError:
                    continue
            for directory in owned_directories:
                try:
                    directory.rmdir()
                except OSError:
                    continue
            with _LOCK:
                _DISK_LRU.pop(str(root), None)
                _DISK_LRU_BYTES.pop(str(root), None)
        _emit("disk_clear", files=disk_files, bytes=disk_bytes)

    if reset_metrics:
        reset_pygfx_mesh_buffer_cache_metrics()
    get_pygfx_mesh_buffer_cache_info()
    return {
        "memory_entries": memory_entries,
        "memory_bytes": memory_bytes,
        "disk_files": disk_files,
        "disk_bytes": disk_bytes,
    }
