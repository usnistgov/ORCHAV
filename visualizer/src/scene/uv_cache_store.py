"""Scene-level persistent store for generated architectural UV arrays."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shared.logging import get_logger

from ..diagnostics.cache_telemetry import record_cache_event, set_cache_inventory

logger = get_logger("orchav.scene.uv_cache")

_SCHEMA_VERSION = 1
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_LOCK = threading.RLock()
_STORES: dict[str, "UVCacheStore"] = {}


def _store_byte_budget() -> int:
    """Return the maximum logical BLOB bytes retained by one scenario store."""
    raw = os.environ.get("ORCHAV_UV_CACHE_STORE_MAX_BYTES")
    try:
        return max(0, int(float(raw))) if raw is not None else _DEFAULT_MAX_BYTES
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_MAX_BYTES


@dataclass(slots=True)
class UVCacheStore:
    """One SQLite-backed UV store for a scenario/source namespace."""

    path: Path
    _connection: sqlite3.Connection | None = None
    _pending: dict[str, tuple[np.ndarray, str]] = field(default_factory=dict)

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is not None and int(row[0]) != _SCHEMA_VERSION:
            connection.execute("DROP TABLE IF EXISTS uv_entries")
            connection.execute("DELETE FROM metadata")
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        connection.execute("""
            CREATE TABLE IF NOT EXISTS uv_entries (
                cache_key TEXT PRIMARY KEY,
                rows INTEGER NOT NULL,
                columns_count INTEGER NOT NULL,
                source_signature TEXT NOT NULL,
                data BLOB NOT NULL
            )
            """)
        connection.commit()
        self._connection = connection
        try:
            os.utime(self.path, None)
        except OSError:
            pass
        return connection

    def contains(self, cache_key: str) -> bool:
        """Return whether an entry exists without decoding its array."""
        with _LOCK:
            if cache_key in self._pending:
                return True
            start = time.perf_counter()
            try:
                row = (
                    self._connect()
                    .execute("SELECT 1 FROM uv_entries WHERE cache_key = ?", (cache_key,))
                    .fetchone()
                )
            except sqlite3.Error:
                record_cache_event("uv", "contains_failure")
                return False
            record_cache_event(
                "uv",
                "contains",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
            return row is not None

    def get(
        self,
        cache_key: str,
        *,
        expected_shape: tuple[int, int],
        source_signature: str,
    ) -> np.ndarray | None:
        """Load and validate one generated UV array."""
        with _LOCK:
            pending = self._pending.get(cache_key)
            if pending is not None:
                array, pending_signature = pending
                if pending_signature == source_signature and array.shape == expected_shape:
                    record_cache_event("uv", "memory_hit", byte_count=int(array.nbytes))
                    return array

            start = time.perf_counter()
            try:
                row = (
                    self._connect()
                    .execute(
                        """
                    SELECT rows, columns_count, source_signature, data
                    FROM uv_entries WHERE cache_key = ?
                    """,
                        (cache_key,),
                    )
                    .fetchone()
                )
            except sqlite3.Error:
                record_cache_event("uv", "read_failure")
                return None
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if row is None:
                record_cache_event("uv", "miss", elapsed_ms=elapsed_ms)
                return None
            rows, columns_count, stored_signature, raw = row
            shape = (int(rows), int(columns_count))
            if stored_signature != source_signature or shape != expected_shape:
                try:
                    self._connect().execute(
                        "DELETE FROM uv_entries WHERE cache_key = ?", (cache_key,)
                    )
                    self._connect().commit()
                except sqlite3.Error:
                    pass
                record_cache_event("uv", "invalid")
                return None
            array = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            record_cache_event(
                "uv",
                "hit",
                elapsed_ms=elapsed_ms,
                byte_count=len(raw),
            )
            # sqlite returns BLOBs as immutable ``bytes``. Keep that backing so
            # MeshPayload can reuse the float32 snapshot without another copy.
            return array

    def put(self, cache_key: str, uvs: np.ndarray, *, source_signature: str) -> None:
        """Stage one UV array; :func:`flush_uv_cache_stores` commits in one transaction."""
        contiguous = np.ascontiguousarray(uvs, dtype=np.float32)
        storage = contiguous.tobytes(order="C")
        array = np.frombuffer(storage, dtype=np.float32).reshape(contiguous.shape)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("Generated UV arrays must have shape (N, 2)")
        with _LOCK:
            self._pending[cache_key] = (array, source_signature)
        record_cache_event("uv", "staged", byte_count=int(array.nbytes))

    def flush(self) -> int:
        """Commit every pending entry in one SQLite transaction."""
        with _LOCK:
            if not self._pending:
                return 0
            pending = self._pending
            self._pending = {}
            rows = [
                (
                    cache_key,
                    int(array.shape[0]),
                    int(array.shape[1]),
                    signature,
                    sqlite3.Binary(array.tobytes(order="C")),
                )
                for cache_key, (array, signature) in pending.items()
            ]
            start = time.perf_counter()
            try:
                connection = self._connect()
                with connection:
                    connection.executemany(
                        """
                        INSERT OR REPLACE INTO uv_entries(
                            cache_key, rows, columns_count, source_signature, data
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    self._prune_to_budget(connection)
            except sqlite3.Error:
                self._pending.update(pending)
                record_cache_event("uv", "write_failure", count=len(rows))
                return 0
            record_cache_event(
                "uv",
                "write",
                count=len(rows),
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                byte_count=sum(int(array.nbytes) for array, _signature in pending.values()),
            )
            return len(rows)

    @staticmethod
    def _prune_to_budget(connection: sqlite3.Connection) -> None:
        """Evict oldest generated arrays when one scenario exceeds its budget."""
        budget = _store_byte_budget()
        if budget <= 0:
            return
        row = connection.execute("SELECT COALESCE(SUM(length(data)), 0) FROM uv_entries").fetchone()
        total_bytes = int(row[0]) if row is not None else 0
        if total_bytes <= budget:
            return
        removed = 0
        removed_bytes = 0
        oldest = connection.execute(
            "SELECT cache_key, length(data) FROM uv_entries ORDER BY rowid"
        ).fetchall()
        for cache_key, byte_count in oldest:
            if total_bytes <= budget:
                break
            connection.execute(
                "DELETE FROM uv_entries WHERE cache_key = ?",
                (cache_key,),
            )
            size = int(byte_count or 0)
            total_bytes -= size
            removed += 1
            removed_bytes += size
        if removed:
            record_cache_event(
                "uv",
                "eviction",
                count=removed,
                byte_count=removed_bytes,
            )

    def inventory(self) -> tuple[int, int]:
        """Return entry and on-disk byte counts."""
        with _LOCK:
            try:
                count = int(
                    self._connect().execute("SELECT COUNT(*) FROM uv_entries").fetchone()[0]
                )
            except (sqlite3.Error, TypeError, ValueError):
                count = 0
            try:
                byte_count = int(self.path.stat().st_size)
            except OSError:
                byte_count = 0
            return count + len(self._pending), byte_count

    def close(self, *, flush: bool = True) -> None:
        """Close the namespace connection, optionally discarding staged writes."""
        with _LOCK:
            if flush:
                self.flush()
            else:
                self._pending.clear()
            if self._connection is not None:
                self._connection.close()
                self._connection = None


def get_uv_cache_store(root: Path, namespace: str) -> UVCacheStore:
    """Return the process-local store for one safe namespace."""
    path = (root / f"{namespace}.uv.sqlite3").resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"UV cache namespace escaped its root: {namespace}")
    key = str(path)
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = UVCacheStore(path)
            _STORES[key] = store
        return store


def flush_uv_cache_stores() -> int:
    """Flush all pending generated UVs and publish aggregate inventory."""
    with _LOCK:
        stores = tuple(_STORES.values())
    written = sum(store.flush() for store in stores)
    entries = 0
    byte_count = 0
    for store in stores:
        store_entries, store_bytes = store.inventory()
        entries += store_entries
        byte_count += store_bytes
    set_cache_inventory("uv", entries=entries, byte_count=byte_count)
    return written


def close_uv_cache_stores(*, root: Path | None = None, flush: bool = True) -> None:
    """Close selected process-local stores, flushing unless explicitly clearing."""
    resolved_root = root.expanduser().resolve(strict=False) if root is not None else None
    with _LOCK:
        selected_keys = tuple(
            key
            for key in _STORES
            if resolved_root is None or Path(key).is_relative_to(resolved_root)
        )
        stores = tuple(_STORES.pop(key) for key in selected_keys)
    for store in stores:
        store.close(flush=flush)


def get_uv_cache_info(root: Path) -> dict[str, int | str]:
    """Return aggregate disk and staged-memory inventory under *root*.

    Inspection does not create stores or databases.  Active stores are read
    through their existing connection; inactive databases are opened
    read-only so the Performance panel cannot mutate cache lifecycle merely by
    polling telemetry.
    """
    resolved_root = root.expanduser().resolve(strict=False)
    with _LOCK:
        active_stores = {
            Path(key): store
            for key, store in _STORES.items()
            if Path(key).is_relative_to(resolved_root)
        }
        pending_entries = sum(len(store._pending) for store in active_stores.values())
        pending_bytes = sum(
            int(array.nbytes)
            for store in active_stores.values()
            for array, _signature in store._pending.values()
        )

        stored_entries = 0
        data_bytes = 0
        database_paths = tuple(resolved_root.glob("*.uv.sqlite3"))
        for path in database_paths:
            connection: sqlite3.Connection | None = None
            close_connection = False
            try:
                active = active_stores.get(path.resolve(strict=False))
                if active is not None and active._connection is not None:
                    connection = active._connection
                else:
                    connection = sqlite3.connect(
                        f"{path.resolve(strict=False).as_uri()}?mode=ro",
                        uri=True,
                        timeout=1.0,
                    )
                    close_connection = True
                row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(length(data)), 0) FROM uv_entries"
                ).fetchone()
                if row is not None:
                    stored_entries += int(row[0] or 0)
                    data_bytes += int(row[1] or 0)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                record_cache_event("uv", "inventory_failure")
            finally:
                if close_connection and connection is not None:
                    connection.close()

    files = 0
    disk_bytes = 0
    for pattern in ("*.uv.sqlite3", "*.uv.sqlite3-wal", "*.uv.sqlite3-shm", "*.npy"):
        for path in resolved_root.rglob(pattern):
            try:
                disk_bytes += int(path.stat().st_size)
                files += 1
            except OSError:
                continue

    entries = stored_entries + pending_entries
    set_cache_inventory(
        "uv",
        entries=entries,
        byte_count=disk_bytes,
        root=str(resolved_root),
    )
    return {
        "stores": len(database_paths),
        "files": files,
        "entries": entries,
        "stored_entries": stored_entries,
        "pending_entries": pending_entries,
        "bytes": disk_bytes,
        "data_bytes": data_bytes,
        "pending_bytes": pending_bytes,
        "max_bytes_per_store": _store_byte_budget(),
        "root": str(resolved_root),
    }


def clear_uv_cache(root: Path) -> dict[str, int]:
    """Remove generated UV stores under *root* and return deletion counts."""
    # Explicit deletion must not spend time committing staged data that will
    # immediately be removed, and must not close stores under another root.
    close_uv_cache_stores(root=root, flush=False)
    removed_files = 0
    removed_bytes = 0
    for pattern in (
        "*.uv.sqlite3",
        "*.uv.sqlite3-wal",
        "*.uv.sqlite3-shm",
        "*.npy",
        "*.npz",
    ):
        for path in root.rglob(pattern):
            try:
                size = int(path.stat().st_size)
                path.unlink()
                removed_files += 1
                removed_bytes += size
            except OSError:
                continue
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            continue
    record_cache_event("uv", "clear", count=removed_files, byte_count=removed_bytes)
    set_cache_inventory("uv", entries=0, byte_count=0, root=str(root))
    return {"files": removed_files, "bytes": removed_bytes}


def prune_uv_cache(root: Path, *, max_age_days: int, max_bytes: int | None = None) -> int:
    """Prune inactive namespace stores by age and a root byte budget."""
    budget = _DEFAULT_MAX_BYTES if max_bytes is None else max(0, int(max_bytes))
    cutoff = time.time() - max(0, int(max_age_days)) * 24 * 60 * 60
    active_paths = {Path(key) for key in _STORES}
    candidates: list[tuple[float, int, Path]] = []
    for path in root.glob("*.uv.sqlite3"):
        if path.resolve(strict=False) in active_paths:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        candidates.append((float(stat.st_mtime), int(stat.st_size), path))

    removed = 0
    total_bytes = sum(size for _mtime, size, _path in candidates)
    for mtime, size, path in sorted(candidates):
        expired = max_age_days > 0 and mtime < cutoff
        over_budget = budget > 0 and total_bytes > budget
        if not expired and not over_budget:
            continue
        try:
            path.unlink()
            total_bytes -= size
            removed += 1
        except OSError:
            continue
    if removed:
        record_cache_event("uv", "pruned", count=removed)
    set_cache_inventory("uv", byte_count=max(0, total_bytes), root=str(root))
    return removed
