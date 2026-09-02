"""Process-local telemetry for persistent and native asset caches.

Cache implementations report through this renderer-neutral registry so the
performance panel and diagnostics tooling can compare stages without learning
their storage details.  Telemetry is observational and never owns cache data.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(slots=True)
class _CacheNamespaceMetrics:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    elapsed_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    current_entries: int | None = None
    current_bytes: int | None = None
    root: str | None = None


_LOCK = threading.RLock()
_METRICS: dict[str, _CacheNamespaceMetrics] = {}


def record_cache_event(
    namespace: str,
    event: str,
    *,
    count: int = 1,
    elapsed_ms: float | None = None,
    byte_count: int | None = None,
) -> None:
    """Record one cache event and its optional duration/byte volume."""
    with _LOCK:
        metrics = _METRICS.setdefault(str(namespace), _CacheNamespaceMetrics())
        metrics.counters[str(event)] += int(count)
        if elapsed_ms is not None:
            metrics.elapsed_ms[str(event)] += max(0.0, float(elapsed_ms))
        if byte_count is not None:
            metrics.counters[f"{event}_bytes"] += max(0, int(byte_count))


def set_cache_inventory(
    namespace: str,
    *,
    entries: int | None = None,
    byte_count: int | None = None,
    root: str | None = None,
) -> None:
    """Publish the current inventory for a cache namespace."""
    with _LOCK:
        metrics = _METRICS.setdefault(str(namespace), _CacheNamespaceMetrics())
        if entries is not None:
            metrics.current_entries = max(0, int(entries))
        if byte_count is not None:
            metrics.current_bytes = max(0, int(byte_count))
        if root is not None:
            metrics.root = str(root)


@contextmanager
def measure_cache_operation(namespace: str, event: str) -> Iterator[None]:
    """Measure one cache operation and record its elapsed time."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        record_cache_event(namespace, f"{event}_failure")
        raise
    finally:
        record_cache_event(
            namespace,
            event,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )


def cache_telemetry_snapshot() -> dict[str, dict[str, Any]]:
    """Return a detached snapshot suitable for UI and benchmark reporting."""
    with _LOCK:
        return {
            namespace: {
                "counters": dict(metrics.counters),
                "elapsed_ms": dict(metrics.elapsed_ms),
                "current_entries": metrics.current_entries,
                "current_bytes": metrics.current_bytes,
                "root": metrics.root,
            }
            for namespace, metrics in _METRICS.items()
        }


def reset_cache_telemetry(namespace: str | None = None) -> None:
    """Reset one namespace or all process-local cache telemetry."""
    with _LOCK:
        if namespace is None:
            _METRICS.clear()
        else:
            _METRICS.pop(str(namespace), None)
