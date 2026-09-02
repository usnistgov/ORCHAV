"""In-memory cache for raw live generator frames.

The cache stores the dictionaries returned by live generation or dispatcher
bridges before they are normalized into ``StandardMPCFrame`` protobuf payloads.
Admission counts known retained CPU buffers after path data is frozen. TTL,
count, and byte limits are independent; a frame larger than the byte limit is
returned through the server's one-shot path but is not retained here.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from shared.cache_sizing import estimate_retained_bytes
from shared.logging import get_logger

from ...core.propagation import freeze_frame_paths
from ...core.propagation.frozen_paths import FrozenPaths

logger = get_logger(__name__)


class GeneratorFrameCache:
    """Cache CPU-frozen raw frames under TTL, byte, and count limits.

    Size telemetry is known retained payload, not whole-process RSS. Oversized
    frames bypass admission without evicting unrelated entries.
    """

    def __init__(
        self,
        max_frames: int = 240,
        ttl_seconds: float = 300.0,
        max_size_bytes: int = 2 * 1024 * 1024 * 1024,  # 2GB
        mode: str = "streaming",
    ):
        """Initialize the frame cache.

        Args:
            max_frames: Maximum number of frames to keep (count-based eviction)
            ttl_seconds: Time-to-live for cached frames in seconds (0 = no TTL)
            max_size_bytes: Maximum total size in bytes (0 = no limit)
            mode: Diagnostic label reported as ``"streaming"`` or ``"batch"``.
                Both values use the same admission and eviction policy. Invalid
                values are normalized to ``"streaming"``.
        """
        self.frames: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.frame_timestamps: "OrderedDict[int, float]" = OrderedDict()
        self.frame_sizes: "OrderedDict[int, int]" = OrderedDict()
        self.total_frames = 0
        self.duration = 0.0
        self.frame_rate = 0.0
        self.lock = threading.Lock()

        self.max_frames = max(1, int(max_frames))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_size_bytes = max(0, int(max_size_bytes))
        self.mode = mode if mode in ("streaming", "batch") else "streaming"

        self._total_size_bytes = 0
        self._peak_size_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._oversized_bypasses = 0
        self._evictions_count = 0
        self._evictions_ttl = 0
        self._evictions_size = 0

    def _estimate_frame_size(self, frame_data: Dict[str, Any]) -> int:
        """Return known retained bytes for a CPU-frozen raw frame."""

        paths = frame_data.get("paths")
        path_arrays = tuple(paths.iter_arrays()) if isinstance(paths, FrozenPaths) else ()
        return estimate_retained_bytes(frame_data, *path_arrays)

    def _evict_stale_frames(self) -> int:
        """Remove frames older than TTL. Must be called with lock held.

        Returns:
            Number of frames evicted.
        """
        if self.ttl_seconds <= 0:
            return 0

        now = time.time()
        cutoff = now - self.ttl_seconds
        evicted = 0

        # Find frames to evict (iterate over copy to allow modification)
        frames_to_remove = []
        for frame_idx, ts in self.frame_timestamps.items():
            if ts < cutoff:
                frames_to_remove.append(frame_idx)

        # Remove stale frames
        for frame_idx in frames_to_remove:
            if frame_idx in self.frames:
                size = self.frame_sizes.pop(frame_idx, 0)
                self._total_size_bytes -= size
                self.frames.pop(frame_idx)
                self.frame_timestamps.pop(frame_idx, None)
                evicted += 1
                self._evictions_ttl += 1

        return evicted

    def _evict_for_size(self, needed_bytes: int = 0) -> int:
        """Evict oldest frames until under size limit. Must be called with lock held.

        Args:
            needed_bytes: Additional bytes we need to fit.

        Returns:
            Number of frames evicted.
        """
        if self.max_size_bytes <= 0:
            return 0

        target = self.max_size_bytes - needed_bytes
        evicted = 0

        while self._total_size_bytes > target and self.frames:
            oldest_idx, _ = self.frames.popitem(last=False)
            size = self.frame_sizes.pop(oldest_idx, 0)
            self._total_size_bytes -= size
            self.frame_timestamps.pop(oldest_idx, None)
            evicted += 1
            self._evictions_size += 1

        return evicted

    def add_frame(
        self,
        frame_idx: int,
        frame_data: Dict[str, Any],
        timestamp: float | None = None,
    ) -> bool:
        """Add a frame and return whether it was admitted to the cache."""
        with self.lock:
            freeze_frame_paths(frame_data)

            frame_size = self._estimate_frame_size(frame_data)

            if frame_idx in self.frames:
                old_size = self.frame_sizes.pop(frame_idx, 0)
                self._total_size_bytes -= old_size
                self.frames.pop(frame_idx)
                self.frame_timestamps.pop(frame_idx, None)

            self._evict_stale_frames()

            self.total_frames = max(self.total_frames, frame_idx + 1)

            if self.max_size_bytes > 0 and frame_size > self.max_size_bytes:
                self._oversized_bypasses += 1
                logger.warning(
                    "Frame %d retains %d known bytes and exceeds the %d-byte "
                    "generator cache limit; returning it without caching",
                    frame_idx,
                    frame_size,
                    self.max_size_bytes,
                )
                return False

            self._evict_for_size(frame_size)

            self.frames[frame_idx] = frame_data
            self.frame_timestamps[frame_idx] = timestamp or time.time()
            self.frame_sizes[frame_idx] = frame_size
            self._total_size_bytes += frame_size
            self._peak_size_bytes = max(self._peak_size_bytes, self._total_size_bytes)

            while len(self.frames) > self.max_frames:
                oldest_idx, _ = self.frames.popitem(last=False)
                size = self.frame_sizes.pop(oldest_idx, 0)
                self._total_size_bytes -= size
                self.frame_timestamps.pop(oldest_idx, None)
                self._evictions_count += 1

            return True

    def get_frame(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single cached frame by index.

        If TTL-based eviction is enabled and the frame has expired, it is
        silently removed and ``None`` is returned.

        Args:
            frame_idx: Zero-based frame index to look up.

        Returns:
            The cached frame data dictionary, or ``None`` if the frame is
            not present or has expired.
        """
        with self.lock:
            # Check TTL if enabled
            if self.ttl_seconds > 0:
                ts = self.frame_timestamps.get(frame_idx)
                if ts is not None and (time.time() - ts) > self.ttl_seconds:
                    # Frame is stale, remove it
                    self._remove_frame_unlocked(frame_idx)
                    self._evictions_ttl += 1
                    self._cache_misses += 1
                    return None
            if frame_idx in self.frames:
                self._cache_hits += 1
                return self.frames[frame_idx]
            self._cache_misses += 1
            return None

    def _remove_frame_unlocked(self, frame_idx: int) -> bool:
        """Remove a frame without acquiring lock. Must be called with lock held."""
        if frame_idx in self.frames:
            size = self.frame_sizes.pop(frame_idx, 0)
            self._total_size_bytes -= size
            self.frames.pop(frame_idx)
            self.frame_timestamps.pop(frame_idx, None)
            return True
        return False

    def get_frame_range(self, start: int, end: int, step: int = 1) -> List[Dict[str, Any]]:
        """Retrieve a range of cached frames.

        Frames that are missing or have expired (TTL) are silently skipped.

        Args:
            start: First frame index (inclusive).
            end: Last frame index (exclusive).
            step: Step size between indices.

        Returns:
            List of frame data dictionaries for the requested range,
            excluding any missing or expired entries.
        """
        with self.lock:
            frames = []
            now = time.time()
            for i in range(start, end, step):
                if i in self.frames:
                    # Check TTL
                    if self.ttl_seconds > 0:
                        ts = self.frame_timestamps.get(i)
                        if ts is not None and (now - ts) > self.ttl_seconds:
                            continue  # Skip stale frame
                    frames.append(self.frames[i])
            return frames

    def get_available_frames(self) -> List[int]:
        """Return sorted indices of all non-expired cached frames.

        When TTL-based eviction is enabled, only frames whose timestamp
        is within the TTL window are included.

        Returns:
            Sorted list of available frame indices.
        """
        with self.lock:
            # Filter out stale frames if TTL enabled
            if self.ttl_seconds > 0:
                now = time.time()
                cutoff = now - self.ttl_seconds
                return sorted(idx for idx, ts in self.frame_timestamps.items() if ts >= cutoff)
            return sorted(self.frames.keys())

    def remove_frame(self, frame_idx: int) -> bool:
        """Remove a specific frame from the cache.

        Returns:
            True if frame was removed, False if it didn't exist.
        """
        with self.lock:
            return self._remove_frame_unlocked(frame_idx)

    def update_stats(self, duration: float, frame_rate: float) -> None:
        """Update aggregate simulation statistics stored alongside the cache.

        Args:
            duration: Total simulation duration in seconds.
            frame_rate: Average frame generation rate in frames per second.
        """
        with self.lock:
            self.duration = duration
            self.frame_rate = frame_rate

    def clear(self) -> int:
        """Clear cached frames and reset timeline metadata.

        Lifetime cache counters and peak retained bytes remain available for
        diagnostics after a generation-epoch reset.

        Returns:
            The number of frames that were removed.
        """
        with self.lock:
            removed = len(self.frames)
            self.frames = OrderedDict()
            self.frame_timestamps = OrderedDict()
            self.frame_sizes = OrderedDict()
            self._total_size_bytes = 0
            self.total_frames = 0
            self.duration = 0.0
            self.frame_rate = 0.0
        return removed

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for health monitoring.

        Returns:
            Dictionary with cache stats including size, age, eviction counts.
        """
        with self.lock:
            now = time.time()
            oldest_age = 0.0
            if self.frame_timestamps:
                oldest_ts = min(self.frame_timestamps.values())
                oldest_age = now - oldest_ts

            return {
                "cached_frames": len(self.frames),
                "total_frames": self.total_frames,
                "total_size_bytes": self._total_size_bytes,
                "peak_size_bytes": self._peak_size_bytes,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "oversized_bypasses": self._oversized_bypasses,
                "max_frames": self.max_frames,
                "max_size_bytes": self.max_size_bytes,
                "ttl_seconds": self.ttl_seconds,
                "mode": self.mode,
                "oldest_frame_age_seconds": oldest_age,
                "evictions_count": self._evictions_count,
                "evictions_ttl": self._evictions_ttl,
                "evictions_size": self._evictions_size,
            }
