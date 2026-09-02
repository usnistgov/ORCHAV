"""Provider-agnostic loading and caching for ``StandardMPCFrame`` data.

``FrameLoaderService`` sits above ``DataProvider`` implementations. It does not
care whether frames came from local HDF5, remote gRPC, a measurement cache, or a
future provider. Its job is to apply shared cache policy and enforce the
complete-frame type at the provider boundary.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from shared.logging import get_logger

from .provider_base import DataProvider
from .types import StandardMPCFrame

logger = get_logger(__name__)

DEFAULT_FRAME_CACHE_SIZE = 100


class CacheEvictionPolicy(Enum):
    """Entry-order policy used when a frame cache reaches capacity.

    LRU refreshes order on a hit, FIFO preserves insertion order, and LIFO
    evicts the most recently inserted entry.
    """

    LRU = "lru"
    FIFO = "fifo"
    LIFO = "lifo"


@dataclass
class CacheStats:
    """Cumulative request/eviction counters and frame-entry capacity."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheManager:
    """Thread-safe, entry-count-bounded cache of complete canonical frames.

    Keys are frame IDs and ``max_size`` counts frames rather than bytes. The
    selected policy controls eviction order. Eviction callbacks are advisory:
    callback failures are logged without failing the cache operation.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_FRAME_CACHE_SIZE,
        policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
        on_evict: Optional[Callable[[int, StandardMPCFrame], None]] = None,
    ):
        """Initialize the cache manager.

        Args:
            max_size: Maximum number of complete frames to retain.
            policy: Ordering used to select an entry for eviction.
            on_evict: Optional callback receiving ``(frame_id, frame)``.
        """
        self._max_size = max(1, max_size)
        self._policy = policy
        self._on_evict = on_evict
        self._cache: OrderedDict[int, StandardMPCFrame] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=self._max_size)

    @property
    def max_size(self) -> int:
        """Maximum number of frames retained by the cache."""
        return self._max_size

    def set_max_size(self, max_size: int) -> None:
        """Resize the cache capacity and evict if needed.

        Args:
            max_size: New maximum number of frames to keep (minimum 1).
        """
        with self._lock:
            self._max_size = max(1, max_size)
            self._stats.max_size = self._max_size
            while len(self._cache) > self._max_size:
                self._evict_one()

    @property
    def size(self) -> int:
        """Current number of cached frames."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> CacheStats:
        """Return cumulative counters with the current frame-entry count."""
        with self._lock:
            self._stats.current_size = len(self._cache)
            return self._stats

    def get(self, frame_idx: int) -> Optional[StandardMPCFrame]:
        """Get a frame from the cache and record a hit or miss.

        Args:
            frame_idx: Frame ID to retrieve.

        Returns:
            The complete frame if cached, otherwise ``None``.
        """
        with self._lock:
            if frame_idx in self._cache:
                self._stats.hits += 1
                if self._policy == CacheEvictionPolicy.LRU:
                    self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]
            self._stats.misses += 1
            return None

    def put(self, frame_idx: int, frame: StandardMPCFrame) -> None:
        """Store a complete frame, evicting by policy when at capacity.

        Args:
            frame_idx: Frame ID used as the cache key.
            frame: Complete canonical frame to retain.
        """
        with self._lock:
            if frame_idx in self._cache:
                self._cache[frame_idx] = frame
                if self._policy == CacheEvictionPolicy.LRU:
                    self._cache.move_to_end(frame_idx)
                return

            while len(self._cache) >= self._max_size:
                self._evict_one()

            self._cache[frame_idx] = frame

    def _evict_one(self) -> None:
        """Evict one frame according to the eviction policy."""
        if not self._cache:
            return

        if self._policy == CacheEvictionPolicy.LIFO:
            evicted_idx, evicted_frame = self._cache.popitem(last=True)
        else:  # LRU or FIFO: remove oldest
            evicted_idx, evicted_frame = self._cache.popitem(last=False)

        self._stats.evictions += 1

        if self._on_evict:
            try:
                self._on_evict(evicted_idx, evicted_frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Eviction callback failed: %s", exc)

    def remove(self, frame_idx: int) -> Optional[StandardMPCFrame]:
        """
        Remove a specific frame from the cache.

        Args:
            frame_idx: Frame index to remove

        Returns:
            The removed frame, or None if not present
        """
        with self._lock:
            return self._cache.pop(frame_idx, None)

    def clear(self) -> int:
        """
        Clear all cached frames.

        Returns:
            Number of frames that were cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug("Cache cleared: %d frames removed", count)
            return count

    def contains(self, frame_idx: int) -> bool:
        """Check if a frame is in the cache."""
        with self._lock:
            return frame_idx in self._cache

    def keys(self) -> List[int]:
        """Get list of cached frame indices."""
        with self._lock:
            return list(self._cache.keys())

    def __contains__(self, frame_idx: int) -> bool:
        return self.contains(frame_idx)

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        """CacheManager is always truthy, even when empty."""
        return True


@dataclass
class LoaderStats:
    """Statistics about frame loading."""

    frames_loaded: int = 0
    load_errors: int = 0
    total_load_time_ms: float = 0.0
    preload_frames: int = 0
    preload_time_ms: float = 0.0

    @property
    def average_load_time_ms(self) -> float:
        """Average frame load time in milliseconds."""
        return self.total_load_time_ms / self.frames_loaded if self.frames_loaded > 0 else 0.0


class FrameLoaderService:
    """Load complete frames from one provider and cache successful reads.

    The provider owns storage and availability; this service adds explicit
    preloading, invalidation, load counters, and subscriber notification.
    Subscribers receive successful provider reads made by ``get_frame()``, not
    cache hits or preload operations.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache: Optional[CacheManager] = None,
        *,
        default_cache_size: int = 100,
    ):
        """
        Initialize the frame loader service.

        Args:
            provider: Data provider to load frames from
            cache: Optional cache manager (creates one if not provided)
            default_cache_size: Cache size if creating new cache
        """
        self._provider = provider
        self._cache = cache or CacheManager(max_size=default_cache_size)
        self._stats = LoaderStats()
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[int, StandardMPCFrame], None]] = []

    @property
    def provider(self) -> DataProvider:
        """The underlying data provider."""
        return self._provider

    @property
    def cache(self) -> CacheManager:
        """The cache manager."""
        return self._cache

    @property
    def stats(self) -> LoaderStats:
        """Get loading statistics."""
        return self._stats

    def get_frame(
        self,
        frame_idx: int,
        *,
        bypass_cache: bool = False,
        validate: bool = True,
    ) -> Optional[StandardMPCFrame]:
        """Get a complete frame, using the shared cache when permitted.

        Args:
            frame_idx: Frame ID to load.
            bypass_cache: Skip lookup but still cache a successful provider read.
            validate: Require a provider result to be ``StandardMPCFrame`` before
                caching and returning it.

        Returns:
            The complete frame, or ``None`` if unavailable or invalid.
        """
        if not bypass_cache:
            cached = self._cache.get(frame_idx)
            if cached is not None:
                return cached

        start_time = time.perf_counter()
        try:
            frame = self._provider.load_frame(frame_idx)
            if frame is not None:
                if validate and not self._validate_frame(frame_idx, frame):
                    return None
                self._cache.put(frame_idx, frame)
                self._stats.frames_loaded += 1
                self._stats.total_load_time_ms += (time.perf_counter() - start_time) * 1000

                self._notify_subscribers(frame_idx, frame)

                return frame
        except (OSError, KeyError, ValueError) as e:
            self._stats.load_errors += 1
            logger.error("Failed to load frame %d: %s", frame_idx, e)

        return None

    def has_frame(self, frame_idx: int) -> bool:
        """Check if a frame is available (in cache or provider)."""
        if self._cache.contains(frame_idx):
            return True
        return self._provider.has_frame(frame_idx)

    def list_frames(self) -> List[int]:
        """List all available frame indices."""
        return self._provider.list_frames()

    def preload_range(
        self,
        start: int,
        end: int,
        *,
        on_progress: Optional[Callable[[int, int], None]] = None,
        validate: bool = True,
    ) -> int:
        """
        Preload a range of frames into the cache.

        Args:
            start: Start frame index (inclusive)
            end: End frame index (exclusive)
            on_progress: Callback with (current, total) progress
            validate: Require complete canonical frames before caching them.

        Returns:
            Number of frames successfully preloaded
        """
        preload_start = time.perf_counter()
        loaded = 0
        total = end - start

        for i, frame_idx in enumerate(range(start, end)):
            if self._cache.contains(frame_idx):
                loaded += 1
                continue

            try:
                frame = self._provider.load_frame(frame_idx)
                if frame is not None:
                    if validate and not self._validate_frame(frame_idx, frame):
                        continue
                    self._cache.put(frame_idx, frame)
                    loaded += 1
            except (OSError, KeyError, ValueError) as e:
                logger.debug("Preload failed for frame %d: %s", frame_idx, e)

            if on_progress:
                on_progress(i + 1, total)

        self._stats.preload_frames += loaded
        self._stats.preload_time_ms += (time.perf_counter() - preload_start) * 1000
        logger.debug("Preloaded %d/%d frames in range [%d, %d)", loaded, total, start, end)
        return loaded

    def preload_all(
        self,
        *,
        on_progress: Optional[Callable[[int, int], None]] = None,
        validate: bool = True,
    ) -> int:
        """
        Preload all available frames into the cache.

        Args:
            on_progress: Callback with (current, total) progress
            validate: Require complete canonical frames before caching them.

        Returns:
            Number of frames successfully preloaded
        """
        frames = self.list_frames()
        if not frames:
            return 0

        preload_start = time.perf_counter()
        loaded = 0
        total = len(frames)

        for i, frame_idx in enumerate(frames):
            if self._cache.contains(frame_idx):
                loaded += 1
                continue

            try:
                frame = self._provider.load_frame(frame_idx)
                if frame is not None:
                    if validate and not self._validate_frame(frame_idx, frame):
                        continue
                    self._cache.put(frame_idx, frame)
                    loaded += 1
            except (OSError, KeyError, ValueError) as e:
                logger.debug("Preload failed for frame %d: %s", frame_idx, e)

            if on_progress:
                on_progress(i + 1, total)

        self._stats.preload_frames += loaded
        self._stats.preload_time_ms += (time.perf_counter() - preload_start) * 1000
        logger.info("Preloaded %d/%d frames in %.2f ms", loaded, total, self._stats.preload_time_ms)
        return loaded

    def invalidate(self, frame_idx: Optional[int] = None) -> int:
        """
        Invalidate cached frames.

        Args:
            frame_idx: Specific frame to invalidate, or None for all

        Returns:
            Number of frames invalidated
        """
        if frame_idx is None:
            return self._cache.clear()
        else:
            if self._cache.remove(frame_idx):
                return 1
            return 0

    def subscribe(self, callback: Callable[[int, StandardMPCFrame], None]) -> None:
        """
        Subscribe to frame load events.

        Args:
            callback: Function called with (frame_idx, frame) when a frame is loaded
        """
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[int, StandardMPCFrame], None]) -> None:
        """
        Unsubscribe from frame load events.

        Args:
            callback: Previously registered callback to remove
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify_subscribers(self, frame_idx: int, frame: StandardMPCFrame) -> None:
        """Notify all subscribers of a newly loaded frame."""
        with self._lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(frame_idx, frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Subscriber callback failed: %s", exc)

    def _validate_frame(self, frame_idx: int, frame: StandardMPCFrame) -> bool:
        """Reject provider results that are not complete canonical frames."""
        if isinstance(frame, StandardMPCFrame):
            return True
        self._stats.load_errors += 1
        logger.error(
            "Frame %s has invalid provider type %s",
            frame_idx,
            type(frame).__name__,
        )
        return False


def create_loader(
    provider: DataProvider,
    *,
    cache_size: int = 100,
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
) -> FrameLoaderService:
    """
    Create a FrameLoaderService with a new cache.

    Args:
        provider: Data provider to load frames from
        cache_size: Maximum number of frames to cache
        eviction_policy: Cache eviction policy

    Returns:
        Configured FrameLoaderService
    """
    cache = CacheManager(max_size=cache_size, policy=eviction_policy)
    return FrameLoaderService(provider, cache)
