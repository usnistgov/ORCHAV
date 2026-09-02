"""
Tests for shared.frames.loader module.

These tests verify the CacheManager and FrameLoaderService that provide
unified frame caching and loading infrastructure.
"""

import threading
import time

import numpy as np

from shared.frames.loader import (
    CacheEvictionPolicy,
    CacheManager,
    FrameLoaderService,
    create_loader,
)
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.provider_base import DataProvider
from shared.frames.types import StandardMPCFrame

# =============================================================================
# Test Fixtures
# =============================================================================


def make_test_frame(frame_idx: int = 0) -> StandardMPCFrame:
    """Create a minimal test frame."""
    return standard_mpc_frame_from_pair_data(
        frame_index=frame_idx,
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        tx_positions=np.array([[float(frame_idx), 0.0, 0.0]], dtype=np.float64),
        rx_positions=np.array([[0.0, float(frame_idx), 0.0]], dtype=np.float64),
        tx_names=("tx-0",),
        rx_names=("rx-0",),
        vertices_by_pair=[np.empty((1, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.empty((1, 0), dtype=np.int32)],
        path_lengths_by_pair=[np.array([0], dtype=np.int64)],
        provenance={"provider": "test", "frame_idx": frame_idx},
    )


class MockProvider(DataProvider):
    """Mock provider for testing."""

    def __init__(self, frames: dict[int, StandardMPCFrame] = None, delay: float = 0.0):
        self._frames = frames or {}
        self._delay = delay
        self._load_count = 0

    def list_frames(self) -> list[int]:
        return sorted(self._frames.keys())

    def has_frame(self, step: int) -> bool:
        return step in self._frames

    def load_frame(self, step: int) -> StandardMPCFrame:
        self._load_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        if step not in self._frames:
            raise KeyError(f"Frame {step} not found")
        return self._frames[step]

    @property
    def load_count(self) -> int:
        return self._load_count


class InvalidProvider(DataProvider):
    """Provider that violates the complete-frame return contract."""

    def list_frames(self) -> list[int]:
        return [0]

    def has_frame(self, step: int) -> bool:
        return step == 0

    def load_frame(self, step: int) -> StandardMPCFrame:
        return {}  # type: ignore[return-value]


# =============================================================================
# CacheManager Tests
# =============================================================================


class TestCacheManager:
    """Tests for CacheManager."""

    def test_create_cache(self):
        """Can create a cache with specified max size."""
        cache = CacheManager(max_size=50)
        assert cache.max_size == 50
        assert cache.size == 0

    def test_put_and_get(self):
        """Can store and retrieve frames."""
        cache = CacheManager(max_size=10)
        frame = make_test_frame(0)

        cache.put(0, frame)
        assert cache.size == 1

        retrieved = cache.get(0)
        assert retrieved is not None
        assert retrieved.provenance is not None
        assert retrieved.provenance["frame_idx"] == 0

    def test_get_missing_returns_none(self):
        """Getting a missing frame returns None."""
        cache = CacheManager(max_size=10)
        assert cache.get(99) is None

    def test_contains(self):
        """Can check if frame is cached."""
        cache = CacheManager(max_size=10)
        frame = make_test_frame(0)

        assert 0 not in cache
        cache.put(0, frame)
        assert 0 in cache
        assert cache.contains(0)

    def test_remove(self):
        """Can remove a specific frame."""
        cache = CacheManager(max_size=10)
        frame = make_test_frame(0)
        cache.put(0, frame)

        removed = cache.remove(0)
        assert removed is not None
        assert 0 not in cache

    def test_remove_missing_returns_none(self):
        """Removing a missing frame returns None."""
        cache = CacheManager(max_size=10)
        assert cache.remove(99) is None

    def test_clear(self):
        """Can clear all cached frames."""
        cache = CacheManager(max_size=10)
        for i in range(5):
            cache.put(i, make_test_frame(i))

        cleared = cache.clear()
        assert cleared == 5
        assert cache.size == 0

    def test_keys(self):
        """Can get list of cached frame indices."""
        cache = CacheManager(max_size=10)
        for i in [0, 2, 4]:
            cache.put(i, make_test_frame(i))

        keys = cache.keys()
        assert sorted(keys) == [0, 2, 4]

    def test_resize_cache_eviction(self):
        """Resizing down should evict oldest entries."""
        cache = CacheManager(max_size=3)
        for i in range(3):
            cache.put(i, make_test_frame(i))

        cache.set_max_size(2)

        assert cache.max_size == 2
        assert cache.size == 2
        assert 0 not in cache


class TestCacheEviction:
    """Tests for cache eviction policies."""

    def test_lru_eviction(self):
        """LRU policy evicts least recently used frame."""
        cache = CacheManager(max_size=3, policy=CacheEvictionPolicy.LRU)

        # Add 3 frames
        for i in range(3):
            cache.put(i, make_test_frame(i))

        # Access frame 0 (makes it most recently used)
        cache.get(0)

        # Add frame 3 (should evict frame 1, not 0)
        cache.put(3, make_test_frame(3))

        assert 0 in cache  # Still there (was accessed)
        assert 1 not in cache  # Evicted
        assert 2 in cache
        assert 3 in cache

    def test_fifo_eviction(self):
        """FIFO policy evicts first added frame."""
        cache = CacheManager(max_size=3, policy=CacheEvictionPolicy.FIFO)

        for i in range(3):
            cache.put(i, make_test_frame(i))

        # Add frame 3 (should evict frame 0, the first added)
        cache.put(3, make_test_frame(3))

        assert 0 not in cache  # Evicted (first added)
        assert 1 in cache
        assert 2 in cache
        assert 3 in cache

    def test_eviction_callback(self):
        """Eviction callback is called when frames are evicted."""
        evicted = []

        def on_evict(frame_idx, frame):
            evicted.append(frame_idx)

        cache = CacheManager(max_size=2, on_evict=on_evict)

        cache.put(0, make_test_frame(0))
        cache.put(1, make_test_frame(1))
        cache.put(2, make_test_frame(2))  # Triggers eviction

        assert len(evicted) == 1
        assert evicted[0] == 0


class TestCacheStats:
    """Tests for cache statistics."""

    def test_hit_tracking(self):
        """Cache tracks hits."""
        cache = CacheManager(max_size=10)
        cache.put(0, make_test_frame(0))

        cache.get(0)  # Hit
        cache.get(0)  # Hit

        assert cache.stats.hits == 2

    def test_miss_tracking(self):
        """Cache tracks misses."""
        cache = CacheManager(max_size=10)

        cache.get(0)  # Miss
        cache.get(1)  # Miss

        assert cache.stats.misses == 2

    def test_hit_rate(self):
        """Can calculate hit rate."""
        cache = CacheManager(max_size=10)
        cache.put(0, make_test_frame(0))

        cache.get(0)  # Hit
        cache.get(1)  # Miss

        assert cache.stats.hit_rate == 0.5

    def test_eviction_tracking(self):
        """Cache tracks evictions."""
        cache = CacheManager(max_size=2)

        cache.put(0, make_test_frame(0))
        cache.put(1, make_test_frame(1))
        cache.put(2, make_test_frame(2))  # Evicts 0

        assert cache.stats.evictions == 1


class TestCacheThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self):
        """Cache handles concurrent access safely."""
        cache = CacheManager(max_size=100)
        errors = []

        def writer():
            try:
                for i in range(50):
                    cache.put(i, make_test_frame(i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    cache.get(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# FrameLoaderService Tests
# =============================================================================


class TestFrameLoaderService:
    """Tests for FrameLoaderService."""

    def test_create_loader(self):
        """Can create a loader with provider."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = FrameLoaderService(provider)

        assert loader.provider is provider
        assert loader.cache is not None

    def test_get_frame_loads_from_provider(self):
        """get_frame loads from provider on cache miss."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = FrameLoaderService(provider)

        frame = loader.get_frame(0)
        assert frame is not None
        assert provider.load_count == 1

    def test_get_frame_uses_cache(self):
        """get_frame returns cached frame on hit."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = FrameLoaderService(provider)

        loader.get_frame(0)  # Load into cache
        loader.get_frame(0)  # Should use cache

        assert provider.load_count == 1  # Only one load

    def test_get_frame_bypass_cache(self):
        """get_frame with bypass_cache loads from provider."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = FrameLoaderService(provider)

        loader.get_frame(0)  # Load into cache
        loader.get_frame(0, bypass_cache=True)  # Force reload

        assert provider.load_count == 2

    def test_has_frame_checks_cache_and_provider(self):
        """has_frame checks both cache and provider."""
        provider = MockProvider({0: make_test_frame(0), 1: make_test_frame(1)})
        loader = FrameLoaderService(provider)

        # Not in cache, but in provider
        assert loader.has_frame(0)

        # Load into cache
        loader.get_frame(0)

        # Now in cache
        assert loader.has_frame(0)

    def test_list_frames(self):
        """list_frames delegates to provider."""
        provider = MockProvider({0: make_test_frame(0), 5: make_test_frame(5)})
        loader = FrameLoaderService(provider)

        assert loader.list_frames() == [0, 5]

    def test_invalidate_single_frame(self):
        """Can invalidate a single cached frame."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = FrameLoaderService(provider)

        loader.get_frame(0)  # Load into cache
        assert loader.cache.contains(0)

        invalidated = loader.invalidate(0)
        assert invalidated == 1
        assert not loader.cache.contains(0)

    def test_invalidate_all(self):
        """Can invalidate all cached frames."""
        provider = MockProvider({i: make_test_frame(i) for i in range(5)})
        loader = FrameLoaderService(provider)

        for i in range(5):
            loader.get_frame(i)

        invalidated = loader.invalidate()
        assert invalidated == 5
        assert loader.cache.size == 0

    def test_loader_rejects_invalid_frames(self):
        """Invalid frames from providers are not cached and count as errors."""
        provider = InvalidProvider()
        loader = FrameLoaderService(provider)

        result = loader.get_frame(0)
        assert result is None
        assert loader.stats.load_errors == 1


class TestFrameLoaderPreload:
    """Tests for preloading functionality."""

    def test_preload_range(self):
        """Can preload a range of frames."""
        provider = MockProvider({i: make_test_frame(i) for i in range(10)})
        loader = FrameLoaderService(provider)

        loaded = loader.preload_range(0, 5)
        assert loaded == 5

        for i in range(5):
            assert loader.cache.contains(i)

    def test_preload_all(self):
        """Can preload all available frames."""
        provider = MockProvider({i: make_test_frame(i) for i in range(5)})
        loader = FrameLoaderService(provider)

        loaded = loader.preload_all()
        assert loaded == 5
        assert loader.cache.size == 5

    def test_preload_progress_callback(self):
        """Preload reports progress via callback."""
        provider = MockProvider({i: make_test_frame(i) for i in range(5)})
        loader = FrameLoaderService(provider)

        progress = []

        def on_progress(current, total):
            progress.append((current, total))

        loader.preload_all(on_progress=on_progress)

        assert len(progress) == 5
        assert progress[-1] == (5, 5)


class TestFrameLoaderSubscription:
    """Tests for frame load subscription."""

    def test_subscribe_receives_events(self):
        """Subscribers receive frame load events."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = FrameLoaderService(provider)

        events = []

        def callback(idx, frame):
            events.append((idx, frame))

        loader.subscribe(callback)
        loader.get_frame(0)

        assert len(events) == 1
        assert events[0][0] == 0

    def test_unsubscribe(self):
        """Can unsubscribe from events."""
        provider = MockProvider({0: make_test_frame(0), 1: make_test_frame(1)})
        loader = FrameLoaderService(provider)

        events = []

        def callback(idx, frame):
            events.append(idx)

        loader.subscribe(callback)
        loader.get_frame(0)  # Should trigger callback

        loader.unsubscribe(callback)
        loader.get_frame(1)  # Should not trigger callback

        assert events == [0]


class TestCreateLoaderHelper:
    """Tests for create_loader convenience function."""

    def test_create_loader_with_defaults(self):
        """Can create loader with default settings."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = create_loader(provider)

        assert loader.cache.max_size == 100  # Default
        assert loader.provider is provider

    def test_create_loader_with_custom_settings(self):
        """Can create loader with custom settings."""
        provider = MockProvider({0: make_test_frame(0)})
        loader = create_loader(provider, cache_size=50, eviction_policy=CacheEvictionPolicy.FIFO)

        assert loader.cache.max_size == 50
