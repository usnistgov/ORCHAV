"""Unit tests for GeneratorFrameCache with TTL, size limits, and eviction tracking.

Tests cover:
- Count-based eviction (max_frames)
- TTL-based eviction (ttl_seconds)
- Size-based eviction (max_size_bytes)
- Cache statistics (get_cache_stats)
- Mode separation (streaming vs batch)
"""

import time
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from generator.core.propagation.frozen_paths import FrozenPaths

# Import the cache class
from generator.io.grpc.live_server import GeneratorFrameCache


def _make_frame(num_paths: int = 10) -> Dict[str, Any]:
    """Create a mock frame data dictionary."""
    return {
        "frame_idx": 0,
        "paths": {f"path_{i}": {"vertices": [[0, 0, 0]]} for i in range(num_paths)},
        "tx_positions": [[0, 0, 0]],
        "rx_positions": [[1, 1, 1]],
    }


def _make_frozen_array_frame(path_count: int = 100_000) -> tuple[Dict[str, Any], int]:
    """Create a frozen-path frame and return its exact known array bytes."""

    arrays = {
        "valid": np.ones(path_count, dtype=np.bool_),
        "tau": np.zeros(path_count, dtype=np.float64),
        "a": np.zeros(path_count, dtype=np.complex64),
        "phi_t": np.zeros(path_count, dtype=np.float32),
        "theta_t": np.zeros(path_count, dtype=np.float32),
        "phi_r": np.zeros(path_count, dtype=np.float32),
        "theta_r": np.zeros(path_count, dtype=np.float32),
        "doppler": np.zeros(path_count, dtype=np.float32),
        "vertices": np.zeros((path_count, 3), dtype=np.float32),
        "interactions": np.zeros(path_count, dtype=np.uint8),
        "objects": np.zeros(path_count, dtype=np.int32),
    }
    source = SimpleNamespace(total_paths=path_count, **arrays)
    frame = {
        "frame_idx": 0,
        "paths": FrozenPaths(source),
        "tx_positions": np.zeros((1, 3), dtype=np.float64),
        "rx_positions": np.ones((1, 3), dtype=np.float64),
    }
    return frame, sum(array.nbytes for array in arrays.values())


class TestCacheMaxFramesEviction:
    """Tests for count-based eviction."""

    def test_cache_evicts_oldest_when_max_exceeded(self):
        """Verify oldest frames are evicted when max_frames is exceeded."""
        cache = GeneratorFrameCache(max_frames=3, ttl_seconds=0, max_size_bytes=0)

        # Add 5 frames
        for i in range(5):
            cache.add_frame(i, _make_frame())

        # Should only have the 3 most recent frames
        assert len(cache.frames) == 3
        assert cache.get_frame(0) is None  # Evicted
        assert cache.get_frame(1) is None  # Evicted
        assert cache.get_frame(2) is not None
        assert cache.get_frame(3) is not None
        assert cache.get_frame(4) is not None

    def test_cache_replaces_existing_frame(self):
        """Verify replacing an existing frame doesn't increase count."""
        cache = GeneratorFrameCache(max_frames=3, ttl_seconds=0, max_size_bytes=0)

        for i in range(3):
            cache.add_frame(i, _make_frame())

        # Replace frame 1
        cache.add_frame(1, _make_frame(num_paths=20))

        # Still 3 frames
        assert len(cache.frames) == 3

    def test_evictions_count_tracked(self):
        """Verify count-based evictions are tracked in stats."""
        cache = GeneratorFrameCache(max_frames=2, ttl_seconds=0, max_size_bytes=0)

        for i in range(5):
            cache.add_frame(i, _make_frame())

        stats = cache.get_cache_stats()
        assert stats["evictions_count"] == 3  # Frames 0, 1, 2 evicted


class TestCacheTTLEviction:
    """Tests for TTL-based eviction."""

    def test_stale_frames_not_returned(self):
        """Verify stale frames are not returned by get_frame."""
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0.1, max_size_bytes=0)

        cache.add_frame(0, _make_frame())
        assert cache.get_frame(0) is not None

        # Wait for TTL to expire
        time.sleep(0.15)

        # Frame should now be stale
        assert cache.get_frame(0) is None

    def test_stale_frames_evicted_on_add(self):
        """Verify stale frames are evicted when adding new frames."""
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0.1, max_size_bytes=0)

        cache.add_frame(0, _make_frame())
        cache.add_frame(1, _make_frame())

        time.sleep(0.15)

        # Add new frame - should trigger TTL eviction
        cache.add_frame(2, _make_frame())

        stats = cache.get_cache_stats()
        assert stats["evictions_ttl"] >= 2  # Frames 0 and 1 evicted
        assert len(cache.frames) == 1  # Only frame 2 remains

    def test_available_frames_excludes_stale(self):
        """Verify get_available_frames excludes stale frames."""
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0.1, max_size_bytes=0)

        cache.add_frame(0, _make_frame())
        time.sleep(0.15)
        cache.add_frame(1, _make_frame())

        available = cache.get_available_frames()
        assert 0 not in available  # Stale
        assert 1 in available


class TestCacheSizeEviction:
    """Tests for size-based eviction."""

    def test_size_eviction_when_limit_exceeded(self):
        """Verify frames are evicted when size limit would be exceeded."""
        payload = np.zeros(6000, dtype=np.uint8)
        cache = GeneratorFrameCache(
            max_frames=100,
            ttl_seconds=0,
            max_size_bytes=10_000,
        )

        cache.add_frame(0, {"frame_idx": 0, "paths": None, "payload": payload.copy()})
        cache.add_frame(1, {"frame_idx": 1, "paths": None, "payload": payload.copy()})

        stats = cache.get_cache_stats()
        assert stats["evictions_size"] == 1
        assert stats["total_size_bytes"] <= cache.max_size_bytes
        assert cache.get_frame(0) is None
        assert cache.get_frame(1) is not None

    def test_frozen_paths_are_charged_by_known_array_bytes(self):
        cache = GeneratorFrameCache(max_frames=2, ttl_seconds=0, max_size_bytes=0)
        frame, known_array_bytes = _make_frozen_array_frame()

        assert cache.add_frame(0, frame) is True

        stats = cache.get_cache_stats()
        assert stats["total_size_bytes"] >= known_array_bytes
        assert stats["total_size_bytes"] > 5_000_000

    def test_oversized_frame_bypasses_without_evicting_other_entries(self):
        sizing_cache = GeneratorFrameCache(max_frames=3, ttl_seconds=0, max_size_bytes=0)
        small = {"frame_idx": 0, "paths": None, "payload": np.zeros(64, dtype=np.uint8)}
        large = {"frame_idx": 1, "paths": None, "payload": np.zeros(4096, dtype=np.uint8)}
        small_size = sizing_cache._estimate_frame_size(small)
        large_size = sizing_cache._estimate_frame_size(large)
        cache = GeneratorFrameCache(
            max_frames=3,
            ttl_seconds=0,
            max_size_bytes=(small_size + large_size) // 2,
        )

        assert cache.add_frame(0, small) is True
        assert cache.add_frame(1, large) is False

        stats = cache.get_cache_stats()
        assert cache.get_frame(0) is small
        assert cache.get_frame(1) is None
        assert stats["oversized_bypasses"] == 1
        assert stats["cached_frames"] == 1
        assert stats["total_size_bytes"] <= stats["max_size_bytes"]

    def test_oversized_replacement_removes_stale_same_index(self):
        sizing_cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=0)
        small = {"frame_idx": 0, "paths": None, "payload": np.zeros(64, dtype=np.uint8)}
        large = {"frame_idx": 0, "paths": None, "payload": np.zeros(4096, dtype=np.uint8)}
        cap = (
            sizing_cache._estimate_frame_size(small) + sizing_cache._estimate_frame_size(large)
        ) // 2
        cache = GeneratorFrameCache(max_frames=1, ttl_seconds=0, max_size_bytes=cap)

        assert cache.add_frame(0, small) is True
        assert cache.add_frame(0, large) is False

        assert cache.get_frame(0) is None
        assert cache.get_cache_stats()["cached_frames"] == 0

    def test_size_tracking_accurate(self):
        """Verify total_size_bytes is tracked correctly."""
        cache = GeneratorFrameCache(max_frames=100, ttl_seconds=0, max_size_bytes=0)

        cache.add_frame(0, _make_frame())
        stats1 = cache.get_cache_stats()
        assert stats1["total_size_bytes"] > 0

        cache.add_frame(1, _make_frame())
        stats2 = cache.get_cache_stats()
        assert stats2["total_size_bytes"] > stats1["total_size_bytes"]

        # Remove a frame
        cache.remove_frame(0)
        stats3 = cache.get_cache_stats()
        assert stats3["total_size_bytes"] < stats2["total_size_bytes"]


class TestCacheStats:
    """Tests for get_cache_stats method."""

    def test_stats_reflect_configuration(self):
        """Verify stats include configuration values."""
        cache = GeneratorFrameCache(
            max_frames=42,
            ttl_seconds=123.0,
            max_size_bytes=999,
            mode="batch",
        )

        stats = cache.get_cache_stats()
        assert stats["max_frames"] == 42
        assert stats["ttl_seconds"] == 123.0
        assert stats["max_size_bytes"] == 999
        assert stats["mode"] == "batch"

    def test_oldest_frame_age_calculated(self):
        """Verify oldest_frame_age_seconds is calculated correctly."""
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0, max_size_bytes=0)

        cache.add_frame(0, _make_frame())
        time.sleep(0.1)

        stats = cache.get_cache_stats()
        assert stats["oldest_frame_age_seconds"] >= 0.1

    def test_empty_cache_stats(self):
        """Verify stats for empty cache."""
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0, max_size_bytes=0)

        stats = cache.get_cache_stats()
        assert stats["cached_frames"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["oldest_frame_age_seconds"] == 0.0

    def test_peak_hits_misses_and_ttl_eviction_are_reported(self):
        cache = GeneratorFrameCache(max_frames=2, ttl_seconds=0.01, max_size_bytes=0)
        cache.add_frame(0, _make_frame())
        retained = cache.get_cache_stats()["total_size_bytes"]

        assert cache.get_frame(0) is not None
        assert cache.get_frame(99) is None
        time.sleep(0.02)
        assert cache.get_frame(0) is None

        stats = cache.get_cache_stats()
        assert stats["peak_size_bytes"] == retained
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["evictions_ttl"] == 1


class TestCacheClear:
    """Tests for cache clear method."""

    def test_clear_removes_all_frames(self):
        """Verify clear removes all frames and resets stats."""
        cache = GeneratorFrameCache(max_frames=10, ttl_seconds=0, max_size_bytes=0)

        for i in range(5):
            cache.add_frame(i, _make_frame())

        removed = cache.clear()
        assert removed == 5
        assert len(cache.frames) == 0
        stats = cache.get_cache_stats()
        assert stats["total_size_bytes"] == 0
        assert stats["peak_size_bytes"] > 0


class TestCacheModes:
    """Tests for cache mode configuration."""

    def test_streaming_mode_default(self):
        """Verify streaming is the default mode."""
        cache = GeneratorFrameCache()
        assert cache.mode == "streaming"

    def test_batch_mode_accepted(self):
        """Verify batch mode is accepted."""
        cache = GeneratorFrameCache(mode="batch")
        assert cache.mode == "batch"

    def test_invalid_mode_defaults_to_streaming(self):
        """Verify invalid mode falls back to streaming."""
        cache = GeneratorFrameCache(mode="invalid")
        assert cache.mode == "streaming"
