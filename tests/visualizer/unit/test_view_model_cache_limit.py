"""Unit tests for ViewModel cache size limiting."""

from unittest.mock import Mock

from visualizer.src.config import MAX_VIEW_MODEL_CACHE_SIZE
from visualizer.src.pipeline.frame_pipeline import FramePipeline


class TestViewModelCacheLimit:
    """Tests for ViewModel cache size enforcement."""

    def test_cache_limit_not_enforced_when_under_limit(self):
        """Test that cache is not modified when under limit."""
        # Create mock visualizer with small cache
        viz = Mock()
        viz.mpc_view_cache = {f"key_{i}": f"value_{i}" for i in range(50)}

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Enforce limit (should not evict anything)
        pipeline._enforce_view_model_cache_limit()

        # Cache should remain unchanged
        assert len(viz.mpc_view_cache) == 50

    def test_cache_limit_enforced_when_over_limit(self):
        """Test that oldest entries are evicted when cache exceeds limit."""
        # Create mock visualizer with cache exceeding limit
        viz = Mock()
        # Create cache with MAX + 20 entries
        viz.mpc_view_cache = {
            f"key_{i}": f"value_{i}" for i in range(MAX_VIEW_MODEL_CACHE_SIZE + 20)
        }

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Enforce limit
        pipeline._enforce_view_model_cache_limit()

        # Cache should be reduced to MAX_VIEW_MODEL_CACHE_SIZE
        assert len(viz.mpc_view_cache) == MAX_VIEW_MODEL_CACHE_SIZE

    def test_cache_limit_evicts_oldest_entries(self):
        """Test that oldest (first inserted) entries are removed."""
        # Create mock visualizer
        viz = Mock()
        # Create cache with entries in order (key_0 is oldest)
        viz.mpc_view_cache = {
            f"key_{i}": f"value_{i}" for i in range(MAX_VIEW_MODEL_CACHE_SIZE + 10)
        }

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Enforce limit
        pipeline._enforce_view_model_cache_limit()

        # Verify oldest entries (key_0 through key_9) were evicted
        for i in range(10):
            assert f"key_{i}" not in viz.mpc_view_cache

        # Verify newest entries remain
        for i in range(10, MAX_VIEW_MODEL_CACHE_SIZE + 10):
            assert f"key_{i}" in viz.mpc_view_cache

    def test_cache_limit_with_exact_limit(self):
        """Test that cache at exactly the limit is not modified."""
        # Create mock visualizer with cache at exact limit
        viz = Mock()
        viz.mpc_view_cache = {f"key_{i}": f"value_{i}" for i in range(MAX_VIEW_MODEL_CACHE_SIZE)}

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Enforce limit
        pipeline._enforce_view_model_cache_limit()

        # Cache should remain unchanged
        assert len(viz.mpc_view_cache) == MAX_VIEW_MODEL_CACHE_SIZE

    def test_cache_limit_with_no_cache(self):
        """Test that enforcement handles missing cache gracefully."""
        # Create mock visualizer without mpc_view_cache
        viz = Mock(spec=[])  # Empty spec means no attributes

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Enforce limit (should not raise exception)
        pipeline._enforce_view_model_cache_limit()

    def test_cache_limit_with_none_cache(self):
        """Test that enforcement handles None cache gracefully."""
        # Create mock visualizer with None cache
        viz = Mock()
        viz.mpc_view_cache = None

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Enforce limit (should not raise exception)
        pipeline._enforce_view_model_cache_limit()

    def test_cache_limit_preserves_insertion_order(self):
        """Test that LRU eviction removes least recently used entries."""
        # Create mock visualizer
        viz = Mock()
        # Create ordered cache
        viz.mpc_view_cache = {}
        for i in range(MAX_VIEW_MODEL_CACHE_SIZE + 5):
            viz.mpc_view_cache[f"key_{i}"] = f"value_{i}"

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Enforce limit
        pipeline._enforce_view_model_cache_limit()

        # Verify remaining keys are in order
        remaining_keys = list(viz.mpc_view_cache.keys())
        expected_keys = [f"key_{i}" for i in range(5, MAX_VIEW_MODEL_CACHE_SIZE + 5)]
        assert remaining_keys == expected_keys

    def test_cache_limit_multiple_evictions(self):
        """Test that repeated evictions work correctly."""
        # Create mock visualizer
        viz = Mock()
        viz.mpc_view_cache = {f"key_{i}": f"value_{i}" for i in range(50)}

        # Create pipeline
        pipeline = FramePipeline(viz)

        # Add more entries to exceed limit
        for i in range(50, MAX_VIEW_MODEL_CACHE_SIZE + 30):
            viz.mpc_view_cache[f"key_{i}"] = f"value_{i}"

        # Enforce limit
        pipeline._enforce_view_model_cache_limit()

        # Should be at limit
        assert len(viz.mpc_view_cache) == MAX_VIEW_MODEL_CACHE_SIZE

        # Add more entries
        for i in range(MAX_VIEW_MODEL_CACHE_SIZE + 30, MAX_VIEW_MODEL_CACHE_SIZE + 50):
            viz.mpc_view_cache[f"key_{i}"] = f"value_{i}"

        # Enforce limit again
        pipeline._enforce_view_model_cache_limit()

        # Should still be at limit
        assert len(viz.mpc_view_cache) == MAX_VIEW_MODEL_CACHE_SIZE
