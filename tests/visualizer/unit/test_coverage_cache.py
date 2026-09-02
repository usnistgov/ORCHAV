"""Unit tests for coverage cache helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tests.visualizer.fixtures.sample_coverage_maps import create_sample_coverage_data
from visualizer.src.coverage.cache import CoverageMeshCache


@pytest.fixture
def coverage_data():
    return create_sample_coverage_data()


def test_cache_put_get_and_evict(coverage_data):
    cache = CoverageMeshCache(max_cache_size=1)
    key = cache._compute_cache_key(coverage_data, height_index=0)

    vertices = np.random.rand(4, 3).astype(np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.random.rand(4, 3).astype(np.float32)

    cache.put_mesh(key, vertices, triangles, colors)
    cached = cache.get_mesh(key)
    assert cached is not None
    v_cached, t_cached, c_cached = cached
    np.testing.assert_array_equal(v_cached, vertices)
    np.testing.assert_array_equal(t_cached, triangles)
    np.testing.assert_array_equal(c_cached, colors)

    # Inserting a new key should evict the previous one due to max_cache_size=1
    cache.put_mesh("other", vertices, triangles, colors)
    assert cache.get_mesh(key) is None


def test_cache_get_no_copy_returns_cached_arrays(coverage_data):
    cache = CoverageMeshCache(max_cache_size=2)
    key = cache._compute_cache_key(coverage_data, height_index=0)

    vertices = np.random.rand(4, 3).astype(np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.random.rand(4, 3).astype(np.float32)
    cache.put_mesh(key, vertices, triangles, colors)

    copied = cache.get_mesh(key)
    direct = cache.get_mesh(key, copy=False)
    assert copied is not None and direct is not None
    copied_vertices = copied[0]
    direct_vertices = direct[0]

    assert not np.shares_memory(copied_vertices, direct_vertices)
    assert np.shares_memory(direct_vertices, cache._cache[key][0])


def test_cache_key_respects_dataset_fingerprint(coverage_data):
    cache = CoverageMeshCache(max_cache_size=2)
    coverage_data["dataset_fingerprint"] = "dataset-v1"

    key_a = cache._compute_cache_key(coverage_data, 0, interpolation="nearest")
    # Mutate array content without changing fingerprint: key should stay stable.
    coverage_data["values_3d"][0, 0, 0] += 123.0
    key_b = cache._compute_cache_key(coverage_data, 0, interpolation="nearest")
    key_c = cache._compute_cache_key(coverage_data, 1, interpolation="nearest")
    key_d = cache._compute_cache_key(coverage_data, 0, interpolation="linear")

    assert key_a == key_b
    assert key_a != key_c
    assert key_a != key_d


def test_cache_stats(coverage_data):
    cache = CoverageMeshCache(max_cache_size=2)
    key_a = cache._compute_cache_key(coverage_data, 0)
    key_b = cache._compute_cache_key(coverage_data, 1)
    dummy = (
        np.random.rand(4, 3).astype(np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
        np.random.rand(4, 3).astype(np.float32),
    )

    cache.put_mesh(key_a, *dummy)
    cache.get_mesh(key_a)  # hit
    cache.get_mesh("missing")  # miss
    cache.put_mesh(key_b, *dummy)

    stats = cache.get_stats()
    assert stats["cache_size"] == 2
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["total_requests"] == 2


def test_interpolation_modes():
    cache = CoverageMeshCache()
    values = np.arange(9, dtype=np.float64).reshape(3, 3)

    # "none" and legacy "nearest" return the original array unchanged
    np.testing.assert_array_equal(cache.interpolate_coverage_values(values, "none"), values)
    np.testing.assert_array_equal(cache.interpolate_coverage_values(values, "nearest"), values)

    # Linear / cubic smoothing produce different (but same-shape) results
    linear = cache.interpolate_coverage_values(values, "linear")
    assert linear.shape == values.shape
    # The output should be smoothed: different from the raw grid
    assert not np.array_equal(linear, values)

    cubic = cache.interpolate_coverage_values(values, "cubic")
    assert cubic.shape == values.shape
    assert not np.array_equal(cubic, values)


def test_interpolation_nan_aware():
    """Smoothing must not shrink the valid coverage area."""
    cache = CoverageMeshCache()
    values = np.full((5, 5), 1.0)
    values[0, :] = np.nan  # Top row is NaN (no coverage)

    smoothed = cache.interpolate_coverage_values(values, "linear")
    # Valid cells must remain valid (not become NaN)
    assert not np.any(np.isnan(smoothed[1:, :]))
    # NaN cells must stay NaN
    assert np.all(np.isnan(smoothed[0, :]))


def test_interpolation_treats_infinities_as_missing_without_neighbor_bleed():
    cache = CoverageMeshCache()
    values = np.ones((5, 5), dtype=np.float64)
    values[2, 2] = np.inf

    smoothed = cache.interpolate_coverage_values(values, "linear")

    assert np.isnan(smoothed[2, 2])
    finite_positions = np.ones_like(values, dtype=bool)
    finite_positions[2, 2] = False
    assert np.isfinite(smoothed[finite_positions]).all()
