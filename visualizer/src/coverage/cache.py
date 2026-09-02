"""Coverage mesh caching for performance optimization."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal, Optional

import numpy as np

from shared.logging import get_logger

logger = get_logger("orchav")

# Gaussian smoothing sigma values (in grid-cell units).
# A 5 m grid with sigma=1.5 smooths across ~15 m; sigma=3.0 across ~30 m.
COVERAGE_SMOOTH_SIGMA_LINEAR = 1.5
COVERAGE_SMOOTH_SIGMA_CUBIC = 3.0


class CoverageMeshCache:
    """
    Handles caching of coverage mesh computations to avoid expensive regeneration.

    The cache stores pre-computed mesh data (vertices, triangles, colors) for different
    combinations of coverage parameters. This significantly improves performance when
    switching between heights or other coverage settings.
    """

    def __init__(self, max_cache_size: int = 50):
        """
        Initialize the coverage mesh cache.

        Args:
            max_cache_size: Maximum number of cached meshes to keep in memory
        """
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._max_cache_size = max_cache_size
        self._access_times: dict[str, float] = {}

    def get_mesh(
        self, cache_key: str, *, copy: bool = True
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get cached mesh data if available.

        Args:
            cache_key: Unique key identifying the mesh configuration
            copy: If True (default), return defensive copies. If False,
                return cached arrays directly (read-only-by-convention).

        Returns:
            Tuple of (vertices, triangles, colors) if cached, None otherwise
        """
        if cache_key in self._cache:
            self._access_times[cache_key] = time.time()
            self._cache_stats["hits"] += 1
            logger.debug(
                f"Coverage cache HIT for key: {cache_key[:16]}... (cache size: {len(self._cache)})"
            )
            vertices, triangles, colors, _ = self._cache[cache_key]
            if copy:
                return vertices.copy(), triangles.copy(), colors.copy()
            return vertices, triangles, colors
        else:
            self._cache_stats["misses"] += 1
            logger.debug(
                f"Coverage cache MISS for key: {cache_key[:16]}... (cache size: {len(self._cache)}, available keys: {[k[:8] for k in self._cache.keys()]})"
            )
            return None

    def put_mesh(
        self, cache_key: str, vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray
    ):
        """
        Store mesh data in cache.

        Args:
            cache_key: Unique key identifying the mesh configuration
            vertices: Mesh vertices array
            triangles: Mesh triangle indices array
            colors: Mesh colors array
        """
        if len(self._cache) >= self._max_cache_size:
            self._evict_lru()

        now = time.time()
        self._cache[cache_key] = (vertices.copy(), triangles.copy(), colors.copy(), now)
        self._access_times[cache_key] = now

        logger.debug(
            f"Coverage cache STORED for key: {cache_key[:16]}... (cache size: {len(self._cache)})"
        )

    def _evict_lru(self):
        """Evict least recently used cache entry."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])

        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._cache_stats["evictions"] += 1

        logger.debug(f"Coverage cache EVICTED key: {lru_key[:16]}...")

    def _compute_cache_key(
        self,
        coverage_data: dict,
        height_index: int,
        interpolation: str = "nearest",
    ) -> str:
        """Generate a unique cache key from coverage parameters.

        Opacity is intentionally excluded — it is applied as a cheap color
        transform after cache retrieval so that opacity changes never
        invalidate cached meshes.

        Args:
            coverage_data: Coverage data dictionary.
            height_index: Selected height index.
            interpolation: Interpolation method ('nearest', 'linear', 'cubic').

        Returns:
            Unique cache key string.
        """
        dataset_fingerprint = str(coverage_data.get("dataset_fingerprint") or "")
        if not dataset_fingerprint:
            values_3d = coverage_data.get("values_3d", None)
            values_shape = tuple(values_3d.shape) if isinstance(values_3d, np.ndarray) else None
            values_dtype = str(values_3d.dtype) if isinstance(values_3d, np.ndarray) else None
            values_addr = (
                int(values_3d.__array_interface__["data"][0])
                if isinstance(values_3d, np.ndarray)
                else None
            )
            dataset_source = (id(values_3d), values_shape, values_dtype, values_addr)
            cached_source = coverage_data.get("_dataset_fingerprint_source")
            cached_fingerprint = coverage_data.get("_dataset_fingerprint")
            if cached_fingerprint and cached_source == dataset_source:
                dataset_fingerprint = str(cached_fingerprint)
            else:
                dataset_meta = {
                    "grid_origin": np.asarray(coverage_data.get("grid_origin", [])).tolist(),
                    "grid_spacing": np.asarray(coverage_data.get("grid_spacing", [])).tolist(),
                    "grid_shape": np.asarray(coverage_data.get("grid_shape", [])).tolist(),
                    "value_min": coverage_data.get("value_min", None),
                    "value_max": coverage_data.get("value_max", None),
                    "metric_name": coverage_data.get("metric_name", None),
                    "heights": np.asarray(coverage_data.get("heights", [])).tolist(),
                    "values_shape": values_shape,
                    "values_dtype": values_dtype,
                    "values_addr": values_addr,
                }
                dataset_fingerprint = hashlib.md5(
                    str(sorted(dataset_meta.items())).encode()
                ).hexdigest()
                coverage_data["_dataset_fingerprint"] = dataset_fingerprint
                coverage_data["_dataset_fingerprint_source"] = dataset_source

        key_data = {
            "dataset_fingerprint": dataset_fingerprint,
            "metric_name": coverage_data.get("metric_name", None),
            "value_min": coverage_data.get("value_min", None),
            "value_max": coverage_data.get("value_max", None),
            "height_index": height_index,
            "interpolation": interpolation,
        }

        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self):
        """Clear all cached data."""
        cache_size = len(self._cache)
        self._cache.clear()
        self._access_times.clear()
        logger.info(f"Coverage cache cleared ({cache_size} entries removed)")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "evictions": self._cache_stats["evictions"],
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests,
        }

    def interpolate_coverage_values(
        self,
        values_2d: np.ndarray,
        interpolation: Literal["none", "nearest", "linear", "cubic"] = "none",
    ) -> np.ndarray:
        """Apply NaN-aware spatial smoothing to coverage values.

        Uses normalized convolution so that NaN cells (no coverage) do not
        bleed into valid neighbors.

        Args:
            values_2d: 2D array of coverage values for a single height slice.
            interpolation: ``"none"`` (raw), ``"linear"`` (light), or
                ``"cubic"`` (strong).  ``"nearest"`` is treated as ``"none"``
                for backwards compatibility.

        Returns:
            Smoothed values array (same shape as input, NaN mask preserved).
        """
        if interpolation in ("none", "nearest"):
            return values_2d

        sigma = {
            "linear": COVERAGE_SMOOTH_SIGMA_LINEAR,
            "cubic": COVERAGE_SMOOTH_SIGMA_CUBIC,
        }.get(interpolation)
        if sigma is None:
            return values_2d

        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            logger.warning("scipy not available for smoothing, returning raw values")
            return values_2d

        # Non-finite-aware normalized convolution:
        # 1. Replace invalid values with 0, build a weight mask (1=valid, 0=invalid)
        # 2. Smooth both arrays with the same kernel
        # 3. Divide smoothed values by smoothed weights (avoids NaN bleed)
        # 4. Restore the original invalid-cell mask as NaN
        invalid_mask = ~np.isfinite(values_2d)
        filled = np.where(invalid_mask, 0.0, values_2d)
        weights = (~invalid_mask).astype(np.float64)

        smoothed_values = gaussian_filter(filled, sigma=sigma)
        smoothed_weights = gaussian_filter(weights, sigma=sigma)

        # Avoid division by zero in fully-NaN regions
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(smoothed_weights > 1e-10, smoothed_values / smoothed_weights, 0.0)

        result[invalid_mask] = np.nan
        return result

    def log_stats(self):
        """Log current cache statistics."""
        stats = self.get_stats()
        logger.info(
            f"Coverage Cache Stats: {stats['cache_size']}/{stats['max_cache_size']} entries, "
            f"{stats['hit_rate_percent']:.1f}% hit rate ({stats['hits']}/{stats['total_requests']} hits)"
        )
