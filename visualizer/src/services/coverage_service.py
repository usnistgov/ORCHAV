"""Visualizer-side coverage loading, metric selection, and mesh caching.

Coverage HDF5 v2 uses compact storage: it keeps canonical path gain and a small
set of materialized derived layers, while the visualizer exposes a larger
logical metric menu. This service keeps loaded scenario metadata in memory and
derives the selected metric layer lazily from the file when the user switches
metrics.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, cast

import h5py
import numpy as np

from shared.coverage.hdf5 import CoverageHDF5Reader
from shared.coverage.schema import (
    COVERAGE_CANONICAL_VALUE_METRICS,
    COVERAGE_FRAME_GENERATION_ID_ATTR,
    COVERAGE_FRAME_SET_ID_ATTR,
    coverage_available_metrics,
    coverage_metric_base,
    normalise_coverage_tx_names,
    validate_coverage_hdf5_contract,
)
from shared.logging import get_logger

from ..coverage.analysis import coverage_metric_color_scale, coverage_metric_valid_mask
from ..coverage.cache import CoverageMeshCache
from ..io.packed_frame_payload import visualizer_frame_provider
from ..renderers.protocol import renderer_capabilities

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.coverage")

DEFAULT_COVERAGE_OPACITY = 1.0
DEFAULT_COVERAGE_INTERPOLATION = "none"
DEFAULT_COVERAGE_ISOLINE_COUNT = 6
DEFAULT_COVERAGE_HEIGHT_ANIMATION_SPEED = 3

_ISOLINE_CACHE_POLICY = "marching-squares-v1:z-offset=0.06+level-index*0.001"

_SINGLE_TX_REDUNDANT_METRICS = frozenset(
    {
        "best_path_loss_db",
        "best_rss_dbm",
        "sum_rss_dbm",
        "serving_tx",
        "tx_margin_db",
    }
)
_SINGLE_TX_PRIMARY_EQUIVALENTS = {
    "path_gain_linear": "path_gain_linear",
    "path_gain_db": "path_gain_db",
    "path_loss_db": "path_loss_db",
    "rss_w": "rss_w",
    "rss_dbm": "rss_dbm",
    "serving_path_gain_linear": "path_gain_linear",
    "best_path_loss_db": "path_loss_db",
    "best_rss_dbm": "rss_dbm",
    "sum_rss_dbm": "rss_dbm",
    "sinr_linear": "sinr_linear",
    "sinr_db": "sinr_db",
}


def _h5_has(parent: Any, name: str) -> bool:
    """Return whether an HDF5 group/dataset contains *name*."""
    return name in parent


def _h5_child(parent: Any, name: str) -> Any:
    """Return a named HDF5 child while keeping call sites schema-oriented."""
    return parent[name]


def _h5_array(parent: Any, name: str, dtype: Any = None) -> np.ndarray:
    """Read a named HDF5 dataset into a NumPy array."""
    return np.asarray(parent[name], dtype=dtype)


def _identity_text(value: Any) -> str | None:
    """Normalize a provider or HDF5 identity value to nonempty text."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    return text or None


def _coverage_metric_menu(
    metrics: Sequence[str],
    *,
    tx_count: int,
    tx_names: Sequence[str],
) -> list[str]:
    """Return useful visualizer metrics for the available transmitter count.

    Best/sum/serving metrics collapse to an existing per-TX value or cannot
    convey a comparison when exactly one transmitter exists. The visualizer
    omits those redundant choices while leaving the stored dataset unchanged.
    """
    if tx_count != 1:
        return [str(metric) for metric in metrics]
    tx_name = str(tx_names[0])
    result: list[str] = []
    for metric in metrics:
        metric = str(metric)
        base, _ = coverage_metric_base(metric)
        equivalent_base = _SINGLE_TX_PRIMARY_EQUIVALENTS.get(base)
        if equivalent_base is not None:
            metric = f"{equivalent_base}/{tx_name}"
        elif base in _SINGLE_TX_REDUNDANT_METRICS:
            continue
        if metric not in result:
            result.append(metric)
    return result


def _preferred_coverage_metric(
    stored_primary: str,
    available_metrics: Sequence[str],
    *,
    tx_names: Sequence[str],
) -> str:
    """Resolve the stored primary to one useful visualizer metric."""
    available = [str(metric) for metric in available_metrics]
    if len(tx_names) == 1:
        base, _ = coverage_metric_base(stored_primary)
        equivalent_base = _SINGLE_TX_PRIMARY_EQUIVALENTS.get(base)
        if equivalent_base is not None:
            equivalent = f"{equivalent_base}/{tx_names[0]}"
            if equivalent in available:
                return equivalent
    if stored_primary in available:
        return stored_primary
    if "best_path_loss_db" in available:
        return "best_path_loss_db"
    return next(iter(available), "path_gain_linear")


def _uses_stored_primary_range(
    stored_primary: str,
    preferred_metric: str,
    *,
    tx_names: Sequence[str],
) -> bool:
    """Return whether the stored primary range applies to the displayed metric.

    A one-transmitter aggregate such as ``best_path_loss_db`` is identical to
    its per-transmitter view.  The metric menu deliberately replaces the
    redundant aggregate name, but the root HDF5 range remains authoritative
    and avoids introducing float32 round-off while rescanning derived slices.
    """

    if preferred_metric == stored_primary:
        return True
    if len(tx_names) != 1:
        return False
    stored_base, _ = coverage_metric_base(stored_primary)
    preferred_base, preferred_selector = coverage_metric_base(preferred_metric)
    equivalent_base = _SINGLE_TX_PRIMARY_EQUIVALENTS.get(stored_base)
    return equivalent_base == preferred_base and preferred_selector == str(tx_names[0])


def _metric_range_values(values: Any, metric_name: str) -> np.ndarray:
    """Return finite values that can participate in a metric's color range."""
    array = np.asarray(values)
    return array[coverage_metric_valid_mask(array, metric_name)]


class CoverageService:
    """Visualizer-side service for coverage overlay loading, metrics, and cache state."""

    def __init__(self, max_cache_size: int = 20):
        """Initialize the coverage mesh cache with a bounded entry count."""
        self.cache = CoverageMeshCache(max_cache_size=max_cache_size)
        self._metric_slice_cache_max_size = max(1, int(max_cache_size))
        self._isoline_cache_max_size = max(1, int(max_cache_size))
        self._isoline_cache: OrderedDict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]],
        ] = OrderedDict()
        self._isoline_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_mesh(self, cache_key: str, *, copy: bool = True) -> Optional[Tuple[Any, Any, Any]]:
        """Return a cached coverage mesh tuple for a renderer payload."""
        return self.cache.get_mesh(cache_key, copy=copy)

    def put_mesh(
        self,
        cache_key: str,
        vertices: Any,
        triangles: Any,
        colors: Any,
    ) -> None:
        """Store a derived coverage mesh under its cache key."""
        self.cache.put_mesh(cache_key, vertices, triangles, colors)

    def clear(self) -> None:
        """Clear cached coverage meshes and isoline geometry."""
        self.cache.clear_cache()
        self._isoline_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return mesh and isoline cache statistics for diagnostics."""
        stats = self.cache.get_stats()
        isoline_requests = self._isoline_cache_stats["hits"] + self._isoline_cache_stats["misses"]
        isoline_hit_rate = (
            self._isoline_cache_stats["hits"] / isoline_requests * 100.0
            if isoline_requests
            else 0.0
        )
        stats.update(
            {
                "isoline_cache_size": len(self._isoline_cache),
                "isoline_max_cache_size": self._isoline_cache_max_size,
                "isoline_hits": self._isoline_cache_stats["hits"],
                "isoline_misses": self._isoline_cache_stats["misses"],
                "isoline_evictions": self._isoline_cache_stats["evictions"],
                "isoline_hit_rate_percent": isoline_hit_rate,
                "isoline_total_requests": isoline_requests,
            }
        )
        return stats

    def compute_cache_key(
        self,
        coverage_data: dict,
        height_index: int,
        interpolation: str = "nearest",
    ) -> str:
        """Expose cache key generation for callers."""
        return self.cache._compute_cache_key(
            coverage_data,
            height_index,
            interpolation,
        )

    @staticmethod
    def compute_isoline_cache_key(
        mesh_key: str,
        levels: Sequence[float],
        z_level: float,
    ) -> str:
        """Return a stable key for isolines derived from one displayed slice."""
        payload = {
            "mesh_key": str(mesh_key),
            "levels": [float(level) for level in levels],
            "z_level": float(z_level),
            "policy": _ISOLINE_CACHE_POLICY,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.md5(encoded).hexdigest()

    def get_isolines(
        self,
        cache_key: str,
        *,
        copy: bool = False,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]]:
        """Return cached isoline buffers, updating LRU order and hit statistics."""
        cached = self._isoline_cache.get(cache_key)
        if cached is None:
            self._isoline_cache_stats["misses"] += 1
            return None
        self._isoline_cache.move_to_end(cache_key)
        self._isoline_cache_stats["hits"] += 1
        if not copy:
            return cached
        points, lines, colors, levels = cached
        return points.copy(), lines.copy(), colors.copy(), levels

    def put_isolines(
        self,
        cache_key: str,
        points: np.ndarray,
        lines: np.ndarray,
        colors: np.ndarray,
        levels: Sequence[float],
    ) -> None:
        """Store isoline buffers in a bounded least-recently-used cache."""
        self._isoline_cache[cache_key] = (
            np.asarray(points).copy(),
            np.asarray(lines).copy(),
            np.asarray(colors).copy(),
            tuple(float(level) for level in levels),
        )
        self._isoline_cache.move_to_end(cache_key)
        while len(self._isoline_cache) > self._isoline_cache_max_size:
            self._isoline_cache.popitem(last=False)
            self._isoline_cache_stats["evictions"] += 1

    def interpolate_values(
        self, values_2d: Any, interpolation: Literal["nearest", "linear", "cubic"]
    ) -> Any:
        """Delegate interpolation helpers to the underlying cache."""
        return self.cache.interpolate_coverage_values(values_2d, interpolation)

    def log_stats(self) -> None:
        """Log current cache statistics."""
        self.cache.log_stats()
        stats = self.stats()
        logger.info(
            "Coverage isoline cache: %d/%d entries, %.1f%% hit rate (%d requests)",
            stats["isoline_cache_size"],
            stats["isoline_max_cache_size"],
            stats["isoline_hit_rate_percent"],
            stats["isoline_total_requests"],
        )

    def select_metric_layer(self, coverage_data: dict[str, Any], metric_name: str) -> None:
        """Switch the active metric layer in a loaded v2 coverage dataset."""
        self._select_metric_layer(coverage_data, metric_name)

    def select_height_layer(
        self,
        coverage_data: dict[str, Any],
        height_index: int,
    ) -> None:
        """Materialize one file-backed height for the active metric.

        Legacy in-memory coverage dictionaries already contain every height and
        therefore need no activation step. File-backed v2 dictionaries keep
        only the selected slice in ``values_3d`` so ordinary height changes do
        not grow memory with the number of coverage levels.
        """
        if not coverage_data.get("coverage_file"):
            return
        metric_name = str(coverage_data.get("metric_name", ""))
        if not metric_name:
            raise KeyError("coverage metric_name is missing")
        index = self._normalise_height_index(coverage_data, height_index)
        coverage_data["values_3d"] = self._metric_layer_at_height(
            coverage_data,
            metric_name,
            index,
        )
        coverage_data["_active_height_index"] = index

    def metric_layer_at_height(
        self,
        coverage_data: dict[str, Any],
        metric_name: str,
        height_index: int,
    ) -> np.ndarray:
        """Return one metric slice as ``(1, y, x)`` without changing selection."""
        index = self._normalise_height_index(coverage_data, height_index)
        if coverage_data.get("coverage_file"):
            return self._metric_layer_at_height(coverage_data, metric_name, index)

        layers = coverage_data.get("metric_layers", {}) or {}
        values = layers.get(metric_name)
        if values is None and metric_name == coverage_data.get("metric_name"):
            values = coverage_data.get("values_3d")
        array = np.asarray(values)
        if array.ndim != 3:
            raise ValueError(
                f"coverage metric {metric_name!r} must have shape (height, y, x), "
                f"got {array.shape}"
            )
        if index >= array.shape[0]:
            raise IndexError(
                f"coverage height index {index} is outside the loaded metric "
                f"range 0..{max(array.shape[0] - 1, 0)}"
            )
        return np.asarray(array[index : index + 1], dtype=np.float32)

    @staticmethod
    def _decode_string_array(values: Any) -> list[str]:
        """Decode HDF5 byte/string arrays into Python strings."""
        arr = np.asarray(values)
        result: list[str] = []
        for item in arr.tolist():
            if isinstance(item, bytes):
                result.append(item.decode("utf-8"))
            else:
                result.append(str(item))
        return result

    @staticmethod
    def _load_json_attr(attrs: Any, key: str, default: Any) -> Any:
        """Load a JSON-encoded HDF5 attribute with a safe default."""
        value = attrs.get(key)
        if value is None:
            return default
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        try:
            return json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return default

    def _load_v2_coverage_hdf5(self, coverage_file: Path) -> dict[str, Any]:
        """Load coverage schema v2 HDF5 into the visualizer's coverage dict.

        This reads grid/device metadata eagerly but leaves ``metric_layers``
        mostly empty. Individual metric arrays can be large, so
        ``_select_metric_layer`` loads or derives only the active layer.
        """
        with h5py.File(coverage_file, "r") as f:
            schema_version = int(f.attrs.get("coverage_schema_version", 0))
            validate_coverage_hdf5_contract(
                schema_version,
                f.attrs.get("coverage_storage_layout"),
            )

            grid = _h5_child(f, "grid")
            tx_group = _h5_child(f, "tx")
            rx_group = _h5_child(f, "rx")
            values_group = _h5_child(f, "values")
            metadata_group = _h5_child(f, "metadata") if _h5_has(f, "metadata") else None
            derived_group = _h5_child(f, "derived") if _h5_has(f, "derived") else None
            shape_xyz = _h5_array(grid, "shape_xyz", np.int32)
            heights = _h5_array(grid, "heights_m", np.float32).tolist()
            origin = _h5_array(grid, "origin", np.float32)
            spacing = _h5_array(grid, "spacing_xy", np.float32)

            path_gain_shape = _h5_child(values_group, "path_gain_linear").shape
            tx_count = int(path_gain_shape[2]) if len(path_gain_shape) >= 3 else 0
            tx_names = normalise_coverage_tx_names(
                self._decode_string_array(_h5_child(tx_group, "names")[()]),
                tx_count,
            )
            rx_names = self._decode_string_array(_h5_child(rx_group, "names")[()])

            metrics_store = (
                self._load_json_attr(metadata_group.attrs, "metrics_store", [])
                if metadata_group is not None
                else []
            )
            metrics_derived = (
                self._load_json_attr(metadata_group.attrs, "metrics_derived", [])
                if metadata_group is not None
                else []
            )
            materialized_derived = sorted(derived_group.keys()) if derived_group is not None else []
            available_metrics = (
                self._load_json_attr(metadata_group.attrs, "available_metrics", [])
                if metadata_group is not None
                else []
            )
            if not available_metrics:
                available_metrics = coverage_available_metrics(
                    tx_names=tx_names,
                    tx_count=tx_count,
                    metrics_store=metrics_store or COVERAGE_CANONICAL_VALUE_METRICS,
                    metrics_derived=metrics_derived,
                    primary_metric=str(f.attrs.get("metric_name", "best_path_loss_db")),
                    materialized_values=COVERAGE_CANONICAL_VALUE_METRICS,
                    materialized_derived=materialized_derived,
                )
            available_metrics = _coverage_metric_menu(
                available_metrics,
                tx_count=tx_count,
                tx_names=tx_names,
            )
            stored_primary_metric = str(f.attrs.get("metric_name", "best_path_loss_db"))
            preferred_metric = _preferred_coverage_metric(
                stored_primary_metric,
                available_metrics,
                tx_names=tx_names,
            )
            metric_ranges: dict[str, tuple[float, float]] = {}
            if _uses_stored_primary_range(
                stored_primary_metric,
                preferred_metric,
                tx_names=tx_names,
            ):
                try:
                    range_min = float(f.attrs["value_min"])
                    range_max = float(f.attrs["value_max"])
                    valid_range = (
                        np.isfinite(range_min) and np.isfinite(range_max) and range_min <= range_max
                    )
                    if coverage_metric_color_scale(preferred_metric) == "logarithmic":
                        valid_range = valid_range and range_min > 0 and range_max > 0
                    if valid_range:
                        metric_ranges[preferred_metric] = (range_min, range_max)
                except (KeyError, TypeError, ValueError):
                    pass

            data = {
                "schema_version": schema_version,
                "coverage_file": str(coverage_file),
                "grid_origin": origin,
                "grid_spacing": spacing,
                "grid_shape": shape_xyz,
                "heights": heights,
                "tx_positions": _h5_array(tx_group, "positions_m", np.float32),
                "rx_positions": _h5_array(rx_group, "positions_m", np.float32),
                "tx_power_dbm": _h5_array(tx_group, "powers_dbm", np.float32),
                "tx_names": tx_names,
                "rx_names": rx_names,
                # Keep metric arrays lazy. ``available_metrics`` is the menu the
                # UI can present; file-backed raw slices live in the bounded
                # private LRU below. ``metric_layers`` remains the legacy
                # in-memory layer mapping.
                "metric_layers": {},
                # Raw file-backed slices use a bounded LRU separate from the
                # mesh cache. The root range belongs to the primary metric and
                # keeps its colors stable without reading every height.
                "_metric_slice_cache": OrderedDict(),
                "_metric_ranges": metric_ranges,
                "_active_height_index": 0,
                "available_metrics": list(available_metrics),
                "metric_name": preferred_metric,
                "metadata": {
                    "tx_mode": (
                        metadata_group.attrs.get("tx_mode", "per_tx")
                        if metadata_group is not None
                        else "per_tx"
                    ),
                    "schema_version": schema_version,
                    "metrics_store": metrics_store,
                    "metrics_derived": metrics_derived,
                    "materialized_derived": materialized_derived,
                    "noise_power_w": (
                        float(metadata_group.attrs.get("noise_power_w", 0.0))
                        if metadata_group is not None
                        else 0.0
                    ),
                },
            }
            self._select_metric_layer(data, preferred_metric)
            return data

    @staticmethod
    def _active_frame_identity(visualizer: OrchavVisualizer) -> tuple[str | None, str | None]:
        """Return generation and frame-set identities exposed by the active provider."""
        provider = visualizer_frame_provider(visualizer)
        if provider is None:
            return None, None
        try:
            info = provider.info
        except (ConnectionError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError):
            logger.debug("Could not read active frame-provider identity", exc_info=True)
            return None, None
        return (
            _identity_text(getattr(info, "generation_id", None)),
            _identity_text(getattr(info, "frame_set_id", None)),
        )

    def _coverage_matches_active_frames(
        self,
        coverage_file: Path,
        visualizer: OrchavVisualizer,
    ) -> bool:
        """Return whether coverage belongs to the active identified frame set.

        Scene-only and non-identifying providers cannot establish a binding, so
        their coverage remains loadable. Once a provider supplies an identity,
        every supplied value must also be present and equal in the coverage file.
        """
        generation_id, frame_set_id = self._active_frame_identity(visualizer)
        expected = {
            COVERAGE_FRAME_GENERATION_ID_ATTR: generation_id,
            COVERAGE_FRAME_SET_ID_ATTR: frame_set_id,
        }
        expected = {name: value for name, value in expected.items() if value is not None}
        if not expected:
            return True

        with h5py.File(coverage_file, "r") as coverage_h5:
            actual = {name: _identity_text(coverage_h5.attrs.get(name)) for name in expected}
        differences = [
            f"{name}: expected {expected_value!r}, found {actual[name]!r}"
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        ]
        if not differences:
            return True

        logger.warning(
            "Ignoring coverage map %s because its frame binding does not match "
            "the active frame provider (%s). Regenerate coverage for this frame set.",
            coverage_file,
            "; ".join(differences),
        )
        return False

    def _load_metric_layer_from_file(
        self,
        coverage_data: dict[str, Any],
        metric_name: str,
        *,
        height_index: int,
    ) -> np.ndarray:
        """Load one metric height through the shared selective HDF5 reader."""
        coverage_file = coverage_data.get("coverage_file")
        if not coverage_file:
            raise KeyError(metric_name)
        return (
            CoverageHDF5Reader(Path(str(coverage_file)))
            .read_metric(metric_name, height_indices=height_index)
            .astype(
                np.float32,
                copy=False,
            )
        )

    @staticmethod
    def _normalise_height_index(
        coverage_data: dict[str, Any],
        height_index: int,
    ) -> int:
        """Validate one logical coverage-height index."""
        heights = coverage_data.get("heights")
        height_count = len(heights) if heights is not None else 0
        if height_count <= 0:
            grid_shape = np.asarray(coverage_data.get("grid_shape", []), dtype=np.int64)
            height_count = int(grid_shape[2]) if grid_shape.size >= 3 else 0
        index = int(height_index)
        if index < 0 or index >= height_count:
            raise IndexError(
                f"coverage height index {index} is outside the valid range "
                f"0..{max(height_count - 1, 0)}"
            )
        return index

    def _metric_layer_at_height(
        self,
        coverage_data: dict[str, Any],
        metric_name: str,
        height_index: int,
    ) -> np.ndarray:
        """Return one file-backed metric slice through a bounded raw-data LRU."""
        cache = coverage_data.get("_metric_slice_cache")
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            coverage_data["_metric_slice_cache"] = cache
        key = (str(metric_name), int(height_index))
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

        layer = self._load_metric_layer_from_file(
            coverage_data,
            metric_name,
            height_index=height_index,
        )
        if layer.ndim != 3 or layer.shape[0] != 1:
            raise ValueError(
                f"coverage metric {metric_name!r} height {height_index} must "
                f"have shape (1, y, x), got {layer.shape}"
            )
        cache[key] = layer
        cache.move_to_end(key)
        while len(cache) > self._metric_slice_cache_max_size:
            cache.popitem(last=False)
        return layer

    def _metric_range_from_file(
        self,
        coverage_data: dict[str, Any],
        metric_name: str,
        *,
        selected_height_index: int,
        selected_layer: np.ndarray,
    ) -> tuple[float, float]:
        """Return a stable all-height metric range with bounded peak memory.

        Current coverage-v2 files store the primary metric's range at the root.
        For a newly selected derived metric, scan one height at a time and cache
        only the two extrema. This preserves cross-height colors without
        materializing the complete metric tensor.
        """
        ranges = coverage_data.get("_metric_ranges")
        if not isinstance(ranges, dict):
            ranges = {}
            coverage_data["_metric_ranges"] = ranges
        cached = ranges.get(metric_name)
        if cached is not None:
            return float(cached[0]), float(cached[1])

        minimum = np.inf
        maximum = -np.inf
        heights = coverage_data.get("heights")
        height_count = len(heights) if heights is not None else 0
        for height_index in range(height_count):
            layer = (
                selected_layer
                if height_index == selected_height_index
                else self._load_metric_layer_from_file(
                    coverage_data,
                    metric_name,
                    height_index=height_index,
                )
            )
            finite = _metric_range_values(layer, metric_name)
            if finite.size:
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            minimum, maximum = 0.0, 1.0
        result = (float(minimum), float(maximum))
        ranges[metric_name] = result
        return result

    def _select_metric_layer(self, coverage_data: dict[str, Any], metric_name: str) -> None:
        """Set the active coverage layer, falling back to another advertised metric."""
        if coverage_data.get("coverage_file"):
            self._select_file_metric_layer(coverage_data, metric_name)
            return

        layers = coverage_data.get("metric_layers", {}) or {}
        if metric_name not in layers:
            logger.warning("Coverage metric %s is not loaded in memory", metric_name)
            fallback_metrics = [
                candidate
                for candidate in coverage_data.get("available_metrics", [])
                if candidate != metric_name
            ]
            for candidate in fallback_metrics:
                if candidate in layers:
                    metric_name = candidate
                    break
        if metric_name not in layers and layers:
            metric_name = next(iter(layers))
        values_3d = np.asarray(layers.get(metric_name, np.empty((0, 0, 0))), dtype=np.float32)
        valid = _metric_range_values(values_3d, metric_name)
        coverage_data["metric_name"] = metric_name
        coverage_data["values_3d"] = values_3d
        base_name, _ = coverage_metric_base(metric_name)
        if base_name == "serving_tx":
            finite_serving = valid[valid >= 0]
            inferred_tx_count = int(finite_serving.max()) + 1 if finite_serving.size else 0
            tx_count = max(len(coverage_data.get("tx_names", []) or []), inferred_tx_count)
            coverage_data["serving_tx_count"] = tx_count
            coverage_data["value_min"] = 0.0
            coverage_data["value_max"] = float(max(tx_count - 1, 0))
        else:
            coverage_data["value_min"] = float(valid.min()) if valid.size else 0.0
            coverage_data["value_max"] = float(valid.max()) if valid.size else 1.0

    def _select_file_metric_layer(
        self,
        coverage_data: dict[str, Any],
        metric_name: str,
    ) -> None:
        """Activate one logical metric while retaining only its selected height."""
        height_index = self._normalise_height_index(
            coverage_data,
            int(coverage_data.get("_active_height_index", 0)),
        )
        candidates = [
            metric_name,
            *[
                candidate
                for candidate in coverage_data.get("available_metrics", [])
                if candidate != metric_name
            ],
        ]
        selected_metric: str | None = None
        selected_layer: np.ndarray | None = None
        first_error: Exception | None = None
        for candidate in candidates:
            try:
                selected_layer = self._metric_layer_at_height(
                    coverage_data,
                    candidate,
                    height_index,
                )
                selected_metric = str(candidate)
                break
            except (KeyError, OSError, ValueError) as exc:
                if first_error is None:
                    first_error = exc
                continue

        if selected_metric is None or selected_layer is None:
            if first_error is not None:
                raise ValueError(
                    f"no advertised coverage metric could be loaded: {first_error}"
                ) from first_error
            raise ValueError("coverage file advertises no usable metrics")
        if selected_metric != metric_name:
            logger.warning(
                "Could not load coverage metric %s; using %s",
                metric_name,
                selected_metric,
            )

        coverage_data["metric_name"] = selected_metric
        coverage_data["values_3d"] = selected_layer
        coverage_data["_active_height_index"] = height_index
        base_name, _ = coverage_metric_base(selected_metric)
        if base_name == "serving_tx":
            tx_count = len(coverage_data.get("tx_names", []) or [])
            coverage_data["serving_tx_count"] = tx_count
            coverage_data["value_min"] = 0.0
            coverage_data["value_max"] = float(max(tx_count - 1, 0))
            return

        value_min, value_max = self._metric_range_from_file(
            coverage_data,
            selected_metric,
            selected_height_index=height_index,
            selected_layer=selected_layer,
        )
        coverage_data["value_min"] = value_min
        coverage_data["value_max"] = value_max

    @staticmethod
    def _set_coverage_data_available(viz: Any, available: bool) -> None:
        """Synchronize the conditional Coverage tab with loaded-data state."""
        ui_manager = getattr(viz, "ui_manager", None)
        if ui_manager is None:
            return
        setter = getattr(ui_manager, "set_coverage_data_available", None)
        if callable(setter):
            setter(bool(available))
            return
        # Compatibility for shells created before the conditional Coverage tab.
        legacy_setter = getattr(ui_manager, "set_panel_visible", None)
        if callable(legacy_setter):
            legacy_setter("coverage", bool(available))

    def reset_runtime_state(self, visualizer: OrchavVisualizer) -> None:
        """Atomically clear scenario-owned coverage data and view state.

        Scenario transitions call this before scene/frame teardown. Stopping the
        height timer first prevents a hidden timer from mutating the next
        scenario while the remaining state and widgets are reset.
        """
        viz: Any = cast(Any, visualizer)
        ui_controller = getattr(viz, "ui_controller", None)
        reset_controller_state = getattr(ui_controller, "reset_coverage_runtime_state", None)
        if callable(reset_controller_state):
            reset_controller_state()

        viz.coverage_data = None
        viz.coverage_heights = []
        viz.coverage_height_index = 0
        viz.coverage_opacity = DEFAULT_COVERAGE_OPACITY
        viz.coverage_interpolation_method = DEFAULT_COVERAGE_INTERPOLATION
        viz.coverage_metric_name = None
        viz.coverage_threshold_enabled = False
        viz.coverage_threshold_value = None
        viz.coverage_threshold_mask_enabled = False
        viz.coverage_isolines_enabled = False
        viz.coverage_isoline_count = DEFAULT_COVERAGE_ISOLINE_COUNT
        viz._coverage_interpolation_dirty = False

        set_state = getattr(viz, "set_state", None)
        if callable(set_state) and getattr(viz, "app_state", None) is not None:
            set_state(show_coverage=False, coverage_height_index=0)

        self.clear()

        ui_manager = getattr(viz, "ui_manager", None)
        coverage_panel = None
        if ui_manager is not None:
            coverage_panel = getattr(ui_manager, "panels", {}).get("coverage")
        if coverage_panel is not None:
            reset_view_state = getattr(coverage_panel, "reset_view_state", None)
            if callable(reset_view_state):
                reset_view_state()
            update_status = getattr(coverage_panel, "update_coverage_status", None)
            if callable(update_status):
                update_status(False)
        self._set_coverage_data_available(viz, False)

    def load_coverage_map(
        self,
        scenario_root: Path,
        visualizer: OrchavVisualizer,
    ) -> bool:
        """Load the canonical coverage map for one scenario."""
        viz: Any = cast(Any, visualizer)
        self.reset_runtime_state(visualizer)
        try:
            coverage_file = Path(scenario_root) / "coverage" / "coverage_maps.h5"
            if not coverage_file.exists():
                logger.debug("No coverage map file found in scenario directory")
                return False

            if not self._coverage_matches_active_frames(coverage_file, visualizer):
                return False

            logger.info(f"Loading coverage map from: {coverage_file}")
            if hasattr(viz, "_set_status_message"):
                viz._set_status_message("Loading coverage map...")

            if coverage_file.suffix in [".h5", ".hdf5"]:
                viz.coverage_data = self._load_v2_coverage_hdf5(coverage_file)
                viz.coverage_heights = viz.coverage_data.get("heights", [])

            if viz.coverage_data is not None:
                try:
                    file_stat = coverage_file.stat()
                    viz.coverage_data["dataset_fingerprint"] = (
                        f"{coverage_file.resolve()}:{file_stat.st_mtime_ns}:{file_stat.st_size}"
                    )
                except OSError:
                    viz.coverage_data["dataset_fingerprint"] = str(coverage_file.resolve())

            if not viz.coverage_data:
                raise ValueError(f"Coverage file contained no usable data: {coverage_file}")

            if viz.coverage_data:
                cd = viz.coverage_data

                self._select_metric_layer(cd, cd.get("metric_name", "best_path_loss_db"))

                value_min = float(cd["value_min"])
                value_max = float(cd["value_max"])
                metric_name = str(cd["metric_name"])
                viz.coverage_metric_name = metric_name

                # Ensure grid_origin and grid_spacing are numpy arrays
                if "grid_origin" in cd:
                    cd["grid_origin"] = np.asarray(cd["grid_origin"])
                else:
                    cd["grid_origin"] = np.array([0.0, 0.0, 0.0])

                if "grid_spacing" in cd:
                    cd["grid_spacing"] = np.asarray(cd["grid_spacing"])
                elif "metadata_resolution" in cd:
                    res = cd["metadata_resolution"]
                    cd["grid_spacing"] = np.array(
                        [res[0], res[1], 1.0] if len(res) >= 2 else [1.0, 1.0, 1.0]
                    )
                else:
                    cd["grid_spacing"] = np.array([1.0, 1.0, 1.0])

                if "grid_shape" in cd:
                    cd["grid_shape"] = np.asarray(cd["grid_shape"])
                else:
                    cd["grid_shape"] = np.array([1, 1, 1])

                # Reconstruct 3D values from flat array if needed
                if "values_3d" not in cd and "values" in cd:
                    values_flat = cd["values"]
                    grid_shape = cd["grid_shape"]
                    heights = viz.coverage_heights

                    if len(grid_shape) >= 3 and grid_shape[2] > 0:
                        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
                    elif len(heights) > 0:
                        nz = len(heights)
                        total_points = len(values_flat)
                        points_per_slice = total_points // nz if nz > 0 else total_points
                        # Assume square grid for simplicity if not specified
                        nx = ny = int(np.sqrt(points_per_slice))
                    else:
                        # Single height
                        nz = 1
                        nx = ny = int(np.sqrt(len(values_flat)))

                    try:
                        # Reshape: flat -> (nz, ny, nx)
                        cd["values_3d"] = values_flat.reshape((nz, ny, nx))
                        cd["grid_shape"] = np.array([nx, ny, nz])
                        logger.debug(
                            f"Reshaped values {values_flat.shape} -> {cd['values_3d'].shape}"
                        )
                    except ValueError:
                        # Can't reshape - just store as is
                        cd["values_3d"] = values_flat[np.newaxis, np.newaxis, :]
                        logger.warning("Could not reshape values to 3D grid, using 1D fallback")

                # Log success
                grid_shape = cd["grid_shape"]
                if viz.coverage_heights:
                    height_info = (
                        f"heights={viz.coverage_heights[0]:.2f}...{viz.coverage_heights[-1]:.2f}m"
                        f" ({len(viz.coverage_heights)} levels)"
                    )
                else:
                    height_info = "single height"
                grid_info = "×".join(str(int(x)) for x in grid_shape[:2] if x > 0)
                logger.info(
                    f"Coverage map loaded: {metric_name} | Grid: {grid_info} | Range: {value_min:.1f} to {value_max:.1f} | {height_info}"
                )

                if hasattr(viz, "ui_manager") and "coverage" in viz.ui_manager.panels:
                    coverage_panel = viz.ui_manager.panels["coverage"]
                    supports_transparency = renderer_capabilities(viz.renderer).transparency
                    coverage_panel.update_coverage_status(
                        True,
                        viz.coverage_data,
                        supports_transparency=supports_transparency,
                    )
                    if hasattr(coverage_panel, "set_heights"):
                        coverage_panel.set_heights(viz.coverage_heights)
                        coverage_panel.set_height_index(viz.coverage_height_index)
                self._set_coverage_data_available(viz, True)

                if hasattr(viz, "_set_status_message"):
                    viz._set_status_message("Coverage map ready", 5000)
                return True

        except (OSError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Could not load coverage map: {e}")
            if hasattr(viz, "_set_status_message"):
                viz._set_status_message(f"Coverage map failed: {e}", 5000)
            self.reset_runtime_state(visualizer)
            return False
