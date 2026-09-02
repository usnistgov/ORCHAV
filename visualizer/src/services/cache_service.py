"""Cache ownership and typed invalidation for the visualizer data flow.

The visualizer has several cache layers with different lifetimes: raw frame
payloads, canonical MPC arrays, derived ViewModels, coverage meshes, target
geometry, and renderer frame state. ``CacheService`` centralizes invalidation so
callers can name what changed instead of clearing every cache blindly.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Optional

from shared.frames.loader import CacheManager
from shared.logging import get_logger

from ..cache.asset_cache_coordinator import (
    clear_static_asset_caches as clear_reusable_asset_caches,
)
from ..cache.asset_cache_coordinator import (
    collect_static_asset_cache_snapshot,
)
from ..config import MAX_FRAME_CACHE_SIZE
from ..services.base import BaseService
from ..services.target_asset_cache import TargetAssetCache

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.cache_service")


class CacheInvalidationScope(str, Enum):
    """Typed local cache invalidation scopes for visualizer-local data."""

    FRAME_DATA = "frame_data"
    FILTERS = "filters"
    TARGET_GEOMETRY = "target_geometry"
    MATERIALS_COLORS = "materials_colors"
    LABELS = "labels"
    CAMERA = "camera"
    STATIC_SCENE_GEOMETRY = "static_scene_geometry"
    MPC_RENDER_SETTINGS = "mpc_render_settings"
    ALL = "all"


def normalize_cache_invalidation_scopes(
    scopes: CacheInvalidationScope | str | Iterable[CacheInvalidationScope | str],
) -> set[CacheInvalidationScope]:
    """Normalize cache invalidation scope inputs to typed enum values."""
    if isinstance(scopes, (str, CacheInvalidationScope)):
        values: Iterable[CacheInvalidationScope | str] = (scopes,)
    else:
        values = scopes
    normalized: set[CacheInvalidationScope] = set()
    for value in values:
        if isinstance(value, CacheInvalidationScope):
            normalized.add(value)
        else:
            normalized.add(CacheInvalidationScope(str(value)))
    return normalized


def invalidate_visualizer_cache(
    visualizer: Any,
    scopes: CacheInvalidationScope | str | Iterable[CacheInvalidationScope | str],
    *,
    reason: str = "cache_reset",
) -> dict[str, int]:
    """Invalidate typed visualizer caches with a fallback for notebook/test doubles."""
    cache_service = getattr(visualizer, "cache_service", None)
    invalidate = getattr(cache_service, "invalidate", None)
    if callable(invalidate):
        return invalidate(scopes, reason=reason)

    normalized = normalize_cache_invalidation_scopes(scopes)
    result = {
        "frame_cache": 0,
        "view_model_cache": 0,
        "coverage_cache": 0,
        "target_cache": 0,
        "renderer_state": 0,
        "canonical_cache": 0,
    }

    view_model_scopes = {
        CacheInvalidationScope.FRAME_DATA,
        CacheInvalidationScope.FILTERS,
        CacheInvalidationScope.TARGET_GEOMETRY,
        CacheInvalidationScope.MATERIALS_COLORS,
        CacheInvalidationScope.LABELS,
        CacheInvalidationScope.STATIC_SCENE_GEOMETRY,
        CacheInvalidationScope.MPC_RENDER_SETTINGS,
        CacheInvalidationScope.ALL,
    }
    if normalized & view_model_scopes:
        cache = getattr(visualizer, "mpc_view_cache", None)
        if cache is not None and hasattr(cache, "clear"):
            try:
                result["view_model_cache"] = len(cache)
            except TypeError:
                result["view_model_cache"] = 0
            cache.clear()
        if hasattr(visualizer, "last_app_state"):
            visualizer.last_app_state = None

    if (
        CacheInvalidationScope.MATERIALS_COLORS in normalized
        or CacheInvalidationScope.ALL in normalized
    ):
        mpc_core = _get_mpc_core(visualizer)
        invalidate_materials = getattr(mpc_core, "invalidate_material_colors_cache", None)
        if callable(invalidate_materials):
            invalidate_materials()

    if CacheInvalidationScope.FRAME_DATA in normalized or CacheInvalidationScope.ALL in normalized:
        _invalidate_mpc_core_step(_get_mpc_core(visualizer), None)
    return result


def _get_mpc_core(owner: Any) -> Any:
    """Return the canonical MPC cache owner from an app or notebook object."""
    return getattr(owner, "mpc_core", None) or getattr(owner, "_mpc_core", None)


def _invalidate_mpc_core_step(mpc_core: Any, step: Optional[int]) -> None:
    """Invalidate MPCCore canonical data while tolerating simple test doubles."""
    invalidate_step = getattr(mpc_core, "invalidate_step", None)
    if not callable(invalidate_step):
        return
    try:
        invalidate_step(step=step)
    except TypeError:
        invalidate_step(step)


class CacheService(BaseService):
    """Own and clear local caches for frame data and derived render payloads."""

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        *,
        max_frame_cache_size: int = MAX_FRAME_CACHE_SIZE,
    ) -> None:
        """Initialize frame-cache state and preview/preload step tracking."""
        super().__init__()
        self.visualizer = visualizer
        self._default_frame_cache_size = max(1, max_frame_cache_size)
        self._frame_cache = CacheManager(
            max_size=self._default_frame_cache_size,
            on_evict=self._on_frame_evict,
        )
        self._override_steps: set[int] = set()
        self._preloaded_steps: set[int] = set()

    @property
    def frame_cache(self) -> CacheManager:
        """Read-only access to the underlying frame CacheManager."""
        return self._frame_cache

    def get_frame(self, step: int) -> Optional[dict[str, Any]]:
        """Return a cached frame if available."""
        return self._frame_cache.get(step)

    def has_frame(self, step: int) -> bool:
        """Check if a frame is cached."""
        return self._frame_cache.contains(step)

    def store_frame(self, step: int, frame: dict[str, Any], *, source: str = "provider") -> None:
        """Store a raw frame and track whether it came from provider/preload/override."""
        self._frame_cache.put(step, frame)
        if source == "override":
            self._override_steps.add(step)
        elif source == "preload":
            self._preloaded_steps.add(step)
            logger.debug(
                "Stored preload frame step=%d (tracked %d preloaded)",
                step,
                len(self._preloaded_steps),
            )

    def mark_preloaded(self, step: int) -> None:
        """Mark a step as preloaded."""
        self._preloaded_steps.add(step)

    def mark_override(self, step: int) -> None:
        """Mark a step as overridden."""
        self._override_steps.add(step)

    def is_override(self, step: int) -> bool:
        """Return True if a step is backed by override data."""
        return step in self._override_steps

    def is_preloaded(self, step: int) -> bool:
        """Return True if a step was preloaded."""
        return step in self._preloaded_steps

    @property
    def preloaded_frame_count(self) -> int:
        """Number of preloaded frames tracked."""
        return len(self._preloaded_steps)

    @property
    def preloaded_steps(self) -> list[int]:
        """Sorted list of preloaded step indices."""
        return sorted(self._preloaded_steps)

    def get_preloaded_frames(self) -> list[tuple[int, dict[str, Any]]]:
        """Return list of preloaded frames currently in cache."""
        frames: list[tuple[int, dict[str, Any]]] = []
        for step in self.preloaded_steps:
            frame = self._frame_cache.get(step)
            if frame is not None:
                frames.append((step, frame))
        return frames

    def clear_override(self, step: int, *, remove_frame: bool = True) -> bool:
        """Clear override state for a step, optionally removing cached frame."""
        removed = step in self._override_steps
        self._override_steps.discard(step)
        if remove_frame:
            self._frame_cache.remove(step)
        return removed

    def clear_preloaded(self) -> int:
        """Clear preloaded step tracking and return the number removed."""
        count = len(self._preloaded_steps)
        self._preloaded_steps.clear()
        return count

    def invalidate_frame(self, step: int) -> bool:
        """Remove a cached frame and its metadata."""
        removed = self._frame_cache.remove(step) is not None
        self._override_steps.discard(step)
        self._preloaded_steps.discard(step)
        return removed

    def clear_frame_cache(self, reason: str = "cache_reset") -> int:
        """Clear cached frame data and associated metadata."""
        removed = self._frame_cache.clear()
        self._override_steps.clear()
        self._preloaded_steps.clear()
        if removed:
            logger.info("[%s] Cleared frame cache (%s entries)", reason, removed)
        return removed

    def ensure_frame_cache_capacity(self, required_size: int) -> None:
        """Grow frame cache capacity to fit a full preload if needed."""
        if required_size <= 0:
            return
        if required_size > self._frame_cache.max_size:
            self._frame_cache.set_max_size(required_size)
            logger.info("Expanded frame cache to %d entries", required_size)

    def reset_frame_cache_size(self) -> None:
        """Reset frame cache capacity to the default size."""
        self._frame_cache.set_max_size(self._default_frame_cache_size)

    def invalidate(
        self,
        scopes: CacheInvalidationScope | str | Iterable[CacheInvalidationScope | str],
        *,
        reason: str = "cache_reset",
    ) -> dict[str, int]:
        """Invalidate one or more typed local cache scopes.

        ``FRAME_DATA`` removes raw frames and canonical MPC data. Filter,
        material, label, target, and render-setting scopes clear derived
        ViewModels and renderer frame state while preserving raw frames unless
        the raw data itself changed.
        """
        normalized = self._normalize_scopes(scopes)
        if CacheInvalidationScope.ALL in normalized:
            return self._invalidate_all(reason)

        result = {
            "frame_cache": 0,
            "view_model_cache": 0,
            "coverage_cache": 0,
            "target_cache": 0,
            "renderer_state": 0,
            "canonical_cache": 0,
        }

        view_model_scopes = {
            CacheInvalidationScope.FRAME_DATA,
            CacheInvalidationScope.FILTERS,
            CacheInvalidationScope.TARGET_GEOMETRY,
            CacheInvalidationScope.MATERIALS_COLORS,
            CacheInvalidationScope.LABELS,
            CacheInvalidationScope.STATIC_SCENE_GEOMETRY,
            CacheInvalidationScope.MPC_RENDER_SETTINGS,
        }
        if normalized & view_model_scopes:
            result["view_model_cache"] += self._clear_view_model_cache(reason)
            result["renderer_state"] += self._clear_renderer_frame_state(reason)
            self._clear_last_app_state()

        if CacheInvalidationScope.FRAME_DATA in normalized:
            result["canonical_cache"] += self._invalidate_canonical_cache(reason)
            result["frame_cache"] += self.clear_frame_cache(reason=reason)
            self._clear_preload_metadata(reason)
            self._clear_last_app_state()

        if CacheInvalidationScope.MATERIALS_COLORS in normalized:
            self._invalidate_material_color_cache()

        if CacheInvalidationScope.TARGET_GEOMETRY in normalized:
            result["target_cache"] += self._clear_target_geometry_caches(reason)

        if CacheInvalidationScope.STATIC_SCENE_GEOMETRY in normalized:
            result["coverage_cache"] += self._clear_coverage_cache(reason)

        if CacheInvalidationScope.CAMERA in normalized:
            self._clear_last_app_state()

        return result

    def clear_local_frame_caches(self, reason: str = "cache_reset") -> dict[str, int]:
        """Clear transient frame data without evicting reusable asset caches."""
        return self.invalidate(CacheInvalidationScope.FRAME_DATA, reason=reason)

    def clear_static_asset_caches(self, reason: str = "asset_cache_reset") -> dict[str, Any]:
        """Explicitly release inactive target and reusable static asset layers."""
        viz = self.visualizer
        material_service = getattr(viz, "material_pbr_service", None)
        invalidate_materials = getattr(
            material_service,
            "invalidate_material_resolution_cache",
            None,
        )
        if callable(invalidate_materials):
            invalidate_materials()
        target_result: dict[str, int] = {"entries": 0, "bytes": 0, "pending": 0}
        target_cache = getattr(viz, "target_asset_cache", None)
        clear_inactive = getattr(target_cache, "clear_inactive_assets", None)
        if callable(clear_inactive):
            try:
                cleared = clear_inactive() or {}
            except (RuntimeError, TypeError, ValueError, AttributeError):
                logger.warning("Failed to clear inactive target assets", exc_info=True)
            else:
                if isinstance(cleared, dict):
                    target_result.update(
                        {
                            key: max(0, int(value))
                            for key, value in cleared.items()
                            if key in target_result
                        }
                    )

        shared_result = clear_reusable_asset_caches(
            getattr(viz, "renderer", None),
            include_disk=True,
        )
        logger.info(
            "[%s] Cleared reusable asset caches (target=%d entries, target_bytes=%d)",
            reason,
            target_result["entries"],
            target_result["bytes"],
        )
        return {"target_assets": target_result, "layers": shared_result}

    def invalidate_canonical_step(
        self,
        step: int,
        *,
        reason: str = "frame_step",
    ) -> dict[str, int]:
        """Invalidate canonical and ViewModel caches for one updated raw frame."""
        result = {
            "frame_cache": 0,
            "view_model_cache": 0,
            "coverage_cache": 0,
            "target_cache": 0,
            "renderer_state": 0,
            "canonical_cache": 0,
        }
        result["canonical_cache"] += self._invalidate_canonical_step(step, reason)
        result["view_model_cache"] += self._clear_view_model_cache(reason)
        result["renderer_state"] += self._clear_renderer_frame_state(reason)
        self._clear_last_app_state()
        return result

    def get_cache_telemetry(self) -> dict[str, Any]:
        """Return cache counters and byte budgets for diagnostics."""
        stats = self._frame_cache.stats
        viz = self.visualizer
        telemetry: dict[str, Any] = {
            "frame_cache_size": int(getattr(stats, "current_size", self._frame_cache.size)),
            "frame_cache_max_size": int(getattr(stats, "max_size", self._frame_cache.max_size)),
            "frame_cache_hits": int(getattr(stats, "hits", 0)),
            "frame_cache_misses": int(getattr(stats, "misses", 0)),
            "frame_cache_evictions": int(getattr(stats, "evictions", 0)),
            "frame_cache_byte_budget": None,
            "view_model_cache_size": self._safe_len(getattr(viz, "mpc_view_cache", None)),
        }
        target_cache = getattr(viz, "target_asset_cache", None)
        target_telemetry = getattr(target_cache, "telemetry", None)
        if callable(target_telemetry):
            try:
                target_stats = target_telemetry() or {}
            except (RuntimeError, ValueError, TypeError, AttributeError):
                target_stats = {}
            if isinstance(target_stats, dict):
                telemetry.update(
                    {f"target_asset_cache_{key}": value for key, value in target_stats.items()}
                )
        renderer = getattr(viz, "renderer", None)
        get_stats = getattr(renderer, "get_runtime_stats", None)
        if callable(get_stats):
            try:
                renderer_stats = get_stats() or {}
            except (RuntimeError, ValueError, TypeError, AttributeError):
                renderer_stats = {}
            if isinstance(renderer_stats, dict):
                for key in (
                    "mpc_line_cache_entries",
                    "mpc_line_cache_bytes",
                    "mpc_line_cache_max_bytes",
                    "mpc_line_cache_hits",
                    "mpc_line_cache_misses",
                    "mpc_line_cache_evictions",
                ):
                    if key in renderer_stats:
                        telemetry[key] = renderer_stats[key]
        asset_snapshot = collect_static_asset_cache_snapshot(renderer).as_dict()
        telemetry["static_asset_caches"] = asset_snapshot["layers"]
        telemetry["static_asset_cache_aggregate"] = asset_snapshot["aggregate"]
        return telemetry

    def _normalize_scopes(
        self,
        scopes: CacheInvalidationScope | str | Iterable[CacheInvalidationScope | str],
    ) -> set[CacheInvalidationScope]:
        """Coerce public scope inputs before applying invalidation policy."""
        return normalize_cache_invalidation_scopes(scopes)

    def _invalidate_all(self, reason: str) -> dict[str, int]:
        """Clear all visualizer-local caches and restore default frame-cache size."""
        result = {
            "frame_cache": 0,
            "view_model_cache": 0,
            "coverage_cache": 0,
            "target_cache": 0,
            "renderer_state": 0,
            "canonical_cache": 0,
        }
        viz = self.visualizer

        result["canonical_cache"] += self._invalidate_canonical_cache(reason)
        result["frame_cache"] += self.clear_frame_cache(reason=reason)
        self._clear_preload_metadata(reason)
        result["view_model_cache"] += self._clear_view_model_cache(reason)
        result["coverage_cache"] += self._clear_coverage_cache(reason)
        self._clear_last_app_state()
        result["renderer_state"] += self._clear_renderer_frame_state(reason)
        result["target_cache"] += self._clear_target_geometry_caches(reason)
        if hasattr(viz, "_cached_bounce_points"):
            self._reset_bounce_cache("_cached_bounce_points", reason)
        if hasattr(viz, "_cached_bounce_colors"):
            self._reset_bounce_cache("_cached_bounce_colors", reason)

        self.reset_frame_cache_size()
        return result

    @staticmethod
    def _safe_len(value: Any) -> Optional[int]:
        """Return ``len(value)`` for real containers while tolerating test doubles."""
        try:
            return len(value) if value is not None else None
        except TypeError:
            return None

    def _invalidate_canonical_cache(self, reason: str) -> int:
        """Clear all canonical MPC arrays owned by MPCCore."""
        viz = self.visualizer
        mpc_core = getattr(viz, "mpc_core", None)
        invalidate = getattr(mpc_core, "invalidate_step", None)
        if not callable(invalidate):
            return 0
        cache_size_before = self._safe_len(getattr(mpc_core, "_canon_cache", None)) or 0
        invalidate(None)
        if cache_size_before:
            logger.info(
                "[%s] Cleared MPCCore canonical cache (removed %s frames)",
                reason,
                cache_size_before,
            )
        return int(cache_size_before)

    def _invalidate_canonical_step(self, step: int, reason: str) -> int:
        """Clear canonical MPC data for one frame step."""
        mpc_core = getattr(self.visualizer, "mpc_core", None)
        invalidate = getattr(mpc_core, "invalidate_step", None)
        if not callable(invalidate):
            return 0
        cache = getattr(mpc_core, "_canon_cache", None)
        had_step = False
        try:
            had_step = step in cache if cache is not None else False
        except TypeError:
            had_step = False
        invalidate(step)
        if had_step:
            logger.info("[%s] Cleared MPCCore canonical cache for step %s", reason, step)
        return int(had_step)

    def _invalidate_material_color_cache(self) -> None:
        """Clear MPCCore material-color memoization after material filters change."""
        mpc_core = getattr(self.visualizer, "mpc_core", None)
        invalidate = getattr(mpc_core, "invalidate_material_colors_cache", None)
        if callable(invalidate):
            invalidate()

    def _clear_preload_metadata(self, reason: str) -> int:
        """Ask animation preloading to drop metadata tied to raw-frame caches."""
        anim = getattr(self.visualizer, "animation_service", None)
        clear = getattr(anim, "clear_preload_data", None)
        if not callable(clear):
            return 0
        removed = int(clear(reset_cache_size=True) or 0)
        if removed:
            logger.info("[%s] Cleared preloaded metadata (%s entries)", reason, removed)
        return removed

    def _clear_view_model_cache(self, reason: str) -> int:
        """Clear derived ViewModels that depend on frame data or visual filters."""
        cache = getattr(self.visualizer, "mpc_view_cache", None)
        if cache is None or not hasattr(cache, "clear"):
            return 0
        removed = self._safe_len(cache) or 0
        cache.clear()
        if removed:
            logger.info("[%s] Cleared mpc_view_cache (%s entries)", reason, removed)
        return int(removed)

    def _clear_coverage_cache(self, reason: str) -> int:
        """Clear derived coverage meshes for static-scene geometry changes."""
        service = getattr(self.visualizer, "coverage_service", None)
        if service is None or not hasattr(service, "clear"):
            return 0
        service.clear()
        logger.info("[%s] Cleared CoverageService cache", reason)
        return 1

    def _clear_target_geometry_caches(self, reason: str) -> int:
        """Atomically clear the typed target asset and runtime-state owner."""
        owner = getattr(self.visualizer, "target_asset_cache", None)
        if not isinstance(owner, TargetAssetCache):
            return 0
        try:
            removed = int(owner.clear() or 0)
        except (TypeError, ValueError, RuntimeError):
            logger.warning("Failed to clear typed target asset cache", exc_info=True)
            return 0

        if removed:
            logger.info(
                "[%s] Cleared target asset/runtime cache (%s records)",
                reason,
                removed,
            )
        return removed

    def _clear_last_app_state(self) -> None:
        """Forget the last AppState snapshot so pipeline diffing restarts cleanly."""
        if hasattr(self.visualizer, "last_app_state"):
            self.visualizer.last_app_state = None

    def _clear_renderer_frame_state(self, reason: str) -> int:
        """Drop renderer-owned frame state and request a redraw when initialized."""
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is None:
            return 0
        removed = 0
        if hasattr(renderer, "last_frame_packet"):
            if getattr(renderer, "last_frame_packet", None) is not None:
                removed = 1
            renderer.last_frame_packet = None
            logger.info("[%s] Cleared renderer last_frame_packet", reason)
        if getattr(renderer, "vis_initialized", False):
            try:
                update = getattr(renderer, "update_renderer", None)
                if callable(update):
                    update()
            except (RuntimeError, ValueError) as exc:
                logger.warning("Failed to refresh renderer after cache invalidation: %s", exc)
        return removed

    def _reset_bounce_cache(self, attr: str, reason: str) -> None:
        """Helper for resetting cached bounce arrays."""
        import numpy as np

        viz = self.visualizer
        setattr(viz, attr, np.empty((0, 3), dtype=np.float64))
        logger.info("[%s] Cleared %s", reason, attr)

    def _on_frame_evict(self, frame_idx: int, _: Any) -> None:
        """Handle raw frame eviction bookkeeping.

        Canonical MPC data is derived from raw frames, so evicting a raw frame
        also evicts the matching canonical step while broader ViewModel cache
        invalidation remains scope-driven.
        """
        self._override_steps.discard(frame_idx)
        self._preloaded_steps.discard(frame_idx)
        viz = self.visualizer
        mpc_core = getattr(viz, "mpc_core", None)
        invalidate = getattr(mpc_core, "invalidate_step", None)
        if callable(invalidate):
            try:
                invalidate(frame_idx)
            except (RuntimeError, ValueError, AttributeError) as exc:
                logger.debug("Failed to invalidate canonical cache for %s: %s", frame_idx, exc)
