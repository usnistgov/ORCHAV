"""Canonical MPC frame-state construction.

Provider projections supply visual payload dictionaries containing aligned
``CanonicalStepData``. ``MPCCore`` caches that data and derives an
application-facing ``ViewModel``.
``ViewModel.to_render_packet()`` projects only frame-heavy renderer inputs into
the renderer protocol without copying their arrays. Canonical arrays are cached
by animation step; ViewModels are cached by frame plus render-affecting state in
``FramePipeline``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np

from shared.logging import get_logger
from shared.statistics import FrameStats
from shared.statistics.themes import theme_manager

from ..metrics.mpc_canon import (
    MPC_MATERIAL_FALLBACK_GRAY,
    CanonicalStepData,
    _bare_material_type,
    build_filter_mask,
    colorize,
    colorize_segments,
    ensure_luts,
    remap_lines,
)
from ..renderers.protocol import renderer_capabilities
from ..scene.surface_payloads import BeamformingSurface
from ..services.cache_service import CacheInvalidationScope
from ..services.mpc_interaction_style_service import (
    build_mpc_type_palette,
    mpc_interaction_label,
    mpc_interaction_sort_key,
)
from ..state import MpcVisibility
from ..utils.colors import (
    clear_categorical_palette_cache,
    clear_continuous_lut_cache,
    get_categorical_order_palette,
    get_categorical_type_palette,
    get_distinct_material_palette,
)

FramePayload = Dict[str, Any]
MpcRevision = tuple[Any, ...]


def _mpc_array_revision_token(array: Optional[np.ndarray]) -> Optional[tuple[Any, ...]]:
    """Return an O(1) token for a ViewModel-owned immutable MPC array."""
    if array is None:
        return None
    return (
        "ndarray",
        id(array),
        tuple(array.shape),
        array.dtype.str,
        tuple(array.strides),
        int(array.nbytes),
    )


def _freeze_mpc_array(value: Optional[Any]) -> Optional[np.ndarray]:
    """Normalize and freeze one MPC array field for renderer reuse."""
    if value is None:
        return None
    array = value if isinstance(value, np.ndarray) else np.asarray(value)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FrameRenderPacket:
    """Shallow, frame-heavy input accepted by renderer backends.

    Persistent TX/RX, target, and label state intentionally stays out of this
    contract. Array and metadata references are shared with the owning
    ``ViewModel``; constructing a packet performs no coercion or copy.
    """

    mpc_points: np.ndarray
    mpc_lines: np.ndarray
    mpc_colors: np.ndarray
    mpc_visibility: MpcVisibility
    mpc_bounce_points: Optional[np.ndarray]
    mpc_bounce_colors: Optional[np.ndarray]
    mpc_bounce_itypes: Optional[np.ndarray]
    mpc_line_itypes: Optional[np.ndarray]
    mpc_line_itype_codes: tuple[int, ...]
    mpc_line_revision: MpcRevision
    mpc_point_revision: MpcRevision
    canonical_data: Optional[CanonicalStepData]
    path_mask: Optional[np.ndarray]
    segment_mask: Optional[np.ndarray]
    coverage_vertices: Optional[np.ndarray]
    coverage_triangles: Optional[np.ndarray]
    coverage_colors: Optional[np.ndarray]
    coverage_isoline_points: Optional[np.ndarray]
    coverage_isoline_lines: Optional[np.ndarray]
    coverage_isoline_colors: Optional[np.ndarray]
    coverage_signature: Optional[str]
    coverage_metadata: Optional[dict[str, Any]]
    coverage_opacity: float
    show_coverage: bool
    beamforming_meshes: tuple[BeamformingSurface, ...]
    colorbar: Optional[tuple[str, tuple[float, float]]]
    stats_text: str


@dataclass
class ViewModel:
    """Application and service state derived for one visualizer frame.

    Positions are in scene meters. Orientation arrays are ``(N, 3)`` yaw,
    pitch, roll values in radians. MPC line arrays index into ``mpc_points``;
    line and point colors are normalized RGB floats whose dtype is chosen for
    the active renderer. MPC arrays are immutable after construction; renderers
    use the revision fields for O(1) freshness checks instead of scanning array
    contents. Renderer backends never receive this object directly; use
    :meth:`to_render_packet` for the narrow frame-rendering contract.
    """

    tx_positions: np.ndarray  # (n_tx, 3) canonical TX inventory order
    rx_positions: np.ndarray  # (n_rx, 3) canonical RX inventory order
    tx_orientations: np.ndarray  # (n_tx, 3) TX orientations (yaw, pitch, roll) in radians
    rx_orientations: np.ndarray  # (n_rx, 3) RX orientations (yaw, pitch, roll) in radians
    mpc_points: np.ndarray  # (N, 3)
    mpc_lines: np.ndarray  # (M, 2) int
    mpc_colors: np.ndarray  # (M, 3)
    colorbar: Optional[tuple[str, tuple[float, float]]]  # ("Delay (ns)", (min,max))
    stats_text: str
    mpc_visibility: MpcVisibility

    # Target data for the same frame as the MPC payload.
    target_positions: np.ndarray  # (n_targets, 3) target positions
    target_orientations: np.ndarray  # (n_targets, 3) target orientations (yaw, pitch, roll)
    target_mesh_files: list[str]  # List of mesh file names for each target
    target_use_ply_positions: list[bool]  # Whether each target uses PLY position
    target_metadata: list[dict]  # Full target metadata for each target
    mpc_bounce_points: Optional[np.ndarray] = None  # (B, 3) filtered physical bounce markers
    mpc_bounce_colors: Optional[np.ndarray] = None  # (B, 3) colors for bounce markers
    mpc_bounce_itypes: Optional[np.ndarray] = None  # (B,) uint8 interaction type markers

    # Canonical data for statistics computation.
    canonical_data: Optional[CanonicalStepData] = None  # Raw canonical data for stats
    path_mask: Optional[np.ndarray] = None  # bool[P] - which paths survive the current filter
    segment_mask: Optional[np.ndarray] = None  # bool[S] - visible canonical segments

    # Coverage map data (mesh-based).
    coverage_vertices: Optional[np.ndarray] = None  # (K, 3) grid vertices
    coverage_triangles: Optional[np.ndarray] = None  # (L, 3) triangle indices
    coverage_colors: Optional[np.ndarray] = None  # (K, 3) per-vertex colors
    coverage_isoline_points: Optional[np.ndarray] = None  # (C, 3) isoline points
    coverage_isoline_lines: Optional[np.ndarray] = None  # (S, 2) isoline segment indices
    coverage_isoline_colors: Optional[np.ndarray] = None  # (S, 3) per-segment colors
    coverage_signature: Optional[str] = None  # Geometry/data revision token
    coverage_metadata: Optional[dict] = None  # Coverage map metadata
    coverage_opacity: float = 1.0  # Coverage transparency (0.0=transparent, 1.0=opaque)
    show_coverage: bool = False  # Whether to show coverage map
    mpc_line_itypes: Optional[np.ndarray] = None  # (M,) uint8 — itype at segment start
    # Sorted unique interaction codes in the rendered segment payload. Type
    # color mode derives this from its existing histogram; other modes leave it
    # empty so they do not pay for an unused full-segment scan.
    mpc_line_itype_codes: tuple[int, ...] = ()
    beamforming_meshes: tuple[BeamformingSurface, ...] = ()
    beamforming_pairs: Optional[list[dict[str, Any]]] = None
    beamforming_info: Optional[dict[str, Any]] = None
    mpc_line_revision: MpcRevision = field(init=False)
    mpc_point_revision: MpcRevision = field(init=False)

    def __post_init__(self) -> None:
        """Freeze MPC payload arrays and derive renderer-neutral freshness tokens."""
        line_fields = ("mpc_points", "mpc_lines", "mpc_colors", "mpc_line_itypes")
        point_fields = (
            "mpc_bounce_points",
            "mpc_bounce_colors",
            "mpc_bounce_itypes",
        )
        mpc_fields = set(line_fields + point_fields + ("segment_mask", "path_mask"))

        revisions: dict[str, Optional[tuple[Any, ...]]] = {}
        for field_name in mpc_fields:
            frozen = _freeze_mpc_array(getattr(self, field_name))
            setattr(self, field_name, frozen)
            revisions[field_name] = _mpc_array_revision_token(frozen)

        self.mpc_line_itype_codes = tuple(
            sorted({int(value) for value in self.mpc_line_itype_codes})
        )

        self.beamforming_meshes = tuple(self.beamforming_meshes)
        if not all(isinstance(surface, BeamformingSurface) for surface in self.beamforming_meshes):
            raise TypeError("beamforming_meshes must contain BeamformingSurface values")

        self.mpc_line_revision = (
            "mpc-line-v1",
            *(revisions[field_name] for field_name in line_fields),
        )
        self.mpc_point_revision = (
            "mpc-point-v1",
            *(revisions[field_name] for field_name in point_fields),
        )

    def to_render_packet(self) -> FrameRenderPacket:
        """Project renderer-owned frame data without copying any payload."""
        return FrameRenderPacket(
            mpc_points=self.mpc_points,
            mpc_lines=self.mpc_lines,
            mpc_colors=self.mpc_colors,
            mpc_visibility=self.mpc_visibility,
            mpc_bounce_points=self.mpc_bounce_points,
            mpc_bounce_colors=self.mpc_bounce_colors,
            mpc_bounce_itypes=self.mpc_bounce_itypes,
            mpc_line_itypes=self.mpc_line_itypes,
            mpc_line_itype_codes=self.mpc_line_itype_codes,
            mpc_line_revision=self.mpc_line_revision,
            mpc_point_revision=self.mpc_point_revision,
            canonical_data=self.canonical_data,
            path_mask=self.path_mask,
            segment_mask=self.segment_mask,
            coverage_vertices=self.coverage_vertices,
            coverage_triangles=self.coverage_triangles,
            coverage_colors=self.coverage_colors,
            coverage_isoline_points=self.coverage_isoline_points,
            coverage_isoline_lines=self.coverage_isoline_lines,
            coverage_isoline_colors=self.coverage_isoline_colors,
            coverage_signature=self.coverage_signature,
            coverage_metadata=self.coverage_metadata,
            coverage_opacity=self.coverage_opacity,
            show_coverage=self.show_coverage,
            beamforming_meshes=self.beamforming_meshes,
            colorbar=self.colorbar,
            stats_text=self.stats_text,
        )


class MPCCore:
    """Build and cache canonical MPC data and renderer-neutral ViewModels."""

    def __init__(self, logger: Optional[Any] = None, visualizer: Optional[Any] = None) -> None:
        """Initialize canonical caches, material lookup state, and theme palettes."""
        if logger is None:
            self.logger = get_logger("orchav")
        else:
            self.logger = logger

        self.visualizer = visualizer

        self.current_delay_range = None
        self.current_path_loss_range = None

        self.beamforming_service: Optional[Any] = None

        # Canonical per-step cache is a byte-budgeted LRU.
        self._canon_lock = threading.Lock()
        self._canon_cache: OrderedDict[int, CanonicalStepData] = OrderedDict()
        self._canon_sizes: dict[int, int] = {}
        self._canon_bytes_used: int = 0

        import os as _os

        from ..config import DEFAULT_CANON_CACHE_MB

        mb = int(_os.getenv("VIZ_CANON_CACHE_MB", str(DEFAULT_CANON_CACHE_MB)))
        self._canon_max_bytes: int = mb * 1024 * 1024

        # Renderer-aware canonical points dtype. Determined lazily so the
        # renderer is already constructed. Pygfx prefers float32 to avoid GPU
        # conversion; Open3D needs float64 for Vector3dVector copies.
        self._canon_points_dtype: Optional[np.dtype] = None

        # Cached material color map (invalidated on scene load)
        self._material_colors_cache: Optional[dict[str, np.ndarray]] = None
        self._last_viewmodel_breakdown: dict[str, float] = {}
        self._last_canonical_lookup_ms: float = 0.0
        self._last_canonical_cache_hit: bool = False

        self._viridis256 = ensure_luts()
        self._update_palettes_from_theme()
        theme_manager.add_listener(self._on_theme_change)

    @property
    def canon_points_dtype(self) -> np.dtype:
        """Return the preferred float dtype for canonical point arrays."""
        if self._canon_points_dtype is None:
            renderer = getattr(self.visualizer, "renderer", None)
            if renderer_capabilities(renderer).prefer_float32_frame_data:
                self._canon_points_dtype = np.float32
            else:
                self._canon_points_dtype = np.float64
        return self._canon_points_dtype

    def _update_palettes_from_theme(self) -> None:
        """Update palettes from current theme or categorical colormap."""
        theme = theme_manager.current

        # Check if categorical colormap override is active
        cat_order_palette = get_categorical_order_palette(n_colors=7)
        cat_type_palette = get_categorical_type_palette(n_colors=9)

        if cat_order_palette is not None:
            # Use colors from categorical colormap
            self._order_palette = cat_order_palette
        else:
            # Use explicit theme colors (semantic: green=LoS, etc.)
            self._order_palette = np.array(
                theme.reflection_order,
                dtype=np.float32,
            )

        if cat_type_palette is not None:
            # Use colors from categorical colormap
            self._type_palette = cat_type_palette
        else:
            self._type_palette = build_mpc_type_palette(theme.interaction_type)

    def _on_theme_change(self, theme) -> None:
        """Handle theme change notification."""
        # Clear caches so new colormaps are used
        clear_continuous_lut_cache()
        clear_categorical_palette_cache()
        # Update palettes (may use categorical colormap or theme colors)
        self._update_palettes_from_theme()
        self.logger.debug("Updated palettes for theme: %s", theme.name)

        cache_service = getattr(self.visualizer, "cache_service", None)
        if cache_service is not None:
            cache_service.invalidate(CacheInvalidationScope.MATERIALS_COLORS, reason="theme_change")

        # Force update on next frame
        if self.visualizer:
            self.visualizer.force_update_next_frame = True

            has_pipeline = hasattr(self.visualizer, "pipeline")
            has_app_state = hasattr(self.visualizer, "app_state")

            if has_pipeline and has_app_state:
                try:
                    current_step = self.visualizer.app_state.step
                    self.visualizer.pipeline.update(current_step)
                    self.logger.info(
                        "Triggered re-render for theme change at step %d", current_step
                    )
                except (RuntimeError, ValueError, AttributeError) as e:
                    self.logger.warning("Failed to trigger re-render for theme change: %s", e)
            else:
                self.logger.warning(
                    "Cannot trigger re-render: pipeline=%s, app_state=%s",
                    has_pipeline,
                    has_app_state,
                )
        else:
            self.logger.warning("Cannot trigger re-render: visualizer is None")

    # ===== Canonical LRU helpers =====
    def _estimate_canon_bytes(self, canon: CanonicalStepData) -> int:
        """Estimate bytes retained by every canonical ndarray backing store.

        Canonical fields often expose multiple views of one allocation, such as
        segment start/end columns backed by ``lines``. Charging each view's
        ``nbytes`` double-counts that allocation, while charging only the view
        can miss a larger base buffer retained by a small slice. Walk to each
        array's ultimate buffer owner and count it once. Enumerating dataclass
        fields also keeps angle and material arrays inside the cache budget as
        the canonical contract evolves.
        """

        def backing_store(array: np.ndarray) -> tuple[object, int]:
            """Return the ultimate retained buffer owner and its byte size."""
            owner: object = array
            visited: set[int] = set()
            while id(owner) not in visited:
                visited.add(id(owner))
                if isinstance(owner, np.ndarray) and owner.base is not None:
                    owner = owner.base
                    continue
                if isinstance(owner, memoryview) and owner.obj is not owner:
                    owner = owner.obj
                    continue
                break

            if isinstance(owner, np.ndarray):
                size = int(owner.nbytes)
            elif isinstance(owner, memoryview):
                size = int(owner.nbytes)
            elif isinstance(owner, (bytes, bytearray)):
                size = len(owner)
            else:
                # NumPy may expose a third-party buffer owner without a public
                # size. The visible ndarray is the safest bounded estimate.
                size = int(array.nbytes)
            return owner, size

        total = 0
        seen_owners: set[int] = set()
        for canonical_field in fields(canon):
            value = getattr(canon, canonical_field.name)
            if not isinstance(value, np.ndarray):
                continue
            owner, size = backing_store(value)
            owner_id = id(owner)
            if owner_id in seen_owners:
                continue
            seen_owners.add(owner_id)
            total += size
        return total

    def get_last_viewmodel_breakdown(self) -> dict[str, float]:
        """Return the last per-phase create_view_model timing breakdown."""
        return dict(self._last_viewmodel_breakdown)

    def _detailed_viewmodel_profile_enabled(self) -> bool:
        """Return whether benchmark-only MPC profiling details should be emitted."""
        if os.getenv("ORCHAV_BENCH_PROFILE_DETAIL"):
            return True
        pipeline = getattr(self.visualizer, "pipeline", None)
        return getattr(pipeline, "benchmark_recorder", None) is not None

    def _touch_step_lru(self, step: int) -> None:
        """Mark step as most recently used. O(1) with OrderedDict.

        Caller must hold ``_canon_lock``.
        """
        if step in self._canon_cache:
            self._canon_cache.move_to_end(step, last=True)

    def _evict_until_under_budget(self) -> None:
        """Evict oldest (LRU) entries until under budget. O(1) per eviction.

        Caller must hold ``_canon_lock``.
        """
        while self._canon_bytes_used > self._canon_max_bytes and len(self._canon_cache) > 1:
            victim, _ = self._canon_cache.popitem(last=False)
            self._canon_bytes_used -= self._canon_sizes.pop(victim, 0)

    def invalidate_step(self, step: Optional[int] = None) -> None:
        """Invalidate cached canonical data for a specific step or the entire cache.

        Thread-safe: called from the preloader's eviction callback (background
        thread) and from the main thread during scenario/scene changes.
        """
        with self._canon_lock:
            if step is None:
                self._canon_cache.clear()
                self._canon_sizes.clear()
                self._canon_bytes_used = 0
                self._material_colors_cache = None
                return

            if step in self._canon_cache:
                self._canon_bytes_used -= self._canon_sizes.pop(step, 0)
                self._canon_cache.pop(step, None)

    def precompute_canonical(self, step: int, raw_frame: FramePayload) -> None:
        """Precompute and cache canonical data for a frame (thread-safe).

        Called from the preload worker thread to move heavy computation
        off the main thread.  If *step* is already cached this is a no-op.
        """
        with self._canon_lock:
            if step in self._canon_cache:
                return

        canon = self._canonical_from_payload(raw_frame)
        sz = self._estimate_canon_bytes(canon)

        with self._canon_lock:
            # Double-check after releasing/reacquiring lock
            if step in self._canon_cache:
                return
            self._canon_cache[step] = canon
            self._canon_sizes[step] = sz
            self._canon_bytes_used += sz
            self._touch_step_lru(step)
            self._evict_until_under_budget()

    def _canonical_from_payload(self, raw_frame: FramePayload) -> CanonicalStepData:
        """Return canonical data supplied by the frame projection boundary.

        Providers normally supply ``CanonicalStepData`` directly. A dictionary
        form remains valid for renderer-specific sources that serialize the
        same canonical fields. Both forms are coerced to the renderer-preferred
        point dtype before caching.
        """
        persisted = raw_frame.get("canonical_data")
        if isinstance(persisted, CanonicalStepData):
            canon = persisted
        elif isinstance(persisted, dict):
            points = np.asarray(
                persisted.get("points", np.empty((0, 3))), dtype=self.canon_points_dtype
            )
            if not points.flags.c_contiguous:
                points = np.ascontiguousarray(points)

            def _arr(
                name: str, dtype: np.dtype, default_shape: tuple[int, ...] = (0,)
            ) -> np.ndarray:
                """Return a contiguous persisted canonical array or empty default."""
                value = persisted.get(name)
                if value is None:
                    return np.empty(default_shape, dtype=dtype)
                arr = np.asarray(value, dtype=dtype)
                if not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                return arr

            def _opt_arr(name: str, dtype: np.dtype) -> Optional[np.ndarray]:
                """Return a contiguous optional persisted canonical array."""
                value = persisted.get(name)
                if value is None:
                    return None
                arr = np.asarray(value, dtype=dtype)
                if not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                return arr

            canon = CanonicalStepData(
                points=points,
                lines=_arr("lines", np.int32, (0, 2)),
                order=_arr("order", np.uint8),
                itype=_arr("itype", np.uint8),
                delay=_arr("delay", np.float32),
                loss=_arr("loss", np.float32),
                tx_id=_opt_arr("tx_id", np.int16),
                rx_id=_opt_arr("rx_id", np.int16),
                path_id=_opt_arr("path_id", np.int32),
                path_start_indices=_opt_arr("path_start_indices", np.int32),
                path_orders=_opt_arr("path_orders", np.uint8),
                path_delays=_opt_arr("path_delays", np.float32),
                path_losses=_opt_arr("path_losses", np.float32),
                path_tx=_opt_arr("path_tx", np.int16),
                path_rx=_opt_arr("path_rx", np.int16),
                path_delay_is_estimated=_opt_arr("path_delay_is_estimated", np.bool_),
                path_loss_is_estimated=_opt_arr("path_loss_is_estimated", np.bool_),
                path_aoa_az=_opt_arr("path_aoa_az", np.float32),
                path_aoa_el=_opt_arr("path_aoa_el", np.float32),
                path_aod_az=_opt_arr("path_aod_az", np.float32),
                path_aod_el=_opt_arr("path_aod_el", np.float32),
                segment_start_indices=_opt_arr("segment_start_indices", np.int32),
                segment_end_indices=_opt_arr("segment_end_indices", np.int32),
                segment_order=_opt_arr("segment_order", np.uint8),
                segment_itype=_opt_arr("segment_itype", np.uint8),
                segment_delay=_opt_arr("segment_delay", np.float32),
                segment_loss=_opt_arr("segment_loss", np.float32),
                segment_tx_id=_opt_arr("segment_tx_id", np.int16),
                segment_rx_id=_opt_arr("segment_rx_id", np.int16),
                segment_path_id=_opt_arr("segment_path_id", np.int32),
                segment_material_ids=_opt_arr("segment_material_ids", np.int16),
                material_names=None,
                material_ids=_opt_arr("material_ids", np.int16),
                material_itu_types=None,
                material_id_to_name=persisted.get("material_id_to_name"),
                material_id_to_itu=persisted.get("material_id_to_itu"),
                material_id_to_bare=persisted.get("material_id_to_bare"),
                delay_min=float(persisted.get("delay_min", 0.0)),
                delay_max=float(persisted.get("delay_max", 1.0)),
                loss_min=float(persisted.get("loss_min", 0.0)),
                loss_max=float(persisted.get("loss_max", 1.0)),
                aoa_az=_opt_arr("aoa_az", np.float32),
                aoa_el=_opt_arr("aoa_el", np.float32),
                aod_az=_opt_arr("aod_az", np.float32),
                aod_el=_opt_arr("aod_el", np.float32),
                aoa_az_min=float(persisted.get("aoa_az_min", 0.0)),
                aoa_az_max=float(persisted.get("aoa_az_max", 360.0)),
                aoa_el_min=float(persisted.get("aoa_el_min", -90.0)),
                aoa_el_max=float(persisted.get("aoa_el_max", 90.0)),
                aod_az_min=float(persisted.get("aod_az_min", 0.0)),
                aod_az_max=float(persisted.get("aod_az_max", 360.0)),
                aod_el_min=float(persisted.get("aod_el_min", -90.0)),
                aod_el_max=float(persisted.get("aod_el_max", 90.0)),
                profile_ms=None,
            )
        else:
            raise TypeError(
                "Visualizer frame payloads must contain canonical_data from "
                "the provider projection boundary"
            )

        if canon.points.dtype != self.canon_points_dtype:
            canon = CanonicalStepData(
                **{
                    **canon.__dict__,
                    "points": np.asarray(canon.points, dtype=self.canon_points_dtype),
                }
            )
        return canon

    def _get_canonical(self, step: int, raw_frame: FramePayload) -> CanonicalStepData:
        """Return cached canonical data for *step*, populating the LRU on miss."""
        lookup_start = time.perf_counter()
        with self._canon_lock:
            if step in self._canon_cache:
                self._touch_step_lru(step)
                self.logger.debug("Using cached canonical data for step %d", step)
                self._last_canonical_lookup_ms = (time.perf_counter() - lookup_start) * 1000.0
                self._last_canonical_cache_hit = True
                return self._canon_cache[step]

        self.logger.debug("Loading canonical data for step %d (not in cache)", step)
        canon = self._canonical_from_payload(raw_frame)
        sz = self._estimate_canon_bytes(canon)

        with self._canon_lock:
            self._canon_cache[step] = canon
            self._canon_sizes[step] = sz
            self._canon_bytes_used += sz
            self._touch_step_lru(step)
            self._evict_until_under_budget()

        self._last_canonical_lookup_ms = (time.perf_counter() - lookup_start) * 1000.0
        self._last_canonical_cache_hit = False
        self.logger.debug(
            "Canonical cached: step=%d size=%.2f MB, total=%.2f MB (cap=%.0f MB)",
            step,
            sz / 1e6,
            self._canon_bytes_used / 1e6,
            self._canon_max_bytes / 1e6,
        )
        return canon

    def stats(self, payload: FramePayload) -> FrameStats:
        """Compute frame statistics from the canonical data attached to payload."""
        if "canonical_data" not in payload or payload["canonical_data"] is None:
            self.logger.warning("No canonical_data in payload, returning empty stats")
            return FrameStats(
                total_paths=0, orders_hist={}, delay_range_ns=None, path_loss_range=None
            )

        from ..metrics.mpc_stats import MPCStatsComputer

        canon = payload["canonical_data"]
        computer = MPCStatsComputer()

        # Only compute advanced stats if metrics window is visible
        include_advanced = payload.get("metrics_visible", False)

        path_mask = payload.get("path_mask", None)
        self.logger.debug("Using canonical stats system, include_advanced=%s", include_advanced)
        return computer.compute_frame_stats(canon, include_advanced, path_mask=path_mask)

    def discover_tx_rx(self, payload: FramePayload) -> tuple[list[int], list[int]]:
        """Discover available 0-based TX/RX indices from frame payload data."""

        available_tx = []
        available_rx = []

        # Try to get TX/RX info from different possible field names
        num_tx = payload.get("num_tx", None)
        num_rx = payload.get("num_rx", None)

        # If not found, try alternative field names
        if num_tx is None:
            num_tx = payload.get("tx_count", None)
        if num_rx is None:
            num_rx = payload.get("rx_count", None)

        # If still not found, try to infer from tx_rx_pairs
        if num_tx is None or num_rx is None:
            if "tx_rx_pairs" in payload:
                pairs = payload["tx_rx_pairs"]
                if len(pairs) > 0:
                    tx_indices = set()
                    rx_indices = set()
                    for pair in pairs:
                        if len(pair) >= 2:
                            tx_indices.add(pair[0])
                            rx_indices.add(pair[1])

                    if num_tx is None and tx_indices:
                        num_tx = max(tx_indices) + 1  # Convert to count
                    if num_rx is None and rx_indices:
                        num_rx = max(rx_indices) + 1  # Convert to count

        if num_tx is not None and num_tx > 0:
            available_tx = list(range(num_tx))  # 0-based indexing
        if num_rx is not None and num_rx > 0:
            available_rx = list(range(num_rx))  # 0-based indexing

        return available_tx, available_rx

    def create_view_model(
        self,
        step: int,
        raw_frame: dict,
        color_mode: str,
        selected_tx: Union[int, str] = "all",
        selected_rx: Union[int, str] = "all",
        mpc_allowed_orders: Optional[Sequence[int]] = None,
        mpc_allowed_types: Optional[Sequence[int]] = None,
        mpc_allowed_materials: Optional[Sequence[str]] = None,
        mpc_visibility: MpcVisibility = MpcVisibility(),
        topk_render_enabled: bool = False,
        topk_render_max_paths: int = 20000,
        include_targets: bool = True,
        show_tx_segments: bool = True,
        show_beamforming: bool = False,
        beamforming_azimuth_samples: int = 72,
        beamforming_elevation_samples: int = 37,
        beamforming_tx_scale: float = 1.5,
        beamforming_rx_scale: float = 1.5,
        beamforming_tx_node: str = "auto",
        beamforming_rx_node: str = "auto",
        beamforming_db_scale: bool = False,
        beamforming_dynamic_range_db: float = 40.0,
        beamforming_colormap: str = "jet",
        beamforming_element_pattern: str = "isotropic",
        beamforming_tx_element_pattern: Optional[str] = None,
        beamforming_rx_element_pattern: Optional[str] = None,
        # Range filters
        delay_filter_min_ns: Optional[float] = None,
        delay_filter_max_ns: Optional[float] = None,
        power_filter_min_db: Optional[float] = None,
        power_filter_max_db: Optional[float] = None,
        # Angle filters
        aoa_az_filter_min_deg: Optional[float] = None,
        aoa_az_filter_max_deg: Optional[float] = None,
        aoa_el_filter_min_deg: Optional[float] = None,
        aoa_el_filter_max_deg: Optional[float] = None,
        aod_az_filter_min_deg: Optional[float] = None,
        aod_az_filter_max_deg: Optional[float] = None,
        aod_el_filter_min_deg: Optional[float] = None,
        aod_el_filter_max_deg: Optional[float] = None,
        # Distinct material colors
        use_distinct_material_colors: bool = False,
    ) -> Optional[ViewModel]:
        """Create a ViewModel from a visual frame payload and render state.

        The method may annotate ``raw_frame`` with ``_source`` metadata used by
        beamforming and metrics. Statistics keep the full filtered path mask
        before any Top-K render cap so panel values describe the data selection,
        not only the rendered subset.
        """
        # Beamforming services use the source step when resolving frame metadata.
        if "_source" not in raw_frame:
            raw_frame["_source"] = {}
        raw_frame["_source"]["step"] = step

        try:
            breakdown: dict[str, float] = {}
            detailed_profile = self._detailed_viewmodel_profile_enabled()
            canon = self._get_canonical(step, raw_frame)
            self._maybe_update_material_filter_choices(canon)
            breakdown["canonical_lookup_ms"] = self._last_canonical_lookup_ms
            breakdown["canonical_cache_hit"] = 1.0 if self._last_canonical_cache_hit else 0.0
            if detailed_profile:
                breakdown["mpc_raw_points_count"] = float(canon.points.shape[0])
                breakdown["mpc_raw_segments_count"] = float(canon.lines.shape[0])
                if canon.path_orders is not None:
                    raw_path_count = len(canon.path_orders)
                elif canon.path_id is not None and canon.path_id.size:
                    raw_path_count = int(np.unique(canon.path_id).size)
                else:
                    raw_path_count = 0
                breakdown["mpc_raw_paths_count"] = float(raw_path_count)
            if not self._last_canonical_cache_hit:
                profile_ms = getattr(canon, "profile_ms", None)
                if profile_ms:
                    for name, elapsed_ms in profile_ms.items():
                        breakdown[f"canonical_build_{name}"] = float(elapsed_ms)

            def _norm_sel(val, max_count):
                """Normalize TX/RX selection values to ``all`` or a clamped index."""
                if val == "all":
                    return "all"
                try:
                    if isinstance(val, str):
                        m = re.search(r"(\d+)$", val)
                        if m:
                            idx = int(m.group(1)) - 1  # "TX1" -> 0, "TX2" -> 1
                        else:
                            idx = int(val)
                    else:
                        idx = int(val)
                except (ValueError, TypeError):
                    return "all"
                if max_count <= 0:
                    return "all"
                # clamp to valid range
                return max(0, min(idx, max_count - 1))

            selection_start = time.perf_counter()
            tx_positions_all = np.asarray(raw_frame.get("tx_positions", ()))
            rx_positions_all = np.asarray(raw_frame.get("rx_positions", ()))

            tx_orientations_all = np.asarray(raw_frame.get("tx_orientations", ()))
            rx_orientations_all = np.asarray(raw_frame.get("rx_orientations", ()))
            # Use canonical id space to avoid clamping selections when
            # position arrays are shorter than the TX/RX ids present in pairs.
            tx_space = (
                (int(canon.tx_id.max()) + 1)
                if canon.tx_id is not None and canon.tx_id.size
                else len(tx_positions_all)
            )
            rx_space = (
                (int(canon.rx_id.max()) + 1)
                if canon.rx_id is not None and canon.rx_id.size
                else len(rx_positions_all)
            )
            sel_tx = _norm_sel(selected_tx, tx_space)
            sel_rx = _norm_sel(selected_rx, rx_space)

            if mpc_allowed_orders is None:
                max_o = int(canon.order.max()) if canon.order.size else 0
                mpc_allowed_orders = list(range(max_o + 1))
            if mpc_allowed_types is None:
                max_t = int(canon.itype.max()) if canon.itype.size else 0
                mpc_allowed_types = list(range(max_t + 1))

            allowed_materials_list = (
                None if mpc_allowed_materials is None else list(mpc_allowed_materials)
            )

            derived_show_tx = show_tx_segments
            if allowed_materials_list is not None:
                derived_show_tx = "no-material" in allowed_materials_list

            aod_device_orientation = None
            aoa_device_orientation = None

            if sel_tx != "all" and isinstance(sel_tx, int) and len(tx_orientations_all) > sel_tx:
                ori = tx_orientations_all[sel_tx]
                if len(ori) == 3:
                    aod_device_orientation = (float(ori[0]), float(ori[1]), float(ori[2]))

            if sel_rx != "all" and isinstance(sel_rx, int) and len(rx_orientations_all) > sel_rx:
                ori = rx_orientations_all[sel_rx]
                if len(ori) == 3:
                    aoa_device_orientation = (float(ori[0]), float(ori[1]), float(ori[2]))
            breakdown["selection_ms"] = (time.perf_counter() - selection_start) * 1000.0

            filter_start = time.perf_counter()
            filter_profile: dict[str, float] | None = {} if detailed_profile else None
            point_mask, segment_mask = build_filter_mask(
                canon,
                allowed_orders=list(mpc_allowed_orders),
                allowed_types=list(mpc_allowed_types),
                selected_tx=sel_tx,
                selected_rx=sel_rx,
                allowed_materials=allowed_materials_list,
                show_tx_segments=derived_show_tx,
                delay_min_ns=delay_filter_min_ns,
                delay_max_ns=delay_filter_max_ns,
                power_min_db=power_filter_min_db,
                power_max_db=power_filter_max_db,
                aoa_az_min_deg=aoa_az_filter_min_deg,
                aoa_az_max_deg=aoa_az_filter_max_deg,
                aoa_el_min_deg=aoa_el_filter_min_deg,
                aoa_el_max_deg=aoa_el_filter_max_deg,
                aod_az_min_deg=aod_az_filter_min_deg,
                aod_az_max_deg=aod_az_filter_max_deg,
                aod_el_min_deg=aod_el_filter_min_deg,
                aod_el_max_deg=aod_el_filter_max_deg,
                material_filter_scope=getattr(
                    self.visualizer, "mpc_material_filter_scope", "segment"
                ),
                aoa_device_orientation=aoa_device_orientation,
                aod_device_orientation=aod_device_orientation,
                profile=filter_profile,
            )
            breakdown["filter_ms"] = (time.perf_counter() - filter_start) * 1000.0
            if filter_profile is not None:
                breakdown.update(filter_profile)

            path_mask_start = time.perf_counter()
            filtered_path_mask = None
            visible_segment_path_ids = np.empty((0,), dtype=np.int32)
            segment_path_ids_all = None
            if canon.segment_path_id is not None:
                segment_path_ids_all = canon.segment_path_id.astype(np.int32, copy=False)
            elif canon.path_id is not None and canon.lines.size:
                segment_path_ids_all = canon.path_id[canon.lines[:, 0]].astype(np.int32, copy=False)
            if segment_path_ids_all is not None and canon.path_orders is not None:
                n_paths = len(canon.path_orders)
                filtered_path_mask = np.zeros(n_paths, dtype=bool)
                if segment_mask.any():
                    visible_segment_path_ids = segment_path_ids_all[segment_mask]
                    in_range = (visible_segment_path_ids >= 0) & (
                        visible_segment_path_ids < n_paths
                    )
                    filtered_path_mask[visible_segment_path_ids[in_range]] = True
            stats_path_mask = filtered_path_mask.copy() if filtered_path_mask is not None else None
            breakdown["path_mask_ms"] = (time.perf_counter() - path_mask_start) * 1000.0
            if detailed_profile:
                breakdown["mpc_path_mask_ms"] = breakdown["path_mask_ms"]
                breakdown["mpc_prefilter_visible_segments_count"] = float(
                    np.count_nonzero(segment_mask)
                )
                breakdown["mpc_prefilter_visible_paths_count"] = float(
                    np.count_nonzero(filtered_path_mask) if filtered_path_mask is not None else 0
                )

            topk_visible_paths = None
            topk_total_candidates = None
            topk_start = time.perf_counter()
            if topk_render_enabled and filtered_path_mask is not None:
                max_paths = int(topk_render_max_paths)
                if max_paths > 0:
                    visible_path_ids = np.flatnonzero(filtered_path_mask)
                    topk_total_candidates = int(visible_path_ids.size)
                    if topk_total_candidates > max_paths:
                        path_losses = canon.path_losses
                        if (
                            path_losses is not None
                            and path_losses.size > 0
                            and np.max(visible_path_ids) < path_losses.size
                        ):
                            loss_values = path_losses[visible_path_ids].astype(
                                np.float64, copy=False
                            )
                            loss_values = np.where(np.isfinite(loss_values), loss_values, np.inf)
                            keep_idx = np.argpartition(loss_values, max_paths - 1)[:max_paths]
                            keep_path_ids = visible_path_ids[keep_idx]
                        else:
                            keep_path_ids = visible_path_ids[:max_paths]

                        topk_mask = np.zeros_like(filtered_path_mask, dtype=bool)
                        topk_mask[keep_path_ids] = True
                        filtered_path_mask = topk_mask
                        topk_visible_paths = max_paths

                        if segment_path_ids_all is not None and segment_path_ids_all.size:
                            path_ids = segment_path_ids_all
                            in_range = (path_ids >= 0) & (path_ids < len(filtered_path_mask))
                            keep_segments = np.zeros_like(segment_mask, dtype=bool)
                            keep_segments[in_range] = filtered_path_mask[path_ids[in_range]]
                            segment_mask = segment_mask & keep_segments
                    else:
                        topk_visible_paths = topk_total_candidates
            breakdown["topk_ms"] = (time.perf_counter() - topk_start) * 1000.0
            if detailed_profile:
                breakdown["mpc_topk_ms"] = breakdown["topk_ms"]
                breakdown["mpc_topk_candidates_count"] = float(topk_total_candidates or 0)
                breakdown["mpc_topk_rendered_paths_count"] = float(topk_visible_paths or 0)

            geometry_start = time.perf_counter()
            all_segments_visible = bool(
                canon.lines.size
                and segment_mask.size == canon.lines.shape[0]
                and np.count_nonzero(segment_mask) == canon.lines.shape[0]
            )
            line_payload_ms = 0.0
            point_payload_ms = 0.0
            if all_segments_visible:
                line_payload_start = time.perf_counter()
                allowed_lines = canon.lines
                mpc_lines = canon.lines
                line_payload_ms += (time.perf_counter() - line_payload_start) * 1000.0
                point_payload_start = time.perf_counter()
                point_mask = np.ones(canon.points.shape[0], dtype=bool)
                mpc_points = canon.points
                point_payload_ms += (time.perf_counter() - point_payload_start) * 1000.0
            elif canon.lines.size and np.any(segment_mask):
                line_select_start = time.perf_counter()
                allowed_lines = canon.lines[segment_mask]
                line_payload_ms += (time.perf_counter() - line_select_start) * 1000.0
                point_payload_start = time.perf_counter()
                point_mask = np.zeros(canon.points.shape[0], dtype=bool)
                point_mask[allowed_lines[:, 0]] = True
                point_mask[allowed_lines[:, 1]] = True
                mpc_points = canon.points[point_mask]
                point_payload_ms += (time.perf_counter() - point_payload_start) * 1000.0
                line_remap_start = time.perf_counter()
                mpc_lines = remap_lines(allowed_lines, point_mask)
                line_payload_ms += (time.perf_counter() - line_remap_start) * 1000.0
            else:
                line_payload_start = time.perf_counter()
                allowed_lines = canon.lines[:0]
                mpc_lines = np.empty((0, 2), dtype=np.int32)
                line_payload_ms += (time.perf_counter() - line_payload_start) * 1000.0
                point_payload_start = time.perf_counter()
                point_mask = np.zeros(canon.points.shape[0], dtype=bool)
                mpc_points = canon.points[:0]
                point_payload_ms += (time.perf_counter() - point_payload_start) * 1000.0
            if detailed_profile:
                breakdown["mpc_line_payload_build_ms"] = line_payload_ms
                breakdown["mpc_point_payload_build_ms"] = point_payload_ms
                breakdown["mpc_all_segments_visible"] = 1.0 if all_segments_visible else 0.0

            segment_metadata_start = time.perf_counter()
            if canon.segment_start_indices is not None and segment_mask.size:
                line_start_indices = (
                    canon.segment_start_indices
                    if all_segments_visible
                    else canon.segment_start_indices[segment_mask]
                )
            else:
                line_start_indices = (
                    allowed_lines[:, 0] if allowed_lines.size else np.empty((0,), dtype=np.int32)
                )
            if detailed_profile:
                breakdown["mpc_segment_metadata_ms"] = (
                    time.perf_counter() - segment_metadata_start
                ) * 1000.0
            breakdown["geometry_remap_ms"] = (time.perf_counter() - geometry_start) * 1000.0
            if detailed_profile:
                breakdown["mpc_segment_remap_ms"] = breakdown["geometry_remap_ms"]

            color_start = time.perf_counter()
            material_colors_start = time.perf_counter()
            material_colors = self._resolve_material_colors(use_distinct_material_colors)
            if detailed_profile:
                breakdown["mpc_material_colors_ms"] = (
                    time.perf_counter() - material_colors_start
                ) * 1000.0
            if color_mode in ("reflection_order", "mpc_type"):
                point_color_mode = "material"
            else:
                point_color_mode = color_mode

            line_color_start = time.perf_counter()
            mpc_colors = colorize_segments(
                canon=canon,
                segment_mask=segment_mask,
                mode=color_mode,
                order_palette=self._order_palette,
                type_palette=self._type_palette,
                viridis256=self._viridis256,
                material_colors=material_colors,
            )
            if detailed_profile:
                breakdown["mpc_line_color_ms"] = (time.perf_counter() - line_color_start) * 1000.0

            if mpc_visibility.effective_bounce_points:
                bounce_mask_start = time.perf_counter()
                bounce_mask = self._build_bounce_point_mask(
                    canon=canon,
                    point_mask=point_mask,
                    allowed_materials=allowed_materials_list,
                    material_filter_scope=getattr(
                        self.visualizer, "mpc_material_filter_scope", "segment"
                    ),
                )
                if detailed_profile:
                    breakdown["mpc_bounce_mask_ms"] = (
                        time.perf_counter() - bounce_mask_start
                    ) * 1000.0
                bounce_payload_start = time.perf_counter()
                bounce_points = canon.points[bounce_mask]
                if detailed_profile:
                    breakdown["mpc_bounce_payload_build_ms"] = (
                        time.perf_counter() - bounce_payload_start
                    ) * 1000.0
                bounce_color_start = time.perf_counter()
                bounce_colors = colorize(
                    canon=canon,
                    mask=bounce_mask,
                    mode=point_color_mode,
                    order_palette=self._order_palette,
                    type_palette=self._type_palette,
                    viridis256=self._viridis256,
                    material_colors=material_colors,
                )
                if detailed_profile:
                    breakdown["mpc_bounce_color_ms"] = (
                        time.perf_counter() - bounce_color_start
                    ) * 1000.0
            else:
                bounce_mask = np.zeros(canon.points.shape[0], dtype=bool)
                bounce_points = canon.points[:0]
                bounce_colors = np.empty((0, 3), dtype=np.float32)
                if detailed_profile:
                    breakdown["mpc_bounce_mask_ms"] = 0.0
                    breakdown["mpc_bounce_payload_build_ms"] = 0.0
                    breakdown["mpc_bounce_color_ms"] = 0.0

            visible_segment_orders = (
                (canon.segment_order if all_segments_visible else canon.segment_order[segment_mask])
                if canon.segment_order is not None
                else (
                    canon.order[line_start_indices]
                    if line_start_indices.size and canon.order.size
                    else np.empty((0,), dtype=np.uint8)
                )
            )
            visible_segment_types = (
                (canon.segment_itype if all_segments_visible else canon.segment_itype[segment_mask])
                if canon.segment_itype is not None
                else (
                    canon.itype[line_start_indices]
                    if line_start_indices.size and canon.itype.size
                    else np.empty((0,), dtype=np.uint8)
                )
            )

            if visible_segment_orders.size == 0:
                mpc_colors = np.empty((0, 3), dtype=np.float32)

            mpc_line_itypes = (
                visible_segment_types.astype(np.uint8, copy=False)
                if visible_segment_types.size > 0
                else None
            )
            breakdown["colorize_ms"] = (time.perf_counter() - color_start) * 1000.0

            prep_start = time.perf_counter()
            _renderer = getattr(self.visualizer, "renderer", None)
            _prefers_f32 = renderer_capabilities(_renderer).prefer_float32_frame_data
            _float_dtype = np.float32 if _prefers_f32 else np.float64
            points_converted = (
                mpc_points.dtype != _float_dtype or not mpc_points.flags["C_CONTIGUOUS"]
            )
            lines_converted = mpc_lines.dtype != np.int32 or not mpc_lines.flags["C_CONTIGUOUS"]
            line_colors_converted = (
                mpc_colors.dtype != _float_dtype or not mpc_colors.flags["C_CONTIGUOUS"]
            )
            bounce_points_converted = (
                bounce_points.dtype != _float_dtype or not bounce_points.flags["C_CONTIGUOUS"]
            )
            bounce_colors_converted = (
                bounce_colors.dtype != _float_dtype or not bounce_colors.flags["C_CONTIGUOUS"]
            )
            if points_converted:
                mpc_points = np.ascontiguousarray(mpc_points, dtype=_float_dtype)
            if lines_converted:
                mpc_lines = np.ascontiguousarray(mpc_lines, dtype=np.int32)
            if line_colors_converted:
                mpc_colors = np.ascontiguousarray(mpc_colors, dtype=_float_dtype)
            if bounce_points_converted:
                bounce_points = np.ascontiguousarray(bounce_points, dtype=_float_dtype)
            if bounce_colors_converted:
                bounce_colors = np.ascontiguousarray(bounce_colors, dtype=_float_dtype)
            breakdown["renderer_prep_ms"] = (time.perf_counter() - prep_start) * 1000.0
            if detailed_profile:
                breakdown["mpc_renderer_prep_ms"] = breakdown["renderer_prep_ms"]
                breakdown["mpc_renderer_points_converted_count"] = 1.0 if points_converted else 0.0
                breakdown["mpc_renderer_lines_converted_count"] = 1.0 if lines_converted else 0.0
                breakdown["mpc_renderer_line_colors_converted_count"] = (
                    1.0 if line_colors_converted else 0.0
                )
                breakdown["mpc_renderer_bounce_points_converted_count"] = (
                    1.0 if bounce_points_converted else 0.0
                )
                breakdown["mpc_renderer_bounce_colors_converted_count"] = (
                    1.0 if bounce_colors_converted else 0.0
                )

            # Selection filters MPCs and effective visibility only.  Node arrays
            # remain in canonical inventory order so stable node IDs, camera
            # anchors, and orientation frames never depend on a UI selection.
            tx_positions = tx_positions_all
            rx_positions = rx_positions_all
            tx_orientations = tx_orientations_all
            rx_orientations = rx_orientations_all

            # Colorbar ranges
            colorbar = None
            if color_mode == "delay":
                colorbar = ("Delay (ns)", (canon.delay_min, canon.delay_max))
            elif color_mode == "path_loss":
                colorbar = ("Path Loss (dB)", (canon.loss_min, canon.loss_max))

            # Targets (reuse existing logic for extraction)
            targets_start = time.perf_counter()
            target_positions_list: list[Any] = []
            target_orientations_list: list[Any] = []
            target_mesh_files = []
            target_use_ply_positions = []
            target_metadata = []

            if (
                include_targets
                and "targets_metadata" in raw_frame
                and raw_frame["targets_metadata"]
            ):
                frame_target_pos = raw_frame.get("target_pos", None)
                if frame_target_pos is not None and hasattr(frame_target_pos, "shape"):
                    if len(frame_target_pos.shape) == 1:
                        frame_target_pos = frame_target_pos.reshape(1, -1)
                elif frame_target_pos is not None and hasattr(frame_target_pos, "__len__"):
                    frame_target_pos = np.array(frame_target_pos)
                    if len(frame_target_pos.shape) == 1:
                        frame_target_pos = frame_target_pos.reshape(1, -1)
                else:
                    frame_target_pos = None

                for i, target_meta in enumerate(raw_frame["targets_metadata"]):
                    # Keep a validity flag so renderers can hide targets with
                    # missing positions instead of treating [0, 0, 0] as data.
                    current_position = target_meta.get("current_position", None)
                    if (
                        current_position is None
                        and frame_target_pos is not None
                        and i < len(frame_target_pos)
                    ):
                        current_position = frame_target_pos[i]

                    position_valid = False
                    if current_position is not None:
                        pos_arr = np.asarray(current_position, dtype=np.float64).reshape(-1)
                        if pos_arr.size >= 3 and np.all(np.isfinite(pos_arr[:3])):
                            target_positions_list.append(
                                [float(pos_arr[0]), float(pos_arr[1]), float(pos_arr[2])]
                            )
                            position_valid = True
                        else:
                            target_positions_list.append([0.0, 0.0, 0.0])
                    else:
                        target_positions_list.append([0.0, 0.0, 0.0])

                    orientation = target_meta.get("orientation", [0, 0, 0])
                    if hasattr(orientation, "tolist"):
                        orientation = orientation.tolist()
                    target_orientations_list.append(orientation)

                    mesh_file = target_meta.get("mesh_file", "")
                    target_mesh_files.append(mesh_file)

                    use_ply_position = target_meta.get("use_ply_position", False)
                    target_use_ply_positions.append(use_ply_position)

                    target_meta_entry = dict(target_meta)
                    target_meta_entry["position_valid"] = bool(position_valid)
                    target_metadata.append(target_meta_entry)

            target_positions = (
                np.array(target_positions_list)
                if target_positions_list
                else np.empty((0, 3), dtype=np.float64)
            )
            target_orientations = (
                np.array(target_orientations_list)
                if target_orientations_list
                else np.empty((0, 3), dtype=np.float64)
            )
            breakdown["targets_ms"] = (time.perf_counter() - targets_start) * 1000.0

            summary_start = time.perf_counter()
            summary_compute_start = time.perf_counter()
            summary = self._summarize_mpcs(
                canon=canon,
                color_mode=color_mode,
                point_mask=point_mask,
                segment_mask=segment_mask,
                path_mask=filtered_path_mask,
                rendered_line_itypes=mpc_line_itypes,
            )
            mpc_line_itype_codes = tuple(summary["mpc_line_itype_codes"])
            if detailed_profile:
                breakdown["mpc_summary_compute_ms"] = (
                    time.perf_counter() - summary_compute_start
                ) * 1000.0

            summary_text_start = time.perf_counter()
            visible = summary["total_mpcs"]
            total = summary["all_mpcs"]
            if total > 0 and visible != total:
                mpc_status_text = f"MPCs: {visible:,}/{total:,}"
            else:
                mpc_status_text = f"MPCs: {visible:,}"
            line0 = f"{mpc_status_text} | Segments: {summary['total_segments']:,}"
            if topk_render_enabled and topk_visible_paths is not None:
                line0 += f" | Rendered MPCs: {topk_visible_paths:,}"
                if topk_total_candidates is not None and topk_total_candidates > topk_visible_paths:
                    line0 += f"/{topk_total_candidates:,}"

            orders_hist_start = time.perf_counter()
            orders_hist = {}
            if (
                stats_path_mask is not None
                and canon.path_orders is not None
                and canon.path_orders.size
            ):
                orders_hist = self._int_count_dict(
                    self._masked_1d_values(canon.path_orders, stats_path_mask)
                )
            elif len(mpc_points) > 0 and canon.order.size > 0:
                orders_hist = self._int_count_dict(self._masked_1d_values(canon.order, point_mask))
            if detailed_profile:
                breakdown["mpc_summary_order_hist_ms"] = (
                    time.perf_counter() - orders_hist_start
                ) * 1000.0

            # Metrics consumers accept canonical summaries through raw-frame metadata.
            if "reflection_order_counts" not in raw_frame:
                raw_frame["reflection_order_counts"] = orders_hist

            self._last_reflection_order_counts = orders_hist

            if "delay_range" not in raw_frame:
                raw_frame["delay_range"] = (canon.delay_min, canon.delay_max)
            if "path_loss_range" not in raw_frame:
                raw_frame["path_loss_range"] = (canon.loss_min, canon.loss_max)

            if len(tx_positions) > 0 or len(rx_positions) > 0:
                if sel_tx != "all" and sel_rx != "all":
                    # Show actual selected TX/RX numbers (convert 0-based to 1-based for display)
                    line0 += f" | TX{int(sel_tx) + 1}, RX{int(sel_rx) + 1}"
                elif sel_tx != "all":
                    # Show selected TX and total RX count
                    line0 += f" | TX{int(sel_tx) + 1}, RX: {len(rx_positions)}"
                elif sel_rx != "all":
                    # Show total TX count and selected RX
                    line0 += f" | TX: {len(tx_positions)}, RX{int(sel_rx) + 1}"
                else:
                    # Show total counts
                    line0 += f" | TX: {len(tx_positions)}, RX: {len(rx_positions)}"
            if len(target_positions) > 0:
                line0 += f" | Targets: {len(target_positions)}"

            stats_lines = [line0]
            if summary["header"] and summary["lines"]:
                stats_lines.append(summary["header"])
                stats_lines.extend(summary["lines"])

            stats_text = "\n".join(stats_lines)
            breakdown["summary_ms"] = (time.perf_counter() - summary_start) * 1000.0
            if detailed_profile:
                breakdown["mpc_summary_text_ms"] = (
                    time.perf_counter() - summary_text_start
                ) * 1000.0

            beamforming_meshes: tuple[BeamformingSurface, ...] = ()
            beamforming_info = None
            beamforming_pairs_metadata = None
            beamforming_start = time.perf_counter()
            if show_beamforming and self.beamforming_service:
                beamforming_result = self.beamforming_service.build_meshes(
                    raw_frame,
                    tx_positions_all,
                    rx_positions_all,
                    tx_orientations_all,
                    rx_orientations_all,
                    canonical_data=canon,
                    beamforming_tx_node=beamforming_tx_node,
                    beamforming_rx_node=beamforming_rx_node,
                    selected_tx=sel_tx,
                    selected_rx=sel_rx,
                    step=step,
                    beamforming_azimuth_samples=beamforming_azimuth_samples,
                    beamforming_elevation_samples=beamforming_elevation_samples,
                    beamforming_tx_scale=beamforming_tx_scale,
                    beamforming_rx_scale=beamforming_rx_scale,
                    beamforming_db_scale=beamforming_db_scale,
                    beamforming_dynamic_range_db=beamforming_dynamic_range_db,
                    beamforming_colormap=beamforming_colormap,
                    beamforming_element_pattern=beamforming_element_pattern,
                    beamforming_tx_element_pattern=beamforming_tx_element_pattern,
                    beamforming_rx_element_pattern=beamforming_rx_element_pattern,
                )
                if beamforming_result:
                    beamforming_meshes = tuple(beamforming_result.get("meshes") or ())
                    beamforming_info = beamforming_result.get("info")
                    if beamforming_info:
                        beamforming_pairs_metadata = beamforming_info.get("pairs")
            breakdown["beamforming_ms"] = (time.perf_counter() - beamforming_start) * 1000.0

            if not mpc_visibility.effective_paths:
                mpc_points = canon.points[:0]
                mpc_lines = np.empty((0, 2), dtype=np.int32)
                mpc_colors = np.empty((0, 3), dtype=np.float32)
                mpc_line_itypes = None
                mpc_line_itype_codes = ()

            viewmodel_construct_start = time.perf_counter()
            result = ViewModel(
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                tx_orientations=tx_orientations,
                rx_orientations=rx_orientations,
                mpc_points=mpc_points,
                mpc_lines=mpc_lines,
                mpc_colors=mpc_colors,
                mpc_bounce_points=bounce_points,
                mpc_bounce_colors=bounce_colors,
                mpc_bounce_itypes=(
                    canon.itype[bounce_mask].astype(np.uint8, copy=False)
                    if canon.itype is not None and canon.itype.size and bounce_mask.any()
                    else None
                ),
                colorbar=colorbar,
                stats_text=stats_text,
                mpc_visibility=mpc_visibility,
                target_positions=target_positions,
                target_orientations=target_orientations,
                target_mesh_files=target_mesh_files,
                target_use_ply_positions=target_use_ply_positions,
                target_metadata=target_metadata,
                coverage_vertices=None,
                coverage_triangles=None,
                coverage_colors=None,
                coverage_isoline_points=None,
                coverage_isoline_lines=None,
                coverage_isoline_colors=None,
                coverage_signature=None,
                coverage_metadata=None,
                coverage_opacity=1.0,
                show_coverage=False,
                mpc_line_itypes=mpc_line_itypes,
                mpc_line_itype_codes=mpc_line_itype_codes,
                canonical_data=canon,  # Statistics/panels use canonical per-path arrays.
                path_mask=stats_path_mask,  # Keep stats independent of Top-K render cap
                segment_mask=np.array(segment_mask, dtype=bool, copy=True),
                beamforming_meshes=beamforming_meshes,
                beamforming_pairs=beamforming_pairs_metadata,
                beamforming_info=beamforming_info,
            )
            if detailed_profile:
                breakdown["mpc_viewmodel_construct_ms"] = (
                    time.perf_counter() - viewmodel_construct_start
                ) * 1000.0
                breakdown["mpc_visible_points_count"] = float(mpc_points.shape[0])
                breakdown["mpc_visible_segments_count"] = float(mpc_lines.shape[0])
                breakdown["mpc_visible_paths_count"] = float(visible)
                breakdown["mpc_bounce_points_count"] = float(bounce_points.shape[0])
                breakdown["mpc_line_payload_bytes"] = float(
                    mpc_points.nbytes
                    + mpc_lines.nbytes
                    + mpc_colors.nbytes
                    + (0 if mpc_line_itypes is None else mpc_line_itypes.nbytes)
                )
                breakdown["mpc_point_payload_bytes"] = float(
                    bounce_points.nbytes
                    + bounce_colors.nbytes
                    + (0 if result.mpc_bounce_itypes is None else result.mpc_bounce_itypes.nbytes)
                )

            self._last_viewmodel_breakdown = {
                key: round(float(value), 3) for key, value in breakdown.items()
            }
            return result
        except (KeyError, IndexError, ValueError, TypeError) as e:
            self._last_viewmodel_breakdown = {}
            self.logger.error(f"Canonical path failed: {e}")
            return None

    @staticmethod
    def _ordinal(n: int) -> str:
        """Format an integer using an English ordinal suffix."""
        if n <= 0:
            return "0"
        if 10 <= (n % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    @staticmethod
    def _normalize_material_name(value) -> str:
        """Normalize material metadata for summary display."""
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = ""
        name = str(value).strip() if value is not None else ""
        return name if name else "no-material"

    @classmethod
    def _normalize_material_filter_label(
        cls,
        value: Any,
        *,
        include_no_material: bool = False,
    ) -> str:
        """Return the canonical label used by MPC material filter controls."""
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        if value is None:
            return ""

        label = str(value).strip().lower().replace(" ", "_")
        if label in {"", "none", "null"}:
            return "no-material" if include_no_material else ""
        if label in {"no-material", "no_material"}:
            return "no-material" if include_no_material else ""

        prefixes = (
            "mat-itu_",
            "mat_itu_",
            "mat-itu-",
            "mat-ground_",
            "mat_ground_",
            "itu_",
            "itu-",
            "ground_",
            "mat-",
        )
        previous = None
        while label and label != previous:
            previous = label
            for prefix in prefixes:
                if label.startswith(prefix) and len(label) > len(prefix):
                    label = label[len(prefix) :]
                    break
            else:
                label = _bare_material_type(label)

        if label in {"", "none", "null"}:
            return "no-material" if include_no_material else ""
        if label in {"no-material", "no_material"}:
            return "no-material" if include_no_material else ""
        return label

    @classmethod
    def _normalize_material_filter_labels(
        cls,
        values: Iterable[Any],
        *,
        include_no_material: bool = True,
    ) -> set[str]:
        """Normalize a sequence of MPC material filter labels."""
        labels: set[str] = set()
        for value in values:
            label = cls._normalize_material_filter_label(
                value,
                include_no_material=include_no_material,
            )
            if label:
                labels.add(label)
        return labels

    @classmethod
    def _normalize_frame_material_filter_label(cls, name: Any, itu_name: Any = None) -> str:
        """Return the user-facing material family for one frame material entry."""
        label = cls._normalize_material_filter_label(name)
        itu_label = cls._normalize_material_filter_label(itu_name)
        if not label:
            return itu_label
        if itu_label and label != itu_label and label.startswith(f"{itu_label}_"):
            return itu_label
        return label

    def _build_range_summary(self, values: np.ndarray, unit: str, bins: int = 5) -> list[str]:
        """Build compact histogram lines for finite scalar MPC metadata."""
        if values.size == 0:
            return []
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return []
        vmin = float(finite.min())
        vmax = float(finite.max())
        if np.isclose(vmin, vmax):
            return [f"• = {vmin:.2f} {unit}: {len(finite)}"]
        edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float32)
        counts, _ = np.histogram(finite, bins=edges)
        lines = []
        for idx, count in enumerate(counts):
            if count <= 0:
                continue
            left = edges[idx]
            right = edges[idx + 1]
            interval = f"{left:.2f}–{right:.2f}"
            if idx == len(counts) - 1:
                interval = f"{left:.2f}–{right:.2f}"
            lines.append(f"• {interval} {unit}: {int(count)}")
        return lines

    @staticmethod
    def _masked_1d_values(values: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Return 1-D values selected by a matching boolean mask without needless copies."""
        arr = np.asarray(values)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if mask is None:
            return arr
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 1 or mask_arr.shape[0] != arr.shape[0]:
            return arr[:0]
        selected_count = int(np.count_nonzero(mask_arr))
        if selected_count == 0:
            return arr[:0]
        if selected_count == arr.shape[0]:
            return arr
        return arr[mask_arr]

    @staticmethod
    def _int_count_pairs(values: np.ndarray) -> list[tuple[int, int]]:
        """Return sorted integer value/count pairs using bincount when possible."""
        arr = np.asarray(values)
        if arr.size == 0:
            return []
        int_values = arr.reshape(-1).astype(np.int64, copy=False)
        if int_values.size == 0:
            return []
        if np.any(int_values < 0):
            unique_values, counts = np.unique(int_values, return_counts=True)
            return [
                (int(value), int(count))
                for value, count in zip(unique_values, counts)
                if int(count) > 0
            ]
        counts = np.bincount(int_values)
        nonzero = np.flatnonzero(counts)
        return [(int(value), int(counts[value])) for value in nonzero]

    @classmethod
    def _int_count_dict(cls, values: np.ndarray) -> dict[int, int]:
        """Return integer histogram counts as a normal dict."""
        return {value: count for value, count in cls._int_count_pairs(values)}

    def _reflection_order_summary_lines(self, orders: np.ndarray) -> list[str]:
        """Build reflection-order summary lines from path-order values."""
        bucket_counts: dict[int, int] = {}
        for order_val, count in self._int_count_pairs(orders):
            bucket = 6 if order_val >= 6 else order_val
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + count

        lines: list[str] = []
        for order_val in sorted(bucket_counts):
            count = bucket_counts[order_val]
            if order_val == 0:
                label = "LoS"
            elif order_val >= 6:
                label = "6+ Order"
            else:
                label = f"{self._ordinal(order_val)} Order"
            lines.append(f"- {label}: {count}")
        return lines

    def _mpc_type_summary_lines(self, type_values: np.ndarray) -> list[str]:
        """Build MPC-type summary lines from visible segment type values."""
        return self._mpc_type_summary_lines_from_counts(self._int_count_pairs(type_values))

    @staticmethod
    def _mpc_type_summary_lines_from_counts(
        type_counts: Sequence[tuple[int, int]],
    ) -> list[str]:
        """Build MPC-type summary lines from an existing segment histogram."""
        entries: list[tuple[tuple[int, int], str, int]] = []
        for type_val, count in type_counts:
            entries.append(
                (
                    mpc_interaction_sort_key(type_val),
                    mpc_interaction_label(type_val, explicit_unknown=True),
                    count,
                )
            )
        entries.sort(key=lambda item: item[0])
        return [f"- {label}: {count}" for _, label, count in entries]

    def _build_bounce_point_mask(
        self,
        *,
        canon: CanonicalStepData,
        point_mask: np.ndarray,
        allowed_materials: Optional[Sequence[str]],
        material_filter_scope: str,
    ) -> np.ndarray:
        """Return a point mask for physical bounce markers only.

        TX and RX endpoints are removed so bounce markers represent physical
        interactions. Segment-scoped material filtering also applies here so
        the bounce cloud matches visible MPC segments.
        """
        bounce_mask = np.zeros(canon.points.shape[0], dtype=bool)
        if not point_mask.any():
            return bounce_mask

        interior = point_mask.copy()
        if canon.path_start_indices is not None and canon.path_start_indices.size:
            starts = canon.path_start_indices.astype(np.int32, copy=False)
            valid_starts = starts[(starts >= 0) & (starts < canon.points.shape[0])]
            interior[valid_starts] = False
            if valid_starts.size:
                sorted_starts = np.sort(valid_starts)
                ends = np.empty_like(sorted_starts)
                ends[:-1] = sorted_starts[1:] - 1
                ends[-1] = canon.points.shape[0] - 1
                ends = ends[(ends >= 0) & (ends < canon.points.shape[0])]
                interior[ends] = False
        elif canon.material_ids is not None and canon.material_ids.size == canon.points.shape[0]:
            interior &= canon.material_ids != 0

        if (
            allowed_materials is not None
            and str(material_filter_scope).lower() not in {"path", "mpc", "whole_path", "whole-mpc"}
            and canon.material_ids is not None
            and canon.material_id_to_name is not None
            and canon.material_id_to_bare is not None
        ):
            allowed_set = set(allowed_materials)
            canonical_allowed = {_bare_material_type(str(name)) for name in allowed_set}
            allowed_ids: list[int] = []
            for mid, name in canon.material_id_to_name.items():
                mid_int = int(mid)
                if mid_int == 0:
                    continue
                if (
                    name in allowed_set
                    or canon.material_id_to_bare.get(mid_int, "") in canonical_allowed
                ):
                    allowed_ids.append(mid_int)
                    continue
                if canon.material_id_to_itu is not None:
                    itu = str(canon.material_id_to_itu.get(mid_int, "")).strip().lower()
                    if itu and itu in canonical_allowed:
                        allowed_ids.append(mid_int)
            if allowed_ids:
                material_ok = np.isin(canon.material_ids, np.asarray(allowed_ids, dtype=np.int16))
            else:
                material_ok = np.zeros(canon.points.shape[0], dtype=bool)
            if "no-material" in allowed_set:
                material_ok |= canon.material_ids == 0
            interior &= material_ok

        bounce_mask[interior] = True
        return bounce_mask

    def _summarize_mpcs(
        self,
        canon: CanonicalStepData,
        color_mode: str,
        point_mask: np.ndarray,
        segment_mask: np.ndarray,
        path_mask: Optional[np.ndarray] = None,
        rendered_line_itypes: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Summarize filtered MPC counts and color-mode ranges for panels."""
        total_segments = int(np.count_nonzero(segment_mask)) if canon.lines.size else 0

        # Total unfiltered path count is already available in the per-path arrays.
        if canon.path_orders is not None:
            all_path_count = int(len(canon.path_orders))
        elif canon.path_id is not None:
            all_path_count = int(np.unique(canon.path_id).size)
        else:
            all_path_count = 0

        summary: dict[str, object] = {
            "total_mpcs": 0,
            "all_mpcs": all_path_count,
            "total_segments": total_segments,
            "header": None,
            "lines": [],
            "mpc_line_itype_codes": (),
        }

        type_counts: list[tuple[int, int]] = []
        if color_mode == "mpc_type" and total_segments > 0:
            if rendered_line_itypes is not None:
                type_values = np.asarray(rendered_line_itypes)
            elif canon.segment_itype is not None:
                type_values = canon.segment_itype[segment_mask]
            else:
                type_values = canon.itype[canon.lines[segment_mask, 0]]
            type_counts = self._int_count_pairs(type_values)
            summary["mpc_line_itype_codes"] = tuple(value for value, _count in type_counts)
            if type_counts:
                summary["header"] = "MPC Type — Segments"
                summary["lines"] = self._mpc_type_summary_lines_from_counts(type_counts)

        path_mask_arr: Optional[np.ndarray] = None
        if path_mask is not None and canon.path_orders is not None:
            candidate = np.asarray(path_mask, dtype=bool)
            if candidate.ndim == 1 and candidate.shape[0] == all_path_count:
                path_mask_arr = candidate

        if canon.path_id is None and path_mask_arr is None:
            return summary

        unique_path_ids: Optional[np.ndarray] = None
        if path_mask_arr is not None:
            visible_path_count = int(np.count_nonzero(path_mask_arr))
        else:
            path_ids_from_segments = np.empty((0,), dtype=np.int32)
            if total_segments > 0 and canon.path_id is not None:
                if canon.segment_path_id is not None:
                    path_ids_from_segments = canon.segment_path_id[segment_mask].astype(
                        np.int32, copy=False
                    )
                else:
                    segment_start_nodes = canon.lines[segment_mask, 0]
                    path_ids_from_segments = canon.path_id[segment_start_nodes].astype(
                        np.int32, copy=False
                    )

            if path_ids_from_segments.size == 0 and canon.path_id is not None and point_mask.any():
                path_ids_from_segments = canon.path_id[point_mask].astype(np.int32, copy=False)

            unique_path_ids = np.unique(path_ids_from_segments)
            visible_path_count = int(unique_path_ids.size)

        summary["total_mpcs"] = visible_path_count

        if visible_path_count == 0 and total_segments == 0:
            return summary

        if color_mode == "reflection_order" and canon.path_orders is not None:
            orders = (
                self._masked_1d_values(canon.path_orders, path_mask_arr)
                if path_mask_arr is not None
                else (
                    canon.path_orders[unique_path_ids]
                    if unique_path_ids is not None and unique_path_ids.size
                    else np.empty((0,), dtype=np.uint8)
                )
            )
            if orders.size:
                summary["header"] = "Reflection Order — MPCs"
                summary["lines"] = self._reflection_order_summary_lines(orders)
            return summary

        if color_mode == "mpc_type":
            return summary

        if color_mode == "material" and (
            canon.material_ids is not None or canon.material_names is not None
        ):
            if total_segments == 0:
                return summary
            segment_start_nodes = (
                canon.segment_start_indices[segment_mask]
                if canon.segment_start_indices is not None
                else canon.lines[segment_mask, 0]
            )
            material_counts: dict[str, int] = {}
            if canon.material_ids is not None and canon.material_id_to_name is not None:
                segment_material_ids = canon.material_ids[segment_start_nodes].astype(
                    np.int32, copy=False
                )
                for mid, count in self._int_count_pairs(segment_material_ids):
                    label = self._normalize_material_name(
                        canon.material_id_to_name.get(int(mid), "")
                    )
                    material_counts[label] = material_counts.get(label, 0) + int(count)
            elif canon.material_names is not None:
                raw_names = canon.material_names[segment_start_nodes]
                for name in raw_names:
                    norm = self._normalize_material_name(name)
                    material_counts[norm] = material_counts.get(norm, 0) + 1
            if material_counts:
                ordered_materials = sorted(material_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                top = ordered_materials[:6]
                remainder = ordered_materials[6:]
                lines = [f"• {label}: {count}" for label, count in top]
                if remainder:
                    lines.append(f"• others: {sum(c for _, c in remainder)}")
                summary["header"] = "Material — Segments"
                summary["lines"] = lines
            return summary

        if color_mode == "delay" and canon.path_delays is not None:
            values = (
                self._masked_1d_values(canon.path_delays, path_mask_arr)
                if path_mask_arr is not None
                else (
                    canon.path_delays[unique_path_ids]
                    if unique_path_ids is not None and unique_path_ids.size
                    else np.empty((0,), dtype=np.float32)
                )
            )
            lines = self._build_range_summary(values, "ns")
            if lines:
                summary["header"] = "Delay — MPCs"
                summary["lines"] = lines
            return summary

        if color_mode == "path_loss" and canon.path_losses is not None:
            values = (
                self._masked_1d_values(canon.path_losses, path_mask_arr)
                if path_mask_arr is not None
                else (
                    canon.path_losses[unique_path_ids]
                    if unique_path_ids is not None and unique_path_ids.size
                    else np.empty((0,), dtype=np.float32)
                )
            )
            lines = self._build_range_summary(values, "dB")
            if lines:
                summary["header"] = "Path Loss — MPCs"
                summary["lines"] = lines
            return summary

        # Fallback: provide reflection order summary if none matched and data exists
        if visible_path_count and canon.path_orders is not None:
            orders = (
                self._masked_1d_values(canon.path_orders, path_mask_arr)
                if path_mask_arr is not None
                else (
                    canon.path_orders[unique_path_ids]
                    if unique_path_ids is not None and unique_path_ids.size
                    else np.empty((0,), dtype=np.uint8)
                )
            )
            if orders.size:
                summary["header"] = "Reflection Order — MPCs"
                summary["lines"] = self._reflection_order_summary_lines(orders)

        return summary

    def _resolve_material_colors(self, use_distinct: bool) -> Optional[dict[str, np.ndarray]]:
        """Return material color dict — ITU colors or distinct palette.

        Args:
            use_distinct: If True, replace ITU colors with maximally distinct
                          colors from tab20/tab20b/HSV colormaps.

        Returns:
            Mapping from material name to (3,) float32 RGB array, or None.
        """
        base_colors = dict(self._get_material_colors() or {})
        frame_materials = sorted(getattr(self.visualizer, "_mpc_material_filter_choices", set()))
        for name in frame_materials:
            if name == "no-material":
                base_colors[name] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                continue
            if name not in base_colors:
                existing_color = self._lookup_material_color(base_colors, name)
                if existing_color is not None:
                    base_colors[name] = existing_color
                    continue
                preset_color = self._preset_material_color(name)
                if preset_color is not None:
                    base_colors[name] = preset_color

        if not base_colors:
            return None
        if not use_distinct:
            return base_colors

        advertised_groups = {
            self._material_color_group_key(name)
            for name in frame_materials
            if name != "no-material"
        }
        advertised_groups.discard("")
        groups: dict[str, set[str]] = {
            group: set(self._material_color_group_aliases(group))
            for group in sorted(advertised_groups)
        }

        for name in base_colors:
            if name == "no-material":
                continue
            group = self._material_color_group_key(name)
            if not group:
                continue
            family = self._material_color_family_for_groups(group, advertised_groups)
            if family:
                groups.setdefault(family, set()).add(name)
            elif not advertised_groups:
                groups.setdefault(group, set()).add(name)

        names = sorted(groups)
        palette = get_distinct_material_palette(len(names))
        distinct: dict[str, np.ndarray] = {}
        for i, group in enumerate(names):
            color = palette[i]
            for alias in self._material_color_group_aliases(group):
                distinct[alias] = color
            for alias in groups[group]:
                distinct[alias] = color
        if "no-material" in base_colors:
            distinct["no-material"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return distinct

    def material_legend_items(
        self,
        use_distinct: bool = False,
        *,
        active_only: bool = False,
    ) -> list[tuple[str, np.ndarray]]:
        """Return canonical material labels and colors in display order.

        ``active_only=True`` exposes only an explicit MPC material allow-list
        for HUD filter swatches. A normalized ``None`` allow-list means that no
        material filter is active and therefore returns no rows.
        """
        colors = self._resolve_material_colors(use_distinct) or {}
        if active_only:
            allowed = getattr(self.visualizer, "mpc_allowed_materials", None)
            labels = [] if allowed is None else sorted(str(label) for label in allowed)
        else:
            choices = sorted(getattr(self.visualizer, "_mpc_material_filter_choices", set()))
            labels = choices if choices else sorted(colors.keys())
        items: list[tuple[str, np.ndarray]] = []
        for label in labels:
            color = self._lookup_material_color(colors, label)
            if color is None:
                color = np.full(3, MPC_MATERIAL_FALLBACK_GRAY, dtype=np.float32)
            items.append((label, color))
        return items

    @staticmethod
    def _lookup_material_color(colors: dict[str, np.ndarray], label: str) -> Optional[np.ndarray]:
        """Resolve color for a user-facing material label and common aliases."""
        candidates = [
            label,
            _bare_material_type(label),
            f"mat-itu_{label}",
            f"mat-{label}",
            f"itu_{label}",
            f"ground_{label}",
            f"mat-ground_{label}",
        ]
        for candidate in candidates:
            if candidate in colors:
                return np.asarray(colors[candidate], dtype=np.float32)
        return None

    @staticmethod
    def _material_color_group_key(name: str) -> str:
        """Collapse scene material aliases to the label used by MPC filters."""
        key = str(name).strip()
        while key:
            bare = _bare_material_type(key)
            if bare == key:
                return key
            key = bare
        return ""

    @staticmethod
    def _material_color_family_for_groups(group: str, families: set[str]) -> str:
        """Map a raw material-color group to an advertised MPC material family."""
        if not group or not families:
            return ""
        if group in families:
            return group
        normalized = str(group).strip().lower()
        for family in sorted(families, key=len, reverse=True):
            if normalized.startswith(f"{family}_"):
                return family
        return ""

    @staticmethod
    def _material_color_group_aliases(group: str) -> tuple[str, ...]:
        """Return common aliases that should share one distinct material color."""
        aliases = (
            group,
            f"mat-itu_{group}",
            f"itu_{group}",
            f"ground_{group}",
            f"mat-ground_{group}",
        )
        return tuple(dict.fromkeys(alias for alias in aliases if alias))

    @staticmethod
    def _preset_material_color(name: str) -> Optional[np.ndarray]:
        """Return a stable material color from ITU/PBR presets."""
        try:
            from ..materials.catalog import is_known_material_type, material_preset
        except ImportError:
            return None
        bare = _bare_material_type(name)
        candidates = [name, bare, f"ground_{bare}"]
        for candidate in candidates:
            if not is_known_material_type(candidate):
                continue
            preset = material_preset(candidate)
            color = preset.get("color")
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                return np.asarray(color[:3], dtype=np.float32)
        return None

    def invalidate_material_colors_cache(self) -> None:
        """Clear cached scene material colors after scene/material changes."""
        self._material_colors_cache = None

    def _get_material_colors(self) -> Optional[dict[str, np.ndarray]]:
        """
        Extract material colors from the visualizer's mesh entries.
        Returns a mapping from material names to RGB colors.
        Results are cached per scene; call ``invalidate_material_colors_cache()``
        on scene load.
        """
        if self._material_colors_cache is not None:
            return self._material_colors_cache

        if self.visualizer is None:
            self.logger.debug("No visualizer reference available")
            return None

        material_colors = {}

        if hasattr(self.visualizer, "mesh_entries"):
            for entry in self.visualizer.mesh_entries:
                if "material_id" in entry and "color" in entry:
                    # Preserve exact scene material ids for renderer/panel lookups.
                    material_id = str(entry["material_id"]).strip()
                    color = entry["color"]
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        material_colors[material_id] = np.array(color[:3], dtype=np.float32)

        if hasattr(self.visualizer, "target_entries"):
            for entry in self.visualizer.target_entries:
                if entry is not None and "material_id" in entry and "color" in entry:
                    # Preserve exact scene material ids for renderer/panel lookups.
                    material_id = str(entry["material_id"]).strip()
                    color = entry["color"]
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        material_colors[material_id] = np.array(color[:3], dtype=np.float32)

        if material_colors:
            self.logger.debug(
                f"Found {len(material_colors)} materials with colors: {list(material_colors.keys())}"
            )
            for material_id, color in material_colors.items():
                self.logger.debug(f"  {material_id}: RGB{tuple(color)}")
        else:
            self.logger.debug("No material colors found")

        result = material_colors if material_colors else None
        self._material_colors_cache = result
        return result

    def _maybe_update_material_filter_choices(self, canon: CanonicalStepData) -> None:
        """Refresh the MPC material checkbox choices from frame material data."""
        if self.visualizer is None:
            return
        names = set(self._get_all_environment_materials())
        names.update(self._get_all_frame_materials(canon))
        if not names:
            return
        current = getattr(self.visualizer, "_mpc_material_filter_choices", set())
        if names == set(current):
            return
        sorted_names = tuple(sorted(names))
        self.visualizer._mpc_material_filter_choices = set(sorted_names)
        self.visualizer._last_material_keys = sorted_names
        ui_controller = getattr(self.visualizer, "ui_controller", None)
        populate = getattr(ui_controller, "populate_material_filters", None)
        if callable(populate):
            populate()

    def _get_all_frame_materials(self, canon: CanonicalStepData) -> list[str]:
        """Return material filter labels advertised by the current frame."""
        names: set[str] = set()
        if canon.material_id_to_name:
            for material_id, name in canon.material_id_to_name.items():
                if int(material_id) == 0:
                    continue
                itu_name = (
                    canon.material_id_to_itu.get(material_id, "")
                    if canon.material_id_to_itu
                    else ""
                )
                bare = self._normalize_frame_material_filter_label(name, itu_name)
                if bare:
                    names.add(bare)
        if not names and canon.material_id_to_itu:
            for material_id, name in canon.material_id_to_itu.items():
                if int(material_id) == 0:
                    continue
                bare = self._normalize_material_filter_label(name)
                if bare:
                    names.add(bare)
        if not names and canon.material_names is not None:
            for raw in np.asarray(canon.material_names, dtype=object).ravel():
                bare = self._normalize_material_filter_label(raw)
                if bare and bare != "no-material":
                    names.add(bare)
        if (
            canon.material_ids is not None
            and canon.material_ids.size
            and np.any(canon.material_ids == 0)
        ):
            names.add("no-material")
        return sorted(names)

    def _get_all_environment_materials(self) -> list[str]:
        """
        Extract all material IDs from the environment (mesh entries and target entries).
        This includes materials that may not have colors defined.
        Returns a list of all unique material IDs found in the environment.
        """
        if self.visualizer is None:
            self.logger.debug("No visualizer reference available")
            return []

        material_ids = set()

        def _entry_material_label(entry: dict[str, Any]) -> str:
            material_type = self._normalize_material_filter_label(entry.get("material_type"))
            if material_type and material_type != "default":
                return material_type
            return self._normalize_material_filter_label(entry.get("material_id"))

        if hasattr(self.visualizer, "mesh_entries"):
            self.logger.debug(
                f"Checking {len(self.visualizer.mesh_entries)} mesh entries for materials"
            )
            for entry in self.visualizer.mesh_entries:
                if entry is not None:
                    label = _entry_material_label(entry)
                    if label:
                        material_ids.add(label)

        if hasattr(self.visualizer, "target_entries"):
            self.logger.debug(
                "Checking %d target entries for materials",
                len(self.visualizer.target_entries),
            )
            for entry in self.visualizer.target_entries:
                if entry is not None:
                    label = _entry_material_label(entry)
                    if label:
                        material_ids.add(label)
                        self.logger.debug("Found target material: %s", label)

        # Add special pseudo-material for TX segments
        material_ids.add("no-material")

        result = sorted(list(material_ids))
        self.logger.debug("Found %d total environment materials: %s", len(result), result)

        return result
