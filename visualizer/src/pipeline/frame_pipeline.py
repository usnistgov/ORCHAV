"""Frame update pipeline for the ORCHAV visualizer.

``FramePipeline.update`` is the foreground data-flow path:
frame source/live preview -> raw frame cache -> ViewModel cache -> renderer
payload -> services and UI mirrors. It keeps the renderer-facing contract in
``ViewModel`` while allowing raw frames, canonical MPC arrays, coverage meshes,
and renderer-side caches to invalidate on separate scopes.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from shared.logging import get_logger

if TYPE_CHECKING:
    from ..benchmarking.recorder import BenchmarkRecorder
from ..config import MAX_VIEW_MODEL_CACHE_SIZE
from ..coverage.analysis import (
    build_coverage_isoline,
    compute_coverage_threshold_mask,
    coverage_metric_color_scale,
    coverage_metric_colormap,
    coverage_metric_valid_mask,
    is_serving_tx_metric,
    serving_tx_color_rgb,
    serving_tx_labels,
    serving_tx_valid_mask,
    supports_coverage_threshold,
)
from ..extensions import sync_runtime_extensions
from ..io.packed_frame_payload import (
    PACKED_PROJECTION_KEY,
    packed_payload_satisfies,
    standard_frame_to_visual_frame,
    try_load_packed_visual_frame,
    try_upgrade_packed_visual_frame,
    visual_frame_read_request_for_visualizer,
    visualizer_frame_provider,
)
from ..services.cache_service import CacheInvalidationScope, invalidate_visualizer_cache
from ..services.coverage_service import CoverageService
from ..services.metrics_service import MetricsService
from ..state import MpcVisibility
from .core import ViewModel

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..state import AppState

logger = get_logger("orchav")


def build_vm_cache_key(
    step: int,
    state: AppState,
    mats_key: Optional[tuple],
) -> tuple:
    """Build the ViewModel cache key for a given step and application state.

    This is the single source of truth for cache-key construction. Both
    :meth:`FramePipeline._derive_view_model` and the
    :class:`~visualizer.src.services.viewmodel_warmer.ViewModelWarmer` must
    use this function to ensure consistent cache look-ups. Include only inputs
    that affect the derived renderer payload; raw frame content is represented
    by the step and invalidated through ``CacheService``.
    """
    return (
        step,
        state.selected_tx,
        state.selected_rx,
        state.mpc_visibility,
        tuple(sorted(state.mpc_allowed_orders)),
        tuple(sorted(state.mpc_allowed_types)),
        mats_key,
        state.color_mode,
        state.topk_render_enabled,
        state.topk_render_max_paths,
        state.show_beamforming,
        state.beamforming_azimuth_samples,
        state.beamforming_elevation_samples,
        state.beamforming_tx_scale,
        state.beamforming_rx_scale,
        state.beamforming_tx_node,
        state.beamforming_rx_node,
        # Standalone beamforming parameters
        state.standalone_beamforming_mode,
        state.standalone_antenna_rows,
        state.standalone_antenna_cols,
        state.standalone_horizontal_spacing_m,
        state.standalone_vertical_spacing_m,
        state.standalone_carrier_frequency_ghz,
        state.standalone_steering_strategy,
        state.standalone_azimuth_deg,
        state.standalone_elevation_deg,
        # Beam pattern display options
        state.beamforming_db_scale,
        state.beamforming_dynamic_range_db,
        state.beamforming_colormap,
        state.beamforming_element_pattern,
        state.beamforming_tx_element_pattern,
        state.beamforming_rx_element_pattern,
        # Range filters
        state.delay_filter_min_ns,
        state.delay_filter_max_ns,
        state.power_filter_min_db,
        state.power_filter_max_db,
        # Angle filters
        state.aoa_az_filter_min_deg,
        state.aoa_az_filter_max_deg,
        state.aoa_el_filter_min_deg,
        state.aoa_el_filter_max_deg,
        state.aod_az_filter_min_deg,
        state.aod_az_filter_max_deg,
        state.aod_el_filter_min_deg,
        state.aod_el_filter_max_deg,
        # Distinct material colors
        state.use_distinct_material_colors,
    )


class FramePipeline:
    """Coordinate raw frame loading, ViewModel derivation, and renderer apply."""

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        *,
        coverage_service: Optional[CoverageService] = None,
        metrics_service: Optional[MetricsService] = None,
        benchmark_recorder: Optional[BenchmarkRecorder] = None,
    ) -> None:
        """Store app services used during raw-frame to renderer-frame updates."""
        self.visualizer = visualizer
        self.coverage_service = (
            coverage_service
            if coverage_service is not None
            else getattr(visualizer, "coverage_service", CoverageService())
        )
        self.metrics_service = metrics_service or getattr(visualizer, "metrics_service", None)
        self.benchmark_recorder = benchmark_recorder
        self._last_viewmodel_cache_hit = False
        # Optional pop-out consumers subscribe only while visible. The accepted
        # frame hot path below performs one nullable callback check when closed;
        # it never constructs Explorer state or inspects canonical MPC arrays.
        self._mpc_explorer_presented_callback: Optional[Callable[..., None]] = None

    def set_mpc_explorer_presented_callback(
        self,
        callback: Optional[Callable[..., None]],
    ) -> None:
        """Install the sole visible MPC Explorer presented-frame consumer."""
        current = self._mpc_explorer_presented_callback
        if callback is not None and current is not None and current is not callback:
            raise RuntimeError("An MPC Explorer presented-frame consumer is already active")
        self._mpc_explorer_presented_callback = callback

    def clear_mpc_explorer_presented_callback(
        self,
        callback: Optional[Callable[..., None]] = None,
    ) -> None:
        """Remove the consumer without allowing a stale session to clear a newer one."""
        current = self._mpc_explorer_presented_callback
        if callback is None or current is callback:
            self._mpc_explorer_presented_callback = None

    def _publish_mpc_explorer_presented_frame(
        self,
        *,
        callback: Callable[..., None],
        step: int,
        view_model: ViewModel,
        render_packet: Any,
    ) -> None:
        """Notify the visible Explorer after renderer acceptance."""
        source_epoch = int(getattr(self.visualizer, "_mpc_presented_source_epoch", 0))
        try:
            callback(source_epoch, int(step), view_model, render_packet)
        except Exception:
            # An optional inspection window must not turn a successfully
            # presented renderer frame into a failed core pipeline update.
            logger.exception("MPC Explorer rejected a presented-frame notification")

    def _sync_mpc_type_legend(self, render_packet: Any) -> None:
        """Mirror rendered interaction types to the shared Paths panel."""
        ui_manager = getattr(self.visualizer, "ui_manager", None)
        panels = getattr(ui_manager, "panels", None)
        if not isinstance(panels, dict):
            return
        panel = panels.get("mpc")
        setter = getattr(panel, "set_present_mpc_type_codes", None)
        if callable(setter):
            try:
                setter(tuple(getattr(render_packet, "mpc_line_itype_codes", ()) or ()))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # A panel may be closing while the renderer accepts its frame.
                # Legend synchronization is informative and must not invalidate
                # a frame that is already visible.
                logger.exception("Paths legend rejected a presented-frame update")

    @staticmethod
    def _array_signature(values: Any) -> tuple[tuple[int, ...], bytes] | None:
        """Return a compact value signature for small TX/RX state arrays."""
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if arr.size == 0:
            return (tuple(arr.shape), b"")
        arr = np.ascontiguousarray(arr)
        return (tuple(arr.shape), arr.tobytes())

    @classmethod
    def _node_orientation_signature(cls, tx_orientations: Any, rx_orientations: Any):
        """Return a stable signature for the current node orientation arrays."""
        return (
            cls._array_signature(tx_orientations),
            cls._array_signature(rx_orientations),
        )

    @staticmethod
    def _mpc_materials_cache_key(viz: Any) -> Optional[tuple]:
        """Return the material-filter portion of the ViewModel cache key."""
        material_scope = str(getattr(viz, "mpc_material_filter_scope", "segment"))
        if not hasattr(viz, "mpc_allowed_materials"):
            return None
        if viz.mpc_allowed_materials is None:
            return None
        if len(viz.mpc_allowed_materials) == 0:
            return (material_scope, "__EMPTY__")
        return (material_scope, *tuple(sorted(viz.mpc_allowed_materials)))

    def clear_coverage_cache(self) -> None:
        """Clear the coverage mesh cache."""
        self.coverage_service.clear()

    def get_coverage_cache_stats(self) -> dict[str, Any]:
        """Get coverage cache statistics."""
        return self.coverage_service.stats()

    @staticmethod
    def _empty_scene_view_model() -> ViewModel:
        """Return an empty frame payload that can carry a scene-only coverage layer."""
        empty_points = np.empty((0, 3), dtype=np.float32)
        return ViewModel(
            tx_positions=empty_points,
            rx_positions=empty_points,
            tx_orientations=empty_points,
            rx_orientations=empty_points,
            mpc_points=empty_points,
            mpc_lines=np.empty((0, 2), dtype=np.int32),
            mpc_colors=empty_points,
            colorbar=None,
            stats_text="",
            mpc_visibility=MpcVisibility(enabled=False, paths=False, bounce_points=False),
            target_positions=empty_points,
            target_orientations=empty_points,
            target_mesh_files=[],
            target_use_ply_positions=[],
            target_metadata=[],
        )

    def update_coverage_overlay(self) -> bool:
        """Apply coverage without requiring MPC frame data.

        Static-scene scenarios can own a valid coverage HDF5 file even when no
        frame source exists. This narrow transaction updates only renderer-owned
        frame overlays and leaves persistent scene geometry untouched.
        """
        viz = self.visualizer
        if not hasattr(viz, "renderer") or not hasattr(viz, "app_state"):
            return False

        view_model = self._empty_scene_view_model()
        if bool(viz.app_state.show_coverage) and getattr(viz, "coverage_data", None) is not None:
            view_model = self._add_coverage_to_view_model(view_model)
        viz.current_view_model = view_model

        renderer = viz.renderer
        render_packet = view_model.to_render_packet()
        frame_started = False
        try:
            renderer.begin_frame_update()
            frame_started = True
            apply_result = renderer.apply_frame(render_packet)
            frame_started = False
            submission_result = renderer.end_frame_update()
        finally:
            if frame_started:
                renderer.end_frame_update()

        if not isinstance(apply_result, bool):
            logger.error(
                "Renderer apply_frame() returned %s instead of bool",
                type(apply_result).__name__,
            )
            return False
        if not isinstance(submission_result, bool):
            logger.error(
                "Renderer end_frame_update() returned %s instead of bool",
                type(submission_result).__name__,
            )
            return False
        accepted = apply_result and submission_result
        if accepted:
            self._sync_mpc_type_legend(render_packet)
            explorer_callback = self._mpc_explorer_presented_callback
            if explorer_callback is not None:
                self._publish_mpc_explorer_presented_frame(
                    callback=explorer_callback,
                    step=int(getattr(viz.app_state, "step", 0)),
                    view_model=view_model,
                    render_packet=render_packet,
                )
        return accepted

    def precache_coverage_heights(self) -> tuple[int, int]:
        """Build missing base meshes for every height without changing the displayed slice.

        Returns:
            ``(generated, reused)`` mesh counts. Contours and threshold masks are
            intentionally excluded because they are independent, on-demand
            visual layers.
        """
        viz = self.visualizer
        coverage_data = getattr(viz, "coverage_data", None)
        if not isinstance(coverage_data, dict):
            return 0, 0

        file_backed = bool(coverage_data.get("coverage_file"))
        values_3d = coverage_data.get("values_3d")
        if values_3d is None:
            return 0, 0
        values_3d = np.asarray(values_3d, dtype=np.float32)
        if values_3d.ndim != 3:
            return 0, 0

        grid_origin = np.asarray(coverage_data.get("grid_origin", []), dtype=np.float32)
        grid_spacing = np.asarray(coverage_data.get("grid_spacing", []), dtype=np.float32)
        grid_shape = np.asarray(coverage_data.get("grid_shape", []), dtype=np.int32)
        if grid_shape.size < 2:
            return 0, 0
        metric_name = str(coverage_data.get("metric_name", "coverage"))
        heights = list(getattr(viz, "coverage_heights", None) or coverage_data.get("heights") or [])
        required_height_count = len(heights) if file_backed else values_3d.shape[0]
        if len(heights) < required_height_count:
            base_z = float(grid_origin[2]) if grid_origin.size >= 3 else 0.0
            heights.extend(base_z for _ in range(required_height_count - len(heights)))

        interpolation_method = getattr(viz, "coverage_interpolation_method", "none")
        if is_serving_tx_metric(metric_name):
            interpolation_method = "none"

        generated = 0
        reused = 0
        for height_index in range(required_height_count):
            cache_key = self.coverage_service.compute_cache_key(
                coverage_data,
                height_index,
                interpolation_method,
            )
            if self.coverage_service.get_mesh(cache_key, copy=False) is not None:
                reused += 1
                continue
            if file_backed:
                mesh_values = self.coverage_service.metric_layer_at_height(
                    coverage_data,
                    metric_name,
                    height_index,
                )
                mesh_heights = [heights[height_index]]
                slice_plan = [(0, True)]
            else:
                mesh_values = values_3d
                mesh_heights = heights
                slice_plan = [(height_index, True)]
            vertices, triangles, colors = self._build_coverage_mesh(
                grid_origin,
                grid_spacing,
                grid_shape,
                mesh_values,
                float(coverage_data.get("value_min", 0.0)),
                float(coverage_data.get("value_max", 1.0)),
                metric_name,
                mesh_heights,
                slice_plan,
            )
            self.coverage_service.put_mesh(cache_key, vertices, triangles, colors)
            generated += 1
        return generated, reused

    def _sync_node_device_names_from_frame(self, raw_frame: dict[str, Any]) -> None:
        """Copy optional frame TX/RX names into state for name-mode labels."""

        def _names(key: str) -> tuple[str, ...]:
            """Return decoded non-crashing device names for one frame key."""
            values = raw_frame.get(key)
            if values is None:
                return ()
            try:
                names = tuple(
                    (
                        (
                            value.decode("utf-8", errors="replace")
                            if isinstance(value, (bytes, np.bytes_))
                            else str(value)
                        )
                        if value
                        else ""
                    )
                    for value in values
                )
            except TypeError:
                return ()
            return names if any(names) else ()

        tx_names = _names("tx_names")
        rx_names = _names("rx_names")
        if not tx_names and not rx_names:
            return

        viz = self.visualizer
        state = getattr(viz, "app_state", None)
        updates: dict[str, tuple[str, ...]] = {}
        if tx_names and tx_names != tuple(getattr(state, "tx_device_names", ()) or ()):
            updates["tx_device_names"] = tx_names
        if rx_names and rx_names != tuple(getattr(state, "rx_device_names", ()) or ()):
            updates["rx_device_names"] = rx_names
        if not updates:
            return
        if hasattr(viz, "set_state"):
            viz.set_state(**updates)
        elif state is not None:
            for key, value in updates.items():
                setattr(state, key, value)
        invalidate_visualizer_cache(
            viz,
            CacheInvalidationScope.LABELS,
            reason="frame_node_names",
        )

    @staticmethod
    def _safe_len(values: Any) -> int:
        """Return ``len(values)`` when available, otherwise ``0``."""
        if values is None:
            return 0
        try:
            return len(values)
        except TypeError:
            return 0

    def _target_focus_dropdown_signature(
        self,
        viz: Any,
        view_model: ViewModel,
    ) -> tuple[Any, ...]:
        """Build a stable signature for camera target-focus dropdown contents."""
        target_metadata = getattr(view_model, "target_metadata", None) or ()
        target_names: list[str] = []
        for idx, item in enumerate(target_metadata):
            if isinstance(item, dict):
                name = item.get("name") or item.get("target_name")
            else:
                name = getattr(item, "name", None) or getattr(item, "target_name", None)
            target_names.append(str(name or f"target_{idx}"))

        app_state = getattr(viz, "app_state", None)
        tx_labels = tuple(getattr(app_state, "tx_labels", ()) or ())
        rx_labels = tuple(getattr(app_state, "rx_labels", ()) or ())
        tx_device_names = tuple(getattr(app_state, "tx_device_names", ()) or ())
        rx_device_names = tuple(getattr(app_state, "rx_device_names", ()) or ())
        return (
            tuple(target_names),
            self._safe_len(getattr(view_model, "tx_positions", None)),
            self._safe_len(getattr(view_model, "rx_positions", None)),
            getattr(app_state, "node_label_mode", "role"),
            tx_labels,
            rx_labels,
            tx_device_names,
            rx_device_names,
        )

    def _maybe_refresh_target_focus_dropdown(
        self,
        viz: Any,
        view_model: ViewModel,
        *,
        force: bool = False,
    ) -> None:
        """Refresh the camera target-focus dropdown only when its contents changed."""
        controller = getattr(viz, "camera_controller", None)
        refresh = getattr(controller, "update_target_focus_dropdown", None)
        if not callable(refresh):
            return

        signature = self._target_focus_dropdown_signature(viz, view_model)
        if not force and getattr(viz, "_last_target_focus_dropdown_signature", None) == signature:
            return

        refresh()
        viz._last_target_focus_dropdown_signature = signature

    @staticmethod
    def _orientation_overlays_enabled(viz: Any) -> bool:
        """Return whether any node/target orientation overlay is currently enabled."""
        return any(
            bool(getattr(viz, attr, False))
            for attr in ("show_tx_orientation", "show_rx_orientation", "show_target_orientation")
        )

    def _enforce_view_model_cache_limit(self) -> None:
        """Evict oldest ViewModel cache entries when the LRU exceeds its limit."""
        viz = self.visualizer
        if not hasattr(viz, "mpc_view_cache") or viz.mpc_view_cache is None:
            return

        cache_size = len(viz.mpc_view_cache)
        if cache_size <= MAX_VIEW_MODEL_CACHE_SIZE:
            return

        num_to_evict = cache_size - MAX_VIEW_MODEL_CACHE_SIZE
        evicted_keys = []

        for key in list(viz.mpc_view_cache.keys())[:num_to_evict]:
            del viz.mpc_view_cache[key]
            evicted_keys.append(key)

        logger.debug(
            "Evicted %d old ViewModel entries (was %d/%d, now %d/%d)",
            num_to_evict,
            cache_size,
            MAX_VIEW_MODEL_CACHE_SIZE,
            len(viz.mpc_view_cache),
            MAX_VIEW_MODEL_CACHE_SIZE,
        )

    @staticmethod
    def _resolve_unique_node_counts(
        raw_frame: dict[str, Any], view_model: ViewModel
    ) -> tuple[int, int]:
        """Resolve unique TX/RX node counts from frame metadata or ViewModel arrays."""

        def _safe_count(value: Any) -> int:
            """Coerce count-like values to non-negative integers."""
            try:
                count = int(value)
            except (TypeError, ValueError):
                return 0
            return count if count >= 0 else 0

        tx_count = _safe_count(raw_frame.get("num_tx"))
        rx_count = _safe_count(raw_frame.get("num_rx"))

        if tx_count == 0:
            tx_count = _safe_count(len(getattr(view_model, "tx_positions", ())))
        if rx_count == 0:
            rx_count = _safe_count(len(getattr(view_model, "rx_positions", ())))

        return tx_count, rx_count

    def update(self, step: int) -> bool:
        """Process one animation step and report accepted submission or no-op."""

        viz = self.visualizer
        frame_start = time.perf_counter()
        bench = self.benchmark_recorder

        def _fail_pending_beamforming(reason: str) -> None:
            """Terminate a beam status that cannot reach result synchronization."""
            controller = getattr(viz, "beamforming_ui_controller", None)
            fail_computation = getattr(controller, "fail_computation", None)
            if callable(fail_computation):
                fail_computation(reason)

        if bench is not None:
            bench.begin_frame(step)

        if not hasattr(viz, "app_state"):
            logger.warning("App state not initialized, skipping update")
            self._record_frame_timing(step, frame_start, completed=False)
            return False

        if not hasattr(viz, "renderer"):
            logger.warning("Renderer not initialized, skipping update")
            _fail_pending_beamforming("Renderer is not initialized")
            self._record_frame_timing(step, frame_start, completed=False)
            return False

        if not viz.ready and bool(getattr(viz, "_scene_only_mode", False)):
            completed = self.update_coverage_overlay()
            self._record_frame_timing(step, frame_start, completed=completed)
            return completed

        if not viz.ready:
            logger.debug("Pipeline: Frame source not ready, skipping update")
            _fail_pending_beamforming("Frame source is not ready")
            self._record_frame_timing(step, frame_start, completed=False)
            return False

        # ``last_app_state`` suppresses redundant work, while dirty flags and
        # forced updates represent external cache/renderer changes.
        force_update = bool(getattr(viz, "force_update_next_frame", False))

        def _retry_forced_frame(reason: str) -> None:
            """Retry transient forced-frame failures without an unbounded loop."""

            if not force_update:
                return
            schedule_retry = getattr(viz, "schedule_frame_retry", None)
            if callable(schedule_retry):
                schedule_retry(reason)
                return
            viz.force_update_next_frame = True
            viz.schedule_update()

        if (
            not force_update
            and hasattr(viz, "last_app_state")
            and viz.last_app_state == viz.app_state
            and not getattr(viz, "_material_filter_dirty", False)
            and not getattr(viz, "_coverage_interpolation_dirty", False)
        ):
            logger.debug("Pipeline: App state unchanged, skipping update")
            self._record_frame_timing(step, frame_start, completed=False)
            return True
        if force_update:
            logger.debug("Pipeline: Processing forced update")

        logger.debug(
            "Pipeline: App state step=%s, mpc_visibility=%s",
            viz.app_state.step,
            viz.app_state.mpc_visibility,
        )
        logger.debug("Pipeline: Renderer type: %s", type(viz.renderer))

        load_start = time.perf_counter()
        raw_frame = self.load_frame(step)
        load_elapsed = (time.perf_counter() - load_start) * 1000
        logger.debug("Load frame: %.2f ms", load_elapsed)
        if bench is not None:
            bench.record_load(load_elapsed)

        if raw_frame is None:
            try:
                _retry_forced_frame("frame data could not be loaded")
            except (RuntimeError, AttributeError):
                logger.debug(
                    "Pipeline: Unable to re-schedule update after load failure", exc_info=True
                )
            _fail_pending_beamforming("Frame data could not be loaded")
            self._record_frame_timing(step, frame_start, completed=False)
            return False
        if force_update:
            viz.force_update_next_frame = False

        logger.debug("Processing frame %s", step)

        vm_start = time.perf_counter()
        view_model = self._derive_view_model(step, raw_frame)
        vm_elapsed = (time.perf_counter() - vm_start) * 1000
        logger.debug("Derive view model: %.2f ms", vm_elapsed)
        if bench is not None:
            bench.record_viewmodel(vm_elapsed)
            bench.record_breakdown(
                "viewmodel_cache_hit", 1.0 if self._last_viewmodel_cache_hit else 0.0
            )
            breakdown_fn = getattr(viz.mpc_core, "get_last_viewmodel_breakdown", None)
            if callable(breakdown_fn) and not self._last_viewmodel_cache_hit:
                bench.record_breakdowns(breakdown_fn())

        if view_model is None:
            try:
                _retry_forced_frame("beam view model could not be computed")
            except (RuntimeError, AttributeError):
                logger.debug(
                    "Pipeline: Unable to re-schedule update after view-model failure",
                    exc_info=True,
                )
            _fail_pending_beamforming("Beam view model could not be computed")
            self._record_frame_timing(step, frame_start, completed=False)
            return False

        # Application services keep the full frame model; renderers receive a
        # narrow shallow projection below.
        viz.current_view_model = view_model
        logger.debug("Pipeline: Set current_view_model early for camera focus")
        render_packet = view_model.to_render_packet()

        # Camera, persistent objects, and frame-heavy payloads are one renderer
        # transaction for every supported backend.
        frame_update_started = False
        frame_submission_succeeded = True
        persistent_entities_succeeded = True
        runtime_extensions_succeeded = True

        def _record_entity_sync_result(result: Any, domain: str) -> None:
            """Aggregate one persistent-entity synchronization result."""
            nonlocal persistent_entities_succeeded
            if not isinstance(result, bool):
                logger.error(
                    "%s synchronization returned %s instead of bool",
                    domain,
                    type(result).__name__,
                )
                persistent_entities_succeeded = False
                return
            persistent_entities_succeeded = result and persistent_entities_succeeded

        def _end_renderer_frame_update() -> bool:
            """Finish one backend transaction without hiding submission failure."""
            result = viz.renderer.end_frame_update()
            if not isinstance(result, bool):
                logger.error(
                    "Renderer end_frame_update() returned %s instead of bool",
                    type(result).__name__,
                )
                return False
            return result

        try:
            viz.renderer.begin_frame_update()
            frame_update_started = True

            # Update Follow/POV camera BEFORE geometry so the complete frame is
            # submitted together by end_frame_update().
            # Camera and geometry are submitted together when end_frame_update() is called.
            t_camera = time.perf_counter()
            self._update_camera_before_frame(viz)
            t_after_camera = time.perf_counter()

            logger.debug(
                "Pipeline: About to call renderer.apply_frame with %d MPC points",
                len(render_packet.mpc_points),
            )
            apply_start = time.perf_counter()
            apply_succeeded = viz.renderer.apply_frame(render_packet)
            apply_elapsed = (time.perf_counter() - apply_start) * 1000
            if not isinstance(apply_succeeded, bool):
                logger.error(
                    "Renderer apply_frame() returned %s instead of bool",
                    type(apply_succeeded).__name__,
                )
                apply_succeeded = False
            logger.debug("Renderer apply_frame: %.2f ms", apply_elapsed)
            if apply_succeeded:
                logger.debug("Pipeline: Renderer accepted frame payload")
                # Remove the temporary empty-scene camera anchor once real content is visible.
                if getattr(viz, "_empty_scene_anchor", None) is not None:
                    viz.scene_service.remove_empty_scene_anchor()
                if bench is not None:
                    bench.record_render(apply_elapsed)
                    bench.record_geometry(
                        len(render_packet.mpc_points), len(render_packet.mpc_lines)
                    )
                    bench.record_breakdown("renderer_apply_ms", apply_elapsed)
            if not apply_succeeded:
                logger.warning("Pipeline: Renderer rejected frame payload for step %s", step)
                _fail_pending_beamforming("Renderer rejected the frame update")
                try:
                    _retry_forced_frame("renderer rejected the frame update")
                except (RuntimeError, AttributeError):
                    logger.debug(
                        "Pipeline: Unable to re-schedule rejected renderer frame",
                        exc_info=True,
                    )
                if frame_update_started:
                    frame_update_started = False
                    _end_renderer_frame_update()
                self._record_frame_timing(step, frame_start, completed=False)
                return False
            t_after_apply = time.perf_counter()

            if self.metrics_service is not None:
                self.metrics_service.update_metrics(view_model)

            beamforming_ui = viz.beamforming_ui_controller

            # Store canonical TX/RX positions and orientations. Empty arrays are
            # meaningful: they hide inventory-owned nodes until anchors return.
            # Position updates use object identity to skip unchanged frame arrays;
            # orientations need value comparison because static nodes can rotate
            # in place via look-at rules.
            _positions_changed = False
            if hasattr(view_model, "tx_positions") and hasattr(view_model, "rx_positions"):
                tx_pos = view_model.tx_positions
                rx_pos = view_model.rx_positions
                previous_tx = getattr(viz, "current_tx_positions", None)
                previous_rx = getattr(viz, "current_rx_positions", None)
                if previous_tx is not tx_pos:
                    viz.current_tx_positions = tx_pos
                    _positions_changed = True
                if previous_rx is not rx_pos:
                    viz.current_rx_positions = rx_pos
                    _positions_changed = True
            _orientations_changed = False
            if hasattr(view_model, "tx_orientations") and hasattr(view_model, "rx_orientations"):
                tx_orient = view_model.tx_orientations
                rx_orient = view_model.rx_orientations
                viz.current_tx_orientations = tx_orient
                viz.current_rx_orientations = rx_orient
                orientation_signature = self._node_orientation_signature(tx_orient, rx_orient)
                if orientation_signature != getattr(viz, "_last_node_orientation_signature", None):
                    viz._last_node_orientation_signature = orientation_signature
                    _orientations_changed = True
            t_after_positions = time.perf_counter()

            node_inventory_changed = False
            node_service = viz.node_service
            num_tx = raw_frame.get("num_tx")
            num_rx = raw_frame.get("num_rx")
            if (num_tx is None or num_tx == 0) and hasattr(view_model, "tx_positions"):
                try:
                    num_tx = int(len(view_model.tx_positions))
                except (TypeError, ValueError, AttributeError):
                    num_tx = None
            if (num_rx is None or num_rx == 0) and hasattr(view_model, "rx_positions"):
                try:
                    num_rx = int(len(view_model.rx_positions))
                except (TypeError, ValueError, AttributeError):
                    num_rx = None
            node_inventory_changed = bool(
                node_service.update_available_tx_rx_from_frame(num_tx, num_rx)
            )

            beamforming_ui.apply_selector_state()

            beamforming_ui.update_standalone_buttons_state()

            # Skip marker sync when neither position, orientation, nor inventory changed.
            if _positions_changed or _orientations_changed or node_inventory_changed:
                if hasattr(viz, "tx_markers") and hasattr(viz, "rx_markers"):
                    if len(viz.tx_markers) > 0 or len(viz.rx_markers) > 0:
                        logger.debug(
                            "Pipeline: Updating marker transforms from ViewModel node state"
                        )
                        _record_entity_sync_result(
                            node_service.update_tx_rx_positions(
                                view_model.tx_positions,
                                view_model.rx_positions,
                            ),
                            "TX/RX entity",
                        )
            elif getattr(viz, "tx_markers", None) or getattr(viz, "rx_markers", None):
                _record_entity_sync_result(
                    node_service.retry_pending_node_syncs(),
                    "TX/RX pending entity",
                )

            # The focus dropdown signature also depends on target names and node labels,
            # so refresh gating must not be coupled only to TX/RX motion.
            self._maybe_refresh_target_focus_dropdown(
                viz,
                view_model,
                force=node_inventory_changed,
            )
            t_after_markers = time.perf_counter()

            if hasattr(viz, "target_entries") and viz.target_entries:
                _record_entity_sync_result(
                    viz.target_service.process_targets_from_view_model(step, view_model),
                    "Target entity",
                )
            t_after_targets = time.perf_counter()

            if self._orientation_overlays_enabled(viz):
                orientation_synced = node_service.create_orientation_frames(step)
                _record_entity_sync_result(
                    orientation_synced,
                    "Orientation-frame entity",
                )
                if orientation_synced is True:
                    logger.debug("Pipeline: Successfully updated orientation frames")
            else:
                logger.debug("Pipeline: Skipping orientation work; overlays disabled")
            t_after_orientations = time.perf_counter()

            aperture_service = getattr(viz, "aperture_service", None)
            state = viz.app_state
            if aperture_service is not None and (
                state.show_aoa_aperture
                or state.show_aod_aperture
                or getattr(state, "show_global_angular_reference", False)
                or getattr(state, "show_local_angular_reference", False)
            ):
                _record_entity_sync_result(
                    aperture_service.update_apertures(),
                    "Aperture entity",
                )

            runtime_extensions_succeeded = self._sync_optional_runtime_extensions(
                viz,
                raw_frame,
                step,
            )

            t_before_end = time.perf_counter()

            logger.debug(
                "[pygfx-telemetry] pipeline: camera=%.1fms apply=%.1fms "
                "positions=%.1fms markers=%.1fms targets=%.1fms "
                "orientations=%.1fms total_before_end=%.1fms",
                (t_after_camera - t_camera) * 1000,
                (t_after_apply - t_after_camera) * 1000,
                (t_after_positions - t_after_apply) * 1000,
                (t_after_markers - t_after_positions) * 1000,
                (t_after_targets - t_after_markers) * 1000,
                (t_after_orientations - t_after_targets) * 1000,
                (t_before_end - t_camera) * 1000,
            )

            if bench is not None:
                bench.record_breakdown("camera_update_ms", (t_after_camera - t_camera) * 1000.0)
                bench.record_breakdown(
                    "tx_rx_positions_ms", (t_after_positions - t_after_apply) * 1000.0
                )
                bench.record_breakdown(
                    "tx_rx_markers_ms", (t_after_markers - t_after_positions) * 1000.0
                )
                bench.record_breakdown(
                    "target_update_ms", (t_after_targets - t_after_markers) * 1000.0
                )
                bench.record_breakdown(
                    "orientation_update_ms",
                    (t_after_orientations - t_after_targets) * 1000.0,
                )
                orientation_breakdown = getattr(viz, "_orientation_frame_breakdown", None)
                if isinstance(orientation_breakdown, dict):
                    bench.record_breakdowns(orientation_breakdown)
                target_service = getattr(viz, "target_service", None)
                target_breakdown_fn = getattr(target_service, "get_last_runtime_breakdown", None)
                if callable(target_breakdown_fn):
                    bench.record_breakdowns(target_breakdown_fn())
                total_before_end_ms = (t_before_end - frame_start) * 1000.0
                bench.record_total_before_end(total_before_end_ms)
                bench.record_breakdown("total_before_end_ms", total_before_end_ms)

            if frame_update_started:
                frame_update_started = False
                frame_submission_succeeded = _end_renderer_frame_update()

            t_after_end = time.perf_counter()
            if bench is not None:
                bench.record_breakdown("end_frame_update_ms", (t_after_end - t_before_end) * 1000.0)
                end_breakdown_fn = getattr(
                    viz.renderer, "get_last_end_frame_update_breakdown", None
                )
                if callable(end_breakdown_fn):
                    bench.record_breakdowns(end_breakdown_fn())
                end_breakdown_bytes_fn = getattr(
                    viz.renderer, "get_last_end_frame_update_breakdown_bytes", None
                )
                if callable(end_breakdown_bytes_fn):
                    bench.record_breakdown_bytes_many(end_breakdown_bytes_fn())

            if not frame_submission_succeeded:
                logger.warning("Pipeline: Frame %s was not accepted by the renderer", step)
                _fail_pending_beamforming("Renderer did not present the frame update")
                self._record_frame_timing(step, frame_start, completed=False)
                return False

            self._sync_mpc_type_legend(render_packet)

            explorer_callback = self._mpc_explorer_presented_callback
            if explorer_callback is not None:
                self._publish_mpc_explorer_presented_frame(
                    callback=explorer_callback,
                    step=step,
                    view_model=view_model,
                    render_packet=render_packet,
                )

            # Only publish beam metadata after the renderer has accepted the
            # frame that owns the corresponding surfaces.
            beamforming_ui.update_node_options(
                getattr(view_model, "beamforming_info", None),
                getattr(view_model, "beamforming_pairs", None),
            )

            if not persistent_entities_succeeded:
                logger.warning("Pipeline: Frame %s has incomplete persistent entities", step)
                self._record_frame_timing(step, frame_start, completed=False)
                return False

            if not runtime_extensions_succeeded:
                logger.warning("Pipeline: Frame %s has incomplete runtime extensions", step)
                self._record_frame_timing(step, frame_start, completed=False)
                return False

            if (
                viz._scene_boot_start is not None
                and not viz._scene_boot_logged
                and hasattr(viz, "set_startup_first_frame_timing")
            ):
                if hasattr(viz, "set_startup_detail_timing"):
                    get_end_breakdown = getattr(
                        viz.renderer, "get_last_end_frame_update_breakdown", None
                    )
                    if callable(get_end_breakdown):
                        end_breakdown = get_end_breakdown()
                        if end_breakdown:
                            viz.set_startup_detail_timing(
                                "first_frame_end_update_breakdown_ms",
                                end_breakdown,
                            )
                viz.set_startup_first_frame_timing(
                    {
                        "load_ms": load_elapsed,
                        "viewmodel_ms": vm_elapsed,
                        "camera_ms": (t_after_camera - t_camera) * 1000.0,
                        "apply_ms": apply_elapsed,
                        "positions_ms": (t_after_positions - t_after_apply) * 1000.0,
                        "tx_rx_update_ms": (t_after_markers - t_after_positions) * 1000.0,
                        "target_update_ms": (t_after_targets - t_after_markers) * 1000.0,
                        "orientation_update_ms": (t_after_orientations - t_after_targets) * 1000.0,
                        "post_orientation_ms": (t_before_end - t_after_orientations) * 1000.0,
                        "end_frame_update_ms": (t_after_end - t_before_end) * 1000.0,
                        "total_before_record_ms": (t_after_end - frame_start) * 1000.0,
                    }
                )

            viz.ui_controller.update_frame_context(
                step,
                raw_frame=raw_frame,
                view_model=view_model,
            )

            viz.last_app_state = viz.app_state
            reset_retry = getattr(viz, "reset_frame_retry_state", None)
            if callable(reset_retry):
                reset_retry()
            self._record_frame_timing(step, frame_start, completed=True)
            return True
        except Exception:
            # Preserve the original exception while ensuring a synchronous
            # pipeline failure cannot strand the Antennas panel on Computing.
            _fail_pending_beamforming("Frame update failed before presentation")
            raise
        finally:
            if frame_update_started:
                frame_update_started = False
                _end_renderer_frame_update()

    @staticmethod
    def _sync_optional_runtime_extensions(
        viz: Any,
        raw_frame: dict[str, Any],
        step: int,
    ) -> bool:
        """Synchronize optional extensions without masking core defects.

        Extension data may be absent or malformed independently of the
        maintained propagation frame. Expected extension data/runtime errors
        make the frame incomplete so an identical request can retry. Programmer
        errors such as assertion failures are intentionally allowed to
        propagate.
        """
        try:
            return bool(sync_runtime_extensions(viz, raw_frame, step))
        except (RuntimeError, ValueError, KeyError) as exc:
            logger.warning("Pipeline: Error updating optional runtime extension: %s", exc)
            return False

    def load_frame(self, step: int) -> Optional[dict[str, Any]]:
        """Load the raw frame for *step* from preview, cache, loader, or source.

        Live preview data takes precedence for its matching step. Otherwise the
        raw frame cache is checked before falling back to ``FrameLoaderService``
        or the active frame source. Loaded provider frames are stored in the raw
        frame cache; derived canonical and ViewModel data are cached elsewhere.
        """

        viz = self.visualizer

        preview_frame = getattr(viz, "_live_preview_frame", None)
        preview_step = getattr(viz, "_live_preview_step", None)
        if preview_frame is not None and preview_step == step:
            logger.debug("Pipeline: Using live preview frame for step %s", step)
            return preview_frame

        cache_service = viz.cache_service
        cached_frame = cache_service.get_frame(step)
        if cached_frame is not None:
            request = visual_frame_read_request_for_visualizer(viz)
            if PACKED_PROJECTION_KEY in cached_frame and not packed_payload_satisfies(
                cached_frame, request
            ):
                provider = self._compact_frame_provider(viz)
                if provider is not None:
                    try:
                        upgraded = try_upgrade_packed_visual_frame(
                            provider,
                            step,
                            cached_frame,
                            request=request,
                            points_dtype=self._canonical_points_dtype(viz),
                        )
                    except (OSError, KeyError, ValueError) as exc:
                        logger.warning(
                            "Unable to upgrade compact frame %s for active overlays: %s",
                            step,
                            exc,
                        )
                    else:
                        if upgraded is not None:
                            cache_service.store_frame(step, upgraded)
                            cached_frame = upgraded
            logger.debug("Pipeline: Using frame cache for step %s", step)
            return cached_frame

        if not viz.ready:
            logger.debug("No frame source ready yet; skipping frame load")
            return None

        raw_frame: Optional[dict[str, Any]] = None
        loader = getattr(viz, "frame_loader", None)
        try:
            provider = self._compact_frame_provider(viz)
            request = visual_frame_read_request_for_visualizer(viz)
            points_dtype = self._canonical_points_dtype(viz)
            if provider is not None:
                raw_frame = try_load_packed_visual_frame(
                    provider,
                    step,
                    request=request,
                    points_dtype=points_dtype,
                )
            if raw_frame is not None:
                logger.debug("Loaded projected visual frame %s", step)
            elif loader is not None:
                raw_frame = standard_frame_to_visual_frame(
                    loader.get_frame(step, bypass_cache=True),
                    request=request,
                    points_dtype=points_dtype,
                )
                logger.debug("Loaded frame %s via FrameLoaderService", step)
            else:
                frame_source = getattr(viz, "frame_source", None)
                if frame_source is None:
                    logger.warning("No frame source available")
                    return None
                raw_frame = standard_frame_to_visual_frame(
                    frame_source.load_frame(step),
                    request=request,
                    points_dtype=points_dtype,
                )
                frame_source_type = type(frame_source).__name__
                logger.debug("Loaded frame %s from %s", step, frame_source_type)
        except (OSError, KeyError, ValueError) as exc:
            logger.error("Error loading frame %s: %s", step, exc)
            if hasattr(viz, "_set_status_message"):
                viz._set_status_message(f"Failed to load frame {step}: {exc}", 5000)
            return None

        if raw_frame is not None:
            if not cache_service.has_frame(step):
                cache_service.store_frame(step, raw_frame)
            logger.debug("Stored raw frame data for step %s", step)

        return raw_frame

    @staticmethod
    def _compact_frame_provider(viz: Any) -> Any | None:
        """Return the active shared provider when the source exposes one."""
        return visualizer_frame_provider(viz)

    @staticmethod
    def _canonical_points_dtype(viz: Any) -> np.dtype:
        """Return the renderer-selected canonical point dtype."""
        core = getattr(viz, "mpc_core", None)
        return np.dtype(getattr(core, "canon_points_dtype", np.float32))

    def _update_camera_before_frame(self, viz: Any) -> None:
        """Update the camera before geometry processing.

        Geometry updates can trigger redraws before the end-frame submission.
        Updating the camera first keeps the view correct when that render occurs.
        """
        camera_mode = getattr(getattr(viz, "app_state", None), "camera_mode", None)

        if camera_mode == "follow" and hasattr(viz, "camera_controller"):
            viz.camera_controller.update_follow_camera_focus()
            logger.debug("Pipeline: Updated camera for Follow mode")
        elif camera_mode == "pov" and hasattr(viz, "camera_controller"):
            viz.camera_controller.set_pov_camera(defer_redraw=True)
            logger.debug("Pipeline: Updated camera for POV mode")

    def _record_frame_timing(self, step: int, frame_start: float, completed: bool) -> None:
        """Record timing metrics for frame processing."""

        viz = self.visualizer
        elapsed_sec = time.perf_counter() - frame_start
        elapsed_ms = elapsed_sec * 1000.0

        if completed:
            if self.benchmark_recorder is not None:
                self.benchmark_recorder.end_frame(elapsed_ms)
            viz.last_frame_duration_ms = elapsed_ms
            viz.frame_times.append(elapsed_sec)
            if len(viz.frame_times) > 240:
                viz.frame_times = viz.frame_times[-240:]

            logger.debug("Frame %s computed in %.2f ms", step, elapsed_ms)

            if viz._scene_boot_start is not None and not viz._scene_boot_logged:
                viz.scene_boot_duration_ms = (time.perf_counter() - viz._scene_boot_start) * 1000.0
                viz._scene_boot_logged = True
                viz._scene_boot_start = None
                scene_msg = f"Scene ready in {viz.scene_boot_duration_ms:.1f} ms"
                logger.info("%s", scene_msg)
                if hasattr(viz, "_set_status_message"):
                    viz._set_status_message(scene_msg, 3000)
                on_scene_ready = getattr(viz, "_on_scene_boot_completed", None)
                if callable(on_scene_ready):
                    on_scene_ready()

            viz.ui_controller.handle_frame_timing_update(step, elapsed_sec)
        else:
            logger.debug(
                "Frame %s aborted after %.2f ms (pipeline exit before completion)",
                step,
                elapsed_ms,
            )

    def _derive_view_model(self, step: int, raw_frame: dict[str, Any]) -> Optional[ViewModel]:
        """Derive or reuse the ViewModel for one raw frame and current state.

        This method owns ViewModel-cache lookup. ``MPCCore`` owns canonical MPC
        construction and the actual payload build. The disabled-MPC path keeps a
        separate cache key because it can still need node, target, bounce, or
        beamforming payloads even when MPC lines are hidden.
        """

        viz = self.visualizer
        self._last_viewmodel_cache_hit = False
        frame_beamforming_data = raw_frame.get("beamforming")
        frame_beamforming_available = bool(
            isinstance(frame_beamforming_data, dict) and frame_beamforming_data.get("pairs")
        )
        viz.beamforming_ui_controller.set_frame_beamforming_available(frame_beamforming_available)
        self._sync_node_device_names_from_frame(raw_frame)
        view_frame = dict(raw_frame)
        source_metadata = raw_frame.get("_source")
        if isinstance(source_metadata, dict):
            view_frame["_source"] = dict(source_metadata)
        current_state = viz.app_state
        beam_az_samples = current_state.beamforming_azimuth_samples
        beam_el_samples = current_state.beamforming_elevation_samples
        beam_tx_scale = current_state.beamforming_tx_scale
        beam_rx_scale = current_state.beamforming_rx_scale
        beam_tx_node = current_state.beamforming_tx_node
        beam_rx_node = current_state.beamforming_rx_node
        beam_db_scale = current_state.beamforming_db_scale
        beam_dynamic_range_db = current_state.beamforming_dynamic_range_db
        beam_colormap = current_state.beamforming_colormap
        beam_element_pattern = current_state.beamforming_element_pattern
        beam_tx_element_pattern = current_state.beamforming_tx_element_pattern
        beam_rx_element_pattern = current_state.beamforming_rx_element_pattern
        mats_key = self._mpc_materials_cache_key(viz)

        # Inject standalone parameters when beam patterns are computed outside frame metadata.
        if current_state.show_beamforming and current_state.standalone_beamforming_mode != "frame":
            view_frame["standalone_beamforming_mode"] = current_state.standalone_beamforming_mode
            view_frame["standalone_beamforming_params"] = {
                "antenna_rows": current_state.standalone_antenna_rows,
                "antenna_cols": current_state.standalone_antenna_cols,
                "horizontal_spacing_m": current_state.standalone_horizontal_spacing_m,
                "vertical_spacing_m": current_state.standalone_vertical_spacing_m,
                "carrier_frequency_ghz": current_state.standalone_carrier_frequency_ghz,
                # TX/RX positions and orientations remain frame-owned.
                "steering_strategy": current_state.standalone_steering_strategy,
                "azimuth_deg": current_state.standalone_azimuth_deg,
                "elevation_deg": current_state.standalone_elevation_deg,
            }
            if not isinstance(view_frame.get("_source"), dict):
                view_frame["_source"] = {}
            view_source = view_frame["_source"]
            # Try to get duration and num_steps from provider or visualizer
            provider = getattr(viz, "provider", None)
            if provider:
                if hasattr(provider, "_get_frame_info"):
                    frame_info = provider._get_frame_info()
                    if frame_info:
                        view_source["duration"] = frame_info.get("duration")
                        view_source["num_steps"] = len(frame_info.get("available_frames", []))
            if "duration" not in view_source:
                view_source["duration"] = getattr(viz, "_frame_duration", None)
            if "num_steps" not in view_source:
                view_source["num_steps"] = getattr(viz, "total_animation_steps", None)
        else:
            view_frame["standalone_beamforming_mode"] = "frame"

        view_cache_key = build_vm_cache_key(step, current_state, mats_key)

        logger.debug(
            "Cache key for frame %s: selected_tx=%s, selected_rx=%s",
            step,
            current_state.selected_tx,
            current_state.selected_rx,
        )

        if view_cache_key not in viz.mpc_view_cache:
            logger.debug(
                "Cache miss - creating new ViewModel for frame %s (color_mode=%s, tx=%s, rx=%s)",
                step,
                current_state.color_mode,
                current_state.selected_tx,
                current_state.selected_rx,
            )
            view_model = viz.mpc_core.create_view_model(
                step=step,
                raw_frame=view_frame,
                color_mode=current_state.color_mode,
                selected_tx=current_state.selected_tx,
                selected_rx=current_state.selected_rx,
                mpc_allowed_orders=current_state.mpc_allowed_orders,
                mpc_allowed_types=current_state.mpc_allowed_types,
                mpc_allowed_materials=getattr(viz, "mpc_allowed_materials", None),
                mpc_visibility=current_state.mpc_visibility,
                topk_render_enabled=current_state.topk_render_enabled,
                topk_render_max_paths=current_state.topk_render_max_paths,
                show_beamforming=current_state.show_beamforming,
                beamforming_azimuth_samples=beam_az_samples,
                beamforming_elevation_samples=beam_el_samples,
                beamforming_tx_scale=beam_tx_scale,
                beamforming_rx_scale=beam_rx_scale,
                beamforming_tx_node=beam_tx_node,
                beamforming_rx_node=beam_rx_node,
                beamforming_db_scale=beam_db_scale,
                beamforming_dynamic_range_db=beam_dynamic_range_db,
                beamforming_colormap=beam_colormap,
                beamforming_element_pattern=beam_element_pattern,
                beamforming_tx_element_pattern=beam_tx_element_pattern,
                beamforming_rx_element_pattern=beam_rx_element_pattern,
                include_targets=True,
                show_tx_segments=getattr(viz, "show_tx_segments", True),
                # Range filters
                delay_filter_min_ns=current_state.delay_filter_min_ns,
                delay_filter_max_ns=current_state.delay_filter_max_ns,
                power_filter_min_db=current_state.power_filter_min_db,
                power_filter_max_db=current_state.power_filter_max_db,
                # Angle filters
                aoa_az_filter_min_deg=current_state.aoa_az_filter_min_deg,
                aoa_az_filter_max_deg=current_state.aoa_az_filter_max_deg,
                aoa_el_filter_min_deg=current_state.aoa_el_filter_min_deg,
                aoa_el_filter_max_deg=current_state.aoa_el_filter_max_deg,
                aod_az_filter_min_deg=current_state.aod_az_filter_min_deg,
                aod_az_filter_max_deg=current_state.aod_az_filter_max_deg,
                aod_el_filter_min_deg=current_state.aod_el_filter_min_deg,
                aod_el_filter_max_deg=current_state.aod_el_filter_max_deg,
                # Distinct material colors
                use_distinct_material_colors=current_state.use_distinct_material_colors,
            )
            if view_model is None:
                logger.error("Failed to create ViewModel for frame %s", step)
                return None
            viz.mpc_view_cache[view_cache_key] = view_model
            self._enforce_view_model_cache_limit()
        else:
            viz.mpc_view_cache.move_to_end(view_cache_key)
            view_model = viz.mpc_view_cache[view_cache_key]
            self._last_viewmodel_cache_hit = True
            logger.debug(
                "Cache hit - using cached ViewModel for frame %s (color_mode=%s, tx=%s, rx=%s)",
                step,
                current_state.color_mode,
                current_state.selected_tx,
                current_state.selected_rx,
            )

        view_model = replace(view_model)

        tx_count, rx_count = self._resolve_unique_node_counts(raw_frame, view_model)
        if tx_count > 0 or rx_count > 0:
            logger.debug(
                "Frame: Creating markers for %s TX and %s RX nodes (unique node count)",
                tx_count,
                rx_count,
            )
            viz.node_service.ensure_tx_rx_markers_created(tx_count, rx_count)
            logger.debug("Markers created - positions will be updated after ViewModel processing")

        if hasattr(viz, "_material_filter_dirty"):
            viz._material_filter_dirty = False

        if hasattr(viz, "_coverage_interpolation_dirty"):
            viz._coverage_interpolation_dirty = False

        if current_state.show_coverage and viz.coverage_data is not None:
            view_model = self._add_coverage_to_view_model(view_model)
        else:
            view_model.coverage_vertices = None
            view_model.coverage_triangles = None
            view_model.coverage_colors = None
            view_model.coverage_isoline_points = None
            view_model.coverage_isoline_lines = None
            view_model.coverage_isoline_colors = None
            view_model.coverage_signature = None
            view_model.coverage_metadata = None
            view_model.show_coverage = False

        return view_model

    @staticmethod
    def _coverage_serving_tx_count(
        coverage_data: dict[str, Any],
        values: np.ndarray | None = None,
    ) -> int:
        """Return the TX category count for a serving-TX coverage layer."""
        counts: list[int] = []
        try:
            counts.append(int(coverage_data.get("serving_tx_count", 0)))
        except (TypeError, ValueError):
            pass
        counts.append(len(coverage_data.get("tx_names", []) or []))

        if values is not None:
            arr = np.asarray(values, dtype=np.float32)
            finite_serving = arr[np.isfinite(arr) & (arr >= 0)]
            if finite_serving.size:
                counts.append(int(np.max(finite_serving)) + 1)

        return max(counts or [0], default=0)

    @staticmethod
    def _apply_threshold_mask_to_mesh_colors(
        colors: np.ndarray,
        threshold_mask: np.ndarray,
        grid_shape: np.ndarray,
        render_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Dim coverage mesh cells that do not satisfy the threshold mask."""
        mesh_colors = np.asarray(colors, dtype=np.float64).copy()
        mask = np.asarray(threshold_mask, dtype=bool)
        nx = int(grid_shape[0])
        ny = int(grid_shape[1])
        if mask.shape != (ny, nx):
            raise ValueError("coverage threshold mask shape mismatch")

        cell_mask = mask.T.ravel()
        if render_mask is not None:
            visible = np.asarray(render_mask, dtype=bool)
            if visible.shape != (ny, nx):
                raise ValueError("coverage render mask shape mismatch")
            cell_mask = cell_mask[visible.T.ravel()]
        vertex_mask = np.repeat(cell_mask, 4)
        if vertex_mask.size != mesh_colors.shape[0]:
            raise ValueError("coverage threshold mask does not match mesh color count")
        fail = ~vertex_mask
        if np.any(fail):
            fail_colors = mesh_colors[fail]
            gray = np.mean(fail_colors, axis=1, keepdims=True)
            mesh_colors[fail] = fail_colors * 0.32 + gray * 0.68
            mesh_colors[fail] *= 0.55
        return np.clip(mesh_colors, 0.0, 1.0)

    @staticmethod
    def _coverage_effective_color_range(
        values: np.ndarray,
        value_min: Any,
        value_max: Any,
        metric_name: str,
    ) -> Optional[tuple[float, float]]:
        """Return one usable range shared by mesh colors and coverage legends."""
        try:
            lower = float(value_min)
            upper = float(value_max)
        except (TypeError, ValueError):
            lower = upper = float("nan")
        valid_bounds = np.isfinite(lower) and np.isfinite(upper) and lower <= upper
        if coverage_metric_color_scale(metric_name) == "logarithmic":
            valid_bounds = valid_bounds and lower > 0.0 and upper > 0.0
        if valid_bounds:
            return lower, upper

        array = np.asarray(values, dtype=np.float32)
        usable = array[coverage_metric_valid_mask(array, metric_name)]
        if usable.size == 0:
            return None
        return float(usable.min()), float(usable.max())

    @staticmethod
    def _coverage_isoline_levels(
        values_2d: np.ndarray,
        level_count: int,
        metric_name: str,
    ) -> list[float]:
        """Return isoline levels spaced according to the metric's color scale."""
        values = np.asarray(values_2d, dtype=np.float32)
        valid = values[coverage_metric_valid_mask(values, metric_name)]
        if valid.size == 0:
            return []
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            return []
        count = max(2, min(int(level_count), 12))
        if coverage_metric_color_scale(metric_name) == "logarithmic":
            levels = np.geomspace(vmin, vmax, count + 2)[1:-1]
        else:
            levels = np.linspace(vmin, vmax, count + 2)[1:-1]
        return [float(level) for level in levels]

    def _build_coverage_isolines(
        self,
        values_2d: np.ndarray,
        *,
        grid_origin: np.ndarray,
        grid_spacing: np.ndarray,
        z_level: float,
        metric_name: str,
        level_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Build auto isolines for one continuous coverage metric slice."""
        if not supports_coverage_threshold(metric_name):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 3), dtype=np.float64),
                [],
            )

        levels = self._coverage_isoline_levels(values_2d, level_count, metric_name)
        points_accum: list[np.ndarray] = []
        lines_accum: list[np.ndarray] = []
        colors_accum: list[np.ndarray] = []
        point_offset = 0

        for level_index, level in enumerate(levels):
            points, lines = build_coverage_isoline(
                values_2d,
                grid_origin=grid_origin,
                grid_spacing=grid_spacing,
                z_level=z_level,
                level=level,
                z_offset=0.06 + level_index * 0.001,
            )
            if points.shape[0] == 0 or lines.shape[0] == 0:
                continue
            points_accum.append(points)
            lines_accum.append(lines + point_offset)
            colors_accum.append(
                np.tile(
                    np.array([[0.05, 0.05, 0.05]], dtype=np.float64),
                    (lines.shape[0], 1),
                )
            )
            point_offset += points.shape[0]

        if not points_accum:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 3), dtype=np.float64),
                [],
            )

        return (
            np.vstack(points_accum),
            np.vstack(lines_accum),
            np.vstack(colors_accum),
            levels,
        )

    def _add_coverage_to_view_model(self, view_model: ViewModel) -> ViewModel:
        """Add coverage map data to the supplied ViewModel."""

        viz = self.visualizer

        coverage_data = viz.coverage_data
        grid_origin = coverage_data["grid_origin"]
        grid_spacing = coverage_data["grid_spacing"]
        grid_shape = coverage_data["grid_shape"]
        value_min = coverage_data["value_min"]
        value_max = coverage_data["value_max"]
        metric_name = coverage_data["metric_name"]

        values_3d = coverage_data.get("values_3d")
        if values_3d is None:
            values_flat = coverage_data.get("values")
            if values_flat is None or grid_shape.size < 3:
                raise ValueError("Coverage data missing values for mesh construction")
            nz = int(grid_shape[2])
            ny = int(grid_shape[1])
            nx = int(grid_shape[0])
            values_3d = np.asarray(values_flat, dtype=np.float32).reshape((nz, ny, nx))
            viz.coverage_data["values_3d"] = values_3d
        values_3d = np.asarray(values_3d, dtype=np.float32)

        heights = viz.coverage_heights or coverage_data.get("heights") or []
        if not heights:
            base_z = float(grid_origin[2]) if grid_origin.size >= 3 else 0.0
            heights = [base_z for _ in range(values_3d.shape[0])]
            viz.coverage_heights = heights

        state_height_index = getattr(
            viz.app_state, "coverage_height_index", viz.coverage_height_index
        )
        max_index = max(0, len(heights) - 1)
        selected_index = max(0, min(state_height_index, max_index))
        if selected_index != viz.coverage_height_index:
            viz.coverage_height_index = selected_index
            if hasattr(viz, "ui_manager") and "coverage" in viz.ui_manager.panels:
                panel = viz.ui_manager.panels["coverage"]
                if hasattr(panel, "set_height_index"):
                    panel.set_height_index(selected_index)
        if selected_index != state_height_index:
            viz.set_state(coverage_height_index=selected_index)

        file_backed = bool(coverage_data.get("coverage_file"))
        if file_backed:
            self.coverage_service.select_height_layer(coverage_data, selected_index)
            values_3d = np.asarray(coverage_data["values_3d"], dtype=np.float32)
            if values_3d.ndim != 3 or values_3d.shape[0] != 1:
                raise ValueError(
                    "file-backed coverage values must contain exactly one active "
                    f"height, got {values_3d.shape}"
                )
            values_index = 0
            mesh_heights = [heights[selected_index]]
        else:
            values_index = selected_index
            mesh_heights = heights

        slice_plan = [(values_index, True)]
        selected_values = np.asarray(values_3d[values_index], dtype=np.float32)
        total_count = int(selected_values.size)
        is_serving_tx = is_serving_tx_metric(metric_name)
        color_scale = coverage_metric_color_scale(metric_name)
        tx_count = (
            self._coverage_serving_tx_count(coverage_data, values_3d)
            if is_serving_tx
            else len(coverage_data.get("tx_names", []) or [])
        )
        tx_names = serving_tx_labels(coverage_data.get("tx_names", []) or [], tx_count)
        if is_serving_tx:
            finite_count = int(serving_tx_valid_mask(selected_values, tx_count).sum())
            effective_range = (float(value_min), float(value_max))
        else:
            valid_values = coverage_metric_valid_mask(selected_values, metric_name)
            finite_count = int(valid_values.sum())
            effective_range = self._coverage_effective_color_range(
                values_3d,
                value_min,
                value_max,
                metric_name,
            )
            if effective_range is None:
                effective_range = (1.0, 1.0) if color_scale == "logarithmic" else (0.0, 1.0)
            value_min, value_max = effective_range
            coverage_data["value_min"] = value_min
            coverage_data["value_max"] = value_max
        no_data_fraction = 1.0 - (finite_count / total_count) if total_count > 0 else 0.0
        color_range_available = bool(is_serving_tx or finite_count > 0)

        # Categorical transmitter IDs must never be spatially interpolated.
        interpolation_method = getattr(viz, "coverage_interpolation_method", "none")
        if is_serving_tx:
            interpolation_method = "none"
            viz.coverage_interpolation_method = "none"
        render_values = np.where(
            coverage_metric_valid_mask(selected_values, metric_name),
            selected_values,
            np.nan,
        )
        display_values = self.coverage_service.interpolate_values(
            render_values,
            interpolation_method,
        )
        cache_key = self.coverage_service.compute_cache_key(
            coverage_data,
            selected_index,
            interpolation_method,
        )

        cached_mesh = self.coverage_service.get_mesh(cache_key, copy=False)
        if cached_mesh is not None:
            vertices, triangles, colors = cached_mesh
        else:
            vertices, triangles, colors = self._build_coverage_mesh(
                grid_origin,
                grid_spacing,
                grid_shape,
                values_3d,
                value_min,
                value_max,
                metric_name,
                mesh_heights,
                slice_plan,
            )
            self.coverage_service.put_mesh(cache_key, vertices, triangles, colors)

        threshold_enabled = bool(getattr(viz, "coverage_threshold_enabled", False))
        threshold_value = getattr(viz, "coverage_threshold_value", None)
        threshold_mask_enabled = bool(getattr(viz, "coverage_threshold_mask_enabled", False))
        isolines_enabled = bool(getattr(viz, "coverage_isolines_enabled", False))
        isoline_count = int(getattr(viz, "coverage_isoline_count", 6))
        threshold_mask = None
        threshold_mask_applied = False
        isoline_points = np.empty((0, 3), dtype=np.float64)
        isoline_lines = np.empty((0, 2), dtype=np.int32)
        isoline_colors = np.empty((0, 3), dtype=np.float64)
        selected_height_value = (
            heights[selected_index] if 0 <= selected_index < len(heights) else None
        )

        if threshold_enabled and threshold_value is not None:
            try:
                threshold_mask = compute_coverage_threshold_mask(
                    display_values,
                    metric_name=metric_name,
                    threshold=float(threshold_value),
                )
                if threshold_mask_enabled:
                    colors = self._apply_threshold_mask_to_mesh_colors(
                        colors,
                        threshold_mask,
                        grid_shape,
                        render_mask=np.isfinite(display_values),
                    )
                    threshold_mask_applied = True
            except (TypeError, ValueError, IndexError) as exc:
                logger.debug("Coverage threshold visual unavailable: %s", exc)
                threshold_mask = None
                threshold_mask_applied = False

        if isolines_enabled and selected_height_value is not None:
            try:
                planned_levels = self._coverage_isoline_levels(
                    display_values,
                    isoline_count,
                    metric_name,
                )
                isoline_cache_key = self.coverage_service.compute_isoline_cache_key(
                    cache_key,
                    planned_levels,
                    float(selected_height_value),
                )
                cached_isolines = self.coverage_service.get_isolines(
                    isoline_cache_key,
                    copy=False,
                )
                if cached_isolines is None:
                    isoline_points, isoline_lines, isoline_colors, isoline_levels = (
                        self._build_coverage_isolines(
                            display_values,
                            grid_origin=grid_origin,
                            grid_spacing=grid_spacing,
                            z_level=float(selected_height_value),
                            metric_name=metric_name,
                            level_count=isoline_count,
                        )
                    )
                    self.coverage_service.put_isolines(
                        isoline_cache_key,
                        isoline_points,
                        isoline_lines,
                        isoline_colors,
                        isoline_levels,
                    )
                else:
                    (
                        isoline_points,
                        isoline_lines,
                        isoline_colors,
                        cached_levels,
                    ) = cached_isolines
                    isoline_levels = list(cached_levels)
            except (TypeError, ValueError, IndexError) as exc:
                logger.debug("Coverage isolines unavailable: %s", exc)
                isoline_levels = []
                isoline_cache_key = None
        else:
            isoline_levels = []
            isoline_cache_key = None

        mask_signature = (
            f"{float(threshold_value)}"
            if threshold_mask_applied and threshold_value is not None
            else "off"
        )
        mesh_signature = f"{cache_key}|mask={mask_signature}"
        isolines_applied = bool(isolines_enabled and isoline_levels)
        isoline_signature = isoline_cache_key if isolines_applied else None

        view_model.coverage_vertices = vertices
        view_model.coverage_triangles = triangles
        view_model.coverage_colors = colors
        view_model.coverage_isoline_points = isoline_points
        view_model.coverage_isoline_lines = isoline_lines
        view_model.coverage_isoline_colors = isoline_colors
        view_model.coverage_signature = mesh_signature
        view_model.coverage_opacity = float(viz.coverage_opacity)
        view_model.coverage_metadata = {
            "metric_name": metric_name,
            "value_min": value_min if color_range_available else None,
            "value_max": value_max if color_range_available else None,
            "grid_shape": grid_shape,
            "opacity": viz.coverage_opacity,
            "heights": heights,
            "selected_height_index": selected_index,
            "selected_height_value": selected_height_value,
            "valid_cell_count": finite_count,
            "total_cell_count": total_count,
            "no_data_fraction": no_data_fraction,
            "tx_names": tx_names,
            "tx_count": tx_count,
            "threshold_enabled": threshold_enabled,
            "threshold_value": threshold_value,
            "threshold_mask_enabled": threshold_mask_enabled,
            "threshold_mask_applied": threshold_mask_applied,
            "isolines_enabled": isolines_enabled,
            "isolines_applied": isolines_applied,
            "isoline_count": isoline_count,
            "isoline_levels": isoline_levels,
            "isoline_segments": int(isoline_lines.shape[0]),
            "isoline_signature": isoline_signature,
            "colormap": coverage_metric_colormap(metric_name),
            "color_scale": color_scale,
        }
        view_model.show_coverage = True

        logger.debug(
            "Coverage mesh added: %d vertices, %d triangles",
            len(vertices),
            len(triangles),
        )

        return view_model

    def _build_coverage_mesh(
        self,
        grid_origin: np.ndarray,
        grid_spacing: np.ndarray,
        grid_shape: np.ndarray,
        values_3d: np.ndarray,
        value_min: float,
        value_max: float,
        metric_name: str,
        heights: list[float],
        slice_plan: list[tuple[int, bool]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a discrete grid mesh from structured grid data for one or multiple slices."""

        from matplotlib import colormaps
        from matplotlib.colors import LogNorm, Normalize

        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        ox = float(grid_origin[0]) if grid_origin.size >= 1 else 0.0
        oy = float(grid_origin[1]) if grid_origin.size >= 2 else 0.0
        dx = float(grid_spacing[0]) if grid_spacing.size >= 1 else 1.0
        dy = float(grid_spacing[1]) if grid_spacing.size >= 2 else 1.0

        value_min = float(value_min)
        value_max = float(value_max)
        # Coverage colors encode metric semantics rather than the generic scene
        # theme: green always means the favorable end of the displayed range.
        cmap_name = coverage_metric_colormap(metric_name) or "viridis"
        cmap = colormaps[cmap_name]
        coverage_data = getattr(self.visualizer, "coverage_data", {}) or {}
        is_serving_tx = is_serving_tx_metric(metric_name)
        color_scale = coverage_metric_color_scale(metric_name)
        serving_tx_count = self._coverage_serving_tx_count(coverage_data, values_3d)
        color_min = value_min
        color_max = value_max
        if color_scale == "logarithmic" and color_min > 0 and color_max > 0:
            if color_max == color_min:
                factor = 1.05
                norm = LogNorm(vmin=color_min / factor, vmax=color_max * factor)
            else:
                norm = LogNorm(vmin=color_min, vmax=color_max)
        elif value_max == value_min:
            epsilon = max(abs(value_min) * 0.05, np.finfo(np.float64).eps)
            norm = Normalize(vmin=value_min - epsilon, vmax=value_max + epsilon)
        else:
            norm = Normalize(vmin=value_min, vmax=value_max)

        vertices_accum = []
        triangles_accum = []
        colors_accum = []
        vertex_offset = 0

        viz = self.visualizer

        for slice_index, highlight in slice_plan:
            if slice_index < 0 or slice_index >= values_3d.shape[0]:
                continue

            values_2d = values_3d[slice_index]

            values_2d = np.where(
                coverage_metric_valid_mask(values_2d, metric_name),
                values_2d,
                np.nan,
            )

            # Apply spatial interpolation for smoother visualization
            interpolation_method = getattr(viz, "coverage_interpolation_method", "none")
            if is_serving_tx:
                interpolation_method = "none"
            values_2d = self.coverage_service.interpolate_values(values_2d, interpolation_method)

            z_level = (
                heights[slice_index]
                if 0 <= slice_index < len(heights)
                else float(grid_origin[2]) if grid_origin.size >= 3 else 0.0
            )

            if is_serving_tx:
                valid_mask = serving_tx_valid_mask(values_2d, serving_tx_count)
                colors_2d = np.zeros((*values_2d.shape, 3), dtype=np.float64)
                tx_indices = np.full(values_2d.shape, -1, dtype=np.int32)
                tx_indices[valid_mask] = values_2d[valid_mask].astype(np.int32)
                for tx_index in range(serving_tx_count):
                    colors_2d[tx_indices == tx_index] = serving_tx_color_rgb(tx_index)
                render_mask = valid_mask
            else:
                valid_mask = coverage_metric_valid_mask(values_2d, metric_name)
                normalized_values = np.full_like(values_2d, 0.5, dtype=np.float32)
                normalized_values[valid_mask] = norm(values_2d[valid_mask])
                colors_2d = cmap(np.clip(normalized_values, 0.0, 1.0))[:, :, :3]
                colors_2d[~valid_mask] = 0.0
                render_mask = valid_mask

            # Vectorized grid build keeps large coverage slices out of Python loops.
            grid_cell_count = nx * ny
            if grid_cell_count == 0:
                continue

            # Cell corner coordinates via broadcasting
            ix = np.arange(nx, dtype=np.float64)
            jy = np.arange(ny, dtype=np.float64)
            jj, ii = np.meshgrid(jy, ix)  # ii: (nx, ny), jj: (nx, ny)
            ii_flat = ii.ravel()
            jj_flat = jj.ravel()
            cell_render_mask = render_mask[jj_flat.astype(np.intp), ii_flat.astype(np.intp)]
            ii_flat = ii_flat[cell_render_mask]
            jj_flat = jj_flat[cell_render_mask]
            n_cells = ii_flat.size
            if n_cells == 0:
                continue

            x0 = ox + ii_flat * dx
            x1 = x0 + dx
            y0 = oy + jj_flat * dy
            y1 = y0 + dy
            z = np.full(n_cells, z_level, dtype=np.float64)

            # 4 vertices per cell: (v0, v1, v2, v3) = (x0y0, x1y0, x1y1, x0y1)
            slice_vertices = np.empty((n_cells * 4, 3), dtype=np.float64)
            slice_vertices[0::4, 0] = x0
            slice_vertices[0::4, 1] = y0
            slice_vertices[0::4, 2] = z
            slice_vertices[1::4, 0] = x1
            slice_vertices[1::4, 1] = y0
            slice_vertices[1::4, 2] = z
            slice_vertices[2::4, 0] = x1
            slice_vertices[2::4, 1] = y1
            slice_vertices[2::4, 2] = z
            slice_vertices[3::4, 0] = x0
            slice_vertices[3::4, 1] = y1
            slice_vertices[3::4, 2] = z

            # 2 triangles per cell referencing local vertex indices
            base = np.arange(n_cells, dtype=np.int32) * 4
            slice_triangles = np.empty((n_cells * 2, 3), dtype=np.int32)
            slice_triangles[0::2, 0] = base
            slice_triangles[0::2, 1] = base + 1
            slice_triangles[0::2, 2] = base + 2
            slice_triangles[1::2, 0] = base
            slice_triangles[1::2, 1] = base + 2
            slice_triangles[1::2, 2] = base + 3
            slice_triangles += vertex_offset

            # Per-vertex colors: each cell's 4 vertices share the same color
            # colors_2d is (ny, nx, 3); index with (jj_flat, ii_flat) for each cell
            cell_colors = colors_2d[jj_flat.astype(np.intp), ii_flat.astype(np.intp)]
            slice_colors = np.repeat(cell_colors, 4, axis=0).astype(np.float64)

            vertices_accum.append(slice_vertices)
            triangles_accum.append(slice_triangles)
            colors_accum.append(slice_colors)

            vertex_offset += slice_vertices.shape[0]

        if not vertices_accum:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.int32),
                np.empty((0, 3), dtype=np.float64),
            )

        vertices = np.vstack(vertices_accum)
        triangles = np.vstack(triangles_accum)
        colors = np.vstack(colors_accum)

        return vertices, triangles, colors
