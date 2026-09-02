"""Incremental ViewModel pre-warming service.

After background preloading finishes for each frame, the warmer builds the
corresponding :class:`~visualizer.src.pipeline.core.ViewModel` on the main thread
via ``QTimer(0)`` idle ticks so that the first playback hits the
``mpc_view_cache`` instead of computing ViewModels on-the-fly. It uses the same
cache key builder as ``FramePipeline`` and reads only raw frames that
``CacheService`` has already stored.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, QTimer, Slot

from shared.logging import get_logger

from ..config import MAX_VIEW_MODEL_CACHE_SIZE
from ..pipeline.frame_pipeline import build_vm_cache_key

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..pipeline.core import ViewModel

logger = get_logger(__name__)

# Maximum wall-clock time (ms) spent per idle tick to avoid UI stutter.
_TICK_BUDGET_MS = 8.0


class ViewModelWarmer(QObject):
    """Pre-warm the ViewModel cache as frames finish preloading.

    All work runs on the **main (Qt) thread** via ``QTimer(0)`` so there is
    no need for locks on ``mpc_view_cache`` or ``mpc_core``.

    This class inherits from :class:`QObject` so that cross-thread signal
    connections use ``QueuedConnection`` automatically — the ``step_ready``
    signal is emitted from the background preload thread, and ``enqueue``
    must execute on the main thread (QTimer requires its owning thread).

    Lifecycle:
        * ``enqueue(step)`` — called from the ``step_ready`` signal.
        * ``invalidate(reason)`` — called when state changes make queued work
          stale (e.g. ``mpc_view_cache.clear()``).
        * ``stop()`` — full reset on scenario change or shutdown.
    """

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Initialize the main-thread warming queue and idle timer."""
        super().__init__()
        self._viz = visualizer
        self._queue: deque[int] = deque()
        self._timer = QTimer(self)
        self._timer.setInterval(0)  # idle tick
        self._timer.timeout.connect(self._process_tick)
        self._active = False
        self._warmed_count = 0
        self._renderer_warmed_count = 0

        # Snapshot of the non-step cache-key fields taken when warming starts.
        # If the current state drifts from this snapshot the queue is stale.
        self._snapshot_key: Optional[tuple] = None

    # Public API

    @Slot(int)
    def enqueue(self, step: int) -> None:
        """Schedule *step* for ViewModel warming.

        Decorated as a ``Slot`` so that cross-thread signal connections
        automatically use ``QueuedConnection``, ensuring execution on the main
        thread that owns the QTimer.

        With ``QueuedConnection``, ``step_ready`` signals arrive one-at-a-time
        through the Qt event loop.  The timer may have drained the queue and
        paused between deliveries, so it restarts without resetting the
        session (snapshot / warmed_count are preserved).
        """
        if self._standalone_beamforming_is_visible():
            logger.debug(
                "ViewModel warmer: skipping step %d while standalone beamforming is visible",
                step,
            )
            return

        self._queue.append(step)
        if not self._active:
            self._activate()
        elif not self._timer.isActive():
            # Session still open but timer paused after queue drained — resume.
            self._timer.start()

    def invalidate(self, reason: str = "") -> None:
        """Discard all queued work because the cache key changed."""
        if not self._active and not self._queue:
            return
        count = len(self._queue)
        self._queue.clear()
        self._timer.stop()
        self._active = False
        self._snapshot_key = None
        if count > 0:
            logger.debug(
                "ViewModel warmer invalidated (%s): discarded %d queued steps",
                reason,
                count,
            )

    def stop(self) -> None:
        """Full reset (scenario change / shutdown).

        This is the *only* path that ends a warming session and logs the
        final count.  Called from ``reset_preloading_state()`` and
        ``closeEvent()``.
        """
        self._timer.stop()
        if self._active and self._warmed_count > 0:
            logger.info(
                "ViewModel warming complete (%d VMs, %d renderer caches pre-warmed)",
                self._warmed_count,
                self._renderer_warmed_count,
            )
        self._queue.clear()
        self._active = False
        self._snapshot_key = None
        self._warmed_count = 0
        self._renderer_warmed_count = 0

    # Internal helpers

    def _activate(self) -> None:
        """Take a state snapshot and start the idle timer."""
        self._snapshot_key = self._current_snapshot_key()
        self._active = True
        self._warmed_count = 0
        self._renderer_warmed_count = 0
        self._timer.start()
        logger.info("ViewModel warmer activated (snapshot taken)")

    def _current_snapshot_key(self) -> tuple:
        """Return the non-step portion of the VM cache key for the current state."""
        viz = self._viz
        state = viz.app_state
        mats_key = self._materials_key()
        # Step 0 is a sentinel because only request fields shared by all steps matter.
        full_key = build_vm_cache_key(0, state, mats_key)
        # Strip the step (first element) to get the "shape" of the key.
        return full_key[1:]

    def _materials_key(self) -> Optional[tuple]:
        """Return the material-filter portion of the ViewModel cache key."""
        mats = getattr(self._viz, "mpc_allowed_materials", None)
        if mats is None:
            return None
        material_scope = str(getattr(self._viz, "mpc_material_filter_scope", "segment"))
        if len(mats) == 0:
            return (material_scope, "__EMPTY__")
        return (material_scope, *tuple(sorted(mats)))

    def _is_state_still_valid(self) -> bool:
        """Check whether the current state still matches the activation snapshot."""
        if self._snapshot_key is None:
            return False
        return self._current_snapshot_key() == self._snapshot_key

    def _standalone_beamforming_is_visible(self) -> bool:
        """Return whether warming would require the costly standalone solver.

        The warmer intentionally derives frame-backed ViewModels.  Caching one
        of those results under a standalone-beam cache key would make the
        normal frame pipeline reuse a ViewModel that never computed the
        requested beam.  Standalone beams are therefore left to the regular
        on-demand path.
        """
        state = self._viz.app_state
        return bool(state.show_beamforming and state.standalone_beamforming_mode != "frame")

    def _prewarm_renderer_cache(self, step: int, vm: ViewModel) -> None:
        """Give renderers a chance to warm backend-specific caches for a ViewModel."""
        renderer = getattr(self._viz, "renderer", None)
        prewarm = getattr(renderer, "prewarm_mpc_line_cache", None)
        if not callable(prewarm):
            return
        try:
            if prewarm(vm.to_render_packet()):
                self._renderer_warmed_count += 1
                logger.debug(
                    "Renderer MPC line cache pre-warmed for step %d (total: %d)",
                    step,
                    self._renderer_warmed_count,
                )
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
            logger.debug("Renderer cache prewarm failed for step %d: %s", step, exc)

    def _process_tick(self) -> None:
        """Process queued steps within the time budget."""
        if not self._queue:
            self._finish()
            return

        # Yield to animation — don't compete with the render loop.
        if getattr(self._viz, "animation_running", False):
            return

        if self._standalone_beamforming_is_visible():
            self.invalidate(reason="standalone beamforming is visible")
            return

        if not self._is_state_still_valid():
            self.invalidate(reason="state drifted")
            return

        viz = self._viz
        cache = viz.mpc_view_cache
        state = viz.app_state
        mats_key = self._materials_key()
        tick_start = time.perf_counter()

        while self._queue:
            elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
            if elapsed_ms >= _TICK_BUDGET_MS:
                break

            step = self._queue.popleft()

            key = build_vm_cache_key(step, state, mats_key)
            if key in cache:
                self._prewarm_renderer_cache(step, cache[key])
                continue  # already warm

            if len(cache) >= MAX_VIEW_MODEL_CACHE_SIZE:
                logger.debug("ViewModel warmer: cache at capacity, pausing")
                self._queue.clear()
                break

            raw_frame = viz.cache_service.get_frame(step)
            if raw_frame is None:
                logger.debug("ViewModel warmer: no raw frame for step %d, skipping", step)
                continue

            # Warmed frames come from frame data; do not inject standalone array params.
            view_frame = dict(raw_frame)
            view_frame["standalone_beamforming_mode"] = "frame"

            vm_start = time.perf_counter()
            try:
                vm = viz.mpc_core.create_view_model(
                    step=step,
                    raw_frame=view_frame,
                    color_mode=state.color_mode,
                    selected_tx=state.selected_tx,
                    selected_rx=state.selected_rx,
                    mpc_allowed_orders=state.mpc_allowed_orders,
                    mpc_allowed_types=state.mpc_allowed_types,
                    mpc_allowed_materials=getattr(viz, "mpc_allowed_materials", None),
                    mpc_visibility=state.mpc_visibility,
                    topk_render_enabled=state.topk_render_enabled,
                    topk_render_max_paths=state.topk_render_max_paths,
                    show_beamforming=state.show_beamforming,
                    beamforming_azimuth_samples=state.beamforming_azimuth_samples,
                    beamforming_elevation_samples=state.beamforming_elevation_samples,
                    beamforming_tx_scale=state.beamforming_tx_scale,
                    beamforming_rx_scale=state.beamforming_rx_scale,
                    beamforming_tx_node=state.beamforming_tx_node,
                    beamforming_rx_node=state.beamforming_rx_node,
                    beamforming_db_scale=state.beamforming_db_scale,
                    beamforming_dynamic_range_db=state.beamforming_dynamic_range_db,
                    beamforming_colormap=state.beamforming_colormap,
                    beamforming_element_pattern=state.beamforming_element_pattern,
                    beamforming_tx_element_pattern=state.beamforming_tx_element_pattern,
                    beamforming_rx_element_pattern=state.beamforming_rx_element_pattern,
                    include_targets=True,
                    show_tx_segments=getattr(viz, "show_tx_segments", True),
                    delay_filter_min_ns=state.delay_filter_min_ns,
                    delay_filter_max_ns=state.delay_filter_max_ns,
                    power_filter_min_db=state.power_filter_min_db,
                    power_filter_max_db=state.power_filter_max_db,
                    aoa_az_filter_min_deg=state.aoa_az_filter_min_deg,
                    aoa_az_filter_max_deg=state.aoa_az_filter_max_deg,
                    aoa_el_filter_min_deg=state.aoa_el_filter_min_deg,
                    aoa_el_filter_max_deg=state.aoa_el_filter_max_deg,
                    aod_az_filter_min_deg=state.aod_az_filter_min_deg,
                    aod_az_filter_max_deg=state.aod_az_filter_max_deg,
                    aod_el_filter_min_deg=state.aod_el_filter_min_deg,
                    aod_el_filter_max_deg=state.aod_el_filter_max_deg,
                    use_distinct_material_colors=state.use_distinct_material_colors,
                )
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
                logger.debug("ViewModel warmer: failed for step %d: %s", step, exc)
                continue

            vm_elapsed = (time.perf_counter() - vm_start) * 1000.0
            if vm is not None:
                cache[key] = vm
                self._warmed_count += 1
                self._prewarm_renderer_cache(step, vm)
                logger.info(
                    "VM warmed: step %d in %.0fms (total warmed: %d)",
                    step,
                    vm_elapsed,
                    self._warmed_count,
                )

        if not self._queue:
            self._finish()

    def _finish(self) -> None:
        """Pause the timer when the queue is temporarily drained.

        The session stays active (snapshot & warmed_count preserved) because
        more ``step_ready`` signals may still be in-flight via
        ``QueuedConnection``.  ``enqueue()`` restarts the timer when the next
        step arrives.  The session only ends when ``stop()`` or
        ``invalidate()`` is called explicitly.
        """
        self._timer.stop()
