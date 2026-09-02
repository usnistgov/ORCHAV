"""Open3D GUI event-pump and redraw scheduling helpers.

Open3D's native GUI is separate from the Qt control shell. This mixin bridges
that boundary by ticking the Open3D application from Qt timers, suppressing
intermediate presents during frame transactions, and coalescing redraws.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

import open3d.visualization.gui as gui

from shared.logging import get_logger

try:
    from PySide6.QtCore import QTimer

    HAS_QTIMER = True
except ImportError:
    QTimer = None  # type: ignore[assignment]
    HAS_QTIMER = False

logger = get_logger("orchav.renderer_open3d")


class Open3DRuntimeMixin:
    """Own Open3D timer ticks, redraw requests, and frame transactions."""

    _FRAME_UPDATE_TIMEOUT_S: float = 5.0

    def get_runtime_stats(self) -> dict[str, Any]:
        """Return draw-pump telemetry without claiming physical presentation."""
        stats = super().get_runtime_stats()
        baseline = getattr(self, "_benchmark_telemetry_baseline", {})

        def _benchmark_delta(name: str, current: int) -> int | None:
            if name not in baseline:
                return None
            return max(0, int(current) - int(baseline[name]))

        stats.update(
            {
                "presentation_observable": False,
                "present_attempts": None,
                "present_successes": None,
                "draw_pump_attempts": int(getattr(self, "_draw_pump_attempts", 0)),
                "draw_pump_alive": int(getattr(self, "_draw_pump_alive", 0)),
                "benchmark_event_pump_calls": _benchmark_delta(
                    "event_pump_calls",
                    int(getattr(self, "_event_pump_calls", 0)),
                ),
                "benchmark_redraw_requests": _benchmark_delta(
                    "redraw_requests",
                    int(getattr(self, "_redraw_requests", 0)),
                ),
                "benchmark_frame_submissions": int(
                    getattr(self, "_benchmark_frame_submissions", 0)
                ),
                "benchmark_redraw_pump_attempts": int(
                    getattr(self, "_benchmark_redraw_pump_attempts", 0)
                ),
                "benchmark_redraw_pump_alive": int(
                    getattr(self, "_benchmark_redraw_pump_alive", 0)
                ),
            }
        )

        def _rect_size(rect: Any) -> list[float] | None:
            if rect is None:
                return None
            try:
                width = float(rect.width)
                height = float(rect.height)
            except (AttributeError, TypeError, ValueError):
                return None
            if width <= 0.0 or height <= 0.0:
                return None
            return [width, height]

        visualizer = self._o3d_vis
        stats["renderer_content_size"] = _rect_size(
            getattr(visualizer, "content_rect", None) if visualizer is not None else None
        )
        stats["renderer_window_size"] = _rect_size(
            getattr(visualizer, "os_frame", None) if visualizer is not None else None
        )
        return stats

    def _tick_o3d_gui(self) -> None:
        """Advance the Open3D GUI from the Qt timer."""
        if self._frame_update_in_progress:
            elapsed = time.monotonic() - self._frame_update_start_time
            if elapsed > self._FRAME_UPDATE_TIMEOUT_S:
                logger.warning(
                    "Frame update suppression exceeded %.1fs (%.1fs elapsed) "
                    "- forcing end_frame_update()",
                    self._FRAME_UPDATE_TIMEOUT_S,
                    elapsed,
                )
                self.end_frame_update()
            return

        if self._gui_initialized and self._o3d_vis is not None:
            redraw_pending = bool(getattr(self, "_native_redraw_pending", False))
            try:
                self._observe_camera_state("gui_tick_pre")
                self._record_event_pump()
                running = gui.Application.instance.run_one_tick()
                self._observe_camera_state("gui_tick_post")
            except RuntimeError as exc:
                if redraw_pending:
                    self._record_draw_pump(False, benchmark_turn=True)
                # Retain both the redraw and render-turn waiters. A later native
                # timer tick may recover, but playback must not advance without
                # a normal Open3D event-loop turn.
                logger.debug("Open3D GUI tick failed; retaining playback permit: %s", exc)
                return

            if redraw_pending:
                self._record_draw_pump(running is not False, benchmark_turn=True)
            if running is False:
                self._invalidate_native_pump("Open3D GUI event loop stopped")
                return

            if redraw_pending:
                self._native_redraw_pending = False
            self._release_deferred_render_turn()

    def begin_frame_update(self) -> None:
        """Begin a frame transaction and suppress intermediate redraw requests."""
        self._frame_update_in_progress = True
        self._frame_update_start_time = time.monotonic()
        self._frame_redraw_pending = False
        logger.debug("Open3DRenderer: Begin frame update")

    def _flush_pending_object_visibility(self) -> bool:
        """Flush geometry-mixin visibility bookkeeping when it is composed."""
        flush = getattr(self, "_flush_visualizer_visibility_updates", None)
        return bool(flush()) if callable(flush) else True

    def end_frame_update(self) -> bool:
        """End a frame transaction and report whether its redraw was queued."""
        self._frame_update_in_progress = False

        self._frame_redraw_pending = False
        if self._o3d_vis is None:
            return False

        # Visibility changes use the low-level scene API while a frame is
        # open. Synchronize O3DVisualizer's geometry tree only after every
        # object has reached its final state, before the final frame redraw.
        self._flush_pending_object_visibility()
        self._set_far_clipping_plane()
        self._force_scene_redraw()
        queued = self._post_redraw()
        if queued:
            if bool(getattr(self, "_benchmark_telemetry_active", False)):
                self._benchmark_frame_submissions = (
                    int(getattr(self, "_benchmark_frame_submissions", 0)) + 1
                )
            logger.debug("Open3DRenderer: End frame update (single redraw queued)")
        return queued

    def defer_until_next_render_turn(self, callback: Callable[[], None]) -> bool:
        """Release ``callback`` after the next separately pumped Open3D turn.

        Open3D's Python API does not expose Filament frame acceptance or a
        physical-present callback. Waiting for the next native GUI turn still
        prevents Maximum playback from repeatedly mutating the scene before
        Open3D has had an independent opportunity to consume the queued redraw.
        """
        timer = self._gui_timer
        if self._o3d_vis is None or timer is None:
            return False
        is_active = getattr(timer, "isActive", None)
        if callable(is_active) and not bool(is_active()):
            return False
        # All accepted waiters belong to the same next native turn. Controller
        # generations decide whether their individual continuations are stale.
        self._deferred_render_turn_callbacks.append(callback)
        return True

    def _release_deferred_render_turn(self) -> None:
        """Release queued renderer-turn waiters after ``run_one_tick`` returns."""
        callbacks = list(getattr(self, "_deferred_render_turn_callbacks", ()))
        if not callbacks:
            return
        self._deferred_render_turn_callbacks.clear()
        generation = int(getattr(self, "_render_turn_lifecycle_generation", 0))

        def _run_if_current() -> None:
            for callback in callbacks:
                if (
                    generation != int(getattr(self, "_render_turn_lifecycle_generation", 0))
                    or self._o3d_vis is None
                ):
                    return
                try:
                    callback()
                except (RuntimeError, AttributeError, TypeError) as exc:
                    logger.debug("Deferred Open3D render-turn callback failed: %s", exc)

        try:
            if HAS_QTIMER and QTimer is not None:
                QTimer.singleShot(0, _run_if_current)
            else:
                _run_if_current()
        except (RuntimeError, AttributeError, TypeError) as exc:
            logger.debug("Failed to queue deferred Open3D render-turn callbacks: %s", exc)

    def _invalidate_native_pump(self, reason: str) -> None:
        """Stop a dead native pump and invalidate callbacks without running them."""
        timer = self._gui_timer
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                logger.debug("Failed to stop dead Open3D GUI timer", exc_info=True)
        self._render_turn_lifecycle_generation += 1
        self._deferred_render_turn_callbacks.clear()
        self._native_redraw_pending = False
        self._visibility_settle_redraw_pending = False
        logger.warning("%s; playback render-turn permits were invalidated", reason)

    def update_renderer(self) -> None:
        """Force a render pass unless a frame transaction is in progress."""
        if self._frame_update_in_progress:
            self._frame_redraw_pending = True
            return

        if self._o3d_vis is not None:
            self._flush_pending_object_visibility()
            self._set_far_clipping_plane()
            self._submit_redraw_now()
            return
        self._record_redraw_request()
        self._record_draw_pump(False)

    def poll_events(self) -> None:
        """No-op compatibility hook; Qt timer ticks the Open3D GUI."""
        pass

    def request_redraw(self) -> None:
        """Request a redraw without forcing an immediate render."""
        self._post_redraw()

    def refresh_viewport_hud(self) -> None:
        """No-op because the Open3D backend does not provide a viewport HUD."""
        return None

    def _post_redraw(self) -> bool:
        """Request a redraw, coalescing it behind active batch/frame scopes."""
        if self._frame_update_in_progress:
            self._frame_redraw_pending = True
            return True
        if self._batch_mode:
            self._batch_redraw_pending = True
            return True

        if self._o3d_vis is not None:
            self._record_redraw_request()
            self._o3d_vis.post_redraw()
            self._native_redraw_pending = True
            return True
        return False

    def _submit_redraw_now(self) -> bool:
        """Queue and process one redraw before returning to frame playback.

        ``post_redraw()`` posts the native window event that asks Open3D to
        draw.  The event must exist before ``run_one_tick()`` pumps the Open3D
        loop; reversing those calls reports frame completion while leaving the
        requested draw queued behind the next scenario-frame mutation.
        """
        if self._o3d_vis is None:
            self._record_draw_pump(False)
            return False
        self._force_scene_redraw()
        if not self._post_redraw():
            self._record_draw_pump(False)
            return False
        try:
            self._record_event_pump()
            running = gui.Application.instance.run_one_tick()
        except RuntimeError as exc:
            self._record_draw_pump(False)
            logger.debug("Open3D redraw pump failed: %s", exc)
            return False
        alive = running is not False
        self._record_draw_pump(alive)
        if alive:
            self._native_redraw_pending = False
        else:
            self._invalidate_native_pump("Open3D GUI event loop stopped during redraw")
        return alive

    def _record_draw_pump(self, alive: bool, *, benchmark_turn: bool = False) -> None:
        """Record redraw-bearing event pumps, not unobservable presents."""
        self._draw_pump_attempts = int(getattr(self, "_draw_pump_attempts", 0)) + 1
        if alive:
            self._draw_pump_alive = int(getattr(self, "_draw_pump_alive", 0)) + 1
        if benchmark_turn and bool(getattr(self, "_benchmark_telemetry_active", False)):
            self._benchmark_redraw_pump_attempts = (
                int(getattr(self, "_benchmark_redraw_pump_attempts", 0)) + 1
            )
            if alive:
                self._benchmark_redraw_pump_alive = (
                    int(getattr(self, "_benchmark_redraw_pump_alive", 0)) + 1
                )

    def begin_benchmark_telemetry(self) -> None:
        """Reset benchmark-local submission and native-pump counters."""
        self._benchmark_telemetry_active = True
        self._benchmark_frame_submissions = 0
        self._benchmark_redraw_pump_attempts = 0
        self._benchmark_redraw_pump_alive = 0
        self._benchmark_telemetry_baseline = {
            "event_pump_calls": int(getattr(self, "_event_pump_calls", 0)),
            "redraw_requests": int(getattr(self, "_redraw_requests", 0)),
        }

    def _render_debug(self, message: str, **fields: Any) -> None:
        """Emit detailed render logs when ORCHAV_RENDER_DEBUG is enabled."""
        if not (self._render_debug_enabled or logger.isEnabledFor(logging.DEBUG)):
            return
        if fields:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            logger.debug("RenderTrace: %s %s", message, details)
        else:
            logger.debug("RenderTrace: %s", message)

    def _request_visibility_settle_redraw(self, reason: str) -> None:
        """Post one scene-settling redraw after a separate Open3D native turn.

        On Windows, visibility, material, and persistent geometry mutations can
        update Open3D's scene state before Filament has presented every command.
        An immediate ``post_redraw()`` may therefore display an intermediate
        scene; mouse input fixes it only because it requests another draw later.
        O3DVisualizer exposes no presentation-complete callback for these
        changes.  The historic method name is retained because visibility was
        the first affected operation.

        The shared render-turn queue is released only by the independently
        scheduled Qt pump, after ``run_one_tick()`` returns. Its callback posts
        this redraw without pumping immediately, so the following native turn
        consumes it. Synchronous batch/update pumps cannot release the callback
        and accidentally collapse both draws into one turn. The boolean
        coalesces a whole label/object batch into one extra frame.
        """
        if self._o3d_vis is None or self._visibility_settle_redraw_pending:
            return
        self._visibility_settle_redraw_pending = True
        if not self.defer_until_next_render_turn(self._post_visibility_settle_redraw):
            self._visibility_settle_redraw_pending = False
            self._render_debug("reject_visibility_settle_redraw", reason=reason)
            return
        self._render_debug("request_visibility_settle_redraw", reason=reason)

    def _post_visibility_settle_redraw(self) -> None:
        """Post the coalesced redraw after the original native turn returned."""
        # Reset/close can invalidate a callback after it was extracted from the
        # shared queue but before Qt executes it.
        if not self._visibility_settle_redraw_pending:
            return
        self._visibility_settle_redraw_pending = False
        if self._o3d_vis is None:
            return
        if self._post_redraw():
            self._render_debug("post_visibility_settle_redraw")

    def _force_scene_redraw(self) -> None:
        """Mark the Filament scene dirty using Open3D's IBL setter."""
        if self._o3d_vis is not None:
            self._o3d_vis.set_ibl_intensity(self._ibl_intensity)

    @contextmanager
    def batch_updates(self) -> Generator[None, None, None]:
        """Batch multiple geometry updates behind one redraw."""
        was_in_batch = self._batch_mode
        prev_pending = self._batch_redraw_pending
        self._batch_mode = True
        if not was_in_batch:
            self._batch_redraw_pending = False
        try:
            yield
        finally:
            self._batch_mode = was_in_batch
            if was_in_batch:
                self._batch_redraw_pending = bool(self._batch_redraw_pending or prev_pending)
            else:
                if not self._frame_update_in_progress:
                    self._flush_pending_object_visibility()
                if self._batch_redraw_pending:
                    self._batch_redraw_pending = False
                    if self._o3d_vis is not None and not self._frame_update_in_progress:
                        self._submit_redraw_now()
                        logger.debug("Open3DRenderer: Batch updates complete, single redraw issued")

    def flush_redraw(self) -> None:
        """Request a redraw and process one GUI tick when possible."""
        if self._frame_update_in_progress:
            self._frame_redraw_pending = True
            return

        if self._o3d_vis is not None:
            self._submit_redraw_now()
