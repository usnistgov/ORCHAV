"""Runtime loop and telemetry helpers for the pygfx renderer.

pygfx owns its draw loop through the canvas. This mixin tracks activity, pauses
presentation while frame geometry is mutating, coalesces redraw requests inside
batch updates, and exposes timing data for visualizer benchmarks.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

import numpy as np

from .canvas import _env_flag

logger = logging.getLogger(__name__)


class PygfxRuntimeMixin:
    """Own on-demand redraw scheduling and frame-update telemetry."""

    _RECENT_PRESENT_INTERVAL_LIMIT = 60
    _IDLE_PRESENT_GAP_S = 0.5

    @staticmethod
    def _read_max_fps() -> float:
        """Read the optional renderer-level FPS cap for runtime telemetry."""
        raw = os.environ.get("ORCHAV_PYGFX_MAX_FPS", "0")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        if value < 0.0:
            return 0.0
        return value

    def _request_canvas_draw(self) -> bool:
        """Request one canvas draw unless rendering is paused for frame mutation."""
        if (
            getattr(self, "_qt_window_closed", False)
            or not self._initialized
            or self._canvas is None
            or self._frame_update_paused
        ):
            return False
        self._redraw_requests += 1
        try:
            self._canvas.request_draw()
        except RuntimeError:
            mark_closed = getattr(self, "_mark_qt_window_closed", None)
            if callable(mark_closed):
                mark_closed()
            logger.debug("PygfxRenderer: canvas draw request ignored after Qt close")
            return False
        return True

    def _record_recent_present_interval(self, interval_s: float, *, animating: bool) -> None:
        """Record recent draw cadence without folding idle gaps into live FPS."""
        previous_animating = self._last_present_was_animating
        state_changed = previous_animating is not None and previous_animating != animating
        idle_gap = not animating and interval_s > self._IDLE_PRESENT_GAP_S
        if state_changed or idle_gap:
            self._recent_present_intervals_s.clear()
        else:
            self._recent_present_intervals_s.append(interval_s)
            if len(self._recent_present_intervals_s) > self._RECENT_PRESENT_INTERVAL_LIMIT:
                self._recent_present_intervals_s = self._recent_present_intervals_s[
                    -self._RECENT_PRESENT_INTERVAL_LIMIT :
                ]
        self._last_present_was_animating = animating

    def _animate(self) -> None:
        """Render one canvas frame requested by application or controller state."""
        if (
            not self._initialized
            or getattr(self, "_qt_window_closed", False)
            or self._renderer is None
            or self._scene is None
            or self._camera is None
        ):
            return
        callback_start = time.perf_counter()
        self._draw_callbacks_received += 1
        self._render_attempts += 1

        telemetry_tick = logger.isEnabledFor(logging.DEBUG) and self._render_attempts % 120 == 1

        if telemetry_tick and self._scene is not None:
            obj_count = [0]

            def _count(obj: Any) -> None:
                """Count every scene child visited by pygfx traversal."""
                obj_count[0] += 1

            self._scene.traverse(_count)
            n_objects = obj_count[0]
            n_static = 0
            if self._static_group is not None:
                static_count = [0]

                def _count_static(obj: Any) -> None:
                    """Count static scene children for telemetry breakdowns."""
                    static_count[0] += 1

                self._static_group.traverse(_count_static)
                n_static = static_count[0] - 1
            prefix_counts: dict[str, int] = {}
            for gname in self._name_to_handle:
                prefix = gname.rsplit("_", 1)[0] if "_" in gname else gname
                for p in (
                    "merged_group_",
                    "scene_outline_",
                    "scene_merged_outline_",
                    "vm_label_",
                    "tx_label_",
                    "rx_label_",
                    "target_",
                    "orientation_",
                    "bldg_label_",
                ):
                    if gname.startswith(p):
                        prefix = p.rstrip("_")
                        break
                else:
                    prefix = gname
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(prefix_counts.items()))
            logger.debug(
                "[pygfx-telemetry] scene: %d total objects (%d static/scene), "
                "%d named geometries: %s",
                n_objects,
                n_static,
                len(self._name_to_handle),
                breakdown,
            )

        try:
            self._update_headlight_pose()
            if _env_flag("ORCHAV_PYGFX_LIGHT_DEBUG", False):
                self._dump_lighting_state()
            render_start = time.perf_counter()
            minimap_drawn = False
            if self._minimap_enabled:
                self._renderer.render(
                    self._scene,
                    self._camera,
                    flush=False,
                )
                minimap_drawn = self._render_minimap()
            if not minimap_drawn:
                self._renderer.render(
                    self._scene,
                    self._camera,
                    flush=True,
                )
            render_end = time.perf_counter()
            self._render_successes += 1

            draw_dur = render_end - render_start
            self._last_renderer_submit_ms = draw_dur * 1000.0
            self._draw_durations.append(draw_dur)
            if len(self._draw_durations) > 240:
                self._draw_durations = self._draw_durations[-240:]

            if telemetry_tick:
                avg_draw = (
                    sum(self._draw_durations) / len(self._draw_durations)
                    if self._draw_durations
                    else 0.0
                )
                max_draw = max(self._draw_durations) if self._draw_durations else 0.0
                logger.debug(
                    "[pygfx-telemetry] render: %.1fms this frame, "
                    "avg=%.1fms max=%.1fms over last %d frames, "
                    "paused=%s",
                    draw_dur * 1000,
                    avg_draw * 1000,
                    max_draw * 1000,
                    len(self._draw_durations),
                    self._frame_update_paused,
                )

            if self._pending_update_start is not None:
                utp = render_end - self._pending_update_start
                self._update_to_present_times.append(utp)
                if len(self._update_to_present_times) > 240:
                    self._update_to_present_times = self._update_to_present_times[-240:]
                self._pending_update_start = None

            now = render_end
            if self._first_present_at is None:
                self._first_present_at = now
                self._schedule_deferred_default_ibl_load()
            if self._last_present_success_at > 0.0:
                dt = now - self._last_present_success_at
                if dt > 0.0:
                    self._present_interval_sum_s += dt
                    self._present_interval_sq_sum_s += dt * dt
                    self._present_interval_samples += 1
                    if dt > self._present_interval_max_s:
                        self._present_interval_max_s = dt
                    target_dt = self._min_frame_dt_s if self._min_frame_dt_s > 0 else (1.0 / 60.0)
                    if dt > 2.0 * target_dt:
                        self._frame_drop_count += 1
                    self._record_recent_present_interval(
                        dt,
                        animating=bool(getattr(self.visualizer, "animation_running", False)),
                    )
            else:
                self._last_present_was_animating = bool(
                    getattr(self.visualizer, "animation_running", False)
                )
            self._last_present_success_at = now
        except Exception as exc:
            self._render_failures += 1
            logger.debug("PygfxRenderer: render failed: %s", exc)

        callback_total = time.perf_counter() - callback_start
        self._last_draw_callback_total_ms = callback_total * 1000.0
        self._draw_callback_total_durations.append(callback_total)
        if len(self._draw_callback_total_durations) > 240:
            self._draw_callback_total_durations = self._draw_callback_total_durations[-240:]

    def update_renderer(self) -> None:
        """Request the canvas to schedule a draw on the next Qt event loop tick."""
        if not self._initialized or getattr(self, "_qt_window_closed", False):
            return
        if self._batch_mode:
            self._batch_redraw_pending = True
            return
        now = time.perf_counter()
        if self._last_update_call_at > 0.0:
            dt = now - self._last_update_call_at
            if dt > 0.0:
                self._update_interval_sum_s += dt
                self._update_interval_samples += 1
        self._last_update_call_at = now

        is_animating = getattr(self.visualizer, "animation_running", False)
        if is_animating:
            self._update_calls_while_animating += 1
        else:
            self._update_calls_while_animating = 0

        self._pending_update_start = now

        self._tick_count += 1
        if self._last_tick_at > 0.0:
            tick_dt = now - self._last_tick_at
            if tick_dt > 0.0:
                self._tick_interval_sum_s += tick_dt
                self._tick_interval_samples += 1
        self._last_tick_at = now

        self._request_canvas_draw()

    def request_redraw(self) -> None:
        """Request a draw while respecting active batch-update coalescing."""
        if getattr(self, "_qt_window_closed", False):
            return
        if self._batch_mode:
            self._batch_redraw_pending = True
            return
        self._request_canvas_draw()

    def defer_until_next_render_turn(self, callback: Callable[[], None]) -> bool:
        """Decline an extra pacing boundary for the canvas-owned render loop.

        pygfx submits through rendercanvas' scheduler, so playback can continue
        immediately after frame submission. The callback remains owned by the
        caller when this method returns ``False``.
        """
        del callback
        return False

    @contextmanager
    def batch_updates(self) -> Generator[None, None, None]:
        """Coalesce redraws while a group of renderer mutations is applied."""
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
            elif self._initialized and self._batch_redraw_pending:
                self._batch_redraw_pending = False
                self.update_renderer()

    def begin_frame_update(self) -> None:
        """Pause presentation while geometry is being updated."""
        self._frame_update_paused = True
        self._frame_update_start = time.perf_counter()
        self._last_end_frame_update_breakdown = {}
        self._frame_update_metrics = {}
        self._last_end_frame_update_breakdown_bytes = {}
        self._frame_update_bytes = {}

    def _record_frame_update_metric(self, name: str, elapsed_ms: float) -> None:
        """Accumulate one frame-update timing metric in milliseconds."""
        if elapsed_ms <= 0.0:
            return
        self._frame_update_metrics[name] = self._frame_update_metrics.get(name, 0.0) + elapsed_ms

    def _profile_detail_enabled(self) -> bool:
        """Return whether detailed benchmark profiling should collect sub-metrics."""
        if _env_flag("ORCHAV_BENCH_PROFILE_DETAIL", False):
            return True
        pipeline = getattr(self.visualizer, "pipeline", None)
        return getattr(pipeline, "benchmark_recorder", None) is not None

    def _record_profile_metric(self, name: str, elapsed_ms: float) -> None:
        """Record an optional profiling metric for the current frame update."""
        if self._profile_detail_enabled():
            self._record_frame_update_metric(name, elapsed_ms)

    def _record_profile_bytes(self, name: str, value: float) -> None:
        """Record optional byte counts for benchmark upload accounting."""
        if value <= 0.0 or not self._profile_detail_enabled():
            return
        self._frame_update_bytes[name] = self._frame_update_bytes.get(name, 0.0) + value

    def _record_profile_array_bytes(self, name: str, *arrays: Any) -> None:
        """Record total byte size for numpy-like arrays when profiling is enabled."""
        if not self._profile_detail_enabled():
            return
        total = 0.0
        for array in arrays:
            if array is None:
                continue
            try:
                total += float(np.asarray(array).nbytes)
            except (TypeError, ValueError):
                continue
        self._record_profile_bytes(name, total)

    def end_frame_update(self) -> bool:
        """Present or request one draw and report whether it was accepted."""
        if getattr(self, "_qt_window_closed", False):
            self._frame_update_paused = True
            self._last_end_frame_update_breakdown = {}
            self._last_end_frame_update_breakdown_bytes = {}
            return self._complete_mpc_path_inspection_transition(False)
        self._frame_update_paused = False
        if not self._initialized or self._canvas is None:
            self._last_end_frame_update_breakdown = {}
            self._last_end_frame_update_breakdown_bytes = {}
            return self._complete_mpc_path_inspection_transition(False)
        benchmark_present_mode = getattr(self, "_benchmark_present_mode", "blocking")
        t_total_start = time.perf_counter()
        frame_update_dur = 0.0
        if hasattr(self, "_frame_update_start"):
            frame_update_dur = (time.perf_counter() - self._frame_update_start) * 1000
        breakdown = {
            "geometry_update_ms": frame_update_dur,
        }
        breakdown.update(self._frame_update_metrics)
        bytes_breakdown = dict(self._frame_update_bytes)
        t_start = time.perf_counter()
        if benchmark_present_mode == "blocking" and hasattr(self._canvas, "force_draw"):
            try:
                callbacks_before = self._draw_callbacks_received
                self._redraw_requests += 1
                self._canvas.force_draw()
                t_end = time.perf_counter()
                force_draw_ms = (t_end - t_start) * 1000.0
                callback_count = self._draw_callbacks_received - callbacks_before
                self._blocking_frame_count += 1
                self._blocking_force_draw_callbacks += callback_count
                breakdown["force_draw_ms"] = force_draw_ms
                breakdown["force_draw_callback_count"] = float(callback_count)
                if callback_count == 1:
                    callback_ms = self._last_draw_callback_total_ms
                    breakdown["draw_callback_total_ms"] = callback_ms
                    breakdown["renderer_submit_ms"] = self._last_renderer_submit_ms
                    breakdown["canvas_present_residual_ms"] = max(
                        0.0,
                        force_draw_ms - callback_ms,
                    )
                else:
                    self._blocking_force_draw_contaminated += 1
                breakdown["total_ms"] = (t_end - t_total_start) * 1000.0
                self._last_end_frame_update_breakdown = breakdown
                self._last_end_frame_update_breakdown_bytes = bytes_breakdown
                logger.debug(
                    "[pygfx-telemetry] end_frame_update: "
                    "geometry_update=%.1fms "
                    "force_draw=%.1fms total=%.1fms",
                    breakdown["geometry_update_ms"],
                    breakdown["force_draw_ms"],
                    breakdown["total_ms"],
                )
                return self._complete_mpc_path_inspection_transition(True)
            except (RuntimeError, AttributeError):
                pass
        accepted = self._request_canvas_draw()
        t_end = time.perf_counter()
        breakdown["request_draw_ms"] = (t_end - t_start) * 1000.0
        breakdown["total_ms"] = (t_end - t_total_start) * 1000.0
        self._last_end_frame_update_breakdown = breakdown
        self._last_end_frame_update_breakdown_bytes = bytes_breakdown
        logger.debug(
            "[pygfx-telemetry] end_frame_update: "
            "geometry_update=%.1fms "
            "request_draw=%.1fms total=%.1fms",
            breakdown["geometry_update_ms"],
            breakdown["request_draw_ms"],
            breakdown["total_ms"],
        )
        return self._complete_mpc_path_inspection_transition(accepted)

    def _complete_mpc_path_inspection_transition(self, accepted: bool) -> bool:
        """Finalize renderer-local selection state at the presentation boundary."""
        finish = getattr(
            self,
            "_finish_mpc_path_inspection_frame_transition",
            None,
        )
        if callable(finish):
            try:
                finish(presented=bool(accepted))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.warning(
                    "PygfxRenderer: failed to finalize MPC selection frame transition",
                    exc_info=True,
                )
        return bool(accepted)

    def get_last_end_frame_update_breakdown(self) -> dict[str, float]:
        """Return the most recent end-of-frame renderer timing breakdown."""
        return dict(self._last_end_frame_update_breakdown)

    def get_last_end_frame_update_breakdown_bytes(self) -> dict[str, float]:
        """Return the most recent end-of-frame upload byte breakdown."""
        return dict(self._last_end_frame_update_breakdown_bytes)

    def begin_benchmark_telemetry(self) -> None:
        """Snapshot cumulative counters before benchmark stepping begins."""
        # Blocking-draw counters describe the benchmark interval itself,
        # unlike the renderer lifetime counters snapshotted below. Reset them
        # so startup/deferred draws cannot contaminate the one-draw-per-frame
        # invariant reported for play-mode benchmarks.
        self._blocking_frame_count = 0
        self._blocking_force_draw_callbacks = 0
        self._blocking_force_draw_contaminated = 0
        self._benchmark_telemetry_baseline = {
            "draw_callbacks_received": int(self._draw_callbacks_received),
            "present_attempts": int(self._render_attempts),
            "present_successes": int(self._render_successes),
            "redraw_requests": int(self._redraw_requests),
        }

    def _line_cache_replay_frame_count_hint(self) -> int:
        """Estimate animation length for MPC expanded-line cache sizing hints."""
        for attr in ("total_animation_steps", "total_steps"):
            value = getattr(self.visualizer, attr, None)
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                return count
        panel_manager = getattr(self.visualizer, "panel_manager", None)
        value = getattr(panel_manager, "total_steps", None)
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        return max(0, count)

    @staticmethod
    def _bytes_to_mb_ceil(value: int) -> int:
        """Convert bytes to a rounded-up MiB count for human telemetry."""
        if value <= 0:
            return 0
        return int((int(value) + 1024 * 1024 - 1) // (1024 * 1024))

    def _line_cache_fit_status(
        self,
        *,
        suggested_full_loop_bytes: int,
        hit_rate: Optional[float],
    ) -> str:
        """Classify whether the expanded-line cache fits the replay workload."""
        max_bytes = int(self._mpc_expanded_line_cache_max_bytes)
        largest = int(self._mpc_expanded_line_cache_largest_entry_bytes)
        if max_bytes <= 0:
            return "disabled"
        if largest <= 0:
            return "no_mpc_lines_seen"
        if (
            self._mpc_expanded_line_cache_rejected_oversize > 0
            and self._mpc_expanded_line_cache_stores == 0
        ):
            return "entry_exceeds_budget"
        if self._mpc_expanded_line_cache_evictions > 0 and hit_rate is not None and hit_rate < 0.25:
            return "churning"
        if suggested_full_loop_bytes > 0 and max_bytes < suggested_full_loop_bytes:
            return "undersized_for_full_loop"
        if (
            self._mpc_expanded_line_cache_evictions == 0
            and self._mpc_expanded_line_cache_stores > 0
        ):
            return "fits_observed_workload"
        return "active"

    def get_runtime_stats(self) -> dict[str, Any]:
        """Return renderer telemetry and MPC line-cache health counters."""
        startup_ms = None
        if self._first_present_at is not None:
            startup_ms = (self._first_present_at - self._created_at) * 1000.0
        avg_present_interval_ms = (
            (self._present_interval_sum_s / self._present_interval_samples) * 1000.0
            if self._present_interval_samples > 0
            else None
        )
        effective_present_fps = (
            (1.0 / (self._present_interval_sum_s / self._present_interval_samples))
            if self._present_interval_samples > 0 and self._present_interval_sum_s > 0.0
            else None
        )
        recent_present_interval_ms = None
        recent_present_fps = None
        if self._recent_present_intervals_s:
            recent_window = self._recent_present_intervals_s[
                -min(len(self._recent_present_intervals_s), 30) :
            ]
            recent_mean_s = sum(recent_window) / len(recent_window)
            recent_present_interval_ms = recent_mean_s * 1000.0
            if recent_mean_s > 0.0:
                recent_present_fps = 1.0 / recent_mean_s
        avg_update_interval_ms = (
            (self._update_interval_sum_s / self._update_interval_samples) * 1000.0
            if self._update_interval_samples > 0
            else None
        )

        avg_draw_ms = None
        if self._draw_durations:
            window = self._draw_durations[-min(len(self._draw_durations), 30) :]
            avg_draw_ms = round((sum(window) / len(window)) * 1000.0, 3)

        avg_draw_callback_total_ms = None
        if self._draw_callback_total_durations:
            window = self._draw_callback_total_durations[
                -min(len(self._draw_callback_total_durations), 30) :
            ]
            avg_draw_callback_total_ms = round((sum(window) / len(window)) * 1000.0, 3)

        avg_utp_ms = None
        effective_utp_fps = None
        if self._update_to_present_times:
            window = self._update_to_present_times[-min(len(self._update_to_present_times), 30) :]
            avg_utp_ms = round((sum(window) / len(window)) * 1000.0, 3)
            if avg_utp_ms > 0:
                effective_utp_fps = round(1000.0 / avg_utp_ms, 3)

        jitter_ms = None
        if self._present_interval_samples > 1:
            mean = self._present_interval_sum_s / self._present_interval_samples
            variance = (
                self._present_interval_sq_sum_s / self._present_interval_samples
            ) - mean * mean
            if variance > 0:
                jitter_ms = round((variance**0.5) * 1000.0, 3)

        frame_drop_rate = None
        if self._present_interval_samples > 0:
            frame_drop_rate = round(self._frame_drop_count / self._present_interval_samples, 4)

        tick_fps = None
        avg_tick_interval_ms = None
        if self._tick_interval_samples > 0 and self._tick_interval_sum_s > 0.0:
            avg_tick_s = self._tick_interval_sum_s / self._tick_interval_samples
            avg_tick_interval_ms = round(avg_tick_s * 1000.0, 3)
            tick_fps = round(1.0 / avg_tick_s, 3)

        cache_accesses = self._mpc_expanded_line_cache_hits + self._mpc_expanded_line_cache_misses
        cache_hit_rate = (
            round(float(self._mpc_expanded_line_cache_hits) / float(cache_accesses), 4)
            if cache_accesses > 0
            else None
        )
        prewarm_avg_ms = (
            self._mpc_expanded_line_cache_prewarm_total_ms
            / self._mpc_expanded_line_cache_prewarm_attempts
            if self._mpc_expanded_line_cache_prewarm_attempts > 0
            else None
        )
        replay_frame_count_hint = self._line_cache_replay_frame_count_hint()
        suggested_full_loop_bytes = (
            int(self._mpc_expanded_line_cache_largest_entry_bytes) * replay_frame_count_hint
            if replay_frame_count_hint > 0
            else 0
        )
        cache_fit_status = self._line_cache_fit_status(
            suggested_full_loop_bytes=suggested_full_loop_bytes,
            hit_rate=cache_hit_rate,
        )
        baseline = self._benchmark_telemetry_baseline

        def _benchmark_delta(name: str, current: int) -> Optional[int]:
            """Return a counter delta when benchmark telemetry was initialized."""
            if name not in baseline:
                return None
            return max(0, int(current) - int(baseline[name]))

        def _canvas_value(method_name: str) -> Any:
            """Read optional rendercanvas size metadata without affecting rendering."""
            method = getattr(self._canvas, method_name, None)
            if not callable(method):
                return None
            try:
                value = method()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return None
            if isinstance(value, (tuple, list)):
                return [float(item) for item in value]
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        def _renderer_value(attribute_name: str) -> Any:
            """Read optional pygfx renderer metadata without triggering draws."""
            try:
                value = getattr(self._renderer, attribute_name)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return None
            if isinstance(value, (tuple, list)):
                return [float(item) for item in value]
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        canvas_logical_size = _canvas_value("get_logical_size")
        canvas_physical_size = _canvas_value("get_physical_size")

        return {
            "event_pump_calls": int(self._event_pump_calls),
            "redraw_requests": int(self._redraw_requests),
            "present_attempts": int(self._render_attempts),
            "present_successes": int(self._render_successes),
            "present_failures": int(self._render_failures),
            "startup_to_first_frame_ms": None if startup_ms is None else round(startup_ms, 3),
            "initial_present_attempted": bool(getattr(self, "_initial_present_attempted", False)),
            "initial_present_succeeded": getattr(
                self,
                "_initial_present_succeeded",
                None,
            ),
            "initial_present_duration_ms": (
                None
                if getattr(self, "_initial_present_duration_ms", None) is None
                else round(float(self._initial_present_duration_ms), 3)
            ),
            "initial_present_error": getattr(self, "_initial_present_error", None),
            "max_fps_cap": float(self._max_fps),
            "canvas_update_mode": str(getattr(self, "_canvas_update_mode", "")),
            "canvas_max_fps": getattr(self, "_canvas_max_fps", None),
            "canvas_vsync": getattr(self, "_canvas_vsync", None),
            "canvas_present_method_requested": str(
                getattr(self, "_canvas_present_method_requested", "unresolved")
            ),
            "canvas_present_method": str(getattr(self, "_canvas_present_method", "unresolved")),
            "canvas_present_fallback_reason": getattr(
                self,
                "_canvas_present_fallback_reason",
                None,
            ),
            "canvas_refresh_rate_hz": float(getattr(self, "_canvas_refresh_rate_hz", 60.0)),
            "canvas_uses_display_refresh": bool(
                getattr(self, "_canvas_uses_display_refresh", False)
            ),
            "canvas_logical_size": canvas_logical_size,
            "canvas_physical_size": canvas_physical_size,
            "canvas_pixel_ratio": _canvas_value("get_pixel_ratio"),
            "renderer_content_size": canvas_physical_size,
            "renderer_window_size": canvas_logical_size,
            "renderer_internal_size": _renderer_value("physical_size"),
            "renderer_pixel_scale": _renderer_value("pixel_scale"),
            "renderer_pixel_ratio": _renderer_value("pixel_ratio"),
            "canvas_schedule_applied": bool(getattr(self, "_canvas_schedule_applied", False)),
            "continuous_redraw": self.capabilities.continuous_redraw,
            "avg_present_interval_ms": (
                None if avg_present_interval_ms is None else round(avg_present_interval_ms, 3)
            ),
            "effective_present_fps": (
                None if effective_present_fps is None else round(effective_present_fps, 3)
            ),
            "lifetime_present_fps": (
                None if effective_present_fps is None else round(effective_present_fps, 3)
            ),
            "recent_present_interval_ms": (
                None if recent_present_interval_ms is None else round(recent_present_interval_ms, 3)
            ),
            "recent_present_fps": (
                None if recent_present_fps is None else round(recent_present_fps, 3)
            ),
            "avg_update_interval_ms": (
                None if avg_update_interval_ms is None else round(avg_update_interval_ms, 3)
            ),
            "update_calls_while_animating": int(self._update_calls_while_animating),
            "draw_callbacks_received": int(self._draw_callbacks_received),
            "benchmark_draw_callbacks": _benchmark_delta(
                "draw_callbacks_received",
                self._draw_callbacks_received,
            ),
            "benchmark_present_attempts": _benchmark_delta(
                "present_attempts",
                self._render_attempts,
            ),
            "benchmark_present_successes": _benchmark_delta(
                "present_successes",
                self._render_successes,
            ),
            "benchmark_redraw_requests": _benchmark_delta(
                "redraw_requests",
                self._redraw_requests,
            ),
            "forced_draw_fallbacks": int(self._forced_draw_fallbacks),
            "avg_draw_ms": avg_draw_ms,
            "avg_draw_callback_total_ms": avg_draw_callback_total_ms,
            "blocking_frame_count": int(self._blocking_frame_count),
            "blocking_force_draw_callbacks": int(self._blocking_force_draw_callbacks),
            "blocking_force_draw_contaminated": int(self._blocking_force_draw_contaminated),
            "avg_update_to_present_ms": avg_utp_ms,
            "effective_utp_fps": effective_utp_fps,
            "present_jitter_ms": jitter_ms,
            "idle_loop_active": False,
            "present_interval_max_ms": (
                round(self._present_interval_max_s * 1000.0, 3)
                if self._present_interval_max_s > 0
                else None
            ),
            "frame_drop_rate": frame_drop_rate,
            "tick_count": int(self._tick_count),
            "avg_tick_interval_ms": avg_tick_interval_ms,
            "tick_fps": tick_fps,
            "mpc_line_cache_entries": int(len(self._mpc_expanded_line_cache)),
            "mpc_line_cache_bytes": int(self._mpc_expanded_line_cache_bytes),
            "mpc_line_cache_max_bytes": int(self._mpc_expanded_line_cache_max_bytes),
            "mpc_line_cache_hits": int(self._mpc_expanded_line_cache_hits),
            "mpc_line_cache_misses": int(self._mpc_expanded_line_cache_misses),
            "mpc_line_cache_stores": int(self._mpc_expanded_line_cache_stores),
            "mpc_line_cache_evictions": int(self._mpc_expanded_line_cache_evictions),
            "mpc_line_cache_hit_rate": cache_hit_rate,
            "mpc_line_cache_fit_status": cache_fit_status,
            "mpc_line_cache_rejected_oversize": int(
                self._mpc_expanded_line_cache_rejected_oversize
            ),
            "mpc_line_cache_prewarm_enabled": bool(
                self._mpc_expanded_line_cache_prewarm_enabled
                and self._mpc_expanded_line_cache_max_bytes > 0
            ),
            "mpc_line_cache_prewarm_attempts": int(self._mpc_expanded_line_cache_prewarm_attempts),
            "mpc_line_cache_prewarm_stores": int(self._mpc_expanded_line_cache_prewarm_stores),
            "mpc_line_cache_prewarm_existing": int(self._mpc_expanded_line_cache_prewarm_existing),
            "mpc_line_cache_prewarm_skips": int(self._mpc_expanded_line_cache_prewarm_skips),
            "mpc_line_cache_prewarm_total_ms": round(
                self._mpc_expanded_line_cache_prewarm_total_ms,
                3,
            ),
            "mpc_line_cache_prewarm_avg_ms": (
                None if prewarm_avg_ms is None else round(prewarm_avg_ms, 3)
            ),
            "mpc_line_cache_last_entry_bytes": int(self._mpc_expanded_line_cache_last_entry_bytes),
            "mpc_line_cache_largest_entry_bytes": int(
                self._mpc_expanded_line_cache_largest_entry_bytes
            ),
            "mpc_line_cache_largest_entry_mb": self._bytes_to_mb_ceil(
                self._mpc_expanded_line_cache_largest_entry_bytes
            ),
            "mpc_line_cache_replay_frame_count_hint": replay_frame_count_hint,
            "mpc_line_cache_suggested_full_loop_bytes": suggested_full_loop_bytes,
            "mpc_line_cache_suggested_full_loop_mb": self._bytes_to_mb_ceil(
                suggested_full_loop_bytes
            ),
            "mpc_line_segment_capacity": int(self._mpc_segment_capacity),
            "mpc_line_segment_capacity_hint": int(self._mpc_segment_capacity_hint),
            "mpc_point_capacity": int(self._mpc_point_capacity),
            "mpc_point_capacity_hint": int(self._mpc_point_capacity_hint),
        }
