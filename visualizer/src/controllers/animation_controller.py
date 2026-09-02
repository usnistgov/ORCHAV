"""Playback and live-frame control for the visualizer UI.

``AnimationController`` translates buttons, sliders, timer ticks, and live gRPC
panel actions into frame updates. It delegates frame loading and cache warming
to ``AnimationService``/``FramePipeline`` and keeps controller-owned policy to
playback cadence, sparse-step traversal, live frame requests, and temporary
buffering while the raw-frame cache catches up.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Protocol

from PySide6.QtCore import QTimer

from shared.logging import get_logger

from ..playback import (
    DEFAULT_FIXED_PLAYBACK_FPS,
    PlaybackMode,
    fixed_fps_interval_ms,
    real_time_interval_ms,
    timestamp_interval_ms,
)
from ..services.animation_service import AnimationService

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.animation_controller")


class PlaybackControls(Protocol):
    """Narrow playback-control surface consumed by ``AnimationController``."""

    def live_speed_multiplier(self) -> float:
        """Return the selected live-playback speed multiplier."""
        ...

    def live_play_as_available(self) -> bool:
        """Return whether live playback should step through available frames."""
        ...

    def frame_stride(self) -> int:
        """Return the selected frame stride."""
        ...

    def playback_mode(self) -> PlaybackMode:
        """Return the selected scenario-frame playback policy."""
        ...

    def fixed_playback_fps(self) -> int:
        """Return the requested scenario-frame rate for Fixed FPS mode."""
        ...

    def loop_enabled(self) -> bool:
        """Return whether playback wraps at sequence boundaries."""
        ...

    def is_online_mode(self) -> bool:
        """Return whether navigation should use live gRPC request semantics."""
        ...

    def request_frame_if_needed(self, frame_idx: int) -> bool:
        """Request a live frame through the animation panel when appropriate."""
        ...

    def prefetch_enabled(self, *, playing: bool) -> bool:
        """Return whether live prefetch should currently be enabled."""
        ...


class _VisualizerPlaybackControls:
    """Read playback-control state from the visualizer's Qt widgets."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        self._visualizer = visualizer

    def _animation_panel(self) -> Any:
        ui_manager = getattr(self._visualizer, "ui_manager", None)
        panels = getattr(ui_manager, "panels", None)
        if not panels:
            return None
        return panels.get("animation")

    def _data_source_panel(self) -> Any:
        """Return the panel that owns live-stream prefetch policy controls."""
        ui_manager = getattr(self._visualizer, "ui_manager", None)
        panels = getattr(ui_manager, "panels", None)
        if not panels:
            return None
        return panels.get("data_source")

    @staticmethod
    def _is_checked(widget: Any, *, default: bool) -> bool:
        if widget is None:
            return default
        try:
            return bool(widget.isChecked())
        except (RuntimeError, AttributeError):
            return default

    def live_speed_multiplier(self) -> float:
        speed_widget = getattr(self._visualizer, "live_playback_speed", None)
        if speed_widget is None:
            return 1.0
        try:
            speed_text = str(speed_widget.currentText())
            return float(speed_text.replace("x", ""))
        except (RuntimeError, AttributeError, ValueError, TypeError):
            logger.debug("Failed to parse live playback speed; defaulting to 1x")
            return 1.0

    def live_play_as_available(self) -> bool:
        checkbox = getattr(self._visualizer, "live_play_as_available_cb", None)
        return self._is_checked(checkbox, default=True)

    def frame_stride(self) -> int:
        combo = getattr(self._visualizer, "stride_combo", None)
        if combo is None:
            return 1
        try:
            text = str(combo.currentText())
        except (RuntimeError, AttributeError):
            return 1
        if text == "Mesh":
            panel = self._animation_panel()
            if panel is None or not hasattr(panel, "compute_mesh_stride"):
                return 1
            try:
                return max(1, int(panel.compute_mesh_stride()))
            except (RuntimeError, ValueError, TypeError):
                return 1
        try:
            return max(1, int(text.replace("x", "")))
        except (ValueError, TypeError):
            return 1

    def playback_mode(self) -> PlaybackMode:
        panel = self._animation_panel()
        if panel is None or not hasattr(panel, "playback_mode"):
            return PlaybackMode.MAXIMUM
        try:
            return PlaybackMode(panel.playback_mode())
        except (RuntimeError, AttributeError, ValueError, TypeError):
            return PlaybackMode.MAXIMUM

    def fixed_playback_fps(self) -> int:
        panel = self._animation_panel()
        if panel is None or not hasattr(panel, "fixed_playback_fps"):
            return DEFAULT_FIXED_PLAYBACK_FPS
        try:
            return max(1, int(panel.fixed_playback_fps()))
        except (RuntimeError, AttributeError, ValueError, TypeError):
            return DEFAULT_FIXED_PLAYBACK_FPS

    def loop_enabled(self) -> bool:
        return self._is_checked(getattr(self._visualizer, "loop_cb", None), default=True)

    def is_online_mode(self) -> bool:
        panel = self._animation_panel()
        if panel is None or not hasattr(panel, "is_in_online_mode"):
            return False
        try:
            return bool(panel.is_in_online_mode())
        except (RuntimeError, AttributeError):
            return False

    def request_frame_if_needed(self, frame_idx: int) -> bool:
        panel = self._animation_panel()
        if panel is None or not hasattr(panel, "request_frame_if_needed"):
            return False
        try:
            return bool(panel.request_frame_if_needed(frame_idx))
        except (OSError, RuntimeError, AttributeError):
            return False

    def prefetch_enabled(self, *, playing: bool) -> bool:
        panel = self._data_source_panel()
        widgets = (getattr(panel, "widgets", {}) if panel is not None else {}) or {}
        enable_cb = widgets.get("prefetch_enable")
        pause_cb = widgets.get("prefetch_pause_when_paused")
        enabled = self._is_checked(enable_cb, default=True)
        if self._is_checked(pause_cb, default=False) and not playing:
            return False
        return enabled


class AnimationController:
    """Coordinate playback controls without owning frame derivation."""

    BUFFER_CHECK_INTERVAL_MS = 100

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        animation_service: AnimationService,
        timer_factory: Optional[Callable[[], QTimer]] = None,
        playback_controls: Optional[PlaybackControls] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        """Store playback collaborators and initialize live-buffering state."""
        self.visualizer = visualizer
        self.animation_service = animation_service
        self._timer_factory = timer_factory or QTimer
        self._playback_controls = playback_controls or _VisualizerPlaybackControls(visualizer)
        self._clock = clock or time.perf_counter
        self._real_time_next_deadline_s: Optional[float] = None
        self._fixed_next_deadline_s: Optional[float] = None
        self._playback_schedule_generation: int = 0
        self._renderer_turn_pending: bool = False
        self._restart_after_renderer_turn: bool = False
        self._playback_tick_step: Optional[int] = None
        self._playback_tick_recorded = False
        self._live_playback_timer: Optional[QTimer] = None
        self._buffering: bool = False
        self._buffering_step: Optional[int] = None
        self._buffer_check_timer: Optional[QTimer] = None

    def prepare_step(self, step: int) -> None:
        """Ask the animation service to load/cache one raw frame."""
        self.animation_service.load_step(step)

    def start_playback(self, start_step: int = 0) -> None:
        """Start service-level playback from a known step."""
        self.animation_service.start(start_step=start_step)

    def stop_playback(self) -> None:
        """Stop any running animation."""
        self.animation_service.stop()

    def advance(self) -> Optional[dict[str, Any]]:
        """Advance the animation service and return its loaded frame, if any."""
        return self.animation_service.advance()

    def preload_steps(self, steps: Iterable[int]) -> None:
        """Warm the raw-frame cache with upcoming steps."""
        self.animation_service.preload(steps)

    @property
    def is_buffering(self) -> bool:
        """True when playback is paused waiting for an unloaded frame."""
        return self._buffering

    def _is_online_mode(self) -> bool:
        """Return True when the animation panel is driving live gRPC requests."""
        return self._playback_controls.is_online_mode()

    def _start_buffering(self, step: int) -> None:
        """Pause playback and poll until *step* reaches the raw-frame cache."""
        self._buffering = True
        self._buffering_step = step
        # Do not let an arbitrary cache wait become a playback-rate sample.
        # Reset again when buffering ends so only post-resume completions are
        # compared with one another.
        self.reset_playback_cadence()
        logger.info("Buffering: frame %d not cached, pausing playback", step)

        viz = self.visualizer
        viz.animation_timer.stop()
        self._playback_schedule_generation += 1
        self._real_time_next_deadline_s = None
        self._fixed_next_deadline_s = None
        if hasattr(viz, "_set_status_message"):
            viz._set_status_message(f"Buffering... (frame {step} not loaded yet)")

        if self._buffer_check_timer is None:
            timer = self._timer_factory()
            timer.timeout.connect(self._check_buffer_ready)
            self._buffer_check_timer = timer
        self._buffer_check_timer.start(self.BUFFER_CHECK_INTERVAL_MS)

    def _check_buffer_ready(self) -> None:
        """Resume playback work once the awaited raw frame has been preloaded."""
        step = self._buffering_step
        if step is None:
            self._stop_buffering()
            return

        cache_service = getattr(self.animation_service, "cache_service", None)
        if cache_service is not None and cache_service.has_frame(step):
            logger.info("Buffering resolved: frame %d now available", step)
            self._stop_buffering()
            viz = self.visualizer
            if viz.animation_running:
                if not self._apply_playback_frame(step):
                    self._stop_after_failed_playback_frame(step)
                    return
                viz.ui_controller.update_performance_display()
                self._schedule_next_playback_tick(self._playback_controls.playback_mode())

    def _stop_buffering(self) -> None:
        """Clear buffering state and stop the poll timer."""
        was_buffering = self._buffering
        self._buffering = False
        self._buffering_step = None
        if self._buffer_check_timer is not None:
            self._buffer_check_timer.stop()
        if was_buffering:
            self.reset_playback_cadence()
            self._restore_scenario_summary()

    # Live playback helpers
    def handle_live_play_toggle(self, playing: bool) -> None:
        """Toggle live playback depending on checkbox state."""
        if playing:
            self.start_live_playback()
        else:
            self.stop_live_playback()

    def start_live_playback(self) -> None:
        """Start timer-driven playback over the active live frame source."""
        logger.info("Live streaming: Starting live playback")
        viz = self.visualizer
        frame_source = getattr(viz, "frame_source", None)

        if frame_source and hasattr(frame_source, "subscribe_to_frames"):
            frame_source.subscribe_to_frames(self.on_new_frame_available)

        timer = self._ensure_live_timer()
        speed_multiplier = self._playback_controls.live_speed_multiplier()
        base_interval = 1000
        timer.start(int(base_interval / speed_multiplier))
        logger.info("Live streaming: Playback started at %.2fx speed", speed_multiplier)

    def stop_live_playback(self) -> None:
        """Stop the live playback timer and detach listeners."""
        logger.info("Live streaming: Stopping live playback")
        timer = self._live_playback_timer
        if timer is not None:
            timer.stop()

        frame_source = getattr(self.visualizer, "frame_source", None)
        if frame_source and hasattr(frame_source, "unsubscribe_from_frames"):
            frame_source.unsubscribe_from_frames(self.on_new_frame_available)

    def on_new_frame_available(self, frame_idx: int, frame_data=None) -> None:
        """Observe streaming-source delivery; timer policy chooses what to show."""
        logger.debug("Live streaming: New frame %s available (ignored by controller)", frame_idx)

    def _ensure_live_timer(self) -> QTimer:
        """Create the live playback timer and attach its stepping callback."""
        if self._live_playback_timer is None:
            timer = self._timer_factory()
            timer.timeout.connect(self._live_playback_step)
            self._live_playback_timer = timer
        return self._live_playback_timer

    def _live_playback_step(self) -> None:
        """Advance live playback by available-frame or sequential policy."""
        viz = self.visualizer
        frame_source = getattr(viz, "frame_source", None)
        if not frame_source:
            return

        play_as_available = self._playback_controls.live_play_as_available()
        if play_as_available and hasattr(frame_source, "list_frames"):
            available_frames = frame_source.list_frames()
            if not available_frames:
                logger.debug("Live streaming: No frames available")
                return
            current_frame = getattr(viz, "animation_step", 0)
            next_frames = [frame for frame in available_frames if frame > current_frame]
            if next_frames:
                next_frame = min(next_frames)
                logger.debug("Live streaming: Playing next available frame %s", next_frame)
                viz.update_frame(next_frame)
            else:
                logger.debug("Live streaming: No new frames available")
            return

        logger.debug("Live streaming: Sequential playback step")
        self.next_frame()

    # User-requested streaming controls
    def request_live_frame(self, frame_idx: int) -> None:
        """Handle explicit frame fetches from the live panel."""
        logger.info("Live streaming: Frame %s requested", frame_idx)
        viz = self.visualizer
        frame_source = getattr(viz, "frame_source", None)
        if not frame_source or not hasattr(frame_source, "request_frame"):
            logger.warning(
                "Live streaming: No frame source available or request_frame not supported"
            )
            return

        success = frame_source.request_frame(frame_idx)
        if success:
            viz.update_frame(frame_idx)
            logger.info("Live streaming: Successfully loaded frame %s", frame_idx)
        else:
            logger.warning("Live streaming: Failed to load frame %s", frame_idx)

    def reconnect_live_stream(self) -> bool:
        """Reconnect to the streaming provider if possible."""
        viz = self.visualizer
        frame_source = getattr(viz, "frame_source", None)
        connection_manager = getattr(frame_source, "connection_manager", None)
        if connection_manager is None:
            logger.warning("Live streaming: No connection manager available for reconnect")
            return False

        logger.info("Live streaming: Reconnecting to gRPC server")
        try:
            connection_manager.close()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
            logger.debug("Live streaming: Failed to close existing connection: %s", exc)
        success = bool(connection_manager.ensure_connection())
        if success:
            logger.info("Live streaming: Reconnection successful")
        else:
            logger.error("Live streaming: Reconnection failed")
        return success

    def clear_live_buffer(self) -> None:
        """Clear remote gRPC caches when possible, otherwise local buffers."""
        viz = self.visualizer
        frame_source = getattr(viz, "frame_source", None)
        if frame_source and hasattr(frame_source, "request_cache_flush"):
            frame_source.request_cache_flush(reason="Live panel clear buffer")
            logger.info("Live streaming: Cache flush requested")
            return
        if frame_source and hasattr(frame_source, "clear_buffer"):
            frame_source.clear_buffer()
            logger.info("Live streaming: Buffer cleared (local only)")
            return
        logger.warning("Live streaming: No buffer clear API available")

    # Stride helpers
    def get_current_stride(self) -> int:
        """Return the currently selected playback stride."""
        return self._playback_controls.frame_stride()

    @staticmethod
    def _frame_timestamp_seconds(frame: Any) -> Optional[float]:
        """Read an explicitly unit-tagged semantic timestamp from a cached frame."""
        if not isinstance(frame, dict):
            return None
        if frame.get("timestamp_ns") is not None:
            try:
                return float(frame["timestamp_ns"]) / 1_000_000_000.0
            except (TypeError, ValueError):
                return None
        if frame.get("timestamp_s") is not None:
            try:
                return float(frame["timestamp_s"])
            except (TypeError, ValueError):
                return None
        return None

    def _cached_timestamp_interval_ms(self, current_step: int, next_step: int) -> Optional[int]:
        """Return timestamp cadence when both adjacent raw frames are cached."""
        cache_service = getattr(self.animation_service, "cache_service", None)
        if cache_service is None:
            cache_service = getattr(self.visualizer, "cache_service", None)
        get_frame = getattr(cache_service, "get_frame", None)
        if not callable(get_frame):
            return None
        try:
            current_frame = get_frame(current_step)
            next_frame = get_frame(next_step)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None
        current_timestamp = self._frame_timestamp_seconds(current_frame)
        next_timestamp = self._frame_timestamp_seconds(next_frame)
        if current_timestamp is None or next_timestamp is None:
            return None
        return timestamp_interval_ms(current_timestamp, next_timestamp)

    def _real_time_timer_interval_ms(self) -> int:
        """Resolve real-time cadence from timestamps, then scenario duration."""
        viz = self.visualizer
        try:
            available_steps = viz.get_available_animation_steps()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            available_steps = []
        stride = self._playback_controls.frame_stride()
        fallback_frame_count = len(available_steps)
        if self._playback_controls.is_online_mode():
            try:
                declared_frame_count = int(getattr(viz, "total_animation_steps", 0))
            except (TypeError, ValueError):
                declared_frame_count = 0
            if declared_frame_count >= 2:
                fallback_frame_count = declared_frame_count
        fallback = real_time_interval_ms(
            getattr(viz, "_frame_duration", None),
            fallback_frame_count,
            frame_stride=stride,
        )
        if len(available_steps) < 2:
            return fallback

        try:
            current_index = viz.get_animation_step_index(viz.animation_step)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return fallback
        delta = -stride if getattr(viz, "play_direction", 1) < 0 else stride
        next_index = current_index + delta
        if next_index < 0 or next_index >= len(available_steps):
            return fallback

        timestamp_ms = self._cached_timestamp_interval_ms(
            available_steps[current_index],
            available_steps[next_index],
        )
        return timestamp_ms if timestamp_ms is not None else fallback

    def playback_timer_interval_ms(self) -> int:
        """Return the Qt timer interval for the selected playback policy."""
        mode = self._playback_controls.playback_mode()
        if mode is PlaybackMode.MAXIMUM:
            return 0
        if mode is PlaybackMode.REAL_TIME:
            return self._real_time_timer_interval_ms()
        return fixed_fps_interval_ms(self._playback_controls.fixed_playback_fps())

    def restart_playback_timer(self) -> None:
        """Arm one serial playback tick for the active cadence policy."""
        viz = self.visualizer
        if not getattr(viz, "animation_running", False) or self._buffering:
            return
        viz.animation_timer.stop()
        self._playback_schedule_generation += 1
        self._real_time_next_deadline_s = None
        self._fixed_next_deadline_s = None
        if self._renderer_turn_pending:
            # A timing or direction change may invalidate the old controller
            # generation, but it must not bypass the native turn already owed
            # to the frame that just completed.
            self._restart_after_renderer_turn = True
            logger.debug("Playback restart deferred until pending renderer turn")
            return
        self._restart_after_renderer_turn = False
        self._arm_playback_timer()

    def _arm_playback_timer(self) -> None:
        """Arm playback from the current policy after all renderer gates clear."""
        viz = self.visualizer
        if not getattr(viz, "animation_running", False) or self._buffering:
            return
        mode = self._playback_controls.playback_mode()
        interval_ms = self.playback_timer_interval_ms()
        if mode is PlaybackMode.REAL_TIME:
            self._real_time_next_deadline_s = self._clock() + interval_ms / 1000.0
            self._fixed_next_deadline_s = None
        elif mode is PlaybackMode.FIXED_FPS:
            self._real_time_next_deadline_s = None
            period_s = 1.0 / float(max(1, self._playback_controls.fixed_playback_fps()))
            self._fixed_next_deadline_s = self._clock() + period_s
        else:
            self._real_time_next_deadline_s = None
            self._fixed_next_deadline_s = None
        viz.animation_timer.start(interval_ms)
        logger.debug(
            "Playback timer: mode=%s interval=%sms",
            mode.value,
            interval_ms,
        )

    def _schedule_next_playback_tick(self, mode: PlaybackMode) -> None:
        """Arm the next serial tick after cadence and renderer permits agree.

        A separately pumped backend may retain the callback until it has
        serviced one native render-loop turn. The cadence calculation runs
        only after that permit arrives, so Fixed FPS and Real time wait for the
        later of their deadline and backend readiness. Backends that decline
        the hook retain the immediate behavior used by pygfx.
        """
        viz = self.visualizer
        generation = self._playback_schedule_generation
        released = False

        if self._renderer_turn_pending:
            logger.debug("Playback tick already has an outstanding renderer-turn permit")
            return

        def _after_renderer_turn() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._renderer_turn_pending = False
            if self._restart_after_renderer_turn:
                self._restart_after_renderer_turn = False
                if getattr(viz, "animation_running", False) and not self._buffering:
                    self._arm_playback_timer()
                return
            if (
                generation != self._playback_schedule_generation
                or not getattr(viz, "animation_running", False)
                or self._buffering
                or self._playback_controls.playback_mode() is not mode
            ):
                return
            if mode is PlaybackMode.REAL_TIME:
                self._schedule_next_real_time_tick()
            elif mode is PlaybackMode.FIXED_FPS:
                self._schedule_next_fixed_tick()
            else:
                viz.animation_timer.start(0)

        renderer = getattr(viz, "renderer", None)
        defer = getattr(renderer, "defer_until_next_render_turn", None)
        if callable(defer):
            accepted = False
            self._renderer_turn_pending = True
            try:
                accepted = bool(defer(_after_renderer_turn))
                if accepted:
                    return
            except (RuntimeError, AttributeError, TypeError) as exc:
                logger.debug("Renderer playback permit failed: %s", exc)
            finally:
                if not accepted:
                    self._renderer_turn_pending = False
        _after_renderer_turn()

    def _schedule_next_fixed_tick(self) -> None:
        """Rearm Fixed FPS against an absolute, non-rounded deadline."""
        viz = self.visualizer
        if not getattr(viz, "animation_running", False) or self._buffering:
            self._fixed_next_deadline_s = None
            return

        period_s = 1.0 / float(max(1, self._playback_controls.fixed_playback_fps()))
        now_s = self._clock()
        prior_deadline_s = self._fixed_next_deadline_s
        if prior_deadline_s is None:
            deadline_s = now_s + period_s
        else:
            deadline_s = prior_deadline_s + period_s
            # Do not replay a long backlog after a breakpoint, stall, or
            # buffering pause. A single immediate tick is enough to resume.
            if deadline_s < now_s - period_s:
                deadline_s = now_s
        self._fixed_next_deadline_s = deadline_s
        delay_ms = max(0, round((deadline_s - now_s) * 1000.0))
        viz.animation_timer.start(delay_ms)
        logger.debug(
            "Fixed playback deadline: period=%.3fms remaining=%sms",
            period_s * 1000.0,
            delay_ms,
        )

    def _schedule_next_real_time_tick(self) -> None:
        """Schedule the next scenario-time deadline without adding frame work."""
        viz = self.visualizer
        if not getattr(viz, "animation_running", False) or self._buffering:
            self._real_time_next_deadline_s = None
            return

        interval_ms = self._real_time_timer_interval_ms()
        interval_s = interval_ms / 1000.0
        now_s = self._clock()
        prior_deadline_s = self._real_time_next_deadline_s
        if prior_deadline_s is None:
            deadline_s = now_s + interval_s
        else:
            deadline_s = prior_deadline_s + interval_s
            # Preserve one immediate catch-up opportunity, but do not replay
            # every missed scenario deadline after a breakpoint, modal dialog,
            # or other long event-loop stall.
            if deadline_s <= now_s - interval_s:
                deadline_s = now_s
        self._real_time_next_deadline_s = deadline_s
        delay_ms = max(0, round((deadline_s - now_s) * 1000.0))
        viz.animation_timer.start(delay_ms)
        logger.debug(
            "Real-time playback deadline: interval=%sms remaining=%sms",
            interval_ms,
            delay_ms,
        )

    def reset_playback_cadence(self) -> None:
        """Discard stale completed-frame samples before a new playback regime."""
        tracker = getattr(self.visualizer, "playback_cadence", None)
        if tracker is not None:
            tracker.reset()

    def shutdown(self) -> None:
        """Invalidate queued playback work before renderer/application teardown."""
        viz = self.visualizer
        timer = getattr(viz, "animation_timer", None)
        if timer is not None:
            timer.stop()
        self._playback_schedule_generation += 1
        self._renderer_turn_pending = False
        self._restart_after_renderer_turn = False
        self._real_time_next_deadline_s = None
        self._fixed_next_deadline_s = None
        self._playback_tick_step = None
        self._playback_tick_recorded = False
        setattr(viz, "animation_running", False)
        if self._buffering:
            self._stop_buffering()
        elif self._buffer_check_timer is not None:
            self._buffer_check_timer.stop()

    def _apply_playback_frame(self, step: int) -> bool:
        """Apply one timer-driven frame and report pipeline acceptance."""
        self._playback_tick_step = int(step)
        self._playback_tick_recorded = False
        try:
            result = self.visualizer.update_frame(step)
            return result is not False
        finally:
            self._playback_tick_step = None

    def _stop_after_failed_playback_frame(self, step: int) -> None:
        """Stop serial playback when a frame transaction was not accepted."""
        viz = self.visualizer
        viz.animation_timer.stop()
        self._playback_schedule_generation += 1
        self._restart_after_renderer_turn = False
        self._real_time_next_deadline_s = None
        self._fixed_next_deadline_s = None
        viz.animation_running = False
        self._update_play_button_states()
        logger.error("Playback stopped: frame %s was not accepted by the renderer", step)
        if hasattr(viz, "_set_status_message"):
            viz._set_status_message(f"Playback stopped: frame {step + 1} was not rendered")

    def record_completed_playback_tick(self, step: int) -> bool:
        """Record one successful pipeline completion for the active timer tick."""
        if (
            not getattr(self.visualizer, "animation_running", False)
            or self._playback_tick_step != int(step)
            or self._playback_tick_recorded
        ):
            return False
        tracker = getattr(self.visualizer, "playback_cadence", None)
        if tracker is None:
            return False
        tracker.record_completion()
        self._playback_tick_recorded = True
        return True

    # Animation button workflows
    def _cancel_pending_slider_scrub(self) -> None:
        """Discard delayed timeline intent before a competing navigation action."""
        cancel = getattr(self.visualizer, "cancel_pending_slider_scrub", None)
        if callable(cancel):
            cancel()

    def handle_slider_commit(self, value: int) -> None:
        """Apply a coalesced display index to the corresponding frame step."""
        viz = self.visualizer
        logger.debug("Committing slider scrub to frame %s", value)
        step = viz.resolve_animation_step(value)

        if self._playback_controls.is_online_mode():
            self._playback_controls.request_frame_if_needed(step)

        viz.update_frame(step)

    def toggle_animation(self, direction: Optional[int] = None) -> None:
        """Toggle playback, preserving direction changes as lightweight updates."""
        self._cancel_pending_slider_scrub()
        viz = self.visualizer
        requested_direction = viz.play_direction if direction is None else direction
        if viz.animation_running:
            if direction is not None and requested_direction != viz.play_direction:
                viz.play_direction = requested_direction
                self.restart_playback_timer()
                self._update_play_button_states()
                logger.debug(
                    "Direction switched to %s",
                    "backward" if viz.play_direction < 0 else "forward",
                )
                return
            viz.animation_timer.stop()
            self._playback_schedule_generation += 1
            self._restart_after_renderer_turn = False
            self._real_time_next_deadline_s = None
            self._fixed_next_deadline_s = None
            viz.animation_running = False
            if self._buffering:
                self._stop_buffering()
            self._update_play_button_states()
            self._restore_scenario_summary()
            viz.update_timer.start(viz._idle_poll_interval_ms)
        else:
            viz.play_direction = requested_direction
            self.reset_playback_cadence()
            viz.animation_running = True
            self.restart_playback_timer()
            self._update_play_button_states()
            viz.update_timer.start(viz._active_poll_interval_ms)

        self.sync_prefetch_settings()
        viz.ui_controller.refresh_status_telemetry()

    def play_backward(self) -> None:
        """Convenience helper for reverse playback."""
        self.toggle_animation(direction=-1)

    def _restore_scenario_summary(self) -> None:
        """Restore the scenario summary text in the status bar after playback stops."""
        viz = self.visualizer
        telemetry = getattr(getattr(viz, "ui_controller", None), "_telemetry_ctrl", None)
        if telemetry is not None:
            saved = getattr(telemetry, "_scenario_summary_text", "")
            if saved and hasattr(viz, "_set_status_message"):
                viz._set_status_message(saved)

    def reset_animation(self) -> None:
        """Reset playback to the first available frame, including sparse sources."""
        self._cancel_pending_slider_scrub()
        viz = self.visualizer
        if viz.animation_running:
            self.toggle_animation()

        start_step = 0
        frame_source = getattr(viz, "frame_source", None)
        if frame_source and hasattr(frame_source, "list_frames"):
            try:
                frames = frame_source.list_frames()
                if frames:
                    start_step = min(frames)
            except OSError:
                pass

        viz.update_frame(start_step)
        viz.play_direction = 1
        self._update_play_button_states()

    def next_frame(self) -> None:
        """Advance forward by stride over the current sparse-step list."""
        self._cancel_pending_slider_scrub()
        viz = self.visualizer
        stride = self._playback_controls.frame_stride()
        available_steps = viz.get_available_animation_steps()
        if not available_steps:
            self._stop_for_empty_frame_plan()
            return
        current_index = viz.get_animation_step_index(viz.animation_step)
        next_index = current_index + stride
        next_step = viz.animation_step
        logger.debug(
            "next_frame: current=%s, next=%s, total=%s, stride=%s",
            viz.animation_step,
            next_index,
            viz.total_animation_steps,
            stride,
        )
        if next_index >= len(available_steps):
            if self._playback_controls.is_online_mode():
                request_step = viz.animation_step + stride
                if self._playback_controls.request_frame_if_needed(request_step):
                    logger.debug(
                        "Animation: successfully requested frame %s in live gRPC mode",
                        request_step,
                    )
                else:
                    next_step = (
                        available_steps[0]
                        if self._playback_controls.loop_enabled()
                        else viz.animation_step
                    )
            else:
                next_step = (
                    available_steps[0]
                    if self._playback_controls.loop_enabled()
                    else viz.animation_step
                )
        else:
            next_step = available_steps[next_index]
            if self._playback_controls.is_online_mode():
                self._playback_controls.request_frame_if_needed(next_step)

        viz.update_frame(next_step)

    def previous_frame(self) -> None:
        """Advance backward by stride over the current sparse-step list."""
        self._cancel_pending_slider_scrub()
        viz = self.visualizer
        stride = self._playback_controls.frame_stride()
        available_steps = viz.get_available_animation_steps()
        if not available_steps:
            self._stop_for_empty_frame_plan()
            return
        current_index = viz.get_animation_step_index(viz.animation_step)
        next_index = current_index - stride
        if next_index < 0:
            next_step = (
                available_steps[-1]
                if self._playback_controls.loop_enabled()
                else viz.animation_step
            )
        else:
            next_step = available_steps[next_index]
            if self._playback_controls.is_online_mode():
                self._playback_controls.request_frame_if_needed(next_step)

        viz.update_frame(next_step)

    def handle_animation_tick(self) -> None:
        """Advance one playback tick and refresh presentation/performance state."""
        if self._buffering:
            return

        viz = self.visualizer
        mode = self._playback_controls.playback_mode()
        if mode is PlaybackMode.REAL_TIME:
            # Real-time mode is deadline-driven. Stop the repeating Qt timer
            # while this frame is applied, then schedule only the remaining
            # wall-clock time to the next scenario deadline.
            viz.animation_timer.stop()
        stride = self._playback_controls.frame_stride()
        available_steps = viz.get_available_animation_steps()
        if not available_steps:
            self._stop_for_empty_frame_plan()
            return
        current_index = viz.get_animation_step_index(viz.animation_step)
        delta = -stride if viz.play_direction < 0 else stride
        next_index = current_index + delta
        next_step = viz.animation_step
        logger.debug(
            "Animation tick: current=%s, next=%s, total=%s, direction=%s, stride=%s",
            viz.animation_step,
            next_index,
            viz.total_animation_steps,
            "backward" if delta < 0 else "forward",
            stride,
        )

        if delta >= 0 and next_index >= len(available_steps):
            if self._playback_controls.loop_enabled():
                next_step = available_steps[next_index % len(available_steps)]
                logger.debug(
                    "Loop mode: wrapping to frame %s (was at frame %s)",
                    next_step,
                    viz.animation_step,
                )
                if self._playback_controls.is_online_mode():
                    self._playback_controls.request_frame_if_needed(next_step)
            elif self._playback_controls.is_online_mode():
                request_step = viz.animation_step + stride
                if self._playback_controls.request_frame_if_needed(request_step):
                    logger.debug(
                        "Live gRPC mode: requested frame %s (dynamic expansion)",
                        request_step,
                    )
                else:
                    logger.debug("Reached end, stopping animation (loop disabled)")
                    self.toggle_animation()
                    return
            else:
                logger.debug("Reached end, stopping animation (loop disabled)")
                self.toggle_animation()
                return
        elif delta < 0 and next_index < 0:
            if self._playback_controls.loop_enabled():
                next_step = available_steps[next_index % len(available_steps)]
                logger.debug("Loop mode: wrapping to last frame %s", next_step)
                if self._playback_controls.is_online_mode():
                    self._playback_controls.request_frame_if_needed(next_step)
            else:
                logger.debug("Reached beginning, stopping animation (loop disabled)")
                self.toggle_animation()
                return
        else:
            next_step = available_steps[next_index]
            if self._playback_controls.is_online_mode():
                self._playback_controls.request_frame_if_needed(next_step)

        # File-backed playback pauses at cache misses until preload catches up;
        # live gRPC sources own their request/buffer policy.
        if not self._is_online_mode():
            cache_service = getattr(self.animation_service, "cache_service", None)
            if cache_service is not None and not cache_service.has_frame(next_step):
                self._start_buffering(next_step)
                return

        logger.debug("Timer advancing to frame %s", next_step)
        if not self._apply_playback_frame(next_step):
            self._stop_after_failed_playback_frame(next_step)
            return

        viz.ui_controller.update_performance_display()

        if viz.animation_running:
            self._schedule_next_playback_tick(mode)

    def _stop_for_empty_frame_plan(self) -> None:
        """Leave playback idle when the active source currently has no frames."""
        viz = self.visualizer
        timer = getattr(viz, "animation_timer", None)
        if timer is not None:
            timer.stop()
        self._playback_schedule_generation += 1
        self._renderer_turn_pending = False
        self._restart_after_renderer_turn = False
        self._real_time_next_deadline_s = None
        self._fixed_next_deadline_s = None
        setattr(viz, "animation_running", False)
        if hasattr(viz, "play_direction"):
            self._update_play_button_states()
        set_status = getattr(viz, "_set_status_message", None)
        if callable(set_status):
            set_status("No animation frames are available")

    def _update_play_button_states(self) -> None:
        """Sync play/pause button states with the current playback state."""
        viz = self.visualizer
        playing_forward = viz.animation_running and viz.play_direction >= 0
        playing_backward = viz.animation_running and viz.play_direction < 0
        if hasattr(viz, "play_btn") and viz.play_btn:
            viz.play_btn.setChecked(playing_forward)
            viz.play_btn.setText("⏸" if playing_forward else "⏵")
        if hasattr(viz, "reverse_play_btn") and viz.reverse_play_btn:
            viz.reverse_play_btn.setChecked(playing_backward)
            viz.reverse_play_btn.setText("⏸" if playing_backward else "⏴")

    def sync_prefetch_settings(self) -> None:
        """Apply the live prefetch controls and current playback direction."""
        try:
            from ..io.frame_sources import LiveGrpcSource
        except ImportError:  # pragma: no cover - optional dependency
            return

        viz = self.visualizer
        frame_source = getattr(viz, "frame_source", None)
        if not isinstance(frame_source, LiveGrpcSource):
            return

        provider = frame_source.provider
        if not provider:
            return

        enabled = self._playback_controls.prefetch_enabled(playing=bool(viz.animation_running))
        if hasattr(provider, "set_play_direction"):
            provider.set_play_direction(viz.play_direction)
        if hasattr(provider, "set_prefetch_enabled"):
            provider.set_prefetch_enabled(enabled)
