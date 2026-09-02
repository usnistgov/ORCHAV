"""Frame context, performance display, and source-summary telemetry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from ..playback import format_playback_fps

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = logging.getLogger(__name__)


class TelemetryController:
    """Synchronize lightweight telemetry widgets with app/runtime state.

    This controller formats status-bar text and frame tooltips. It reads cache,
    renderer, frame-source, and ViewModel state, but it does not mutate frame
    data or control renderer timing.
    """

    def __init__(self, parent: Any) -> None:
        """Store the parent controller and reset cached scenario summary text."""
        self._parent = parent
        self._scenario_summary_text: str = ""

    @property
    def visualizer(self) -> OrchavVisualizer:
        """Shortcut to the parent's visualizer instance."""
        return self._parent.visualizer

    @staticmethod
    def _cache_step_flag(cache_service: Any, method_name: str, step: int) -> bool:
        """Return one optional cache source flag without requiring the full service."""
        predicate = getattr(cache_service, method_name, None)
        if not callable(predicate):
            return False
        try:
            return bool(predicate(step))
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Unable to read cache telemetry flag %s(%s): %s", method_name, step, exc)
            return False

    def update_frame_context(
        self, step: int, raw_frame: Optional[dict] = None, view_model: Optional[Any] = None
    ) -> None:
        """Refresh slider tooltip context for the displayed frame.

        Args:
            step: Zero-based animation step index.
            raw_frame: Optional raw frame dictionary with ``timestamp_ns`` etc.
            view_model: Optional view-model carrying canonical data.
        """
        viz = self.visualizer
        total_steps = getattr(viz, "total_animation_steps", 0)
        display_index = step + 1
        if total_steps > 0:
            frame_summary = f"Frame {display_index} / {total_steps}"
        else:
            frame_summary = f"Frame {display_index}"

        if getattr(viz, "total_steps_label", None) and total_steps > 0:
            viz.total_steps_label.setText(f"/ {total_steps}")

        tooltip_lines: List[str] = [frame_summary]

        if raw_frame:
            timestamp = raw_frame.get("timestamp_ns", raw_frame.get("timestamp"))
            if timestamp is not None:
                try:
                    timestamp_val = float(timestamp)
                    tooltip_lines.append(f"Timestamp: {timestamp_val:.0f} ns")
                except (TypeError, ValueError):
                    tooltip_lines.append(f"Timestamp: {timestamp}")

        total_paths: Optional[int] = None
        if view_model and getattr(view_model, "canonical_data", None) is not None:
            canon = view_model.canonical_data
            try:
                if hasattr(canon, "lines") and canon.lines is not None:
                    total_paths = int(len(canon.lines))
            except (ValueError, TypeError, IndexError):
                total_paths = None

        if total_paths is None and raw_frame:
            paths_obj = raw_frame.get("paths")
            if hasattr(paths_obj, "total_paths"):
                try:
                    total_paths = int(paths_obj.total_paths)
                except (ValueError, TypeError):
                    total_paths = None

        if total_paths is not None:
            tooltip_lines.append(f"MPC paths: {total_paths}")

        source_flags: List[str] = []
        cache_service = getattr(viz, "cache_service", None)
        # Source flags are presentation hints only; CacheService remains the
        # authority for whether a frame came from live override, preload, or
        # ordinary frame cache.
        if cache_service is not None:
            if self._cache_step_flag(cache_service, "is_override", step):
                source_flags.append("source: override")
            elif getattr(viz, "use_preload_mode", False) and self._cache_step_flag(
                cache_service,
                "is_preloaded",
                step,
            ):
                source_flags.append("source: preloaded")
            elif self._cache_step_flag(cache_service, "has_frame", step):
                source_flags.append("source: cached")

        duration_note: Optional[str] = None
        if getattr(viz, "last_frame_duration_ms", None) is not None:
            duration_note = f"{viz.last_frame_duration_ms:.1f} ms"

        combined_notes = list(source_flags)
        if duration_note:
            combined_notes.append(duration_note)

        if combined_notes:
            tooltip_lines.append(", ".join(combined_notes))

        viz._last_frame_tooltip_context = {
            "step": step,
            "lines": list(tooltip_lines),
            "source_flags": list(source_flags),
            "duration_note": duration_note,
        }

        if hasattr(viz, "step_slider") and viz.step_slider is not None:
            viz.step_slider.setToolTip("\n".join(tooltip_lines))

    def refresh_status_telemetry(self) -> None:
        """Update backend-neutral playback-update cadence in the status bar."""
        viz = self.visualizer
        fps_label = getattr(viz, "status_fps_label", None)
        if fps_label is None:
            return

        tracker = getattr(viz, "playback_cadence", None)
        fps_value = tracker.frames_per_second() if tracker is not None else None
        fps_text = format_playback_fps(fps_value)
        if getattr(viz, "animation_running", False):
            if fps_text is None:
                fps_label.setText("Playback updates: measuring")
            else:
                fps_label.setText(f"Playback updates: {fps_text}/s")
            fps_label.setToolTip(
                "Completed scenario-frame pipelines per wall-clock second. This is an "
                "application update rate, not renderer callback or display FPS."
            )
        elif fps_text is not None:
            fps_label.setText(f"Playback updates: paused (last {fps_text}/s)")
            fps_label.setToolTip(
                "Last completed scenario-frame pipeline rate before playback paused; "
                "this is not display FPS."
            )
        else:
            fps_label.setText("Playback updates: paused")
            fps_label.setToolTip(
                "Start playback to measure completed scenario-frame pipelines per second."
            )

    def update_performance_display(self) -> None:
        """Recompute rolling performance stats and refresh the telemetry label."""
        viz = self.visualizer
        frame_times = getattr(viz, "frame_times", None)
        if frame_times:
            avg_time_sec = sum(frame_times) / len(frame_times)
            viz._last_avg_frame_ms = float(avg_time_sec * 1000.0)
        else:
            viz._last_avg_frame_ms = None

        self.refresh_status_telemetry()

    def update_file_source_summary(self) -> None:
        """Display concise scenario and frame-source context in the status bar."""
        from ..io.frame_sources import FileSource, LiveGrpcSource, RemoteHdf5Source

        viz = self.visualizer
        manager = getattr(viz, "ui_manager", None)
        source_panel = (
            getattr(manager, "panels", {}).get("data_source") if manager is not None else None
        )
        refresh_source = getattr(source_panel, "refresh_source_status", None)
        if callable(refresh_source):
            refresh_source()

        status_label = getattr(viz, "status_scenario_label", None)
        if status_label is None:
            return

        label_text = ""
        tooltip_text = ""

        try:
            scenario = getattr(viz, "scenario_config", None)
            scene_name = None
            scene_source = None
            if scenario is not None:
                scene_name = getattr(scenario, "scene_id", None)
                scene_source = getattr(scenario, "scene_source", None)

                scene_spec = getattr(scenario, "scene_spec", {})
                if not scene_name and isinstance(scene_spec, dict):
                    scene_name = scene_spec.get("id")
                if not scene_source and isinstance(scene_spec, dict):
                    scene_source = scene_spec.get("source")

            if scene_name and scene_source:
                scene_context = f"{scene_name} ({scene_source})"
            else:
                scene_context = scene_name
            frame_source = getattr(viz, "frame_source", None)

            # The status label is intentionally brief; the tooltip carries the
            # endpoint/root path so the status bar can stay stable while users
            # switch between file, live gRPC, remote HDF5, and custom sources.
            if isinstance(frame_source, LiveGrpcSource):
                fmt_tag = "Online"
                frame_info = "streaming"
                endpoint = getattr(frame_source, "endpoint", "")
                if endpoint:
                    tooltip_text = str(endpoint)
                if scene_context:
                    label_text = f"{scene_context} \u00b7 {fmt_tag} \u00b7 {frame_info}"
                else:
                    label_text = f"{fmt_tag} \u00b7 {frame_info}"
            elif isinstance(frame_source, FileSource):
                root_path = Path(frame_source.root)
                tooltip_text = str(root_path)
                fmt_tag = (frame_source.fmt or "h5").upper()
                if fmt_tag in ("H5", "HDF5"):
                    fmt_tag = "HDF5"

                try:
                    frames = frame_source.list_frames()
                except OSError as exc:
                    logger.debug("Unable to list frames for status summary: %s", exc)
                    frames = []

                frame_count = len(frames) if frames else 0
                frame_info = f"{frame_count} frames" if frame_count else "0 frames"

                if not scene_context:
                    scene_context = root_path.name
                label_text = f"{scene_context} \u00b7 {fmt_tag} \u00b7 {frame_info}"
            elif isinstance(frame_source, RemoteHdf5Source):
                fmt_tag = "Remote"
                try:
                    frames = frame_source.list_frames()
                except OSError:
                    frames = []
                frame_count = len(frames) if frames else 0
                frame_info = f"{frame_count} frames" if frame_count else "0 frames"
                if not scene_context:
                    scene_context = "remote"
                label_text = f"{scene_context} \u00b7 {fmt_tag} \u00b7 {frame_info}"
            elif frame_source is not None:
                fmt_tag = str(
                    getattr(frame_source, "frame_source_label", type(frame_source).__name__)
                )
                list_frames = getattr(frame_source, "list_frames", None)
                if callable(list_frames):
                    try:
                        frames = list_frames()
                    except OSError:
                        frames = []
                    frame_count = len(frames) if frames else 0
                    frame_info = f"{frame_count} frames" if frame_count else "0 frames"
                else:
                    frame_info = "custom source"

                source_root = getattr(frame_source, "scenario_root", None) or getattr(
                    frame_source, "root", None
                )
                if source_root:
                    tooltip_text = str(source_root)
                if not scene_context:
                    scene_context = fmt_tag.lower()
                label_text = f"{scene_context} \u00b7 {fmt_tag} \u00b7 {frame_info}"
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            logger.debug("Failed to build file source summary: %s", exc)
            label_text = ""
            tooltip_text = ""

        self._scenario_summary_text = label_text
        status_label.setText(label_text)
        status_label.setToolTip(tooltip_text)

    def handle_frame_timing_update(self, step: int, elapsed_sec: float) -> None:
        """Update rolling averages and frame tooltips when timings arrive.

        Args:
            step: Zero-based animation step that just completed.
            elapsed_sec: Wall-clock seconds spent rendering this frame.
        """
        viz = self.visualizer

        frame_times = getattr(viz, "frame_times", None)
        if frame_times:
            window = frame_times[-min(len(frame_times), 60) :]
            if window:
                avg = sum(window) / len(window)
                viz._last_avg_frame_ms = avg * 1000.0
            else:
                viz._last_avg_frame_ms = None
        else:
            viz._last_avg_frame_ms = None

        last_frame_ms = elapsed_sec * 1000.0
        viz.last_frame_duration_ms = last_frame_ms
        animation_controller = getattr(viz, "animation_controller", None)
        record_playback_tick = getattr(
            animation_controller,
            "record_completed_playback_tick",
            None,
        )
        if callable(record_playback_tick):
            record_playback_tick(step)
        self.refresh_status_telemetry()

        context = getattr(viz, "_last_frame_tooltip_context", None)
        duration_text = f"{last_frame_ms:.1f} ms"
        if context and context.get("step") == step:
            source_flags = list(context.get("source_flags", []))
            previous_duration = context.get("duration_note")
            previous_combined = list(source_flags)
            if previous_duration:
                previous_combined.append(previous_duration)

            lines = list(context.get("lines", []))
            if previous_combined:
                prev_line = ", ".join(previous_combined)
                if lines and lines[-1] == prev_line:
                    lines.pop()

            combined = list(source_flags)
            combined.append(duration_text)
            if combined:
                lines.append(", ".join(combined))

            context["lines"] = lines
            context["duration_note"] = duration_text
            viz._last_frame_tooltip_context = context

            if hasattr(viz, "step_slider") and viz.step_slider is not None:
                viz.step_slider.setToolTip("\n".join(lines))
