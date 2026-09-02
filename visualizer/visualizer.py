#!/usr/bin/env python3
"""Qt application entry module for the ORCHAV visualizer.

This module provides the public CLI hand-off and the ``OrchavVisualizer``
``QMainWindow`` composition shell used by controllers and panels. Application
composition lives under ``visualizer.src.app``: startup workflow
parses CLI arguments, state bootstrap initializes runtime fields, services
construct the service graph, scenario workflow owns load/cleanup, and renderer
lifecycle boots the selected backend.

Per-frame data flows through ``FramePipeline`` and ``ViewModel`` before being
applied through the renderer-neutral protocol in ``visualizer.src.renderers``.
Keep this module import-light and add behavior to the owning service or
controller instead of growing the shell with new compatibility delegates.
"""

import json
import platform
import re
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path

from shared.logging import configure_logging, get_logger

if __name__ == "__main__" and __package__ is None:
    raise SystemExit("Run with `python -m visualizer` from the repo root.")

# Configure logging early so module-level logging calls are handled
configure_logging()
logger = get_logger(__name__)

DEFAULT_ANIMATION_CACHE_SIZE = 32


def _open_export_location(output_path: str) -> None:
    """Open the exported file's location without invoking a command shell."""
    if platform.system() == "Darwin":
        subprocess.Popen(["open", "-R", output_path])
    elif platform.system() == "Windows":
        subprocess.Popen(["explorer.exe", f"/select,{output_path}"])
    else:
        subprocess.Popen(["xdg-open", str(Path(output_path).parent)])


from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QProgressDialog,
)


class ProgressReporter:
    """Minimal CLI progress helper."""

    def __init__(self, enabled: bool = False, stream=None):
        """Configure whether progress messages are emitted and where they go."""
        self.enabled = enabled
        self.stream = stream or sys.stdout

    class _Task:
        """Context manager that logs one progress task boundary."""

        def __init__(self, reporter, description: str):
            """Store the reporter and human-readable task label."""
            self.reporter = reporter
            self.description = description

        def __enter__(self):
            """Log task start when progress reporting is enabled."""
            if self.reporter.enabled:
                logger.debug("Starting: %s", self.description)
            return self

        def __exit__(self, exc_type, exc, tb):
            """Log task completion or failure and propagate exceptions."""
            if not self.reporter.enabled:
                return False
            if exc_type is None:
                logger.debug("Done: %s", self.description)
            else:
                logger.error("Failed: %s: %s", self.description, exc)
            # propagate exceptions
            return False

    def task(self, description: str):
        """Return a task context for one progress-reporting scope."""
        return ProgressReporter._Task(self, description)

    def note(self, message: str):
        """Emit a progress note when reporting is enabled."""
        if self.enabled:
            logger.debug("%s", message)


from .src.app.composition import finalize_deferred_setup
from .src.app.lifecycle import shutdown_visualizer
from .src.app.scenario_workflow import (
    ScenarioOpenOutcome,
    ScenarioOpenStatus,
)
from .src.app.scenario_workflow import (
    open_scenario as app_open_scenario,
)
from .src.app.services import construct_services
from .src.app.startup_ui import (
    apply_visualizer_shell,
    build_main_panels,
    install_loading_placeholder,
    make_loading_updater,
)
from .src.app.startup_workflow import run_visualizer_cli
from .src.app.state_bootstrap import initialize_runtime_state, set_minimal_startup_flags
from .src.renderers.protocol import renderer_capabilities
from .src.services.cache_service import CacheInvalidationScope
from .src.state import AppState, update_state

# Target 60 FPS for smooth UI updates while preventing excessive rendering
UPDATE_DEBOUNCE_MS = 16  # ~60 FPS (1000ms / 60fps ≈ 16.67ms)
MAX_FORCED_FRAME_RETRIES = 24
MAX_FORCED_FRAME_RETRY_DELAY_MS = 250


def _current_canonical_scenario_root(visualizer: Any) -> Path | None:
    """Return the active scenario root when it owns ``scenario.yaml``."""

    current = getattr(visualizer, "current_scenario_path", None)
    if not current:
        return None
    candidate = Path(current).expanduser()
    scenario_root = candidate.parent if candidate.suffix.lower() in {".yaml", ".yml"} else candidate
    return scenario_root if (scenario_root / "scenario.yaml").is_file() else None


class OrchavVisualizer(QMainWindow):
    """Main Qt window coordinating controllers/services for the visualizer.

    The class exposes a thin API for Qt widgets, while delegating real work to:
      * Services (scene, node, animation, cache, overrides, appearance, etc.)
      * Controllers (UI, animation, camera, main, selection manager)
    Keeping those responsibilities split encourages incremental refactors while still
    giving contributors a single place to inspect when learning the application.
    """

    def __init__(
        self,
        progress: Optional[ProgressReporter] = None,
        renderer_type: str = "pygfx",
        layout_profile: str = "auto",
        viewport_mode: str = "auto",
    ):
        """Create the shell window and defer heavy service/renderer setup."""
        super().__init__()

        self._layout_profile = layout_profile
        resolved_viewport_mode = str(viewport_mode)
        if resolved_viewport_mode == "auto":
            detached_host_required = renderer_type == "open3d" or layout_profile in {
                "capture-renderer",
                "capture-workspace",
            }
            resolved_viewport_mode = "detached" if detached_host_required else "embedded"
        if resolved_viewport_mode not in {"embedded", "detached"}:
            raise ValueError(
                f"Unknown viewport mode {viewport_mode!r}; expected auto, embedded, or detached"
            )
        self._viewport_mode = resolved_viewport_mode
        self._last_scenario_attempt: str | None = None
        self._last_scenario_data_mode_override: str | None = None
        self._last_scenario_grpc_port_override: int | None = None
        apply_visualizer_shell(
            self,
            layout_profile=layout_profile,
            viewport_mode=resolved_viewport_mode,
        )

        self.progress = progress if progress is not None else ProgressReporter(enabled=False)
        self.progress.note("Initializing ORCHAV")
        self._renderer_type = renderer_type

        set_minimal_startup_flags(self)

        install_loading_placeholder(self)

    # Deferred initialization: heavy setup performed after show()

    def _deferred_init(self, pending_camera: Optional[dict] = None) -> None:
        """Complete all heavy initialization (services, panels, timers).

        Called *after* ``show()`` so the user sees the loading placeholder
        immediately while services and UI panels are being created.
        """
        self._pending_camera = pending_camera

        update_loading = make_loading_updater(self)

        def ensure_startup_active() -> None:
            """Abort staged initialization after a nested Qt close event."""

            if bool(getattr(self, "_shutdown_started", False)):
                raise RuntimeError("Visualizer initialization cancelled during shutdown")

        update_loading("Initializing core state...")
        initialize_runtime_state(self, default_animation_cache_size=DEFAULT_ANIMATION_CACHE_SIZE)
        ensure_startup_active()

        update_loading("Creating services...")
        construct_services(self, default_animation_cache_size=DEFAULT_ANIMATION_CACHE_SIZE)
        ensure_startup_active()

        update_loading("Building UI panels...")
        finalize_deferred_setup(self)
        ensure_startup_active()

        from .src.authoring.feature import scenario_builder_enabled

        if scenario_builder_enabled() and renderer_capabilities(self.renderer).scenario_authoring:
            from .src.authoring.generation_controller import QtGenerationController
            from .src.authoring.mode_controller import WorkspaceModeController

            self.workspace_mode_controller = WorkspaceModeController(self)
            self.authoring_generation_controller = QtGenerationController(
                save_callback=self.workspace_mode_controller.save,
                workspace_provider=lambda: self.workspace_mode_controller.workspace,
                parent=self,
            )
            self._refresh_authoring_actions()

    def set_state(self, **changes):
        """Update the authoritative AppState snapshot using state helpers."""
        previous_state = self.app_state
        self.app_state = update_state(self.app_state, **changes)
        logger.debug(f"AppState updated: {changes}")

        ui_manager = getattr(self, "ui_manager", None)
        refresh_context = getattr(ui_manager, "refresh_global_context", None)
        if callable(refresh_context):
            refresh_context(self.app_state)

        prev_fly_mode = bool(getattr(previous_state, "fly_mode", False))
        fly_mode = bool(getattr(self.app_state, "fly_mode", False))
        if prev_fly_mode != fly_mode:
            widget = getattr(self, "fly_mode_cb", None)
            if widget is not None and bool(widget.isChecked()) != fly_mode:
                widget.blockSignals(True)
                widget.setChecked(fly_mode)
                widget.blockSignals(False)
            renderer = getattr(self, "renderer", None)
            if renderer is not None and renderer_capabilities(renderer).fly_mode:
                try:
                    renderer.set_fly_mode(fly_mode)
                except (AttributeError, RuntimeError, ValueError):
                    logger.debug("Could not sync fly mode to renderer", exc_info=True)

        if "show_camera_minimap" in changes:
            minimap_visible = bool(self.app_state.show_camera_minimap)
            widget = getattr(self, "camera_minimap_cb", None)
            if widget is not None and bool(widget.isChecked()) != minimap_visible:
                widget.blockSignals(True)
                widget.setChecked(minimap_visible)
                widget.blockSignals(False)
            renderer = getattr(self, "renderer", None)
            if renderer is not None and renderer_capabilities(renderer).camera_minimap:
                try:
                    renderer.set_camera_minimap_visible(minimap_visible)
                except (AttributeError, RuntimeError, ValueError):
                    logger.debug("Could not sync minimap state to renderer", exc_info=True)

        if hasattr(self, "beam_azimuth_spin") and self.beam_azimuth_spin is not None:
            self.beamforming_ui_controller.update_resolution_controls()

    def schedule_update(self):
        """Coalesce rapid UI changes into one delayed render request."""
        logger.debug(f"Schedule update: update_pending={self.update_pending}")
        if not self.update_pending:
            self.update_pending = True
            self.update_timer_coalesce.start(UPDATE_DEBOUNCE_MS)
            logger.debug("Schedule update: Started debounced timer")
        else:
            logger.debug("Schedule update: Update already pending")

    def reset_frame_retry_state(self) -> None:
        """Invalidate pending forced-frame retries after success or scenario change."""

        self._frame_retry_token = int(getattr(self, "_frame_retry_token", 0)) + 1
        self._frame_retry_count = 0
        self._frame_retry_pending = False

    def schedule_frame_retry(self, reason: str) -> bool:
        """Schedule a bounded, backed-off retry for transient frame failures."""

        if bool(getattr(self, "_shutdown_started", False)):
            return False
        if bool(getattr(self, "_frame_retry_pending", False)):
            return True
        count = int(getattr(self, "_frame_retry_count", 0)) + 1
        self._frame_retry_count = count
        if count > MAX_FORCED_FRAME_RETRIES:
            self.force_update_next_frame = False
            self._frame_retry_pending = False
            message = f"Initial frame could not be displayed: {reason}"
            self._set_status_message(message)
            host = getattr(self, "_viewport_host", None)
            if host is not None and self._viewport_mode == "embedded":
                from .src.app.viewport_workspace import ViewportState

                host.set_state(ViewportState.ERROR, message)
            logger.error("Frame retry limit reached after %d attempts: %s", count - 1, reason)
            return False

        token = int(getattr(self, "_frame_retry_token", 0))
        self._frame_retry_pending = True
        delay_ms = min(
            UPDATE_DEBOUNCE_MS * (2 ** min(count - 1, 4)),
            MAX_FORCED_FRAME_RETRY_DELAY_MS,
        )

        def _retry() -> None:
            if token != int(getattr(self, "_frame_retry_token", 0)):
                return
            if bool(getattr(self, "_shutdown_started", False)):
                self._frame_retry_pending = False
                return
            self.force_update_next_frame = True
            self.schedule_update()

        QTimer.singleShot(delay_ms, _retry)
        return True

    def cancel_scheduled_update(self) -> None:
        """Drop any coalesced pipeline update that has not run yet."""
        self.update_pending = False
        timer = getattr(self, "update_timer_coalesce", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                logger.debug("Could not stop coalesced update timer", exc_info=True)

    def set_background_update_enabled(self, enabled: bool) -> None:
        """Enable or disable the lightweight background update timer."""
        timer = getattr(self, "update_timer", None)
        if timer is None:
            return
        if enabled:
            try:
                if not timer.isActive():
                    timer.start(self._idle_poll_interval_ms)
            except RuntimeError:
                logger.debug("Could not start background update timer", exc_info=True)
        else:
            try:
                timer.stop()
            except RuntimeError:
                logger.debug("Could not stop background update timer", exc_info=True)

    def _flush_update(self):
        """Process the pending update through the unified pipeline"""
        self.update_pending = False

        # Scene-only scenarios may still own a coverage layer even though no
        # FrameSource will ever become ready. Let those updates reach the
        # pipeline's coverage-only transaction.
        if not self.ready and not bool(getattr(self, "_scene_only_mode", False)):
            logger.debug("Flush update: FrameSource not ready yet, skipping update")
            self._frame_retry_pending = False
            return

        logger.debug("Flush update: processing")
        if self.vis_initialized and hasattr(self, "app_state"):
            logger.debug("Flush update: Calling unified pipeline step")
            self._frame_retry_pending = False
            self._process_frame_step(self.animation_step)
        else:
            logger.warning("Flush update: Skipping update - conditions not met")

    def reset_startup_timing_profile(self) -> None:
        """Clear startup-stage timings for a new scenario load."""
        self.startup_stage_timings_ms = OrderedDict()
        self.startup_first_frame_timings_ms = {}
        self.startup_detail_timings_ms = OrderedDict()

    def record_startup_stage_timing(self, name: str, elapsed_ms: float) -> None:
        """Record a named startup stage timing in milliseconds."""
        self.startup_stage_timings_ms[str(name)] = float(elapsed_ms)

    def set_startup_first_frame_timing(self, timings: dict[str, float]) -> None:
        """Store the first completed frame breakdown captured during scene boot."""
        self.startup_first_frame_timings_ms = {
            str(key): float(value) for key, value in timings.items()
        }

    def set_startup_detail_timing(self, group_name: str, timings: dict[str, float]) -> None:
        """Store a nested startup timing breakdown under a named group."""
        self.startup_detail_timings_ms[str(group_name)] = {
            str(key): float(value) for key, value in timings.items()
        }

    def get_startup_timing_profile(self) -> dict[str, dict[str, float]]:
        """Return rounded startup timing breakdown for benchmark export."""
        profile: dict[str, dict[str, float]] = {}
        if self.startup_stage_timings_ms:
            profile["scenario_open_stages_ms"] = {
                key: round(float(value), 3) for key, value in self.startup_stage_timings_ms.items()
            }
        if self.startup_first_frame_timings_ms:
            profile["first_frame_pipeline_ms"] = {
                key: round(float(value), 3)
                for key, value in self.startup_first_frame_timings_ms.items()
            }
        for group_name, timings in self.startup_detail_timings_ms.items():
            profile[str(group_name)] = {
                key: round(float(value), 3) for key, value in timings.items()
            }
        return profile

    def _process_frame_step(self, step: int) -> bool:
        """Cache and submit one frame, returning backend acceptance."""
        bench = getattr(getattr(self, "pipeline", None), "benchmark_recorder", None)
        if hasattr(self, "animation_controller"):
            # Keep the cache warm before the FramePipeline tries to load the same frame.
            t_prepare_start = time.perf_counter()
            self.animation_controller.prepare_step(step)
            if bench is not None:
                bench.record_prepare_step((time.perf_counter() - t_prepare_start) * 1000.0)
        return self.pipeline.update(step) is not False

    def _init_ui(self):
        """Build the main Qt panels through the app composition helper."""
        build_main_panels(
            self,
            metrics_available=bool(getattr(self.metrics_service, "available", False)),
        )

    def resizeEvent(self, event):
        """Switch to compact mode when the window is narrow."""
        super().resizeEvent(event)
        if not hasattr(self, "ui_manager") or self.ui_manager is None:
            return
        controls = getattr(self, "ctrl_panel", None)
        controls_width = controls.width() if controls is not None else event.size().width()
        compact = controls_width < 480
        if compact != self._compact_mode:
            self._compact_mode = compact
            for panel in (self.ui_manager.panels.get(k) for k in ("animation", "camera")):
                if panel is not None and hasattr(panel, "set_compact_mode"):
                    panel.set_compact_mode(compact)

    def _on_viewport_resized(self, width: int, height: int) -> None:
        """Keep renderer camera and overlay sizing aligned with the canvas slot."""

        if self._viewport_mode != "embedded" or not self.vis_initialized:
            return
        resize = getattr(self.renderer, "resize", None)
        if callable(resize) and width > 0 and height > 0:
            resize(int(width), int(height))

    def _on_viewport_screen_changed(self, _screen: Any) -> None:
        """Refresh embedded physical sizing after a monitor/DPI transition."""

        host = getattr(self, "_viewport_host", None)
        if host is None:
            return
        size = host.canvas_parent.size()
        self._on_viewport_resized(size.width(), size.height())

    def retry_last_scenario(self) -> bool:
        """Retry the most recent scenario request from the integrated error page."""

        if not self._last_scenario_attempt:
            return False
        outcome = self.open_scenario(
            self._last_scenario_attempt,
            data_mode_override=self._last_scenario_data_mode_override,
            grpc_port_override=self._last_scenario_grpc_port_override,
        )
        return bool(getattr(outcome, "succeeded", False))

    def _setup_keyboard_shortcuts(self):
        """Set up viewport playback and HUD keyboard shortcuts."""
        from .src.app.shortcuts import shortcut

        # Animation control shortcuts
        # Space: Play/Pause toggle
        space_shortcut = QShortcut(shortcut("play_pause").key_sequence(), self)
        if hasattr(space_shortcut, "setContext"):
            space_shortcut.setContext(Qt.WindowShortcut)
        space_shortcut.activated.connect(
            lambda: self._run_viewport_shortcut(self._toggle_play_pause)
        )

        # Left Arrow: Previous frame
        left_shortcut = QShortcut(shortcut("previous_frame").key_sequence(), self)
        if hasattr(left_shortcut, "setContext"):
            left_shortcut.setContext(Qt.WindowShortcut)
        left_shortcut.activated.connect(
            lambda: self._run_viewport_shortcut(self.animation_controller.previous_frame)
        )

        # Right Arrow: Next frame
        right_shortcut = QShortcut(shortcut("next_frame").key_sequence(), self)
        if hasattr(right_shortcut, "setContext"):
            right_shortcut.setContext(Qt.WindowShortcut)
        right_shortcut.activated.connect(
            lambda: self._run_viewport_shortcut(self.animation_controller.next_frame)
        )

        hud_shortcut = QShortcut(shortcut("toggle_hud").key_sequence(), self)
        if hasattr(hud_shortcut, "setContext"):
            hud_shortcut.setContext(Qt.WindowShortcut)
        # HUD visibility remains useful in authoring mode; only playback
        # shortcuts are suppressed by ``_run_viewport_shortcut`` there.
        hud_shortcut.activated.connect(self.ui_controller.toggle_viewport_hud)

        logger.debug(
            "Viewport shortcuts set up "
            "(Space=Play/Pause, Left/Right=Prev/Next frame, Ctrl+H=Toggle HUD)"
        )

    def _run_viewport_shortcut(self, callback: Any) -> None:
        """Run playback shortcuts throughout visualization workspace focus."""

        mode_controller = getattr(self, "workspace_mode_controller", None)
        mode = getattr(mode_controller, "mode", None)
        if getattr(mode, "value", mode) == "authoring":
            return
        callback()

    def _show_help_dialog(self) -> None:
        """Open the keyboard shortcut cheat sheet dialog."""
        from visualizer.src.panels.help_dialog import HelpDialog

        dialog = HelpDialog(self)
        dialog.exec()

    def _toggle_play_pause(self):
        """Toggle play/pause state via keyboard shortcut."""
        self.animation_controller.toggle_animation(direction=1)

        # Sync the play button UI state
        if hasattr(self, "play_btn") and self.play_btn is not None:
            self.play_btn.setChecked(self.animation_running)

    def save_session_dialog(self):
        """Save a named snapshot of the current scenario workspace."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        if not getattr(self, "current_scenario_path", None):
            QMessageBox.information(
                self,
                "No Scenario Open",
                "Open a scenario before saving a workspace snapshot.",
            )
            return

        from pathlib import Path

        scenario_path = Path(self.current_scenario_path)
        scenario_root = (
            scenario_path.parent
            if scenario_path.suffix.lower() in {".yaml", ".yml"}
            else scenario_path
        )
        default_name = f"{scenario_root.name}_workspace"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workspace Snapshot",
            str(self.session_service.session_dir / f"{default_name}.json"),
            "Workspace Snapshots (*.json)",
        )

        if file_path:
            try:
                saved_path = self.session_service.save_session(path=Path(file_path))
                QMessageBox.information(
                    self,
                    "Workspace Snapshot Saved",
                    f"Workspace snapshot saved to:\n{saved_path}",
                )
                self.ui_controller.update_recent_sessions_menu()
                logger.info("Workspace snapshot saved successfully to %s", saved_path)
            except (OSError, IOError, PermissionError, ValueError) as e:
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Failed to save workspace snapshot:\n{e}",
                )
                logger.error("Failed to save workspace snapshot: %s", e, exc_info=True)

    def load_session_dialog(self):
        """Show a file dialog for restoring a workspace snapshot."""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Workspace Snapshot",
            str(self.session_service.session_dir),
            "Workspace Snapshots (*.json)",
        )

        if file_path:
            from pathlib import Path

            self.load_session_file(Path(file_path))

    def load_session_file(self, session_path):
        """Restore workspace state from a snapshot file.

        Args:
            session_path: Path to session JSON file
        """
        from pathlib import Path

        from PySide6.QtWidgets import QMessageBox

        try:
            success = self.session_service.load_session(Path(session_path))
            if success:
                QMessageBox.information(
                    self,
                    "Workspace Restored",
                    f"Workspace restored from:\n{session_path}",
                )
                self.ui_controller.update_recent_sessions_menu()
                logger.info("Workspace restored successfully from %s", session_path)
            else:
                QMessageBox.warning(
                    self,
                    "Restore Failed",
                    f"Failed to restore workspace from:\n{session_path}",
                )
        except (OSError, IOError, ValueError, KeyError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Restore Error", f"Error restoring workspace:\n{e}")
            logger.error("Error restoring workspace from %s: %s", session_path, e, exc_info=True)

    def start_preloading(self):
        """Start preloading all frame data via AnimationService."""
        if not hasattr(self, "animation_service"):
            logger.warning("Animation service unavailable; cannot preload")
            return False

        progress_bar = getattr(self, "status_progress_bar", None)
        _preload_re = re.compile(r"(\d+)\s*/\s*(\d+)")

        def _update_progress(text: str) -> None:
            """Mirror preload progress text into status widgets."""
            if getattr(self, "preload_status_label", None):
                self.preload_status_label.setText(text)
            if progress_bar is not None:
                m = _preload_re.search(text)
                if m:
                    loaded, total = int(m.group(1)), int(m.group(2))
                    progress_bar.setMaximum(total)
                    progress_bar.setValue(loaded)
                    progress_bar.setFormat(f"Loading {loaded}/{total}")
                    if not progress_bar.isVisible():
                        progress_bar.setVisible(True)

        def _on_complete(frames_list: list[tuple[int, dict]], duration: float) -> None:
            """Finalize preload UI and stop ViewModel warmer accounting."""
            logger.info(
                "_on_complete entered: %d frames, preload took %.1fs", len(frames_list), duration
            )
            # End the warming session — all step_ready signals have been
            # delivered by now, so stop() logs the final count and resets.
            self._vm_warmer.stop()

            msg = f"Preload complete: {len(frames_list)} frames in {duration:.1f}s"
            if getattr(self, "preload_status_label", None):
                self.preload_status_label.setText(msg)
            self._set_status_message(msg, 5000)
            if progress_bar is not None:
                progress_bar.setValue(progress_bar.maximum())
                progress_bar.setFormat("Loading complete")
                QTimer.singleShot(2000, lambda: progress_bar.setVisible(False))
            if (
                hasattr(self, "ui_manager")
                and getattr(self.ui_manager, "panels", None)
                and "data_source" in self.ui_manager.panels
            ):
                panel = self.ui_manager.panels["data_source"]
                if panel:
                    panel._update_status()
            self.detect_mpc_frames()

        _update_progress("Preloading frames...")
        return self.animation_service.start_preloading(
            on_progress=_update_progress,
            on_complete=_on_complete,
            on_step_ready=self._vm_warmer.enqueue,
        )

    def request_startup_preload(self, *, delay_ms: int = 1000) -> None:
        """Schedule background preloading after the first completed frame."""
        if not self.use_preload_mode:
            return
        self._startup_preload_requested = True
        self._startup_preload_delay_ms = max(0, int(delay_ms))
        if self._scene_boot_logged:
            self._schedule_startup_preload()

    def cancel_startup_preload(self) -> None:
        """Cancel any delayed startup preload request."""
        self._startup_preload_requested = False
        timer = getattr(self, "_startup_preload_timer", None)
        if timer is None:
            return
        try:
            timer.stop()
        except RuntimeError:
            logger.debug("Could not stop startup preload timer", exc_info=True)

    def _schedule_startup_preload(self) -> None:
        """Arm the delayed preload timer once the first frame is complete."""
        if not self._startup_preload_requested:
            return
        timer = getattr(self, "_startup_preload_timer", None)
        if timer is None:
            return
        try:
            if timer.isActive():
                return
            timer.start(self._startup_preload_delay_ms)
        except RuntimeError:
            logger.debug("Could not start startup preload timer", exc_info=True)

    def _run_startup_preload(self) -> None:
        """Execute the deferred startup preload."""
        if not self._startup_preload_requested:
            return
        self._startup_preload_requested = False
        self.start_preloading()

    def _on_scene_boot_completed(self) -> None:
        """Kick off deferred preload once the scene is actually ready."""
        self._schedule_startup_preload()

    def reset_preloading_state(self):
        """Reset preloading state to allow retrying"""
        self._vm_warmer.stop()
        self.animation_service.reset_preloading_state()

    def detect_mpc_frames(self):
        """Detect the number of available MPC frames and update total_animation_steps"""

        # In live gRPC mode, the server metadata owns the total frame count.
        is_online_mode = False
        if hasattr(self, "ui_manager") and "animation" in self.ui_manager.panels:
            animation_panel = self.ui_manager.panels["animation"]
            is_online_mode = animation_panel.is_in_online_mode()
            logger.debug(
                "detect_mpc_frames: animation_panel found, is_online_mode=%s", is_online_mode
            )
        else:
            logger.debug("detect_mpc_frames: no animation panel found")

        if is_online_mode:
            logger.debug(
                "Live gRPC mode detected - skipping frame count detection, total_steps=%d",
                self.total_animation_steps,
            )
            return

        def on_frame_count_updated(new_total_steps: int):
            """Commit a pending live-mode slider value after frame-count refresh."""
            if new_total_steps != self.total_animation_steps:
                logger.debug(
                    "Detected %d MPC frames (was %d)", new_total_steps, self.total_animation_steps
                )
                self.total_animation_steps = new_total_steps

                if hasattr(self, "ui_manager") and hasattr(self.ui_manager, "panels"):
                    self.ui_manager.update_total_steps(self.total_animation_steps)

                    if hasattr(self, "step_slider"):
                        self.step_slider.setMaximum(self.total_animation_steps - 1)

                    if hasattr(self, "total_steps_label") and self.total_steps_label is not None:
                        self.total_steps_label.setText(f"/ {self.total_animation_steps}")
                self.ui_controller.update_file_source_summary()
            else:
                logger.debug(
                    "MPC frame count (%d) matches current total_animation_steps", new_total_steps
                )

        self.animation_service.detect_mpc_frames(on_frame_count_updated=on_frame_count_updated)

    # Object/label/highlight state helpers

    def update_mpc_visualization(self):
        """Refresh MPC rendering for the current frame through the unified pipeline."""
        if not self.app_state.mpc_visibility.enabled:
            self.mpc_info_label.setText("MPC layer: Disabled")
            return

        if self.force_update_next_frame:
            self.cache_service.invalidate(
                CacheInvalidationScope.MPC_RENDER_SETTINGS,
                reason="force_mpc_update",
            )
            self.force_update_next_frame = False
            # Force immediate update using new pipeline
            if self.vis_initialized:
                self._process_frame_step(self.animation_step)
            return

        if self.vis_initialized:
            self._process_frame_step(self.animation_step)

    def closeEvent(self, event):
        """Persist state and run the shared idempotent application teardown."""
        mode_controller = getattr(self, "workspace_mode_controller", None)
        request_close = getattr(mode_controller, "request_application_close", None)
        if callable(request_close) and not request_close():
            event.ignore()
            return
        try:
            workspace = getattr(self, "_visualization_workspace", None)
            save_settings = getattr(workspace, "save_settings", None)
            if callable(save_settings):
                save_settings()
            generation_controller = getattr(self, "authoring_generation_controller", None)
            stop_generation = getattr(generation_controller, "shutdown", None)
            if callable(stop_generation):
                stop_generation()
            close_modes = getattr(mode_controller, "close", None)
            if callable(close_modes):
                close_modes()
            shutdown_visualizer(self, persist_state=True)
        finally:
            event.accept()

    # Frame updates & animation pipeline

    def _sync_animation_navigation_widgets(self, display_index: int) -> None:
        """Mirror one committed display index into timeline widgets without signals."""
        display_index = max(0, int(display_index))
        step_label = getattr(self, "step_label", None)
        if step_label is not None:
            step_label.setText(str(display_index + 1))

        for widget, value in (
            (getattr(self, "step_slider", None), display_index),
            (getattr(self, "frame_input", None), display_index + 1),
        ):
            if widget is None:
                continue
            signals_were_blocked = widget.blockSignals(True)
            try:
                widget.setValue(value)
            finally:
                widget.blockSignals(signals_were_blocked)

    def _take_pending_slider_scrub(self) -> Optional[int]:
        """Stop the scrub timer and atomically remove its latest pending index."""
        timer = getattr(self, "slider_scrub_timer", None)
        if timer is not None:
            timer.stop()
        pending = getattr(self, "_pending_slider_scrub_index", None)
        self._pending_slider_scrub_index = None
        return None if pending is None else int(pending)

    def flush_pending_slider_scrub(self) -> bool:
        """Commit the latest pending timeline position unless shutdown has begun."""
        display_index = self._take_pending_slider_scrub()
        if display_index is None or bool(getattr(self, "_shutdown_started", False)):
            return False
        controller = getattr(self, "animation_controller", None)
        commit = getattr(controller, "handle_slider_commit", None)
        if not callable(commit):
            logger.debug("Dropping slider scrub because the animation controller is unavailable")
            return False
        commit(display_index)
        return True

    def cancel_pending_slider_scrub(self) -> bool:
        """Discard a queued scrub and restore widgets to the committed frame."""
        display_index = self._take_pending_slider_scrub()
        if display_index is None:
            return False
        if bool(getattr(self, "_shutdown_started", False)):
            return True
        committed_index = self.get_animation_step_index(self.animation_step)
        self._sync_animation_navigation_widgets(committed_index)
        return True

    def update_frame(self, step) -> bool:
        """Synchronize AppState, widgets, and renderer output to one frame.

        Timers, buttons, and slider commits all enter here. Widget updates are
        signal-blocked so a programmatic frame change does not recursively issue
        another frame request.
        """
        # Any authoritative frame source supersedes delayed manual scrub intent.
        # Competing UI actions cancel explicitly before navigation so they can
        # restore widget state; direct live/export/video updates only need this
        # non-restoring clear before publishing their own committed position.
        self._take_pending_slider_scrub()
        logger.debug("Updating to frame %d", step)

        self.animation_step = step
        self.set_state(step=step)

        display_index = self.get_animation_step_index(step)

        self._sync_animation_navigation_widgets(display_index)
        self.ui_controller.update_frame_context(step)

        # The pipeline updates camera after deriving the ViewModel and before
        # submitting geometry in the common renderer transaction.
        frame_completed = False
        if self.vis_initialized:
            frame_completed = self._process_frame_step(step)
        else:
            logger.warning("Visualizer not initialized, skipping frame update")

        if hasattr(self, "ui_manager") and "trajectory" in self.ui_manager.panels:
            trajectory_panel = self.ui_manager.panels.get("trajectory")
            if trajectory_panel and hasattr(trajectory_panel, "set_current_frame"):
                trajectory_panel.set_current_frame(step)

        return frame_completed

    def get_available_animation_steps(self) -> list[int]:
        """Return animation steps in playback order."""
        frame_source = getattr(self, "frame_source", None)
        if frame_source is not None and hasattr(frame_source, "list_frames"):
            try:
                available_frames = frame_source.list_frames()
            except (OSError, ValueError):
                available_frames = []
            if available_frames:
                return list(available_frames)
        total_steps = max(0, int(getattr(self, "total_animation_steps", 0)))
        return list(range(total_steps))

    def resolve_animation_step(self, display_index: int) -> int:
        """Map a 0-based UI frame index to the actual frame step."""
        steps = self.get_available_animation_steps()
        if not steps:
            return max(0, int(display_index))
        clamped = max(0, min(int(display_index), len(steps) - 1))
        return int(steps[clamped])

    def get_animation_step_index(self, step: int) -> int:
        """Map an actual frame step to the 0-based UI/playback index."""
        steps = self.get_available_animation_steps()
        if not steps:
            return max(0, int(step))
        try:
            return steps.index(int(step))
        except ValueError:
            if 0 <= int(step) < len(steps):
                return int(step)
            for idx, candidate in enumerate(steps):
                if candidate >= int(step):
                    return idx
            return len(steps) - 1

    # HUD, status bar, and telemetry helpers

    def _set_status_message(self, text: str, timeout: int = 0) -> None:
        """Show a transient message in the status bar scenario label.

        The message replaces the scenario context text. If *timeout* > 0 the
        original text is restored automatically after *timeout* milliseconds,
        unless a newer status has replaced it.
        """
        label = getattr(self, "status_scenario_label", None)
        if label is None:
            return
        revision = int(getattr(self, "_status_message_revision", 0)) + 1
        self._status_message_revision = revision
        try:
            label.setText(text)
            if timeout > 0:
                telemetry = getattr(self.ui_controller, "_telemetry_ctrl", None)
                saved = telemetry._scenario_summary_text if telemetry else ""

                def restore_if_current() -> None:
                    if getattr(self, "_status_message_revision", 0) != revision:
                        return
                    try:
                        label.setText(saved)
                    except RuntimeError as exc:
                        logger.debug("Failed to restore status bar message: %s", exc)

                QTimer.singleShot(timeout, restore_if_current)
        except RuntimeError as exc:
            logger.debug("Failed to update status bar message: %s", exc)
        self._last_status_message = text

    # Animation controls & timer callbacks

    def update_animation(self):
        """Delegate animation timer tick to AnimationController."""
        self.animation_controller.handle_animation_tick()

    # Scenario lifecycle & environment loading

    def _set_frame_data_available(self, available: bool) -> None:
        """Synchronize UI controls with frame-backed or scene-only mode."""
        ui_manager = getattr(self, "ui_manager", None)
        if ui_manager is not None and hasattr(ui_manager, "set_frame_data_available"):
            ui_manager.set_frame_data_available(bool(available))

        anim_panel = None
        if ui_manager is not None:
            anim_panel = ui_manager.panels.get("animation")
        if anim_panel is not None and hasattr(anim_panel, "set_scene_only_mode"):
            anim_panel.set_scene_only_mode(not bool(available))

    def export_video(
        self,
        output_path: str,
        fps: int = 30,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        stride: Optional[int] = None,
        resolution_scale: float = 1.0,
        include_hud: bool = False,
    ) -> bool:
        """Export animation as MP4 or GIF video.

        Args:
            output_path: Output file path (.mp4 or .gif)
            fps: Frames per second for the video
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (inclusive), None = last frame
            stride: Frame stride for export, None = current animation stride
            resolution_scale: Post-capture output scale for captured frames
                (1.0 = native renderer pixels; pygfx uses bilinear resizing).
            include_hud: Composite visible viewport HUD widgets into each frame.

        Returns:
            True if export succeeded, False otherwise
        """
        self.last_video_export_error = ""

        def fail(message: str) -> bool:
            """Record a video-export failure message and return ``False``."""
            self.last_video_export_error = message
            logger.error(message)
            return False

        if not renderer_capabilities(getattr(self, "renderer", None)).screenshot_export:
            return fail("Video export is unavailable for the active renderer")

        # Validate frame range
        if end_frame is None:
            end_frame = self.total_animation_steps - 1

        if start_frame < 0 or end_frame >= self.total_animation_steps:
            return fail(
                "Invalid frame range: "
                f"{start_frame}-{end_frame} (total frames: {self.total_animation_steps})"
            )

        if start_frame > end_frame:
            return fail(f"Start frame ({start_frame}) > end frame ({end_frame})")

        if stride is None:
            stride = self.animation_controller.get_current_stride()
        try:
            stride = max(1, int(stride))
        except (TypeError, ValueError):
            return fail(f"Invalid export stride: {stride}")

        available_steps = self.get_available_animation_steps()
        if available_steps:
            frame_indices = list(available_steps[start_frame : end_frame + 1 : stride])
        else:
            frame_indices = list(range(start_frame, end_frame + 1, stride))
        total_frames = len(frame_indices)
        if total_frames == 0:
            return fail(
                "Export produced no frames for range "
                f"{start_frame}-{end_frame} with stride {stride}"
            )

        progress = QProgressDialog("Exporting video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately

        ext = Path(output_path).suffix.lower()
        writer = None
        original_frame = self.app_state.step
        playback_was_running = bool(getattr(self, "animation_running", False))
        original_play_direction = int(getattr(self, "play_direction", 1))
        playback_toggle = getattr(
            getattr(self, "animation_controller", None),
            "toggle_animation",
            None,
        )

        logger.info(
            "Starting video export: %s (%d frames @ %d FPS, stride=%d)",
            output_path,
            total_frames,
            fps,
            stride,
        )

        try:
            # QProgressDialog.processEvents() is required for cancellation and
            # progress painting, but it also dispatches animation timer ticks.
            # Suspend timer-driven playback before selecting explicit export
            # frames, then restore the user's prior direction and running state
            # in the common finally path.
            if playback_was_running:
                if not callable(playback_toggle):
                    return fail(
                        "Video export could not pause active playback because "
                        "the animation controller is unavailable."
                    )
                try:
                    playback_toggle()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    return fail(f"Video export could not pause active playback: {exc}")
                if bool(getattr(self, "animation_running", False)):
                    return fail("Video export could not pause active playback.")

            import imageio.v2 as imageio

            if ext == ".gif":
                writer = imageio.get_writer(output_path, mode="I", fps=fps, loop=0)
            elif ext == ".mp4":
                writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
            else:
                return fail(f"Unsupported format: {ext}. Use .mp4 or .gif")

            for progress_idx, frame_idx in enumerate(frame_indices, start=1):
                if progress.wasCanceled():
                    logger.info("Video export canceled by user")
                    return fail("Video export canceled by user.")

                self.update_frame(frame_idx)
                QApplication.processEvents()  # Allow UI to update

                image = self.renderer.export_screenshot_to_array(
                    resolution_scale=resolution_scale,
                    include_hud=include_hud,
                )

                writer.append_data(image)

                progress.setValue(progress_idx)
                QApplication.processEvents()

            writer.close()
            writer = None

            logger.info(f"Video exported successfully to {output_path}")

            # Auto-open file location
            try:
                _open_export_location(output_path)
            except OSError as exc:
                logger.warning(f"Could not auto-open file location: {exc}")

            return True

        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if ext == ".mp4":
                return fail(
                    "MP4 export failed. Install an MP4 writer backend such as "
                    "'imageio[ffmpeg]' or 'pyav', then retry. Details: "
                    f"{exc}"
                )
            return fail(f"Video export failed: {exc}")
        finally:
            if writer is not None:
                try:
                    writer.close()
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning("Video writer cleanup failed: %s", exc)
            progress.close()
            try:
                self.update_frame(original_frame)
            finally:
                if (
                    playback_was_running
                    and callable(playback_toggle)
                    and not bool(getattr(self, "animation_running", False))
                ):
                    try:
                        self.play_direction = original_play_direction
                        playback_toggle(original_play_direction)
                    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        logger.error(
                            "Could not restore playback after video export: %s",
                            exc,
                        )

    def reset_camera_to_overview(self):
        """Reset to the scenario default camera or fit the whole scene."""
        renderer = getattr(self, "renderer", None)
        if not self.vis_initialized or renderer is None:
            return

        try:
            logger.debug("Resetting camera to overview position...")
            camera = getattr(self, "camera_controller", None)
            overview_button = getattr(self, "overview_mode_rb", None)
            if overview_button is not None and hasattr(overview_button, "setChecked"):
                overview_button.setChecked(True)
            elif hasattr(self, "set_state"):
                self.set_state(camera_mode="overview")

            if camera is not None and hasattr(camera, "restore_pov_entity_visibility"):
                try:
                    camera.restore_pov_entity_visibility(update_renderer=False)
                except TypeError:
                    camera.restore_pov_entity_visibility()
            if hasattr(self, "set_state"):
                self.set_state(camera_mode="overview", pov_hidden_node=None)
            if camera is not None and hasattr(camera, "clear_pre_pov_camera_state"):
                camera.clear_pre_pov_camera_state()
            if hasattr(renderer, "reset_follow_state"):
                renderer.reset_follow_state()

            default_camera_view = getattr(self, "default_camera_view", None)
            view_to_apply = default_camera_view or "isometric"
            if hasattr(self, "apply_camera_view"):
                if self.apply_camera_view(
                    view_to_apply,
                    camera_dist=(
                        getattr(self, "default_camera_dist", None) if default_camera_view else None
                    ),
                    fov=(
                        getattr(self, "default_camera_fov", None) if default_camera_view else None
                    ),
                ):
                    return

            renderer.reset_camera_bounds()
            logger.debug("Camera reset: Fit to whole-scene bounds")

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error(f"Error resetting camera: {e}")

    def update_visualizer(self):
        """Poll renderer events and flush a requested redraw."""
        if not self.vis_initialized or self.vis is None:
            return

        try:
            if renderer_capabilities(self.renderer).external_event_pump:
                self.renderer.poll_events()

            if self.force_update_next_frame and not bool(
                getattr(self, "_frame_retry_pending", False)
            ):
                try:
                    self.renderer.update_renderer()
                except RuntimeError as e:
                    # Qt teardown can invalidate GLFW context before timer callbacks drain.
                    if "GLFW" in str(e) or "context" in str(e).lower():
                        pass  # Ignore GLFW context errors
                    else:
                        logger.error(f"Renderer update error: {e}")
                self.force_update_next_frame = False
        except RuntimeError as e:
            # Qt teardown can invalidate GLFW context before timer callbacks drain.
            if "GLFW" in str(e) or "context" in str(e).lower():
                pass  # Ignore GLFW context errors
            else:
                logger.error(f"Error updating visualizer: {e}")

    def _handle_camera_preset_clicked(self, preset_num: int):
        """Handle saved-view button click."""
        if not hasattr(self, "camera_controller"):
            return

        if getattr(self, "_camera_preset_save_mode", False):
            self._save_camera_preset(preset_num)
            self._set_camera_preset_save_mode(False)
            return

        success = self.camera_controller.load_camera_preset(preset_num)
        if not success:
            logger.debug(
                "Camera view %s not set. Click Save, then %s to store it.",
                preset_num,
                preset_num,
            )

    def _save_camera_preset(self, preset_num: int) -> bool:
        """Save current camera to preset slot."""
        if not hasattr(self, "camera_controller"):
            return False

        success = self.camera_controller.save_camera_preset(preset_num)
        if success:
            self._update_camera_preset_buttons()
        return bool(success)

    def _set_camera_preset_save_mode(self, enabled: bool) -> None:
        """Toggle saved-view slot buttons between load and save mode."""
        self._camera_preset_save_mode = bool(enabled)

        save_btn = getattr(self, "camera_preset_save_btn", None)
        if save_btn is not None:
            if bool(save_btn.isChecked()) != self._camera_preset_save_mode:
                save_btn.blockSignals(True)
                save_btn.setChecked(self._camera_preset_save_mode)
                save_btn.blockSignals(False)
            if self._camera_preset_save_mode:
                save_btn.setToolTip("Choose a view number to save the current view")
                save_btn.setStyleSheet("QPushButton { font-weight: bold; }")
            else:
                save_btn.setToolTip("Click, then choose a view number to save the current view")
                save_btn.setStyleSheet("")

        self._update_camera_preset_buttons()

    def _update_camera_preset_buttons(self):
        """Update camera preset button appearance based on saved presets."""
        if not hasattr(self, "camera_preset_buttons") or not self.camera_preset_buttons:
            return
        if not hasattr(self, "camera_controller"):
            return

        presets = self.camera_controller.get_all_presets()
        save_mode = bool(getattr(self, "_camera_preset_save_mode", False))
        base_style = "QPushButton { font-size: 10px; padding: 2px; }"

        for btn in self.camera_preset_buttons:
            preset_num = btn.property("preset_num")
            if save_mode:
                btn.setToolTip(f"Click to save current view to slot {preset_num}")
                btn.setStyleSheet(
                    base_style
                    + " QPushButton { border: 1px solid #b8860b; background-color: #fff8dc; }"
                )
            elif preset_num in presets:
                preset_name = presets[preset_num]
                btn.setToolTip(f"View {preset_num}: {preset_name}\n" "Click to load")
                btn.setStyleSheet(
                    base_style + " QPushButton { border: 1px solid #8c98a8; font-weight: bold; }"
                )
            else:
                btn.setToolTip(
                    f"View {preset_num}: empty. Click Save, then {preset_num} to store current view"
                )
                btn.setStyleSheet(base_style)

    def apply_camera_view(
        self,
        view: str,
        camera_dist: Optional[float] = None,
        fov: Optional[float] = None,
    ) -> bool:
        """Apply a named overview camera view (top/side/front/isometric)."""
        if not view:
            return False
        # Force Overview mode so the view isn't overridden by Follow/POV.
        if hasattr(self, "overview_mode_rb") and self.overview_mode_rb is not None:
            self.overview_mode_rb.setChecked(True)
        else:
            try:
                self.set_state(camera_mode="overview")
            except (TypeError, ValueError, AttributeError):
                pass
        return self.camera_controller.set_overview_view(view, camera_dist=camera_dist, fov=fov)

    def _apply_view_defaults(
        self, view_defaults: Dict[str, Any], skip_camera: bool = False
    ) -> None:
        """Apply scenario-provided view defaults (state + camera).

        Args:
            view_defaults: Dict of AppState fields and camera overrides.
            skip_camera: When True, skip camera application (already applied
                during boot to avoid a visible jump).
        """
        if not view_defaults:
            return

        # Only pass AppState fields into set_state
        state_fields = {field.name for field in fields(AppState)}
        state_updates = {k: v for k, v in view_defaults.items() if k in state_fields}
        if state_updates:
            try:
                self.set_state(**state_updates)
                logger.debug("Applied AppState view defaults: %s", state_updates.keys())
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Could not apply AppState view defaults: %s", exc)

        if skip_camera:
            return

        # Camera view overrides
        camera_view = view_defaults.get("camera_view")
        camera_dist = view_defaults.get("camera_dist")
        camera_fov = view_defaults.get("fov")

        if camera_view:
            self.default_camera_view = camera_view
            self.default_camera_dist = camera_dist
            self.default_camera_fov = camera_fov
            self.apply_camera_view(camera_view, camera_dist=camera_dist, fov=camera_fov)

    def open_scenario(
        self,
        scenario_path: str,
        pending_camera: Optional[dict] = None,
        *,
        autorun_initial_frame: bool = True,
        data_mode_override: str | None = None,
        grpc_port_override: int | None = None,
    ) -> ScenarioOpenOutcome:
        """Open a scenario through the application lifecycle workflow."""
        self._last_scenario_attempt = str(scenario_path)
        self._last_scenario_data_mode_override = data_mode_override
        self._last_scenario_grpc_port_override = grpc_port_override
        mode_controller = getattr(self, "workspace_mode_controller", None)
        prepare_normal_open = getattr(
            mode_controller,
            "prepare_normal_scenario_open",
            None,
        )
        if callable(prepare_normal_open) and not prepare_normal_open():
            return ScenarioOpenOutcome(
                ScenarioOpenStatus.CANCELLED,
                message="Scenario open cancelled while leaving authoring mode.",
            )
        reset_retry = getattr(self, "reset_frame_retry_state", None)
        if callable(reset_retry):
            reset_retry()
        source_overrides: dict[str, Any] = {}
        if data_mode_override is not None:
            source_overrides["data_mode_override"] = data_mode_override
        if grpc_port_override is not None:
            source_overrides["grpc_port_override"] = grpc_port_override
        return app_open_scenario(
            self,
            scenario_path,
            pending_camera=pending_camera,
            autorun_initial_frame=autorun_initial_frame,
            **source_overrides,
        )

    def _authoring_mode_controller(self):
        """Return the gated workspace controller, constructing it lazily."""
        from .src.authoring.feature import scenario_builder_enabled

        if not scenario_builder_enabled():
            raise RuntimeError("Scenario Builder requires ORCHAV_ENABLE_SCENARIO_BUILDER=1")
        if (
            self._renderer_type != "pygfx"
            or not renderer_capabilities(self.renderer).scenario_authoring
        ):
            raise RuntimeError("Scenario Builder requires the pygfx renderer")
        controller = getattr(self, "workspace_mode_controller", None)
        if controller is None:
            from .src.authoring.mode_controller import WorkspaceModeController

            controller = WorkspaceModeController(self)
            self.workspace_mode_controller = controller
        if getattr(self, "authoring_generation_controller", None) is None:
            from .src.authoring.generation_controller import QtGenerationController

            self.authoring_generation_controller = QtGenerationController(
                save_callback=controller.save,
                workspace_provider=lambda: controller.workspace,
                parent=self,
            )
        self._refresh_authoring_actions()
        return controller

    def new_authoring_scenario(self) -> bool:
        """Enter the Scenario Builder with a new unsaved document."""
        return bool(self._authoring_mode_controller().new_document())

    def open_for_authoring(self, scenario_path: str) -> bool:
        """Open one canonical compatible scenario in the embedded builder."""
        return bool(self._authoring_mode_controller().open_document(scenario_path))

    def open_for_authoring_dialog(self) -> bool:
        """Choose a scenario YAML or directory and route it through ownership policy."""
        from PySide6.QtWidgets import QFileDialog

        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open for Authoring",
            str(Path.cwd()),
            "ORCHAV Scenario (scenario.yaml);;All Files (*.*)",
        )
        return bool(path and self.open_for_authoring(path))

    def edit_current_scenario(self) -> bool:
        """Open the active visualization scenario in the embedded builder."""

        scenario_root = _current_canonical_scenario_root(self)
        if scenario_root is None:
            return False
        return self.open_for_authoring(str(scenario_root))

    def copy_current_scenario_and_edit(self) -> bool:
        """Create and enter an editable copy of the active scenario."""

        scenario_root = _current_canonical_scenario_root(self)
        if scenario_root is None:
            return False
        return bool(
            self._authoring_mode_controller().create_copy_for_authoring(
                scenario_root / "scenario.yaml"
            )
        )

    def resume_authoring_draft(self) -> bool:
        """Return from result playback to the preserved in-memory draft."""

        return bool(self._authoring_mode_controller().resume_document())

    def save_authoring_scenario(self) -> bool:
        """Save the active authoring document to its canonical path."""

        return bool(self._authoring_mode_controller().save())

    def save_authoring_scenario_as(self) -> bool:
        """Choose a directory and write exactly ``<directory>/scenario.yaml``."""

        return bool(self._authoring_mode_controller().save_as())

    def _refresh_authoring_actions(self) -> None:
        """Synchronize Scenario Builder actions with draft, mode, and job state."""

        controller = getattr(self, "workspace_mode_controller", None)
        document = getattr(controller, "authoring_document", None)
        mode = getattr(getattr(controller, "mode", None), "value", "visualization")
        in_authoring = mode == "authoring"
        generation = getattr(self, "authoring_generation_controller", None)
        running = bool(getattr(generation, "running", False))
        writable = document is not None and not bool(getattr(document, "read_only", False))
        workspace = getattr(controller, "workspace", None)
        pending_waypoints = bool(getattr(workspace, "has_pending_waypoint_session", False))
        current_scenario_editable = _current_canonical_scenario_root(self) is not None
        action_states = (
            (
                "new_authoring_scenario_action",
                not running and not pending_waypoints,
            ),
            (
                "open_for_authoring_action",
                not running and not pending_waypoints,
            ),
            (
                "resume_authoring_draft_action",
                document is not None and not in_authoring and not running and not pending_waypoints,
            ),
            (
                "edit_current_scenario_action",
                current_scenario_editable and not in_authoring and not running,
            ),
            (
                "copy_current_scenario_action",
                current_scenario_editable and not in_authoring and not running,
            ),
            (
                "save_authoring_scenario_action",
                writable and in_authoring and not running and not pending_waypoints,
            ),
            (
                "save_authoring_scenario_as_action",
                writable and in_authoring and not running and not pending_waypoints,
            ),
            (
                "return_to_visualization_action",
                in_authoring and not running and not pending_waypoints,
            ),
        )
        for attribute, enabled in action_states:
            action = getattr(self, attribute, None)
            if action is not None:
                action.setEnabled(bool(enabled))

    def return_to_visualization(self) -> bool:
        """Leave authoring while retaining its document and undo history."""
        controller = getattr(self, "workspace_mode_controller", None)
        return bool(controller is None or controller.request_leave_authoring())


def main():
    """Run the visualizer CLI entry point."""
    run_visualizer_cli(OrchavVisualizer, ProgressReporter)


if __name__ == "__main__":
    main()
