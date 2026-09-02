"""Application-wide visualizer shutdown.

The window, benchmark, batch-render, and failed-startup paths all delegate
here so background work is retired before native renderer resources.  Each
step is best-effort: one broken collaborator must not leave later workers or
the Qt application alive.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shared.logging import get_logger

from ..io.config_handlers import RecentFilesHandler
from .frame_loader import teardown_frame_loader
from .renderer_lifecycle import stop_renderer_session

logger = get_logger(__name__)

_ROOT_TIMER_NAMES = (
    "_benchmark_timer",
    "_batch_render_timer",
    "_startup_preload_timer",
    "update_timer_coalesce",
    "slider_scrub_timer",
    "update_timer",
)


def shutdown_visualizer(viz: Any, *, persist_state: bool = False) -> bool:
    """Release application resources once, leaving the renderer until last.

    Args:
        viz: Visualizer window or a partially initialized visualizer-like object.
        persist_state: Save recent files and the autosave session before runtime
            scene resources are detached.  Window close uses this; CLI-driven
            benchmark and batch exits do not.

    Returns:
        ``True`` when every attempted step completed successfully.  Shutdown is
        intentionally best-effort, so failures are logged instead of raised.
    """
    if bool(getattr(viz, "_shutdown_complete", False)):
        return bool(getattr(viz, "_shutdown_succeeded", True))
    if bool(getattr(viz, "_shutdown_started", False)):
        return False

    viz._shutdown_started = True
    # A callback already queued in Qt may still run after its timer is stopped.
    # Mark frame data unavailable before detaching sources so such callbacks
    # return without trying to reload from a closed scenario.
    viz.ready = False
    failures: list[str] = []

    def run_step(name: str, operation: Callable[[], Any] | None) -> bool:
        if not callable(operation):
            return True
        try:
            operation()
        except Exception as exc:  # shutdown must continue through independent resources
            failures.append(name)
            logger.warning("Visualizer shutdown step '%s' failed: %s", name, exc, exc_info=True)
            return False
        return True

    # Prevent queued callbacks from scheduling more work while collaborators
    # are being detached.  The named timers cover both interactive and
    # CLI-driven roots; controller/service-owned timers are stopped below.
    viz.update_pending = False
    viz._startup_preload_requested = False
    for timer_name in _ROOT_TIMER_NAMES:
        timer = getattr(viz, timer_name, None)
        run_step(f"timer:{timer_name}", getattr(timer, "stop", None))
    viz._pending_slider_scrub_index = None

    animation_controller = getattr(viz, "animation_controller", None)
    animation_stopped = run_step(
        "animation_controller",
        getattr(animation_controller, "shutdown", None),
    )
    if not animation_stopped or animation_controller is None:
        timer = getattr(viz, "animation_timer", None)
        run_step("timer:animation_timer", getattr(timer, "stop", None))
    run_step(
        "live_playback",
        getattr(animation_controller, "stop_live_playback", None),
    )

    # UI controllers own a few unparented debounce/height timers which are not
    # covered by the root timer set.
    ui_controller = getattr(viz, "ui_controller", None)
    for timer_name in ("_tx_marker_size_timer", "_rx_marker_size_timer"):
        timer = getattr(ui_controller, timer_name, None)
        run_step(f"ui_timer:{timer_name}", getattr(timer, "stop", None))
    coverage_controller = getattr(ui_controller, "_coverage_ctrl", None)
    coverage_timer = getattr(coverage_controller, "_height_animation_timer", None)
    run_step("ui_timer:coverage_height", getattr(coverage_timer, "stop", None))

    animation_service = getattr(viz, "animation_service", None)
    run_step("animation_service", getattr(animation_service, "stop", None))
    run_step(
        "frame_preloader",
        getattr(animation_service, "reset_preloading_state", None),
    )
    warmer = getattr(viz, "_vm_warmer", None)
    run_step("viewmodel_warmer", getattr(warmer, "stop", None))

    live_preview = getattr(viz, "live_preview_service", None)
    stop_live_preview = getattr(live_preview, "stop", None)
    if not callable(stop_live_preview):
        stop_live_preview = getattr(live_preview, "reset", None)
    run_step("live_preview", stop_live_preview)

    explorer_session = getattr(viz, "_mpc_explorer_session", None)
    run_step("mpc_explorer", getattr(explorer_session, "shutdown", None))
    viz._mpc_explorer_session = None

    # Persist while semantic scenario state is still attached.  Persistence
    # failure must not block worker, source, or renderer teardown.
    if persist_state:
        config_file = getattr(viz, "config_file", None)
        recent_files = getattr(viz, "recent_files", None)
        if config_file is not None and recent_files is not None:
            run_step(
                "recent_files",
                lambda: RecentFilesHandler.save_recent_files(config_file, recent_files),
            )
        session_service = getattr(viz, "session_service", None)
        run_step(
            "session_autosave",
            getattr(session_service, "auto_save_on_exit", None),
        )

    # The preview is only a signal consumer; the coordinator is the one owner
    # of trajectory source reads, snapshots, and its background worker.
    ui_manager = getattr(viz, "ui_manager", None)
    panels = getattr(ui_manager, "panels", {}) or {}
    trajectory_panel = panels.get("trajectory")
    panel_cleanup = getattr(ui_manager, "cleanup", None)
    if callable(panel_cleanup):
        run_step("ui_panels", panel_cleanup)
    else:
        run_step("trajectory_preview", getattr(trajectory_panel, "cleanup", None))
    trajectory_coordinator = getattr(viz, "trajectory_load_coordinator", None)
    run_step("trajectory", getattr(trajectory_coordinator, "shutdown", None))

    run_step("frame_loader", lambda: teardown_frame_loader(viz))

    frame_source = getattr(viz, "frame_source", None)
    run_step("frame_source", getattr(frame_source, "close", None))
    viz.frame_source = None
    mpc_core = getattr(viz, "mpc_core", None)
    set_frame_source = getattr(mpc_core, "set_frame_source", None)
    run_step(
        "mpc_frame_source", lambda: set_frame_source(None) if callable(set_frame_source) else None
    )

    target_asset_cache = getattr(viz, "target_asset_cache", None)
    run_step("target_asset_cache", getattr(target_asset_cache, "close", None))

    metrics_window = getattr(viz, "metrics_window", None)
    run_step("metrics_window", getattr(metrics_window, "close", None))

    # Native renderer/window resources are always the final application-owned
    # resource.  This keeps trajectory removal and renderer callbacks valid
    # throughout the preceding teardown steps.
    run_step("renderer", lambda: stop_renderer_session(viz))

    viz._shutdown_failures = tuple(failures)
    viz._shutdown_succeeded = not failures
    viz._shutdown_complete = True
    return not failures
