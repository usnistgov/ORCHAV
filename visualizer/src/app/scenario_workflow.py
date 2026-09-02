"""Scenario lifecycle orchestration for the visualizer app shell.

This module owns the high-level transition from one scenario state to another:
teardown of old scene/frame resources, scenario configuration load, optional
service activation, frame-loader configuration, initial render, camera restore,
and final UI mode updates. Focused services are called directly; the Qt window
remains a composition root rather than a compatibility facade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog

from shared.logging import get_logger

from ..extensions import clear_runtime_extension_services, configure_runtime_extensions
from ..io.frame_sources import LiveGrpcSource
from ..services.cache_service import CacheInvalidationScope
from ..services.scenario_loader_service import (
    ScenarioFrameSourcePreparation,
    ScenarioLoadResult,
)
from ..state import get_beamforming_state_defaults
from .frame_loader import configure_frame_loader, teardown_frame_loader
from .viewport_workspace import ViewportState

logger = get_logger(__name__)


class ScenarioOpenStatus(str, Enum):
    """Terminal state of one app-level scenario-open request."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ScenarioOpenOutcome:
    """Typed scenario-open result consumed by startup and UI callers."""

    status: ScenarioOpenStatus
    scenario_root: Optional[Path] = None
    frame_source_ready: bool = False
    message: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """Return whether the request committed a usable scenario."""
        return self.status is ScenarioOpenStatus.SUCCEEDED

    @classmethod
    def success(cls, scenario_root: Path, *, frame_source_ready: bool) -> ScenarioOpenOutcome:
        """Build a successful outcome with canonical scenario identity."""
        return cls(
            ScenarioOpenStatus.SUCCEEDED,
            scenario_root=scenario_root,
            frame_source_ready=bool(frame_source_ready),
        )

    @classmethod
    def failure(cls, message: str) -> ScenarioOpenOutcome:
        """Build a handled failure outcome."""
        return cls(ScenarioOpenStatus.FAILED, message=message)

    @classmethod
    def cancelled(cls, message: Optional[str] = None) -> ScenarioOpenOutcome:
        """Build an outcome for a caller-aborted mode transition."""
        return cls(ScenarioOpenStatus.CANCELLED, message=message)


class _ScenarioProgressReporter(Protocol):
    """Internal progress-reporting boundary for scenario-open orchestration."""

    def update(self, step: int, message: str) -> None:
        """Report one scenario-open progress step."""
        ...

    def close(self) -> None:
        """Release any progress UI resources."""
        ...


class _QtScenarioProgressReporter:
    """Qt progress dialog adapter used by normal interactive scenario opens."""

    def __init__(self, viz: Any) -> None:
        """Create the modal progress dialog and prime Qt event processing."""
        self._viz = viz
        self._dialog = QProgressDialog("Loading scenario...", None, 0, 8, viz)
        self._dialog.setWindowTitle("Opening Scenario")
        self._dialog.setWindowModality(Qt.WindowModal)
        self._dialog.setMinimumDuration(0)
        self._dialog.setValue(0)
        self._dialog.setAutoClose(True)
        self._dialog.setAutoReset(False)
        QApplication.processEvents()
        self._last_process_events = time.perf_counter()

    def update(self, step: int, message: str) -> None:
        """Mirror scenario-load progress into the dialog and status bar."""
        self._dialog.setLabelText(message)
        self._dialog.setValue(step)
        self._viz._set_status_message(message)
        now = time.perf_counter()
        if now - self._last_process_events > 0.2:
            QApplication.processEvents()
            self._last_process_events = now

    def close(self) -> None:
        """Close the progress dialog."""
        self._dialog.close()


class _EmbeddedScenarioProgressReporter:
    """Progress adapter rendered inside the persistent viewport host."""

    def __init__(self, viz: Any) -> None:
        self._viz = viz
        self._host = viz._viewport_host
        self._last_process_events = time.perf_counter()
        self._host.set_loading_progress(0, 8, "Loading scenario…")

    def update(self, step: int, message: str) -> None:
        self._host.set_loading_progress(step, 8, message)
        self._viz._set_status_message(message)
        now = time.perf_counter()
        if now - self._last_process_events > 0.2:
            QApplication.processEvents()
            self._last_process_events = now

    def close(self) -> None:
        """Keep the host page alive; terminal workflow state replaces it."""


class _StatusScenarioProgressReporter:
    """Non-modal reporter for benchmark and frame-render startup."""

    def __init__(self, viz: Any) -> None:
        self._viz = viz

    def update(self, _step: int, message: str) -> None:
        self._viz._set_status_message(message)
        progress = getattr(self._viz, "progress", None)
        note = getattr(progress, "note", None)
        if callable(note):
            note(message)

    def close(self) -> None:
        """No top-level progress resource is owned."""


class _ScenarioOpenCancelled(RuntimeError):
    """Internal control flow for shutdown during nested Qt event processing."""


def _default_progress_reporter(viz: Any) -> _ScenarioProgressReporter:
    if bool(getattr(viz, "_cli_driven_frame_run", False)):
        return _StatusScenarioProgressReporter(viz)
    if (
        getattr(viz, "_viewport_mode", "detached") == "embedded"
        and getattr(viz, "_viewport_host", None) is not None
    ):
        return _EmbeddedScenarioProgressReporter(viz)
    return _QtScenarioProgressReporter(viz)


def _set_viewport_terminal_state(
    viz: Any,
    state: ViewportState,
    message: str | None = None,
) -> None:
    if getattr(viz, "_viewport_mode", "detached") != "embedded":
        return
    host = getattr(viz, "_viewport_host", None)
    if host is not None:
        host.set_state(state, message)


def cleanup_previous_scene(viz: Any) -> None:
    """Reset app-owned scene state before loading another scenario.

    Cleanup cancels background work before removing scene resources, then
    clears frame/trajectory state and hides frame-backed UI affordances. Keep
    this ordering aligned with ``tests/visualizer/unit/test_scenario_workflow.py``.
    """
    reset_retry = getattr(viz, "reset_frame_retry_state", None)
    if callable(reset_retry):
        reset_retry()
    viz.cancel_startup_preload()
    viz.reset_preloading_state()
    statistics_service = getattr(viz, "scenario_statistics_service", None)
    cancel_statistics = getattr(statistics_service, "cancel_collection", None)
    if callable(cancel_statistics):
        cancel_statistics()
    viz._scene_boot_logged = False
    viz._scene_boot_start = None
    coverage_reset = False
    coverage_service = getattr(viz, "coverage_service", None)
    reset_coverage = getattr(coverage_service, "reset_runtime_state", None)
    if callable(reset_coverage):
        reset_coverage(viz)
        coverage_reset = True
    # Retire the accepted MPC frame before its source or renderer-owned
    # overlays are touched. A failed preflight never reaches this point, so
    # the committed epoch does not invalidate the still-active scenario.
    viz._mpc_presented_source_epoch = int(getattr(viz, "_mpc_presented_source_epoch", 0)) + 1
    explorer_session = getattr(viz, "_mpc_explorer_session", None)
    explorer_teardown = getattr(explorer_session, "on_scenario_teardown", None)
    if callable(explorer_teardown):
        explorer_teardown()
    # Retire trajectory reads before scene cleanup closes the active source.
    viz.trajectory_load_coordinator.reset()
    viz.scene_service.cleanup_previous_scene()
    clear_runtime_extension_services(viz)
    live_preview = getattr(viz, "live_preview_service", None)
    reset_preview = getattr(live_preview, "reset", None)
    if callable(reset_preview):
        reset_preview()
    viz.scenario_config = None
    viz._scene_only_mode = False
    teardown_frame_loader(viz)
    viz._set_frame_data_available(False)
    progress_bar = getattr(viz, "status_progress_bar", None)
    if progress_bar is not None:
        progress_bar.setValue(0)
        progress_bar.setVisible(False)
    if not coverage_reset and getattr(viz, "ui_manager", None):
        set_coverage_available = getattr(viz.ui_manager, "set_coverage_data_available", None)
        if callable(set_coverage_available):
            set_coverage_available(False)
        else:
            viz.ui_manager.set_panel_visible("coverage", False)
    _clear_scenario_identity(viz)


def _clear_scenario_identity(viz: Any) -> None:
    """Clear scenario identity after teardown or a failed transition."""
    viz.scenario = None
    viz.scenario_config = None
    viz.frame_source = None
    viz.current_scenario_path = None
    viz.current_scenario_policy = None
    viz.current_project_root = None
    viz.current_base_dir = None
    viz.ready = False
    viz.force_update_next_frame = False
    refresh_authoring_actions = getattr(viz, "_refresh_authoring_actions", None)
    if callable(refresh_authoring_actions):
        refresh_authoring_actions()

    mpc_core = getattr(viz, "mpc_core", None)
    set_frame_source = getattr(mpc_core, "set_frame_source", None)
    if callable(set_frame_source):
        set_frame_source(None)
    _refresh_edit_panel(viz)


def _refresh_edit_panel(viz: Any) -> None:
    """Refresh source-specific edit controls after a scenario transition."""
    panels = getattr(getattr(viz, "ui_manager", None), "panels", {})
    nodes_panel = panels.get("nodes")
    refresh = getattr(nodes_panel, "refresh_live_preview_state", None)
    if callable(refresh):
        refresh()


def _rollback_failed_scenario(viz: Any) -> None:
    """Retire resources created by an incomplete scenario-open attempt."""
    try:
        cleanup_previous_scene(viz)
    except Exception:  # broad catch: preserve the original scenario-load failure
        logger.exception("Failed to fully roll back an incomplete scenario load")
        _clear_scenario_identity(viz)


def panel_enabled(viz: Any, key: str, default: bool = True) -> bool:
    """Return the scenario-level visibility setting for one optional panel."""
    try:
        config = getattr(viz, "scenario_config", None)
        if not config:
            return default
        viz_config = getattr(config, "visualizer_cfg", {}) or {}
        panels_cfg = viz_config.get("panels", {}) if isinstance(viz_config, dict) else {}
        panel_cfg = panels_cfg.get(key, None)
        if isinstance(panel_cfg, dict):
            return bool(panel_cfg.get("enabled", default))
        if isinstance(panel_cfg, bool):
            return panel_cfg
    except (KeyError, AttributeError, TypeError) as exc:
        logger.debug("Could not check panel config for %s: %s", key, exc)
    return default


def configure_optional_services(viz: Any) -> tuple[str, ...]:
    """Configure externally registered runtime services for this scenario."""
    return configure_runtime_extensions(viz)


def configure_visual_profiles(viz: Any, scenario: Any) -> None:
    """Install scenario visual-material rules before scene assets are resolved.

    Profiles affect the initial material binding of both static meshes and
    targets, so configuring them after either scene path has rendered is too
    late. Passing an empty list clears scenario-owned rules before the next
    scenario is installed while retaining the service's built-in rules.
    """
    service = getattr(viz, "visual_profile_service", None)
    load_rules = getattr(service, "load_scenario_rules", None)
    if not callable(load_rules):
        return
    view_defaults = getattr(scenario, "view_defaults", None) or {}
    rules = view_defaults.get("visual_profiles", []) if isinstance(view_defaults, dict) else []
    load_rules(rules or [])


def _record_recent_scenario(viz: Any, scenario: Any) -> None:
    """Persist the canonical scenario root after a complete successful open."""
    add_recent_file = getattr(getattr(viz, "ui_controller", None), "add_recent_file", None)
    scenario_root = getattr(scenario, "root", None)
    if not callable(add_recent_file) or scenario_root is None:
        return

    try:
        canonical_root = Path(scenario_root).expanduser().resolve(strict=False)
        add_recent_file(str(canonical_root))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        # Recent-file persistence is ancillary UI state. A config or Qt menu
        # failure must not roll back a scenario that is already ready to use.
        logger.warning("Could not record recent scenario %s: %s", scenario_root, exc)


def open_scenario(
    viz: Any,
    scenario_path: str,
    pending_camera: Optional[dict] = None,
    *,
    autorun_initial_frame: bool = True,
    data_mode_override: str | None = None,
    grpc_port_override: int | None = None,
    _progress_factory: Optional[Callable[[Any], _ScenarioProgressReporter]] = None,
) -> ScenarioOpenOutcome:
    """Open a scenario folder or YAML file through app-level orchestration.

    Coverage is scenario-owned and loads independently of MPC frame readiness.
    Frame-backed scenarios then load TX/RX metadata and targets before scheduling
    the first pipeline update. Scene-only scenarios retain coverage while leaving
    frame-driven controls disabled. Process-local source overrides participate in
    preflight but are not persisted to the scenario or workspace.
    """
    if bool(getattr(viz, "_scenario_load_in_progress", False)):
        return ScenarioOpenOutcome.cancelled("Another scenario is already loading.")
    cancel_scrub = getattr(viz, "cancel_pending_slider_scrub", None)
    if callable(cancel_scrub):
        cancel_scrub()
    viz._scenario_load_in_progress = True
    viz._scenario_load_generation = int(getattr(viz, "_scenario_load_generation", 0)) + 1
    load_generation = viz._scenario_load_generation
    had_active_scenario = bool(
        getattr(viz, "current_scenario_path", None) and getattr(viz, "vis_initialized", False)
    )
    progress: _ScenarioProgressReporter | None = None
    progress_closed = False
    replacement_started = False
    prepared_frame_source: ScenarioFrameSourcePreparation | None = None
    prepared_source_transferred = False

    def close_progress() -> None:
        """Close the progress reporter at most once."""
        nonlocal progress_closed
        if progress_closed or progress is None:
            return
        progress.close()
        progress_closed = True

    def ensure_load_is_current() -> None:
        """Abort work superseded by shutdown or another load generation."""
        if bool(getattr(viz, "_shutdown_started", False)):
            raise _ScenarioOpenCancelled("Scenario loading cancelled during shutdown.")
        if int(getattr(viz, "_scenario_load_generation", 0)) != load_generation:
            raise _ScenarioOpenCancelled("Scenario loading was superseded.")

    def update_progress(step: int, message: str) -> None:
        if progress is None:
            raise RuntimeError("Scenario progress reporter is not initialized")
        progress.update(step, message)
        ensure_load_is_current()

    try:
        progress = (
            _progress_factory(viz)
            if _progress_factory is not None
            else _default_progress_reporter(viz)
        )
        update_progress(1, "Validating scenario...")
        preflight_overrides: dict[str, Any] = {}
        if data_mode_override is not None:
            preflight_overrides["data_mode_override"] = data_mode_override
        if grpc_port_override is not None:
            preflight_overrides["grpc_port_override"] = grpc_port_override
        preflight = viz.scenario_loader_service.preflight_scenario(
            scenario_path,
            **preflight_overrides,
        )
        if preflight is None:
            message = f"Failed to load scenario: {scenario_path}"
            viz._set_status_message(message, 5000)
            _set_viewport_terminal_state(
                viz,
                ViewportState.ACTIVE if had_active_scenario else ViewportState.ERROR,
                message,
            )
            return ScenarioOpenOutcome.failure(message)

        prepare_frame_source = getattr(
            viz.scenario_loader_service,
            "prepare_frame_source",
            None,
        )
        if callable(prepare_frame_source):
            with viz.progress.task("Preparing frame source"):
                prepared_frame_source = prepare_frame_source(preflight)
            ensure_load_is_current()
            if prepared_frame_source is None:
                preparation_error = getattr(
                    viz.scenario_loader_service,
                    "last_frame_source_preparation_error",
                    None,
                )
                message = (
                    preparation_error
                    if isinstance(preparation_error, str) and preparation_error
                    else f"Failed to prepare scenario data source: {scenario_path}"
                )
                viz._set_status_message(message, 5000)
                _set_viewport_terminal_state(
                    viz,
                    ViewportState.ACTIVE if had_active_scenario else ViewportState.ERROR,
                    message,
                )
                return ScenarioOpenOutcome.failure(message)
            if not isinstance(prepared_frame_source, ScenarioFrameSourcePreparation):
                raise TypeError(
                    "ScenarioLoaderService.prepare_frame_source() must return "
                    "ScenarioFrameSourcePreparation"
                )

        update_progress(2, "Cleaning up previous scene...")
        logger.debug("Cleaning up previous scene and clearing caches")
        viz.reset_startup_timing_profile()
        cleanup_start = time.perf_counter()
        replacement_started = True
        cleanup_previous_scene(viz)
        viz.record_startup_stage_timing(
            "cleanup_previous_scene_ms",
            (time.perf_counter() - cleanup_start) * 1000.0,
        )

        update_progress(3, "Loading scenario configuration...")
        load_scenario_start = time.perf_counter()
        load_kwargs: dict[str, Any] = {
            "cleanup_scene_first": False,
            "preflight": preflight,
        }
        if prepared_frame_source is not None:
            load_kwargs["prepared_frame_source"] = prepared_frame_source
            prepared_source_transferred = True
        result = viz.scenario_loader_service.load_scenario(scenario_path, **load_kwargs)
        viz.record_startup_stage_timing(
            "load_scenario_service_ms",
            (time.perf_counter() - load_scenario_start) * 1000.0,
        )
        if result is None:
            _rollback_failed_scenario(viz)
            message = f"Failed to load scenario: {scenario_path}"
            viz._set_status_message(message, 5000)
            _set_viewport_terminal_state(viz, ViewportState.ERROR, message)
            return ScenarioOpenOutcome.failure(message)

        if not isinstance(result, ScenarioLoadResult):
            raise TypeError("ScenarioLoaderService.load_scenario() must return ScenarioLoadResult")
        scenario = result.scenario
        frame_source = result.frame_source
        frame_source_ready = result.frame_source_ready
        viz.scenario_config = scenario
        viz.frame_source = frame_source
        _refresh_edit_panel(viz)
        configure_visual_profiles(viz, scenario)
        viz.set_state(**get_beamforming_state_defaults(scenario))
        configure_services_start = time.perf_counter()
        enabled_extensions = configure_optional_services(viz)
        viz.record_startup_stage_timing(
            "configure_optional_services_ms",
            (time.perf_counter() - configure_services_start) * 1000.0,
        )
        if hasattr(viz, "ui_manager"):
            for extension_key in enabled_extensions:
                viz.ui_manager.ensure_panel(extension_key)
        configure_frame_loader_start = time.perf_counter()
        configure_frame_loader(viz, frame_source)
        viz.record_startup_stage_timing(
            "configure_frame_loader_ms",
            (time.perf_counter() - configure_frame_loader_start) * 1000.0,
        )
        viz._set_frame_data_available(frame_source_ready)

        _load_scenario_coverage(viz, scenario, update_progress=update_progress)

        if frame_source_ready:
            _open_frame_backed_scenario(
                viz,
                scenario,
                frame_source,
                pending_camera,
                autorun_initial_frame=autorun_initial_frame,
                update_progress=update_progress,
            )
        else:
            _open_scene_only_scenario(
                viz,
                scenario,
                pending_camera,
                update_progress=update_progress,
            )

        if scenario.view_defaults:
            viz._apply_view_defaults(scenario.view_defaults, skip_camera=True)

        logger.debug("Frame update pipeline scheduled")
        _configure_animation_panel_mode(viz)
        _configure_trajectory_controls(viz)
        _configure_source_color_modes(viz, frame_source)
        _configure_trajectory_panel_visibility(viz)

        scenario_root = Path(scenario.root).expanduser().resolve(strict=False)
        logger.info("Scenario opened: %s", scenario_root)
        # Commit identity and ancillary recent-file state only after every
        # required scene/UI synchronization step has succeeded.
        viz.current_scenario_path = str(scenario_root)
        refresh_authoring_actions = getattr(viz, "_refresh_authoring_actions", None)
        if callable(refresh_authoring_actions):
            refresh_authoring_actions()
        _record_recent_scenario(viz, scenario)

        status_msg = (
            "Scene loaded — scene inspector mode (no frame data)"
            if viz._scene_only_mode
            else "Scenario loaded successfully"
        )
        viz._set_status_message(status_msg, 5000)
        _set_viewport_terminal_state(viz, ViewportState.ACTIVE)
        return ScenarioOpenOutcome.success(
            scenario_root,
            frame_source_ready=frame_source_ready,
        )
    except _ScenarioOpenCancelled as exc:
        if replacement_started and not bool(getattr(viz, "_shutdown_started", False)):
            _rollback_failed_scenario(viz)
            _set_viewport_terminal_state(viz, ViewportState.ERROR, str(exc))
        return ScenarioOpenOutcome.cancelled(str(exc))
    except Exception as exc:
        if replacement_started:
            _rollback_failed_scenario(viz)
        _set_viewport_terminal_state(viz, ViewportState.ERROR, str(exc))
        raise
    finally:
        if prepared_frame_source is not None and not prepared_source_transferred:
            close_source = getattr(prepared_frame_source.frame_source, "close", None)
            if callable(close_source):
                try:
                    close_source()
                except Exception:
                    logger.warning(
                        "Could not close uncommitted candidate frame source", exc_info=True
                    )
        try:
            close_progress()
        except Exception:
            # Progress UI is ancillary. A deleted Qt dialog/host must not mask
            # the load outcome or permanently wedge the reentrancy guard.
            logger.warning("Could not close scenario progress reporter", exc_info=True)
        finally:
            if int(getattr(viz, "_scenario_load_generation", 0)) == load_generation:
                viz._scenario_load_in_progress = False


def _load_scenario_coverage(viz: Any, scenario: Any, *, update_progress: Any) -> bool:
    """Load scenario coverage regardless of MPC frame-source availability."""
    update_progress(4, "Loading coverage data...")
    coverage_start = time.perf_counter()
    with viz.progress.task("Loading coverage data"):
        loaded = viz.coverage_service.load_coverage_map(
            scenario.root,
            viz,
        )
    viz.record_startup_stage_timing(
        "coverage_load_ms",
        (time.perf_counter() - coverage_start) * 1000.0,
    )
    return bool(loaded)


def _open_frame_backed_scenario(
    viz: Any,
    scenario: Any,
    frame_source: Any,
    pending_camera: Optional[dict],
    *,
    autorun_initial_frame: bool,
    update_progress: Any,
) -> None:
    """Load frame-backed resources before enabling the frame update pipeline."""
    update_progress(5, "Discovering TX/RX metadata...")
    txrx_start = time.perf_counter()
    with viz.progress.task("Discovering TX/RX metadata"):
        viz.node_service.discover_available_tx_rx()
    viz.record_startup_stage_timing(
        "tx_rx_discovery_ms",
        (time.perf_counter() - txrx_start) * 1000.0,
    )
    logger.debug("TX/RX discovery completed")

    update_progress(6, "Loading target meshes...")
    target_load_start = time.perf_counter()
    with viz.progress.task("Loading target meshes"):
        viz.target_service.load_target_models()
    viz.record_startup_stage_timing(
        "target_mesh_load_ms",
        (time.perf_counter() - target_load_start) * 1000.0,
    )
    _refresh_target_ui(viz)

    update_progress(7, "Rendering initial scene...")
    render_scene_start = time.perf_counter()
    with viz.progress.task("Rendering initial scene"):
        viz.scene_service.render_scene()
    viz.record_startup_stage_timing(
        "render_initial_scene_ms",
        (time.perf_counter() - render_scene_start) * 1000.0,
    )

    _apply_initial_camera(viz, scenario, pending_camera)

    update_progress(8, "Populating material filters...")
    material_filters_start = time.perf_counter()
    with viz.progress.task("Populating material filters"):
        viz.ui_controller.populate_material_filters()
    viz.record_startup_stage_timing(
        "populate_material_filters_ms",
        (time.perf_counter() - material_filters_start) * 1000.0,
    )

    _sync_initial_frame(viz, frame_source)

    viz.ready = True
    _start_scenario_statistics(viz, frame_source)
    viz.request_startup_preload()
    if autorun_initial_frame:
        viz.force_update_next_frame = True
        viz.schedule_update()
        logger.debug("FrameSource ready, update pipeline scheduled")
    else:
        viz.force_update_next_frame = False
        viz.cancel_scheduled_update()
        logger.debug("FrameSource ready; initial frame autorun deferred to caller")


def _start_scenario_statistics(viz: Any, frame_source: Any) -> bool:
    """Start statistics from the provider without waiting for frame preload."""

    benchmark_active = (
        getattr(getattr(viz, "pipeline", None), "benchmark_recorder", None) is not None
    )
    if benchmark_active or not panel_enabled(viz, "statistics", default=True):
        return False

    ui_manager = getattr(viz, "ui_manager", None)
    panels = getattr(ui_manager, "panels", {}) if ui_manager is not None else {}
    statistics_panel = panels.get("statistics")
    bind_source = getattr(statistics_panel, "set_statistics_source", None)
    if not callable(bind_source):
        return False
    try:
        return bool(bind_source(frame_source))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Could not start scenario statistics: %s", exc)
        return False


def _open_scene_only_scenario(
    viz: Any,
    scenario: Any,
    pending_camera: Optional[dict],
    *,
    update_progress: Any,
) -> None:
    """Render a static scene while leaving frame-dependent controls disabled."""
    viz._scene_only_mode = True

    update_progress(5, "Rendering scene (no frame data)...")
    scene_only_render_start = time.perf_counter()
    with viz.progress.task("Rendering scene (scene-only)"):
        viz.scene_service.render_scene()
    viz.record_startup_stage_timing(
        "scene_only_render_scene_ms",
        (time.perf_counter() - scene_only_render_start) * 1000.0,
    )

    _apply_initial_camera(viz, scenario, pending_camera)

    if hasattr(viz, "ui_manager") and "animation" in viz.ui_manager.panels:
        anim_panel = viz.ui_manager.panels["animation"]
        anim_panel.set_scene_only_mode(True)

    logger.info("Scene-only mode: no frame data, scene inspector active")


def _refresh_target_ui(viz: Any) -> None:
    """Refresh target-dependent controls after target meshes are loaded."""
    target_refresh_start = time.perf_counter()
    viz.camera_controller.update_target_focus_dropdown()
    viz.cache_service.invalidate(
        CacheInvalidationScope.MATERIALS_COLORS,
        reason="target_materials_loaded",
    )

    if hasattr(viz, "ui_manager") and viz.ui_manager:
        materials_panel = viz.ui_manager.panels.get("materials")
        if materials_panel is not None and hasattr(materials_panel, "refresh_material_list"):
            materials_panel.refresh_material_list()
    viz.record_startup_stage_timing(
        "target_ui_refresh_ms",
        (time.perf_counter() - target_refresh_start) * 1000.0,
    )


def _apply_initial_camera(viz: Any, scenario: Any, pending_camera: Optional[dict]) -> None:
    """Apply restored session camera before scenario defaults to avoid jumps."""
    camera_start = time.perf_counter()
    if pending_camera is not None:
        viz.session_service._restore_camera_state(pending_camera)
        logger.debug("Applied pre-read session camera (no jump)")
    elif scenario.view_defaults:
        cam_view = scenario.view_defaults.get("camera_view")
        cam_dist = scenario.view_defaults.get("camera_dist")
        cam_fov = scenario.view_defaults.get("fov")
        if cam_view:
            viz.default_camera_view = cam_view
            viz.default_camera_dist = cam_dist
            viz.default_camera_fov = cam_fov
            viz.apply_camera_view(cam_view, camera_dist=cam_dist, fov=cam_fov)
            logger.debug("Applied scenario view_defaults camera")
    viz.record_startup_stage_timing(
        "apply_initial_camera_ms",
        (time.perf_counter() - camera_start) * 1000.0,
    )


def _sync_initial_frame(viz: Any, frame_source: Any) -> None:
    """Pin the UI to the first available frame step from the active source."""
    first_frame = None
    if frame_source and hasattr(frame_source, "list_frames"):
        try:
            available_frames = frame_source.list_frames()
            if available_frames:
                first_frame = min(available_frames)
        except (OSError, ValueError) as exc:
            logger.debug("Unable to determine initial frame: %s", exc)

    if first_frame is None or first_frame == viz.animation_step:
        return

    logger.debug("Setting initial animation step to %d", first_frame)
    viz.animation_step = first_frame
    viz.set_state(step=first_frame)
    first_frame_index = viz.get_animation_step_index(first_frame)
    if hasattr(viz, "step_slider") and viz.step_slider is not None:
        viz.step_slider.blockSignals(True)
        viz.step_slider.setValue(first_frame_index)
        viz.step_slider.blockSignals(False)
    if hasattr(viz, "step_label") and viz.step_label is not None:
        viz.step_label.setText(str(first_frame_index + 1))


def _configure_animation_panel_mode(viz: Any) -> None:
    """Switch animation controls between offline and live gRPC semantics."""
    if not (hasattr(viz, "ui_manager") and "animation" in viz.ui_manager.panels):
        return
    animation_panel = viz.ui_manager.panels["animation"]
    is_online = isinstance(viz.frame_source, LiveGrpcSource) if viz.frame_source else False
    logger.debug("Frame source type: %s, is_online: %s", type(viz.frame_source), is_online)
    animation_panel.set_online_mode(is_online)
    if is_online:
        logger.debug("Animation panel: live gRPC mode enabled")
    else:
        logger.debug("Animation panel: Offline mode")


def _configure_trajectory_controls(viz: Any) -> None:
    """Enable trajectory controls only for coordinator-supported sources."""
    if hasattr(viz, "ui_controller"):
        is_file_mode = viz.trajectory_load_coordinator.supports_source(viz.frame_source)
        viz.ui_controller.configure_trajectory_checkboxes(enabled=is_file_mode)


def _configure_source_color_modes(viz: Any, frame_source: Any) -> None:
    """Expose color modes that are supported by the selected frame source."""
    if hasattr(viz, "ui_manager") and "mpc" in viz.ui_manager.panels:
        mpc_panel = viz.ui_manager.panels["mpc"]
        supports_reconstruction_type = bool(
            getattr(frame_source, "supports_reconstruction_type_color", False)
        )
        mpc_panel.set_reconstruction_type_visible(supports_reconstruction_type)
        if supports_reconstruction_type:
            logger.debug("Source-specific reconstruction color mode enabled")


def _configure_trajectory_panel_visibility(viz: Any) -> None:
    """Show precomputed trajectory preview only when the panel has data to display."""
    if not (hasattr(viz, "ui_manager") and "trajectory" in viz.ui_manager.panels):
        return
    trajectory_panel = viz.ui_manager.panels["trajectory"]
    if trajectory_panel and hasattr(trajectory_panel, "should_be_visible"):
        should_show = trajectory_panel.should_be_visible()
        viz.ui_manager.set_panel_visible("trajectory", should_show)
        if should_show:
            logger.debug("Trajectory preview panel enabled (pre-computed data mode)")
        else:
            logger.debug("Trajectory preview panel hidden (live gRPC mode)")
