"""Private visualizer registration for sensing UI and frame overlays."""

from __future__ import annotations

from typing import Any
from weakref import WeakKeyDictionary

from shared.logging import get_logger

from ..extensions import (
    RuntimeFeatureExtension,
    extension_state_value,
    register_runtime_extension,
    set_extension_state,
)

logger = get_logger(__name__)

FEATURE_KEY = "sensing"
STATE_DEFAULTS: dict[str, Any] = {
    "show_detections": False,
    "highlight_selection": True,
    "scene_mode": "detections",
    "history_frames": 12,
    "full_history": True,
    "show_velocity_arrows": False,
    "show_track_labels": False,
    "show_truth": True,
    "use_marker_glyphs": True,
    "pin_freshest": False,
    "show_rt_context": True,
    "show_tracking_uncertainty": False,
    "uncertainty_sigma": 1.0,
    "uncertainty_selected_only": True,
}
_ANNOTATION_ADAPTERS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


def state_value(state: Any, name: str) -> Any:
    """Read one private runtime value with its canonical default."""
    return extension_state_value(state, FEATURE_KEY, name, STATE_DEFAULTS.get(name))


def update_state(viz: Any, **changes: Any) -> None:
    """Update the private runtime namespace through immutable AppState replacement."""
    set_extension_state(viz, FEATURE_KEY, **changes)


def _panel_enabled(viz: Any) -> bool:
    """Return whether the current scenario activates the private feature."""
    scenario = getattr(viz, "scenario_config", None)
    default = bool(
        scenario is not None
        and isinstance(getattr(scenario, "sensing", None), dict)
        and scenario.sensing.get("enabled", False)
    )
    if scenario is None:
        return default
    visualizer_cfg = getattr(scenario, "visualizer_cfg", {}) or {}
    panels = visualizer_cfg.get("panels", {}) if isinstance(visualizer_cfg, dict) else {}
    configured = panels.get(FEATURE_KEY)
    if isinstance(configured, dict):
        return bool(configured.get("enabled", default))
    if isinstance(configured, bool):
        return configured
    return default


def _annotation_adapter(renderer: Any) -> Any | None:
    """Return the private pygfx presentation adapter when supported."""
    if getattr(renderer, "renderer_id", None) != "pygfx":
        return None
    adapter = _ANNOTATION_ADAPTERS.get(renderer)
    if adapter is None:
        from ..renderers.pygfx.sensing import PygfxSensingAdapter

        adapter = PygfxSensingAdapter(renderer)
        _ANNOTATION_ADAPTERS[renderer] = adapter
    return adapter


def _prepare_requested_scene(viz: Any) -> Any:
    """Start explicitly enabled scene preparation before either render-stage reader."""
    panels = getattr(getattr(viz, "ui_manager", None), "panels", {})
    panel = panels.get("sensing")
    prepare_scene = getattr(panel, "start_scene_preparation", None)
    if bool(state_value(getattr(viz, "app_state", None), "show_detections")) and callable(
        prepare_scene
    ):
        # Demos and manual workspace restoration can enable overlays directly.
        # Establish the background-preparation gate before any scene reader runs.
        prepare_scene()
    return panel


def _sync_frame(viz: Any, service: Any, raw_frame: dict[str, Any], step: int) -> bool:
    """Combine native measurements and separately saved tracks under one control set."""
    from ..services.evidence_focus import focus_identity

    panel = _prepare_requested_scene(viz)
    viz._evidence_focus_presented_step = step
    focus = getattr(viz, "_evidence_focus", None)
    if focus is not None and focus.active:
        focus.refresh(focus_identity(viz, step))
    if getattr(service, "preparation_pending", False):
        if focus is not None:
            focus.stop()
        refresh_tracking_uncertainty(viz, clear=True)
        return True
    succeeded = _sync_native_frame(viz, service, raw_frame, step)
    scene = getattr(panel, "saved_track_scene", None)
    if scene is None:
        _sync_tracking_uncertainty(viz, service, step, raw_frame)
        if focus is not None and focus.active:
            focus.refresh(focus_identity(viz, step))
        return succeeded
    fusion = getattr(viz, "fusion_window", None)
    if fusion is not None and not fusion.closed and fusion.world_aligned:
        if focus is not None:
            focus.stop()
        refresh_tracking_uncertainty(viz, clear=True)
        return scene.overlay.clear(viz.renderer) and succeeded
    from ..io.packed_frame_payload import visualizer_frame_provider
    from .dashboard_data import DashboardSession

    provider = visualizer_frame_provider(viz)
    snapshots = scene.prepare(
        getattr(viz, "current_scenario_path", None),
        DashboardSession.provider_identity(provider),
        step,
        viz.app_state,
        service,
        result_path=getattr(panel, "saved_result_path", None),
    )
    from ..services.label_callout_service import objects_with_callouts

    succeeded = (
        scene.overlay.sync(viz.renderer, objects_with_callouts(snapshots, viz)) and succeeded
    )
    label = panel.widgets.get("scene_status")
    if label is not None:
        label.setText(scene.status or "Visibility only; saved detections and tracks are unchanged.")
    adapter = _annotation_adapter(viz.renderer)
    if adapter is not None and scene.layers:
        # Preserve native detection picking while adding saved-state identities.
        layers = list(adapter._selection_layers.values()) + scene.layers
        adapter.apply_sensing_scene_annotations(adapter._selection_frame, layers)
        adapter.bind_selection(adapter._selection_frame, layers)
        adapter.highlight_selection(
            adapter._selection_frame, layers, getattr(viz, "sensing_selection", None)
        )
    _sync_tracking_uncertainty(viz, service, step, raw_frame)
    if focus is not None and focus.active:
        focus.refresh(focus_identity(viz, step))
    return succeeded


def _sync_native_frame(viz: Any, service: Any, raw_frame: dict[str, Any], step: int) -> bool:
    """Synchronize private layers after the core frame transaction is prepared."""
    context = getattr(service, "path_context", None)
    if context is not None and not context.sync(viz.renderer):
        return False
    adapter = _annotation_adapter(viz.renderer)
    fusion = getattr(viz, "fusion_window", None)
    if fusion is not None and not fusion.closed and fusion.reader is not None:
        fusion.follow_scene(step)
        if fusion.world_aligned:
            service.clear_scene_overlays(viz.renderer)
            fusion._draw_world()
            return True
    if not bool(state_value(viz.app_state, "show_detections")):
        succeeded = bool(service.clear_scene_overlays(viz.renderer))
        if adapter is not None:
            adapter.clear_sensing_scene_annotations()
            adapter.clear_selection_bindings()
            adapter.update_scene_legend()
        return succeeded

    prepared = service.resolve_scene_overlay_layers(
        raw_frame,
        step=step,
        mode=str(state_value(viz.app_state, "scene_mode")),
        history_frames=int(state_value(viz.app_state, "history_frames")),
        use_full_history=bool(state_value(viz.app_state, "full_history")),
        show_velocity_arrows=bool(state_value(viz.app_state, "show_velocity_arrows")),
        pin_freshest=bool(state_value(viz.app_state, "pin_freshest")),
    )
    if prepared is None:
        succeeded = bool(service.clear_scene_overlays(viz.renderer))
        if adapter is not None:
            adapter.clear_sensing_scene_annotations()
            adapter.clear_selection_bindings()
            adapter.update_scene_legend()
        return succeeded

    detection_frame, layers = prepared
    if not bool(state_value(viz.app_state, "show_truth")):
        layers = [layer for layer in layers if not layer.name.startswith("sensing_truth_")]
    succeeded = bool(service.sync_scene_overlays(viz.renderer, layers))
    if not succeeded:
        logger.warning("Private scene overlays were not fully synchronized")
        return False
    if adapter is not None:
        adapter.apply_sensing_scene_annotations(detection_frame, layers)
        adapter.bind_selection(detection_frame, layers)
        adapter.highlight_selection(
            detection_frame, layers, getattr(viz, "sensing_selection", None)
        )
        adapter.update_scene_legend()
    return True


def refresh_scene_selection(viz: Any, selection: Any) -> None:
    """Refresh an optional renderer's estimate emphasis without reading frames."""
    from ..services.evidence_focus import stop_evidence_focus

    if selection != getattr(viz, "sensing_selection", None):
        stop_evidence_focus(viz)
    viz.sensing_selection = selection
    adapter = _annotation_adapter(viz.renderer)
    if adapter is not None:
        adapter.refresh_selection(selection)
    refresh_tracking_uncertainty(viz)


def locate_scene_selection(viz: Any) -> bool:
    """Center the camera only when the saved estimate has a world position."""
    position = selected_scene_position(viz)
    if position is None:
        return False
    controller = getattr(viz, "camera_controller", None)
    tools = (
        controller.presentation() if controller and hasattr(controller, "presentation") else None
    )
    if tools is not None:
        tools.focus(position)
        return True
    return bool(viz.renderer.focus_camera(position))


def selected_scene_position(viz: Any):
    """Expose the selected saved 3D estimate to optional camera inspection tools."""
    adapter = _annotation_adapter(viz.renderer)
    if adapter is None:
        return None
    return adapter.selected_position(getattr(viz, "sensing_selection", None))


def _status_chips(viz: Any, service: Any, _packet: Any) -> tuple[str, ...]:
    """Identify historical geometry without repeating the chosen sensing controls."""
    context = getattr(service, "path_context", None)
    visibility = getattr(viz.app_state, "mpc_visibility", None)
    historical = (
        bool(getattr(context, "historical_paths_visible", False))
        and bool(state_value(viz.app_state, "show_rt_context"))
        and bool(getattr(visibility, "effective_paths", False))
    )
    return ("Historical paths",) if historical else ()


def _refresh_hud(viz: Any, _service: Any) -> None:
    """Restore an existing scene legend after a HUD toggle without frame reads."""
    renderer = getattr(viz, "renderer", None)
    if getattr(renderer, "renderer_id", None) != "pygfx":
        return
    adapter = _ANNOTATION_ADAPTERS.get(renderer)
    if adapter is not None:
        adapter.update_scene_legend()


def _prepare_render_packet(viz: Any, service: Any, raw: dict, step: int, packet: Any) -> Any:
    """Prepare the optional historical-geometry cue before renderer submission."""
    _prepare_requested_scene(viz)
    if getattr(service, "preparation_pending", False):
        return packet
    return service.path_context.prepare(viz, service, raw, step, packet)


def _create_panel(parent: Any) -> Any:
    """Import and create the private panel only when its tab is requested."""
    from ..panels.sensing_panel import SensingPanel

    return SensingPanel(parent)


def _create_service(viz: Any) -> Any:
    """Import and create the private service only for an enabled scenario."""
    from ..services.sensing_service import SensingService

    views = getattr(getattr(viz, "scenario_config", None), "view_defaults", {}) or {}
    defaults = views.get("sensing") or {}
    changes = {
        key: value for key, value in defaults.items() if key in STATE_DEFAULTS and value is not None
    }
    if changes:
        update_state(viz, **changes)
    return SensingService(viz)


def _retire_service(viz: Any, service: Any) -> None:
    """Retire sensing presentation even when the next scenario disables it."""
    from ..services.evidence_focus import stop_evidence_focus

    stop_evidence_focus(viz)
    preparation = getattr(service, "preparation", None)
    if preparation is not None:
        preparation.close()
    viz.sensing_selection = None
    refresh_tracking_uncertainty(viz, clear=True)
    viz._tracking_uncertainty_scene = None
    renderer = getattr(viz, "renderer", None)
    if renderer is None:
        return
    service.clear_scene_overlays(renderer)
    adapter = _ANNOTATION_ADAPTERS.pop(renderer, None)
    if adapter is not None:
        adapter.retire()


def register() -> None:
    """Register the sensing runtime panel, service, and frame hooks."""
    register_runtime_extension(
        RuntimeFeatureExtension(
            key=FEATURE_KEY,
            panel_title="Sensing",
            tab_label="Sensing",
            panel_factory=_create_panel,
            service_factory=_create_service,
            enabled=_panel_enabled,
            sync_frame=_sync_frame,
            status_chips=_status_chips,
            refresh_hud=_refresh_hud,
            prepare_render_packet=_prepare_render_packet,
            retire_service=_retire_service,
            start_open=True,
        )
    )


def bind_fusion_layers(renderer, layers):
    """Reuse renderer-specific picking through the sensing extension boundary."""
    adapter = _annotation_adapter(renderer)
    if adapter is not None:
        adapter.bind_selection({}, layers)


def refresh_tracking_uncertainty(viz, *, clear=False):
    """Selection/toggle-only redraw from cached states; never fetch frame products."""
    scene = getattr(viz, "_tracking_uncertainty_scene", None)
    enabled = (
        not clear
        and bool(state_value(getattr(viz, "app_state", None), "show_tracking_uncertainty"))
        and bool(state_value(getattr(viz, "app_state", None), "show_detections"))
    )
    if scene is None:
        return
    if not enabled:
        if not scene.overlay.owned_ids and scene.frame is None:
            return
        scene.clear(viz.renderer)
    else:
        scene.refresh(
            viz.renderer,
            selection=getattr(viz, "sensing_selection", None),
            sigma=state_value(getattr(viz, "app_state", None), "uncertainty_sigma"),
            selected_only=state_value(getattr(viz, "app_state", None), "uncertainty_selected_only"),
        )
    panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("sensing")
    controls = getattr(panel, "uncertainty_controls", None)
    if controls is not None:
        controls.note.setText(scene.status)
    canvas = getattr(viz.renderer, "_canvas", None)
    if canvas is not None:
        canvas.request_draw()


def _sync_tracking_uncertainty(viz, service, step, raw_frame=None):
    """Use the native overlay's accepted frame or an exact verified world epoch."""
    enabled = bool(
        state_value(getattr(viz, "app_state", None), "show_tracking_uncertainty")
    ) and bool(state_value(getattr(viz, "app_state", None), "show_detections"))
    if not enabled:
        if getattr(viz, "_tracking_uncertainty_scene", None) is not None:
            refresh_tracking_uncertainty(viz, clear=True)
        return
    from .uncertainty_scene import TrackingUncertaintyScene

    scene = getattr(viz, "_tracking_uncertainty_scene", None)
    if scene is None:
        scene = viz._tracking_uncertainty_scene = TrackingUncertaintyScene()
    panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("sensing")
    from ..io.packed_frame_payload import visualizer_frame_provider
    from .dashboard_data import DashboardSession

    source = DashboardSession.provider_identity(visualizer_frame_provider(viz))
    frame = {}
    if raw_frame is not None and hasattr(service, "resolve_display_sensing_frame"):
        resolved = service.resolve_display_sensing_frame(
            raw_frame,
            scene_frame_id=step,
            storage_frame_id=step,
            product_names=("detections",),
            pin_freshest=False,
        )
        if resolved is not None:
            frame = resolved.display_frame
    window = getattr(panel, "_sensing_window", None)
    request = getattr(window, "_last_request", None)
    if (
        request is not None
        and request.source_identity == source
        and request.effective_display_step == step
    ):
        frame = request.display_frame
    # A stripped marker frame is not sufficient evidence for covariance. Only
    # a complete accepted native frame or the exact loaded dashboard request is used.
    result = (frame.get("sensing") or {}).get("temporal", {}).get("result", {})
    timestamp = result.get("reference_time_s")
    viz._tracking_uncertainty_step = step
    saved_scene = getattr(panel, "saved_track_scene", None)
    binding = getattr(saved_scene, "binding", None)
    if binding is not None:
        lineage = binding.reader["sources"][binding.sensor_id].get("lineage", ())
        exact = [i for i, row in enumerate(lineage) if row["scene_frame_id"] == step]
        timestamp = float(binding.reader.times[exact[0]]) if len(exact) == 1 else None
    scene.sync(
        viz.renderer,
        frame,
        binding=binding,
        time_s=timestamp,
        source=source,
        selection=getattr(viz, "sensing_selection", None),
        sigma=state_value(getattr(viz, "app_state", None), "uncertainty_sigma"),
        selected_only=state_value(getattr(viz, "app_state", None), "uncertainty_selected_only"),
    )
    controls = getattr(panel, "uncertainty_controls", None)
    if controls is not None:
        controls.note.setText(scene.status)
