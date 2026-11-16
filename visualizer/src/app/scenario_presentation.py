"""Apply portable scenario presentation through the ordinary GUI owners."""

import json

from shared.logging import get_logger

from ..materials.texture_policy import textures_locked_off
from ..renderers.protocol import renderer_capabilities
from ..types.camera_state import CameraState

logger = get_logger(__name__)


def load_appearance_recipe(defaults, policy, service):
    """Read and validate a portable recipe for either visualizer workspace."""
    recipe = defaults.get("appearance_recipe")
    if not recipe or service is None:
        return None
    path = policy.resolve_path(recipe)
    if path.stat().st_size > 65536:
        raise ValueError("Scenario appearance recipe is too large")
    data = service.validate(json.loads(path.read_text()))
    if textures_locked_off():
        data["textures"] = False
    return data


def apply_camera_pose(renderer, defaults):
    """Apply the shared explicit-pose contract and report whether one was given."""
    pose = defaults.get("camera_pose")
    if not pose:
        return False
    renderer.set_camera_state(CameraState.from_dict({"up": [0, 0, 1], "fov": 60, **pose}))
    return True


def apply_camera_defaults(viz, defaults):
    """Set an initial view without disabling navigation or starting camera motion."""
    if apply_camera_pose(viz.renderer, defaults):
        return
    view = defaults.get("camera_view")
    if view:
        viz.default_camera_view = view
        viz.default_camera_dist = defaults.get("camera_dist")
        viz.default_camera_fov = defaults.get("fov")
        viz.apply_camera_view(view, camera_dist=viz.default_camera_dist, fov=viz.default_camera_fov)


def apply_presentation_defaults(viz, defaults):
    """Restore optional local appearance and cadence; keep playback paused."""
    renderer = getattr(viz, "renderer", None)
    if renderer_capabilities(renderer).dusk_lighting:
        renderer.configure_dusk(defaults.get("dusk"))
    recipe = defaults.get("appearance_recipe")
    service = getattr(viz, "appearance_preset_service", None)
    if recipe and service is not None:
        try:
            data = load_appearance_recipe(defaults, viz.current_scenario_policy, service)
            service.restore(data, name="Scenario appearance")
            viz.renderer.refocus_shadows()
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Could not apply scenario appearance %s: %s", recipe, exc)
    playback = defaults.get("playback")
    panels = getattr(getattr(viz, "ui_manager", None), "panels", {})
    if playback and "animation" in panels:
        widgets = panels["animation"].widgets
        mode = widgets["playback_mode_combo"]
        mode.setCurrentIndex(mode.findData(playback.get("mode", "fixed_fps")))
        widgets["playback_fps_spinbox"].setValue(playback.get("fps", 25))
        stride = f"x{playback.get('stride', 1)}"
        plan = getattr(viz, "sensing_playback", None)
        if plan is not None:
            plan.set_full_timeline_stride(stride)
        else:
            widgets["stride_combo"].setCurrentText(stride)
    render_panel = panels.get("render")
    sync_effects = getattr(render_panel, "sync_presentation_effects", None)
    if callable(sync_effects):
        sync_effects()
    nodes = panels.get("nodes")
    controls = nodes.widgets.get("label_style_controls") if nodes is not None else None
    if controls is not None:
        controls.sync_from_state()
    refresh = getattr(nodes, "refresh_display_controls", None)
    if callable(refresh):
        refresh()
