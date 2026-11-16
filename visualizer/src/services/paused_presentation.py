"""Stop optional paused-frame effects at the existing playback boundary."""

from PySide6.QtCore import QObject
from shiboken6 import isValid


def _alive(obj):
    return obj is not None and (not isinstance(obj, QObject) or isValid(obj))


def frames_advancing(viz):
    """Include normal playback, live playback and the service-level clock."""
    controller = getattr(viz, "animation_controller", None)
    timer = getattr(controller, "_live_playback_timer", None)
    return bool(
        getattr(viz, "animation_running", False)
        or (_alive(timer) and timer.isActive())
        or getattr(viz, "_presentation_playback_active", False)
    )


def playback_changed(viz, playing):
    """No timers of our own; notify only already-created optional controls."""
    viz._presentation_playback_active = bool(playing)
    focus = getattr(viz, "_evidence_focus", None)
    if focus is not None:
        focus.set_playing(bool(playing))
    panels = getattr(getattr(viz, "ui_manager", None), "panels", {})
    panel = panels.get("sensing")
    if panel is not None:
        focus_control = getattr(panel, "widgets", {}).get("evidence_focus")
        if _alive(focus_control):
            focus_control.setEnabled(not playing)
        controls = getattr(panel, "uncertainty_controls", None)
        if _alive(controls):
            controls.set_playing(bool(playing))
        window = getattr(panel, "_sensing_window", None)
        reveal = getattr(window, "scene_explanation", None)
        if _alive(reveal) and hasattr(reveal, "set_playing"):
            reveal.set_playing(bool(playing))
    service = getattr(viz, "extension_services", {}).get("experiments")
    if service is not None and hasattr(service, "set_presentation_playing"):
        service.set_presentation_playing(bool(playing))
