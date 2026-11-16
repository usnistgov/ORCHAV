"""Reversible presentation dimming of known evidence overlays on a paused frame."""

import math
from html import escape

from PySide6.QtCore import QSignalBlocker
from shiboken6 import isValid

from .paused_presentation import frames_advancing


class EvidenceFocus:
    """No timers, file reads, geometry edits or changes to physical scene materials."""

    DIM = 0.15
    HUD = "evidence_focus"
    TRACK_PREFIXES = (
        "sensing_track_point_",
        "sensing_track_trail_",
        "sensing_track_arrow_",
        "sensing_track_label_",
        "sensing_saved_track_point_",
        "sensing_saved_track_trail_",
        "sensing_saved_track_arrow_",
        "sensing_saved_track_label_",
    )

    def __init__(self, viz, control=None):
        self.viz = viz
        self.control = control
        self.active = False
        self.renderer = None
        self.identity = None
        self.saved = {}
        self.selected_track_id = None
        self.status = "Focus off"

    def _background(self, name):
        if name in ("mpc_lines", "mpc_points", "sensing_detections", "sensing_evidence"):
            return True
        if name.startswith("sensing_uncertainty_") and self.selected_track_id is not None:
            suffix = name[len("sensing_uncertainty_") :]
            selected = str(self.selected_track_id)
            return suffix != selected and not suffix.startswith(selected + "_")
        if name.startswith("sensing_truth_"):
            return True
        for prefix in self.TRACK_PREFIXES:
            if name.startswith(prefix):
                return name[len(prefix) :] != str(self.selected_track_id)
        return False

    @staticmethod
    def _materials(obj):
        nodes = obj.iter() if hasattr(obj, "iter") else (obj,)
        return {id(m): m for node in nodes if (m := getattr(node, "material", None)) is not None}

    def start(self, *, identity, selected_track_id=None, result_time_s=None):
        self.stop()
        renderer = getattr(self.viz, "renderer", None)
        if (
            identity[1] is None
            or frames_advancing(self.viz)
            or getattr(renderer, "renderer_id", None) != "pygfx"
        ):
            self.status = "Pause playback and use pygfx to focus selected evidence"
            return False
        if getattr(renderer, "_qt_window_closed", False):
            return False
        self.renderer = renderer
        self.identity = identity
        self.selected_track_id = selected_track_id
        self.active = True
        if self.control is not None and isValid(self.control):
            with QSignalBlocker(self.control):
                self.control.setChecked(True)
        self.refresh(identity)
        if not self.active:
            return False
        source, step = identity[:2]
        time_label = "result time unavailable"
        if isinstance(result_time_s, (int, float)) and math.isfinite(result_time_s):
            time_label = f"saved result {result_time_s:.6g} s"
        track = "selected evidence" if selected_track_id is None else f"track {selected_track_id}"
        self.status = f"Focus: {track} · source epoch {source} · frame {step} · {time_label}"
        renderer.set_hud_overlay(
            self.HUD,
            html=escape(self.status)
            + "<br>Bright: selected / explanatory overlays · Dim: background evidence; physical scene unchanged",
            visible=True,
            role="status",
            corner="top_left",
            priority=30,
        )
        renderer.request_redraw()
        return True

    def refresh(self, identity):
        if not self.active:
            return
        if (
            identity != self.identity
            or frames_advancing(self.viz)
            or getattr(self.viz, "renderer", None) is not self.renderer
            or getattr(self.renderer, "_qt_window_closed", False)
        ):
            self.stop()
            return
        objects = getattr(self.renderer, "_objects", {})
        protected, background = {}, {}
        for name, obj in objects.items():
            target = background if self._background(str(name)) else protected
            target.update(self._materials(obj))
        # Never mutate a material shared with physical geometry or selected evidence.
        eligible = {key: material for key, material in background.items() if key not in protected}
        changed = False
        for key, (material, original, applied) in list(self.saved.items()):
            if key not in eligible:
                changed = self._restore(material, original, applied) or changed
                self.saved.pop(key)
        for key, material in eligible.items():
            opacity = float(material.opacity)
            previous = self.saved.get(key)
            if previous is not None and math.isclose(opacity, previous[2], abs_tol=1e-7):
                continue
            # A later appearance edit becomes the new original, never stale state.
            applied = opacity * self.DIM
            material.opacity = applied
            self.saved[key] = (material, opacity, float(material.opacity))
            changed = True
        if changed:
            self.renderer.request_redraw()

    @staticmethod
    def _restore(material, original, applied):
        if math.isclose(float(material.opacity), applied, abs_tol=1e-7):
            material.opacity = original
            return True
        return False

    def stop(self):
        was_active = self.active
        changed = False
        for material, original, applied in self.saved.values():
            changed = self._restore(material, original, applied) or changed
        self.saved.clear()
        renderer = self.renderer
        self.renderer = None
        self.identity = None
        self.active = False
        if self.control is not None and isValid(self.control):
            with QSignalBlocker(self.control):
                self.control.setChecked(False)
        if renderer is not None:
            renderer.clear_hud_overlay(self.HUD)
            if (changed or was_active) and not getattr(renderer, "_qt_window_closed", False):
                renderer.request_redraw()
        self.status = "Focus off"

    close = stop

    def set_playing(self, playing):
        if playing:
            self.stop()
        if self.control is not None and isValid(self.control):
            self.control.setEnabled(not playing)


def focus_identity(viz, step=None):
    """Already accepted source epoch and frame, not a new provider read."""
    if step is None:
        step = getattr(viz, "_evidence_focus_presented_step", None)
    panels = getattr(getattr(viz, "ui_manager", None), "panels", {})
    panel = panels.get("sensing")
    request = getattr(getattr(panel, "_sensing_window", None), "_last_request", None)
    return (getattr(viz, "_mpc_presented_source_epoch", 0), step, id(request))


def stop_evidence_focus(viz):
    focus = getattr(viz, "_evidence_focus", None)
    if focus is not None:
        focus.stop()


def selected_focus_track(selection, request, source, step, saved_scene=None):
    """Keep a track bright only when its saved source/run/epoch identity matches."""
    from ..sensing.fusion_inspection import FusionSelection
    from ..sensing.selection import SensingSelection, selection_scope

    if isinstance(selection, FusionSelection):
        binding = getattr(saved_scene, "binding", None)
        source_key = getattr(saved_scene, "source_key", ()) or ()
        if binding is None or tuple(source_key[:-1]) != tuple(source):
            return None
        if selection.run_id != binding.reader.inspection_id:
            return None
        lineage = binding.reader["sources"][binding.sensor_id].get("lineage", ())
        exact = [index for index, row in enumerate(lineage) if row["scene_frame_id"] == step]
        return selection.track_id if exact == [selection.epoch_index] else None
    if not isinstance(selection, SensingSelection) or request is None:
        return None
    if (
        request.source_identity != source
        or request.effective_display_step != step
        or selection.source != source
    ):
        return None
    sensing = request.display_frame.get("sensing", {})
    if selection.scope != selection_scope(sensing):
        return None
    if selection.detection_index is not None:
        acquisition = sensing.get("temporal", {}).get("acquisition", {})
        if selection.acquisition_id != acquisition.get(
            "acquisition_id"
        ) or selection.detection_collection != sensing.get(
            "_display_detection_collection", "detections"
        ):
            return None
    return selection.track_id
