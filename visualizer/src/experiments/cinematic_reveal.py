"""Presentation of the existing saved-bin reveal; camera-only interpolation."""

from dataclasses import replace
from time import perf_counter

import numpy as np
from PySide6.QtCore import QSignalBlocker, QTimer
from shiboken6 import isValid

from ..services.paused_presentation import frames_advancing
from ..types.camera_state import CameraState
from .range_geometry import build_range_locus
from .report_geometry import build_angular_cell


class CinematicReveal:
    """Reuse CameraPresentation transitions and the existing reveal's phase clock."""

    def __init__(self, viz, controls):
        self.viz, self.controls = viz, controls
        self.overlay = None
        self.camera = None
        self.original = None
        self.original_mode = None
        self.original_options = None
        self.identity = None
        self.source = None
        self.stages = {}
        self.phase = None
        self.internal_camera_action = False
        self.fade = QTimer(controls)
        self.fade.setInterval(16)
        self.fade.timeout.connect(self._fade_tick)
        self.fade_start = 0.0
        self.fade_from = {}
        self.fade_to = {}
        self.paused_at = None
        self.play_generation = None
        controls.destroyed.connect(self.close)
        controls.restore_camera_requested.connect(self.restore)
        controls.takeover_requested.connect(self.takeover)
        controls.pause_changed.connect(self.pause)

    def _camera(self):
        if self.camera is None:
            owner = getattr(self.viz, "camera_controller", None)
            self.camera = owner.presentation() if owner is not None else None
            if self.camera is None:
                raise ValueError("Smooth camera presentation is unavailable")
            self.camera.status_changed.connect(self._camera_status)
        return self.camera

    def sync(self, request, selection, dashboard):
        if frames_advancing(self.viz):
            self.controls.set_playing(True)
            self.clear()
            return
        if self.source != request.source_identity:
            self.clear(keep_restore=False)
            self.source = request.source_identity
        identity = self.controls.context
        if self.identity != identity:
            self.clear()
            self.identity = identity
        phase = self.controls.phase.currentData()
        if phase == "off":
            self.clear()
            return
        if self.play_generation != self.controls.play_generation:
            # Each deliberate replay starts from the user's current view, even
            # when the only available stage is again the saved position.
            self.clear(keep_restore=False)
            self.play_generation = self.controls.play_generation
        if not self.stages:
            self._prepare(request, selection, dashboard)
        available = tuple(self.stages)
        self.controls.available_phases = available
        if not available:
            raise ValueError("No recorded range, coarse-angle or spatial-estimate geometry")
        if phase not in self.stages:
            order = ("range", "angle", "position")
            remaining = [p for p in available if order.index(p) >= order.index(phase)]
            if not remaining:
                self.controls.timer.stop()
                self.controls.play.setText("Play reveal")
                return
            phase = remaining[0]
            with QSignalBlocker(self.controls.phase):
                self.controls.phase.setCurrentIndex(self.controls.phase.findData(phase))
        if self.phase == phase:
            return
        self.phase = phase
        center, extent, label = self.stages[phase]
        camera = self._camera()
        if self.original is None:
            self.original = self.viz.renderer.get_camera_state()
            if self.original is None:
                raise ValueError("Camera is not ready")
            self.original_mode = getattr(self.viz.app_state, "camera_mode", "overview")
            self.original_options = (camera.smooth, camera.duration)
            self.controls.restore_camera.setEnabled(True)
        current = self.viz.renderer.get_camera_state()
        direction = np.asarray(self.original.eye) - self.original.lookat
        direction /= np.linalg.norm(direction)
        # Keep the current viewing direction and a conservative standoff. No
        # orbit or target-mesh intersection assumptions are introduced.
        radius = max(8.0, float(extent) / max(np.sin(np.radians(current.fov_deg / 2)), 0.1) * 1.25)
        goal = CameraState(
            tuple(center + direction * radius), tuple(center), current.up, current.fov_deg
        )
        self.internal_camera_action = True
        try:
            camera.prepare()
            camera.smooth, camera.duration = True, 0.85
            self.viz.set_state(camera_mode="overview")
            self.viz.renderer.set_camera_state(goal)
        finally:
            self.internal_camera_action = False
        targets = {
            "range": 0.16 if phase == "angle" else (1.0 if phase == "range" else 0.0),
            "angle": 1.0 if phase == "angle" else 0.0,
            "position": 1.0 if phase == "position" else 0.0,
            "track": 1.0 if phase == "position" else 0.0,
        }
        self.fade_from = {name: self._opacity(name) for name in targets}
        self.fade_to = targets
        self.fade_start = perf_counter()
        self.fade.start()
        self.controls.note.setText(label)
        self.viz.renderer.set_hud_overlay(
            "cinematic_sensing_reveal",
            html=label,
            visible=True,
            role="status",
            corner="top_left",
            priority=20,
        )

    def _prepare(self, request, selection, dashboard):
        from ..renderers.pygfx.experiments import ExperimentOverlay

        self.overlay = ExperimentOverlay(self.viz.renderer)
        try:
            locus = build_range_locus(request, selection)
            self.overlay.range_locus(locus)
            points = np.concatenate([p for p, _ in locus.surfaces])
            self.stages["range"] = (
                np.asarray(locus.center),
                np.linalg.norm(np.ptp(points, axis=0)) / 2,
                f"RANGE — recorded range bin · {locus.label}",
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        try:
            cell = build_angular_cell(request, selection)
            self.overlay.angular_cell(cell)
            self.stages["angle"] = (
                np.mean(cell.vertices, axis=0),
                np.linalg.norm(np.ptp(cell.vertices, axis=0)) / 2,
                f"ANGLE — recorded coarse angular bin · {cell.label}",
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        point = dashboard.selected_world_position()
        if point is not None and np.shape(point) == (3,) and np.isfinite(point).all():
            self.overlay.report_position(point)
            coordinates = ", ".join(f"{float(value):.3g}" for value in point)
            label = (
                f"POSITION — saved spatial estimate · world ({coordinates}) m"
                " · not actual target geometry"
            )
            center, extent = np.asarray(point), 3.0
            if selection is not None and selection.track_id is not None:
                track_point = self._associated_track_position(request, selection)
                label += f" · TRACK — associated saved track {selection.track_id}"
                if track_point is not None:
                    from ..sensing.display_style import identity_color

                    self.overlay.track_position(track_point, identity_color(selection.track_id))
                    center = (np.asarray(point) + track_point) / 2
                    extent = max(3.0, float(np.linalg.norm(track_point - point)) / 2 + 1.5)
                    values = ", ".join(f"{float(value):.3g}" for value in track_point)
                    label += f" · world ({values}) m"
                else:
                    label += " (3D state unavailable)"
            self.stages["position"] = (center, extent, label)
        for name in self.overlay.layers:
            self._set_opacity(name, 0.0)

    @staticmethod
    def _associated_track_position(request, selection):
        """Use only an exact native association, never match world-run IDs by number."""
        from ..sensing.selection import selection_scope
        from ..sensing.selection_geometry import selected_world_position
        from ..sensing.tracking_presentation import tracking_snapshot
        from ..sensing.tracking_uncertainty import native_epoch_valid

        sensing = request.display_frame.get("sensing") or {}
        temporal = sensing.get("temporal", {})
        if (
            selection.source != request.source_identity
            or selection.scope != selection_scope(sensing)
            or selection.acquisition_id != temporal.get("acquisition", {}).get("acquisition_id")
            or selection.detection_collection
            != sensing.get("_display_detection_collection", "detections")
        ):
            return None
        try:
            snapshot = tracking_snapshot(sensing)
            if (
                snapshot.complete_states
                and not native_epoch_valid(
                    sensing, temporal.get("result", {}).get("reference_time_s")
                )[0]
            ):
                return None
            state = next(
                (row for row in snapshot.states if row["track_id"] == selection.track_id), None
            )
            if state is None or np.shape(state.get("position_m")) != (3,):
                return None
            return selected_world_position(
                request, replace(selection, detection_index=None), track=state
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    def _opacity(self, name):
        obj = self.overlay.layers.get(name) if self.overlay is not None else None
        return float(getattr(obj, "_reveal_opacity", 0.0)) if obj is not None else 0.0

    def _set_opacity(self, name, value):
        obj = self.overlay.layers.get(name) if self.overlay is not None else None
        if obj is None:
            return
        obj._reveal_opacity = value
        obj.visible = value > 0.001
        for node in obj.iter():
            material = getattr(node, "material", None)
            if material is not None:
                if not hasattr(node, "_reveal_base_opacity"):
                    node._reveal_base_opacity = material.opacity
                material.opacity = node._reveal_base_opacity * value

    def _fade_tick(self, now=None):
        if not isValid(self.controls) or frames_advancing(self.viz):
            self.clear()
            return
        t = min(1.0, ((perf_counter() if now is None else now) - self.fade_start) / 0.4)
        t = t * t * (3 - 2 * t)
        for name, end in self.fade_to.items():
            self._set_opacity(name, self.fade_from[name] * (1 - t) + end * t)
        if t >= 1:
            self.fade.stop()
        self.viz.renderer.request_redraw()

    def pause(self, paused):
        if not isValid(self.controls) or not isValid(self.fade):
            self.close()
            return
        if paused:
            self.paused_at = perf_counter()
            self.fade.stop()
            if self.camera is not None:
                self.camera.timer.stop()
        elif self.paused_at is not None:
            delay = perf_counter() - self.paused_at
            self.fade_start += delay
            if self.camera is not None and self.camera.transition is not None:
                self.camera.transition[2] += delay
                self.camera.timer.start()
            if self.fade_to:
                self.fade.start()
            self.paused_at = None

    def _camera_status(self, status):
        if not isValid(self.controls):
            self.close()
            return
        if status == "Stopped" and not self.internal_camera_action and self.phase is not None:
            self.takeover()
            self.controls.stop()

    def takeover(self):
        self.clear(keep_restore=True)

    def clear(self, *, keep_restore=True):
        fade_valid = isValid(self.fade)
        was_active = self.phase is not None or (fade_valid and self.fade.isActive())
        if fade_valid:
            self.fade.stop()
        self.paused_at = None
        self.fade_from.clear()
        self.fade_to.clear()
        self.phase = None
        if self.camera is not None and isValid(self.camera) and was_active:
            self.internal_camera_action = True
            self.camera.stop()
            self.internal_camera_action = False
            if self.original_options is not None:
                self.camera.smooth, self.camera.duration = self.original_options
        if self.overlay is not None:
            self.overlay.clear()
        self.overlay = None
        self.stages.clear()
        self.viz.renderer.clear_hud_overlay("cinematic_sensing_reveal")
        if not keep_restore:
            self.original = self.original_mode = self.original_options = None
            if isValid(self.controls):
                self.controls.restore_camera.setEnabled(False)

    def restore(self):
        if not isValid(self.controls):
            self.close()
            return
        original, mode = self.original, self.original_mode
        self.controls.stop()
        self.clear(keep_restore=False)
        if original is not None:
            camera = self._camera()
            camera.apply(original)
            self.viz.set_state(camera_mode=mode)

    def close(self):
        # The control widget owns the fade timer, so it may already be deleted.
        camera = self.camera
        if camera is not None and isValid(camera):
            try:
                camera.status_changed.disconnect(self._camera_status)
            except (RuntimeError, TypeError):
                pass
        self.clear(keep_restore=False)
        self.camera = None
