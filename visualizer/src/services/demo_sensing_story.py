"""Alternate recorded sensing playback with an explanation of one saved report.

The ordinary sensing runtime owns detections and tracks. This presentation owns
only temporary visibility, a bounded saved-result reader and a paused-frame
camera reveal; it never runs a detector, tracker or signal reconstruction.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

from PySide6.QtWidgets import QCheckBox, QWidget
from shiboken6 import isValid

from shared.logging import get_logger

from ..experiments.cinematic_reveal import CinematicReveal
from ..io.packed_frame_payload import visualizer_frame_provider
from ..sensing.dashboard_data import DashboardSession
from ..sensing.runtime import state_value, update_state
from ..sensing.scene_explanation import SceneExplanationControls
from ..sensing.selection import inspection_cell, select_estimate
from ..sensing.selection_geometry import selected_world_position

logger = get_logger("orchav.demo.sensing")

_OVERLAY_SETTINGS = dict(
    show_detections=True,
    scene_mode="full",
    history_frames=64,
    full_history=True,
    show_velocity_arrows=True,
    show_track_labels=True,
    use_marker_glyphs=True,
)
_PHASE_SECONDS = 2.5
_READ_TIMEOUT_SECONDS = 15.0
_PLAYBACK_CAPTION = "Saved detections and tracks · recorded playback"


class _RevealOwner:
    """Keep cleanup bound to the renderer that created the temporary geometry."""

    def __init__(self, viz):
        self._viz = viz
        self.renderer = viz.renderer
        self.camera_controller = viz.camera_controller

    def __getattr__(self, name):
        return getattr(self._viz, name)


class DemoSensingStory:
    """A playback/reveal clock with no dependency on a visible sensing dashboard."""

    def __init__(self, viz, interval=12.0):
        self.viz = viz
        self.interval = max(1.0, float(interval))
        self.caption = _PLAYBACK_CAPTION
        self.session = None
        self.controls = None
        self.reveal = None
        self.request = None
        self.selection = None
        self._owner = _RevealOwner(viz)
        self._provider = None
        self._identity = None
        self._settings = None
        self._phase = "closed"
        self._next = 0.0
        self._deadline = 0.0
        self._report_index = 0
        self._camera_paused = False
        self._phases = ()
        self._phase_index = 0
        self._camera_history = None
        self._owned_transition = None

    @staticmethod
    def available(viz) -> bool:
        """Use already prepared recording metadata; never scan files in the UI."""
        if (
            getattr(getattr(viz, "renderer", None), "renderer_id", None) != "pygfx"
            or getattr(getattr(viz, "scenario", None), "data_mode", None) != "files"
            or visualizer_frame_provider(viz) is None
        ):
            return False
        service = getattr(viz, "extension_services", {}).get("sensing")
        if service is None or getattr(service, "preparation_pending", False):
            return False
        prepared = getattr(service, "prepared_scene", None)
        if prepared is not None and getattr(prepared, "records", ()):
            return True
        return bool(getattr(getattr(viz, "sensing_playback", None), "steps", ()))

    def start(self, now=None) -> None:
        """Capture display choices and start a small independent saved-result reader."""
        if self._phase != "closed":
            return
        if not self.available(self.viz):
            raise ValueError("Sensing story needs a prepared local sensing recording")
        now = perf_counter() if now is None else now
        viz = self.viz
        self._provider = visualizer_frame_provider(viz)
        self._identity = DashboardSession.provider_identity(self._provider)
        service = viz.extension_services["sensing"]
        panel = getattr(getattr(viz, "ui_manager", None), "panels", {}).get("sensing")
        self._settings = {key: state_value(viz.app_state, key) for key in _OVERLAY_SETTINGS}
        self.session = DashboardSession(
            self._provider,
            source_path=getattr(viz, "current_scenario_path", None),
            saved_result_path=getattr(panel, "saved_result_path", None),
            prepared=getattr(service, "prepared_sensing", None),
            cache_entry_limit=2,
            cache_byte_limit=64 * 1024 * 1024,
        )
        parent = viz if isinstance(viz, QWidget) else None
        shell = QCheckBox()
        self.controls = SceneExplanationControls(shell, parent)
        shell.setParent(self.controls)
        shell.hide()
        self.controls.hide()
        self.reveal = CinematicReveal(self._owner, self.controls)
        update_state(viz, **_OVERLAY_SETTINGS)
        viz.schedule_update()
        self._phase = "playback"
        self._next = now + self.interval
        self._set_playback(True)

    def _same_source(self):
        return (
            self.viz.renderer is self._owner.renderer
            and DashboardSession.provider_identity(visualizer_frame_provider(self.viz))
            == self._identity
        )

    def _set_playback(self, enabled):
        if bool(getattr(self.viz, "animation_running", False)) != enabled:
            self.viz.animation_controller.toggle_animation()

    def tick(self, now=None) -> None:
        """Consume ready results on Qt's thread and keep native work off this clock."""
        if self._phase == "closed":
            return
        now = perf_counter() if now is None else now
        if not self._same_source():
            self.close(restore=False)
            return
        if self._camera_paused:
            if self.session.pending is not None and self.session.pending.done():
                self.session.pending = None
            return
        try:
            if self._phase == "playback" and now >= self._next:
                if self.session.pending is not None:
                    if not self.session.pending.done():
                        return
                    self.session.pending = None
                step = self.viz.animation_step
                # Live mode plus one named product reads saved RD axes without
                # preparing dashboard plots, histories or detector diagnostics.
                self.session.request((step, "live", "primary", True, "auto", ("range_doppler",)))
                self._phase = "loading"
                self._deadline = now + _READ_TIMEOUT_SECONDS
            if self._phase == "loading":
                future = self.session.pending
                if future is not None and future.done():
                    self.session.pending = None
                    request = future.result()
                    if request is None or not self._prepare_report(request):
                        self._return_to_playback(now, restore_camera=False)
                        return
                    self._set_playback(False)
                    step = request.effective_display_step
                    steps = self.viz.get_available_animation_steps()
                    if step not in steps:
                        self._return_to_playback(now, restore_camera=False)
                        return
                    self.viz.cancel_pending_slider_scrub()
                    self.viz.ui_controller.handle_frame_input_changed(steps.index(step) + 1)
                    self._phase = "waiting_scene"
                    self._deadline = now + _READ_TIMEOUT_SECONDS
                elif now >= self._deadline:
                    self._return_to_playback(now, restore_camera=False)
                    return
            if self._phase == "waiting_scene":
                accepted = getattr(self.viz.extension_services.get("experiments"), "step", None)
                if accepted == self.request.effective_display_step:
                    self._start_reveal(now)
                elif now >= self._deadline:
                    self._return_to_playback(now, restore_camera=False)
            elif self._phase == "reveal" and now >= self._next:
                self._phase_index += 1
                if self._phase_index >= len(self._phases):
                    self._return_to_playback(now, restore_camera=True)
                else:
                    self._show_phase(self._phases[self._phase_index])
                    self._next = now + _PHASE_SECONDS
        except (
            CancelledError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            OSError,
        ) as exc:
            logger.warning("Skipping sensing Demo reveal: %s", exc)
            self._return_to_playback(now, restore_camera=True)

    def _prepare_report(self, request):
        if request.source_identity != self._identity:
            return False
        sensing = request.display_frame.get("sensing") or {}
        detections = sensing.get("detections") or []
        if not detections:
            return False
        index = self._report_index % len(detections)
        self._report_index += 1
        row = detections[index]
        cell = None
        r, d = row.get("range_bin"), row.get("doppler_bin")
        if isinstance(r, int) and not isinstance(r, bool):
            cell = inspection_cell(request, r, d, origin="detection")
        self.request = request
        self.selection = replace(
            select_estimate(sensing, detection_index=index), source=self._identity, cell=cell
        )
        return True

    def _start_reveal(self, now):
        tools = self._owner.camera_controller.presentation()
        self._camera_history = list(tools.history)
        self.controls.set_playing(False)
        self.controls.bind_context(self.request, self.selection)
        self.controls.play_generation += 1
        self._show_phase("range")
        self._phases = tuple(self.reveal.stages)
        if not self._phases:
            self._return_to_playback(now, restore_camera=False)
            return
        self._phase_index = self._phases.index(self.reveal.phase)
        self._phase = "reveal"
        self._next = now + _PHASE_SECONDS

    def _show_phase(self, phase):
        self.controls.phase.setCurrentIndex(self.controls.phase.findData(phase))
        dashboard = SimpleNamespace(
            selected_world_position=lambda: selected_world_position(self.request, self.selection)
        )
        self.reveal.sync(self.request, self.selection, dashboard)
        self._owned_transition = (
            self.reveal.camera.transition if self.reveal.camera is not None else None
        )
        self.caption = self.controls.note.text()

    def _return_to_playback(self, now, *, restore_camera):
        if self.reveal is not None:
            if restore_camera and not self._camera_paused and self.reveal.original is not None:
                self.reveal.restore()
            else:
                self.reveal.clear(keep_restore=False)
        if self.controls is not None:
            self.controls.stop(notify=False)
        self._restore_camera_history()
        self._owned_transition = None
        self.request = self.selection = None
        self._phase = "playback"
        self._next = now + self.interval
        self.caption = _PLAYBACK_CAPTION
        if self._same_source():
            self._set_playback(True)

    def pause_camera(self) -> None:
        """Keep manual framing until Resume camera explicitly permits another reveal."""
        self._camera_paused = True
        if self._phase != "closed":
            self._return_to_playback(perf_counter(), restore_camera=False)

    def resume_camera(self) -> None:
        self._camera_paused = False
        self._next = perf_counter() + self.interval

    def _restore_camera_history(self):
        if self._camera_history is not None:
            self._owner.camera_controller.presentation().history = self._camera_history
            self._camera_history = None

    def close(self, restore=True) -> None:
        """Retire owned work and restore display settings only into the same source."""
        same_source = self._identity is not None and self._same_source()
        if self.reveal is not None:
            if (
                restore
                and same_source
                and not self._camera_paused
                and self.reveal.original is not None
            ):
                self.reveal.restore()
            if not restore or not same_source:
                # The replacement scene may already own this camera. Retire
                # only our transition; never replay its old pose or options.
                camera = self.reveal.camera
                if camera is not None and isValid(camera):
                    try:
                        camera.status_changed.disconnect(self.reveal._camera_status)
                    except (RuntimeError, TypeError):
                        pass  # The owned controls may already have been destroyed.
                    if (
                        self._owned_transition is not None
                        and camera.transition is self._owned_transition
                    ):
                        camera.stop()
                self.reveal.camera = None
            self.reveal.close()
            self.reveal = None
        if restore and same_source:
            self._restore_camera_history()
        else:
            self._camera_history = None
        if self.controls is not None:
            self.controls.deleteLater()
            self.controls = None
        if self.session is not None:
            self.session.close()
            self.session = None
        if restore and same_source and self._settings is not None:
            update_state(self.viz, **self._settings)
            self.viz.schedule_update()
        self._settings = None
        self._owned_transition = None
        self.request = self.selection = None
        self._phase = "closed"
