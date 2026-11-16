"""Cancellable startup preparation; only the Qt poller touches the panel."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from threading import Event

from PySide6.QtCore import QTimer

from shared.frames.providers import Hdf5Provider
from shared.logging import get_logger

from ..io.packed_frame_payload import visualizer_frame_provider
from .dashboard_data import DashboardSession
from .playback import prepared_playback_plan
from .preparation import prepare_sensing
from .scene_recording import provider_token

logger = get_logger(__name__)


class SensingPreparation:
    """Keep costly source preparation outside playback and view-change callbacks."""

    def __init__(self, panel, *, prepare_playback=True):
        self.panel = panel
        self.cancelled = Event()
        self.future = self.executor = None
        self.progress = ("Preparing sensing", 0, 0)
        self.timer = QTimer(panel._container)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.service = panel._get_service()
        self.provider = visualizer_frame_provider(panel.parent)
        if self.service is None or not isinstance(self.provider, Hdf5Provider):
            return
        self.token = provider_token(self.provider)
        # Explicit 3D loading must leave an already active metadata timeline alone.
        self.playback_plan = Future() if prepare_playback else None
        playback = getattr(panel, "playback", None)
        if playback is not None and self.playback_plan is not None:
            playback.prepare_for_scenario(self.provider, self.playback_plan)
        self.service.preparation = self
        self.service.preparation_pending = True
        retained = (
            getattr(self.provider, "_prepared_sensing_dashboard", None)
            or getattr(panel, "_dashboard_cache", None)
            or getattr(panel, "_dashboard_session", None)
        )
        prepared = getattr(retained, "prepared", None)
        if prepared is not None and prepared.scene.matches_provider(self.provider):
            self.progress = (
                "Sensing ready",
                len(prepared.scene.records),
                len(prepared.scene.records),
            )
            self._scene_ready(prepared.scene)
            self.future = Future()
            self.future.set_result(prepared)
            self.poll()
            return
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensing-prepare")
        self.future = self.executor.submit(self._prepare)
        self._display_progress()
        self.timer.start()

    def _prepare(self):
        """Publish navigation before separately saved tracking finishes."""
        if self.playback_plan is not None and not self.playback_plan.set_running_or_notify_cancel():
            raise CancelledError()
        try:
            return prepare_sensing(self.provider, self.cancelled, self._progress, self._scene_ready)
        except Exception as exc:
            if self.playback_plan is not None and not self.playback_plan.done():
                self.playback_plan.set_exception(exc)
            raise

    def _scene_ready(self, scene):
        if self.cancelled.is_set() or provider_token(self.provider) != self.token:
            raise CancelledError()
        if self.playback_plan is not None:
            self.playback_plan.set_result(prepared_playback_plan(scene))

    def _progress(self, stage, done, total):
        self.progress = stage, done, total

    def _display_progress(self):
        checkbox = self.panel.widgets.get("show_sensing_detections")
        if checkbox is not None:
            stage, done, total = self.progress
            checkbox.setText(f"{stage}…" + (f" {done}/{total}" if total else ""))
            checkbox.setEnabled(False)

    def poll(self):
        if self.future is None:
            return
        if (
            self.cancelled.is_set()
            or self.panel._closed
            or visualizer_frame_provider(self.panel.parent) is not self.provider
            or provider_token(self.provider) != self.token
        ):
            self.close()
            return
        if not self.future.done():
            self._display_progress()
            return
        future, self.future = self.future, None
        self.timer.stop()
        if self.executor is not None:
            self.executor.shutdown(wait=False)
        self.executor = None
        error = None
        try:
            prepared = future.result()
            self.service.prepared_scene = prepared.scene
            self.service.prepared_sensing = prepared
            session = getattr(self.panel, "_dashboard_session", None)
            if session is not None and session.provider is self.provider:
                session.prepared = prepared
            self.panel.saved_track_scene.install(
                prepared.tracking,
                prepared.tracking_error,
                DashboardSession.provider_identity(self.provider),
            )
        except CancelledError:
            pass
        except Exception as exc:  # Worker boundary: retain the validated on-demand reader.
            logger.exception("Could not prepare sensing display cache")
            error = f"Sensing preparation unavailable: {exc}"
        self.service.preparation_pending = False
        checkbox = self.panel.widgets.get("show_sensing_detections")
        if checkbox is not None:
            checkbox.setText("Show 3D sensing")
            checkbox.setEnabled(True)
            if error:
                checkbox.setToolTip(error + ". Views will load on demand.")
        viz = self.panel.parent
        if error:
            viz._set_status_message(error, 8000)
        viz.force_update_next_frame = True
        viz.schedule_update()

    def close(self):
        if self.cancelled.is_set():
            return
        self.cancelled.set()
        plan = getattr(self, "playback_plan", None)
        if plan is not None:
            plan.cancel()
        self.timer.stop()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = self.future = None
        if self.service is not None:
            self.service.preparation_pending = False
