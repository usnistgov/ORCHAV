"""Optional playback over saved sensing scene bindings, with selective preload.

Indexing uses a detached service on a worker. Only the Qt polling callback
changes navigation widgets; cancelled or replaced recordings cannot install
their results into the current timeline.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from PySide6.QtCore import QSignalBlocker, QTimer

from shared.frames import FrameComponent, FrameReadRequest

from ..io.packed_frame_payload import visualizer_frame_provider
from ..playback import timestamp_interval_ms
from ..services.sensing_service import SensingService


def _provider_token(provider) -> tuple:
    info = provider.info
    return (
        id(provider),
        info.generation_id,
        info.frame_set_id,
        getattr(provider, "_generation_epoch", None),
        id(provider.frame_ids_snapshot),
    )


def load_sensing_playback_plan(provider, cancelled: Event) -> tuple[tuple[int, ...], dict]:
    """Read result bindings and scene timestamps without signal/geometry arrays."""
    proxy = SimpleNamespace(frame_source=SimpleNamespace(provider=provider))
    service = SensingService(proxy)
    service.configure_background_reader(cancelled, lambda *_: None)
    steps = service.sensing_playback_frame_ids()
    timestamps = {}
    request = FrameReadRequest(components=frozenset({FrameComponent.PROVENANCE}))
    projections = provider.iter_frame_projections(steps, request)
    try:
        for step, projection in zip(steps, projections, strict=True):
            service.report_read_progress("Indexing sensing frames", 0, len(steps))
            if projection.frame.frame_index != step:
                raise ValueError("Sensing timeline projection changed frame order")
            timestamps[step] = projection.frame.timestamp_s
    finally:
        close = getattr(projections, "close", None)
        if callable(close):
            close()
    return steps, timestamps


def prepared_playback_plan(scene) -> tuple[tuple[int, ...], dict[int, float]]:
    """Reuse verified bound-scene timestamps, including zero-detection results."""
    timestamps = {
        frame["_sensing_display_provenance"]["bound_scene_frame_id"]: frame["timestamp_s"]
        for frame in scene.records
        if isinstance(frame["_sensing_display_provenance"]["bound_scene_frame_id"], int)
        and not isinstance(frame["_sensing_display_provenance"]["bound_scene_frame_id"], bool)
    }
    return tuple(sorted(timestamps)), timestamps


class SensingPlayback:
    """Own one recording's optional sensing-only navigation and preload plan."""

    def __init__(self, visualizer, checkbox) -> None:
        self.visualizer = visualizer
        self.checkbox = checkbox
        self.steps: tuple[int, ...] | None = None
        self.timestamps: dict[int, float] = {}
        self._future: Future | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._cancelled = Event()
        self._token: tuple | None = None
        self._cached_plan: tuple | None = None
        self._prepared_future: Future | None = None
        self._prepared_token: tuple | None = None
        self._automatic = False
        self._full_timeline_stride = None
        self._startup_error: Callable[[str], None] | None = None
        self._timer = QTimer(checkbox)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        checkbox.toggled.connect(self.set_enabled)

    @property
    def preparing(self) -> bool:
        """Wait until the saved timeline is installed on the GUI thread."""
        return self.steps == () and self._future is not None

    def interval_ms(self, current: int, following: int) -> int | None:
        """Keep physical cadence even before the selected frames are cached."""
        return timestamp_interval_ms(self.timestamps.get(current), self.timestamps.get(following))

    def enable_for_startup(self, *, on_error: Callable[[str], None]) -> None:
        """Enable before preload; a CLI failure must never load the full recording."""
        self._startup_error = on_error
        with QSignalBlocker(self.checkbox):
            self.checkbox.setChecked(True)
        self.set_enabled(True)

    def prepare_for_scenario(self, provider, future: Future) -> None:
        """Default local recordings to saved results while startup prepares them.

        The shared preparation worker owns this future. Toggling the filter
        never cancels that work or repeats its source indexing.
        """
        self._prepared_token = _provider_token(provider)
        self._prepared_future = future
        if getattr(self.visualizer, "_cli_driven_frame_run", False):
            return
        with QSignalBlocker(self.checkbox):
            self.checkbox.setChecked(True)
        self.set_enabled(True, automatic=True)

    def _cancel_read(self) -> None:
        self._cancelled.set()
        if self._future is not None and self._future is not self._prepared_future:
            self._future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._future = None
        self._executor = None
        self._timer.stop()

    def close(self) -> None:
        """Detach work before the scenario's provider or widgets are retired."""
        self._cancel_read()
        self._startup_error = None
        self._prepared_future = self._cached_plan = None
        self._prepared_token = None
        self.steps = None
        self.timestamps.clear()
        self._token = None
        self._restore_stride()
        with QSignalBlocker(self.checkbox):
            self.checkbox.setChecked(False)
            self.checkbox.setText("Play sensing frames only")
        if getattr(self.visualizer, "sensing_playback", None) is self:
            self.visualizer.sensing_playback = None

    def set_enabled(self, enabled: bool, *, automatic: bool = False) -> None:
        viz = self.visualizer
        self._cancel_read()
        self._automatic = automatic
        if viz.animation_running:
            viz.animation_controller.toggle_animation()
        viz.cancel_startup_preload()
        viz.reset_preloading_state()
        self.timestamps.clear()
        if enabled and self.steps is None:
            stride = getattr(viz, "stride_combo", None)
            if stride is not None:
                self._full_timeline_stride = stride.currentText()
        self.steps = () if enabled else None
        if not enabled:
            self._restore_stride()
            self._startup_error = None
            self.checkbox.setText("Play sensing frames only")
            self._apply_navigation()
            return
        provider = visualizer_frame_provider(viz)
        if provider is None or provider.frame_ids_snapshot is None:
            self._unavailable("Sensing-only playback requires a saved recording.")
            return
        self._token = _provider_token(provider)
        self.checkbox.setText("Indexing sensing frames…")
        if self._cached_plan is not None and self._cached_plan[0] == self._token:
            self._future = Future()
            self._future.set_result(self._cached_plan[1])
            self._timer.start()
            self._poll()
            return
        if self._prepared_token == self._token and self._prepared_future is not None:
            self._future = self._prepared_future
            self._timer.start()
            self._poll()
            return
        self._cancelled = Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensing-playback")
        self._future = self._executor.submit(load_sensing_playback_plan, provider, self._cancelled)
        self._timer.start()

    def _unavailable(self, message: str, *, quiet: bool = False) -> None:
        on_error = self._startup_error
        if on_error is not None:
            self._startup_error = None
            self._cancel_read()
            on_error(message)
            return
        with QSignalBlocker(self.checkbox):
            self.checkbox.setChecked(False)
        self.set_enabled(False, automatic=self._automatic)
        if not quiet:
            self.visualizer._set_status_message(message, 6000)

    def _poll(self) -> None:
        future = self._future
        if future is None or not future.done():
            return
        # Scenario construction pumps Qt events before targets/navigation exist.
        if not getattr(self.visualizer, "ready", True):
            return
        provider = visualizer_frame_provider(self.visualizer)
        if provider is None or _provider_token(provider) != self._token:
            self._unavailable("Recording changed; select sensing-only playback again.")
            return
        try:
            steps, timestamps = future.result()
        except (CancelledError, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            self._unavailable(f"Cannot index sensing frames: {exc}")
            return
        self._cached_plan = self._token, (steps, timestamps)
        self._cancel_read()
        if not steps:
            self._unavailable(
                "No saved sensing results have an available scene binding.", quiet=self._automatic
            )
            return
        self.steps, self.timestamps = steps, dict(timestamps)
        self._startup_error = None
        self.checkbox.setText(f"Play sensing frames only ({len(steps)})")
        stride = getattr(self.visualizer, "stride_combo", None)
        if stride is not None:
            with QSignalBlocker(stride):
                stride.setCurrentText("x1")
                stride.setEnabled(False)
        self._apply_navigation()

    def set_full_timeline_stride(self, text):
        """Retain authored cadence separately from sensing-result ordinals."""
        stride = getattr(self.visualizer, "stride_combo", None)
        if self.steps is not None:
            self._full_timeline_stride = text
        elif stride is not None:
            stride.setCurrentText(text)

    def _restore_stride(self):
        stride = getattr(self.visualizer, "stride_combo", None)
        if stride is not None:
            with QSignalBlocker(stride):
                if self._full_timeline_stride is not None:
                    stride.setCurrentText(self._full_timeline_stride)
                stride.setEnabled(True)
        self._full_timeline_stride = None

    def _apply_navigation(self) -> None:
        viz = self.visualizer
        if not getattr(viz, "ready", True):
            return
        steps = viz.get_available_animation_steps()
        if not steps:
            return
        # Changing slider bounds can itself emit a seek; block it until the
        # filtered timeline and its chosen current step have both been installed.
        blockers = [
            QSignalBlocker(widget)
            for name in ("step_slider", "frame_input")
            if (widget := getattr(viz, name, None)) is not None
        ]
        try:
            viz.total_animation_steps = len(steps)
            viz.ui_manager.update_total_steps(len(steps))
            panel = viz.ui_manager.panels.get("animation")
            if panel is not None:
                panel.set_recorded_frame_context(steps)
            step = min(steps, key=lambda value: abs(value - viz.animation_step))
            viz.force_update_next_frame = True
            viz.update_frame(step)
        finally:
            for blocker in blockers:
                blocker.unblock()
        refresh = getattr(getattr(viz, "ui_controller", None), "refresh_status_telemetry", None)
        if callable(refresh):
            refresh()
        if not self._automatic or getattr(viz, "use_preload_mode", True):
            viz.start_preloading()
