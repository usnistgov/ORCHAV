"""Frame playback, cache warming, and background preloading service."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional

from PySide6.QtCore import QObject, Qt, Signal

from shared.frames.loader import FrameLoaderService
from shared.logging import get_logger

from ..io.packed_frame_payload import (
    standard_frame_to_visual_frame,
    try_load_packed_visual_frame,
    visual_frame_read_request,
    visual_frame_read_request_for_visualizer,
)
from ..pipeline.frame_pipeline import FramePipeline
from ..services.base import BaseService

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer


PRELOAD_WORKERS = 4
"""Number of parallel I/O threads used for frame preloading."""


class _PreloadSignals(QObject):
    """Qt signals for thread-safe preload progress/completion."""

    progress = Signal(str)  # status text
    complete = Signal(int, int, float, list)  # loaded, failed, duration, loaded_steps
    step_ready = Signal(int)  # step number that just finished preloading


class _PreloadWorker:
    """Loads frames on a background thread using a pool for parallel I/O.

    Frame reads are submitted to a :class:`ThreadPoolExecutor` so that
    multiple HDF5/file reads overlap.  Cache writes and canonical
    precomputation are serialized in the consumer loop to avoid contention
    on non-thread-safe structures (``_preloaded_steps`` set, eviction
    callbacks).
    """

    def __init__(
        self,
        loader: Optional[FrameLoaderService],
        frame_source: Optional[Any],
        cache_service: Optional[Any],
        frame_numbers: List[int],
        mpc_core: Optional[Any] = None,
        *,
        is_current: Callable[[], bool],
        run_if_current: Callable[[Callable[[], None]], bool],
    ) -> None:
        """Initialize a preload worker for a fixed list of frame numbers."""
        self.signals = _PreloadSignals()
        self._loader = loader
        self._frame_source = frame_source
        self._cache_service = cache_service
        self._frame_numbers = frame_numbers
        self._mpc_core = mpc_core
        self._canonical_points_dtype = getattr(mpc_core, "canon_points_dtype", None)
        self._is_current = is_current
        self._run_if_current = run_if_current
        self._stop_requested = threading.Event()
        self._logger = get_logger(__name__)

    def request_stop(self) -> None:
        """Ask the preload loop to stop before scheduling more work."""
        self._stop_requested.set()

    def _should_stop(self) -> bool:
        """Return whether this worker was stopped or superseded."""
        return self._stop_requested.is_set() or not self._is_current()

    def _load_one(self, step: int) -> tuple[int, Any]:
        """Perform I/O for a single frame (runs inside the thread pool).

        Returns:
            ``(step, frame_data)`` on success, ``(step, None)`` on failure.
        """
        try:
            data = None
            if self._loader is not None:
                data = try_load_packed_visual_frame(
                    self._loader.provider,
                    step,
                    request=visual_frame_read_request(),
                    points_dtype=self._canonical_points_dtype or "float32",
                )
                if data is None:
                    data = standard_frame_to_visual_frame(
                        self._loader.get_frame(step),
                        request=visual_frame_read_request(),
                        points_dtype=self._canonical_points_dtype or "float32",
                    )
            elif self._frame_source is not None:
                data = standard_frame_to_visual_frame(
                    self._frame_source.load_frame(step),
                    request=visual_frame_read_request(),
                    points_dtype=self._canonical_points_dtype or "float32",
                )
            return (step, data)
        except (OSError, IOError, ValueError, KeyError, AttributeError, TypeError) as exc:
            self._logger.warning("Failed to preload frame %d: %s", step, exc)
            return (step, None)

    def run(self) -> None:
        """Load requested frames, store cache entries, and emit preload progress."""
        total = len(self._frame_numbers)
        loaded = 0
        failed = 0
        loaded_steps: list[int] = []
        start_time = time.time()

        if not self._loader and not self._frame_source:
            self._logger.warning("No loader available for preloading")
            if not self._should_stop():
                self.signals.complete.emit(0, total, time.time() - start_time, [])
            return

        futures: dict[Future[tuple[int, Any]], int] = {}
        with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as pool:
            for step in self._frame_numbers:
                if self._should_stop():
                    break
                fut = pool.submit(self._load_one, step)
                futures[fut] = step

            for fut in as_completed(futures):
                if self._should_stop():
                    # Cancel any remaining pending futures
                    for pending in futures:
                        pending.cancel()
                    break

                step = futures[fut]
                try:
                    _step, data = fut.result()
                except (OSError, IOError, ValueError, KeyError, AttributeError) as exc:
                    self._logger.warning("Failed to preload frame %d: %s", step, exc)
                    failed += 1
                    data = None

                if data is not None:

                    def store_frame() -> None:
                        if self._cache_service is not None:
                            self._cache_service.store_frame(step, data, source="preload")

                    if not self._run_if_current(store_frame):
                        for pending in futures:
                            pending.cancel()
                        break

                    if self._mpc_core is not None:
                        try:
                            t0 = time.time()
                            self._mpc_core.precompute_canonical(step, data)
                            self._logger.info(
                                "Canonical precompute step %d: %.2fs", step, time.time() - t0
                            )
                        except (ValueError, TypeError, KeyError, IndexError) as exc:
                            self._logger.debug(
                                "Failed to precompute canonical for step %d: %s",
                                step,
                                exc,
                            )
                    loaded += 1
                    loaded_steps.append(step)
                    if self._is_current():
                        self.signals.step_ready.emit(step)
                else:
                    failed += 1

                if total > 0 and self._is_current():
                    pct = int((loaded + failed) / total * 100)
                    self.signals.progress.emit(f"Preload: {loaded}/{total} ({pct}%)")

        duration = time.time() - start_time
        if not self._should_stop():
            self.signals.complete.emit(loaded, failed, duration, loaded_steps)


class _ThreadedPreloader:
    """Non-blocking preloader that runs frame I/O on a daemon thread.

    The main Qt thread never blocks regardless of frame size. Progress and
    completion are delivered via Qt signals (auto-queued across threads).
    """

    def __init__(self, animation_service: "AnimationService", generation: int) -> None:
        """Bind the threaded preloader to its owning animation service."""
        self._service = animation_service
        self._generation = generation
        self._worker: Optional[_PreloadWorker] = None
        self._thread: Optional[threading.Thread] = None
        self._on_progress: Optional[Callable[[str], None]] = None
        self._on_complete: Optional[Callable[[List[tuple[int, dict]], float], None]] = None
        self._on_step_ready: Optional[Callable[[int], None]] = None
        self._logger = get_logger(__name__)

    def _is_current(self) -> bool:
        """Return whether this preloader still owns the active generation."""
        return self._service._is_current_preloader(self, self._generation)

    def _run_if_current(self, operation: Callable[[], None]) -> bool:
        """Run one worker-side mutation only while this generation is active."""
        return self._service._run_if_current_preloader(
            self,
            self._generation,
            operation,
        )

    def start(
        self,
        frame_numbers: List[int],
        on_progress: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[List[tuple[int, dict]], float], None]] = None,
        on_step_ready: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Start non-blocking preloading of frames in a background thread."""
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._on_step_ready = on_step_ready

        if not frame_numbers:
            self._handle_complete(0, 0, 0.0)
            return

        if on_progress:
            on_progress(f"Preload: Loading {len(frame_numbers)} files...")

        viz = self._service.visualizer
        loader = self._service.frame_loader or getattr(viz, "frame_loader", None)
        frame_source = getattr(viz, "frame_source", None)
        cache_service = self._service.cache_service
        mpc_core = getattr(viz, "mpc_core", None)

        self._worker = _PreloadWorker(
            loader,
            frame_source,
            cache_service,
            frame_numbers,
            mpc_core=mpc_core,
            is_current=self._is_current,
            run_if_current=self._run_if_current,
        )
        self._worker.signals.progress.connect(self._handle_progress)
        self._worker.signals.complete.connect(self._handle_complete)
        if self._on_step_ready is not None:
            self._worker.signals.step_ready.connect(
                self._handle_step_ready,
                Qt.QueuedConnection,
            )

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop preloading and wait for the background thread to finish."""
        worker = self._worker
        if worker is not None:
            worker.request_stop()
            self._disconnect_worker(worker)
        thread = self._thread
        if thread is not None:
            if thread.is_alive():
                thread.join(timeout=2.0)
            if not thread.is_alive():
                self._thread = None
                self._worker = None

    def _disconnect_worker(self, worker: _PreloadWorker) -> None:
        """Disconnect callbacks while retaining identity guards for queued signals."""
        connections = [
            (worker.signals.progress, self._handle_progress),
            (worker.signals.complete, self._handle_complete),
        ]
        if self._on_step_ready is not None:
            connections.append((worker.signals.step_ready, self._handle_step_ready))
        for signal, callback in connections:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                # A signal may already have disconnected during Qt teardown.
                pass

    def _handle_progress(self, text: str) -> None:
        """Forward worker progress text to the caller callback."""
        if self._is_current() and self._on_progress:
            self._on_progress(text)

    def _handle_step_ready(self, step: int) -> None:
        """Forward a completed step only for the active preload generation."""
        if self._is_current() and self._on_step_ready:
            self._on_step_ready(step)

    def _handle_complete(
        self, loaded: int, failed: int, duration: float, loaded_steps: list
    ) -> None:
        """Finalize preload state and reconcile worker/cache step tracking."""
        if not self._is_current():
            return

        end_time = time.time()
        self._service._preload_end_time = end_time
        self._service._preloading_completed = True

        if self._on_progress:
            self._on_progress(f"Preload: Complete ({loaded} files)")
            if not self._is_current():
                return

        self._logger.info(
            "Preloading completed: %d loaded, %d failed in %.2f seconds",
            loaded,
            failed,
            duration,
        )

        # Reconcile: ensure every step the worker loaded is tracked as preloaded
        cache_service = self._service.cache_service
        if cache_service is not None and loaded_steps:
            tracked = cache_service.preloaded_frame_count
            if tracked != len(loaded_steps):
                self._logger.warning(
                    "Preload reconcile: worker loaded %d steps but cache tracks %d; "
                    "marking missing steps",
                    len(loaded_steps),
                    tracked,
                )
                for step in loaded_steps:
                    if not cache_service.is_preloaded(step):
                        cache_service.mark_preloaded(step)

        if self._on_complete and self._is_current():
            frames_list = cache_service.get_preloaded_frames() if cache_service is not None else []
            self._on_complete(frames_list, duration)


class AnimationService(BaseService):
    """
    Frame-loading coordination service.

    Delegates caching to the shared frame cache and coordinates frame pipeline
    updates. Also handles preloading of frames for offline scenarios.
    """

    def __init__(
        self,
        pipeline: FramePipeline,
        visualizer: OrchavVisualizer,
        *,
        max_cached_steps: int = 10,
        on_frame: Optional[Callable[[int, dict[str, Any]], None]] = None,
        frame_loader: Optional[FrameLoaderService] = None,
        cache_service: Optional[Any] = None,
    ):
        """Initialize playback cursor, cache service, and preload bookkeeping."""
        super().__init__()
        self.pipeline = pipeline
        self.visualizer = visualizer
        self.max_cached_steps = max_cached_steps
        self._current_step = 0
        self._on_frame = on_frame
        self.frame_loader = frame_loader
        self.cache_service = cache_service or getattr(visualizer, "cache_service", None)

        # Preloading state
        self._preloading_started = False
        self._preloading_completed = False
        self._preload_start_time: Optional[float] = None
        self._preload_end_time: Optional[float] = None
        self._threaded_preloader: Optional[_ThreadedPreloader] = None
        self._preload_generation = 0
        self._preload_generation_lock = threading.RLock()

    def _install_preloader(self) -> _ThreadedPreloader:
        """Create and install the sole owner of a new preload generation."""
        with self._preload_generation_lock:
            self._preload_generation += 1
            preloader = _ThreadedPreloader(self, self._preload_generation)
            self._threaded_preloader = preloader
            return preloader

    def _is_current_preloader(
        self,
        preloader: _ThreadedPreloader,
        generation: int,
    ) -> bool:
        """Check both generation and object identity for a preload callback."""
        with self._preload_generation_lock:
            return generation == self._preload_generation and preloader is self._threaded_preloader

    def _run_if_current_preloader(
        self,
        preloader: _ThreadedPreloader,
        generation: int,
        operation: Callable[[], None],
    ) -> bool:
        """Serialize an active-worker mutation against generation retirement."""
        with self._preload_generation_lock:
            if generation != self._preload_generation or preloader is not self._threaded_preloader:
                return False
            operation()
            return True

    def _retire_current_preloader(self) -> Optional[_ThreadedPreloader]:
        """Invalidate and detach the current preloader before asking it to stop."""
        with self._preload_generation_lock:
            self._preload_generation += 1
            preloader = self._threaded_preloader
            self._threaded_preloader = None
            return preloader

    def _stop_current_preloader(self) -> None:
        """Retire the active generation, then request bounded worker shutdown."""
        preloader = self._retire_current_preloader()
        if preloader is not None:
            preloader.stop()

    @property
    def preloading_started(self) -> bool:
        """Return whether a background preload has been started."""
        return self._preloading_started

    @property
    def preloading_completed(self) -> bool:
        """Return whether the current preload lifecycle has completed."""
        return self._preloading_completed

    @property
    def preload_frame_count(self) -> int:
        """Return the number of frames marked as preloaded in the cache service."""
        cache_service = self.cache_service
        if cache_service is None:
            return 0
        return cache_service.preloaded_frame_count

    def preload_duration(self) -> Optional[float]:
        """Return elapsed preload time for active or completed preloads."""
        if self._preload_start_time is None:
            return None
        end_time = self._preload_end_time or time.time()
        return end_time - self._preload_start_time

    def set_frame_loader(self, loader: Optional[FrameLoaderService]) -> None:
        """Update the frame loader reference when scenarios change."""
        self.frame_loader = loader

    def start(self, start_step: int = 0) -> None:
        """Start playback from a given step without loading a frame yet."""
        self._current_step = start_step
        super().start()

    def advance(self) -> Optional[dict[str, Any]]:
        """Load the current step, run the pipeline, and advance the cursor."""
        if not self.is_running():
            return None

        frame = self._load_frame(self._current_step)
        if frame is None:
            return None

        self.pipeline.update(self._current_step)
        self._current_step += 1

        if self._on_frame:
            self._on_frame(self._current_step - 1, frame)

        return frame

    def _load_frame(self, step: int) -> Optional[dict[str, Any]]:
        """Load one frame from cache, frame loader, or frame source."""
        cache_service = self.cache_service
        if cache_service is not None:
            cached = cache_service.get_frame(step)
            if cached is not None:
                return cached

        loader = self.frame_loader or getattr(self.visualizer, "frame_loader", None)
        frame: Optional[dict[str, Any]] = None
        request = visual_frame_read_request_for_visualizer(self.visualizer)
        points_dtype = getattr(
            getattr(self.visualizer, "mpc_core", None),
            "canon_points_dtype",
            "float32",
        )
        if loader is not None:
            frame = try_load_packed_visual_frame(
                loader.provider,
                step,
                request=request,
                points_dtype=points_dtype,
            )
            if frame is None:
                frame = standard_frame_to_visual_frame(
                    loader.get_frame(step),
                    request=request,
                    points_dtype=points_dtype,
                )
        else:
            frame_source = getattr(self.visualizer, "frame_source", None)
            if frame_source is None:
                return None
            try:
                frame = standard_frame_to_visual_frame(
                    frame_source.load_frame(step),
                    request=request,
                    points_dtype=points_dtype,
                )
            except FileNotFoundError as exc:
                if getattr(self.visualizer, "_scene_only_mode", False) or (
                    "scene-only mode" in str(exc)
                ):
                    get_logger(__name__).debug(
                        "No frames available in scene-only mode; skipping frame load"
                    )
                    return None
                raise

        if frame is None:
            return None

        if cache_service is not None and not cache_service.has_frame(step):
            cache_service.store_frame(step, frame)
        return frame

    def preload(self, steps: Iterable[int]) -> None:
        """Try to preload a sequence of future steps into the cache."""
        for step in steps:
            cache_service = self.cache_service
            if cache_service is None or not cache_service.has_frame(step):
                self._load_frame(step)

    def ensure_step_cached(self, step: int) -> Optional[dict[str, Any]]:
        """Load and cache a specific step without advancing playback state."""
        return self._load_frame(step)

    def clear_cache(self) -> None:
        """Clear frame-cache entries owned by the cache service."""
        cache_service = self.cache_service
        if cache_service is not None:
            cache_service.clear_frame_cache(reason="animation_service")

    def load_step(self, step: int) -> Optional[dict[str, Any]]:
        """
        Load and cache a specific animation step without advancing the playback cursor.
        This method keeps the cache warm before the pipeline consumes the frame.
        """
        frame = self.ensure_step_cached(step)
        if frame is None:
            return None
        self._current_step = step + 1
        return frame

    def get_cached_frame(self, step: int) -> Optional[dict[str, Any]]:
        """Return a cached frame if available (used by the frame pipeline)."""
        cache_service = self.cache_service
        if cache_service is None:
            return None
        return cache_service.get_frame(step)

    def start_preloading(
        self,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[List[tuple[int, dict]], float], None]] = None,
        on_step_ready: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Preload all available frames from the frame source.

        Frame I/O runs on a background thread so the UI never blocks,
        regardless of per-frame size.  Progress and completion callbacks
        are delivered on the main thread via Qt signals.

        Args:
            on_progress: Callback for progress updates (receives status string).
            on_complete: Callback when complete (receives frames list and duration).
            on_step_ready: Callback fired for each step after it finishes
                preloading (receives step number).

        Returns:
            True if preloading started, False if skipped/already running.
        """
        logger = getattr(self.visualizer, "logger", None)
        if not logger:
            logger = get_logger(__name__)

        # STATE GUARD: Prevent multiple preloading calls
        if self._preloading_started:
            if logger:
                logger.debug("Preloading already in progress; skipping duplicate call")
            return False

        if self._preloading_completed:
            if logger:
                logger.debug("Preloading already completed; skipping duplicate call")
            return False

        # Guard: don't try to preload until frame source is ready
        frame_source = getattr(self.visualizer, "frame_source", None)
        loader = self.frame_loader or getattr(self.visualizer, "frame_loader", None)
        if not frame_source:
            if logger:
                logger.debug("No frame source yet; skipping preloading")
            return False

        # Live gRPC frames are requested on demand, so local preloading is unnecessary.
        try:
            from ..io.frame_sources import LiveGrpcSource

            if isinstance(frame_source, LiveGrpcSource):
                if logger:
                    logger.debug("Skipping preloading in live gRPC mode (frames load on demand)")
                self._preloading_started = False
                self._preloading_completed = True
                if on_progress:
                    on_progress("Preload: Not needed (live gRPC mode)")
                return False
        except ImportError:
            pass

        if loader is None:
            try:
                available_frames = frame_source.list_frames()
            except (OSError, ValueError, AttributeError) as exc:
                if logger:
                    logger.debug("Unable to enumerate frames for preloading: %s", exc)
                return False
            if not available_frames:
                if logger:
                    logger.debug("No frames available yet; skipping preloading")
                return False

        # Mark preloading as started
        self._preloading_started = True
        self._preload_start_time = time.time()
        self._preload_end_time = None
        if logger:
            logger.info("Starting non-blocking preloading")

        try:
            # Find all frame data using the frame source
            if loader is not None:
                frame_numbers = loader.list_frames()
            else:
                frame_numbers = frame_source.list_frames()

            if not frame_numbers:
                if logger:
                    logger.debug("No frame data found for preloading")
                if on_progress:
                    on_progress("Preload: No frame data found")
                self._preloading_started = False
                self._preloading_completed = True
                self._preload_end_time = time.time()
                if self._preload_start_time:
                    duration = self._preload_end_time - self._preload_start_time
                    if logger:
                        logger.info(
                            "Preloading completed (no frames to load) in %.2f seconds",
                            duration,
                        )
                return True

            cache_service = self.cache_service
            if cache_service is not None:
                cache_service.ensure_frame_cache_capacity(len(frame_numbers))

            # Prioritize frames near the current playback position so the
            # buffer zone around the active step fills first.
            current_step = getattr(self.visualizer, "animation_step", 0)
            frame_numbers = sorted(frame_numbers, key=lambda s: abs(s - current_step))

            preloader = self._install_preloader()
            preloader.start(
                frame_numbers=frame_numbers,
                on_progress=on_progress,
                on_complete=on_complete,
                on_step_ready=on_step_ready,
            )

            if logger:
                logger.info(
                    "Background preload started for %d frames (nearest-first from step %d)",
                    len(frame_numbers),
                    current_step,
                )

            return True

        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            self._stop_current_preloader()
            if logger:
                logger.error(f"Preloading failed to start: {e}")
            if on_progress:
                on_progress("Preload: Failed to start")
            # Reset state on failure so it can be retried
            self._preloading_started = False
            self._preload_end_time = None
            self._preload_start_time = None
            return False

    def reset_preloading_state(self) -> None:
        """Reset preloading state to allow retrying."""
        self._stop_current_preloader()
        self._preloading_started = False
        self._preloading_completed = False
        self._preload_start_time = None
        self._preload_end_time = None
        logger = getattr(self.visualizer, "logger", None)
        if logger:
            logger.debug("Preloading state reset")

    def detect_mpc_frames(
        self, *, on_frame_count_updated: Optional[Callable[[int], None]] = None
    ) -> Optional[int]:
        """
        Detect the number of available MPC frames.

        Args:
            on_frame_count_updated: Callback when frame count is detected (receives count)

        Returns:
            Total number of frames detected, or None if detection failed/skipped
        """
        logger = getattr(self.visualizer, "logger", None)

        # Guard: don't try to detect frames until frame source is ready
        frame_source = getattr(self.visualizer, "frame_source", None)
        if not frame_source:
            if logger:
                logger.debug("No frame source yet; skipping frame detection")
            return None

        if not frame_source.has_frame(0):
            if logger:
                logger.debug("Frame 0 not available yet; skipping frame detection")
            return None

        try:
            # Find all frame data using the frame source
            frame_numbers = frame_source.list_frames()
            if not frame_numbers:
                if logger:
                    logger.debug("No MPC frame files found")
                return None

            # Find the maximum step number
            max_step = max(frame_numbers)

            if max_step >= 0:
                # Frame count is max_step + 1 (frames are 0-indexed)
                new_total_steps = max_step + 1

                if logger:
                    logger.info(f"Detected {new_total_steps} MPC frames (max index: {max_step})")

                # Notify callback
                if on_frame_count_updated:
                    on_frame_count_updated(new_total_steps)

                return new_total_steps

        except (OSError, ValueError, KeyError, AttributeError) as e:
            if logger:
                logger.warning(f"Could not detect MPC frames: {e}")

        return None

    def clear_preload_data(self, *, reset_cache_size: bool = False) -> int:
        """Clear preloaded tracking state.

        Returns number of entries cleared and stops any in-progress preloading.
        """
        self._stop_current_preloader()
        cache_service = self.cache_service
        count = cache_service.preloaded_frame_count if cache_service is not None else 0
        if cache_service is not None:
            cache_service.clear_preloaded()
            if reset_cache_size:
                cache_service.reset_frame_cache_size()
        return count
