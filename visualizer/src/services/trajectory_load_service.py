"""Single-owner background loading for 2D and 3D trajectory views.

The coordinator performs each frame-source scan once and publishes immutable
snapshots to every consumer.  A monotonically increasing generation prevents
callbacks from a retired scenario or worker from replacing current data.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

from PySide6.QtCore import QObject, Qt, Signal, Slot

from shared.frames import StandardMPCFrame, project_standard_mpc_frame
from shared.logging import get_logger

from ..io.frame_sources import FileSource, RemoteHdf5Source
from ..io.packed_frame_payload import frame_source_provider
from ..io.trajectory_frame_projection import (
    TRAJECTORY_READ_REQUEST,
    try_load_packed_trajectory_frame,
)

logger = get_logger(__name__)

TRAJECTORY_UNAVAILABLE_MESSAGE = "Only available for pre-computed data"

TrajectoryPoint = tuple[int, float, float, float]
IndexedTrajectoryTracks = Mapping[int, tuple[TrajectoryPoint, ...]]
NamedTrajectoryTracks = Mapping[str, tuple[TrajectoryPoint, ...]]


def _freeze_point(point: Any) -> TrajectoryPoint:
    """Copy one frame/XYZ sample into an immutable primitive tuple."""
    return (
        int(point[0]),
        float(point[1]),
        float(point[2]),
        float(point[3]),
    )


def _freeze_indexed_tracks(raw_tracks: Mapping[Any, Any]) -> IndexedTrajectoryTracks:
    """Deep-freeze integer-keyed TX or RX tracks."""
    return MappingProxyType(
        {
            int(track_id): tuple(_freeze_point(point) for point in points)
            for track_id, points in raw_tracks.items()
        }
    )


def _freeze_named_tracks(raw_tracks: Mapping[Any, Any]) -> NamedTrajectoryTracks:
    """Deep-freeze target-name tracks."""
    return MappingProxyType(
        {
            str(track_name): tuple(_freeze_point(point) for point in points)
            for track_name, points in raw_tracks.items()
        }
    )


@dataclass(frozen=True, slots=True)
class TrajectorySnapshot(Mapping[str, object]):
    """Deeply immutable trajectory data shared by all presentation consumers."""

    tx_positions: IndexedTrajectoryTracks
    rx_positions: IndexedTrajectoryTracks
    target_positions: NamedTrajectoryTracks
    frames_loaded: tuple[int, ...]
    total_frames: int

    _KEYS: ClassVar[tuple[str, ...]] = (
        "tx_positions",
        "rx_positions",
        "target_positions",
        "frames_loaded",
        "total_frames",
    )

    def __post_init__(self) -> None:
        """Defensively freeze direct construction as well as worker snapshots."""
        object.__setattr__(self, "tx_positions", _freeze_indexed_tracks(self.tx_positions))
        object.__setattr__(self, "rx_positions", _freeze_indexed_tracks(self.rx_positions))
        object.__setattr__(
            self,
            "target_positions",
            _freeze_named_tracks(self.target_positions),
        )
        object.__setattr__(
            self,
            "frames_loaded",
            tuple(int(frame) for frame in self.frames_loaded),
        )
        object.__setattr__(self, "total_frames", int(self.total_frames))

    @classmethod
    def from_mutable(cls, trajectories: Mapping[str, Any]) -> "TrajectorySnapshot":
        """Deep-copy a worker-owned mutable accumulator into a stable snapshot."""
        return cls(
            tx_positions=trajectories["tx_positions"],
            rx_positions=trajectories["rx_positions"],
            target_positions=trajectories["target_positions"],
            frames_loaded=trajectories["frames_loaded"],
            total_frames=int(trajectories["total_frames"]),
        )

    def __getitem__(self, key: str) -> object:
        """Provide the existing mapping-shaped read contract without mutability."""
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the stable trajectory field names."""
        return iter(self._KEYS)

    def __len__(self) -> int:
        """Return the number of mapping fields."""
        return len(self._KEYS)


class _TrajectoryLoadSignals(QObject):
    """Worker-to-coordinator signals carrying the owning generation."""

    progress_updated = Signal(int, int, int)  # generation, loaded, total
    snapshot_updated = Signal(int, object)  # generation, TrajectorySnapshot
    loading_complete = Signal(int, object)  # generation, TrajectorySnapshot
    error_occurred = Signal(int, str)  # generation, message


class _TrajectoryLoadWorker:
    """Extract trajectories progressively from one fixed frame source."""

    def __init__(
        self,
        generation: int,
        frame_source: Any,
        *,
        is_current: Callable[[], bool],
    ) -> None:
        """Bind the worker to a source and coordinator generation."""
        self.generation = generation
        self.frame_source = frame_source
        self.signals = _TrajectoryLoadSignals()
        self._is_current = is_current
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Ask extraction to stop before the next mutation or publication."""
        self._stop_requested.set()

    def _should_stop(self) -> bool:
        """Return whether this worker was stopped or superseded."""
        return self._stop_requested.is_set() or not self._is_current()

    def _publish_snapshot(self, trajectories: Mapping[str, Any], *, complete: bool) -> None:
        """Freeze and publish the accumulator only while this run is current."""
        if self._should_stop():
            return
        snapshot = TrajectorySnapshot.from_mutable(trajectories)
        if complete:
            self.signals.loading_complete.emit(self.generation, snapshot)
        else:
            self.signals.snapshot_updated.emit(self.generation, snapshot)

    def run(self) -> None:
        """Sample frames for a quick preview, then fill the remaining frames."""
        try:
            if self._should_stop():
                return
            frames = list(self.frame_source.list_frames())
            if self._should_stop():
                return
            total_frames = len(frames)
            if total_frames == 0:
                if not self._should_stop():
                    self.signals.error_occurred.emit(
                        self.generation,
                        "No frames available",
                    )
                return

            trajectories: dict[str, Any] = {
                "tx_positions": {},
                "rx_positions": {},
                "target_positions": {},
                "frames_loaded": [],
                "total_frames": total_frames,
            }

            sample_step = max(1, total_frames // 20)
            sampled_frames = frames[::sample_step]
            if not self._load_frames(
                sampled_frames,
                trajectories,
                update_interval=max(1, len(sampled_frames) // 4),
            ):
                return

            self._publish_snapshot(trajectories, complete=False)

            loaded_frames = set(trajectories["frames_loaded"])
            remaining_frames = [frame for frame in frames if frame not in loaded_frames]
            if not self._load_frames(
                remaining_frames,
                trajectories,
                update_interval=max(1, len(remaining_frames) // 20),
            ):
                return

            self._publish_snapshot(trajectories, complete=True)
        except (
            AttributeError,
            IndexError,
            OSError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error("Trajectory extraction failed: %s", exc)
            if not self._should_stop():
                self.signals.error_occurred.emit(self.generation, str(exc))

    def _load_frames(
        self,
        frames: list[int],
        trajectories: dict[str, Any],
        *,
        update_interval: int,
    ) -> bool:
        """Load one extraction phase, returning false when the run is retired."""
        total_frames = int(trajectories["total_frames"])
        for index, frame_idx in enumerate(frames):
            if self._should_stop():
                return False
            try:
                provider = frame_source_provider(self.frame_source)
                frame_data = try_load_packed_trajectory_frame(provider, frame_idx)
                if frame_data is None:
                    frame_data = self.frame_source.load_frame(frame_idx)
                if self._should_stop():
                    return False
                self._extract_positions(frame_idx, frame_data, trajectories)
                trajectories["frames_loaded"].append(frame_idx)
                self.signals.progress_updated.emit(
                    self.generation,
                    len(trajectories["frames_loaded"]),
                    total_frames,
                )
                if index % update_interval == 0:
                    self._publish_snapshot(trajectories, complete=False)
            except (
                AttributeError,
                IndexError,
                OSError,
                RuntimeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning("Failed to load frame %d: %s", frame_idx, exc)
        return not self._should_stop()

    @staticmethod
    def _extract_positions(
        frame_idx: int,
        frame_data: Mapping[str, Any] | StandardMPCFrame,
        trajectories: dict[str, Any],
    ) -> None:
        """Append TX, RX, and named target positions from one frame."""
        if isinstance(frame_data, StandardMPCFrame):
            projected = project_standard_mpc_frame(
                frame_data,
                TRAJECTORY_READ_REQUEST,
            ).frame
            tx_pos = projected.tx_positions
            rx_pos = projected.rx_positions
            targets_meta = projected.targets_metadata
        else:
            tx_pos = frame_data.get("tx_positions")
            rx_pos = frame_data.get("rx_positions")
            targets_meta = frame_data.get("targets_metadata")

        if tx_pos is not None and len(tx_pos) > 0:
            for tx_idx, pos in enumerate(tx_pos):
                trajectories["tx_positions"].setdefault(tx_idx, []).append(
                    (frame_idx, float(pos[0]), float(pos[1]), float(pos[2]))
                )

        if rx_pos is not None and len(rx_pos) > 0:
            for rx_idx, pos in enumerate(rx_pos):
                trajectories["rx_positions"].setdefault(rx_idx, []).append(
                    (frame_idx, float(pos[0]), float(pos[1]), float(pos[2]))
                )

        if targets_meta:
            for target in targets_meta:
                target_name = target.get("name", "Target")
                pos = target.get("current_position")
                if pos is not None and len(pos) >= 3:
                    trajectories["target_positions"].setdefault(target_name, []).append(
                        (frame_idx, float(pos[0]), float(pos[1]), float(pos[2]))
                    )


class TrajectoryLoadCoordinator(QObject):
    """Own one trajectory load generation and share its immutable snapshots."""

    progress_updated = Signal(int, int)
    snapshot_updated = Signal(object)  # TrajectorySnapshot
    loading_complete = Signal(object)  # TrajectorySnapshot
    error_occurred = Signal(str)
    cleared = Signal()

    def __init__(self) -> None:
        """Initialize an idle, reusable coordinator."""
        super().__init__()
        self._lock = threading.RLock()
        self._generation = 0
        self._source: Any = None
        self._snapshot: TrajectorySnapshot | None = None
        self._progress: tuple[int, int] = (0, 0)
        self._complete = False
        self._shutdown = False
        self._worker: _TrajectoryLoadWorker | None = None
        self._thread: threading.Thread | None = None
        self._retired: list[tuple[_TrajectoryLoadWorker, threading.Thread]] = []

    @staticmethod
    def supports_source(frame_source: Any) -> bool:
        """Return whether trajectory extraction is supported for this source."""
        return isinstance(frame_source, (FileSource, RemoteHdf5Source)) or (
            frame_source_provider(frame_source) is not None
        )

    @property
    def snapshot(self) -> TrajectorySnapshot | None:
        """Return the latest stable snapshot for the active generation."""
        with self._lock:
            return self._snapshot

    @property
    def progress(self) -> tuple[int, int]:
        """Return ``(loaded, total)`` for the active generation."""
        with self._lock:
            return self._progress

    @property
    def is_loading(self) -> bool:
        """Return whether the active generation is still extracting frames."""
        with self._lock:
            return self._worker is not None and not self._complete

    @property
    def is_complete(self) -> bool:
        """Return whether the active source produced a final snapshot."""
        with self._lock:
            return self._complete

    @property
    def is_shutdown(self) -> bool:
        """Return whether final application shutdown retired this coordinator."""
        with self._lock:
            return self._shutdown

    def load(self, frame_source: Any) -> bool:
        """Start one load unless this source is already loading or complete.

        Returns:
            ``True`` when a new worker was started, otherwise ``False``.
        """
        if not self.supports_source(frame_source):
            self.error_occurred.emit(TRAJECTORY_UNAVAILABLE_MESSAGE)
            return False

        with self._lock:
            if self._shutdown:
                raise RuntimeError("Trajectory coordinator is shut down")
            if self._source is frame_source and (self._worker is not None or self._complete):
                return False

        self._replace_active_load(frame_source)
        return True

    def _replace_active_load(self, frame_source: Any) -> None:
        """Retire prior work and launch a new guarded generation."""
        with self._lock:
            prior = self._detach_active_locked()
            if prior is not None:
                self._retired.append(prior)
            self._generation += 1
            generation = self._generation
            self._source = frame_source
            self._snapshot = None
            self._progress = (0, 0)
            self._complete = False
            worker = _TrajectoryLoadWorker(
                generation,
                frame_source,
                is_current=lambda: self._is_current_generation(generation),
            )
            worker.signals.progress_updated.connect(
                self._on_worker_progress,
                Qt.QueuedConnection,
            )
            worker.signals.snapshot_updated.connect(
                self._on_worker_snapshot,
                Qt.QueuedConnection,
            )
            worker.signals.loading_complete.connect(
                self._on_worker_complete,
                Qt.QueuedConnection,
            )
            worker.signals.error_occurred.connect(
                self._on_worker_error,
                Qt.QueuedConnection,
            )
            thread = threading.Thread(target=worker.run, daemon=True)
            self._worker = worker
            self._thread = thread

        if prior is not None:
            prior[0].request_stop()
        self._reap_retired_workers()
        thread.start()
        self.cleared.emit()

    def _is_current_generation(self, generation: int) -> bool:
        """Return whether a worker still owns publication rights."""
        with self._lock:
            return not self._shutdown and generation == self._generation

    @Slot(int, int, int)
    def _on_worker_progress(self, generation: int, loaded: int, total: int) -> None:
        """Publish progress only for the active generation."""
        if not self._is_current_generation(generation):
            return
        with self._lock:
            if self._complete:
                return
            self._progress = (loaded, total)
        self.progress_updated.emit(loaded, total)

    @Slot(int, object)
    def _on_worker_snapshot(self, generation: int, snapshot: object) -> None:
        """Store and publish a partial immutable snapshot when current."""
        if not self._is_current_generation(generation) or not isinstance(
            snapshot,
            TrajectorySnapshot,
        ):
            return
        with self._lock:
            if self._complete:
                return
            self._snapshot = snapshot
        self.snapshot_updated.emit(snapshot)

    @Slot(int, object)
    def _on_worker_complete(self, generation: int, snapshot: object) -> None:
        """Commit the final snapshot only for the active generation."""
        if not self._is_current_generation(generation) or not isinstance(
            snapshot,
            TrajectorySnapshot,
        ):
            return
        with self._lock:
            self._snapshot = snapshot
            self._progress = (len(snapshot.frames_loaded), snapshot.total_frames)
            self._complete = True
        self.loading_complete.emit(snapshot)

    @Slot(int, str)
    def _on_worker_error(self, generation: int, message: str) -> None:
        """Publish an extraction failure only for the active generation."""
        if not self._is_current_generation(generation):
            return
        with self._lock:
            if self._complete:
                return
            self._worker = None
            self._thread = None
        self.error_occurred.emit(message)

    def reset(self, *, timeout: float = 2.0) -> bool:
        """Invalidate current data and stop workers while remaining reusable."""
        return self._retire_all(timeout=timeout, permanent=False)

    def shutdown(self, *, timeout: float = 2.0) -> bool:
        """Permanently retire all workers during application shutdown."""
        return self._retire_all(timeout=timeout, permanent=True)

    def _retire_all(self, *, timeout: float, permanent: bool) -> bool:
        """Clear active state and stop every retained worker generation."""
        with self._lock:
            if self._shutdown:
                return True
            self._shutdown = permanent
            self._generation += 1
            active = self._detach_active_locked()
            workers = list(self._retired)
            self._retired.clear()
            if active is not None:
                workers.append(active)
            self._source = None
            self._snapshot = None
            self._progress = (0, 0)
            self._complete = False
        stopped = self._stop_workers(workers, timeout=timeout)
        self.cleared.emit()
        return stopped

    def _detach_active_locked(
        self,
    ) -> tuple[_TrajectoryLoadWorker, threading.Thread] | None:
        """Detach and return the active worker pair while holding the lock."""
        worker = self._worker
        thread = self._thread
        self._worker = None
        self._thread = None
        if worker is None or thread is None:
            return None
        return worker, thread

    def _reap_retired_workers(self) -> None:
        """Drop completed retired workers without blocking the UI thread."""
        with self._lock:
            self._retired = [pair for pair in self._retired if pair[1].is_alive()]

    @staticmethod
    def _stop_workers(
        workers: list[tuple[_TrajectoryLoadWorker, threading.Thread]],
        *,
        timeout: float,
    ) -> bool:
        """Request cancellation and join workers within one total deadline."""
        for worker, _thread in workers:
            worker.request_stop()
        deadline = time.monotonic() + max(0.0, timeout)
        all_stopped = True
        for _worker, thread in workers:
            remaining = max(0.0, deadline - time.monotonic())
            if thread.is_alive() and remaining > 0.0:
                thread.join(timeout=remaining)
            if thread.is_alive():
                all_stopped = False
                logger.warning("Trajectory worker did not stop within %.1f seconds", timeout)
        return all_stopped
