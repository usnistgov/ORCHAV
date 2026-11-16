"""Focused tests for single-owner trajectory loading and publication."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from shared.frames.contracts import FrameReadRequest
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed import FrameProjection
from shared.frames.provider_base import DataProvider
from shared.frames.providers import Hdf5Provider
from shared.frames.types import StandardMPCFrame
from visualizer.src.io.frame_sources import FileSource
from visualizer.src.panels.trajectory_preview_panel import TrajectoryPreviewPanel
from visualizer.src.services.trajectory_load_service import (
    TrajectoryLoadCoordinator,
    TrajectorySnapshot,
)


class _MemoryFileSource(FileSource):
    """FileSource-shaped in-memory source with optional blocking and delay."""

    def __init__(
        self,
        frame_ids: list[int],
        *,
        x_offset: float = 0.0,
        release: threading.Event | None = None,
        entered: threading.Event | None = None,
        load_delay: float = 0.0,
    ) -> None:
        self.frame_ids = frame_ids
        self.x_offset = x_offset
        self.release = release
        self.entered = entered
        self.load_delay = load_delay
        self.load_calls: list[int] = []

    def list_frames(self) -> list[int]:
        return list(self.frame_ids)

    def load_frame(self, step: int) -> StandardMPCFrame:
        self.load_calls.append(step)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(timeout=2.0)
        if self.load_delay:
            time.sleep(self.load_delay)
        x = self.x_offset + float(step)
        return standard_mpc_frame_from_pair_data(
            frame_index=step,
            tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
            tx_positions=np.asarray([[x, 1.0, 2.0]], dtype=np.float64),
            rx_positions=np.asarray([[x, 3.0, 4.0]], dtype=np.float64),
            tx_names=("tx_0",),
            rx_names=("rx_0",),
            vertices_by_pair=[np.empty((0, 0, 3), dtype=np.float32)],
            interactions_by_pair=[np.empty((0, 0), dtype=np.int32)],
            path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
            target_positions_m=np.asarray([[x, 5.0, 6.0]], dtype=np.float64),
            targets_metadata=[
                {
                    "name": "pedestrian",
                    "current_position": [x, 5.0, 6.0],
                }
            ],
        )


class _MappedProviderSource(DataProvider):
    """Provider-shaped source whose public step must win over nested storage."""

    def __init__(self) -> None:
        self.projection_calls: list[int] = []
        self.load_calls: list[int] = []
        self.provider = _UnexpectedNestedProvider()

    def list_frames(self) -> list[int]:
        return [0]

    def has_frame(self, step: int) -> bool:
        return step == 0

    def load_frame(self, step: int) -> StandardMPCFrame:
        assert step == 0
        self.load_calls.append(step)
        return standard_mpc_frame_from_pair_data(
            frame_index=step,
            tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
            tx_positions=np.asarray([[101.0, 1.0, 2.0]], dtype=np.float64),
            rx_positions=np.asarray([[202.0, 3.0, 4.0]], dtype=np.float64),
            tx_names=("tx_0",),
            rx_names=("rx_0",),
            vertices_by_pair=[np.empty((0, 0, 3), dtype=np.float32)],
            interactions_by_pair=[np.empty((0, 0), dtype=np.int32)],
            path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
            target_positions_m=np.asarray([[303.0, 5.0, 6.0]], dtype=np.float64),
            targets_metadata=[{"name": "mapped", "current_position": [303.0, 5.0, 6.0]}],
        )

    def load_frame_projection(
        self,
        step: int,
        request: FrameReadRequest,
    ) -> FrameProjection:
        self.projection_calls.append(step)
        return super().load_frame_projection(step, request)


class _UnexpectedNestedProvider(DataProvider):
    def list_frames(self) -> list[int]:
        return [91]

    def has_frame(self, step: int) -> bool:
        return False

    def load_frame(self, step: int) -> StandardMPCFrame:
        raise AssertionError("nested measurement ids must not replace public source steps")


@pytest.fixture
def coordinator() -> Iterator[TrajectoryLoadCoordinator]:
    """Provide a coordinator that cannot leak a worker beyond its test."""
    service = TrajectoryLoadCoordinator()
    yield service
    service.shutdown()


def _join_and_deliver(thread: threading.Thread | None, *, timeout: float = 2.0) -> None:
    """Join one worker and deliver its queued Qt publications without a nested loop."""
    assert thread is not None
    thread.join(timeout=timeout)
    assert not thread.is_alive()
    QCoreApplication.processEvents()


def test_two_consumers_share_one_source_read_and_snapshot(
    coordinator: TrajectoryLoadCoordinator,
    qapp,
) -> None:
    source = _MemoryFileSource([0, 1, 2])
    first_consumer: list[TrajectorySnapshot] = []
    second_consumer: list[TrajectorySnapshot] = []
    coordinator.loading_complete.connect(first_consumer.append)
    coordinator.loading_complete.connect(second_consumer.append)

    assert coordinator.load(source) is True
    assert coordinator.load(source) is False
    _join_and_deliver(coordinator._thread)

    assert source.load_calls == [0, 1, 2]
    assert len(first_consumer) == 1
    assert first_consumer[0] is second_consumer[0]
    assert coordinator.snapshot is first_consumer[0]


def test_provider_source_uses_projection_without_bypassing_public_step_mapping(
    coordinator: TrajectoryLoadCoordinator,
    qapp,
) -> None:
    source = _MappedProviderSource()

    assert coordinator.load(source) is True
    _join_and_deliver(coordinator._thread)

    assert source.projection_calls == [0]
    assert source.load_calls == [0]
    snapshot = coordinator.snapshot
    assert snapshot is not None
    assert snapshot.tx_positions[0] == ((0, 101.0, 1.0, 2.0),)
    assert snapshot.rx_positions[0] == ((0, 202.0, 3.0, 4.0),)
    assert snapshot.target_positions["mapped"] == ((0, 303.0, 5.0, 6.0),)


def test_packed_trajectory_uses_batched_positions_and_bounded_progress(
    coordinator, qapp, tmp_path, monkeypatch
):
    from shared.frames import FrameComponent

    steps = list(range(0, 240, 2))
    source = _MemoryFileSource(steps)
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for step in steps:
        writer.append(source.load_frame(step))
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    original = provider.iter_frame_projections
    batches = []
    progress = []

    def projections(requested, request):
        assert request.components == frozenset({FrameComponent.DEVICES, FrameComponent.TARGETS})
        batches.append(list(requested))
        return original(requested, request)

    def forbidden(*args, **kwargs):
        pytest.fail("Trajectory scans must not use scalar or complete HDF5 frame reads")

    monkeypatch.setattr(provider, "iter_frame_projections", projections)
    monkeypatch.setattr(provider, "load_frame_projection", forbidden)
    monkeypatch.setattr(provider, "load_frame", forbidden)
    coordinator.progress_updated.connect(lambda *values: progress.append(values))
    try:
        coordinator.load(provider)
        _join_and_deliver(coordinator._thread)
        snapshot = coordinator.snapshot
        assert coordinator.is_complete
        assert len(batches) == 2  # Quick overview, then all remaining recorded samples.
        assert sorted(snapshot.frames_loaded) == steps
        assert sorted(snapshot.target_positions["pedestrian"]) == [
            (step, float(step), 5.0, 6.0) for step in steps
        ]
        assert progress[-1] == (len(steps), len(steps))
        assert len(progress) <= 30
    finally:
        coordinator.shutdown()
        provider.close()


def test_failed_batch_resumes_scalar_reads_without_losing_later_frames(coordinator, qapp):
    class Provider(_MappedProviderSource):
        def list_frames(self):
            return [0, 1, 2, 3]

        def load_frame(self, step):
            self.load_calls.append(step)
            if step == 2:
                raise OSError("unreadable frame")
            return _MemoryFileSource([step]).load_frame(step)

        def iter_frame_projections(self, steps, request):
            yield self.load_frame_projection(steps[0], request)
            raise OSError("batch failed")

    provider = Provider()
    coordinator.load(provider)
    _join_and_deliver(coordinator._thread)
    snapshot = coordinator.snapshot
    assert coordinator.is_complete
    assert snapshot.frames_loaded == (0, 1, 3)
    assert snapshot.target_positions["pedestrian"][-1] == (3, 3.0, 5.0, 6.0)
    assert provider.load_calls.count(0) == 1


def test_partial_snapshots_are_deeply_immutable_and_stable(
    coordinator: TrajectoryLoadCoordinator,
    qapp,
) -> None:
    source = _MemoryFileSource(list(range(6)))
    partials: list[TrajectorySnapshot] = []
    completed: list[TrajectorySnapshot] = []
    coordinator.snapshot_updated.connect(partials.append)
    coordinator.loading_complete.connect(completed.append)

    coordinator.load(source)
    _join_and_deliver(coordinator._thread)

    assert completed
    first_partial = partials[0]
    assert first_partial.frames_loaded == (0,)
    assert first_partial.tx_positions[0] == ((0, 0.0, 1.0, 2.0),)
    assert completed[0].frames_loaded == (0, 1, 2, 3, 4, 5)
    assert first_partial.frames_loaded == (0,)

    with pytest.raises(TypeError):
        first_partial.tx_positions[0] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        first_partial["tx_positions"] = {}  # type: ignore[index]


def test_stale_worker_completion_cannot_replace_new_generation(
    coordinator: TrajectoryLoadCoordinator,
    qapp,
) -> None:
    old_entered = threading.Event()
    old_release = threading.Event()
    old_source = _MemoryFileSource(
        [0],
        x_offset=10.0,
        entered=old_entered,
        release=old_release,
    )
    new_source = _MemoryFileSource([0], x_offset=200.0)
    completed: list[TrajectorySnapshot] = []
    coordinator.loading_complete.connect(completed.append)

    assert coordinator.load(old_source) is True
    old_thread = coordinator._thread
    assert old_entered.wait(timeout=1.0)
    assert coordinator.load(new_source) is True
    _join_and_deliver(coordinator._thread)
    assert completed
    active_snapshot = completed[-1]
    old_release.set()
    _join_and_deliver(old_thread)

    assert len(completed) == 1
    assert coordinator.snapshot is active_snapshot
    assert active_snapshot.tx_positions[0][0][1] == 200.0


def test_shutdown_stops_publication_and_is_idempotent(
    coordinator: TrajectoryLoadCoordinator,
    qapp,
) -> None:
    entered = threading.Event()
    source = _MemoryFileSource(
        list(range(100)),
        entered=entered,
        load_delay=0.01,
    )
    publications: list[TrajectorySnapshot] = []
    coordinator.snapshot_updated.connect(publications.append)

    coordinator.load(source)
    assert entered.wait(timeout=1.0)
    assert coordinator.shutdown(timeout=1.0) is True
    publication_count = len(publications)
    QCoreApplication.processEvents()

    assert coordinator.is_shutdown is True
    assert coordinator.snapshot is None
    assert len(publications) == publication_count
    assert coordinator.shutdown() is True
    with pytest.raises(RuntimeError, match="shut down"):
        coordinator.load(source)


@pytest.mark.parametrize("owned", [False, True])
def test_trajectory_scan_closes_only_explicitly_owned_sources(coordinator, qapp, owned):
    source = _MemoryFileSource([0, 1, 2])
    closed = []
    source.close = lambda: closed.append(True)
    coordinator.load(source, close_source=owned)
    _join_and_deliver(coordinator._thread)
    assert coordinator.is_complete
    assert closed == ([True] if owned else [])
    assert len(coordinator.snapshot.frames_loaded) == 3


def test_preview_progress_is_safe_while_panel_controls_are_being_built(
    coordinator: TrajectoryLoadCoordinator,
    qtbot,
) -> None:
    """Coordinator callbacks must not assume controls exist during startup."""

    class _Parent:
        trajectory_load_coordinator = coordinator
        frame_source = None

    panel = TrajectoryPreviewPanel(_Parent())

    assert panel._coordinator_connected is False
    panel._on_progress(1, 4)

    group = panel.create_panel()
    qtbot.addWidget(group)
    assert panel._coordinator_connected is True

    coordinator.progress_updated.emit(2, 4)
    assert panel.widgets["progress_bar"].value() == 50
    assert panel.widgets["status_label"].text() == "Loading: 2/4 frames"

    panel.cleanup()


def test_preview_catches_up_when_load_completed_before_panel_creation(
    coordinator: TrajectoryLoadCoordinator,
    qtbot,
) -> None:
    """A lazily built preview should render the coordinator's current snapshot."""
    completed: list[TrajectorySnapshot] = []
    coordinator.loading_complete.connect(completed.append)
    assert coordinator.load(_MemoryFileSource([0, 1, 2])) is True
    _join_and_deliver(coordinator._thread)
    assert completed

    class _Parent:
        trajectory_load_coordinator = coordinator
        frame_source = None

    panel = TrajectoryPreviewPanel(_Parent())
    group = panel.create_panel()
    qtbot.addWidget(group)

    assert panel._trajectories is completed[0]
    assert panel.widgets["status_label"].text() == "Loaded 3 frames"
    assert panel.widgets["progress_bar"].isHidden()

    panel.cleanup()
