"""Provider, cache, cancellation, and generation tests for scenario statistics."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from shared.frames import FrameProjection, ProjectedMPCFrame
from shared.frames.contracts import (
    PATH_METRIC_ORDER,
    PATH_METRIC_VALIDITY_BITS,
    FrameReadRequest,
)
from shared.frames.provider_base import DataProvider, ProviderInfo
from shared.frames.types import StandardMPCFrame
from visualizer.src.metrics.scenario_statistics import SCENARIO_STATISTICS_REQUEST
from visualizer.src.services.scenario_statistics_service import (
    ScenarioStatisticsService,
)
from visualizer.src.services.statistics_cache_service import StatisticsCacheService


def _projection(frame_index: int, *, path_loss_db: float = 10.0) -> FrameProjection:
    validity = sum(PATH_METRIC_VALIDITY_BITS.values())
    frame = ProjectedMPCFrame(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.asarray([0, 1], dtype=np.int64),
        bounce_offsets=np.asarray([0, 0], dtype=np.int64),
        interactions=np.empty((0,), dtype=np.uint8),
        delays_ns=np.asarray([2.0], dtype=np.float32),
        path_loss_db=np.asarray([path_loss_db], dtype=np.float32),
        aoa_az_deg=np.asarray([3.0], dtype=np.float32),
        aoa_el_deg=np.asarray([4.0], dtype=np.float32),
        aod_az_deg=np.asarray([5.0], dtype=np.float32),
        aod_el_deg=np.asarray([6.0], dtype=np.float32),
        metric_valid_bits=np.asarray([validity], dtype=np.uint8),
    )
    return FrameProjection.from_request(frame, SCENARIO_STATISTICS_REQUEST)


class _ProjectionProvider(DataProvider):
    def __init__(
        self,
        *,
        generation_id: str,
        frame_set_id: str,
        frame_indices: tuple[int, ...] = (0,),
        started: threading.Event | None = None,
        release: threading.Event | None = None,
        finished: threading.Event | None = None,
    ) -> None:
        self._info = ProviderInfo(
            name="projection-test",
            source="memory",
            total_frames=len(frame_indices),
            generation_id=generation_id,
            frame_set_id=frame_set_id,
        )
        self.frame_indices = frame_indices
        self.started = started
        self.release = release
        self.finished = finished
        self.list_calls = 0
        self.load_calls = 0
        self.projection_requests: list[FrameReadRequest] = []

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def list_frames(self) -> list[int]:
        self.list_calls += 1
        return list(self.frame_indices)

    def has_frame(self, step: int) -> bool:
        return step in self.frame_indices

    def load_frame(self, step: int) -> StandardMPCFrame:
        self.load_calls += 1
        raise AssertionError(f"full frame {step} must not be loaded")

    def iter_frame_projections(self, steps, request):
        self.projection_requests.append(request)
        try:
            for step in steps:
                if self.started is not None:
                    self.started.set()
                if self.release is not None:
                    assert self.release.wait(5.0)
                yield _projection(int(step), path_loss_db=10.0 + int(step))
        finally:
            if self.finished is not None:
                self.finished.set()


def _services(tmp_path: Path) -> tuple[StatisticsCacheService, ScenarioStatisticsService, Any]:
    scenario = SimpleNamespace(
        root=tmp_path / "scenario",
        frames_dir=tmp_path / "scenario" / "frames",
    )
    visualizer = SimpleNamespace(scenario=scenario)
    cache = StatisticsCacheService(visualizer)
    return cache, ScenarioStatisticsService(cache), scenario


def test_cache_hit_performs_zero_provider_frame_reads(tmp_path: Path):
    cache, service, scenario = _services(tmp_path)
    provider = _ProjectionProvider(generation_id="generation-a", frame_set_id="set-a")
    cached_stats = {"total_frames": 1, "total_mpcs": 7}
    assert cache.save_cached_stats(cached_stats, scenario, provider=provider) is not None

    result = service.collect(provider, scenario=scenario)

    assert result.from_cache is True
    assert result.stats["total_mpcs"] == 7
    assert provider.list_calls == 0
    assert provider.projection_requests == []
    assert provider.load_calls == 0


def test_cache_identity_invalidates_generation_and_frame_set(tmp_path: Path):
    cache, _, scenario = _services(tmp_path)
    original = _ProjectionProvider(generation_id="generation-a", frame_set_id="set-a")
    assert cache.save_cached_stats({"total_frames": 0}, scenario, provider=original) is not None

    assert cache.load_cached_stats(scenario, provider=original) is not None
    assert (
        cache.load_cached_stats(
            scenario,
            provider=_ProjectionProvider(
                generation_id="generation-b",
                frame_set_id="set-a",
            ),
        )
        is None
    )
    assert (
        cache.load_cached_stats(
            scenario,
            provider=_ProjectionProvider(
                generation_id="generation-a",
                frame_set_id="set-b",
            ),
        )
        is None
    )


def test_streaming_scan_uses_exact_statistics_projection(tmp_path: Path):
    _, service, scenario = _services(tmp_path)
    provider = _ProjectionProvider(
        generation_id="generation-a",
        frame_set_id="set-a",
        frame_indices=(4, 9),
    )
    progress: list[tuple[int, int]] = []

    result = service.collect(
        provider,
        scenario=scenario,
        force=True,
        on_progress=lambda current, total: progress.append((current, total)),
    )

    assert result.from_cache is False
    assert result.stats["frame_indices"] == [4, 9]
    assert provider.projection_requests == [SCENARIO_STATISTICS_REQUEST]
    assert provider.load_calls == 0
    assert progress == [(1, 2), (2, 2)]
    assert SCENARIO_STATISTICS_REQUEST.metrics == frozenset(PATH_METRIC_ORDER)


def test_new_generation_cancels_and_suppresses_stale_callbacks(tmp_path: Path):
    _, service, scenario = _services(tmp_path)
    first_started = threading.Event()
    release_first = threading.Event()
    first_finished = threading.Event()
    first = _ProjectionProvider(
        generation_id="generation-a",
        frame_set_id="set-a",
        started=first_started,
        release=release_first,
        finished=first_finished,
    )
    second = _ProjectionProvider(
        generation_id="generation-b",
        frame_set_id="set-b",
        frame_indices=(5,),
    )
    completions = []
    errors = []

    first_generation = service.start_collection(
        first,
        scenario=scenario,
        force=True,
        on_complete=completions.append,
        on_error=errors.append,
    )
    assert first_started.wait(5.0)

    second_generation = service.start_collection(
        second,
        scenario=scenario,
        force=True,
        on_complete=completions.append,
        on_error=errors.append,
    )
    assert second_generation > first_generation
    assert service.wait_for_idle(5.0)
    release_first.set()
    assert first_finished.wait(5.0)

    assert errors == []
    assert len(completions) == 1
    assert completions[0].generation == second_generation
    assert completions[0].stats["frame_indices"] == [5]
