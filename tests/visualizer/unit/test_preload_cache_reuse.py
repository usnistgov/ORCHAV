"""Resident-frame reuse across selected preload runs and cache eviction."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication

from shared.frames import FrameComponent, FrameReadRequest, project_standard_mpc_frame
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.io.packed_frame_payload import (
    projection_to_visual_frame,
    visual_frame_read_request,
)
from visualizer.src.services.animation_service import AnimationService, _PreloadWorker
from visualizer.src.services.cache_service import CacheService


class CountingProvider:
    """Real projection conversion with explicit provider read counts."""

    def __init__(self):
        self.batches = []
        self.individual = []
        self.before_batch = lambda: None

    def list_frames(self):
        return [0, 1, 2, 3]

    def iter_frame_projections(self, steps, request):
        self.batches.append(list(steps))
        self.before_batch()
        for step in steps:
            yield project_standard_mpc_frame(build_standard_mpc_frame(frame_idx=step), request)

    def load_frame_projection(self, step, request):
        self.individual.append(step)
        return project_standard_mpc_frame(build_standard_mpc_frame(frame_idx=step), request)


def visual_frame(step):
    return projection_to_visual_frame(
        project_standard_mpc_frame(
            build_standard_mpc_frame(frame_idx=step), visual_frame_read_request()
        )
    )


def worker_for(cache, provider, steps, **kwargs):
    return _PreloadWorker(
        SimpleNamespace(provider=provider),
        None,
        cache,
        steps,
        is_current=kwargs.pop("is_current", lambda: True),
        run_if_current=kwargs.pop("run_if_current", lambda action: action() or True),
        **kwargs,
    )


@pytest.mark.parametrize("partial_inventory", [False, True])
def test_preload_reuses_complete_frames_and_reads_incomplete_payload(qapp, partial_inventory):
    cache = CacheService(SimpleNamespace())
    provider = CountingProvider()
    resident = {step: visual_frame(step) for step in (3, 5)}
    for step, frame in resident.items():
        cache.store_frame(step, frame)
    incomplete = {"tx_positions": [[0, 0, 0]]}
    if partial_inventory:
        request = visual_frame_read_request()
        incomplete = projection_to_visual_frame(
            project_standard_mpc_frame(
                build_standard_mpc_frame(frame_idx=4),
                FrameReadRequest(
                    components=request.components - {FrameComponent.PROVENANCE},
                    metrics=request.metrics,
                ),
            )
        )
    cache.store_frame(4, incomplete)
    warmed, ready, completed = [], [], []
    worker = worker_for(
        cache,
        provider,
        [3, 4, 5],
        mpc_core=SimpleNamespace(precompute_canonical=lambda step, _: warmed.append(step)),
    )
    worker.signals.step_ready.connect(ready.append)
    worker.signals.complete.connect(lambda loaded, failed, *_: completed.append((loaded, failed)))

    worker.run()

    assert provider.batches == [[4]]
    assert provider.individual == []
    assert ready == warmed == [3, 4, 5]
    assert completed == [(3, 0)]
    assert cache.preloaded_steps == [3, 4, 5]
    assert cache.get_frame(3) is resident[3]
    assert cache.get_frame(5) is resident[5]
    assert cache.get_frame(4)["_source"]["frame_idx"] == 4


def test_evicted_resident_frame_falls_back_without_consuming_batch(qapp):
    cache = CacheService(SimpleNamespace(), max_frame_cache_size=2)
    provider = CountingProvider()
    cache.store_frame(3, visual_frame(3))
    cache.store_frame(5, visual_frame(5))
    provider.before_batch = lambda: cache.store_frame(8, visual_frame(8))
    worker = worker_for(cache, provider, [3, 4, 5, 6])

    frames = list(worker._iter_frame_data())

    assert provider.batches == [[4, 6]]
    assert provider.individual == [5]
    assert cache.frame_cache.stats.evictions == 1
    assert [step for step, _ in frames] == [3, 4, 5, 6]
    assert [frame["_source"]["frame_idx"] for _, frame in frames] == [3, 4, 5, 6]


@pytest.mark.parametrize("evicted", [False, True])
def test_retired_generation_does_not_publish_cached_frame(qapp, evicted):
    current = True
    cache = CacheService(SimpleNamespace())
    cache.store_frame(3, visual_frame(3))
    provider = CountingProvider()
    ready = []
    worker = worker_for(cache, provider, [3], is_current=lambda: current)
    original_get = worker._cached_visual_frame
    lookups = 0

    def retire_during_reuse(step):
        nonlocal current, lookups
        lookups += 1
        if lookups == 2:
            current = False
            if evicted:
                return None
        return original_get(step)

    worker._cached_visual_frame = retire_during_reuse
    worker.signals.step_ready.connect(ready.append)

    worker.run()

    assert ready == []
    assert cache.preloaded_steps == []
    assert provider.batches == provider.individual == []


def test_same_source_timeline_toggle_reuses_resident_selected_frames(qapp):
    provider = CountingProvider()
    loader = SimpleNamespace(provider=provider, list_frames=provider.list_frames)
    viz = SimpleNamespace(
        frame_source=SimpleNamespace(list_frames=provider.list_frames),
        frame_loader=loader,
        sensing_playback=SimpleNamespace(steps=None),
        animation_step=0,
    )
    cache = CacheService(viz)
    viz.cache_service = cache
    service = AnimationService(SimpleNamespace(), viz, frame_loader=loader, cache_service=cache)

    def preload_ready_steps():
        ready = []
        assert service.start_preloading(on_step_ready=ready.append)
        service._threaded_preloader._thread.join(timeout=2.0)
        assert not service._threaded_preloader._thread.is_alive()
        QCoreApplication.processEvents()
        assert service.preloading_completed
        return ready

    try:
        assert preload_ready_steps() == [0, 1, 2, 3]
        resident = {step: cache.get_frame(step) for step in provider.list_frames()}
        service.reset_preloading_state()
        viz.sensing_playback.steps = (1, 3)
        assert preload_ready_steps() == [1, 3]
        service.reset_preloading_state()
        viz.sensing_playback.steps = None
        assert preload_ready_steps() == [0, 1, 2, 3]
        assert provider.batches == [[0, 1, 2, 3]]
        assert provider.individual == []
        assert all(cache.get_frame(step) is frame for step, frame in resident.items())
    finally:
        service.reset_preloading_state()
