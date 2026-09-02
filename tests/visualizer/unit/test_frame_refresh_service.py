"""Tests for current-frame refresh ownership."""

from __future__ import annotations

from types import SimpleNamespace

from visualizer.src.panels.data_source.raytracing_section import RaytracingControlSection
from visualizer.src.services.cache_service import CacheInvalidationScope
from visualizer.src.services.frame_refresh_service import FrameRefreshService


class _CacheService:
    def __init__(self) -> None:
        self.invalidations = []
        self.invalidated_frames = []

    def invalidate(self, scopes, *, reason: str) -> None:
        self.invalidations.append((scopes, reason))

    def invalidate_frame(self, frame: int) -> bool:
        self.invalidated_frames.append(frame)
        return True


def test_refresh_current_frame_invalidates_and_updates_public_frame_api() -> None:
    cache_service = _CacheService()
    updates = []
    scheduled = []
    viz = SimpleNamespace(
        animation_step=4,
        cache_service=cache_service,
        last_app_state=object(),
        force_update_next_frame=False,
        update_frame=lambda frame: updates.append(frame),
    )
    service = FrameRefreshService(
        viz,
        timer_single_shot=lambda delay_ms, callback: scheduled.append(delay_ms) or callback(),
    )

    assert service.refresh_current_frame_after_data_change(reason="PARAM_UPDATE") is True

    assert cache_service.invalidations == [(CacheInvalidationScope.FRAME_DATA, "PARAM_UPDATE")]
    assert cache_service.invalidated_frames == [4]
    assert viz.last_app_state is None
    assert viz.force_update_next_frame is True
    assert scheduled == [300]
    assert updates == [4]


def test_refresh_current_frame_returns_false_without_current_frame() -> None:
    cache_service = _CacheService()
    viz = SimpleNamespace(animation_step=None, cache_service=cache_service)
    service = FrameRefreshService(viz, timer_single_shot=lambda _delay, _callback: None)

    assert service.refresh_current_frame_after_data_change(reason="PARAM_UPDATE") is False
    assert cache_service.invalidations == []
    assert cache_service.invalidated_frames == []


def test_raytracing_section_delegates_reload_to_frame_refresh_service() -> None:
    refresh_calls = []
    private_calls = []
    refresh_service = SimpleNamespace(
        refresh_current_frame_after_data_change=lambda *, reason: refresh_calls.append(reason)
        or True
    )
    parent = SimpleNamespace(
        frame_refresh_service=refresh_service,
        _process_frame_step=lambda frame: private_calls.append(frame),
    )
    section = RaytracingControlSection(parent, {}, lambda: "")

    assert section._reload_current_frame() is True
    assert refresh_calls == ["PARAM_UPDATE"]
    assert private_calls == []
