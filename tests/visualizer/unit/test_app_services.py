"""Observable contracts for visualizer service composition."""

from __future__ import annotations

from visualizer.src.app import services as app_services
from visualizer.src.services.target_asset_cache import TargetAssetCache


class _RecordingViz:
    """Tiny visualizer double that records composition-root installation."""

    def __init__(self) -> None:
        object.__setattr__(self, "assignments", [])
        object.__setattr__(self, "_record_assignments", False)
        self._frame_loader_cache_size = 7
        self._renderer_type = "fake-renderer"
        self.target_asset_cache = TargetAssetCache()
        object.__setattr__(self, "_record_assignments", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_record_assignments", False):
            self.assignments.append(name)
        object.__setattr__(self, name, value)


class _FakeRenderer:
    pass


class _FakeSessionService:
    def __init__(self, viz) -> None:
        self.viz = viz


class _FakePresetService:
    pass


class _FakeMaterialPBRService:
    def __init__(self, viz) -> None:
        self.visualizer = viz


def _patch_safe_construction(monkeypatch):
    renderer = _FakeRenderer()
    create_renderer_calls = []

    def _create_renderer(viz, renderer_type):
        create_renderer_calls.append((viz, renderer_type))
        return renderer

    monkeypatch.setattr(app_services, "create_renderer", _create_renderer)
    monkeypatch.setattr(app_services, "SessionService", _FakeSessionService)
    monkeypatch.setattr(app_services, "PresetService", _FakePresetService)
    monkeypatch.setattr(app_services, "MaterialPBRService", _FakeMaterialPBRService)
    return renderer, create_renderer_calls


def test_build_service_bundle_is_non_mutating_and_shares_core_owners(monkeypatch):
    renderer, create_renderer_calls = _patch_safe_construction(monkeypatch)
    viz = _RecordingViz()
    try:
        bundle = app_services.build_service_bundle(viz, default_animation_cache_size=12)

        assert isinstance(bundle, app_services.AppServiceBundle)
        assert create_renderer_calls == [(viz, "fake-renderer")]
        assert viz.assignments == []
        assert not hasattr(viz, "cache_service")
        assert bundle.renderer is renderer
        assert bundle.pipeline.metrics_service is bundle.metrics_service
        assert bundle.pipeline.coverage_service is bundle.coverage_service
        assert bundle.animation_service.pipeline is bundle.pipeline
        assert bundle.animation_service.cache_service is bundle.cache_service
        assert bundle.node_service.target_service is bundle.target_service
    finally:
        viz.target_asset_cache.close()


def test_construct_services_installs_entry_collaborators_and_loader(monkeypatch):
    renderer, create_renderer_calls = _patch_safe_construction(monkeypatch)
    viz = _RecordingViz()
    try:
        result = app_services.construct_services(viz, default_animation_cache_size=9)

        assert result is None
        assert create_renderer_calls == [(viz, "fake-renderer")]
        assert viz.renderer is renderer
        assert viz.pipeline.metrics_service is viz.metrics_service
        assert viz.pipeline.coverage_service is viz.coverage_service
        assert viz.animation_service.pipeline is viz.pipeline
        assert viz.animation_service.cache_service is viz.cache_service
        assert viz.node_service.target_service is viz.target_service
        assert viz.ui_controller._menu_ctrl._scenario_loader is viz.scenario_loader_service
    finally:
        viz.target_asset_cache.close()
