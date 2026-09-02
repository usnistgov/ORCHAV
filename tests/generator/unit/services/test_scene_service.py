from unittest.mock import MagicMock

import pytest

from generator.core import services
from generator.core.services.actor_state_service import ActorStateService
from generator.core.services.coverage_service import CoverageService
from generator.core.services.raytracing_service import RayTracingService
from generator.core.services.scene_service import SceneService


def test_cleanup_releases_scene_references():
    service = SceneService(MagicMock())
    service.scene = MagicMock()
    service.tx_list = [MagicMock()]
    service.rx_list = [MagicMock()]
    service.target_managers = [MagicMock()]
    service.target_objects = [MagicMock()]

    service.cleanup()

    assert service.scene is None
    assert service.tx_list == []
    assert service.rx_list == []
    assert service.target_managers == []
    assert service.target_objects == []


def test_services_package_exports_run_scoped_services():
    assert services.SceneService is SceneService
    assert services.ActorStateService is ActorStateService
    assert services.RayTracingService is RayTracingService
    assert services.CoverageService is CoverageService


def test_requested_scene_material_default_failure_raises(monkeypatch):
    import generator.core.materials as materials

    cfg = MagicMock()
    cfg.scene_material_scattering_coefficient_preset = "itu"
    service = SceneService(cfg)
    service.scene = object()

    def _raise_material_failure(*_args, **_kwargs):
        raise RuntimeError("material default failed")

    monkeypatch.setattr(materials, "apply_material_settings", _raise_material_failure)

    with pytest.raises(RuntimeError, match="material default failed"):
        service._apply_scene_material_defaults()


def test_unrequested_scene_material_default_is_noop(monkeypatch):
    import generator.core.materials as materials

    cfg = MagicMock()
    cfg.scene_material_scattering_coefficient_preset = "none"
    service = SceneService(cfg)
    service.scene = object()

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("material helper should not be called")

    monkeypatch.setattr(materials, "apply_material_settings", _raise_if_called)

    service._apply_scene_material_defaults()


def test_requested_material_override_failure_raises(monkeypatch):
    import generator.core.materials as materials

    cfg = MagicMock()
    cfg.material_overrides = {"concrete": {"scattering_coefficient": 0.5}}
    service = SceneService(cfg)
    service.scene = object()

    def _raise_material_failure(*_args, **_kwargs):
        raise RuntimeError("material override failed")

    monkeypatch.setattr(materials, "apply_material_settings", _raise_material_failure)

    with pytest.raises(RuntimeError, match="material override failed"):
        service._apply_material_overrides()
