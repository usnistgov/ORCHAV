from pathlib import Path
from types import SimpleNamespace
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


@pytest.mark.parametrize("scene_metal", [False, True])
def test_target_family_override_does_not_report_missing_scene_material(caplog, scene_metal):
    from sionna.rt import load_scene
    from sionna.rt.radio_materials import ITURadioMaterial

    from generator.core.target import TargetConfig

    cfg = SimpleNamespace(material_overrides={"metal": {"scattering_coefficient": 0.99}})
    service = SceneService(cfg)
    service.scene = load_scene()
    if scene_metal:
        service.scene.add(ITURadioMaterial(name="metal", itu_type="metal", thickness=0.1))
    mesh_dir = Path(__file__).resolve().parents[4] / "libraries/targets/cube"
    target = TargetConfig(
        name="Person",
        mobility=None,
        mesh_pattern="cube.ply",
        mesh_directory=str(mesh_dir),
        resolved_mesh_directory=mesh_dir,
        material_type="metal",
        _initial_position=(0.0, 0.0, 0.0),
    )
    service._create_targets([target])
    material = service.target_managers[0].target_material
    assert float(material.scattering_coefficient[0]) == pytest.approx(0.99)

    with caplog.at_level("WARNING"):
        service._apply_material_overrides()
    assert "not found in scene" not in caplog.text
    assert float(material.scattering_coefficient[0]) == pytest.approx(0.99)
    if scene_metal:
        assert float(
            service.scene.radio_materials["metal"].scattering_coefficient[0]
        ) == pytest.approx(0.99)

    cfg.material_overrides["missing_material"] = {"scattering_coefficient": 0.5}
    with caplog.at_level("WARNING"):
        service._apply_material_overrides()
    assert "Material override 'missing_material' not found in scene" in caplog.text
