from pathlib import Path

from generator.core.configuration import build_simulation_config
from shared.scenarios import load_scenario_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_ROOT = "scenarios/generator/propagation_and_materials/scene_diffuse_scattering"


def _load_simulation_config(relative_scenario: str):
    scenario = load_scenario_configuration(PROJECT_ROOT / relative_scenario)
    return build_simulation_config(scenario)


def _assert_diffuse_solving_enabled(relative_scenario: str) -> None:
    scenario = load_scenario_configuration(PROJECT_ROOT / relative_scenario)

    assert scenario.raytracing is not None
    quality = scenario.raytracing["quality"]
    assert quality["custom"]["diffuse_reflection"] is True


def test_diffuse_solver_switch_fixture_leaves_scene_materials_unchanged():
    scenario_path = f"{SCENARIO_ROOT}/scenario.yaml"
    cfg = _load_simulation_config(scenario_path)

    assert cfg.scene_material_scattering_coefficient_preset == "none"
    assert cfg.material_overrides is None
    _assert_diffuse_solving_enabled(scenario_path)


def test_diffuse_itu_preset_fixture_assigns_family_preset():
    scenario_path = f"{SCENARIO_ROOT}/itu_preset/scenario.yaml"
    cfg = _load_simulation_config(scenario_path)

    assert cfg.scene_material_scattering_coefficient_preset == "itu"
    assert cfg.material_overrides is None
    _assert_diffuse_solving_enabled(scenario_path)


def test_diffuse_itu_preset_concrete_override_fixture_applies_both_layers():
    scenario_path = f"{SCENARIO_ROOT}/itu_preset_concrete_override/scenario.yaml"
    cfg = _load_simulation_config(scenario_path)

    assert cfg.scene_material_scattering_coefficient_preset == "itu"
    assert cfg.material_overrides["itu_concrete"]["scattering_coefficient"] == 0.0
    _assert_diffuse_solving_enabled(scenario_path)


def test_diffuse_explicit_concrete_fixture_uses_material_override_without_preset():
    scenario_path = f"{SCENARIO_ROOT}/explicit_concrete/scenario.yaml"
    cfg = _load_simulation_config(scenario_path)

    assert cfg.scene_material_scattering_coefficient_preset == "none"
    assert cfg.material_overrides["itu_concrete"]["scattering_coefficient"] == 0.1
    _assert_diffuse_solving_enabled(scenario_path)
