from types import SimpleNamespace

import pytest

from generator.core.materials import (
    apply_material_settings,
    overrides,
    resolve_scene_material_family,
)
from generator.core.materials.defaults import DEFAULT_SCATTERING_COEFFICIENTS


class _DummyPattern:
    def __init__(self, name: str):
        self.name = name
        if name in {"directive", "backscattering"}:
            self.alpha_r = 1
        if name == "g-rer":
            self.alpha_g = 0.0


class _DummyMaterial:
    def __init__(self):
        self.relative_permittivity = 1.0
        self.conductivity = 0.0
        self.scattering_coefficient = 0.0
        self.scattering_pattern = "lambertian"
        self.xpd_coefficient = 0.0
        self.thickness = 0.01


class _PatternWithoutAlpha:
    def __init__(self, name: str):
        self.name = name


def _patch_pattern_registry(monkeypatch):
    monkeypatch.setattr(overrides, "_register_custom_scattering_patterns", lambda: None)
    monkeypatch.setattr(
        overrides,
        "_get_scattering_pattern_factory",
        lambda pattern_name: (lambda: _DummyPattern(pattern_name)),
    )


def test_apply_material_overrides_sets_extended_material_fields(monkeypatch):
    _patch_pattern_registry(monkeypatch)
    material = _DummyMaterial()
    scene = SimpleNamespace(radio_materials={"mat-wood": material})

    overrides.apply_material_overrides(
        scene,
        {
            "wood": {
                "relative_permittivity": 4.2,
                "conductivity": 0.31,
                "scattering_coefficient": 0.25,
                "xpd_coefficient": 0.6,
                "thickness": 0.02,
            }
        },
    )

    assert material.relative_permittivity == 4.2
    assert material.conductivity == 0.31
    assert material.scattering_coefficient == 0.25
    assert material.xpd_coefficient == 0.6
    assert material.thickness == 0.02


def test_apply_material_overrides_canonicalizes_custom_pattern_aliases(monkeypatch):
    _patch_pattern_registry(monkeypatch)
    material = _DummyMaterial()
    scene = SimpleNamespace(radio_materials={"mat-metal": material})

    overrides.apply_material_overrides(
        scene,
        {
            "metal": {
                "scattering_pattern": "erisotropic",
            }
        },
    )

    assert isinstance(material.scattering_pattern, _DummyPattern)
    assert material.scattering_pattern.name == "er-isotropic"


def test_resolve_material_name_accepts_mat_prefixed_itu_alias():
    scene = SimpleNamespace(radio_materials={"mat-itu_concrete": _DummyMaterial()})

    assert overrides.resolve_material_name(scene, "mat-concrete") == "mat-itu_concrete"


def test_apply_material_overrides_routes_alpha_g_to_scattering_pattern(monkeypatch):
    _patch_pattern_registry(monkeypatch)
    material = _DummyMaterial()
    scene = SimpleNamespace(radio_materials={"mat-glass": material})

    overrides.apply_material_overrides(
        scene,
        {
            "glass": {
                "scattering_pattern": "grer",
                "alpha_g": 7.5,
            }
        },
    )

    assert isinstance(material.scattering_pattern, _DummyPattern)
    assert material.scattering_pattern.name == "g-rer"
    assert material.scattering_pattern.alpha_g == 7.5


def test_public_supported_scattering_patterns_register_without_private_extension(monkeypatch):
    import generator.core.materials.scattering_patterns as scattering_patterns

    def _raise_missing_private(_module_name: str):
        raise ImportError("private extension intentionally unavailable")

    monkeypatch.setattr(scattering_patterns, "_PRIVATE_EXTENSIONS_LOADED", False)
    monkeypatch.setattr(scattering_patterns, "_SUPPORTED_PATTERNS_REGISTERED", False)
    monkeypatch.setattr(scattering_patterns, "import_module", _raise_missing_private)

    scattering_patterns.register_extra_scattering_patterns()
    material = _DummyMaterial()

    overrides.set_material_property(material, "scattering_pattern", "g-rer")
    assert type(material.scattering_pattern).__name__ == "GaussianRERPattern"
    assert hasattr(material.scattering_pattern, "alpha_g")

    overrides.set_material_property(material, "scattering_pattern", "isotropic")
    assert type(material.scattering_pattern).__name__ == "IsotropicPattern"

    overrides.set_material_property(material, "scattering_pattern", "er-isotropic")
    assert type(material.scattering_pattern).__name__ == "ERIsotropicPattern"


def test_apply_material_overrides_routes_alpha_r_to_scattering_pattern(monkeypatch):
    _patch_pattern_registry(monkeypatch)
    material = _DummyMaterial()
    scene = SimpleNamespace(radio_materials={"mat-concrete": material})

    overrides.apply_material_overrides(
        scene,
        {
            "concrete": {
                "scattering_pattern": "backscattering",
                "alpha_r": 8.6,
            }
        },
    )

    assert isinstance(material.scattering_pattern, _DummyPattern)
    assert material.scattering_pattern.name == "backscattering"
    assert material.scattering_pattern.alpha_r == 9


def test_set_material_property_invalid_scattering_pattern_raises(monkeypatch):
    def _raise_unknown_pattern(pattern_name: str):
        raise ValueError(f"Unknown scattering_pattern '{pattern_name}'")

    monkeypatch.setattr(overrides, "_register_custom_scattering_patterns", lambda: None)
    monkeypatch.setattr(
        overrides,
        "_get_scattering_pattern_factory",
        _raise_unknown_pattern,
    )
    material = _DummyMaterial()

    with pytest.raises(ValueError, match="Unknown scattering_pattern 'not-a-pattern'"):
        overrides.set_material_property(material, "scattering_pattern", "not-a-pattern")

    assert material.scattering_pattern == "lambertian"


def test_set_material_property_empty_scattering_pattern_raises(monkeypatch):
    monkeypatch.setattr(overrides, "_register_custom_scattering_patterns", lambda: None)
    material = _DummyMaterial()

    with pytest.raises(ValueError, match="Empty scattering_pattern is not allowed"):
        overrides.set_material_property(material, "scattering_pattern", "   ")

    assert material.scattering_pattern == "lambertian"


def test_alpha_r_failed_instantiation_does_not_mutate_material(monkeypatch):
    monkeypatch.setattr(overrides, "_register_custom_scattering_patterns", lambda: None)
    monkeypatch.setattr(
        overrides,
        "_get_scattering_pattern_factory",
        lambda pattern_name: (lambda: _PatternWithoutAlpha(pattern_name)),
    )
    material = _DummyMaterial()
    material.scattering_pattern = "backscattering"

    with pytest.raises(AttributeError, match="alpha-capable scattering pattern"):
        overrides.set_material_property(material, "alpha_r", 4.0)

    assert material.scattering_pattern == "backscattering"


def test_alpha_g_failed_instantiation_does_not_mutate_material(monkeypatch):
    monkeypatch.setattr(overrides, "_register_custom_scattering_patterns", lambda: None)
    monkeypatch.setattr(
        overrides,
        "_get_scattering_pattern_factory",
        lambda pattern_name: (lambda: _PatternWithoutAlpha(pattern_name)),
    )
    material = _DummyMaterial()
    material.scattering_pattern = "g-rer"

    with pytest.raises(AttributeError, match="alpha-capable scattering pattern"):
        overrides.set_material_property(material, "alpha_g", 4.0)

    assert material.scattering_pattern == "g-rer"


def test_apply_material_settings_applies_overrides_when_diffusion_disabled():
    override_material = _DummyMaterial()
    mapped_material = _DummyMaterial()
    scene = SimpleNamespace(
        radio_materials={"mat-wood": override_material},
        objects={"Wall": SimpleNamespace(radio_material=mapped_material)},
    )

    apply_material_settings(
        scene,
        default_scattering_preset="none",
        material_overrides={"wood": {"scattering_coefficient": 0.42}},
    )

    assert override_material.scattering_coefficient == 0.42
    assert mapped_material.scattering_coefficient == 0.0


def test_resolve_scene_material_family_accepts_sionna_aliases():
    assert resolve_scene_material_family("concrete") == "concrete"
    assert resolve_scene_material_family("itu_concrete") == "concrete"
    assert resolve_scene_material_family("mat-itu_concrete") == "concrete"
    assert resolve_scene_material_family("scene-material-itu_concrete") == "concrete"
    assert resolve_scene_material_family("mat-itu_glass_Target") is None
    assert resolve_scene_material_family("mat-itu_brick") is None


def test_apply_material_settings_leaves_scene_materials_unchanged_when_preset_none():
    concrete = _DummyMaterial()
    concrete.scattering_coefficient = 0.17
    wood = _DummyMaterial()
    scene = SimpleNamespace(
        radio_materials={"mat-itu_concrete": concrete, "mat-itu_wood": wood},
        objects={},
    )

    apply_material_settings(scene, default_scattering_preset="none")

    assert concrete.scattering_coefficient == 0.17
    assert wood.scattering_coefficient == 0.0


def test_apply_material_settings_applies_itu_defaults_by_material_family():
    concrete = _DummyMaterial()
    wood = _DummyMaterial()
    brick = _DummyMaterial()
    target = _DummyMaterial()
    target.scattering_coefficient = 0.3
    scene = SimpleNamespace(
        radio_materials={
            "mat-itu_concrete": concrete,
            "itu_wood": wood,
            "mat-itu_brick": brick,
            "mat-itu_glass_Target": target,
        },
        objects={},
    )

    apply_material_settings(scene, default_scattering_preset="itu")

    assert concrete.scattering_coefficient == DEFAULT_SCATTERING_COEFFICIENTS["concrete"]
    assert wood.scattering_coefficient == DEFAULT_SCATTERING_COEFFICIENTS["wood"]
    assert brick.scattering_coefficient == 0.0
    assert target.scattering_coefficient == 0.3


def test_apply_material_settings_explicit_overrides_win_over_default_scattering():
    material = _DummyMaterial()
    scene = SimpleNamespace(
        radio_materials={"mat-plasterboard": material},
        objects={"Wall": SimpleNamespace(radio_material=material)},
    )

    apply_material_settings(
        scene,
        default_scattering_preset="itu",
        material_overrides={"plasterboard": {"scattering_coefficient": 0.12}},
    )

    assert material.scattering_coefficient == 0.12


def test_apply_material_settings_explicit_zero_override_wins_over_default_scattering():
    material = _DummyMaterial()
    scene = SimpleNamespace(radio_materials={"mat-itu_concrete": material}, objects={})

    apply_material_settings(
        scene,
        default_scattering_preset="itu",
        material_overrides={"concrete": {"scattering_coefficient": 0.0}},
    )

    assert material.scattering_coefficient == 0.0
