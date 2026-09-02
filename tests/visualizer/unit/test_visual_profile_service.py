"""Unit tests for VisualProfileService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from visualizer.src.services.visual_profile_service import (
    BUILTIN_RULES,
    VisualProfileService,
    _compile_rule,
)


@pytest.fixture()
def service():
    viz = MagicMock()
    return VisualProfileService(viz)


# ------------------------------------------------------------------
# Rule compilation
# ------------------------------------------------------------------


class TestRuleCompilation:
    def test_compile_valid_rule(self):
        rule = {"match": {"name": "human.*", "type": "target"}, "preset": "Skin"}
        compiled = _compile_rule(rule, source="test")
        assert compiled is not None
        assert compiled.preset == "Skin"
        assert compiled.entry_type == "target"
        assert compiled.name_re is not None

    def test_compile_rule_no_preset(self):
        rule = {"match": {"name": "human.*"}}
        assert _compile_rule(rule, source="test") is None

    def test_compile_rule_bad_regex(self):
        rule = {"match": {"name": "[invalid"}, "preset": "Skin"}
        assert _compile_rule(rule, source="test") is None

    def test_compile_rule_bad_type(self):
        rule = {"match": {"type": "invalid"}, "preset": "Skin"}
        assert _compile_rule(rule, source="test") is None

    def test_compile_rule_itu_lowered(self):
        rule = {"match": {"itu": "Metal"}, "preset": "Chrome"}
        compiled = _compile_rule(rule, source="test")
        assert compiled is not None
        assert compiled.itu == "metal"

    def test_compile_rule_color_override(self):
        rule = {"match": {}, "preset": "Plastic", "color": [0.1, 0.2, 0.3]}
        compiled = _compile_rule(rule, source="test")
        assert compiled is not None
        assert compiled.color_override == [0.1, 0.2, 0.3]

    def test_compile_rule_no_match_dict(self):
        rule = {"preset": "Skin"}
        compiled = _compile_rule(rule, source="test")
        assert compiled is not None  # empty match = match everything


# ------------------------------------------------------------------
# Matching
# ------------------------------------------------------------------


class TestMatching:
    def test_match_by_name(self, service):
        service.load_scenario_rules([{"match": {"name": "human.*"}, "preset": "Skin"}])
        result = service.resolve("human_walking", "concrete", "target")
        assert result is not None
        assert result["roughness"] == pytest.approx(0.55)  # Skin preset

    def test_match_by_name_case_insensitive(self, service):
        service.load_scenario_rules([{"match": {"name": "HUMAN.*"}, "preset": "Skin"}])
        result = service.resolve("human_walking", "concrete", "target")
        assert result is not None

    def test_match_by_itu(self, service):
        service.load_scenario_rules([{"match": {"itu": "glass"}, "preset": "Clear Glass"}])
        result = service.resolve("window_01", "glass", "mesh")
        assert result is not None

    def test_match_by_type_target(self, service):
        service.load_scenario_rules([{"match": {"type": "target"}, "preset": "Matte"}])
        assert service.resolve("anything", "concrete", "target") is not None
        assert service.resolve("anything", "concrete", "mesh") is None

    def test_match_and_logic(self, service):
        """All conditions must match (AND logic)."""
        service.load_scenario_rules(
            [{"match": {"name": "xdrone.*", "itu": "metal"}, "preset": "Chrome"}]
        )
        # Both match
        assert service.resolve("xdrone_01", "metal", "target") is not None
        # Name matches but itu doesn't — no builtin for xdrone, so None
        assert service.resolve("xdrone_01", "glass", "target") is None
        # ITU matches but name doesn't
        assert service.resolve("xcar_01", "metal", "target") is None

    def test_no_match_returns_none(self, service):
        service.load_scenario_rules([{"match": {"name": "xyz_never"}, "preset": "Skin"}])
        assert service.resolve("human_walking", "concrete", "target") is not None
        # ^ Still matches because builtins have human.* → Skin

    def test_no_match_no_builtins(self, service):
        service.load_scenario_rules([])
        assert service.resolve("xyz_unique_name", "concrete", "mesh") is None

    def test_drone_target_no_longer_has_builtin_visual_override(self, service):
        service.load_scenario_rules([])
        assert service.resolve("Drone_01", "metal", "target") is None

    def test_scenario_rule_can_still_override_drone_target_explicitly(self, service):
        service.load_scenario_rules(
            [{"match": {"name": "drone.*", "type": "target"}, "preset": "Plastic"}]
        )
        result = service.resolve("Drone_01", "metal", "target")
        assert result is not None
        assert result["roughness"] == pytest.approx(0.35)

    def test_color_override_applied(self, service):
        service.load_scenario_rules(
            [{"match": {"name": "car.*"}, "preset": "Glossy", "color": [1, 0, 0]}]
        )
        result = service.resolve("car_body", "metal", "target")
        assert result is not None
        assert result["color"] == [1.0, 0.0, 0.0]

    def test_invalid_preset_skipped(self, service):
        service.load_scenario_rules(
            [
                {"match": {"name": "test.*"}, "preset": "NonexistentPreset"},
                {"match": {"name": "test.*"}, "preset": "Skin"},
            ]
        )
        result = service.resolve("test_obj", "concrete", "target")
        assert result is not None
        # Should skip the invalid preset and match the second rule
        assert result["roughness"] == pytest.approx(0.55)
        binding = service.resolve_binding("test_obj", "concrete", "target")
        assert binding is not None
        assert binding.preset == "Skin"


# ------------------------------------------------------------------
# Priority
# ------------------------------------------------------------------


class TestPriority:
    def test_scenario_rules_override_builtins(self, service):
        """Scenario rules take precedence over builtin defaults."""
        service.load_scenario_rules(
            [{"match": {"name": "human.*", "type": "target"}, "preset": "Leather"}]
        )
        result = service.resolve("human_walking", "concrete", "target")
        assert result is not None
        # Leather roughness (0.6) not Skin roughness (0.55)
        assert result["roughness"] == pytest.approx(0.6)

    def test_first_matching_rule_wins(self, service):
        service.load_scenario_rules(
            [
                {"match": {"name": "human.*"}, "preset": "Leather"},
                {"match": {"name": "human.*"}, "preset": "Fabric"},
            ]
        )
        result = service.resolve("human_01", "concrete", "target")
        assert result is not None
        assert result["roughness"] == pytest.approx(0.6)  # Leather, not Fabric

    def test_builtins_always_present(self, service):
        """Even with no scenario rules, builtins match."""
        result = service.resolve("human_walking", "concrete", "target")
        assert result is not None


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------


class TestLifecycle:
    def test_clear_scenario_rules(self, service):
        service.load_scenario_rules([{"match": {"name": "custom.*"}, "preset": "Gold"}])
        assert service.resolve("custom_obj", "metal", "target") is not None
        service.clear_scenario_rules()
        assert service.resolve("custom_obj", "metal", "target") is None

    def test_get_active_rules_round_trip(self, service):
        rules = [{"match": {"name": "test.*"}, "preset": "Skin"}]
        service.load_scenario_rules(rules)
        active = service.get_active_rules()
        assert active == rules

    def test_load_replaces_previous_rules(self, service):
        # Use names that don't match any builtin rule
        service.load_scenario_rules([{"match": {"name": "alpha.*"}, "preset": "Gold"}])
        service.load_scenario_rules([{"match": {"name": "beta.*"}, "preset": "Silver"}])
        assert service.resolve("alpha_obj", "concrete", "mesh") is None
        assert service.resolve("beta_obj", "concrete", "mesh") is not None


# ------------------------------------------------------------------
# Builtin rules sanity
# ------------------------------------------------------------------


class TestBuiltinRules:
    def test_builtin_rules_all_valid(self):
        """Every builtin rule should compile and reference a valid preset."""
        from visualizer.src.services.material_pbr_service import MaterialPBRService

        for rule in BUILTIN_RULES:
            compiled = _compile_rule(rule, source="builtin")
            assert compiled is not None, f"Builtin rule failed to compile: {rule}"
            assert (
                compiled.preset in MaterialPBRService.PRESETS
            ), f"Builtin preset {compiled.preset!r} not in PRESETS"
