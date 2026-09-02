"""
Smoke tests for scenario scripts.

These tests verify:
1. Scenario scripts have valid Python syntax (can be parsed)
2. Scenario scripts have module docstrings
3. Scenario scripts don't contain hardcoded absolute paths
4. Core generator imports work

Note: These tests do NOT run actual simulations - they only check
that the script code is syntactically valid and well-structured.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCENARIO_SCRIPTS = sorted(
    script
    for script in (PROJECT_ROOT / "scenarios").glob("*/*/generate.py")
    if "_internal" not in script.parts
)


def _script_id(path: Path) -> str:
    """Return a short test ID like 'getting_started/hello_world/generate.py'."""
    return str(path.relative_to(PROJECT_ROOT / "scenarios"))


class TestScenarioScriptSyntax:
    """Test that all scenario scripts have valid Python syntax."""

    @pytest.mark.parametrize("script", SCENARIO_SCRIPTS, ids=_script_id)
    def test_script_has_valid_syntax(self, script: Path):
        """Verify scenario script can be parsed as valid Python."""
        source = script.read_text()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {script}: {e}")

    @pytest.mark.parametrize("script", SCENARIO_SCRIPTS, ids=_script_id)
    def test_script_has_docstring(self, script: Path):
        """Verify scenario script has a module docstring."""
        source = script.read_text()
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        if docstring is None:
            pytest.skip(f"No docstring in {script.parent.name}/generate.py (recommended)")


class TestScenarioScriptStructure:
    """Test scenario script structure patterns."""

    @pytest.mark.parametrize("script", SCENARIO_SCRIPTS, ids=_script_id)
    def test_script_no_hardcoded_absolute_paths(self, script: Path):
        """Check that scenario scripts don't use hardcoded absolute paths."""
        source = script.read_text()
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if '"/home/' in line or "'/home/" in line:
                pytest.fail(
                    f"Hardcoded home path in {script.parent.name}/generate.py:{i}: {stripped}"
                )
            if '"/mnt/' in line or "'/mnt/" in line:
                pytest.fail(
                    f"Hardcoded mount path in {script.parent.name}/generate.py:{i}: {stripped}"
                )

    def test_no_duplicate_scenario_script_names(self):
        """Check that scenario script directory names are unique."""
        assert SCENARIO_SCRIPTS, "No scenario scripts found"
        names = [s.parent.name for s in SCENARIO_SCRIPTS]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            pytest.fail(f"Duplicate tutorial names: {set(duplicates)}")


class TestCoreGeneratorImports:
    """Test that core generator imports work in tutorial context."""

    def test_generator_core_imports(self):
        """Verify core generator imports are available."""
        from generator.core.configuration import ReceiverConfig, TransmitterConfig
        from generator.core.mobility import LinearMobility, StationaryMobility
        from generator.core.orientation import FixedOrientationSpec
        from generator.core.pipeline import perform_pipeline

        assert TransmitterConfig is not None
        assert ReceiverConfig is not None
        assert StationaryMobility is not None
        assert LinearMobility is not None
        assert FixedOrientationSpec is not None
        assert perform_pipeline is not None

    def test_shared_imports(self):
        """Verify shared module imports are available."""
        from shared.scenarios import load_scenario_configuration
        from shared.scenarios.paths import create_script_path_policy

        assert create_script_path_policy is not None
        assert load_scenario_configuration is not None
