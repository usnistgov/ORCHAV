"""Tests for PresetService."""

import json
from pathlib import Path

import pytest

from visualizer.src.services.preset_service import PresetService
from visualizer.src.state import MPC_ORDER_VALUES, MPC_TYPE_VALUES


@pytest.fixture
def preset_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PresetService:
    """Create a PresetService with a temporary preset directory."""
    # Override the preset directory to use tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return PresetService()


@pytest.fixture
def user_preset_path(preset_service: PresetService) -> Path:
    """Get the path to user presets directory."""
    return preset_service.preset_dir


# =============================================================================
# Built-in Presets Tests
# =============================================================================


def test_built_in_presets_exist(preset_service: PresetService) -> None:
    """Verify built-in presets are created on initialization."""
    assert len(preset_service.built_in_presets) > 0


def test_built_in_presets_have_required_fields(preset_service: PresetService) -> None:
    """Verify all built-in presets have required structure."""
    required_fields = ["description", "mpc_allowed_orders", "mpc_allowed_types", "materials"]

    for name, preset in preset_service.built_in_presets.items():
        for field in required_fields:
            assert field in preset, f"Preset '{name}' missing required field '{field}'"


def test_all_mpcs_preset_shows_everything(preset_service: PresetService) -> None:
    """Verify 'All MPCs (Reset)' shows all MPC types."""
    preset = preset_service.built_in_presets.get("All MPCs (Reset)")
    assert preset is not None
    assert preset["mpc_allowed_orders"] == list(MPC_ORDER_VALUES)
    assert preset["mpc_allowed_types"] == list(MPC_TYPE_VALUES)
    assert 99 in preset["mpc_allowed_types"]


def test_los_only_preset(preset_service: PresetService) -> None:
    """Verify 'LoS Only' preset shows only line-of-sight MPCs."""
    preset = preset_service.built_in_presets.get("LoS Only")
    assert preset is not None
    assert preset["mpc_allowed_orders"] == [0]
    assert preset["mpc_allowed_types"] == [0]


def test_single_bounce_preset(preset_service: PresetService) -> None:
    """Verify 'Single Bounce' preset shows LoS and single-bounce MPCs."""
    preset = preset_service.built_in_presets.get("Single Bounce")
    assert preset is not None
    assert preset["mpc_allowed_orders"] == [0, 1]


def test_specular_only_preset(preset_service: PresetService) -> None:
    """Verify 'Specular Only' preset excludes LoS."""
    preset = preset_service.built_in_presets.get("Specular Only")
    assert preset is not None
    assert 0 not in preset["mpc_allowed_orders"]  # LoS order excluded
    assert preset["mpc_allowed_types"] == [1]  # Only specular type


# =============================================================================
# Load Preset Tests
# =============================================================================


def test_load_built_in_preset(preset_service: PresetService) -> None:
    """Verify loading a built-in preset returns correct data."""
    preset = preset_service.load_preset("LoS Only")
    assert preset is not None
    assert preset["mpc_allowed_types"] == [0]


def test_load_nonexistent_preset_returns_none(preset_service: PresetService) -> None:
    """Verify loading a non-existent preset returns None."""
    preset = preset_service.load_preset("NonExistent Preset")
    assert preset is None


def test_load_user_preset(preset_service: PresetService, user_preset_path: Path) -> None:
    """Verify loading a user preset works."""
    # Create a user preset file
    user_preset = {
        "name": "My Custom Preset",
        "description": "Custom test preset",
        "mpc_allowed_orders": [1, 2],
        "mpc_allowed_types": [1],
        "materials": {},
    }
    preset_file = user_preset_path / "My Custom Preset.json"
    preset_file.write_text(json.dumps(user_preset))

    # Load it
    loaded = preset_service.load_preset("My Custom Preset")
    assert loaded is not None
    assert loaded["description"] == "Custom test preset"
    assert loaded["mpc_allowed_orders"] == [1, 2]


def test_load_user_preset_with_invalid_json(
    preset_service: PresetService, user_preset_path: Path
) -> None:
    """Verify loading a corrupted preset file returns None."""
    preset_file = user_preset_path / "Corrupted.json"
    preset_file.write_text("not valid json {{{")

    result = preset_service.load_preset("Corrupted")
    assert result is None


# =============================================================================
# Save Preset Tests
# =============================================================================


def test_save_preset_creates_file(preset_service: PresetService, user_preset_path: Path) -> None:
    """Verify saving a preset creates a JSON file."""
    filter_state = {
        "mpc_allowed_orders": [0, 1, 2],
        "mpc_allowed_types": [0, 1],
        "materials": {},
    }

    result = preset_service.save_preset("My Test Preset", filter_state)
    assert result is True

    preset_file = user_preset_path / "My Test Preset.json"
    assert preset_file.exists()

    # Verify content
    content = json.loads(preset_file.read_text())
    assert content["name"] == "My Test Preset"
    assert content["mpc_allowed_orders"] == [0, 1, 2]


def test_save_preset_sanitizes_filename(
    preset_service: PresetService, user_preset_path: Path
) -> None:
    """Verify save_preset sanitizes special characters in filename."""
    filter_state = {"mpc_allowed_orders": [0], "mpc_allowed_types": [0], "materials": {}}

    result = preset_service.save_preset("Test/Preset:Name", filter_state)
    assert result is True

    # Should create file with sanitized name
    preset_file = user_preset_path / "TestPresetName.json"
    assert preset_file.exists()


def test_save_preset_with_empty_name_fails(preset_service: PresetService) -> None:
    """Verify save_preset fails for names that sanitize to empty string."""
    filter_state = {"mpc_allowed_orders": [0], "mpc_allowed_types": [0], "materials": {}}

    # Name with only special characters
    result = preset_service.save_preset("///", filter_state)
    assert result is False


def test_save_preset_includes_description(
    preset_service: PresetService, user_preset_path: Path
) -> None:
    """Verify save_preset includes custom description."""
    filter_state = {
        "description": "My custom description",
        "mpc_allowed_orders": [0],
        "mpc_allowed_types": [0],
        "materials": {},
    }

    preset_service.save_preset("Described Preset", filter_state)

    preset_file = user_preset_path / "Described Preset.json"
    content = json.loads(preset_file.read_text())
    assert content["description"] == "My custom description"


# =============================================================================
# Delete Preset Tests
# =============================================================================


def test_delete_user_preset(preset_service: PresetService, user_preset_path: Path) -> None:
    """Verify deleting a user preset removes the file."""
    # Create a preset first
    preset_file = user_preset_path / "ToDelete.json"
    preset_file.write_text('{"name": "ToDelete"}')

    result = preset_service.delete_preset("ToDelete")
    assert result is True
    assert not preset_file.exists()


def test_delete_built_in_preset_fails(preset_service: PresetService) -> None:
    """Verify built-in presets cannot be deleted."""
    result = preset_service.delete_preset("LoS Only")
    assert result is False
    # Preset should still exist
    assert "LoS Only" in preset_service.built_in_presets


def test_delete_nonexistent_preset_returns_false(preset_service: PresetService) -> None:
    """Verify deleting a non-existent preset returns False."""
    result = preset_service.delete_preset("Does Not Exist")
    assert result is False


# =============================================================================
# List Presets Tests
# =============================================================================


def test_list_presets_includes_built_in(preset_service: PresetService) -> None:
    """Verify list_presets includes built-in presets."""
    presets = preset_service.list_presets()

    assert "All MPCs (Reset)" in presets
    assert "LoS Only" in presets
    assert "Specular Only" in presets


def test_list_presets_includes_user_presets(
    preset_service: PresetService, user_preset_path: Path
) -> None:
    """Verify list_presets includes user presets."""
    # Create user preset
    preset_file = user_preset_path / "UserPreset.json"
    preset_file.write_text('{"name": "UserPreset"}')

    presets = preset_service.list_presets()
    assert "UserPreset" in presets


# =============================================================================
# Utility Method Tests
# =============================================================================


def test_is_built_in_returns_true_for_built_in(preset_service: PresetService) -> None:
    """Verify is_built_in returns True for built-in presets."""
    assert preset_service.is_built_in("LoS Only") is True
    assert preset_service.is_built_in("All MPCs (Reset)") is True


def test_is_built_in_returns_false_for_user_preset(preset_service: PresetService) -> None:
    """Verify is_built_in returns False for user presets."""
    assert preset_service.is_built_in("My Custom Preset") is False


def test_get_preset_description_for_built_in(preset_service: PresetService) -> None:
    """Verify get_preset_description returns description for built-in presets."""
    desc = preset_service.get_preset_description("LoS Only")
    assert desc is not None
    assert "line-of-sight" in desc.lower()


def test_get_preset_description_for_nonexistent(preset_service: PresetService) -> None:
    """Verify get_preset_description returns None for non-existent presets."""
    desc = preset_service.get_preset_description("Does Not Exist")
    assert desc is None
