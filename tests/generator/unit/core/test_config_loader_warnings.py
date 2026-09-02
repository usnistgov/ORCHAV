import logging

import pytest

from shared.scenarios import validate_scenario_data


def test_pydantic_validation_unknown_keys():
    """Verify that unknown keys in scenario dict raise ValueError."""

    invalid_data = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0.0},
        "scene": {"id": "test", "source": "local"},
        "typo_key_at_root": "should fail",
        "raytracing": {"enabled": True, "typo_key_nested": 123},
    }

    with pytest.raises(ValueError, match="typo_key_at_root"):
        validate_scenario_data(invalid_data)


def test_pydantic_validation_valid(caplog):
    """Verify valid config passes without errors."""
    valid_data = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0.0},
        "scene": {"id": "test", "source": "local"},
        "raytracing": {"enabled": True},
        "view_defaults": {
            "color_mode": "reflection_order",
            "selected_tx": "all",
            "mpc_visibility": {
                "enabled": True,
                "paths": True,
                "bounce_points": False,
            },
        },
    }

    with caplog.at_level(logging.ERROR):
        validate_scenario_data(valid_data)

    assert "Validation Failed" not in caplog.text
