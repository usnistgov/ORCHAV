import json

import pytest

from visualizer.src.io.config_handlers import ConfigFileHandler


def test_config_file_handler_saves_parseable_json(tmp_path):
    """Saved visualizer configs are JSON, not Python dict repr strings."""
    config_path = tmp_path / "nested" / "visualizer_config.json"
    config_data = {
        "scenario": "scenarios/_internal/visualizer/config_serialization_roundtrip",
        "recent_files": ["scenarios/generator/targets/mesh_targets"],
        "visualizer": {
            "renderer": "pygfx",
            "mpc_visibility": {
                "enabled": False,
                "paths": True,
                "bounce_points": True,
            },
            "resolution_scale": 2,
        },
    }

    ConfigFileHandler.save_config(str(config_path), config_data)

    content = config_path.read_text(encoding="utf-8")
    assert json.loads(content) == config_data
    assert "'scenario'" not in content
    assert "False" not in content
    assert ConfigFileHandler.load_config(str(config_path)) == config_data


def test_config_file_handler_rejects_non_json_content(tmp_path):
    """Configuration syntax errors are reported instead of silently reinterpreted."""
    config_path = tmp_path / "invalid_config.txt"
    config_path.write_text("{'scenario': 'not-json'}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        ConfigFileHandler.load_config(str(config_path))
