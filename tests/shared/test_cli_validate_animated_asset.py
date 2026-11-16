"""Strict public CLI validation authenticates native animated manifest resources."""

import pytest
import yaml

from shared.animated_target_workflow import _write_synthetic_asset
from shared.cli.validate import _check_referenced_paths, validate_scenario


def _scenario(tmp_path):
    manifest, _ = _write_synthetic_asset(tmp_path / "asset")
    (tmp_path / "empty.xml").write_text('<scene version="3.0.0"/>', encoding="utf-8")
    (tmp_path / "frames").mkdir()
    data = {
        "schema_version": 2,
        "timeline": {"steps": 2, "duration_s": 1.0},
        "scene": {"source": "local", "id": "empty.xml"},
        "raytracing": {"enabled": False},
        "actors": {
            "targets": [
                {
                    "name": "Walker",
                    "mobility": {"type": "stationary", "position_m": [0.0, 0.0, 0.0]},
                    "asset": {
                        "source": "animated_asset",
                        "path": "asset/animated_target.json",
                        "playback": {
                            "scenario_clock_id": "scenario:test",
                            "motion_clock_id": "motion:test",
                            "playback_rate": 1.0,
                            "playback_offset_s": 0.0,
                        },
                    },
                }
            ]
        },
    }
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return manifest, scenario_path, data


def test_strict_validator_accepts_existing_authenticated_animated_manifest(tmp_path):
    _, path, data = _scenario(tmp_path)
    assert _check_referenced_paths(data, tmp_path, check_frame_paths=True) == []
    assert validate_scenario(path, strict=True) == 0


@pytest.mark.parametrize("damage", ("missing_manifest", "malformed_manifest", "changed_resource"))
def test_strict_validator_rejects_unreadable_or_unauthenticated_motion(tmp_path, damage):
    manifest, path, data = _scenario(tmp_path)
    if damage == "missing_manifest":
        manifest.unlink()
    elif damage == "malformed_manifest":
        manifest.write_text("{}", encoding="utf-8")
    else:
        (tmp_path / "asset" / "poses" / "z_neutral_stance.json").write_text("{}", encoding="utf-8")
    warnings = _check_referenced_paths(data, tmp_path, check_frame_paths=True)
    assert len(warnings) == 1 and "actors.targets[0].asset.path" in warnings[0]
    assert validate_scenario(path, strict=True) == 1


def test_ordinary_mesh_path_validation_does_not_import_animated_runtime(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "shared.animated_target":
            raise AssertionError("ordinary assets must not load the animated contract")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    (tmp_path / "ordinary.ply").write_text("ply", encoding="utf-8")
    data = {"actors": {"targets": [{"asset": {"source": "file", "path": "ordinary.ply"}}]}}
    assert _check_referenced_paths(data, tmp_path) == []
