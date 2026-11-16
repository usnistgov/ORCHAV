"""Derived frames retain a truthful local scene binding without changing sources."""

from pathlib import Path

import pytest
import yaml

from shared.sensing.result_scenario import write_result_scenario


def test_missing_scene_is_not_invented(tmp_path):
    assert write_result_scenario(tmp_path, tmp_path) is None


@pytest.mark.parametrize("explicit_source", [False, True])
def test_library_geometry_binding_accepts_compact_yaml(tmp_path, explicit_source):
    scene_file = tmp_path / "libraries/scenes/empty/empty.xml"
    scene_file.parent.mkdir(parents=True)
    scene_file.write_text("<scene/>")
    source = tmp_path / "scenarios/cube"
    source.mkdir(parents=True)
    scene = {"id": "empty/empty.xml"}
    if explicit_source:
        scene["source"] = "library"
    original = yaml.safe_dump({"schema_version": 2, "scene": scene})
    (source / "scenario.yaml").write_text(original)
    output = tmp_path / "result"
    output.mkdir()

    saved = yaml.safe_load(write_result_scenario(source / "frames", output).read_text())

    assert saved["scene"] == {"source": "local", "id": str(scene_file)}
    assert Path(saved["scene"]["id"]).is_file()
    assert saved["data"]["files"]["directory"] == "frames"
    assert not saved["raytracing"]["enabled"]
    assert (source / "scenario.yaml").read_text() == original


def test_relative_geometry_and_existing_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "scene": {"source": "local", "id": "scene.xml"},
        "actors": {"targets": [{"asset": {"path": "target.ply"}}]},
        "raytracing": {"enabled": True},
    }
    original = yaml.safe_dump(config)
    (source / "scenario.yaml").write_text(original)
    output = tmp_path / "result"
    output.mkdir()
    path = write_result_scenario(source / "frames", output)
    saved = yaml.safe_load(path.read_text())
    assert Path(saved["scene"]["id"]) == source / "scene.xml"
    assert Path(saved["actors"]["targets"][0]["asset"]["path"]) == source / "target.ply"
    assert saved["data"]["files"]["directory"] == "frames"
    assert not saved["raytracing"]["enabled"]
    assert (source / "scenario.yaml").read_text() == original
    with pytest.raises(FileExistsError):
        write_result_scenario(source, output)
