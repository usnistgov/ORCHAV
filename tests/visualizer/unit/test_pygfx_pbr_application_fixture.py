"""The native reference must remain a real scenario and explicit visual-only data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from shared.frames.manifest import load_frame_manifest
from shared.scenarios import load_scenario_configuration
from visualizer.src.scene.io import XMLSceneHandler, build_scene_from_root


def test_native_pbr_fixture_opens_as_local_scene_and_canonical_frames(tmp_path):
    root = Path(__file__).resolve().parents[3]
    path = root / "examples/visualizer/pygfx_pbr_reference/generate.py"
    spec = importlib.util.spec_from_file_location("native_pbr_fixture_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = module.generate(tmp_path / "reference")
    configuration = load_scenario_configuration(directory, project_root=root)
    assert configuration.scene_source == "local"
    assert configuration.raytracing["enabled"] is False
    manifest = load_frame_manifest(directory / "frames")
    assert manifest is not None
    assert manifest.frame_ids == (0,)
    entries = build_scene_from_root(
        XMLSceneHandler.load_xml_scene(str(configuration.scene_xml)),
        str(configuration.scene_xml),
    )
    assert len(entries) == 19
    coat = [entry for entry in entries if "clearcoat" in entry["material_id"]]
    assert len(coat) == 4
    assert all(entry["pbr_properties"]["clearcoat"] == 0.4 for entry in coat)
