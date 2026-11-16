"""Tests for OSM scene reuse/regeneration decisions in the scenario loader."""

from __future__ import annotations

import json
from pathlib import Path

import generator.osm as osm_module
from generator.osm.metadata import OSM_METADATA_SCHEMA_VERSION, normalize_osm_config
from shared.scenarios.parsers import parse_scene_config


def _write_generated_osm_scene(scenario_root: Path, osm_cfg: dict) -> None:
    (scenario_root / "mesh").mkdir(parents=True, exist_ok=True)
    (scenario_root / "scene.xml").write_text("<scene />", encoding="utf-8")
    (scenario_root / "street_network.graphml").write_text("<graphml />", encoding="utf-8")
    (scenario_root / "mesh" / "ground.ply").write_text("ply\n", encoding="utf-8")
    metadata = {
        "schema_version": OSM_METADATA_SCHEMA_VERSION,
        "osm_config": normalize_osm_config(osm_cfg),
        "artifacts": {
            "scene_xml": "scene.xml",
            "street_network_graphml": "street_network.graphml",
            "mesh_files": ["mesh/ground.ply"],
        },
        "terrain_grid": {"enabled": False},
        "acquisition": {"complete": True, "layers": {}},
    }
    (scenario_root / "osm_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def test_parse_scene_config_reuses_generated_osm_scene(tmp_path: Path):
    scenario_root = tmp_path / "osm_scene"
    scenario_root.mkdir()
    osm_cfg = {"center": [48.853, 2.349], "size_m": 500.0}
    _write_generated_osm_scene(scenario_root, osm_cfg)

    scene_id, scene_source, scene_xml = parse_scene_config(
        {"scene": {"source": "  osm  ", "osm": osm_cfg}},
        scenario_root,
        tmp_path,
    )

    assert scene_id == "scene.xml"
    assert scene_source == "osm"
    assert scene_xml == scenario_root / "scene.xml"


def test_parse_scene_config_rebuilds_when_osm_config_changes(tmp_path: Path, monkeypatch):
    scenario_root = tmp_path / "osm_scene"
    scenario_root.mkdir()
    existing_cfg = {"center": [48.853, 2.349], "size_m": 500.0}
    new_cfg = {"center": [48.854, 2.349], "size_m": 500.0}
    _write_generated_osm_scene(scenario_root, existing_cfg)

    calls = []

    def fake_build_scene_from_center(**kwargs):
        calls.append(kwargs)
        return str(scenario_root / "scene.xml")

    monkeypatch.setattr(osm_module, "build_scene_from_center", fake_build_scene_from_center)

    _, _, scene_xml = parse_scene_config(
        {"scene": {"source": "osm", "osm": new_cfg}},
        scenario_root,
        tmp_path,
    )

    assert scene_xml == scenario_root / "scene.xml"
    assert len(calls) == 1
    assert calls[0]["metadata_config"] == normalize_osm_config(new_cfg)


def test_parse_scene_config_rebuilds_when_feature_flag_toggled(tmp_path: Path, monkeypatch):
    scenario_root = tmp_path / "osm_scene"
    scenario_root.mkdir()
    existing_cfg = {"center": [48.853, 2.349], "size_m": 500.0, "vegetation": True}
    new_cfg = {"center": [48.853, 2.349], "size_m": 500.0, "vegetation": False}
    _write_generated_osm_scene(scenario_root, existing_cfg)

    calls = []

    def fake_build_scene_from_center(**kwargs):
        calls.append(kwargs)
        return str(scenario_root / "scene.xml")

    monkeypatch.setattr(osm_module, "build_scene_from_center", fake_build_scene_from_center)

    parse_scene_config(
        {"scene": {"source": "osm", "osm": new_cfg}},
        scenario_root,
        tmp_path,
    )

    assert len(calls) == 1, "toggling vegetation flag must trigger a rebuild"
    assert calls[0]["vegetation"] is False


def test_parse_scene_config_rebuilds_when_metadata_missing(tmp_path: Path, monkeypatch):
    scenario_root = tmp_path / "osm_scene"
    scenario_root.mkdir()
    (scenario_root / "scene.xml").write_text("<scene />", encoding="utf-8")

    calls = []

    def fake_build_scene(*, bbox, output_dir, metadata_config, **kwargs):
        calls.append(
            {
                "bbox": bbox,
                "output_dir": output_dir,
                "metadata_config": metadata_config,
                "kwargs": kwargs,
            }
        )
        return str(scenario_root / "scene.xml")

    monkeypatch.setattr(osm_module, "build_scene", fake_build_scene)

    osm_cfg = {"bbox": [48.852, 2.348, 48.854, 2.350], "terrain": True}
    _, _, scene_xml = parse_scene_config(
        {"scene": {"source": "osm", "osm": osm_cfg}},
        scenario_root,
        tmp_path,
    )

    assert scene_xml == scenario_root / "scene.xml"
    assert len(calls) == 1
    assert calls[0]["metadata_config"] == normalize_osm_config(osm_cfg)


def test_parse_scene_explicit_refresh_bypasses_asset_reuse(tmp_path, monkeypatch):
    cfg = {"center": [48.853, 2.349], "size_m": 500}
    _write_generated_osm_scene(tmp_path, cfg)
    calls = []
    monkeypatch.setattr(
        osm_module,
        "build_scene_from_center",
        lambda **kw: calls.append(kw) or str(tmp_path / "scene.xml"),
    )
    parse_scene_config(
        {"scene": {"source": "osm", "osm": {**cfg, "refresh_cache": True}}}, tmp_path, tmp_path
    )
    assert len(calls) == 1
    assert calls[0]["osm_refresh_cache"] is True


def test_parse_scene_environment_refresh_bypasses_asset_reuse(tmp_path, monkeypatch):
    cfg = {"center": [48.853, 2.349]}
    _write_generated_osm_scene(tmp_path, cfg)
    monkeypatch.setenv("OSM_REFRESH_CACHE", "1")
    calls = []
    monkeypatch.setattr(
        osm_module,
        "build_scene_from_center",
        lambda **kw: calls.append(kw) or str(tmp_path / "scene.xml"),
    )
    parse_scene_config({"scene": {"source": "osm", "osm": cfg}}, tmp_path, tmp_path)
    assert len(calls) == 1


def test_incomplete_scene_rebuilds_on_next_load(tmp_path, monkeypatch):
    cfg = {"center": [48.853, 2.349]}
    _write_generated_osm_scene(tmp_path, cfg)
    path = tmp_path / "osm_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"]["complete"] = False
    path.write_text(json.dumps(metadata))
    calls = []
    monkeypatch.setattr(
        osm_module,
        "build_scene_from_center",
        lambda **kw: calls.append(kw) or str(tmp_path / "scene.xml"),
    )
    parse_scene_config({"scene": {"source": "osm", "osm": cfg}}, tmp_path, tmp_path)
    assert len(calls) == 1


def test_old_vegetation_default_requires_rebuild_but_explicit_false_reuses(tmp_path, monkeypatch):
    cfg = {"center": [38.8887, -77.0261]}
    _write_generated_osm_scene(tmp_path, {**cfg, "green_vegetation": False})
    calls = []
    monkeypatch.setattr(
        osm_module,
        "build_scene_from_center",
        lambda **kw: calls.append(kw) or str(tmp_path / "scene.xml"),
    )
    parse_scene_config(
        {"scene": {"source": "osm", "osm": {**cfg, "green_vegetation": False}}},
        tmp_path,
        tmp_path,
    )
    assert not calls
    parse_scene_config({"scene": {"source": "osm", "osm": cfg}}, tmp_path, tmp_path)
    assert len(calls) == 1 and calls[0]["green_vegetation"] is True
    assert calls[0]["metadata_config"]["green_vegetation"] is True
