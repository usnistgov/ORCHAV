"""Focused contracts shared by scenario validation and runtime parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import shared.cli.validate as validate_cli
from shared.coverage.schema import resolve_coverage_tx_index
from shared.scenarios import load_scenario_configuration, load_scenario_yaml
from shared.scenarios.defaults import DEFAULT_COVERAGE_TX_MODE
from shared.scenarios.model import CoverageTxModel
from shared.scenarios.parsers import parse_coverage_config


def _write_scenario(path: Path, **overrides: object) -> None:
    document: dict[str, object] = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0.0},
    }
    document.update(overrides)
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_coverage_tx_mode_has_one_schema_and_runtime_default(tmp_path: Path) -> None:
    assert CoverageTxModel().mode == DEFAULT_COVERAGE_TX_MODE
    assert parse_coverage_config({}, tmp_path)["tx_mode"] == DEFAULT_COVERAGE_TX_MODE


def test_coverage_figure_runtime_config_has_no_height_selector(tmp_path: Path) -> None:
    config = parse_coverage_config(
        {
            "coverage": {
                "save": {
                    "figure": {
                        "enabled": True,
                        "distribution": {"enabled": True},
                    }
                }
            }
        },
        tmp_path,
    )

    figure = config["save"]["figure"]
    assert "height_index" not in figure
    assert "height_index" not in figure["distribution"]


def test_scene_null_uses_the_omitted_scene_defaults(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    _write_scenario(scenario_root / "scenario.yaml", scene=None)

    config = load_scenario_configuration(scenario_root, project_root=tmp_path)

    assert config.scene_id == "default"
    assert config.scene_source == "library"
    assert config.scene_xml == (tmp_path / "libraries" / "scenes" / "default").resolve()


def test_scene_parser_consumes_the_schema_normalized_source(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    _write_scenario(
        scenario_root / "scenario.yaml",
        scene={"id": "warehouse", "source": "  library  "},
    )

    config = load_scenario_configuration(scenario_root, project_root=tmp_path)

    assert config.scene_source == "library"
    assert config.scene_xml == (tmp_path / "libraries" / "scenes" / "warehouse").resolve()


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_schema_rejects_nonpositive_chunk_size(tmp_path: Path, chunk_size: int) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    _write_scenario(
        scenario_root / "scenario.yaml",
        data={"files": {"chunk_size": chunk_size}},
    )

    with pytest.raises(ValueError, match=r"data\.files\.chunk_size"):
        load_scenario_configuration(scenario_root, project_root=tmp_path)


def test_scenario_yaml_is_read_as_utf8(tmp_path: Path) -> None:
    scenario_yaml = tmp_path / "scenario.yaml"
    _write_scenario(
        scenario_yaml,
        scene={"source": "local", "id": "scène-東京.xml"},
    )

    data = load_scenario_yaml(scenario_yaml)

    assert data["scene"]["id"] == "scène-東京.xml"


def test_project_root_marker_resolves_local_scene_and_read_frames(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    scenario_root = project_root / "scenarios" / "reader"
    scene_xml = project_root / "assets" / "scene.xml"
    frames_dir = project_root / "recordings" / "frames"
    scenario_root.mkdir(parents=True)
    scene_xml.parent.mkdir()
    scene_xml.write_text("<scene />", encoding="utf-8")
    frames_dir.mkdir(parents=True)
    _write_scenario(
        scenario_root / "scenario.yaml",
        scene={"source": "local", "id": "${PROJECT_ROOT}/assets/scene.xml"},
        data={
            "mode": "files",
            "files": {"directory": "${PROJECT_ROOT}/recordings/frames"},
        },
    )

    config = load_scenario_configuration(scenario_root, project_root=project_root)

    assert config.scene_xml == scene_xml.resolve()
    assert config.frames_dir == frames_dir.resolve()


def test_validator_path_checks_expand_project_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    scenario_root = project_root / "scenarios" / "reader"
    scene_xml = project_root / "assets" / "scene.xml"
    frames_dir = project_root / "recordings" / "frames"
    scenario_root.mkdir(parents=True)
    scene_xml.parent.mkdir()
    scene_xml.write_text("<scene />", encoding="utf-8")
    frames_dir.mkdir(parents=True)
    monkeypatch.setattr(validate_cli, "find_project_root", lambda _path: project_root)

    warnings = validate_cli._check_referenced_paths(
        {
            "scene": {"source": "local", "id": "${PROJECT_ROOT}/assets/scene.xml"},
            "data": {
                "mode": "files",
                "files": {"directory": "${PROJECT_ROOT}/recordings/frames"},
            },
        },
        scenario_root,
        check_frame_paths=True,
    )

    assert warnings == []


def test_numeric_coverage_selectors_are_zero_based() -> None:
    names = ["West", "Center", "East"]

    assert resolve_coverage_tx_index("0", names, 3) == 0
    assert resolve_coverage_tx_index("2", names, 3) == 2
    assert resolve_coverage_tx_index("tx3", names, 3) == 2
    assert resolve_coverage_tx_index("East", names, 3) == 2
    with pytest.raises(ValueError, match="unknown TX selector"):
        resolve_coverage_tx_index("3", names, 3)
