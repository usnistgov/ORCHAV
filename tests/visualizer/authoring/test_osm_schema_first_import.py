"""Scenario Builder dependency checks for OSM-derived scene artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.visualizer.authoring.test_schema_first_import import (
    _mapping,
    _materialize_dependencies,
    _MergedCompiler,
)
from visualizer.src.authoring.compiler import ScenarioCompiler
from visualizer.src.authoring.persistence import (
    create_scenario_copy,
    load_for_authoring,
    save_document,
)


@pytest.mark.parametrize(
    ("contents", "diagnostic"),
    [("{not-json", "OSM metadata is unreadable"), ("[]", "OSM metadata must contain an object")],
)
def test_unreadable_osm_metadata_opens_but_blocks_import_save(
    tmp_path: Path, contents: str, diagnostic: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mapping = _mapping()
    mapping["scene"] = {
        "source": "osm",
        "osm": {"center": [48.853, 2.349], "size_m": 500.0},
    }
    scenario_yaml = source / "scenario.yaml"
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    _materialize_dependencies(source)
    (source / "osm_metadata.json").write_text(contents, encoding="utf-8")

    document = create_scenario_copy(scenario_yaml, tmp_path / "copy")
    metadata = next(
        dependency
        for dependency in document.scenario.dependencies
        if dependency.kind == "osm_metadata"
    )

    assert diagnostic in metadata.problem
    compiled = ScenarioCompiler().compile(document.scenario, scenario_directory=tmp_path / "copy")
    assert any(
        issue.message == metadata.problem and issue.path == "scene.osm" for issue in compiled.issues
    )
    with pytest.raises(ValueError, match=diagnostic):
        save_document(document, compiler=_MergedCompiler())
    assert not (tmp_path / "copy" / "scenario.yaml").exists()

    # Repairing the original metadata must unblock this same unsaved document.
    (source / "osm_metadata.json").write_text("{}", encoding="utf-8")
    saved = save_document(document, compiler=_MergedCompiler())
    assert saved == tmp_path / "copy" / "scenario.yaml"
    assert (saved.parent / "osm_metadata.json").read_text(encoding="utf-8") == "{}"


def test_osm_artifact_parent_traversal_is_a_copy_blocker(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mapping = _mapping()
    mapping["scene"] = {
        "source": "osm",
        "osm": {"center": [48.853, 2.349], "size_m": 500.0},
    }
    scenario_yaml = source / "scenario.yaml"
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    _materialize_dependencies(source)
    (source / "osm_metadata.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "scene_xml": "../outside.xml",
                    "street_network_graphml": "street_network.graphml",
                    "mesh_files": [],
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "outside.xml").write_text("<scene />", encoding="utf-8")

    loaded = load_for_authoring(scenario_yaml)

    assert loaded.scenario is not None
    problem = next(
        dependency.problem
        for dependency in loaded.scenario.dependencies
        if dependency.origin_path.endswith("../outside.xml")
    )
    assert "parent traversal" in problem
    document = create_scenario_copy(scenario_yaml, tmp_path / "copy")
    with pytest.raises(ValueError, match="parent traversal"):
        save_document(document, compiler=_MergedCompiler())
