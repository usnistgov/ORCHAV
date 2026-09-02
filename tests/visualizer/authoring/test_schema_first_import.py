"""Schema-first loading and semantic scenario-copy behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visualizer.src.authoring.compiler import (
    CompilationResult,
    canonical_yaml,
    merged_scenario_mapping,
)
from visualizer.src.authoring.persistence import (
    LoadDisposition,
    create_scenario_copy,
    load_for_authoring,
    save_document,
)


def _mapping() -> dict:
    return {
        "schema_version": 2,
        "scene": {"source": "library", "id": "empty/empty.xml"},
        "timeline": {"steps": 12, "duration_s": 4.0},
        "raytracing": {
            "enabled": True,
            "export_path_metrics": False,
            "quality": {"preset": "ultra-low"},
        },
        "actors": {
            "tx": [
                {
                    "name": "TX",
                    "power_dbm": -3.0,
                    "mobility": {
                        "type": "network_route",
                        "graph_path": "street_network.graphml",
                        "route": "random_walk",
                        "seed": 7,
                    },
                }
            ],
            "rx": [
                {
                    "name": "RX",
                    "mobility": {
                        "type": "stationary",
                        "position_m": [1.0, 0.0, 1.0],
                    },
                }
            ],
            "targets": [
                {
                    "name": "Target",
                    "asset": {
                        "source": "directory",
                        "path": "target",
                        "pattern": "frame_*.ply",
                        "material_type": "brick",
                        "scale": 2.0,
                        "start_index": 1,
                        "frame_stride": 2,
                    },
                    "mobility": {
                        "type": "mesh_sequence",
                        "positions_path": "positions.yaml",
                    },
                }
            ],
        },
        "data": {
            "mode": "files",
            "files": {
                "directory": "results",
                "pattern": "chunk_*.h5",
                "format": "h5",
                "compression": "lzf",
                "chunk_size": 12,
            },
        },
        "generator_summary": {
            "enabled": True,
            "create": ["scene2d", "speed"],
            "output": {"format": "svg"},
        },
        "visualizer": {"theme": "dark"},
    }


class _MergedCompiler:
    def compile(self, scenario, *, scenario_directory=None):
        mapping = merged_scenario_mapping(scenario)
        return CompilationResult(mapping, canonical_yaml(mapping), (), {})


def _materialize_dependencies(root: Path) -> None:
    (root / "street_network.graphml").write_text("<graphml />", encoding="utf-8")
    (root / "positions.yaml").write_text("positions: [[0, 0, 0]]", encoding="utf-8")
    target = root / "target"
    target.mkdir(exist_ok=True)
    (target / "frame_000.ply").write_text("ply", encoding="utf-8")


def test_full_source_snapshot_round_trips_outside_builder_marker(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    scenario_yaml = source / "scenario.yaml"
    mapping = _mapping()
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    _materialize_dependencies(source)
    (source / "target" / "frame_001.ply").write_text("ply", encoding="utf-8")
    (source / "target" / "metadata.json").write_text("{}", encoding="utf-8")

    loaded = load_for_authoring(scenario_yaml)

    assert loaded.disposition is LoadDisposition.OWNED_EDITABLE
    assert loaded.document is not None
    assert loaded.scenario is not None
    merged = merged_scenario_mapping(loaded.scenario)
    marker = merged["visualizer"].pop("scenario_builder")
    assert marker == {"document_version": 2}
    assert merged == mapping
    assert loaded.scenario.actors[0].id in {
        binding.id for binding in loaded.scenario.source_snapshot.actor_bindings
    }


def test_scenario_copy_promotes_only_yaml_and_semantic_dependencies(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mapping = _mapping()
    scenario_yaml = source / "scenario.yaml"
    source_text = yaml.safe_dump(mapping, sort_keys=False)
    scenario_yaml.write_text(source_text, encoding="utf-8")
    _materialize_dependencies(source)
    (source / "target" / "metadata.json").write_text("{}", encoding="utf-8")
    (source / "unrelated.txt").write_text("do not copy", encoding="utf-8")
    (source / "frames").mkdir()
    (source / "frames" / "old.h5").write_text("do not copy", encoding="utf-8")
    for derived_directory in ("summary", "coverage", "cache", "diagnostics"):
        derived = source / derived_directory
        derived.mkdir()
        (derived / "sentinel.txt").write_text("do not copy", encoding="utf-8")

    destination = tmp_path / "copy"
    document = create_scenario_copy(scenario_yaml, destination)
    saved = save_document(document, compiler=_MergedCompiler())

    assert scenario_yaml.read_text(encoding="utf-8") == source_text
    assert saved == destination / "scenario.yaml"
    assert (destination / "street_network.graphml").is_file()
    assert (destination / "positions.yaml").is_file()
    assert (destination / "target" / "frame_000.ply").is_file()
    assert (destination / "target" / "metadata.json").is_file()
    assert not (destination / "unrelated.txt").exists()
    assert not (destination / "frames").exists()
    assert not (destination / "summary").exists()
    assert not (destination / "coverage").exists()
    assert not (destination / "cache").exists()
    assert not (destination / "diagnostics").exists()


def test_transitive_scene_parent_traversal_opens_but_blocks_import_save(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mapping = _mapping()
    mapping["scene"] = {"source": "local", "id": "scene.xml"}
    scenario_yaml = source / "scenario.yaml"
    scenario_yaml.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    _materialize_dependencies(source)
    (source / "scene.xml").write_text(
        '<scene><include filename="../outside.xml" /></scene>',
        encoding="utf-8",
    )
    (tmp_path / "outside.xml").write_text("<scene />", encoding="utf-8")

    document = create_scenario_copy(scenario_yaml, tmp_path / "copy")
    problem = next(
        dependency.problem
        for dependency in document.scenario.dependencies
        if dependency.origin_path.endswith(":../outside.xml")
    )

    assert "parent traversal" in problem
    with pytest.raises(ValueError, match="parent traversal"):
        save_document(document, compiler=_MergedCompiler())
    assert not (tmp_path / "copy" / "scenario.yaml").exists()
