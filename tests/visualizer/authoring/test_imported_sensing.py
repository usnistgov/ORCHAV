"""Imported sensing scenarios keep their source contract in the Builder."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from generator.core.scenario_actors.animated_asset import prepare_animated_asset
from generator.core.scenario_actors.types import Timeline
from shared.animated_target import animated_target_revision_sha256
from shared.animated_target_workflow import _write_synthetic_asset
from visualizer.src.authoring import (
    ActorRole,
    ScenarioCompiler,
    ScenarioDocument,
    ScenarioSourceSnapshot,
    canonical_scenario_mapping,
    load_for_authoring,
    save_document,
)
from visualizer.src.authoring.persistence import scenario_from_mapping
from visualizer.src.authoring.sensing_inspector import SensingInspector

PROJECT = Path(__file__).resolve().parents[3]


def imported_sensing_scenario(root: Path) -> Path:
    """Author a public synthetic import with route-owned and asset-owned motion."""
    source = PROJECT / "scenarios/sensing/cube_radar/moving/scenario.yaml"
    mapping = yaml.safe_load(source.read_text())
    mapping["actors"]["targets"] = [
        {
            "name": "Car",
            "asset": {"source": "directory", "path": "external_car", "pattern": "car.ply"},
            "mobility": {
                "type": "network_route",
                "travel_mode": "car",
                "start_node": "a",
                "end_node": "b",
                "graph_path": "street_network.graphml",
                "traversal": {"type": "constant_speed", "speed_mps": 3.0},
            },
        },
        {
            "name": "Pedestrian",
            "asset": {
                "source": "animated_asset",
                "path": "assets/walker/animated_target.json",
                "playback": {
                    "scenario_clock_id": "synthetic_scene",
                    "motion_clock_id": "synthetic_motion",
                    "playback_rate": 1.0,
                    "playback_offset_s": 0.0,
                },
            },
            "mobility": {"type": "stationary", "position_m": [0.0, 0.0, 0.0]},
        },
    ]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "scenario.yaml"
    path.write_text(yaml.safe_dump(mapping, sort_keys=False))
    return path


@pytest.mark.parametrize("prepared", [False, True])
def test_synthetic_import_keeps_sensing_and_locks_asset_owned_motion(tmp_path, prepared):
    imported_sensing_scenario(tmp_path)
    # Real walker assets are generated locally and are absent in a fresh clone.
    # Exercise both states without depending on a private motion dataset.
    if prepared:
        manifest_path, _ = _write_synthetic_asset(tmp_path / "assets/walker")
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["source_ids"].append("adapter_provenance:synthetic-test")
        manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest))
        manifest_path.with_name("adapter_provenance.json").write_text("{}")
    result = load_for_authoring(tmp_path)
    assert result.editable, result.issues
    document = result.document
    assert not document.dirty
    pedestrian = document.scenario.actor_by_name("Pedestrian")
    assert pedestrian.target.source == "animated_asset"
    assert canonical_scenario_mapping(document.scenario)["sensing"] == result.raw_mapping["sensing"]
    assert (
        canonical_scenario_mapping(document.scenario)["actors"]["targets"]
        == result.raw_mapping["actors"]["targets"]
    )
    with pytest.raises(PermissionError, match="owned"):
        document.update_actor(
            pedestrian.id, mobility=pedestrian.mobility.model_copy(update={"position_m": (1, 0, 0)})
        )
    dependencies = {d.relative_path for d in document.scenario.dependencies}
    assert "assets/walker/animated_target.json" in dependencies
    assert ("assets/walker/adapter_provenance.json" in dependencies) is prepared
    if prepared:
        assert {f"assets/walker/{r['path']}" for r in manifest["resources"]} <= dependencies


def test_animated_preview_and_portable_save_without_rt_mesh_generation(tmp_path, monkeypatch):
    from generator.core.scenario_actors import animated_asset

    root = tmp_path / "source"
    _write_synthetic_asset(root / "asset")
    document = ScenarioDocument.new()
    document.add_default_actor(ActorRole.TX)
    document.add_default_actor(ActorRole.RX)
    mapping = canonical_scenario_mapping(document.scenario)
    (root / "scene.xml").write_text('<scene version="3.0.0"><shape type="cube"/></scene>')
    mapping["scene"] = {"source": "local", "id": "scene.xml"}
    mapping["timeline"] = {"steps": 5, "duration_s": 1.0}
    asset = {
        "source": "animated_asset",
        "path": "asset/animated_target.json",
        "playback": {
            "scenario_clock_id": "test:scene",
            "motion_clock_id": "test:motion",
            "playback_rate": 1.0,
            "playback_offset_s": 0.0,
        },
    }
    mapping["actors"]["targets"] = [
        {
            "name": "Walker",
            "asset": asset,
            "mobility": {"type": "stationary", "position_m": [0, 0, 0]},
        }
    ]
    (root / "scenario.yaml").write_text(yaml.safe_dump(mapping))
    document = load_for_authoring(root).document
    monkeypatch.setattr(
        animated_asset,
        "materialize_animated_meshes",
        lambda *_: pytest.fail("Preview must not write RT meshes"),
    )
    compiler = ScenarioCompiler(PROJECT)
    result = compiler.compile(document, scenario_directory=root)
    assert result.valid, result.issues
    actor = document.scenario.actor_by_name("Walker")
    expected = prepare_animated_asset(asset, Timeline(5, 1.0), target_name="Walker", base_dir=root)
    for index, frame in enumerate(result.target_assets[actor.id]):
        np.testing.assert_allclose(
            frame.mesh.vertices + result.samples[actor.id].positions[index],
            expected.binding.samples[index].vertices_m,
            atol=1e-6,
        )
    interactive = compiler.compile(document, scenario_directory=root, preview_only=True)
    assert interactive.valid, interactive.issues
    provider = interactive.pose_sources[actor.id]
    np.testing.assert_allclose(provider.trajectory_positions, expected.centers_m)
    for index in range(5):
        frame = interactive.target_assets[actor.id][index]
        np.testing.assert_allclose(
            frame.mesh.vertices + provider.sample(index).positions[0],
            expected.binding.samples[index].vertices_m,
            atol=1e-6,
        )
    assert len(provider._samples) <= 3
    import visualizer.src.authoring.animated_preview as preview_module

    monkeypatch.setattr(
        preview_module,
        "load_animated_target",
        lambda *_: pytest.fail("Unchanged animation was loaded again"),
    )
    refreshed = compiler.compile(document, scenario_directory=root, preview_only=True)
    assert refreshed.pose_sources[actor.id] is provider
    destination = save_document(document, tmp_path / "copy", compiler=compiler)
    reopened = load_for_authoring(destination)
    assert reopened.editable
    assert compiler.compile(reopened.document, scenario_directory=destination.parent).valid
    assert not list(tmp_path.rglob("sample_*.ply"))


@pytest.mark.parametrize("rate,offset", [(1.0, 0.0), (0.5, 0.25)])
def test_animation_path_matches_canonical_binding_with_bounded_geometry(
    tmp_path, monkeypatch, rate, offset
):
    from shared.animated_target import AnimatedTargetAsset
    from visualizer.src.authoring.animated_preview import AnimatedPreview

    path, _ = _write_synthetic_asset(tmp_path / "asset")
    spec = {
        "path": str(path),
        "playback": {
            "scenario_clock_id": "test:scene",
            "motion_clock_id": "test:motion",
            "playback_rate": rate,
            "playback_offset_s": offset,
        },
    }
    timeline = Timeline(129, 1.0)
    expected = prepare_animated_asset(spec, timeline, target_name="Walker", base_dir=tmp_path)
    bind = AnimatedTargetAsset.bind
    batch_sizes = []

    def bounded_bind(asset, context, times):
        batch_sizes.append(len(times))
        return bind(asset, context, times)

    monkeypatch.setattr(AnimatedTargetAsset, "bind", bounded_bind)
    progress = []
    provider = AnimatedPreview(
        SimpleNamespace(
            name="Walker",
            target=SimpleNamespace(path=str(path), playback=SimpleNamespace(**spec["playback"])),
        ),
        timeline,
        tmp_path,
        SimpleNamespace(project_root=PROJECT),
        progress=progress.append,
    )
    np.testing.assert_allclose(provider.trajectory_positions, expected.centers_m)
    assert len(set(provider.trajectory_positions)) > 1
    assert max(batch_sizes) == 1
    assert len(batch_sizes) == 1  # Only the displayed frame needs full binding and normals.
    assert len(provider._samples) <= 3
    assert "129/129 positions" in progress[-1]
    prepared_count = len(batch_sizes)
    for index in (0, 64, 128):
        assert provider.sample(index).positions == (provider.trajectory_positions[index],)
    assert len(batch_sizes) == prepared_count  # Inspecting a path never loads display meshes.


def test_numeric_keys_and_hold_last_survive_import():
    snapshot = ScenarioSourceSnapshot.from_mapping(
        {"view_defaults": {"labels": {2: "node", 3.5: "other"}}}
    )
    assert snapshot.to_mapping()["view_defaults"]["labels"] == {2: "node", 3.5: "other"}
    mapping = canonical_scenario_mapping(ScenarioDocument.new().scenario)
    mapping["actors"]["targets"] = [
        {
            "name": "Car",
            "mobility": {"type": "stationary", "position_m": [0, 0, 0]},
            "asset": {"source": "catalog", "id": "cube", "mesh_end_behavior": "hold_last"},
        }
    ]
    scenario = scenario_from_mapping(mapping)
    assert canonical_scenario_mapping(scenario)["actors"]["targets"] == mapping["actors"]["targets"]


def test_sensing_inspector_shows_imported_settings_without_mutation(qtbot):
    widget = SensingInspector()
    qtbot.addWidget(widget)
    mapping = {
        "sensing": {"enabled": True, "acquisition": {"bandwidth_hz": 5e8}},
        "raytracing": {"carrier_frequency_hz": 28e9},
    }
    widget.set_mapping(mapping)
    assert "read-only" in widget.summary.text()
    assert widget.tree.topLevelItemCount() == 2
    assert widget.tree.topLevelItem(0).child(1).child(0).text(1) == "500 MHz"
    assert mapping["sensing"]["acquisition"]["bandwidth_hz"] == 5e8
