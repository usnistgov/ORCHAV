"""Interactive preparation stays bounded, canonical, and resource-aware."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from shared.scenarios.actors import LinearMobilitySpec, MeshSequenceMobilitySpec
from tests.visualizer.authoring.test_persistence import _external_mapping
from tests.visualizer.authoring.test_preview_assets import _scenario, _write_triangle_ply
from visualizer.src.authoring import ActorRole, AuthoringActor, ScenarioCompiler, TargetAsset
from visualizer.src.authoring.dependencies import ScenarioDependencyCollector
from visualizer.src.authoring.frame_geometry import FrameGeometry
from visualizer.src.authoring.orientation_models import actor_look_at_orientation
from visualizer.src.authoring.persistence import create_scenario_copy, load_for_authoring
from visualizer.src.authoring.recording import RecordingPreview

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("kind", ["relative", "absolute", "project", "library"])
def test_directory_targets_use_shared_path_semantics(tmp_path, kind):
    project = tmp_path / "project"
    root = project / "scenario"
    root.mkdir(parents=True)
    directory = root / "meshes" if kind == "relative" else project / "libraries/targets/drone"
    directory.mkdir(parents=True)
    _write_triangle_ply(directory / "drone.ply", 1)
    path = {
        "relative": "meshes",
        "absolute": str(directory),
        "project": "${PROJECT_ROOT}/libraries/targets/drone",
        "library": "libraries/targets/drone",
    }[kind]
    actor = AuthoringActor.create(
        ActorRole.TARGET,
        "Drone",
        target=TargetAsset(
            asset_id="drone",
            mesh_directory=path,
            source="directory",
            path=path,
            mesh_pattern="drone.ply",
            mesh_animation=False,
        ),
    )
    compiler = ScenarioCompiler(project)
    result = compiler.prepare_actor_preview(
        _scenario(target=actor), actor.id, scenario_directory=root
    )
    assert result.valid, result.issues
    assert Path(result.target_assets[actor.id][0].mesh_path) == directory / "drone.ply"
    compiler.close()


def test_long_timeline_does_not_load_every_mesh(tmp_path, monkeypatch):
    import visualizer.src.authoring.assets as assets
    from visualizer.src.services.target_asset_cache import TargetAssetCache

    paths = []
    for i in range(8):
        path = tmp_path / f"pose{i}.ply"
        _write_triangle_ply(path, i + 1)
        paths.append(path)
    actor = AuthoringActor.create(
        ActorRole.TARGET,
        "Actor",
        target=TargetAsset(
            asset_id="poses",
            mesh_directory=str(tmp_path),
            source="directory",
            path=str(tmp_path),
            mesh_pattern="*.ply",
            mesh_animation=True,
        ),
    )
    compiler = assets.PreviewAssetCompiler(ROOT)
    compiler.target_cache.close()
    compiler.target_cache = TargetAssetCache(max_entries=4)
    # Disable speculative reads here so the visible working set is deterministic.
    monkeypatch.setattr(compiler.target_cache, "prefetch_after", lambda *a, **k: None)
    calls = []
    original = assets.load_mesh_payload

    def load(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(assets, "load_mesh_payload", load)
    samples = SimpleNamespace(
        positions=((0.0, 0.0, 0.0),) * 10240, orientations=((0.0, 0.0, 0.0),) * 10240
    )
    frames = compiler.build_target_assets(actor, tuple(paths), samples)
    assert isinstance(frames, FrameGeometry)
    assert len(frames) == 10240
    assert len(calls) == 1
    for index in range(8):
        assert frames[index].mesh_path == str(paths[index])
    assert compiler.target_cache.telemetry()["entries"] <= 4
    compiler.target_cache.close()


def test_edit_prepares_only_reference_component_and_reuses_other_poses(tmp_path, monkeypatch):
    import visualizer.src.authoring.compiler as module

    mover = AuthoringActor.create(ActorRole.RX, "Mover").with_changes(
        mobility=LinearMobilitySpec(start_m=(0, 0, 0), end_m=(10, 0, 0))
    )
    observer = AuthoringActor.create(ActorRole.TX, "Observer").with_changes(
        orientation=actor_look_at_orientation(mover.id)
    )
    other = AuthoringActor.create(ActorRole.RX, "Unrelated")
    scenario = replace(_scenario(), actors=(mover, observer, other))
    compiler = ScenarioCompiler(ROOT)
    calls = []
    original = module.prepare_scenario

    def prepare(model, **kwargs):
        calls.append(model)
        return original(model, **kwargs)

    monkeypatch.setattr(module, "prepare_scenario", prepare)
    monkeypatch.setattr(
        compiler,
        "_compile_scene_preview_assets",
        lambda *a: pytest.fail("edit loaded the environment"),
    )
    first = compiler.prepare_actor_preview(scenario, mover.id, scenario_directory=tmp_path)
    assert first.valid, first.issues
    assert set(first.samples) == {mover.id, observer.id}
    second = compiler.prepare_actor_preview(scenario, mover.id, scenario_directory=tmp_path)
    assert second.samples == first.samples
    assert len(calls) == 1
    compiler.close()


def test_relative_pose_resource_revision_invalidates_cache(tmp_path):
    path = tmp_path / "positions.npy"
    np.save(path, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    actor = AuthoringActor.create(
        ActorRole.TARGET, "Target", target=TargetAsset.from_catalog_id("cube", mesh_animation=False)
    ).with_changes(mobility=MeshSequenceMobilitySpec(positions_path="positions.npy"))
    scenario = replace(_scenario(), actors=(actor,))
    compiler = ScenarioCompiler(ROOT)
    first = compiler.prepare_actor_preview(scenario, actor.id, scenario_directory=tmp_path)
    assert first.valid, first.issues
    np.save(path, np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]))
    second = compiler.prepare_actor_preview(scenario, actor.id, scenario_directory=tmp_path)
    assert second.valid, second.issues
    assert first.samples[actor.id].positions[-1][0] == 2
    assert second.samples[actor.id].positions[-1][0] == 8
    compiler.close()


def test_save_as_collects_target_surface_and_recipe_closure(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "skins").mkdir()
    surface = {
        "schema": "orchav.target_surface.v1",
        "texture": "albedo.png",
        "uv_map": "uv.npz",
        "material": {"normal_map_path": "normal.png"},
    }
    (source / "skins/surface.json").write_text(json.dumps(surface))
    for filename in ("albedo.png", "uv.npz", "normal.png"):
        (source / "skins" / filename).write_bytes(b"fixture")
    (source / "appearance.json").write_text("{}")
    mapping = _external_mapping()
    mapping["view_defaults"] = {
        "appearance_recipe": "appearance.json",
        "visual_profiles": [
            {"match": {"name": "Actor"}, "preset": "Matte", "surface": "skins/surface.json"}
        ],
    }
    path = source / "scenario.yaml"
    path.write_text(yaml.safe_dump(mapping))
    deps = ScenarioDependencyCollector(ROOT).collect(mapping, source)
    assert {d.relative_path for d in deps} >= {
        "appearance.json",
        "skins/surface.json",
        "skins/uv.npz",
        "skins/albedo.png",
        "skins/normal.png",
    }
    copy = create_scenario_copy(path, tmp_path / "copy")
    assert {d.relative_path for d in copy.scenario.dependencies} >= {"skins/uv.npz"}


def test_recorded_profile_is_read_only_and_uses_bounded_frame_projections(tmp_path, monkeypatch):
    import visualizer.src.authoring.recording as module

    mapping = _external_mapping()
    path = tmp_path / "scenario.yaml"
    path.write_text("# Saved frames own target poses.\n" + yaml.safe_dump(mapping))
    monkeypatch.setattr(
        ScenarioDependencyCollector,
        "collect",
        lambda *args: pytest.fail("Recorded playback scanned authoring dependencies"),
    )
    loaded = load_for_authoring(path)
    assert loaded.document.read_only
    assert loaded.document.scenario.recorded_poses
    requests = []

    class Reader:
        frame_ids_snapshot = tuple(range(10240))

        def __init__(self, *args, **kwargs):
            pass

        def load_frame_projection(self, index, request):
            requests.append(request)
            return SimpleNamespace(
                frame=SimpleNamespace(
                    tx_names=["TX1"],
                    tx_positions=[(index, 5, 8)],
                    tx_orientations=[(np.pi / 2, 0, 0)],
                )
            )

        def close(self):
            pass

    monkeypatch.setattr(module, "HDF5FormatHandler", Reader)
    preview = RecordingPreview(loaded.document.scenario, tmp_path, None)
    actor = loaded.document.scenario.actors[0]
    for step in range(10):
        sample = preview.sample(actor, step)
        assert sample.positions[0] == (step, 5, 8)
        assert sample.orientations[0][0] == pytest.approx(90)
    assert len(preview._frames) == 3
    assert all(r.components == frozenset({"devices", "targets"}) for r in requests)
    preview.close()
