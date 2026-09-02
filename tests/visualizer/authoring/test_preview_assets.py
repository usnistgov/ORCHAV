"""Compiler-owned renderer-neutral preview asset tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import visualizer.src.authoring.assets as preview_assets_module
from visualizer.src.authoring import (
    ActorRole,
    ActorSamples,
    AuthoringActor,
    AuthoringScenario,
    FixedOrientation,
    PreviewAssetCompiler,
    ScenarioCompiler,
    SceneReference,
    Stationary,
    TargetAsset,
    TimelineSettings,
    prepared_actor_pose,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _scenario(
    *,
    scene: SceneReference | None = None,
    target: AuthoringActor | None = None,
    steps: int = 3,
) -> AuthoringScenario:
    actors = [
        AuthoringActor.create(ActorRole.TX, "TX1", position=(-2.0, 0.0, 1.0)),
        AuthoringActor.create(ActorRole.RX, "RX1", position=(2.0, 0.0, 1.0)),
    ]
    if target is not None:
        actors.append(target)
    return AuthoringScenario(
        scene=scene or SceneReference("library", "empty/empty.xml"),
        timeline=TimelineSettings(steps=steps, duration_s=3.0),
        actors=tuple(actors),
    )


def _target(
    asset_id: str,
    pattern: str,
    *,
    scale: float = 1.0,
    material: str = "glass",
    mesh_animation: bool = True,
) -> AuthoringActor:
    return replace(
        AuthoringActor.create(
            ActorRole.TARGET,
            "Target1",
            target=TargetAsset.from_catalog_id(
                asset_id,
                mesh_pattern=pattern,
                material=material,
                scale=scale,
                mesh_animation=mesh_animation,
            ),
        ),
        mobility=Stationary(position_m=(3.0, -2.0, 1.5)),
        orientation=FixedOrientation(
            yaw_deg=31.0,
            pitch_deg=-12.0,
            roll_deg=4.0,
        ),
    )


def _world_aabb_center(mesh_vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    world = (transform[:3, :3] @ vertices.T).T + transform[:3, 3]
    return (world.min(axis=0) + world.max(axis=0)) / 2.0


def _write_triangle_ply(path: Path, width: float) -> None:
    path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                f"{width} 0 0",
                "0 1 0",
                "3 0 1 2",
            )
        ),
        encoding="utf-8",
    )


def test_library_scene_assets_preserve_reference_and_reuse_revision_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = ScenarioCompiler(PROJECT_ROOT)
    scenario = _scenario(scene=SceneReference("library", "ground/ground_60x50.xml"))

    first = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)

    def unexpected_reload(*_args, **_kwargs):
        raise AssertionError("unchanged scene revision should use the compiler cache")

    monkeypatch.setattr(
        preview_assets_module.XMLSceneHandler,
        "load_xml_scene",
        unexpected_reload,
    )
    second = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)

    assert first.valid, first.issues
    assert first.resolved_scene_path == str(
        (PROJECT_ROOT / "libraries/scenes/ground/ground_60x50.xml").resolve()
    )
    assert len(first.scene_assets) == 1
    asset = first.scene_assets[0]
    assert (asset.scene_source, asset.scene_id) == ("library", "ground/ground_60x50.xml")
    assert len(asset.mesh.vertices) == 4
    assert asset.material.base_color[:3] == pytest.approx((0.18, 0.18, 0.18))
    assert second.scene_assets is first.scene_assets
    assert second.scene_assets[0].mesh is first.scene_assets[0].mesh


def test_target_animation_order_and_rotated_aabb_center_match_prepared_pose() -> None:
    target = _target(
        "nist_human_walking",
        "fitted_Image_Psm_01_0000[12].ply",
        scale=1.25,
        mesh_animation=True,
    )
    scenario = _scenario(target=target, steps=3)

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert result.valid, result.issues
    frames = result.target_assets[target.id]
    assert [Path(frame.mesh_path).name for frame in frames] == [
        "fitted_Image_Psm_01_00001.ply",
        "fitted_Image_Psm_01_00002.ply",
        "fitted_Image_Psm_01_00001.ply",
    ]
    samples = result.samples[target.id]
    for frame, position, orientation in zip(
        frames,
        samples.positions,
        samples.orientations,
    ):
        world_transform = prepared_actor_pose(position, orientation) @ frame.local_to_actor
        np.testing.assert_allclose(
            _world_aabb_center(frame.mesh.vertices, world_transform),
            position,
            atol=1e-6,
        )


def test_target_mesh_switch_disabled_uses_sorted_first_frame() -> None:
    target = _target(
        "nist_human_walking",
        "fitted_Image_Psm_01_0000[12].ply",
        mesh_animation=False,
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        _scenario(target=target, steps=3),
        scenario_directory=PROJECT_ROOT,
    )

    assert result.valid, result.issues
    frames = result.target_assets[target.id]
    assert {Path(frame.mesh_path).name for frame in frames} == {"fitted_Image_Psm_01_00001.ply"}
    assert all(frame.mesh is frames[0].mesh for frame in frames)


def test_target_material_scale_and_file_revision_caches_are_semantic() -> None:
    compiler = ScenarioCompiler(PROJECT_ROOT)
    target = _target("cube", "cube.ply", scale=2.0, material="metal")
    scenario = _scenario(target=target, steps=2)

    first = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)
    second = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)

    assert first.valid and second.valid
    first_frame = first.target_assets[target.id][0]
    second_frame = second.target_assets[target.id][0]
    assert first_frame.mesh is second_frame.mesh
    assert first_frame.material is second_frame.material
    np.testing.assert_allclose(np.ptp(first_frame.mesh.vertices, axis=0), (2.0, 2.0, 2.0))

    glass_target = replace(
        target,
        target=replace(target.target, material="glass") if target.target is not None else None,
    )
    glass = compiler.compile(
        replace(scenario, actors=(*scenario.actors[:2], glass_target)),
        scenario_directory=PROJECT_ROOT,
    )
    assert glass.valid, glass.issues
    glass_frame = glass.target_assets[target.id][0]
    assert glass_frame.mesh is first_frame.mesh
    assert glass_frame.material is not first_frame.material

    larger_target = replace(
        target,
        target=replace(target.target, scale=3.0) if target.target is not None else None,
    )
    larger = compiler.compile(
        replace(scenario, actors=(*scenario.actors[:2], larger_target)),
        scenario_directory=PROJECT_ROOT,
    )
    assert larger.valid, larger.issues
    larger_frame = larger.target_assets[target.id][0]
    assert larger_frame.mesh is not first_frame.mesh
    np.testing.assert_allclose(np.ptp(larger_frame.mesh.vertices, axis=0), (3.0, 3.0, 3.0))


def test_target_mesh_cache_invalidates_when_source_revision_changes(tmp_path: Path) -> None:
    mesh_path = tmp_path / "changing.ply"
    _write_triangle_ply(mesh_path, 1.0)
    target = _target("cube", "cube.ply")
    samples = ActorSamples(positions=((0.0, 0.0, 0.0),), orientations=((0.0, 0.0, 0.0),))
    asset_compiler = PreviewAssetCompiler(PROJECT_ROOT)

    first = asset_compiler.build_target_assets(target, (mesh_path,), samples)[0]
    _write_triangle_ply(mesh_path, 10.0)
    second = asset_compiler.build_target_assets(target, (mesh_path,), samples)[0]

    assert second.mesh is not first.mesh
    assert second.cache_key != first.cache_key
    assert np.ptp(second.mesh.vertices[:, 0]) == pytest.approx(10.0)


def test_incomplete_document_still_exposes_valid_scene_and_target() -> None:
    target = _target("cube", "cube.ply")
    scenario = AuthoringScenario(
        scene=SceneReference("library", "ground/ground_60x50.xml"),
        actors=(AuthoringActor.create(ActorRole.TX, "TX1"), target),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert not result.valid
    assert result.resolved_scene_path == str(
        (PROJECT_ROOT / "libraries/scenes/ground/ground_60x50.xml").resolve()
    )
    assert len(result.scene_assets) == 1
    assert len(result.target_assets[target.id]) == 30
    assert not any(issue.code == "target.preview.samples_missing" for issue in result.issues)


def test_missing_scene_does_not_suppress_valid_target_preview() -> None:
    target = _target("cube", "cube.ply")
    scenario = AuthoringScenario(
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1"),
            target,
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert not result.valid
    assert result.runtime is None
    assert any(issue.code == "scene.required" for issue in result.issues)
    assert len(result.target_assets[target.id]) == 30
    assert not any(issue.code == "target.preview.samples_missing" for issue in result.issues)


def test_invalid_target_does_not_suppress_valid_peer_preview() -> None:
    valid_target = _target("cube", "cube.ply")
    invalid_target = _target("cube", "cube.ply").with_changes(
        name="InvalidTarget",
        target=replace(
            TargetAsset.from_catalog_id("cube", mesh_pattern="cube.ply"),
            scale=0.0,
        ),
    )
    scenario = _scenario(target=valid_target)
    scenario = replace(scenario, actors=(*scenario.actors, invalid_target))

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert not result.valid
    assert set(result.target_assets) == {valid_target.id}
    assert len(result.target_assets[valid_target.id]) == 3
    assert any(
        issue.code == "target.scale.positive" and issue.actor_id == invalid_target.id
        for issue in result.issues
    )
    assert not any(issue.code == "target.preview.samples_missing" for issue in result.issues)


def test_invalid_scene_never_exposes_preview_assets() -> None:
    scenario = AuthoringScenario(
        scene=SceneReference("library", "missing/not-there.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX1"),
            AuthoringActor.create(ActorRole.RX, "RX1"),
        ),
    )

    result = ScenarioCompiler(PROJECT_ROOT).compile(
        scenario,
        scenario_directory=PROJECT_ROOT,
    )

    assert not result.valid
    assert result.resolved_scene_path is None
    assert result.scene_assets == ()
    assert result.target_assets == {}


def test_preview_loader_failures_are_structured_and_clear_all_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = ScenarioCompiler(PROJECT_ROOT)
    scenario = _scenario(scene=SceneReference("library", "ground/ground_60x50.xml"))

    def fail_scene(*_args, **_kwargs):
        raise ValueError("broken scene payload")

    monkeypatch.setattr(compiler._preview_assets, "build_scene_assets", fail_scene)
    result = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)

    issue = next(issue for issue in result.issues if issue.code == "scene.asset.load_failed")
    assert issue.path == "scene.id"
    assert "broken scene payload" in issue.message
    assert not result.valid
    assert result.scene_assets == ()
    assert result.target_assets == {}


def test_target_preview_failure_is_structured_and_never_exposes_partial_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = ScenarioCompiler(PROJECT_ROOT)
    target = _target("cube", "cube.ply")
    scenario = _scenario(target=target, steps=2)

    def fail_target(*_args, **_kwargs):
        raise ValueError("broken target payload")

    monkeypatch.setattr(compiler._preview_assets, "build_target_assets", fail_target)
    result = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)

    issue = next(issue for issue in result.issues if issue.code == "target.asset.load_failed")
    assert issue.path == "actors.targets.0.asset.id"
    assert issue.actor_id == target.id
    assert "broken target payload" in issue.message
    assert not result.valid
    assert result.target_assets == {}


def test_target_preview_failure_preserves_successful_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = ScenarioCompiler(PROJECT_ROOT)
    valid_target = _target("cube", "cube.ply")
    failed_target = _target("cube", "cube.ply").with_changes(name="FailedTarget")
    scenario = _scenario(target=valid_target, steps=2)
    scenario = replace(scenario, actors=(*scenario.actors, failed_target))
    build_target_assets = compiler._preview_assets.build_target_assets

    def fail_one_target(actor, *args, **kwargs):
        if actor.id == failed_target.id:
            raise ValueError("broken target payload")
        return build_target_assets(actor, *args, **kwargs)

    monkeypatch.setattr(compiler._preview_assets, "build_target_assets", fail_one_target)
    result = compiler.compile(scenario, scenario_directory=PROJECT_ROOT)

    issue = next(issue for issue in result.issues if issue.code == "target.asset.load_failed")
    assert issue.actor_id == failed_target.id
    assert not result.valid
    assert set(result.target_assets) == {valid_target.id}
    assert len(result.target_assets[valid_target.id]) == 2
