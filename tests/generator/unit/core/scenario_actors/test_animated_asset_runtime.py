import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.scenario_actors.animated_asset import (
    materialize_animated_meshes,
    prepare_animated_asset,
)
from generator.core.scenario_actors.runtime import prepare_actor_runtime
from generator.core.scenario_actors.types import Timeline
from shared.animated_target_workflow import _write_synthetic_asset
from shared.scenarios.actors import TargetActorSpec


def _actor(path="asset/animated_target.json", *, rate=1.0, offset=0.0):
    return TargetActorSpec.model_validate(
        {
            "name": "Walker",
            "mobility": {"type": "stationary", "position_m": [0, 0, 0]},
            "asset": {
                "source": "animated_asset",
                "path": path,
                "playback": {
                    "scenario_clock_id": "scenario:test",
                    "motion_clock_id": "motion:test",
                    "playback_rate": rate,
                    "playback_offset_s": offset,
                },
            },
        }
    )


def _configuration(tmp_path, actor, *, steps=5, duration=1.0):
    return SimpleNamespace(
        root=tmp_path,
        project_root=tmp_path,
        timeline=SimpleNamespace(steps=steps, duration_s=duration),
        actors=SimpleNamespace(tx=(), rx=(), targets=(actor,)),
        groups=(),
    )


def test_animated_asset_requires_explicit_clocks_and_single_transform_owner():
    actor = _actor().model_dump()
    for missing in ("scenario_clock_id", "motion_clock_id", "playback_rate", "playback_offset_s"):
        candidate = _actor().model_dump()
        del candidate["asset"]["playback"][missing]
        with pytest.raises(ValueError):
            TargetActorSpec.model_validate(candidate)
    actor["mobility"]["position_m"] = (1, 0, 0)
    with pytest.raises(ValueError, match="owns root motion"):
        TargetActorSpec.model_validate(actor)
    actor = _actor().model_dump()
    actor["orientation"]["yaw_deg"] = 90
    with pytest.raises(ValueError, match="owns root orientation"):
        TargetActorSpec.model_validate(actor)
    actor = _actor().model_dump()
    actor["asset"]["scale"] = 2
    with pytest.raises(ValueError):
        TargetActorSpec.model_validate(actor)


def test_exact_clock_binding_centers_and_root_velocity(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    runtime = prepare_actor_runtime(
        _configuration(tmp_path, _actor(rate=2.0), steps=5, duration=0.5)
    )
    target = runtime.targets[0]
    prepared = target.animated_asset
    assert [s.scenario_time_s for s in prepared.binding.samples] == [0, 0.125, 0.25, 0.375, 0.5]
    assert [s.motion_time_s for s in prepared.binding.samples] == [0, 0.25, 0.5, 0.75, 1]
    assert [s.pose_sample_id for s in prepared.binding.samples] == [
        "pose:neutral_stance",
        "pose:neutral_stance",
        "pose:left_reach",
        "pose:left_reach",
        "pose:asymmetric_settle",
    ]
    np.testing.assert_allclose(target.mobility.prepared_velocities()[0], [2.4, 0.2, 0])
    np.testing.assert_allclose(target.mobility.prepared_velocities()[2], [2.4, 0.4, 0.4])
    assert target.orientation.orientations() == [(0, 0, 0)] * 5
    assert target.scale == 1 and not target.use_ply_position
    assert target.initial_position == prepared.centers_m[0]
    assert runtime.prepared.actor("Walker").positions_m == prepared.centers_m
    for index, sample in enumerate(prepared.binding.samples):
        payload = (target.resolved_mesh_directory / f"sample_{index:09d}.ply").read_bytes()
        header, body = payload.split(b"end_header\n", 1)
        assert b"format binary_little_endian 1.0" in header
        local = np.frombuffer(body, dtype="<f8", count=len(sample.vertices_m) * 3).reshape(-1, 3)
        np.testing.assert_array_equal(local, sample.vertices_m - prepared.centers_m[index])
        faces = np.frombuffer(
            body, dtype=[("count", "u1"), ("indices", "<i4", (3,))], offset=local.nbytes
        )
        assert np.all(faces["count"] == 3)
        np.testing.assert_array_equal(faces["indices"], prepared.binding.faces)
        np.testing.assert_allclose(local + prepared.centers_m[index], sample.vertices_m, atol=1e-14)
        np.testing.assert_allclose(np.min(local, axis=0) + np.max(local, axis=0), 0, atol=1e-14)
    manifest = json.loads((target.resolved_mesh_directory / "rt_geometry.json").read_text())
    assert manifest["ply_encoding"] == "binary_little_endian"
    assert manifest["vertex_precision"] == "float64"
    for artifact in manifest["artifacts"]:
        assert (
            artifact["sha256"]
            == hashlib.sha256(
                (target.resolved_mesh_directory / artifact["path"]).read_bytes()
            ).hexdigest()
        )
    repeated = prepare_actor_runtime(
        _configuration(tmp_path, _actor(rate=2.0), steps=5, duration=0.5)
    )
    assert repeated.targets[0].animated_asset.binding.binding_id == prepared.binding.binding_id
    assert repeated.targets[0].resolved_mesh_directory == target.resolved_mesh_directory


def test_out_of_range_and_corrupted_cache_fail_before_rt(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    with pytest.raises(ValueError, match="outside"):
        prepare_actor_runtime(_configuration(tmp_path, _actor(), duration=2.0))
    target = prepare_actor_runtime(_configuration(tmp_path, _actor())).targets[0]
    mesh = target.resolved_mesh_directory / "sample_000000000.ply"
    mesh.write_text("damaged")
    with pytest.raises(ValueError, match="cache content mismatch"):
        materialize_animated_meshes(target.animated_asset, tmp_path)


def test_runtime_progress_preserves_binding_and_cached_meshes(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    configuration = _configuration(tmp_path, _actor())
    baseline = prepare_actor_runtime(configuration).targets[0]
    cache = baseline.resolved_mesh_directory
    expected_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cache.iterdir()
    }
    events = []

    def report(stage, done, total):
        events.append((stage, done, total))
        if stage.endswith("Preparing ray-tracing mesh cache") and done == total:
            assert (cache / "binding.json").is_file()
            assert (cache / "rt_geometry.json").is_file()

    target = prepare_actor_runtime(configuration, on_progress=report).targets[0]
    assert events == [
        (f"Walker: {stage}", done, total)
        for stage, total in [
            ("Verifying pose resources", len(baseline.animated_asset.asset.resources)),
            ("Validating pose geometry", len(baseline.animated_asset.asset.pose_samples)),
            ("Binding animation poses", 5),
            ("Computing motion velocities", 5),
            ("Preparing ray-tracing mesh cache", 5),
        ]
        for done in range(total + 1)
    ]
    assert target.animated_asset.binding.to_canonical_json() == (
        baseline.animated_asset.binding.to_canonical_json()
    )
    assert target.animated_asset.centers_m == baseline.animated_asset.centers_m
    assert target.animated_asset.root_velocities_mps == baseline.animated_asset.root_velocities_mps
    assert target.resolved_mesh_directory == cache
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cache.iterdir()
    } == expected_hashes


def test_cache_progress_waits_for_authenticated_sidecars(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    target = prepare_actor_runtime(_configuration(tmp_path, _actor())).targets[0]
    sidecar = target.resolved_mesh_directory / "rt_geometry.json"
    sidecar.write_bytes(b"damaged cache metadata")
    events = []
    with pytest.raises(ValueError, match="cache content mismatch"):
        materialize_animated_meshes(
            target.animated_asset,
            tmp_path,
            on_progress=lambda stage, done, total: events.append((stage, done, total)),
        )
    assert events == [("Walker: Preparing ray-tracing mesh cache", done, 5) for done in range(5)]
    assert sidecar.read_bytes() == b"damaged cache metadata"


def test_runtime_binds_once_but_reauthenticates_on_the_next_run(tmp_path, monkeypatch):
    from generator.core.scenario_actors import animated_asset as module

    _write_synthetic_asset(tmp_path / "asset")
    original = module.prepare_animated_asset
    calls = []

    def counted(*args, **kwargs):
        prepared = original(*args, **kwargs)
        calls.append(prepared)
        return prepared

    monkeypatch.setattr(module, "prepare_animated_asset", counted)
    config = _configuration(tmp_path, _actor())
    runtime = prepare_actor_runtime(config)
    assert len(calls) == 1
    assert runtime.targets[0].animated_asset is calls[0]
    repeated = prepare_actor_runtime(config)
    assert len(calls) == 2
    assert repeated.targets[0].animated_asset is calls[1]
    assert calls[0].binding.binding_id == calls[1].binding.binding_id
    manifest = json.loads((tmp_path / "asset/animated_target.json").read_text())
    resource = tmp_path / "asset" / manifest["resources"][0]["path"]
    resource.write_bytes(resource.read_bytes() + b" ")
    with pytest.raises(ValueError, match="(SHA-256|size|hash)"):
        prepare_actor_runtime(config)


def test_binding_identity_changes_with_clock_rate_offset_and_asset(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    timeline = Timeline(3, 0.25)
    first = prepare_animated_asset(
        _actor().asset, timeline, target_name="Walker", base_dir=tmp_path
    )
    shifted = prepare_animated_asset(
        _actor(offset=0.25).asset, timeline, target_name="Walker", base_dir=tmp_path
    )
    fast = prepare_animated_asset(
        _actor(rate=2).asset, timeline, target_name="Walker", base_dir=tmp_path
    )
    assert len({first.binding.binding_id, shifted.binding.binding_id, fast.binding.binding_id}) == 3
    assert first.catalog_metadata()["asset_revision_sha256"] == first.asset.revision_sha256
    assert shifted.sample_metadata(0)["sample"]["motion_time_s"] == 0.25


def test_cache_rejects_linked_ancestor_and_concurrent_content_change(tmp_path, monkeypatch):
    from generator.core.scenario_actors import animated_asset as module

    _write_synthetic_asset(tmp_path / "asset")
    prepared = prepare_animated_asset(
        _actor().asset, Timeline(2, 1), target_name="Walker", base_dir=tmp_path
    )
    original_is_symlink = Path.is_symlink
    with monkeypatch.context() as patch:
        patch.setattr(
            Path, "is_symlink", lambda path: path.name == ".orchav" or original_is_symlink(path)
        )
        with pytest.raises(ValueError, match="links or junctions"):
            materialize_animated_meshes(prepared, tmp_path)
    real_link = module.os.link

    def raced_link(source, destination):
        destination.write_bytes(b"concurrent unrelated data")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", raced_link)
    with pytest.raises(ValueError, match="cache content mismatch"):
        materialize_animated_meshes(prepared, tmp_path)
    cache = (
        tmp_path / ".orchav" / "animated_targets" / (prepared.binding.binding_id + "-ply-binary-v1")
    )
    assert (cache / "sample_000000000.ply").read_bytes() == b"concurrent unrelated data"


@pytest.mark.skipif(os.name != "nt", reason="Windows rename has no-overwrite semantics")
def test_cache_falls_back_to_no_overwrite_rename_if_hardlink_denied(tmp_path, monkeypatch):
    from generator.core.scenario_actors import animated_asset as module

    _write_synthetic_asset(tmp_path / "asset")
    prepared = prepare_animated_asset(
        _actor().asset, Timeline(2, 1), target_name="Walker", base_dir=tmp_path
    )

    def denied(*_args):
        raise PermissionError("hard links are unavailable")

    monkeypatch.setattr(module.os, "link", denied)
    cache = materialize_animated_meshes(prepared, tmp_path)
    first = cache / "sample_000000000.ply"
    original = first.read_bytes()
    assert original
    assert materialize_animated_meshes(prepared, tmp_path) == cache
    assert first.read_bytes() == original
    assert not list(cache.glob(".tmp-*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows rename has no-overwrite semantics")
def test_cache_fallback_does_not_replace_competing_writer(tmp_path, monkeypatch):
    from generator.core.scenario_actors import animated_asset as module

    _write_synthetic_asset(tmp_path / "asset")
    prepared = prepare_animated_asset(
        _actor().asset, Timeline(2, 1), target_name="Walker", base_dir=tmp_path
    )
    real_rename = module.os.rename

    def denied(*_args):
        raise PermissionError("hard links are unavailable")

    def raced_rename(source, destination):
        destination.write_bytes(b"competing writer")
        return real_rename(source, destination)

    monkeypatch.setattr(module.os, "link", denied)
    monkeypatch.setattr(module.os, "rename", raced_rename)
    with pytest.raises(ValueError, match="cache content mismatch"):
        materialize_animated_meshes(prepared, tmp_path)
    cache = (
        tmp_path / ".orchav" / "animated_targets" / (prepared.binding.binding_id + "-ply-binary-v1")
    )
    assert (cache / "sample_000000000.ply").read_bytes() == b"competing writer"
    assert not list(cache.glob(".tmp-*"))


@pytest.mark.parametrize("platform", ["posix", "nt"])
def test_cache_hardlink_permission_failure_dispatch_and_cleanup(tmp_path, monkeypatch, platform):
    from generator.core.scenario_actors import animated_asset as module

    _write_synthetic_asset(tmp_path / "asset")
    prepared = prepare_animated_asset(
        _actor().asset, Timeline(2, 1), target_name="Walker", base_dir=tmp_path
    )
    renamed = []

    def denied(*_args):
        raise PermissionError("hard links are unavailable")

    def no_overwrite_rename(source, destination):
        # Exercise dispatch on either host; native Windows semantics are
        # checked separately above. Exclusive creation models a missing target.
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
        source.unlink()
        renamed.append(destination)

    monkeypatch.setattr(
        module, "os", SimpleNamespace(name=platform, link=denied, rename=no_overwrite_rename)
    )
    if platform == "posix":
        with pytest.raises(PermissionError, match="hard links are unavailable"):
            materialize_animated_meshes(prepared, tmp_path)
        assert not renamed
    else:
        cache = materialize_animated_meshes(prepared, tmp_path)
        assert len(renamed) == len(prepared.binding.samples) + 2
        assert materialize_animated_meshes(prepared, tmp_path) == cache
        assert len(renamed) == len(prepared.binding.samples) + 2
    assert not list((tmp_path / ".orchav").rglob(".tmp-*"))


def test_binary_cache_leaves_existing_ascii_acquisition_resources_untouched(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    prepared = prepare_animated_asset(
        _actor().asset, Timeline(2, 1), target_name="Walker", base_dir=tmp_path
    )
    previous = tmp_path / ".orchav" / "animated_targets" / prepared.binding.binding_id
    previous.mkdir(parents=True)
    old_mesh = previous / "sample_000000000.ply"
    old_mesh.write_bytes(b"existing saved-acquisition resource")
    new_cache = materialize_animated_meshes(prepared, tmp_path)
    assert new_cache != previous
    assert old_mesh.read_bytes() == b"existing saved-acquisition resource"


@pytest.mark.optional_runtime
def test_native_rt_mesh_geometry_root_velocity_and_solve_selection(tmp_path):
    import mitsuba as mi
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    from generator.core.propagation.actor_state_application import apply_target_state_to_scene
    from generator.core.target.manager import TargetManager
    from generator.io.frames.builder import _process_target_metadata

    _write_synthetic_asset(tmp_path / "asset")
    target = prepare_actor_runtime(_configuration(tmp_path, _actor())).targets[0]
    scene = load_scene()
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = scene.tx_array
    scene.add(Transmitter("tx", position=[-2, -4, 0.5]))
    scene.add(Receiver("rx", position=[3, -4, 0.5]))
    manager = TargetManager(target, scene)
    manager.create_target()
    for step, cir_steps, selected in ((0, 4, 0), (1, 4, 0), (4, 4, 4), (1, 1, 1), (3, 1, 3)):
        apply_target_state_to_scene(
            [manager],
            [target.mobility.prepared_positions()[step]],
            [(0, 0, 0)],
            step,
            step_velocities=[target.mobility.prepared_velocities()[step]],
            simulation_config=SimpleNamespace(
                mesh_update_interval_s=100, num_steps=5, duration=1, cir_time_steps=cir_steps
            ),
        )
        assert manager.current_mesh_idx == selected
        params = mi.traverse(scene.mi_scene)
        key = next(key for key in params.keys() if key.endswith(".vertex_positions"))
        vertices = np.asarray(params[key]).reshape(-1, 3)
        np.testing.assert_allclose(
            vertices,
            target.animated_asset.binding.samples[selected].vertices_m,
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            np.asarray(manager.target_object.velocity).reshape(3),
            target.animated_asset.root_velocities_mps[selected],
            atol=1e-6,
        )
        before = vertices.copy()
        paths = PathSolver()(scene, max_depth=1, samples_per_src=256, refraction=False)
        coefficients, delays = paths.cir(
            sampling_frequency=2000, num_time_steps=cir_steps, out_type="numpy"
        )
        assert np.all(np.isfinite(coefficients)) and coefficients.size > 0
        assert np.any(np.abs(coefficients) > 0)
        np.testing.assert_array_equal(
            np.asarray(mi.traverse(scene.mi_scene)[key]).reshape(-1, 3), before
        )
        positions, metadata = _process_target_metadata([manager.target_object], [manager], step)
        evidence = metadata[0]["animated_asset"]
        assert evidence["state"]["sample"]["scenario_time_s"] == selected * 0.25
        assert (
            evidence["state"]["sample"]["pose_sample_id"]
            == target.animated_asset.binding.samples[selected].pose_sample_id
        )


def test_public_build_rejects_only_selected_animated_source(monkeypatch):
    import importlib.util

    from shared.scenarios.actors import FileAssetSpec

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert FileAssetSpec(path="ordinary.ply").source == "file"
    with pytest.raises(ValueError, match="v0.2 animated-target runtime"):
        _actor()


def test_single_epoch_preserves_physical_root_velocity(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    prepared = prepare_animated_asset(
        _actor(rate=2).asset, Timeline(1, 0), target_name="Walker", base_dir=tmp_path
    )
    np.testing.assert_allclose(prepared.mobility(Timeline(1, 0)).velocities_mps[0], [2.4, 0.2, 0])


def test_native_motion_clock_selects_every_pose_and_preserves_before_boundary(tmp_path):
    from shared.animated_target import (
        AnimatedTargetBindingTime,
        animated_target_revision_sha256,
    )
    from shared.scenarios.actors import TimelineSpec

    _write_synthetic_asset(tmp_path / "asset")
    path = tmp_path / "asset/animated_target.json"
    manifest = json.loads(path.read_text())
    original_poses = manifest["motion"]["pose"]["samples"]
    original_root = manifest["motion"]["root"]["samples"][0]
    times = np.arange(150, dtype=np.float64) / 30
    manifest["motion"]["pose"]["samples"] = [
        {**original_poses[i % 3], "sample_id": f"pose:time:{float(t).hex()}", "time_s": float(t)}
        for i, t in enumerate(times)
    ]
    manifest["motion"]["root"]["samples"] = [
        {**original_root, "sample_id": f"root:time:{float(t).hex()}", "time_s": float(t)}
        for i, t in enumerate(times)
    ]
    manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
    path.write_text(json.dumps(manifest))
    config = _configuration(tmp_path, _actor(), steps=150, duration=149 / 30)
    config.timeline = TimelineSpec(steps=150, sample_rate_hz=30)
    runtime = prepare_actor_runtime(config)
    prepared = runtime.targets[0].animated_asset
    assert [s.pose_sample_id for s in prepared.binding.samples] == [
        f"pose:time:{float(t).hex()}" for t in times
    ]
    assert [s.scenario_time_s for s in prepared.binding.samples] == times.tolist()
    # An explicitly earlier physical time is still earlier: the sampler has no
    # epsilon that could silently advance a caller's requested pose boundary.
    before = float(np.nextafter(times[23], -np.inf))
    exact = float(times[23])
    binding = prepared.asset.bind(
        prepared.binding.context,
        [
            AnimatedTargetBindingTime("before", before, before),
            AnimatedTargetBindingTime("exact", exact, exact),
        ],
    )
    assert [s.pose_sample_id for s in binding.samples] == [
        f"pose:time:{float(times[i]).hex()}" for i in (22, 23)
    ]
    assert binding.samples[0].motion_time_s == before
