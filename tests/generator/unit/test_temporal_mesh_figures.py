"""Temporal depiction checks using authored public scenarios and native poses."""

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from generator.core.scenario_actors.quaternion import Quaternion
from generator.figures.temporal_mesh import ALPHAS, ROOT, load_temporal_inputs, quaternion_matrix
from tests.generator.unit.test_topology_figures import authored as authored


def test_installed_figure_imports_do_not_require_scenario_packages():
    # Wheels install generator/shared/visualizer, while scenario assets are
    # explicit source inputs. Block scenario imports even in a full checkout.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.abc, sys\n"
            "class NoScenarios(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'scenarios' or fullname.startswith('scenarios.'):\n"
            "            raise ModuleNotFoundError(fullname)\n"
            "sys.meta_path.insert(0, NoScenarios())\n"
            "import generator.figures.temporal_mesh\n"
            "import generator.figures.paris_mesh_context\n",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_rotation_uses_native_quaternion_convention():
    rotation = quaternion_matrix(Quaternion.from_euler_deg(90))
    assert np.allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "relative,steps,duration",
    [
        ("cube_radar/two_moving", 1024, 1.023),
        ("cube_radar/azimuth", 1024, 1.023),
    ],
)
def test_authored_sample_positions_scales_and_poses(relative, steps, duration):
    config, _, actors, _, indices, times, _ = load_temporal_inputs(
        ROOT / "scenarios/sensing" / relative
    )
    assert config["timeline"]["steps"] == steps
    assert indices == (0, steps // 2, steps - 1)
    assert times[0] == 0
    assert times[-1] == pytest.approx(duration)
    assert times[1] == pytest.approx(duration * (steps // 2) / (steps - 1))
    assert ALPHAS == (1.0, 0.24, 1.0)
    for actor in actors:
        for mesh, index, euler in zip(actor["meshes"], indices, actor["orientation_deg"]):
            rotation = quaternion_matrix(Quaternion.from_euler_deg(*euler))
            local = (mesh.reshape(-1, 3) - actor["positions"][index]) @ rotation
            world = mesh.reshape(-1, 3)
            assert np.allclose(
                (world.min(0) + world.max(0)) / 2, actor["positions"][index], atol=1e-6
            )
            assert np.allclose(np.ptp(local, axis=0), actor["extent_m"], atol=1e-6)
        assert np.allclose(
            actor["positions"][indices[1]],
            actor["positions"][0]
            + (actor["positions"][-1] - actor["positions"][0]) * indices[1] / (steps - 1),
            atol=1e-6,
        )
        authored_target = next(t for t in config["actors"]["targets"] if t["name"] == actor["name"])
        assert actor["scale"] == authored_target["asset"].get("scale", 1.0)


def test_public_paris_authored_timeline_and_articulated_target():
    config = yaml.safe_load(
        (ROOT / "scenarios/showcases/paris_opera_multitarget/scenario.yaml").read_text()
    )
    assert config["timeline"] == {"steps": 10240, "sample_rate_hz": 2000.0}
    pedestrian = next(
        target for target in config["actors"]["targets"] if target["name"] == "Pedestrian"
    )
    assert pedestrian["asset"]["source"] == "animated_asset"
    assert pedestrian["mobility"] == {"type": "stationary", "position_m": [0.0, 0.0, 0.0]}


@pytest.fixture
def temporal_animation(authored, tmp_path):
    from shared.animated_target_workflow import _write_synthetic_asset

    path, config = authored
    manifest, _ = _write_synthetic_asset(tmp_path / "animation")
    config["actors"]["targets"][0] = dict(
        name="Reflector",
        mobility=dict(type="stationary", position_m=[0, 0, 0]),
        asset=dict(
            source="animated_asset",
            path=str(manifest),
            playback=dict(
                scenario_clock_id="scenario:temporal",
                motion_clock_id="motion:temporal",
                playback_rate=1.0,
                playback_offset_s=0.0,
            ),
        ),
    )
    path.write_text(yaml.safe_dump(config))
    return path.parent, manifest


def test_temporal_animation_preserves_exact_world_meshes_at_each_timestamp(temporal_animation):
    from shared.animated_target import (
        AnimatedTargetBindingContext,
        AnimatedTargetBindingTime,
        load_animated_target,
    )

    scenario, manifest = temporal_animation
    original = {p: p.read_bytes() for p in scenario.rglob("*") if p.is_file()}
    _, _, actors, radar, indices, times, sources = load_temporal_inputs(scenario)
    assert indices == (0, 1, 2)
    assert times == [0.0, 0.5, 1.0]
    np.testing.assert_array_equal(radar, [0, 0, 1])
    binding = load_animated_target(manifest).bind(
        AnimatedTargetBindingContext(
            "Reflector",
            "scenario:temporal",
            "motion:temporal",
            None,
            1.0,
            0.0,
            "affine_scenario_to_motion",
        ),
        [AnimatedTargetBindingTime(f"scenario:temporal:time:{t.hex()}", t, t) for t in times],
    )
    actor = actors[0]
    assert actor["animated"]
    assert actor["orientation_deg"] is None
    assert actor["extent_definition"] == "initial_world_aabb"
    for mesh, sample, affine, evidence in zip(
        actor["meshes"], binding.samples, actor["root_affines"], actor["animated_samples"]
    ):
        np.testing.assert_array_equal(mesh, sample.vertices_m[binding.faces])
        np.testing.assert_array_equal(affine, sample.resolved_root_affine_4x4)
        assert evidence == sample.evidence_dict()
    np.testing.assert_array_equal(
        actor["positions"], [s.resolved_root_affine_4x4[:3, 3] for s in binding.samples]
    )
    assert manifest in sources
    assert all(p.read_bytes() == data for p, data in original.items())
    assert not (scenario / ".orchav").exists()
    json.dumps(actor["animated_samples"])


def test_temporal_scene_uses_canonical_authored_transforms(authored):
    path, _ = authored
    (path.parent / "scene.xml").write_text(
        '<scene version="3.0.0"><shape type="ply" id="translated">'
        '<string name="filename" value="target/tri.ply"/>'
        '<transform name="to_world"><translate x="4" y="2" z="3"/></transform>'
        "</shape></scene>"
    )
    _, meshes, _, _, _, _, sources = load_temporal_inputs(path.parent)
    assert len(meshes) == 1
    assert meshes[0][0] == "translated"
    np.testing.assert_array_equal(meshes[0][1], [[3, 1, 3], [5, 1, 3], [4, 3, 3]])
    assert path.parent / "target/tri.ply" in sources
