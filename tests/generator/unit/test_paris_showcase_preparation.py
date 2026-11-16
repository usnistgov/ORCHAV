"""Portable Paris inputs and source-neutral walking preparation contracts."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scenarios.showcases.paris_opera_multitarget.scripts.walking import (
    WALK_START,
    WALK_V,
    prepare_walking,
)
from shared.animated_target import (
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    load_animated_target,
)
from shared.animated_target_export import export_animated_target


@pytest.mark.parametrize(
    ("option", "existing_directory", "message"),
    [
        ("--walking-asset", False, "existing animated_target.json file"),
        ("--walking-asset", True, "existing animated_target.json file"),
        ("--walk-library", False, "existing walking library directory"),
        ("--walk-library", True, "mesh_library.json"),
    ],
)
def test_prepare_cli_rejects_missing_walking_input_before_creating_output(
    tmp_path, option, existing_directory, message
):
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "walking-input"
    if existing_directory:
        source.mkdir()
    output = tmp_path / "uncreated" / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scenarios/showcases/paris_opera_multitarget/prepare.py"),
            option,
            str(source),
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert f"error: {option}" in completed.stderr
    assert message in completed.stderr
    assert str(source) in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""
    assert not output.parent.exists()


@pytest.fixture
def supplied_motion(tmp_path):
    poses = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, height]] for height in (1.0, 1.5, 2.0)]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    manifest = export_animated_target(
        tmp_path / "source",
        asset_id="test:nonhuman",
        vertices=poses,
        faces=faces,
        times_s=[0.0, 0.25, 0.5],
        root_translation=np.full((3, 3), 100.0),
        root_rotation_xyzw=np.tile([0, 0, 0, 1], (3, 1)),
        source_type="synthetic",
        source_records={
            "numeric_fixture": {
                "sha256": hashlib.sha256(poses.tobytes()).hexdigest(),
                "size_bytes": poses.nbytes,
            }
        },
        adapter_metadata={},
        front_axis="+X",
        up_axis="+Z",
    )
    return manifest, poses


def test_walking_rejects_implicit_loop_and_period_that_drops_last_slot(tmp_path, supplied_motion):
    source, _ = supplied_motion
    with pytest.raises(ValueError, match="too short"):
        prepare_walking(source, tmp_path / "short", 1.4)
    with pytest.raises(ValueError, match="greater than"):
        prepare_walking(source, tmp_path / "period", 1.4, pose_period_s=0.5)
    assert not (tmp_path / "short").exists()


def test_walking_preserves_all_poses_and_continuous_route_across_wraps(tmp_path, supplied_motion):
    source, poses = supplied_motion
    original = source.read_bytes()
    output = prepare_walking(source, tmp_path / "walk", 1.4, pose_period_s=0.75)
    asset = load_animated_target(output)
    mapping = json.loads((output.parent / "pose_mapping.json").read_text())
    assert [row["source_time_s"] for row in mapping] == [0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 0.5]
    assert [row["cycle"] for row in mapping] == [0, 0, 0, 1, 1, 1, 1]
    context = AnimatedTargetBindingContext(
        "walk", "scenario", "motion", None, 1.0, 0.0, "affine_scenario_to_motion"
    )
    times = [0.0, 0.499, 0.5, 0.749, 0.75, 0.751, 1.4]
    binding = asset.bind(
        context, [AnimatedTargetBindingTime(f"time:{t.hex()}", t, t) for i, t in enumerate(times)]
    )
    yaw = np.arctan2(WALK_V[1], WALK_V[0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    for sample, time, index in zip(binding.samples, times, [0, 1, 2, 2, 0, 0, 2]):
        expected = poses[index] @ rotation.T + WALK_START + time * WALK_V
        np.testing.assert_allclose(sample.vertices_m, expected, atol=1e-12)
    assert source.read_bytes() == original


def test_walking_preserves_no_repeat_and_explicit_offset(tmp_path, supplied_motion):
    source, poses = supplied_motion
    output = prepare_walking(source, tmp_path / "walk", 0.4, offset_m=(0, 0, 1))
    asset = load_animated_target(output)
    np.testing.assert_array_equal(asset.pose_samples[0].vertices, poses[0])
    np.testing.assert_allclose(asset.root_samples[0].translation, WALK_START + [0, 0, 1])
    assert asset.pose_out_of_range == "error"


def test_paris_template_has_full_acquisition_and_portable_actor_bindings():
    import yaml

    root = Path(__file__).resolve().parents[3]
    template = yaml.safe_load(
        (root / "scenarios/showcases/paris_opera_multitarget/scenario.yaml").read_text()
    )
    assert template["timeline"] == {"steps": 10240, "sample_rate_hz": 2000.0}
    assert template["raytracing"]["quality"]["custom"]["samples_per_src"] == 800000
    assert template["sensing"]["recording"]["profile"] == "reprocessing"
    assert len(template["actors"]["targets"]) == 5
    for target in template["actors"]["targets"]:
        assert not Path(target["asset"]["path"]).is_absolute()
        assert "runs/" not in target["asset"]["path"]
