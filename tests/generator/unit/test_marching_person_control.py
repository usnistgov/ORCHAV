"""The articulated comparison keeps clocks and channel settings controlled."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from generator.core.configuration.loader import resolve_sensing_document
from generator.core.scenario_actors.animated_asset import prepare_animated_asset
from generator.core.scenario_actors.types import Timeline
from scenarios.sensing.targets_motion.marching_person.prepare import _linear_root_motion, prepare
from shared.animated_target_export import export_animated_target, source_affine
from shared.scenarios import validate_scenario_data
from shared.scenarios.actors import TargetActorSpec

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "scenarios/sensing/targets_motion/marching_person"


def test_matched_articulation_acquisitions_keep_scene_and_processing_equal():
    documents = [
        yaml.safe_load((FAMILY / mode / "scenario.yaml").read_text())
        for mode in ("coherent", "sequential")
    ]
    models = [validate_scenario_data(data) for data in documents]
    for key in ("scene", "actors", "timeline", "raytracing"):
        assert documents[0][key] == documents[1][key]
    sensing = [resolve_sensing_document(data)["sensing"] for data in documents]
    for key in (
        "bandwidth",
        "prf_hz",
        "fft_size_range",
        "fft_size_doppler",
        "window_doppler",
        "range_response",
        "coherent_clutter_mode",
    ):
        assert sensing[0][key] == sensing[1][key]
    assert sensing[0]["prf_hz"] == 4000
    assert sensing[0]["fft_size_doppler"] == 256
    assert sensing[0]["range_response"]["model"] == "cfr_ifft"
    assert models[0].timeline.steps == models[1].timeline.steps == 8192
    times = np.arange(8192) / 4000
    runtime_times = np.asarray(Timeline(8192, 8191 / 4000).timestamps_s)
    # Step-selected explicit surfaces must never fall back to the preceding
    # pose because the scenario and preparation clocks round differently.
    np.testing.assert_array_equal(
        np.searchsorted(times, runtime_times, side="right") - 1, np.arange(8192)
    )
    coherent = times.reshape(32, 256)
    sequential = np.lib.stride_tricks.sliding_window_view(times, 256)
    assert sequential.shape == (7937, 256)
    np.testing.assert_array_equal(coherent, sequential[::256])
    assert np.mean(coherent[0]) == pytest.approx(0.031875)
    assert np.mean(coherent[-1]) == pytest.approx(2.015875)


def test_untrusted_model_is_rejected_before_creating_outputs(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="trusted-model"):
        prepare(tmp_path / "absent.npz", tmp_path / "models", output)
    assert not output.exists()


def test_preparation_does_not_extend_or_loop_a_short_source(tmp_path):
    motion = tmp_path / "motion.npz"
    np.savez(
        motion,
        mocap_framerate=120.0,
        gender="male",
        poses=np.zeros((2, 156)),
        trans=np.zeros((2, 3)),
    )
    model = tmp_path / "models/smpl/SMPL_MALE.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"never deserialized")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="beyond the source"):
        prepare(motion, model.parents[1], output, trusted_model=True)
    assert not output.exists()


def test_approach_world_trajectory_has_no_wrap_or_double_translation():
    times = np.arange(8192) / 4000
    roots = _linear_root_motion(
        times, floor_offset_m=1.1, approach_speed_m_s=1.4, start_range_m=8.0
    )
    basis = source_affine("-Y", "+Z")[:3, :3]
    world = roots @ basis.T
    np.testing.assert_allclose(world[:, 0], 8.0 - 1.4 * times, atol=1e-14)
    np.testing.assert_array_equal(world[:, 1], 0.0)
    np.testing.assert_array_equal(world[:, 2], 1.1)
    assert world[-1, 0] == pytest.approx(5.13315)
    np.testing.assert_allclose(np.diff(world[:, 0]), -1.4 / 4000, atol=1e-14)


@pytest.mark.parametrize("sample_rate", [50.0, 119.995, 120.005, 60.0, 120.0, 4000.0])
def test_preparation_requires_evaluated_pose_times_to_match_declared_pulses(
    tmp_path, monkeypatch, sample_rate
):
    from generator.tools import smpl_to_ply

    motion = tmp_path / "motion.npz"
    np.savez(
        motion,
        mocap_framerate=120.0,
        gender="male",
        poses=np.zeros((300, 156)),
        trans=np.ones((300, 3)),
    )
    model = tmp_path / "models/smpl/SMPL_MALE.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"never deserialized")
    output = tmp_path / "output"
    evaluated = []

    def stop_before_model_evaluation(argv):
        evaluated.append(argv)
        raise RuntimeError("validated grid reached evaluator")

    monkeypatch.setattr(smpl_to_ply, "main", stop_before_model_evaluation)
    valid_grid = sample_rate in (60.0, 120.0, 4000.0)
    error = RuntimeError if valid_grid else ValueError
    message = "validated grid reached evaluator" if valid_grid else "uniform pulse times"
    with pytest.raises(error, match=message):
        prepare(
            motion,
            model.parents[1],
            output,
            samples=8192 if sample_rate == 4000.0 else 4,
            sample_rate_hz=sample_rate,
            trusted_model=True,
        )
    assert bool(evaluated) == valid_grid
    assert output.exists() == valid_grid
    # Neither rejection nor successful preflight mutates the supplied motion.
    with np.load(motion, allow_pickle=False) as original:
        np.testing.assert_array_equal(original["trans"], np.ones((300, 3)))


def test_articulating_geometry_keeps_manifest_root_velocity_and_single_transform(tmp_path):
    times = np.array([0.0, 0.2, 0.4])
    poses = np.repeat([[[-0.3, 0, -1], [0.3, 0, -1], [0.3, 0, 1], [-0.3, 0, 1]]], 3, axis=0).astype(
        float
    )
    # Change the surface's bounding center without changing the root trajectory.
    poses[1, 2, 1] = 0.2
    poses[2, 2, 1] = -0.3
    poses[2, :2, 2] -= 0.05
    roots = _linear_root_motion(
        times, floor_offset_m=-poses[0, :, 2].min(), approach_speed_m_s=1.4, start_range_m=8
    )
    manifest = export_animated_target(
        tmp_path / "asset",
        asset_id="test:marching:linear_approach",
        vertices=poses,
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        times_s=times,
        root_translation=roots,
        root_rotation_xyzw=np.repeat([[0, 0, 1, 0]], 3, axis=0),
        source_type="synthetic_test",
        source_records={
            "synthetic_pose_array": {
                "sha256": hashlib.sha256(poses.tobytes()).hexdigest(),
                "size_bytes": poses.nbytes,
            }
        },
        adapter_metadata={},
        front_axis="-Y",
        up_axis="+Z",
        out_of_range="error",
    )
    actor = TargetActorSpec.model_validate(
        {
            "name": "Person",
            "mobility": {"type": "stationary", "position_m": [0, 0, 0]},
            "asset": {
                "source": "animated_asset",
                "path": str(manifest),
                "playback": {
                    "scenario_clock_id": "scenario:test",
                    "motion_clock_id": "motion:test",
                    "playback_rate": 1,
                    "playback_offset_s": 0,
                },
            },
        }
    )
    prepared = prepare_animated_asset(
        actor.asset, Timeline(3, 0.4), target_name="Person", base_dir=tmp_path
    )
    for index, sample in enumerate(prepared.binding.samples):
        # Independent composition: the 180-degree facing rotation and -Y-front
        # basis map local (x,y,z) to (y,-x,z), then apply the root exactly once.
        expected = poses[index][:, [1, 0, 2]] * [1, -1, 1]
        expected += [8 - 1.4 * times[index], 0, 1]
        np.testing.assert_allclose(sample.vertices_m, expected, atol=1e-14)
    np.testing.assert_allclose(prepared.root_velocities_mps, [[-1.4, 0, 0]] * 3, atol=1e-14)
    # One initial floor offset preserves subsequent vertical articulation.
    assert prepared.binding.samples[2].vertices_m[:, 2].min() == pytest.approx(-0.05)
    assert not np.allclose(np.diff(np.asarray(prepared.centers_m)[:, 0]) / 0.2, -1.4)


@pytest.mark.parametrize("speed,start", [(float("nan"), 8), (-1, 8), (1, 0), (4, 1)])
def test_invalid_or_radar_crossing_root_trajectory_is_rejected(speed, start):
    with pytest.raises(ValueError):
        _linear_root_motion(
            np.array([0.0, 0.5]),
            floor_offset_m=1.1,
            approach_speed_m_s=speed,
            start_range_m=start,
        )
