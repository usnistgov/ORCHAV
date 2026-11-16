"""Configuration behavior and the progressive sensing lesson boundary."""

from pathlib import Path

import yaml

from generator.core.configuration.loader import resolve_sensing_document

ROOT = Path(__file__).resolve().parents[2]
CUBES = ROOT / "scenarios/sensing/cube_radar"


def config(name):
    return resolve_sensing_document(yaml.safe_load((CUBES / name / "scenario.yaml").read_text()))


def test_ordinary_tracking_is_configured_during_acquisition():
    for name in ("moving_tracking", "two_moving"):
        sensing = config(name)["sensing"]
        assert sensing["tracker_type"] == "kalman"
        assert sensing["track_confirmation_m"] == 2
        assert sensing["track_confirmation_n"] == 3
        assert sensing["track_association_range_gate_m"] == 0.5
        assert sensing["track_association_velocity_gate_m_s"] == 0.3
        assert sensing["persist_tracking_diagnostics"]
    assert len(config("two_moving")["actors"]["targets"]) == 2


def test_direction_is_introduced_before_cartesian_tracking():
    horizontal = yaml.safe_load((CUBES / "azimuth/scenario.yaml").read_text())
    planar = yaml.safe_load((CUBES / "azimuth/elevation/scenario.yaml").read_text())
    tracked = yaml.safe_load((CUBES / "moving_tracking_3d/scenario.yaml").read_text())
    assert horizontal["raytracing"]["antenna"]["rx"] == {"num_rows": 1, "num_cols": 16}
    assert "tracking" not in horizontal["sensing"]
    assert planar["raytracing"]["antenna"]["rx"] == {"num_rows": 8, "num_cols": 8}
    assert "tracking" not in planar["sensing"]
    assert tracked["sensing"]["tracking"]["type"] == "cartesian3d"
    for key in ("actors", "timeline", "raytracing"):
        assert planar[key] == tracked[key]
    for key in ("acquisition", "processing", "detection"):
        assert planar["sensing"][key] == tracked["sensing"][key]
    assert horizontal["sensing"]["detection"]["mode"] == "beam_scan"


def test_rigid_person_uses_one_bundled_pose_before_articulation():
    folder = ROOT / "scenarios/sensing/targets_motion/marching_person"
    rigid = yaml.safe_load((folder / "rigid/scenario.yaml").read_text())
    asset = rigid["actors"]["targets"][0]["asset"]
    assert asset["source"] == "directory"
    assert asset["switch_meshes"] is False
    assert (ROOT / asset["path"] / asset["pattern"]).is_file()
    assert resolve_sensing_document(rigid)["sensing"]["tracker_type"] == "kalman"
    for mode in ("coherent", "sequential"):
        animated = yaml.safe_load((folder / mode / "scenario.yaml").read_text())
        assert animated["actors"]["targets"][0]["asset"]["source"] == "animated_asset"


def test_window_comparison_retains_the_needed_complex_observations():
    assert config("moving")["sensing"]["recording_profile"] == "reprocessing"
    assert config("stationary")["sensing"]["recording_profile"] == "maps"


def test_paris_generation_includes_the_documented_standard_summaries():
    path = ROOT / "scenarios/showcases/paris_opera_multitarget/scenario.yaml"
    summary = yaml.safe_load(path.read_text())["generator_summary"]
    assert summary["enabled"]
    assert {"topology", "sensing"} <= set(summary["create"])
    assert {"range_doppler", "tracking_xyz"} <= set(summary["sensing"]["figures"])


def test_oscillation_has_two_distinct_acquisitions_with_equal_windows():
    coherent, sequential = (config("oscillating_" + mode) for mode in ("coherent", "sequential"))
    assert coherent["actors"] == sequential["actors"]
    assert coherent["timeline"] == sequential["timeline"]
    assert coherent["sensing"]["processing_mode"] == "coherent_cpi"
    assert sequential["sensing"]["processing_mode"] == "sequential_stft"
    assert coherent["sensing"]["fft_size_doppler"] == sequential["sensing"]["fft_size_doppler"]
