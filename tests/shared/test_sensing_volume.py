"""Conditional outcomes must not become probabilities or fabricated misses."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.volume.reader import SensingVolume


@pytest.fixture
def volume_data():
    trial = {
        "id": "a",
        "index_zyx": [0, 0, 0],
        "heading": 0,
        "position_m": [10, 20, 30],
        "velocity_m_s": [0, 0, 0],
        "status": "measured",
        "hit": True,
        "tp": 1,
        "fn": 0,
        "fp": 2,
        "radial_closing_velocity_m_s": 0,
        "matched_error_m": 1,
        "native": "native",
        "configuration_sha256": "not_yet_verified",
        "source_hashes": {"frames.h5": "not_yet_verified"},
    }
    return {
        "schema_version": 1,
        "kind": "conditional_detection_outcomes",
        "axes_m": {"x": [10, 20], "y": [20, 40], "z": [30, 60]},
        "world_frame": {
            "id": "test",
            "origin": "test origin",
            "axes": ["x", "y", "z"],
            "handedness": "right",
            "length_unit": "m",
        },
        "headings": [{"name": "Hover", "velocity_m_s": [0, 0, 0]}],
        "sensor_position_m": [0, 0, 0],
        "position_gate_m": 5,
        "scene_scenario": "scenario.yaml",
        "trials": [trial],
    }


def write_volume(tmp_path, data):
    path = tmp_path / "sensing_volume.json"
    path.write_text(json.dumps(data))
    return path


def test_unknown_and_excluded_cells_remain_unknown(tmp_path, volume_data):
    miss = deepcopy(volume_data["trials"][0])
    miss.update(
        id="b",
        index_zyx=[0, 0, 1],
        position_m=[20, 20, 30],
        hit=False,
        tp=0,
        fn=1,
        matched_error_m=None,
    )
    excluded = deepcopy(miss)
    excluded.update(
        id="c", index_zyx=[0, 1, 0], position_m=[10, 40, 30], status="excluded_building"
    )
    volume_data["trials"] += [miss, excluded]
    reader = SensingVolume(write_volume(tmp_path, volume_data))
    grid = reader.outcome_grid(0)
    assert grid[0, 0, 0] == 1 and grid[0, 0, 1] == 0
    assert np.isnan(grid[0, 1, 0]) and np.isnan(grid[1]).all()
    assert np.isfinite(grid).sum() == 2
    assert not grid.flags.writeable
    volume_data["truth_for_evaluation"] = {"irrelevant": "changed truth is not a display input"}
    other = SensingVolume(write_volume(tmp_path, volume_data))
    np.testing.assert_array_equal(other.outcome_grid(0), grid)


def test_geometry_can_disagree_with_detection_and_preserves_exclusions(tmp_path, volume_data):
    trial = volume_data["trials"][0]
    trial["direct_path_blocked"] = True
    reader = SensingVolume(write_volume(tmp_path, volume_data))
    assert reader.outcome_grid(0)[0, 0, 0] == 1
    assert reader.geometry_grid(0)[0, 0, 0] == 1
    trial["status"] = "pending"
    reader = SensingVolume(write_volume(tmp_path, volume_data))
    assert np.isnan(reader.outcome_grid(0)).all()
    assert reader.geometry_grid(0)[0, 0, 0] == 1
    trial["status"] = "excluded_geometry"
    reader = SensingVolume(write_volume(tmp_path, volume_data))
    assert np.isnan(reader.geometry_grid(0)).all()
    trial["direct_path_blocked"] = 0.5
    with pytest.raises(ValueError, match="Direct-path geometry"):
        SensingVolume(write_volume(tmp_path, volume_data))


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "outside",
        "position",
        "probability",
        "counts",
        "error",
        "radial",
        "world",
        "axis",
    ],
)
def test_invalid_volume_refused_before_rendering(tmp_path, volume_data, mutation):
    trial = volume_data["trials"][0]
    if mutation == "duplicate":
        volume_data["trials"].append(deepcopy(trial))
    elif mutation == "outside":
        trial["index_zyx"][0] = 99
    elif mutation == "position":
        trial["position_m"][0] += 1
    elif mutation == "probability":
        trial["hit"] = 0.8
    elif mutation == "counts":
        trial["fn"] = 1
    elif mutation == "error":
        trial["matched_error_m"] = 6
    elif mutation == "radial":
        trial["radial_closing_velocity_m_s"] = 1
    elif mutation == "world":
        volume_data.pop("world_frame")
    else:
        volume_data["axes_m"]["x"] = [10, 20, 99]
    with pytest.raises(ValueError):
        SensingVolume(write_volume(tmp_path, volume_data))


def test_scene_and_native_hashes_detect_replacement(tmp_path, volume_data):
    from shared.sensing.fusion.experiment import fingerprints

    (tmp_path / "scenario.yaml").write_text("scene: test")
    volume_data["scene_configuration_sha256"] = hashlib.sha256(
        (tmp_path / "scenario.yaml").read_bytes()
    ).hexdigest()
    native = tmp_path / "native"
    frames = native / "frames"
    frames.mkdir(parents=True)
    (frames / "frames_manifest.json").write_text("{}")
    (native / "scenario.yaml").write_text("trial: test")
    trial = volume_data["trials"][0]
    trial["source_hashes"] = fingerprints(frames)
    trial["configuration_sha256"] = hashlib.sha256(
        (native / "scenario.yaml").read_bytes()
    ).hexdigest()
    reader = SensingVolume(write_volume(tmp_path, volume_data))
    assert reader.verify_scene() == tmp_path / "scenario.yaml"
    assert reader.verify_native((0, 0, 0, 0)) == native
    (frames / "frames_manifest.json").write_text('{"modified": true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.verify_native((0, 0, 0, 0))
    (tmp_path / "scenario.yaml").write_text("different scene")
    with pytest.raises(ValueError, match="world binding"):
        reader.verify_scene()


@pytest.mark.parametrize(
    "assets",
    [
        {},
        [],
        {"../outside.ply": "a" * 64},
        {"/outside.ply": "a" * 64},
        {"meshes/arc.ply": "invalid digest"},
    ],
)
def test_invalid_scene_asset_bindings_rejected(tmp_path, volume_data, assets):
    volume_data["scene_asset_hashes"] = assets
    with pytest.raises(ValueError, match="Scene asset bindings"):
        SensingVolume(write_volume(tmp_path, volume_data))


def test_different_cells_cannot_share_a_trial_identity(tmp_path, volume_data):
    other = deepcopy(volume_data["trials"][0])
    other.update(index_zyx=[0, 0, 1], position_m=[20, 20, 30])
    volume_data["trials"].append(other)
    with pytest.raises(ValueError, match="Trial IDs"):
        SensingVolume(write_volume(tmp_path, volume_data))


def test_native_availability_checks_inventory_without_reading_hdf5(tmp_path, volume_data):
    native = tmp_path / "native"
    native.mkdir()
    (native / "scenario.yaml").write_text("{}")
    reader = SensingVolume(write_volume(tmp_path, volume_data))
    key = (0, 0, 0, 0)
    assert "manifest missing" in reader.native_availability(key)
    frames = native / "frames"
    frames.mkdir()
    (frames / "frames_manifest.json").write_text("{}")
    assert "frame files missing" in reader.native_availability(key)
    # Deliberately not valid HDF5: an availability query must only stat it.
    (frames / "frames.h5").write_bytes(b"Only verification on opening reads evidence")
    assert reader.native_availability(key) == "available"
