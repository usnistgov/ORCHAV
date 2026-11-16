"""Configuration and authored geometry checks; no fabricated scientific frames."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import yaml

from shared.scenarios import validate_scenario_data

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "visualizer_experiments"
NAMES = (
    "arrival_courtyard",
    "angular_corridor",
    "bistatic_range_locus",
    "monostatic_range_locus",
)


def _config(name: str) -> dict:
    return yaml.safe_load((EXAMPLES / name / "scenario.yaml").read_text(encoding="utf-8"))


def _position(config: dict, role: str, index: int = 0) -> np.ndarray:
    return np.asarray(config["actors"][role][index]["mobility"]["position_m"], dtype=float)


def _mesh_corners(name: str, mesh: str) -> np.ndarray:
    content = (EXAMPLES / name / "meshes" / f"{mesh}.ply").read_text(encoding="utf-8")
    header, vertices_and_faces = content.split("end_header\n")
    assert "element vertex 4" in header
    assert "element face 2" in header
    return np.asarray([row.split() for row in vertices_and_faces.splitlines()[:4]], dtype=float)


def _image_length(tx: np.ndarray, rx: np.ndarray, axis: int, coordinate: float) -> float:
    mirrored_tx = tx.copy()
    mirrored_tx[axis] = 2 * coordinate - tx[axis]
    return float(np.linalg.norm(mirrored_tx - rx))


@pytest.mark.parametrize("name", NAMES)
def test_experiment_configs_use_ordinary_local_generation(name):
    config = _config(name)
    scenario = validate_scenario_data(config)
    assert config["scene"] == {"source": "local", "id": "scene.xml"}
    assert config["data"]["mode"] == "files"
    assert config["data"]["files"]["directory"] == "frames"
    assert scenario.raytracing.export_path_metrics is True
    assert "rcs" not in config["raytracing"]
    assert not config["actors"].get("targets")
    quality = config["raytracing"]["quality"]["custom"]
    assert quality["max_depth"] == 1
    assert quality["specular_reflection"] is True
    assert quality["diffuse_reflection"] is False
    assert quality["refraction"] is False
    for role in ("tx", "rx"):
        assert all(actor["mobility"]["type"] == "stationary" for actor in config["actors"][role])


@pytest.mark.parametrize("name", NAMES)
def test_scene_assets_are_self_contained_small_meshes(name):
    scene = ElementTree.parse(EXAMPLES / name / "scene.xml").getroot()
    material = scene.find("bsdf")
    assert material.attrib["type"] == "itu-radio-material"
    assert material.find("string").attrib == {"name": "type", "value": "metal"}
    for shape in scene.findall("shape"):
        assert shape.attrib["type"] == "ply"
        filename = Path(shape.find("string").attrib["value"])
        assert not filename.is_absolute() and ".." not in filename.parts
        mesh = EXAMPLES / name / filename
        assert mesh.is_file() and mesh.stat().st_size < 1000
        assert _mesh_corners(name, mesh.stem).shape == (4, 3)


def test_checked_in_inputs_match_procedural_builder(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "visual_experiment_builder", EXAMPLES / "build_scenarios.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(module.SCENARIO_NAMES) == NAMES
    module.build_scenarios(tmp_path)
    for name in NAMES:
        for generated in (tmp_path / name).rglob("*"):
            if generated.is_file():
                assert (
                    generated.read_bytes()
                    == (EXAMPLES / name / generated.relative_to(tmp_path / name)).read_bytes()
                )


def test_arrival_geometry_has_separated_delays_and_second_pair():
    config = _config("arrival_courtyard")
    tx, rx = _position(config, "tx"), _position(config, "rx")
    near = _mesh_corners("arrival_courtyard", "near_wall")
    far = _mesh_corners("arrival_courtyard", "far_wall")
    assert np.all(near[:, 1] == 3) and np.all(far[:, 1] == -6)
    assert np.linalg.norm(tx - rx) == pytest.approx(8)
    assert _image_length(tx, rx, 1, 3) == pytest.approx(10)
    assert _image_length(tx, rx, 1, -6) == pytest.approx(math.sqrt(208))
    offset = _position(config, "rx", 1)
    assert offset[1] == 2
    assert np.linalg.norm(tx - offset) == pytest.approx(math.sqrt(68))
    assert _image_length(tx, offset, 1, 3) == pytest.approx(math.sqrt(80))
    assert _image_length(tx, offset, 1, -6) == pytest.approx(math.sqrt(260))


def test_corridor_equal_delay_reflections_have_distinct_directions():
    config = _config("angular_corridor")
    tx, rx = _position(config, "tx"), _position(config, "rx")
    for wall, y in (("north_wall", 3), ("south_wall", -3)):
        assert np.all(_mesh_corners("angular_corridor", wall)[:, 1] == y)
        assert _image_length(tx, rx, 1, y) == pytest.approx(10)
    assert np.all(_mesh_corners("angular_corridor", "end_wall")[:, 0] == 10)
    assert _image_length(tx, rx, 0, 10) == pytest.approx(20)
    directions = [np.array([0, y, 1.5]) - rx for y in (3, -3)]
    azimuths = [math.degrees(math.atan2(v[1], v[0])) % 360 for v in directions]
    assert azimuths == pytest.approx([143.130102354, 216.869897646])


@pytest.mark.parametrize("name", NAMES[2:])
def test_range_examples_request_single_static_coherent_cpi(name):
    config = _config(name)
    rt, sensing = config["raytracing"], config["sensing"]
    assert len(config["actors"]["tx"]) == len(config["actors"]["rx"]) == 1
    assert config["timeline"] == {"steps": 8, "sample_rate_hz": 1000.0}
    assert rt["cir_time_steps"] == sensing["fft_size_doppler"] == 8
    assert rt["cir_sampling_frequency_hz"] == sensing["prf_hz"] == 1000.0
    assert sensing["processing_mode"] == "coherent_cpi"
    assert sensing["range_response"]["model"] == "cfr_ifft"
    assert sensing["range_response"]["num_frequency_bins"] == sensing["fft_size_range"] == 128
    assert sensing["bandwidth"] == rt["bandwidth_hz"] == 300e6
    assert sensing["noise_enabled"] is False
    assert sensing["output_detections"] is True
    assert sensing["persist_detection_stage_artifacts"] is True
    assert rt["quality"]["custom"]["los"] is False
    assert np.all(_mesh_corners(name, "reflecting_wall")[:, 1] == 8)
    length = _image_length(_position(config, "tx"), _position(config, "rx"), 1, 8)
    if name == "monostatic_range_locus":
        assert sensing["range_mode"] == "monostatic"
        np.testing.assert_array_equal(_position(config, "tx"), _position(config, "rx"))
        assert length == pytest.approx(16)
        assert length / 2 == pytest.approx(8)
    else:
        assert sensing["range_mode"] == "bistatic"
        assert length == pytest.approx(2 * math.sqrt(73))
