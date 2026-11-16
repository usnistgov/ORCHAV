"""Live mesh-only geometry checks; no RT 2.2 RCS target dependencies."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from generator.core.configuration.loader import load_simulation_config, resolve_sensing_document
from shared.scenarios.model import ScenarioModel
from shared.sensing.range_response import C0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("ORCHAV_RUN_LIVE_CFR") != "1",
        reason="Set ORCHAV_RUN_LIVE_CFR=1 for live Sionna geometry checks",
    ),
]
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "scenario,cpi",
    [("stationary_specular_cube", 0)]
    + [(s, i) for s in ("moving_specular_cube", "static_and_moving_cubes") for i in range(3)],
)
def test_authored_specular_faces_return_at_every_cpi(scenario, cpi):
    import drjit as dr
    import sionna.rt as rt

    dr.set_thread_count(2)
    cfg = yaml.safe_load(
        (ROOT / "scenarios/sensing/foundations" / scenario / "scenario.yaml").read_text()
    )
    cfg = resolve_sensing_document(ScenarioModel.model_validate(cfg).model_dump(exclude_none=True))
    assert cfg["sensing"]["coherent_clutter_mode"] == "none"
    assert not cfg["raytracing"]["quality"]["custom"].get("diffuse_reflection", False)
    scene = rt.load_scene(str(ROOT / "libraries/scenes/empty/empty.xml"))
    scene.frequency = load_simulation_config(SimpleNamespace(**cfg)).carrier_frequency_hz
    for key in ("tx_array", "rx_array"):
        setattr(
            scene,
            key,
            rt.PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="V",
            ),
        )
    radar = cfg["actors"]["tx"][0]["mobility"]["position_m"]
    scene.add(rt.Transmitter("tx", position=radar))
    scene.add(rt.Receiver("rx", position=radar))
    expected = []
    for target in cfg["actors"]["targets"]:
        mobility = target["mobility"]
        velocity = np.zeros(3)
        if mobility["type"] == "linear":
            position = np.asarray(mobility["start_m"], float)
            velocity = (np.asarray(mobility["end_m"]) - position) / (
                (cfg["timeline"]["steps"] - 1) / cfg["timeline"]["sample_rate_hz"]
            )
            position = position + velocity * cpi * 0.064
        else:
            position = np.asarray(mobility["position_m"], float)
        obj = rt.SceneObject(
            name=target["name"],
            fname=str(ROOT / target["asset"]["path"] / target["asset"]["pattern"]),
            radio_material=rt.ITURadioMaterial(target["name"] + "_metal", "metal", thickness=0.01),
        )
        scene.edit(add=obj)
        obj.position = position.tolist()
        obj.velocity = velocity.tolist()
        obj.orientation = [float(np.deg2rad(target["orientation"]["yaw_deg"])), 0, 0]
        offset = position - radar
        distance = np.linalg.norm(offset)
        expected.append((distance - 0.5, -np.dot(velocity, offset) / distance))
    paths = rt.PathSolver()(scene, **cfg["raytracing"]["quality"]["custom"])
    a, tau = paths.cir(
        sampling_frequency=1000, num_time_steps=64, normalize_delays=False, out_type="numpy"
    )
    valid = tau[0, 0] >= 0
    assert valid.sum() == len(expected)
    ranges = tau[0, 0, valid] * C0 / 2
    signals = a[0, 0, 0, 0, valid]
    order = np.argsort(ranges)
    for index, (distance, speed) in zip(order, sorted(expected)):
        assert ranges[index] == pytest.approx(distance, abs=3e-6)
        tone = np.exp(2j * np.pi * (2 * speed * 28e9 / C0) * np.arange(64) / 1000)
        np.testing.assert_allclose(signals[index] / signals[index, 0], tone, atol=4e-5, rtol=0)
