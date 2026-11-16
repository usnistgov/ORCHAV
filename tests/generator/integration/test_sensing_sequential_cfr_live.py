"""Controlled CUDA phase checks for a face-on, specular-only moving cube.

These checks qualify this rigid control only. They make no coherence claim
about arbitrary diffuse scattering, changing path topology, or articulated meshes.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ORCHAV_RUN_LIVE_SEQUENTIAL_CFR") != "1",
    reason="Set ORCHAV_RUN_LIVE_SEQUENTIAL_CFR=1 for the controlled CUDA phase checks",
)


@pytest.fixture(scope="module")
def cuda_backend():
    import mitsuba as mi

    # Must precede Sionna import; LLVM is not evidence for this CUDA check.
    mi.set_variant("cuda_ad_mono_polarized")
    import sionna.rt as rt

    assert mi.variant() == "cuda_ad_mono_polarized"
    print(
        f"selected_mitsuba_variant={mi.variant()}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    return rt


@pytest.mark.parametrize("seed", [7, 42, 113])
@pytest.mark.parametrize("ray_budget", [1000, 10000])
def test_cuda_independent_specular_pulses_keep_carrier_phase(cuda_backend, seed, ray_budget):
    from generator.core.configuration import SensingConfig
    from generator.core.sensing.acquisition import resolve_sensing_execution_values
    from generator.core.sensing.processor import create_sensing_processor
    from shared.sensing.range_response import RangeResponseConfig

    rt = cuda_backend
    carrier = 28e9
    bandwidth = 500e6
    prf = 1000.0
    count = 9
    scene = rt.load_scene()
    scene.frequency = carrier
    for name in ("tx_array", "rx_array"):
        setattr(
            scene,
            name,
            rt.PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="V",
            ),
        )
    scene.add(rt.Transmitter("tx", position=[0, 0, 0.5]))
    scene.add(rt.Receiver("rx", position=[0, 0, 0.5]))
    cube = rt.SceneObject(
        fname=str(Path(__file__).resolve().parents[3] / "libraries/targets/cube/cube.ply"),
        name="cube",
        radio_material=rt.ITURadioMaterial("metal", "metal", thickness=0.01),
    )
    scene.edit(add=cube)
    solver = rt.PathSolver()
    config = SensingConfig(
        enabled=True,
        processing_mode="sequential_stft",
        bandwidth=bandwidth,
        prf_hz=prf,
        fft_size_range=128,
        fft_size_doppler=8,
        window_range="boxcar",
        window_doppler="hann",
        range_mode="monostatic",
        range_response=RangeResponseConfig("cfr_ifft", 128, "hann"),
        coherent_clutter_mode="none",
        raw_payload_policy="always",
        output_detections=False,
        output_range_profile=True,
        output_range_doppler=True,
    )
    execution = resolve_sensing_execution_values(
        config, cir_time_steps=1, cir_sampling_frequency_hz=None, frame_dt_s=1 / prf
    )
    for motion in ("static", "oscillating"):
        processor = create_sensing_processor(config, carrier, execution)
        delays = []
        normalized_gains = []
        expected_observations = []
        for index in range(count):
            time_s = index / prf
            displacement = 0.15 * np.sin(2 * np.pi * 2 * time_s) if motion == "oscillating" else 0.0
            speed = (
                -0.15 * 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * time_s)
                if motion == "oscillating"
                else 0.0
            )
            cube.position = [0, float(5 - displacement), 0.5]
            cube.velocity = [0, float(speed), 0]
            paths = solver(
                scene,
                max_depth=1,
                samples_per_src=ray_budget,
                seed=seed,
                los=False,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=False,
            )
            a, tau = paths.cir(
                num_time_steps=1, sampling_frequency=prf, normalize_delays=False, out_type="numpy"
            )
            valid = tau[0, 0] >= 0
            assert np.count_nonzero(valid) == 1, (motion, index, seed, ray_budget)
            delay = float(tau[0, 0, valid][0])
            coefficient = complex(a[0, 0, 0, 0, valid, 0][0])
            passband = complex(
                np.asarray(paths.a[0])[0, 0, 0, 0, valid][0]
                + 1j * np.asarray(paths.a[1])[0, 0, 0, 0, valid][0]
            )
            normalized_gains.append(coefficient / passband)
            delays.append(delay)
            frequencies = (np.arange(128) - 64) * bandwidth / 128
            expected_observations.append(coefficient * np.exp(-2j * np.pi * frequencies * delay))
            result = processor.process(None, index, paths, source_time_s=time_s)
            assert result is not None
            np.testing.assert_allclose(
                result["cfr_by_pair_rx_element"][0, 0, -1],
                expected_observations[-1],
                atol=1e-11,
                rtol=2e-5,
            )
        normalized_gains = np.asarray(normalized_gains)
        predicted = np.exp(-2j * np.pi * carrier * np.asarray(delays))
        error_rad = np.angle(normalized_gains * predicted.conj())
        step_error_rad = np.angle(
            normalized_gains[1:]
            * normalized_gains[:-1].conj()
            * np.exp(2j * np.pi * carrier * np.diff(delays))
        )
        assert np.max(abs(error_rad)) < 5e-4
        assert np.max(abs(step_error_rad)) < 8e-4
        if motion == "static":
            np.testing.assert_array_equal(delays, np.full(count, delays[0]))
            np.testing.assert_array_equal(normalized_gains, np.full(count, normalized_gains[0]))
        else:
            assert np.all(np.diff(delays) < 0)
        np.testing.assert_allclose(
            result["cfr_by_pair_rx_element"][0, 0],
            expected_observations[-8:],
            atol=1e-11,
            rtol=2e-5,
        )
        print(
            json.dumps(
                {
                    "motion": motion,
                    "seed": seed,
                    "samples_per_src": ray_budget,
                    "pulses": count,
                    "max_carrier_phase_error_rad": float(np.max(abs(error_rad))),
                    "max_pulse_phase_step_error_rad": float(np.max(abs(step_error_rad))),
                    "variant": "cuda_ad_mono_polarized",
                    "scattering": "specular_only_face_on_cube",
                },
                sort_keys=True,
            )
        )
