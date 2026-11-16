"""T24: actual installed Sionna backend parity, never replaced by a mock."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ORCHAV_RUN_LIVE_CFR") != "1",
    reason="Set ORCHAV_RUN_LIVE_CFR=1 to execute the installed Sionna backend",
)


@pytest.mark.parametrize(
    "rows,columns,spacing",
    [(1, 1, 0.5), (1, 4, 0.5), (2, 2, 0.5), (4, 4, 0.25), (4, 4, 0.4), (1, 16, 0.5), (4, 8, 0.5)],
)
def test_T24_live_sionna_cfr(rows, columns, spacing):
    import mitsuba as mi
    import sionna.rt as rt

    from shared.sensing.range_response import (
        RangeResponseConfig,
        frequency_grid,
        paths_to_frequency,
    )

    scene = rt.load_scene()
    scene.frequency = 28e9
    scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=rows,
        num_cols=columns,
        vertical_spacing=spacing,
        horizontal_spacing=spacing,
        pattern="iso",
        polarization="V",
    )
    scene.add(rt.Transmitter("tx", position=[0, 0, 1]))
    scene.add(rt.Receiver("rx", position=[10, 2, 1], velocity=[0.1, 0, 0]))
    paths = rt.PathSolver()(scene, max_depth=1, samples_per_src=100, synthetic_array=True)
    a, tau = paths.cir(
        sampling_frequency=1000, num_time_steps=8, normalize_delays=False, out_type="numpy"
    )
    cfg = RangeResponseConfig("cfr_ifft", 32)
    expected = paths.cfr(
        mi.Float(frequency_grid(100e6, 32)),
        sampling_frequency=1000,
        num_time_steps=8,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )
    observed, _ = paths_to_frequency(
        a[0, :, 0, 0], tau[0, 0], tau[0, 0] >= 0, np.arange(8) / 1000, bandwidth=100e6, config=cfg
    )
    np.testing.assert_allclose(observed, expected[0, :, 0, 0], rtol=2e-5, atol=1e-10)


def test_live_monostatic_cube_round_trip_phase_and_intra_cpi_coherence():
    """A specular front face isolates factor two from diffuse interference.

    This is a live backend diagnostic, not a replacement for the recorded
    diffuse cube. Check its frozen-delay/carrier-Doppler contract explicitly.
    """
    import sionna.rt as rt

    from shared.sensing.range_response import C0, RangeResponseConfig, paths_to_frequency

    scene = rt.load_scene()
    scene.frequency = 28e9
    scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.add(rt.Transmitter("tx", position=[0, 0, 0]))
    scene.add(rt.Receiver("rx", position=[0, 0, 0]))
    mesh = Path(__file__).resolve().parents[3] / "libraries/targets/cube/cube.ply"
    cube = rt.SceneObject(
        fname=str(mesh),
        name="cube",
        radio_material=rt.ITURadioMaterial("metal", "metal", thickness=0.01),
    )
    scene.edit(add=cube)
    cube.position = [0, 5, 0]
    cube.velocity = [0, -1.8, 0]
    paths = rt.PathSolver()(
        scene,
        max_depth=1,
        samples_per_src=10000,
        los=False,
        specular_reflection=True,
        diffuse_reflection=False,
    )
    a, tau = paths.cir(
        sampling_frequency=1000, num_time_steps=64, normalize_delays=False, out_type="numpy"
    )
    valid = tau[0, 0] >= 0
    assert np.count_nonzero(valid) > 0
    coefficients = a[0, 0, 0, 0, valid]
    np.testing.assert_allclose(tau[0, 0, valid], 2 * 4.5 / C0, rtol=2e-6)
    doppler = 2 * 1.8 * 28e9 / C0
    tone = np.exp(2j * np.pi * doppler * np.arange(64) / 1000)
    np.testing.assert_allclose(
        coefficients / coefficients[:, :1],
        np.broadcast_to(tone, coefficients.shape),
        atol=3e-5,
        rtol=0,
    )
    # Baseband carrier phase occurs exactly once, on top of the passband path gain.
    gains = paths.a
    passband = (
        np.asarray(gains[0])[0, 0, 0, 0, valid] + 1j * np.asarray(gains[1])[0, 0, 0, 0, valid]
    )
    np.testing.assert_allclose(
        coefficients[:, 0] / passband,
        np.exp(-2j * np.pi * 28e9 * tau[0, 0, valid].astype(float)),
        atol=5e-4,
        rtol=0,
    )
    h, _ = paths_to_frequency(
        a[0, :, 0, 0],
        tau[0, 0],
        valid,
        np.arange(64) / 1000,
        bandwidth=500e6,
        config=RangeResponseConfig("cfr_ifft", 128),
    )
    # Every sampled frequency uses the same carrier Doppler in one solve:
    # this deliberately does not assert the full migrating-delay model.
    np.testing.assert_allclose(
        h[0] / h[0, :1], tone[:, None] * np.ones((1, 128)), atol=3e-5, rtol=0
    )
