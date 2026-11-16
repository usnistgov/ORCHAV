"""The exported RD view and GUI use one display-only zoom policy."""

import numpy as np
import pytest

from shared.sensing.plotting import viz as sensing_viz
from shared.sensing.plotting.limits import range_doppler_zoom_limits


def test_zoom_includes_weak_overlays_and_neighboring_range_samples():
    power = np.full((5, 8), -200.0)
    power[2, 3] = -80
    ranges = np.arange(8.0) * 0.075
    velocity = np.arange(-2, 3.0) * 0.08
    before = power.copy()
    bounds = range_doppler_zoom_limits(power, ranges, velocity, extra_points=[(0.45, 0.08)])
    assert bounds[0] == pytest.approx((0.1125, 0.5625))
    assert bounds[1] == pytest.approx((-0.016, 0.096))
    np.testing.assert_array_equal(power, before)
    assert range_doppler_zoom_limits(np.full_like(power, np.nan), ranges, velocity) is None


@pytest.mark.parametrize("zoom", [True, False])
def test_exported_zoom_and_full_views_use_same_pixels(tmp_path, monkeypatch, zoom):
    power = np.full((5, 8), 1e-20)
    power[2, 3] = 1e-8
    ranges = np.arange(8.0) * 0.075
    doppler = np.arange(-2, 3.0) * 15.625
    velocity = sensing_viz.doppler_axis_to_velocity_m_s(doppler, 28e9)
    fig, ax = sensing_viz.plt.subplots()
    monkeypatch.setattr(sensing_viz.plt, "subplots", lambda **kw: (fig, ax))
    monkeypatch.setattr(sensing_viz.plt, "close", lambda _: None)
    sensing_viz.create_range_doppler_array_figure(
        power_linear=power,
        range_axis_m=ranges,
        doppler_axis_hz=doppler,
        carrier_freq_hz=28e9,
        range_mode="monostatic",
        zoom_to_data=zoom,
        display_config={"display_range_xlim": [100, 200], "display_velocity_ylim": [100, 200]},
        output_path=tmp_path / "rd.png",
    )
    np.testing.assert_allclose(ax.collections[0].get_array(), 10 * np.log10(power))
    if zoom:
        bounds = range_doppler_zoom_limits(10 * np.log10(power), ranges, velocity)
        np.testing.assert_allclose(ax.get_xlim(), bounds[0])
        np.testing.assert_allclose(ax.get_ylim(), bounds[1])
    else:
        np.testing.assert_allclose(ax.get_xlim(), [0, ranges[-1] + 0.075 / 2])
        assert ax.get_ylim()[0] < velocity[0] and ax.get_ylim()[1] > velocity[-1]
    fig.clear()
