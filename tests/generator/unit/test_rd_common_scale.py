"""Shared display-scale controls must not modify stored scientific values."""

import numpy as np
import pytest

from shared.sensing.plotting import viz as viz


def draw(tmp_path, monkeypatch, limits, interpolation="nearest"):
    seen = {}
    original = viz.plt.subplots

    def spy(*a, **kw):
        fig, ax = original(*a, **kw)
        seen["ax"] = ax
        return fig, ax

    monkeypatch.setattr(viz.plt, "subplots", spy)
    power = np.array([[1e-12, 1e-10], [1e-11, 1e-9]])
    copy = power.copy()
    result = viz.create_range_doppler_array_figure(
        power_linear=power,
        range_axis_m=np.array([1.0, 2.0]),
        doppler_axis_hz=np.array([-1.0, 1.0]),
        range_mode="monostatic",
        carrier_freq_hz=3.5e9,
        output_path=tmp_path / "rd.png",
        power_limits_db=limits,
        interpolation=interpolation,
    )
    assert result.is_file()
    np.testing.assert_array_equal(copy, power)
    np.testing.assert_allclose(seen["ax"].collections[0].get_array(), 10.0 * np.log10(power))
    return seen["ax"].collections[0].get_clim()


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_explicit_shared_limits(tmp_path, monkeypatch, interpolation):
    assert draw(tmp_path, monkeypatch, (-130.0, -80.0), interpolation) == (-130.0, -80.0)


def test_default_peak_relative_limits(tmp_path, monkeypatch):
    assert draw(tmp_path, monkeypatch, None) == (-150.0, -90.0)


def test_display_clipping_does_not_clip_power_values(tmp_path, monkeypatch):
    assert draw(tmp_path, monkeypatch, (-110.0, -100.0)) == (-110.0, -100.0)


@pytest.mark.parametrize("representation", ["power_linear", "complex_voltage"])
def test_typed_rd_plot_forwards_shared_limits(tmp_path, monkeypatch, representation):
    power = np.array([[1e-12, 1e-10], [1e-11, 1e-9]])
    values = power.copy() if representation == "power_linear" else np.sqrt(power) * 1j
    original = values.copy()
    ranges = np.array([1.0, 2.0])
    dopplers = np.array([-1.0, 1.0])
    monkeypatch.setattr(viz, "_require_resolved_plot_config", lambda _: ("monostatic", 3.5e9))
    monkeypatch.setattr(
        viz,
        "_require_single_pair_product",
        lambda *_a, **_kw: (values, (dopplers, ranges), {"representation": representation}),
    )
    monkeypatch.setattr(viz, "_canonical_truth_points", lambda _: ())
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(viz, "create_range_doppler_array_figure", capture)
    destination = tmp_path / "typed.png"
    assert (
        viz.create_range_doppler_figure(
            {"config_resolved": {}}, destination, power_limits_db=(-130.0, -80.0)
        )
        == destination
    )
    assert seen["power_limits_db"] == (-130.0, -80.0)
    np.testing.assert_allclose(seen["power_linear"], power)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(seen["range_axis_m"], ranges)
    np.testing.assert_array_equal(seen["doppler_axis_hz"], dopplers)


@pytest.mark.parametrize(
    "limits", [(0.0, 0.0), (1.0, 0.0), (np.nan, 0.0), (0.0, np.inf), (0.0,), (0.0, 1.0, 2.0)]
)
def test_invalid_limits_rejected(tmp_path, limits):
    with pytest.raises(ValueError, match="power_limits_db"):
        viz.create_range_doppler_array_figure(
            power_linear=np.ones((2, 2)),
            range_axis_m=np.array([1.0, 2.0]),
            doppler_axis_hz=np.array([-1.0, 1.0]),
            range_mode="monostatic",
            carrier_freq_hz=3.5e9,
            output_path=tmp_path / "invalid.png",
            power_limits_db=limits,
        )
